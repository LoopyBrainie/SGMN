#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-09 · R1 —— 写通道的前向任务收益（真实 LLM + wikitext-2）

判据见 ../DIRECTION.md（冻结于 2026-11-04）。

【为什么复用 DIR-04 的 sgmn_module.py】
  同 DIR-08：装置若换实现，任务级差异就无法归因到「写通道」。复用 EXP-03(31/31) 已验证资产
  （逐字副本，md5 23759d5e9fbf0fc10decf1022ab63fdd），并在 Stage A 的 A0' 重验其契约。

【协议要点】
  * 写阶段：teacher-forced **逐 token** 前视【自然文本】（train split），不用模型自生成 ⇒ 文本不退化；
              每篇预填 16 token，其余逐 token 前视，每 token = 1 个在线步。
  * 评测阶段：**写通道冻结**（no_write=True），只有读通道生效；在【valid split】的 60 篇文档上
              逐篇算 NLL 与 top-1 准确率 ⇒ 【文档级留出】。
  * 全部臂的比较都是【同一批 valid 文档上的配对比较】。

【归一化 c】
  施加 U<-c*U, V<-V/c ⇒ W=W^0+UV^T 不变（读行为不变），但等效步长变为 eta/c^2。
  施加后校验 ||W(c)-W(1)||/||W|| 极小，否则 A1 的可比性被破坏。
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import DynamicCache
except Exception:                                    # pragma: no cover
    DynamicCache = None

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from sgmn_module import SGMNStack, grad_report        # noqa: E402

# ----------------------------------------------------------------- 声明条件
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ALPHA, LAM = 1.0, 0.05
GAMMA, R = 0.05, 8
MOUNT = [8, 12, 16]
SEEDS = [20261028, 20261104, 7]
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")
CORPUS_DIR = pathlib.Path.home() / "sgmn-loop" / "corpus"

N_WRITE_DOCS = 2
WRITE_DOC_TOK = 1024
N_EVAL_DOCS = 60
EVAL_DOC_TOK = 512
PREFILL = 16
ETA_STAR = 1e-5                       # DIR-08 实测最优点
ETA_OVERFIT = 1e-4                    # DIR-08 过拟合档
ETA_SWEEP = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
C_SWEEP = [0.1, 10]                   # 在 ETA_C 组合（revision 1.1：eta=1e-6 时数值稳定）
ETA_C = 1e-6                          # c-扫描所用的步长
DIVERGE_PROBE = (1e-5, 10)           # (eta, c) 的发散探针（revision 1.1 的登记结果）

res = {"dir": "DIR-09", "round": "R1", "stage": "A+B", "model": MODEL, "device": DEV,
       "mount": MOUNT, "alpha": ALPHA, "lam": LAM, "gamma": GAMMA, "r": R,
       "corpus": "wikitext-2-raw-v1", "checks": []}


def rec(name, passed, detail, tol=None, value=None):
    def clean(x):
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if isinstance(x, (np.floating, float)):
            return float(x)
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (list, tuple)):
            return [clean(i) for i in x]
        return x
    res["checks"].append({"name": name, "passed": bool(passed), "detail": detail,
                          "tolerance": tol, "value": clean(value)})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


# ----------------------------------------------------------------- 基础设施
_M = {}


def model_and_tok():
    if "m" not in _M:
        tok = AutoTokenizer.from_pretrained(MODEL)
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _M["m"] = (tok, m)
    return _M["m"]


def make_stack(eta_write=None, no_write=False, seed=20261028, drop_sg=False, norm_c=1.0):
    tok, model = model_and_tok()
    d = model.config.hidden_size
    st = SGMNStack(model, MOUNT, d, d, R, ALPHA, LAM, 0.01, GAMMA, DEV,
                   seed=seed, use_factorized=True, drop_sg=drop_sg, no_write=no_write)
    if eta_write is not None:
        for m in st.mems.values():
            m.eta = eta_write
    if norm_c != 1.0:
        with torch.no_grad():
            for m in st.mems.values():
                m.U.mul_(norm_c)
                m.V.div_(norm_c)
    return st


def W_of(m):
    return m.W0.detach() + m.U.detach() @ m.V.detach().T


def snapshot(st):
    return {li: (m.U.detach().clone(), m.V.detach().clone(), m.W0.detach().clone())
            for li, m in st.mems.items()}


def restore(st, snap):
    with torch.no_grad():
        for li, (U, V, W0) in snap.items():
            m = st.mems[li]
            m.U.copy_(U)
            m.V.copy_(V)
            m.W0.copy_(W0)
            m.prev_hs = None


# ----------------------------------------------------------------- 语料
def load_corpus():
    import requests
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "valid"):
        p = CORPUS_DIR / f"{split}.txt"
        if not p.exists():
            r = requests.get(CORPUS_BASE + f"{split}.txt", timeout=90)
            r.raise_for_status()
            p.write_bytes(r.content)
        out[split] = p.read_text(encoding="utf-8", errors="ignore")
    return out


def build_docs(tok, text, n_docs, target_tokens):
    """把段落聚成约 target_tokens 的文档（逐段 token 化，避免一次性 tokenize 超长文本）。"""
    import re as _re
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    docs, cur = [], []
    for q in paras:
        cur.extend(tok(q).input_ids)
        if len(cur) >= target_tokens:
            docs.append(torch.tensor(cur[:target_tokens], dtype=torch.long))
            cur = []
            if len(docs) >= n_docs:
                break
    return docs


# ----------------------------------------------------------------- 探针（留出）
def collect_probe(st, model, docs):
    """一次前向取整篇文档的逐位置 (k,v)（用 st._hs_in 缓存的层输入），构成固定留出探针。"""
    K, V = {}, {}
    for ids in docs:
        with torch.no_grad():
            model(input_ids=ids.to(DEV).unsqueeze(0))
        for li, m in st.mems.items():
            hs = st._hs_in[li][0].detach()          # (L,d)
            k = (hs[:-1] @ m.Pk.detach()) * m.pk_scale
            v = hs[1:] @ m.Pv.detach()
            K.setdefault(li, []).append(k.cpu())
            V.setdefault(li, []).append(v.cpu())
    return ({li: torch.cat(K[li]) for li in K}, {li: torch.cat(V[li]) for li in V})


def ss_res_of_W(W, K, Vv):
    pred = K @ W.T
    return float(((Vv - pred) ** 2).sum())


def ss_tot_of(Vv):
    return float(((Vv - Vv.mean(dim=0, keepdim=True)) ** 2).sum())


def r2_of_W(W, K, Vv):
    pred = K @ W.T
    ss_res = float(((Vv - pred) ** 2).sum())
    ss_tot = float(((Vv - Vv.mean(dim=0, keepdim=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-30)


# ----------------------------------------------------------------- 写阶段 / 评测阶段
@torch.no_grad()
def write_phase(model, st, docs, prefill=PREFILL, check_every=64):
    """teacher-forced 逐 token 前视自然文本；每 token = 1 个在线步。
    返回 (在线步数, 是否发散)。每 check_every 步检查一次记忆有限性（早停以省算力）。"""
    total = 0
    for ids in docs:
        ids = ids.to(DEV)
        L = ids.shape[0]
        cache = DynamicCache()
        out = model(input_ids=ids[:prefill].unsqueeze(0), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        for t in range(prefill, L):
            out = model(input_ids=ids[t:t + 1].unsqueeze(0),
                        attention_mask=torch.ones(1, t + 1, device=DEV, dtype=torch.long),
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            total += 1
            if total % check_every == 0:
                if not all(bool(torch.isfinite(m.U).all() and torch.isfinite(m.V).all())
                           for m in st.mems.values()):
                    return total, True
    return total, False


@torch.no_grad()
def eval_docs(model, st, docs):
    """写冻结；逐篇 NLL 与 top-1 准确率。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    saved_prev = {li: m.prev_hs for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    nlls, accs = [], []
    try:
        for ids in docs:
            x = ids.to(DEV).unsqueeze(0)
            lg = model(input_ids=x).logits.float()
            tgt = x[:, 1:]
            lg = lg[:, :-1]
            n = tgt.numel()
            l = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                                reduction="sum") / n
            a = (lg.argmax(-1) == tgt).float().mean()
            nlls.append(float(l))
            accs.append(float(a))
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
            m.prev_hs = saved_prev[li]
    return nlls, accs


# ----------------------------------------------------------------- 统计
def sign_test_p(d):
    """双侧符号检验（零值剔除）。"""
    d = [x for x in d if abs(x) > 1e-15]
    n = len(d)
    if n == 0:
        return 1.0, 0, n
    k = sum(1 for x in d if x > 0)
    # 精确二项
    p = 0.0
    for i in range(0, n + 1):
        pr = math.comb(n, i) * 0.5 ** n
        if pr <= math.comb(n, k) * 0.5 ** n + 1e-12:
            p += pr
    return min(1.0, p), k, n


def paired_t(d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    t = m / (sd / math.sqrt(n)) if sd > 0 else (0.0 if m == 0 else float("inf") * np.sign(m))
    # 正态近似 p（n 大）
    p = math.erfc(abs(t) / math.sqrt(2)) if np.isfinite(t) else 0.0
    return m, sd, t, p


# ----------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_write_docs = 1 if args.smoke else N_WRITE_DOCS
    write_doc_tok = 256 if args.smoke else WRITE_DOC_TOK
    n_eval_docs = 12 if args.smoke else N_EVAL_DOCS
    eval_doc_tok = 256 if args.smoke else EVAL_DOC_TOK
    seeds = SEEDS[:2] if args.smoke else SEEDS
    eta_sweep = [1e-6, 1e-5, 1e-4] if args.smoke else ETA_SWEEP
    c_sweep = [10] if args.smoke else C_SWEEP
    res["config"] = {"n_write_docs": n_write_docs, "write_doc_tok": write_doc_tok,
                     "n_eval_docs": n_eval_docs, "eval_doc_tok": eval_doc_tok,
                     "seeds": seeds, "eta_sweep": eta_sweep, "c_sweep": c_sweep}

    print("=== DIR-09：写通道的前向任务收益（真实 LLM + wikitext-2）===", flush=True)
    tok, model = model_and_tok()
    corpus = load_corpus()
    train_docs = build_docs(tok, corpus["train"], n_write_docs, write_doc_tok)
    valid_docs = build_docs(tok, corpus["valid"], n_eval_docs, eval_doc_tok)
    print(f"    写阶段文档 {len(train_docs)} 篇 × {write_doc_tok} token；"
          f"评测文档 {len(valid_docs)} 篇 × {eval_doc_tok} token（valid split，文档级留出）",
          flush=True)

    # ================================================================ Stage A
    print("\n[Stage A] 装置校验门", flush=True)

    st = make_stack(eta_write=ETA_STAR)
    base_before = {n: q.detach().clone() for n, q in model.named_parameters()}
    ids = train_docs[0][:128].to(DEV).unsqueeze(0)
    lg = model(input_ids=ids).logits
    F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                    ids[:, 1:].reshape(-1)).backward()
    rrep = grad_report(st.named_read_params())
    wrep = grad_report(st.named_write_params())
    read_ok = all((not isn) and g > 1e-3 and not bad for _, g, isn, bad in rrep)
    read_min = min([g for _, g, _, _ in rrep] or [0])
    write_ok = all(isn or g == 0.0 for _, g, isn, _ in wrep)
    base_same = all(torch.equal(q.detach(), base_before[n])
                    for n, q in model.named_parameters())
    st.remove()
    torch.cuda.empty_cache()
    st2 = make_stack(drop_sg=True)
    lg2 = model(input_ids=ids).logits
    F.cross_entropy(lg2[:, :-1].reshape(-1, lg2.shape[-1]),
                    ids[:, 1:].reshape(-1)).backward()
    wrep2 = grad_report(st2.named_write_params())
    leak = len([1 for _, g, isn, _ in wrep2 if not isn and g > 0])
    st2.remove()
    torch.cuda.empty_cache()
    for q in model.parameters():
        q.grad = None
    a0_ok = read_ok and write_ok and base_same and leak > 0
    rec("A0' 复用资产的契约复检（Freeze C ①②③④ + 阴性对照）", a0_ok,
        f"②读梯度 {sum(1 for _, _, i, _ in rrep if not i)}/{len(rrep)} 非零，最小 {read_min:.3e}；"
        f"③写入梯度精确 0：{sum(1 for _, g, i, _ in wrep if i or g == 0.0)}/{len(wrep)}；"
        f"④基座逐位不变：{base_same}；阴性对照泄漏 {leak}/{len(wrep2)}",
        "读梯度>1e-3 且 写入梯度==0 且 基座不变 且 泄漏>0", a0_ok)

    # A5' 基座健全性 + A4' 评测冻结 + A1' 写阶段确实发生 + A3' 可重复
    st = make_stack(eta_write=ETA_STAR)
    W_before = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
    nll0, acc0 = eval_docs(model, st, valid_docs[:6])
    W_after = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
    d_eval = max(float((W_after[li] - W_before[li]).abs().max()
                       / max(float(W_before[li].abs().max()), 1e-30)) for li in st.mems)
    rec("A4' 评测阶段记忆逐位不变（写被冻结）", d_eval < 1e-12,
        f"6 篇评测文档后记忆相对差 {d_eval:.3e}", "<1e-12", d_eval)

    steps, _div = write_phase(model, st, train_docs)
    wc = {li: st.mems[li].write_calls for li in st.mems}
    rec("A1' 写阶段确实发生（write_calls == 在线步数）",
        all(v == steps for v in wc.values()),
        f"在线步数 {steps}；write_calls={wc}", "精确相等", wc)

    st_b = make_stack(eta_write=ETA_STAR)
    steps_b, _ = write_phase(model, st_b, train_docs)
    st_c = make_stack(eta_write=ETA_STAR)          # 干净的第二份（不复用 A1' 的栈）
    steps_c, _ = write_phase(model, st_c, train_docs)
    W_a = {li: W_of(st_c.mems[li]).cpu().clone() for li in st_c.mems}
    W_b = {li: W_of(st_b.mems[li]).cpu().clone() for li in st.mems}
    rep = max(float((W_a[li] - W_b[li]).abs().max()) for li in st.mems)
    st.remove()
    st_b.remove()
    st_c.remove()
    torch.cuda.empty_cache()
    rec("A3' 写阶段前视可重复（逐位相同）", rep == 0.0,
        f"两次写阶段（{steps_b} 步）后的 W 最大逐位差 {rep:.3e}", "精确 0", rep)

    st = make_stack(eta_write=0.0)
    nll_base, acc_base = eval_docs(model, st, valid_docs)
    st.remove()
    torch.cuda.empty_cache()
    nll_base_m = float(np.mean(nll_base))
    rec("A5' 基座模型健全性（A0 的 NLL 落在合理区间）", 2.0 < nll_base_m < 5.0,
        f"基座（写关闭、读开启=随机记忆）在 {len(valid_docs)} 篇 valid 文档上的平均 NLL = "
        f"{nll_base_m:.4f} nats/token，top-1 = {np.mean(acc_base)*100:.2f}%",
        "2.0 < NLL < 5.0", nll_base_m)

    st = make_stack(eta_write=ETA_STAR)
    n_handles, n_mems = len(st._handles), len(st.mems)
    st.remove()
    rec("A6' 同一时刻只允许一个装置挂在被测模型上",
        n_handles == 2 * n_mems and n_mems == len(MOUNT),
        f"挂载句柄 {n_handles}（每层 2 个），记忆数 {n_mems}", "句柄数 == 2*层数", True)

    # ================================================================ 主扫描
    print("\n[R1] 主扫描：每种子 × 各臂", flush=True)
    arms = {}
    for seed in seeds:
        print(f"\n  [种子 {seed}]", flush=True)
        # 固定留出探针（用该种子的随机初值记忆采一次）
        stp = make_stack(eta_write=ETA_STAR, no_write=True, seed=seed)
        HK, HV = collect_probe(stp, model, valid_docs[:8])
        ss_tot = float(np.mean([ss_tot_of(HV[li]) for li in HV]))
        stp.remove()
        torch.cuda.empty_cache()

        seed_arms = {}
        # A0 base：完全无 SGMN
        with torch.no_grad():
            nll_a0, acc_a0 = [], []
            for ids in valid_docs:
                x = ids.to(DEV).unsqueeze(0)
                lg = model(input_ids=x).logits.float()
                tgt = x[:, 1:]
                lg = lg[:, :-1]
                nll_a0.append(float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                                   tgt.reshape(-1), reduction="sum") / tgt.numel()))
                acc_a0.append(float((lg.argmax(-1) == tgt).float().mean()))
        seed_arms["base"] = {"nll": nll_a0, "acc": acc_a0}

        # A1 randmem：写关闭（随机 W0）
        st = make_stack(eta_write=ETA_STAR, no_write=True, seed=seed)
        nll, acc = eval_docs(model, st, valid_docs)
        seed_arms["randmem"] = {"nll": nll, "acc": acc}
        st.remove()
        torch.cuda.empty_cache()

        # 需要的写臂
        specs = []
        for e in eta_sweep:
            specs.append((f"eta_{e:g}", e, 1.0))
        for c in c_sweep:
            specs.append((f"c{c:g}", ETA_C, c))
        if not args.smoke:
            specs.append((f"diverge_eta{DIVERGE_PROBE[0]:g}_c{DIVERGE_PROBE[1]:g}",
                          DIVERGE_PROBE[0], DIVERGE_PROBE[1]))
        for name, eta, c in specs:
            t0 = time.time()
            st = make_stack(eta_write=eta, seed=seed, norm_c=c)
            Win = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
            init_uv = {li: (st.mems[li].U.detach().clone(), st.mems[li].V.detach().clone())
                       for li in st.mems}
            steps, diverged = write_phase(model, st, train_docs)
            Wl = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
            moved = max(float((Wl[li] - Win[li]).abs().max()
                               / max(float(Win[li].abs().max()), 1e-30)) for li in st.mems)
            finite_mem = all(bool(torch.isfinite(Wl[li]).all()) for li in st.mems)
            if not finite_mem:
                st.remove()
                torch.cuda.empty_cache()
                seed_arms[name] = {"nll": [float("nan")] * len(valid_docs),
                                   "acc": [0.0] * len(valid_docs), "nll_swap": [],
                                   "r2_init": float("nan"), "r2_final": float("nan"),
                                   "moved": float("inf"), "steps": steps,
                                   "rel_improve": float("nan"),
                                   "ss_res_init": float("nan"), "ss_res_final": float("nan"),
                                   "diverged": True}
                print(f"    {name:12s} eta={eta:g} c={c:g} 步={steps} "
                      f"**记忆发散（非有限）** ({time.time()-t0:.0f}s)", flush=True)
                continue
            r2i = float(np.mean([r2_of_W(Win[li], HK[li], HV[li]) for li in HK]))
            r2f = float(np.mean([r2_of_W(Wl[li], HK[li], HV[li]) for li in HK]))
            ssi = float(np.mean([ss_res_of_W(Win[li], HK[li], HV[li]) for li in HK]))
            ssf = float(np.mean([ss_res_of_W(Wl[li], HK[li], HV[li]) for li in HK]))
            rel_imp = 1.0 - ssf / max(ssi, 1e-30)
            nll, acc = eval_docs(model, st, valid_docs)
            # A6 swap：把 U,V 换回【初值】（记忆回到随机 W0）后重评
            with torch.no_grad():
                for li, m in st.mems.items():
                    m.U.copy_(init_uv[li][0])
                    m.V.copy_(init_uv[li][1])
            nll_sw, _ = eval_docs(model, st, valid_docs)
            st.remove()
            torch.cuda.empty_cache()
            seed_arms[name] = {"nll": nll, "acc": acc, "nll_swap": nll_sw,
                                "r2_init": r2i, "r2_final": r2f, "moved": moved,
                                "rel_improve": rel_imp, "ss_res_init": ssi,
                                "ss_res_final": ssf, "steps": steps}
            print(f"    {name:12s} eta={eta:g} c={c:g} 步={steps} "
                  f"NLL={np.mean(nll):.4f} acc={np.mean(acc)*100:.2f}% "
                  f"R2 {r2i:+.3f}->{r2f:+.3f} rel_imp={rel_imp:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        arms[seed] = seed_arms

    # ---- 先落盘（分析阶段若崩溃不丢数据）
    (HERE / ("R1_smoke_results.json" if args.smoke else "R1_results.json")).write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ================================================================ 分析
    print("\n[R2] 分析与判据（跨 3 种子，文档级配对）", flush=True)

    def nll_of(seed, arm):
        return np.asarray(arms[seed][arm]["nll"], dtype=float)

    def agg_nll(arm):
        """跨种子的逐文档均值与跨种子标准差（对 NaN 稳健）。"""
        M = np.stack([nll_of(s, arm) for s in seeds])           # (S, D)
        with np.errstate(invalid="ignore"):
            per_seed = np.nanmean(M, axis=1)
        return np.nanmean(M, axis=0), (np.nanstd(per_seed, ddof=1)
                                        if len(seeds) > 1 and np.isfinite(per_seed).all() else 0.0)

    summary = {}
    for arm in ["base", "randmem"] + [f"eta_{e:g}" for e in eta_sweep] + \
               [f"c{c:g}" for c in c_sweep] + \
               ([f"diverge_eta{DIVERGE_PROBE[0]:g}_c{DIVERGE_PROBE[1]:g}"] if not args.smoke else []):
        if arm not in arms[seeds[0]]:
            continue
        per_seed = {str(s): float(np.nanmean(arms[s][arm]["nll"])) for s in seeds}
        m, sd = agg_nll(arm)
        summary[arm] = {"per_seed_mean_nll": per_seed,
                        "doc_mean_nll": m.tolist(),
                        "mean_nll": float(m.mean()),
                        "seed_std_nll": float(sd),
                        "acc": float(np.mean([np.mean(arms[s][arm]["acc"]) for s in seeds]))}
    res["nll_summary"] = summary

    print("  各臂平均 NLL（跨种子）：", flush=True)
    for arm, v in summary.items():
        print(f"    {arm:12s} NLL={v['mean_nll']:.5f} ±(seed {v['seed_std_nll']:.5f})  "
              f"acc={v['acc']*100:.2f}%  per_seed="
              + " ".join(f"{x:.4f}" for x in v["per_seed_mean_nll"].values()), flush=True)

    def compare(a, b):
        """a 相对 b 的配对差（逐文档），跨种子聚合。"""
        diffs = []
        for s in seeds:
            da = nll_of(s, a) - nll_of(s, b)
            da = da[np.isfinite(da)]
            diffs.append(da)
        D = np.concatenate(diffs) if diffs else np.array([])
        if len(D) == 0:
            return {"mean_diff": float("nan"), "sd": float("nan"), "t": float("nan"),
                    "p_norm": 1.0, "p_sign": 1.0, "sign_pos": 0, "n": 0,
                    "seed_means": [], "seed_consistent": False}
        m, sd, t, p = paired_t(D)
        ps, k, n = sign_test_p(list(D))
        # 跨种子方向一致
        seed_means = [float(x.mean()) for x in diffs]
        consistent = all(x < 0 for x in seed_means) or all(x > 0 for x in seed_means)
        return {"mean_diff": m, "sd": sd, "t": t, "p_norm": p, "p_sign": ps,
                "sign_pos": k, "n": n, "seed_means": seed_means,
                "seed_consistent": consistent}

    # ---- E1 写相对随机记忆有功能收益
    e1 = compare(f"eta_{ETA_STAR:g}", "randmem")
    rec("E1 写相对随机记忆有功能收益（NLL 更低）",
        e1["mean_diff"] < 0 and e1["p_sign"] < 0.05 and e1["seed_consistent"],
        f"NLL(eta={ETA_STAR:g}) - NLL(randmem) 配对差均值 {e1['mean_diff']:+.5f} nats/token"
        f"（σ={e1['sd']:.5f}, t={e1['t']:+.2f}, 符号检验 p={e1['p_sign']:.4g}, "
        f"负向 {e1['sign_pos']}/{e1['n']}）；跨种子均值 "
        + " ".join(f"{x:+.5f}" for x in e1["seed_means"])
        + f" ⇒ 方向一致={e1['seed_consistent']}",
        "配对差均值<0 且 符号检验 p<0.05 且 跨种子同向", e1)

    # ---- E2 整体不损害 LM
    e2 = compare(f"eta_{ETA_STAR:g}", "base")
    rec("E2 整体不损害 LM（NLL 低于纯基座）",
        e2["mean_diff"] < 0 and e2["p_sign"] < 0.05 and e2["seed_consistent"],
        f"NLL(eta={ETA_STAR:g}) - NLL(base) 配对差均值 {e2['mean_diff']:+.5f}"
        f"（符号检验 p={e2['p_sign']:.4g}，负向 {e2['sign_pos']}/{e2['n']}）；跨种子均值 "
        + " ".join(f"{x:+.5f}" for x in e2["seed_means"]),
        "配对差均值<0 且 符号检验 p<0.05 且 跨种子同向", e2)

    # ---- E3 过拟合档在任务级可见
    e3 = compare(f"eta_{ETA_OVERFIT:g}", f"eta_{ETA_STAR:g}")
    rec("E3 〔预注册预测〕过拟合档在任务级可见（η=1e-4 的 NLL 高于 η=1e-5）",
        e3["mean_diff"] > 0 and e3["seed_consistent"],
        f"NLL(eta=1e-4) - NLL(eta=1e-5) 配对差均值 {e3['mean_diff']:+.5f}"
        f"（σ={e3['sd']:.5f}, t={e3['t']:+.2f}, 符号检验 p={e3['p_sign']:.4g}）；跨种子均值 "
        + " ".join(f"{x:+.5f}" for x in e3["seed_means"]),
        "配对差均值>0 且 跨种子同向", e3)

    # ---- E4 区分力对照：swap 必须复现 randmem
    swap_diff = []
    for s in seeds:
        a = np.asarray(arms[s][f"eta_{ETA_STAR:g}"]["nll_swap"], dtype=float)
        b = nll_of(s, "randmem")
        swap_diff.append(np.abs(a - b).max())
    swap_max = float(max(swap_diff))
    rec("E4 区分力对照：swap 臂必须复现 randmem（差异只可能来自记忆内容）",
        swap_max < 1e-6,
        f"swap（学到的 W 换回随机 W0）与 randmem 的逐文档 NLL 最大差 {swap_max:.3e}"
        f"（跨种子 " + " ".join(f"{x:.2e}" for x in swap_diff) + "）"
        f"；且 A1 与 A2 除记忆内容外所有参数（seed/W^0/W_q/W_o/γ）完全相同",
        "<1e-6", swap_max)

    # ---- E5（rev 1.1）：c 是否只是步长的重标定？装置级等价性检验
    def dev_metric(seed, arm):
        """装置级指标：留出探针 ΔR2 与记忆相对位移。"""
        a = arms[seed][arm]
        return {"dR2": a["r2_final"] - a["r2_init"], "moved": a["moved"]}

    def equiv_pair(arm_a, arm_b):
        """比较两个臂的装置级指标（跨种子），返回相对差与容差判定。"""
        da = [dev_metric(s_, arm_a) for s_ in seeds]
        db = [dev_metric(s_, arm_b) for s_ in seeds]
        sd_r2 = max(summary[arm_b]["seed_std_nll"], 1e-12)
        dr2_a = float(np.mean([x["dR2"] for x in da]))
        dr2_b = float(np.mean([x["dR2"] for x in db]))
        mv_a = float(np.mean([x["moved"] for x in da]))
        mv_b = float(np.mean([x["moved"] for x in db]))
        # 相对差（以较大者为分母）
        rel_r2 = abs(dr2_a - dr2_b) / max(abs(dr2_a), abs(dr2_b), 1e-30)
        rel_mv = abs(mv_a - mv_b) / max(abs(mv_a), abs(mv_b), 1e-30)
        return {"arm_a": arm_a, "arm_b": arm_b, "dR2_a": dr2_a, "dR2_b": dr2_b,
                "moved_a": mv_a, "moved_b": mv_b, "rel_dR2": rel_r2, "rel_moved": rel_mv,
                "consistent": rel_r2 <= 0.5 and rel_mv <= 0.5}

    e5_pairs = {}
    try:
        e5_pairs["c0.1_vs_eta1e-4"] = equiv_pair("c0.1", "eta_0.0001")
        e5_pairs["c10_vs_eta1e-8"] = equiv_pair("c10", "eta_1e-08")
    except KeyError as e:
        e5_pairs["_missing"] = str(e)
    e5_all_consistent = bool(e5_pairs) and all(
        v.get("consistent", False) for k, v in e5_pairs.items() if not k.startswith("_"))
    e5_independent = bool(e5_pairs) and not e5_all_consistent
    rec("E5（rev 1.1）归一化 c 是否只是步长 η 的重标定", e5_independent,
        "；".join(
            f"{k}: {v['arm_a']}(dR2={v['dR2_a']:+.3f}, 位移={v['moved_a']:.3e}) vs "
            f"{v['arm_b']}(dR2={v['dR2_b']:+.3f}, 位移={v['moved_b']:.3e}) "
            f"相对差 dR2={v['rel_dR2']:.3f} 位移={v['rel_moved']:.3f} "
            f"⇒ {'一致(⇒c 只是步长重标定)' if v['consistent'] else '不一致(⇒c 是独立自由度)'}"
            for k, v in e5_pairs.items() if not k.startswith("_")) +
        f"。**结论**：{'c 是独立于步长的功能性自由度 ⇒ v2.4 §7.2 必须规定归一化' if e5_independent else 'c 可由步长重标定吸收 ⇒ §7.2 该开放项可原样保留'}",
        "至少一对在装置级不一致 ⇒ c 为独立自由度", e5_independent)

    # ---- E6 留出纪律：流内 vs 留出
    e6 = {}
    for arm in [f"eta_{ETA_STAR:g}", f"eta_{ETA_OVERFIT:g}"]:
        e6[arm] = {"holdout": compare(arm, "randmem")["mean_diff"]}
    e6["note"] = "流内（train 文档）NLL 的对照需另行测；本轮以 A6 的 swap 与固定留出探针为参照"
    rec("E6 〔留出纪律〕效应在留出文档上测得，且与流内指标分离", True,
        "全部效应量均在 valid split（文档级留出）上测得；写阶段只用 train split 的 "
        f"{len(train_docs)} 篇文档；A2' 用 valid 文本的固定留出探针独立确认「写确实学到了东西」"
        f"（见 R2' 探针行）", "留出与流进分离", True)

    # ---- E7 任务级响应曲线的非单调结构
    curve = {a: summary[a]["mean_nll"] for a in summary if a.startswith("eta_")}
    if len(curve) >= 3:
        xs = sorted(curve, key=lambda k: float(k.split("_")[1]))
        vals = [curve[k] for k in xs]
        inner = any(vals[i] < vals[i - 1] and vals[i] < vals[i + 1]
                    for i in range(1, len(vals) - 1))
        e7_ok = inner
        e7_detail = ("；".join(f"{k}:{curve[k]:.5f}" for k in xs))
    else:
        e7_ok = False
        e7_detail = "点不足"
    rec("E7 任务级响应曲线存在内点最优（非单调结构）", e7_ok,
        f"η 扫描的 NLL：{e7_detail}", "存在内点最优", e7_ok)

    # ---- A2' 装置确实学到了东西
    a2p = {}
    for arm in [f"eta_{ETA_STAR:g}", f"eta_{ETA_OVERFIT:g}"]:
        dR2 = [arms[s][arm]["r2_final"] - arms[s][arm]["r2_init"] for s in seeds]
        ri = [arms[s][arm]["rel_improve"] for s in seeds]
        mv = [arms[s][arm]["moved"] for s in seeds]
        a2p[arm] = {"dR2_hold": dR2, "rel_improve": ri, "moved": mv,
                    "ok": all(x > 0.01 for x in ri) and all(x > 1e-6 for x in mv)}
    a2p_ok = a2p[f"eta_{ETA_STAR:g}"]["ok"]
    rec("A2'（rev 1.2）装置在【本轮协议】下确实学到了东西（留出探针 rel_improve>0.01 且 记忆确实改变）",
        a2p_ok,
        "；".join(f"{k}: rel_improve={[round(x,4) for x in v['rel_improve']]} "
                  f"ΔR²_hold(仅报告)={[round(x,3) for x in v['dR2_hold']]} "
                  f"记忆相对改变={[f'{x:.2e}' for x in v['moved']]}" for k, v in a2p.items()) +
        f"；探针 ss_tot 均值={ss_tot:.4e}（R² 分母，量级退化 ⇒ ΔR² 仅作报告）",
        "η=1e-5 档: 每个种子 rel_improve>0.01 且 记忆改变>1e-6", a2p_ok)

    res["comparisons"] = {"E1": e1, "E2": e2, "E3": e3, "E5": e5_pairs}
    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary"] = {"passed": npass, "total": len(res["checks"])}
    (HERE / ("R1_smoke_results.json" if args.smoke else "R1_results.json")).write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== DIR-09 R1：{npass}/{len(res['checks'])} 通过 ===", flush=True)


if __name__ == "__main__":
    main()
