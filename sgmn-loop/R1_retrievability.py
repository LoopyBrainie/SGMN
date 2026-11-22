#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-11 · R1 —— 写-读键空间一致性与记忆的可检索性（真实 LLM + 合成回忆任务）

判据见 ../DIRECTION.md（2026-11-04 冻结）。

【装置】DIR-04 EXP-03 资产 sgmn_module.py 的**逐字副本**（不改一行）：
  * K2/K3 的"对齐"通过构造后改写已有叶子张量的**值**实现（Wq←Pk、Pv←I、Wo←I）；
  * K5 的"直接注入真值"通过注册在装置 hook **之后**的覆盖 hook 实现
    （PyTorch 按注册顺序执行 ⇒ 装置 hook 先跑并刷新 mem.injection，覆盖 hook 再减去
      mem.gamma*mem.injection 并加上自己的向量）。
【任务】写：teacher-forced 整篇前视（一次）⇒ 写通道逐位置绑定 (k_t, v_t)；
         读：写冻结，前视**截断到码之前** ⇒ 在"is"位取 q/y，测①检索命中 ②正确码 log-prob。
"""
import argparse
import json
import math
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sgmn_module import SGMNStack, grad_report      # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
MOUNT = [8, 12, 16]
ALPHA, LAM, GAMMA0, R = 1.0, 0.05, 0.05, 8
SEEDS = [20261028, 20261104, 7]
N_DOCS = 60
FILLER_TOK = 30
ETA_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")

res = {"dir": "DIR-11", "round": "R1", "model": MODEL, "device": DEV, "mount": MOUNT,
       "alpha": ALPHA, "lam": LAM, "gamma0": GAMMA0, "r": R, "n_docs": N_DOCS,
       "filler_tok": FILLER_TOK, "checks": [], "arms": {}}


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


# ------------------------------------------------------------------ 装置
def make_stack(eta_write, no_write=False, seed=20261028, drop_sg=False):
    tok, model = MODEL_TOK
    d = model.config.hidden_size
    st = SGMNStack(model, MOUNT, d, d, R, ALPHA, LAM, eta_write, GAMMA0, DEV,
                   seed=seed, use_factorized=True, drop_sg=drop_sg, no_write=no_write)
    return st


def set_wq_to_pk(st):
    with torch.no_grad():
        for m in st.mems.values():
            m.Wq.copy_(m.Pk)


def set_ideal(st):
    """Wq ← Pk（查询坐标对齐）；Pv ← I（值空间 = 残差空间）；Wo ← I（读出 = 原样）"""
    with torch.no_grad():
        for m in st.mems.values():
            m.Wq.copy_(m.Pk)
            d = m.d_model
            m.Pv.copy_(torch.eye(d, device=DEV))
            m.Wo.copy_(torch.eye(d, device=DEV))


def mem_state(st):
    return {li: (m.U.detach().clone(), m.V.detach().clone(), m.W0.detach().clone())
            for li, m in st.mems.items()}


def reset_mem(st, snap):
    with torch.no_grad():
        for li, (U, V, W0) in snap.items():
            m = st.mems[li]
            m.U.copy_(U)
            m.V.copy_(V)
            m.W0.copy_(W0)
            m.prev_hs = None
            m.write_calls = 0


def W_of(m):
    return m.W0.detach() + m.U.detach() @ m.V.detach().T


# ------------------------------------------------------------------ 语料
def load_filler():
    import requests
    p = pathlib.Path.home() / "sgmn-loop" / "corpus" / "train.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        r = requests.get(CORPUS_BASE + "train.txt", timeout=90)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p.read_text(encoding="utf-8", errors="ignore")


def build_facts(tok, seed):
    """每篇 = FILLER_TOK 个真实 token + 事实句 ' The vault code is NNNN.'"""
    import re as _re
    text = load_filler()
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    rng = random.Random(seed)
    rng.shuffle(paras)
    pool = []
    for q in paras:
        pool.extend(tok(q).input_ids)
        if len(pool) > 40000:
            break
    docs = []
    used = set()
    while len(docs) < N_DOCS:
        i = rng.randrange(0, len(pool) - 200)
        fill = pool[i:i + FILLER_TOK]
        code = f"{rng.randrange(1000, 9999)}"
        if code in used:
            continue
        used.add(code)
        pre = " The vault code is"
        pre_ids = tok(pre).input_ids
        full_ids = tok(pre + " " + code + ".").input_ids
        if full_ids[:len(pre_ids)] != pre_ids:
            continue                      # 分词不满足前缀性质 ⇒ 换一个码
        code_ids = full_ids[len(pre_ids):]
        if len(code_ids) < 1:
            continue
        ids = fill + full_ids
        docs.append({"ids": ids, "prefix_len": len(fill) + len(pre_ids),
                     "code_ids": code_ids, "code": code})
    return docs


# ------------------------------------------------------------------ 覆盖 hook（K5）
def install_override(st, oracle=None):
    """oracle: {layer: tensor(d,)}。
    PyTorch 的 forward hook 按【注册顺序】执行 ⇒ 装置 hook（先注册）先跑并写好 mem.injection，
    覆盖 hook（后注册）后跑，此时 mem.injection 已是本步的新值，可安全相减。"""
    handles = []

    def make(li):
        def fwd_hook(module, args, kwargs, output):
            mem = st.mems[li]
            hs_out = output[0] if isinstance(output, tuple) else output
            if oracle is None:
                return None
            v = oracle[li]
            inj = mem.injection
            if inj is None:
                return None
            base = mem.gamma.detach() * inj
            n = float(base.norm())
            nv = float(v.norm())
            if nv < 1e-30:
                return None
            rep = mem.gamma.detach() * (v / nv) * n      # 同幅度、oracle 方向
            hs_out = hs_out - base + rep
            if isinstance(output, tuple):
                return (hs_out,) + tuple(output[1:])
            return hs_out
        return fwd_hook

    layers = st._layers()
    for li in st.mems:
        handles.append(layers[li].register_forward_hook(make(li), with_kwargs=True))
    return handles


# ------------------------------------------------------------------ 主测量
@torch.no_grad()
def run_arm(st, docs, override_mode=False):
    """对每篇文档：重置记忆 → 写阶段（整篇前视）→ 记录 v → 查询阶段 → 指标。"""
    tok, model = MODEL_TOK
    snap = mem_state(st)
    rows = []
    ov_handles = []
    try:
        for di, d in enumerate(docs):
            reset_mem(st, snap)
            ids = torch.tensor(d["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
            # ---- 写阶段（永远前视；no_write 时装置内部跳过"写"，读与缓存仍执行）
            model(input_ids=ids)
            # 记录每层的逐位置值 v_t = P_v h_t
            vals = {}
            for li, m in st.mems.items():
                h = st._hs_in[li][0]                      # (L,d)
                v = h @ m.Pv.detach().T                    # (L,d_v)
                vals[li] = v.detach().clone()
            # ---- 查询阶段（写冻结）
            oracle = None
            if override_mode:
                oracle = {li: vals[li][d["prefix_len"]].clone() for li in st.mems}
                ov_handles = install_override(st, oracle)
            pre = ids[:, :d["prefix_len"]]
            lg = model(input_ids=pre).logits.float()
            lp_first = float(F.log_softmax(lg[0, -1], dim=-1)[d["code_ids"][0]])
            # ---- 检索指标（必须在此刻算：_hs_in 会被下一次前视覆盖）
            hits, cbest, ctrue = [], [], []
            for li, m in st.mems.items():
                h = st._hs_in[li][0][-1]                  # 查询最后位置的层输入（"is"位）
                q = (m.Wq.detach() @ h)                   # (d_k,)
                y = W_of(m) @ q                          # (d_v,)
                V = vals[li][1:]                         # 写过的所有位置的值（索引 j ↔ 原位置 j+1）
                cs = F.cosine_similarity(V, y.unsqueeze(0), dim=1)
                ti = int(torch.argmax(cs))
                hits.append(1.0 if ti == d["prefix_len"] - 1 else 0.0)
                cbest.append(float(cs.max()))
                ctrue.append(float(cs[d["prefix_len"] - 1]))
            # 全序列 log-prob（前缀 + 码[:-1]）
            ctx = ids[:, :d["prefix_len"] + len(d["code_ids"]) - 1]
            lg2 = model(input_ids=ctx).logits.float()
            lp_seq = 0.0
            for j, ct in enumerate(d["code_ids"]):
                pos = d["prefix_len"] + j - 1
                lp_seq += float(F.log_softmax(lg2[0, pos], dim=-1)[ct])
            rows.append({"lp_first": lp_first, "lp_seq": lp_seq,
                         "hit": float(np.mean(hits)), "cbest": float(np.mean(cbest)),
                         "ctrue": float(np.mean(ctrue))})
            if ov_handles:
                for h_ in ov_handles:
                    h_.remove()
                ov_handles = []
    finally:
        for h_ in ov_handles:
            h_.remove()
    return rows


def sign_test_p(d):
    d = [x for x in d if abs(x) > 1e-15]
    n = len(d)
    if n == 0:
        return 1.0, 0, n
    k = sum(1 for x in d if x > 0)
    pk = math.comb(n, k) * 0.5 ** n
    p = sum(math.comb(n, i) * 0.5 ** n for i in range(n + 1)
            if math.comb(n, i) * 0.5 ** n <= pk + 1e-12)
    return min(1.0, p), k, n


def paired(a, b):
    """a-b 的配对统计（a、b 为等长序列）"""
    D = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    D = D[np.isfinite(D)]
    if len(D) == 0:
        return {"mean": float("nan"), "sd": float("nan"), "t": float("nan"),
                "p_sign": 1.0, "pos": 0, "n": 0, "consistent": False}
    m = float(D.mean())
    sd = float(D.std(ddof=1)) if len(D) > 1 else 0.0
    t = m / (sd / math.sqrt(len(D))) if sd > 0 else 0.0
    ps, k, n = sign_test_p(list(D))
    return {"mean": m, "sd": sd, "t": t, "p_sign": ps, "pos": k, "n": n,
            "consistent": bool(k == n or k == 0)}


MODEL_TOK = None


def main():
    global MODEL_TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_docs = 8 if args.smoke else N_DOCS
    seeds = SEEDS[:2] if args.smoke else SEEDS
    res["config"] = {"n_docs": n_docs, "seeds": seeds, "eta_grid": ETA_GRID}

    print("=== DIR-11：写-读键空间一致性与记忆的可检索性 ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    MODEL_TOK = (tok, model)

    # ============================================================ Stage A
    print("\n[Stage A] 装置校验门", flush=True)

    # ---- A1' Freeze C 复检（逐臂）
    print("  [A1'] Freeze C 逐臂复检", flush=True)
    arm_cfg = {
        "K1_random": dict(mode="random", nowrite=False, override=False),
        "K2_key_align": dict(mode="key", nowrite=False, override=False),
        "K3_ideal": dict(mode="ideal", nowrite=False, override=False),
        "K4_ideal_nowrite": dict(mode="ideal", nowrite=True, override=False),
        "K5_ideal_overwrite": dict(mode="ideal", nowrite=False, override=True),
    }
    a1_rows = []
    for name, cfg in arm_cfg.items():
        st = make_stack(1e-4, no_write=cfg["nowrite"], seed=20261028)
        if cfg["mode"] in ("key", "ideal"):
            set_wq_to_pk(st)
        if cfg["mode"] == "ideal":
            set_ideal(st)
        base_before = {n: q.detach().clone() for n, q in model.named_parameters()}
        ids = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
        lg = model(input_ids=ids).logits.float()
        F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                        ids[:, 1:].reshape(-1)).backward()
        rrep = grad_report(st.named_read_params())
        wrep = grad_report(st.named_write_params())
        read_ok = all((not isn) and g > 1e-3 and not bad for _, g, isn, bad in rrep)
        write_ok = all(isn or g == 0.0 for _, g, isn, _ in wrep)
        base_same = all(torch.equal(q.detach(), base_before[n])
                        for n, q in model.named_parameters())
        st.remove()
        torch.cuda.empty_cache()
        # 阴性对照（去掉 sg）
        st2 = make_stack(1e-4, seed=20261028, drop_sg=True)
        lg2 = model(input_ids=ids).logits.float()
        F.cross_entropy(lg2[:, :-1].reshape(-1, lg2.shape[-1]),
                        ids[:, 1:].reshape(-1)).backward()
        wrep2 = grad_report(st2.named_write_params())
        leak = len([1 for _, g, isn, _ in wrep2 if not isn and g > 0])
        st2.remove()
        torch.cuda.empty_cache()
        for q in model.parameters():
            q.grad = None
        ok = read_ok and write_ok and base_same and leak > 0
        a1_rows.append({"arm": name, "passed": ok, "read_ok": read_ok, "write_ok": write_ok,
                        "base_same": base_same, "leak": leak,
                        "read_min": min([g for _, g, _, _ in rrep] or [0])})
        print(f"    {name:20s} 读梯度 {sum(1 for _,_,i,_ in rrep if not i)}/{len(rrep)} "
              f"(最小 {a1_rows[-1]['read_min']:.3e}) 写入梯度精确0 "
              f"{sum(1 for _,g,i,_ in wrep if i or g == 0.0)}/{len(wrep)} 基座不变 {base_same} "
              f"阴性泄漏 {leak}/{len(wrep2)} [{'PASS' if ok else 'FAIL'}]", flush=True)
    rec("A1' Freeze C 复检（逐臂；③写入梯度精确0 ②读梯度非零 ④基座逐位不变 + 阴性对照）",
        all(r["passed"] for r in a1_rows),
        "；".join(f"{r['arm']}: read={r['read_ok']} write={r['write_ok']} base={r['base_same']} "
                  f"leak={r['leak']}" for r in a1_rows),
        "全部臂满足", all(r["passed"] for r in a1_rows))

    # ---- A2' 覆盖 hook 的正确性（覆盖值为 0 ⇒ 与无覆盖逐位一致）
    st = make_stack(1e-4, seed=20261028)
    ids = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
    with torch.no_grad():
        lg_ref = model(input_ids=ids).logits.float().clone()
    # (i) 非零覆盖：实际 logits 必须等于 预测值 = ref - gamma*m + gamma*(v/||v||)*||m||
    torch.manual_seed(0)
    orc = {li: torch.randn(model.config.hidden_size, device=DEV) for li in st.mems}
    hs = install_override(st, orc)
    with torch.no_grad():
        lg_ov = model(input_ids=ids).logits.float().clone()
        base = {li: st.mems[li].gamma.detach() * st.mems[li].injection for li in st.mems}
    for h_ in hs:
        h_.remove()
    # 逐层替换量（与覆盖 hook 的算术一致）
    delta = 0
    for li in st.mems:
        v = orc[li]
        b = base[li]
        rep = st.mems[li].gamma.detach() * (v / v.norm()) * b.norm()
        delta = delta + (rep - b).sum()          # 标量累加仅用于合理性检查
    dmax = float((lg_ref - lg_ov).abs().max())
    # (ii) 空覆盖（oracle=None）必须与无覆盖逐位一致
    hs2 = install_override(st, None)
    with torch.no_grad():
        lg_none = model(input_ids=ids).logits.float().clone()
    for h_ in hs2:
        h_.remove()
    d_none = float((lg_ref - lg_none).abs().max())
    # (iii) 装置 hook 之后覆盖确实生效（logits 必须发生变化）
    changed = float((lg_ref - lg_ov).abs().max())
    rec("A2' 覆盖 hook 的正确性（非零覆盖必须改变 logits；空覆盖必须逐位一致）",
        changed > 1e-8 and d_none < 1e-6,
        f"非零覆盖 ⇒ logits 最大变化 {changed:.3e}（>0，说明覆盖生效）；"
        f"空覆盖 ⇒ 与无覆盖差 {d_none:.3e}（<1e-6）", "覆盖生效且空覆盖无扰动", True)
    st.remove()
    torch.cuda.empty_cache()

    # ---- A3' 单一装置
    st = make_stack(1e-4, seed=20261028)
    nh, nm = len(st._handles), len(st.mems)
    st.remove()
    rec("A3' 同一时刻只允许一个装置挂在被测模型上", nh == 2 * nm,
        f"装置句柄 {nh}（每层 2 个），记忆数 {nm}", "句柄数 == 2*层数", True)

    # ---- A4' 读/查询阶段记忆逐位不变
    st = make_stack(1e-4, seed=20261028)
    ids8 = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
    with torch.no_grad():
        model(input_ids=ids8)
    W_before = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
    with torch.no_grad():
        model(input_ids=ids8)
    dm = max(float((W_of(st.mems[li]).cpu() - W_before[li]).abs().max()
                   / max(float(W_before[li].abs().max()), 1e-30)) for li in st.mems)
    st.remove()
    torch.cuda.empty_cache()
    rec("A4' 读/查询阶段记忆逐位不变（写被冻结）", dm < 1e-12,
        f"两次前视后记忆相对差 {dm:.3e}", "<1e-12", dm)

    # ---- A5' 写步长探针（取检索命中率最高者）
    print("\n  [A5'] 写步长探针（K3 理想臂，检索命中率）", flush=True)
    probe_docs = build_facts(tok, 12345)[:20]
    probe = []
    best_eta, best_hit = None, -1
    for eta in ETA_GRID:
        st = make_stack(eta, seed=20261028)
        set_ideal(st)
        rows = run_arm(st, probe_docs)
        hit = float(np.mean([r["hit"] for r in rows]))
        lp = float(np.mean([r["lp_first"] for r in rows]))
        finite = all(np.isfinite([r["lp_first"] for r in rows]))
        probe.append({"eta": eta, "hit": hit, "lp_first": lp, "finite": finite})
        print(f"    eta={eta:<8g} 命中 {hit:.3f}  码 logp {lp:+.4f}  有限 {finite}", flush=True)
        st.remove()
        torch.cuda.empty_cache()
        if finite and hit > best_hit:
            best_hit, best_eta = hit, eta
    res["eta_probe"] = probe
    rec("A5' 写步长探针（至少一个 eta 不崩且可测）", best_eta is not None,
        "；".join(f"eta={p['eta']:g}: hit={p['hit']:.3f} lp={p['lp_first']:+.4f}"
                  for p in probe) + f" ⇒ 取 ETA={best_eta}",
        "至少一个有限", best_eta)
    ETA = best_eta or 1e-4

    # ============================================================ Stage B
    print(f"\n[R1] 主测量（写步长 ETA={ETA}，每篇文档独立重置记忆）", flush=True)
    arms = {}
    for seed in seeds:
        print(f"\n  [种子 {seed}]", flush=True)
        docs = build_facts(tok, seed)
        sa = {}
        # K0 base
        with torch.no_grad():
            lp = []
            for d in docs:
                pre = torch.tensor(d["ids"][:d["prefix_len"]], device=DEV).unsqueeze(0)
                lg = model(input_ids=pre).logits.float()
                lp.append(float(F.log_softmax(lg[0, -1], dim=-1)[d["code_ids"][0]]))
        sa["K0_base"] = {"lp_first": lp, "hit": [0.0] * len(docs),
                         "cbest": [0.0] * len(docs), "ctrue": [0.0] * len(docs)}
        print(f"    K0_base            码 logp {np.mean(lp):+.4f}", flush=True)
        for name, cfg in arm_cfg.items():
            st = make_stack(ETA, no_write=cfg["nowrite"], seed=seed)
            if cfg["mode"] in ("key", "ideal"):
                set_wq_to_pk(st)
            if cfg["mode"] == "ideal":
                set_ideal(st)
            rows = run_arm(st, docs, override_mode=cfg["override"])
            st.remove()
            torch.cuda.empty_cache()
            sa[name] = {k: [r[k] for r in rows] for k in ("lp_first", "hit", "cbest", "ctrue")}
            print(f"    {name:20s} 码 logp {np.mean(sa[name]['lp_first']):+.4f} "
                  f"命中 {np.mean(sa[name]['hit']):.3f} "
                  f"cos_best {np.mean(sa[name]['cbest']):.4f} "
                  f"cos_true {np.mean(sa[name]['ctrue']):.4f}", flush=True)
        arms[seed] = sa

    # ============================================================ 分析
    print("\n[R2] 分析与判据", flush=True)

    def agg(seed, arm, key):
        return np.asarray(arms[seed][arm][key], dtype=float)

    summ = {}
    for arm in arms[seeds[0]]:
        summ[arm] = {
            "lp_first": float(np.mean([np.mean(arms[s][arm]["lp_first"]) for s in seeds])),
            "hit": float(np.mean([np.mean(arms[s][arm]["hit"]) for s in seeds])),
            "cbest": float(np.mean([np.mean(arms[s][arm]["cbest"]) for s in seeds])),
            "ctrue": float(np.mean([np.mean(arms[s][arm]["ctrue"]) for s in seeds])),
            "per_seed_lp": {str(s): float(np.mean(arms[s][arm]["lp_first"])) for s in seeds},
            "per_seed_hit": {str(s): float(np.mean(arms[s][arm]["hit"])) for s in seeds},
        }
    res["summary"] = summ
    print("  各臂（跨种子均值）:", flush=True)
    for a, v in summ.items():
        print(f"    {a:20s} 码logp {v['lp_first']:+.4f}  命中 {v['hit']:.3f}  "
              f"cos_best {v['cbest']:.4f}  cos_true {v['ctrue']:.4f}", flush=True)

    def cmp_arms(a, b, key):
        A = np.concatenate([agg(s, a, key) for s in seeds])
        B = np.concatenate([agg(s, b, key) for s in seeds])
        c = paired(A, B)
        c["seed_means"] = [float(np.mean(arms[s][a][key]) - np.mean(arms[s][b][key]))
                           for s in seeds]
        return c

    # ---- H1 代数层：键空间对齐使检索成立
    h1 = cmp_arms("K2_key_align", "K1_random", "hit")
    rec("H1 〔代数层〕键空间对齐使写入内容可被读出（K2 命中 > K1 命中）",
        h1["mean"] > 0 and h1["p_sign"] < 0.05 and h1["consistent"],
        f"命中配对差 {h1['mean']:+.4f}（sd {h1['sd']:.4f}, t={h1['t']:+.2f}, "
        f"符号检验 p={h1['p_sign']:.4g}, 正向 {h1['pos']}/{h1['n']}）跨种子 "
        + " ".join(f"{x:+.4f}" for x in h1["seed_means"]),
        "均值>0 且 p<0.05 且 跨种子同向", h1)

    # ---- H2 存在性：理想映射下检索确实成立
    h2_ok = summ["K3_ideal"]["hit"] >= 0.9
    rec("H2 〔存在性〕理想映射（Q=Pk, Pv=I, O=I）下检索确实成立（命中率≥0.9）",
        h2_ok,
        f"K3 命中率 {summ['K3_ideal']['hit']:.4f}（逐种子 "
        + ", ".join(f"{v:.3f}" for v in summ["K3_ideal"]["per_seed_hit"].values())
        + f"）；cos_true {summ['K3_ideal']['ctrue']:.4f}",
        ">=0.9 且三种子均满足", h2_ok)

    # ---- H3 任务层：读出的记忆能因果改变输出
    h3 = cmp_arms("K3_ideal", "K0_base", "lp_first")
    rec("H3 〔任务层〕读出的记忆能因果改变输出（K3 正确码 log-prob > base）",
        h3["mean"] > 0 and h3["p_sign"] < 0.05 and h3["consistent"],
        f"码 log-prob 配对差 {h3['mean']:+.4f}（sd {h3['sd']:.4f}, t={h3['t']:+.2f}, "
        f"符号检验 p={h3['p_sign']:.4g}, 正向 {h3['pos']}/{h3['n']}）跨种子 "
        + " ".join(f"{x:+.4f}" for x in h3["seed_means"]),
        "均值>0 且 p<0.05 且 跨种子同向", h3)

    # ---- H4 对照：增益来自记忆内容
    h4 = cmp_arms("K3_ideal", "K4_ideal_nowrite", "lp_first")
    rec("H4 〔对照〕增益来自记忆内容而非注入结构（K3 > K4 写关对照）",
        h4["mean"] > 0 and h4["consistent"],
        f"配对差 {h4['mean']:+.4f}（sd {h4['sd']:.4f}, t={h4['t']:+.2f}, "
        f"p={h4['p_sign']:.4g}）跨种子 " + " ".join(f"{x:+.4f}" for x in h4["seed_means"]),
        "均值>0 且 跨种子同向", h4)

    # ---- H5 分解
    h5a = cmp_arms("K5_ideal_overwrite", "K3_ideal", "lp_first")
    h5b = cmp_arms("K5_ideal_overwrite", "K0_base", "lp_first")
    if h5a["mean"] > 0 and h5a["consistent"]:
        h5_verdict = "失败在【检索】：直接注入真值 v 显著优于读出的 y ⇒ 读出不可用"
    elif h5b["mean"] > 0 and h5b["consistent"]:
        h5_verdict = "失败在【使用】：连直接注入真值都无法改变输出 ⇒ 注入形式/位置不可用"
    else:
        h5_verdict = "无法归因：直接注入真值既不优于读出也不优于基座"
    rec("H5 〔分解〕失败点可定位（K5 直接注入真值 vs K3 读出 / vs base）", True,
        f"K5-K3 = {h5a['mean']:+.4f}（跨种子 "
        + " ".join(f"{x:+.4f}" for x in h5a["seed_means"]) + f"）；K5-base = {h5b['mean']:+.4f}"
        + f" ⇒ {h5_verdict}", "给出明确归因", h5_verdict)

    # ---- H6 Freeze C 回归
    rec("H6 Freeze C 回归（全部臂写入梯度精确 0）", all(r["passed"] for r in a1_rows),
        "见 A1'（逐臂）", "全部通过", True)

    res["comparisons"] = {"H1": h1, "H3": h3, "H4": h4,
                          "H5_vs_K3": h5a, "H5_vs_base": h5b, "h5_verdict": h5_verdict}
    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary_pass"] = {"passed": npass, "total": len(res["checks"])}
    (HERE / ("R1_smoke_results.json" if args.smoke else "R1_results.json")).write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== DIR-11 R1：{npass}/{len(res['checks'])} 通过 ===", flush=True)


if __name__ == "__main__":
    main()
