#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-18 · R1 —— v2.4 的**实际写规则**（Hebbian）能否支撑已验证的「寻址 → 读出 → 任务收益」链条？

判据见 ../DIRECTION.md（H1–H6，冻结于 rev 1.0）。

装置：`sgmn_module_hebb.py`（DIR-17 装置的**逐字派生**，只新增 `HebbianCore`）：
  * `hebbian_full` —— §3.2 全矩阵参考步：W <- W - ηG，G = α(Wk-v)k⊤ + λ(W-W⁰)（§3.1）
  * `hebbian_fact` —— §3.3 同步因子步：U <- U-ηGV，V <- V-ηG⊤U（**用旧 U、旧 V**）
  * 地址索引（keys/sigs）+ **地址选键 + §2.2 线性读**：y = sg(W)q，q = 地址匹配选出的键

臂（唯一变量 = 记忆核的写规则）：
  W0_verbatim（逐字存 (k,v) ⇒ 上界天花板）｜W1_hebb_full（§3.2）｜W2_hebb_fact（§3.3）
  W3_hebb_full_eta6/eta4（只改 η）｜W4_hebb_full_trainPv（任务损失同时更新 $P_v$）

任务、协议、判据实现、统计口径**全部沿用 DIR-17**（逐字派生）：
  每篇样本 = 目标文档 + 干扰文档（同线索、异码、异上下文）；码 = 单 token 稀有词、**在文档末尾**。
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
from sgmn_module_hebb import SGMNStack as SGS, grad_report   # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOUNT = [8, 12, 16]
ALPHA, LAM, GAMMA0, R = 1.0, 0.05, 0.05, 8
ETA = 1e-5
BETA = 64.0
# 【rev 1.3】目标文档与干扰文档各约 37 条 ⇒ 合计 ~74 条 ⇒ N_ENTRIES 必须 ≥ 74，
#   否则核心槽位回绕 ⇒ 命中指标与真值下标错位（D30 的同类缺陷）。128 ⇒ 槽位 = 写入序号。
N_ENTRIES = 128
SEEDS = [20261028, 20261104, 7]
N_EVAL, N_TRAIN, FILLER_TOK = 60, 200, 30
CUE_A = " The vault code is"
CUE_B = " The safe combination is"
MIN_FILLER_GAP = 400                            # 目标/干扰填充切片的最短间隔 ⇒ 签名必不同
# 【脚本 v1.2 · rev 1.4】码 = **单个稀有词 token**，且**位于文档的最后一个位置**。
#   理由（★ 关键）：§2.3 的因果写把 $h_t$ 存进 $v_t$ ⇒ **码之后任一位置的 $h_t$ 都含码的身份**
#   ⇒ 「位置就近」的邻位条目也能解出码 ⇒ 任何「读对条目 vs 读邻位」的对照都会双双饱和
#   （smoke#5 实测：E2 命中 0 却 nll=0.0000）。
#   ⇒ 把码放在文档末尾 ⇒ 只有**码自己那一条**持有其身份 ⇒ 邻位条目不含码 ⇒ 对照才有效。
MIN_FILLER_GAP = MIN_FILLER_GAP
SAT_EPS = 0.05                                 # 饱和门：E1/E2/E3 的 NLL 极差须 ≥ 此值
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")

res = {"dir": "DIR-17", "round": "R1", "checks": [], "arms": {}, "meta": {}}


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


def load_filler():
    import requests
    p = pathlib.Path.home() / "sgmn-loop" / "corpus" / "train.txt"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(CORPUS_BASE + "train.txt", timeout=90)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p.read_text(encoding="utf-8", errors="ignore")


def build_docs(tok, seed, n_train, n_eval):
    """每篇样本 = 目标文档 + 干扰文档（同线索 CUE_A、异码、异上下文）。
    码全局唯一 ⇒ 训练码与评测码**零重叠**（rev 1.3 的「评测码留出」）。"""
    import re as _re
    text = load_filler()
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    rng = random.Random(seed)
    rng.shuffle(paras)
    pool = []
    for q in paras:
        pool.extend(tok(q).input_ids)
        if len(pool) > 160000:
            break
    cueA = tok(CUE_A).input_ids
    code_pool = rare_word_tokens(tok, pool)
    out, used = [], set()
    guard = 0
    need = n_train + n_eval
    while len(out) < need and guard < 80 * need:
        guard += 1
        i = rng.randrange(0, len(pool) - 900)
        j = rng.randrange(0, len(pool) - 900)
        if abs(i - j) < MIN_FILLER_GAP:
            continue
        code_id = rng.choice(code_pool)
        dcode_id = rng.choice(code_pool)
        if code_id == dcode_id:
            continue
        used.add(code_id); used.add(dcode_id)
        fill = pool[i:i + FILLER_TOK]
        fill2 = pool[j:j + FILLER_TOK]
        doc = fill + cueA + [code_id]          # ★ 码 = 文档的最后一个 token
        dist = fill2 + cueA + [dcode_id]       # ★ 干扰文档也以码结尾（⇒ ctx 混合里有两份「码分量」）
        p1 = len(fill) + len(cueA)
        out.append({"ids": doc, "prefix": fill + cueA, "code_ids": [code_id],
                    "true_idx": p1 - 1, "code": str(code_id),
                    "dist_ids": dist, "dist_code": str(dcode_id),
                    "n_entries_doc": len(doc) - 1, "n_entries_dist": len(dist) - 1})
    tr, ev = out[:n_train], out[n_train:need]
    assert len(tr) == n_train and len(ev) == n_eval
    return tr, ev


_RARE_CACHE = {}


def rare_word_tokens(tok, corpus_ids=None, n_max=4000):
    """从词表里挑**单 token 的稀有词**（带前导空格、纯字母、长度 ≥ 6、往返可编码），
    并按**语料频次**升序取最稀有的 n_max 个。
    稀有 ⇒ 基座几乎不可能猜中 ⇒ 目标必须靠记忆读出（A2'⑨ 会实测基座 NLL ≥ 3）。"""
    key = (id(tok), n_max)
    if key in _RARE_CACHE:
        return _RARE_CACHE[key]
    vocab = tok.get_vocab()
    size = max(vocab.values()) + 1
    cand = []
    for i in range(1000, size):
        # ★ 注意：tok.get_vocab() 的键是**token 字符串**，不是 id ⇒ 不可用 `i not in vocab` 过滤
        try:
            s = tok.decode([i])
        except Exception:
            continue
        if not s or not s[0].isspace():
            continue
        w = s.strip()
        if len(w) < 6 or not w.isalpha():
            continue
        if tok(s).input_ids != [i]:
            continue
        cand.append((i, w))
    from collections import Counter
    cnt = Counter(corpus_ids) if corpus_ids is not None else Counter()
    cand.sort(key=lambda t: (cnt.get(t[0], 0), t[0]))
    out = [i for i, _ in cand[:n_max]]
    _RARE_CACHE[key] = out
    return out


def _norm_cdf(x):
    """标准正态 CDF（无 scipy ⇒ 用 math.erf）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sign_test_p(d):
    d = [x for x in d if abs(x) > 1e-15]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0
    k = sum(1 for x in d if x > 0)
    pk = math.comb(n, k) * 0.5 ** n
    p = sum(math.comb(n, i) * 0.5 ** n for i in range(n + 1)
            if math.comb(n, i) * 0.5 ** n <= pk + 1e-12)
    return min(1.0, p), k, n


def paired(A, B):
    """返回 B − A。对 NLL（越低越好）：>0 ⇒ A 更好；对 log-prob（越高越好）：>0 ⇒ B 更好。"""
    D = np.asarray(B, dtype=float) - np.asarray(A, dtype=float)
    D = D[np.isfinite(D)]
    if len(D) == 0:
        return {"mean": float("nan"), "t": float("nan"), "p": 1.0, "pos": 0, "n": 0,
                "consistent": False}
    if np.all(D == 0.0):
        return {"mean": 0.0, "sd": 0.0, "t": 0.0, "p": 1.0, "pos": 0, "n": len(D),
                "consistent": True}
    m = float(D.mean())
    sd = float(D.std(ddof=1))
    t = m / (sd / math.sqrt(len(D))) if sd > 0 else 0.0
    ps, k, n = sign_test_p(list(D))
    return {"mean": m, "sd": sd, "t": t, "p": ps, "pos": k, "n": n,
            "consistent": bool(k == n or k == 0)}


ARMS = [
    # name, query_mode, mem_kind, eta, train, train_pv, key_mode
    ("W0_verbatim",          "cue", "explicit_kv", None,  True,  False, "linear"),
    ("W1_hebb_full_eta5",    "cue", "hebbian_full", 1e-5, True,  False, "linear"),   # 发散对照（D43）
    ("W2_hebb_fact_eta5",    "cue", "hebbian_fact", 1e-5, True,  False, "linear"),   # 发散对照
    ("W3_hebb_full_eta6",    "cue", "hebbian_full", 1e-6, True,  False, "linear"),
    ("W3_hebb_full_eta7",    "cue", "hebbian_full", 1e-7, True,  False, "linear"),   # ★ 主 Hebbian 臂
    ("W3_hebb_full_eta8",    "cue", "hebbian_full", 1e-8, True,  False, "linear"),
    ("W4_hebb_full_trainPv", "cue", "hebbian_full", 1e-7, True,  True,  "linear"),   # 可训 P_v
    ("W5_hebb_full_ortho",   "cue", "hebbian_full", 1e-7, True,  False, "ortho"),    # ★ 正交键
    ("E4_cue_notrain",       "cue", "explicit_kv", None,  False, False, "linear"),
    ("E5_base",              None, "explicit_kv", None,  False, False, "linear"),
]


def make_stack(model, d, query_mode, seed=20261028, mem_kind="explicit_kv", eta=None,
               key_mode="linear"):
    st = SGS(model, MOUNT, d, d, R, ALPHA, LAM, (eta if eta is not None else ETA), GAMMA0, DEV,
             seed=seed, use_factorized=False, no_write=False, perpos=True,
             mem_kind=mem_kind, n_entries=N_ENTRIES, key_mode=key_mode)
    for m in st.mems.values():
        m.key_mode = key_mode
        m.query_mode = query_mode
        m.addr_mode = "sig"
        m.read_beta = BETA
    return st


def write_both(st, d):
    """写阶段：目标文档 + 干扰文档（都 no_grad）。顺序固定 ⇒ 目标条目在前。"""
    ids = torch.tensor(d["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
    dids = torch.tensor(d["dist_ids"], dtype=torch.long, device=DEV).unsqueeze(0)
    with torch.no_grad():
        st.model(input_ids=ids)
        st.model(input_ids=dids)


def train_arm(st, train_docs, mode, steps=None, train_pv=False):
    """两阶段：写阶段（no_grad）→ 冻结写 → 读阶段（梯度开）→ 目标码首 token 的 NLL → 反向 → 步。

    【DIR-18】train_pv=True ⇒ **任务损失同时更新写侧的值映射 $P_v$**（其余臂 $P_v$ 冻结）
    ⇒ ⇒ 「受保护的联合训练」的**最小形态**：写侧的**核状态 $W$** 仍是 no_grad（梯度不流入）。
    """
    snap0 = st.snapshot_all()
    params = [t for _, t in st.read_params_for(mode)]
    if train_pv:
        for li, m in st.mems.items():
            params.append(m.Pv)
    opt = torch.optim.Adam(params, lr=1e-3)
    w0 = {n: t.detach().clone() for n, t in
          [(f"L{li}.{k}", getattr(m, k)) for li, m in st.mems.items() for k in ("Wo", "gamma", "Wq")]}
    traj = []
    n = len(train_docs) if steps is None else min(steps, len(train_docs))
    for d in train_docs[:n]:
        st.unfreeze_write()          # 【D36】每篇开始前解冻写（否则上一轮遗留的 no_write=True 使写阶段空转）
        st.restore_all(snap0)
        write_both(st, d)            # ① 写阶段（目标 + 干扰）
        st.freeze_write()            # ② 冻结写
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        opt.zero_grad(set_to_none=True)
        lg = st.model(input_ids=pre).logits.float()   # ③ 读阶段（梯度开）
        tgt = torch.tensor([d["code_ids"][0]], dtype=torch.long, device=DEV)
        loss = F.cross_entropy(lg[0, -1:, :], tgt)
        loss.backward()
        gnorm = float(sum(float(p_.grad.norm()) ** 2 for p_ in params
                          if p_.grad is not None) ** 0.5)
        opt.step()
        traj.append((float(loss), gnorm))
    rep = {}
    for li, m in st.mems.items():
        rep[f"L{li}.Wo"] = {
            "changed": float((m.Wo.detach() - w0[f"L{li}.Wo"]).abs().max()),
            "max_abs": float(m.Wo.detach().abs().max()),
            "grad": None if m.Wo.grad is None else float(m.Wo.grad.norm()),
        }
        rep[f"L{li}.gamma"] = {
            "changed": float((m.gamma.detach() - w0[f"L{li}.gamma"]).abs().max()),
            "value": float(m.gamma.detach()),
            "grad": None if m.gamma.grad is None else float(m.gamma.grad.norm()),
        }
        rep[f"L{li}.Wq"] = {
            "changed": float((m.Wq.detach() - w0[f"L{li}.Wq"]).abs().max()),
            "grad": None if m.Wq.grad is None else float(m.Wq.grad.norm()),
        }
    st.unfreeze_write()
    return traj, rep


def eval_arm(st, eval_docs):
    rows = []
    snap0 = st.snapshot_all()
    for d in eval_docs:
        st.unfreeze_write()          # 【D36】同上
        st.restore_all(snap0)
        write_both(st, d)
        # 【D35】**写阶段一结束就快照写入值并清空列表**（与 DIR-16 的 run_arm 逐位同构）
        #   ⇒ 读阶段的 last_yr 与写阶段的 V 处在同一「代」上 ⇒ 命中指标才有意义。
        vals = {}
        for li, m in st.mems.items():
            vals[li] = torch.stack(m.vals_written) if m.vals_written else None
            m.vals_written.clear(); m.keys_written.clear()
        st.freeze_write()
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            lg = st.model(input_ids=pre).logits.float()
        lp = float(F.log_softmax(lg[0, -1], dim=-1)[d["code_ids"][0]])
        hits = []
        fids = []
        addr_ok = []
        for li, m in st.mems.items():
            yr = m.last_yr[0] if m.last_yr is not None else None
            V = vals[li]
            # ★【DIR-18 · rev 1.1 修订 51】地址准确率：签名匹配选中的槽 == 真值槽
            j_sel = m.last_cue_idx
            addr_ok.append(1.0 if (j_sel is not None and int(j_sel) == int(d["true_idx"])) else 0.0)
            if yr is None or V is None or V.shape[0] == 0:
                hits.append(0.0); fids.append(float("nan")); continue
            cs = F.cosine_similarity(V, yr.unsqueeze(0), dim=1)
            hits.append(1.0 if int(torch.argmax(cs)) == int(d["true_idx"]) else 0.0)
            # ★【DIR-18】写保真度：cos(W k_j, v_j)，j = 地址匹配选中的槽
            if hasattr(m.core, "fidelity"):
                j = m.last_cue_idx if m.last_cue_idx is not None else 0
                fids.append(float(m.core.fidelity(int(j))))
            else:
                fids.append(float("nan"))
        rows.append({"lp": lp, "nll": -lp, "hit": float(np.mean(hits)),
                     "addr": float(np.mean(addr_ok)) if addr_ok else float("nan"),
                     "fid": (float(np.nanmean(fids)) if (len(fids) and not all(math.isnan(f) for f in fids)) else float("nan")),
                     "n_entries": int(vals[list(vals)[0]].shape[0]) if vals and vals[list(vals)[0]] is not None else 0})
    st.unfreeze_write()
    return rows


def eval_base(model, eval_docs):
    rows = []
    for d in eval_docs:
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            lg = model(input_ids=pre).logits.float()
        lp = float(F.log_softmax(lg[0, -1], dim=-1)[d["code_ids"][0]])
        rows.append({"lp": lp, "nll": -lp})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_eval = 12 if args.smoke else N_EVAL
    n_train = 40 if args.smoke else N_TRAIN
    seeds = SEEDS[:2] if args.smoke else SEEDS

    print("=== DIR-17 R1（脚本 v1.1 · 卡片 rev 1.3 任务构造）：可寻址读取 + 读侧训练 ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size

    # ---------------- Stage A
    print("\n[Stage A]", flush=True)
    ids8 = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
    # A1' Freeze C（cue 模式）
    st = make_stack(model, d, "cue")
    base_before = {n: q.detach().clone() for n, q in model.named_parameters()}
    with torch.no_grad():
        st.model(input_ids=ids8)
    st.freeze_write()
    lg = model(input_ids=ids8).logits.float()
    F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids8[:, 1:].reshape(-1)).backward()
    rrep = grad_report(st.named_read_params())
    wrep = grad_report(st.named_write_params())
    gam = [gg for n, gg, i, b in rrep if n.endswith("gamma")]
    read_ok = (all((not b) for _, _, _, b in rrep) and all(gg > 0 for gg in gam)
               and max(gg for _, gg, _, _ in rrep) > 1e-6)
    write_ok = all(isn or gg == 0.0 for _, gg, isn, _ in wrep)
    base_same = all(torch.equal(q.detach(), base_before[n]) for n, q in model.named_parameters())
    nh, nm = len(st._handles), len(st.mems)
    st.remove(); torch.cuda.empty_cache()
    # 阴性对照
    st2 = SGS(model, MOUNT, d, d, R, ALPHA, LAM, ETA, GAMMA0, DEV, seed=20261028,
              use_factorized=False, perpos=True, drop_sg=True, mem_kind="explicit_kv",
              n_entries=N_ENTRIES)
    with torch.no_grad():
        st2.model(input_ids=ids8)
    for m in st2.mems.values():
        m.no_write = True
    lg2 = model(input_ids=ids8).logits.float()
    F.cross_entropy(lg2[:, :-1].reshape(-1, lg2.shape[-1]), ids8[:, 1:].reshape(-1)).backward()
    wrep2 = grad_report(st2.named_write_params())
    leak = len([1 for _, gg, isn, _ in wrep2 if not isn and gg > 0])
    st2.remove(); torch.cuda.empty_cache()
    for q in model.parameters():
        q.grad = None
    rec("A1' Freeze C（读梯度非零 / 写梯度精确 0 / 基座逐位不变 / 阴性对照泄漏）",
        read_ok and write_ok and base_same and leak > 0,
        f"读梯度非零 {sum(1 for _,_,i,_ in rrep if not i)}/{len(rrep)}；"
        f"写梯度精确0 {sum(1 for _,gg,i,_ in wrep if i or gg==0.0)}/{len(wrep)}；"
        f"基座逐位不变 {base_same}；阴性对照泄漏 {leak}/{len(wrep2)}",
        "读梯度>0 且 写梯度==0 且 基座不变 且 泄漏>0",
        {"read": read_ok, "write": write_ok, "base": base_same, "leak": leak})
    rec("A1' 句柄数 = 2×层数", nh == 2 * nm, f"handles={nh} mems={nm}", tol=2 * nm, value=nh)

    # A2' 训练的生效性
    st = make_stack(model, d, "cue")
    tr, ev = build_docs(tok, SEEDS[0], 4, 2)
    traj, rep = train_arm(st, tr, "cue")
    wo_ch = max(v["changed"] for k, v in rep.items() if k.endswith("Wo"))
    ga_ch = max(v["changed"] for k, v in rep.items() if k.endswith("gamma"))
    ga_gr = max(v["grad"] for k, v in rep.items() if k.endswith("gamma")
                and v["grad"] is not None)
    wq_gr = max((v["grad"] if v["grad"] is not None else 0.0)
                for k, v in rep.items() if k.endswith("Wq"))
    wq_ch = max(v["changed"] for k, v in rep.items() if k.endswith("Wq"))
    a2_ok = (wo_ch > 0 and ga_ch > 0 and ga_gr > 0 and wq_gr == 0.0 and wq_ch == 0.0)
    rec("A2' 训练生效性：Wo/gamma 改变且 gamma 梯度非零；cue 模式下 Wq 梯度恒 0 且不变", a2_ok,
        f"Δmax|Wo|={wo_ch:.4g}；Δmax|gamma|={ga_ch:.4g}；‖g_gamma‖={ga_gr:.4g}；"
        f"‖g_Wq‖={wq_gr:.4g}；Δ|Wq|={wq_ch:.4g}", value=a2_ok)
    # ===== ★【DIR-18】三条新检查（卡片 rev 1.0 的 A2'⑩⑪⑫）=====
    # A2' ⑩【Hebbian 核的自检】写一步后 W 确实朝 v 移动，且不爆炸
    st_h = make_stack(model, d, "cue", mem_kind="hebbian_full")
    core0 = st_h.mems[list(st_h.mems)[0]].core
    kk = torch.randn(core0.d_k, device=DEV, generator=torch.Generator(device=DEV).manual_seed(7))
    vv = torch.randn(core0.d_v, device=DEV, generator=torch.Generator(device=DEV).manual_seed(8))
    with torch.no_grad():
        d_before = float((core0._W() @ kk - vv).norm())
    core0.write(kk, vv)
    with torch.no_grad():
        d_after = float((core0._W() @ kk - vv).norm())
        wnorm = float(core0._W().norm())
    st_h.remove(); torch.cuda.empty_cache()
    rec("A2' ⑩ Hebbian 核自检：写一步后 W 朝 v 移动且不发散",
        (d_after < d_before) and math.isfinite(wnorm) and (wnorm < 1e6),
        f"‖Wk−v‖：{d_before:.4f} → {d_after:.4f}（须下降）；‖W‖={wnorm:.4g}（须有限且 < 1e6）",
        tol=d_before, value=d_after)
    # A2' ⑪【梯度隔离】Hebbian 臂：任务损失的梯度不流入核状态 W，且 P_v 的梯度恒 0
    st_h2 = make_stack(model, d, "cue", mem_kind="hebbian_full")
    tr_h, ev_h = build_docs(tok, SEEDS[0], 3, 2)
    train_arm(st_h2, tr_h, "cue")
    wgrad = []
    for li, m in st_h2.mems.items():
        wgrad.append(None if m.core.W.grad is None else float(m.core.W.grad.norm()))
    pvgrad = []
    for li, m in st_h2.mems.items():
        pvgrad.append(None if m.Pv.grad is None else float(m.Pv.grad.norm()))
    st_h2.remove(); torch.cuda.empty_cache()
    leak_w = [g for g in wgrad if g is not None and g != 0.0]
    leak_pv = [g for g in pvgrad if g is not None and g != 0.0]
    rec("A2' ⑪ 梯度隔离：Hebbian 臂中任务梯度不流入核状态 W，且（非 W4 臂）P_v 梯度恒 0",
        (len(leak_w) == 0) and (len(leak_pv) == 0),
        f"核状态 W 的梯度泄漏 {len(leak_w)}/{len(wgrad)}（须 0）；P_v 的梯度泄漏 {len(leak_pv)}/{len(pvgrad)}（须 0）",
        value={"W_leak": len(leak_w), "Pv_leak": len(leak_pv)})
    # A2' ④：读阶段冻结写 ⇒ 记忆逐位不变
    st.freeze_write()
    s_before = st.snapshot_all()
    with torch.no_grad():
        st.model(input_ids=torch.tensor(ev[0]["prefix"], dtype=torch.long,
                                       device=DEV).unsqueeze(0))
    s_after = st.snapshot_all()
    same = all(torch.equal(s_before[li][0], s_after[li][0]) for li in s_before)
    rec("A2' ④ 读阶段冻结写 ⇒ 记忆状态逐位不变", same, f"equal={same}", value=same)
    # A2' ⑤【跨方向逐位对照探针】评测期的命中必须复现 DIR-16 的 C1 = 1.0000
    st_p = make_stack(model, d, "cue", mem_kind="explicit_kv")
    tr_p, ev_p = build_docs(tok, SEEDS[0], 4, 6)
    rows_p = eval_arm(st_p, ev_p)
    hit_p = float(np.mean([r["hit"] for r in rows_p]))
    nent_p = int(np.mean([r["n_entries"] for r in rows_p]))
    st_p.remove(); torch.cuda.empty_cache()
    rec("A2' ⑤ 跨方向对照探针：未训练的 cue 臂命中 = 1.0（复现 DIR-16 的 C1）",
        hit_p >= 0.9, f"hit={hit_p:.4f}（DIR-16 的 C1 = 1.0000）；每篇写入条目数={nent_p}",
        tol=0.9, value=hit_p)
    # A2' ⑥【rev 1.3】写入条目数不得超过核心槽位数（否则槽位回绕 ⇒ 命中指标错位）
    rec("A2' ⑥ 槽位不回绕：每篇写入条目数 ≤ N_ENTRIES", nent_p <= N_ENTRIES,
        f"每篇 {nent_p} 条（N_ENTRIES={N_ENTRIES}）", tol=N_ENTRIES, value=nent_p)
    # A2' ⑦【rev 1.3】评测码留出：训练码与评测码零重叠
    ov = set(x["code"] for x in tr_p) & set(x["code"] for x in ev_p)
    ovd = set(x["code"] for x in tr_p) & set(x["dist_code"] for x in ev_p)
    rec("A2' ⑦ 评测码留出：训练码与评测码（含干扰码）零重叠",
        len(ov) == 0 and len(ovd) == 0,
        f"目标码重叠 {len(ov)}；与干扰码重叠 {len(ovd)}", value=len(ov) + len(ovd))
    # A2' ⑧【rev 1.4】干扰文档必须与目标文档**同线索、异码、异上下文**；且**码必须在文档末尾**
    cue_ids = tok(CUE_A).input_ids
    ok8 = True
    for x in list(tr_p[:2]) + list(ev_p[:2]):
        cid = int(x["code_ids"][0]); did = int(x["dist_code"])
        # 线索紧贴码之前
        same_cue = (x["ids"][len(x["ids"]) - len(cue_ids) - 1:len(x["ids"]) - 1] == cue_ids
                    and x["dist_ids"][len(x["dist_ids"]) - len(cue_ids) - 1:
                                     len(x["dist_ids"]) - 1] == cue_ids)
        code_last = (x["ids"][-1] == cid and x["dist_ids"][-1] == did)
        diff_code = cid != did
        diff_ctx = x["ids"][:FILLER_TOK] != x["dist_ids"][:FILLER_TOK]
        ok8 = ok8 and same_cue and code_last and diff_code and diff_ctx
    rec("A2' ⑧ 干扰项构造：同线索 + 异码 + 异上下文 + 码在文档末尾（唯一持有其身份的条目）", ok8,
        f"线索 = CUE_A（目标与干扰相同）；码不同（全局唯一）；填充切片不同（间隔 ≥ {MIN_FILLER_GAP} token）"
        f"⇒ 上下文签名必不同；码为文档最后一个 token ⇒ 邻位条目**不含**码的身份；抽样 4 篇通过={ok8}",
        value=1.0 if ok8 else 0.0)
    # A2' ⑨【rev 1.4】目标必须**不可被基座猜中**（否则任务平凡 ⇒ 又是饱和）
    rows_b = eval_base(model, ev_p[:8])
    nll_b = float(np.mean([r["nll"] for r in rows_b]))
    rec("A2' ⑨ 目标不可猜中：基座 NLL ≥ 3（否则任务平凡）", nll_b >= 3.0,
        f"base NLL = {nll_b:.4f}（阈值 3.0）；码为稀有单 token 词", tol=3.0, value=nll_b)
    # A2' ⑫【读路径一致性】W0_verbatim 的读出 y 必须≈槽内 v_j（cos ≥ 0.999）
    st_v = make_stack(model, d, "cue", mem_kind="explicit_kv")
    tr_v, ev_v = build_docs(tok, SEEDS[0], 2, 3)
    rows_v = eval_arm(st_v, ev_v)
    cs_min = []
    for li, m in st_v.mems.items():
        pass
    # 逐槽核对：读阶段 last_yr 与其来源槽的 v_j 的余弦
    snap_v = st_v.snapshot_all()
    for x in ev_v:
        st_v.unfreeze_write(); st_v.restore_all(snap_v)
        write_both(st_v, x)
        vals_v = {li: (torch.stack(m.vals_written) if m.vals_written else None)
                  for li, m in st_v.mems.items()}
        for li, m in st_v.mems.items():
            m.vals_written.clear(); m.keys_written.clear()
        st_v.freeze_write()
        pre_v = torch.tensor(x["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            st_v.model(input_ids=pre_v)
        for li, m in st_v.mems.items():
            yr = m.last_yr[0] if m.last_yr is not None else None
            Vv = vals_v[li]
            if yr is None or Vv is None or Vv.shape[0] == 0:
                cs_min.append(0.0); continue
            cs = F.cosine_similarity(Vv, yr.unsqueeze(0), dim=1)
            cs_min.append(float(cs.max()))
    st_v.remove(); torch.cuda.empty_cache()
    cs_min_mean = float(np.mean(cs_min)) if cs_min else 0.0
    rec("A2' ⑫ 读路径一致性：W0_verbatim 的读出 y 与其来源槽 v_j 的余弦 ≥ 0.999",
        cs_min_mean >= 0.999,
        f"cos(y, v_j) 均值 = {cs_min_mean:.6f}（阈值 0.999，{len(cs_min)} 个槽）",
        tol=0.999, value=cs_min_mean)

    # A4' 训练不发散
    losses = [t[0] for t in traj]
    finite = all(np.isfinite(losses))
    max_wo = max(v["max_abs"] for k, v in rep.items() if k.endswith("Wo"))
    gam_fin = all(np.isfinite(v["value"]) for k, v in rep.items() if k.endswith("gamma"))
    rec("A4' 训练不发散（损失有限、max|Wo| 有限、gamma 有限）",
        finite and np.isfinite(max_wo) and gam_fin,
        f"loss 首/末 = {losses[0]:.4f}/{losses[-1]:.4f}；max|Wo|={max_wo:.4g}；"
        f"gamma 终值={[round(v['value'],4) for k,v in rep.items() if k.endswith('gamma')]}",
        value=max_wo)
    res["meta"]["a4_loss"] = {"first": losses[0], "last": losses[-1]}
    st.remove(); torch.cuda.empty_cache()

    # ---------------- 主循环
    print("\n[主循环]", flush=True)
    all_rows = {}
    for seed in seeds:
        tr, ev = build_docs(tok, seed, n_train, n_eval)
        for name, qm, mkind, aeta, train, train_pv, kmode in ARMS:
            if qm is None:
                rows = eval_base(model, ev)
            else:
                st = make_stack(model, d, qm, seed=seed, mem_kind=mkind, eta=aeta,
                                key_mode=kmode)
                if train:
                    traj, rep = train_arm(st, tr, qm, train_pv=train_pv)
                else:
                    traj, rep = [], {}
                rows = eval_arm(st, ev)
                st.remove(); torch.cuda.empty_cache()
            all_rows.setdefault(name, {})[seed] = rows
            print(f"  seed={seed} {name:22s} nll={np.mean([r['nll'] for r in rows]):.4f} "
                  f"lp={np.mean([r['lp'] for r in rows]):+.4f} "
                  f"hit={np.mean([r.get('hit', np.nan) for r in rows]):.4f} "
                  f"addr={np.nanmean([r.get('addr', np.nan) for r in rows]):.4f} "
                  f"fid={np.nanmean([r.get('fid', np.nan) for r in rows]):.4f}", flush=True)

    # ---------------- 汇总
    print("\n[汇总与判据]", flush=True)
    agg = {}
    for name, per_seed in all_rows.items():
        lps = [r["lp"] for rs in per_seed.values() for r in rs]
        nlls = [r["nll"] for rs in per_seed.values() for r in rs]
        hits = [r["hit"] for rs in per_seed.values() for r in rs if "hit" in r]
        fids = [r["fid"] for rs in per_seed.values() for r in rs if "fid" in r]
        addrs = [r["addr"] for rs in per_seed.values() for r in rs if "addr" in r]
        agg[name] = {"lp": float(np.mean(lps)), "nll": float(np.mean(nlls)),
                     "fid": (float(np.nanmean(fids)) if fids else None),
                     "addr": (float(np.nanmean(addrs)) if addrs else None),
                     "lp_by_seed": [float(np.mean([r["lp"] for r in rs])) for rs in per_seed.values()],
                     "nll_by_seed": [float(np.mean([r["nll"] for r in rs])) for rs in per_seed.values()],
                     "hit": (float(np.mean(hits)) if hits else None), "n": len(lps)}
        print(f"  {name:22s} nll={agg[name]['nll']:.4f} lp={agg[name]['lp']:+.4f} "
              f"hit={agg[name]['hit']} addr={agg[name]['addr']} fid={agg[name]['fid']}", flush=True)
    res["arms"] = agg

    def flat(name, field):
        return [r[field] for rs in all_rows[name].values() for r in rs if field in r]

    def seed_delta(name_a, name_b, field="nll"):
        return [b - a for a, b in zip(agg[name_a][f"{field}_by_seed"], agg[name_b][f"{field}_by_seed"])]

    def check_pair(hname, label, a, b):
        """★ rev 1.3：统一 **NLL 口径**（越低越好）。paired(a,b) = b − a ⇒ >0 ⇒ a 更好。
        （此前用 log-prob 口径时符号方向写反 ⇒ D33。）"""
        dd = paired(flat(a, "nll"), flat(b, "nll"))
        sd = seed_delta(a, b, "nll")
        ok = (dd["mean"] > 0 and dd["p"] < 0.05 and all(x > 0 for x in sd))
        rec(f"{hname} {label}", ok,
            f"Δ(Δnll)={dd['mean']:+.6f} t={dd['t']:.2f} p={dd['p']:.3g} "
            f"pos={dd['pos']}/{dd['n']} 逐种子={['%+.4f'%x for x in sd]} "
            f"（nll: {a}={agg[a]['nll']:.4f} vs {b}={agg[b]['nll']:.4f}）",
            tol=0.0, value=dd["mean"])
        return dd, sd

    # ===== ★【DIR-18】判据 H1–H7（卡片 rev 1.0 冻结；rev 1.1/1.2 修订实现与主臂口径）=====
    MAIN_H = "W3_hebb_full_eta7"          # ★ 主 Hebbian 臂（rev 1.1 修订 55：η=1e-7 稳定）

    def check_pair_th(hname, label, a, b, thresh=0.0, pmax=0.05):
        """同 DIR-17 的 NLL 口径（paired(a,b) = b − a ⇒ >0 ⇒ a 更好），另加**幅度阈值**。"""
        dd = paired(flat(a, "nll"), flat(b, "nll"))
        sd = seed_delta(a, b, "nll")
        ok = (dd["mean"] > thresh and dd["p"] < pmax and all(x > 0 for x in sd))
        rec(f"{hname} {label}", ok,
            f"Δ(Δnll)={dd['mean']:+.6f}（阈值 > {thresh}） t={dd['t']:.2f} p={dd['p']:.3g} "
            f"pos={dd['pos']}/{dd['n']} 逐种子={['%+.4f' % x for x in sd]} "
            f"（nll: {a}={agg[a]['nll']:.4f} vs {b}={agg[b]['nll']:.4f}）",
            tol=thresh, value=dd["mean"])
        return dd, sd

    # H1〔天花板可复现〕W0_verbatim（逐字存储）相对 base 的收益 ≥ 4 nats
    d1, s1 = check_pair_th("H1", "〔★ 天花板可复现〕W0_verbatim（逐字存储）显著优于 base",
                            "W0_verbatim", "E5_base", thresh=4.0, pmax=1e-6)
    # H2〔★ v2.4 写规则可用〕主 Hebbian 臂相对 base 有显著收益
    d2, s2 = check_pair_th("H2", "〔★ v2.4 的实际写规则（Hebbian）可用〕主臂显著优于 base",
                            MAIN_H, "E5_base", thresh=0.0, pmax=0.05)
    # H3〔★ 写规则是瓶颈〕W0 − 主臂 ≥ 1.0 nats
    d3, s3 = check_pair_th("H3", "〔★ 写规则是瓶颈〕W0_verbatim 显著优于主 Hebbian 臂",
                            "W0_verbatim", MAIN_H, thresh=1.0, pmax=0.05)
    # H4〔★ 值映射是瓶颈〕W4（可训 P_v）− 主臂 ≥ 0.5 nats
    d4, s4 = check_pair_th("H4", "〔★ 值映射 P_v 是瓶颈〕W4_hebb_full_trainPv 显著优于主臂",
                            "W4_hebb_full_trainPv", MAIN_H, thresh=0.5, pmax=0.05)
    # H5〔地址与写规则正交〕Hebbian 臂的**地址准确率** ≥ 0.95（rev 1.1 修订 51：口径由 hit 改为 addr）
    heb_arms = ["W3_hebb_full_eta7", "W4_hebb_full_trainPv", "W5_hebb_full_ortho"]
    adr = {k: agg[k]["addr"] for k in heb_arms}
    ok5 = all((v is not None and v >= 0.95) for v in adr.values())
    rec("H5 〔地址与写规则正交〕Hebbian 臂的**地址准确率** ≥ 0.95", ok5,
        f"地址准确率：" + "、".join(f"{k.split('_hebb')[0]}={adr[k]}" for k in heb_arms) +
        "（阈值 0.95；★ 口径：地址匹配选中的槽 == 真值槽）",
        tol=0.95, value=min(v for v in adr.values() if v is not None))
    # H6〔写保真度的机制读数〕主臂的 cos(W k_j, v_j) 显著 > 0
    fid1 = [r["fid"] for rs in all_rows[MAIN_H].values() for r in rs
            if "fid" in r and not math.isnan(r["fid"])]
    if fid1:
        m_f = float(np.mean(fid1)); sd_f = float(np.std(fid1, ddof=1)); n_f = len(fid1)
        t_f = m_f / (sd_f / math.sqrt(n_f)) if sd_f > 0 else 0.0
        pos = sum(1 for x in fid1 if x > 0)
        p_f = 0.0 if pos == n_f else 2.0 * (1.0 - _norm_cdf(abs(t_f)))
        ok6 = (m_f > 0.05) and (p_f < 0.05)
        rec("H6 〔★ 写保真度〕主臂的 cos(W k_j, v_j) 显著 > 0（写有做功）", ok6,
            f"均值={m_f:.6f}（阈值 0.05） t={t_f:.3f} p≈{p_f:.3g} 正向 {pos}/{n_f}",
            tol=0.05, value=m_f)
    else:
        rec("H6 〔★ 写保真度〕主臂的 cos(W k_j, v_j) 显著 > 0（写有做功）", False,
            "无有效样本（fidelity 全为 NaN）", value=0.0)
    # H7〔★ 正交键能否救回写规则〕rev 1.1 修订 53
    fid_o = [r["fid"] for rs in all_rows["W5_hebb_full_ortho"].values() for r in rs
             if "fid" in r and not math.isnan(r["fid"])]
    m_o = float(np.mean(fid_o)) if fid_o else float("nan")
    d7, s7 = check_pair_th("H7", "〔★ 正交键能否救回写规则〕W5（正交键）优于主臂（线性键）",
                            "W5_hebb_full_ortho", MAIN_H, thresh=0.5, pmax=0.05)
    ok7 = (m_o == m_o) and (m_o > 0.05) and (d7["mean"] > 0.5) and (d7["p"] < 0.05)
    rec("H7 〔★ 正交键能否救回写规则〕W5（正交键）的写保真度 > 0.05 且任务收益优于主臂 ≥ 0.5 nats", ok7,
        f"fid(ortho)={m_o:.6f}（阈值 0.05）；fid(linear)={agg[MAIN_H]['fid']}；"
        f"Δ(Δnll)={d7['mean']:+.6f}（阈值 > 0.5） p={d7['p']:.3g} 逐种子={['%+.4f' % x for x in s7]}",
        tol=0.05, value=m_o)

    # 发散对照报告项（D43 的正面证据）
    for arm in ("W1_hebb_full_eta5", "W2_hebb_fact_eta5", "W3_hebb_full_eta6",
                "W3_hebb_full_eta7", "W3_hebb_full_eta8"):
        if arm in agg:
            nf = int(np.sum([1 for rs in all_rows[arm].values() for r in rs
                             if math.isfinite(r["nll"])]))
            tot = int(sum(len(rs) for rs in all_rows[arm].values()))
            print(f"  [发散报告] {arm:22s} 有限样本 {nf}/{tot}；nll={agg[arm]['nll']:.4f}；"
                  f"addr={agg[arm]['addr']}；fid={agg[arm]['fid']}", flush=True)

    # A3' 饱和预检（纪律 ㉔）：W0 / 主臂 / W5 三臂的 NLL 极差 ≥ 阈值
    trio = [agg[k]["nll"] for k in ("W0_verbatim", MAIN_H, "W5_hebb_full_ortho")]
    trio_ok = all(math.isfinite(x) for x in trio)
    spread = (float(max(trio) - min(trio)) if trio_ok else 0.0)
    res["meta"]["saturation"] = {"trio_nll": trio, "spread": spread, "eps": SAT_EPS}
    rec("A3' 饱和预检：W0/主臂/W5 的 NLL 极差 ≥ eps ⇒ 任务具备分辨率（否则判据无信息量）",
        trio_ok and spread >= SAT_EPS,
        f"nll(W0)={trio[0]:.4f} nll(主臂)={trio[1]:.4f} nll(W5)={trio[2]:.4f}；"
        f"极差={spread:.4f}（阈值 {SAT_EPS}）", tol=SAT_EPS, value=spread)

    passed = sum(1 for c in res["checks"] if c["passed"])
    res["summary_pass"] = {"passed": passed, "total": len(res["checks"])}
    print(f"\n=== 判据通过 {passed}/{len(res['checks'])} ===", flush=True)
    out = HERE.parent / ("R18_smoke_results.json" if args.smoke else "R18_results.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
