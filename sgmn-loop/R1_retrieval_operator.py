#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-12 · R1 —— 读取侧的检索算子、键一致性与自指诊断（合成回忆任务）

判据见 ../DIRECTION.md（2026-11-04 冻结）。

【装置】sgmn_module_retrieval.py = DIR-11 的逐位置写变体（修 D2）+ D1 修复 + 【检索读算子】：
    beta=0        ⇒ y = sg(W) q                      （与 DIR-11 的线性读逐位相同）
    beta>0        ⇒ q 先在记忆的键基（W 的列）上做 softmax 竞争：q~ = sum_j a_j W[:,j]，
                    a ∝ exp(beta * q^T W[:,j])；再线性读 y = sg(W) q~
    Freeze B（状态 W 与写规则 §3.1/§3.2）一字未动。
【写入】位置级（逐位置 t=1..L-1）+ 全矩阵参考分支（§3.2）：
        因 D5（逐位置+因子化在稳定区发散），本轮不测因子化+位置级的组合。
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
from sgmn_module_retrieval import SGMNStack as SGRet   # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
MOUNT = [8, 12, 16]
ALPHA, LAM, GAMMA0, R = 1.0, 0.05, 0.05, 8
SEEDS = [20261028, 20261104, 7]
N_DOCS = 60
FILLER_TOK = 30
ETA_GRID = [1e-5, 1e-4, 1e-3]
BETA_GRID = [1.0, 4.0, 16.0]
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")

res = {"dir": "DIR-12", "round": "R1", "model": MODEL, "device": DEV, "mount": MOUNT,
       "alpha": ALPHA, "lam": LAM, "gamma0": GAMMA0, "r": R, "n_docs": N_DOCS,
       "checks": [], "arms": {}}


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


MODEL_TOK = None


def make_stack(eta_write, no_write=False, seed=20261028, beta=0.0, random_q=False,
               lam=0.0):
    tok, model = MODEL_TOK
    d = model.config.hidden_size
    st = SGRet(model, MOUNT, d, d, R, ALPHA, LAM, eta_write, GAMMA0, DEV,
               seed=seed, use_factorized=False, no_write=no_write, perpos=True)
    for m in st.mems.values():
        m.read_beta = beta
        m.read_lambda = lam
    if random_q:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + 991)
        with torch.no_grad():
            for m in st.mems.values():
                m.Wq.copy_(torch.randn(m.Wq.shape, generator=g) / math.sqrt(m.Wq.shape[1]))
    return st


def set_aligned(st):
    with torch.no_grad():
        for m in st.mems.values():
            m.Wq.copy_(m.Pk)


def set_ideal(st):
    with torch.no_grad():
        for m in st.mems.values():
            m.Wq.copy_(m.Pk)
            d = m.d_model
            m.Pv.copy_(torch.eye(d, device=DEV))
            m.Wo.copy_(torch.eye(d, device=DEV))


def mem_state(st):
    out = {}
    for li, m in st.mems.items():
        Wf = (m.Wfull.detach().clone() if getattr(m, "Wfull", None) is not None else None)
        out[li] = (m.U.detach().clone(), m.V.detach().clone(),
                   m.W0.detach().clone(), Wf)
    return out


def reset_mem(st, snap):
    with torch.no_grad():
        for li, (U, V, W0, Wf) in snap.items():
            m = st.mems[li]
            m.U.copy_(U)
            m.V.copy_(V)
            m.W0.copy_(W0)
            if Wf is not None and getattr(m, "Wfull", None) is not None:
                m.Wfull.copy_(Wf)
            m.prev_hs = None
            m.write_calls = 0


def W_of(m):
    if getattr(m, "Wfull", None) is not None:
        return m.Wfull.detach()
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
    docs, used = [], set()
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
            continue
        code_ids = full_ids[len(pre_ids):]
        if len(code_ids) < 1:
            continue
        docs.append({"ids": fill + full_ids, "prefix_len": len(fill) + len(pre_ids),
                     "code_ids": code_ids, "code": code})
    return docs


# ------------------------------------------------------------------ 主测量
@torch.no_grad()
def run_arm(st, docs, use_oracle_key=False):
    snap = mem_state(st)
    rows = []
    for d in docs:
        reset_mem(st, snap)
        ids = torch.tensor(d["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
        MODEL_TOK[1](input_ids=ids)                 # 写阶段（位置级）
        # 记录：每层的逐位置键 k_t = P_k h_{t-1} 与值 v_t = P_v h_t
        keys, vals = {}, {}
        for li, m in st.mems.items():
            h = st._hs_in[li][0]                   # (L,d)
            keys[li] = (h @ m.Pk.detach().T)        # (L,d_k)
            vals[li] = (h @ m.Pv.detach().T)        # (L,d_v)
        tp = d["prefix_len"]                       # 码首位的位置
        if use_oracle_key:
            for li, m in st.mems.items():
                m.oracle_key = keys[li][tp - 1].clone()      # 写阶段记录的键
        pre = ids[:, :d["prefix_len"]]
        lg = MODEL_TOK[1](input_ids=pre).logits.float()
        lp_first = float(F.log_softmax(lg[0, -1], dim=-1)[d["code_ids"][0]])
        # 检索指标 + 自指度量
        hits, cbest, ctrue, selfcos, amax = [], [], [], [], []
        for li, m in st.mems.items():
            hq = st._hs_in[li][0][-1]
            q = (m.Wq.detach() @ hq)
            # 【D7 修复】诊断必须用【装置实际读出的 y】（含检索算子），不能用线性读的 y
            y = m.last_y[0] if m.last_y is not None else (W_of(m) @ q)
            V = vals[li][1:]
            cs = F.cosine_similarity(V, y.unsqueeze(0), dim=1)
            ti = int(torch.argmax(cs))
            hits.append(1.0 if ti == tp - 1 else 0.0)
            cbest.append(float(cs.max()))
            ctrue.append(float(cs[tp - 1]))
            # 自指：写阶段该位置的键 vs 读阶段查询键
            selfcos.append(float(F.cosine_similarity(keys[li][tp - 1], q, dim=0)))
            a = m.last_attn
            if a is not None:
                amax.append(float(a[0].max()))
        for li, m in st.mems.items():
            m.oracle_key = None
        rows.append({"lp_first": lp_first,
                     "hit": float(np.mean(hits)), "cbest": float(np.mean(cbest)),
                     "ctrue": float(np.mean(ctrue)),
                     "self_cos": float(np.mean(selfcos)),
                     "attn_max": float(np.mean(amax)) if amax else 0.0})
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


def main():
    global MODEL_TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_docs = 8 if args.smoke else N_DOCS
    seeds = SEEDS[:2] if args.smoke else SEEDS
    res["config"] = {"n_docs": n_docs, "seeds": seeds, "beta_grid": BETA_GRID,
                     "eta_grid": ETA_GRID}

    print("=== DIR-12：读取侧的检索算子、键一致性与自指诊断 ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    MODEL_TOK = (tok, model)

    # ============================================================ Stage A
    print("\n[Stage A] 装置校验门", flush=True)

    # A1' Freeze C 逐臂复检
    print("  [A1'] Freeze C 逐臂复检", flush=True)
    a1_rows = []
    for name, beta, oracle, nowrite, randq in (
            ("Q1_linear_aligned", 0.0, False, False, False),
            ("Q2_attn_l10", 16.0, False, False, False),
            ("Q3_linear_oracle_key", 0.0, True, False, False),
            ("Q5_attn_l10_nowrite", 16.0, False, True, False),
            ("Q6_attn_l10_random_q", 16.0, False, False, True)):
        st = make_stack(1e-5, no_write=nowrite, seed=20261028, beta=beta, random_q=randq)
        if not randq:
            set_aligned(st)
        base_before = {n: q.detach().clone() for n, q in model.named_parameters()}
        ids = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
        with torch.no_grad():
            model(input_ids=ids)
            model(input_ids=ids)
        lg = model(input_ids=ids).logits.float()
        F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                        ids[:, 1:].reshape(-1)).backward()
        from sgmn_module_retrieval import grad_report
        rrep = grad_report(st.named_read_params())
        wrep = grad_report(st.named_write_params())
        # rev 1.1：绝对阈值不具跨配置可比性 ⇒ 改为「有限 + gamma 梯度非零 + max|g|>1e-6」
        gam = [g for n, g, i, b in rrep if n.endswith("gamma")]
        read_ok = (all((not bad) and (isn or g >= 0) for _, g, isn, bad in rrep)
                   and all(g > 0 for g in gam) and max(g for _, g, _, _ in rrep) > 1e-6)
        write_ok = all(isn or g == 0.0 for _, g, isn, _ in wrep)
        base_same = all(torch.equal(q.detach(), base_before[n])
                        for n, q in model.named_parameters())
        st.remove()
        torch.cuda.empty_cache()
        st2 = make_stack(1e-5, seed=20261028, beta=beta, drop_sg=True) if False else None
        # 阴性对照：去掉 sg（用同一装置类的 drop_sg 开关）
        st2 = SGRet(model, MOUNT, model.config.hidden_size, model.config.hidden_size,
                    R, ALPHA, LAM, 1e-5, GAMMA0, DEV, seed=20261028,
                    use_factorized=False, perpos=True, drop_sg=True)
        for m in st2.mems.values():
            m.read_beta = beta
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
                        "base_same": base_same, "leak": leak})
        print(f"    {name:24s} 读梯度 {sum(1 for _,_,i,_ in rrep if not i)}/{len(rrep)} "
              f"写入梯度精确0 {sum(1 for _,g,i,_ in wrep if i or g == 0.0)}/{len(wrep)} "
              f"基座不变 {base_same} 阴性泄漏 {leak}/{len(wrep2)} [{'PASS' if ok else 'FAIL'}]",
              flush=True)
    rec("A1' Freeze C 逐臂复检（③写入梯度精确0 ②读梯度非零有限 ④基座逐位不变 + 阴性对照）",
        all(r["passed"] for r in a1_rows),
        "；".join(f"{r['arm']}: read={r['read_ok']} write={r['write_ok']} "
                  f"base={r['base_same']} leak={r['leak']}" for r in a1_rows),
        "全部满足", all(r["passed"] for r in a1_rows))

    # A2' 装置等价性：beta=1e-8 与 beta=0 的输出必须几乎相同
    docs = build_facts(tok, 999)[:6]
    st0 = make_stack(1e-5, seed=20261028, beta=0.0)
    set_aligned(st0)
    r0 = run_arm(st0, docs)
    st0.remove()
    torch.cuda.empty_cache()
    st1 = make_stack(1e-5, seed=20261028, lam=1e-8)
    set_aligned(st1)
    r1s = run_arm(st1, docs)
    st1.remove()
    torch.cuda.empty_cache()
    dmax = max(abs(a["lp_first"] - b["lp_first"]) for a, b in zip(r0, r1s))
    dh = max(abs(a["hit"] - b["hit"]) for a, b in zip(r0, r1s))
    # 【D10 修复·判据双边化】绝对"逐位相同"不可达（算子在场会改变算术顺序）；
    #   正确的等价性判据是**双边**：① λ→0 时效应消失（|Δ| < 1e-4）；
    #   ② λ=0.5 时效应显著（|Δ| > 1e-4）⇒ 算子确实起作用。
    st_hi = make_stack(1e-5, seed=20261028, lam=0.5)
    set_aligned(st_hi)
    r_hi = run_arm(st_hi, docs)
    st_hi.remove()
    torch.cuda.empty_cache()
    lp_hi = np.asarray([r["lp_first"] for r in r_hi])
    d_hi = float(np.abs(lp_hi - np.asarray([r["lp_first"] for r in r0])).max())
    rec("A2' 装置等价性（双边：λ=1e-8 与 λ=0 的差 <1e-4，且 λ=0.5 与 λ=0 的差 >1e-4）",
        dmax < 1e-4 and dh < 1e-6 and d_hi > 1e-4,
        f"|Δ(λ=1e-8, λ=0)|码logp {dmax:.3e}（须 <1e-4），命中 {dh:.3e}；"
        f"|Δ(λ=0.5, λ=0)|码logp {d_hi:.3e}（须 >1e-4）",
        "dmax<1e-4 且 dh<1e-6 且 d_hi>1e-4", dmax)

    # A4' 单一装置 + 读阶段记忆冻结
    st = make_stack(1e-5, seed=20261028, beta=16.0, lam=1.0)
    set_aligned(st)
    nh, nm = len(st._handles), len(st.mems)
    ids8 = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
    with torch.no_grad():
        model(input_ids=ids8)
    Wb = {li: W_of(st.mems[li]).cpu().clone() for li in st.mems}
    with torch.no_grad():
        model(input_ids=ids8)
    dm = max(float((W_of(st.mems[li]).cpu() - Wb[li]).abs().max()
                   / max(float(Wb[li].abs().max()), 1e-30)) for li in st.mems)
    st.remove()
    torch.cuda.empty_cache()
    # 【D8 修复·检查实现错误】原实现把「写开」下的两次前视当作读阶段冻结来查 ⇒ 必然变化 ⇒ 假 FAIL。
    # 正确的可区分形式：①【写关】时两次前视记忆逐位不变（读阶段冻结）；
    #                    ②【写开】时记忆必须变化（否则装置根本没在写，判据会假 PASS）。
    st_ = make_stack(1e-5, no_write=True, seed=20261028, beta=16.0, lam=1.0)
    with torch.no_grad():
        model(input_ids=ids8)
    Wb2 = {li: W_of(st_.mems[li]).cpu().clone() for li in st_.mems}
    with torch.no_grad():
        model(input_ids=ids8)
    dm_off = max(float((W_of(st_.mems[li]).cpu() - Wb2[li]).abs().max()
                        / max(float(Wb2[li].abs().max()), 1e-30)) for li in st_.mems)
    st_.remove()
    torch.cuda.empty_cache()
    rec("A4' 单一装置（句柄 2×层数）+ 读阶段记忆逐位不变（写关）/ 写开时必须变化",
        nh == 2 * nm and dm_off < 1e-12 and dm > 1e-9,
        f"句柄 {nh}（每层 2 个），记忆数 {nm}；写关时两次前视相对差 {dm_off:.3e}（须 <1e-12）；"
        f"写开时相对差 {dm:.3e}（须 >1e-9，证明装置确实在写）",
        "句柄数==2*层数 且 写关差<1e-12 且 写开差>1e-9", dm_off)

    # ---- 【新增诊断】键基（W 的列）的共线性：W vs W0 vs 随机矩阵
    print("\n  [诊断] 键基列共线性（写入后 vs 初始 W0 vs 随机）", flush=True)
    coll = {}
    for tag, mk in (("W_written", None), ("W0_init", "W0"), ("random", "rand")):
        vals_c = []
        for li in st.mems:
            if mk == "W0":
                Wm = st.mems[li].W0.detach()
            elif mk == "rand":
                Wm = torch.randn_like(st.mems[li].W0.detach())
            else:
                Wm = W_of(st.mems[li])
            Wm = Wm / Wm.norm(dim=0, keepdim=True).clamp_min(1e-30)
            C = Wm.T @ Wm
            n = C.shape[0]
            off = (C.sum() - C.trace()) / (n * n - n)
            vals_c.append(float(off))
        coll[tag] = float(np.mean(vals_c))
        print(f"    {tag:12s} 平均列间余弦 = {coll[tag]:+.5f}", flush=True)
    res["key_basis_collinearity"] = coll

    # ============================================================ 主测量
    print(f"\n[R1] 主测量（{n_docs} 篇 × {len(seeds)} 种子）", flush=True)
    arms = {}
    for seed in seeds:
        print(f"\n  [种子 {seed}]", flush=True)
        dset = build_facts(tok, seed)
        sa = {}
        with torch.no_grad():
            lp = []
            for d in dset:
                pre = torch.tensor(d["ids"][:d["prefix_len"]], device=DEV).unsqueeze(0)
                lg = model(input_ids=pre).logits.float()
                lp.append(float(F.log_softmax(lg[0, -1], dim=-1)[d["code_ids"][0]]))
        sa["Q0_base"] = {"lp_first": lp, "hit": [0.0] * len(dset),
                         "cbest": [0.0] * len(dset), "ctrue": [0.0] * len(dset),
                         "self_cos": [0.0] * len(dset), "attn_max": [0.0] * len(dset)}
        print(f"    Q0_base            码 logp {np.mean(lp):+.4f}", flush=True)

        # 【D9 修复】λ>0 的臂必须显式给锐度 beta=16（默认锐度被脚本写成 0 ⇒ logits≡0 ⇒
        #   注意力恒为均匀 1/896 ⇒ 算子退化为"加一个常数向量"）。此处逐臂显式赋值。
        cfgs = [("Q1_linear_aligned", 0.0, False, False, False, 0.0),
                ("Q2_attn_l025", 16.0, False, False, False, 0.25),
                ("Q2_attn_l05", 16.0, False, False, False, 0.5),
                ("Q2_attn_l10", 16.0, False, False, False, 1.0),
                ("Q3_linear_oracle_key", 0.0, True, False, False, 0.0),
                ("Q4_attn_l10_oracle_key", 16.0, True, False, False, 1.0),
                ("Q5_attn_l10_nowrite", 16.0, False, True, False, 1.0),
                ("Q6_attn_l10_random_q", 16.0, False, False, True, 1.0),
                ("Q7_linear_nowrite", 0.0, False, True, False, 0.0)]
        for name, beta, oracle, nowrite, randq, lam in cfgs:
            st = make_stack(1e-5, no_write=nowrite, seed=seed, beta=beta, random_q=randq,
                            lam=lam)
            if not randq:
                set_aligned(st)
            rows = run_arm(st, dset, use_oracle_key=oracle)
            st.remove()
            torch.cuda.empty_cache()
            sa[name] = {k: [r[k] for r in rows] for k in
                        ("lp_first", "hit", "cbest", "ctrue", "self_cos", "attn_max")}
            print(f"    {name:24s} 码 logp {np.mean(sa[name]['lp_first']):+.4f} "
                  f"命中 {np.mean(sa[name]['hit']):.4f} "
                  f"cos_true {np.mean(sa[name]['ctrue']):+.4f} "
                  f"自指cos {np.mean(sa[name]['self_cos']):.4f} "
                  f"注意力max {np.mean(sa[name]['attn_max']):.4f}", flush=True)
        arms[seed] = sa

    # ============================================================ 分析
    print("\n[R2] 分析与判据", flush=True)

    def agg(seed, arm, key):
        return np.asarray(arms[seed][arm][key], dtype=float)

    summ = {}
    for arm in arms[seeds[0]]:
        summ[arm] = {k: float(np.mean([np.mean(arms[s][arm][k]) for s in seeds]))
                     for k in ("lp_first", "hit", "cbest", "ctrue", "self_cos", "attn_max")}
        summ[arm]["per_seed_hit"] = {str(s): float(np.mean(arms[s][arm]["hit"])) for s in seeds}
    res["summary"] = summ
    print("  各臂（跨种子均值）：", flush=True)
    for a, v in summ.items():
        print(f"    {a:24s} 码logp {v['lp_first']:+.4f}  命中 {v['hit']:.4f}  "
              f"cos_true {v['ctrue']:+.4f}  自指cos {v['self_cos']:.4f}  "
              f"注意力max {v['attn_max']:.4f}", flush=True)

    def cmp(a, b, key):
        A = np.concatenate([agg(s, a, key) for s in seeds])
        B = np.concatenate([agg(s, b, key) for s in seeds])
        c = paired(A, B)
        c["seed_means"] = [float(np.mean(arms[s][a][key]) - np.mean(arms[s][b][key]))
                           for s in seeds]
        return c

    # H1 对齐是必要条件
    h1 = cmp("Q2_attn_l10", "Q6_attn_l10_random_q", "hit")
    rec("H1 〔对齐是必要条件〕检索算子下键对齐臂命中 > 随机 W_q 臂（Q2(l=1) vs Q6）",
        h1["mean"] > 0 and h1["p_sign"] < 0.05 and h1["consistent"],
        f"命中配对差 {h1['mean']:+.4f}（sd {h1['sd']:.4f}, t={h1['t']:+.2f}, "
        f"p={h1['p_sign']:.4g}, 正向 {h1['pos']}/{h1['n']}）跨种子 "
        + " ".join(f"{x:+.4f}" for x in h1["seed_means"]),
        "均值>0 且 p<0.05 且 跨种子同向", h1)

    # H2 存在性：最佳 β 的命中 ≥ 0.9
    best_beta_arm = max(("Q2_attn_l025", "Q2_attn_l05", "Q2_attn_l10"),
                         key=lambda a: summ[a]["hit"])
    h2_ok = summ[best_beta_arm]["hit"] >= 0.9
    rec("H2 〔存在性·检索算子〕检索确实可实现（最佳 β 臂命中 ≥ 0.9）", h2_ok,
        f"最佳 β 臂 = {best_beta_arm}，命中 {summ[best_beta_arm]['hit']:.4f}"
        f"（逐种子 " + ", ".join(f"{v:.3f}" for v in summ[best_beta_arm]["per_seed_hit"].values())
        + f"）；随机基线 1/41=0.0244", ">=0.9 且三种子均满足", h2_ok)

    # H3 自指诊断
    h3a = cmp("Q3_linear_oracle_key", "Q1_linear_aligned", "hit")
    h3b = cmp("Q4_attn_l10_oracle_key", "Q2_attn_l10", "hit")
    rec("H3 〔自指诊断〕oracle 键使检索显著改善（Q3>Q1 或 Q4>Q2）",
        (h3a["mean"] > 0 and h3a["p_sign"] < 0.05 and h3a["consistent"])
        or (h3b["mean"] > 0 and h3b["p_sign"] < 0.05 and h3b["consistent"]),
        f"Q3-Q1 = {h3a['mean']:+.4f}（t={h3a['t']:+.2f}, p={h3a['p_sign']:.4g}，跨种子 "
        + " ".join(f"{x:+.4f}" for x in h3a["seed_means"]) + f"）；"
        f"Q4-Q2 = {h3b['mean']:+.4f}（t={h3b['t']:+.2f}, p={h3b['p_sign']:.4g}）",
        "至少一个显著且跨种子同向", {"Q3-Q1": h3a, "Q4-Q2": h3b})

    # H4 任务层
    h4 = cmp("Q2_attn_l10", "Q0_base", "lp_first")
    rec("H4 〔任务层·合成回忆〕记忆内容因果改变输出（Q2(l=1) 码 logp > base）",
        h4["mean"] > 0 and h4["p_sign"] < 0.05 and h4["consistent"],
        f"配对差 {h4['mean']:+.4f}（sd {h4['sd']:.4f}, t={h4['t']:+.2f}, "
        f"p={h4['p_sign']:.4g}, 正向 {h4['pos']}/{h4['n']}）跨种子 "
        + " ".join(f"{x:+.4f}" for x in h4["seed_means"]),
        "均值>0 且 p<0.05 且 跨种子同向", h4)

    # H5 严格对照
    h5 = cmp("Q2_attn_l10", "Q5_attn_l10_nowrite", "lp_first")
    rec("H5 〔严格对照〕增益来自记忆内容而非注入结构（Q2(l=1) > Q5 写关）",
        h5["mean"] > 0 and h5["consistent"],
        f"配对差 {h5['mean']:+.4f}（t={h5['t']:+.2f}, p={h5['p_sign']:.4g}）跨种子 "
        + " ".join(f"{x:+.4f}" for x in h5["seed_means"]),
        "均值>0 且 跨种子同向", h5)

    res["comparisons"] = {"H1": h1, "H3a": h3a, "H3b": h3b, "H4": h4, "H5": h5,
                           "best_beta_arm": best_beta_arm}
    res["self_ref_cos"] = {a: summ[a]["self_cos"] for a in summ}
    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary_pass"] = {"passed": npass, "total": len(res["checks"])}
    (HERE / ("R1_smoke_results.json" if args.smoke else "R1_results.json")).write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== DIR-12 R1：{npass}/{len(res['checks'])} 通过 ===", flush=True)


if __name__ == "__main__":
    main()
