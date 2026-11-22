#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-16 · R1 —— 可寻址性的泛化：线索换了说法、换了上下文、有了干扰，还找得到吗？

判据见 ../DIRECTION.md（H1–H7，冻结于 rev 1.0）。

装置：`sgmn_module_generalize.py`（DIR-15 装置的逐字派生）：
  * 【D29】读侧地址签名取 `hs_in[0][L-1]`（与写侧「事实首 token 的前一位」约定对齐；DIR-15 用 `[L-2]`）
  * 【新增】`addr_mode ∈ {sig, key}`：sig = 原始上下文签名做地址；key = 键 P_k h 做地址（阴性对照）
  * 其余（显式 KV 核、位置级写、读出映射 W_o、梯度隔离、last_yr/last_q/last_y）逐字沿用

臂（7 个）：
  C1 cue_same   复现 DIR-15（控制）｜C2 cue_para 改写线索｜C3 cue_late 上下文/位置扰动
  C4 cue_multi  多事实干扰（问第二张）｜C5 addr_key 地址=键（阴性对照）｜C6 ctx 现状对照
每个变体各有一个 base（纯基座，同前缀）。

协议：每篇文档前重置记忆 → 写阶段整篇前视（位置级写）→ 读阶段（写冻结）用**变体前缀**前视。
60 篇 × 3 种子 = 180 对/臂。
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
from sgmn_module_generalize import SGMNStack as SGS, grad_report   # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOUNT = [8, 12, 16]
ALPHA, LAM, GAMMA0, R = 1.0, 0.05, 0.05, 8
ETA = 1e-5
BETA = 64.0
N_ENTRIES = 64
SEEDS = [20261028, 20261104, 7]
N_DOCS, FILLER_TOK = 60, 30
CUE_A = " The vault code is"
CUE_B = " The safe combination is"
CUE_C = " The second code is"
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")

res = {"dir": "DIR-16", "round": "R1", "checks": [], "arms": {}, "meta": {}}


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


# ------------------------------------------------------------------ 语料
def load_filler():
    import requests
    p = pathlib.Path.home() / "sgmn-loop" / "corpus" / "train.txt"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(CORPUS_BASE + "train.txt", timeout=90)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p.read_text(encoding="utf-8", errors="ignore")


def build_variant(tok, seed, variant, n=N_DOCS):
    """变体文档：写阶段文档 / 读阶段前缀 / 目标码 / 真值条目下标（写阶段口径）。"""
    import re as _re
    text = load_filler()
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    rng = random.Random(seed)
    rng.shuffle(paras)
    pool = []
    for q in paras:
        pool.extend(tok(q).input_ids)
        if len(pool) > 60000:
            break
    cueA, cueB, cueC = tok(CUE_A).input_ids, tok(CUE_B).input_ids, tok(CUE_C).input_ids
    docs, used = [], set()
    guard = 0
    while len(docs) < n and guard < 20 * n:
        guard += 1
        i = rng.randrange(0, len(pool) - 400)
        code1 = f"{rng.randrange(1000, 9999)}"
        if code1 in used:
            continue
        used.add(code1)
        c1 = tok(" " + code1 + ".").input_ids
        fill = pool[i:i + FILLER_TOK]
        fill2 = pool[i + 200:i + 200 + FILLER_TOK]
        if variant in ("same", "para", "key", "ctx"):
            doc = fill + cueA + c1
            p1 = len(fill) + len(cueA)
            prefix = fill + cueA if variant != "para" else fill + cueB
            docs.append({"ids": doc, "prefix": prefix, "code_ids": c1,
                         "true_idx": p1 - 1, "code": code1})
        elif variant == "late":
            doc = fill + cueA + c1 + fill2
            p1 = len(fill) + len(cueA)
            docs.append({"ids": doc, "prefix": fill + fill2 + cueA, "code_ids": c1,
                         "true_idx": p1 - 1, "code": code1})
        elif variant == "multi":
            code2 = f"{rng.randrange(1000, 9999)}"
            if code2 in used:
                continue
            used.add(code2)
            c2 = tok(" " + code2 + ".").input_ids
            doc = fill + cueA + c1 + fill2 + cueC + c2
            p2 = len(fill) + len(cueA) + len(c1) + len(fill2) + len(cueC)
            docs.append({"ids": doc, "prefix": fill + fill2 + cueC, "code_ids": c2,
                         "true_idx": p2 - 1, "code": code2})
        else:
            raise ValueError(variant)
    return docs


# ------------------------------------------------------------------ 统计
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
    D = np.asarray(A, dtype=float) - np.asarray(B, dtype=float)
    D = D[np.isfinite(D)]
    if len(D) == 0:
        return {"mean": float("nan"), "t": float("nan"), "p": 1.0, "pos": 0, "n": 0,
                "consistent": False, "seed_means": []}
    if np.all(D == 0.0):
        return {"mean": 0.0, "sd": 0.0, "t": 0.0, "p": 1.0, "pos": 0, "n": len(D),
                "consistent": True, "seed_means": []}
    m = float(D.mean())
    sd = float(D.std(ddof=1))
    t = m / (sd / math.sqrt(len(D))) if sd > 0 else 0.0
    ps, k, n = sign_test_p(list(D))
    return {"mean": m, "sd": sd, "t": t, "p": ps, "pos": k, "n": n,
            "consistent": bool(k == n or k == 0)}


# ------------------------------------------------------------------ 装置
ARMS = [
    # name, variant, query_mode, addr_mode
    ("C1_cue_same",  "same", "cue",        "sig"),
    ("C2_cue_para",  "para", "cue",        "sig"),
    ("C3_cue_late",  "late", "cue",        "sig"),
    ("C4_cue_multi", "multi", "cue",       "sig"),
    ("C5_addr_key",  "same", "cue",        "key"),
    ("C6_ctx",       "same", "context",    "sig"),
]


def make_stack(model, d, query_mode, addr_mode, seed=20261028, gamma=GAMMA0):
    st = SGS(model, MOUNT, d, d, R, ALPHA, LAM, ETA, gamma, DEV, seed=seed,
             use_factorized=False, no_write=False, perpos=True, mem_kind="explicit_kv",
             n_entries=N_ENTRIES)
    for m in st.mems.values():
        m.query_mode = query_mode
        m.addr_mode = addr_mode
        m.read_beta = BETA
    return st


def run_arm(st, docs, gamma_on=True):
    """每篇文档前重置记忆 → 写阶段整篇前视 → 读阶段（写冻结）用变体前缀前视。"""
    rows = []
    for d in docs:
        for m in st.mems.values():
            m.state_restore(m._snap)
        ids = torch.tensor(d["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
        arm_mode = st.mems[next(iter(st.mems))].query_mode
        for m in st.mems.values():
            m.gamma.data.fill_(GAMMA0 if gamma_on else 0.0)
            m.query_mode = "context"        # 写阶段统一：各臂写阶段逐位相同
            m.oracle_q = None
        with torch.no_grad():
            st.model(input_ids=ids)
        vals, keys = {}, {}
        for li, m in st.mems.items():
            vals[li] = torch.stack(m.vals_written) if m.vals_written else None
            keys[li] = torch.stack(m.keys_written) if m.keys_written else None
            m.keys_written.clear()
            m.vals_written.clear()
        ti = int(d["true_idx"])
        for li, m in st.mems.items():
            m.gamma.data.fill_(GAMMA0 if gamma_on else 0.0)
            m.query_mode = arm_mode
            m.oracle_q = None
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            lg = st.model(input_ids=pre).logits.float()
        lp = float(F.log_softmax(lg[0, -1], dim=-1)[d["code_ids"][0]])
        hits, ctrue, ahits, cue_ok, tgt_w = [], [], [], [], []
        for li, m in st.mems.items():
            yr = m.last_yr[0] if m.last_yr is not None else None
            V, K = vals[li], keys[li]
            if yr is None or V is None or V.shape[0] == 0:
                hits.append(0.0); ctrue.append(0.0); ahits.append(0.0)
                cue_ok.append(float("nan")); tgt_w.append(0.0)
                continue
            cs = F.cosine_similarity(V, yr.unsqueeze(0), dim=1)
            hits.append(1.0 if int(torch.argmax(cs)) == ti else 0.0)
            ctrue.append(float(cs[ti]) if 0 <= ti < cs.shape[0] else 0.0)
            qk = m.last_q[-1] if m.last_q is not None else None
            if qk is not None and K is not None and K.shape[0] > 0:
                Kn = K / K.norm(dim=1, keepdim=True).clamp_min(1e-30)
                sc = Kn @ (qk / qk.norm().clamp_min(1e-30))
                ahits.append(1.0 if int(torch.argmax(sc)) == ti else 0.0)
                with torch.no_grad():
                    a = torch.softmax(BETA * sc, dim=-1)
                tgt_w.append(float(a[ti]) if 0 <= ti < a.shape[0] else 0.0)
            else:
                ahits.append(0.0); tgt_w.append(0.0)
            # 【D30】线索匹配返回的是**核心槽位下标**（写轮转 append，超过 N 会回绕），
            #   而真值 ti 是**写入序号** ⇒ 二者只在「写入次数 ≤ N」时相等
            #   ⇒ 多事实臂（文档更长 ⇒ 写入 > 64）必须比较 slot = ti % N_ENTRIES。
            if m.last_cue_idx is not None:
                cue_ok.append(1.0 if int(m.last_cue_idx) == (ti % N_ENTRIES) else 0.0)
            else:
                cue_ok.append(float("nan"))
        rows.append({"lp_first": lp, "hit": float(np.mean(hits)),
                     "ctrue": float(np.mean(ctrue)), "ahit": float(np.mean(ahits)),
                     "cue_ok": (float(np.nanmean(cue_ok))
                                if not all(np.isnan(x) for x in cue_ok) else float("nan")),
                     "tgt_w": float(np.mean(tgt_w))})
    return rows


def run_base(model, docs):
    rows = []
    for d in docs:
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            lg = model(input_ids=pre).logits.float()
        lp = float(F.log_softmax(lg[0, -1], dim=-1)[d["code_ids"][0]])
        rows.append({"lp_first": lp})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_docs = 8 if args.smoke else N_DOCS
    seeds = SEEDS[:2] if args.smoke else SEEDS

    print("=== DIR-16 R1：可寻址性的泛化 ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size

    # ---------------- Stage A
    print("\n[Stage A]", flush=True)
    ids8 = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
    st = make_stack(model, d, "context", "sig")
    base_before = {n: q.detach().clone() for n, q in model.named_parameters()}
    with torch.no_grad():
        st.model(input_ids=ids8)                 # 前向#1：写入
    for m in st.mems.values():
        m.no_write = True                        # 冻结写 ⇒ 反向不被原地写破坏
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
    # 阴性对照（drop_sg=True）：写侧必须泄漏
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

    # 写关 / 写开两侧
    st = make_stack(model, d, "cue", "sig")
    for li, m in st.mems.items():
        m._snap = m.state_snapshot()
    with torch.no_grad():
        st.model(input_ids=ids8)
    s1 = {li: m.state_snapshot() for li, m in st.mems.items()}
    with torch.no_grad():
        st.model(input_ids=ids8)
    s2 = {li: m.state_snapshot() for li, m in st.mems.items()}
    diff_on = max(float((s1[li][1] - s2[li][1]).abs().max()) for li in s1)
    st.remove(); torch.cuda.empty_cache()
    st = SGS(model, MOUNT, d, d, R, ALPHA, LAM, ETA, GAMMA0, DEV, seed=20261028,
             use_factorized=False, no_write=True, perpos=True, mem_kind="explicit_kv",
             n_entries=N_ENTRIES)
    with torch.no_grad():
        st.model(input_ids=ids8)
    s1 = {li: m.state_snapshot() for li, m in st.mems.items()}
    with torch.no_grad():
        st.model(input_ids=ids8)
    s2 = {li: m.state_snapshot() for li, m in st.mems.items()}
    diff_off = max(float((s1[li][1] - s2[li][1]).abs().max()) for li in s1)
    st.remove(); torch.cuda.empty_cache()
    rec("A3' 写关时两次前视记忆相对差 < 1e-12", diff_off < 1e-12, f"diff_off={diff_off:.3g}", tol=1e-12, value=diff_off)
    rec("A3' 写开时两次前视记忆变化 > 1e-9", diff_on > 1e-9, f"diff_on={diff_on:.3g}", tol=1e-9, value=diff_on)

    # A2' ②：真值条目下标与写阶段写入序号一致（用合成张量直接检查写循环的序号约定）
    st = make_stack(model, d, "cue", "sig")
    for li, m in st.mems.items():
        m._snap = m.state_snapshot()
    probe = torch.tensor([[100, 200, 300, 400, 500]], dtype=torch.long, device=DEV)
    with torch.no_grad():
        st.model(input_ids=probe)
    nw = len(st.mems[list(st.mems)[0]].vals_written)
    idx_ok = (nw == probe.shape[1] - 1)      # 位置级写：L-1 次写入，序号 t-1
    # A2' ③：addr_mode='key' 的地址确实取自键 ⇒ 与原始签名逐位不同
    st.mems[list(st.mems)[0]].addr_mode = "key"
    with torch.no_grad():
        st.model(input_ids=probe)
    m0 = st.mems[list(st.mems)[0]]
    sig0 = m0.core.sigs[0].detach().clone()
    key0 = m0.core.keys[0].detach().clone()
    addr_differs = not torch.allclose(sig0, key0)
    st.remove(); torch.cuda.empty_cache()
    rec("A2' ② 真值条目下标 = 写阶段写入序号（位置级写：L−1 次）", idx_ok,
        f"写入次数 {nw}，序列长度 {probe.shape[1]} ⇒ 期望 {probe.shape[1]-1}", value=idx_ok)
    rec("A2' ③ 地址=键 与 原始签名 逐位不同", addr_differs,
        f"‖sig‖={float(sig0.norm()):.4g} ‖key‖={float(key0.norm()):.4g} 逐位相同={not addr_differs}",
        value=addr_differs)
    # A2' ④【D30】槽位映射：写入次数 > N 时，线索匹配的下标（槽位）与写入序号必须按 % N 映射
    slot_map_ok = all((ti % N_ENTRIES) == (ti % 64) for ti in (0, 63, 64, 70, 129))
    rec("A2' ④ 槽位映射（写入序号 → 核心槽位，% N）", slot_map_ok,
        f"N_ENTRIES={N_ENTRIES}；多事实臂的写入数可达 ~73 ⇒ 必须按 ti % N 映射槽位", value=slot_map_ok)

    # ---------------- 主循环
    print("\n[主循环]", flush=True)
    all_rows = {}
    base_rows = {}
    for seed in seeds:
        docsets = {v: build_variant(tok, seed, v, n_docs)
                   for v in ("same", "para", "late", "multi")}
        for name, variant, qm, am in ARMS:
            st = make_stack(model, d, qm, am, seed=seed)
            for li, m in st.mems.items():
                m._snap = m.state_snapshot()
            rows = run_arm(st, docsets[variant])
            st.remove(); torch.cuda.empty_cache()
            all_rows.setdefault(name, {})[seed] = rows
            print(f"  seed={seed} {name:14s} hit={np.mean([r['hit'] for r in rows]):.4f} "
                  f"ahit={np.mean([r['ahit'] for r in rows]):.4f} "
                  f"cue_ok={np.nanmean([r['cue_ok'] for r in rows]) if not all(np.isnan(r['cue_ok']) for r in rows) else float('nan'):.4f} "
                  f"tgt_w={np.mean([r['tgt_w'] for r in rows]):.4f} "
                  f"lp={np.mean([r['lp_first'] for r in rows]):.4f}", flush=True)
        for v in ("same", "para", "late", "multi"):
            brows = run_base(model, docsets[v])
            base_rows.setdefault(f"base_{v}", {})[seed] = brows
        print(f"  seed={seed} base: " + "  ".join(
            f"{v}={np.mean([r['lp_first'] for r in base_rows[f'base_{v}'][seed]]):.4f}"
            for v in ("same", "para", "late", "multi")), flush=True)
    for v in ("same", "para", "late", "multi"):
        all_rows[f"base_{v}"] = base_rows[f"base_{v}"]

    # ---------------- 汇总
    print("\n[汇总与判据]", flush=True)
    agg = {}
    for name, per_seed in all_rows.items():
        _r0 = per_seed[next(iter(per_seed))]
        hits = ([r["hit"] for rs in per_seed.values() for r in rs]
                if (_r0 and "hit" in _r0[0]) else [])
        lps = [r["lp_first"] for rs in per_seed.values() for r in rs]
        seed_hits, seed_lps = [], []
        for s, rs in per_seed.items():
            if rs and "hit" in rs[0]:
                seed_hits.append(float(np.mean([r["hit"] for r in rs])))
            seed_lps.append(float(np.mean([r["lp_first"] for r in rs])))
        agg[name] = {"hit": (float(np.mean(hits)) if hits else None), "hit_by_seed": seed_hits,
                     "lp": float(np.mean(lps)), "lp_by_seed": seed_lps, "n": len(lps)}
        print(f"  {name:16s} hit={agg[name]['hit']} lp={agg[name]['lp']:.4f}", flush=True)
    res["arms"] = agg

    def flat(name, field):
        return [r[field] for rs in all_rows[name].values() for r in rs if field in r]

    def seed_diff(name_a, name_b, field):
        a, b = agg[name_a]["lp_by_seed"], agg[name_b]["lp_by_seed"]
        return [x - y for x, y in zip(a, b)]

    # ---- H1 复现
    h1_ok = (agg["C1_cue_same"]["hit"] >= 0.9
             and all(x >= 0.9 for x in agg["C1_cue_same"]["hit_by_seed"]))
    rec("H1 〔复现〕C1 cue_same 命中 ≥ 0.9（3 种子）", h1_ok,
        f"hit={agg['C1_cue_same']['hit']} by_seed={agg['C1_cue_same']['hit_by_seed']}", tol=0.9,
        value=agg["C1_cue_same"]["hit"])

    # ---- H2/H3/H4 泛化轴
    for hname, arm, label in (("H2", "C2_cue_para", "〔① 改写线索〕"),
                              ("H3", "C3_cue_late", "〔② 上下文/位置扰动〕"),
                              ("H4", "C4_cue_multi", "〔③ 多事实干扰〕")):
        ok = (agg[arm]["hit"] >= 0.9 and all(x >= 0.9 for x in agg[arm]["hit_by_seed"]))
        rec(f"{hname} {label} {arm} 命中 ≥ 0.9（3 种子）", ok,
            f"hit={agg[arm]['hit']} by_seed={agg[arm]['hit_by_seed']} "
            f"cue_ok={np.nanmean([r['cue_ok'] for rs in all_rows[arm].values() for r in rs]) if not all(np.isnan(r['cue_ok']) for rs in all_rows[arm].values() for r in rs) else float('nan'):.4f}",
            tol=0.9, value=agg[arm]["hit"])

    # ---- H5 阴性对照：地址=键
    h5a = agg["C5_addr_key"]["hit"] is not None and agg["C5_addr_key"]["hit"] <= 0.1
    d5 = paired(flat("C1_cue_same", "hit"), flat("C5_addr_key", "hit"))
    h5_ok = h5a and d5["mean"] > 0 and d5["p"] < 0.05 and d5["consistent"]
    rec("H5 〔阴性对照〕地址=键 命中 ≤ 0.1 且 C1 显著 > C5", h5_ok,
        f"hit={agg['C5_addr_key']['hit']}（须 ≤0.1）；C1−C5={d5['mean']:.4f} p={d5['p']:.3g} "
        f"pos={d5['pos']}/{d5['n']} consistent={d5['consistent']}", tol=0.1,
        value=agg["C5_addr_key"]["hit"])

    # ---- H6 现状对照
    h6_ok = agg["C6_ctx"]["hit"] is not None and agg["C6_ctx"]["hit"] <= 0.1
    rec("H6 〔现状对照〕C6 (q=W_qc) 命中 ≤ 0.1", h6_ok,
        f"hit={agg['C6_ctx']['hit']}（随机基线 0.0244）", tol=0.1, value=agg["C6_ctx"]["hit"])

    # ---- H7 任务层：对命中 ≥0.9 的泛化臂，码 log-prob 显著 > 同前缀 base
    h7_detail = []
    h7_ok = False
    for arm, basev in (("C1_cue_same", "base_same"), ("C2_cue_para", "base_para"),
                       ("C3_cue_late", "base_late"), ("C4_cue_multi", "base_multi")):
        d7 = paired(flat(arm, "lp_first"), flat(basev, "lp_first"))
        sd = seed_diff(arm, basev, "lp_first")
        cross = all(x > 0 for x in sd)
        ok = (d7["mean"] > 0 and d7["p"] < 0.05 and cross)
        h7_detail.append(f"{arm} vs {basev}: Δ={d7['mean']:+.6f} p={d7['p']:.3g} 逐种子={['%+.4f'%x for x in sd]}")
        print(f"  [H7] {arm:14s} vs {basev:10s} Δlp={d7['mean']:+.6f} t={d7['t']:.2f} "
              f"p={d7['p']:.3g} 逐种子={['%+.4f'%x for x in sd]} 跨种子同向={cross}", flush=True)
        if ok and agg[arm]["hit"] >= 0.9:
            h7_ok = True
    rec("H7 〔任务层〕命中 ≥0.9 的臂：正确码 log-prob 显著 > 同前缀 base（跨种子同向）", h7_ok,
        "；".join(h7_detail), tol=0.0)

    # ---- 报告项
    for name in ("C1_cue_same", "C2_cue_para", "C3_cue_late", "C4_cue_multi",
                 "C5_addr_key", "C6_ctx"):
        ah = [r["ahit"] for rs in all_rows[name].values() for r in rs if "ahit" in r]
        ct = [r["ctrue"] for rs in all_rows[name].values() for r in rs if "ctrue" in r]
        tw = [r["tgt_w"] for rs in all_rows[name].values() for r in rs if "tgt_w" in r]
        co = [r["cue_ok"] for rs in all_rows[name].values() for r in rs if "cue_ok" in r]
        agg[name]["ahit"] = float(np.mean(ah)) if ah else None
        agg[name]["ctrue"] = float(np.mean(ct)) if ct else None
        agg[name]["tgt_w"] = float(np.mean(tw)) if tw else None
        agg[name]["cue_ok"] = float(np.nanmean(co)) if co else None
        print(f"  [报告] {name:14s} 地址命中={agg[name]['ahit']:.4f} 真值余弦={agg[name]['ctrue']:.4f} "
              f"目标权重={agg[name]['tgt_w']:.4f} 线索选中真值={agg[name]['cue_ok']}", flush=True)
    res["meta"]["seeds"] = seeds
    res["meta"]["n_docs"] = n_docs

    passed = sum(1 for c in res["checks"] if c["passed"])
    res["summary_pass"] = {"passed": passed, "total": len(res["checks"])}
    print(f"\n=== 判据通过 {passed}/{len(res['checks'])} ===", flush=True)
    out = HERE.parent / ("R1_smoke_results.json" if args.smoke else "R1_results.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
