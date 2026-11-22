#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-15 · R1 —— 内容寻址式的读取：查询能否由「要取回的内容」构造？

判据见 ../DIRECTION.md（H1–H6，冻结于 rev 1.0）。

装置：`sgmn_module_cue.py`（DIR-14 装置的逐字派生，base md5 = 51e870607c85c6f026cc9de239a55da7）：
  * 显式键-值分离核（写**完全正确**：逐条存 (k,v)）
  * 每条关联额外存**上下文签名** h_{t-1}（与键 k 的同一输入）
  * `query_mode ∈ {context, oracle, cue, misaddress}`

臂（唯一变量 = 查询怎么来）：
  B0 base         无 SGMN
  B1 ctx          q = W_q · c（v2.4 现状）                    —— 对照（应失败）
  B2 oracle       q = 写阶段记录的目标键                        —— 判定性上界（诊断，非机制）
  B3 cue          q = 线索匹配（上下文签名余弦）选中的条目的键     —— 可实现的正面构造
  B4 cue_nogamma  同 B3，但【全臂 γ=0】（写/读都不注入）           —— 测 §6.1 耦合是否破坏寻址
  B5 misaddr      q = 线索匹配结果的邻居条目的键                  —— 严格对照

协议（与 DIR-11/12/13/14 完全相同）：
  每篇文档前重置记忆 → 写阶段整篇前视（位置级写）→ 读阶段（写冻结）截断到码之前。
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
from sgmn_module_cue import SGMNStack as SGS, grad_report   # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOUNT = [8, 12, 16]
ALPHA, LAM, GAMMA0, R = 1.0, 0.05, 0.05, 8
ETA = 1e-5
BETA = 64.0                 # 地址 softmax 锐度（显式 KV 读）
N_ENTRIES = 64
SEEDS = [20261028, 20261104, 7]
N_DOCS, FILLER_TOK = 60, 30
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")

res = {"dir": "DIR-15", "round": "R1", "checks": [], "arms": {}, "meta": {}}


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


def build_facts(tok, seed, n=N_DOCS):
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
    while len(docs) < n:
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
        docs.append({"ids": fill + full_ids, "code_ids": full_ids[len(pre_ids):],
                     "prefix_len": len(fill) + len(pre_ids)})
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
    if np.all(D == 0.0):          # 【D26】两臂逐位相同 ⇒ 差为 0（不是 nan）
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
    # name, query_mode, gamma_on, gamma_scale
    ("B1_ctx",         "context",    True,  1.0),
    ("B2_oracle",      "oracle",     True,  1.0),
    ("B3_cue",         "cue",        True,  1.0),
    ("B4_cue_nogamma", "cue",        False, 1.0),
    ("B5_misaddr",     "misaddress", True,  1.0),
    # 【rev 1.1·H3】γ 扫描：注入增益是任务层效应的量纲因子（v2.4 §4：γ 是可训练参数）
    ("B6_cue_g05",     "cue",        True,  10.0),
    ("B7_cue_g20",     "cue",        True,  40.0),
    # 【rev 1.2】H3 的内容对照臂：同 γ、同装置、同注入强度，只把「取回的内容」换成邻居的内容
    ("B8_misaddr_g05", "misaddress", True,  10.0),
    ("B9_misaddr_g20", "misaddress", True,  40.0),
]


def make_stack(model, d, query_mode, seed=20261028, gamma=GAMMA0, gscale=1.0):
    st = SGS(model, MOUNT, d, d, R, ALPHA, LAM, ETA, gamma * gscale, DEV, seed=seed,
             use_factorized=False, no_write=False, perpos=True, mem_kind="explicit_kv",
             n_entries=N_ENTRIES)
    for m in st.mems.values():
        m.query_mode = query_mode
        m.read_beta = BETA
        m._gscale = gscale
    return st


def run_arm(st, docs, gamma_on=True):
    """每篇文档前重置记忆 → 写阶段整篇前视 → 读阶段（写冻结）截断到码之前。"""
    rows = []
    for d in docs:
        for m in st.mems.values():
            m.state_restore(m._snap)
        ids = torch.tensor(d["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
        # ---- 写阶段（γ 按臂设定；**查询方式固定为 context** ⇒ 各臂的写阶段逐位相同）
        arm_mode = st.mems[next(iter(st.mems))].query_mode
        gval = GAMMA0 * getattr(st.mems[next(iter(st.mems))], "_gscale", 1.0)
        for m in st.mems.values():
            m.gamma.data.fill_(gval if gamma_on else 0.0)
            m.query_mode = "context"
            m.oracle_q = None
        with torch.no_grad():
            st.model(input_ids=ids)
        vals, keys = {}, {}
        for li, m in st.mems.items():
            vals[li] = torch.stack(m.vals_written) if m.vals_written else None
            keys[li] = torch.stack(m.keys_written) if m.keys_written else None
            m.keys_written.clear()
            m.vals_written.clear()
        # ---- 读阶段（写冻结；γ 恢复）
        tp = d["prefix_len"] - 1
        ti = tp - 1                                   # 真值条目在 vals_written 里的下标
        for li, m in st.mems.items():
            m.gamma.data.fill_(gval if gamma_on else 0.0)
            m.query_mode = arm_mode                     # 读阶段才切到该臂的查询方式
            if arm_mode == "oracle":
                m.oracle_q = keys[li][ti].detach().clone() if keys[li] is not None else None
            else:
                m.oracle_q = None
        pre = ids[:, :d["prefix_len"]]
        with torch.no_grad():
            lg = st.model(input_ids=pre).logits.float()
        lp = float(F.log_softmax(lg[0, -1], dim=-1)[d["code_ids"][0]])
        hits, hits_inj, ctrue, ahits, cue_ok, tgt_w, oracle_exact = [], [], [], [], [], [], []
        for li, m in st.mems.items():
            # 【D22】命中指标的对象必须是**记忆读出 y**（与写入值同空间），不是注入量 Wo·y
            yr = m.last_yr[0] if m.last_yr is not None else None
            y = m.last_y[0] if m.last_y is not None else None
            V, K = vals[li], keys[li]
            if yr is None or V is None or V.shape[0] == 0:
                hits.append(0.0); hits_inj.append(0.0); ctrue.append(0.0); ahits.append(0.0)
                cue_ok.append(0.0); tgt_w.append(0.0); oracle_exact.append(0.0)
                continue
            cs = F.cosine_similarity(V, yr.unsqueeze(0), dim=1)
            hits.append(1.0 if int(torch.argmax(cs)) == ti else 0.0)
            if y is not None:
                csi = F.cosine_similarity(V, y.unsqueeze(0), dim=1)
                hits_inj.append(1.0 if int(torch.argmax(csi)) == ti else 0.0)
            else:
                hits_inj.append(0.0)
            ctrue.append(float(cs[ti]) if 0 <= ti < cs.shape[0] else 0.0)
            # 地址命中：query 键在实际写入键中的 argmax（纯寻址，与写的内容无关）
            qk = m.last_q[-1] if m.last_q is not None else None
            if qk is not None and K is not None and K.shape[0] > 0:
                Kn = K / K.norm(dim=1, keepdim=True).clamp_min(1e-30)
                sc = Kn @ (qk / qk.norm().clamp_min(1e-30))
                ahits.append(1.0 if int(torch.argmax(sc)) == ti else 0.0)
                # 目标条目在读 softmax 里的权重
                with torch.no_grad():
                    a = torch.softmax(BETA * sc, dim=-1)
                tgt_w.append(float(a[ti]) if 0 <= ti < a.shape[0] else 0.0)
            else:
                ahits.append(0.0); tgt_w.append(0.0)
            # 线索匹配是否选中真值条目
            if m.last_cue_idx is not None:
                cue_ok.append(1.0 if int(m.last_cue_idx) == ti else 0.0)
            else:
                cue_ok.append(float("nan"))
            # oracle 模式：查询键是否与写阶段记录的目标键逐位相等
            if m.query_mode == "oracle" and K is not None and qk is not None:
                oracle_exact.append(1.0 if torch.equal(qk, K[ti]) else 0.0)
            else:
                oracle_exact.append(float("nan"))
        rows.append({"lp_first": lp, "hit": float(np.mean(hits)),
                     "hit_inj": float(np.mean(hits_inj)),
                     "ctrue": float(np.mean(ctrue)), "ahit": float(np.mean(ahits)),
                     "cue_ok": float(np.nanmean(cue_ok)) if not all(
                         np.isnan(x) for x in cue_ok) else float("nan"),
                     "tgt_w": float(np.mean(tgt_w)),
                     "oracle_exact": float(np.nanmean(oracle_exact)) if not all(
                         np.isnan(x) for x in oracle_exact) else float("nan")})
    return rows


def run_base(model, docs):
    rows = []
    for d in docs:
        ids = torch.tensor(d["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
        pre = ids[:, :d["prefix_len"]]
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

    print("=== DIR-15 R1：内容寻址式的读取 ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size

    # ---------------- Stage A
    print("\n[Stage A]", flush=True)
    # ---- A1' Freeze C
    # 【D21·装置】Stage A 必须用**正式轮实际使用的那个核心**（显式 KV），且反向必须跨过一次
    #   「已写入」的前向 ⇒ 写成：① 前向#1（no_grad，写入）② 冻结写（no_write=True）后前向#2（带反向）
    #   ⇒ 避免「写侧原地更新」破坏计算图（version counter）。
    ids8 = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
    st = make_stack(model, d, "context")
    base_before = {n: q.detach().clone() for n, q in model.named_parameters()}
    with torch.no_grad():                      # 前向#1：写入
        st.model(input_ids=ids8)
    for m in st.mems.values():
        m.no_write = True                      # 冻结写 ⇒ 反向不被原地写破坏
    lg = model(input_ids=ids8).logits.float()
    F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids8[:, 1:].reshape(-1)).backward()
    rrep = grad_report(st.named_read_params())
    wrep = grad_report(st.named_write_params())
    gam = [g for n, g, i, b in rrep if n.endswith("gamma")]
    read_ok = (all((not b) for _, _, _, b in rrep) and all(g > 0 for g in gam)
               and max(g for _, g, _, _ in rrep) > 1e-6)
    write_ok = all(isn or g == 0.0 for _, g, isn, _ in wrep)
    base_same = all(torch.equal(q.detach(), base_before[n]) for n, q in model.named_parameters())
    nh, nm = len(st._handles), len(st.mems)
    st.remove(); torch.cuda.empty_cache()
    # 阴性对照（drop_sg=True）：写侧必须**泄漏**梯度 ⇒ 否则「写梯度恒 0」没有区分力（DIR-14 D18）
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
    leak = len([1 for _, g, isn, _ in wrep2 if not isn and g > 0])
    st2.remove(); torch.cuda.empty_cache()
    for q in model.parameters():
        q.grad = None
    rec("A1' Freeze C（读梯度非零 / 写梯度精确 0 / 基座逐位不变 / 阴性对照泄漏）",
        read_ok and write_ok and base_same and leak > 0,
        f"读梯度非零 {sum(1 for _,_,i,_ in rrep if not i)}/{len(rrep)}；"
        f"写梯度精确0 {sum(1 for _,g,i,_ in wrep if i or g==0.0)}/{len(wrep)}；"
        f"基座逐位不变 {base_same}；阴性对照泄漏 {leak}/{len(wrep2)}",
        "读梯度>0 且 写梯度==0 且 基座不变 且 泄漏>0",
        {"read": read_ok, "write": write_ok, "base": base_same, "leak": leak})
    rec("A1' 句柄数 = 2×层数", nh == 2 * len(MOUNT), f"handles={nh} mems={nm}", tol=2 * len(MOUNT), value=nh)

    # ---- A3' 写关/写开两侧的「两次前视记忆不变/变化」
    st = make_stack(model, d, "context")
    for li, m in st.mems.items():
        m._snap = m.state_snapshot()
    idsa = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
    with torch.no_grad():
        st.model(input_ids=idsa)
    s1 = {li: m.state_snapshot() for li, m in st.mems.items()}
    with torch.no_grad():
        st.model(input_ids=idsa)
    s2 = {li: m.state_snapshot() for li, m in st.mems.items()}
    diff_on = max(float((s1[li][1] - s2[li][1]).abs().max()) if isinstance(s1[li], tuple)
                  else float((s1[li] - s2[li]).abs().max()) for li in s1)
    st.remove(); torch.cuda.empty_cache()
    # 写关侧
    st = SGS(model, MOUNT, d, d, R, ALPHA, LAM, ETA, GAMMA0, DEV, seed=20261028,
             use_factorized=False, no_write=True, perpos=True, mem_kind="explicit_kv",
             n_entries=N_ENTRIES)
    for li, m in st.mems.items():
        m._snap = m.state_snapshot()
    with torch.no_grad():
        st.model(input_ids=idsa)
    s1 = {li: m.state_snapshot() for li, m in st.mems.items()}
    with torch.no_grad():
        st.model(input_ids=idsa)
    s2 = {li: m.state_snapshot() for li, m in st.mems.items()}
    diff_off = max(float((s1[li][1] - s2[li][1]).abs().max()) if isinstance(s1[li], tuple)
                   else float((s1[li] - s2[li]).abs().max()) for li in s1)
    st.remove(); torch.cuda.empty_cache()
    rec("A3' 写关时两次前视记忆相对差 < 1e-12", diff_off < 1e-12, f"diff_off={diff_off:.3g}", tol=1e-12, value=diff_off)
    rec("A3' 写开时两次前视记忆变化 > 1e-9", diff_on > 1e-9, f"diff_on={diff_on:.3g}", tol=1e-9, value=diff_on)

    # ---------------- 主循环
    print("\n[主循环]", flush=True)
    all_rows = {}
    for seed in seeds:
        docs = build_facts(tok, seed, n_docs)
        for name, qm, gamma_on, gscale in ARMS:
            st = make_stack(model, d, qm, seed=seed, gscale=gscale)
            for li, m in st.mems.items():
                m._snap = m.state_snapshot()
            rows = run_arm(st, docs, gamma_on=gamma_on)
            st.remove(); torch.cuda.empty_cache()
            all_rows.setdefault(name, {})[seed] = rows
            print(f"  seed={seed} {name:18s} hit={np.mean([r['hit'] for r in rows]):.4f} "
                  f"hit_inj={np.mean([r['hit_inj'] for r in rows]):.4f} "
                  f"ahit={np.mean([r['ahit'] for r in rows]):.4f} "
                  f"cue_ok={np.nanmean([r['cue_ok'] for r in rows]) if not all(np.isnan(r['cue_ok']) for r in rows) else float('nan'):.4f} "
                  f"tgt_w={np.mean([r['tgt_w'] for r in rows]):.4f} "
                  f"lp={np.mean([r['lp_first'] for r in rows]):.4f}", flush=True)
        brows = run_base(model, docs)
        all_rows.setdefault("B0_base", {})[seed] = brows
        print(f"  seed={seed} {'B0_base':18s} lp={np.mean([r['lp_first'] for r in brows]):.4f}", flush=True)

    # ---------------- 汇总
    print("\n[汇总与判据]", flush=True)
    agg = {}
    for name, per_seed in all_rows.items():
        _rows0 = per_seed[next(iter(per_seed))]
        hits = ([r["hit"] for rs in per_seed.values() for r in rs]
                if (_rows0 and "hit" in _rows0[0]) else [])
        lps = [r["lp_first"] for rs in per_seed.values() for r in rs]
        seed_hits, seed_lps = [], []
        for s, rs in per_seed.items():
            if rs and "hit" in rs[0]:
                seed_hits.append(float(np.mean([r["hit"] for r in rs])))
            seed_lps.append(float(np.mean([r["lp_first"] for r in rs])))
        agg[name] = {"hit": float(np.mean(hits)) if hits else None,
                     "hit_by_seed": seed_hits,
                     "lp": float(np.mean(lps)), "lp_by_seed": seed_lps,
                     "n": len(lps)}
        print(f"  {name:18s} hit={agg[name]['hit']}  lp={agg[name]['lp']:.4f}", flush=True)
    res["arms"] = agg

    def flat(name, field):
        return [r[field] for rs in all_rows[name].values() for r in rs if field in r]

    # ---- H1：oracle 臂命中 ≥ 0.9
    h1_ok = (agg["B2_oracle"]["hit"] is not None and agg["B2_oracle"]["hit"] >= 0.9
             and all(x >= 0.9 for x in agg["B2_oracle"]["hit_by_seed"]))
    oe = [x for x in flat("B2_oracle", "oracle_exact") if not math.isnan(x)]
    oe_ok = len(oe) > 0 and all(x == 1.0 for x in oe)
    rec("H1 〔判定性上界〕oracle 键命中 ≥ 0.9（3 种子）", h1_ok,
        f"hit={agg['B2_oracle']['hit']} by_seed={agg['B2_oracle']['hit_by_seed']} "
        f"oracle_key_bitwise_exact={oe_ok}", tol=0.9, value=agg["B2_oracle"]["hit"])
    res["meta"]["oracle_bitwise_exact"] = oe_ok

    # ---- H2：cue 臂命中 ≥ 0.9
    h2_ok = (agg["B3_cue"]["hit"] is not None and agg["B3_cue"]["hit"] >= 0.9
             and all(x >= 0.9 for x in agg["B3_cue"]["hit_by_seed"]))
    rec("H2 〔可实现的正面构造〕cue 线索寻址命中 ≥ 0.9（3 种子）", h2_ok,
        f"hit={agg['B3_cue']['hit']} by_seed={agg['B3_cue']['hit_by_seed']}", tol=0.9,
        value=agg["B3_cue"]["hit"])

    # ---- H3（rev 1.2）：**内容对照** —— 同 γ、同装置、同注入强度下，
    #      「取回真值内容」臂的正确码 log-prob 显著 > 「取回邻居内容」臂。
    #      （rev 1.1 的「与 base 比」把「有无记忆」与「内容对不对」混在一起 ⇒ 撤销）
    h3_scan = {}
    h3_ok = False
    for cg, mg in (("B3_cue", "B5_misaddr"), ("B6_cue_g05", "B8_misaddr_g05"),
                   ("B7_cue_g20", "B9_misaddr_g20")):
        d3 = paired(flat(cg, "lp_first"), flat(mg, "lp_first"))
        dbase = paired(flat(cg, "lp_first"), flat("B0_base", "lp_first"))
        h3_scan[cg] = {"content": d3, "vs_base": dbase}
        ok3 = (d3["mean"] > 0 and d3["p"] < 0.05 and d3["consistent"])
        print(f"  [H3 内容对照] {cg:14s} vs {mg:16s} Δlp={d3['mean']:+.6f} t={d3['t']:.2f} "
              f"p={d3['p']:.3g} pos={d3['pos']}/{d3['n']} consistent={d3['consistent']}"
              f" ｜ vs base Δ={dbase['mean']:+.6f}", flush=True)
        if ok3:
            h3_ok = True
    rec("H3 〔任务层·内容对照〕存在 γ 使 cue 臂正确码 log-prob 显著 > misaddr 臂", h3_ok,
        "；".join(f"{k}: Δ内容={v['content']['mean']:+.6f} p={v['content']['p']:.3g} "
                  f"consistent={v['content']['consistent']}；Δ对base={v['vs_base']['mean']:+.6f}"
                  for k, v in h3_scan.items()), tol=0.0,
        value=max([v["content"]["mean"] for v in h3_scan.values()]) if h3_scan else None)
    res["meta"]["h3_scan"] = h3_scan

    # ---- H4：cue_nogamma 命中 > cue 命中（§6.1 耦合是否破坏寻址）
    d4 = paired(flat("B4_cue_nogamma", "hit"), flat("B3_cue", "hit"))
    h4_ok = (d4["mean"] > 0 and d4["p"] < 0.05 and d4["consistent"])
    rec("H4 〔§6.1 耦合是否是元凶？〕cue_nogamma 命中显著 > cue 命中", h4_ok,
        f"mean={d4['mean']:.6f} t={d4['t']:.2f} p={d4['p']:.3g} pos={d4['pos']}/{d4['n']} "
        f"consistent={d4['consistent']}", tol=0.0, value=d4["mean"])

    # ---- H5：misaddr 命中显著 < cue 命中
    d5 = paired(flat("B3_cue", "hit"), flat("B5_misaddr", "hit"))
    h5_ok = (d5["mean"] > 0 and d5["p"] < 0.05 and d5["consistent"])
    rec("H5 〔严格对照〕cue 命中显著 > misaddr 命中", h5_ok,
        f"mean={d5['mean']:.6f} t={d5['t']:.2f} p={d5['p']:.3g} pos={d5['pos']}/{d5['n']} "
        f"consistent={d5['consistent']}", tol=0.0, value=d5["mean"])

    # ---- A2'：查询构造的生效性
    cue_oks = [x for x in flat("B3_cue", "cue_ok") if not math.isnan(x)]
    cue_hit_rate = float(np.mean(cue_oks)) if cue_oks else 0.0
    n_distinct = 0
    for s, rs in all_rows["B3_cue"].items():
        n_distinct = max(n_distinct, len({round(r["cue_ok"], 6) for r in rs}))
    a2_ok = oe_ok and (cue_hit_rate > 0 or agg["B3_cue"]["hit"] != agg["B5_misaddr"]["hit"])
    rec("A2' ① oracle 查询键逐位等于写阶段记录的目标键", oe_ok,
        f"n={len(oe)} all_exact={oe_ok}", value=oe_ok)
    rec("A2' ②/③ 线索匹配与 misaddr 的区分力", a2_ok,
        f"cue_hit_rate={cue_hit_rate:.4f} cue_hit={agg['B3_cue']['hit']} "
        f"misaddr_hit={agg['B5_misaddr']['hit']}")

    # ---- 报告项：地址命中（两因子分解）
    for name in ("B1_ctx", "B2_oracle", "B3_cue", "B4_cue_nogamma", "B5_misaddr",
                 "B6_cue_g05", "B7_cue_g20", "B8_misaddr_g05", "B9_misaddr_g20"):
        hi = [r["hit_inj"] for rs in all_rows[name].values() for r in rs if "hit_inj" in r]
        ah = [r["ahit"] for rs in all_rows[name].values() for r in rs if "ahit" in r]
        ct = [r["ctrue"] for rs in all_rows[name].values() for r in rs if "ctrue" in r]
        tw = [r["tgt_w"] for rs in all_rows[name].values() for r in rs if "tgt_w" in r]
        print(f"  [报告] {name:18s} 地址命中={np.mean(ah):.4f}  真值余弦={np.mean(ct):.4f}  "
              f"目标条目权重={np.mean(tw):.4f}", flush=True)
        agg[name]["hit_inj"] = float(np.mean(hi)) if hi else None
        agg[name]["ahit"] = float(np.mean(ah)) if ah else None
        agg[name]["ctrue"] = float(np.mean(ct)) if ct else None
        agg[name]["tgt_w"] = float(np.mean(tw)) if tw else None

    # ---- 对照：ctx 臂应当复现 DIR-14 的量级（命中 ≈ 随机基线）
    d1 = paired(flat("B3_cue", "hit"), flat("B1_ctx", "hit"))
    print(f"  [报告] B3_cue − B1_ctx 命中差 = {d1['mean']:.4f} (p={d1['p']:.3g})", flush=True)
    res["meta"]["cue_minus_ctx"] = d1
    res["meta"]["lp_base"] = agg["B0_base"]["lp"]
    for name in agg:
        res["meta"].setdefault("lp_by_arm", {})[name] = agg[name]["lp_by_seed"]

    passed = sum(1 for c in res["checks"] if c["passed"])
    res["summary_pass"] = {"passed": passed, "total": len(res["checks"])}
    print(f"\n=== 判据通过 {passed}/{len(res['checks'])} ===", flush=True)

    out = HERE.parent / ("R1_smoke_results.json" if args.smoke else "R1_results.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
