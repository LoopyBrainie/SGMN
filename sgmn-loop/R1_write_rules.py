#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-14 · R1 —— 替代写规则/记忆结构能不能让记忆变得可检索？

判据见 ../DIRECTION.md（H1–H5，冻结于 rev 1.0）。

装置：`sgmn_module_structures.py`（DIR-13 装置的逐字派生 + 可插拔的记忆核心）：
  dense（v2.4 §3.2 稠密矩阵）/ multi_slot（多槽）/ sparse_dict（稀疏字典）/ explicit_kv（显式键-值分离）

臂（8 个）：
  A1 dense_linear（v2.4 原样，基线）｜A2 dense_sparse｜A3 slot_linear｜A4 slot_sparse
  A5 dict_sparse｜A6 kv_sparse｜A7 kv_linear｜A8 slot_sparse_nowrite（严格对照）
写量校准：按「每次写入的状态变化范数」把各臂的 η 缩放到与稠密基线同量级（卡片 §4-2）。
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
from sgmn_module_structures import SGMNStack as SGS, grad_report   # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MOUNT = [8, 12, 16]
ALPHA, LAM, GAMMA0, R = 1.0, 0.05, 0.05, 8
ETA_BASE = 1e-5
TOPK = 32
N_SLOTS, N_PROTO, N_ENTRIES = 64, 64, 64
SEEDS = [20261028, 20261104, 7]
N_DOCS, FILLER_TOK = 60, 30
CORPUS_BASE = ("https://raw.githubusercontent.com/pytorch/examples/main/"
               "word_language_model/data/wikitext-2/")

res = {"dir": "DIR-14", "round": "R1", "checks": [], "arms": {}, "calib": {}}


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
        r.raise_status()
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
    m = float(D.mean())
    sd = float(D.std(ddof=1))
    t = m / (sd / math.sqrt(len(D))) if sd > 0 else 0.0
    ps, k, n = sign_test_p(list(D))
    return {"mean": m, "sd": sd, "t": t, "p": ps, "pos": k, "n": n,
            "consistent": bool(k == n or k == 0)}


# ------------------------------------------------------------------ 装置
ARMS = [
    ("A1_dense_linear", "dense", "linear", False, 16.0),        # v2.4 原样（基线）
    ("A2_dense_sparse", "dense", "sparse", False, 16.0),
    ("A3_slot_sparse", "multi_slot", "sparse", False, 16.0),     # 多槽（每槽 rank-8 因子化）
    ("A4_dict_sparse", "sparse_dict", "sparse", False, 16.0),   # 稀疏字典
    ("A5_kv_sparse", "explicit_kv", "sparse", False, 16.0),     # 显式键-值分离
    ("A6_kv_linear", "explicit_kv", "linear", False, 16.0),
    ("A7_slot_sparse_nowrite", "multi_slot", "sparse", True, 16.0),   # 严格对照
    ("A8_kv_sparse_b64", "explicit_kv", "sparse", False, 64.0),
    ("A9_kv_sparse_b256", "explicit_kv", "sparse", False, 256.0),
    ("A10_kv_sparse_b1024", "explicit_kv", "sparse", False, 1024.0),
    ("A11_kv_linear_b1024", "explicit_kv", "linear", False, 1024.0),
]


def make_stack(model, d, kind, key_mode, seed=20261028, no_write=False):
    st = SGS(model, MOUNT, d, d, R, ALPHA, LAM, ETA_BASE, GAMMA0, DEV, seed=seed,
             use_factorized=False, no_write=no_write, perpos=True, mem_kind=kind,
             n_slots=N_SLOTS, n_proto=N_PROTO, n_entries=N_ENTRIES)
    for m in st.mems.values():
        m.key_mode = key_mode
        m.key_topk = TOPK
    return st


def probe_delta(st, ids):
    """跑一次前视，返回每次写入的状态变化范数的均值（用于写量校准）。"""
    ds = []
    with torch.no_grad():
        st.model(input_ids=ids)
    for m in st.mems.values():
        if m.core.last_delta > 0:
            ds.append(m.core.last_delta)
    return float(np.mean(ds)) if ds else 0.0


def calibrate(model, d, kind, key_mode, ids, ref_delta, tag):
    """按「每次写入的状态变化范数」把 η 缩放到与稠密基线同量级（卡片 §4-2）。"""
    st = make_stack(model, d, kind, key_mode)
    eta = ETA_BASE
    for m in st.mems.values():
        m.eta = eta
    snap0 = {li: m.state_snapshot() for li, m in st.mems.items()}   # 校准前必须是【原始状态】
    delta = probe_delta(st, ids)
    iters = 0
    for it in range(6):
        if delta <= 0 or ref_delta <= 0:
            break
        ratio = ref_delta / delta
        if 0.5 <= ratio <= 2.0:
            break
        eta *= ratio
        for li, m in st.mems.items():
            m.eta = eta
            m.state_restore(snap0[li])
        delta = probe_delta(st, ids)
        iters += 1
    for li, m in st.mems.items():
        m.eta = eta
        m.state_restore(snap0[li])
    st.remove()
    torch.cuda.empty_cache()
    res["calib"][tag] = {"eta": eta, "factor": eta / ETA_BASE, "delta": delta,
                         "ref_delta": ref_delta, "iters": iters}
    return eta


def run_arm(st, docs):
    """每篇文档前重置记忆（DIR-13 D12）→ 写阶段整篇前视 → 读阶段（写冻结）截断到码之前。"""
    rows = []
    for d in docs:
        for m in st.mems.values():
            m.state_restore(m._snap)
        ids = torch.tensor(d["ids"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            st.model(input_ids=ids)
        vals, keys = {}, {}
        for li, m in st.mems.items():
            vals[li] = torch.stack(m.vals_written) if m.vals_written else None
            keys[li] = torch.stack(m.keys_written) if m.keys_written else None
            m.keys_written.clear()
            m.vals_written.clear()
        pre = ids[:, :d["prefix_len"]]
        with torch.no_grad():
            lg = st.model(input_ids=pre).logits.float()
        lp = float(F.log_softmax(lg[0, -1], dim=-1)[d["code_ids"][0]])
        tp = d["prefix_len"] - 1
        hits, ctrue, ahits = [], [], []
        for li, m in st.mems.items():
            y = m.last_y[0] if m.last_y is not None else None
            V, K = vals[li], keys[li]
            ti = tp - 1
            if y is None or V is None or V.shape[0] == 0:
                hits.append(0.0); ctrue.append(0.0); ahits.append(0.0); continue
            cs = F.cosine_similarity(V, y.unsqueeze(0), dim=1)
            hits.append(1.0 if int(torch.argmax(cs)) == ti else 0.0)
            ctrue.append(float(cs[ti]) if 0 <= ti < cs.shape[0] else 0.0)
            # 【D19·分解指标】地址命中：query 键在【实际写入的键】中的 argmax
            #   ⇒ 只测"能不能寻址"，与"写没存进去"无关 ⇒ 跨臂统一、可与内容命中对照
            qk = m.last_q[-1] if m.last_q is not None else None
            if qk is not None and K is not None and K.shape[0] > 0:
                Kn = K / K.norm(dim=1, keepdim=True).clamp_min(1e-30)
                sc = (Kn @ (qk / qk.norm().clamp_min(1e-30)))
                ahits.append(1.0 if int(torch.argmax(sc)) == ti else 0.0)
            else:
                ahits.append(0.0)
        rows.append({"lp_first": lp, "hit": float(np.mean(hits)),
                     "ctrue": float(np.mean(ctrue)), "ahit": float(np.mean(ahits))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_docs = 8 if args.smoke else N_DOCS
    seeds = SEEDS[:2] if args.smoke else SEEDS

    print("=== DIR-14 R1：替代写规则/记忆结构 vs 可检索性 ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size
    docs0 = build_facts(tok, SEEDS[0], 2)
    probe_ids = torch.tensor(docs0[0]["ids"], dtype=torch.long, device=DEV).unsqueeze(0)

    # ---------------- 写量校准：先测稠密基线的每次写入变化量
    print("\n[校准] 写量对齐（按每次写入的状态变化范数）", flush=True)
    st = make_stack(model, d, "dense", "linear")
    for m in st.mems.values():
        m.eta = ETA_BASE
    ref_delta = probe_delta(st, probe_ids)
    st.remove(); torch.cuda.empty_cache()
    print(f"  稠密基线每次写入变化量 = {ref_delta:.6g}", flush=True)
    eta_by_arm = {}
    for name, kind, km, nowrite, beta in ARMS:
        if name == "A1_dense_linear":
            eta_by_arm[name] = ETA_BASE
            res["calib"][name] = {"eta": ETA_BASE, "factor": 1.0, "delta": ref_delta,
                                   "ref_delta": ref_delta, "iters": 0}
            print(f"  {name:22s} η={ETA_BASE:.4g}（基线） 每次写入变化量={ref_delta:.4g}", flush=True)
            continue
        if nowrite:
            eta_by_arm[name] = ETA_BASE
            res["calib"][name] = {"eta": ETA_BASE, "factor": 1.0, "delta": 0.0,
                                   "ref_delta": ref_delta, "iters": 0}
            continue
        if kind == "dense":
            # 稠密臂的更新对 η 线性 ⇒ 可【一步】按写量对齐（稀疏键的 ‖k‖ 不同 ⇒ 写量不同）
            st = make_stack(model, d, kind, km)
            for m in st.mems.values():
                m.eta = ETA_BASE
            snap0 = {li: m.state_snapshot() for li, m in st.mems.items()}
            d0 = probe_delta(st, probe_ids)
            eta = ETA_BASE * (ref_delta / d0) if d0 > 0 else ETA_BASE
            for li, m in st.mems.items():
                m.eta = eta
                m.state_restore(snap0[li])
            dd = probe_delta(st, probe_ids)
            st.remove(); torch.cuda.empty_cache()
            eta_by_arm[name] = eta
            res["calib"][name] = {"eta": eta, "factor": eta / ETA_BASE, "delta": dd,
                                   "ref_delta": ref_delta, "iters": 1}
            print(f"  {name:22s} η={eta:.4g}（×{eta/ETA_BASE:.3g}，一步对齐）"
                  f" 每次写入变化量={dd:.4g}", flush=True)
            continue
        # 【D16】其余结构的更新语义不同（因子化双线性 / 字典加权和 / 显式复制）⇒
        #   无法用单一标量对齐写量 ⇒ **固定声明 η=1e-5**，并把写量作为诊断报告。
        eta_by_arm[name] = ETA_BASE
        st = make_stack(model, d, kind, km)
        for m in st.mems.values():
            m.eta = ETA_BASE
            m.read_beta = beta
        snap0 = {li: m.state_snapshot() for li, m in st.mems.items()}
        dd = probe_delta(st, probe_ids)
        st.remove(); torch.cuda.empty_cache()
        res["calib"][name] = {"eta": ETA_BASE, "factor": 1.0, "delta": dd,
                               "ref_delta": ref_delta, "iters": 0,
                               "note": "固定声明 η；写量仅作诊断报告"}
        print(f"  {name:22s} η={ETA_BASE:.4g}（声明值） 每次写入变化量={dd:.4g}"
              f"（与基线比 {dd/ref_delta if ref_delta else float('nan'):.3g}）", flush=True)

    # ---------------- Stage A
    print("\n[Stage A]", flush=True)
    st = make_stack(model, d, "dense", "linear")
    base_before = {n: q.detach().clone() for n, q in model.named_parameters()}
    ids8 = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]], device=DEV)
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
    st2 = SGS(model, MOUNT, d, d, R, ALPHA, LAM, ETA_BASE, GAMMA0, DEV, seed=20261028,
              use_factorized=False, perpos=True, drop_sg=True, mem_kind="dense")
    for m in st2.mems.values():
        m.key_mode = "linear"
    lg2 = model(input_ids=ids8).logits.float()
    F.cross_entropy(lg2[:, :-1].reshape(-1, lg2.shape[-1]), ids8[:, 1:].reshape(-1)).backward()
    wrep2 = grad_report(st2.named_write_params())
    leak = len([1 for _, g, isn, _ in wrep2 if not isn and g > 0])
    st2.remove(); torch.cuda.empty_cache()
    for q in model.parameters():
        q.grad = None
    rec("A1' Freeze C（写入梯度精确0 / 读梯度有限非零 / 基座逐位不变 / 阴性对照）",
        read_ok and write_ok and base_same and leak > 0,
        f"读梯度 {sum(1 for _,_,i,_ in rrep if not i)}/{len(rrep)} 非零且有限；"
        f"写入梯度精确0 {sum(1 for _,g,i,_ in wrep if i or g==0.0)}/{len(wrep)}；"
        f"基座逐位不变 {base_same}；阴性对照泄漏 {leak}/{len(wrep2)}",
        "读梯度>0 且 写入梯度==0 且 基座不变 且 泄漏>0",
        {"read": read_ok, "write": write_ok, "base": base_same, "leak": leak})

    # A3'：单一装置 + 写关/写开两侧
    st = make_stack(model, d, "dense", "linear")
    nh, nm = len(st._handles), len(st.mems)
    with torch.no_grad():
        model(input_ids=ids8)
    Wb = {li: st.mems[li].state_snapshot().cpu().clone() for li in st.mems}
    with torch.no_grad():
        model(input_ids=ids8)
    dm_on = max(float((st.mems[li].state_snapshot().cpu() - Wb[li]).abs().max()
                       / max(float(Wb[li].abs().max()), 1e-30)) for li in st.mems)
    st.remove(); torch.cuda.empty_cache()
    st = make_stack(model, d, "dense", "linear", no_write=True)
    with torch.no_grad():
        model(input_ids=ids8)
    Wb2 = {li: st.mems[li].state_snapshot().cpu().clone() for li in st.mems}
    with torch.no_grad():
        model(input_ids=ids8)
    dm_off = max(float((st.mems[li].state_snapshot().cpu() - Wb2[li]).abs().max()
                        / max(float(Wb2[li].abs().max()), 1e-30)) for li in st.mems)
    st.remove(); torch.cuda.empty_cache()
    rec("A3' 单一装置（句柄 2×层数）+ 写关记忆冻结 / 写开必须变化",
        nh == 2 * nm and dm_off < 1e-12 and dm_on > 1e-9,
        f"句柄 {nh}；写关相对差 {dm_off:.3e}（须 <1e-12）；写开相对差 {dm_on:.3e}（须 >1e-9）",
        "句柄==2*层数 且 写关<1e-12 且 写开>1e-9", {"off": dm_off, "on": dm_on})

    # A4'：写量校准生效
    # 【D16】只有稠密臂能做写量对齐（更新对 η 线性）⇒ A4' 改为：稠密臂写量比值 ∈ [0.2,5]，
    #   其余臂的写量作为诊断报告（如实记录，不作通过条件）
    ratios = {k: (v["delta"] / v["ref_delta"]) for k, v in res["calib"].items()
              if v["delta"] > 0 and k.startswith("A")}
    dense_r = {k: v for k, v in ratios.items() if k.startswith("A1") or k.startswith("A2")}
    ok_cal = all(0.2 <= r <= 5.0 for r in dense_r.values())
    rec("A4' 写量对齐（稠密臂一步对齐，比值 ∈ [0.2,5]；其余臂写量作诊断报告）",
        ok_cal,
        "稠密：" + "；".join(f"{k}: {res['calib'][k]['delta']:.4g}"
                            f"（基线 {res['calib'][k]['ref_delta']:.4g}，比值 {v:.3g}）"
                            for k, v in dense_r.items())
        + " ｜ 其余：" + "；".join(f"{k}: {v:.3g}×基线" for k, v in ratios.items() if k not in dense_r),
        "稠密臂比值 ∈ [0.2, 5]", ratios)

    # ---------------- 主测量
    arms = {}
    for seed in seeds:
        print(f"\n  [种子 {seed}]", flush=True)
        docs = build_facts(tok, seed, n_docs)
        sa = {}
        for name, kind, km, nowrite, beta in ARMS:
            st = make_stack(model, d, kind, km, seed=seed, no_write=nowrite)
            for li, m in st.mems.items():
                m.eta = eta_by_arm[name]
                m.read_beta = beta
                m._snap = m.state_snapshot()
            rows = run_arm(st, docs)
            stats = {li: m.core.stats() for li, m in st.mems.items()}
            st.remove(); torch.cuda.empty_cache()
            sa[name] = {k: [r[k] for r in rows] for k in ("lp_first", "hit", "ctrue", "ahit")}
            sa[name]["stats"] = stats
            print(f"    {name:22s} 码logp {np.mean(sa[name]['lp_first']):+.4f} "
                  f"命中 {np.mean(sa[name]['hit']):.4f} "
                  f"地址命中 {np.mean(sa[name]['ahit']):.4f} "
                  f"槽/条目 {np.mean([v['slots_used'] for v in stats.values()]):.1f}"
                  f"/{np.mean([v['entries'] for v in stats.values()]):.1f}", flush=True)
        arms[seed] = sa

    # ---------------- A2'：结构生效性
    print("\n[A2'] 结构生效性", flush=True)
    eff = {}
    for name, kind, km, nowrite, beta in ARMS:
        if nowrite:
            continue
        stats = arms[seeds[0]][name]["stats"]
        su = float(np.mean([v["slots_used"] for v in stats.values()]))
        en = float(np.mean([v["entries"] for v in stats.values()]))
        if kind == "dense":
            ok = en > 0
            eff[name] = {"slots_used": su, "entries": en, "ok": bool(ok)}
        elif kind == "multi_slot":
            ok = su >= 4
            eff[name] = {"slots_used": su, "entries": en, "ok": bool(ok)}
        elif kind == "sparse_dict":
            ok = su >= 4
            eff[name] = {"slots_used": su, "entries": en, "ok": bool(ok)}
        else:
            ok = en >= N_ENTRIES
            eff[name] = {"slots_used": su, "entries": en, "ok": bool(ok)}
    res["effectiveness"] = eff
    rec("A2' 结构生效性（多槽/字典占用 ≥4，KV 写满 64 条，稠密有写入）",
        all(v["ok"] for v in eff.values()),
        "；".join(f"{k}: 槽 {v['slots_used']:.0f} 条目 {v['entries']:.0f} ⇒ {v['ok']}"
                 for k, v in eff.items()),
        "逐臂满足", eff)

    # ---------------- 分析
    print("\n[R1] 分析", flush=True)

    def agg(seed, arm, key):
        return np.asarray(arms[seed][arm][key], dtype=float)

    def cmp(a, b, key="hit"):
        A = np.concatenate([agg(s, a, key) for s in seeds])
        B = np.concatenate([agg(s, b, key) for s in seeds])
        c = paired(A, B)
        c["seed_means"] = [float(np.mean(arms[s][a][key]) - np.mean(arms[s][b][key])) for s in seeds]
        return c

    summ = {a: {k: float(np.mean([np.mean(arms[s][a][k]) for s in seeds]))
                for k in ("lp_first", "hit", "ahit")} for a in arms[seeds[0]]}
    res["summary"] = summ
    print("  各臂：", json.dumps(summ, ensure_ascii=False), flush=True)

    alt = [a for a in summ if a not in ("A1_dense_linear", "A2_dense_sparse",
                                        "A7_slot_sparse_nowrite")]
    best = max(alt, key=lambda a: summ[a]["hit"])
    res["best_alt"] = best
    h1 = {"best": best, "hit": summ[best]["hit"],
          "per_seed": [float(np.mean(arms[s][best]["hit"])) for s in seeds]}
    ctrl = "A7_slot_sparse_nowrite" if best.startswith("A3") else None
    h2 = cmp(best, ctrl, "lp_first") if ctrl else None
    h3 = cmp(best, ctrl, "hit") if ctrl else None
    h4 = cmp(best, "A1_dense_linear", "hit")

    rec("H1 〔存在性〕至少一个替代结构的检索命中 ≥ 0.9（随机基线 0.0244）",
        h1["hit"] >= 0.9 and all(x >= 0.9 for x in h1["per_seed"]),
        f"最佳替代臂 {best}，命中 {h1['hit']:.4f}（逐种子 "
        + " ".join(f"{x:.4f}" for x in h1["per_seed"]) + "）",
        "≥0.9 且 3 种子均满足", h1)
    if h2 is not None:
        rec("H2 〔任务层〕最佳替代臂的码 log-prob 显著 > 写关对照 A7",
            h2["mean"] > 0 and h2["p"] < 0.05 and h2["consistent"],
            f"{best}-A8 配对差 {h2['mean']:+.4f}（t={h2['t']:+.2f}, p={h2['p']:.4g}, "
            f"正向 {h2['pos']}/{h2['n']}）跨种子 " + " ".join(f"{x:+.4f}" for x in h2["seed_means"]),
            "均值>0 且 p<0.05 且 跨种子同向", h2)
        rec("H3 〔严格对照〕最佳替代臂的命中显著 > 写关对照 A7",
            h3["mean"] > 0 and h3["p"] < 0.05 and h3["consistent"],
            f"{best}-A8 配对差 {h3['mean']:+.4f}（t={h3['t']:+.2f}, p={h3['p']:.4g}, "
            f"正向 {h3['pos']}/{h3['n']}）",
            "均值>0 且 p<0.05 且 跨种子同向", h3)
    rec("H4 〔对比 v2.4〕最佳替代臂的命中显著 > 稠密基线 A1",
        h4["mean"] > 0 and h4["p"] < 0.05 and h4["consistent"],
        f"{best}-A1 配对差 {h4['mean']:+.4f}（t={h4['t']:+.2f}, p={h4['p']:.4g}, "
        f"正向 {h4['pos']}/{h4['n']}）跨种子 " + " ".join(f"{x:+.4f}" for x in h4["seed_means"]),
        "均值>0 且 p<0.05 且 跨种子同向", h4)

    # H6：〔地址选择性〕显式 KV（写完美）下，地址命中随 β 上升 ⇒ 存在 β 使其 ≥ 0.9
    kv_b = [("A5_kv_sparse", 16.0), ("A8_kv_sparse_b64", 64.0), ("A9_kv_sparse_b256", 256.0),
            ("A10_kv_sparse_b1024", 1024.0)]
    ah_curve = [(b, summ[a]["ahit"]) for a, b in kv_b if a in summ]
    best_a = max(ah_curve, key=lambda x: x[1]) if ah_curve else (None, 0.0)
    h6 = {"curve": ah_curve, "best_beta": best_a[0], "best_ahit": best_a[1]}
    rec("H6 〔地址选择性·新增〕显式 KV 结构的地址命中随 β 上升，且存在 β 使其 ≥ 0.9",
        best_a[1] >= 0.9,
        "β→地址命中：" + " ".join(f"{b:g}:{v:.4f}" for b, v in ah_curve)
        + f"（最佳 β={best_a[0]}，{best_a[1]:.4f}）",
        "存在 β 使地址命中 ≥ 0.9", h6)
    # H7：〔写 vs 地址的分解〕内容命中 vs 地址命中
    h7 = {a: {"content_hit": summ[a]["hit"], "address_hit": summ[a]["ahit"]} for a in summ}
    rec("H7 〔分解·新增〕报告内容命中与地址命中的分离（写 vs 寻址）",
        True,
        "；".join(f"{a}: 内容 {h7[a]['content_hit']:.4f} / 地址 {h7[a]['address_hit']:.4f}"
                 for a in sorted(h7)),
        "报告项", h7)

    res["comparisons"] = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H6": h6, "H7": h7}
    npass = sum(1 for c in res["checks"] if c["passed"])
    res["summary_pass"] = {"passed": npass, "total": len(res["checks"])}
    (HERE / ("R1_smoke_results.json" if args.smoke else "R1_results.json")).write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== DIR-14 R1：{npass}/{len(res['checks'])} 通过 ===", flush=True)


if __name__ == "__main__":
    main()
