#!/opt/pytorch/bin/python
r"""DIR-41 · **只读收口核查**（**不改任何已冻结内容、不重跑正式轮**）

三件事（Advisor 复核要求）：
  ① `A41-11` 的 PI「跨实例逐位不同」断言：**实际比较对数**是否覆盖 `C(4,2)=6`
     （run2 的实现是「与首元素比」⇒ 只有 3 对 ⇒ **抓不住尾部互相重复**）⇒ 本脚本**独立地**
     枚举**全部**对，并额外在更大样本上复核。
  ② **探针**用的共享图种子 / 实例种子段，是否与**正式轮**的 5 张图 / 5 个段**相交**
     ⇒ 若相交 ⇒ 「5/5」就不是 5 次未被预看的抽取。
  ③ **独立重算**裁决：从 18 个 `DIR41r10_cell_*.json` 的**原始读数**（不经 `merge()`）
     重算 P1/P2/P3、通过数、对比检验与判定表分支 ⇒ 与 `DIR41r10_formal.json` 的结局**比对**。

用法：readonly_checks41.py
"""
import json
import math
import os
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir37")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir40")
import R1_task33 as R
import R1_grad as M
import R1_task37 as T37
import R1_task40 as T40

OUT = "/home/ec2-user/sgmn-loop/dir41/out"
SEED0 = M.SEED0
SHARED = {"SH1": 20262101, "SH2": 20262102, "SH3": 20262103,
          "SH4": 20262104, "SH5": 20262105}
CELLS = tuple(SHARED) + ("PI",)
ARMS = ("oracle", "rand", "learned")
SIG_K, FRAC = 3.0, 0.20

# 探针用过的种子 / 段（`dir41/PROBE41.md` + `dir41/probe41.py`）
PROBE_PK1_SEEDS = {"S1": 20261001, "S2": 20261988}
PROBE_OFF = {"smoke": (20400, 20412), "train": (20500, 20595), "eval": (20600, 20999)}
DIR40_PK1_SEED = 20261112
FORMAL_OFF = {"train": (21000, 21095), "smoke": (21100, 21111), "eval": (21500, 21899)}


def _stat(d):
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    se = sd / len(d) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": SIG_K * se, "n": len(d)}


def check1_a41_11_pairs():
    r"""① PI 格的 `Pk1`：**枚举全部对**（不止「与首元素比」）⇒ 报出比较对数与全异结论。"""
    out = {}
    for n_inst in (4, 20):
        seeds = [SEED0 + 32000 + i for i in range(n_inst)]   # 全新实例种子（只读结构检查）
        mats = []
        for sd in seeds:
            P = T37._mk(sd, None)
            mats.append(P["Pk1"])
        pairs, diff, same = 0, 0, []
        for i in range(len(mats)):
            for j in range(i + 1, len(mats)):
                pairs += 1
                d = float((mats[i] - mats[j]).abs().max())
                if d > 0.0:
                    diff += 1
                else:
                    same.append((i, j))
        out[f"n_instances={n_inst}"] = {
            "pairs_checked": pairs, "pairs_expected_C(n,2)": n_inst * (n_inst - 1) // 2,
            "pairs_differing": diff, "pairs_identical": same,
            "all_pairs_differ": (diff == pairs and pairs == n_inst * (n_inst - 1) // 2)}
    out["note"] = ("run2 的实现只把其余 3 个与首元素比（3 对）⇒ 抓不住尾部互相重复；"
                   "本脚本枚举全部 C(n,2) 对。")
    return out


def check2_seed_independence():
    r"""② 探针 / DIR-40 的共享图种子与实例段，是否与正式轮相交。"""
    formal_pk1 = set(SHARED.values())
    probe_pk1 = set(PROBE_PK1_SEEDS.values())
    out = {
        "probe_pk1_seeds": sorted(probe_pk1), "formal_pk1_seeds": sorted(formal_pk1),
        "dir40_pk1_seed": DIR40_PK1_SEED,
        "probe_pk1_inter_formal": sorted(probe_pk1 & formal_pk1),
        "dir40_pk1_inter_formal": [DIR40_PK1_SEED] if DIR40_PK1_SEED in formal_pk1 else [],
    }
    inter = []
    for k, (a, b) in PROBE_OFF.items():
        for k2, (a2, b2) in FORMAL_OFF.items():
            lo, hi = max(a, a2), min(b, b2)
            if lo <= hi:
                inter.append(f"{k}({a}-{b}) ∩ {k2}({a2}-{b2}) = {lo}-{hi}")
    out["probe_offsets_inter_formal_offsets"] = inter
    out["all_disjoint"] = (not out["probe_pk1_inter_formal"] and not inter
                           and not out["dir40_pk1_inter_formal"])
    return out


def check3_recompute_verdict():
    r"""③ 从 18 个原始 JSON **独立重算**裁决（不经 `merge()`）⇒ 与归档结局比对。"""
    arms, series = {}, {}
    for cell in CELLS:
        for arm in ARMS:
            p = os.path.join(OUT, f"DIR41r10_cell_{cell}_{arm}.json")
            d = json.load(open(p, encoding="utf-8"))
            for k, v in d["arms"].items():
                arms[(cell, k)] = v
                series[(cell, k)] = d["L_series"][k]
    res = {}
    for cell in CELLS:
        Lo = arms[(cell, "oracle")]["mean_L"]
        Lr = arms[(cell, "rand")]["mean_L"]
        Ll = arms[(cell, "learned")]["mean_L"]
        Lrc = arms[(cell, "learned_rc")]["mean_L"]
        Rj = Lr - Lo
        pd = _stat([a - b for a, b in zip(series[(cell, "rand")], series[(cell, "learned")])])
        pc = _stat([a - b for a, b in zip(series[(cell, "learned_rc")], series[(cell, "learned")])])
        delta = Lr - Ll
        p1 = delta >= pd["3se"]
        p2 = delta >= FRAC * Rj
        p3 = pc["mean"] >= pc["3se"]
        res[cell] = {"R": Rj, "floor": FRAC * Rj, "delta": delta,
                     "delta_3se": pd["3se"], "delta_se": pd["se"], "delta_sd": pd["sd"],
                     "content_mean": pc["mean"], "content_3se": pc["3se"],
                     "P1": bool(p1), "P2": bool(p2), "P3": bool(p3),
                     "pass": bool(p1 and p2 and p3),
                     "L_oracle": Lo, "L_rand": Lr, "L_learned": Ll, "L_learned_rc": Lrc}
    # 对比检验（逐图、逐实例配对）
    d_pi = [a - b for a, b in zip(series[("PI", "rand")], series[("PI", "learned")])]
    contrast = {}
    for cell in SHARED:
        d_j = [a - b for a, b in zip(series[(cell, "rand")], series[(cell, "learned")])]
        st = _stat([x - y for x, y in zip(d_j, d_pi)])
        contrast[cell] = {"mean": st["mean"], "3se": st["3se"], "pass": bool(st["mean"] >= st["3se"])}
    shared_pass = sum(1 for c in SHARED if res[c]["pass"])
    contrast_pass = sum(1 for c in SHARED if contrast[c]["pass"])
    pi = res["PI"]
    finite = all(math.isfinite(v) for c in res.values() for v in
                 (c["R"], c["delta"], c["delta_se"], c["content_mean"], c["content_3se"]))
    # 判定表（按序命中）
    if not finite:
        outcome = "DIVERGED"
    elif any(res[c]["R"] <= 0 for c in res):
        outcome = "INDETERMINATE"
    elif pi["P1"] and pi["P3"]:
        outcome = "NOT-NECESSARY"
    elif shared_pass < 4:
        outcome = "SHARED-NOT-REPLICATED"
    elif contrast_pass < 4:
        outcome = "CONTRAST-FAIL"
    else:
        outcome = "PASS"
    # 相对极差（**分项**，不做捆绑）
    def spread(vals):
        lo, hi = min(vals), max(vals)
        return {"min": lo, "max": hi, "rel_spread_pct": (hi - lo) / lo * 100 if lo else None}
    out = {
        "cells": res, "contrast": contrast,
        "shared_pass_count": shared_pass, "contrast_pass_count": contrast_pass,
        "recomputed_outcome": outcome,
        "spread_by_quantity": {
            "closure_CL": spread([res[c]["delta"] / res[c]["R"] for c in SHARED]),
            "effect_over_floor": spread([res[c]["delta"] / res[c]["floor"] for c in SHARED]),
            "L_oracle": spread([res[c]["L_oracle"] for c in SHARED]),
            "R": spread([res[c]["R"] for c in SHARED]),
            "delta": spread([res[c]["delta"] for c in SHARED]),
        },
        "pi_decomposition": {
            "delta": pi["delta"], "delta_3se_bar": pi["delta_3se"],
            "abs_delta_over_3se_bar": abs(pi["delta"]) / pi["delta_3se"],
            "abs_delta_over_se": abs(pi["delta"]) / pi["delta_se"],
            "content_mean": pi["content_mean"], "content_3se": pi["content_3se"],
        },
        "contrast_decomposition": {
            c: {"contrast_mean": contrast[c]["mean"],
                "shared_delta_component": res[c]["delta"],
                "minus_delta_PI_component": -pi["delta"]}
            for c in SHARED},
    }
    arch = json.load(open(os.path.join(OUT, "DIR41r10_formal.json"), encoding="utf-8"))
    out["archived_outcome"] = arch["outcome"]
    out["archived_counted"] = arch["counted"]
    out["agrees_with_archived"] = (arch["outcome"] == outcome)
    out["archived_vs_recomputed_shared_pass"] = [arch["shared_pass_count"], shared_pass]
    out["archived_vs_recomputed_contrast_pass"] = [arch["contrast_pass_count"], contrast_pass]
    return out


def main():
    res = {"A41_11_pairs": check1_a41_11_pairs(),
           "seed_independence": check2_seed_independence(),
           "recomputed_verdict": check3_recompute_verdict()}
    # Stage A 归档读数复核（直接从 `DIR41r10_stageA.json` 取，列出关键闸门值）
    sa = json.load(open(os.path.join(OUT, "DIR41r10_stageA.json"), encoding="utf-8"))
    g = sa["guards"]
    res["stageA_recheck"] = {
        "ok": sa["ok"],
        "A41-4": {k: g[k] for k in g if k.startswith("A41-4")},
        "A41-13": {k: g[k] for k in g if k.startswith("A41-13")},
        "A41-11": {k: g[k] for k in g if k.startswith("A41-11")},
        "A41-5_max_abs_diff": g["A41-5_max_abs_diff"],
        "A41-6_max_abs_diff": g["A41-6_max_abs_diff"],
        "A41-9_meta": g["A41-9_meta"],
    }
    with open(os.path.join(OUT, "DIR41r10_readonly_checks.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("── ① A41-11 的比较对数 ──")
    print(json.dumps(res["A41_11_pairs"], ensure_ascii=False, indent=1))
    print("── ② 种子独立性 ──")
    print(json.dumps(res["seed_independence"], ensure_ascii=False, indent=1))
    r3 = res["recomputed_verdict"]
    print("── ③ 独立重算裁决 ──")
    print("  重算结局 =", r3["recomputed_outcome"], " 归档结局 =", r3["archived_outcome"],
          " 一致 =", r3["agrees_with_archived"])
    print("  共享通过数 归档/重算 =", r3["archived_vs_recomputed_shared_pass"],
          " 对比通过数 归档/重算 =", r3["archived_vs_recomputed_contrast_pass"])
    print("  分项相对极差 =", json.dumps(r3["spread_by_quantity"], ensure_ascii=False, indent=1))
    print("  PI 分解 =", json.dumps(r3["pi_decomposition"], ensure_ascii=False, indent=1))
    print("  对比分解 =", json.dumps(r3["contrast_decomposition"], ensure_ascii=False, indent=1))
    print("── Stage A 归档复核 ──")
    print(json.dumps(res["stageA_recheck"], ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    main()
