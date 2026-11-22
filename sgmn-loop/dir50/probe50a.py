#!/opt/pytorch/bin/python
r"""DIR-50 · **诊断探针 A**（读数**不进判据**）—— 两层装置的可测性体检。

回答四个问题（决定卡形）：
1. **结构不变量**：臂 `hit2` 的**层 1 轨迹**是否与 `none` **逐位相同**（层 2 的记忆不影响层 1）？
2. **直接伤害是否可测**：同一剂量下，层 1 与层 2 的 `rel1`（干扰步后的目标余弦/初始余弦）——
   ⇒ 「模态无关性」的判据量 `|D_1 − D_2|` 的量级与噪声。
3. **跨模态级联是否存在**：层 1 被命中后，**层 2** 的存活曲线 `rel_2(d)`（`d ≥ 2`）相对 `none` 是否下降？
   ⇒ 级联量 `CASC` 的量级与噪声。
4. **中介通道**：层 1 的读出 `y_1` 在命中后偏移多少（`ρ1 := ‖Δy_1‖/‖y_1‖`）⇒ 级联的**通道**是否真的打开。

用法：`probe50a.py --n 200 --seed0 31000 --out dir50/out/DIR50_probe_a.json`
"""
import argparse
import json
import math
import os
import sys
import time

import torch

ROOT = "/home/ec2-user/sgmn-loop"
sys.path.insert(0, os.path.join(ROOT, "dir50"))
import dev50 as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=31000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    acc = {}
    ovk = {0.0: [], 1.0: []}
    nbad = {"none": 0, "hit1": 0, "hit2": 0}
    rho1 = {0.0: [], 1.0: []}
    identical_l1 = 0
    d_rows = []
    for i in range(a.n):
        sd = a.seed0 + i
        P = D.build(sd)
        cont = D.content(sd)
        base = D.rollout(P, cont, "none", None)
        for li in (0, 1):
            for d, c in enumerate(base["curve"][li], start=1):
                pass
        # 1) 结构不变量：`hit2` 的层 1 轨迹 vs `none`
        h2 = D.rollout(P, cont, "hit2", 1.0)
        same = all((x == y) or (isinstance(x, float) and math.isnan(x) and math.isnan(y))
                   for x, y in zip(base["curve"][0], h2["curve"][0]))
        same0 = (base["c0"][0] == h2["c0"][0]) or (math.isnan(base["c0"][0]) and math.isnan(h2["c0"][0]))
        if same and same0:
            identical_l1 += 1
        # 2) 各臂/剂量的层内读数
        for arm in ("none", "hit1", "hit2"):
            for dose in ((None,) if arm == "none" else D.DOSES):
                rr = D.rollout(P, cont, arm, dose)
                rel = [D.rel_curve(rr, li) for li in (0, 1)]
                if any(r[0] is None for r in rel):
                    nbad[arm] += 1
                if arm != "none":
                    ovk[dose].append(rr["ov_k"])
                    if arm == "hit1":
                        y1n = base["y1"][1]        # step 5 的 `y_1`（`none` 轨迹）
                        y1h = rr["y1"][1]
                        den = float(y1n.norm())
                        if den > 0 and math.isfinite(den):
                            rho1[dose].append(float((y1h - y1n).norm()) / den)
                d_rows.append({"seed": sd, "arm": arm, "dose": dose,
                               "rel": rel, "c0": list(base["c0"]),
                               "ov_k": rr["ov_k"],
                               "med": {kk: [x.clone() for x in vv] for kk, vv in rr["med"].items()},
                               "med_base": {kk: [x.clone() for x in vv] for kk, vv in base["med"].items()}})
    # 汇总
    def agg(arm, dose, li, d):
        vals = [r["rel"][li][d - 1] for r in d_rows
                if r["arm"] == arm and r["dose"] == dose and r["rel"][li][d - 1] is not None]
        if not vals:
            return None
        m = sum(vals) / len(vals)
        n = len(vals)
        sd = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
        return {"mean": m, "3se": 3 * sd / math.sqrt(n) if n else None, "n": n}

    # ★ 配对差（同实例、同层、同步）：实例级方差抵消 ⇒ 判据量的分辨力来源
    paired = {}
    for li in (0, 1):
        arm_hit = "hit1" if li == 0 else "hit2"
        for dose in D.DOSES:
            for d in range(1, D.W_AFTER + 1):
                diffs = []
                for r in d_rows:
                    if r["arm"] != arm_hit or r["dose"] != dose:
                        continue
                    base_r = next(x for x in d_rows if x["seed"] == r["seed"] and x["arm"] == "none")
                    x, y = r["rel"][li][d - 1], base_r["rel"][li][d - 1]
                    if x is not None and y is not None:
                        diffs.append(x - y)
                if not diffs:
                    continue
                m = sum(diffs) / len(diffs)
                nn = len(diffs)
                sd = (sum((z - m) ** 2 for z in diffs) / (nn - 1)) ** 0.5 if nn > 1 else 0.0
                paired[f"hit{arm_hit}|{dose}|L{li+1}|d{d}"] = {
                    "mean_diff": m, "3se": 3 * sd / math.sqrt(nn) if nn else None, "n": nn,
                    "frac_neg": sum(1 for z in diffs if z < 0) / nn}
    # ★ 跨臂配对差：`hit1` 作用在层 2 上 = **纯级联**；`hit2` 作用在层 1 上 = **恒 0**（结构不变量）
    cross = {}
    for li, arm_hit in ((0, "hit2"), (1, "hit1")):
        for dose in D.DOSES:
            for d in range(1, D.W_AFTER + 1):
                diffs = []
                for r in d_rows:
                    if r["arm"] != arm_hit or r["dose"] != dose:
                        continue
                    base_r = next(x for x in d_rows if x["seed"] == r["seed"] and x["arm"] == "none")
                    x, y = r["rel"][li][d - 1], base_r["rel"][li][d - 1]
                    if x is not None and y is not None:
                        diffs.append(x - y)
                if not diffs:
                    continue
                m = sum(diffs) / len(diffs); nn = len(diffs)
                sd = (sum((z - m) ** 2 for z in diffs) / (nn - 1)) ** 0.5 if nn > 1 else 0.0
                cross[f"{arm_hit}|{dose}|L{li+1}|d{d}"] = {
                    "mean_diff": m, "3se": 3 * sd / math.sqrt(nn) if nn else None, "n": nn,
                    "max_abs": max(abs(z) for z in diffs)}
    # ★ 中介衰减链：命中层 1 后，层 1 读出 → 层 1 注入后隐 → 层 2 注入前隐 → 层 2 的 (k2,v2)
    # ★ 中介链：**逐步**、**NaN-safe**（发散实例的 NaN/inf 必须剔除，否则中位数被排序污染）
    chain = {}
    for dose in D.DOSES:
        for kk in ("y1", "h1p", "h2pre", "k2", "v2"):
            per_step = {}
            for j in range(D.W_AFTER + 1):        # med[] 索引 j ⇒ step T_HIT + j
                vals = []
                for r in d_rows:
                    if r["arm"] != "hit1" or r["dose"] != dose:
                        continue
                    b = r.get("med_base")
                    if b is None or j >= len(r["med"][kk]) or j >= len(b[kk]):
                        continue
                    x, y = r["med"][kk][j], b[kk][j]
                    den = float(y.norm())
                    if not (den > 0 and math.isfinite(den)):
                        continue
                    val = float((x - y).norm()) / den
                    if math.isfinite(val):
                        vals.append(val)
                if vals:
                    sv = sorted(vals)
                    per_step[f"step{D.T_HIT + j}"] = {
                        "median": sv[len(sv)//2], "mean": sum(vals)/len(vals),
                        "q25": sv[len(sv)//4], "q75": sv[(3*len(sv))//4],
                        "max": sv[-1], "n": len(vals),
                        "frac_zero": sum(1 for z in vals if z == 0.0)/len(vals)}
            if per_step:
                chain[f"{dose}|{kk}"] = per_step
    out = {"n": a.n, "seed0": a.seed0, "paired_diffs": paired, "cross_diffs": cross, "chain": chain,
           "identical_l1_hit2_vs_none": identical_l1,
           "ov_k": {str(k): {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "n": len(v)}
                    for k, v in ovk.items() if v},
           "rho1_read_shift": {str(k): {"mean": sum(v) / len(v), "max": max(v), "n": len(v)}
                               for k, v in rho1.items() if v},
           "n_bad": nbad,
           "rel_by_arm_dose_layer_d": {}}
    for arm in ("none", "hit1", "hit2"):
        for dose in ((None,) if arm == "none" else D.DOSES):
            for li in (0, 1):
                for d in range(1, D.W_AFTER + 1):
                    g = agg(arm, dose, li, d)
                    if g:
                        out["rel_by_arm_dose_layer_d"][f"{arm}|{dose}|L{li+1}|d{d}"] = g
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"墙钟 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
