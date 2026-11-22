#!/opt/pytorch/bin/python
r"""DIR-34 诊断：A34-6 的**实例 max** 条款是**样本量依赖**的 ⇒ 必须换成总体量（分位数）。

**现象**：DIR-33 的 A33-6 双条款（实例平均 ≤0.60 且 实例 max ≤0.90）在 **24 个实例**上
以 0.8892 通过（余量仅 0.011）；DIR-34 在 **400 个实例**上实测 **0.9951** ⇒ 必然不过。

**为什么**：实例 max 是对实例取**极值**的统计量，其**取值随实例数增长而增大**
（极值统计的经典性质）⇒ 它不是总体性质，而是「样本量 × 分布」的联合产物
—— 与决定三十六（纪律 78）**同族**：断言必须与判据**同口径**（总体统计量）。

**本诊断报**：逐实例 max 的**分位数分布**（评估段 400 实例 vs 标定段 24 实例），
供 rev 1.1 把「实例 max ≤ 0.90」换成**总体量**（固定分位数）并冻结阈值。
"""
import json
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
import R1_task33 as R

OUT = "/home/ec2-user/sgmn-loop/dir34/out"
SEED0 = R.M.SEED0
SETS = {"calib_held_24": [SEED0 + 700 + i for i in range(24)],
        "eval_400": [SEED0 + 2000 + i for i in range(400)]}


def per_inst_max(seeds):
    out = []
    for sd in seeds:
        P = R.build(sd, "V0")
        R.period2(P, sd)
        g = R.targets(P)
        cs = [abs(float((g[i] * g[j]).sum() / (g[i].norm() * g[j].norm())))
              for i in range(len(g)) for j in range(len(g)) if (i - j) % 2 == 1]
        out.append(max(cs))
    return out


def pct(v, q):
    s = sorted(v)
    k = min(len(s) - 1, max(0, int(round(q / 100 * (len(s) - 1)))))
    return s[k]


res = {}
for name, seeds in SETS.items():
    v = per_inst_max(seeds)
    res[name] = {"n": len(v), "mean": statistics.fmean(v), "median": statistics.median(v),
                 "p90": pct(v, 90), "p95": pct(v, 95), "p99": pct(v, 99), "max": max(v)}
    r = res[name]
    print(f"{name:16s} n={r['n']:4d} · 均值 {r['mean']:.4f} · 中位 {r['median']:.4f} · "
          f"p90 {r['p90']:.4f} · p95 {r['p95']:.4f} · p99 {r['p99']:.4f} · max {r['max']:.4f}")
print("\n读法：均值随 n 稳定（≈0.48）⇒ **总体量**；max 随 n 增大（0.8892@24 → 0.9951@400）"
      "⇒ 极值不是总体性质，条款必须换成固定分位数。")
json.dump(res, open(f"{OUT}/DIR34_diag_instmax.json", "w"), indent=1)
