#!/usr/bin/env python3
"""`DIR-128` 前置探针（**开卡前**，`P127-1` 要求的「机械验证合并不变性」）。

把全部 8+1 个剂量的 **(cell, rng) → (n1, n2, e1, e2)** 四元组一次 dump 成 JSON。
之后**任何**聚合口径的候选量都可在该张量上纯算术算出，不必再跑装置。

★ 张量结构（由 `mech125.strata_of` / `stratum_obs` 决定）：
   - 层 = **9 个 cell**（= 9 个训练 seed），**与子集 `S` 无关** ⇒ `S` 只改 rng，不改层成员。
   - `N1_c(S) = Σ_{r∈S} n1(r)`、`N2_c(S) = Σ_{r∈S} n2(r)` —— **与 cell 无关**（同层同值）。
   - `e1_c(S) = Σ_{r∈S} e1(c,r)` —— 逐 cell 各不同。
   ⇒ 故只需存 `n1(r)/n2(r)`（4×2 个数）与 `e1(c,r)/e2(c,r)`（9×4×2 个数）。
"""
import json
import sys

sys.path.insert(0, "dir127")
sys.path.insert(0, "dir126")
sys.path.insert(0, "dir125")
sys.path.insert(0, ".")
import mech127 as MM      # noqa: E402
import mech126 as M126    # noqa: E402

M = MM.M

out = {"doses": {}, "cells_per_dose": {}, "note": "cell=训练seed层；n只由rng定、与cell无关"}
for d in M126.DOSE_ALL9:
    avail, cells = MM.available_rngs(d)
    per = {}
    for r in avail:
        ctx = dict(MM._supports(d))
        ctx["rngs"] = (r,)
        st = M.strata_of(ctx)
        obs = M.stratum_obs(ctx, st)
        n1, n2 = int(obs[0]["N1"]), int(obs[0]["N2"])
        assert all(int(o["N1"]) == n1 and int(o["N2"]) == n2 for o in obs), "N 与 cell 无关被破坏"
        per[str(r)] = {"n1": n1, "n2": n2,
                       "e1": [int(o["e1"]) for o in obs],
                       "e2": [int(o["e2"]) for o in obs]}
    out["doses"][str(d)] = {"avail": list(map(str, avail)), "per_rng": per}
    out["cells_per_dose"][str(d)] = len(cells)
    print("dose=%-4d avail=%s n_cells=%d  n1/n2=%s  Sum e1=%s Sum e2=%s"
          % (d, list(map(str, avail)), len(cells),
             {r: (v["n1"], v["n2"]) for r, v in per.items()},
             sum(sum(v["e1"]) for v in per.values()),
             sum(sum(v["e2"]) for v in per.values())))
json.dump(out, open("dir128/out/tensor128.json", "w"), indent=1, sort_keys=True)
print("\n已写 dir128/out/tensor128.json")
