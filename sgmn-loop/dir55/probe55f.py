#!/opt/pytorch/bin/python
r"""DIR-55 · §0 定向探针（第六支）：为 (a) 条款找**有功效**的形态（`DEF-55-5` 修正后 $M$ 的 sd 太大：$0.750$ ⇒ $3\mathrm{se}@96=0.230\ge0.20$）。

两个候选形态（都在**同一初始状态**上做独立重复，写入次数可预注册）：
  (A1) $R{=}8$ 条**独立**自然写入的**平均**同损差分 $\bar M_8$（每条都从**同一**初始状态出发，互不叠加）；
  (A2) **相对**同损 $M^{\text{rel}}=\Delta c_0/|\Delta c_0| - \Delta c_1/|\Delta c_1|$（只比方向，不比幅度）。
本支只报**跨种子 sd** 与投影 `3se@96`，用于在**正式轮之前**判定哪一形态有功效。
"""
import argparse
import json
import math
import os
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48"):
    sys.path.insert(0, os.path.join("/home/ec2-user/sgmn-loop", _d))
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir55")
import probe55a as PA
import probe55e as PE


def fmean(xs):
    return math.fsum(xs) / len(xs)


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = fmean(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    M8, Mrel, drops = [], [], []
    for s in range(a.seeds):
        seed = 55000 + s
        P, keys, vals, U, Vm, _ = PE.store(seed, 2)
        C0 = PA.cmat(U, Vm, keys, vals)
        d0s, d1s = [], []
        for r in range(a.reps):
            kk, vv = PE.natural_pair_valid(P, seed * 100 + r)   # 每条自然写入**独立内容种子**
            assert float(kk.norm()) > 0.1, float(kk.norm())     # 守卫：非退化
            U2, V2 = PA.write_seq(U, Vm, [(kk, vv)], reps=1)
            C1 = PA.cmat(U2, V2, keys, vals)
            d0s.append(C1[0][0] - C0[0][0])
            d1s.append(C1[1][1] - C0[1][1])
        m0, m1 = fmean(d0s), fmean(d1s)
        M8.append(m0 - m1)
        Mrel.append((1.0 if m0 >= 0 else -1.0) - (1.0 if m1 >= 0 else -1.0))
        drops.append((abs(m0) + abs(m1)) / 2)
    sd8 = stdev(M8)
    res = {"phase": "probe55f", "seeds": a.seeds, "reps": a.reps, "secs": time.time() - t0,
           "A1_mean_of_8": {"M8_mean": fmean(M8), "M8_sd": sd8,
                            "M8_3se_at_96": 3 * sd8 / math.sqrt(96),
                            "powered_0.20": bool(3 * sd8 / math.sqrt(96) < 0.20)},
           "A2_relative": {"Mrel_mean": fmean(Mrel), "Mrel_sd": stdev(Mrel),
                           "Mrel_3se_at_96": 3 * stdev(Mrel) / math.sqrt(96)},
           "mean_drop_per_write": fmean(drops)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()