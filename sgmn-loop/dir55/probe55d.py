#!/opt/pytorch/bin/python
r"""DIR-55 · §0 定向探针（第四支）：**按卡片将要冻结的三个量**在 24 个种子上先量一遍（只判可行性/可分辨）。

  (a) $n{=}2$ + 一条非人格写入：同损差分 $M$（已有 `probe55c` 的 $M=-0.061\pm0.171$；本支复核）
  (b) $n{=}3$：幸存者数分布 + "幸存者 = 最近写入者"的占比（两种顺序）
  (c) $n{=}3$：**非定向**写入后 $\Pr(\text{幸存者不变})$ vs **定向**强化后 $\Pr(\text{幸存者} = \text{trait }0)$
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


def fmean(xs):
    return math.fsum(xs) / len(xs)


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = fmean(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def survivors(cos, thr=0.5):
    return [i for i, x in enumerate(cos) if x >= thr]


def store(seed, n, order=None, reps=64):
    P, keys0, vals0 = PA.build_traits(seed, n, "ortho")
    idx = list(range(n)) if order is None else [int(c) for c in order]
    keys = [keys0[i] for i in idx]
    vals = [vals0[i] for i in idx]
    U, Vm = P["U1"].clone(), P["V1"].clone()
    U, Vm = PA.write_seq(U, Vm, list(zip(keys, vals)), reps=reps)
    return P, keys, vals, U, Vm, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()

    # (a)
    Ms, c0, c1, xs = [], [], [], []
    # (b)
    cnt1, last_ok, ns = [], [], []
    # (c)
    nat_keep, dir_hit = [], []
    for s in range(a.seeds):
        seed = 55000 + s
        # ── (a) n=2
        P, keys, vals, U, Vm, _ = store(seed, 2)
        C0 = PA.cmat(U, Vm, keys, vals)
        U2, V2 = PA.write_seq(U, Vm, PA.natural_pairs(P, seed, 1), reps=1)
        C1 = PA.cmat(U2, V2, keys, vals)
        d = [C1[i][i] - C0[i][i] for i in range(2)]
        Ms.append(d[0] - d[1])
        c0.append(C1[0][0])
        c1.append(C1[1][1])
        xs.append(max(C1[0][1], C1[1][0]))
        # ── (b) n=3 两种顺序
        for order in ("012", "210"):
            P3, k3, v3, U3, V3, idx = store(seed, 3, order)
            C = PA.cmat(U3, V3, k3, v3)
            sv = survivors(PA.diag(C))
            cnt1.append(len(sv) == 1)
            if len(sv) == 1:
                last_ok.append(idx[sv[0]] == idx[-1])   # 幸存者的**原始索引** = 最后写入者的原始索引
            ns.append(len(sv))
        # ── (c) n=3 基准（顺序 012）
        P3, k3, v3, U3, V3, idx = store(seed, 3, "012")
        base = survivors(PA.diag(PA.cmat(U3, V3, k3, v3)))
        # 非定向：1 条自然内容
        Un, Vn = PA.write_seq(U3, V3, PA.natural_pairs(P3, seed, 1), reps=1)
        nat = survivors(PA.diag(PA.cmat(Un, Vn, k3, v3)))
        nat_keep.append(nat == base)
        # 定向：再写一次 trait 0
        Ud, Vd = PA.write_seq(U3, V3, [(k3[0], v3[0])], reps=1)
        dd = PA.diag(PA.cmat(Ud, Vd, k3, v3))
        dir_hit.append(0 in survivors(dd))

    res = {"phase": "probe55d", "seeds": a.seeds, "secs": time.time() - t0,
           "a_n2_m1": {"M_mean": fmean(Ms), "M_sd": stdev(Ms),
                       "M_3se_at_96": 3 * stdev(Ms) / math.sqrt(96),
                       "c0_after": fmean(c0), "c1_after": fmean(c1),
                       "cross_absmax_mean": fmean(xs)},
           "b_n3": {"P_one_survivor": fmean([1.0 if x else 0.0 for x in cnt1]),
                    "n_cells": len(cnt1), "mean_survivors": fmean(ns),
                    "P_survivor_is_last": fmean([1.0 if x else 0.0 for x in last_ok]),
                    "n_unique": len(last_ok)},
           "c_directed": {"P_natural_keeps_set": fmean([1.0 if x else 0.0 for x in nat_keep]),
                          "P_reinforce_makes_trait0_survivor": fmean([1.0 if x else 0.0 for x in dir_hit])}}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()