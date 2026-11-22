#!/opt/pytorch/bin/python
r"""DIR-55 · §0 定向探针（第三支）：把**卡片要冻结的两个量**在 24 个种子上先量一遍（**只判可行性/可分辨**，不设判据）。

  (P1) 容量内 $n{=}2$、**一条**非人格写入后的**同损差分** $M=\Delta c_0-\Delta c_1$ 的分布与 sd
       ⇒ 决定 (a) 条款的 `3se` 与等价带是否**有功效**（3se 必须 < 等价带）。
  (P2) 超容量 $n{=}3$：**写入顺序置换**下"幸存集 = 最后写入的 $r$ 个"这一预测的**一致率**，
       以及"模式占比"（零假设 $1/\binom{3}{2}=1/3$）⇒ 决定 (b) 条款的 0.60 门限是否有意义。
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


def store(seed, n, reps=64):
    P, keys, vals = PA.build_traits(seed, n, "ortho")
    U, Vm = P["U1"].clone(), P["V1"].clone()
    U, Vm = PA.write_seq(U, Vm, list(zip(keys, vals)), reps=reps)
    return P, keys, vals, U, Vm


def one_natural(P, seed, U, Vm):
    pair = PA.natural_pairs(P, seed, 1)[0]
    return PA.write_seq(U, Vm, [pair], reps=1)


def pattern(cos, thr=0.5):
    return "".join("1" if x >= thr else "0" for x in cos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    # ── (P1) n=2：一条非人格写入后的同损差分
    Ms, c0, c1, xmax = [], [], [], []
    for s in range(a.seeds):
        seed = 55000 + s
        P, keys, vals, U, Vm = store(seed, 2)
        C0 = PA.cmat(U, Vm, keys, vals)
        U2, V2 = one_natural(P, seed, U, Vm)
        C1 = PA.cmat(U2, V2, keys, vals)
        d = [C1[i][i] - C0[i][i] for i in range(2)]
        Ms.append(d[0] - d[1])
        c0.append(C1[0][0])
        c1.append(C1[1][1])
        xmax.append(max(C1[0][1], C1[1][0]))
    sd_M = stdev(Ms)
    # ── (P2) n=3：两种顺序下"最后 r 个"预测的一致率
    agree, pats = {"012": [], "210": []}, {"012": [], "210": []}
    for s in range(a.seeds):
        seed = 55000 + s
        for order in ("012", "210"):
            idx = [int(ch) for ch in order]
            P, keys0, vals0 = PA.build_traits(seed, 3, "ortho")
            keys = [keys0[i] for i in idx]
            vals = [vals0[i] for i in idx]
            U, Vm = P["U1"].clone(), P["V1"].clone()
            U, Vm = PA.write_seq(U, Vm, list(zip(keys, vals)), reps=64)
            C = PA.cmat(U, Vm, keys, vals)
            d = PA.diag(C)
            pat = pattern(d)
            pats[order].append(pat)
            # 预测：**最后写入的 r=2 个**（即 order 的后两位）应幸存
            pred = "".join("1" if i >= 1 else "0" for i in range(3))
            agree[order].append(pat == pred)
    res = {"phase": "probe55c", "seeds": a.seeds, "secs": time.time() - t0,
           "P1_n2_m1": {"M_mean": fmean(Ms), "M_sd": sd_M,
                        "M_3se_at_n": 3 * sd_M / math.sqrt(a.seeds),
                        "M_3se_at_96": 3 * sd_M / math.sqrt(96),
                        "c0_after_mean": fmean(c0), "c1_after_mean": fmean(c1),
                        "cross_absmax_mean": fmean(xmax),
                        "powered_band_0.20": bool(3 * sd_M / math.sqrt(96) < 0.20)},
           "P2_n3_order": {o: {"agree_rate": fmean([1.0 if x else 0.0 for x in agree[o]]),
                               "pattern_counts": {p: pats[o].count(p) for p in sorted(set(pats[o]))}}
                           for o in ("012", "210")}}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()