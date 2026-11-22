#!/opt/pytorch/bin/python
r"""DIR-55 · §0 定向探针（第二支）：**幸存集结构** + **剂量响应** + **噪声底**。

第一支（`probe55a.py`）已给出三条装置事实：
  (a) $n\le r$ 时装置能把 trait **精确**存进去并分辨（$n{=}2$ 正交键/正交值：$\mathrm{diag}\,C=0.994$、$|\mathrm{off}|_{\max}=0.066$）；
  (b) $n>r$ 时出现**驱逐**：$n{=}3$ 收敛后 $\mathrm{diag}=[0.00002,\,0.9986,\,0.99999]$（**恰 $r$ 个幸存**）；
  (c) 定向强化与非定向写入的**差分**方向相反（$+1.53$ vs $+0.54$，$n{=}3$ 正交，$m{=}8$）。

本支只回答三个 formal 轮**必须先知道**的问题：
  (S1) **幸存集是否结构稳定**：跨种子，"谁被驱逐"是同一组索引，还是随机漂移？
       —— 若随机漂移，则"幸存集"不是身份结构，而是噪声 ⇒ 该方向的主判据不成立。
  (S2) **剂量响应**：$m\in\{1,2,4,8,16\}$ 下定向差分 $D(m)=\Delta\cos_0-\overline{\Delta\cos_{j>0}}$ 的形状，
       以及**未饱和窗口**在哪（formal 轮的 $m$ 必须落在"效应可测但不撞天花板/地板"的区间）。
  (S3) **噪声底**：跨种子的 $\mathrm{sd}(D)$ ⇒ formal 轮 `3se` 门限的**唯一合法来源**（不得由效应值反推）。
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


def converge(seed, n, key_mode, reps=64):
    P, keys, vals = PA.build_traits(seed, n, key_mode)
    U, Vm = P["U1"].clone(), P["V1"].clone()
    U, Vm = PA.write_seq(U, Vm, list(zip(keys, vals)), reps=reps)
    return P, keys, vals, U, Vm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--reps-store", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()

    # ── (S1) 幸存集跨种子结构
    surv = {}
    for n in (2, 3, 4):
        pats, diags = [], []
        for s in range(a.seeds):
            seed = 55000 + s
            _, keys, vals, U, Vm = converge(seed, n, "ortho", a.reps_store)
            C = PA.cmat(U, Vm, keys, vals)
            d = PA.diag(C)
            diags.append(d)
            pats.append("".join("1" if x >= 0.5 else "0" for x in d))
        surv[n] = {"patterns": pats,
                   "pattern_counts": {p: pats.count(p) for p in sorted(set(pats))},
                   "diag_mean": [fmean([d[i] for d in diags]) for i in range(n)],
                   "diag_sd": [stdev([d[i] for d in diags]) for i in range(n)],
                   "n_survivors_mean": fmean([p.count("1") for p in pats])}

    # ── (S2)(S3) 剂量响应 + 噪声底
    dose = {}
    for n in (2, 3):
        for m in (1, 2, 4, 8, 16):
            for arm in ("reinforce", "natural"):
                Ds, d0s, djs = [], [], []
                for s in range(a.seeds):
                    seed = 55000 + s
                    r = PA.run_cell(seed, n, "ortho", arm, m=m, reps_store=a.reps_store)
                    dd = r["delta_diag"]
                    Ds.append(dd[0] - fmean(dd[1:]))
                    d0s.append(r["diag1"][0])
                    djs.append(fmean(r["diag1"][1:]))
                key = f"n{n}_m{m}_{arm}"
                dose[key] = {"n": n, "m": m, "arm": arm,
                             "D_mean": fmean(Ds), "D_sd": stdev(Ds),
                             "D_3se": 3 * stdev(Ds) / math.sqrt(a.seeds),
                             "diag0_after_mean": fmean(d0s), "diag_others_after_mean": fmean(djs)}
    # 噪声底：两个臂在同一 (n,m) 上的 D 的合并 sd（不随 m 单调，取最大者作为保守底）
    floor = max(v["D_sd"] for v in dose.values())

    res = {"phase": "probe55b", "seeds": a.seeds, "reps_store": a.reps_store,
           "secs": time.time() - t0,
           "S1_survivors": surv, "S2_dose": dose, "S3_noise_floor_sd": floor}
    print(json.dumps({"S1": {str(k): v["pattern_counts"] for k, v in surv.items()},
                      "S1_nsurv": {str(k): v["n_survivors_mean"] for k, v in surv.items()},
                      "S3_noise_floor_sd": floor}, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()