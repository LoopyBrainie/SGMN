#!/opt/pytorch/bin/python
r"""DIR-55 · §0 定向探针（第五支）：**修正 `DEF-55-5` 后**复测 (a) 与 (c)。

★ `DEF-55-5`（本轮实测）：装置的自然内容生成式 `T48.content` 的第 0 项 $h_{-1}=0$ ⇒ $k_0=\tanh(0)=\mathbf 0$
   —— 一条"自然写入"若取 $t{=}0$，它的 $G=\alpha(Wk-v)k^\top+\lambda W=\lambda W$ **只是正则在衰减**，**不含内容**。
   实测（seed 55000）：$\|k_0\|=0.000000$、$\|k_1\|=0.869553$。
   本支把自然对改成**第 1 项**（$t{=}1$，$\|k\|>0.8$），并在输出里报告 $\|k\|$ 作为守卫。
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
import R1_task48 as T48
import R1_grad as M


def fmean(xs):
    return math.fsum(xs) / len(xs)


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = fmean(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def natural_pair_valid(P, seed):
    r"""**非退化**自然对：取 `T48.content` 的第 1 项（$h_0\neq0$ ⇒ $k\neq0$）。"""
    a = T48.content(seed + 7919, 2)
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(2)]
    k = torch.tanh(H[0] @ P["Pk1"])
    v = torch.tanh(H[1] @ P["Pv1"])
    return k, v


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
    Ms, ks, c0, c1, nk = [], [], [], [], []
    nat0, dir0, nat_keep = [], [], []
    for s in range(a.seeds):
        seed = 55000 + s
        # (a) n=2
        P, keys, vals, U, Vm, _ = store(seed, 2)
        C0 = PA.cmat(U, Vm, keys, vals)
        kk, vv = natural_pair_valid(P, seed)
        nk.append(float(kk.norm()))
        U2, V2 = PA.write_seq(U, Vm, [(kk, vv)], reps=1)
        C1 = PA.cmat(U2, V2, keys, vals)
        d = [C1[i][i] - C0[i][i] for i in range(2)]
        Ms.append(d[0] - d[1])
        c0.append(C1[0][0])
        c1.append(C1[1][1])
        # (c) n=3
        P3, k3, v3, U3, V3, idx = store(seed, 3, "012")
        base = survivors(PA.diag(PA.cmat(U3, V3, k3, v3)))
        kn, vn = natural_pair_valid(P3, seed)
        Un, Vn = PA.write_seq(U3, V3, [(kn, vn)], reps=1)
        nat = survivors(PA.diag(PA.cmat(Un, Vn, k3, v3)))
        nat0.append(0 in nat)
        nat_keep.append(nat == base)
        Ud, Vd = PA.write_seq(U3, V3, [(k3[0], v3[0])], reps=1)
        dir0.append(0 in survivors(PA.diag(PA.cmat(Ud, Vd, k3, v3))))
    res = {"phase": "probe55e", "seeds": a.seeds, "secs": time.time() - t0,
           "guard_k_norm_min": min(nk), "guard_k_norm_mean": fmean(nk),
           "a_n2_m1_natural_valid": {"M_mean": fmean(Ms), "M_sd": stdev(Ms),
                                     "M_3se_at_96": 3 * stdev(Ms) / math.sqrt(96),
                                     "c0_after": fmean(c0), "c1_after": fmean(c1)},
           "c_n3": {"P_trait0_in_S_natural": fmean([1.0 if x else 0.0 for x in nat0]),
                    "P_trait0_in_S_reinforce": fmean([1.0 if x else 0.0 for x in dir0]),
                    "D_c": fmean([1.0 if x else 0.0 for x in dir0]) - fmean([1.0 if x else 0.0 for x in nat0]),
                    "P_natural_keeps_set": fmean([1.0 if x else 0.0 for x in nat_keep])}}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()