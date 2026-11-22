#!/opt/pytorch/bin/python
r"""DIR-47 · 工作例（**独立第二链**）：不用 `R1_task47` 的任何函数，从零手写同一批计算。

★ 目的：给卡 §3.3 与 `A47-2` 提供**逐位**对照 —— 若两条链的子公式有分歧（漏乘、下标错、`picks` 口径错），
   这里会当场不一致。本文件**只**用装置常量（`M.ALPHA/LAM/ETA`、`R.build`）与 `D_H`；
   `content`、`K/V`、写循环、槽位取回、`hit/rest` 分组**全部重写**。

用法：`workex47.py`（打印比对结果；非零退出码 = 不一致）
"""
import math
import sys

import torch

torch.set_default_dtype(torch.float64)

ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir47"):
    sys.path.insert(0, ROOT + "/" + _d)

import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task47 as T47      # noqa: E402   ★ 只用来取「被比对的那条链」

CTX, S, RR, NREP = 3, 8, 2, 16
D_H = R.D_H
TAG_C, TAG_R, OFF_R = 7777, 9899, 17


def indep_content(sd, n):
    g = torch.Generator().manual_seed(int(sd) + TAG_C)
    x = torch.randn(n, D_H, generator=g)
    for t in range(n):                        # ★ 逐行归一化（写法与实现不同，结果须相同）
        x[t] = x[t] / torch.linalg.norm(x[t]) * math.sqrt(D_H)
    return x


def indep_picks(sd):
    g = torch.Generator().manual_seed(int(sd) + TAG_R + OFF_R)
    out = []
    for _ in range(RR):
        out.append(int(torch.randint(low=1, high=S, size=(1,), generator=g).item()))
    return out


def indep_run(sd, arm):
    P = R.build(sd, "V0")
    n = CTX + S + RR
    a = indep_content(sd, n)
    Hh = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(n)]
    ks, vs = [], []
    for t in range(n):
        src = torch.zeros(D_H) if t == 0 else Hh[t - 1]
        ks.append(torch.tanh(src @ P["Pk1"]))
        vs.append(torch.tanh(Hh[t] @ P["Pv1"]))
    U, V = P["U1"].clone(), P["V1"].clone()
    order = list(range(CTX + S))
    if arm == "nor":
        order += list(range(CTX + S, n))
    else:
        order += [CTX + s for s in indep_picks(sd)]
    for t in order:
        for _ in range(NREP):
            W = M._W(U, V)                     # 只用装置 `_W`（不自己拼 `U Vᵀ`）
            GG = M.ALPHA * torch.outer(W @ ks[t] - vs[t], ks[t]) + M.LAM * W
            U = U - M.ETA * (GG @ V)
            V = V - M.ETA * (GG.t() @ U)
    W = M._W(U, V)
    c = {}
    for s in range(1, S):
        t = CTX + s
        den = torch.linalg.norm(W @ ks[t]) * torch.linalg.norm(vs[t])
        c[s] = float((W @ ks[t]) @ vs[t] / den) if float(den) > 0 else float("nan")
    return c, indep_picks(sd)


def main():
    seeds = [25200, 25201, 25202, 25203]
    worst = 0.0
    bad = 0
    for sd in seeds:
        for arm in ("nor", "rep_on"):
            ind, pk_ind = indep_run(sd, arm)
            got = T47.trace_one(sd, arm)
            assert pk_ind == got["picks"], (sd, arm, pk_ind, got["picks"])
            for s, v in ind.items():
                g = got["slots"][str(s)]
                d = abs(g - v)
                worst = max(worst, d)
                if d > 1e-12:
                    bad += 1
                    print(f"✗ sd={sd} arm={arm} s={s} indep={v!r} impl={g!r} |Δ|={d:.3e}")
        print(f"  sd={sd} 比对完（两臂 × 7 槽位）", flush=True)
    print(f"\n工作例：{len(seeds)*2} 次运行 × 7 槽位 = {len(seeds)*2*7} 个单元；"
          f"max|Δ| = {worst:.3e}；不一致 {bad} 个")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
