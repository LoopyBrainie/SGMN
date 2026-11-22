#!/opt/pytorch/bin/python
r"""DIR-48 · 工作例（**独立第二链**）：不用 `R1_task48` 的任何函数，从零手写同一批计算。

★ 目的：给卡 §3.3 与 `A48-2` 提供**逐位**对照。`content`、`K/V`、写循环、读数、`L` 判据**全部重写**；
   只借用装置常量（`M.ALPHA/LAM/ETA`、`R.build`）与窗口/阈值定义。

用法：`workex48.py`（非零退出码 = 不一致）
"""
import math
import sys

import torch

torch.set_default_dtype(torch.float64)

ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir48"):
    sys.path.insert(0, ROOT + "/" + _d)

import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task48 as T48      # noqa: E402   ★ 只取「被比对的那条链」

CTX, NREP, RHO = 3, 16, 0.5
WIN = {"quiet": 16, "decay": 32, "busy": 8}
D_H = R.D_H
TAG_C = 7777
NMARK = CTX + 1 + max(WIN.values())      # ★ 与实现同口径（`DEF-48-4`）


def indep_content(sd, n):
    g = torch.Generator().manual_seed(int(sd) + TAG_C)
    x = torch.randn(n, D_H, generator=g)
    for t in range(n):
        x[t] = x[t] / torch.linalg.norm(x[t]) * math.sqrt(D_H)
    return x


def indep_run(sd, arm):
    W = WIN[arm]
    P = R.build(sd, "V0")
    n = CTX + W
    a = indep_content(sd, NMARK)[:n]        # ★ 固定长度生成后切片 ⇒ 各臂共享同一前缀
    Hh = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(n)]
    ks, vs = [], []
    for t in range(n):
        src = torch.zeros(D_H) if t == 0 else Hh[t - 1]
        ks.append(torch.tanh(src @ P["Pk1"]))
        vs.append(torch.tanh(Hh[t] @ P["Pv1"]))
    U, V = P["U1"].clone(), P["V1"].clone()

    def w_of():
        return M._W(U, V)

    def advance(k, v, alpha):
        nonlocal U, V
        for _ in range(NREP):
            Wm = w_of()
            GG = alpha * torch.outer(Wm @ k - v, k) + M.LAM * Wm
            U = U - M.ETA * (GG @ V)
            V = V - M.ETA * (GG.t() @ U)

    def rd():
        Wm = w_of()
        den = torch.linalg.norm(Wm @ ks[CTX]) * torch.linalg.norm(vs[CTX])
        num = (Wm @ ks[CTX]) @ vs[CTX]
        return (float(num / den) if float(den) > 0 else float("nan")), float(torch.linalg.norm(Wm))

    for t in range(CTX + 1):
        advance(ks[t], vs[t], 1.0)
    cs, ws = [], []
    cc, ww = rd(); cs.append(cc); ws.append(ww)
    for d in range(1, W):
        if arm == "decay":
            advance(ks[CTX + d], vs[CTX + d], 0.0)
        elif arm == "busy":
            advance(ks[CTX + d], vs[CTX + d], 1.0)
        cc, ww = rd(); cs.append(cc); ws.append(ww)
    L, cens = W, True
    if math.isfinite(cs[0]) and cs[0] > 0:
        for d in range(1, W):
            if cs[d] < RHO * cs[0]:
                L, cens = d, False
                break
    return cs, ws, L, cens


def main():
    seeds = [26200, 26201, 26202, 26203]
    worst_c = worst_w = 0.0
    bad = 0
    for sd in seeds:
        for arm in ("quiet", "decay", "busy"):
            cs, ws, L, cens = indep_run(sd, arm)
            got = T48.trace_one(sd, arm)
            if got["undef"]:
                print(f"  (sd={sd} arm={arm} 实现判 undef，跳过逐位比对)")
                continue
            if (L, cens) != (got["L"], got["censored"]):
                bad += 1
                print(f"✗ sd={sd} arm={arm} L/cens indep=({L},{cens}) impl=({got['L']},{got['censored']})")
            for d in range(len(cs)):
                dc = abs(cs[d] - got["c"][d]); dw = abs(ws[d] - got["wnorm"][d])
                worst_c = max(worst_c, dc); worst_w = max(worst_w, dw)
                if dc > 1e-12 or dw > 1e-12:
                    bad += 1
                    print(f"✗ sd={sd} arm={arm} d={d} c|Δ|={dc:.2e} w|Δ|={dw:.2e}")
        print(f"  sd={sd} 比对完（3 臂）", flush=True)
    ncell = sum(WIN[a] for a in WIN) * len(seeds)
    print(f"\n工作例：{len(seeds)} 实例 × 3 臂 = {ncell} 个 (d, c/‖W‖) 单元；"
          f"max|Δc| = {worst_c:.3e}，max|Δ‖W‖| = {worst_w:.3e}；不一致 {bad}")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
