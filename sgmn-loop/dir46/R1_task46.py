#!/opt/pytorch/bin/python
r"""DIR-46 · 实现：外积记忆的容量/长期干扰曲线 `C(N_w)`。

卡：`revolve/DIR-46/DIRECTION_rev46_v10.md`（rev 1.0，md5 `19ace86942e89c0c4608b07174a2770a`，**阈值已冻结**）。
★ **逐字派生**：写入规则、`ALPHA/LAM/ETA`、`Pk/Pv/A1/b1/U1/V1` 全部沿用 `dir31/R1_grad.py` + `dir33/R1_task33.py`；
  本文件**只加两个自变量轴**：`mode ∈ {period2, perstep}`（内容持续度）与 `N_w ∈ {2,4,8,16,32,64,128}`（写入次数）。
★ **无读器、无训练、无梯度** ⇒ 判据是**前向量** `cos(W·k_s, v_s)`（卡 §0/§1）。

用法：`R1_task46.py --mode {period2,perstep} --n 400 --seed0 N --out PATH [--tag r10]`
"""
import argparse
import json
import math
import os
import sys
import time

import torch

torch.set_default_dtype(torch.float64)

ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M       # noqa: E402  ★ 装置常量（导入 R1_task33 后已被其 revision 覆盖：ETA=0.2/LAM=0.05/D_K=16）
import R1_task33 as R     # noqa: E402  ★ 装置构造 `build()`（含 KEY_SCALE=0.5）

# ══════ 冻结常数（卡 §2 / §6；**开卡后一字不改**）
NW_GRID = (2, 4, 8, 16, 32, 64, 128)   # 判据点 = 8；其余点只作诊断（卡 §4 的 `*` 格）
N_REP = 16                              # 装置写入协议（每对重复次数）
MODES = ("period2", "perstep")
CRIT_NW = 8
ANCHOR_NW = 2
D_H = R.D_H
CONTENT_TAG = 7777                      # 内容生成器偏移（与 probe46a 逐字同流）


def content(sd, n_w, mode):
    r"""长度 `n_w` 的内容序列。

    `perstep`：逐步独立 `x_t ~ N(0,1)^{D_H}`，行归一化到 `√D_H`；
    `period2`：同流，但 `a[t] := x[t mod 2]`（装置原样，只有 2 个状态）。
    ★ 生成器种子 `sd + 7777` 与 `probe46a` **逐字相同** ⇒ `A46-1` 位级锚可复现。
    """
    g = torch.Generator().manual_seed(int(sd) + CONTENT_TAG)
    x = torch.randn(n_w, D_H, generator=g)
    x = x / x.norm(dim=1, keepdim=True) * (D_H ** 0.5)
    if mode == "period2":
        x = torch.stack([x[t % 2] for t in range(n_w)])
    return x


def trace_one(sd, mode, n_w):
    r"""单个 `(seed, mode, N_w)`：写入 `n_w` 对后测全部**可读**槽位的取回余弦。

    ★ `s=0` 恒跳过：`k_0 = 0` ⇒ `W·k_0 = 0` ⇒ 该槽永不可读、且第 0 步只走 `LAM` 项（卡 §3.2 / `DEF-46-3`）。
    ★ 删失（`W` 非有限）：`finite=False`、`n_read=0`、`n_skip=n_w-1`；**不计入均值但必须计数**（卡 §3.1）。
    """
    P = R.build(sd, "V0")
    a = content(sd, n_w, mode)
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(n_w)]
    zero = torch.zeros(D_H)
    K = [torch.tanh((zero if s == 0 else H[s - 1]) @ P["Pk1"]) for s in range(n_w)]
    Vv = [torch.tanh(H[s] @ P["Pv1"]) for s in range(n_w)]
    U, V = P["U1"].clone(), P["V1"].clone()
    for s in range(n_w):
        k, v = K[s], Vv[s]
        for _ in range(N_REP):
            W = M._W(U, V)
            G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
            U = U - M.ETA * (G @ V)
            V = V - M.ETA * (G.t() @ U)
    W = M._W(U, V)
    finite = bool(torch.isfinite(W).all())
    cos, n_skip = [], 0
    if finite:
        for s in range(1, n_w):
            num = float(W @ K[s] @ Vv[s])
            den = float((W @ K[s]).norm() * Vv[s].norm())
            c = num / den if den > 0 else float("nan")
            if math.isfinite(c):
                cos.append(c)
            else:
                n_skip += 1
    else:
        n_skip = n_w - 1
    return {"n_w": n_w, "finite": finite,
            "wnorm": (float(W.norm()) if finite else None),
            "cos": cos,
            "C": (sum(cos) / len(cos)) if cos else None,
            "n_read": len(cos), "n_skip": n_skip}


def one_instance(sd, mode):
    r"""一个实例在全部 `N_w` 上的曲线（每格**独立重算**，与 worked example 逐位可比）。"""
    cells = {}
    for n_w in NW_GRID:
        c = trace_one(sd, mode, n_w)
        # ★ A46-5 计数恒等式（**每格当场断言**，残差必须为 0）
        assert c["n_read"] + c["n_skip"] == n_w - 1, (sd, mode, n_w, c)
        assert c["finite"] or c["n_read"] == 0, (sd, mode, n_w, "删失格却有可读槽")
        cells[str(n_w)] = c
    return {"seed": sd, "mode": mode, "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="r10")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            sd = args.seed0 + i
            row = one_instance(sd, args.mode)
            row["tag"] = args.tag
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"[{args.mode}] {i+1}/{args.n}  {time.time()-t0:.1f}s", flush=True)
    print(f"DONE {args.mode} n={args.n} seed0={args.seed0} → {args.out}  "
          f"墙钟 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
