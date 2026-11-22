#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""workex46.py — DIR-46 的**worked example 校验**（卡 §3 必需的冻结前置）。

★ 目的：把测量对象 `C(N_w)` **用一条独立写法**（**不 import** `probe46a`，**不用** `M._W`，
  而是显式 `W = U @ V.T`、`U -= ETA·(G @ V)`、`V -= ETA·(G.T @ U)`，逐行手写）重算一遍，
  与 `probe46a` 落盘的值**逐位比对**（`ATOL=1e-12`）。
★ 若不一致 ⇒ 测量对象没定义清楚 ⇒ **不得**进卡。
"""

import json
import os
import sys

import torch

torch.set_default_dtype(torch.float64)

ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M       # noqa: E402  ★ 只用常量（ALPHA/LAM/ETA/SEED0）
import R1_task33 as R     # noqa: E402  ★ 只用 build/D_H/T（装置构造）

ATOL = 1e-12
N_REP = 16
CASES = (("perstep", 4), ("perstep", 8), ("period2", 8))


def content(sd, n_w, mode):
    g = torch.Generator().manual_seed(int(sd) + 7777)
    x = torch.randn(n_w, R.D_H, generator=g)
    x = x / x.norm(dim=1, keepdim=True) * (R.D_H ** 0.5)
    if mode == "period2":
        x = torch.stack([x[t % 2] for t in range(n_w)])
    return x


def c_indep(sd, mode, n_w):
    r"""**独立实现**：手写 U/V 更新，不用 `M._W`，不用 `probe46a`。"""
    P = R.build(sd, "V0")
    a = content(sd, n_w, mode)
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(n_w)]
    zero = torch.zeros(R.D_H)
    K = [torch.tanh((zero if s == 0 else H[s - 1]) @ P["Pk1"]) for s in range(n_w)]
    Vv = [torch.tanh(H[s] @ P["Pv1"]) for s in range(n_w)]
    U = P["U1"].clone()
    Vm = P["V1"].clone()
    for s in range(n_w):
        k, v = K[s], Vv[s]
        for _ in range(N_REP):
            W = U @ Vm.transpose(0, 1)
            G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
            U = U - M.ETA * (G @ Vm)
            Vm = Vm - M.ETA * (G.transpose(0, 1) @ U)
    W = U @ Vm.transpose(0, 1)
    finite = bool(torch.isfinite(W).all())
    cs = []
    if finite:
        for s in range(1, n_w):
            num = float(W @ K[s] @ Vv[s])
            den = float((W @ K[s]).norm() * Vv[s].norm())
            cs.append(num / den if den > 0 else float("nan"))
    return {"finite": finite,
            "C": (sum(cs) / len(cs)) if cs else None,
            "wnorm": (float(W.norm()) if finite else None)}


def main():
    ref = json.load(open(os.path.join(ROOT, "dir46", "out", "DIR46_probe_a.json")))
    ok, bad = 0, []
    for mode, n_w in CASES:
        cell = next(c for c in ref["cells"] if c["mode"] == mode and c["n_w"] == n_w)
        for row in cell["rows"]:
            sd = row["seed"]
            got = c_indep(sd, mode, n_w)
            d_fin = got["finite"] != row["finite"]
            d_c = (got["C"] is None) != (row["C"] is None)
            dc = (0.0 if (got["C"] is None or row["C"] is None)
                  else abs(got["C"] - row["C"]))
            dw = (0.0 if (got["wnorm"] is None or row["wnorm"] is None)
                  else abs(got["wnorm"] - row["wnorm"]))
            if d_fin or d_c or dc > ATOL or dw > ATOL:
                bad.append({"mode": mode, "n_w": n_w, "seed": sd, "dC": dc, "dW": dw,
                            "fin_mismatch": d_fin, "none_mismatch": d_c})
            else:
                ok += 1
    print(f"逐位比对：一致 {ok} / 不一致 {len(bad)}（ATOL={ATOL:g}）")
    for b in bad[:8]:
        print("  ✗", b)
    # 抽样打印，供人眼核对
    for mode, n_w in CASES:
        cell = next(c for c in ref["cells"] if c["mode"] == mode and c["n_w"] == n_w)
        r0 = cell["rows"][0]
        g0 = c_indep(r0["seed"], mode, n_w)
        print(f"  样例 {mode} N_w={n_w} seed={r0['seed']}: "
              f"参考 C={r0['C']} ‖W‖={r0['wnorm']} | 独立 C={g0['C']} ‖W‖={g0['wnorm']}")
    print("⇒ ⇒ " + ("测量对象定义**通过**（进卡）" if not bad else "★ 未通过 ⇒ **不得**进卡"))


if __name__ == "__main__":
    main()
