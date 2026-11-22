r"""DIR-61 算例（worked example，**不是**正式轮）：验证「死块」驻点与逃逸时间律。

装置（§5.1 的局部二次目标，退化到单对、精确 C/B）：
  k = e1, v = e2, d = 2, r = 1, λ = 0.05, α ≡ 1
  J(W) = ½‖Wk − v‖² + λ/2‖W − W0‖²_F
  G = (Wk − v) kᵀ + λ(W − W0)
  因子化（同步）：U ← U − η G V,  V ← V − η Gᵀ U,  W = U Vᵀ
  全矩阵参考：W ← W − η G

死块构造：U0 ∝ e1（⊥ v）, V0 ∝ e2（⊥ k）⇒ W0 = U0V0ᵀ 满足 W0k = 0、GV = 0、GᵀU = 0 ⇒ 永久驻在 W = W0。
带投影 θ：V0 = cosθ·e2 + sinθ·e1 ⇒ kᵀV0 = sinθ ⇒ 逃逸速率一阶 ⇒ 预测 τ ∝ |sinθ|^{-1}。
"""
import argparse
import json
import math

import numpy as np

LAM = 0.05
ETA = 0.05
K = np.array([[1.0], [0.0]])      # e1
V = np.array([[0.0], [1.0]])      # e2
NMAX = 400_000


def J(W, W0):
    r = W @ K - V
    return 0.5 * float(np.sum(r * r)) + 0.5 * LAM * float(np.sum((W - W0) ** 2))


def run_fac(theta, eta=ETA, nmax=NMAX):
    """返回 (tau, J_end, J_0, trace_head)。tau = 首次达到 ½(J0−Jss_healthy) 的步数。"""
    U = np.array([[1.0], [0.0]])                       # ∝ e1 ⇒ ⊥ v
    VV = np.array([[math.sin(theta)], [math.cos(theta)]])   # kᵀVV = sinθ
    W0 = U @ VV.T
    j0 = J(W0, W0)
    jss = J(np.array([[0.0, 0.0], [1.0 / (1.0 + LAM), 0.0]]), W0)   # 闭式最优（r=1 恰好够）
    thresh = j0 + 0.5 * (jss - j0)
    tau = None
    head = []
    for t in range(1, nmax + 1):
        W = U @ VV.T
        G = (W @ K - V) @ K.T + LAM * (W - W0)
        U, VV = U - eta * (G @ VV), VV - eta * (G.T @ U)   # 同步：用同一 G
        if t % 2000 == 0 and len(head) < 40:
            head.append((t, J(U @ VV.T, W0)))
        if tau is None and J(U @ VV.T, W0) <= thresh:
            tau = t
            break
    return tau, J(U @ VV.T, W0), j0, jss, head


def run_full(theta, eta=ETA, nmax=NMAX):
    U = np.array([[1.0], [0.0]])
    VV = np.array([[math.sin(theta)], [math.cos(theta)]])
    W0 = U @ VV.T
    W = W0.copy()
    j0 = J(W, W0)
    jss = J(np.array([[0.0, 0.0], [1.0 / (1.0 + LAM), 0.0]]), W0)
    thresh = j0 + 0.5 * (jss - j0)
    tau = None
    for t in range(1, nmax + 1):
        G = (W @ K - V) @ K.T + LAM * (W - W0)
        W = W - eta * G
        if tau is None and J(W, W0) <= thresh:
            tau = t
            break
    return tau, J(W, W0), j0, jss


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    out = {"lam": LAM, "eta": ETA, "cells": []}
    for theta in (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, math.pi / 2):
        t_f, j_f, j0, jss, _ = run_fac(theta)
        t_m, j_m, _, _ = run_full(theta)
        out["cells"].append({"theta": theta, "sin_theta": math.sin(theta),
                             "tau_fac": t_f, "J_end_fac": j_f,
                             "tau_full": t_m, "J_end_full": j_m,
                             "J0": j0, "Jss": jss})
    # 斜率：log τ vs log sinθ（跳过 θ=0 与 π/2 端点）
    pts = [(math.log(c["sin_theta"]), math.log(c["tau_fac"]))
           for c in out["cells"] if 0 < c["sin_theta"] < 0.999 and c["tau_fac"]]
    n = len(pts)
    if n >= 3:
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        out["slope_loglog"] = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
        out["slope_n"] = n
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        print("  θ        sinθ      τ_fac    τ_full   J_end_fac  J_end_full   J0     Jss")
        for c in out["cells"]:
            print("%8.4g %8.4g %9s %9s %10.6f %10.6f %.4f %.6f"
                  % (c["theta"], c["sin_theta"], c["tau_fac"], c["tau_full"],
                     c["J_end_fac"], c["J_end_full"], c["J0"], c["Jss"]))
        if "slope_loglog" in out:
            print("★ log-log 斜率 (log τ vs log sinθ) = %+.3f（n=%d；预测 −1.00 ⇒ 一阶逃逸）"
                  % (out["slope_loglog"], out["slope_n"]))
