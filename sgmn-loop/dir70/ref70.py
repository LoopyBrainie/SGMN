r"""DIR-70 装置：DIR-69 的 ref69 逐字派生；**唯一改动** = 加 `mode` 参数与 `keys` 分支。
mode="target"：B1 = B + gamma*cb*W2（D-90 原样）。
mode="keys"  ：kt_i = k_i + gamma*W2 k_i；层 1 的 (C1,B1) 用 kt_i（值为 P k_i）；层 2 键 = W1 kt_i。
gamma=0 时两分支都不加项 ⇒ 逐位同 ref65。
"""
import numpy as np
D, R, LAM = 6, 2, 0.05
P = np.zeros((D, D)); P[0, 0] = P[1, 1] = 1.0

def keys(c):
    return [np.sqrt(c) * np.eye(D)[i] for i in range(D)]

def _div(W, thr=1e3):
    return (not np.isfinite(W).all()) or float(np.linalg.norm(W)) > thr

def run_stack(c, gamma, eta, n, seed=20261118, init=0.1, trace_every=100, U0=None, V0=None, mode="target"):
    ks = keys(c); tg = [P @ k for k in ks]
    C = sum(np.outer(k, k) for k in ks) / D
    B = sum(np.outer(v, k) for v, k in zip(tg, ks)) / D
    cb = c / D
    if U0 is None:
        rng = np.random.default_rng(seed)
        U1 = rng.normal(size=(D, R)) * init; V1 = rng.normal(size=(D, R)) * init
        U2 = rng.normal(size=(D, R)) * init; V2 = rng.normal(size=(D, R)) * init
    else:
        U1, V1 = U0.copy(), V0.copy(); U2, V2 = U0.copy(), V0.copy()
    trace, td = [], None
    for t in range(1, n + 1):
        W1 = U1 @ V1.T; W2 = U2 @ V2.T
        if gamma == 0.0:
            C1, B1, kin = C, B, ks
        elif mode == "target":
            C1, B1, kin = C, B + gamma * cb * W2, ks
        else:
            kin = [k + gamma * (W2 @ k) for k in ks]                  # ← 键被反馈扰动
            C1 = sum(np.outer(k, k) for k in kin) / D
            B1 = sum(np.outer(v, k) for v, k in zip(tg, kin)) / D
        G1 = (W1 @ C1 - B1) + LAM * W1
        U1 = U1 - eta * (G1 @ V1); V1 = V1 - eta * (G1.T @ U1)
        k2 = [W1 @ k for k in kin]                                    # ← 层 2 的输入 = 层 1 对（扰动）键的读出
        C2 = sum(np.outer(k, k) for k in k2) / D
        B2 = sum(np.outer(P @ k, k) for k in k2) / D
        G2 = (W2 @ C2 - B2) + LAM * W2
        U2 = U2 - eta * (G2 @ V2); V2 = V2 - eta * (G2.T @ U2)
        if t % trace_every == 0:
            nu, nv = float(np.linalg.norm(U1)), float(np.linalg.norm(V1))
            trace.append((t, nu / nv if nv > 0 else np.inf, nu * nv))
        if t % 25 == 0 or t == n:
            if _div(U1 @ V1.T) or _div(U2 @ V2.T):
                td = t; break
    W1, W2 = U1 @ V1.T, U2 @ V2.T
    return dict(div=td is not None, td=td, steps=(td or n), trace=trace, W1=W1, W2=W2,
                u1=float(W1[0, 0]), u2=float(W2[0, 0]), gamma=gamma, eta=eta, c=c, n=n, mode=mode)

def pred_target(c, g):
    cb, lam = c / D, LAM
    u2 = 1.0
    for _ in range(200):
        u1 = cb * (1 + g * u2) / (cb + lam)
        u2 = u1 * u1 * cb / (u1 * u1 * cb + lam)
    return dict(u1=cb * (1 + g * u2) / (cb + lam), u2=u2)

def pred_keys(c, g):
    cb, lam = c / D, LAM
    s = 1.0 + g * 0.95
    for _ in range(500):
        m = s * s / (s * s + lam)
        u2 = m * m / (m * m + lam)
        s = 1.0 + g * u2
    return dict(u1=s / (s * s + lam), u2=u2, s=s)
