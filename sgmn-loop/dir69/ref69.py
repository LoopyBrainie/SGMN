r"""DIR-69 装置：双层 fac_batch + 跨层反馈 γ（唯一新增自由度）。
逐字派生 dir65/ref65.py：**层 1 的更新式与 ref65 一字不差**（γ=0 时分支跳过反馈项 ⇒ 逐位同轨迹）；
层 2 是同一份 fac_batch 更新，键取层 1 的读出 k2_i = W1 k_i、值 v2_i = P k2_i（闭环由 γ 项完成）。
"""
import numpy as np
D, R, LAM = 6, 2, 0.05
P = np.zeros((D, D)); P[0, 0] = P[1, 1] = 1.0

def keys(c):
    return [np.sqrt(c) * np.eye(D)[i] for i in range(D)]

def _div(W, thr=1e3):
    return (not np.isfinite(W).all()) or float(np.linalg.norm(W)) > thr

def run_stack(c, gamma, eta, n, seed=20261118, init=0.1, trace_every=100, U0=None, V0=None):
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
        B1 = B if gamma == 0.0 else B + gamma * cb * W2          # ← 唯一新增项（闭环）
        G1 = (W1 @ C - B1) + LAM * W1
        U1 = U1 - eta * (G1 @ V1); V1 = V1 - eta * (G1.T @ U1)
        k2 = [W1 @ k for k in ks]                                # ← 层 2 的输入 = 层 1 的读出（用步首 W1）
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
                u1=float(W1[0, 0]), u2=float(W2[0, 0]), gamma=gamma, eta=eta, c=c, n=n)

def bisect(gamma, c, n, rel=0.005, lo=1e-3, hi=3.0, seed=20261118, init=0.1):
    import math
    fa = run_stack(c, gamma, lo, n, seed=seed, init=init)["div"]
    fb = run_stack(c, gamma, hi, n, seed=seed, init=init)["div"]
    if fa or not fb:
        return dict(eta=float("nan"), ok=False, note="两端同号")
    a, b = math.log(lo), math.log(hi)
    for _ in range(80):
        if (b - a) < rel * (a + b) / 2: break
        m = 0.5 * (a + b)
        if run_stack(c, gamma, math.exp(m), n, seed=seed, init=init)["div"]: b = m
        else: a = m
    return dict(eta=float(math.exp(0.5 * (a + b))), ok=True, note="")

def pred(gamma):
    """卡片 §2 的代数预言：u1*, u2*, r_pred"""
    a = 6 / (6 + 6 * LAM)
    u1 = a
    for _ in range(2000):
        u2 = 6 * u1 * u1 / (6 * u1 * u1 + D * LAM)
        n1 = a * (1 + gamma * u2)
        if abs(n1 - u1) < 1e-15: break
        u1 = n1
    u2 = 6 * u1 * u1 / (6 * u1 * u1 + D * LAM)
    r = np.log(1e3 / u1) / np.log(1e3 / a)
    return dict(u1=u1, u2=u2, r=float(r))
