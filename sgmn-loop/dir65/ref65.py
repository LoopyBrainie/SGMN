r"""DIR-65 装置：fac_batch 边界 0.794 的机制（唯一自变量 = 时域 N 与 η）
逐字派生 DIR-64 ref64（同 d=6,r=2,λ=0.05,P=diag(1,1,0…),v=Pk,键 k_i=√c e_i）；
只把**测量层**换成 T_div / 不平衡比 t_k / 多 N 的边界，装置与更新式一字未改。
"""
import numpy as np

D, R, LAM = 6, 2, 0.05
P = np.zeros((D, D)); P[0, 0] = P[1, 1] = 1.0

def keys(c):
    return [np.sqrt(c) * np.eye(D)[i] for i in range(D)]

def Wstar(c):
    return c * P / (c + D * LAM)

def _div(W):
    return (not np.isfinite(W).all()) or float(np.linalg.norm(W)) > 1e3

def run(c, eta, n, seed=20261118, init=0.3, trace_every=100, U0=None, V0=None):
    """fac_batch 一步式：U,V ← U − η(GV), V − η(GᵀU)。返回 div/steps/td/trace/W/UV"""
    ks = keys(c); tg = [P @ k for k in ks]
    C = sum(np.outer(k, k) for k in ks) / D
    B = sum(np.outer(v, k) for v, k in zip(tg, ks)) / D
    if U0 is None:
        rng = np.random.default_rng(seed)
        U = rng.normal(size=(D, R)) * init
        V = rng.normal(size=(D, R)) * init
    else:
        U, V = U0.copy(), V0.copy()
    trace, td = [], None
    for t in range(1, n + 1):
        W = U @ V.T
        G = (W @ C - B) + LAM * W
        U = U - eta * (G @ V)
        V = V - eta * (G.T @ U)
        if t % trace_every == 0:
            nu, nv = float(np.linalg.norm(U)), float(np.linalg.norm(V))
            trace.append((t, nu / nv if nv > 0 else np.inf, nu * nv))
        if t % 25 == 0 or t == n:
            if _div(U @ V.T):
                td = t
                break
    W = U @ V.T
    return dict(div=td is not None, td=td, steps=(td or n), trace=trace, W=W,
                U=U, V=V, eta=eta, c=c, n=n)

def bisect(c, n, rel=0.002, lo=1e-3, hi=3.0):
    """在 log η 上二分到相对精度 rel。返回 (η*, 带符号的两端)"""
    import math
    a, b = math.log(lo), math.log(hi)
    fa = run(c, lo, n)["div"]; fb = run(c, hi, n)["div"]
    if fa or not fb:
        return dict(eta=float("nan"), lo=lo, hi=hi, ok=False, note="两端同号")
    for _ in range(60):
        if (b - a) < rel * (a + b) / 2:
            break
        m = 0.5 * (a + b)
        if run(c, math.exp(m), n)["div"]:
            b = m
        else:
            a = m
    return dict(eta=float(math.exp(0.5 * (a + b))), lo=float(math.exp(a)), hi=float(math.exp(b)), ok=True, note="")
