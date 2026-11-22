r"""DIR-64 装置：四种更新格式的**操作域边界**（唯一自变量 = 键集尺度 c 与更新序）
装置逐字派生 DIR-63 ref63（同 d=6,r=2,λ=0.05,P=diag(1,1,0…),v=Pk），只把**更新格式**变成自变量。
"""
import numpy as np

D, R, LAM = 6, 2, 0.05
P = np.zeros((D, D)); P[0, 0] = P[1, 1] = 1.0

def keys(c):
    return [np.sqrt(c) * np.eye(D)[i] for i in range(D)]

def Wstar(c):
    return c * P / (c + D * LAM)

def Jfun(W, c):
    ks = keys(c)
    s = sum(float((W @ k - P @ k) @ (W @ k - P @ k)) for k in ks)
    return 0.5 * s / D + 0.5 * LAM * float(np.sum(W * W))

def Jstar(c):
    return Jfun(Wstar(c), c)

def _div(W):
    return (not np.isfinite(W).all()) or float(np.linalg.norm(W)) > 1e3

def run(fmt, c, eta, n, seed=20261118, init=0.3, tail=0):
    """返回 dict(div, steps, W, J, tail_J)。tail>0 时记录最后 tail 步的 J（逐键格式：每步都记）"""
    ks = keys(c); tg = [P @ k for k in ks]
    C = sum(np.outer(k, k) for k in ks) / D
    B = sum(np.outer(v, k) for v, k in zip(tg, ks)) / D
    rng = np.random.default_rng(seed)
    tailJ = []
    tailW = []
    if fmt in ("mat_batch", "mat_cycle"):
        W = np.zeros((D, D))
        for t in range(n):
            if fmt == "mat_batch":
                W = W - eta * ((W @ C - B) + LAM * W)
            else:
                i = t % D
                W = W - eta * (np.outer(W @ ks[i] - tg[i], ks[i]) + LAM * W)
            if t % 25 == 0 and _div(W):
                return dict(div=True, steps=t)
            if t >= n - tail:
                tailJ.append(Jfun(W, c)); tailW.append(W.copy())
        return dict(div=False, steps=n, W=W, J=Jfun(W, c), tail_J=tailJ, tail_W=tailW)
    U = rng.normal(size=(D, R)) * init; V = rng.normal(size=(D, R)) * init
    for t in range(n):
        W = U @ V.T
        if fmt == "fac_batch":
            G = (W @ C - B) + LAM * W
            U = U - eta * (G @ V); V = V - eta * (G.T @ U)
        else:  # fac_cycle
            i = t % D
            G = np.outer(W @ ks[i] - tg[i], ks[i]) + LAM * W
            U = U - eta * (G @ V); V = V - eta * (G.T @ U)
        if t % 25 == 0 and _div(U @ V.T):
            return dict(div=True, steps=t)
        if t >= n - tail:
            tailJ.append(Jfun(U @ V.T, c)); tailW.append((U @ V.T).copy())
    W = U @ V.T
    return dict(div=False, steps=n, W=W, J=Jfun(W, c), tail_J=tailJ, tail_W=tailW)

def bisect_init(fmt, c, init, n=100000, rel=0.005, lo=1e-4, hi=8.0, seed=20261118):
    """同 bisect，但指定初始化尺度（用于分离线性极限与盆地限制）"""
    import math
    if run(fmt, c, lo, min(n, 20000), seed, init=init)["div"]:
        return float("nan")
    if not run(fmt, c, hi, min(n, 20000), seed, init=init)["div"]:
        return float("nan")
    a, b = math.log(lo), math.log(hi)
    while b - a > rel:
        m = 0.5 * (a + b)
        if run(fmt, c, math.exp(m), n, seed, init=init)["div"]:
            b = m
        else:
            a = m
    return float(math.exp(0.5 * (a + b)))


def bisect(fmt, c, n=100000, rel=0.005, lo=1e-4, hi=8.0, seed=20261118):
    """在 log η 上二分边界（假设单调：下方有界、上方发散）"""
    import math
    assert not run(fmt, c, lo, min(n, 20000), seed)["div"], "下界已发散"
    if not run(fmt, c, hi, min(n, 20000), seed)["div"]:
        return dict(eta=float("nan"), note="上界未发散")
    a, b = math.log(lo), math.log(hi)
    while b - a > rel:
        m = 0.5 * (a + b)
        if run(fmt, c, math.exp(m), n, seed)["div"]:
            b = m
        else:
            a = m
    return dict(eta=float(math.exp(0.5 * (a + b))))
