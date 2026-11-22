"""DIR-79 测量层：48 维因子空间的中心差分 Jacobian + 谱半径穿越。
派生 dir77/jac77.py（逐字同 flat/unflat 约定：U1,V1,U2,V2 顺序按列展平），装置取 dir70/ref70.py。
与 dir77 的唯一差别：本卡的自变量是 gamma>0 的闭环（W2 动态），且代表点由 warm78.warm_fp 跑到不动点。
"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir78")
import numpy as np, step72

D, R = 6, 2
SH = D * R  # 12

def flat(x):
    U1, V1, U2, V2 = x
    return np.concatenate([U1.ravel(), V1.ravel(), U2.ravel(), V2.ravel()])

def unflat(v):
    return [v[k * SH:(k + 1) * SH].reshape(D, R).copy() for k in range(4)]

def step48(v, eta, mode, g, c=6.0):
    U1, V1, U2, V2 = unflat(v)
    n1, n2, n3, n4 = step72.step(U1, V1, U2, V2, eta, mode, g, c=c)
    return np.concatenate([n1.ravel(), n2.ravel(), n3.ravel(), n4.ravel()])

def jac(x, eta, mode, g, h=1e-6, c=6.0):
    x0 = flat(x); n = x0.size; J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        J[:, j] = (step48(x0 + e, eta, mode, g, c) - step48(x0 - e, eta, mode, g, c)) / (2 * h)
    return J

def lmax(J):
    return float(np.max(np.abs(np.linalg.eigvals(J))))

def rho_scan(x, mode, g, lo=1e-6, hi=1.0, rel=2e-3, tol=1e-6, c=6.0, h=1e-6):
    """首个 lam_max > 1+tol 的 eta（log 二分）。返回 (eta_cross, 诊断)。"""
    import math
    def over(eta):
        return lmax(jac(x, eta, mode, g, h=h, c=c)) > 1.0 + tol
    a, b = math.log(lo), math.log(hi)
    fa, fb = over(math.exp(a)), over(math.exp(b))
    if fa or not fb:
        return float("nan"), dict(ok=False, f_lo=fa, f_hi=fb)
    for _ in range(60):
        if (b - a) < rel * (a + b) / 2: break
        m = 0.5 * (a + b)
        if over(math.exp(m)): b = m
        else: a = m
    return float(math.exp(0.5 * (a + b))), dict(ok=True, f_lo=fa, f_hi=fb)
