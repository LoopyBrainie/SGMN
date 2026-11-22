"""DIR-78 量测层：rho_lambda(warm/bal 代表点) —— 逐字沿用 dir77/diag77.py 的"首个 λ>1+1e-6 细扫+二分"路径。"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir77"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, step72, jac77
TOL = 1e-6

def lam_of(J):
    return float(abs(jac77.top_pair(J)[0]))

def power_lam(J, iters=20):
    v = np.ones(J.shape[0]); v /= np.linalg.norm(v)
    for _ in range(iters):
        v = J @ v; n = np.linalg.norm(v)
        if n == 0: return 0.0
        v /= n
    return float(np.linalg.norm(J @ v))

def rho_lam(xb, eta_lin, mode, g, lo=0.30, hi=1.30, step=0.005, h=1e-9):
    """细扫找首个 λ>1+TOL，再二分到机器精度。返回 (rho, lam(1.05), lam(1.10), ln λ 的 dλ/dρ)。"""
    lam = lambda r: lam_of(jac77.jac(xb, r * eta_lin, mode, g, h=h))
    xs = [lo + step * i for i in range(int(round((hi - lo) / step)) + 1)]
    ys = [lam(r) for r in xs]
    rho = float("nan")
    for i in range(len(xs) - 1):
        if ys[i] <= 1 + TOL < ys[i + 1]:
            a, b = xs[i], xs[i + 1]
            for _ in range(42):
                m = 0.5 * (a + b)
                if lam(m) > 1 + TOL: b = m
                else: a = m
            rho = 0.5 * (a + b); break
    l105, l110 = lam(1.05), lam(1.10)
    slope = float("nan") if rho != rho else (lam(rho + 0.01) - lam(rho - 0.01)) / 0.02
    return rho, l105, l110, slope

def toy_rho(b, lo=0.05, hi=2.0, step=0.005):
    """闭式工作量例：x'=(1-ηb)x ⇒ J=1-ηb，穿越 η=1/b。走同一条细扫+二分路径。"""
    lam = lambda r: abs(1.0 - r * b)
    xs = [lo + step * i for i in range(int(round((hi - lo) / step)) + 1)]
    ys = [lam(r) for r in xs]
    for i in range(len(xs) - 1):
        if ys[i] <= 1 + TOL < ys[i + 1]:
            a, b_ = xs[i], xs[i + 1]
            for _ in range(60):
                m = 0.5 * (a + b_)
                if lam(m) > 1 + TOL: b_ = m
                else: a = m
            return 0.5 * (a + b_)
    return float("nan")
