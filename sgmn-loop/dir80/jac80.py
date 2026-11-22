"""DIR-80 测量层：`dir79/jac79.py` 逐字派生。
唯一改动 = `rho_scan` 的终止判据：`(b-a) < rel*(a+b)/2` → `(b-a) <= rel`。
理由（`DEF-79-4`）：log 坐标下区间宽度就是 b-a，而 (a+b)/2 在 eta<1 时为负 ⇒ 右端为负
⇒ rel 判据永不触发，ref69/ref70/jac79 都被 `for _ in range(80/60)` 的上限掩盖，
实际是跑满到机器精度。本派生版把上限变成**可检测的 cap_hit**（由 Stage A 的 A8 断言）。
flat/unflat/step48/jac 直接 import jac79（同一约定），不复制。
"""
import sys
for q in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79", "dir80"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import math
import numpy as np
import jac79

flat, unflat, step48, jac, lmax = jac79.flat, jac79.unflat, jac79.step48, jac79.jac, jac79.lmax


def rho_scan(x, mode, g, lo=1e-6, hi=1.0, rel=2e-3, tol=1e-6, c=6.0, h=1e-6, nmax=60):
    """首个 lam_max > 1+tol 的 eta（log 二分）。返回 (eta_cross, 诊断)。
    与 jac79.rho_scan 的唯一差别：终止判据用 (b-a) <= rel（DEF-79-4）。"""
    def over(eta):
        return lmax(jac(x, eta, mode, g, h=h, c=c)) > 1.0 + tol
    a, b = math.log(lo), math.log(hi)
    fa, fb = over(math.exp(a)), over(math.exp(b))
    if fa or not fb:
        return float("nan"), dict(ok=False, f_lo=fa, f_hi=fb)
    nev, cap = 0, False
    while True:
        if (b - a) <= rel: break
        if nev >= nmax: cap = True; break
        m = 0.5 * (a + b); nev += 1
        if over(math.exp(m)): b = m
        else: a = m
    eta = float(math.exp(0.5 * (a + b)))
    return eta, dict(ok=True, f_lo=fa, f_hi=fb, nev=nev, cap_hit=cap, width=b - a,
                     lmax_at=float(lmax(jac(x, eta, mode, g, h=h, c=c))),
                     lmax_half=float(lmax(jac(x, 0.5 * eta, mode, g, h=h, c=c))))


def rho_nn(J, tol=1e-6):
    """**非中性**谱半径：绕开 8 个规范中性模（|λ| ≡ 1）——DEF-79-2 的教训。"""
    w = np.linalg.eigvals(J)
    nn = np.abs(w[np.abs(w - 1.0) > tol])
    return float(np.max(nn)) if nn.size else float("nan")


def supjt(J, T=5000, marks=(1, 10, 100, 1000, 5000)):
    """sup_{1<=t<=T} ||J^t||_2 与峰时。DEF-79-8：lambda_max 含 8 个中性模 ≡1 ⇒ 不除 lambda^t，
    直接报 ||J^t||（这就是"相对中性模的放大"）。"""
    D = J.shape[0]
    M = np.eye(D)
    best, bt = 1.0, 0
    rec = {}
    for t in range(1, T + 1):
        M = J @ M
        v = float(np.linalg.norm(M, 2))
        if v > best: best, bt = v, t
        if t in marks: rec[str(t)] = v
    return dict(sup=best, peak_t=bt, at_t=rec, interior_peak=bool(bt < T * 0.8))
