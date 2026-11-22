"""DIR-83 共享测量层。Stage A 与正式轮**共用**这些函数，使 A5/A7 的结构断言
检验的正是正式轮实际调用的代码（否则「结构断言」只是注释）。

只读沿用 ref70/step72/jac79/jac81/jac82，装置一字未改。
"""
import sys, os
for q in ("dir70", "dir72", "dir73", "dir78", "dir79", "dir80", "dir81", "dir82", "dir83"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np
import ref70, step72, jac79, jac81, jac82

OUT = "/home/ec2-user/sgmn-loop/dir83/out"
C, D, R = 6.0, 6, 2
DIV_ABS = 1e3
CAP = 1e12
SD, INIT = 20261118, 0.1
SD_DELTA = 7777

# 冻结常量（卡 §1）。A8 要求**数值匹配**，不得用 %g 拼键。
FROZEN = [
    dict(g=3.0,  eta_th=0.064974679,   rho=0.068867243,    W=5.373535),
    dict(g=10.0, eta_th=0.0073034051,  rho=0.0091167253,   W=14.809432),
    dict(g=20.0, eta_th=0.0015740607,  rho=0.0025007558,   W=28.280904),
    dict(g=30.0, eta_th=0.00057804665, rho=0.001147340655, W=41.750654),
]
GAM = [3.0, 10.0, 20.0, 30.0]


def lookup(g, tol=1e-9):
    """按 γ **数值匹配**取冻结常量（A8）。"""
    hit = [r for r in FROZEN if abs(r["g"] - float(g)) <= tol]
    if len(hit) != 1:
        raise KeyError("frozen lookup failed for gamma=%r (hits=%d)" % (g, len(hit)))
    return hit[0]


def eta_mid(g):
    r = lookup(g)
    return 0.5 * (r["eta_th"] + r["rho"])


def load_fp(g):
    z = np.load(os.path.join(OUT, "fp_target_g%g.npz" % g))
    return jac82.gfix_qr((z["U1"], z["V1"], z["U2"], z["V2"]))


def wnorm(x):
    U1, V1, U2, V2 = x
    return float(np.linalg.norm(U1 @ V1.T)), float(np.linalg.norm(U2 @ V2.T))


def is_finite_state(v):
    return bool(np.isfinite(v).all())


def diverged(w1, w2, div_abs=DIV_ABS):
    """真发散判定。非有限必须走 `not isfinite` 分支，不得当数值比较（A5 / DEF-81-1）。"""
    if not (np.isfinite(w1) and np.isfinite(w2)):
        return True
    return bool(w1 > div_abs or w2 > div_abs)


def mu_of_jac(J, tol=1e-6):
    """μ = ln max{|λ| : |λ−1| > tol}；不用 eig 的特征向量基（DEF-81-6）。"""
    ev = np.linalg.eigvals(J)
    sel = np.abs(ev[np.abs(ev - 1.0) > tol])
    if sel.size == 0:
        return float("-inf"), 0, float("nan")
    return float(np.log(float(sel.max()))), int(sel.size), float(sel.max())


def lam_sup(J, T, cap=CAP):
    """sup_{t<=T} ||J^t||_2；溢出走 capped 分支而非抛异常（A7 / DEF-83-3）。"""
    n = J.shape[0]
    M = np.eye(n)
    best, bt, mk = 1.0, 0, {}
    for t in range(1, T + 1):
        try:
            M = J @ M
            s = float(np.linalg.norm(M, 2))
        except np.linalg.LinAlgError:
            return dict(sup=best, peak_t=bt, marks=mk, overflow_t=t, ok=False, capped=False)
        if not np.isfinite(s) or s > cap:
            return dict(sup=float(max(best, s if np.isfinite(s) else cap)),
                        peak_t=(t if s > best else bt), marks=mk,
                        overflow_t=t, ok=True, capped=True)
        if s > best:
            best, bt = s, t
        if t in (2000, 5000, 20000):
            mk[str(t)] = s
    return dict(sup=best, peak_t=bt, marks=mk, overflow_t=None, ok=True, capped=False)


def flat(x):
    return jac81.flat(x)


def unflat(v):
    return jac81.unflat(v)


def delta_hat(n, seed=SD_DELTA):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=n)
    return d / float(np.linalg.norm(d))


def traj_eps(g, eta, eps, N, seed=SD_DELTA):
    """从 gfix_qr(x*) + eps*||v0||*delta_hat 出发，N 步。"""
    xc = load_fp(g)
    v0 = flat(xc)
    nv0 = float(np.linalg.norm(v0))
    v = v0 + eps * nv0 * delta_hat(v0.size, seed)
    w1r, _ = wnorm(unflat(v0))
    mx1, t_nan, t_div = 0.0, None, None
    for t in range(1, N + 1):
        vn = jac79.step48(v, eta, "target", g, c=C)
        if not is_finite_state(vn):
            t_nan = t
            break
        v = vn
        w1, w2 = wnorm(unflat(v))
        if w1 > mx1:
            mx1 = w1
        if t_div is None and diverged(w1, w2):
            t_div = t
    return dict(eps=eps, eta=eta, finite=bool(t_nan is None), t_nan=t_nan, t_div=t_div,
                mx_rel=max(mx1, w1r) / w1r, w_ref=w1r, steps=(t_nan or N), N=N)


def traj_init(g, eta, N, seed=SD, init=INIT, every=None):
    """从 init=0.1 出发（逐字同 ref70.run_stack 的抽签与更新顺序）。"""
    U1, V1, U2, V2 = jac81.draw_init(init, seed)
    every = every or max(1, N // 200)
    ns, t_nan, t_div, mx = [], None, None, 0.0
    for t in range(1, N + 1):
        U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, eta, "target", g, C)
        w1 = float(np.linalg.norm(U1 @ V1.T))
        w2 = float(np.linalg.norm(U2 @ V2.T))
        if not (np.isfinite(w1) and np.isfinite(w2)):
            t_nan = t
            break
        if w1 > mx:
            mx = w1
        if t_div is None and diverged(w1, w2):
            t_div = t
        if t % every == 0:
            ns.append((t, w1))
    return dict(t_nan=t_nan, finite=bool(t_nan is None), t_div=t_div, mx=mx, ns=ns, N=N)


def mu_tail(ns, N, frac=0.8):
    tail = [(t, w) for (t, w) in ns if t >= frac * N]
    if len(tail) < 3 or any(w <= 0 for _, w in tail):
        return None
    tt = np.array([t for t, _ in tail], float)
    yy = np.log(np.array([w for _, w in tail], float))
    v = float(np.polyfit(tt, yy, 1)[0])
    return v if np.isfinite(v) else None