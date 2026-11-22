"""DIR-73 量测层：**近不动点起点**的对数二分。
派生：`dir72/step72.py`（原样，`step()` 与 `ref70.run_stack` 逐位同）+ `dir71/R2_bisect71.py`（log 二分、rel=0.02、n=2e5、t%25 or t==n 查 `_div`）。
唯一改动 = 初值参数化 `x0 = x*_FP + δ·ξ`（ξ 固定种子、每格只生成一次）。
"""
import math, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, step72, ref70

N_DEF, SD, REL = 200000, 20261118, 0.02
LIN = {("target", 3.0): 0.0656379, ("target", 10.0): 0.0086533, ("target", 30.0): 0.00108872,
       ("keys", 3.0): 0.247608, ("keys", 10.0): 0.082765}

def eta_lin(mode, gamma):
    return LIN[(mode, float(gamma))]

def freeze_fp(mode, gamma, eta_warm, seed=SD):
    fp = step72.fp_factors(mode, gamma, eta_warm=eta_warm, seed=seed)
    return (fp["U1"], fp["V1"], fp["U2"], fp["V2"]), fp

def perturb(xstar, delta, seed=SD):
    """每个因子矩阵独立加 δ·‖X‖·ξ/‖ξ‖（相对 Frobenius 扰动 = δ）。返回 x0 与实测相对扰动。"""
    rng = np.random.default_rng(seed)
    out, meas = [], []
    for X in xstar:
        xi = rng.normal(size=X.shape)
        dX = delta * np.linalg.norm(X) * xi / np.linalg.norm(xi)
        out.append(X + dX)
        meas.append(float(np.linalg.norm(dX) / np.linalg.norm(X)))
    return tuple(out), meas

def run_from(x0, eta, mode, gamma, n=N_DEF, c=6.0):
    U1, V1, U2, V2 = (np.array(X, dtype=float, copy=True) for X in x0)
    td = None
    for t in range(1, n + 1):
        U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, eta, mode, gamma, c)
        if t % 25 == 0 or t == n:
            if ref70._div(U1 @ V1.T) or ref70._div(U2 @ V2.T):
                td = t; break
    return dict(div=td is not None, td=td, steps=(td or n))

def bisect_near(mode, gamma, x0, lo, hi, n=N_DEF, rel=REL, verbose=True):
    a, b = math.log(lo), math.log(hi)
    fa = run_from(x0, lo, mode, gamma, n)["div"]
    fb = run_from(x0, hi, mode, gamma, n)["div"]
    hist = [(float(lo), bool(fa)), (float(hi), bool(fb))]; ne = 2
    if verbose: print("  [near %s γ=%g] 端检 lo=%.6g div=%s / hi=%.6g div=%s" % (mode, gamma, lo, fa, hi, fb), flush=True)
    if fa or not fb:
        return dict(mode=mode, gamma=gamma, eta=float("nan"), lo=float("nan"), hi=float("nan"),
                    ok=False, note=("lo也发散" if fa else "hi也不发散"), n_eval=ne, hist=hist)
    for _ in range(80):
        if (b - a) <= math.log(1.0 + rel): break
        m = 0.5 * (a + b)
        d = run_from(x0, math.exp(m), mode, gamma, n)["div"]
        ne += 1; hist.append((float(math.exp(m)), bool(d)))
        if verbose: print("   试 η=%.6g div=%s" % (math.exp(m), d), flush=True)
        if d: b = m
        else: a = m
    return dict(mode=mode, gamma=gamma, eta=float(math.exp(0.5 * (a + b))),
                lo=float(math.exp(a)), hi=float(math.exp(b)), ok=True, note="", n_eval=ne, hist=hist)
