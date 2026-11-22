r"""DIR-68 测量层：受控几何方向（PAR / ORTH / MIX），幅度按 ‖U0‖,‖V0‖ 精确配平。
装置不改：调用 ref65.run(U0=...,V0=...)（逐字同一份 fac_batch 更新式）。"""
import sys, math, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir65")
import ref65 as DEV
D, R, LAM = DEV.D, DEV.R, DEV.LAM

def mix_draw(a, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(D, R)) * a, rng.normal(size=(D, R)) * a

def _proj_rescale(M, keep, ref):
    """投影到 keep 行，再按 ref 的范数重标"""
    Q = np.zeros_like(M); Q[keep] = M[keep]
    n = np.linalg.norm(Q)
    if n == 0: raise ValueError("投影为零，无法重标")
    return Q * (np.linalg.norm(ref) / n)

def init_dir(kind, a, seed):
    U, V = mix_draw(a, seed)
    if kind == "MIX":  return U, V
    keep = [0, 1] if kind == "PAR" else [2, 3, 4, 5]
    return _proj_rescale(U, keep, U), _proj_rescale(V, keep, V)

def run_dir(kind, c, eta, a, n, seed=20261118, trace_every=25):
    U0, V0 = init_dir(kind, a, seed)
    o = DEV.run(c, eta, n, seed=seed, trace_every=trace_every, U0=U0, V0=V0)
    o["maxW"] = float(np.max(np.abs(o["W"]))) if np.all(np.isfinite(o["W"])) else float("inf")
    o["U0"], o["V0"], o["kind"] = U0, V0, kind
    return o

def bisect_dir(kind, c, a, n, rel=0.005, lo=1e-3, hi=2.5, seed=20261118):
    fa = run_dir(kind, c, lo, a, n, seed)["div"]; fb = run_dir(kind, c, hi, a, n, seed)["div"]
    if fa or not fb: return dict(eta=float("nan"), ok=False, note="两端同号")
    x, y = math.log(lo), math.log(hi)
    for _ in range(80):
        if (y - x) < rel * (x + y) / 2: break
        m = 0.5 * (x + y)
        if run_dir(kind, c, math.exp(m), a, n, seed)["div"]: y = m
        else: x = m
    return dict(eta=float(math.exp(0.5 * (x + y))), lo=float(math.exp(x)), hi=float(math.exp(y)), ok=True, note="")
