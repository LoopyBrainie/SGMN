r"""DIR-66 测量层：η*(a) 与逃逸位置 t_esc。
装置**不改**：直接 import DIR-65 的 `ref65.run`（逐字同一份 fac_batch 更新式）。
本文件只加两件事：按初值幅度 a 扫描 + 取「最后一条有限记录」的 t（纪律 159）。
"""
import sys, math, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir65")
import ref65 as DEV

D, R, LAM = DEV.D, DEV.R, DEV.LAM

def run_a(c, eta, a, n, seed=20261118, trace_every=25):
    """a = 初值幅度（U,V ~ a·N(0,1)）。返回 t_esc（最后有限记录的 ‖U‖/‖V‖）与 T_div。"""
    o = DEV.run(c, eta, n, seed=seed, init=a, trace_every=trace_every)
    finite = [x for x in o["trace"] if np.isfinite(x[1]) and np.isfinite(x[2])]
    o["t_esc"] = finite[-1][1] if finite else float("nan")
    o["t_esc_step"] = finite[-1][0] if finite else None
    o["a"] = a
    return o

def bisect_a(c, a, n, rel=0.005, lo=1e-3, hi=3.0, seed=20261118):
    """在 log η 上二分；返回 η*(a) 与两端（0.5% 相对精度）"""
    fa = run_a(c, lo, a, n, seed)["div"]; fb = run_a(c, hi, a, n, seed)["div"]
    if fa or not fb:
        return dict(eta=float("nan"), ok=False, note="两端同号", a=a)
    x, y = math.log(lo), math.log(hi)
    for _ in range(80):
        if (y - x) < rel * (x + y) / 2: break
        m = 0.5 * (x + y)
        if run_a(c, math.exp(m), a, n, seed)["div"]: y = m
        else: x = m
    return dict(eta=float(math.exp(0.5*(x+y))), lo=float(math.exp(x)), hi=float(math.exp(y)),
                ok=True, note="", a=a)
