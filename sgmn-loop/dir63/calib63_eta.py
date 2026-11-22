r"""标定 η（不进判据）：因子化步在 C_x=I（||k||²=6）下的稳定域；输出用于 rev 1.1 冻结"""
import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir63")
from ref63 import D, LAM, KEYS, P, v1, v0, J_layer, Jd_analytic, Jc_analytic
tgt1 = [v1(k) for k in KEYS]
Jd = Jd_analytic()
print("η        t=2e4 rel       t=1e5 rel       t=2e5 rel      finite")
for eta in (0.05, 0.02, 0.01, 0.005, 0.002):
    rng = np.random.default_rng(20261119)
    U = rng.normal(size=(D, 2)) * 0.3; V = rng.normal(size=(D, 2)) * 0.3
    out = []
    for t in range(1, 200001):
        k = KEYS[t % D]
        G = ((U @ V.T) @ k - tgt1[t % D]) @ k.T + LAM * (U @ V.T)
        U = U - eta * (G @ V); V = V - eta * (G.T @ U)
        if t in (20000, 100000, 200000):
            J = J_layer(U @ V.T, KEYS, tgt1)
            out.append(abs(J - Jd) / Jd if np.isfinite(J) else np.nan)
    print("%-8.3f %-14.3e %-14.3e %-14.3e %s" % (eta, out[0], out[1], out[2], "yes" if all(np.isfinite(out)) else "NO"))
