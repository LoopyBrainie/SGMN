r"""A63-3 FAIL 的机制诊断：收敛曲线 + 尾部斜率（不是判据，只用来定 N）"""
import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir63")
from ref63 import D, LAM, ETA, KEYS, P, v1, J_layer, Jd_analytic, Jc_analytic, generic
tgt = [v1(k) for k in KEYS]
Jd = Jd_analytic()
for init in (0.3, 2.0):
    rng = np.random.default_rng(20261119)
    U = rng.normal(size=(D, 2)) * init; V = rng.normal(size=(D, 2)) * init
    print("--- init=%.1f ---" % init)
    marks = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    hist = []
    for t in range(1, 1000001):
        k = KEYS[t % D]
        G = ((U @ V.T) @ k - tgt[t % D]) @ k.T + LAM * (U @ V.T)
        U = U - ETA * (G @ V); V = V - ETA * (G.T @ U)
        if t in marks:
            J = J_layer(U @ V.T, KEYS, tgt); rel = abs(J - Jd) / Jd
            hist.append((t, J, rel)); print("  t=%-8d J=%.9f rel=%.3e" % (t, J, rel))
    (t1, J1, r1), (t2, J2, r2) = hist[-2], hist[-1]
    print("  尾部斜率 log(rel) per decade = %.2f ⇒ 外推 rel<1e-6 需 t≈%.1e"
          % (np.log10(r2 / r1), t2 * (1e-6 / r2) ** (np.log(t2 / t1) / np.log(r2 / r1))))
