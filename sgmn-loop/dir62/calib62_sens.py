r"""标定 A62-6 的扰动尺度 ε（参照层计算，不进判据）：g1(ε) 随非正交度增长的量级"""
import sys
import numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir62")
from ref62 import D, LAM, k, v, preload, closed, svd_r
import R1_preload62 as R
E = lambda i: np.eye(D)[:, i:i + 1]
for eps in (1e-2, 0.05, 0.1, 0.3, 0.5, 1.0):
    kP = E(2) + eps * E(0)
    pairs = [(1.40, kP, E(0)), (0.50, E(3), E(1)), (0.15, E(4), E(4))]
    W4 = svd_r(closed(preload(pairs)) @ R.RSYM, 4) @ np.linalg.inv(R.RSYM)
    g1 = float((W4 @ kP).T @ E(0))
    print("ε=%.3f  k^TkP=%.4f  gain_1=%.8f  dev=%.4e" % (eps, eps * 2, g1, abs(g1 - 1.40)))
