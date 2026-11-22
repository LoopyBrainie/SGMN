r"""DIR-62 预卡试跑：因子化动力学能否到达 §5.1 的秩-r 最优？（只用于定 η/N，不进判据）"""
import numpy as np
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir62")
from ref62 import D, LAM, k, v, preload, J, closed, svd_r

E = lambda i: np.eye(D)[:, i:i + 1]
P3 = [(1.40, E(2), E(0)), (0.60, E(3), E(1)), (0.50, E(4), E(4))]
W0 = preload(P3)
C = k @ k.T; B = v @ k.T
Rsym = np.linalg.cholesky(C + LAM * np.eye(D))
Ws = closed(W0)


def Jref(r):
    Wr = svd_r(Ws @ Rsym, r) @ np.linalg.inv(Rsym)
    return J(Wr, W0), Wr


def run(r, eta, N, seed):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(D, r)) * 0.3
    V = rng.normal(size=(D, r)) * 0.3
    for t in range(N):
        W = U @ V.T
        G = (W @ k - v) @ k.T + LAM * (W - W0)
        U = U - eta * (G @ V)
        V = V - eta * (G.T @ U)
    W = U @ V.T
    return J(W, W0), W


for r in (1, 2, 3, 4):
    Jr, Wr = Jref(r)
    for eta, N in ((0.05, 20000), (0.02, 40000)):
        outs = []
        for seed in (1, 2, 3):
            jn, W = run(r, eta, N, seed)
            outs.append((jn, abs(jn - Jr) / Jr))
        g = [float((W @ E(2)).T @ E(0)), float((W @ E(3)).T @ E(1)), float((W @ E(4)).T @ E(4))]
        print("r=%d η=%.2f N=%5d | J_ref=%.10f  J_N=%s | rel=%s | gains=%s"
              % (r, eta, N, Jr, ["%.8f" % o[0] for o in outs], ["%.1e" % o[1] for o in outs],
                 ["%.6f" % x for x in g]))
