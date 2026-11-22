r"""DIR-62 预卡试跑 B：两种预载强度下的收敛与逐级截断（定 γ/η/N，不进判据）"""
import sys
import numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir62")
from ref62 import D, LAM, k, v, preload, J, closed, svd_r

E = lambda i: np.eye(D)[:, i:i + 1]
C = k @ k.T
Rsym = np.linalg.cholesky(C + LAM * np.eye(D))
REG = {"WEAK": [(1.40, E(2), E(0)), (0.50, E(3), E(1)), (0.15, E(4), E(4))],
       "STRONG": [(8.00, E(2), E(0)), (6.00, E(3), E(1)), (1.00, E(4), E(4))]}


def run(U, V, W0, eta, N):
    for _ in range(N):
        W = U @ V.T
        G = (W @ k - v) @ k.T + LAM * (W - W0)
        U = U - eta * (G @ V); V = V - eta * (G.T @ U)
    return U @ V.T


for tag, pairs in REG.items():
    W0 = preload(pairs); Ws = closed(W0)
    sig = np.linalg.svd(Ws @ Rsym, compute_uv=False)
    print("\n===== %s  预载 γ=%s   σ(W*R)=%s" % (tag, [p[0] for p in pairs], np.round(sig, 6)))
    for r in (1, 2, 3, 4):
        Wr = svd_r(Ws @ Rsym, r) @ np.linalg.inv(Rsym)
        Jr = J(Wr, W0)
        gref = [float((Wr @ kj).T @ vj) / float(vj.T @ vj) for (_, kj, vj) in pairs]
        gdat = float((Wr @ k).T @ v) / float(v.T @ v)
        for eta, N in ((0.05, 20000), (0.05, 200000)):
            rels, gs, gd = [], [], []
            for seed in (1, 2, 3):
                rng = np.random.default_rng(seed)
                W = run(rng.normal(size=(D, r)) * 0.3, rng.normal(size=(D, r)) * 0.3, W0, eta, N)
                rels.append(abs(J(W, W0) - Jr) / Jr)
                gs.append([float((W @ kj).T @ vj) / float(vj.T @ vj) for (_, kj, vj) in pairs])
                gd.append(float((W @ k).T @ v) / float(v.T @ v))
            print("  r=%d η=%.2f N=%6d | J_ref=%.8f rel=%s | gain_data: ref=%.6f got_min/max=%.6f/%.6f | pre: ref=%s got=%s"
                  % (r, eta, N, Jr, ["%.0e" % x for x in rels], gdat, min(gd), max(gd),
                     ["%.3f" % x for x in gref], [["%.4f" % x for x in row] for row in gs[:1]]))
