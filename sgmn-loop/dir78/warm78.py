"""DIR-78：warm 代表点跑到收敛（修 DEF-78-4）。从 near73.freeze_fp 出发，继续 step72.step 迭代至
‖step(x) − x‖/‖x‖ ≤ tol。只迭代、不改装置。"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, step72, near73

def warm_fp(mode, gamma, eta_warm, tol=1e-15, nmax=10_000_000):
    x, fp0 = near73.freeze_fp(mode, gamma, eta_warm)
    U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in x)
    res = float("inf")
    for k in range(nmax):
        n1, n2, n3, n4 = step72.step(U1, V1, U2, V2, eta_warm, mode, gamma)
        num = max(np.linalg.norm(n1 - U1), np.linalg.norm(n2 - V1), np.linalg.norm(n3 - U2), np.linalg.norm(n4 - V2))
        den = max(np.linalg.norm(U1), np.linalg.norm(V1), np.linalg.norm(U2), np.linalg.norm(V2))
        U1, V1, U2, V2 = n1, n2, n3, n4
        res = num / den
        if res <= tol: break
    return (U1, V1, U2, V2), dict(res=res, steps=k + 1, res0=fp0["res"])
