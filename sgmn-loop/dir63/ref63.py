r"""DIR-63 参照层（rev 1.0 §2 的闭式；不含动力学）"""
import numpy as np

D, LAM, ETA, N = 6, 0.05, 0.05, 200000
SEEDS = [20261119 + i for i in range(5)]
THETAS = [0.0, 30.0, 45.0, 60.0, 90.0]
KEYS = [np.sqrt(D) * np.eye(D)[:, i:i + 1] for i in range(D)]   # C_x = I 精确
I = np.eye(D)
P = np.zeros((D, D)); P[0, 0] = P[1, 1] = 1.0                    # v1 = P k
e = lambda i: np.eye(D)[:, i:i + 1]


def P0(th):
    a = np.cos(th) * e(0) + np.sin(th) * e(2)
    return e(0) @ a.T + e(1) @ e(1).T


def v0(k, th): return P0(th) @ k
def v1(k): return P @ k


def J_layer(W, keys, targets):
    s = 0.0
    for k, v in zip(keys, targets):
        r = W @ k - v
        s += 0.5 * float(r.T @ r)
    return s / len(keys) + 0.5 * LAM * float(np.sum(W * W))


def Jd_analytic(): return LAM / (1 + LAM)


def Jc_analytic(th):
    return 1 - (1 + np.cos(th) ** 2) / (2 * (1 + LAM * (1 + LAM) ** 2))


def generic(th):
    """通用矩阵闭式：W0* = P0 (Cx+λI)^{-1}；C1 = W0 Cx W0^T；W1* = B1 (C1+λI)^{-1}"""
    Cx = I
    W0 = P0(th) @ np.linalg.inv(Cx + LAM * I)
    W1d = P @ np.linalg.inv(Cx + LAM * I)
    Jd = J_layer(W1d, KEYS, [v1(k) for k in KEYS])
    J0 = J_layer(W0, KEYS, [v0(k, th) for k in KEYS])
    C1 = W0 @ Cx @ W0.T
    B1 = P @ Cx @ W0.T
    W1 = B1 @ np.linalg.inv(C1 + LAM * I)
    keys1 = [W0 @ k for k in KEYS]
    J1 = J_layer(W1, keys1, [v1(k) for k in KEYS])
    return dict(W0=W0, W1=W1, W1d=W1d, J0=J0, J1=J1, Jd=Jd,
                r0=int(np.linalg.matrix_rank(W0, tol=1e-9)),
                r1=int(np.linalg.matrix_rank(W1, tol=1e-9)))
