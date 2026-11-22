"""DIR-72 · 单步映射（**逐字**取自 `ref70.run_stack` 的循环体；唯一新增 = `field()` 把同一梯度场作用在 (W1,W2) 上）。
`step()` 与 `ref70.run_stack` 的差别：只有"取一步"与"取 n 步"（A6 用 20000 步逐位校验）。
"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, ref70
D, R, LAM, P = ref70.D, ref70.R, ref70.LAM, ref70.P

def _consts(c):
    ks = ref70.keys(c); tg = [P @ k for k in ks]
    C = sum(np.outer(k, k) for k in ks) / D
    B = sum(np.outer(v, k) for v, k in zip(tg, ks)) / D
    return ks, tg, C, B, c / D

def step(U1, V1, U2, V2, eta, mode, gamma, c=6.0):
    ks, tg, C, B, cb = _consts(c)
    W1 = U1 @ V1.T; W2 = U2 @ V2.T
    if gamma == 0.0:
        C1, B1, kin = C, B, ks
    elif mode == "target":
        C1, B1, kin = C, B + gamma * cb * W2, ks
    else:
        kin = [k + gamma * (W2 @ k) for k in ks]
        C1 = sum(np.outer(k, k) for k in kin) / D
        B1 = sum(np.outer(v, k) for v, k in zip(tg, kin)) / D
    G1 = (W1 @ C1 - B1) + LAM * W1
    U1 = U1 - eta * (G1 @ V1); V1 = V1 - eta * (G1.T @ U1)
    k2 = [W1 @ k for k in kin]
    C2 = sum(np.outer(k, k) for k in k2) / D
    B2 = sum(np.outer(P @ k, k) for k in k2) / D
    G2 = (W2 @ C2 - B2) + LAM * W2
    U2 = U2 - eta * (G2 @ V2); V2 = V2 - eta * (G2.T @ U2)
    return U1, V1, U2, V2

def field(W1, W2, eta, mode, gamma, c=6.0):
    """同一梯度场作用在**矩阵对**上（层 2 的输入用**旧** W1，与 step() 的 `k2 = W1 @ k` 一致）。"""
    ks, tg, C, B, cb = _consts(c)
    if gamma == 0.0:
        C1, B1, kin = C, B, ks
    elif mode == "target":
        C1, B1, kin = C, B + gamma * cb * W2, ks
    else:
        kin = [k + gamma * (W2 @ k) for k in ks]
        C1 = sum(np.outer(k, k) for k in kin) / D
        B1 = sum(np.outer(v, k) for v, k in zip(tg, kin)) / D
    G1 = (W1 @ C1 - B1) + LAM * W1
    W1n = W1 - eta * G1
    k2 = [W1 @ k for k in kin]
    C2 = sum(np.outer(k, k) for k in k2) / D
    B2 = sum(np.outer(P @ k, k) for k in k2) / D
    G2 = (W2 @ C2 - B2) + LAM * W2
    W2n = W2 - eta * G2
    return W1n, W2n

def fp_factors(mode, gamma, eta_warm, c=6.0, n_max=500000, tol=1e-11, seed=20261118, init=0.1):
    """以安全步长迭代到不动点，返回实际到达的因子化 (U1,V1,U2,V2) 与残差。"""
    p = ref70.pred_target(c, gamma) if mode == "target" else ref70.pred_keys(c, gamma)
    W1s = p["u1"] * P; W2s = p["u2"] * P
    rng = np.random.default_rng(seed)
    U1 = rng.normal(size=(D, R)) * init; V1 = rng.normal(size=(D, R)) * init
    U2 = rng.normal(size=(D, R)) * init; V2 = rng.normal(size=(D, R)) * init
    res = np.inf
    for t in range(1, n_max + 1):
        U1, V1, U2, V2 = step(U1, V1, U2, V2, eta_warm, mode, gamma, c)
        if t % 200 == 0 or t == 1:
            res = max(float(np.linalg.norm(U1 @ V1.T - W1s) / np.linalg.norm(W1s)),
                      float(np.linalg.norm(U2 @ V2.T - W2s) / np.linalg.norm(W2s)))
            if res < tol:
                break
    return dict(U1=U1, V1=V1, U2=U2, V2=V2, res=res, steps=t, u1=p["u1"], u2=p["u2"])
