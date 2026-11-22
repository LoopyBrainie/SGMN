r"""诊断 2：改成**批梯度**（6 键平均）后，退化谱 w=(1,1) 是否还留锯齿？η 取多少？（用于 rev 1.1）"""
import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir63")
from ref63 import D, LAM, KEYS, P, P0, v0, v1, J_layer, Jd_analytic, Jc_analytic
def batch_direct(eta, n, seed=20261119, track=()):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(D, 2)) * 0.3; V = rng.normal(size=(D, 2)) * 0.3
    C = sum(k @ k.T for k in KEYS) / len(KEYS)
    B = sum(v1(k) @ k.T for k in KEYS) / len(KEYS)
    tgt = [v1(k) for k in KEYS]; out = {}
    for t in range(1, n + 1):
        W = U @ V.T
        G = (W @ C - B) + LAM * W
        U = U - eta * (G @ V); V = V - eta * (G.T @ U)
        if t in track:
            J = J_layer(U @ V.T, KEYS, tgt); out[t] = abs(J - Jd_analytic()) / Jd_analytic()
    return out
Jd = Jd_analytic()
print("批梯度·直连臂 (Jd=%.9f)" % Jd)
for eta in (0.05, 0.02, 0.01):
    o = batch_direct(eta, 200000, track=(2000, 10000, 50000, 200000))
    print("  η=%.3f " % eta + "  ".join("t=%d:%.2e" % (t, r) for t, r in o.items()))
# 堆叠臂 θ=0 与 90
def batch_stack(th, eta, n, seed=20261119, track=()):
    rng = np.random.default_rng(seed)
    U0 = rng.normal(size=(D, 2)) * 0.3; V0 = rng.normal(size=(D, 2)) * 0.3
    U1 = rng.normal(size=(D, 2)) * 0.3; V1 = rng.normal(size=(D, 2)) * 0.3
    t0 = [v0(k, th) for k in KEYS]; t1 = [v1(k) for k in KEYS]; out = {}
    for t in range(1, n + 1):
        W0 = U0 @ V0.T
        C0 = sum(k @ k.T for k in KEYS) / len(KEYS); B0 = sum(t0[i] @ KEYS[i].T for i in range(D)) / D
        G0 = (W0 @ C0 - B0) + LAM * W0
        U0 = U0 - eta * (G0 @ V0); V0 = V0 - eta * (G0.T @ U0)
        W0 = U0 @ V0.T
        hs = [W0 @ k for k in KEYS]
        C1 = sum(h @ h.T for h in hs) / D; B1 = sum(t1[i] @ hs[i].T for i in range(D)) / D
        W1 = U1 @ V1.T
        G1 = (W1 @ C1 - B1) + LAM * W1
        U1 = U1 - eta * (G1 @ V1); V1 = V1 - eta * (G1.T @ U1)
        if t in track:
            W1 = U1 @ V1.T; J = J_layer(W1, [W0 @ k for k in KEYS], t1)
            out[t] = abs(J - Jc_analytic(th)) / Jc_analytic(th)
    return out
for th in (0.0, np.pi / 2):
    o = batch_stack(th, 0.02, 200000, track=(2000, 10000, 50000, 200000))
    print("堆叠 θ=%2.0f° η=0.02 " % np.rad2deg(th) + "  ".join("t=%d:%.2e" % (t, r) for t, r in o.items()))
