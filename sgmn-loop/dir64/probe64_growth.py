import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir64")
import ref64 as R
D, LAM = R.D, R.LAM
# ① 完整谱
c = 6.0; Ws = R.Wstar(c)
U0, s, Vt = np.linalg.svd(Ws); U = U0[:, :2] * np.sqrt(s[:2]); V = Vt[:2, :].T * np.sqrt(s[:2])
ks = R.keys(c); tg = [R.P @ k for k in ks]
C = sum(np.outer(k, k) for k in ks) / D; B = sum(np.outer(v, k) for v, k in zip(tg, ks)) / D
n, h, A = 4 * D, 1e-6, np.eye(4 * D)
for j in range(n):
    dU = np.zeros((D, 2)); dV = np.zeros((D, 2))
    if j < 2 * D: dU.flat[j] = h
    else: dV.flat[j - 2 * D] = h
    o = []
    for sg in (1, -1):
        Ux, Vx = U + sg * dU, V + sg * dV
        G = ((Ux @ Vx.T) @ C - B) + LAM * (Ux @ Vx.T)
        o.append(np.concatenate([(Ux - G @ Vx).ravel(), (Vx - G.T @ Ux).ravel()]))
    A[:, j] = np.eye(n)[j] - (o[0] - o[1]) / (2 * h)
ev = np.sort_complex(np.linalg.eigvals(A))
print("A 的谱（去重到 4 位）：")
print("  ", np.unique(np.round(ev, 4)))
print("  max Re=%.6f   max|λ|=%.6f" % (ev.real.max(), np.abs(ev).max()))
# ② 增长曲线
print("\nη       ‖W‖_F at t=1e3 / 3e3 / 1e4 / 3e4 / 1e5   （指数⇒等比；代数⇒缓增）")
for eta in (0.3, 0.5, 0.8, 0.86, 0.9, 1.1):
    rng = np.random.default_rng(20261118)
    Uu = rng.normal(size=(D, 2)) * 0.3; Vv = rng.normal(size=(D, 2)) * 0.3
    marks = {1000, 3000, 10000, 30000, 100000}; got = {}
    for t in range(1, 100001):
        W = Uu @ Vv.T
        G = (W @ C - B) + LAM * W
        Uu = Uu - eta * (G @ Vv); Vv = Vv - eta * (G.T @ Uu)
        if t in marks:
            got[t] = float(np.linalg.norm(Uu @ Vv.T))
            if not np.isfinite(got[t]) or got[t] > 1e6: break
    print("%-5.2f  %s" % (eta, "  ".join("%9.3e" % got.get(t, float("nan")) for t in (1000, 3000, 10000, 30000, 100000))))
