import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir64")
import ref64 as R
D, LAM = R.D, R.LAM
print("① η=0.8 的前 120 步轨迹（init=0.3）")
c = 6.0; ks = R.keys(c); tg = [R.P @ k for k in ks]
C = sum(np.outer(k, k) for k in ks) / D; B = sum(np.outer(v, k) for v, k in zip(tg, ks)) / D
rng = np.random.default_rng(20261118)
U = rng.normal(size=(D, 2)) * 0.3; V = rng.normal(size=(D, 2)) * 0.3
tr = []
for t in range(1, 121):
    W = U @ V.T; G = (W @ C - B) + LAM * W
    U = U - 0.8 * (G @ V); V = V - 0.8 * (G.T @ U)
    if t % 10 == 0: tr.append((t, float(np.linalg.norm(U @ V.T)), float(np.linalg.norm(U)), float(np.linalg.norm(V))))
print("  ", "  ".join("t=%d:%.2e/%.2e" % (t, w, nu) for t, w, nu, nv in tr))
print("\n② fac_batch 边界随 N 的移动（c=6, init=0.3）")
for n in (10000, 100000, 1000000):
    b = R.bisect("fac_batch", 6.0, n=n, rel=0.01, hi=2.0)
    print("   N=%-8d η*=%.5f" % (n, b["eta"]))
print("\n③ fac_cycle 边界随 N 的移动（c=6, init=0.3）")
for n in (10000, 100000):
    b = R.bisect("fac_cycle", 6.0, n=n, rel=0.01, hi=2.0)
    print("   N=%-8d η*=%.5f" % (n, b["eta"]))
