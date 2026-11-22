import sys, os, json, math
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir71"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, ref70
OUT = "/home/ec2-user/sgmn-loop/dir71/out/grid"
mode, gamma, eta, n = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 200000
P = ref70.P
p = ref70.pred_target(6.0, gamma) if mode == "target" else ref70.pred_keys(6.0, gamma)
Wstar = p["u1"] * P
ks = ref70.keys(6.0); tg = [P @ k for k in ks]
C = sum(np.outer(k, k) for k in ks) / 6.0
B = sum(np.outer(v, k) for v, k in zip(tg, ks)) / 6.0
cb = 1.0
rng = np.random.default_rng(20261118)
U1 = rng.normal(size=(6, 2)) * 0.1; V1 = rng.normal(size=(6, 2)) * 0.1
U2 = rng.normal(size=(6, 2)) * 0.1; V2 = rng.normal(size=(6, 2)) * 0.1
rec = []
def dist(W): return float(np.linalg.norm(W - Wstar) / np.linalg.norm(Wstar))
div = False; td = None
for t in range(1, n + 1):
    W1 = U1 @ V1.T; W2 = U2 @ V2.T
    if gamma == 0.0: C1, B1, kin = C, B, ks
    elif mode == "target": C1, B1, kin = C, B + gamma * cb * W2, ks
    else:
        kin = [k + gamma * (W2 @ k) for k in ks]
        C1 = sum(np.outer(k, k) for k in kin) / 6.0
        B1 = sum(np.outer(v, k) for v, k in zip(tg, kin)) / 6.0
    G1 = (W1 @ C1 - B1) + ref70.LAM * W1
    U1 = U1 - eta * (G1 @ V1); V1 = V1 - eta * (G1.T @ U1)
    k2 = [W1 @ k for k in kin]
    C2 = sum(np.outer(k, k) for k in k2) / 6.0
    B2 = sum(np.outer(P @ k, k) for k in k2) / 6.0
    G2 = (W2 @ C2 - B2) + ref70.LAM * W2
    U2 = U2 - eta * (G2 @ V2); V2 = V2 - eta * (G2.T @ U2)
    if t % 500 == 0: rec.append((t, dist(U1 @ V1.T)))
    if t % 25 == 0 or t == n:
        if ref70._div(U1 @ V1.T) or ref70._div(U2 @ V2.T): div = True; td = t; break
half = [r for r in rec if r[0] >= 0.5 * n]
if len(half) >= 3 and not div:
    ts = np.array([r[0] for r in half], float); ls = np.log(np.array([max(r[1], 1e-300) for r in half]))
    A_ = np.vstack([ts, np.ones_like(ts)]).T
    slope = float(np.linalg.lstsq(A_, ls, rcond=None)[0][0])
    d5, dN = half[0][1], half[-1][1]
    ratio = float(dN / d5) if d5 > 0 else float("inf")
else:
    slope, d5, dN, ratio = float("nan"), float("nan"), float("nan"), float("nan")
verdict = "DRIFT" if (not div and math.isfinite(slope) and slope > 1e-5 and ratio > 1.05) else ("DIV" if div else "CONVERGED")
res = dict(mode=mode, gamma=gamma, eta=eta, n=n, div=div, td=td, slope=slope, d_half=d5, d_end=dN, ratio=ratio,
           u1_star=p["u1"], verdict=verdict, tail=rec[-5:])
json.dump(res, open(os.path.join(OUT, "drift_%s_g%g.json" % (mode, gamma)), "w"), indent=1)
print("DONE drift %s γ=%g η=%.6g div=%s slope=%.3e ratio=%.3f d_end=%.3e ⇒ %s"
      % (mode, gamma, eta, div, slope, ratio, dN, verdict), flush=True)
