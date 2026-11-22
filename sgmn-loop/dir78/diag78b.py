"""DIR-78 诊断 rev b（修 DEF-78-6）：target 30 · bal · η=1.06η*_lin（贴近阈值使 t_div~1e3），
逐**步**调用装置自己的 `ref70._div(W,1e3)` 判据 ⇒ t_div 分辨率 = 1 步。5 种子，t_div vs ln(1/|cos|)。"""
import sys, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir78"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir77")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72, jac77
mode, g, RATIO, NMAX = "target", 30.0, 1.06, 200000
SEEDS = [20261118, 20261119, 20261120, 20261121, 20261122]
el = near73.eta_lin(mode, g); eta = RATIO * el
p = ref70.pred_target(6.0, g)
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
bal = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
J = jac77.jac(bal, eta, mode, g)
lam, v = jac77.top_pair(J); lam = abs(lam)
rows = []
for sd in SEEDS:
    x0, _ = near73.perturb(bal, 1e-3, seed=sd)
    U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in x0)
    td, div = None, False
    for k in range(1, NMAX + 1):
        U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, eta, mode, g)
        if ref70._div(U1 @ V1.T, 1e3) or ref70._div(U2 @ V2.T, 1e3):
            td, div = k, True; break
    pv = jac77.flat(x0) - jac77.flat(bal)
    c = float(abs(np.vdot(pv, v)) / (np.linalg.norm(pv) * np.linalg.norm(v)))
    rows.append(dict(seed=sd, cos=c, td=td, div=div))
    print("seed=%d |cos|=%.6f ln(1/|cos|)=%.4f t_div=%s" % (sd, c, np.log(1 / c), td), flush=True)
xs = np.array([np.log(1.0 / r["cos"]) for r in rows]); ys = np.array([r["td"] for r in rows], float)
A = np.vstack([xs, np.ones_like(xs)]).T
sol, *_ = np.linalg.lstsq(A, ys, rcond=None); pred = A @ sol
ss = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
ps = 1 / np.log(lam)
print("FIT 斜率=%.3f 截距=%.1f R²=%.5f | 预测 1/lnλ(1.06η*)=%.3f ±15%%=[%.2f,%.2f] | λ=%.6f"
      % (sol[0], sol[1], ss, ps, 0.85 * ps, 1.15 * ps, lam), flush=True)
json.dump(dict(ratio=RATIO, lam=float(lam), slope=float(sol[0]), icpt=float(sol[1]), r2=float(ss),
               pred_slope=float(ps), rows=rows), open("/home/ec2-user/sgmn-loop/dir78/out/diag_tdiv_b.json", "w"), indent=1)
