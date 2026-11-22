"""DIR-78 诊断（DEF-77-1 的更正测法）：target 30 · bal · η=1.10η*_lin，5 种子。t_div vs ln(1/|cos|)。"""
import sys, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir78"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir77")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, jac77, rho78
mode, g = "target", 30.0
SEEDS = [20261118, 20261119, 20261120, 20261121, 20261122]
el = near73.eta_lin(mode, g); eta = 1.10 * el
p = ref70.pred_target(6.0, g)
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
bal = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
J = jac77.jac(bal, eta, mode, g)
lam, v = jac77.top_pair(J); lam = abs(lam)
rows = []
for sd in SEEDS:
    x0, _ = near73.perturb(bal, 1e-3, seed=sd)
    r = near73.run_from(x0, eta, mode, g, n=200000)
    pv = jac77.flat(x0) - jac77.flat(bal)
    c = float(abs(np.vdot(pv, v)) / (np.linalg.norm(pv) * np.linalg.norm(v)))
    rows.append(dict(seed=sd, cos=c, td=r["td"], div=r["div"]))
    print("seed=%d |cos|=%.6f t_div=%s div=%s" % (sd, c, r["td"], r["div"]), flush=True)
xs = np.array([np.log(1.0 / r["cos"]) for r in rows]); ys = np.array([r["td"] for r in rows], float)
A = np.vstack([xs, np.ones_like(xs)]).T
sol, *_ = np.linalg.lstsq(A, ys, rcond=None)
pred = A @ sol; ss = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
print("FIT 斜率=%.4f  截距=%.3f  R²=%.5f  预测斜率 1/lnλ=%.4f（λ=%.6f）  ±15%%=[%.3f,%.3f]"
      % (sol[0], sol[1], ss, 1 / np.log(lam), lam, (1 / np.log(lam)) * 0.85, (1 / np.log(lam)) * 1.15), flush=True)
json.dump(dict(eta_ratio=1.10, lam=lam, slope=float(sol[0]), icpt=float(sol[1]), r2=float(ss),
               pred_slope=float(1 / np.log(lam)), rows=rows),
          open("/home/ec2-user/sgmn-loop/dir78/out/diag_tdiv.json", "w"), indent=1)
