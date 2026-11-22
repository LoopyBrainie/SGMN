"""DIR-77 正式轮：keys γ=10 固定格 × 5 种子（唯一自变量 = 种子）。装置逐字派生 dir76/R1_near76.py。"""
import sys, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir77"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir74")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, jac77
mode, g, sd = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
LO, HI = 0.85, 1.50
el = near73.eta_lin(mode, g)
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
p = ref70.pred_keys(6.0, g)
xb = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
x0, meas = near73.perturb(xb, 1e-3, seed=sd)
print("REP %s γ=%g seed=%d bal  u1*=%.9f u2*=%.9f  ‖W1‖=%.6e ‖Δ‖=%.1e" %
      (mode, g, sd, p["u1"], p["u2"], np.linalg.norm(xb[0] @ xb[1].T), max(meas)), flush=True)
bs = near73.bisect_near(mode, g, x0, LO * el, HI * el, rel=0.02)
ratio = bs["eta"] / el
J = jac77.jac(xb, bs["eta"], mode, g) if bs["ok"] else None
lam, v = jac77.top_pair(J) if J is not None else (float("nan"), None)
# |cos|：扰动向量 p 与增长方向 v
pert_vec = jac77.flat(x0) - jac77.flat(xb)
cosab = float(abs(np.vdot(pert_vec, v)) / (np.linalg.norm(pert_vec) * np.linalg.norm(v))) if v is not None else float("nan")
out = dict(mode=mode, gamma=g, seed=sd, rep="bal", u1=p["u1"], u2=p["u2"], eta_lin=el,
           eta=bs["eta"], lo=bs["lo"], hi=bs["hi"], ratio=ratio, ok=bs["ok"], note=bs["note"],
           n_eval=bs["n_eval"], lam_max=float(abs(lam)), cos=cosab,
           hist=[dict(eta=a, eta_l=a / el, div=b) for a, b in bs["hist"]])
json.dump(out, open("/home/ec2-user/sgmn-loop/dir77/out/%s_g%g_s%d.json" % (mode, g, sd), "w"), indent=1)
print("DONE seed=%d ρ=%.4f η_th=%.6e 括号[%.4f,%.4f]·η*_lin n_eval=%d ok=%s |cos|=%.6f λ_max=%.6f"
      % (sd, ratio, bs["eta"], bs["lo"] / el, bs["hi"] / el, bs["n_eval"], bs["ok"], cosab, abs(lam)), flush=True)
