"""DIR-78 正式轮（warm 行）：每格 warm 代表点 → 扰动(种子 20261118, δ=1e-3) → 对数二分阈值 + ρ_λ。
装置：near73.* / step72 / ref70 全原样；jac77/rho78 逐字。"""
import sys, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir78"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir77")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, jac77, rho78, warm78
mode, g = sys.argv[1], float(sys.argv[2])
LO, HI = 0.30, 1.50
el = near73.eta_lin(mode, g)
xw, info = warm78.warm_fp(mode, g, 0.3 * el)
p = (ref70.pred_target if mode == "target" else ref70.pred_keys)(6.0, g)
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
bal = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
dW = max(np.linalg.norm(bal[0] @ bal[1].T - xw[0] @ xw[1].T) / np.linalg.norm(xw[0] @ xw[1].T),
         np.linalg.norm(bal[2] @ bal[3].T - xw[2] @ xw[3].T) / np.linalg.norm(xw[2] @ xw[3].T))
x0, meas = near73.perturb(xw, 1e-3, seed=20261118)
print("REP %s γ=%g warm  u*=%.6f/% .6f  ‖W1‖=%.6e  ΔW=%.2e  warmup %d 步 res=%.1e  ‖Δ‖=%.1e"
      % (mode, g, p["u1"], p["u2"], np.linalg.norm(xw[0] @ xw[1].T), dW, info["steps"], info["res"], max(meas)), flush=True)
rho, l105, l110, slope = rho78.rho_lam(xw, el, mode, g)
print("RHO %s γ=%g ρ_λ^warm=%.6f  λ(1.05)=%.6f λ(1.10)=%.6f  dλ/dρ=%.4f" % (mode, g, rho, l105, l110, slope), flush=True)
bs = near73.bisect_near(mode, g, x0, LO * el, HI * el, rel=0.02)
ratio = bs["eta"] / el
out = dict(mode=mode, gamma=g, gauge="warm", seed=20261118, eta_lin=el, eta=bs["eta"], lo=bs["lo"], hi=bs["hi"],
           rho_div=ratio, ok=bs["ok"], note=bs["note"], n_eval=bs["n_eval"],
           rho_lam=rho, lam_105=l105, lam_110=l110, dlam_drho=slope,
           dW=dW, warm_steps=info["steps"], warm_res=info["res"], u1=p["u1"], u2=p["u2"],
           W1=float(np.linalg.norm(xw[0] @ xw[1].T)),
           hist=[dict(eta=a, eta_l=a / el, div=b) for a, b in bs["hist"]])
json.dump(out, open("/home/ec2-user/sgmn-loop/dir78/out/warm_%s_g%g.json" % (mode, g), "w"), indent=1)
print("DONE warm %s γ=%g ρ_div=%.4f [%.4f,%.4f]·η*_lin n_eval=%d ok=%s | ρ_λ=%.4f | 相对差=%.2f%%"
      % (mode, g, ratio, bs["lo"] / el, bs["hi"] / el, bs["n_eval"], bs["ok"], rho, 100 * (ratio - rho) / rho), flush=True)
