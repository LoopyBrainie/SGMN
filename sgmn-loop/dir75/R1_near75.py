"""DIR-75 正式轮：平衡 gauge 上的近 FP 对数二分。`near73.bisect_near` 原样，唯一改动 = 代表点。"""
import sys, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir74"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70
mode, g = sys.argv[1], float(sys.argv[2])
LO, HI = (0.90, 1.60) if mode == "target" else (0.90, 1.30)
el = near73.eta_lin(mode, g)
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
p = ref70.pred_target(6.0, g) if mode == "target" else ref70.pred_keys(6.0, g)
xb = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
x0, meas = near73.perturb(xb, 1e-3)
print("REP %s γ=%g bal  U1=U2^T=√u1*E, V=同  u1*=%.9f u2*=%.9f  ‖W1‖=%.6e ‖Δ‖=%.1e" %
      (mode, g, p["u1"], p["u2"], np.linalg.norm(xb[0] @ xb[1].T), max(meas)), flush=True)
bs = near73.bisect_near(mode, g, x0, LO * el, HI * el, rel=0.02)
ratio = bs["eta"] / el
out = dict(mode=mode, gamma=g, rep="bal", u1=p["u1"], u2=p["u2"], eta_lin=el,
           eta=bs["eta"], lo=bs["lo"], hi=bs["hi"], ratio=ratio, ok=bs["ok"], note=bs["note"],
           n_eval=bs["n_eval"],
           hist=[dict(eta=a, eta_l=a / el, div=b) for a, b in bs["hist"]])
json.dump(out, open("/home/ec2-user/sgmn-loop/dir75/out/%s_g%g.json" % (mode, g), "w"), indent=1)
print("DONE %-6s γ=%-3g bal：η_th=%.6e（ρ=%.4f，括号 [%.4f, %.4f]·η*_lin）n_eval=%d ok=%s note=%s"
      % (mode, g, bs["eta"], ratio, bs["lo"] / el, bs["hi"] / el, bs["n_eval"], bs["ok"], bs["note"] or "-"), flush=True)
