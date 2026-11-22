"""DIR-74 正式轮：单格线性放大曲线。派生 spec74.curves（原样）；参数 (mode, gamma, f=η/η*_lin, rep)。"""
import sys, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir74"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, spec74, near73, ref70
mode, g, f, rep = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
el = near73.eta_lin(mode, g); eta = f * el
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
if rep == "bal":
    p = ref70.pred_target(6.0, g) if mode == "target" else ref70.pred_keys(6.0, g)
    xs = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
else:
    xs, _fp = near73.freeze_fp(mode, g, eta_warm=0.3 * el)
Rm = spec74.readout(xs); Rw, r = spec74.whiten(Rm)
d0, meas = spec74.delta0(xs)
best, curve = spec74.curves(xs, mode, g, eta, d0, nmax=200000, want_G=True, Rw=Rw)
out = dict(mode=mode, gamma=g, f=f, rep=rep, eta=eta, eta_lin=el, pert=meas, rank=r,
           g_max=best["g"], t_star=best["t"], absmax=best["absmax"], Den=best["Den"], g_req=best["g_req"],
           ratio=best["g"] / best["g_req"], G_max=best["G"], t_G=best["tG"], n_eval=best["n_eval"],
           curve=curve,
           M_spec_radius=float(max(abs(np.linalg.eigvals(spec74.spec72.jac_factor(spec74.spec72.pack(*xs), mode, g, eta)))))
           )
json.dump(out, open("/home/ec2-user/sgmn-loop/dir74/out/grid/gain_%s_g%g_f%.4f_%s.json" % (mode, g, f, rep), "w"), indent=1)
print("DONE %-6s γ=%-3g f=%.4f %-4s | g_max=%.3e t*=%-6s G_max=%.3e tG=%-6s | G_req=%.3e ratio=%.3g | |ρ|=%.6f"
      % (mode, g, f, rep, best["g"], best["t"], best["G"], best["tG"], best["g_req"], best["g"] / best["g_req"], out["M_spec_radius"]), flush=True)
