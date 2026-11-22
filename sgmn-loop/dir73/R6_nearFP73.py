"""DIR-73 正式轮单格：冻结 FP → 扰动 δ → 二分 → 比值 vs D-93 的 η*_lin。
用法: R6_nearFP73.py <mode> <gamma> <lo> <hi> [delta]
"""
import sys, os, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
import near73
mode, gamma = sys.argv[1], float(sys.argv[2])
lo, hi = float(sys.argv[3]), float(sys.argv[4])
delta = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
OUT = "/home/ec2-user/sgmn-loop/dir73/out/grid"; os.makedirs(OUT, exist_ok=True)
el = near73.eta_lin(mode, gamma)
xs, fp = near73.freeze_fp(mode, gamma, eta_warm=0.3 * el)
x0, meas = near73.perturb(xs, delta)
r = near73.bisect_near(mode, gamma, x0, lo, hi)
rec = dict(mode=mode, gamma=gamma, delta=delta, eta_lin=el, eta_near=r["eta"],
           ratio=(r["eta"] / el if r["ok"] else None), ok=r["ok"], note=r["note"],
           n_eval=r["n_eval"], bracket=[r["lo"], r["hi"]], hist=r["hist"],
           fp_res=fp["res"], fp_steps=fp["steps"], pert_meas=meas, eta_warm=0.3 * el)
tag = "near_%s_g%g_d%g" % (mode, gamma, delta)
json.dump(rec, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
print("DONE %s | η*_near=%.6g η*_lin=%.6g ratio=%.4f ok=%s n_eval=%d bracket=[%.6g,%.6g]"
      % (tag, r["eta"], el, (r["eta"] / el if r["ok"] else float("nan")), r["ok"], r["n_eval"], r["lo"], r["hi"]), flush=True)
