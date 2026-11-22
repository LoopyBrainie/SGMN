"""DIR-73 §7 补充 B（不进判据）：解析精确因子化（平衡代表点）上把边界往上探。"""
import sys, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70
mode, g = "target", 30.0
el = near73.eta_lin(mode, g); p = ref70.pred_target(6.0, g)
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
xs = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
x0, meas = near73.perturb(xs, 1e-3)
out = {}
for f in (1.5, 2.0, 4.0, 8.0):
    r = near73.run_from(x0, f * el, mode, g)
    out["%.1f" % f] = dict(div=r["div"], td=r["td"])
    print("  η=%.1f·η*_lin = %.6g  div=%s td=%s" % (f, f * el, r["div"], r["td"]), flush=True)
json.dump(dict(mode=mode, gamma=g, eta_lin=el, grid=out, pert=meas, note="解析精确因子化(平衡)；不进判据"),
          open("/home/ec2-user/sgmn-loop/dir73/out/grid/supp2_target_g30.json", "w"), indent=1)
print("SUPP2 done", flush=True)
