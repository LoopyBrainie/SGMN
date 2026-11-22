"""DIR-73 §7 补充（**不进判据**）：target γ=30 的 `fp_res=1.4e-4` 来自慢模未收敛。
改用**解析精确因子化** x* = (√u1*·E, √u1*·E, √u2*·E, √u2*·E), E=[e1|e2]（同一 W* 的另一代表点，
流形上且 G=0 精确成立），只做 4 个 η 点的 div 判定 ⇒ 边界区间。
"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72
P = ref70.P
mode, g = "target", 30.0
el = near73.eta_lin(mode, g)
p = ref70.pred_target(6.0, g)
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
xs = (np.sqrt(p["u1"]) * E, np.sqrt(p["u1"]) * E, np.sqrt(p["u2"]) * E, np.sqrt(p["u2"]) * E)
# 自检：x* 在流形上且是精确不动点（走 1000 步 η=1e-7 漂移必须 ~0）
U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in xs)
for t in range(1000): U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, 1e-7, mode, g)
drift = max(float(np.linalg.norm(U1 - xs[0])), float(np.linalg.norm(V1 - xs[1])))
print("自检 解析不动点漂移(1000 步 η=1e-7) = %.2e ；u1*=%.6f u2*=%.6f" % (drift, p["u1"], p["u2"]), flush=True)
x0, meas = near73.perturb(xs, 1e-3)
W1s = p["u1"] * P
print("起点 rel(W1) = %.3e" % (np.linalg.norm(x0[0] @ x0[1].T - W1s) / np.linalg.norm(W1s)), flush=True)
res = {}
for f in (0.95, 0.99, 1.01, 1.05):
    r = near73.run_from(x0, f * el, mode, g)
    res["%.2f" % f] = r["div"]
    print("  η=%.2f·η*_lin = %.6g  div=%s (td=%s)" % (f, f * el, r["div"], r["td"]), flush=True)
print("SUPP 边界区间 in [%.2f, %.2f]·η*_lin" % (max([f for f in (0.95, 0.99, 1.01, 1.05) if not res["%.2f" % f]], default=float("nan")),
                                               min([f for f in (0.95, 0.99, 1.01, 1.05) if res["%.2f" % f]], default=float("nan"))), flush=True)
import json; json.dump(dict(mode=mode, gamma=g, eta_lin=el, grid=res, drift=drift, pert=meas,
                           note="解析精确因子化代表点；不进判据"), open("/home/ec2-user/sgmn-loop/dir73/out/grid/supp_target_g30.json", "w"), indent=1)
