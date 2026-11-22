"""DIR-77 诊断 D-λ（rev 1.0 的实现修正：中性模 |ρ|≡1 使粗糙内插失效 ⇒ 改用 λ_max>1+1e-6 的首次穿越 + 二分）。
不进判据，只报读数。"""
import sys, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir77"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, jac77
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
CELLS = [("target", 3.0), ("target", 10.0), ("target", 30.0), ("keys", 3.0), ("keys", 10.0)]
TOL = 1e-6
out = {}
for mode, g in CELLS:
    el = near73.eta_lin(mode, g)
    p = (ref70.pred_target if mode == "target" else ref70.pred_keys)(6.0, g)
    xb = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
    lam = lambda r: float(abs(jac77.top_pair(jac77.jac(xb, r * el, mode, g))[0]))
    xs = [1.00 + 0.005 * i for i in range(0, 61)]           # 1.00 … 1.30
    ys = [lam(r) for r in xs]
    rl, curve = float("nan"), list(zip([round(x, 4) for x in xs], [round(y, 6) for y in ys]))
    for i in range(len(xs) - 1):
        if ys[i] <= 1 + TOL < ys[i + 1]:
            a, b = xs[i], xs[i + 1]
            for _ in range(40):                              # 二分 λ_max = 1+TOL
                m = 0.5 * (a + b)
                if lam(m) > 1 + TOL: b = m
                else: a = m
            rl = 0.5 * (a + b); break
    out["%s_%g" % (mode, g)] = dict(rho_lambda=rl, lam_at_1_05=lam(1.05), lam_at_1_10=lam(1.10), curve=curve)
    print("%-6s γ=%-3g ρ_λ=%s  λ(1.05η*)=%.6f λ(1.10η*)=%.6f" %
          (mode, g, ("%.4f" % rl) if rl == rl else "n/a", lam(1.05), lam(1.10)), flush=True)
json.dump(out, open("/home/ec2-user/sgmn-loop/dir77/out/diag_lam.json", "w"), indent=1)
