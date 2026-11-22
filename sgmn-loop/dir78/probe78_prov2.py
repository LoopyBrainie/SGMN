"""D-99 出处核验 2：LIN 表 = 因子空间 48 维 bound，在哪一点算的（bal？warm？）？"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir78"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, spec72, warm78
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
for mode, g in [("target", 3.0), ("target", 10.0), ("target", 30.0), ("keys", 3.0), ("keys", 10.0)]:
    el = near73.eta_lin(mode, g)
    p = (ref70.pred_target if mode == "target" else ref70.pred_keys)(6.0, g)
    bal = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
    xw, _ = warm78.warm_fp(mode, g, 0.3 * el)
    res = {}
    for tag, x in (("bal", bal), ("warm", xw)):
        xs = spec72.pack(*[np.array(X, float) for X in x])
        b, dg = spec72.bound(spec72.jac_factor(xs, mode, g, 1e-3), 1e-3)
        res[tag] = (b, dg["argmin"], dg["zeros"])
    print("%-6s γ=%-3g LIN=%.7g | bal界=%.7g (×%.4f) | warm界=%.7g (×%.4f) | 零模=%d/%d"
          % (mode, g, el, res["bal"][0], res["bal"][0] / el, res["warm"][0], res["warm"][0] / el,
             res["bal"][2], res["warm"][2]), flush=True)
