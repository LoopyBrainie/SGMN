"""D-99 出处核验：LIN 表（eta_lin）是否 = spec72.bound(jac_field(bal W), η)？"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, spec72
for mode, g in [("target", 3.0), ("target", 10.0), ("target", 30.0), ("keys", 3.0), ("keys", 10.0)]:
    p = (ref70.pred_target if mode == "target" else ref70.pred_keys)(6.0, g)
    u1, u2 = p["u1"], p["u2"]
    W1 = np.zeros((6, 6)); W1[0, 0] = W1[1, 1] = u1
    W2 = np.zeros((6, 6)); W2[0, 0] = W2[1, 1] = u2
    out = []
    for eta in (1e-3, 1e-4):
        J = spec72.jac_field(W1, W2, mode, g, eta)
        b, diag = spec72.bound(J, eta)
        out.append(b)
    print("%-6s γ=%-3g eta_lin表=%.7g | field界(η=1e-3)=%.7g | (η=1e-4)=%.7g | 相对差=%.2e | 零模=%d 绑定λ=%s"
          % (mode, g, near73.eta_lin(mode, g), out[0], out[1], abs(out[0] / near73.eta_lin(mode, g) - 1),
             diag["zeros"], np.round(diag["argmin"], 6)), flush=True)
