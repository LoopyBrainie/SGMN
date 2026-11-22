"""诊断 DEF-78-5 假设：target 30 的 5.5e-6 是不是**解析 bal 点自身的离流形残差**？"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir78"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72, warm78
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
for mode, g in [("target", 3.0), ("target", 10.0), ("target", 30.0), ("keys", 3.0), ("keys", 10.0)]:
    el = near73.eta_lin(mode, g)
    p = (ref70.pred_target if mode == "target" else ref70.pred_keys)(6.0, g)
    bal = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
    xw, info = warm78.warm_fp(mode, g, 0.3 * el)
    a1, b1, a2, b2 = step72.step(*[np.array(X, float) for X in bal], 0.0, mode, g)   # η=0 只是取符号占位
    a1, b1, a2, b2 = step72.step(*[np.array(X, float) for X in bal], 1e-9, mode, g)
    num = max(np.linalg.norm(a1 - bal[0]), np.linalg.norm(b1 - bal[1]), np.linalg.norm(a2 - bal[2]), np.linalg.norm(b2 - bal[3]))
    den = max(np.linalg.norm(bal[0]), np.linalg.norm(bal[1]), np.linalg.norm(bal[2]), np.linalg.norm(bal[3]))
    res_bal = num / (1e-9 * den)
    dW = max(np.linalg.norm(bal[0] @ bal[1].T - xw[0] @ xw[1].T) / np.linalg.norm(xw[0] @ xw[1].T),
             np.linalg.norm(bal[2] @ bal[3].T - xw[2] @ xw[3].T) / np.linalg.norm(xw[2] @ xw[3].T))
    print("%-6s γ=%-3g res_bal=%.3e  ΔW/‖W‖=%.3e  ΔW/res_bal=%6.2f  warm_res0=%.2e→%.2e (%d 步)"
          % (mode, g, res_bal, dW, dW / max(res_bal, 1e-30), info["res0"], info["res"], info["steps"]), flush=True)
