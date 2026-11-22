import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir74"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
for mode, g in (("keys", 10.0), ("target", 30.0)):
    p = ref70.pred_target(6.0, g) if mode == "target" else ref70.pred_keys(6.0, g)
    pts = {"bal": tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))}
    pts["warm"], fp = near73.freeze_fp(mode, g, 0.3 * near73.eta_lin(mode, g))
    for tag, xb in pts.items():
        U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in xb)
        for t in range(100):
            U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, 1e-7, mode, g)
        d = max(np.linalg.norm(U1 - xb[0]), np.linalg.norm(V1 - xb[1]), np.linalg.norm(U2 - xb[2]), np.linalg.norm(V2 - xb[3]))
        # 一步残差 ‖G‖ 的近似：用 η=1e-7 一步的位移 / η
        Ua, Va, Ub, Vb = (np.array(X, float, copy=True) for X in xb)
        Ua, Va, Ub, Vb = step72.step(Ua, Va, Ub, Vb, 1e-7, mode, g)
        g1 = max(np.linalg.norm(Ua - xb[0]), np.linalg.norm(Va - xb[1])) / 1e-7
        print("%-6s %-4g %-4s  drift100=%.2e  ‖G‖≈%.2e  ‖W*‖=%.2e" % (mode, g, tag, d, g1,
              np.linalg.norm(xb[0] @ xb[1].T)))
