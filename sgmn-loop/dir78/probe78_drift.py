"""DEF-78-5 判别：target 30 的 warmup 在 3e6 步内 ΔW(t) 是**收敛**还是**停在第二个不动点**。"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
mode, g = "target", 30.0
el = near73.eta_lin(mode, g)
p = ref70.pred_target(6.0, g)
bal = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
Wb1, Wb2 = bal[0] @ bal[1].T, bal[2] @ bal[3].T
x, _ = near73.freeze_fp(mode, g, 0.3 * el)
U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in x)
marks = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
k = 0
for step in range(1, 3_000_001):
    n1, n2, n3, n4 = step72.step(U1, V1, U2, V2, 0.3 * el, mode, g)
    num = max(np.linalg.norm(n1 - U1), np.linalg.norm(n2 - V1), np.linalg.norm(n3 - U2), np.linalg.norm(n4 - V2))
    den = max(np.linalg.norm(U1), np.linalg.norm(V1), np.linalg.norm(U2), np.linalg.norm(V2))
    U1, V1, U2, V2 = n1, n2, n3, n4
    if step == marks[k]:
        dW = max(np.linalg.norm(U1 @ V1.T - Wb1) / np.linalg.norm(Wb1), np.linalg.norm(U2 @ V2.T - Wb2) / np.linalg.norm(Wb2))
        print("t=%9d  res=%.3e  ΔW/‖W‖=%.6e" % (step, num / den, dW), flush=True)
        k += 1
        if k == len(marks): break
