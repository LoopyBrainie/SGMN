"""DIR-73 · A2 单独重跑（DEF-73-1：stageA73.py 里 `U1e, V1e, _, _ = step(...)` 把 U2/V2 都绑到同一个 `_`，
层 2 状态被写成 (V2,V2)，而 target 模式层 2 反馈进层 1 ⇒ 那条 2e5 步轨迹是坏的）。
本脚本用**正确解包**重算，断言与卡 §5 A2 一字不差。"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72
P = ref70.P
el = near73.eta_lin("target", 30.0)
xs, fp = near73.freeze_fp("target", 30.0, eta_warm=0.3 * el)
x0, meas = near73.perturb(xs, 1e-3)
eta = 1e-4 * el
U1, V1, U2, V2 = (np.array(X, dtype=float, copy=True) for X in x0)
for t in range(200000):
    U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, eta, "target", 30.0)
Ws1 = fp["u1"] * P; Ws2 = fp["u2"] * P
r1 = float(np.linalg.norm(U1 @ V1.T - Ws1) / np.linalg.norm(Ws1))
r2 = float(np.linalg.norm(U2 @ V2.T - Ws2) / np.linalg.norm(Ws2))
r0 = float(np.linalg.norm(x0[0] @ x0[1].T - Ws1) / np.linalg.norm(Ws1))
ok = (r1 < 1e-10) and (r2 < 1e-10)
print("%-58s %s | rel(W1)=%.3e rel(W2)=%.3e （起点 rel=%.3e）" % ("A2 η→0 从 x0 收敛到 FP（<1e-10）", "PASS" if ok else "FAIL", r1, r2, r0), flush=True)
