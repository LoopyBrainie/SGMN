r"""诊断：循环键序 ⇒ 单相位读数带锯齿；测"整周期均值"是否几何收敛（用于 rev 1.1 的量测协议）"""
import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir63")
from ref63 import D, LAM, KEYS, v1, J_layer, Jd_analytic
tgt1 = [v1(k) for k in KEYS]; Jd = Jd_analytic()
for eta in (0.005, 0.002):
    rng = np.random.default_rng(20261119)
    U = rng.normal(size=(D, 2)) * 0.3; V = rng.normal(size=(D, 2)) * 0.3
    print("--- η=%.3f ---" % eta)
    for t in range(1, 200001):
        k = KEYS[t % D]
        G = ((U @ V.T) @ k - tgt1[t % D]) @ k.T + LAM * (U @ V.T)
        U = U - eta * (G @ V); V = V - eta * (G.T @ U)
        if t % 20000 == 0 and t >= 20000:
            js = []
            for ph in range(D):
                kk = KEYS[ph]
                G = ((U @ V.T) @ kk - tgt1[ph]) @ kk.T + LAM * (U @ V.T)
                U2 = U - eta * (G @ V); V2 = V - eta * (G.T @ U)
                js.append(J_layer(U2 @ V2.T, KEYS, tgt1))
            rels = [abs(j - Jd) / Jd for j in js]
            print("  t=%-7d 相位均值 rel=%.3e  锯齿幅度(max-min)/J=%.3e"
                  % (t, abs(np.mean(js) - Jd) / Jd, (max(js) - min(js)) / Jd))
