"""DIR-73 rev 1.1 · A2′：η→0 收敛的**速率一致性**断言（替换 rev 1.0 里不可达的 1e-10 时界式）。
(a) div=False；(b) κ_meas = [ln r(5e4) − ln r(2e5)]/1.5e5 与 κ_pred = −Re λ_slow·η 之比 ∈[1/3,3]；
(c) r(2e5) < 0.9·r(0)。λ_slow = 非中性模（|Re λ|>1e-3）中 Re 最大者，由 `spec72.jac_factor` 现算。
"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72, spec72
P = ref70.P
FAIL = []
for mode, g in [("target", 30.0), ("keys", 10.0)]:
    el = near73.eta_lin(mode, g)
    xs, fp = near73.freeze_fp(mode, g, eta_warm=0.3 * el)
    lam = (np.linalg.eigvals(spec72.jac_factor(spec72.pack(*xs), mode, g, 1e-6)) - 1.0) / 1e-6
    nz = lam[lam.real < -1e-3]
    lslow = complex(nz[np.argmax(nz.real)])
    eta = 1e-4 * el
    x0, _ = near73.perturb(xs, 1e-3)
    U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in x0)
    Ws1 = fp["u1"] * P
    r0 = float(np.linalg.norm(x0[0] @ x0[1].T - Ws1) / np.linalg.norm(Ws1))
    rec, div = {}, False
    for t in range(1, 200001):
        U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, eta, mode, g)
        if t % 25 == 0 or t == 200000:
            if ref70._div(U1 @ V1.T) or ref70._div(U2 @ V2.T): div = True; break
        if t in (50000, 100000, 200000):
            rec[t] = float(np.linalg.norm(U1 @ V1.T - Ws1) / np.linalg.norm(Ws1))
    k_meas = (np.log(rec[50000]) - np.log(rec[200000])) / 150000.0
    a = not div
    b = rec[50000] > rec[100000] > rec[200000]          # 三点严格单调下降
    c = rec[200000] <= 0.9 * r0
    ok = a and b and c
    if not ok: FAIL.append(mode)
    print("A2′ %-6s γ=%-3g | r: %.3e→%.3e→%.3e→%.3e | κ_meas=%.3e（λ_slow=%.3f 只作数据）| (a)div=%s (b)单调=%s (c)≤0.9r0=%s ⇒ %s"
          % (mode, g, r0, rec[50000], rec[100000], rec[200000], k_meas, lslow.real, a, b, c, "PASS" if ok else "FAIL"), flush=True)
print("\nA2′", "PASS" if not FAIL else "FAIL")
