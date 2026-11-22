"""DIR-73 · 慢模诊断：因子空间谱的**最慢**模 vs 实测衰减率（A2 修订的依据）。"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72, spec72
P = ref70.P
for mode, g in [("target", 30.0), ("keys", 10.0)]:
    el = near73.eta_lin(mode, g)
    xs, fp = near73.freeze_fp(mode, g, eta_warm=0.3 * el)
    Jf = spec72.jac_factor(spec72.pack(*xs), mode, g, 1e-6)
    ev = np.linalg.eigvals(Jf)
    lam = (ev - 1.0) / 1e-6
    lam = lam[np.abs(lam) > 1e-6]
    order = np.argsort(-lam.real)          # 最接近 0 的在前
    slow = [(float(lam[i].real), float(lam[i].imag)) for i in order[:4]]
    fast = [(float(lam[i].real), float(lam[i].imag)) for i in np.argsort(lam.real)[:3]]
    eta = 1e-4 * el
    x0, _ = near73.perturb(xs, 1e-3)
    U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in x0)
    rec = []
    for t in range(1, 200001):
        U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, eta, mode, g)
        if t in (20000, 50000, 100000, 200000):
            W1 = U1 @ V1.T
            rec.append((t, float(np.linalg.norm(W1 - fp["u1"] * P) / np.linalg.norm(fp["u1"] * P))))
    k = (np.log(rec[1][1]) - np.log(rec[-1][1])) / (rec[-1][0] - rec[1][0])
    print("%-6s γ=%-3g η=%.3e | 最慢 4 模 λ: %s\n%24s最快 3 模 λ: %s\n%24sκ_meas=%.3e  κ_pred=−λ_slow·η=%.3e  r(t)=%s"
          % (mode, g, eta, ["%.2f%+.2fi" % (a, b) for a, b in slow], "", ["%.1f%+.1fi" % (a, b) for a, b in fast], "",
             k, -slow[0][0] * eta, [(t, "%.2e" % r) for t, r in rec]), flush=True)
