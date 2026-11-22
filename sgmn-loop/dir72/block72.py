import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, ref70, step72
P = ref70.P; eta = 1e-3; h = 1e-6
for mode, gamma in (("target", 3.0), ("target", 10.0), ("keys", 3.0)):
    p = ref70.pred_target(6.0, gamma) if mode == "target" else ref70.pred_keys(6.0, gamma)
    W1s, W2s = p["u1"] * P, p["u2"] * P
    # 方向 1: δW1 = hP, δW2 = 0
    a1, b1 = step72.field(W1s + h * P, W2s, eta, mode, gamma)
    a0, b0 = step72.field(W1s - h * P, W2s, eta, mode, gamma)
    dW1 = (a1 - a0) / (2 * h); dW2 = (b1 - b0) / (2 * h)
    M11 = (1.0 - np.trace(dW1 @ P) / 2.0) / eta; M21 = -(np.trace(dW2 @ P) / 2.0) / eta
    # 方向 2: δW1 = 0, δW2 = hP
    a1, b1 = step72.field(W1s, W2s + h * P, eta, mode, gamma)
    a0, b0 = step72.field(W1s, W2s - h * P, eta, mode, gamma)
    dW1b = (a1 - a0) / (2 * h); dW2b = (b1 - b0) / (2 * h)
    M12 = -(np.trace(dW1b @ P) / 2.0) / eta; M22 = (1.0 - np.trace(dW2b @ P) / 2.0) / eta
    Mn = np.array([[M11, M12], [M21, M22]])
    u1, u2, cb, lam = p["u1"], p["u2"], 1.0, ref70.LAM
    Ma = np.array([[cb + lam, -gamma * cb], [2 * cb * u1 * (u2 - 1.0), cb * u1 * u1 + lam]])
    print("%-6s γ=%-4g FD块=%s" % (mode, gamma, np.array2string(Mn, precision=5)))
    print("            代数=%s  Δλmax=%.2e" % (np.array2string(Ma, precision=5),
          abs(max(np.linalg.eigvals(Mn).real) - max(np.linalg.eigvals(Ma).real)) / max(np.linalg.eigvals(Ma).real)))
