r"""DIR-63 开卡前的代数自检（参照层，不进装置）：
两层线性堆叠，C_x=I（键 = √6 e_i 循环），
  层0 目标 v0 = (a^T x) e1 + x2 e2 ,  a = cosθ e1 + sinθ e3
  层1 目标 v1 = x1 e1 + x2 e2 ,      键 = h0 = W0 x (sg)
  直连  层1 目标同上，键 = x
解析预言（手推）：
  J_d = λ/(1+λ)
  J_c(θ) = 1 - (1+cos²θ) / (2(1+λ(1+λ)²))
"""
import numpy as np
D, LAM = 6, 0.05
P = np.zeros((D, D)); P[0,0] = P[1,1] = 1.0          # v1 的映射（也是 Q1）
def analytic(th):
    Jd = LAM/(1+LAM)
    Jc = 1 - (1+np.cos(th)**2)/(2*(1+LAM*(1+LAM)**2))
    return Jd, Jc
def generic(th):
    a = np.zeros(D); a[0] = np.cos(th); a[2] = np.sin(th)
    P0 = np.outer(np.eye(D)[:,0], a) + np.outer(np.eye(D)[:,1], np.eye(D)[:,1])
    Cx = np.eye(D)
    # 层0：W0* = P0 (Cx+λI)^{-1}
    W0 = P0 @ np.linalg.inv(Cx + LAM*np.eye(D))
    # 直连臂：W1d* = P (Cx+λI)^{-1}
    W1d = P @ np.linalg.inv(Cx + LAM*np.eye(D))
    Jd = 0.5*np.trace(W1d@Cx@W1d.T) - np.trace(W1d@(P@Cx).T) + 0.5*np.trace(P@Cx@P.T) + 0.5*LAM*np.trace(W1d@W1d.T)
    # 堆叠臂：C1 = W0 Cx W0^T, B1 = P Cx W0^T
    C1 = W0 @ Cx @ W0.T; B1 = P @ Cx @ W0.T
    W1 = B1 @ np.linalg.inv(C1 + LAM*np.eye(D))
    Jc = 0.5*np.trace(W1@C1@W1.T) - np.trace(W1@B1.T) + 0.5*np.trace(P@Cx@P.T) + 0.5*LAM*np.trace(W1@W1.T)
    return Jd, Jc, np.linalg.matrix_rank(W0, tol=1e-9), np.linalg.matrix_rank(W1, tol=1e-9)
print("θ°    Jd(解析)  Jd(矩阵)   Jc(解析)  Jc(矩阵)   |ΔJc|      rank(W0,W1)")
for th in np.deg2rad([0, 30, 45, 60, 90]):
    (a1, a2) = analytic(th); (g1, g2, r0, r1) = generic(th)
    print("%5.1f  %.9f %.9f  %.9f %.9f  %.2e   %d,%d" % (np.rad2deg(th), a1, g1, a2, g2, abs(a2-g2), r0, r1))
j0 = analytic(0)[1]; j90 = analytic(np.pi/2)[1]; jd = analytic(0)[0]
print("\nΔJ(0)=%.6f  ΔJ(90°)=%.6f  ΔJ(90)/Jd=%.3f" % (j0-jd, j90-jd, (j90-jd)/jd))
