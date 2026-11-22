r"""DIR-64 代数预言（开卡前，纯解析+线性化，不跑动力学）
装置：d=6,r=2,键 k_i=√c e_i(||k||²=c),λ=0.05,目标 v1=Pk,P=diag(1,1,0,0,0,0)
固定点 W*=(c+λ)^{-1}P；线性化 δ⁺=(I−ηA)δ（批）/ ∏_i(I−ηA_i)（循环，周期=6）
⇒ 边界 η* 由 ρ=1 给出。矩阵形式边界 = 2/(c+λ)（解析）。
"""
import numpy as np
D, R, LAM = 6, 2, 0.05
P = np.zeros((D, D)); P[0, 0] = P[1, 1] = 1.0
KEYS = lambda c: [np.sqrt(c) * np.eye(D)[i] for i in range(D)]

# 键 k_i=√c e_i ⇒ C=Σkkᵀ/D=(c/D)I ⇒ B=P·C=(c/D)P ⇒ W*=B(C+λ)^{-1}=cP/(c+Dλ)
def fixed_point(c):
    return c * P / (c + D * LAM)

def balanced(c):
    Ws = fixed_point(c); U0, s, Vt = np.linalg.svd(Ws)
    return U0[:, :R] * np.sqrt(s[:R]), Vt[:R, :].T * np.sqrt(s[:R])

def jac_map(c, kind):
    """kind='batch' ⇒ 单步雅可比；kind='key_i' ⇒ 第 i 键单步雅可比。返回 A 使 δ⁺=δ−ηAδ"""
    U, V = balanced(c); keys = KEYS(c)
    n = 2 * D * R; h = 1e-6
    if kind == "batch":
        C = sum(np.outer(k, k) for k in keys) / D
        B = sum(np.outer(P @ k, k) for k in keys) / D
        def G(W): return (W @ C - B) + LAM * W
    else:
        k = keys[kind]
        def G(W, k=k): return np.outer(W @ k - P @ k, k) + LAM * W
    A = np.eye(n)
    for j in range(n):
        dU = np.zeros((D, R)); dV = np.zeros((D, R))
        if j < D * R: dU.flat[j] = h
        else: dV.flat[j - D * R] = h
        outs = []
        for sg in (+1, -1):
            Ux, Vx = U + sg * dU, V + sg * dV
            Gx = G(Ux @ Vx.T)
            outs.append(np.concatenate([(Ux - (Gx @ Vx)).ravel(), (Vx - (Gx.T @ Ux)).ravel()]))
        d = (outs[0] - outs[1]) / (2 * h)
        A[:, j] = np.eye(n)[j] - d
    return A

def eta_star_batch(c):
    mu = np.linalg.eigvals(jac_map(c, "batch"))
    cand = [2 * m.real / abs(m) ** 2 for m in mu if m.real > 1e-12]
    return (min(cand) if cand else np.inf), mu

def rho_cycle(c, eta):
    Ais = [jac_map(c, i) for i in range(D)]
    M = np.eye(2 * D * R)
    for Ai in Ais: M = M @ (np.eye(2 * D * R) - eta * Ai)
    # 剔除数值上恰好=1 的边际模（U,V 的缩放/旋转规范方向），只报真正增长
    ev = np.linalg.eigvals(M)
    return max(abs(ev))

def eta_star_cycle(c):
    lo, hi = 1e-4, 3.0
    if rho_cycle(c, lo) > 1 + 1e-8: return lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rho_cycle(c, mid) > 1 + 1e-8: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)

print("  c    η*_mat(解析)  η*_fac(批,数)  η*_fac(批,解析 D/c)  η*_fac(循环)  ρ_cyc(0.05)")
for c in (1.0, 2.0, 3.0, 6.0, 12.0, 24.0):
    esb, mu = eta_star_batch(c)
    esc = eta_star_cycle(c)
    print("%-6.1f %-14.4f %-15.4f %-21.4f %-13.4f %-12.4f"
          % (c, 2 / (c / D + LAM), esb, D / c, esc, rho_cycle(c, 0.05)))
print()
print("预测（卡片 §2 用）：")
for c in (1.0, 2.0, 3.0, 6.0, 12.0, 24.0):
    esb, _ = eta_star_batch(c); esc = eta_star_cycle(c)
    print("  c=%-4.1f 循环/批=%.3f  循环/矩阵=%.3f  批/矩阵=%.3f"
          % (c, esc / esb, esc / (2 / (c / D + LAM)), esb / (2 / (c / D + LAM))))
