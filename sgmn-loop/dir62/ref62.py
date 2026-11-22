r"""DIR-62 参照：W^0 作为「预载通道」的闭式代数（§5.1 两条秩公式的对照）

装置（d=6, r 可变, 单键恒定输入）：
  k = e1+e2 (s=2),  v = e3+e4          ← 训练对
  预载：W^0 = Σ_j γ_j (v_j k_j^T)/s_j，  k_j ∈ {e3,e4,e5} ⊥ k,  v_j ∈ {e1,e2,e5} ⊥ v
  目标：J(W) = ½||Wk−v||² + λ/2||W−W^0||²
  W* = (vk^T + λW^0)(kk^T+λI)^{-1}
  读数：gain(k_j) = (Wk_j)·v_j/||v_j||²；gain_data = (Wk)·v/||v||²
"""
import numpy as np

D = 6
LAM = 0.05
k = np.zeros((D, 1)); k[0, 0] = k[1, 0] = 1.0
v = np.zeros((D, 1)); v[2, 0] = v[3, 0] = 1.0
S = float(k.T @ k)


def preload(pairs):
    W0 = np.zeros((D, D))
    for gamma, kj, vj in pairs:
        sj = float(kj.T @ kj)
        W0 = W0 + gamma * (vj @ kj.T) / sj
    return W0


def J(W, W0):
    res = W @ k - v
    return 0.5 * float(res.T @ res) + 0.5 * LAM * float(np.sum((W - W0) ** 2))


def closed(W0):
    C = k @ k.T
    B = v @ k.T
    return (B + LAM * W0) @ np.linalg.inv(C + LAM * np.eye(D))


def svd_r(M, r):
    U, sp, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * sp[:r]) @ Vt[:r]


def report(pairs, r, tag):
    W0 = preload(pairs)
    Ws = closed(W0)
    R = np.linalg.cholesky(k @ k.T + LAM * np.eye(D)).T          # R = (C+λI)^{1/2}，对称正定取 cholesky 上三角亦可行
    Rsym = np.linalg.cholesky(k @ k.T + LAM * np.eye(D))
    _ = R
    Mbar = Ws - W0
    sig_WR = np.linalg.svd(Ws @ Rsym, compute_uv=False)
    sig_MR = np.linalg.svd(Mbar @ Rsym, compute_uv=False)
    L_B = 0.5 * float(np.sum(sig_WR[r:] ** 2))        # W 参数化：秩预算压在 W 上
    L_A = 0.5 * float(np.sum(sig_MR[r:] ** 2))        # ΔW 参数化：秩预算压在 W−W^0 上
    Wr = svd_r(Ws @ Rsym, r) @ np.linalg.inv(Rsym)    # 秩-r 最优（W 参数化）
    Wa = W0 + svd_r(Mbar @ Rsym, r) @ np.linalg.inv(Rsym)   # 秩-r 最优（ΔW 参数化）
    print("\n[%s]  r=%d  rank(W*)=%d" % (tag, r, int(np.sum(sig_WR > 1e-12))))
    print("  σ(W*R)   =", np.round(sig_WR, 6))
    print("  σ(M̄R)    =", np.round(sig_MR, 6))
    print("  L_B(W秩) = %.10f    L_A(ΔW秩) = %.10f    L_A/L_B = %.6f" % (L_B, L_A, L_A / L_B if L_B else float("nan")))
    print("  J(W*)    = %.10f   J(秩r, W参数化) = %.10f   J(秩r, ΔW参数化) = %.10f"
          % (J(Ws, W0), J(Wr, W0), J(Wa, W0)))
    print("  恒等式核对：J(Wr)−J(W*) − L_B = %.3e ; J(Wa)−J(W*) − L_A = %.3e"
          % (J(Wr, W0) - J(Ws, W0) - L_B, J(Wa, W0) - J(Ws, W0) - L_A))
    gd = float((Wr @ k).T @ v) / float(v.T @ v)
    print("  gain_data(W_r) = %.10f   期望 s/(s+λ) = %.10f" % (gd, S / (S + LAM)))
    for j, (gamma, kj, vj) in enumerate(pairs):
        g = float((Wr @ kj).T @ vj) / float(vj.T @ vj)
        print("  gain(pre_%d)：γ=%.2f  秩r下实际 = %.10f  全秩 = %.10f" % (j + 1, gamma, g, gamma))
    return dict(sig_WR=sig_WR, sig_MR=sig_MR, L_A=L_A, L_B=L_B, gf=gd)


if __name__ == "__main__":
    e = lambda i: np.eye(D)[:, i:i + 1]
    P3 = [(1.40, e(2), e(0)), (0.60, e(3), e(1)), (0.50, e(4), e(4))]
    print("=" * 96); print("★ 全秩参照（无秩压力）")
    report(P3, 4, "p=3, r=4")
    for r in (1, 2, 3):
        print("-" * 96); report(P3, r, "p=3, r=%d" % r)
    print("\n" + "=" * 96); print("★ 预载落在训练键上（稀释律）：W^0 = γ v_P k^T/s")
    vP = e(0)
    W0 = preload([(1.0, k, vP)])
    Ws = closed(W0)
    read = Ws @ k
    cf = float(read.T @ v) / float(v.T @ v)
    cp = float(read.T @ vP) / float(vP.T @ vP)
    print("  W*k 在 {v, v_P} 上的系数：data=%.12f  preload=%.12f" % (cf, cp))
    print("  预测 data=s/(s+λ)=%.12f  preload=λ/(s+λ)=%.12f" % (S / (S + LAM), LAM / (S + LAM)))
    print("  残差：%.3e  %.3e" % (abs(cf - S / (S + LAM)), abs(cp - LAM / (S + LAM))))
