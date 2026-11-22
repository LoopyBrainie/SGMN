"""DIR-82 测量层：`dir81/jac81.py` 逐字派生。
**唯一新增** = `gfix_qr` / `gfix_polar` / `orth_err` / `ker_dim`（归一化与仪器检查）。
`flat/unflat/step48/jac/lmax/rho_scan/warm_from_init/supjt` 全部 import 沿用，不复制。
见卡 §2：任何"把 V 正交化 + 令 U=W V"的归一化都落在同一 O(2) 轨道；
而 D-102 已实测 rho 在 O(2) 上精确不变 ⇒ rho(gfix(x)) 是 W 的良定义函数。
"""
import sys
for q in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79", "dir80", "dir81", "dir82"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np
import jac81

flat, unflat, step48, jac, lmax = jac81.flat, jac81.unflat, jac81.step48, jac81.jac, jac81.lmax
rho_scan, warm_from_init, draw_init = jac81.rho_scan, jac81.warm_from_init, jac81.draw_init
D, R = jac81.D, jac81.R


def _layer_qr(U, V):
    """V = Q R；取 D=diag(sign(diag R)) 使 diag(R')>0；Q'=QD, R'=DR；U'=U R'^T。
    核：U' V'^T = U R'^T Q'^T = U (Q'R')^T = U (QD DR)^T = U (QR)^T = U V^T。"""
    Q, Rm = np.linalg.qr(V)                      # reduced: 6x2 , 2x2
    d = np.sign(np.diag(Rm))
    d = np.where(d == 0.0, 1.0, d)               # 退化列不翻符号（不会发生，但设防）
    G = np.diag(d)
    return U @ (G @ Rm).T, Q @ G, G @ Rm


def _layer_polar(U, V):
    """V = P S Z^T；V' = P Z^T（正交归一）；M = Z S Z^T（对称）；U' = U M。
    核：U' V'^T = U M Z P^T = U Z S Z^T Z P^T = U Z S P^T = U V^T。"""
    P, S, Zt = np.linalg.svd(V, full_matrices=False)
    M = (Zt.T * S) @ Zt                          # Z diag(S) Z^T
    return U @ M.T, P @ Zt, M


def gfix_qr(x):
    U1, V1, U2, V2 = x
    a, b, _ = _layer_qr(U1, V1)
    c, d, _ = _layer_qr(U2, V2)
    return (a, b, c, d)


def gfix_polar(x):
    U1, V1, U2, V2 = x
    a, b, _ = _layer_polar(U1, V1)
    c, d, _ = _layer_polar(U2, V2)
    return (a, b, c, d)


def orth_err(V):
    return float(np.linalg.norm(V.T @ V - np.eye(V.shape[1])))


def ker_dim(x, eta, mode, g, tol=1e-8, c=6.0):
    """dim ker(J-I) 用 SVD（不用 eig 的特征向量基，DEF-81-6）。
    返回 (dim, 谱隙 = 最小"大"奇异值)。"""
    J = jac(x, eta, mode, g, c=c)
    s = np.linalg.svd(J - np.eye(J.shape[0]), compute_uv=False)
    small = s[s < tol]
    big = s[s >= tol]
    return int(small.size), (float(big.min()) if big.size else float("inf"))


def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


ORTHO_G = [("I", np.eye(2)), ("rot30", rot(np.pi / 6)), ("rot71", rot(1.24)),
           ("refl", np.diag([1.0, -1.0]))]


def apply_gauge(x, Gm):
    """U -> U Gm , V -> V Gm^{-T}（GL(2) 参数化变换；W 不变）。"""
    U1, V1, U2, V2 = x
    Gi = np.linalg.inv(Gm)
    return (U1 @ Gm, V1 @ Gi.T, U2 @ Gm, V2 @ Gi.T)
