"""DIR-72 · 求谱与稳定界（因子空间 48 维 / 矩阵空间 72 维）。
稳定界取**离散 Euler 精确形式**：λ_M = (λ_J − 1)/η，η*_pred = min_{Re λ_M>0} 2Re λ_M/|λ_M|²。
（实特征值 ⇒ 2/λ；复特征对 ⇒ 2a/(a²+b²)，在 |b|≳a 时比 2/a 小一半。见卡 §1/§3。）
"""
import numpy as np
from step72 import D, R, step, field

def pack(U1, V1, U2, V2):
    return np.concatenate([U1.ravel(), V1.ravel(), U2.ravel(), V2.ravel()])

def unpack(x):
    i = 0; out = []
    for (r, cc) in ((D, R), (D, R), (D, R), (D, R)):
        out.append(x[i:i + r * cc].reshape(r, cc)); i += r * cc
    return out

def jac_factor(xs, mode, gamma, eta, h=1e-6, c=6.0):
    n = xs.size; J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        fp = step(*unpack(xs + e), eta=eta, mode=mode, gamma=gamma, c=c)
        fm = step(*unpack(xs - e), eta=eta, mode=mode, gamma=gamma, c=c)
        J[:, j] = (pack(*fp) - pack(*fm)) / (2 * h)
    return J

def jac_field(W1, W2, mode, gamma, eta, h=1e-6, c=6.0):
    x0 = np.concatenate([W1.ravel(), W2.ravel()]); n = x0.size; J = np.zeros((n, n))
    def f(x):
        a = x[:D * D].reshape(D, D); b = x[D * D:].reshape(D, D)
        r1, r2 = field(a, b, eta, mode, gamma, c)
        return np.concatenate([r1.ravel(), r2.ravel()])
    for j in range(n):
        e = np.zeros(n); e[j] = h
        J[:, j] = (f(x0 + e) - f(x0 - e)) / (2 * h)
    return J

def bound(J, eta, zero_tol=1e-8):
    """λ_M = (λ_J − 1)/η（映射是 x' = x − ηG ⇒ λ_M 为**负**）。
    离散 Euler 精确稳定界：η* = min_{Re λ_M<0} 2(−Re λ_M)/|λ_M|²。
    （实特征值 ⇒ 2/|λ|；复特征对 ⇒ 2|a|/(a²+b²)，|b|≳|a| 时比 2/|a| 小。）"""
    lj = np.linalg.eigvals(J)
    lm = (lj - 1.0) / eta
    zeros = int(np.sum(np.abs(lj - 1.0) < zero_tol))
    neg = lm[np.real(lm) < 0]
    diag = dict(zeros=zeros, n_neg=int(neg.size), n_zero=int(lm.size - zeros),
                dim=int(lm.size), lmax_decay=float(-np.min(np.real(lm))) if lm.size else float("nan"))
    if neg.size == 0:
        diag["bound"] = float("inf")
        return float("inf"), diag
    b = 2.0 * (-np.real(neg)) / (np.abs(neg) ** 2)
    k = int(np.argmin(b))
    diag["bound"] = float(b[k]); diag["argmin"] = complex(neg[k])
    diag["top5"] = [(float(bi), complex(li)) for bi, li in sorted(zip(b, neg), key=lambda z: z[0])[:5]]
    return float(np.min(b)), diag


def verify_bound(J, eta_field, eta_pred):
    """独立验证：线性映射 (I + η·λ_M) 在 0.99η* 稳定、在 1.01η* 不稳定（λ_M 为场谱）。"""
    import numpy as _np
    lam = (_np.linalg.eigvals(J) - 1.0) / eta_field
    lo = float(_np.max(_np.abs(1.0 + 0.99 * eta_pred * lam)))
    hi = float(_np.max(_np.abs(1.0 + 1.01 * eta_pred * lam)))
    return dict(rad_at_099=lo, rad_at_101=hi, ok=bool(lo <= 1.0 + 1e-6 and hi > 1.0 + 1e-6))
