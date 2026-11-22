import numpy as np
D, LAM, R = 6, 0.05, 2
E = np.zeros((D, R)); E[0, 0] = E[1, 1] = 1.0

def mats(c):
    ks = [np.sqrt(c) * np.eye(D)[:, i] for i in range(D)]
    C = sum(np.outer(k, k) for k in ks) / D
    B = sum(np.outer(E @ (E.T @ k), k) for k in ks) / D   # v = P k, P = E E^T
    return C, B

def F(UV, c, eta):
    U, V = UV[:D * R].reshape(D, R), UV[D * R:].reshape(D, R)
    G = (U @ V.T) @ mats(c)[0] - mats(c)[1] + LAM * (U @ V.T)
    return np.concatenate([(U - eta * (G @ V)).ravel(), (V - eta * (G.T @ U)).ravel()])

def fp(c, t):
    s = np.sqrt(c / (c + D * LAM))
    return np.concatenate([(s * np.sqrt(t) * E).ravel(), (s / np.sqrt(t) * E).ravel()])

def jac(UV, c, eta, h=1e-6):
    n = UV.size; J = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = h
        J[:, i] = (F(UV + e, c, eta) - F(UV - e, c, eta)) / (2 * h)
    return J

print("① 固定点集是双曲线 ab=p* 吗（F(fp)=fp）")
for c in (2.0, 6.0, 24.0):
    for t in (1.0, 3.0, 30.0):
        d = np.abs(F(fp(c, t), c, 0.5) - fp(c, t)).max()
        print("   c=%-5.1f t=%-5.1f  ‖F(fp)-fp‖∞=%.2e" % (c, t, d))

print("\n② 谱随不平衡 t 的变化（c=6, η=0.8）")
c = 6.0
for t in (1.0, 1.5, 2.0, 4.0, 10.0):
    ev = np.abs(np.linalg.eigvals(jac(fp(c, t), c, 0.8)))
    print("   t=%-5.1f max|λ|=%.6f   前 4 大=%s" % (t, ev.max(), np.round(np.sort(ev)[::-1][:4], 5)))

print("\n③ 每个 t 的边界（max|λ|=1），以及 min_t")
def bound(c, t, lo=0.3, hi=1.6, it=42):
    for _ in range(it):
        m = 0.5 * (lo + hi)
        if np.abs(np.linalg.eigvals(jac(fp(c, t), c, m))).max() < 1: lo = m
        else: hi = m
    return 0.5 * (lo + hi)
for c in (2.0, 6.0, 24.0):
    bs = {t: bound(c, t) for t in (1.0, 1.5, 2.0, 4.0, 10.0, 30.0)}
    print("   c=%-5.1f " % c + "  ".join("t=%-4.1f:%.5f" % (t, b) for t, b in bs.items()) + "  ⇒ min=%.5f  D/c=%.4f" % (min(bs.values()), D / c))
