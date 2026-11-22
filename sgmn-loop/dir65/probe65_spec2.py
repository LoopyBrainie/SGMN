import numpy as np
D, LAM, R = 6, 0.05, 2
E = np.zeros((D, R)); E[0, 0] = E[1, 1] = 1.0
def mats(c):
    ks = [np.sqrt(c) * np.eye(D)[:, i] for i in range(D)]
    return (sum(np.outer(k, k) for k in ks) / D,
            sum(np.outer(E @ (E.T @ k), k) for k in ks) / D)
def F(UV, c, eta):
    U, V = UV[:D*R].reshape(D, R), UV[D*R:].reshape(D, R)
    W = U @ V.T; C, B = mats(c); G = W @ C - B + LAM * W
    return np.concatenate([(U - eta*(G @ V)).ravel(), (V - eta*(G.T @ U)).ravel()])
def fp(c, t=1.0):
    s = np.sqrt(c/(c + D*LAM))
    return np.concatenate([(s*np.sqrt(t)*E).ravel(), (s/np.sqrt(t)*E).ravel()])
def jac(UV, c, eta, h=1e-6):
    n = UV.size; J = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = h
        J[:, i] = (F(UV+e, c, eta) - F(UV-e, c, eta))/(2*h)
    return J
def spec(c, eta, t=1.0):
    ev = np.abs(np.linalg.eigvals(jac(fp(c, t), c, eta)))
    return np.sort(ev)[::-1]
print("① 平衡点谱随 η（c=6）：前 8 大 |λ|")
for eta in (0.5, 0.7, 0.75, 0.78, 0.79, 0.80, 0.85, 0.9, 1.0):
    ev = spec(6.0, eta)
    print("   η=%-5.2f %s" % (eta, np.round(ev[:8], 6)))
print("\n② 去掉恰为 1 的 4 个中性值后的最大 |λ|，及其过 1 的 η（多 c）")
for c in (2.0, 3.0, 6.0, 12.0, 24.0):
    def m(eta):
        ev = spec(c, eta); return max([x for x in ev if x < 1 - 1e-7] or [0]), ev.sum()
    lo, hi = 0.2, 3.0
    for _ in range(50):
        mid = 0.5*(lo+hi)
        if m(mid)[0] < 1: lo = mid
        else: hi = mid
    b = 0.5*(lo+hi)
    print("   c=%-5.1f 边界=%.5f   D/c=%.4f  比值=%.4f   最大非中性|λ|(η=0.5)=%.4f" % (c, b, D/c, b/(D/c), m(0.5)[0]))
print("\n③ 哪个特征值先过 1（c=6, η=0.79 vs 0.80）")
for eta in (0.79, 0.80):
    ev = spec(6.0, eta); print("   η=%.2f  非中性前 3=%s" % (eta, np.round([x for x in ev if x < 1-1e-7][:3], 6)))
