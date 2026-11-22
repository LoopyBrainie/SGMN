r"""DIR-61 算例 B（worked example，**不是**正式轮）：d=4 上的 2×2 陷阱判别。

k = e1+e2, v = e3+e4, d=4, r=2, lam=0.05, eta=0.05
G = (Wk - v)k^T + lam(W - W0);  同步  U<-U-eta*G V,  V<-V-eta*G^T U,  W=U V^T
步数 N=2000；判据量是 max_t ||W_t - W0||_F。
"""
import numpy as np

d, r, LAM, ETA, N = 4, 2, 0.05, 0.05, 2000
k = np.zeros((d, 1)); k[0, 0] = k[1, 0] = 1.0
v = np.zeros((d, 1)); v[2, 0] = v[3, 0] = 1.0
kv = k / np.linalg.norm(k); vv = v / np.linalg.norm(v)

def J(W, W0):
    res = W @ k - v
    return 0.5 * float(np.sum(res * res)) + 0.5 * LAM * float(np.sum((W - W0) ** 2))

def run(U, V, n=N):
    W0 = U @ V.T
    j0 = J(W0, W0); dmax = 0.0; Wprev = W0.copy(); head = []
    for t in range(1, n + 1):
        W = U @ V.T
        G = (W @ k - v) @ k.T + LAM * (W - W0)
        U, V = U - ETA * (G @ V), V - ETA * (G.T @ U)
        Wn = U @ V.T
        dmax = max(dmax, float(np.linalg.norm(Wn - W0)))
        if t in (1, 10, 100, 500, 2000):
            head.append((t, float(np.linalg.norm(Wn - W0)), J(Wn, W0)))
        Wprev = Wn
    return dmax, J(Wprev, W0), j0, head

def run_full(W0, n=N):
    W = W0.copy(); dmax = 0.0
    for _ in range(n):
        G = (W @ k - v) @ k.T + LAM * (W - W0)
        W = W - ETA * G
        dmax = max(dmax, float(np.linalg.norm(W - W0)))
    return dmax, J(W, W0)

rng = np.random.default_rng(20261117)
Pk = np.zeros((d, 2)); Pk[2, 0] = Pk[3, 1] = 1.0     # 列 ⊥ k  （span{e3,e4}）
Pv = np.zeros((d, 2)); Pv[0, 0] = Pv[1, 1] = 1.0     # 列 ⊥ v  （span{e1,e2}）
A  = lambda: rng.normal(size=(2, 2)) + 2.0 * np.eye(2)   # 保证可逆、非退化

arms = {}
arms["ZERO"]   = (np.zeros((d, 2)), np.zeros((d, 2)), "frozen")
arms["ORTHO"]  = (Pv @ A(), Pk @ A(), "frozen")
arms["VNULL"]  = (Pv @ A() + 0.6 * vv @ np.ones((1, 2)), Pk @ A(), "move")   # 键侧对齐、值侧不对齐
arms["UNULL"]  = (Pv @ A(), Pk @ A() + 0.6 * kv @ np.ones((1, 2)), "move")   # 值侧对齐、键侧不对齐
U0, V0 = Pv @ A(), Pk @ A()
V0[:, 1] += 0.6 * kv[:, 0]          # 第 2 列键侧不对齐 ⇒ 整体不再是陷阱
arms["DEADBLK"] = (U0, V0, "move")
arms["RAND"]   = (rng.normal(size=(d, 2)), rng.normal(size=(d, 2)), "move")

print("%-8s %-6s %-10s %-10s %-11s %-9s %-9s %-11s" %
      ("arm", "pred", "|k^T V|", "|U^T v|", "dW_fac", "J0", "J_end", "dW_full"))
res = {}
for name, (U, V, pred) in arms.items():
    dm, jend, j0, head = run(U, V)
    dmf, jf = run_full(U @ V.T)
    ktv = float(np.abs(k.T @ V).max()); utv = float(np.abs(U.T @ v).max())
    res[name] = (pred, ktv, utv, dm, j0, jend, dmf)
    print("%-8s %-6s %-10.2e %-10.2e %-11.4e %-9.4f %-9.4f %-11.4e"
          % (name, pred, ktv, utv, dm, j0, jend, dmf))
print()
for name in ("ZERO", "ORTHO", "DEADBLK", "RAND"):
    _, _, _, _, _, _, _ = res[name]
ok = (res["ZERO"][3] <= 1e-12 and res["ORTHO"][3] <= 1e-12
      and all(res[a][3] >= 1e-2 for a in ("VNULL", "UNULL", "DEADBLK", "RAND")))
print("★ 算例 B 结论：冻结支 ZERO/ORTHO 的 dW = %.2e / %.2e；逃逸支 %s ⇒ %s"
      % (res["ZERO"][3], res["ORTHO"][3],
         " ".join("%s=%.2e" % (a, res[a][3]) for a in ("VNULL", "UNULL", "DEADBLK", "RAND")),
         "2×2 判别成立（陷阱 ⇔ 两侧同时对齐）" if ok else "判别**不成立**"))
