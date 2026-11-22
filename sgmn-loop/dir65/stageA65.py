import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir65")
import ref65 as R
D, Rr, LAM = R.D, R.R, R.LAM
ok_all = True
def chk(name, ok, msg=""):
    global ok_all
    ok_all = ok_all and ok
    print("%-44s %s | %s" % (name, "PASS" if ok else "FAIL", msg))

# A65-1 线性谱（代数）
E = np.zeros((D, Rr)); E[0, 0] = E[1, 1] = 1.0
def mats(c):
    ks = R.keys(c)
    return (sum(np.outer(k, k) for k in ks) / D, sum(np.outer(E @ (E.T @ k), k) for k in ks) / D)
def F(UV, c, eta):
    U, V = UV[:D*Rr].reshape(D, Rr), UV[D*Rr:].reshape(D, Rr)
    W = U @ V.T; C, B = mats(c); G = W @ C - B + LAM * W
    return np.concatenate([(U - eta*(G @ V)).ravel(), (V - eta*(G.T @ U)).ravel()])
c0 = 6.0; eta0 = 0.9 * D / c0
s = np.sqrt(c0 / (c0 + D * LAM))
fp = np.concatenate([(s*E).ravel(), (s*E).ravel()])
J = np.zeros((24, 24)); h = 1e-6
for i in range(24):
    e = np.zeros(24); e[i] = h
    J[:, i] = (F(fp+e, c0, eta0) - F(fp-e, c0, eta0)) / (2*h)
ev = np.abs(np.linalg.eigvals(J))
n_neu = int(np.sum(np.abs(ev - 1) <= 1e-9))
pred = abs(1 - 2*eta0*c0/D)
rest = np.sort(ev[np.abs(ev - 1) > 1e-9])
chk("A65-1 线性稳定（4×1 + 非中性 max|λ|<1）", n_neu == 4 and len(rest) == 20 and np.max(rest) <= 1 - 1e-9,
    "中性=%d 非中性 max|λ|=%.6f（η=0.9D/c）" % (n_neu, np.max(rest) if len(rest) else -1))

# A65-2 二分可执行
b_ok = (not R.run(c0, 0.1, 10000)["div"]) and R.run(c0, 2.0, 10000)["div"]
chk("A65-2 二分可执行（0.1 有界 / 2.0 发散）", b_ok, "")
# A65-3 T_div 可达
o = R.run(c0, 1.5, 2000)
chk("A65-3 T_div 可达（η=1.5 时 <1e3 步）", o["div"] and o["td"] < 1000, "T_div=%s" % o["td"])
# A65-4 不平衡可测（构造 t0=4）
U0 = 2*s*E; V0 = 0.5*s*E
o = R.run(c0, 0.8, 200, U0=U0, V0=V0, trace_every=10)
tr = [x[1] for x in o["trace"]]
chk("A65-4 不平衡可测（变化≥1% 且有限）", len(tr) >= 5 and np.isfinite(tr).all() and abs(tr[-1] - tr[0]) / tr[0] >= 0.01,
    "t: %.3f→%.3f" % (tr[0], tr[-1]))
# A65-5 确定性
a = R.run(c0, 0.5, 2000)["W"]; b = R.run(c0, 0.5, 2000)["W"]
chk("A65-5 确定性（逐位相同）", np.array_equal(a, b), "max|Δ|=%.1e" % np.max(np.abs(a - b)))
# A65-6 分辨力：相邻 η 的二分能分辨 0.2%
x1 = R.bisect(c0, 3000, rel=0.002, lo=0.6, hi=0.9)
x2 = R.bisect(c0, 3000, rel=0.002, lo=0.6, hi=0.9)
chk("A65-6 分辨力（重跑一致 + 区间宽 ≤0.2%）", x1["ok"] and x2["ok"] and abs(x1["eta"]-x2["eta"])/x1["eta"] <= 2e-3 and (x1["hi"]-x1["lo"])/x1["eta"] <= 2e-3,
    "η*=%.5f (%.3f,%.3f)" % (x1["eta"], x1["lo"], x1["hi"]))
print("\n★ Stage A: %d/6 PASS" % (6 if ok_all else -1))
