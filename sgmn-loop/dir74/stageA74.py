"""DIR-74 Stage A（rev 1.0 §5 字面）：A1 2×2 工作例 / A2 正常算子 / A3 读出 FD / A4 矩阵幂 / A5 η=0 / A6 流形漂移 / A7 正控峰位。"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir74"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, spec74, near73, ref70, step72
OK = []
def chk(tag, cond, val):
    OK.append(bool(cond)); print("%-4s %-46s %s" % (tag, val, "PASS" if cond else "FAIL"), flush=True)

# A1 解析 2×2 工作例
a, b = 0.8, 10.0
M = np.array([[a, b], [0.0, a]])
Mp = np.eye(2); errs = []
for t in range(1, 51):
    Mp = M @ Mp
    closed = np.array([[a ** t, b * t * a ** (t - 1)], [0.0, a ** t]])
    errs.append(np.linalg.norm(Mp - closed) / max(np.linalg.norm(closed), 1e-300))
chk("A1", max(errs) < 1e-12, "2×2 闭式 t=1..50 max rel=%.2e" % max(errs))

# A2 正常算子：(I+ηJ)^t 的奇异值 = |λ|^t
Md = np.diag([0.9, 0.5, -0.3]); Mp = np.eye(3); e2 = []
for t in range(1, 31):
    Mp = Md @ Mp
    e2.append(abs(np.linalg.svd(Mp, compute_uv=False)[0] - max(abs(np.diag(Md))) ** t) / max(abs(np.diag(Md)) ** t))
chk("A2", max(e2) < 1e-12, "对角 M：σ_max=|λ_max|^t  max rel=%.2e" % max(e2))

# A3 读出 R vs 中心差分
mode, g = "keys", 10.0
xs, fp = near73.freeze_fp(mode, g, eta_warm=0.3 * near73.eta_lin(mode, g))
Rm = spec74.readout(xs)
rng = np.random.default_rng(7); e3 = []
for _ in range(5):
    d = rng.normal(size=48); d /= np.linalg.norm(d)
    eps = 1e-7
    xp = spec72_pack = np.concatenate([X.ravel() for X in xs])
    def Wvec(x):
        Us = spec74.spec72.unpack(x)   # 复用 spec72.unpack
        return np.concatenate([(Us[0] @ Us[1].T).ravel(), (Us[2] @ Us[3].T).ravel()])
    fd = (Wvec(xp + eps * d) - Wvec(xp - eps * d)) / (2 * eps)
    an = Rm @ d
    e3.append(np.linalg.norm(fd - an) / max(np.linalg.norm(an), 1e-300))
chk("A3", max(e3) < 1e-6, "读出 R vs 中心差分  max rel=%.2e" % max(e3))

# A4 M_t 递推 vs matrix_power
Mm = spec74.spec72.jac_factor(spec74.spec72.pack(*xs), mode, g, 0.5 * near73.eta_lin(mode, g))
Mt = np.eye(48); e4 = []
for t in range(1, 101):
    Mt = Mm @ Mt
    if t in (1, 7, 100):
        e4.append(np.abs(Mt - np.linalg.matrix_power(Mm, t)).max())
chk("A4", max(e4) < 1e-10, "M_t 递推 vs matrix_power  max|Δ|=%.2e" % max(e4))

# A5 η=0 ⇒ g≡1
M0 = spec74.spec72.jac_factor(spec74.spec72.pack(*xs), mode, g, 0.0)
d0, _ = spec74.delta0(xs)
b5, _ = spec74.curves(xs, mode, g, 0.0, d0, nmax=300, want_G=False)
chk("A5", abs(b5["g"] - 1.0) < 1e-8 and np.abs(M0 - np.eye(48)).max() < 1e-8,
    "η=0：g=%.15f, max|M−I|=%.1e" % (b5["g"], np.abs(M0 - np.eye(48)).max()))

# A6 两代表点都在流形上
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
drifts = []
for (mode, g) in (("keys", 10.0), ("target", 30.0)):
    p = ref70.pred_target(6.0, g) if mode == "target" else ref70.pred_keys(6.0, g)
    xb = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
    U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in xb)
    for t in range(100):
        U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, 1e-7, mode, g)
    drifts.append(max(np.linalg.norm(U1 - xb[0]), np.linalg.norm(V1 - xb[1])))
    xw, _ = near73.freeze_fp(mode, g, eta_warm=0.3 * near73.eta_lin(mode, g))
    U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in xw)
    for t in range(100):
        U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, 1e-7, mode, g)
    drifts.append(max(np.linalg.norm(U1 - xw[0]), np.linalg.norm(V1 - xw[1])))
chk("A6", all(d < 1e-12 for d in (drifts[0], drifts[1], drifts[2])) and drifts[3] < 1e-8,
    "漂移 bal/warm: keys %.1e/%.1e · target %.1e/%.1e（target 暖机档 1e-8）" % tuple(drifts))

# A7 正控峰位：0.9/5 的闭式峰位
Mb = np.array([[0.9, 5.0], [0.0, 0.9]]); Mp = np.eye(2); gg = []
for t in range(1, 400):
    Mp = Mb @ Mp; gg.append(np.linalg.svd(Mp, compute_uv=False)[0])
tstar = int(np.argmax(gg)) + 1
# 闭式：σ_max ≈ sqrt(a^{2t} + (t b a^{t-1})^2)，峰位由 d/dt 给出 ⇒ t* = 1/ln(1/a) 附近
tana = int(round(-1.0 / np.log(0.9)))
chk("A7", abs(tstar - tana) <= 1, "正控峰位 t*=%d（闭式 %d）" % (tstar, tana))
print("\nA（共 %d 项）%s" % (len(OK), "PASS" if all(OK) else "FAIL"), flush=True)
