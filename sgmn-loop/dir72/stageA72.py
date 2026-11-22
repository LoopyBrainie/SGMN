"""DIR-72 · Stage A（守卫；任一 FAIL 阻断正式轮）。卡 rev 1.0 §5。
"""
import sys, json, math
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, ref70
import step72, spec72
from step72 import D, R, P, LAM, step
ok = True
def chk(name, cond, msg=""):
    global ok; ok = ok and bool(cond)
    print("%-56s %s | %s" % (name, "PASS" if cond else "FAIL", msg), flush=True)

# ---------- A1 玩具闭式 ----------
c_t, b_t, lam_t = 1.0, 1.0, 0.05
def toy_step(u, v, eta):
    G = (c_t + lam_t) * u * v - b_t
    u2 = u - eta * G * v; v2 = v - eta * G * u2
    return u2, v2
w_star = b_t / (c_t + lam_t); s = math.sqrt(w_star)
eta_t = 1e-3; h = 1e-6
Jt = np.zeros((2, 2))
for j in range(2):
    e = np.zeros(2); e[j] = h
    fp = np.array(toy_step(*(np.array([s, s]) + e), eta_t)); fm = np.array(toy_step(*(np.array([s, s]) - e), eta_t))
    Jt[:, j] = (fp - fm) / (2 * h)
ev_t = np.sort(np.linalg.eigvals(Jt).real)
want = np.sort(np.array([1 - 2 * eta_t * b_t, 1.0]))
chk("A1 玩具 FD 谱 == 闭式 {1−2ηb, 1}", np.max(np.abs(ev_t - want)) < 1e-8, "max|Δλ|=%.2e" % np.max(np.abs(ev_t - want)))
# A1 第二半 = **公式检查**：把闭式谱直接喂给 bound（不经 FD）
J_exact = np.diag([1.0 - 2 * eta_t * b_t, 1.0])
bnd_t, diag_t = spec72.bound(J_exact, eta_t)
chk("A1 玩具 η*=1/b（闭式谱 → bound）", abs(bnd_t - 1.0 / b_t) < 1e-12, "η*=%.13f vs %.13f" % (bnd_t, 1.0 / b_t))

# ---------- A2 矩阵空间 FD vs 解析 2x2（target γ=3）----------
alg = json.load(open("/home/ec2-user/sgmn-loop/dir72/out/algebra72_v11.json"))
r3 = [r for r in alg["rows"] if r["mode"] == "target" and r["gamma"] == 3][0]
p3 = ref70.pred_target(6.0, 3.0)
W1s, W2s = p3["u1"] * P, p3["u2"] * P
eta_f = 1e-3
Jf = spec72.jac_field(W1s, W2s, "target", 3.0, eta_f)
lm_f = (np.linalg.eigvals(Jf) - 1.0) / eta_f
lmax_f = float(np.max(np.real(lm_f)))
dec_f = float(-np.min(np.real(lm_f)))
chk("A2 矩阵空间 FD 最大衰减率 == 解析", abs(dec_f - r3["lmax"]) / r3["lmax"] < 1e-6,
    "FD=%.6f 解析=%.6f rel=%.2e" % (dec_f, r3["lmax"], abs(dec_f - r3["lmax"]) / r3["lmax"]))

# ---------- 取 target γ=3 的因子空间不动点 ----------
fp3 = step72.fp_factors("target", 3.0, eta_warm=0.3 * 0.0654901)
chk("A7 不动点可达 + Jacobian 全有限", np.isfinite(fp3["res"]) and fp3["res"] < 1e-11, "res=%.2e steps=%d" % (fp3["res"], fp3["steps"]))
xs = spec72.pack(fp3["U1"], fp3["V1"], fp3["U2"], fp3["V2"])
Jfac = spec72.jac_factor(xs, "target", 3.0, eta_f)
chk("A7 因子空间 Jacobian 全有限", bool(np.isfinite(Jfac).all()) and bool(np.isfinite(np.linalg.eigvals(Jfac)).all()), "shape=%s" % (Jfac.shape,))

# ---------- A3 零模计数 ----------
ljf = np.linalg.eigvals(Jfac)
zeros = int(np.sum(np.abs(ljf - 1.0) < 1e-8))
chk("A3 因子空间零模 ∈[8,12] 且非零 ≥36", 8 <= zeros <= 12 and (ljf.size - zeros) >= 36,
    "zeros=%d nonzero=%d (dim=%d)" % (zeros, ljf.size - zeros, ljf.size))

# ---------- A4 步长/标度不变性 ----------
b1, _ = spec72.bound(Jfac, eta_f)
Jh = spec72.jac_factor(xs, "target", 3.0, eta_f, h=5e-7)
b2, _ = spec72.bound(Jh, eta_f)
chk("A4a h 减半：η* 相对变化 <1e-4", abs(b2 - b1) / b1 < 1e-4, "%.9f vs %.9f rel=%.2e" % (b1, b2, abs(b2 - b1) / b1))
Jg = spec72.jac_factor(xs, "target", 3.0, 1e-4)
b3, _ = spec72.bound(Jg, 1e-4)
chk("A4b 求谱 η 改 10 倍：η* 相对变化 <1e-6", abs(b3 - b1) / b1 < 1e-6, "%.9f vs %.9f rel=%.2e" % (b1, b3, abs(b3 - b1) / b1))

# ---------- A5 γ=0 锚 ----------
fp0 = step72.fp_factors("target", 0.0, eta_warm=0.3 * 0.88751)
xs0 = spec72.pack(fp0["U1"], fp0["V1"], fp0["U2"], fp0["V2"])
J0 = spec72.jac_factor(xs0, "target", 0.0, eta_f)
b0, d0 = spec72.bound(J0, eta_f)
rec = 0.88751
chk("A5 γ=0 锚：预言与记录值差 ≤25%", abs(b0 / rec - 1.0) <= 0.25, "pred=%.6f rec=%.5f rel=%+.3f zeros=%d" % (b0, rec, b0 / rec - 1.0, d0["zeros"]))

# ---------- A6 γ=0 逐位 ----------
for mode in ("target", "keys"):
    r = ref70.run_stack(6.0, 0.0, 0.5, 20000, seed=20261118, init=0.1, mode=mode)
    rng = np.random.default_rng(20261118)
    U1 = rng.normal(size=(D, R)) * 0.1; V1 = rng.normal(size=(D, R)) * 0.1
    U2 = rng.normal(size=(D, R)) * 0.1; V2 = rng.normal(size=(D, R)) * 0.1
    for _ in range(20000):
        U1, V1, U2, V2 = step(U1, V1, U2, V2, 0.5, mode, 0.0)
    chk("A6 γ=0 逐位复现 ref70 [%s]" % mode, float(np.max(np.abs(U1 @ V1.T - r["W1"]))) == 0.0,
        "max|ΔW1|=%.1e" % float(np.max(np.abs(U1 @ V1.T - r["W1"]))))

# ---------- A8 稳定界自洽（0.99η* 稳定 / 1.01η* 不稳定）----------
vb = spec72.verify_bound(Jfac, eta_f, b1)
chk("A8 0.99η* 稳定 且 1.01η* 不稳定", vb["ok"], "rad(0.99)=%.12f rad(1.01)=%.12f" % (vb["rad_at_099"], vb["rad_at_101"]))

print("\nSTAGE_A", "PASS" if ok else "FAIL")
json.dump(dict(ok=ok, zeros=zeros, eta_pred_g3=b1, eta_pred_g0=b0, eta_field_g3=lmax_f), open("/home/ec2-user/sgmn-loop/dir72/out/stageA72.json", "w"), indent=1)
sys.exit(0 if ok else 1)
