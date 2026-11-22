"""DIR-79 Stage A（A1–A7）。任一失败即阻断正式轮。"""
import sys, math, json
for p in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79"): sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + p)
import numpy as np, ref69, ref70, step72, warm78, jac79

GAM = [("target", g) for g in (0.0, 3.0, 10.0, 20.0, 30.0)]
R = {}
def rec(k, ok, **kw):
    R[k] = dict(ok=bool(ok), **kw); print(("  ✓ " if ok else "  ✗ ") + k, json.dumps(kw, ensure_ascii=False)[:220], flush=True)

# ---- A1 gamma=0 逐位保真 ----
a = ref69.run_stack(6.0, 0.0, 0.5, 500); b = ref70.run_stack(6.0, 0.0, 0.5, n=500, mode="target")
d1 = float(np.max(np.abs(a["W1"] - b["W1"]))); d2 = float(np.max(np.abs(a["W2"] - b["W2"])))
rec("A1_gamma0_bitwise", d1 == 0.0 and d2 == 0.0 and a["steps"] == b["steps"], dW1=d1, dW2=d2, steps=a["steps"])


# ---- A1b step72.step ≡ ref70 内步（逐位）----
rng = np.random.default_rng(20261118)
st = [rng.normal(size=(6, 2)) * 0.1 for _ in range(4)]
U1, V1, U2, V2 = (X.copy() for X in st)
for _ in range(500):
    U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, 1e-4, "target", 10.0)
ref = ref70.run_stack(6.0, 10.0, 1e-4, 500, seed=20261118, init=0.1, mode="target")
d = max(float(np.max(np.abs(U1 @ V1.T - ref["W1"]))), float(np.max(np.abs(U2 @ V2.T - ref["W2"]))))
rec("A1b_step72_eq_ref70", d == 0.0, dW=d, steps=ref["steps"])

# ---- 先求 eta_th 的最小需求：用 D-90 括号作为 warm 的 eta 依据，随后正式轮再二分 ----
idx = {g: v for g, v in zip([0.0, 3.0, 10.0, 20.0, 30.0], [0.876, 0.05, 0.005, 0.0015, 0.0008])}
FP, TH = {}, {}
for mode, g in GAM:
    e0 = idx[g]
    x, info = warm78.warm_fp(mode, g, 0.3 * e0)
    FP[g] = x
    # A2 不动点
    n1, n2, n3, n4 = step72.step(*x, 0.3 * e0, mode, g)
    num = max(np.linalg.norm(n1 - x[0]), np.linalg.norm(n2 - x[1]), np.linalg.norm(n3 - x[2]), np.linalg.norm(n4 - x[3]))
    den = max(np.linalg.norm(X) for X in x)
    rec("A2_fp_residual_g%g" % g, num / den <= 1e-13, res=num / den, steps=info["steps"])
    # 记录 warm eta
    TH[g] = 0.3 * e0

# ---- A3 h 收敛 ----
worst = (0.0, -1)
for mode, g in GAM:
    x = FP[g]; eta = TH[g]
    J1 = jac79.jac(x, eta, mode, g, h=1e-6); J2 = jac79.jac(x, eta, mode, g, h=2e-6)
    sc = max(1e-12, float(np.max(np.abs(J1))))
    rel = float(np.max(np.abs(J1 - J2))) / sc
    if rel > worst[0] if isinstance(worst, tuple) else rel: worst = (rel, g)
    rec("A3_hconv_g%g" % g, rel <= 1e-3, rel=rel, Jmax=float(np.max(np.abs(J1))), normF=float(np.linalg.norm(J1)))
rec("A3_worst", worst[0] <= 1e-3, rel=worst[0], at_g=worst[1])

# ---- A4 特征对残差（lam_max 处前 6 个）----
worst = (0.0, -1)
for mode, g in (("target", 10.0), ("target", 20.0), ("target", 30.0)):
    x = FP[g]; eta = TH[g]
    J = jac79.jac(x, eta, mode, g)
    w, V = np.linalg.eig(J)
    o = np.argsort(-np.abs(w))[:6]
    res = max(float(np.linalg.norm(J @ V[:, i] - w[i] * V[:, i]) / max(1e-12, abs(w[i]))) for i in o)
    if res > (worst[0] if isinstance(worst, tuple) else worst): worst = (res, g)
    rec("A4_eigres_g%g" % g, res <= 1e-9, res=res, lmax=float(np.max(np.abs(w))))

# ---- A5 闭式例：gamma=0 时 J 下块三角 ⇒ 谱 = 两对角块之并；非中性部分与 D-90 锚一致 ----
mode, g = "target", 0.0
J = jac79.jac(FP[0.0], TH[0.0], mode, 0.0)
off = float(np.max(np.abs(J[:24, 24:])))            # 层 1 对层 2 的敏感度必须为 0（gamma=0 无反馈项）
blk1, blk2 = J[:24, :24], J[24:, 24:]
def nonneutral_max(M):
    w = np.linalg.eigvals(M)
    m = np.abs(w[np.abs(w - 1.0) > 1e-6])
    return float(np.max(m)) if m.size else float("nan")
m_all = nonneutral_max(J); m_blocks = max(nonneutral_max(blk1), nonneutral_max(blk2))
rec("A5_blocktriangular", off <= 1e-10 and abs(m_all - m_blocks) <= 1e-9,
    offblock_L1_wrt_L2=off, nonneutral_all=m_all, nonneutral_blocks=m_blocks,
    nz_all=int(np.sum(np.abs(np.linalg.eigvals(J) - 1.0) <= 1e-8)))
# A5b 谱并集（含中性模计数相加）
w_all = np.abs(np.linalg.eigvals(J) - 1.0)
w_b = np.concatenate([np.abs(np.linalg.eigvals(blk1) - 1.0), np.abs(np.linalg.eigvals(blk2) - 1.0)])
rec("A5b_spec_union", abs(float(np.sum(w_all <= 1e-8)) - float(np.sum(w_b <= 1e-8))) <= 1e-12,
    n_zero_all=int(np.sum(w_all <= 1e-8)), n_zero_blocks=int(np.sum(w_b <= 1e-8)))
# A5c 锚（交换检查用，不是判据）：gamma=0 的谱穿越应与 D-90 ③ 的 gamma=0 二分值 0.876 相容
r0, d0 = jac79.rho_scan(FP[0.0], "target", 0.0, lo=0.2, hi=2.0)
print("  ○ O1 观测（非门禁）：gamma=0 rho_lambda=%.6f vs D-90 轨迹锚 0.876 ⇒ 相对差 %+.2f%%" % (r0, 100 * (r0 / 0.876 - 1)), flush=True)
R["O1_gamma0_rho_vs_trajectory"] = dict(ok=True, rho=float(r0), ref=0.876, note="观测，不作门禁")

# ---- A6 中性模 ----
J = jac79.jac(FP[10.0], TH[10.0], "target", 10.0)
w = np.linalg.eigvals(J)
nz = int(np.sum(np.abs(w - 1.0) <= 1e-8))
rec("A6_neutral_modes", nz >= 8, n_zero=nz, n=w.size)

# ---- A7 括号（用 warm 点做快速预检；正式括号在正式轮二分内断言）----
okA7 = True; det = {}
for mode, g in (("target", 10.0), ("target", 20.0), ("target", 30.0)):
    lo = ref70.run_stack(6.0, g, 1e-6, 2000, mode=mode)["div"]; hi = ref70.run_stack(6.0, g, 1.0, 2000, mode=mode)["div"]
    det[g] = (bool(lo), bool(hi)); okA7 &= (not lo) and hi
rec("A7_brackets_precheck", okA7, detail=str(det))

json.dump(R, open("/home/ec2-user/sgmn-loop/dir79/out/stageA79.json", "w"), ensure_ascii=False, indent=1)
bad = [k for k, v in R.items() if not v["ok"]]
print(("Stage A 全部通过 (%d/%d)" % (len(R) - len(bad), len(R))) if not bad else ("Stage A 失败: " + ", ".join(bad)))
