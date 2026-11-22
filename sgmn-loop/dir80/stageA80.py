"""DIR-80 Stage A（A1–A9）。任一失败即阻断正式轮。
守卫只检验仪器、不携带假设方向（纪律 158）：本文件**不含**任何"ρ 应等于某历史值"的断言
（DEF-79-3 的教训）；先验 η 只用于挑一个可收敛的 warm 点来测仪器。
"""
import sys, math, json, time
for q in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79", "dir80"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np, ref69, ref70, step72, warm78, jac80

MODE, C = "keys", 6.0
GAM = (0.0, 3.0, 10.0, 20.0, 30.0)
# 先验 η（D-91 括号 + 局域指数 2.00）仅用于**挑 warm 点**，不是判据
IDX = {3.0: 0.12, 10.0: 0.015, 20.0: 0.004, 30.0: 0.0015}
R = {}
def rec(k, ok, **kw):
    R[k] = dict(ok=bool(ok), **kw)
    print(("  ✓ " if ok else "  ✗ ") + k + " " + json.dumps(kw, ensure_ascii=False)[:230], flush=True)

T0 = time.time()

# ---- A1 γ=0 逐位保真 ----
a = ref69.run_stack(C, 0.0, 0.5, 500)
b = ref70.run_stack(C, 0.0, 0.5, n=500, mode=MODE)
d1, d2 = float(np.max(np.abs(a["W1"] - b["W1"]))), float(np.max(np.abs(a["W2"] - b["W2"])))
rec("A1_gamma0_bitwise_keys", d1 == 0.0 and d2 == 0.0 and a["steps"] == b["steps"], dW1=d1, dW2=d2, steps=a["steps"])

# ---- A1b step72.step ≡ ref70 内步 ----
rng = np.random.default_rng(20261118)
U1, V1, U2, V2 = (rng.normal(size=(6, 2)) * 0.1 for _ in range(4))
for _ in range(500):
    U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, 1e-4, MODE, 10.0)
ref = ref70.run_stack(C, 10.0, 1e-4, 500, seed=20261118, init=0.1, mode=MODE)
d = max(float(np.max(np.abs(U1 @ V1.T - ref["W1"]))), float(np.max(np.abs(U2 @ V2.T - ref["W2"]))))
rec("A1b_step72_eq_ref70_keys", d == 0.0, dW=d, steps=ref["steps"])

# ---- A5 γ=0 的 keys 仍块三角（层 1 不看层 2）----
x0, i0 = warm78.warm_fp(MODE, 0.0, 0.5)
J0 = jac80.jac(x0, 0.5, MODE, 0.0, c=C)
blk = float(np.max(np.abs(J0[:24, 24:])))
rec("A5_gamma0_block_triangular_keys", blk == 0.0, max_abs_J_upper=blk)

# ---- 各 γ 的仪器检查（用先验 η 挑 warm 点）----
FP = {}
import os
CACHE = "/home/ec2-user/sgmn-loop/dir80/out/sa_fp_%s_g%g.npz"
for g in (3.0, 10.0, 20.0, 30.0):
    eta_w = 0.3 * IDX[g]
    cp = CACHE % (MODE, g)
    if os.path.exists(cp):
        z = np.load(cp)
        if abs(float(z["eta"]) - eta_w) < 1e-18:
            x = [z[k].copy() for k in ("U1", "V1", "U2", "V2")]
            info = dict(res=float(z["res"]), steps=int(z["steps"]))
            print("  · 复用缓存 %s" % cp, flush=True)
        else:
            x, info = warm78.warm_fp(MODE, g, eta_w); z = None
    else:
        x, info = warm78.warm_fp(MODE, g, eta_w); z = None
    if not os.path.exists(cp) or abs(float(np.load(cp)["eta"]) - eta_w) >= 1e-18:
        np.savez(cp, U1=x[0], V1=x[1], U2=x[2], V2=x[3], eta=eta_w, res=info["res"], steps=info["steps"])
    FP[g] = (x, eta_w)
    v = jac80.flat(x)
    num = float(np.linalg.norm(jac80.step48(v, eta_w, MODE, g, C) - v))
    rec("A2_fp_residual_g%g" % g, num / float(np.linalg.norm(v)) <= 1e-13,
        res=num / float(np.linalg.norm(v)), steps=info["steps"], eta_warm=eta_w)
    J = jac80.jac(x, eta_w, MODE, g, c=C)
    # A3 h-收敛
    J2 = jac80.jac(x, eta_w, MODE, g, h=2e-6, c=C)
    den = max(float(np.max(np.abs(J))), 1e-300)
    relh = float(np.max(np.abs(J - J2))) / den
    rec("A3_h_convergence_g%g" % g, relh <= 1e-3, rel=relh, normF=float(np.linalg.norm(J)))
    # A4 方向导数一致性（5 个固定随机方向）
    rr = np.random.default_rng(700 + int(g)); h = 1e-6; worst = 0.0
    for _ in range(5):
        dvec = rr.normal(size=v.size); dvec /= np.linalg.norm(dvec)
        an = J @ dvec
        fd = (jac80.step48(v + h * dvec, eta_w, MODE, g, C) - jac80.step48(v - h * dvec, eta_w, MODE, g, C)) / (2 * h)
        worst = max(worst, float(np.linalg.norm(an - fd)) / max(float(np.linalg.norm(an)), 1e-300))
    rec("A4_directional_g%g" % g, worst <= 1e-8, rel=worst)
    # A6 中性模
    w = np.linalg.eigvals(J)
    nz = int(np.sum(np.abs(w - 1.0) <= 1e-8))
    rec("A6_neutral_modes_g%g" % g, nz >= 8, n_neutral=nz, dim=w.size,
        nonneutral_max=float(np.max(np.abs(w[np.abs(w - 1.0) > 1e-6]))) if np.any(np.abs(w - 1.0) > 1e-6) else None)
    # A9 谱扫描自洽（rev 1.1 重写：用**非中性**谱半径 + ±10% 三点点测）
    # 旧版两处错（DEF-80-1）：(i) 断言 lmax(0.5ρ) ≤ 1 而 lmax 含 8 个中性模 ≡1（DEF-79-2 原样复发）；
    # (ii) 断言 lmax(ρ) ≥ 1+1e-6，但 ρ 是二分区间中点、可能落在穿越点下方，且容差与分辨率同量级。
    ec, dg = jac80.rho_scan(x, MODE, g, lo=1e-6, hi=1.0)
    r05, r09, r11 = (jac80.rho_nn(jac80.jac(x, f * ec, MODE, g, c=C)) for f in (0.5, 0.9, 1.1))
    rec("A9_spectral_scan_g%g" % g, (r05 < 1.0) and (r09 <= 1.0 + 1e-6) and (r11 > 1.0 + 1e-6),
        rho=ec, rnn_half=r05, rnn_lo=r09, rnn_hi=r11, nev=dg.get("nev"), cap_hit=dg.get("cap_hit"))

# ---- A8 二分终止判据的回归测试（N 缩小 10x；只测协议是否触发，不取 η 值）----
def bisect_probe(g, N=20000, lo=1e-6, hi=1.0, rel=0.005):
    f_lo = ref70.run_stack(C, g, lo, N, mode=MODE)["div"]
    f_hi = ref70.run_stack(C, g, hi, N, mode=MODE)["div"]
    if f_lo or not f_hi: return dict(ok=False, brackets=(bool(f_lo), bool(f_hi)))
    a, b, nev, cap = math.log(lo), math.log(hi), 0, False
    while True:
        if (b - a) <= rel: break
        if nev >= 60: cap = True; break
        m = 0.5 * (a + b); nev += 1
        if ref70.run_stack(C, g, math.exp(m), N, mode=MODE)["div"]: b = m
        else: a = m
    return dict(ok=True, brackets=(False, True), nev=nev, cap_hit=cap, width=b - a, eta=math.exp(0.5 * (a + b)))

bp = {}
for g in (3.0, 10.0):
    bp[g] = bisect_probe(g)
    rec("A7_bisect_brackets_g%g" % g, bp[g].get("ok", False), **{k: bp[g][k] for k in bp[g] if k != "ok"})
    rec("A8_bisect_terminates_g%g" % g, (not bp[g].get("cap_hit", True)) and bp[g].get("width", 9) <= 0.005,
        cap_hit=bp[g].get("cap_hit"), width=bp[g].get("width"), nev=bp[g].get("nev"))

json.dump(R, open("/home/ec2-user/sgmn-loop/dir80/out/stageA80.json", "w"), ensure_ascii=False, indent=1)
nf = [k for k, v in R.items() if not v["ok"]]
print("\n== Stage A: %d/%d PASS  失败项=%s  用时 %.0fs ==" % (len(R) - len(nf), len(R), nf or "无", time.time() - T0), flush=True)
