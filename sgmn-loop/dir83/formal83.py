"""DIR-83 正式轮：卡 §3 的四项门禁判据（阈值冻结，不得改动）。
① μ(η_mid) < 0 四格   ② Λ(η_mid) ≤ 2.0 四格（T=2e4）
★③ 在 η_mid 上 ε=3e-2 的 ‖W‖ 始终有限、且 ε=1e-1 变为非有限（括号不越出）四格
④ 阴性对照：η=0.5ρ 上 ε=1 收敛（有限且 max‖W‖/‖W*‖ ≤ 3）四格
"""
import json, os, sys, time
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir83")
import numpy as np
import dev83 as dv
import jac79

OUT = dv.OUT
N = 200000
T_LAM = 20000
RES = {}
T0 = time.time()


def flush():
    json.dump(RES, open(os.path.join(OUT, "formal83.json.tmp"), "w"), ensure_ascii=False, indent=1)
    os.replace(os.path.join(OUT, "formal83.json.tmp"), os.path.join(OUT, "formal83.json"))


print("=" * 96)
print("DIR-83 正式轮 · 四判据 × 四 γ（阈值冻结）")
print("=" * 96)

rows = {}
for g in dv.GAM:
    r = dv.lookup(g)
    xc = dv.load_fp(g)
    w1s, w2s = dv.wnorm(xc)
    m = dv.eta_mid(g)
    eta_th, rho = r["eta_th"], r["rho"]
    row = dict(g=g, eta_th=eta_th, rho=rho, eta_mid=m, W_ref=w1s, W2_ref=w2s)
    print("\n--- γ=%g  (η_th=%.9g, ρ=%.9g, η_mid=%.9g, ‖W1*‖=%.6f) ---" %
          (g, eta_th, rho, m, w1s), flush=True)

    # μ 在三个 η 上（主判据用 η_mid；另两个作次级读数 c 的支撑）
    mus = {}
    for tag, eta in (("half_rho", 0.5 * rho), ("eta_th", eta_th), ("eta_mid", m)):
        J = jac79.jac(xc, eta, "target", g, c=dv.C)
        mu, nn, lmax = dv.mu_of_jac(J)
        mus[tag] = dict(eta=eta, mu=mu, n_nonneutral=nn, lmax=lmax)
        print("    μ: %-9s η=%.9g  max非中性|λ|=%.9f  μ=%+.4e" % (tag, eta, lmax, mu), flush=True)
    row["mu"] = mus

    # ① 线性稳定
    row["c1_ok"] = bool(mus["eta_mid"]["mu"] < 0)
    print("    ① μ(η_mid)=%+.4e < 0 ? %s" % (mus["eta_mid"]["mu"], row["c1_ok"]), flush=True)

    # Λ 在三个 η 上
    lams = {}
    for tag, eta in (("half_rho", 0.5 * rho), ("eta_th", eta_th), ("eta_mid", m)):
        J = jac79.jac(xc, eta, "target", g, c=dv.C)
        d = dv.lam_sup(J, T_LAM)
        lams[tag] = d
        print("    Λ: %-9s sup=%.6f peak_t=%-6s capped=%s" %
              (tag, d["sup"], d["peak_t"], d["capped"]), flush=True)
    row["lam"] = lams

    # ② 瞬态有界
    row["c2_ok"] = bool(lams["eta_mid"]["sup"] <= 2.0)
    print("    ② Λ(η_mid)=%.6f ≤ 2.0 ? %s" % (lams["eta_mid"]["sup"], row["c2_ok"]), flush=True)

    # ★③ 有限幅度：η_mid 上 ε=3e-2 必须始终有限，ε=1e-1 必须变非有限
    e_small = dv.traj_eps(g, m, 3e-2, N)
    print("    ③ ε=3e-2 : finite=%s t_nan=%s t_div=%s mx_rel=%.4f" %
          (e_small["finite"], e_small["t_nan"], e_small["t_div"], e_small["mx_rel"]), flush=True)
    e_big = dv.traj_eps(g, m, 1e-1, N)
    print("    ③ ε=1e-1 : finite=%s t_nan=%s t_div=%s mx_rel=%.4f" %
          (e_big["finite"], e_big["t_nan"], e_big["t_div"], e_big["mx_rel"]), flush=True)
    row["eps_small"], row["eps_big"] = e_small, e_big
    row["c3_ok"] = bool(e_small["finite"] and (not e_big["finite"]))
    print("    ③ 括号 [3e-2,1e-1] 未越出 ? %s" % row["c3_ok"], flush=True)

    # ④ 阴性对照：η=0.5ρ 上 ε=1 也应收敛
    ctrl = dv.traj_eps(g, 0.5 * rho, 1.0, N)
    print("    ④ 0.5ρ,ε=1 : finite=%s t_nan=%s mx_rel=%.4f" %
          (ctrl["finite"], ctrl["t_nan"], ctrl["mx_rel"]), flush=True)
    row["ctrl"] = ctrl
    row["c4_ok"] = bool(ctrl["finite"] and ctrl["mx_rel"] <= 3.0)
    print("    ④ 有限且 mx_rel≤3 ? %s" % row["c4_ok"], flush=True)

    rows[str(g)] = row
    RES["rows"] = rows
    RES["elapsed"] = time.time() - T0
    flush()

# ---- 汇总 ----
print("\n" + "=" * 96)
hdr = "%-6s %-14s %-12s %-34s %-12s" % ("γ", "①μ(η_mid)", "②Λ(η_mid)", "★③ ε=3e-2 finite / 1e-1 finite", "④对照")
print(hdr)
allok = {"1": True, "2": True, "3": True, "4": True}
for g in dv.GAM:
    r = rows[str(g)]
    print("%-6g %-14s %-12s %-34s %-12s" % (
        g, "%+.3e(%s)" % (r["mu"]["eta_mid"]["mu"], "P" if r["c1_ok"] else "F"),
        "%.4f(%s)" % (r["lam"]["eta_mid"]["sup"], "P" if r["c2_ok"] else "F"),
        "%s / %s (%s)" % (r["eps_small"]["finite"], r["eps_big"]["finite"], "P" if r["c3_ok"] else "F"),
        "%.3f(%s)" % (r["ctrl"]["mx_rel"], "P" if r["c4_ok"] else "F")))
    for k, key in (("1", "c1_ok"), ("2", "c2_ok"), ("3", "c3_ok"), ("4", "c4_ok")):
        allok[k] = allok[k] and r[key]

# 次级读数
mu_floor = {g: float(np.log(1e3 / dv.lookup(g)["W"]) / N) for g in dv.GAM}
sec_b = {str(g): dict(mu_eta_th=rows[str(g)]["mu"]["eta_th"]["mu"], mu_floor=mu_floor[g],
                      ratio=abs(rows[str(g)]["mu"]["eta_th"]["mu"]) / mu_floor[g]) for g in dv.GAM}
sec_c = {str(g): {t: rows[str(g)]["lam"][t]["sup"] for t in ("half_rho", "eta_th", "eta_mid")} for g in dv.GAM}
sec_d = {str(g): dict(eps_3e2_finite=rows[str(g)]["eps_small"]["finite"],
                      eps_1e1_finite=rows[str(g)]["eps_big"]["finite"]) for g in dv.GAM}

RES.update(allok=allok, sec_b=sec_b, sec_c=sec_c, sec_d=sec_d,
           elapsed=time.time() - T0,
           h83_confirmed=all(allok.values()))
flush()

print("\n次级读数 b：|μ(η_th)|/μ_floor =", " / ".join("%.2f" % sec_b[str(g)]["ratio"] for g in dv.GAM))
print("次级读数 c：Λ 在 η/ρ∈{0.5, η_th, η_mid} 上的极差 =",
      " / ".join("%.3f" % (max(sec_c[str(g)].values()) - min(sec_c[str(g)].values())) for g in dv.GAM))
print("\n判据：① %s ② %s ③ %s ④ %s" % tuple("PASS" if allok[k] else "FAIL" for k in "1234"))
print("H83（有限幅度）%s" % ("CONFIRMED" if RES["h83_confirmed"] else "FALSIFIED"))
print("耗时 %.1f s" % RES["elapsed"])
print("=" * 96)