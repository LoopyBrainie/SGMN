"""DIR-78 Stage A：A1 warm 不动点 / A2 同一 W / A3 J(0)=I / A4 eig vs 幂迭代 / A5 窗口可穿越 /
A6 正负控 / A7 二分单调 / A8 闭式工作量例。"""
import sys, math
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir78"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir77")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir74"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72, jac77, rho78, warm78
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
CELLS = [("target", 3.0), ("target", 10.0), ("target", 30.0), ("keys", 3.0), ("keys", 10.0)]
OK = []
def chk(t, ok, m): OK.append(bool(ok)); print("%-4s %-72s %s" % (t, m, "PASS" if ok else "FAIL"), flush=True)

reps = {}
for mode, g in CELLS:
    el = near73.eta_lin(mode, g)
    p = (ref70.pred_target if mode == "target" else ref70.pred_keys)(6.0, g)
    bal = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
    xw, fp = warm78.warm_fp(mode, g, 0.3 * el)
    reps[(mode, g)] = dict(el=el, bal=bal, warm=xw, fres=fp["res"])
    print("     [%s %g] η*_lin=%.7g warm 收敛: fp_res0=%.2e → %.2e（%d 步）" % (mode, g, el, fp["res0"], fp["res"], fp["steps"]), flush=True)

# A1 warm 是该场的不动点（用装置自己的场）
bad1 = []
for (mode, g), r in reps.items():
    U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in r["warm"])
    a1, b1, a2, b2 = step72.step(U1, V1, U2, V2, 1e-9, mode, g)
    num = max(np.linalg.norm(a1 - U1), np.linalg.norm(b1 - V1), np.linalg.norm(a2 - U2), np.linalg.norm(b2 - V2))
    den = max(np.linalg.norm(U1), np.linalg.norm(V1), np.linalg.norm(U2), np.linalg.norm(V2))
    rel = num / (1e-9 * den)
    if rel > 1e-6: bad1.append((mode, g, rel))
chk("A1", not bad1, "warm 相对场 ≤1e-06 5/5（异常=%s）" % (bad1 or "-"))

# A2 同一 W*
bad2 = []
for (mode, g), r in reps.items():
    d = max(np.linalg.norm(r["bal"][0] @ r["bal"][1].T - r["warm"][0] @ r["warm"][1].T) / np.linalg.norm(r["warm"][0] @ r["warm"][1].T),
            np.linalg.norm(r["bal"][2] @ r["bal"][3].T - r["warm"][2] @ r["warm"][3].T) / np.linalg.norm(r["warm"][2] @ r["warm"][3].T))
    if d > 1e-9: bad2.append((mode, g, d))
print("     [A2 逐格] " + " · ".join("%s%g:%.2e" % (m[0], g, max(np.linalg.norm(r["bal"][0] @ r["bal"][1].T - r["warm"][0] @ r["warm"][1].T) / np.linalg.norm(r["warm"][0] @ r["warm"][1].T), np.linalg.norm(r["bal"][2] @ r["bal"][3].T - r["warm"][2] @ r["warm"][3].T) / np.linalg.norm(r["warm"][2] @ r["warm"][3].T))) for (m, g), r in reps.items()), flush=True)
chk("A2", not bad2, "‖W_bal−W_warm‖/‖W_warm‖ ≤1e-09 5/5（异常=%s）" % (bad2 or "-"))

# A3 J(0)=I（10 点）
bad3 = []
for (mode, g), r in reps.items():
    for gname in ("bal", "warm"):
        J0 = jac77.jac(r[gname], 0.0, mode, g)
        e = float(np.max(np.abs(J0 - np.eye(48))))
        if e > 1e-6: bad3.append((mode, g, gname, e))
chk("A3", not bad3, "J(η=0)=I₄₈（max|J−I|≤1e-06，差分本底 2.2e-07）10/10（异常=%s）" % (bad3 or "-"))

# A4 特征对残差（rev 1.1）
bad4 = []
for mode, g, gname in [("target", 30.0, "warm"), ("keys", 10.0, "warm"), ("target", 30.0, "bal")]:
    r = reps[(mode, g)]; J = jac77.jac(r[gname], 1.10 * r["el"], mode, g)
    w, V = np.linalg.eig(J); k = int(np.argmax(np.abs(w)))
    lamv, v = w[k], V[:, k]
    rres = float(np.linalg.norm(J @ v - lamv * v) / np.linalg.norm(v))
    i2 = np.argsort(-np.abs(w)); gap = float(abs(w[i2[1]]))
    print("     [A4 %s %g %s] λ=%.12f 残差=%.1e |λ2|=%.6f" % (mode, g, gname, abs(lamv), rres, gap), flush=True)
    if not rres <= 1e-10: bad4.append((mode, g, gname, rres))
chk("A4", not bad4, "特征对残差 ‖Jv−λv‖/‖v‖ ≤1e-10 3/3（异常=%s）" % (bad4 or "-"))

# A5 窗口可穿越
bad5 = []
for (mode, g), r in reps.items():
    l = rho78.lam_of(jac77.jac(r["warm"], 1.30 * r["el"], mode, g))
    if not l > 1 + 1e-6: bad5.append((mode, g, l))
chk("A5", not bad5, "warm 在 1.30η*_lin 处 λ_max>1+1e-06 5/5（异常=%s）" % (bad5 or "-"))

# A6 正负控
bad6 = []
for (mode, g), r in reps.items():
    x0, _ = near73.perturb(r["warm"], 1e-3, seed=20261118)
    r_lo = near73.run_from(x0, 0.30 * r["el"], mode, g, n=2000)
    r_hi = near73.run_from(x0, 1.50 * r["el"], mode, g, n=2000)
    ok = (not r_lo["div"]) and r_hi["div"] and (r_hi["td"] or 9999) <= 2000
    print("     [A6 %s %g] 0.30η*:div=%s · 1.50η*:div=%s td=%s" % (mode, g, r_lo["div"], r_hi["div"], r_hi["td"]), flush=True)
    if not ok: bad6.append((mode, g, r_lo["div"], r_hi["div"]))
chk("A6", not bad6, "0.30η* 不发散 且 1.50η* ≤2000 步发散 5/5（异常=%s）" % (bad6 or "-"))

# A7 二分单调（warm target 30，rel=0.5）
r = reps[("target", 30.0)]
x0, _ = near73.perturb(r["warm"], 1e-3, seed=20261118)
bs = near73.bisect_near("target", 30.0, x0, 0.30 * r["el"], 1.50 * r["el"], rel=0.5, verbose=False)
a, b, mid_ok, shr = math.log(0.30 * r["el"]), math.log(1.50 * r["el"]), True, True
for eta_t, d in bs["hist"][2:]:
    mid_ok &= abs(math.log(eta_t) - 0.5 * (a + b)) <= 1e-12
    b0, a0 = b, a
    if d: b = math.log(eta_t)
    else: a = math.log(eta_t)
    shr &= (b - a) < (b0 - a0)
chk("A7", bs["ok"] and bs["lo"] < bs["eta"] < bs["hi"] and mid_ok and shr,
    "ok=%s lo<η<hi=%s 中点=%s 收缩=%s hist=%s" % (bs["ok"], bs["lo"] < bs["eta"] < bs["hi"], mid_ok, shr,
                                                [(round(x / r["el"], 3), d) for x, d in bs["hist"]]))

# A8 闭式工作量例
rho_toy, b = rho78.toy_rho(3.7), 3.7
ref8 = (2.0 + rho78.TOL) / b
rel8 = abs(rho_toy - ref8) / ref8
at_half = abs(abs(1.0 - (1.0 / b) * b))
chk("A8", rel8 <= 1e-12 and at_half <= 1e-15,
    "标量松弛 b=3.7：ρ_toy=%.15f vs (2+TOL)/b=%.15f 相对差=%.1e；|λ(1/b)|=%.1e" % (rho_toy, ref8, rel8, at_half))

print("\nStage A（共 %d 项）%s" % (len(OK), "PASS" if all(OK) else "FAIL"), flush=True)
sys.exit(0 if all(OK) else 1)
