"""DIR-77 Stage A：8 条守卫（5 种子 + 装置身份 + J 可诊断性）。装置 = near73/step72/ref70 原样。"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir77"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir74")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72, jac77
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
G, MODE, EL = 10.0, "keys", near73.eta_lin("keys", 10.0)
SEEDS = [20261118, 20261119, 20261120, 20261121, 20261122]
OK = []
def chk(tag, ok, msg):
    OK.append(bool(ok)); print("%-4s %-66s %s" % (tag, msg, "PASS" if ok else "FAIL"), flush=True)

p = ref70.pred_keys(6.0, G)
xb = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
print("     [bal keys γ=10] η*_lin=%.7g u1*=%.9f u2*=%.9f ‖W1*‖=%.6e" % (EL, p["u1"], p["u2"], np.linalg.norm(xb[0] @ xb[1].T)), flush=True)

# A2 用装置自己的场
U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in xb)
a1, b1, a2, b2 = step72.step(U1, V1, U2, V2, 1e-9, MODE, G)
num = max(np.linalg.norm(a1 - U1), np.linalg.norm(b1 - V1), np.linalg.norm(a2 - U2), np.linalg.norm(b2 - V2))
den = max(np.linalg.norm(U1), np.linalg.norm(V1), np.linalg.norm(U2), np.linalg.norm(V2))
relf = num / (1e-9 * den); chk("A2", relf <= 1e-6, "bal 相对场 %.2e ≤ 1e-06" % relf)

# A3 5 种子可复现
x0s, bad = {}, []
for sd in SEEDS:
    x1, m1 = near73.perturb(xb, 1e-3, seed=sd); x2, m2 = near73.perturb(xb, 1e-3, seed=sd)
    same = all(np.array_equal(A, B) for A, B in zip(x1, x2)) and list(m1) == list(m2)
    d = max(abs(np.array(m1) - 1e-3))
    x0s[sd] = x1
    if not (same and d <= 1e-12): bad.append((sd, same, d))
chk("A3", not bad, "5 种子逐位可复现 + 相对扰动=1e-3  （异常=%s）" % (bad or "-"))

# A4 正控
divs = []
for sd in SEEDS:
    r = near73.run_from(x0s[sd], 2.0 * EL, MODE, G, n=2000); divs.append((sd, r["div"], r["td"]))
chk("A4", all(d and (t or 99999) <= 2000 for _, d, t in divs), "2η*_lin：%s" % [(s % 100, t) for s, d, t in divs])

# A5
U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in x0s[SEEDS[1]])
for _ in range(1000):
    U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, 0.0, MODE, G)
d0 = max(np.abs(U1 - x0s[SEEDS[1]][0]).max(), np.abs(V1 - x0s[SEEDS[1]][1]).max(),
         np.abs(U2 - x0s[SEEDS[1]][2]).max(), np.abs(V2 - x0s[SEEDS[1]][3]).max())
chk("A5", d0 == 0.0, "η=0 走 1000 步 max|ΔX|=%.1e" % d0)

# A6 gauge
xw, fp = near73.freeze_fp(MODE, G, 0.3 * EL)
dW = max(np.linalg.norm(xb[0] @ xb[1].T - xw[0] @ xw[1].T) / np.linalg.norm(xw[0] @ xw[1].T),
         np.linalg.norm(xb[2] @ xb[3].T - xw[2] @ xw[3].T) / np.linalg.norm(xw[2] @ xw[3].T))
chk("A6", dW <= 1e-9, "‖ΔW‖/‖W‖=%.2e ≤ 1e-09 (fp_res=%.2e)" % (dW, fp["res"]))

# A1 装置身份
xw10, _ = near73.freeze_fp("keys", 10.0, 0.3 * EL); x0w, _ = near73.perturb(xw10, 1e-3)
r95 = near73.run_from(x0w, 0.95 * EL, "keys", 10.0); r50 = near73.run_from(x0w, 0.50 * EL, "keys", 10.0)
chk("A1", r95["div"] and not r50["div"], "warm 复现 D-94：ρ=0.95 div=%s(td=%s) · ρ=0.50 div=%s" % (r95["div"], r95["td"], r50["div"]))

# A7 中点/收缩复算（rel=0.5，种子 20261119）
import math
bs = near73.bisect_near(MODE, G, x0s[20261119], 0.85 * EL, 1.50 * EL, rel=0.5, verbose=False)
a, b, mid_ok, shrink_ok = math.log(0.85 * EL), math.log(1.50 * EL), True, True
for eta_t, d in bs["hist"][2:]:
    mid_ok &= abs(math.log(eta_t) - 0.5 * (a + b)) <= 1e-12
    b0, a0 = b, a
    if d: b = math.log(eta_t)
    else: a = math.log(eta_t)
    shrink_ok &= (b - a) < (b0 - a0)
chk("A7", bs["ok"] and bs["lo"] < bs["eta"] < bs["hi"] and mid_ok and shrink_ok,
    "ok=%s lo<η<hi=%s 中点=%s 收缩=%s hist=%s" % (bs["ok"], bs["lo"] < bs["eta"] < bs["hi"], mid_ok, shrink_ok,
                                                    [(round(x / EL, 3), d) for x, d in bs["hist"]]))

# A8 J 可诊断性
J = jac77.jac(xb, 1.02 * EL, MODE, G, h=1e-9)
w95, _ = jac77.top_pair(J)
J2 = jac77.jac(xb, 1.02 * EL, MODE, G, h=1e-8); w98, _ = jac77.top_pair(J2)
wl = jac77.top_pair(jac77.jac(xb, 0.5 * EL, MODE, G))[0]
wh = jac77.top_pair(jac77.jac(xb, 2.0 * EL, MODE, G))[0]
relh = abs(abs(w98) - abs(w95)) / abs(w95)
chk("A8", np.all(np.isfinite(J)) and J.shape == (48, 48) and abs(wl) <= 1 + 1e-6 and abs(wh) > 1 + 1e-6 and relh <= 1e-3,
    "48×48 有限 · λ(0.5η*)=%.12f ≤1+1e-6 · λ(2η*)=%.6f >1 · h 相对差=%.1e ≤1e-3" % (abs(wl), abs(wh), relh))

print("\nStage A（共 %d 项）%s" % (len(OK), "PASS" if all(OK) else "FAIL"), flush=True)
sys.exit(0 if all(OK) else 1)
