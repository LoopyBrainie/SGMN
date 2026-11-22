"""DIR-76 Stage A：7 条守卫（3 个新格 + 装置身份）。装置 = dir75/dir73 原样；只换格清单。"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir74"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
OK = []
def chk(tag, ok, msg):
    OK.append(ok); print("%-4s %-62s %s" % (tag, msg, "PASS" if ok else "FAIL"), flush=True)

def field_rel(xs, mode, g, eta=1e-9):
    U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in xs)
    a1, b1, a2, b2 = step72.step(U1, V1, U2, V2, eta, mode, g)
    num = max(np.linalg.norm(a1 - U1), np.linalg.norm(b1 - V1), np.linalg.norm(a2 - U2), np.linalg.norm(b2 - V2))
    den = max(np.linalg.norm(U1), np.linalg.norm(V1), np.linalg.norm(U2), np.linalg.norm(V2))
    return num / (eta * den)

NEW = (("target", 3.0), ("target", 10.0), ("keys", 3.0))
reps = {}
for mode, g in NEW:
    el = near73.eta_lin(mode, g)
    p = ref70.pred_target(6.0, g) if mode == "target" else ref70.pred_keys(6.0, g)
    xb = tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2"))
    xw, fp = near73.freeze_fp(mode, g, 0.3 * el)
    fb, fw = field_rel(xb, mode, g), field_rel(xw, mode, g)
    dW = max(np.linalg.norm(xb[0] @ xb[1].T - xw[0] @ xw[1].T) / np.linalg.norm(xw[0] @ xw[1].T),
             np.linalg.norm(xb[2] @ xb[3].T - xw[2] @ xw[3].T) / np.linalg.norm(xw[2] @ xw[3].T))
    lim2 = 1e-6 if mode == "keys" else 1e-3
    lim6 = 1e-9 if mode == "keys" else max(1e-3, 10.0 * fp["res"])
    print("     [%s %g] η*_lin=%.7g  u1*=%.9f u2*=%.9f  ‖W1*‖=%.6e  fp_res=%.2e  ‖ΔW‖/‖W‖=%.2e"
          % (mode, g, el, p["u1"], p["u2"], np.linalg.norm(xb[0] @ xb[1].T), fp["res"], dW), flush=True)
    chk("A2", fb <= lim2 and fw <= lim2, "[%s %g] 相对场 bal=%.2e warm=%.2e (≤%.0e)" % (mode, g, fb, fw, lim2))
    chk("A6", dW <= lim6, "[%s %g] gauge 差 %.2e ≤ %.1e (max(1e-3,10·%.1e))" % (mode, g, dW, lim6, fp["res"]))
    reps[(mode, g)] = (xb, xw, p, fp)

# A1 装置身份：warm keys 10 复现 D-94
xw10, _fp10 = near73.freeze_fp("keys", 10.0, 0.3 * near73.eta_lin("keys", 10.0))
el10 = near73.eta_lin("keys", 10.0)
x0w, _ = near73.perturb(xw10, 1e-3)
r95 = near73.run_from(x0w, 0.95 * el10, "keys", 10.0); r50 = near73.run_from(x0w, 0.50 * el10, "keys", 10.0)
chk("A1", r95["div"] and not r50["div"], "warm keys10 复现 D-94：ρ=0.95 div=%s(td=%s) · ρ=0.50 div=%s" % (r95["div"], r95["td"], r50["div"]))

# A3
x0a, ma = near73.perturb(reps[("keys", 3.0)][0], 1e-3); x0b, mb = near73.perturb(reps[("keys", 3.0)][0], 1e-3)
same = all(np.array_equal(A, B) for A, B in zip(x0a, x0b))
chk("A3", max(abs(np.array(ma) - 1e-3)) <= 1e-12 and same, "相对扰动 max|meas−1e-3|=%.1e · 逐位相同=%s" % (max(abs(np.array(ma) - 1e-3)), same))

# A4 正控 2η*_lin、n=2000
for mode, g in NEW:
    xb, xw = reps[(mode, g)][0], reps[(mode, g)][1]
    e = near73.eta_lin(mode, g); res = []
    for tag, xs in (("bal", xb), ("warm", xw)):
        x0, _ = near73.perturb(xs, 1e-3); r = near73.run_from(x0, 2.0 * e, mode, g, n=2000)
        res.append((tag, r["div"], r["td"]))
    chk("A4", all(d and (t or 9999) <= 2000 for _, d, t in res), "[%s %g] 2η*_lin：%s" % (mode, g, res))

# A5
x0z, _ = near73.perturb(reps[("target", 3.0)][0], 1e-3)
U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in x0z)
for _ in range(1000):
    U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, 0.0, "target", 3.0)
d0 = max(np.abs(U1 - x0z[0]).max(), np.abs(V1 - x0z[1]).max(), np.abs(U2 - x0z[2]).max(), np.abs(V2 - x0z[3]).max())
chk("A5", d0 == 0.0, "η=0 走 1000 步 max|ΔX|=%.1e" % d0)

# A7 coarse 二分
el3 = near73.eta_lin("keys", 3.0)
x0k, _ = near73.perturb(reps[("keys", 3.0)][0], 1e-3)
bs = near73.bisect_near("keys", 3.0, x0k, 0.85 * el3, 1.45 * el3, rel=0.5, verbose=False)
h = bs["hist"]
# 按卡片文字：ok 且 lo<η<hi；真正的"单调" = 每次试探都是当前括号的 log 中点、且括号严格收缩（不是"div 必须交替"）
import math
a, b, mid_ok, shrink_ok = math.log(0.85 * el3), math.log(1.45 * el3), True, True
for eta_t, d in h[2:]:
    mid_ok &= abs(math.log(eta_t) - 0.5 * (a + b)) <= 1e-12
    b0, a0 = b, a
    if d: b = math.log(eta_t)
    else: a = math.log(eta_t)
    shrink_ok &= (b - a) < (b0 - a0)
ok7 = bool(bs["ok"]) and bs["lo"] < bs["eta"] < bs["hi"] and mid_ok and shrink_ok
chk("A7", ok7, "coarse 二分 ok=%s lo<η<hi=%s 中点=%s 收缩=%s · hist=%s"
    % (bs["ok"], bs["lo"] < bs["eta"] < bs["hi"], mid_ok, shrink_ok, [(round(a_ / el3, 3), d) for a_, d in h]))

print("\nStage A（共 %d 项）%s" % (len(OK), "PASS" if all(OK) else "FAIL"), flush=True)
sys.exit(0 if all(OK) else 1)
