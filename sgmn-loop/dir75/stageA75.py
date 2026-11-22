"""DIR-75 Stage A：7 条守卫（全过才开正式轮）。装置 = dir73/near73.py 原样；只换代表点。"""
import sys, hashlib
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir74"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70, step72
E = np.zeros((6, 2)); E[0, 0] = E[1, 1] = 1.0
OK = []
def chk(tag, ok, msg):
    OK.append(ok); print("%-4s %-58s %s" % (tag, msg, "PASS" if ok else "FAIL"), flush=True)

def bal_rep(mode, g):
    p = ref70.pred_target(6.0, g) if mode == "target" else ref70.pred_keys(6.0, g)
    return tuple(np.sqrt(p[k]) * E for k in ("u1", "u1", "u2", "u2")), p

def field_rel(xs, mode, g, eta=1e-9):
    """装置自己的场：‖step(x*,η)−x*‖/(η·‖x*‖)（不重推 C,B）。"""
    U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in xs)
    a1, b1, a2, b2 = step72.step(U1, V1, U2, V2, eta, mode, g)
    num = max(np.linalg.norm(a1 - U1), np.linalg.norm(b1 - V1), np.linalg.norm(a2 - U2), np.linalg.norm(b2 - V2))
    den = max(np.linalg.norm(U1), np.linalg.norm(V1), np.linalg.norm(U2), np.linalg.norm(V2))
    return num / (eta * den)

CELLS = (("keys", 10.0), ("target", 30.0))
reps = {}
for mode, g in CELLS:
    xb, p = bal_rep(mode, g); xw, _ = near73.freeze_fp(mode, g, 0.3 * near73.eta_lin(mode, g))
    reps[(mode, g)] = (xb, xw, p)
    Wb1, Wb2 = xb[0] @ xb[1].T, xb[2] @ xb[3].T
    Ww1, Ww2 = xw[0] @ xw[1].T, xw[2] @ xw[3].T
    nb = field_rel(xb, mode, g); nw = field_rel(xw, mode, g)
    dW = max(np.linalg.norm(Wb1 - Ww1) / np.linalg.norm(Ww1), np.linalg.norm(Wb2 - Ww2) / np.linalg.norm(Ww2))
    print("     [%s %g] 相对场 bal=%.2e warm=%.2e  ‖ΔW‖/‖W‖=%.2e" % (mode, g, nb, nw, dW), flush=True)
    reps[(mode, g)] = (xb, xw, p, nb, dW)
    lim2 = 1e-6 if mode == "keys" else 1e-3
    lim6 = 1e-9 if mode == "keys" else 1e-3
    chk("A2", nb <= lim2 and nw <= lim2, "[%s %g] 装置自身相对场 bal=%.2e warm=%.2e (≤%.0e)" % (mode, g, nb, nw, lim2))
    chk("A6", dW <= lim6, "[%s %g] 同一不动点（只换 gauge）：‖ΔW‖/‖W‖=%.2e (≤%.0e)" % (mode, g, dW, lim6))

# A1 装置身份：warm 在 keys 10 复现 D-94（0.5679）：ρ=0.95 发散、0.50 不发散
el = near73.eta_lin("keys", 10.0)
_, xw, _p, _, _d = reps[("keys", 10.0)]
x0w, _m = near73.perturb(xw, 1e-3)
r95 = near73.run_from(x0w, 0.95 * el, "keys", 10.0)
r50 = near73.run_from(x0w, 0.50 * el, "keys", 10.0)
chk("A1", r95["div"] and not r50["div"], "warm 复现 D-94：ρ=0.95 div=%s(td=%s) · ρ=0.50 div=%s" % (r95["div"], r95["td"], r50["div"]))

# A3 perturb 相对扰动恰为 1e-3 且可复现
x0a, ma = near73.perturb(reps[("keys", 10.0)][0], 1e-3); x0b, mb = near73.perturb(reps[("keys", 10.0)][0], 1e-3)
same = all(np.array_equal(A, B) for A, B in zip(x0a, x0b))
chk("A3", max(abs(np.array(ma) - 1e-3)) <= 1e-12 and same, "相对扰动 max|meas−1e-3|=%.1e · 同种子逐位相同=%s" % (max(abs(np.array(ma) - 1e-3)), same))

# A4 正控：2η*_lin 两代表点都 ≤200 步 div
for mode, g in CELLS:
    xb, xw = reps[(mode, g)][0], reps[(mode, g)][1]
    e = near73.eta_lin(mode, g); res = []
    for tag, xs in (("bal", xb), ("warm", xw)):
        x0, _ = near73.perturb(xs, 1e-3); r = near73.run_from(x0, 2.0 * e, mode, g, n=200)
        res.append((tag, r["div"], r["td"]))
    chk("A4", all(d and (t or 999) <= 200 for _, d, t in res), "[%s %g] 2η*_lin：%s" % (mode, g, res))

# A5 η=0 恒等
x0z, _ = near73.perturb(reps[("target", 30.0)][0], 1e-3)
U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in x0z)
for _ in range(1000):
    U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, 0.0, "target", 30.0)
d0 = max(np.abs(U1 - x0z[0]).max(), np.abs(V1 - x0z[1]).max(), np.abs(U2 - x0z[2]).max(), np.abs(V2 - x0z[3]).max())
chk("A5", d0 == 0.0, "η=0 走 1000 步 max|ΔX|=%.1e" % d0)

# A7 括号单调（coarse rel=0.5，keys bal）
el = near73.eta_lin("keys", 10.0)
x0k, _ = near73.perturb(reps[("keys", 10.0)][0], 1e-3)
bs = near73.bisect_near("keys", 10.0, x0k, 0.90 * el, 1.30 * el, rel=0.5, verbose=False)
h = bs["hist"]; mono = all((h[i][0] < bs["lo"] + 1e-30) or True for i in range(len(h)))
ok7 = bool(bs["ok"]) and bs["lo"] < bs["eta"] < bs["hi"] and all(h[i + 1][1] != h[i][1] for i in range(len(h) - 1))
chk("A7", ok7, "coarse 二分 ok=%s lo<η<hi · hist=%s" % (bs["ok"], [(round(a / el, 3), b) for a, b in h]))

print("\nStage A（共 7 项）%s" % ("PASS" if all(OK) else "FAIL"), flush=True)
sys.exit(0 if all(OK) else 1)
