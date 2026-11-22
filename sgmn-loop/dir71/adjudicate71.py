"""DIR-71 裁决器。读法约定（**写于任何二分数据存在之前**）：
 · 指数：α ≡ ln[η*(3)/η*(10)]/ln(10/3)，用**二分值**（不是括号中点）。
 · prefactor：C(mode,γ) ≡ η*(γ)·γ²，直接用二分值。
 · 判据③：η*(100) 用**预注册网格的括号几何中点**；预言用 C̄ = sqrt(C_keys(3)·C_keys(10))。
 · 预测核对（§3）：target 三点、keys 两点，容差 ±10%（预测容差，非判据）。
"""
import json, math, os, sys
ROOT = "/home/ec2-user/sgmn-loop/dir71/out/grid"
def J(fn):
    p = os.path.join(ROOT, fn)
    if not os.path.exists(p): return None
    return json.load(open(p))
def bis(mode, g):
    r = J("bisect_%s_g%g.json" % (mode, g))
    if r is None or not r.get("ok") or not math.isfinite(r["eta"]): return None
    return r
def cells(mode, g):
    r = J("%s_g%g.json" % (mode, g))
    return None if r is None else r["cells"]
def bracket_from(cs, order):
    last = None
    for e in order:
        k = "%.0e" % e
        if k not in cs: return None
        if cs[k]["div"]: return (last, e)
        last = e
    return (last, None)
T = {g: bis("target", g) for g in (3, 10, 30)}
K = {g: bis("keys", g) for g in (3, 10)}
for nm, D in (("target", T), ("keys", K)):
    for g, r in D.items():
        if r is None: print("✗ 缺 %s γ=%g 的二分（或 ok=False / 非有限）" % (nm, g), flush=True); sys.exit(2)
E = lambda r: r["eta"]
alpha = lambda D: math.log(E(D[3]) / E(D[10])) / math.log(10.0 / 3.0)
C = lambda r: E(r) * r["gamma"] ** 2
aT, aK = alpha(T), alpha(K)
R3 = C(K[3]) / C(T[3]); R10 = C(K[10]) / C(T[10])
Rbar = 0.5 * (R3 + R10)
dr = abs(R3 - R10) / Rbar
c1 = (1.9 <= aT <= 2.1) and (1.9 <= aK <= 2.1)
c2 = dr < 0.25
CbarK = math.sqrt(C(K[3]) * C(K[10]))
pred100 = CbarK * 1e-4
cs100 = cells("keys", 100)
g100 = bracket_from(cs100, [4e-5, 8e-5, 1.2e-4, 1.7e-4, 2.4e-4, 3.4e-4]) if cs100 else None
if g100 and g100[0] and g100[1]: mid100 = math.sqrt(g100[0] * g100[1])
elif g100 and g100[0] and not g100[1]: mid100 = g100[0] * math.sqrt(1.5)
elif g100 and g100[1] and not g100[0]: mid100 = g100[1] / math.sqrt(1.5)
else: mid100 = float("nan")
c3 = math.isfinite(mid100) and abs(mid100 / pred100 - 1.0) <= 0.40
print("=" * 84)
print("[二分值] target: " + " · ".join("γ=%g η*=%.5g C=%.4g" % (g, E(T[g]), C(T[g])) for g in (3, 10, 30)))
print("[二分值] keys  : " + " · ".join("γ=%g η*=%.5g C=%.4g" % (g, E(K[g]), C(K[g])) for g in (3, 10)))
print("-" * 84)
print("判据① 指数  target=%.3f  keys=%.3f  (都须 ∈[1.9,2.1])  ⇒ %s" % (aT, aK, "PASS" if c1 else "FAIL"))
print("判据② R(3)=%.3f  R(10)=%.3f  相对差=%.1f%%  (<25%%)      ⇒ %s" % (R3, R10, 100 * dr, "PASS" if c2 else "FAIL"))
print("判据③ keys γ=100：C̄=%.4g ⇒ 预言 %.4g；网格括号=%s ⇒ 实测(中点) %.4g；偏差 %s  ⇒ %s"
      % (CbarK, pred100, g100, mid100, ("%.1f%%" % (100 * (mid100 / pred100 - 1))) if math.isfinite(mid100) else "nan", "PASS" if c3 else "FAIL"))
print("-" * 84)
print("预测核对（±10%，报告项）：")
for g, p in ((3, 0.0611), (10, 5.50e-3), (30, 6.11e-4)):
    d = 100 * (E(T[g]) / p - 1); print("   target γ=%-3g 实测 %.5g 预言 %.5g 偏差 %+.1f%%" % (g, E(T[g]), p, d))
for g, p in ((3, 0.1825), (10, 1.645e-2)):
    d = 100 * (E(K[g]) / p - 1); print("   keys   γ=%-3g 实测 %.5g 预言 %.5g 偏差 %+.1f%%" % (g, E(K[g]), p, d))
cs30 = cells("keys", 30); g30 = bracket_from(cs30, [1e-3, 1.4e-3, 1.9e-3, 2.6e-3, 3.5e-3]) if cs30 else None
m30 = math.sqrt(g30[0] * g30[1]) if (g30 and g30[0] and g30[1]) else float("nan")
print("   keys   γ=30  网格括号 %s 中点 %.4g；预言 C̄/900=%.4g" % (g30, m30, CbarK / 900.0))
verdict = "SUCCEEDED" if (c1 and c2 and c3) else ("PARTIAL" if (c1 or c2 or c3) else "FALSIFIED")
print("-" * 84); print("★ 裁决 = %s" % verdict)
json.dump(dict(alpha=dict(target=aT, keys=aK), R=dict(r3=R3, r10=R10, reldiff=dr), c1=c1, c2=c2, c3=c3,
               Cbar_keys=CbarK, pred100=pred100, mid100=mid100, g100=g100, g30=g30,
               eta={m: {str(g): E(D[g]) for g in D} for m, D in (("target", T), ("keys", K))},
               verdict=verdict), open("/home/ec2-user/sgmn-loop/dir71/out/verdict71.json", "w"), indent=1)
