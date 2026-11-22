"""DIR-83 Stage A：硬门禁 A1–A8（卡 §5）。任一 FAIL 阻断正式轮。
A5/A7/A8 检验的是 dev83 里正式轮**实际调用**的函数。
"""
import json, os, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir83")
import numpy as np
import dev83 as dv
import ref70, step72, jac82

OUT = dv.OUT
RES = {}


def rec(k, ok, detail):
    RES[k] = dict(ok=bool(ok), detail=detail)
    print("  %-4s %-5s %s" % (k, "PASS" if ok else "FAIL", detail), flush=True)


print("=" * 90)
print("DIR-83 Stage A（卡 §5，阈值冻结）")
print("=" * 90)

# ---- A1: gfix_qr(x*) 的固定点残差（四个 γ）----
a1 = []
for g in dv.GAM:
    xc = dv.load_fp(g)
    r = dv.lookup(g)
    eta_w = 0.1 * r["eta_th"]
    n = step72.step(*xc, eta_w, "target", g, dv.C)
    num = max(float(np.linalg.norm(n[k] - xc[k])) for k in range(4))
    den = max(float(np.linalg.norm(xc[k])) for k in range(4))
    res = num / den
    # 诊断：η_mid 上的残差（不动点应与 η 无关）
    n2 = step72.step(*xc, dv.eta_mid(g), "target", g, dv.C)
    num2 = max(float(np.linalg.norm(n2[k] - xc[k])) for k in range(4))
    den2 = max(float(np.linalg.norm(xc[k])) for k in range(4))
    a1.append(dict(g=g, eta_w=eta_w, res=res, res_mid=num2 / den2))
    print("    γ=%-4g res(η_w)=%.3e  res(η_mid)=%.3e" % (g, res, num2 / den2), flush=True)
rec("A1", all(x["res"] <= 1e-13 for x in a1),
    "四格 res ≤ 1e-13（max=%.2e）" % max(x["res"] for x in a1))
RES["A1"]["rows"] = a1

# ---- A2: ||W*||_F / (sqrt2 * u1_pred) - 1（四个 γ，显式带 sqrt2，DEF-83-1）----
a2 = []
for g in dv.GAM:
    xc = dv.load_fp(g)
    w1, w2 = dv.wnorm(xc)
    u1p = float(ref70.pred_target(dv.C, g)["u1"])
    pred = np.sqrt(2.0) * u1p
    rel = w1 / pred - 1.0
    a2.append(dict(g=g, w1=w1, u1_pred=u1p, pred_fro=pred, rel=rel, w2=w2))
    print("    γ=%-4g ‖W1‖_F=%.6f  √2u1=%.6f  rel=%+.2e" % (g, w1, pred, rel), flush=True)
rec("A2", all(abs(x["rel"]) <= 1e-4 for x in a2),
    "四格 |rel| ≤ 1e-4（max=%.2e）" % max(abs(x["rel"]) for x in a2))
RES["A2"]["rows"] = a2

# ---- A3: gfix_qr 后 V 的正交误差 ----
a3 = []
for g in dv.GAM:
    a, b, c_, d_ = dv.load_fp(g)
    e1, e2 = jac82.orth_err(b), jac82.orth_err(d_)
    a3.append(dict(g=g, V1=e1, V2=e2))
    print("    γ=%-4g ‖V1ᵀV1−I‖=%.2e  ‖V2ᵀV2−I‖=%.2e" % (g, e1, e2), flush=True)
rec("A3", all(max(x["V1"], x["V2"]) <= 1e-13 for x in a3),
    "四格 ≤ 1e-13（max=%.2e）" % max(max(x["V1"], x["V2"]) for x in a3))
RES["A3"]["rows"] = a3

# ---- A4: 算例 γ=30, η=η_th 的 μ_tail，对 −4.79e-9 相对差 ≤5%（T 加倍自洽）----
REF_MU = -4.79e-9
r30 = dv.lookup(30.0)
a4rows = []
for N in (200000, 400000):
    tr = dv.traj_init(30.0, r30["eta_th"], N)
    mu = dv.mu_tail(tr["ns"], N)
    a4rows.append(dict(N=N, mu=mu, t_nan=tr["t_nan"], t_div=tr["t_div"], mx=tr["mx"]))
    if mu is None:
        print("    N=%-7d μ_tail=None（尾部样本不足）" % N, flush=True)
    else:
        print("    N=%-7d μ_tail=%+.6e  相对差=%s" %
              (N, mu, ("%+.3f%%" % (100 * (mu - REF_MU) / REF_MU)) if mu and REF_MU else "n/a"), flush=True)
mu_main = a4rows[0]["mu"]
ok_a4 = (mu_main is not None) and (abs((mu_main - REF_MU) / REF_MU) <= 0.05)
rec("A4", ok_a4,
    "N=2e5 的 μ_tail=%s，对 %.2e 相对差=%s（≤5%%）" %
    (("%+.6e" % mu_main) if mu_main is not None else "None", REF_MU,
     ("%+.3f%%" % (100 * (mu_main - REF_MU) / REF_MU)) if mu_main is not None else "n/a"))
RES["A4"]["rows"] = a4rows

# ---- A5: 结构断言 —— 非有限必须走 not-isfinite 分支，不当数值比较 ----
nan, inf = float("nan"), float("inf")
cases = [((nan, 1.0), True), ((1.0, nan), True), ((inf, 1.0), True), ((1.0, inf), True),
         ((1.0, 1.0), False), ((1e4, 1.0), True), ((1e3, 1.0), False)]
a5ok, a5det = True, []
for (w1, w2), want in cases:
    try:
        got = dv.diverged(w1, w2)
    except Exception as e:                                    # 数值比较遇 NaN 会静默 False 或抛错
        a5ok = False
        a5det.append("%r→异常(%s)" % ((w1, w2), type(e).__name__))
        continue
    good = (got is want)
    a5ok = a5ok and good
    a5det.append("%r→%s%s" % ((w1, w2), got, "" if good else "(期望%s)" % want))
rec("A5", a5ok, "非有限走 not-isfinite 分支；" + "; ".join(a5det))
RES["A5"]["cases"] = a5det

# ---- A6: η_th < η_mid < ρ_λ（四格）----
a6 = []
for g in dv.GAM:
    r = dv.lookup(g)
    m = dv.eta_mid(g)
    a6.append(dict(g=g, eta_th=r["eta_th"], eta_mid=m, rho=r["rho"],
                   ok=bool(r["eta_th"] < m < r["rho"]),
                   ratio_th=m / r["eta_th"], ratio_rho=m / r["rho"]))
    print("    γ=%-4g η_mid=%.9g  /η_th=%.4f  /ρ=%.4f" %
          (g, m, m / r["eta_th"], m / r["rho"]), flush=True)
rec("A6", all(x["ok"] for x in a6), "四格严格 η_th<η_mid<ρ_λ")
RES["A6"]["rows"] = a6

# ---- A7: 结构断言 —— Λ 溢出走 capped 分支，不抛异常 ----
a7ok, a7det = True, []
try:
    J1 = 2.0 * np.eye(8)                       # ||J^t||=2^t 越 1e12
    d1 = dv.lam_sup(J1, 200)
    a7ok = a7ok and d1["ok"] and d1["capped"] and d1["overflow_t"] is not None
    a7det.append("2I:T=200→capped=%s,overflow_t=%s" % (d1["capped"], d1["overflow_t"]))
except Exception as e:
    a7ok = False
    a7det.append("2I→异常(%s)" % type(e).__name__)
try:
    J2 = 1e150 * np.eye(8)                     # 两步即非有限
    d2 = dv.lam_sup(J2, 50)
    a7ok = a7ok and d2["overflow_t"] is not None
    a7det.append("1e150I:capped=%s,overflow_t=%s" % (d2["capped"], d2["overflow_t"]))
except Exception as e:
    a7ok = False
    a7det.append("1e150I→异常(%s)" % type(e).__name__)
try:                                            # 正常格不得被误判为 capped
    J3 = 0.5 * np.eye(8)
    d3 = dv.lam_sup(J3, 100)
    a7ok = a7ok and d3["ok"] and (not d3["capped"]) and abs(d3["sup"] - 1.0) < 1e-12
    a7det.append("0.5I:sup=%.3f,capped=%s" % (d3["sup"], d3["capped"]))
except Exception as e:
    a7ok = False
    a7det.append("0.5I→异常(%s)" % type(e).__name__)
rec("A7", a7ok, "; ".join(a7det))
RES["A7"]["cases"] = a7det

# ---- A8: 历史读数一律数值匹配，不得用 %g 拼键（纪律 192）----
a8ok, a8det = True, []
for g in dv.GAM:
    try:
        r = dv.lookup(g)
        a8ok = a8ok and (abs(r["g"] - g) <= 1e-9)
        a8det.append("γ=%g→η_th=%.9g" % (g, r["eta_th"]))
    except Exception as e:
        a8ok = False
        a8det.append("γ=%g→%s" % (g, type(e).__name__))
# 演示风险真实存在：json 的浮点键是 str(float)，而 "%g" 会给出不同字符串
mism = [(g, str(g), "%g" % g) for g in dv.GAM if str(g) != ("%g" % g)]
a8ok = a8ok and len(mism) > 0
a8det.append("%d/%d 个 γ 上 str(γ)≠%%g(γ)（如 %s vs %s）⇒ 拼键必错" %
             (len(mism), len(dv.GAM), mism[0][1] if mism else "-", mism[0][2] if mism else "-"))
try:
    dv.lookup(7.0)
    a8ok = False
    a8det.append("γ=7 应 KeyError 却命中")
except KeyError:
    a8det.append("γ=7 正确 KeyError")
rec("A8", a8ok, "; ".join(a8det))
RES["A8"]["detail"] = a8det

# ---- 汇总 ----
fails = [k for k, v in RES.items() if k.startswith("A") and not v["ok"]]
print("=" * 90)
print("Stage A 结果：%d/%d PASS" % (8 - len(fails), 8), "| FAIL:", fails or "无")
print("门禁：", "放行正式轮" if not fails else "★ 阻断正式轮 ★")
print("=" * 90)
json.dump(dict(rows=RES, fails=fails, gate_open=(not fails)),
          open(os.path.join(OUT, "stageA83.json"), "w"), ensure_ascii=False, indent=1)
sys.exit(0 if not fails else 1)