"""DIR-79 正式轮：gamma in {0,3,10,20,30} × (轨迹二分 eta_th + warm 代表点谱穿越 rho_lambda)。
判据 ①②③ 见卡 rev 1.1（阈值冻结）。"""
import sys, math, json
for q in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79"): sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np, ref70, warm78, jac79, os, time
OUT = "/home/ec2-user/sgmn-loop/dir79/out/DIR79_formal.json"
def _flush(d):
    json.dump(d, open(OUT + ".tmp", "w"), ensure_ascii=False, indent=1); os.replace(OUT + ".tmp", OUT)

N, SD, INIT, REL, C = 200000, 20261118, 0.1, 0.005, 6.0
GAM = [0.0, 3.0, 10.0, 20.0, 30.0]

def bisect79(g, lo=1e-6, hi=1.0, c=C, n=N, rel=REL, seed=SD, init=INIT):
    f_lo = ref70.run_stack(c, g, lo, n, seed=seed, init=init, mode="target")["div"]
    f_hi = ref70.run_stack(c, g, hi, n, seed=seed, init=init, mode="target")["div"]
    if f_lo or not f_hi:
        return dict(eta=float("nan"), ok=False, brackets=(bool(f_lo), bool(f_hi)), nev=0)
    a, b, nev = math.log(lo), math.log(hi), 0
    while True:
        # DEF-79-4 修正：log 坐标下区间宽度 b-a 就是相对宽度；ref69 的 `(b-a) < rel*(a+b)/2`
        # 在 eta<1 时右端为负 ⇒ 永不触发（被 for _ in range(80) 掩盖）。改为 b-a <= rel，并设上限。
        if (b - a) <= rel or nev >= 60: break
        m = 0.5 * (a + b); nev += 1
        dv = ref70.run_stack(c, g, math.exp(m), n, seed=seed, init=init, mode="target")["div"]
        if nev % 4 == 0: print("    bisect γ=%g nev=%d eta=%.4g div=%s t=%.0fs" % (g, nev, math.exp(m), dv, time.time() - T0), flush=True)
        if dv: b = m
        else: a = m
    return dict(eta=float(math.exp(0.5 * (a + b))), ok=True, brackets=(False, True), nev=nev)

out = {}
T0 = time.time()
for g in GAM:
    # warm 代表点：用 0.1*eta_th 的步长（比卡里 0.3 更保守，避免落在阈值附近）；找不到就退 0.3
    bt = bisect79(g)
    eta_ref = bt["eta"] if bt["ok"] else 0.3
    x, info = warm78.warm_fp("target", g, 0.1 * eta_ref)
    rho, dg = jac79.rho_scan(x, "target", g, lo=1e-6, hi=1.0)
    J = jac79.jac(x, 0.1 * eta_ref, "target", g)
    w = np.linalg.eigvals(J)
    nn = np.abs(w[np.abs(w - 1.0) > 1e-6])
    out[str(g)] = dict(eta_th=bt["eta"], bisect_ok=bt["ok"], brackets=list(bt["brackets"]), nev=bt["nev"],
                       rho=float(rho), rho_diag=dg, eta_warm=0.1 * eta_ref,
                       fp_res=info["res"], fp_steps=info["steps"],
                       lmax_at_warm=float(np.max(np.abs(w))), nonneutral_max=float(np.max(nn)) if nn.size else None,
                       )
    r = out[str(g)]
    _flush(out)
    print("γ=%-4g eta_th=%.6g (nev=%d) rho=%.6g 比=%.5f fp_res=%.1e steps=%d  t=%.0fs"
          % (g, r["eta_th"], r["nev"], r["rho"], r["rho"] / r["eta_th"], r["fp_res"], r["fp_steps"],
             time.time() - T0), flush=True)

# 判据
cells = [10.0, 20.0, 30.0]
rel = [abs(out[str(g)]["rho"] / out[str(g)]["eta_th"] - 1) for g in cells]
c1 = all(r <= 0.03 for r in rel)
lg = np.log(cells); lr = np.log([out[str(g)]["rho"] for g in cells])
slope = float(np.polyfit(lg, lr, 1)[0])
Cfit = float(np.exp(np.mean(np.log([out[str(g)]["rho"] for g in cells]) + 2 * lg)))
pred = dict(C1=dict(ok=c1, rel=[float(r) for r in rel], thr=0.03),
            C2=dict(ok=(-2.6 <= slope <= -1.4), slope=slope, band=[-2.6, -1.4]),
            C3=dict(ok=(0.40 <= Cfit <= 0.75), C=Cfit, ref=0.55))
out["verdict"] = pred
_flush(out)
print(json.dumps({k: v for k, v in pred.items()}, ensure_ascii=False, indent=1))
print("判据: ① %s ② %s ③ %s" % ("PASS" if c1 else "FAIL", "PASS" if pred["C2"]["ok"] else "FAIL", "PASS" if pred["C3"]["ok"] else "FAIL"))
