"""DIR-80 单格执行器：python cell80.py <gamma>
一个 `keys` 格 = 二分 eta_th → warm 不动点(0.1*eta_th) → 谱穿越 rho_lambda → sup_t||J^t||(T=5000)。
逐格增量落盘 + 心跳（DEF-79-6 / 纪律 163/171：首结果产生前就能落盘）。
"""
import sys, os, math, json, time
for q in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79", "dir80"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np, ref70, warm78, jac80

OUT = "/home/ec2-user/sgmn-loop/dir80/out"
N, SD, INIT, REL, C, T = 200000, 20261118, 0.1, 0.005, 6.0, 5000
MODE = "keys"
g = float(sys.argv[1])
PATH = os.path.join(OUT, "cell_%g.json" % g)
T0 = time.time()
S = {}


def flush(**kw):
    S.update(kw)
    S["secs"] = round(time.time() - T0, 1)
    json.dump(S, open(PATH + ".tmp", "w"), ensure_ascii=False, indent=1)
    os.replace(PATH + ".tmp", PATH)


def hb(msg):
    print("[%6.0fs] γ=%-3g " % (time.time() - T0, g) + msg, flush=True)


def run(eta):
    return ref70.run_stack(C, g, eta, N, seed=SD, init=INIT, mode=MODE)


# ---------- 1. 二分 eta_th（修正后的终止判据） ----------
_prev = {}
if os.path.exists(PATH):
    try: _prev = json.load(open(PATH))
    except Exception: _prev = {}
if _prev.get("eta_th") and _prev.get("brackets") == [False, True]:
    S = dict(_prev)
    eta_th = float(_prev["eta_th"]); nev = _prev.get("nev", -1); cap = _prev.get("cap_hit", False)
    print("  · 复用二分结果 eta_th=%.8g（nev=%s）" % (eta_th, nev), flush=True)
    flush(stage="bisect_cached", eta_th=eta_th)
else:
    flush(g=g, mode=MODE, stage="bisect", eta_th=None)
    eta_th = None
    lo, hi = 1e-6, 1.0
    f_lo, f_hi = run(lo)["div"], run(hi)["div"]
    flush(brackets=[bool(f_lo), bool(f_hi)])
    if f_lo or not f_hi:
        flush(stage="bisect_failed", ok=False); hb("括号无效"); sys.exit(0)
    a, b, nev, cap, steps_tot = math.log(lo), math.log(hi), 0, False, 0
    while True:
        if (b - a) <= REL: break
        if nev >= 60: cap = True; break
        m = 0.5 * (a + b); nev += 1
        r = run(math.exp(m)); steps_tot += r["steps"]
        if r["div"]: b = m
        else: a = m
        flush(theta=math.exp(0.5 * (a + b)), nev=nev, width=b - a, cap_hit=cap)
        if nev % 3 == 0: hb("bisect nev=%d eta=%.5g div=%s" % (nev, math.exp(m), r["div"]))
    eta_th = float(math.exp(0.5 * (a + b)))
    flush(stage="bisect_done", eta_th=eta_th, nev=nev, cap_hit=cap, width=b - a, steps_bisect=steps_tot)
    hb("eta_th=%.6g (nev=%d cap=%s width=%.4f)" % (eta_th, nev, cap, b - a))

# ---------- 2. warm 不动点 ----------
eta_w = 0.1 * eta_th
x, info = warm78.warm_fp(MODE, g, eta_w)
_v = jac80.flat(x)
res = float(np.linalg.norm(jac80.step48(_v, eta_w, MODE, g, C) - _v)) / float(np.linalg.norm(_v))
pk = ref70.pred_keys(C, g) if MODE == "keys" else ref70.pred_target(C, g)
u1w = float((x[0] @ x[1].T)[0, 0]); u2w = float((x[2] @ x[3].T)[0, 0])
flush(stage="warm_done", eta_warm=eta_w, fp_res=res, fp_steps=info["steps"],
      u1_warm=u1w, u2_warm=u2w, u1_pred=float(pk["u1"]), u2_pred=float(pk["u2"]),
      normU1=float(np.linalg.norm(x[0])), normV1=float(np.linalg.norm(x[1])))
hb("warm fp_res=%.2e steps=%d" % (res, info["steps"]))

# ---------- 3. 谱穿越 ----------
eta_cross, dg = jac80.rho_scan(x, MODE, g, lo=1e-6, hi=1.0)
flush(stage="rho_done", rho=eta_cross, rho_diag=dg)
hb("rho_lambda=%.6g (nev=%d cap=%s) 比=%.5f" % (eta_cross, dg["nev"], dg["cap_hit"], eta_cross / eta_th))

# ---------- 4. sup_t||J^t|| ----------
J = jac80.jac(x, eta_w, MODE, g, c=C)
sp = jac80.supjt(J, T=T)
flush(stage="done", sup=sp)
hb("sup=%.4f @t=%d  interior=%s  ||J^5000||=%.4f" % (sp["sup"], sp["peak_t"], sp["interior_peak"], sp["at_t"]["5000"]))
print("DONE γ=%g eta_th=%.8g rho=%.8g gap=%.6f sup=%.5f" % (g, eta_th, eta_cross, eta_cross / eta_th - 1, sp["sup"]), flush=True)
