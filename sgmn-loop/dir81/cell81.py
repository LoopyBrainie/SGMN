"""DIR-81 单格执行器：python cell81.py <mode> <init>
一个格 = 二分 eta_th(init) → 有效性检查(t_div @1.05 eta_th) → warm_from_init(0.1 eta_th)
        → 谱穿越 rho_lambda → sup_t||J^t||(T=5000, 非门禁观测)。
逐 eta 缓存 + 逐阶段增量落盘 + 心跳（DEF-79-6 / 163 / 171 / 176）。
"""
import sys, os, math, json, time
for q in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79", "dir80", "dir81"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np, ref70, jac81

OUT = "/home/ec2-user/sgmn-loop/dir81/out"
N, SD, REL, C, G, T = 200000, 20261118, 0.005, 6.0, 10.0, 5000
MODE = sys.argv[1]; INIT = float(sys.argv[2])
TAG = "%s_%03d" % (MODE, int(round(INIT * 100)))
PATH = os.path.join(OUT, "cell_%s.json" % TAG)
T0 = time.time()
S = {}


def flush(**kw):
    S.update(kw)
    S["secs"] = round(time.time() - T0, 1)
    json.dump(S, open(PATH + ".tmp", "w"), ensure_ascii=False, indent=1)
    os.replace(PATH + ".tmp", PATH)


def hb(msg):
    print("[%6.0fs] %s init=%-5g " % (time.time() - T0, TAG, INIT) + msg, flush=True)


def run(eta):
    r = ref70.run_stack(C, G, eta, N, seed=SD, init=INIT, mode=MODE)
    return dict(div=bool(r["div"]), td=r["td"], steps=int(r["steps"]))


# ---------- 0. 复用 ----------
_prev = {}
if os.path.exists(PATH):
    try: _prev = json.load(open(PATH))
    except Exception: _prev = {}
S = dict(_prev)
cache = {k: v for k, v in S.get("eta_cache", {}).items()}
flush(mode=MODE, g=G, init=INIT, n=N, rel=REL, stage=S.get("stage", "start"))


def run_cached(eta):
    k = "%.12g" % eta
    if k in cache:
        return cache[k]
    r = run(eta)
    cache[k] = r
    flush(eta_cache=cache)
    return r


# ---------- 1. 二分 eta_th ----------
if S.get("stage") in ("bisect_done", "valid_done", "warm_done", "rho_done", "done", "done_all") and S.get("eta_th"):
    eta_th = float(S["eta_th"])
    print("  · 复用二分 eta_th=%.8g" % eta_th, flush=True)
else:
    flush(stage="bisect", eta_th=None)
    lo, hi = 1e-6, 1.0
    f_lo, f_hi = run_cached(lo)["div"], run_cached(hi)["div"]
    flush(brackets=[bool(f_lo), bool(f_hi)])
    if f_lo or not f_hi:
        flush(stage="bisect_failed", ok=False)
        hb("括号无效 lo_div=%s hi_div=%s ⇒ 该格判据从未被计算" % (f_lo, f_hi)); sys.exit(0)
    a, b, nev, cap, steps_tot = math.log(lo), math.log(hi), 0, False, 0
    while True:
        if (b - a) <= REL: break
        if nev >= 60: cap = True; break
        m = 0.5 * (a + b); nev += 1
        r = run_cached(math.exp(m)); steps_tot += r["steps"]
        if r["div"]: b = m
        else: a = m
        flush(theta=math.exp(0.5 * (a + b)), nev=nev, width=b - a, cap_hit=cap)
        if nev % 3 == 0: hb("bisect nev=%d eta=%.5g div=%s" % (nev, math.exp(m), r["div"]))
    eta_th = float(math.exp(0.5 * (a + b)))
    flush(stage="bisect_done", eta_th=eta_th, nev=nev, cap_hit=cap, width=b - a, steps_bisect=steps_tot)
    hb("eta_th=%.6g (nev=%d cap=%s width=%.4f)" % (eta_th, nev, cap, b - a))

# ---------- 2. 有效性：1.05 eta_th 处 t_div ----------
if S.get("stage") in ("valid_done", "warm_done", "rho_done", "done", "done_all") and "t_div_probe" in S:
    valid = S["valid"]; t_div = S["t_div_probe"]
else:
    eta_p = 1.05 * eta_th
    r = run_cached(eta_p)
    t_div = r["td"] if r["div"] else None
    # 非截断限制：t_div <= N/10；括号内 1.05*eta_th 必须发散
    valid = bool(r["div"] and t_div is not None and t_div <= N // 10)
    flush(stage="valid_done", t_div_probe=t_div, valid=valid, valid_note=(
        "ok" if valid else ("not_diverged_at_1.05" if not r["div"] else "truncation_limited")))
    hb("1.05*eta_th: div=%s t_div=%s ⇒ valid=%s" % (r["div"], t_div, valid))

# ---------- 3. warm 不动点（从本格自己的 init 出发） ----------
FPPATH = os.path.join(OUT, "fp_%s.npz" % TAG)
eta_w = float(S.get("eta_warm") or 0.1 * eta_th)


def load_fp():
    if not os.path.exists(FPPATH): return None
    z = np.load(FPPATH)
    return tuple(np.array(z[k], float) for k in ("U1", "V1", "U2", "V2"))


x = load_fp()
if x is not None and S.get("stage") in ("warm_done", "rho_done", "done", "done_all"):
    print("  · 复用 warm npz eta_warm=%.6g 步数=%s" % (eta_w, S.get("fp_steps")), flush=True)
elif x is not None:
    print("  · 复用 warm npz（阶段=%s）" % S.get("stage"), flush=True)
else:
    flush(stage="warm", eta_warm=eta_w)
    x, info = jac81.warm_from_init(MODE, G, eta_w, INIT, seed=SD)
    np.savez(FPPATH, U1=x[0], V1=x[1], U2=x[2], V2=x[3])
    flush(stage="warm_done", eta_warm=eta_w, fp_res=info["res_fp"], fp_steps=info["steps"],
          warm_info=info)
    hb("warm steps=%d res=%.2e res_fp=%.2e u1=%.6g dW1_norm=%.4g" % (
        info["steps"], info["res"], info["res_fp"], info["u1_warm"], info["normW1"]))

warm_conv = bool(S.get("warm_info", {}).get("converged", True))

# ---------- 4. 谱穿越 ----------
if S.get("stage") in ("rho_done", "done", "done_all") and S.get("rho") is not None:
    rho = float(S["rho"])
    print("  · 复用 rho=%.8g" % rho, flush=True)
elif not warm_conv:
    flush(stage="rho_skipped", rho=None, rho_note="warm_not_converged")
    hb("warm 未收敛 ⇒ 判据 ② 该格从未被计算"); sys.exit(0)
else:
    rho, dg = jac81.rho_scan(x, MODE, G, lo=1e-6, hi=1.0)
    J = jac81.jac(x, eta_w, MODE, G, c=C)
    flush(stage="rho_done", rho=rho, rho_diag=dg, rho_nn=jac81.rho_nn(J))
    hb("rho_lambda=%.6g (nev=%d cap=%s) ratio=%.5f rho_nn=%.6g" % (
        rho, dg["nev"], dg["cap_hit"], rho / eta_th, S["rho_nn"]))

# ---------- 5. sup_t||J^t||（非门禁观测） ----------
if S.get("stage") in ("done", "done_all") and S.get("sup") is not None:
    sp = S["sup"]
else:
    J = jac81.jac(x, eta_w, MODE, G, c=C)
    sp = jac81.supjt(J, T=T)
    flush(stage="done", sup=sp)
    hb("sup=%.4f @t=%d interior=%s" % (sp["sup"], sp["peak_t"], sp["interior_peak"]))

flush(stage="done_all")
print("DONE %s eta_th=%.8g rho=%.8g gap=%.6f valid=%s t_div=%s warm_steps=%s" % (
    TAG, eta_th, rho, rho / eta_th - 1, S.get("valid"), S.get("t_div_probe"),
    S.get("fp_steps")), flush=True)
