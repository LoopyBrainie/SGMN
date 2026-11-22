"""DIR-82 单格执行器
python cell82.py <mode> <gamma> <init> <warmrule> <label>
  warmrule = fp   → warm78.warm_fp（DIR-79/80 记录 rho 时用的源规则）
            = init → jac81.warm_from_init（DIR-81 的 init 扫描规则）
一个格测：raw rho（2e-3 与 1e-8 两档）+ gfix_qr / gfix_polar 的 rho + lmax 函数级比对
        + 仪器（幂等/W不变/正交/不动点/核维/O(2)）。不跑轨迹（eta_th 冻结在 D-79/D-80）。
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jac82, warm78

T0 = time.time()
MODE, G, INIT, RULE, LABEL = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
OUT = "dir82/out/cell_%s.json" % LABEL
TAG = "%s %s" % (LABEL, MODE)


def log(m):
    print("[%6.0fs] %-22s %s" % (time.time() - T0, TAG, m), flush=True)


def flush(**kw):
    os.makedirs("dir82/out", exist_ok=True)
    d = dict(label=LABEL, mode=MODE, gamma=G, init=INIT, rule=RULE,
             elapsed=round(time.time() - T0, 1), stage=kw.pop("stage", "run"), **kw)
    json.dump(d, open(OUT, "w"), ensure_ascii=False, indent=1)


flush(stage="start")


def pick(tab, g):
    for k, v in tab.items():
        if not isinstance(v, dict) or "eta_th" not in v:
            continue
        try:
            kv = float(k)
        except ValueError:
            continue
        if abs(kv - g) < 1e-9:
            return v
    raise KeyError("找不到 gamma=%g" % g)


def eta_th_of(mode, g):
    path, kind = {"target": ("dir79/out/DIR79_formal.json", "mech"),
                  "keys": ("dir80/out/DIR80_formal.json", "cells")}[mode]
    d = json.load(open(path))
    return float(pick(d if kind == "mech" else d["cells"], g)["eta_th"]), path


def lmax_at(x, etas):
    return np.array([jac82.lmax(jac82.jac(x, e, MODE, G, c=6.0)) for e in etas])


eta_th, src = eta_th_of(MODE, G)
eta_w = 0.1 * eta_th
log("eta_th=%.12g eta_w=%.6g rule=%s" % (eta_th, eta_w, RULE))

# ---------- 1. warm ----------
if RULE == "fp":
    x, info = warm78.warm_fp(MODE, G, eta_w)
    wsteps = info["steps"]
else:
    x, info = jac82.warm_from_init(MODE, G, eta_w, INIT, seed=20261118)
    wsteps = info["steps"]
log("warm steps=%d res=%.2e" % (wsteps, info["res"]))
flush(stage="warm", eta_th=eta_th, eta_warm=eta_w, warm_steps=wsteps, warm_res=info["res"])

# ---------- 2. raw rho 两档分辨率 ----------
rho_raw2, dg_raw2 = jac82.rho_scan(x, MODE, G, lo=1e-6, hi=1.0, rel=2e-3)
rho_raw8, dg_raw8 = jac82.rho_scan(x, MODE, G, lo=1e-6, hi=1.0, rel=1e-8)
log("raw rho(2e-3)=%.12g  rho(1e-8)=%.12g" % (rho_raw2, rho_raw8))

# ---------- 3. 归一化 ----------
xq = jac82.gfix_qr(x)
xp = jac82.gfix_polar(x)


def maxabs(a, b):
    for v in (a, b):
        if not np.all(np.isfinite(np.asarray(v, float))):
            raise AssertionError("non-finite")
    return max(float(np.max(np.abs(np.asarray(p, float) - np.asarray(q, float))))
               for p, q in zip(a, b))


g_idem = maxabs(jac82.gfix_qr(xq), xq)
w_err = max(float(np.linalg.norm(xq[k] @ xq[k + 1].T - x[k] @ x[k + 1].T)) for k in (0, 2))
w_rel = w_err / max(float(np.linalg.norm(x[k] @ x[k + 1].T)) for k in (0, 2))
orth_qr = max(jac82.orth_err(xq[1]), jac82.orth_err(xq[3]))
orth_pol = max(jac82.orth_err(xp[1]), jac82.orth_err(xp[3]))
v_q = jac82.flat(xq)
fp_after = float(np.linalg.norm(jac82.step48(v_q, eta_w, MODE, G, 6.0) - v_q)) / float(np.linalg.norm(v_q))
kd_raw, gap_raw = jac82.ker_dim(x, eta_w, MODE, G)
kd_qr, gap_qr = jac82.ker_dim(xq, eta_w, MODE, G)
log("gfix idem=%.1e w_rel=%.1e orth=%.1e fp_after=%.1e ker=%d->%d"
    % (g_idem, w_rel, max(orth_qr, orth_pol), fp_after, kd_raw, kd_qr))

# ---------- 4. lmax 函数级比对（判据 ① 的精确版：不经二分） ----------
etas = np.array([1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
la_q = lmax_at(xq, etas)
la_p = lmax_at(xp, etas)
fn_rel = float(np.max(np.abs(la_q - la_p) / np.maximum(np.abs(la_q), 1e-300)))
log("lmax 函数级 qr-vs-polar 相对差=%.3e" % fn_rel)

# ---------- 5. gfix 后 rho ----------
rho_g2, dg_g2 = jac82.rho_scan(xq, MODE, G, lo=1e-6, hi=1.0, rel=2e-3)
rho_g8, dg_g8 = jac82.rho_scan(xq, MODE, G, lo=1e-6, hi=1.0, rel=1e-8)
rho_p10, dg_p10 = jac82.rho_scan(xp, MODE, G, lo=1e-6, hi=1.0, rel=1e-10)
rho_q10, dg_q10 = jac82.rho_scan(xq, MODE, G, lo=1e-6, hi=1.0, rel=1e-10)
two_rel = abs(rho_q10 - rho_p10) / abs(rho_q10)
log("gfix rho(2e-3)=%.12g rho(1e-8)=%.12g  两归一化 rel=%.2e" % (rho_g2, rho_g8, two_rel))

# ---------- 6. O(2) 不变性 ----------
rho_o2 = {}
for tag, Gm in jac82.ORTHO_G:
    r, _ = jac82.rho_scan(jac82.apply_gauge(xq, Gm), MODE, G, lo=1e-6, hi=1.0, rel=1e-8)
    rho_o2[tag] = r
vv = np.array([v for v in rho_o2.values() if np.isfinite(v)])
o2_spread = float((vv.max() - vv.min()) / vv.mean()) if vv.size else float("nan")

flush(stage="done", eta_th=eta_th, eta_warm=eta_w, warm_steps=wsteps,
      rho_raw_2e3=rho_raw2, dg_raw_2e3=dg_raw2,
      rho_raw_1e8=rho_raw8, dg_raw_1e8=dg_raw8,
      rho_qr_2e3=rho_g2, dg_qr_2e3=dg_g2,
      rho_qr_1e8=rho_g8, dg_qr_1e8=dg_g8,
      rho_qr_1e10=rho_q10, rho_polar_1e10=rho_p10, two_norm_rel=two_rel,
      lmax_fn_rel=fn_rel, lmax_etas=etas.tolist(), lmax_qr=la_q.tolist(), lmax_polar=la_p.tolist(),
      gfix=dict(idem=g_idem, w_rel=w_rel, orth_qr=orth_qr, orth_pol=orth_pol,
                fp_after=fp_after, ker_raw=kd_raw, ker_qr=kd_qr, gap_raw=gap_raw, gap_qr=gap_qr,
                condV1=float(np.linalg.cond(x[1]))),
      rho_o2=rho_o2, o2_spread=o2_spread,
      ratio_raw=rho_raw2 / eta_th, ratio_raw8=rho_raw8 / eta_th,
      ratio_qr=rho_g2 / eta_th, ratio_qr8=rho_g8 / eta_th)
log("DONE ratio_raw=%.4f ratio_qr=%.4f" % (rho_raw2 / eta_th, rho_g2 / eta_th))
