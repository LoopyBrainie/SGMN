r"""DIR-61 · ESC 逃逸律（d2r1 主装置 = 算例 A 原样；d4r1/d4r2 = 新装置）

动力学逐字取自 dir61/workex61.py 与 dir28/R1_couple.py：
  G = (Wk - v) k^T + lam (W - W0);  同步  U <- U - eta G V,  V <- V - eta G^T U,  W = U V^T
tau = 首次到达 1/2 (J0 + Jss) 的步数；未到达记 CENSORED（**不**记成 Nmax）。
"""
import argparse
import json
import math
import os

import numpy as np

LAM, ETA = 0.05, 0.05
NMAX = 400_000
NREF = 20_000


def device(variant):
    """返回 (d, r, k, v, U0_builder, V0_builder, Jss_provider)。"""
    if variant == "d2r1":
        d, r = 2, 1
        k = np.array([[1.0], [0.0]]); v = np.array([[0.0], [1.0]])
    elif variant == "d4r1":
        d, r = 4, 1
        k = np.zeros((4, 1)); k[0, 0] = k[1, 0] = 1.0
        v = np.zeros((4, 1)); v[2, 0] = v[3, 0] = 1.0
    elif variant == "d4r2":
        d, r = 4, 2
        k = np.zeros((4, 1)); k[0, 0] = k[1, 0] = 1.0
        v = np.zeros((4, 1)); v[2, 0] = v[3, 0] = 1.0
    else:
        raise SystemExit("variant?")
    return d, r, k, v


def inits(variant, theta):
    """死块 + 投影 theta；返回 (U0, V0)。
    d2r1: U0 ∝ e1 (⊥v), V0 = cos·e2 + sin·e1   （workex61 原样）
    d4r1: U0 ∝ e1 (⊥v), V0 = cos·e3 + sin·k̂    （同构嵌入 d=4）
    d4r2: 两列同倾角：U0 = [e1, e2]（均 ⊥ v）, V0 = [cos·e3 + sin·k̂, cos·e4 + sin·k̂]
    """
    if variant == "d2r1":
        U0 = np.array([[1.0], [0.0]])
        V0 = np.array([[math.sin(theta)], [math.cos(theta)]])
        return U0, V0
    d, r, k, v = device(variant)
    kh = k / np.linalg.norm(k)
    if variant == "d4r1":
        U0 = np.zeros((4, 1)); U0[0, 0] = 1.0
        V0 = math.cos(theta) * np.array([[0.0], [0.0], [1.0], [0.0]]) + math.sin(theta) * kh
        return U0, V0
    U0 = np.zeros((4, 2)); U0[0, 0] = 1.0; U0[1, 1] = 1.0
    e3 = np.array([[0.0], [0.0], [1.0], [0.0]]); e4 = np.array([[0.0], [0.0], [0.0], [1.0]])
    V0 = np.hstack([math.cos(theta) * e3 + math.sin(theta) * kh,
                    math.cos(theta) * e4 + math.sin(theta) * kh])
    return U0, V0


def J(W, W0, k, v):
    res = W @ k - v
    return 0.5 * float(np.sum(res * res)) + 0.5 * LAM * float(np.sum((W - W0) ** 2))


def run_fac(variant, theta, nmax=NMAX, eta=ETA, sync=True):
    d, r, k, v = device(variant)
    U, V = inits(variant, theta)
    W0 = U @ V.T
    j0 = J(W0, W0, k, v)
    align, cosv, vtu = [], [], []
    tau = None
    for t in range(1, nmax + 1):
        W = U @ V.T
        G = (W @ k - v) @ k.T + LAM * (W - W0)
        if sync:
            U, V = U - eta * (G @ V), V - eta * (G.T @ U)
        else:                                  # 异步（负对照）：V 用更新后 U 重算的 G
            U = U - eta * (G @ V)
            W2 = U @ V.T
            G2 = (W2 @ k - v) @ k.T + LAM * (W2 - W0)
            V = V - eta * (G2.T @ U)
        Wn = U @ V.T
        nV = float(np.linalg.norm(V))
        align.append(float(np.linalg.norm(k.T @ V)))
        cosv.append(float(np.linalg.norm(k.T @ V)) / nV if nV > 0 else 0.0)
        vtu.append(float(np.linalg.norm(U.T @ v)))
        if tau is None and j0 > 1e-300:
            pass
        if tau is None and J(Wn, W0, k, v) <= 0.5 * (j0 + jss_of(variant, W0)):
            tau = t
            break
    return {"tau": tau, "censored": tau is None, "J0": j0,
            "J_end": J(U @ V.T, W0, k, v), "align": align, "cos": cosv, "vtu": vtu,
            "W0": W0, "U": U, "V": V}


_JSS_CACHE = {}


def jss_of(variant, W0):
    """Jss = 同装置、同 W0、theta=pi/2 的长期终态 J（参照水位，**预注册**；非判据阈值）。"""
    key = variant
    if key not in _JSS_CACHE:
        d, r, k, v = device(variant)
        U, V = inits(variant, math.pi / 2)
        W0h = U @ V.T
        for _ in range(NREF):
            W = U @ V.T
            G = (W @ k - v) @ k.T + LAM * (W - W0h)
            U, V = U - ETA * (G @ V), V - ETA * (G.T @ U)
        _JSS_CACHE[key] = J(U @ V.T, W0h, k, v)
    return _JSS_CACHE[key]


def run_full(theta, nmax=NMAX):
    """d2r1 全矩阵参考（workex61.run_full 原样）。"""
    U, V = inits("d2r1", theta)
    k = np.array([[1.0], [0.0]]); v = np.array([[0.0], [1.0]])
    W0 = U @ V.T
    W = W0.copy()
    j0 = J(W, W0, k, v)
    jss = J(np.array([[0.0, 0.0], [1.0 / (1.0 + LAM), 0.0]]), W0, k, v)
    for t in range(1, nmax + 1):
        G = (W @ k - v) @ k.T + LAM * (W - W0)
        W = W - ETA * G
        if J(W, W0, k, v) <= 0.5 * (j0 + jss):
            return t
    return None


def theta_grid():
    return [10 ** (q / 4.0) for q in range(-16, 1)]


def r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - yhat) ** 2)) / ss if ss > 0 else float("nan")


def fits(rows, lmin=None):
    """两族模型在**同一响应尺度 τ** 上比较（rev 1.1 修正 DEF-61-3）。
    log 族: tau = a + b*L, L = ln(1/sin)          （指数逃逸 ⇒ b>0, 对 L 线性）
    pow 族: ln tau = c + alpha*ln(sin) ⇒ tau = e^c sin^alpha  （幂律）
    """
    pts = [(r["theta"], r["tau"]) for r in rows if r["tau"] is not None and r["tau"] > 0]
    if lmin is not None:
        pts = [(t, tau) for t, tau in pts if math.log(1.0 / math.sin(t)) >= lmin]
    if len(pts) < 3:
        return {"n": len(pts), "log": None, "pow": None}
    s = np.array([math.sin(t) for t, _ in pts]); tt = np.array([float(x) for _, x in pts])
    L = np.log(1.0 / s)
    A = np.vstack([np.ones_like(L), L]).T
    cl, *_ = np.linalg.lstsq(A, tt, rcond=None)                 # 直接拟合 tau
    pred_log = A @ cl
    B = np.vstack([np.ones_like(s), np.log(s)]).T
    cp, *_ = np.linalg.lstsq(B, np.log(tt), rcond=None)         # 拟合 ln tau
    pred_pow = np.exp(B @ cp)
    # 退化诊断（DEF-61-3 的留档证据）：把 log 族也放到 ln tau 尺度上，与 pow 族完全同一回归
    Al = np.vstack([np.ones_like(np.log(s)), np.log(1.0 / s)]).T
    cl2, *_ = np.linalg.lstsq(Al, np.log(tt), rcond=None)
    r2log_lnscale = r2(np.log(tt), Al @ cl2)
    return {"n": len(pts),
            "log": {"a": float(cl[0]), "b": float(cl[1]), "R2": float(r2(tt, pred_log))},
            "pow": {"c": float(cp[0]), "alpha": float(cp[1]), "R2": float(r2(tt, pred_pow))},
            "R2_gap": float(r2(tt, pred_log) - r2(tt, pred_pow)),
            "DEFECT_degenerate": {"R2_log_lnscale": float(r2log_lnscale),
                                  "identical_to_pow": bool(abs(r2log_lnscale - r2(np.log(tt), B @ cp)) < 1e-12)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="d2r1")
    ap.add_argument("--nmax", type=int, default=NMAX)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = []
    for th in theta_grid():
        r = run_fac(a.variant, th, nmax=a.nmax)
        al = r.pop("align"); cs = r.pop("cos"); vt = r.pop("vtu")
        rows.append({"theta": th, "sin_theta": math.sin(th), "tau": r["tau"],
                     "censored": r["censored"], "J0": r["J0"], "J_end": r["J_end"],
                     "align_min": float(np.min(al)) if al else None,
                     "align_max": float(np.max(al)) if al else None,
                     "align_monotone": bool(all(al[i] <= al[i + 1] * (1 + 1e-12) + 1e-15 for i in range(len(al) - 1))),
                     "align_n_monotone_viol": int(sum(1 for i in range(len(al) - 1) if al[i] > al[i + 1] * (1 + 1e-12) + 1e-15)),
                     "cos_min": float(np.min(cs)) if cs else None,
                     "align_max_rel_drop": float(max([(al[i] - al[i + 1]) / max(al[i], 1e-300)
                                                     for i in range(len(al) - 1)] + [0.0])),
                     "vtu_monotone": bool(all(vt[i] <= vt[i + 1] * (1 + 1e-12) + 1e-15 for i in range(len(vt) - 1))),
                     "vtu_n_viol": int(sum(1 for i in range(len(vt) - 1) if vt[i] > vt[i + 1] * (1 + 1e-12) + 1e-15)),
                     "vtu_start": vt[0] if vt else None, "vtu_end": vt[-1] if vt else None,
                     "n_steps": len(al)})
        print("%-6s theta=%.3e  tau=%s  J0=%.4f  J_end=%.4f  align %s→%s  mono=%s"
              % (a.variant, th, r["tau"], r["J0"], r["J_end"],
                 "%.2e" % rows[-1]["align_min"], "%.2e" % rows[-1]["align_max"],
                 rows[-1]["align_monotone"]))
    f = fits(rows)
    f_hi = fits(rows, lmin=4.605)          # 仅小 θ（L>=4.6）的**诊断**，不进判据
    print("★ fits(all):", json.dumps(f))
    print("★ fits(L>=4.6, diag):", json.dumps(f_hi))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"variant": a.variant, "nmax": a.nmax, "rows": rows, "fits": f, "fits_diag_hiL": f_hi},
                  open(a.out, "w"), indent=1)
