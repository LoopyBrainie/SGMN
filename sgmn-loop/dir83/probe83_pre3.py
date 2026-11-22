"""DIR-83 开卡前预言探针 · 第 3 部分（补 D(γ=10)、重做 E′、重做 F 带溢出保护）。

修正两处**自己的**设计错误：
  (i) E 原用 η = 1.25η_th，但 γ=3 时 ρ_gfix/η_th 仅 **1.0599** ⇒ 1.25η_th 已在 ρ_λ **之上**，
      测的是"线性不稳"，与缺口问题无关（`DEF-83-2`）。改为 η = 1.10η_th 且只取 γ∈{10,20,30}。
  (ii) F 在 η=ρ_gfix 上 Λ 溢出（||J^t|| 真发散）⇒ SVD 不收敛崩溃。改为带 1e12 上限 + 异常捕获。
"""
import sys, os, json, time, math
for q in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79", "dir80", "dir81", "dir82", "dir83"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np
import ref70, step72, jac79, jac81, jac82

OUT = "/home/ec2-user/sgmn-loop/dir83/out"
C, D, R = 6.0, 6, 2
N = 200000
DIV_ABS = 1e3
CAP = 1e12
SD, INIT = 20261118, 0.1

ETA_TH = {3.0: 0.064974679, 10.0: 0.0073034051, 20.0: 0.0015740607, 30.0: 0.00057804665}
RHO_GFIX = {3.0: 0.068867243, 10.0: 0.0091167253, 20.0: 0.0025007558, 30.0: 0.001147340655}


def flush(name, d):
    p = os.path.join(OUT, name)
    json.dump(d, open(p + ".tmp", "w"), ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


def hb(m):
    print("    " + m, flush=True)


def load_fp(g):
    z = np.load(os.path.join(OUT, "fp_target_g%g.npz" % g))
    return jac82.gfix_qr((z["U1"], z["V1"], z["U2"], z["V2"]))


def wn(x):
    U1, V1, U2, V2 = x
    return float(np.linalg.norm(U1 @ V1.T))


# ---------------- F. Λ(η,T)，带溢出保护 ----------------
def supT(J, T, cap=CAP):
    n = J.shape[0]
    M = np.eye(n); best, bt, mk = 1.0, 0, {}
    for t in range(1, T + 1):
        try:
            M = J @ M
            s = float(np.linalg.norm(M, 2))
        except np.linalg.LinAlgError:
            return dict(sup=best, peak_t=bt, marks=mk, overflow_t=t, ok=False)
        if not np.isfinite(s) or s > cap:
            return dict(sup=max(best, s if np.isfinite(s) else cap), peak_t=(t if s > best else bt),
                        marks=mk, overflow_t=t, ok=True, capped=True)
        if s > best: best, bt = s, t
        if t in (2000, 5000, 20000): mk[str(t)] = s
    return dict(sup=best, peak_t=bt, marks=mk, overflow_t=None, ok=True)


LAMV = {}
for g in (3.0, 10.0, 20.0, 30.0):
    xc = load_fp(g)
    rows = {}
    for tag, eta in (("eta_th", ETA_TH[g]), ("1.10th", 1.10 * ETA_TH[g]),
                     ("1.25th", 1.25 * ETA_TH[g]), ("0.5rho", 0.5 * RHO_GFIX[g])):
        J = jac79.jac(xc, eta, "target", g, c=C)
        r = supT(J, 20000)
        r.update(eta=eta, ratio_eta=eta / ETA_TH[g], below_rho=bool(eta < RHO_GFIX[g]))
        rows[tag] = r
        hb("F γ=%-4g %-8s η/η_th=%.2f(<ρ:%s)  Λ=%.4g peak_t=%-6s overflow_t=%s"
           % (g, tag, eta / ETA_TH[g], eta < RHO_GFIX[g], r["sup"], r["peak_t"], r["overflow_t"]))
        flush("probe83_F2.json", LAMV)
    LAMV[str(g)] = rows


# ---------------- D 补 γ=10 ----------------
def traj(g, eta):
    rng = np.random.default_rng(SD)
    U1 = rng.normal(size=(D, R)) * INIT; V1 = rng.normal(size=(D, R)) * INIT
    U2 = rng.normal(size=(D, R)) * INIT; V2 = rng.normal(size=(D, R)) * INIT
    every = max(1, N // 200)
    ns, tdiv, mx, bad, tcap = [], None, 0.0, False, None
    for t in range(1, N + 1):
        U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, eta, "target", g, C)
        r1 = float(np.linalg.norm(U1)); r2 = float(np.linalg.norm(V1))
        r3 = float(np.linalg.norm(U2)); r4 = float(np.linalg.norm(V2))
        if not (np.isfinite(r1) and np.isfinite(r2) and np.isfinite(r3) and np.isfinite(r4)):
            bad = True; tdiv = t; break
        w1 = float(np.linalg.norm(U1 @ V1.T)); w2 = float(np.linalg.norm(U2 @ V2.T))
        if w1 > mx: mx = w1
        if (w1 > DIV_ABS or w2 > DIV_ABS) and tdiv is None: tdiv = t
        if max(r1, r3) > CAP and tcap is None: tcap = t; break
        if t % every == 0: ns.append((t, w1))
    tail = [(t, w) for (t, w) in ns if t >= 0.8 * N]
    mu_t = None
    if len(tail) >= 3 and all(w > 0 for _, w in tail):
        tt = np.array([t for t, _ in tail], float); yy = np.log(np.array([w for _, w in tail], float))
        v = float(np.polyfit(tt, yy, 1)[0]); mu_t = v if np.isfinite(v) else None
    return dict(tdiv=tdiv, nan=bad, cap_t=tcap, mx=mx, mu_tail=mu_t)


DG = {}
for g in (10.0, 20.0):
    xc = load_fp(g); nW = wn(xc)
    rows = {}
    for mult in (1.0, 1.10, 1.25, 1.50):
        eta = mult * ETA_TH[g]
        t0 = time.time()
        r = traj(g, eta)
        rows["x%.2f" % mult] = dict(r, eta=eta, mult=mult, mx_rel=r["mx"] / nW, normW=nW,
                                    below_rho=bool(eta < RHO_GFIX[g]), secs=time.time() - t0)
        hb("D γ=%-4g %.2f·η_th (below_rho=%s) max‖W‖/‖W*‖=%9.3g t_div=%-6s μ_tail=%s (%.0fs)"
           % (g, mult, eta < RHO_GFIX[g], r["mx"] / nW, r["tdiv"],
              ("%.2e" % r["mu_tail"]) if r["mu_tail"] is not None else "None", time.time() - t0))
        flush("probe83_D2.json", DG)
    DG[str(g)] = rows


# ---------------- E′. ε 阶梯（正确 η） ----------------
EP = {}
for g in (10.0, 20.0, 30.0):
    xc = load_fp(g); nW = wn(xc)
    v0 = jac81.flat(xc); nv0 = float(np.linalg.norm(v0))
    rng = np.random.default_rng(7777)
    d = rng.normal(size=v0.size); d /= np.linalg.norm(d)
    eta = 1.10 * ETA_TH[g]
    rows = {}
    for e in (0.0, 1e-6, 1e-3, 3e-2, 1e-1, 3e-1, 1.0):
        v = v0 + e * nv0 * d
        mx, tdiv, bad = 0.0, None, False
        for t in range(1, N + 1):
            vn = jac79.step48(v, eta, "target", g, c=C)
            fin = bool(np.isfinite(vn).all())
            v = vn
            if not fin:
                bad = True; tdiv = t; break
            if t % 20 == 0:
                p = jac81.unflat(v)
                w1 = float(np.linalg.norm(p[0] @ p[1].T)); w2 = float(np.linalg.norm(p[2] @ p[3].T))
                if w1 > mx: mx = w1
                if (w1 > DIV_ABS or w2 > DIV_ABS) and tdiv is None: tdiv = t
        rows["%.0e" % e] = dict(eps=e, eta=eta, tdiv=tdiv, nan=bad, mx_rel=mx / nW,
                               below_rho=bool(eta < RHO_GFIX[g]))
        hb("E' γ=%-4g ε=%-8.0e η/η_th=1.10 max‖W‖/‖W*‖=%9.3g t_div=%-6s nan=%s"
           % (g, e, mx / nW, tdiv, bad))
        flush("probe83_E2.json", EP)
    EP[str(g)] = rows

print("PROBE83C DONE", flush=True)