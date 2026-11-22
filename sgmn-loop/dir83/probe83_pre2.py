"""DIR-83 开卡前预言探针 · 第 2 部分（A/B 已完成，不动点已缓存）。

只做三件：
 D. 轨迹真相：η/η_th ∈ {1.0, 1.25, 1.5, 2.0} 上跑满 N=2e5，记录
    max‖W‖/‖W*‖（过冲）、末段增长率 μ_tail、是否 NaN/超阈。── 判"真发散"还是"过冲后有界"。
 E. ε 阶梯：x* + εδ，η=1.25η_th。── 判逃逸是否需要**有限幅度**（非正规/有限幅度 vs 线性）。
 F. Λ(η,T) = sup_{t≤T}‖J^t‖，T∈{2000,5000,20000}，在 gfix 规范代表上。
    + 正确的谱斜率 c（用 ρ_gfix 作端点，不再用 ρ_raw）。
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
SD, INIT = 20261118, 0.1

ETA_TH = {3.0: 0.064974679, 10.0: 0.0073034051, 20.0: 0.0015740607, 30.0: 0.00057804665}
RHO_GFIX = {3.0: 0.068867243, 10.0: 0.0091167253, 20.0: 0.0025007558, 30.0: 0.001147340655}
GAM = [3.0, 10.0, 20.0, 30.0]


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


# ---------------- D. 轨迹真相 ----------------
DIVOFF = {}
for g in GAM:
    xc = load_fp(g); nW = wn(xc)
    rows = {}
    for mult in (1.0, 1.25, 1.5, 2.0):
        eta = mult * ETA_TH[g]
        ks, tg, Cm, Bm, cb = step72._consts(C)
        rng = np.random.default_rng(SD)
        U1 = rng.normal(size=(D, R)) * INIT; V1 = rng.normal(size=(D, R)) * INIT
        U2 = rng.normal(size=(D, R)) * INIT; V2 = rng.normal(size=(D, R)) * INIT
        every = max(1, N // 200)
        ns, tdiv, mx, bad = [], None, 0.0, False
        t0 = time.time()
        for t in range(1, N + 1):
            U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, eta, "target", g, C)
            if not np.isfinite(U1).all():
                bad = True; tdiv = t; break
            w1 = float(np.linalg.norm(U1 @ V1.T)); w2 = float(np.linalg.norm(U2 @ V2.T))
            if w1 > mx: mx = w1
            if (w1 > DIV_ABS or w2 > DIV_ABS) and tdiv is None: tdiv = t
            if t % every == 0:
                ns.append((t, w1))
        tail = [(t, w) for (t, w) in ns if t >= 0.8 * N]
        mu_t = None
        if len(tail) >= 3 and all(w > 0 for _, w in tail):
            tt = np.array([t for t, _ in tail], float)
            yy = np.log(np.array([w for _, w in tail], float))
            v = float(np.polyfit(tt, yy, 1)[0])
            mu_t = v if np.isfinite(v) else None
        rows["x%.2f" % mult] = dict(eta=eta, mult=mult, tdiv=tdiv, nan=bad,
                                    mx=mx, mx_rel=mx / nW, mu_tail=mu_t,
                                    w_final=(float(np.linalg.norm(U1 @ V1.T)) if not bad else None),
                                    normW= nW, secs=time.time() - t0)
        hb("D γ=%-4g %.2f·η_th  max‖W‖/‖W*‖=%8.2f  t_div=%-7s μ_tail=%-10s nan=%s (%.0fs)"
           % (g, mult, mx / nW, tdiv, ("%.3e" % mu_t) if mu_t is not None else "None", bad,
              time.time() - t0))
        flush("probe83_D.json", DIVOFF)
    DIVOFF[str(g)] = rows


# ---------------- E. ε 阶梯 ----------------
EPS = {}
for g in GAM:
    xc = load_fp(g); nW = wn(xc)
    v0 = jac81.flat(xc)
    rng = np.random.default_rng(7777)
    d = rng.normal(size=v0.size); d /= np.linalg.norm(d)
    eta = 1.25 * ETA_TH[g]
    rows = {}
    for e in (0.0, 1e-8, 1e-4, 1e-2, 1e0):
        v = v0 + e * np.linalg.norm(v0) * d
        mx, tdiv, bad = 0.0, None, False
        for t in range(1, N + 1):
            vn = jac79.step48(v, eta, "target", g, c=C)
            if not np.isfinite(vn).all():
                bad = True; tdiv = t; v = vn; break
            v = vn
            if t % 20 == 0:
                p = jac81.unflat(v)
                w1 = float(np.linalg.norm(p[0] @ p[1].T)); w2 = float(np.linalg.norm(p[2] @ p[3].T))
                if w1 > mx: mx = w1
                if (w1 > DIV_ABS or w2 > DIV_ABS) and tdiv is None: tdiv = t
        rows["%.0e" % e] = dict(eps=e, eta=eta, tdiv=tdiv, nan=bad, mx_rel=mx / nW)
        hb("E γ=%-4g ε=%-6.0e  max‖W‖/‖W*‖=%8.3f  t_div=%-7s nan=%s" % (g, e, mx / nW, tdiv, bad))
        flush("probe83_E.json", EPS)
    EPS[str(g)] = rows


# ---------------- F. Λ(η,T) ----------------
def supT(J, T):
    M = np.eye(J.shape[0]); best, bt, mk = 1.0, 0, {}
    for t in range(1, T + 1):
        M = J @ M
        s = float(np.linalg.norm(M, 2))
        if s > best: best, bt = s, t
        if t in (2000, 5000): mk[str(t)] = s
    mk["20000"] = s
    return best, bt, mk


LAMV = {}
for g in GAM:
    xc = load_fp(g)
    rows = {}
    for tag, eta in (("eta_th", ETA_TH[g]), ("1.25th", 1.25 * ETA_TH[g]),
                     ("1.50th", 1.50 * ETA_TH[g]), ("rho_gfix", RHO_GFIX[g])):
        J = jac79.jac(xc, eta, "target", g, c=C)
        sup, bt, mk = supT(J, 20000)
        rows[tag] = dict(eta=eta, ratio_eta=eta / ETA_TH[g], sup=sup, peak_t=bt, marks=mk)
        hb("F γ=%-4g %-9s Λ=%.4f peak_t=%-6d Λ(2k)=%.3f Λ(5k)=%.3f"
           % (g, tag, sup, bt, mk["2000"], mk["5000"]))
        flush("probe83_F.json", LAMV)
    LAMV[str(g)] = rows


# ---------------- C′. 正确的谱斜率（用 ρ_gfix 作右端点） ----------------
CC = {}
for g in GAM:
    xc = load_fp(g); nW = wn(xc)
    mu_floor = math.log(DIV_ABS / nW) / N
    def mu(eta):
        J = jac79.jac(xc, eta, "target", g, c=C)
        w = np.linalg.eigvals(J); a = np.abs(w)
        nn = a[np.abs(w - 1.0) > 1e-6]
        return float(np.log(np.max(nn))) if nn.size else float("nan")
    m95, m100 = mu(0.95 * RHO_GFIX[g]), mu(RHO_GFIX[g])
    c = (m100 - m95) / (0.05 * RHO_GFIX[g])
    gap_obs = RHO_GFIX[g] / ETA_TH[g] - 1.0
    CC[str(g)] = dict(normW=nW, mu_floor=mu_floor, mu95=m95, mu100=m100, slope_c=float(c),
                      gap_obs=float(gap_obs),
                      c_needed=float(mu_floor / (ETA_TH[g] * gap_obs)) if gap_obs > 0 else None)
    hb("C' γ=%-4g μ(0.95ρ)=%.3e μ(ρ)=%.3e c=%.4e | μ_floor=%.3e gap_obs=%.4f c_needed=%.4e"
       % (g, m95, m100, c, mu_floor, gap_obs, CC[str(g)]["c_needed"] or float("nan")))
    flush("probe83_C2.json", CC)

print("PROBE83B DONE", flush=True)