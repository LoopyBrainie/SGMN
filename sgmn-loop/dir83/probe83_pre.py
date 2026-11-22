"""DIR-83 开卡前的代数/数值预言探针（`DEF-64-1` 义务）。

目的**不是**产出判据读数，而是回答"阈值该设在哪"以及"哪些候选机制由代数直接排除"。
全部只读沿用 `dir82/jac82.py`（`gfix_qr` 等）与 `ref70/step72/jac79`，不改装置。

六件事：
 A. 不动点与范数：x*（源规则 warm78.warm_fp）、‖W*‖、gfix 前后残差。
 B. 谱在 η 上的**斜率**：μ(η) = ln|λ|_max_nonneutral(η)，及 η_th 邻域的 c = dμ/dη。
 C. **缺口预测**：gap_pred = μ_floor / (c·ρ_λ)，其中 μ_floor = ln(1e3/‖W*‖)/N，N=200000。
    —— 若 gap_pred 复现实测 gap，则"缺口"是**探测协议的分辨率极限**，不是 basin/非正规性。
 D. 轨迹真相：在 η 阶梯上跑长轨迹，记录 max‖W‖/‖W*‖、末段增长率符号
    —— 区分"真指数发散"vs"过冲后有界"vs"慢收敛未完成"。
 E. ε 阶梯：x*+εδ，看逃逸是否需要有限幅度（区分非正规瞬态 vs 不动点不是极限）。
 F. Λ(η,T) = sup_{t≤T}‖J(η)^t‖，T∈{2000,5000,20000}（T 加倍收敛核查，`D-101` 边界 4 的欠账）。
"""
import sys, os, json, time, math
for q in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79", "dir80", "dir81", "dir82", "dir83"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np
import ref70, step72, jac79, jac81, jac82, warm78

OUT = "/home/ec2-user/sgmn-loop/dir83/out"
os.makedirs(OUT, exist_ok=True)
C, D, R, LAM = 6.0, 6, 2, ref70.LAM
N_BIS, SD, INIT = 200000, 20261118, 0.1
DIV_ABS = 1e3

# dir79 冻结读数（绝对阈值口径）
ETA_TH = {3.0: 0.064974679, 10.0: 0.0073034051, 20.0: 0.0015740607, 30.0: 0.00057804665}
RHO_RAW = {3.0: 0.06565421, 10.0: 0.0086535658, 20.0: 0.0023729778, 30.0: 0.0010888393}
GAM = [3.0, 10.0, 20.0, 30.0]


def flush(name, d):
    p = os.path.join(OUT, name)
    json.dump(d, open(p + ".tmp", "w"), ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


def hb(msg):
    print("    " + msg, flush=True)


# ---------------- A. 不动点 ----------------
def get_fp(g):
    """源规则（与 dir79/formal79 逐字一致）：η_warm = 0.1·η_th，warm78.warm_fp。npz 缓存。"""
    npz = os.path.join(OUT, "fp_target_g%g.npz" % g)
    if os.path.exists(npz):
        z = np.load(npz)
        return (z["U1"], z["V1"], z["U2"], z["V2"]), dict(cached=True, **{k: float(z[k]) for k in ("steps", "res")})
    t0 = time.time()
    x, info = warm78.warm_fp("target", g, 0.1 * ETA_TH[g])
    U1, V1, U2, V2 = x
    np.savez(npz, U1=U1, V1=V1, U2=U2, V2=V2, steps=info["steps"], res=info["res"])
    return x, dict(cached=False, steps=info["steps"], res=info["res"], secs=time.time() - t0)


def wnorm(x):
    U1, V1, U2, V2 = x
    return float(np.linalg.norm(U1 @ V1.T)), float(np.linalg.norm(U2 @ V2.T))


def resid(x, eta, g):
    v = jac81.flat(x)
    return float(np.linalg.norm(jac79.step48(v, eta, "target", g, c=C) - v)) / float(np.linalg.norm(v))


RES = {}
for g in GAM:
    x, info = get_fp(g)
    nW1, nW2 = wnorm(x)
    p = ref70.pred_target(C, g)
    xc = jac82.gfix_qr(x)
    r_raw = resid(x, 0.1 * ETA_TH[g], g)
    r_fix = resid(xc, 0.1 * ETA_TH[g], g)
    RES[str(g)] = dict(fp=info, normW1=nW1, normW2=nW2, u1_pred=p["u1"], u2_pred=p["u2"],
                       relW1=abs(nW1 - p["u1"]) / p["u1"],
                       resid_raw=r_raw, resid_gfix=r_fix,
                       orth_err_V1=jac82.orth_err(xc[1]), orth_err_V2=jac82.orth_err(xc[3]))
    hb("A γ=%-4g ‖W1‖=%.6f (pred %.6f rel %.1e) ‖W2‖=%.6f  resid raw=%.2e gfix=%.2e  steps=%s"
       % (g, nW1, p["u1"], RES[str(g)]["relW1"], nW2, r_raw, r_fix, info.get("steps", "cached")))
    flush("probe83_A.json", RES)


# ---------------- B. 谱在 η 上的斜率 ----------------
def spec(x, eta, g):
    """返回非中性谱半径与全谱半径。不用 eig 的特征向量基（DEF-81-6），只用模。"""
    J = jac79.jac(x, eta, "target", g, c=C)
    w = np.linalg.eigvals(J)
    a = np.abs(w)
    nn = a[np.abs(w - 1.0) > 1e-6]
    return float(np.max(nn)) if nn.size else float("nan"), float(np.max(a)), J


SPEC = {}
for g in GAM:
    x, _ = get_fp(g)
    xc = jac82.gfix_qr(x)
    rows = {}
    for frac in (0.10, 0.30, 0.50, 0.70, 0.85, 0.95, 1.00, 1.05):
        eta = frac * RHO_RAW[g]
        r_nn, r_all, _ = spec(xc, eta, g)
        rows["%.2f" % frac] = dict(eta=eta, rho_nn=r_nn, rho_all=r_all,
                                   mu=float(np.log(r_nn)) if r_nn == r_nn and r_nn > 0 else float("nan"))
    # η_th 处的读数
    r_nn_th, r_all_th, _ = spec(xc, ETA_TH[g], g)
    rows["eta_th"] = dict(eta=ETA_TH[g], rho_nn=r_nn_th, rho_all=r_all_th,
                          mu=float(np.log(r_nn_th)) if r_nn_th > 0 else float("nan"))
    SPEC[str(g)] = rows
    hb("B γ=%-4g μ(η_th)=%.3e (|λ|=%.8f)  μ(0.5ρ)=%.3e  μ(0.95ρ)=%.3e"
       % (g, rows["eta_th"]["mu"], r_nn_th, rows["0.50"]["mu"], rows["0.95"]["mu"]))
    flush("probe83_B.json", SPEC)


# ---------------- C. 缺口预测 ----------------
PRED = {}
for g in GAM:
    x, _ = get_fp(g)
    xc = jac82.gfix_qr(x)
    nW1, _ = wnorm(xc)
    mu_floor = math.log(DIV_ABS / nW1) / N_BIS
    # 用 0.85ρ 与 1.00ρ 之间的割线斜率作 c
    r85 = SPEC[str(g)]["0.85"]; r100 = SPEC[str(g)]["1.00"]
    c = (r100["mu"] - r85["mu"]) / (r100["eta"] - r85["eta"])
    gap_pred = mu_floor / (c * RHO_RAW[g]) if c > 0 else float("nan")
    gap_obs = RHO_RAW[g] / ETA_TH[g] - 1.0
    PRED[str(g)] = dict(normW1=nW1, mu_floor=mu_floor, slope_c=float(c),
                        gap_pred=float(gap_pred), gap_obs=float(gap_obs),
                        sqrt_mu=float(math.sqrt(mu_floor / abs(c))) if c != 0 else None)
    hb("C γ=%-4g μ_floor=%.3e c=%.4e ⇒ gap_pred=%.4f vs gap_obs=%.4f"
       % (g, mu_floor, c, gap_pred, gap_obs))
    flush("probe83_C.json", PRED)


# ---------------- D. 轨迹真相 ----------------
def run_diag(g, eta, n=400_000, init=INIT, seed=SD, nsamp=200):
    """跑 run_stack 等价的长轨迹，记录 ‖W‖ 的轨迹概要与其相对 ‖W*‖ 的规模。"""
    ks, tg, Cm, Bm, cb = step72._consts(C)
    rng = np.random.default_rng(seed)
    U1 = rng.normal(size=(D, R)) * init; V1 = rng.normal(size=(D, R)) * init
    U2 = rng.normal(size=(D, R)) * init; V2 = rng.normal(size=(D, R)) * init
    every = max(1, n // nsamp)
    ns, tdiv, mx = [], None, 0.0
    for t in range(1, n + 1):
        U1, V1, U2, V2 = step72.step(U1, V1, U2, V2, eta, "target", g, C)
        if not np.isfinite(U1).all():
            tdiv = t; break
        w1 = float(np.linalg.norm(U1 @ V1.T)); w2 = float(np.linalg.norm(U2 @ V2.T))
        if w1 > mx: mx = w1
        # 与 ref70._div 同口径：两层任一超阈即判发散
        if (w1 > DIV_ABS or w2 > DIV_ABS) and tdiv is None: tdiv = t
        if t % every == 0:
            ns.append((t, w))
    W1 = U1 @ V1.T
    # 末段增长率：对最后 20% 采样点拟合 ln‖W‖
    tail = [(t, w) for (t, w) in ns if t >= 0.8 * n]
    mu_t = float("nan")
    if len(tail) >= 3 and all(w > 0 for _, w in tail):
        tt = np.array([t for t, _ in tail], float); yy = np.log(np.array([w for _, w in tail], float))
        mu_t = float(np.polyfit(tt, yy, 1)[0])
        mu_t = None if not np.isfinite(mu_t) else mu_t
    return dict(eta=eta, tdiv=tdiv, mx=mx, mu_tail=mu_t, finite=bool(np.isfinite(W1).all()),
                w_final=float(np.linalg.norm(W1)), trace=ns)


TRAJ = {}
for g in (3.0, 30.0):
    x, _ = get_fp(g)
    xc = jac82.gfix_qr(x)
    nW1, _ = wnorm(xc)
    rows = {}
    for mult in (0.8, 1.0, 1.25, 1.5, 2.0):
        eta = mult * ETA_TH[g]
        t0 = time.time()
        r = run_diag(g, eta)
        rows["x%.2f" % mult] = dict(r, normW1=nW1, mx_rel=r["mx"] / nW1, secs=time.time() - t0)
        rows["x%.2f" % mult].pop("trace", None)
        hb("D γ=%-4g η=%.3g·η_th  max‖W‖/‖W*‖=%.2f  t_div=%s  μ_tail=%s  (%.0fs)"
           % (g, mult, r["mx"] / nW1, r["tdiv"], r["mu_tail"], time.time() - t0))
        flush("probe83_D.json", TRAJ)
    TRAJ[str(g)] = rows


# ---------------- E. ε 阶梯 ----------------
EPS = {}
for g in (3.0, 30.0):
    x, _ = get_fp(g)
    xc = jac82.gfix_qr(x)
    nW1, _ = wnorm(xc)
    v0 = jac81.flat(xc)
    rng = np.random.default_rng(7777)
    d = rng.normal(size=v0.size); d /= np.linalg.norm(d)
    eta = 1.25 * ETA_TH[g]
    rows = {}
    for e in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0):
        v = v0 + e * np.linalg.norm(v0) * d
        mx, tdiv = 0.0, None
        for t in range(1, 400_001):
            vn = jac79.step48(v, eta, "target", g, c=C)
            if not np.isfinite(vn).all():
                tdiv = t; v = vn; break
            v = vn
            if t % 20 == 0:
                p = jac81.unflat(v)
                w1 = float(np.linalg.norm(p[0] @ p[1].T)); w2 = float(np.linalg.norm(p[2] @ p[3].T))
                if w1 > mx: mx = w1
                if (w1 > DIV_ABS or w2 > DIV_ABS) and tdiv is None: tdiv = t
        rows["%.0e" % e] = dict(eps=e, tdiv=tdiv, mx_rel=mx / nW1,
                                finite=bool(np.isfinite(v).all()))
        hb("E γ=%-4g ε=%.0e  max‖W‖/‖W*‖=%.3f  t_div=%s" % (g, e, mx / nW1, tdiv))
        flush("probe83_E.json", EPS)
    EPS[str(g)] = rows


# ---------------- F. Λ(η,T) 与 T 加倍 ----------------
def supjt_T(J, T):
    M = np.eye(J.shape[0]); best, bt = 1.0, 0
    marks = {}
    for t in range(1, T + 1):
        M = J @ M
        v = float(np.linalg.norm(M, 2))
        if v > best: best, bt = v, t
        if t in (2000, 5000, 20000): marks[str(t)] = v
    return best, bt, marks

LAMV = {}
for g in GAM:
    x, _ = get_fp(g)
    xc = jac82.gfix_qr(x)
    rows = {}
    for tag, eta in (("eta_th", ETA_TH[g]), ("1.25th", 1.25 * ETA_TH[g]),
                     ("0.5rho", 0.5 * RHO_RAW[g]), ("0.95rho", 0.95 * RHO_RAW[g])):
        J = jac79.jac(xc, eta, "target", g, c=C)
        sup, bt, marks = supjt_T(J, 20000)
        rows[tag] = dict(eta=eta, sup=sup, peak_t=bt, marks=marks)
        hb("F γ=%-4g %-8s Λ=%.4f peak_t=%d  Λ(2k)=%.4f Λ(5k)=%.4f" %
           (g, tag, sup, bt, marks.get("2000", float("nan")), marks.get("5000", float("nan"))))
        flush("probe83_F.json", LAMV)
    LAMV[str(g)] = rows

print("PROBE83 DONE", flush=True)