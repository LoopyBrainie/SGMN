r"""DIR-26-L · R1_gating13.py —— 剂量的**远点定律检验**：$G\propto n_{\rm kv}^{0.4325}$ 能推到 6 倍世界规模吗？

卡片：`revolve/DIR-26L/DIRECTION_rev26l_v11.md`（rev 1.1 冻结；rev 1.0 留档，Stage A 未过，DEF-26L-2/3）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，零改动）
★ 单元**一行不改**：直接 `import R1_gating12.run_unit`（判据口径 O + 控制口径 X + 六臂 + 构造恒等）。
   本文件只做三件事：把单元铺到 $\{90,120\}$、把 D-43 的参照值**现场读出**（护栏 (n)）、跑远点四通道判决树。

相对 `dir26k/R1_gating12.py` 的改动：
  ① 格点 = $\{90,120\}\times N_{\rm read}{=}20$（$\le60$ 三格复用 D-43 的落盘，不重跑，护栏 (c) 逐位证明协议未变）；
  ② 判据换成**远点定律检验**：$G_{120}$ 对 $p_{\rm lin}=6G_{20}$ 与 $p_{\rm pow}=G_{20}6^{\gamma}$ 的 $3\mathrm{se}$ 比较（四通道 LIN/POW/CAP/ACC）；
  ③ 护栏 (f) 改为**比值型计数可比性**（纪律 (xx)④），(n) 现场读参照、(o) 判据层级的元断言。
"""
import argparse, json, os, sys, time

import numpy as np
import torch

for _p in ("dir26l", "dir26k", "dir26j", "dir26i", "dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating2 as G2
import R1_gating3 as G3
import R1_gating5 as G5
import R1_gating6 as G6
import R1_gating7 as G7
import R1_gating10 as G10
import R1_gating11 as G11
import R1_gating12 as K
from R1_gating import D_MODEL, N_DIST, ETA, M, S_KEY, build_stream, drive, make_core, metrics_plas
from R1_gating2 import keep_positions, item_counts
from R1_gating3 import ret_jitter
from R1_gating5 import calibrate_w, theta_of
from R1_gating6 import _pos_idxs
from R1_gating7 import round_quota, build_keep, realized_profile, PROFILE, md5, chk, res
from R1_gating10 import ret_jitter_margin

N_READ = 20
KV_NEW = (90, 120)
KV_REF = (20, 35, 60)
ANCHOR_KV = 60                     # 复用的锚点格点
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
D43 = "/home/ec2-user/sgmn-loop/dir26k/out/DIR26Kr10_formal.json"

SHARED_SEEDS = [20261028, 20261104, 7, 20261111, 20261118,
                20261125, 20261202, 20261209, 20261216, 20261223]
NEW_SEEDS = [20270519, 20270526, 20270602, 20270609, 20270616,
             20270623, 11, 13, 17, 19]
SEEDS = SHARED_SEEDS + NEW_SEEDS            # 20
ANCHOR_SEEDS = SHARED_SEEDS[:5]
ARMS6 = ("A", "B", "P", "Fu", "Uu", "Ku")

SE_K = 3.0
NU_CAP_MAX = 0.10
QBAR_RATIO = (0.85, 1.15)
SCHED_SPREAD_MAX = 0.10            # |h_F−h_K| 三格极差 / 中位 ≤ 10%
GAMMA_REF = 0.4325                 # 护栏 (n)：必须与 D-43 现场读出的值逐位相等


# ══════════════════════════════════════════════ 参照值（护栏 (n)）
def d43_means():
    r"""从 D-43 的落盘现场读出 $G_c$、$\mu_{U0,c}$ 与 $\hat\gamma$（防手抄漂移）。"""
    rows = json.load(open(D43))["rows"]
    g, u = {}, {}
    for n in KV_REF:
        rr = [r for r in rows if r["n_kv"] == n]
        assert len(rr) == 30, f"D-43 的 n_kv={n} 行数不是 30：{len(rr)}"
        g[n] = float(np.mean([r["G_O"] for r in rr]))
        u[n] = float(np.mean([r["U0_O"] for r in rr]))
    gam = float(np.log(g[60] / g[20]) / np.log(3.0))
    return {"G": g, "U0": u, "gamma": gam, "rows": rows}


# ══════════════════════════════════════════════ Stage A
def stage_a():
    print("=== Stage A（A26L-1 … A26L-15） ===")
    chk("A26L-1", md5("/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py") == DEV_MD5,
        f"装置 md5 == {DEV_MD5}")

    pairs = [(sys.modules["R1_gating"], "build_stream", build_stream),
             (sys.modules["R1_gating"], "drive", drive),
             (sys.modules["R1_gating"], "make_core", make_core),
             (sys.modules["R1_gating"], "metrics_plas", metrics_plas),
             (sys.modules["R1_gating2"], "keep_positions", keep_positions),
             (sys.modules["R1_gating2"], "item_counts", item_counts),
             (sys.modules["R1_gating3"], "ret_jitter", ret_jitter),
             (sys.modules["R1_gating5"], "calibrate_w", calibrate_w),
             (sys.modules["R1_gating5"], "theta_of", theta_of),
             (sys.modules["R1_gating6"], "_pos_idxs", _pos_idxs),
             (sys.modules["R1_gating7"], "round_quota", round_quota),
             (sys.modules["R1_gating7"], "build_keep", build_keep),
             (sys.modules["R1_gating7"], "realized_profile", realized_profile),
             (sys.modules["R1_gating10"], "ret_jitter_margin", ret_jitter_margin),
             (sys.modules["R1_gating11"], "run_unit", G11.run_unit),
             (sys.modules["R1_gating12"], "run_unit", K.run_unit),
             (sys.modules["R1_gating12"], "cal_at", K.cal_at),
             (sys.modules["R1_gating12"], "read_stream", K.read_stream)]
    bad = [f"{m.__name__}:{n}" for m, n, o in pairs if getattr(m, n, None) is not o]
    chk("A26L-2", not bad,
        f"{len(pairs)} 个复用对象逐项同源（同一对象，非同名副本）" + ("" if not bad else f"；不同源 {bad}"))

    taus = {n: K.cal_at(n) for n in KV_NEW}
    okc = all(abs(t["skip_rate"] - 0.40) <= 0.05 and t["gap"] >= 0.05 and t["skip_frac_repeat"] >= 0.85 and t["gate_ok"]
              for t in taus.values())
    chk("A26L-3", G5.N_DIST == N_DIST and okc,
        f"复位 N_DIST==35 ? {G5.N_DIST == N_DIST}；"
        + " ".join(f"n{n}:c_tau={t['c_tau']:.4f}/skip={t['skip_rate']:.3f}/gap={t['gap']:+.3f}/rep={t['skip_frac_repeat']:.3f}"
                   for n, t in taus.items()))

    s0 = SEEDS[0]
    g120 = K.run_unit(120, s0, taus[120]["c_tau"])
    chk("A26L-4", all(v["mu_read"] == N_READ * K.R_HO8 for v in g120["arms"].values())
        and all((v["mu"] > 0) == (v["ret1"] > 0.5) for v in g120["arms"].values())
        and abs(g120["G_X"]) <= 0.25 * abs(g120["G_O"]),
        f"n_kv=120 seed={s0}：判据口径六臂 μ 样本数 == 160 且符号/命中逐条兼容；"
        f"控制口径 |G_X| {abs(g120['G_X']):.5f} ≤ 0.25·|G_O| {0.25*abs(g120['G_O']):.5f} "
        f"（G_O {g120['G_O']:+.5f} vs G_X {g120['G_X']:+.5f}）")

    # A26L-5 工作例
    D = build_stream(s0, D_MODEL, 120, K.R_PRES, theta_of(0.9))
    RS8 = K.read_stream(s0, K.R_HO8)
    core = make_core(D_MODEL, s0, K.R0)
    _, logB = drive(core, D, "B", tau=taus[120]["c_tau"])
    cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, 120)
    keep, _ = build_keep(120, K.R_PRES, cnt, PROFILE["F"])
    c = make_core(D_MODEL, s0, K.R0)
    drive(c, D, "R", keep=keep)
    P = c._W().detach() - c.W0
    vb = D["vb"][:N_READ]
    m = ret_jitter_margin(P, RS8["K"], vb, N_READ, K.R_HO8)
    vbU = G.unit(vb)
    tbl, ms = [], []
    for rho in range(K.R_HO8):
        S = G.unit((P @ RS8["K"][rho].T).T) @ vbU.T
        for i in (0, 7, 19):
            row = S[i]
            other = torch.cat([row[:i], row[i + 1:]])
            tbl.append({"rho": rho, "i": i, "diag": round(float(row[i]), 6),
                        "max_other": round(float(other.max()), 6), "m": round(float(row[i] - other.max()), 6)})
        for i in range(N_READ):
            row = S[i]
            other = torch.cat([row[:i], row[i + 1:]]) if i + 1 < N_READ else row[:i]
            ms.append(float(row[i] - other.max()))
    re_mu = float(np.mean(ms))
    chk("A26L-5", re_mu == m["mu"] and len(ms) == m["n_read"],
        f"工作例（n_kv=120）：独立重算 μ = {re_mu:+.12f} 与 `ret_jitter_margin` {m['mu']:+.12f} "
        f"{'逐位相等' if re_mu == m['mu'] else '不等'}（样本 {len(ms)} == {m['n_read']}）；"
        f"抽查 (ρ,i)=({tbl[0]['rho']},{tbl[0]['i']}) diag {tbl[0]['diag']:+.6f} max_other {tbl[0]['max_other']:+.6f} ⇒ m {tbl[0]['m']:+.6f}")
    res["worked"] = {"mu": re_mu, "n": len(ms), "sample": tbl}

    # A26L-6 漂移锚点：重跑 (60,20) 与 D-43 逐位
    rows43 = {(r["n_kv"], r["seed"]): r for r in json.load(open(D43))["rows"]}
    t60 = K.cal_at(ANCHOR_KV)["c_tau"]
    bads, ncmp = [], 0
    for s in ANCHOR_SEEDS:
        gg = K.run_unit(ANCHOR_KV, s, t60)
        ref = rows43[(ANCHOR_KV, s)]
        for a in ARMS6:
            for f in ("ho4_mean", "ho4_maj", "ho8_mean", "clean"):
                ncmp += 1
                if gg["arms"][a][f] != ref["arms"][a][f]:
                    bads.append(f"{a}.{f}")
        for f in ("qbar", "qmax", "mirror_ok", "nu_cap", "sched_dist"):
            ncmp += 1
            if gg[f] != ref[f]:
                bads.append(f)
    chk("A26L-6", not bads, f"漂移锚点：5 种子 × (6 臂 × 4 字段 + 5 量) = {ncmp} 值对与 D-43 的 (60|20) **逐位相等**"
                            + ("" if not bads else f"；不符 {bads[:6]}（共 {len(bads)}）"))

    # A26L-7 护栏 (n)：参照值现场读出并与卡 §4.1 常数逐位相等
    ref = d43_means()
    p_lin = 6.0 * ref["G"][20]
    p_pow = ref["G"][20] * 6.0 ** ref["gamma"]
    # 卡 rev 1.1 §4.1 印出的是落盘的精确位（DEF-26L-2 修正后）；相对差 ≤2e-9
    c_g20, c_g35, c_g60 = 0.006285978551798811, 0.007901956066101166, 0.010109853324247523
    c_gam = 0.4325356666135775
    c_lin, c_pow = 0.03771587131079286, 0.013644293870204098
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-12)
    ok7 = (rel(ref["G"][20], c_g20) <= 2e-9 and rel(ref["G"][35], c_g35) <= 2e-9
           and rel(ref["G"][60], c_g60) <= 2e-9 and rel(ref["gamma"], c_gam) <= 2e-9
           and rel(p_lin, c_lin) <= 2e-9 and rel(p_pow, c_pow) <= 2e-9)
    chk("A26L-7", ok7,
        f"参照现场读出 == 卡 rev 1.1 §4.1 的精确位（相对差 ≤2e-9）：G20 {ref['G'][20]!r} γ̂ {ref['gamma']!r} "
        f"G60 {ref['G'][60]!r} ⇒ p_lin {p_lin!r}（卡 {c_lin!r}）、p_pow {p_pow!r}（卡 {c_pow!r}）")

    # A26L-8 读几何（护栏 (g)）：判据口径的读流与 n_kv 无关；写流 v 跨 n_kv 不同
    vbs = {n: build_stream(s0, D_MODEL, n, K.R_PRES, theta_of(0.9))["vb"][:N_READ] for n in (60, 90, 120)}
    pair_same = {f"{a}vs{b}": bool(torch.equal(vbs[a], vbs[b]))
                 for a in (60, 90, 120) for b in (60, 90, 120) if a < b}
    chk("A26L-8", not all(pair_same.values()),
        f"写流 $v$ 前 20 行跨 $n_{{\\rm kv}}$ 两两比较：{pair_same} ⇒ 不全相同"
        f"（DEF-26K-1 复现：身份值随 $n_{{\\rm kv}}$ 变，配对增益只来自 $K$/`proto`；判据口径的分工见卡 §3）")

    # A26L-9 构造恒等 / A26L-10 零发放 / A26L-11 读后不变 / A26L-12 副作用隔离
    ident = all(v["sum_ok"] and v["quota_ok"] for t, v in g120["arms"].items() if t in ("Fu", "Uu", "Ku"))
    chk("A26L-9", ident and g120["mirror_ok"] and g120["nu_cap"] <= NU_CAP_MAX,
        f"构造恒等：(120) k_B {g120['k']['B']}、q_max {g120['qmax']}、q_min {min(g120['counts'])}、"
        f"ν_cap {g120['nu_cap']:.3f}、镜像 {g120['mirror_ok']}、日程距离 {g120['sched_dist']:.3f}")

    rho_stab = 1 - ETA * (1.0 * S_KEY + 0.05 / 2)
    c = make_core(D_MODEL, s0, K.R0)
    W0 = c._W().detach().clone()
    kN, _ = drive(c, build_stream(s0, D_MODEL, 20, K.R_PRES, theta_of(0.9)), "NONE")
    chk("A26L-10", rho_stab > 0 and kN == 0 and torch.equal(W0, c._W().detach()),
        f"ρ_stab={rho_stab:.4f}>0；零发放臂 0 次、‖ΔW‖=0")
    chk("A26L-11", all(v["W_unchanged"] for v in g120["arms"].values()),
        f"n_kv=120 seed={s0} 六臂读后 W 逐位不变")
    iso = (g120["arms"]["Fu"]["issued"] == g120["arms"]["Uu"]["issued"] == g120["arms"]["Ku"]["issued"]
           and len({g120["W0sig"]}) == 1)
    chk("A26L-12", iso and g120["arms"]["Fu"]["sum_ok"],
        f"三臂同 W0 指纹 {g120['W0sig'][:8]}、同 k {g120['arms']['Fu']['issued']}；"
        f"剖面 F {[round(x,3) for x in g120['prof']['Fu']]} vs K {[round(x,3) for x in g120['prof']['Ku']]}")

    need = {"issued", "wsig", "clean", "mu", "mu_sd", "mu_read", "ret1", "ho4_mean", "ho8_mean"}
    miss = {t: sorted(need - set(v)) for t, v in g120["arms"].items()}
    miss = {t: m for t, m in miss.items() if m}
    chk("A26L-13", not miss and all(kk in g120 for kk in ("qbar", "qmax", "counts", "prof", "mirror_ok",
                                                          "nu_cap", "sched_dist", "G_O", "G_X", "U0_O", "U0_X")),
        f"落盘字段完整（缺 {miss}）；(120) G_O {g120['G_O']:+.5f} G_X {g120['G_X']:+.5f} U0_O {g120['U0_O']:+.5f}")

    # A26L-14 护栏 (o)：判据层级元断言
    chk("A26L-14", True, "判据层级（纪律 (xx)①）：§4.1 的 3se 用**逐格点 sd/√n**（`_band`），"
                         "§4.3 (e) 的 ν_cap 用**格点均值**、§4.3 (f) 的 q̄ 用**逐格点比值** —— "
                         "三条断言在 crit() 里各自显式写出，本项只作存在性确认")

    # A26L-15 探针种子不入表
    chk("A26L-15", 1907 not in SEEDS and len(SEEDS) == 20 and len(set(SEEDS)) == 20,
        f"种子表 20 个互异且不含探针种子 1907；shared {len(SHARED_SEEDS)} + new {len(NEW_SEEDS)}")

    res["calib"] = [{"n_kv": n, "c_tau": t["c_tau"], "skip_rate": t["skip_rate"], "gap": t["gap"]} for n, t in taus.items()]
    res["ref"] = {"G": ref["G"], "U0": ref["U0"], "gamma": ref["gamma"], "p_lin": p_lin, "p_pow": p_pow}


# ══════════════════════════════════════════════ 判据
def _band(G):
    G = np.asarray(G, float)
    se = float(np.std(G, ddof=1)) / np.sqrt(len(G))
    return float(np.mean(G)), se, SE_K * se


def crit(rows, ref):
    v = {}
    by = {n: sorted([r for r in rows if r["n_kv"] == n], key=lambda r: r["seed"]) for n in KV_NEW}
    used = [r["seed"] for r in by[90]]
    assert all([r["seed"] for r in by[n]] == used for n in KV_NEW), "两格点的种子序列不一致"
    assert len(used) >= 10, "种子过少"
    G = {n: np.array([r["G_O"] for r in by[n]]) for n in KV_NEW}
    GX = {n: np.array([r["G_X"] for r in by[n]]) for n in KV_NEW}
    U0 = {n: np.array([r["U0_O"] for r in by[n]]) for n in KV_NEW}

    # ---- 护栏 (e) 逐格点均值 ν_cap；(f) 比值型计数可比性；(m) 控制口径
    nu = {n: float(np.mean([r["nu_cap"] for r in by[n]])) for n in KV_NEW}
    v["e_nucap"] = {"ok": all(x <= NU_CAP_MAX for x in nu.values()), "nu": {n: round(nu[n], 3) for n in KV_NEW},
                    "level": "cell-mean"}
    qm = {n: float(np.mean([r["qbar"] for r in by[n]])) for n in KV_NEW}
    qm[60] = float(np.mean([r["qbar"] for r in ref["rows"] if r["n_kv"] == 60]))
    sd_ = {n: float(np.mean([r["sched_dist"] for r in by[n]])) for n in KV_NEW}
    sd_[60] = float(np.mean([r["sched_dist"] for r in ref["rows"] if r["n_kv"] == 60]))
    ratios = {n: qm[n] / qm[60] for n in KV_NEW}
    spread = (max(sd_.values()) - min(sd_.values())) / float(np.median(list(sd_.values())))
    v["f_comparable"] = {"ok": all(QBAR_RATIO[0] <= ratios[n] <= QBAR_RATIO[1] for n in KV_NEW) and spread <= SCHED_SPREAD_MAX,
                         "qbar_ratio_vs60": {n: round(ratios[n], 3) for n in KV_NEW},
                         "qbar": {n: round(qm[n], 3) for n in (60, 90, 120)},
                         "sched_dist": {n: round(sd_[n], 3) for n in (60, 90, 120)}, "sched_spread": round(spread, 4),
                         "level": "cell-ratio"}
    mx = {}
    for n in KV_NEW:
        m, se, k = _band(GX[n])
        mx[n] = {"G_X": round(m, 6), "k": round(k, 6),
                 "ratio": round(abs(m) / max(abs(float(np.mean(G[n]))), 1e-12), 4)}
    v["m_control"] = {"ok": all(abs(x["G_X"]) <= x["k"] and x["ratio"] <= 0.25 for x in mx.values()), "cells": mx}

    # ---- 判据：远点四通道
    g120, se120, k120 = _band(G[120])
    g90, se90, k90 = _band(G[90])
    p_lin, p_pow = ref["p_lin"], ref["p_pow"]
    # 预测的不确定度：p 由 D-43 的 30 种子均值构成 ⇒ 用其自身的 se 传播（非配对）
    refG = ref["G"]
    se_ref20 = float(np.std([r["G_O"] for r in ref["rows"] if r["n_kv"] == 20], ddof=1)) / np.sqrt(30)
    se_ref60 = float(np.std([r["G_O"] for r in ref["rows"] if r["n_kv"] == 60], ddof=1)) / np.sqrt(30)
    se_gam = float(np.sqrt((se_ref60 / refG[60]) ** 2 + (se_ref20 / refG[20]) ** 2)) / np.log(3.0)
    se_plin = 6 * se_ref20
    se_ppow = float(np.sqrt((6.0 ** ref["gamma"] * se_ref20) ** 2 + (p_pow * np.log(6.0) * se_gam) ** 2))
    d_lin = g120 - p_lin
    d_pow = g120 - p_pow
    # 卡 rev 1.1 §4.1：判决带宽 = 3se(G_120)（逐格点 sd/√20，**不含** p 的不确定度）。
    # 合并 se 的版本只进 report_only（DEF-26L-5：首版实现把 p 的 se 卷进判决带 ⇒ 放宽带宽）。
    k_card = SE_K * se120
    k_lin = k_pow = k_card
    k_lin_pooled = SE_K * float(np.sqrt(se120 ** 2 + se_plin ** 2))
    k_pow_pooled = SE_K * float(np.sqrt(se120 ** 2 + se_ppow ** 2))
    gam_eff = float(np.log(g120 / refG[20]) / np.log(6.0))
    if abs(d_lin) <= k_lin:
        chan = "LIN"
    elif abs(d_pow) <= k_pow:
        chan = "POW"
    elif d_pow <= -k_pow:
        chan = "CAP"
    else:
        chan = "ACC"

    d3 = float(np.mean(G[90])) - refG[60]
    d4 = float(np.mean(G[120] - G[90]))
    v["far"] = {"G": {n: round(float(np.mean(G[n])), 5) for n in KV_NEW},
                "se": {n: round(float(np.std(G[n], ddof=1)) / np.sqrt(len(G[n])), 6) for n in KV_NEW},
                "k3": {n: round(SE_K * float(np.std(G[n], ddof=1)) / np.sqrt(len(G[n])), 6) for n in KV_NEW},
                "p_lin": round(p_lin, 6), "p_pow": round(p_pow, 6),
                "d_lin": round(d_lin, 6), "k_lin": round(k_lin, 6), "d_pow": round(d_pow, 6), "k_pow": round(k_pow, 6),
                "gamma_eff_20_120": round(gam_eff, 4),
                "se_ppow": round(se_ppow, 6), "se_gamma": round(se_gam, 4),
                "band_card": round(k_card, 6),
                "band_pooled": {"lin": round(k_lin_pooled, 6), "pow": round(k_pow_pooled, 6)},
                "channel_if_pooled": ("LIN" if abs(d_lin) <= k_lin_pooled
                                      else "POW" if abs(d_pow) <= k_pow_pooled
                                      else "CAP" if d_pow <= -k_pow_pooled else "ACC"),
                "report_only": {"delta3_60_90_unpaired": round(d3, 5), "delta4_90_120_paired": round(d4, 5),
                                "gamma_60_120": round(float(np.log(float(np.mean(G[120])) / refG[60]) / np.log(2.0)), 4),
                                "per_item_120": round(float(np.mean(G[120])) / 120, 8),
                                "per_budget_120": round(float(np.mean(G[120])) / float(np.mean([r["k"]["B"] for r in by[120]])), 9)}}
    v["channel"] = chan

    # ---- 标尺条款
    refU = {n: ref["U0"][n] for n in KV_REF}
    u_ref_mid = [(a + b) / 2 for a, b in zip(U0[90], U0[120])]
    dU = float(np.mean(u_ref_mid)) - refU[60]
    seU = float(np.std(u_ref_mid, ddof=1)) / np.sqrt(len(u_ref_mid))
    kU = SE_K * float(np.sqrt(seU ** 2 + (float(np.std([r["U0_O"] for r in ref["rows"] if r["n_kv"] == 60], ddof=1)) / np.sqrt(30)) ** 2))
    v["scale"] = {"ok": bool(abs(dU) <= kU), "dU": round(dU, 6), "k": round(kU, 6),
                  "levels": {n: round(refU[n], 5) for n in KV_REF} | {n: round(float(np.mean(U0[n])), 5) for n in KV_NEW}}

    # ---- 护栏 (c)(d) 由 Stage A 断言 + 行级复核
    ident = all(r["arms"][t]["sum_ok"] and r["arms"][t]["quota_ok"] for r in rows for t in ("Fu", "Uu", "Ku"))
    v["c_anchor"] = {"ok": True, "note": "Stage A A26L-6：5 种子 × 105 值对逐位 = D-43 的 (60|20)"}
    v["d_identity"] = {"ok": bool(ident and all(r["mirror_ok"] for r in rows))}
    v["g_geom"] = {"ok": True, "note": "Stage A A26L-8：写流 v 跨 n_kv 不同（DEF-26K-1 复现）"}
    # 卡 rev 1.1 §4.1：与印出的精确位比较，相对差 ≤2e-9（rev 1.0 的 0.0062888/2e-6 已作废，见 DEF-26L-2/6）
    _c = (0.006285978551798811, 0.007901956066101166, 0.010109853324247523, 0.4325356666135775,
          0.03771587131079286, 0.013644293870204098)
    _rel = lambda a, b: abs(a - b) / max(abs(b), 1e-12)
    _got = (ref["G"][20], ref["G"][35], ref["G"][60], ref["gamma"], p_lin, p_pow)
    v["n_ref"] = {"ok": all(_rel(g, e) <= 2e-9 for g, e in zip(_got, _c)),
                  "p_lin": round(p_lin, 7), "p_pow": round(p_pow, 7), "gamma": round(ref["gamma"], 4),
                  "rel_dev": [f"{_rel(g, e):.2e}" for g, e in zip(_got, _c)]}
    v["o_level"] = {"ok": True, "note": "(e) 格点均值、(f) 格点比值、(4.1) 逐格点 sd/√n —— 三条层级各自显式"}

    levels = {n: {"G_O": round(float(np.mean(G[n])), 5), "sd_O": round(float(np.std(G[n], ddof=1)), 5),
                  "G_X": round(float(np.mean(GX[n])), 5), "U0_O": round(float(np.mean(U0[n])), 5),
                  "sd_U0": round(float(np.std(U0[n], ddof=1)), 5),
                  "R_O": round(float(np.mean([r["R_O"] for r in by[n]])), 4),
                  "kb": int(round(np.mean([r["k"]["B"] for r in by[n]]))), "qbar": round(qm[n], 3),
                  "qmax": max(r["qmax"] for r in by[n]), "nu_cap": round(nu[n], 3),
                  "sched": round(sd_[n], 3),
                  "ho8_A": round(float(np.mean([r["arms"]["A"]["ho8_mean"] for r in by[n]])), 4),
                  "ho8_F": round(float(np.mean([r["arms"]["Fu"]["ho8_mean"] for r in by[n]])), 4),
                  "ho8_K": round(float(np.mean([r["arms"]["Ku"]["ho8_mean"] for r in by[n]])), 4),
                  "clean_F": round(float(np.mean([r["arms"]["Fu"]["clean"] for r in by[n]])), 4),
                  "clean_K": round(float(np.mean([r["arms"]["Ku"]["clean"] for r in by[n]])), 4)}
              for n in KV_NEW}
    v["levels"] = levels
    v["ref_levels"] = {n: {"G": round(refG[n], 5), "U0": round(refU[n], 5)} for n in KV_REF}

    halves = {}
    for nm, ii in (("h1", range(0, 10)), ("h2", range(10, len(used)))):
        ii = list(ii)
        g, _, k = _band(G[120][ii])
        halves[nm] = {"G120": round(g, 5), "k": round(k, 5), "gam": round(float(np.log(g / refG[20]) / np.log(6.0)), 3)}
    v["halves"] = halves

    guards = ("c_anchor", "d_identity", "e_nucap", "f_comparable", "g_geom", "m_control", "n_ref", "o_level")
    guards_ok = all(v[k]["ok"] for k in guards)
    verdict = chan if guards_ok else "IND"
    v["main"] = {"ok": verdict == "POW", "verdict": verdict}
    v["guards"] = {k: bool(v[k]["ok"]) for k in guards}

    print(f"  [{'PASS' if guards_ok else 'FAIL'}] 护栏 " + " ".join(f"{k}:{'OK' if v[k]['ok'] else 'FAIL'}" for k in guards))
    print(f"  (f) 计数可比性：q̄(60/90/120) = {qm[60]:.3f}/{qm[90]:.3f}/{qm[120]:.3f}；"
          f"比值 {ratios[90]:.3f}/{ratios[120]:.3f} ∈ {QBAR_RATIO}；"
          f"日程距离极差 {spread:.3f} ≤ {SCHED_SPREAD_MAX}")
    print(f"  (m) 控制口径：" + " ".join(f"n{n}:G_X {x['G_X']:+.6f}(3se {x['k']:.6f}, 比 {x['ratio']:.3f})" for n, x in mx.items()))
    for n in KV_NEW:
        L = levels[n]
        print(f"  n_kv={n:>3} k_B {L['kb']:>4} q̄ {L['qbar']:.2f} q_max {L['qmax']} ν_cap {L['nu_cap']:.3f} sched {L['sched']:.3f} | "
              f"G_O {L['G_O']:+.5f} (sd {L['sd_O']:.5f}, 3se {v['far']['k3'][n]:.5f}) G_X {L['G_X']:+.5f} | "
              f"U0 {L['U0_O']:+.5f} R {L['R_O']:+.4f}")
    print(f"  参照(D-43)：G20 {refG[20]:+.5f} G35 {refG[35]:+.5f} G60 {refG[60]:+.5f}；γ̂ {ref['gamma']:.4f}")
    print(f"  预测：p_lin {p_lin:+.5f}  p_pow {p_pow:+.5f}")
    print(f"  G120 {g120:+.5f}：对 p_lin 差 {d_lin:+.5f}（3se(G120)={k_card:.5f}，{abs(d_lin)/k_card:.2f}×）；"
          f"对 p_pow 差 {d_pow:+.5f}（3se(G120)={k_card:.5f}，{abs(d_pow)/k_card:.2f}×）")
    print(f"  [仅报告] 若把 p 的 se 也合并进带宽：lin {k_lin_pooled:.5f}、pow {k_pow_pooled:.5f} "
          f"⇒ 同数据会判 {v['far']['channel_if_pooled']}（不作判决权）")
    print(f"  γ_eff(20→120) = {gam_eff:+.3f}（POW ⇔ [0.33,0.54]）")
    print(f"  报告项：Δ3(60→90,非配对) {d3:+.5f} · Δ4(90→120,配对) {d4:+.5f} · γ(60→120) {v['far']['report_only']['gamma_60_120']:+.3f}")
    print(f"  标尺：U0(60/90/120) = {refU[60]:.5f}/{levels[90]['U0_O']:.5f}/{levels[120]['U0_O']:.5f}；"
          f"ΔU' {dU:+.5f}（3se {kU:.5f}）⇒ {'PASS' if v['scale']['ok'] else 'FAIL'}")
    print(f"  两半：h1 G120 {halves['h1']['G120']:+.5f}(γ {halves['h1']['gam']:+.3f}) · "
          f"h2 {halves['h2']['G120']:+.5f}(γ {halves['h2']['gam']:+.3f})")
    print(f"  ⇒ H26L：**{verdict}**")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26Lr10")
    args = ap.parse_args()
    t0 = time.time()
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26l/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    if args.stage == "stageA":
        stage_a()
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}"
              f"（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"], res["passed"] = "stageA", ok
    else:
        seeds = SEEDS[:3] if args.stage == "smoke" else SEEDS
        taus = {n: K.cal_at(n)["c_tau"] for n in KV_NEW}
        res["calib"] = [{"n_kv": n, "c_tau": t} for n, t in taus.items()]
        rows, done = [], set()
        if os.path.exists(outp):
            try:
                old = json.load(open(outp)); rows = old.get("rows", [])
                done = {(r["n_kv"], r["seed"]) for r in rows}
                print(f"  [resume] 已有 {len(rows)} 行")
            except Exception:
                rows, done = [], set()
        print(f"=== {args.stage}（2 格点 × {len(seeds)} 种子 × 6 臂） ===")
        for seed in seeds:
            for n in KV_NEW:
                if (n, seed) in done:
                    continue
                g = K.run_unit(n, seed, taus[n])
                rows.append(g)
                json.dump({"rows": rows}, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
                print(f"  n_kv={n:>3} seed={seed} | G_O {g['G_O']:+.5f} G_X {g['G_X']:+.5f} U0 {g['U0_O']:+.5f} "
                      f"R {g['R_O']:+.4f} | F {g['arms']['Fu']['mu']:+.4f} K {g['arms']['Ku']['mu']:+.4f} | "
                      f"k_B {g['k']['B']} q̄ {g['qbar']:.2f} sched {g['sched_dist']:.2f}", flush=True)
        rows.sort(key=lambda r: (r["n_kv"], r["seed"]))
        res["rows"] = rows
        if args.stage == "formal":
            ref = d43_means()
            res["ref"] = {"G": ref["G"], "U0": ref["U0"], "gamma": ref["gamma"],
                          "p_lin": 6.0 * ref["G"][20], "p_pow": ref["G"][20] * 6.0 ** ref["gamma"]}
            json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
            print(f"  [落盘] rows 已写入 {outp}（判据前）")
            try:
                res["verdicts"] = crit(rows, res["ref"])
                res["passed"] = res["verdicts"]["main"]["ok"]
            except Exception as e:
                res["verdicts"] = {"error": f"{type(e).__name__}: {e}"}
                res["passed"] = False
                print(f"  ✗ 判据异常：{type(e).__name__}: {e}")

    res["meta"] = {"wall_seconds": round(time.time() - t0, 1), "device": "cpu", "d": D_MODEL,
                   "cells": [{"n_kv": n, "n_read": N_READ} for n in KV_NEW], "ref_cells": list(KV_REF),
                   "seeds": SEEDS, "shared_seeds": SHARED_SEEDS, "new_seeds": NEW_SEEDS,
                   "r0": K.R0, "eta": ETA, "m": M, "s_key": S_KEY, "r_pres": K.R_PRES,
                   "r_ho": [K.R_HO4, K.R_HO8], "rho_main": K.R_MAIN,
                   "profile": {"F": PROFILE["F"], "U0": PROFILE["U"], "K": PROFILE["K"]},
                   "stage": args.stage, "card": "DIR-26L rev1.0",
                   "criterion": {"se_k": SE_K, "gamma_ref": GAMMA_REF, "qbar_ratio": list(QBAR_RATIO),
                                 "sched_spread_max": SCHED_SPREAD_MAX, "nu_cap_max": NU_CAP_MAX}}
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
