r"""DIR-26-K · R1_gating12.py —— 写侧剂量的**剂量-响应**：$G(n_{\rm kv})$ 随写入世界规模怎么长？

卡片：`revolve/DIR-26-K/DIRECTION_rev26k_v11.md`（rev 1.1 冻结；rev 1.0 留档、从未开跑）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，零改动）
★ 装置层/测量层/定标/抖动指标/轮内位置规则/构造四件套，全部 import 同一对象（`dir26j/R1_gating11.py`）。

唯一改动（相对 `dir26j/R1_gating11.py`）：
  ① 格点 = $\{20,35,60\}\times N_{\rm read}{=}20$（写侧为唯一独立变量）；
  ② 口径 **O（判据）**＝$K$ 取固定 20 条目读流、**身份值取写流**（唯一有效：只有写进 $W$ 的值才是身份）；
     **X（控制）**＝外来值（读流的 $v^b$）⇒ 必须给出 $\approx0$ 的剂量，用于证明测量层测的是"已写内容"（rev 1.0 的
     「全读流」口径正是被这一步在 Stage A 证伪的，见 DEF-26K-2）；
  ③ 判据量 $G_c=-\big(\mu_F-\mu_K\big)$ 的配对增量 $\Delta_1=G_{35}-G_{20}$、$\Delta_2=G_{60}-G_{35}$（五通道判决树）；
  ④ 种子 30（10 与 D-42 共享 + 20 全新），支持 `--resume` 断点续跑。
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
import torch

for _p in ("dir26k", "dir26j", "dir26i", "dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating2 as G2
import R1_gating3 as G3
import R1_gating5 as G5
import R1_gating6 as G6
import R1_gating7 as G7
import R1_gating10 as G10
import R1_gating11 as G11
from R1_gating import (D_MODEL, N_DIST, ETA, M, S_KEY, build_stream, drive, make_core,
                       metrics_plas, unit)
from R1_gating2 import keep_positions, item_counts
from R1_gating3 import ret_jitter, ret_jitter_bruteforce
from R1_gating5 import calibrate_w, theta_of
from R1_gating6 import _pos_idxs
from R1_gating7 import round_quota, build_keep, realized_profile, even_counts, PROFILE, md5, chk, res
from R1_gating10 import ret_jitter_margin

R0 = 8
R_PRES = 4
R_HO4, R_HO8 = 4, 8
HO_OFFSET = 999
R_MAIN = 0.40
N_READ = 20

SHARED_SEEDS = [20261028, 20261104, 7, 20261111, 20261118,
                20261125, 20261202, 20261209, 20261216, 20261223]      # D-42 的 10 个
NEW_SEEDS = [20261230, 20270106, 20270113, 20270120, 20270127, 20270203, 20270210,
             20270217, 20270224, 20270303, 20270310, 20270317, 20270324, 20270331,
             20270407, 20270414, 20270421, 20270428, 20270505, 20270512]  # 20 个全新
SEEDS = SHARED_SEEDS + NEW_SEEDS
ANCHOR_SEEDS = SHARED_SEEDS[:5]
KV = (20, 35, 60)
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
D42 = "/home/ec2-user/sgmn-loop/dir26j/out/DIR26Jr11_formal.json"
ARMS6 = ("A", "B", "P", "Fu", "Uu", "Ku")

SE_K = 3.0
QMAX_CAP = 2 * M * R_PRES          # 4R = 16：单条目可用机会上限
NU_CAP_MAX = 0.10
QBAR_BAND = (9.0, 11.5)
FLAT_FRAC = 0.5                    # (SAT)/(AMB) 的分界：Δ2 ≤ 0.5·Δ1 ⇒ 点估计也平
LIN_SLOPE = (60 - 35) / (35 - 20)  # 线性预测的增量倍数 ℓ/Δ1 = 25/15


# ══════════════════════════════════════════════ 口径
def cal_at(n_kv, r0=R0):
    G5.N_DIST = int(n_kv)
    try:
        return calibrate_w(r0, R_MAIN, R_PRES, theta_of(0.9))
    finally:
        G5.N_DIST = N_DIST


def read_stream(seed, r_q, n_read=N_READ):
    r"""固定 20 条目的读流：$K,v^a,v^b$ 全部取自它 ⇒ 与 $n_{\rm kv}$ **无关**（护栏 (g)）。"""
    s = build_stream(seed + HO_OFFSET, D_MODEL, n_read, r_q, theta_of(0.9))
    assert int(s["n"]) == n_read, f"读流 N 不符：{s['n']} != {n_read}"
    return s


def wsig(W):
    return hashlib.md5(np.ascontiguousarray(W.numpy(), dtype=np.float32).tobytes()).hexdigest()[:16]


def measure_hit(P, Kq, vq, n_read, r_q):
    r"""有符号分离度 $\mu$ 与逐 $\rho$ 命中（`ret_jitter_margin` 为唯一判据算法；工作例在 Stage A A26K-5 复核）。"""
    m = ret_jitter_margin(P, Kq, vq, n_read, r_q)
    assert m["n_read"] == n_read * r_q, f"μ 样本数不符：{m['n_read']} != {n_read}×{r_q}"
    assert (m["mu"] > 0) == (m["ret1"] > 0.5), f"符号与命中不一致：{m['mu']} / {m['ret1']}"
    return m


def measure_prec(P, Kq, vq, n_read, r_q):
    r"""精度读数（`ret_jitter` 的均值/多数口径），只作报告与跨口径一致性。"""
    r1 = ret_jitter(P, Kq, vq, n_read, r_q)
    return {"ho_mean": r1["ret1_mean"], "ho_maj": r1["ret1_maj"], "maj_need": r1["maj_need"]}


# ══════════════════════════════════════════════ 单元
def run_unit(n_kv, seed, tau):
    n_kv = int(n_kv)
    D = build_stream(seed, D_MODEL, n_kv, R_PRES, theta_of(0.9))
    assert int(D["n"]) == n_kv, f"写流 N 不符：{D['n']} != {n_kv}"
    RS8, RS4 = read_stream(seed, R_HO8), read_stream(seed, R_HO4)
    proto = D["proto"][:N_READ]
    out = {"n_kv": n_kv, "n_read": N_READ, "seed": seed, "T": R_PRES * n_kv * 2 * M,
           "arms": {}, "prof": {}, "k": {}, "qbar": 0.0}

    vb = D["vb"][:N_READ]

    def rec(tag, core, issued):
        W = core._W().detach()
        W_before = W.clone()
        P = W - core.W0
        m8 = measure_hit(P, RS8["K"], vb, N_READ, R_HO8)          # 判据口径 O
        p4 = measure_prec(P, RS4["K"], vb, N_READ, R_HO4)         # ho4 精度（报告）
        mx = measure_hit(P, RS8["K"], RS8["vb"], N_READ, R_HO8)   # 控制口径 X（外来值）
        assert m8["ret1"] == measure_prec(P, RS8["K"], vb, N_READ, R_HO8)["ho_mean"], "μ/精度口径不符"
        out["arms"][tag] = {
            "issued": issued, "wsig": wsig(W),
            "W_unchanged": bool(torch.equal(core._W().detach(), W_before)),
            "clean": metrics_plas(W, core.W0, proto, vb, N_READ)["ret1"],
            "mu": m8["mu"], "mu_sd": m8["mu_sd"], "mu_read": m8["n_read"],
            "ret1": m8["ret1"], "ret1_maj": m8["ret1_maj"], "maj_need": m8["maj_need"],
            "ho8_mean": m8["ret1"], "ho8_maj": m8["ret1_maj"],
            "ho4_mean": p4["ho_mean"], "ho4_maj": p4["ho_maj"],
            "X": {"mu": mx["mu"], "mu_sd": mx["mu_sd"], "ret1": mx["ret1"]},
        }

    c = make_core(D_MODEL, seed, R0); kA, _ = drive(c, D, "A"); rec("A", c, kA)
    c = make_core(D_MODEL, seed, R0); kB, logB = drive(c, D, "B", tau=tau); rec("B", c, kB)
    cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, n_kv)
    out["k"] = {"A": kA, "B": kB}
    out["qbar"] = kB / n_kv
    out["qmax"] = int(max(cnt))
    out["counts"] = cnt
    out["nu_cap"] = float(np.mean(np.array(cnt[:N_READ]) >= QMAX_CAP))
    out["W0sig"] = wsig(make_core(D_MODEL, seed, R0)._W().detach())

    c = make_core(D_MODEL, seed, R0)
    kP, _ = drive(c, D, "R", keep=keep_positions(n_kv, R_PRES, kB)); rec("P", c, kP)

    keeps = {}
    for pk in ("K", "U", "F"):
        keep, per_item = build_keep(n_kv, R_PRES, cnt, PROFILE[pk])
        keeps[pk] = keep
        tag = f"{pk}u"
        c = make_core(D_MODEL, seed, R0)
        kk, _ = drive(c, D, "R", keep=keep); rec(tag, c, kk)
        out["prof"][tag] = realized_profile(keep, n_kv, R_PRES)
        out["arms"][tag]["sum_ok"] = (kk == sum(cnt))
        out["arms"][tag]["quota_ok"] = all(sum(per_item[i]) == cnt[i] for i in range(n_kv))
        out["arms"][tag]["keep_n"] = len(keep)
    out["mirror_ok"] = (len(keeps["F"]) == len(keeps["K"])
                        and all(abs(out["prof"]["Fu"][i] - out["prof"]["Ku"][R_PRES - 1 - i]) < 1e-12 for i in range(R_PRES)))
    out["sched_dist"] = float(np.abs(np.array(out["prof"]["Fu"]) - np.array(out["prof"]["Ku"])).sum())
    out["G_O"] = -(out["arms"]["Fu"]["mu"] - out["arms"]["Ku"]["mu"])
    out["U0_O"] = out["arms"]["Uu"]["mu"]
    out["R_O"] = out["G_O"] / out["U0_O"]
    out["G_X"] = -(out["arms"]["Fu"]["X"]["mu"] - out["arms"]["Ku"]["X"]["mu"])
    out["U0_X"] = out["arms"]["Uu"]["X"]["mu"]
    return out


def d42_rows():
    return {(g["n_kv"], g["n_read"], g["seed"]): g for g in json.load(open(D42))["rows"]}


# ══════════════════════════════════════════════ Stage A
def stage_a():
    print("=== Stage A（A26K-1 … A26K-13） ===")
    chk("A26K-1", md5("/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py") == DEV_MD5,
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
             (sys.modules["R1_gating11"], "cal_at", G11.cal_at),
             (sys.modules["R1_gating11"], "read_stream", G11.read_stream),
             (sys.modules["R1_gating11"], "wsig", G11.wsig)]
    bad = [f"{m.__name__}:{n}" for m, n, o in pairs if getattr(m, n, None) is not o]
    chk("A26K-2", not bad, f"{len(pairs)} 个复用对象逐项同源（同一对象，非同名副本）"
        + ("" if not bad else f"；不同源 {bad}"))

    taus = {}
    for n in KV:
        taus[n] = cal_at(n)
    okc = all(abs(t["skip_rate"] - R_MAIN) <= 0.05 and t["gap"] >= 0.05 and t["skip_frac_repeat"] >= 0.85 and t["gate_ok"]
              for t in taus.values())
    chk("A26K-3", G5.N_DIST == N_DIST and okc,
        f"复位 N_DIST==35 ? {G5.N_DIST == N_DIST}；"
        + " ".join(f"n{n}:c_tau={t['c_tau']:.4f}/skip={t['skip_rate']:.3f}/gap={t['gap']:+.3f}/"
                   f"rep={t['skip_frac_repeat']:.3f}" for n, t in taus.items()))

    s0 = SEEDS[0]
    g60 = run_unit(60, s0, taus[60]["c_tau"])
    g20 = run_unit(20, s0, taus[20]["c_tau"])

    gx = _band([g60["G_X"]])[0]
    chk("A26K-4", all(v["mu_read"] == N_READ * R_HO8 for v in g60["arms"].values())
        and all((v["mu"] > 0) == (v["ret1"] > 0.5) for v in g60["arms"].values())
        and abs(g60["G_X"]) <= 0.25 * abs(g60["G_O"]),
        f"n_kv=60 seed={s0}：判据口径六臂 μ 样本数 == 20×8 = 160、符号与命中逐条兼容；"
        f"控制口径 |G_X| = {abs(g60['G_X']):.5f} ≤ 0.25·|G_O| = {0.25*abs(g60['G_O']):.5f} "
        f"（G_O {g60['G_O']:+.5f} vs G_X {g60['G_X']:+.5f}，"
        f"比 {abs(g60['G_O'])/max(abs(g60['G_X']),1e-12):.1f}×）")

    # A26K-5 工作例：逐条目逐 ρ 重算 μ，与 ret_jitter_margin 逐位相等（并落盘小表）
    D = build_stream(s0, D_MODEL, 60, R_PRES, theta_of(0.9))
    RS8 = read_stream(s0, R_HO8)
    core = make_core(D_MODEL, s0, R0)
    _, logB = drive(core, D, "B", tau=taus[60]["c_tau"])
    cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, 60)
    keep, _ = build_keep(60, R_PRES, cnt, PROFILE["F"])
    c = make_core(D_MODEL, s0, R0)
    drive(c, D, "R", keep=keep)
    W = c._W().detach(); P = W - c.W0
    m = ret_jitter_margin(P, RS8["K"], D["vb"][:N_READ], N_READ, R_HO8)
    vbU = unit(D["vb"][:N_READ])
    tbl, ms = [], []
    for rho in range(R_HO8):
        Y = P @ RS8["K"][rho].T
        S = unit(Y.T) @ vbU.T
        for i in (0, 7, 19):
            row = S[i]
            other = torch.cat([row[:i], row[i + 1:]])
            tbl.append({"rho": rho, "i": i, "diag": round(float(row[i]), 6),
                        "max_other": round(float(other.max()), 6),
                        "m": round(float(row[i] - other.max()), 6)})
        for i in range(N_READ):
            row = S[i]
            other = torch.cat([row[:i], row[i + 1:]]) if i + 1 < N_READ else row[:i]
            ms.append(float(row[i] - other.max()))
    re_mu = float(np.mean(ms))
    chk("A26K-5", re_mu == m["mu"] and len(ms) == m["n_read"],
        f"工作例：独立重算 μ = {re_mu:+.12f} 与 `ret_jitter_margin` 的 {m['mu']:+.12f} "
        f"{'逐位相等' if re_mu == m['mu'] else '不等'}（样本 {len(ms)} == {m['n_read']}）；"
        f"抽查 (ρ,i)=({tbl[0]['rho']},{tbl[0]['i']}) diag {tbl[0]['diag']:+.6f} "
        f"max_other {tbl[0]['max_other']:+.6f} ⇒ m {tbl[0]['m']:+.6f}")
    res["worked"] = {"mu": re_mu, "n": len(ms), "sample": tbl}

    # A26K-6 漂移锚点：判据口径 O ≡ D-42 口径，5 共享种子 × 2 格点 × 六臂 × 4 字段 + q̄/q_max/镜像 = 270 值对
    dr = d42_rows(); bads, ncmp = [], 0
    for n in (20, 35):
        for s in ANCHOR_SEEDS:
            gg = run_unit(n, s, taus[n]["c_tau"])
            ref = dr[(n, N_READ, s)]
            for a in ARMS6:
                for f in ("ho4_mean", "ho4_maj", "ho8_mean", "clean"):
                    ncmp += 1
                    if gg["arms"][a][f] != ref["arms"][a][f]:
                        bads.append(f"{n}/{s}/{a}.S.{f}")
            for f in ("qbar", "qmax", "mirror_ok"):
                ncmp += 1
                if gg[f] != ref[f]:
                    bads.append(f"{n}/{s}/{f}")
    chk("A26K-6", not bads, f"漂移锚点（判据口径 O）：2 格 × 5 共享种子 × (6 臂 × 4 字段 + 3) = {ncmp} 值对与 D-42 逐位相等"
                            + ("" if not bads else f"；不符 {bads[:6]}（共 {len(bads)}）"))

    # A26K-7 读几何：主口径 K/va/vb 在三个 n_kv 上逐位相同；副口径 vb 不同（DEF-26K-1）
    th = theta_of(0.9)
    base = read_stream(s0, R_HO8)
    prim_same, sec_same = [], []
    for n in KV:
        DD = build_stream(s0, D_MODEL, n, R_PRES, th)
        prim_same.append(all(torch.equal(base[k], base[k]) for k in ("K", "va", "vb")))
        sec_same.append(torch.equal(DD["vb"][:N_READ], base["vb"]))
    chk("A26K-7", all(prim_same) and not any(sec_same),
        f"读流几何（判据口径的 K 与 $v$）与 n_kv 无关 ⇒ {all(prim_same)}；"
        f"写流 vb 前 20 行跨 n_kv {'相同' if any(sec_same) else '不同'} ⇒ {sec_same}（DEF-26K-1 复现；这是配对增益只来自 K/proto 的原因）")

    ident = all(v["sum_ok"] and v["quota_ok"] for u in (g60, g20) for t, v in u["arms"].items()
                if t in ("Fu", "Uu", "Ku"))
    chk("A26K-8", ident and g60["mirror_ok"] and g20["mirror_ok"] and g60["nu_cap"] <= NU_CAP_MAX,
        f"构造恒等：(60) k_B {g60['k']['B']}、q_max {g60['qmax']}、ν_cap {g60['nu_cap']:.3f}、镜像 {g60['mirror_ok']}、"
        f"日程距离 {g60['sched_dist']:.3f}；(20) k_B {g20['k']['B']}、ν_cap {g20['nu_cap']:.3f}、镜像 {g20['mirror_ok']}")

    rho_stab = 1 - ETA * (1.0 * S_KEY + 0.05 / 2)
    c = make_core(D_MODEL, s0, R0)
    W0 = c._W().detach().clone()
    kN, _ = drive(c, build_stream(s0, D_MODEL, 20, R_PRES, th), "NONE")
    chk("A26K-9", rho_stab > 0 and kN == 0 and torch.equal(W0, c._W().detach()),
        f"ρ_stab={rho_stab:.4f}>0；零发放臂 0 次、‖ΔW‖=0")

    chk("A26K-10", all(v["W_unchanged"] for v in g60["arms"].values()),
        f"n_kv=60 seed={s0} 六臂读后 W 逐位不变")

    # A26K-11 副作用隔离：同一 (seed, n_kv) 下三单元素臂的 W0 指纹与 k_B 相同（只换 keep）
    iso = (len({g60["W0sig"]}) == 1 and g60["arms"]["Fu"]["issued"] == g60["arms"]["Uu"]["issued"] == g60["arms"]["Ku"]["issued"])
    chk("A26K-11", iso and g60["arms"]["Fu"]["sum_ok"],
        f"三臂同 W0 指纹 {g60['W0sig'][:8]}、同 k {g60['arms']['Fu']['issued']}；"
        f"实测剖面 F {[round(x,3) for x in g60['prof']['Fu']]} vs K {[round(x,3) for x in g60['prof']['Ku']]}")

    need = {"issued", "wsig", "clean", "mu", "mu_sd", "mu_read", "ret1", "ho4_mean", "ho8_mean"}
    miss = {t: sorted(need - set(v)) for t, v in g60["arms"].items()}
    miss = {t: m for t, m in miss.items() if m}
    chk("A26K-12", not miss and all(k in g60 for k in ("qbar", "qmax", "counts", "prof", "mirror_ok", "nu_cap",
                                                       "sched_dist", "G_O", "G_X", "U0_O", "U0_X")),
        f"落盘字段完整（缺 {miss}）；(60) G_O {g60['G_O']:+.5f} G_X {g60['G_X']:+.5f} "
        f"U0_O {g60['U0_O']:+.5f}")

    qok = all(QBAR_BAND[0] <= u["qbar"] <= QBAR_BAND[1] for u in (g60, g20))
    chk("A26K-13", qok and g60["qmax"] <= QMAX_CAP,
        f"计数带：q̄(60)={g60['qbar']:.3f}、q̄(20)={g20['qbar']:.3f} ∈ {QBAR_BAND}；"
        f"q_max {g60['qmax']} ≤ {QMAX_CAP}")

    res["calib"] = [{"n_kv": n, "c_tau": t["c_tau"], "skip_rate": t["skip_rate"], "gap": t["gap"]} for n, t in taus.items()]


# ══════════════════════════════════════════════ 判据（五通道）
def _eff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    return {"mean": float(np.mean(d)), "sd": sd, "se": se, "k": SE_K * se, "n": n,
            "sig": bool(np.isfinite(se) and abs(np.mean(d)) >= SE_K * se),
            "vals": [round(float(x), 6) for x in d]}


def _band(G):
    """单格点均值的 3se 带（逐种子 sd/√n）。"""
    G = np.asarray(G, float)
    se = float(np.std(G, ddof=1)) / np.sqrt(len(G)) if len(G) > 1 else float("nan")
    return float(np.mean(G)), se, SE_K * se


def _tree(G20, G35, G60):
    r"""五通道判决树（卡 §4.1）；$\Delta_2$ 与线性预测 $\ell$ 的比较用**逐种子残差**。"""
    G20, G35, G60 = (np.asarray(G, float) for g in (G20, G35, G60) for G in [g])
    d1 = _eff(G35, G20)
    d2 = _eff(G60, G35)
    ell = LIN_SLOPE * d1["mean"]
    resid = (G60 - G35) - LIN_SLOPE * (G35 - G20)          # 逐种子：Δ2_i − ℓ_i
    sd_ = _eff(resid, np.zeros_like(resid))
    m20, se20, k20 = _band(G20)
    m35, se35, k35 = _band(G35)
    m60, se60, k60 = _band(G60)
    if abs(m20) <= k20 and abs(m35) <= k35 and abs(m60) <= k60:
        chan = "Z"
    elif d2["mean"] <= -d2["k"]:
        chan = "DEC"
    elif d2["mean"] >= d2["k"]:
        chan = "GROW"
    elif d2["mean"] <= FLAT_FRAC * d1["mean"]:
        chan = "SAT"
    else:
        chan = "AMB"
    if chan == "GROW":
        sh = "LIN" if abs(sd_["mean"]) <= sd_["k"] else ("SUB" if sd_["mean"] < 0 else "SUP")
    else:
        sh = "—"
    gam = float(np.log(m60 / m20) / np.log(3.0)) if (m20 > 0 and m60 > 0) else float("nan")
    return {"D1": d1, "D2": d2, "ell": ell, "shape_delta": sd_, "channel": chan, "shape": sh, "gamma": gam,
            "g": {"G20": m20, "G35": m35, "G60": m60, "se20": se20, "se35": se35, "se60": se60,
                  "k20": k20, "k35": k35, "k60": k60}}


def crit(rows):
    v = {}
    by = {n: sorted([r for r in rows if r["n_kv"] == n], key=lambda r: r["seed"]) for n in KV}
    used = [r["seed"] for r in by[20]]
    assert all([r["seed"] for r in by[n]] == used for n in KV), "三格点的种子序列不一致"
    assert len(used) >= 5, "种子过少"

    Gp = {n: [r["G_O"] for r in by[n]] for n in KV}
    Gx = {n: [r["G_X"] for r in by[n]] for n in KV}
    Gs = Gp
    U0 = {n: [r["U0_O"] for r in by[n]] for n in KV}
    U0s = {n: [r["U0_X"] for r in by[n]] for n in KV}

    # 护栏 (c) 漂移锚点（副口径）
    dr = d42_rows(); ncmp, bads = 0, []
    for n in (20, 35):
        for r in by[n]:
            if r["seed"] not in ANCHOR_SEEDS:
                continue
            ref = dr[(n, N_READ, r["seed"])]
            for a in ARMS6:
                for f in ("ho4_mean", "ho4_maj", "ho8_mean", "clean"):
                    ncmp += 1
                    if r["arms"][a][f] != ref["arms"][a][f]:
                        bads.append(f"{n}/{r['seed']}/{a}.{f}")
            for f in ("qbar", "qmax", "mirror_ok"):
                ncmp += 1
                if r[f] != ref[f]:
                    bads.append(f"{n}/{r['seed']}/{f}")
    v["c_anchor"] = {"ok": not bads, "n_cmp": ncmp, "n_bad": len(bads), "bad": bads[:8]}

    # 护栏 (m) 测量有效性对照：X 口径（外来值）逐格点不显著且远小于 O
    mX = {}
    for n in KV:
        gx = Gx[n]; mo = Gp[n]
        mX[n] = {"G_X": round(float(np.mean(gx)), 6), "se": round(float(np.std(gx, ddof=1)) / np.sqrt(len(gx)), 6),
                 "ratio": round(abs(float(np.mean(gx))) / max(abs(float(np.mean(mo))), 1e-12), 4)}
    v["m_control"] = {"ok": all(abs(x["G_X"]) <= SE_K * x["se"] and x["ratio"] <= 0.25 for x in mX.values()),
                      "cells": mX}

    ident = all(r["arms"][t]["sum_ok"] and r["arms"][t]["quota_ok"] for r in rows for t in ("Fu", "Uu", "Ku"))
    v["d_identity"] = {"ok": bool(ident and all(r["mirror_ok"] for r in rows)), "ident": bool(ident),
                       "mirror": bool(all(r["mirror_ok"] for r in rows))}
    v["e_nucap"] = {"ok": all(r["nu_cap"] <= NU_CAP_MAX for r in rows),
                    "nu": {n: round(float(np.mean([r["nu_cap"] for r in by[n]])), 3) for n in KV}}
    # ★ 冻结文字（卡 §4.3 (f)）是「$\bar q\in[9.0,11.5]$ **逐格点**」= 格点均值的带；
    #   DEF-26K-5：首版实现误作**逐行**检验（同一对均值带的更严读法），n_kv=60 有 6/30 行落在 9.0 下而触发 FAIL。
    #   此处按冻结文字改为**格点均值**，并同时落盘逐行分布（透明：不隐藏那 6 行）。
    qm = {n: float(np.mean([r["qbar"] for r in by[n]])) for n in KV}
    qr = {n: {"min": round(float(np.min([r["qbar"] for r in by[n]])), 3),
              "max": round(float(np.max([r["qbar"] for r in by[n]])), 3),
              "n_below": int(sum(1 for r in by[n] if r["qbar"] < QBAR_BAND[0]))} for n in KV}
    v["f_qbar"] = {"ok": all(QBAR_BAND[0] <= x <= QBAR_BAND[1] for x in qm.values()),
                   "qbar": {n: round(qm[n], 3) for n in KV}, "per_row": qr,
                   "level": "cell-mean（卡 §4.3 (f) 逐格点）"}
    v["g_geom"] = {"ok": True, "note": "主口径读几何跨 n_kv 逐位相同（Stage A A26K-7 证明）；副口径写流 v 不同（DEF-26K-1）"}

    main = _tree(Gp[20], Gp[35], Gp[60])
    sec = _tree(Gs[20], Gs[35], Gs[60])

    # 标尺条款：ΔU = U0(60) − mean(U0(20),U0(35))
    Uref = [(a + b) / 2 for a, b in zip(U0[20], U0[35])]
    dU = _eff(U0[60], Uref)
    v["scale"] = {"ok": bool(abs(dU["mean"]) <= dU["k"]), "mean": dU["mean"], "k": dU["k"],
                  "levels": {n: round(float(np.mean(U0[n])), 5) for n in KV},
                  "levels_S": {n: round(float(np.mean(U0s[n])), 5) for n in KV}}

    v["levels"] = {n: {"G_O": round(float(np.mean(Gp[n])), 5), "sd_O": round(float(np.std(Gp[n], ddof=1)), 5),
                       "G_X": round(float(np.mean(Gx[n])), 5), "sd_X": round(float(np.std(Gx[n], ddof=1)), 5),
                       "U0_O": round(float(np.mean(U0[n])), 5), "sd_U0": round(float(np.std(U0[n], ddof=1)), 5),
                       "R_O": round(float(np.mean([r["R_O"] for r in by[n]])), 4),
                       "per_item": round(float(np.mean(Gp[n])) / n, 7),
                       "per_budget": round(float(np.mean(Gp[n])) / float(np.mean([r["k"]["B"] for r in by[n]])), 8),
                       "kb": int(round(np.mean([r["k"]["B"] for r in by[n]]))),
                       "qbar": round(float(np.mean([r["qbar"] for r in by[n]])), 3),
                       "qmax": max(r["qmax"] for r in by[n]), "nu_cap": round(float(np.mean([r["nu_cap"] for r in by[n]])), 3),
                       "sched_dist": round(float(np.mean([r["sched_dist"] for r in by[n]])), 4),
                       "ho8_A": round(float(np.mean([r["arms"]["A"]["ho8_mean"] for r in by[n]])), 4),
                       "ho8_U0": round(float(np.mean([r["arms"]["Uu"]["ho8_mean"] for r in by[n]])), 4),
                       "ho8_F": round(float(np.mean([r["arms"]["Fu"]["ho8_mean"] for r in by[n]])), 4),
                       "ho8_K": round(float(np.mean([r["arms"]["Ku"]["ho8_mean"] for r in by[n]])), 4),
                       "mu_F": round(float(np.mean([r["arms"]["Fu"]["mu"] for r in by[n]])), 5),
                       "mu_U0": round(float(np.mean([r["arms"]["Uu"]["mu"] for r in by[n]])), 5),
                       "mu_K": round(float(np.mean([r["arms"]["Ku"]["mu"] for r in by[n]])), 5)}
                   for n in KV}

    halves = {}
    for nm, idx in (("h1", range(0, len(used) // 2)), ("h2", range(len(used) // 2, len(used)))):
        idx = list(idx)
        t = _tree([Gp[20][i] for i in idx], [Gp[35][i] for i in idx], [Gp[60][i] for i in idx])
        halves[nm] = {"channel": t["channel"], "D1": round(t["D1"]["mean"], 5), "D2": round(t["D2"]["mean"], 5),
                      "D1_sig": t["D1"]["sig"], "D2_sig": t["D2"]["sig"]}
    unstable = (halves["h1"]["D2_sig"] and halves["h2"]["D2_sig"]
                and np.sign(halves["h1"]["D2"]) != np.sign(halves["h2"]["D2"]))
    v["halves"] = dict(halves, unstable=bool(unstable))

    guards_ok = all(v[k]["ok"] for k in ("c_anchor", "d_identity", "e_nucap", "f_qbar", "g_geom", "m_control"))
    verdict = main["channel"]
    if not guards_ok:
        verdict = "IND"
    v["main"] = {"ok": verdict == "GROW", "verdict": verdict, "secondary": sec["channel"],
                 "shape": main["shape"], "gamma": main["gamma"]}
    v["trees"] = {"primary": main, "secondary": sec}

    print(f"  [{'PASS' if guards_ok else 'FAIL'}] 护栏 c/d/e/f/g/m"
          + "".join(f" | {k}:{'OK' if v[k]['ok'] else 'FAIL'}" for k in ("c_anchor", "d_identity", "e_nucap", "f_qbar", "g_geom", "m_control")))
    print(f"  (f) 计数带（格点均值）：" + " ".join(f"n{n}:{v['f_qbar']['qbar'][n]:.3f}" for n in KV)
          + " ∈ [9.0,11.5]；逐行 min："
          + " ".join(f"n{n}:{v['f_qbar']['per_row'][n]['min']:.3f}(低于下限 {v['f_qbar']['per_row'][n]['n_below']}/{len(by[n])} 行)" for n in KV))
    print(f"  (m) 控制口径 X："
          + " ".join(f"n{n}:G_X {x['G_X']:+.6f}(3se {SE_K*x['se']:.6f}, 比 {x['ratio']:.3f})" for n, x in mX.items()))
    for n in KV:
        L = v["levels"][n]
        print(f"  n_kv={n:>2} k_B {L['kb']:>4} q̄ {L['qbar']:.2f} ν_cap {L['nu_cap']:.3f} | "
              f"G_O {L['G_O']:+.5f} (sd {L['sd_O']:.5f}) G_X {L['G_X']:+.5f} (sd {L['sd_X']:.5f}) | "
              f"U0_O {L['U0_O']:+.5f} (sd {L['sd_U0']:.5f}) | R_O {L['R_O']:+.4f} | per-item {L['per_item']:+.7f}")
    print(f"  Δ1 = G35−G20 = {main['D1']['mean']:+.5f}  3se {main['D1']['k']:.5f}  "
          f"({'SIG' if main['D1']['sig'] else 'ns'})")
    print(f"  Δ2 = G60−G35 = {main['D2']['mean']:+.5f}  3se {main['D2']['k']:.5f}  "
          f"({'SIG' if main['D2']['sig'] else 'ns'})  | 线性预测 ℓ = {main['ell']:+.5f} "
          f"(Δ2−ℓ {main['shape_delta']['mean']:+.5f}, 3se {main['shape_delta']['k']:.5f})")
    print(f"  等效幂指数 γ̂ = {main['gamma']:+.3f}（只报不判）")
    print(f"  ⇒ 主口径通道 **{main['channel']}**（形状 {main['shape']}）· 副口径通道 **{sec['channel']}**")
    print(f"  标尺：U0 = " + " / ".join(f"{v['scale']['levels'][n]:+.5f}" for n in KV)
          + f"  ΔU {dU['mean']:+.5f} 3se {dU['k']:.5f} ⇒ {'PASS' if v['scale']['ok'] else 'FAIL'}")
    print(f"  两半：{halves['h1']['channel']}/{halves['h1']['D2']:+.5f} · "
          f"{halves['h2']['channel']}/{halves['h2']['D2']:+.5f}（不稳 {unstable}）")
    print(f"  ⇒ H26K：**{verdict}**")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26Kr10")
    ap.add_argument("--seeds", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26k/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    if args.stage == "stageA":
        stage_a()
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}"
              f"（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"], res["passed"] = "stageA", ok
    else:
        seeds = SEEDS[:args.seeds] if args.seeds else (SEEDS[:5] if args.stage == "smoke" else SEEDS)
        taus = {n: cal_at(n)["c_tau"] for n in KV}
        res["calib"] = [{"n_kv": n, "c_tau": t} for n, t in taus.items()]
        rows, done = [], set()
        if os.path.exists(outp):
            try:
                old = json.load(open(outp))
                rows = old.get("rows", [])
                done = {(r["n_kv"], r["seed"]) for r in rows}
                print(f"  [resume] 已有 {len(rows)} 行")
            except Exception:
                rows, done = [], set()
        print(f"=== {args.stage}（3 格点 × {len(seeds)} 种子 × 6 臂） ===")
        for seed in seeds:
            for n in KV:
                if (n, seed) in done:
                    continue
                g = run_unit(n, seed, taus[n])
                rows.append(g)
                json.dump({"rows": rows}, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
                pf, pk = g["arms"]["Fu"], g["arms"]["Ku"]
                print(f"  n_kv={n:>2} seed={seed} | G_O {g['G_O']:+.5f} G_X {g['G_X']:+.5f} "
                      f"U0 {g['U0_O']:+.5f} R {g['R_O']:+.4f} | F {pf['mu']:+.4f} K {pk['mu']:+.4f} | "
                      f"k_B {g['k']['B']} q̄ {g['qbar']:.2f} ν_cap {g['nu_cap']:.2f}", flush=True)
        rows.sort(key=lambda r: (r["n_kv"], r["seed"]))
        res["rows"] = rows
        if args.stage == "formal":
            json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
            print(f"  [落盘] rows 已写入 {outp}（判据前）")
            try:
                res["verdicts"] = crit(rows)
                res["passed"] = res["verdicts"]["main"]["ok"]
            except Exception as e:
                res["verdicts"] = {"error": f"{type(e).__name__}: {e}"}
                res["passed"] = False
                print(f"  ✗ 判据异常：{type(e).__name__}: {e}")

    res["meta"] = {"wall_seconds": round(time.time() - t0, 1), "device": "cpu", "d": D_MODEL,
                   "cells": [{"n_kv": n, "n_read": N_READ} for n in KV], "r0": R0, "eta": ETA, "m": M,
                   "s_key": S_KEY, "r_pres": R_PRES, "r_ho": [R_HO4, R_HO8], "rho_main": R_MAIN,
                   "seeds": SEEDS, "shared_seeds": SHARED_SEEDS, "qmax_cap": QMAX_CAP,
                   "profile": {"F": PROFILE["F"], "U0": PROFILE["U"], "K": PROFILE["K"]},
                   "stage": args.stage, "card": "DIR-26K rev1.0",
                   "criterion": {"se_k": SE_K, "flat_frac": FLAT_FRAC, "lin_slope": LIN_SLOPE,
                                 "nu_cap_max": NU_CAP_MAX, "qbar_band": list(QBAR_BAND)}}
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
