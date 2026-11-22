r"""DIR-26-J · R1_gating11.py —— 剖面效应的**量纲分解**：写侧 = 剂量，读侧 = 标尺？

卡片：`revolve/DIR-26J/DIRECTION_rev26j_v11.md`（rev 1.1 冻结；rev 1.0 `_v10` 留档）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，零改动）
★ 装置层/测量层/定标/抖动指标/轮内位置规则/构造四件套，全部 import 同一对象（`dir26i/R1_gating10.py`）。

唯一改动（相对 `dir26i/R1_gating10.py`）：
  ① 格点变为**真正的 2×2** $(n_{\rm kv},N_{\rm read})\in\{(20,12),(20,20),(35,12),(35,20)\}$（两旋钮独立）；
  ② 判据量换成**量纲分解**：剂量 $D_{\rm abs}=\mu_F-\mu_K$ 与标尺 $\mu_{U0}$ 分开，
     $E_{1..4}$（剂量效应）/ $S_{1..4}$（标尺效应）各自算配对 se，$3\mathrm{se}$ 判定；
  ③ 精度读数降级为报告项（$N_{\rm read}{=}12$ 触顶，跨格点不可比）；
  ④ 种子 10 个（5 个与 D-41 共享作锚点 + 5 个全新）。
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
import torch

for _p in ("dir26i", "dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating2 as G2
import R1_gating3 as G3
import R1_gating5 as G5
import R1_gating6 as G6
import R1_gating7 as G7
import R1_gating10 as G10
from R1_gating import (D_MODEL, N_DIST, ETA, M, S_KEY, build_stream, drive, make_core,
                       metrics_plas, unit)
from R1_gating2 import keep_positions, item_counts
from R1_gating3 import ret_jitter, ret_jitter_bruteforce
from R1_gating5 import calibrate_w, theta_of
from R1_gating6 import _pos_idxs
from R1_gating7 import round_quota, build_keep, realized_profile, even_counts, PROFILE, md5, chk, res

R0 = 8
R_PRES = 4
R_HO4, R_HO8 = 4, 8
HO_OFFSET = 999
R_MAIN = 0.40

SEEDS = [20261028, 20261104, 7, 20261111, 20261118,          # 与 D-41 共享（锚点）
         20261125, 20261202, 20261209, 20261216, 20261223]    # 全新
ANCHOR_SEEDS = SEEDS[:5]
CELLS = ({"n_kv": 20, "n_read": 12}, {"n_kv": 20, "n_read": 20},
         {"n_kv": 35, "n_read": 12}, {"n_kv": 35, "n_read": 20})
CELL_KEY = [f"{c['n_kv']}|{c['n_read']}" for c in CELLS]
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
D41 = "/home/ec2-user/sgmn-loop/dir26i/out/DIR26Ir10_formal.json"
ARMS6 = ("A", "B", "P", "Fu", "Uu", "Ku")

SE_K = 3.0              # 判据带宽：3·se
SIGN_NEED = 8           # A26J-11：逐种子同向 ≥8/10
QMAX_CAP = 16


# ══════════════════════════════════════════════ 单元（结构逐字沿用 R1_gating10）
def cal_at(n_kv, r0=R0):
    G5.N_DIST = int(n_kv)
    try:
        return calibrate_w(r0, R_MAIN, R_PRES, theta_of(0.9))
    finally:
        G5.N_DIST = N_DIST


def read_stream(seed, n_read, r_q):
    s = build_stream(seed + HO_OFFSET, D_MODEL, n_read, r_q, theta_of(0.9))
    assert int(s["n"]) == n_read, f"读流 N 不符：{s['n']} != {n_read}"
    return s


def wsig(W):
    return hashlib.md5(np.ascontiguousarray(W.numpy(), dtype=np.float32).tobytes()).hexdigest()[:16]


def run_unit(cell, seed, tau):
    n_kv, n_read = int(cell["n_kv"]), int(cell["n_read"])
    assert n_read <= n_kv, f"读集大于写集：{n_read} > {n_kv}"
    D = build_stream(seed, D_MODEL, n_kv, R_PRES, theta_of(0.9))
    assert int(D["n"]) == n_kv, f"写流 N 不符：{D['n']} != {n_kv}"
    RS4, RS8 = read_stream(seed, n_read, R_HO4), read_stream(seed, n_read, R_HO8)
    assert int(RS8["n"]) == n_read, f"读流(ho8) N 不符：{RS8['n']} != {n_read}"
    out = {"n_kv": n_kv, "n_read": n_read, "seed": seed, "T": R_PRES * n_kv * 2 * M,
           "arms": {}, "prof": {}, "k": {}, "qbar": 0.0}

    def rec(tag, core, issued):
        W = core._W().detach()
        W_before = W.clone()
        P = W - core.W0
        m4 = ret_jitter(P, RS4["K"], D["vb"][:n_read], n_read, R_HO4)
        m8 = ret_jitter(P, RS8["K"], D["vb"][:n_read], n_read, R_HO8)
        mg = G10.ret_jitter_margin(P, RS8["K"], D["vb"][:n_read], n_read, R_HO8)
        assert mg["ret1"] == m8["ret1_mean"], f"{tag} 边际/精度口径不符：{mg['ret1']} != {m8['ret1_mean']}"
        assert (mg["mu"] > 0) == (mg["ret1"] > 0.5), f"{tag} 符号与命中不一致：{mg['mu']} / {mg['ret1']}"
        out["arms"][tag] = {"issued": issued, "wsig": wsig(W),
                            "clean": metrics_plas(W, core.W0, D["proto"][:n_read], D["vb"][:n_read], n_read)["ret1"],
                            "ho4_mean": m4["ret1_mean"], "ho4_maj": m4["ret1_maj"],
                            "ho8_mean": m8["ret1_mean"], "ho8_maj": m8["ret1_maj"],
                            "mu": mg["mu"], "mu_sd": mg["mu_sd"], "mu_read": mg["n_read"],
                            "maj_need4": m4["maj_need"], "maj_need8": m8["maj_need"],
                            "W_unchanged": bool(torch.equal(core._W().detach(), W_before))}

    c = make_core(D_MODEL, seed, R0); kA, _ = drive(c, D, "A"); rec("A", c, kA)
    c = make_core(D_MODEL, seed, R0); kB, logB = drive(c, D, "B", tau=tau); rec("B", c, kB)
    cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, n_kv)
    out["k"] = {"A": kA, "B": kB}
    out["qbar"] = kB / n_kv
    out["qmax"] = int(max(cnt))

    c = make_core(D_MODEL, seed, R0)
    kP, _ = drive(c, D, "R", keep=keep_positions(n_kv, R_PRES, kB)); rec("P", c, kP)

    keeps = {}
    for pk in ("K", "U", "F"):
        keep, per_item = build_keep(n_kv, R_PRES, cnt, PROFILE[pk])
        keeps[pk] = keep
        tag = f"{pk}u"
        c = make_core(D_MODEL, seed, R0); kk, _ = drive(c, D, "R", keep=keep); rec(tag, c, kk)
        out["prof"][tag] = realized_profile(keep, n_kv, R_PRES)
        out["arms"][tag]["sum_ok"] = (kk == sum(cnt))
        out["arms"][tag]["quota_ok"] = all(sum(per_item[i]) == cnt[i] for i in range(n_kv))
        out["arms"][tag]["keep_n"] = len(keep)
    out["mirror_ok"] = (len(keeps["F"]) == len(keeps["K"])
                        and all(abs(out["prof"]["Fu"][i] - out["prof"]["Ku"][R_PRES - 1 - i]) < 1e-12 for i in range(R_PRES)))
    out["counts"] = cnt
    out["D_abs"] = out["arms"]["Fu"]["mu"] - out["arms"]["Ku"]["mu"]
    out["U0"] = out["arms"]["Uu"]["mu"]
    out["R_mu"] = out["D_abs"] / out["U0"]
    return out


def d41_rows():
    return {(g["n_kv"], g["n_read"], g["seed"]): g for g in json.load(open(D41))["rows"]}


# ══════════════════════════════════════════════ Stage A
def stage_a():
    print("=== Stage A（A26J-1 … A26J-12） ===")
    chk("A26J-1", md5("/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py") == DEV_MD5,
        f"装置 md5 == {DEV_MD5}")

    want = {"R1_gating": ("build_stream", "drive", "make_core", "metrics_plas", "stream_idx", "unit"),
            "R1_gating2": ("keep_positions", "item_counts"),
            "R1_gating3": ("ret_jitter", "ret_jitter_bruteforce"),
            "R1_gating5": ("calibrate_w", "theta_of"),
            "R1_gating6": ("_pos_idxs",),
            "R1_gating7": ("round_quota", "build_keep", "realized_profile", "even_counts")}
    bad = [f"{n}:{f}" for n, fs in want.items() for f in fs
           if getattr(sys.modules[n], f, None) is not getattr(G7, f, getattr(sys.modules[n], f, None))]
    chk("A26J-2", not bad, "17 个复用对象的来源模块逐项核对" if not bad else f"不同源 {bad}")

    ks = sorted({c["n_kv"] for c in CELLS})
    taus = {n: cal_at(n) for n in ks}
    okc = all(abs(t["skip_rate"] - R_MAIN) <= 0.05 and t["gap"] >= 0.05 and t["skip_frac_repeat"] >= 0.85 and t["gate_ok"]
              for t in taus.values())
    chk("A26J-3", G5.N_DIST == N_DIST and okc,
        f"标定后复位 R1_gating5.N_DIST==35 ? {G5.N_DIST == N_DIST}；"
        + " ".join(f"n{n}:c_tau={t['c_tau']:.4f}/skip={t['skip_rate']:.3f}/gap={t['gap']:+.3f}/"
                   f"rep={t['skip_frac_repeat']:.3f}" for n, t in taus.items()))

    s0 = SEEDS[0]
    g35_12 = run_unit({"n_kv": 35, "n_read": 12}, s0, taus[35]["c_tau"])
    g20_20 = run_unit({"n_kv": 20, "n_read": 20}, s0, taus[20]["c_tau"])

    chk("A26J-4", all((v["mu"] > 0) == (v["ho8_mean"] > 0.5) for v in g35_12["arms"].values())
        and all(v["mu_read"] == 12 * R_HO8 for v in g35_12["arms"].values()),
        f"(35,12) seed={s0}：六臂 μ 与命中逐位兼容；μ 样本数 == 12×8 = 96（该格 ho8 已触顶，断言仍成立："
        + " ".join(f"{t}:{g35_12['arms'][t]['ho8_mean']:.3f}" for t in ARMS6) + "）")

    ident = all(v["sum_ok"] and v["quota_ok"] for u in (g35_12, g20_20) for t, v in u["arms"].items()
                if t in ("Fu", "Uu", "Ku"))
    chk("A26J-5", ident and g35_12["qmax"] <= QMAX_CAP and g35_12["mirror_ok"] and g20_20["mirror_ok"],
        f"构造恒等：(35,12) k_B {g35_12['k']['B']}、q_max {g35_12['qmax']}、镜像 {g35_12['mirror_ok']}；"
        f"(20,20) k_B {g20_20['k']['B']}、q_max {g20_20['qmax']}、镜像 {g20_20['mirror_ok']}")

    # A26J-10 写世界共享：同写侧、不同读集 ⇒ 三臂（Fu/Uu/Ku）W 指纹逐位相同
    g20_12 = run_unit({"n_kv": 20, "n_read": 12}, s0, taus[20]["c_tau"])
    g35_20 = run_unit({"n_kv": 35, "n_read": 20}, s0, taus[35]["c_tau"])
    pairs = []
    for nk, a, b in ((20, g20_12, g20_20), (35, g35_12, g35_20)):
        eq = all(a["arms"][t]["wsig"] == b["arms"][t]["wsig"] for t in ("Fu", "Uu", "Ku"))
        pairs.append((f"{nk}|12 vs {nk}|20", eq))
    chk("A26J-10", all(eq for _, eq in pairs),
        f"同写侧、不同读集的 W 指纹逐位相同：{pairs}（种子 {s0}；写世界唯一 ⇒ 读集只改「问什么」）")

    # A26J-6 锚点（(35,20)、(20,20) × D-41 的 5 个共享种子 × 六臂 × 4 字段 + q̄/q_max/镜像）
    dr = d41_rows()
    bads, ncmp = [], 0
    for cell in ({"n_kv": 35, "n_read": 20}, {"n_kv": 20, "n_read": 20}):
        for s in ANCHOR_SEEDS:
            gg = run_unit(cell, s, taus[cell["n_kv"]]["c_tau"])
            ref = dr[(cell["n_kv"], cell["n_read"], s)]
            for a in ARMS6:
                for f in ("ho4_mean", "ho4_maj", "ho8_mean", "clean"):
                    ncmp += 1
                    if gg["arms"][a][f] != ref["arms"][a][f]:
                        bads.append(f"{cell['n_kv']}|{cell['n_read']}/{s}/{a}.{f}")
            for f in ("qbar", "qmax", "mirror_ok"):
                ncmp += 1
                if gg[f] != ref[f]:
                    bads.append(f"{cell['n_kv']}|{cell['n_read']}/{s}/{f}")
    chk("A26J-6", not bads, f"锚点：2 格 × 5 共享种子 × (6 臂 × 4 字段 + q̄/q_max/镜像) = {ncmp} 值对与 D-41 逐位相等"
                            + ("" if not bads else f"；不符 {bads[:6]}（共 {len(bads)}）"))

    chk("A26J-7", g35_12["n_kv"] == 35 and g35_12["n_read"] == 12,
        f"流-读集分离：D.n {g35_12['n_kv']} == 35、RS.n {g35_12['n_read']} == 12")

    rho_stab = 1 - ETA * (1.0 * S_KEY + 0.05 / 2)
    D = build_stream(s0, D_MODEL, 20, R_PRES, theta_of(0.9))
    core = make_core(D_MODEL, s0, R0)
    W0 = core._W().detach().clone()
    kN, _ = drive(core, D, "NONE")
    chk("A26J-8", rho_stab > 0 and kN == 0 and torch.equal(W0, core._W().detach()),
        f"ρ_stab={rho_stab:.4f}>0；零发放臂 0 次、‖ΔW‖=0")

    chk("A26J-9", all(v["W_unchanged"] for v in g35_12["arms"].values()),
        f"(35,12) seed={s0} 六臂读后 W 逐位不变")

    # A26J-11 标尺无天花板（均值排序 + 逐种子同向，用 Stage A 已跑过的 (20,12)/(20,20) 之外再补 (35,12)/(35,20)）
    u12, u20 = [], []
    for s in (SEEDS[0], SEEDS[1]):
        u12.append(run_unit({"n_kv": 35, "n_read": 12}, s, taus[35]["c_tau"])["U0"])
        u20.append(run_unit({"n_kv": 35, "n_read": 20}, s, taus[35]["c_tau"])["U0"])
    same = sum(1 for a, b in zip(u12, u20) if a > b)
    chk("A26J-11", np.mean(u12) > np.mean(u20) and same >= len(u12) - 1,
        f"(35,·) 标尺：U(·,12) {np.mean(u12):+.4f} > U(·,20) {np.mean(u20):+.4f} ? "
        f"{np.mean(u12) > np.mean(u20)}；逐种子同向 {same}/{len(u12)}（Stage A 抽样 2 种子；正式轮按 ≥8/10 判）")

    need = {"issued", "wsig", "clean", "ho4_mean", "ho4_maj", "ho8_mean", "ho8_maj", "mu", "mu_sd",
            "mu_read", "maj_need4", "maj_need8", "W_unchanged"}
    miss = {t: sorted(need - set(v)) for t, v in g35_12["arms"].items()}
    miss = {t: m for t, m in miss.items() if m}
    chk("A26J-12", not miss and all(k in g35_12 for k in ("qbar", "qmax", "counts", "prof", "mirror_ok", "D_abs", "U0", "R_mu")),
        f"落盘字段完整（缺 {miss}）；(35,12) D_abs {g35_12['D_abs']:+.4f}、U0 {g35_12['U0']:+.4f}、"
        f"R_mu {g35_12['R_mu']:+.4f}")
    res["calib"] = [{"n_kv": n, "c_tau": t["c_tau"], "skip_rate": t["skip_rate"], "gap": t["gap"]} for n, t in taus.items()]


# ══════════════════════════════════════════════ 判据（量纲分解）
def _eff(a, b, seeds_idx=None):
    """配对效应 mean/sd/se/se_k（对逐种子差 a−b）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if seeds_idx is not None:
        a, b = a[seeds_idx], b[seeds_idx]
    d = a - b
    n = len(d)
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    return {"mean": float(np.mean(d)), "sd": sd, "se": se, "k": SE_K * se, "n": n,
            "sig": bool(np.isfinite(se) and abs(np.mean(d)) >= SE_K * se), "vals": [round(float(x), 6) for x in d]}


def _pattern(D, U, idx=None):
    E1 = _eff(D["35|20"], D["20|20"], idx); E2 = _eff(D["35|12"], D["20|12"], idx)
    E3 = _eff(D["20|12"], D["20|20"], idx); E4 = _eff(D["35|12"], D["35|20"], idx)
    S1 = _eff(U["35|20"], U["20|20"], idx); S2 = _eff(U["35|12"], U["20|12"], idx)
    S3 = _eff(U["20|12"], U["20|20"], idx); S4 = _eff(U["35|12"], U["35|20"], idx)
    I = _eff([a - b for a, b in zip(D["35|12"], D["35|20"])],
             [a - b for a, b in zip(D["20|12"], D["20|20"])], idx)
    sig_pos = lambda e: e["sig"] and e["mean"] > 0
    dose = (E1["sig"] and E2["sig"] and np.sign(E1["mean"]) == np.sign(E2["mean"])
            and not E3["sig"] and not E4["sig"])
    scale = (sig_pos(S3) and sig_pos(S4) and not S1["sig"] and not S2["sig"])
    fals = (E1["sig"] and E3["sig"] and np.sign(E3["mean"]) == np.sign(E1["mean"])) or \
           (E2["sig"] and E4["sig"] and np.sign(E4["mean"]) == np.sign(E2["mean"]))
    chan = "CONFIRMED" if (dose and scale and not I["sig"]) else ("FALSIFIED" if fals else "MIXED")
    return {"E": {"E1": E1, "E2": E2, "E3": E3, "E4": E4}, "S": {"S1": S1, "S2": S2, "S3": S3, "S4": S4},
            "I": I, "dose": bool(dose), "scale": bool(scale), "fals": bool(fals), "channel": chan}


def crit(rows):
    v = {}
    by = {c: sorted([r for r in rows if f"{r['n_kv']}|{r['n_read']}" == c], key=lambda r: r["seed"]) for c in CELL_KEY}
    used = [r["seed"] for r in by["35|20"]]
    assert all([r["seed"] for r in by[c]] == used for c in CELL_KEY), "四格点的种子序列不一致"

    D = {c: [r["D_abs"] for r in by[c]] for c in CELL_KEY}
    U = {c: [r["U0"] for r in by[c]] for c in CELL_KEY}
    RM = {c: [r["R_mu"] for r in by[c]] for c in CELL_KEY}

    # (c) 锚点（Stage A 已查，这里对正式落盘再查一遍共享 5 种子）
    dr = d41_rows(); ncmp, bads = 0, []
    for cell in ("35|20", "20|20"):
        nk, nr = [int(x) for x in cell.split("|")]
        for r in by[cell]:
            if r["seed"] not in ANCHOR_SEEDS:
                continue
            ref = dr[(nk, nr, r["seed"])]
            for a in ARMS6:
                for f in ("ho4_mean", "ho4_maj", "ho8_mean", "clean"):
                    ncmp += 1
                    if r["arms"][a][f] != ref["arms"][a][f]:
                        bads.append(f"{cell}/{r['seed']}/{a}.{f}")
            for f in ("qbar", "qmax", "mirror_ok"):
                ncmp += 1
                if r[f] != ref[f]:
                    bads.append(f"{cell}/{r['seed']}/{f}")
    v["c_anchor"] = {"ok": not bads, "n_cmp": ncmp, "n_bad": len(bads), "bad": bads[:8]}

    ident = all(r["arms"][t]["sum_ok"] and r["arms"][t]["quota_ok"] for r in rows for t in ("Fu", "Uu", "Ku"))
    mirror = all(r["mirror_ok"] for r in rows)
    v["d_identity"] = {"ok": bool(ident and mirror), "ident": bool(ident), "mirror": bool(mirror),
                       "qbar": {c: round(float(np.mean([r["qbar"] for r in by[c]])), 3) for c in CELL_KEY},
                       "qmax": {c: max(r["qmax"] for r in by[c]) for c in CELL_KEY}}

    # (g) 写世界共享：同写侧两格的三臂 W 指纹相同（逐种子）
    sh_ok = []
    for nk, other in ((20, "20|20"), (35, "35|20")):
        lab12 = f"{nk}|12"
        for s in used:
            a = next(r for r in by[lab12] if r["seed"] == s)
            b = next(r for r in by[other] if r["seed"] == s)
            sh_ok.append(all(a["arms"][t]["wsig"] == b["arms"][t]["wsig"] for t in ("Fu", "Uu", "Ku")))
    v["g_write_shared"] = {"ok": all(sh_ok), "n": len(sh_ok), "n_ok": int(sum(sh_ok)),
                           "note": "同写侧不同读集：Fu/Uu/Ku 三臂 W 指纹逐位相同（写世界唯一）"}

    main = _pattern(D, U)
    h1 = _pattern(D, U, list(range(5)))
    h2 = _pattern(D, U, list(range(5, 10)))
    unstable = ((h1["E"]["E1"]["sig"] and h2["E"]["E1"]["sig"]
                 and np.sign(h1["E"]["E1"]["mean"]) != np.sign(h2["E"]["E1"]["mean"]))
                or {h1["channel"], h2["channel"]} == {"CONFIRMED", "FALSIFIED"})
    v["halves"] = {"h1": h1["channel"], "h2": h2["channel"], "unstable": bool(unstable),
                   "E1_h1": round(h1["E"]["E1"]["mean"], 5), "E1_h2": round(h2["E"]["E1"]["mean"], 5),
                   "E1_h1_sig": h1["E"]["E1"]["sig"], "E1_h2_sig": h2["E"]["E1"]["sig"]}

    v["levels"] = {c: {"D_abs": round(float(np.mean(D[c])), 5), "sd_D": round(float(np.std(D[c], ddof=1)), 5),
                       "U0": round(float(np.mean(U[c])), 5), "sd_U": round(float(np.std(U[c], ddof=1)), 5),
                       "R_mu": round(float(np.mean(RM[c])), 4),
                       "ho8_A": round(float(np.mean([r["arms"]["A"]["ho8_mean"] for r in by[c]])), 4),
                       "ho8_U0": round(float(np.mean([r["arms"]["Uu"]["ho8_mean"] for r in by[c]])), 4),
                       "ho8_F": round(float(np.mean([r["arms"]["Fu"]["ho8_mean"] for r in by[c]])), 4),
                       "ho8_K": round(float(np.mean([r["arms"]["Ku"]["ho8_mean"] for r in by[c]])), 4),
                       "kb": int(round(np.mean([r["k"]["B"] for r in by[c]]))),
                       "qbar": round(float(np.mean([r["qbar"] for r in by[c]])), 3)}
                   for c in CELL_KEY}
    v["eff"] = main
    v["criterion"] = {"se_k": SE_K, "sign_need": SIGN_NEED}

    verdict = ("INDETERMINATE" if not (v["d_identity"]["ok"] and v["c_anchor"]["ok"] and v["g_write_shared"]["ok"])
               else ("UNSTABLE" if unstable else main["channel"]))
    v["main"] = {"ok": verdict == "CONFIRMED", "verdict": verdict}

    for k in ("c_anchor", "d_identity", "g_write_shared"):
        print(f"  [{'PASS' if v[k]['ok'] else 'FAIL'}] {k}")
    for c in CELL_KEY:
        L = v["levels"][c]
        print(f"  {c:>6} k_B {L['kb']:>3} | D_abs {L['D_abs']:+.5f} (sd {L['sd_D']:.5f}) | "
              f"U0 {L['U0']:+.5f} (sd {L['sd_U']:.5f}) | R_mu {L['R_mu']:+.4f} | ho8 A {L['ho8_A']:.3f}")
    for nm, e in v["eff"]["E"].items():
        print(f"  {nm} (剂量) {e['mean']:+.5f}  3se {e['k']:.5f}  {'SIG' if e['sig'] else 'ns'}")
    for nm, e in v["eff"]["S"].items():
        print(f"  {nm} (标尺) {e['mean']:+.5f}  3se {e['k']:.5f}  {'SIG' if e['sig'] else 'ns'}")
    print(f"  交互 I {v['eff']['I']['mean']:+.5f}  3se {v['eff']['I']['k']:.5f}  {'SIG' if v['eff']['I']['sig'] else 'ns'}")
    print(f"  剂量模式 {v['eff']['dose']}  标尺模式 {v['eff']['scale']}  否证触发 {v['eff']['fals']}")
    print(f"  两半 {v['halves']['h1']} / {v['halves']['h2']}（不稳 {v['halves']['unstable']}；"
          f"E1 两半 {v['halves']['E1_h1']:+.5f}/{'SIG' if v['halves']['E1_h1_sig'] else 'ns'}、"
          f"{v['halves']['E1_h2']:+.5f}/{'SIG' if v['halves']['E1_h2_sig'] else 'ns'}）")
    print(f"  ⇒ H26J 判定：**{verdict}**")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26Jr11")
    args = ap.parse_args()
    t0 = time.time()
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26j/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    if args.stage == "stageA":
        stage_a()
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"], res["passed"] = "stageA", ok
    else:
        seeds = [SEEDS[0]] if args.stage == "smoke" else SEEDS
        taus = {n: cal_at(n)["c_tau"] for n in sorted({c["n_kv"] for c in CELLS})}
        res["calib"] = [{"n_kv": n, "c_tau": t} for n, t in taus.items()]
        rows = []
        print(f"=== {args.stage}（{len(CELLS)} 格点 × {len(seeds)} 种子 × 6 臂） ===")
        for cell in CELLS:
            for s in seeds:
                g = run_unit(cell, s, taus[cell["n_kv"]])
                rows.append(g)
                json.dump({"rows": rows}, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
                a = g["arms"]
                print(f"  ({cell['n_kv']:>2}|{cell['n_read']:>2}) seed={s} | D_abs {g['D_abs']:+.5f} "
                      f"U0 {g['U0']:+.5f} R_mu {g['R_mu']:+.4f} | "
                      + " ".join(f"{t} {a[t]['mu']:+.4f}" for t in ("Fu", "Uu", "Ku"))
                      + f" | k_B {g['k']['B']} q̄ {g['qbar']:.2f}")
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
                   "cells": CELL_KEY, "r0": R0, "eta": ETA, "m": M, "s_key": S_KEY, "r_pres": R_PRES,
                   "r_ho": [R_HO4, R_HO8], "rho_main": R_MAIN, "seeds": SEEDS,
                   "profile": {"F": PROFILE["F"], "U0": PROFILE["U"], "K": PROFILE["K"]},
                   "stage": args.stage, "card": "DIR-26J rev1.1",
                   "criterion": {"se_k": SE_K, "sign_need": SIGN_NEED}}
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
