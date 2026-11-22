r"""DIR-26-I · R1_gating10.py —— 剖面效应：**写侧干扰**还是**读侧竞争**？（$n_{\rm kv}$ 与 $N_{\rm read}$ 拆开）

卡片：`revolve/DIR-26I/DIRECTION_rev26i_v10.md`（rev 1.0 冻结）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，零改动）
★ 装置层/测量层/定标/抖动指标/轮内位置规则/构造四件套，全部 import 同一对象（`dir26g/R1_gating7.py`）。

唯一改动（相对 `dir26h/R1_gating8.py`）：
  ① 格点从"一个 $N$"变成"$(n_{\rm kv},N_{\rm read})$ 两旋钮" —— 写流按 $n_{\rm kv}$ 建、读流按 $N_{\rm read}$ 另建，
     读的目标值取写流的前 $N_{\rm read}$ 个 $v^b$；
  ② 新增**边际读数** $\mu$（有符号分离度），与既有精度读数逐位兼容（Stage A A26I-4）；
  ③ 判据量换成相对边际对比 $R_\mu=(\mu_F-\mu_K)/\mu_{U0}$ 与 share。

判据量（逐字）：$\delta_{\rm wr}=R_\mu(35,20)-R_\mu(20,20)$、$\delta_{\rm rd}=R_\mu(35,35)-R_\mu(35,20)$、
$\text{share}=\lvert\delta_{\rm wr}\rvert/(\lvert\delta_{\rm wr}\rvert+\lvert\delta_{\rm rd}\rvert)$；
share $\ge0.75$ ⇒ 写侧主导（H26I 成立）、$\le0.25$ ⇒ 读侧主导（H26I 否证）、中间 ⇒ MIXED。
"""
import argparse, json, os, sys, time

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
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]
CELLS = ({"n_kv": 35, "n_read": 35}, {"n_kv": 35, "n_read": 20},
         {"n_kv": 28, "n_read": 20}, {"n_kv": 20, "n_read": 20})
CELL_KEY = [f"{c['n_kv']}|{c['n_read']}" for c in CELLS]
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
D40 = "/home/ec2-user/sgmn-loop/dir26h/out/DIR26Hr10_formal.json"
ARMS6 = ("A", "B", "P", "Fu", "Uu", "Ku")

SHARE_WR = 0.75          # (i) 写侧主导
SHARE_RD = 0.25          # (ii) 读侧主导
THR_ABS = 0.05           # 前置：锚点细胞的 |R_mu| 下限
SIGN_NEED = 4            # 前置：5 种子里 ≥4 个同号
CEIL, FLOOR = 0.95, 0.35  # (v) 天花板/地板


# ══════════════════════════════════════════════ 边际读数（本卡唯一的测量层新增）
def ret_jitter_margin(P, Kq, vb, n, r_q=8):
    r"""有符号分离度 $m_{i,\rho}=\cos(y_i,v_i)-\max_{j\ne i}\cos(y_i,v_j)$，$\mu=\overline m$。

    $\mu>0\iff\arg\max$ 命中 ⇒ 与 `ret_jitter` 的 `ret1` 逐位一致（A26I-4 断言）。
    """
    vbU = unit(vb)
    m_all = []
    hit = np.zeros((r_q, n), dtype=bool)
    for rho in range(r_q):
        Y = P @ Kq[rho].T
        S = unit(Y.T) @ vbU.T
        for i in range(n):
            row = S[i]
            j = int(torch.argmax(row).item())
            hit[rho, i] = j == i
            other = torch.cat([row[:i], row[i + 1:]]) if i + 1 < n else row[:i]
            m_all.append(float(row[i] - other.max()))
    maj_need = r_q // 2 + 1
    return {"mu": float(np.mean(m_all)), "mu_sd": float(np.std(m_all, ddof=1)),
            "ret1": float(hit.mean()), "ret1_maj": float(np.mean([hit[:, i].sum() >= maj_need for i in range(n)])),
            "maj_need": int(maj_need), "n_hit": int(hit.sum()), "n_read": int(hit.size)}


def cal_at(n_kv, r0=R0):
    r"""定标必须在该格点的**同一个 $n_{\rm kv}$** 上取分位数；调用后复位（A26H-3）。"""
    G5.N_DIST = int(n_kv)
    try:
        return calibrate_w(r0, R_MAIN, R_PRES, theta_of(0.9))
    finally:
        G5.N_DIST = N_DIST


def read_stream(seed, n_read, r_q):
    s = build_stream(seed + HO_OFFSET, D_MODEL, n_read, r_q, theta_of(0.9))
    assert int(s["n"]) == n_read, f"读流 N 不符：{s['n']} != {n_read}"
    return s


def run_unit(cell, seed, tau):
    n_kv, n_read = int(cell["n_kv"]), int(cell["n_read"])
    assert n_read <= n_kv, f"读集大于写集：{n_read} > {n_kv}"
    D = build_stream(seed, D_MODEL, n_kv, R_PRES, theta_of(0.9))
    assert int(D["n"]) == n_kv, f"写流 N 不符：{D['n']} != {n_kv}"
    RS4, RS8 = read_stream(seed, n_read, R_HO4), read_stream(seed, n_read, R_HO8)
    out = {"n_kv": n_kv, "n_read": n_read, "seed": seed, "T": R_PRES * n_kv * 2 * M,
           "arms": {}, "prof": {}, "k": {}, "qbar": 0.0}

    def rec(tag, core, issued):
        W = core._W().detach()
        W_before = W.clone()
        P = W - core.W0
        m4 = ret_jitter(P, RS4["K"], D["vb"][:n_read], n_read, R_HO4)
        m8 = ret_jitter(P, RS8["K"], D["vb"][:n_read], n_read, R_HO8)
        mg = ret_jitter_margin(P, RS8["K"], D["vb"][:n_read], n_read, R_HO8)
        assert mg["ret1"] == m8["ret1_mean"], f"{tag} 边际口径与精度口径不符：{mg['ret1']} != {m8['ret1_mean']}"
        assert (mg["mu"] > 0) == (mg["ret1"] > 0.5), f"{tag} 符号与命中不一致：{mg['mu']} / {mg['ret1']}"
        out["arms"][tag] = {"issued": issued,
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
    return out


def d40_rows():
    """D-40 的锚点细胞按 **(n, seed)** 索引（每个 n 有 5 行，不能只按 n 取）。"""
    return {(g["n"], g["seed"]): g for g in json.load(open(D40))["rows"] if g["n"] in (20, 35)}


# ══════════════════════════════════════════════ Stage A
def stage_a():
    print("=== Stage A（A26I-1 … A26I-12） ===")
    chk("A26I-1", md5("/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py") == DEV_MD5,
        f"装置 md5 == {DEV_MD5}")

    want = {"R1_gating": ("build_stream", "drive", "make_core", "metrics_plas", "stream_idx", "unit"),
            "R1_gating2": ("keep_positions", "item_counts"),
            "R1_gating3": ("ret_jitter", "ret_jitter_bruteforce"),
            "R1_gating5": ("calibrate_w", "theta_of"),
            "R1_gating6": ("_pos_idxs",),
            "R1_gating7": ("round_quota", "build_keep", "realized_profile", "even_counts")}
    bad = [f"{n}:{f}" for n, fs in want.items() for f in fs
           if getattr(sys.modules[n], f, None) is not getattr(G7, f, getattr(sys.modules[n], f, None))]
    chk("A26I-2", not bad, f"17 个复用对象的来源模块逐项核对：{want}" if not bad else f"不同源 {bad}")

    ks = sorted({c["n_kv"] for c in CELLS})
    taus = {n: cal_at(n) for n in ks}
    chk("A26I-3", G5.N_DIST == N_DIST,
        f"标定后模块全局复位：R1_gating5.N_DIST == 35 ? {G5.N_DIST == N_DIST}；c_tau 逐 n_kv = "
        f"{[(n, round(taus[n]['c_tau'], 4)) for n in ks]}")
    okc = all(abs(t["skip_rate"] - R_MAIN) <= 0.05 and t["gap"] >= 0.05 and t["skip_frac_repeat"] >= 0.85 and t["gate_ok"]
              for t in taus.values())
    chk("A26I-3b", okc, "四个 n_kv 上 skip_rate=0.400±0.05、gap≥0.05、repeat≥0.85、gate_ok："
        + " ".join(f"n{n}:{t['skip_rate']:.3f}/{t['gap']:+.3f}/{t['skip_frac_repeat']:.3f}" for n, t in taus.items()))

    s0 = SEEDS[0]

    # A26I-4/5/6/8/10/11 —— 在一个格点 (35,20) 上逐条断言
    g = run_unit({"n_kv": 35, "n_read": 20}, s0, taus[35]["c_tau"])
    bad4 = [t for t, v in g["arms"].items() if not (v["mu"] > 0) == (v["ho8_mean"] > 0.5)]
    chk("A26I-4", not bad4 and all(v["mu_read"] == 20 * R_HO8 for v in g["arms"].values()),
        f"(35,20) seed={s0}：六臂 μ 与命中逐位兼容（不符 {bad4}）；μ 的样本数 == 20×8 = 160")

    ident = all(v["sum_ok"] and v["quota_ok"] for t, v in g["arms"].items() if t in ("Fu", "Uu", "Ku"))
    chk("A26I-5", ident and g["qmax"] <= 16,
        f"(35,20)：Σ发放 == k_B {g['k']['B']} == Σq_i、逐条目配额逐位 True；q_max {g['qmax']} ≤ 16；"
        f"q̄ {g['qbar']:.2f}")

    mir = g["mirror_ok"] and g["arms"]["Ku"]["keep_n"] == g["arms"]["Fu"]["keep_n"]
    chk("A26I-6", mir, f"(35,20)：|keep_F| {g['arms']['Fu']['keep_n']} == |keep_K| {g['arms']['Ku']['keep_n']}；"
                       f"剖面 F{[round(x, 2) for x in g['prof']['Fu']]} vs K{[round(x, 2) for x in g['prof']['Ku']]} 严格镜像")

    chk("A26I-8", g["n_kv"] == 35 and g["n_read"] == 20,
        f"流-读集分离：D.n {g['n_kv']} == 35、RS.n {g['n_read']} == 20；N_read ≤ n_kv")

    # A26I-11 读流的键不随 n_kv 变
    r1 = read_stream(s0, 20, R_HO8)
    r2 = read_stream(s0, 20, R_HO8)
    chk("A26I-11", torch.equal(r1["K"], r2["K"]), f"同一 N_read=20 的读流逐位相同（与 n_kv 无关）")

    # A26I-7 锚点：两格 × 5 种子 × 六臂 × 4 字段 逐位对 D-40
    dr = d40_rows()
    bads, ncmp = [], 0
    for cell, n_ref in (({"n_kv": 35, "n_read": 35}, 35), ({"n_kv": 20, "n_read": 20}, 20)):
        for s in SEEDS:
            gg = run_unit(cell, s, taus[n_ref]["c_tau"])
            ref = dr[(n_ref, s)]
            for a in ARMS6:
                for f in ("ho4_mean", "ho4_maj", "ho8_mean", "clean"):
                    ncmp += 1
                    if gg["arms"][a][f] != ref["arms"][a][f]:
                        bads.append(f"{n_ref}/{s}/{a}.{f}")
            for f in ("qbar", "qmax", "mirror_ok"):
                ncmp += 1
                if gg[f] != ref[f]:
                    bads.append(f"{n_ref}/{s}/{f}")
    chk("A26I-7", not bads, f"锚点：2 格 × 5 种子 × (6 臂 × 4 字段 + q̄/q_max/镜像) = {ncmp} 个值对与 D-40 逐位相等"
                            + ("" if not bads else f"；不符 {bads[:6]}（共 {len(bads)}）"))

    # A26I-9 稳定性 + 零发放
    rho_stab = 1 - ETA * (1.0 * S_KEY + 0.05 / 2)
    D = build_stream(s0, D_MODEL, 20, R_PRES, theta_of(0.9))
    core = make_core(D_MODEL, s0, R0)
    W0 = core._W().detach().clone()
    kN, _ = drive(core, D, "NONE")
    chk("A26I-9", rho_stab > 0 and kN == 0 and torch.equal(W0, core._W().detach()),
        f"ρ_stab={rho_stab:.4f}>0；零发放臂 0 次、‖ΔW‖=0")

    # A26I-10 读后不变
    chk("A26I-10", all(v["W_unchanged"] for v in g["arms"].values()),
        f"(35,20) seed={s0} 六臂读后 W 逐位不变 = {all(v['W_unchanged'] for v in g['arms'].values())}")

    # A26I-12 落盘字段完整性（防止"忘了记一个量"）
    need = {"issued", "clean", "ho4_mean", "ho4_maj", "ho8_mean", "ho8_maj", "mu", "mu_sd",
            "mu_read", "maj_need4", "maj_need8", "W_unchanged"}
    miss = {t2: sorted(need - set(v2)) for t2, v2 in g["arms"].items()}
    miss = {t2: m2 for t2, m2 in miss.items() if m2}
    chk("A26I-12", not miss and all(k in g for k in ("qbar", "qmax", "counts", "prof", "mirror_ok")),
        f"落盘字段完整（缺 {miss}）；本格点 q̄ {g['qbar']:.2f}、q_max {g['qmax']}、k_B {g['k']['B']}、"
        f"μ(U0) {g['arms']['Uu']['mu']:+.4f}")
    res["calib"] = [{"n_kv": n, "c_tau": t["c_tau"], "skip_rate": t["skip_rate"], "gap": t["gap"]} for n, t in taus.items()]


# ══════════════════════════════════════════════ 判据
def crit(rows):
    v = {}
    by = {c: sorted([r for r in rows if f"{r['n_kv']}|{r['n_read']}" == c], key=lambda r: r["seed"]) for c in CELL_KEY}
    ns = {c: len(by[c]) for c in CELL_KEY}

    def rel(cell, field_num, field_den):
        out = []
        for r in by[cell]:
            den = r["arms"]["Uu"][field_den]
            out.append((r["arms"]["Fu"][field_num] - r["arms"]["Ku"][field_num]) / den if den else float("nan"))
        return out

    Rmu = {c: np.array(rel(c, "mu", "mu"), dtype=float) for c in CELL_KEY}
    Racc = {c: np.array(rel(c, "ho8_mean", "ho8_mean"), dtype=float) for c in CELL_KEY}
    M = {c: float(np.mean(Rmu[c])) for c in CELL_KEY}
    A = {c: float(np.mean(Racc[c])) for c in CELL_KEY}

    def share(num_a, num_b, den_a, den_b, d):
        d_wr = d[num_a] - d[num_b]
        d_rd = d[den_a] - d[den_b]
        s = abs(d_wr) / (abs(d_wr) + abs(d_rd)) if (abs(d_wr) + abs(d_rd)) > 0 else float("nan")
        return d_wr, d_rd, s

    dM_wr, dM_rd, sM = share("35|20", "20|20", "35|35", "35|20", M)
    dA_wr, dA_rd, sA = share("35|20", "20|20", "35|35", "35|20", A)

    # (c) 锚点
    dr = d40_rows()
    ncmp, bads = 0, []
    for cell, n_ref in (("35|35", 35), ("20|20", 20)):
        for r in by[cell]:
            ref = dr[(n_ref, r["seed"])]
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

    # (d) 构造恒等
    ident = all(r["arms"][t]["sum_ok"] and r["arms"][t]["quota_ok"]
                for r in rows for t in ("Fu", "Uu", "Ku"))
    mirror = all(r["mirror_ok"] for r in rows)
    v["d_identity"] = {"ok": bool(ident and mirror), "ident": bool(ident), "mirror": bool(mirror),
                       "qbar": {c: round(float(np.mean([r["qbar"] for r in by[c]])), 3) for c in CELL_KEY},
                       "qmax": {c: max(r["qmax"] for r in by[c]) for c in CELL_KEY}}

    # (f) 天花板/地板
    lv = {c: {a: round(float(np.mean([r["arms"][a]["ho8_mean"] for r in by[c]])), 4) for a in ARMS6} for c in CELL_KEY}
    bad_cells = [c for c in CELL_KEY if max(lv[c].values()) >= CEIL or lv[c]["A"] <= FLOOR]
    v["f_headroom"] = {"ok": not bad_cells, "bad_cells": bad_cells, "levels": lv,
                       "guard": {"ceil": CEIL, "floor": FLOOR}}

    # 前置条件
    anchor_cell = "35|35"
    pos = int((Rmu[anchor_cell] < 0).sum())
    neg = int((Rmu[anchor_cell] > 0).sum())
    same = max(pos, neg)
    v["pre"] = {"ok": same >= SIGN_NEED and abs(M[anchor_cell]) >= THR_ABS,
                "same_sign": f"{same}/{ns[anchor_cell]}（负 {pos} / 正 {neg}）",
                "abs_R_anchor": round(abs(M[anchor_cell]), 4), "thr_abs": THR_ABS}

    v["wr_side"] = {"delta_R_mu": round(dM_wr, 4), "cells": ["20|20", "28|20", "35|20"],
                    "trend": [round(M[c], 4) for c in ("20|20", "28|20", "35|20")]}
    v["rd_side"] = {"delta_R_mu": round(dM_rd, 4), "cells": ["35|20", "35|35"],
                    "levels": [round(M[c], 4) for c in ("35|20", "35|35")]}

    def chan(s):
        if not np.isfinite(s):
            return "nan"
        return "WRITE" if s >= SHARE_WR else ("READ" if s <= SHARE_RD else "MIXED")

    v["share"] = {"margin": round(sM, 4), "acc": round(sA, 4), "channel_margin": chan(sM),
                  "channel_acc": chan(sA), "thr": {"write": SHARE_WR, "read": SHARE_RD},
                  "R_mu": {c: round(M[c], 4) for c in CELL_KEY},
                  "R_acc": {c: round(A[c], 4) for c in CELL_KEY},
                  "per_seed_R_mu": {c: [round(float(x), 4) for x in Rmu[c]] for c in CELL_KEY}}
    v["e_secondary"] = {"ok": chan(sM) == chan(sA), "margin": v["share"]["channel_margin"],
                        "acc": v["share"]["channel_acc"]}

    main = None
    if not v["d_identity"]["ok"] or not v["c_anchor"]["ok"] or not v["f_headroom"]["ok"] or not v["pre"]["ok"]:
        main = "INDETERMINATE"
    elif not v["e_secondary"]["ok"]:
        main = "WEAK-INCONSISTENT"
    elif chan(sM) == "WRITE":
        main = "CONFIRMED"
    elif chan(sM) == "READ":
        main = "FALSIFIED"
    else:
        main = "WEAK-MIXED"
    v["main"] = {"ok": main == "CONFIRMED", "verdict": main}

    # 先算后印（DEF-26H-4）
    printed = [("c_anchor", v["c_anchor"]), ("d_identity", v["d_identity"]), ("pre", v["pre"]),
               ("f_headroom", v["f_headroom"]), ("e_secondary", v["e_secondary"])]
    for k, blk in printed:
        tag = blk.get("ok", not blk.get("bad_cells", []))
        print(f"  [{'PASS' if tag else 'FAIL'}] {k}")
    print(f"  R_mu: " + " | ".join(f"{c} {M[c]:+.4f}" for c in CELL_KEY))
    print(f"  R_acc: " + " | ".join(f"{c} {A[c]:+.4f}" for c in CELL_KEY))
    print(f"  δ_wr = {dM_wr:+.4f}（写侧轴 {v['wr_side']['trend']}）、δ_rd = {dM_rd:+.4f}"
          f"（读侧 {v['rd_side']['levels']}）")
    print(f"  share(mu) = {sM:.4f} → {chan(sM)}；share(acc) = {sA:.4f} → {chan(sA)}"
          f"（阈值 {SHARE_RD}/{SHARE_WR}）")
    print(f"  ho8 电平：" + " | ".join(f"{c} A {lv[c]['A']:.3f} U0 {lv[c]['Uu']:.3f}" for c in CELL_KEY))
    print(f"  ⇒ H26I 判定：**{main}**")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26Ir10")
    args = ap.parse_args()
    t0 = time.time()
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26i/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    if args.stage == "stageA":
        stage_a()
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}"
              f"（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"], res["passed"] = "stageA", ok
    else:
        seeds = [SEEDS[0]] if args.stage == "smoke" else SEEDS
        taus = {n: cal_at(n)["c_tau"] for n in sorted({c["n_kv"] for c in CELLS})}
        res["calib"] = [{"n_kv": n, "c_tau": t} for n, t in taus.items()]
        plan = [{"n_kv": 35, "n_read": 35}, {"n_kv": 20, "n_read": 20}] if args.stage == "smoke" else CELLS
        rows = []
        print(f"=== {args.stage}（{len(plan)} 格点 × {len(seeds)} 种子 × 6 臂） ===")
        for cell in plan:
            for s in seeds:
                g = run_unit(cell, s, taus[cell["n_kv"]])
                rows.append(g)
                json.dump({"rows": rows}, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
                a = g["arms"]
                print(f"  ({cell['n_kv']:>2}|{cell['n_read']:>2}) seed={s} | "
                      + " ".join(f"{t} {a[t]['ho8_mean']:.3f}/{a[t]['mu']:+.4f}" for t in ARMS6)
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
                   "cells": [f"{c['n_kv']}|{c['n_read']}" for c in CELLS], "r0": R0, "eta": ETA, "m": M,
                   "s_key": S_KEY, "r_pres": R_PRES, "r_ho": [R_HO4, R_HO8], "rho_main": R_MAIN,
                   "seeds": SEEDS, "profile": {"F": PROFILE["F"], "U0": PROFILE["U"], "K": PROFILE["K"]},
                   "stage": args.stage, "card": "DIR-26I rev1.0",
                   "thresholds": {"share_write": SHARE_WR, "share_read": SHARE_RD,
                                  "thr_abs_R": THR_ABS, "sign_need": SIGN_NEED, "ceil": CEIL, "floor": FLOOR}}
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
