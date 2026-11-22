r"""DIR-26-H · R1_gating8.py —— 写入剖面的效应：条目内顺序（甲）还是条目间干扰（乙）？载荷 $N$ 轴

卡片：`revolve/DIR-26H/DIRECTION_rev26h_v11.md`（rev 1.1 冻结；rev 1.0 留档）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，零改动）
★ 装置层/测量层/定标/抖动指标/轮内位置规则，以及本卡复用的四个构造（`round_quota`/`build_keep`/
  `realized_profile`/`even_counts`）与剖面模板，全部 import 同一对象（`dir26g/R1_gating7.py`）。

唯一独立变量 = **载荷 $N$**（共享秩 8 支撑里的条目数）。$\rho{=}0.4$ 的 $c_\tau$ 标定使
$k_B=0.6T\propto N$ ⇒ $\bar q=k_B/N=9.6$ 与 $N$ 无关（A26H-7），故 $N$ 只改干扰、不改条目内结构。
判据量：$\Delta(N)=\overline{F-K}$（主口径 `ho8`+mean），比率 $R_N=\lvert\Delta(N)\rvert/\lvert\Delta(35)\rvert$。
"""
import argparse, hashlib, json, os, sys, time

import numpy as np
import torch

for _p in ("dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating2 as G2
import R1_gating3 as G3
import R1_gating5 as G5
import R1_gating6 as G6
import R1_gating7 as G7
from R1_gating import (D_MODEL, N_DIST, ETA, M, S_KEY, stream_idx,
                       build_stream, drive, make_core, metrics_plas)
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
N_GRID = (10, 20, 35, 45, 60)
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
D39 = "/home/ec2-user/sgmn-loop/dir26g/out/DIR26Gr11_formal.json"

THR_R10, THR_R20, THR_R60 = 0.65, 0.90, 1.15      # (a) 乙：小载荷衰减 + 未见过的大载荷放大
BAND = (0.75, 1.33)                                # (b) 甲：载荷无关带
THR_D35 = 0.05                                     # (a) 前置：锚点上效应必须存在
QBAR_BAND = (9.0, 12.0)      # rev 1.1（DEF-26H-1）：由 3 点阈值改为覆盖 5 个格点的带
QBAR_SPREAD = 0.8
CEIL = 0.95
ARMS6 = ("A", "B", "P", "Fu", "Uu", "Ku")


def cal_at(n, r0=R0):
    r"""定标必须在**同一个 $n$** 上取分位数（`calibrate_w` 读模块全局 `N_DIST`）；调用后**复位**（A26H-3）。"""
    G5.N_DIST = int(n)
    try:
        return calibrate_w(r0, R_MAIN, R_PRES, theta_of(0.9))
    finally:
        G5.N_DIST = N_DIST


def read_streams(seed, n):
    th = theta_of(0.9)
    return {"ho4": build_stream(seed + HO_OFFSET, D_MODEL, n, R_HO4, th),
            "ho8": build_stream(seed + HO_OFFSET, D_MODEL, n, R_HO8, th)}


def run_unit(cell, seed, tau, RS):
    n = int(cell["n"])
    D = build_stream(seed, D_MODEL, n, R_PRES, theta_of(0.9))
    out = {"n": n, "seed": seed, "T": R_PRES * n * 2 * M, "arms": {}, "prof": {}, "k": {}, "qbar": 0.0}

    def rec(tag, core, issued):
        W = core._W().detach()
        W_before = W.clone()
        P = W - core.W0
        m4 = ret_jitter(P, RS["ho4"]["K"], D["vb"], n, R_HO4)
        m8 = ret_jitter(P, RS["ho8"]["K"], D["vb"], n, R_HO8)
        out["arms"][tag] = {"issued": issued, "clean": metrics_plas(W, core.W0, D["proto"], D["vb"], n)["ret1"],
                            "ho4_mean": m4["ret1_mean"], "ho4_maj": m4["ret1_maj"],
                            "ho8_mean": m8["ret1_mean"], "ho8_maj": m8["ret1_maj"],
                            "maj_need4": m4["maj_need"], "maj_need8": m8["maj_need"],
                            "W_unchanged": bool(torch.equal(core._W().detach(), W_before))}

    c = make_core(D_MODEL, seed, R0); kA, _ = drive(c, D, "A"); rec("A", c, kA)
    c = make_core(D_MODEL, seed, R0); kB, logB = drive(c, D, "B", tau=tau); rec("B", c, kB)
    cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, n)
    out["k"] = {"A": kA, "B": kB}
    out["qbar"] = kB / n
    out["qmax"] = int(max(cnt))

    c = make_core(D_MODEL, seed, R0)
    kP, _ = drive(c, D, "R", keep=keep_positions(n, R_PRES, kB)); rec("P", c, kP)

    keeps = {}
    for pk in ("K", "U", "F"):
        keep, per_item = build_keep(n, R_PRES, cnt, PROFILE[pk])
        keeps[pk] = keep
        tag = f"{pk}u"
        c = make_core(D_MODEL, seed, R0); kk, _ = drive(c, D, "R", keep=keep); rec(tag, c, kk)
        got = item_counts(keep, n)
        out["prof"][tag] = realized_profile(keep, n, R_PRES)
        out["arms"][tag]["sum_ok"] = (kk == sum(got) == kB)
        out["arms"][tag]["quota_ok"] = all(sum(per_item[i]) == cnt[i] for i in range(n))
    out["mirror_ok"] = (len(keeps["F"]) == len(keeps["K"])
                        and all(abs(out["prof"]["Fu"][r] - out["prof"]["Ku"][R_PRES - 1 - r]) < 1e-12 for r in range(R_PRES)))
    out["counts"] = cnt
    return out


def d39_rows():
    return {(g["r0"], g["seed"]): g for g in json.load(open(D39))["rows"] if g["r0"] == R0}


# ══════════════════════════════════════════════ Stage A
def stage_a():
    print(f"=== Stage A（A26H-1 … A26H-12） ===")
    chk("A26H-1", md5("/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py") == DEV_MD5,
        f"装置 md5 == {DEV_MD5}")

    want = {"R1_gating": ("build_stream", "drive", "make_core", "metrics_plas", "stream_idx"),
            "R1_gating2": ("keep_positions", "item_counts"),
            "R1_gating3": ("ret_jitter", "ret_jitter_bruteforce"),
            "R1_gating5": ("calibrate_w", "theta_of"),
            "R1_gating6": ("_pos_idxs",),
            "R1_gating7": ("round_quota", "build_keep", "realized_profile", "even_counts")}
    bad = [f"{n}:{f}" for n, fs in want.items() for f in fs
           if getattr(sys.modules[n], f, None) is not getattr(G7, f, getattr(sys.modules[n], f, None)) and n != "R1_gating7"]
    bad += [f"R1_gating7:{f}" for f in want["R1_gating7"] if getattr(G7, f, None) is None]
    chk("A26H-2", not bad, f"12 个复用对象的来源模块逐项核对：{want}" if not bad else f"不同源 {bad}")

    taus = {n: cal_at(n) for n in N_GRID}
    chk("A26H-3", G5.N_DIST == N_DIST,
        f"标定后模块全局复位：R1_gating5.N_DIST == 35 ? {G5.N_DIST == N_DIST}；"
        f"c_tau 逐 N = {[(n, round(taus[n]['c_tau'], 4)) for n in N_GRID]}")

    # A26H-8 定标
    ok8 = all(abs(t["skip_rate"] - R_MAIN) <= 0.05 and t["gap"] >= 0.05 and t["skip_frac_repeat"] >= 0.85 and t["gate_ok"]
              for t in taus.values())
    chk("A26H-8", ok8, "五个 N 上 skip_rate=0.400±0.05、gap≥0.05、重复占比≥0.85、gate_ok："
        + " ".join(f"N{n}:{t['skip_rate']:.3f}/{t['gap']:+.3f}/{t['skip_frac_repeat']:.3f}" for n, t in taus.items()))

    # A26H-4/5/7：构造恒等 + 镜像 + 载荷无关性（两个 N 抽样，逐位）
    s0 = SEEDS[0]
    qbars, det = [], []
    for n in (10, 60):
        D = build_stream(s0, D_MODEL, n, R_PRES, theta_of(0.9))
        c = make_core(D_MODEL, s0, R0); kB, log = drive(c, D, "B", tau=taus[n]["c_tau"])
        cnt = item_counts({t for t, e in enumerate(log) if e["issued"]}, n)
        qbars.append(kB / n)
        per = {}
        for pk in ("F", "K"):
            keep, pi = build_keep(n, R_PRES, cnt, PROFILE[pk])
            kk, _ = drive(make_core(D_MODEL, s0, R0), D, "R", keep=keep)
            per[pk] = (len(keep), kk, realized_profile(keep, n, R_PRES), all(sum(pi[i]) == cnt[i] for i in range(n)))
        det.append((n, kB, sum(cnt), per))
        chk(f"A26H-4.N{n}", per["F"][1] == per["K"][1] == kB == sum(cnt) and per["F"][3] and per["K"][3],
            f"N={n} seed={s0}: Σ发放 {per['F'][1]}/{per['K'][1]} == k_B {kB} == Σq_i {sum(cnt)}；逐条目配额逐位 True")
        mir = (per["F"][0] == per["K"][0]
               and all(abs(per["F"][2][r] - per["K"][2][R_PRES - 1 - r]) < 1e-12 for r in range(R_PRES)))
        chk(f"A26H-5.N{n}", mir, f"N={n}: |keep_F| {per['F'][0]} == |keep_K| {per['K'][0]}；剖面 F{[round(x, 2) for x in per['F'][2]]} "
                                f"K{[round(x, 2) for x in per['K'][2]]} 严格镜像")
        qbars[-1] = kB / n
    # rev 1.1（DEF-26H-1）：护栏改为"带"（覆盖全部 5 个格点），极差降为**报告项**
    chk("A26H-7", all(QBAR_BAND[0] <= q <= QBAR_BAND[1] for q in qbars),
        f"抽样 N=10/60 的 q̄ = {[round(q, 3) for q in qbars]} ∈ [{QBAR_BAND[0]},{QBAR_BAND[1]}]；"
        f"N=35 的 q̄ = 372/35 = {372 / 35:.3f}（D-39 实测 k_B=372）；★报告：极差 = {max(qbars) - min(qbars):.3f}")

    # A26H-6 锚点逐位（种子 1、N=35）
    dr = d39_rows()
    g = run_unit({"n": 35}, s0, taus[35]["c_tau"], read_streams(s0, 35))
    ref = dr[(R0, s0)]["arms"]
    bads, ncmp = [], 0
    for a in ARMS6:
        for f in ("ho4_mean", "ho4_maj", "ho8_mean", "clean"):
            ncmp += 1
            if g["arms"][a][f] != ref[a][f]:
                bads.append(f"{a}.{f}:{g['arms'][a][f]}!={ref[a][f]}")
    chk("A26H-6", not bads, f"N=35 seed={s0}：六臂 × {{ho4_mean,ho4_maj,ho8_mean,clean}} = {ncmp} 个值对 D-39 逐位相等"
                          + ("" if not bads else f"；不符 {bads}"))

    # A26H-9 稳定性 + 零发放
    rho_stab = 1 - ETA * (1.0 * S_KEY + 0.05 / 2)
    core = make_core(D_MODEL, s0, R0)
    W0 = core._W().detach().clone()
    kN, _ = drive(core, D, "NONE")
    chk("A26H-9", rho_stab > 0 and kN == 0 and torch.equal(W0, core._W().detach()),
        f"ρ_stab={rho_stab:.4f}>0；零发放臂 0 次、‖ΔW‖=0")

    # A26H-10 抖动指标逐位 —— rev 1.1（DEF-26H-2）：必须**显式重建** N=35 的写流，并断言它的 n
    D35 = build_stream(s0, D_MODEL, 35, R_PRES, theta_of(0.9))
    assert int(D35["n"]) == 35, f"跨装置流混用：D35[\"n\"]={D35['n']}"
    RS = read_streams(s0, 35)
    cK = make_core(D_MODEL, s0, R0)
    keepK, _ = build_keep(35, R_PRES, g["counts"], PROFILE["K"])
    drive(cK, D35, "R", keep=keepK)
    PK = cK._W().detach() - cK.W0
    P = PK                      # 只读投影，供两口径共用
    a4 = ret_jitter(PK, RS["ho4"]["K"], D35["vb"], 35, R_HO4); b4 = ret_jitter_bruteforce(PK, RS["ho4"]["K"], D35["vb"], 35, R_HO4)
    a8 = ret_jitter(PK, RS["ho8"]["K"], D35["vb"], 35, R_HO8); b8 = ret_jitter_bruteforce(PK, RS["ho8"]["K"], D35["vb"], 35, R_HO8)
    chk("A26H-10", a4["ret1_mean"] == b4["ret1_mean"] and a8["ret1_mean"] == b8["ret1_mean"]
        and a4["maj_need"] == 3 and a8["maj_need"] == 5,
        f"r_q=4→mean {a4['ret1_mean']:.4f}/maj {a4['ret1_maj']:.4f}(need 3)；r_q=8→mean {a8['ret1_mean']:.4f}/maj {a8['ret1_maj']:.4f}(need 5)")

    # A26H-11 读后不变
    chk("A26H-11", all(v["W_unchanged"] for v in g["arms"].values()),
        f"N=35 seed={s0} 六臂读后 W 逐位不变 = {all(v['W_unchanged'] for v in g['arms'].values())}")

    # A26H-12 N<=d
    okd = True
    for n in N_GRID:
        try:
            Dn = build_stream(s0, D_MODEL, n, R_PRES, theta_of(0.9))
            okd = okd and Dn["K"].shape == (R_PRES, n, D_MODEL) and len(item_counts(set(), n)) == n
        except Exception as e:
            okd = False
            print(f"    N={n} 异常：{type(e).__name__}: {e}")
    chk("A26H-12", okd and max(N_GRID) <= D_MODEL, f"五个 N 均成功建流且 N≤d={D_MODEL}；N=60 的 item_counts 长度 60")

    res["calib"] = [{"n": n, "c_tau": t["c_tau"], "k_B": None, "qbar": None} for n, t in taus.items()]


# ══════════════════════════════════════════════ 判据
def crit(rows):
    v = {}
    by = {n: [g for g in rows if g["n"] == n] for n in N_GRID}
    ns = {n: len(by[n]) for n in N_GRID}

    def mean(n, key, arm):
        return float(np.mean([g["arms"][arm][key] for g in by[n]]))

    Dp = {n: mean(n, "ho8_mean", "Fu") - mean(n, "ho8_mean", "Ku") for n in N_GRID}   # 主口径
    Ds = {n: mean(n, "ho4_mean", "Fu") - mean(n, "ho4_mean", "Ku") for n in N_GRID}   # 副口径
    per = {n: [g["arms"]["Fu"]["ho8_mean"] - g["arms"]["Ku"]["ho8_mean"] for g in by[n]] for n in N_GRID}
    per_s = {n: [g["arms"]["Fu"]["ho4_mean"] - g["arms"]["Ku"]["ho4_mean"] for g in by[n]] for n in N_GRID}

    # (d) 构造恒等
    ident = all(v2.get("sum_ok") and v2.get("quota_ok")
                for g in rows for t, v2 in g["arms"].items() if t in ("Fu", "Uu", "Ku"))
    mirror = all(g["mirror_ok"] for g in rows)
    qbar = {n: float(np.mean([g["qbar"] for g in by[n]])) for n in N_GRID}
    # rev 1.1 §2(d)：只要求**带**；跨 N 极差是**报告项**（DEF-26H-3：首版这里残留了旧阈值）
    d_ok = bool(ident and mirror and all(QBAR_BAND[0] <= q <= QBAR_BAND[1] for q in qbar.values()))
    v["d_identity"] = {"ok": d_ok, "ident": bool(ident), "mirror": bool(mirror),
                       "qbar": {str(n): round(qbar[n], 3) for n in N_GRID},
                       "qbar_spread": round(max(qbar.values()) - min(qbar.values()), 4)}

    # (c) 锚点
    dr = d39_rows()
    ncmp, bads = 0, []
    for g in by[35]:
        ref = dr[(R0, g["seed"])]["arms"]
        for a in ARMS6:
            for f in ("ho4_mean", "ho4_maj", "ho8_mean", "clean"):
                ncmp += 1
                if g["arms"][a][f] != ref[a][f]:
                    bads.append(f"{g['seed']}:{a}.{f}")
    v["c_anchor"] = {"ok": not bads, "n_cmp": ncmp, "bad": bads[:8], "n_bad": len(bads)}

    # (f) 天花板
    ceil_cells = {str(n): round(max(mean(n, "ho8_mean", a) for a in ARMS6), 4) for n in N_GRID
                  if max(mean(n, "ho8_mean", a) for a in ARMS6) >= CEIL}
    v["f_ceiling"] = {"cells": ceil_cells, "n": len(ceil_cells), "hit": len(ceil_cells) >= 2}

    # (a)(b) 比率
    a35 = abs(Dp[35])
    R = {n: (abs(Dp[n]) / a35 if a35 > 0 else float("inf")) for n in N_GRID}
    pos60 = sum(1 for x in per[60] if x < 0)
    pos35 = sum(1 for x in per[35] if x < 0)
    a_ok = (a35 >= THR_D35 and R[10] <= THR_R10 and R[20] <= THR_R20 and R[60] >= THR_R60
            and np.sign(Dp[60]) == np.sign(Dp[35]) and Dp[35] < 0 and pos60 >= 4 and pos35 >= 4)
    b_ok = all(BAND[0] <= R[n] <= BAND[1] for n in (10, 20, 60))
    v["a_main"] = {"ok": bool(a_ok), "R": {str(n): round(R[n], 4) for n in N_GRID},
                   "thr": {"R10": THR_R10, "R20": THR_R20, "R60": THR_R60, "D35": THR_D35},
                   "pos60": f"{pos60}/{ns[60]}", "pos35": f"{pos35}/{ns[35]}"}
    v["b_alt"] = {"ok": bool(b_ok), "band": list(BAND), "R": {str(n): round(R[n], 4) for n in N_GRID}}

    # (e) 副口径一致
    e_detail = [{"n": n, "ho8": round(Dp[n], 4), "ho4": round(Ds[n], 4),
                 "same_sign": bool(np.sign(Dp[n]) == np.sign(Ds[n]))} for n in N_GRID]
    v["e_maj"] = {"ok": all(x["same_sign"] for x in e_detail), "detail": e_detail}

    v["report"] = {
        "D_ho8": {str(n): round(Dp[n], 4) for n in N_GRID},
        "D_ho4": {str(n): round(Ds[n], 4) for n in N_GRID},
        "per_seed_ho8": {str(n): [round(x, 4) for x in per[n]] for n in N_GRID},
        "sd_ho8": {str(n): round(float(np.std(per[n], ddof=1)), 4) for n in N_GRID},
        "levels": {str(n): {a: round(mean(n, "ho8_mean", a), 4) for a in ARMS6} for n in N_GRID},
        "dBP_ho8": {str(n): round(mean(n, "ho8_mean", "B") - mean(n, "ho8_mean", "P"), 4) for n in N_GRID},
        "dFU": {str(n): round(mean(n, "ho8_mean", "Fu") - mean(n, "ho8_mean", "Uu"), 4) for n in N_GRID},
        "dUK": {str(n): round(mean(n, "ho8_mean", "Uu") - mean(n, "ho8_mean", "Ku"), 4) for n in N_GRID},
        "clean": {str(n): {a: round(mean(n, "clean", a), 4) for a in ARMS6} for n in N_GRID},
        "qmax": {str(n): max(g["qmax"] for g in by[n]) for n in N_GRID},
    }

    if (not d_ok) or (not v["c_anchor"]["ok"]) or v["f_ceiling"]["hit"]:
        main = "INDETERMINATE"
    elif b_ok:
        main = "DIRECTION-FALSIFIED"
    elif a_ok and v["e_maj"]["ok"]:
        main = "CONFIRMED"
    else:
        main = "WEAK"
    v["main"] = {"ok": main == "CONFIRMED", "verdict": main}

    for k in ("d_identity", "c_anchor", "a_main", "b_alt", "e_maj", "f_ceiling"):
        tag = v[k].get("ok", not v[k].get("hit", False))
        print(f"  [{'PASS' if tag else 'FAIL'}] {k}" + ("" if k != "f_ceiling" else f" (hit={v[k]['hit']})"))
    for n in N_GRID:
        print(f"    N={n:<3d} F {mean(n, 'ho8_mean', 'Fu'):.4f} U0 {mean(n, 'ho8_mean', 'Uu'):.4f} "
              f"K {mean(n, 'ho8_mean', 'Ku'):.4f} | Δ(F−K) {Dp[n]:+.4f} (±{np.std(per[n], ddof=1):.4f}, "
              f"{sum(1 for x in per[n] if x > 0)}/{ns[n]} 正) | R {R[n]:.3f} | Δho4 {Ds[n]:+.4f} | "
              f"q̄ {qbar[n]:.2f} q_max {max(g['qmax'] for g in by[n])}")
    print(f"  ⇒ H26H 判定：**{main}**")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26Hr10")
    args = ap.parse_args()
    t0 = time.time()
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26h/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    if args.stage == "stageA":
        stage_a()
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}"
              f"（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"], res["passed"] = "stageA", ok
    else:
        seeds = [SEEDS[0]] if args.stage == "smoke" else SEEDS
        taus = {n: cal_at(n)["c_tau"] for n in N_GRID}
        res["calib"] = [{"n": n, "c_tau": t} for n, t in taus.items()]
        plan = [{"n": 10}, {"n": 35}] if args.stage == "smoke" else [{"n": n} for n in N_GRID]
        rows = []
        print(f"=== {args.stage}（{len(plan)} 个载荷 × {len(seeds)} 种子 × 6 臂） ===")
        for cell in plan:
            for s in seeds:
                RS = read_streams(s, cell["n"])
                g = run_unit(cell, s, taus[cell["n"]], RS)
                rows.append(g)
                json.dump({"rows": rows}, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
                a = g["arms"]
                print(f"  N={cell['n']:<3d} seed={s} | " + " ".join(f"{t} {a[t]['ho8_mean']:.3f}" for t in ARMS6)
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
                   "n_grid": list(N_GRID), "r0": R0, "eta": ETA, "m": M, "s_key": S_KEY, "r_pres": R_PRES,
                   "r_ho": [R_HO4, R_HO8], "rho_main": R_MAIN, "seeds": SEEDS,
                   "profile": {"F": PROFILE["F"], "U0": PROFILE["U"], "K": PROFILE["K"]},
                   "stage": args.stage, "card": "DIR-26H rev1.1"}
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
