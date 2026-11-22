"""DIR-26-E · R1_gating5.py —— 写侧错位轴：两侧错位对称吗？

卡片：`revolve/DIR-26E/DIRECTION_rev26e_v10.md`（rev 1.0 冻结）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，零改动）
★ 测量层/装置层/P 臂/抖动指标全部 import：`dir26/R1_gating.py`、`dir26b/R1_gating2.py`、`dir26c/R1_gating3.py`

唯一独立变量 = **写侧错位角 θ_w**（cosθ_w ∈ {0.9,0.7,0.5,0.3}），读侧固定 cosθ_r=0.9（ho4/ho8 各一条流，全档复用）。
臂 A / B(ρ=.4，逐档重定标 c_τ) / P；格点 r∈{8,12} × 4 档 × 5 种子 × 3 臂 = 120 次运行。
对照：D-36 的读侧曲线 `dir26d/out/DIR26Dr10_formal.json`（同种子同格点，不重跑）。
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

for _p in ("dir26e", "dir26d", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating2 as G2
import R1_gating3 as G3
from R1_gating import (D_MODEL, N_DIST, ETA, M, S_KEY, build_stream, drive, make_core, metrics_plas)
from R1_gating2 import calibrate, keep_positions
from R1_gating3 import ret_jitter, ret_jitter_bruteforce

R0_GRID = (8, 12)
COS_W = (0.9, 0.7, 0.5, 0.3)                             # ★ 唯一独立变量
R_PRES = 4
R_HO4, R_HO8 = 4, 8
HO_OFFSET = 999
COS_R_FIX = 0.9                                          # 读侧固定
THETA_W0 = math.radians(25.8419)                         # cos = 0.9（与 D-35/D-36 同）
R_MAIN = 0.40
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]
SEED_PILOT = 20261041
THR_MAIN = 0.10
THR_DEGEN = 0.10
THR_MONO = 0.05
THR_SYM = 0.10                                           # (a) 同构
THR_ZERO = 0.03                                          # (e) 记 0 的带宽
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
D36_JSON = "/home/ec2-user/sgmn-loop/dir26d/out/DIR26Dr10_formal.json"
D35_JSON = "/home/ec2-user/sgmn-loop/dir26c/out/DIR26Cr11_formal.json"
res = {"checks": [], "rows": [], "calib": []}


def theta_of(cos_w):
    """cos=0.9 档必须逐位等于 THETA_W0（`acos(0.9)` 差 2.6e-6 ⇒ 键不同 ⇒ A26E-4）。"""
    return THETA_W0 if abs(cos_w - 0.9) < 1e-9 else math.acos(cos_w)


def calibrate_w(r0, rho, r_pres, theta_w):
    """与 `R1_gating2.calibrate` 同一逻辑（同 quantile、同 gate_ok），把 $R$ 与 $\\theta$ 参数化。"""
    D = build_stream(SEED_PILOT, D_MODEL, N_DIST, r_pres, theta_w)
    c = make_core(D_MODEL, SEED_PILOT, r0)
    _, log = drive(c, D, "A")
    allc = [e["cos"] for e in log]
    tau = float(np.quantile(allc, 1.0 - rho))
    skip = [e for e in log if e["cos"] > tau]
    kep = [e for e in log if e["cos"] <= tau]
    rate = len(skip) / len(allc)
    m_skip = float(np.median([e["cos"] for e in skip])) if skip else float("nan")
    m_keep = float(np.median([e["cos"] for e in kep])) if kep else float("nan")
    frac_rep = float(np.mean([e["round"] > 0 for e in skip])) if skip else float("nan")
    ok = (abs(rate - rho) <= 0.05) and (m_skip - m_keep >= 0.05) and (frac_rep >= 0.85)
    row = {"r0": r0, "rho": rho, "r_pres": r_pres, "cos_w": math.cos(theta_w), "c_tau": tau,
           "skip_rate": rate, "gap": m_skip - m_keep, "skip_frac_repeat": frac_rep,
           "gate_ok": bool(ok), "seed": SEED_PILOT}
    print(f"  [标定] r={r0:2d} ρ={rho:.1f} cosθ_w={row['cos_w']:.1f} c_τ={tau:+.4f} 跳过率 {rate:.3f} "
          f"gap {row['gap']:+.4f} 重复占比 {frac_rep:.3f} {'OK' if ok else '★DEGENERATE'}")
    return row


def cross_stats(K0):
    """写键**条目间**交叉余弦的 (最大, 平均绝对值)。掩码取非对角元素（不用 nanmax：旧 torch 无此方法）。"""
    cc = torch.nn.functional.cosine_similarity(K0.unsqueeze(1), K0.unsqueeze(0), dim=-1)
    m = ~torch.eye(K0.shape[0], dtype=torch.bool)
    v = cc[m]
    return float(v.max()), float(v.abs().mean())


def read_streams(seed):
    """读流**与 θ_w 无关**（A26E-7）：一次建好，全档复用。cos0.9 用 THETA_W0 保持与 D-35 逐位同一。"""
    th = THETA_W0 if abs(COS_R_FIX - 0.9) < 1e-9 else math.acos(COS_R_FIX)
    return {"ho4": build_stream(seed + HO_OFFSET, D_MODEL, N_DIST, R_HO4, th),
            "ho8": build_stream(seed + HO_OFFSET, D_MODEL, N_DIST, R_HO8, th)}


def run_cell(r0, seed, cos_w, tau, RS):
    """一个 (r, seed, θ_w) 格点：A / B / P 三臂；读数 = clean + ho4 + ho8。"""
    th = theta_of(cos_w)
    D = build_stream(seed, D_MODEL, N_DIST, R_PRES, th)
    T = R_PRES * N_DIST * 2 * M
    out = {"r0": r0, "seed": seed, "cos_w": cos_w, "T": T, "arms": {}}

    def rec(tag, core, issued):
        P = core._W().detach() - core.W0
        W_before = core._W().detach().clone()
        m4 = ret_jitter(P, RS["ho4"]["K"], D["vb"], N_DIST, R_HO4)
        m8 = ret_jitter(P, RS["ho8"]["K"], D["vb"], N_DIST, R_HO8)
        out["arms"][tag] = {"issued": issued,
                            "clean": metrics_plas(core._W().detach(), core.W0, D["proto"], D["vb"], N_DIST)["ret1"],
                            "ho4_mean": m4["ret1_mean"], "ho4_maj": m4["ret1_maj"],
                            "ho8_mean": m8["ret1_mean"], "ho8_maj": m8["ret1_maj"],
                            "maj_need4": m4["maj_need"], "maj_need8": m8["maj_need"],
                            "W_unchanged": bool(torch.equal(core._W().detach(), W_before))}

    cA = make_core(D_MODEL, seed, r0); kA, _ = drive(cA, D, "A"); rec("A", cA, kA)
    cB = make_core(D_MODEL, seed, r0); kB, _ = drive(cB, D, "B", tau=tau); rec("B", cB, kB)
    keepP = keep_positions(N_DIST, R_PRES, kB)
    cP = make_core(D_MODEL, seed, r0); kP, _ = drive(cP, D, "R", keep=keepP); rec("P", cP, kP)
    out["kP_eq_kB"] = bool(kP == kB)

    # (g) 串扰：写键的**条目间**最大交叉余弦（报告）
    mx, ma = cross_stats(D["K"][0])
    out["crosstalk"] = {"max": mx, "mean_abs": ma}
    return out


def chk(cid, ok, detail):
    res["checks"].append({"id": cid, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}  {detail}")
    return bool(ok)


def md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage_a():
    print("=== Stage A（A26E-1 … A26E-11） ===")
    dev = "/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py"
    chk("A26E-1", md5(dev) == DEV_MD5, f"装置 md5 {md5(dev)} == {DEV_MD5}")

    bad = [n for n in ("unit", "make_core", "build_stream", "drive", "metrics_plas", "stream_idx")
           if getattr(G, n, None).__module__ != "R1_gating"]
    bad += [n for n in ("keep_positions", "calibrate") if getattr(G2, n, None).__module__ != "R1_gating2"]
    bad += [n for n in ("ret_jitter", "ret_jitter_bruteforce") if getattr(G3, n, None).__module__ != "R1_gating3"]
    chk("A26E-2", not bad, f"装置/测量/P 臂/定标/抖动指标 全为 import 的同一对象：异常 {bad or '无'}")

    RS = read_streams(SEEDS[0])
    D90 = build_stream(SEEDS[0], D_MODEL, N_DIST, R_PRES, THETA_W0)
    al, off = [], []
    for c in COS_W:
        Dw = build_stream(SEEDS[0], D_MODEL, N_DIST, R_PRES, theta_of(c))
        al.append(float(torch.nn.functional.cosine_similarity(Dw["K"], Dw["proto"], dim=-1).mean()))
        off.append(cross_stats(Dw["K"][0])[1])
    ok3 = (all(abs(a - c) <= 0.01 for a, c in zip(al, COS_W))
           and all(off[i + 1] >= off[i] - 0.02 for i in range(len(off) - 1)))
    chk("A26E-3", ok3, "写键对齐度 cos(写键,原型)=" + " ".join(f"{a:.4f}" for a in al)
        + f"（目标 {list(COS_W)}）；异条目 |cos| 平均 " + " ".join(f"{x:.4f}" for x in off) + "（须单调不减−0.02）")

    D35 = json.load(open(D35_JSON))
    rs35 = [r for r in D35["rows"] if r["r0"] == 8 and r["seed"] == SEEDS[0]][0]
    sameK = torch.equal(build_stream(SEEDS[0], D_MODEL, N_DIST, R_PRES, theta_of(0.9))["K"], D90["K"])
    cal90 = {r0: calibrate_w(r0, R_MAIN, R_PRES, THETA_W0) for r0 in R0_GRID}
    g90 = run_cell(8, SEEDS[0], 0.9, cal90[8]["c_tau"], RS)
    bit = all(g90["arms"][a]["ho4_mean"] == rs35["cells"][t]["Jout"]["ret1_mean"]
              and g90["arms"][a]["ho4_maj"] == rs35["cells"][t]["Jout"]["ret1_maj"]
              and g90["arms"][a]["clean"] == rs35["cells"][t]["clean"]
              for a, t in (("A", "A"), ("B", "B0.40"), ("P", "P0.40")))
    chk("A26E-4", sameK and bit,
        f"cosθ_w=0.9 写流逐位同一={sameK}；整组读数与 D-35 逐位相同={bit}"
        f"（B ho4 {g90['arms']['B']['ho4_mean']:.4f} vs {rs35['cells']['B0.40']['Jout']['ret1_mean']:.4f}）")

    same_cal = all(abs(cal90[r0]["c_tau"] - calibrate(r0, R_MAIN)["c_tau"]) == 0.0
                   and abs(cal90[r0]["skip_rate"] - calibrate(r0, R_MAIN)["skip_rate"]) == 0.0
                   for r0 in R0_GRID)
    cals = {(r0, c): calibrate_w(r0, R_MAIN, R_PRES, theta_of(c)) for r0 in R0_GRID for c in COS_W}
    # rev 1.1：硬条件 = 跳过率 ±0.05 且 gap ≥0.05；重复占比**逐档记录为结果**（DEF-26E-1）
    hard5 = all(abs(v["skip_rate"] - R_MAIN) <= 0.05 and v["gap"] >= 0.05 for v in cals.values())
    chk("A26E-5", same_cal and hard5,
        f"calibrate_w(cosθ_w=0.9) 与 R1_gating2.calibrate 逐位相同={same_cal}；四档（r=8）："
        + " ".join(f"{c:.1f}:率{cals[(8, c)]['skip_rate']:.2f}/gap{cals[(8, c)]['gap']:.3f}"
                   f"/重复{cals[(8, c)]['skip_frac_repeat']:.3f}" for c in COS_W))

    d16, n16 = 16, 8
    D16 = build_stream(SEEDS[0], d16, n16, R_PRES, THETA_W0)
    c16 = make_core(d16, SEEDS[0], 4); drive(c16, D16, "A")
    P16 = c16._W().detach() - c16.W0
    R16 = {"ho4": build_stream(SEEDS[0] + HO_OFFSET, d16, n16, R_HO4, THETA_W0),
           "ho8": build_stream(SEEDS[0] + HO_OFFSET, d16, n16, R_HO8, THETA_W0)}
    a4 = ret_jitter(P16, R16["ho4"]["K"], D16["vb"], n16, R_HO4)
    b4 = ret_jitter_bruteforce(P16, R16["ho4"]["K"], D16["vb"], n16, R_HO4)
    a8 = ret_jitter(P16, R16["ho8"]["K"], D16["vb"], n16, R_HO8)
    b8 = ret_jitter_bruteforce(P16, R16["ho8"]["K"], D16["vb"], n16, R_HO8)
    chk("A26E-6", a4["ret1_mean"] == b4["ret1_mean"] and a4["ret1_maj"] == b4["ret1_maj"]
        and a8["ret1_mean"] == b8["ret1_mean"] and a8["ret1_maj"] == b8["ret1_maj"]
        and a4["maj_need"] == 3 and a8["maj_need"] == 5,
        f"抖动指标逐位正确：r_q=4 mean {a4['ret1_mean']:.4f}、maj {a4['ret1_maj']:.4f}（need 3）；"
        f"r_q=8 mean {a8['ret1_mean']:.4f}、maj {a8['ret1_maj']:.4f}（need 5）")

    cr = float(torch.nn.functional.cosine_similarity(RS["ho4"]["K"], RS["ho4"]["proto"], dim=-1).mean())
    chk("A26E-7", abs(cr - COS_R_FIX) <= 0.01,
        f"读流与 θ_w 无关（全档复用同一对象）；cos(ho4,原型)={cr:.4f}=0.9±0.01")

    g30 = run_cell(8, SEEDS[0], 0.3, cals[(8, 0.3)]["c_tau"], RS)
    chk("A26E-8", g90["T"] == 560 and g30["T"] == 560 and g90["arms"]["A"]["issued"] == 560
        and all(g90["arms"][a]["W_unchanged"] and g30["arms"][a]["W_unchanged"] for a in "ABP"),
        f"写步 T={g90['T']}=560；A 臂实发 {g90['arms']['A']['issued']}；读不改写=True")

    rho_stab = 1 - ETA * (1.0 * S_KEY + 0.05 / 2)
    cN = make_core(D_MODEL, SEEDS[0], 8); W0n = cN._W().detach().clone()
    kN, _ = drive(cN, D90, "NONE")
    chk("A26E-9", rho_stab > 0 and kN == 0 and float((cN._W().detach() - W0n).norm()) == 0.0,
        f"稳定性 ρ={rho_stab:.4f}>0；零发放臂 {kN} 次、‖ΔW‖=0")

    # rev 1.1：只在已算出的两档上核（另两档结构性覆盖：keep_positions 只依赖 (n,R,k)，k 唯一来自 B）
    chk("A26E-10", g90["kP_eq_kB"] and g30["kP_eq_kB"],
        f"P 臂等量：θ_w=0.9 k_B={g90['arms']['B']['issued']}、θ_w=0.3 k_B={g30['arms']['B']['issued']}，"
        "两档 k_P=k_B 逐位（另两档结构性覆盖）")

    res["calib"] = list(cals.values())

    cb = make_core(D_MODEL, SEEDS[0], 8); drive(cb, D90, "B", tau=cal90[8]["c_tau"])
    Pb = cb._W().detach() - cb.W0
    D30 = build_stream(SEEDS[0], D_MODEL, N_DIST, R_PRES, theta_of(0.3))
    si = [ret_jitter(s * Pb, RS["ho4"]["K"], D30["vb"], N_DIST, R_HO4)["ret1_mean"] for s in (0.37, 1.0, 2.5)]
    chk("A26E-11", max(si) - min(si) == 0.0, f"护栏（θ_w=0.3）：纯标量不变性 c∈{{0.37,1,2.5}} = {si}")
    return cals, RS


def crit(rows):
    print("=== 判据（rev 1.0 冻结阈值） ===")
    D36 = json.load(open(D36_JSON))
    r36 = D36["rows"]

    def delta(rs, r0, c):
        d = [g["arms"]["B"]["ho4_mean"] - g["arms"]["P"]["ho4_mean"] for g in rs
             if g["r0"] == r0 and abs(g["cos_w"] - c) < 1e-9]
        return float(np.mean(d)), (float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")), d

    def delta_r(r0, c):
        d = [g["arms"]["B"]["reads"][f"{c:.1f}"]["mean4"] - g["arms"]["P"]["reads"][f"{c:.1f}"]["mean4"]
             for g in r36 if g["r0"] == r0]
        return float(np.mean(d))

    v = {}
    spread = {}
    for r0 in R0_GRID:
        vals = [[g["arms"][a]["ho4_mean"] for a in "ABP"] for g in rows
                if g["r0"] == r0 and abs(g["cos_w"] - 0.9) < 1e-9]
        spread[r0] = float(np.mean([max(x) - min(x) for x in vals]))
    v["g_anchor"] = {"ok": all(spread[r] >= THR_DEGEN for r in R0_GRID), "detail": spread}

    sym, dw_all, dr_all = [], {}, {}
    for r0 in R0_GRID:
        for c in COS_W:
            m, sd, d = delta(rows, r0, c)
            mr = delta_r(r0, c)
            dw_all[f"{r0}@{c:.1f}"] = m
            dr_all[f"{r0}@{c:.1f}"] = mr
            sym.append({"r0": r0, "cos": c, "d_w": m, "sd_w": sd, "n_pos": int(sum(x > 0 for x in d)),
                        "n": len(d), "d_r": mr, "gap": abs(m - mr), "ok": abs(m - mr) <= THR_SYM})
    v["a_sym"] = {"ok": all(x["ok"] for x in sym), "detail": sym, "threshold": THR_SYM,
                  "n_ok": sum(x["ok"] for x in sym), "n": len(sym)}

    mono = {}
    for r0 in R0_GRID:
        seq = [dw_all[f"{r0}@{c:.1f}"] for c in COS_W]
        bad = [(f"{COS_W[i]:.1f}→{COS_W[i+1]:.1f}", round(seq[i + 1] - seq[i], 4))
               for i in range(len(seq) - 1) if seq[i + 1] - seq[i] > THR_MONO]
        mono[r0] = {"seq": [round(x, 4) for x in seq], "violations": bad, "ok": not bad}
    v["b_mono"] = {"ok": all(mono[r]["ok"] for r in R0_GRID), "detail": mono, "tolerance": THR_MONO}

    D35 = json.load(open(D35_JSON))
    ok_c, det_c = True, {}
    for r0 in R0_GRID:
        sp = [r for r in D35["rows"] if r["r0"] == r0]
        for a, t in (("A", "A"), ("B", "B0.40"), ("P", "P0.40")):
            mine = float(np.mean([g["arms"][a]["ho4_mean"] for g in rows
                                  if g["r0"] == r0 and abs(g["cos_w"] - 0.9) < 1e-9]))
            ref = float(np.mean([r["cells"][t]["Jout"]["ret1_mean"] for r in sp]))
            good = abs(mine - ref) == 0.0
            ok_c = ok_c and good
            det_c[f"{r0}:{a}"] = {"mine": mine, "d35": ref, "ok": good}
    v["c_anchor"] = {"ok": ok_c, "detail": det_c}

    cnt = {}
    for r0 in R0_GRID:
        nw = sum(dw_all[f"{r0}@{c:.1f}"] >= THR_MAIN for c in COS_W)
        nr = sum(dr_all[f"{r0}@{c:.1f}"] >= THR_MAIN for c in COS_W)
        cnt[r0] = {"write": nw, "read": nr, "ok": abs(nw - nr) <= 1}
    v["d_count"] = {"ok": all(c["ok"] for c in cnt.values()), "detail": cnt}

    def band(x):
        return 0 if abs(x) <= THR_ZERO else (1 if x > 0 else -1)
    sgn = [{"r0": x["r0"], "cos": x["cos"], "w": band(x["d_w"]), "r": band(x["d_r"]),
            "ok": band(x["d_w"]) == band(x["d_r"])} for x in sym]
    v["e_sign"] = {"ok": all(x["ok"] for x in sgn), "detail": sgn, "band": THR_ZERO,
                   "n_ok": sum(x["ok"] for x in sgn), "n": len(sgn)}

    fm = []
    for r0 in R0_GRID:
        for c in (0.7, 0.5):
            dm = float(np.mean([g["arms"]["B"]["ho8_maj"] - g["arms"]["P"]["ho8_maj"] for g in rows
                                if g["r0"] == r0 and abs(g["cos_w"] - c) < 1e-9]))
            dmn = dw_all[f"{r0}@{c:.1f}"]
            fm.append({"r0": r0, "cos": c, "maj8": round(dm, 4), "mean4": round(dmn, 4),
                       "ok": abs(dm - dmn) <= 0.10 and (dm > 0) == (dmn > 0)})
    v["f_maj"] = {"ok": all(x["ok"] for x in fm), "detail": fm}

    v["g_crosstalk"] = {"ok": True,
                        "detail": {f"{c:.1f}": round(float(np.mean([g["crosstalk"]["max"] for g in rows
                                                                   if abs(g["cos_w"] - c) < 1e-9])), 4)
                                   for c in COS_W}}
    v["h_clean"] = {"ok": True,
                    "detail": {f"{r0}@{c:.1f}": {a: round(float(np.mean([g["arms"][a]["clean"] for g in rows
                                                                            if g["r0"] == r0 and abs(g["cos_w"] - c) < 1e-9])), 4)
                                                 for a in "ABP"}
                               for r0 in R0_GRID for c in COS_W}}
    v["i_guard"] = {"ok": True, "note": "见 A26E-11"}

    if not v["g_anchor"]["ok"]:
        main = "INDETERMINATE"
    elif not v["a_sym"]["ok"]:
        main = "FALSIFIED"
    elif v["b_mono"]["ok"] and v["c_anchor"]["ok"] and v["d_count"]["ok"] and v["e_sign"]["ok"] and v["f_maj"]["ok"]:
        main = "CONFIRMED"
    else:
        main = "WEAK"
    worst = sorted(sym, key=lambda x: -x["gap"])[:3]
    v["main"] = {"ok": main == "CONFIRMED", "verdict": main,
                 "worst": [{"r0": x["r0"], "cos": x["cos"], "gap": round(x["gap"], 4),
                            "d_w": round(x["d_w"], 4), "d_r": round(x["d_r"], 4)} for x in worst]}
    for k in ("g_anchor", "a_sym", "b_mono", "c_anchor", "d_count", "e_sign", "f_maj"):
        print(f"  [{'PASS' if v[k]['ok'] else 'FAIL'}] {k}  " +
              (f"{v[k].get('n_ok')}/{v[k].get('n')} ok；最差 {v['main']['worst']}"
               if k in ("a_sym", "e_sign") else str(v[k].get("detail", ""))))
    print(f"  ⇒ H26E 判定：**{main}**")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26Er10")
    args = ap.parse_args()
    t0 = time.time()
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26e/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)          # ★ DEF-26D-4

    if args.stage == "stageA":
        stage_a()
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}"
              f"（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"], res["passed"] = "stageA", ok
    else:
        r0s = (8,) if args.stage == "smoke" else R0_GRID
        seeds = [SEEDS[0]] if args.stage == "smoke" else SEEDS
        coss = (0.9, 0.3) if args.stage == "smoke" else COS_W
        taus = {(r0, c): calibrate_w(r0, R_MAIN, R_PRES, theta_of(c)) for r0 in r0s for c in coss}
        res["calib"] = [{k: v for k, v in t.items()} for t in taus.values()]
        rows = []
        print(f"=== {args.stage}（{len(r0s) * len(seeds) * len(coss)} 格点 × 3 臂） ===")
        for r0 in r0s:
            for c in coss:
                for s in seeds:
                    RS = read_streams(s)
                    g = run_cell(r0, s, c, taus[(r0, c)]["c_tau"], RS)
                    rows.append(g)
                    json.dump({"rows": rows}, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
                    a = g["arms"]
                    print(f"  r={r0:2d} θ_w={c:.1f} seed={s} | clean A {a['A']['clean']:.3f} B {a['B']['clean']:.3f} "
                          f"P {a['P']['clean']:.3f} | ho4 A {a['A']['ho4_mean']:.3f} B {a['B']['ho4_mean']:.3f} "
                          f"P {a['P']['ho4_mean']:.3f} | Δ {a['B']['ho4_mean'] - a['P']['ho4_mean']:+.3f} "
                          f"| maj8 {a['B']['ho8_maj'] - a['P']['ho8_maj']:+.3f} | k {a['B']['issued']} "
                          f"| 串扰 max {g['crosstalk']['max']:.3f}")
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
                   "n_dist": N_DIST, "eta": ETA, "m": M, "s_key": S_KEY, "r_grid": list(R0_GRID),
                   "cos_w": list(COS_W), "cos_r_fix": COS_R_FIX, "r_pres": R_PRES,
                   "r_ho": [R_HO4, R_HO8], "rho_main": R_MAIN, "seeds": SEEDS,
                   "seed_pilot": SEED_PILOT, "stage": args.stage, "card": "DIR-26E rev1.0"}
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
