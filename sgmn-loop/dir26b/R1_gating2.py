"""DIR-26-B · R1_gating2.py —— 内容门控的收益是不是"均衡覆盖"？（P 臂 + $\\rho$ 轴）

卡片：`revolve/DIR-26B/DIRECTION_rev26b_v10.md`（rev 1.0 冻结）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，零改动）
★ 测量层与装置层**从 `dir26/R1_gating.py` 直接 import**（同一对象，非拷贝）⇒ A26B-2 严格成立。

臂：A 全写 · **B 内容门控**（跳过率 $\\rho$）· R 随机等发放数 · **P 位置均衡等发放数（content-blind）** · D 只写首现
轴：$r\\in\\{4,6,8,12\\}$ × $\\rho\\in\\{0.2,0.4,0.6\\}$ × 5 种子 = 220 次；判据格点 $r\\in\\{6,8,12\\}$、主判据 $\\rho{=}0.4$。
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26b")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")

import R1_gating as G                                   # ★ 装置层/测量层的唯一来源（A26B-2）
from R1_gating import (D_MODEL, N_DIST, ETA, M, S_KEY, N_SLOTS, stream_idx,
                       build_stream, drive, make_core, metrics_plas, unit, scalar_invariance)

RHO_GRID = (0.2, 0.4, 0.6)
R0_GRID = (4, 6, 8, 12)
CRIT_R = (6, 8, 12)
R_MAIN = 0.4
R_PRES = 4
THETA = math.radians(25.8419)
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]
SEED_PILOT = 20261041
THR_MAIN = 0.10
THR_FALSIFY = 0.05
THR_RANGE = 0.20                                        # (G) 量程闸门：极差 < 2×主阈值
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
HERE = os.path.dirname(os.path.abspath(__file__))
res = {"checks": [], "rows": [], "calib": []}


# ══════════════════════════════════════════════ P 臂：位置均衡（content-blind，卡片 §2 新 1）
def keep_positions(n, r_pres, k):
    """只依赖 $(n,R,k)$：条目间差额 $\\le1$，条目内跨轮均匀。**不读任何张量。**"""
    per_item_tot = r_pres * 2 * M
    base, rem = divmod(int(k), n)
    keep = set()
    for i in range(n):
        c_i = base + (1 if i < rem else 0)
        idxs = sorted({int(round(x)) for x in np.linspace(0, per_item_tot - 1, max(c_i, 1))})
        j = 0
        while len(idxs) < c_i:
            if j not in idxs:
                idxs.append(j)
                idxs.sort()
            j += 1
        for pos in idxs[:c_i]:
            rho, rest = divmod(pos, 2 * M)
            mbr, m = divmod(rest, M)
            keep.add(stream_idx(rho, i, mbr, m, n))
    return keep


def item_counts(keep, n):
    cnt = [0] * n
    for t in keep:
        cnt[(t // (2 * M)) % n] += 1
    return cnt


# ══════════════════════════════════════════════ $c_\\tau$ 标定（rev 1.2 §C4 规则，pilot 种子）
def calibrate(r0, rho):
    D = build_stream(SEED_PILOT, D_MODEL, N_DIST, R_PRES, THETA)
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
    row = {"r0": r0, "rho": rho, "c_tau": tau, "skip_rate": rate, "gap": m_skip - m_keep,
           "skip_frac_repeat": frac_rep, "gate_ok": bool(ok), "seed": SEED_PILOT}
    res["calib"].append(row)
    print(f"  [标定] r={r0:2d} ρ={rho:.1f} c_τ={tau:+.4f} 跳过率 {rate:.3f} gap {row['gap']:+.4f} "
          f"重复占比 {frac_rep:.3f} {'OK' if ok else '★GATE-DEGENERATE'}")
    return row


# ══════════════════════════════════════════════ 一个格点（同流全部臂）
def run_cell(r0, seed, taus):
    D = build_stream(seed, D_MODEL, N_DIST, R_PRES, THETA)
    T = R_PRES * N_DIST * 2 * M
    out = {"r0": r0, "seed": seed, "T": T, "cells": {}}
    t0 = time.time()

    cA = make_core(D_MODEL, seed, r0); kA, _ = drive(cA, D, "A")
    out["cells"]["A"] = {**metrics_plas(cA._W().detach(), cA.W0, D["proto"], D["vb"], N_DIST), "issued": kA}
    cD = make_core(D_MODEL, seed, r0); kD, _ = drive(cD, D, "D")
    out["cells"]["D"] = {**metrics_plas(cD._W().detach(), cD.W0, D["proto"], D["vb"], N_DIST), "issued": kD}

    for rho in RHO_GRID:
        tau = taus[(r0, rho)]
        tag = f"{rho:.1f}"
        cB = make_core(D_MODEL, seed, r0)
        kB, log = drive(cB, D, "B", tau=tau)
        out["cells"][f"B{tag}"] = {**metrics_plas(cB._W().detach(), cB.W0, D["proto"], D["vb"], N_DIST),
                                   "issued": kB}
        gk = torch.Generator().manual_seed(seed + 7)
        keepR = set(torch.randperm(T, generator=gk)[:int(kB)].tolist())
        cR = make_core(D_MODEL, seed, r0); kR, _ = drive(cR, D, "R", keep=keepR)
        out["cells"][f"R{tag}"] = {**metrics_plas(cR._W().detach(), cR.W0, D["proto"], D["vb"], N_DIST),
                                   "issued": kR}
        keepP = keep_positions(N_DIST, R_PRES, kB)
        cP = make_core(D_MODEL, seed, r0); kP, _ = drive(cP, D, "R", keep=keepP)
        out["cells"][f"P{tag}"] = {**metrics_plas(cP._W().detach(), cP.W0, D["proto"], D["vb"], N_DIST),
                                   "issued": kP}
        cnt = item_counts(keepP, N_DIST)
        out.setdefault("P_balance", {})[tag] = {"k_P": kP, "k_B": kB, "item_min": min(cnt),
                                                "item_max": max(cnt)}
    # 量程闸门 + 护栏（在主 ρ 上）
    vals = [out["cells"][k]["ret1"] for k in out["cells"]]
    out["range"] = float(max(vals) - min(vals))
    si = scalar_invariance(cA._W().detach(), cA.W0, D["proto"], D["vb"], N_DIST)
    out["scalar_invariance"] = {str(k_): v for k_, v in si.items()}
    out["wall"] = round(time.time() - t0, 1)
    return out


# ══════════════════════════════════════════════ Stage A
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
    print("=== Stage A（A26B-1 … A26B-10） ===")
    dev = "/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py"
    chk("A26B-1", md5(dev) == DEV_MD5, f"装置 md5 {md5(dev)} == {DEV_MD5}")

    names = ["unit", "make_core", "build_stream", "drive", "metrics_plas", "stream_idx"]
    bad = [n for n in names if getattr(G, n, None).__module__ != "R1_gating"]
    chk("A26B-2", not bad, f"测量/装置层由 dir26/R1_gating.py import（同一对象）：异常 {bad or '无'}")

    taus = {}
    cals = []
    for r0 in R0_GRID:
        for rho in RHO_GRID:
            c = calibrate(r0, rho)
            cals.append(c)
            taus[(r0, rho)] = c["c_tau"] if c["gate_ok"] else None
    chk("A26B-5", all(c["gate_ok"] for c in cals),
        "ρ 定标自洽：跳过率 "
        + " ".join(f"{c['skip_rate']:.2f}/{c['rho']:.1f}" for c in cals[:6])
        + f" …（共 {len(cals)} 格，阈值 |率−ρ|≤0.05、gap≥0.05、重复占比≥0.85）")

    o = run_cell(8, SEEDS[0], taus)
    D8 = build_stream(SEEDS[0], D_MODEL, N_DIST, R_PRES, THETA)
    c8 = make_core(D_MODEL, SEEDS[0], 8); drive(c8, D8, "A")
    W = c8._W().detach(); P = W - c8.W0
    r6 = int(torch.linalg.matrix_rank(P, rtol=1e-6).item())
    r4 = int(torch.linalg.matrix_rank(P, rtol=1e-4).item())
    rW = int(torch.linalg.matrix_rank(W, rtol=1e-12).item())
    chk("A26B-3", r6 <= 8 and r4 == 8 and rW == D_MODEL,
        f"塑性秩 rank(P,1e-6)={r6}≤8、rank(P,1e-4)={r4}=8；rank(W,1e-12)={rW}=d")

    bal = o["P_balance"]["0.4"]
    k1 = keep_positions(N_DIST, R_PRES, bal["k_B"])
    k2 = keep_positions(N_DIST, R_PRES, bal["k_B"])
    chk("A26B-4", bal["k_P"] == bal["k_B"] and (bal["item_max"] - bal["item_min"]) <= 1 and k1 == k2,
        f"P 臂等量且均衡：k_P={bal['k_P']}==k_B={bal['k_B']}；每条目发放数 [{bal['item_min']},{bal['item_max']}] 差额≤1；"
        f"keep 集可复现 {k1 == k2}")

    chk("A26B-6", o["range"] >= THR_RANGE,
        f"量程闸门（r=8, seed1）：臂间极差 {o['range']:.3f} ≥ {THR_RANGE} ⇒ 非 NO-RANGE")

    T_ = R_PRES * N_DIST * 2 * M
    chk("A26B-7", o["T"] == T_ and o["cells"]["A"]["issued"] == T_,
        f"写步计数 T={o['T']} == R·N·2·m={T_}；A 臂实发 {o['cells']['A']['issued']}")

    cz = make_core(D_MODEL, SEEDS[0], 8)
    Wb = cz._W().detach().clone()
    kz, _ = drive(cz, D8, "NONE")
    chk("A26B-8", kz == 0 and float((cz._W().detach() - Wb).norm()) == 0.0,
        f"零发放臂：发放 {kz}，‖ΔW‖={float((cz._W().detach() - Wb).norm()):.1f} 逐位 0")

    rho_stab = 1 - ETA * (1.0 * S_KEY + 0.05 / 2)
    chk("A26B-9", rho_stab > 0, f"稳定性 ρ=1−η(αS+λ/2)={rho_stab:.4f} > 0")

    si = [v for v in o["scalar_invariance"].values()]
    chk("A26B-10", max(si) - min(si) == 0.0,
        f"纯标量不变性 ret1(cP) c∈{{0.37,1,2.5}} = {si}（逐位相同）")

    res["stage_a_point"] = {k: v for k, v in o["cells"].items()}
    return o, taus


# ══════════════════════════════════════════════ 判据
def pair(rows, r0, tag):
    d = [r["cells"][f"B{tag}"]["ret1"] - r["cells"][f"P{tag}"]["ret1"] for r in rows if r["r0"] == r0]
    return float(np.mean(d)) if d else float("nan"), (float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")), d


def mean_of(rows, r0, key):
    v = [r["cells"][key]["ret1"] for r in rows if r["r0"] == r0]
    return float(np.mean(v)) if v else float("nan")


def criteria(rows):
    print("=== 判据（rev 1.0 冻结阈值） ===")
    v = {}
    free, noir = [], []
    for r0 in R0_GRID:
        rg = [r["range"] for r in rows if r["r0"] == r0]
        rng = float(np.mean(rg))
        (noir if rng < THR_RANGE else free).append(r0)
    v["g_range"] = {"ok": bool(free), "free": free, "no_range": noir, "threshold": THR_RANGE,
                    "detail": {r0: float(np.mean([r["range"] for r in rows if r["r0"] == r0]))
                               for r0 in R0_GRID}}

    crit_free = [r0 for r0 in free if r0 in CRIT_R]
    a_det, ok_a = [], bool(crit_free)
    for r0 in crit_free:
        m, sd, d = pair(rows, r0, f"{R_MAIN:.1f}")
        ok = m >= THR_MAIN
        ok_a = ok_a and ok
        a_det.append({"r0": r0, "d_BP": m, "sd": sd, "n_pos": int(sum(x > 0 for x in d)),
                      "ret1": {k: mean_of(rows, r0, k) for k in (f"B{R_MAIN:.1f}", f"P{R_MAIN:.1f}", "A", "D")},
                      "ok": ok})
    v["a_main"] = {"ok": ok_a, "criterion_cells": crit_free, "details": a_det,
                   "note": "Δt_BP = ret1(B)−ret1(P) ≥ 0.10 @ ρ=0.4，每个 free 判据格点"}

    b_pos = all(x["d_BP"] > 0 for r0 in free for x in [{"d_BP": pair(rows, r0, f"{R_MAIN:.1f}")[0]}])
    v["b_sign"] = {"ok": b_pos,
                   "detail": {r0: pair(rows, r0, f"{R_MAIN:.1f}")[0] for r0 in R0_GRID}}

    curve = []
    for r0 in free:
        for rho in RHO_GRID:
            tag = f"{rho:.1f}"
            m, sd, _ = pair(rows, r0, tag)
            curve.append({"r0": r0, "rho": rho, "d_BP": m, "sd": sd,
                          "ret1_B": mean_of(rows, r0, f"B{tag}"), "ret1_P": mean_of(rows, r0, f"P{tag}"),
                          "ret1_R": mean_of(rows, r0, f"R{tag}"),
                          "ret1_A": mean_of(rows, r0, "A"),
                          "k_B_over_T": float(np.mean([r["cells"][f"B{tag}"]["issued"] / r["T"]
                                                       for r in rows if r["r0"] == r0]))})
    star = {r0: None for r0 in free}
    for r0 in free:
        for rho in RHO_GRID:
            e = [x for x in curve if x["r0"] == r0 and x["rho"] == rho][0]
            if star[r0] is None and e["ret1_B"] >= e["ret1_A"] - 0.02:
                star[r0] = rho
    v["c_curve"] = {"curve": curve, "rho_star": star,
                    "note": "ρ^★ = min{ρ : ret1(B_ρ) ≥ ret1(A) − 0.02}（省写曲线，报告不判）"}

    dbr, dbr_sd, _ = pair_db(rows, 8)
    v["d_align"] = {"ok": abs(dbr - 0.2914) <= 0.10, "d_BR": dbr, "sd": dbr_sd,
                    "ref": 0.2914,
                    "note": "r=8,ρ=0.4 的 ret1(B)−ret1(R) 对齐 D-33 的 +0.2914（差 ≤0.10）"}

    ok_e = all(max(r["scalar_invariance"].values()) - min(r["scalar_invariance"].values()) == 0.0
               for r in rows)
    v["e_guard"] = {"ok": ok_e, "note": "纯标量不变性逐格点逐位（护栏）"}

    falsified = any(x["d_BP"] <= THR_FALSIFY for x in a_det)
    if len(crit_free) < 2 or not (v["d_align"]["ok"] and v["e_guard"]["ok"]):
        main = "INDETERMINATE"
    elif falsified:
        main = "FALSIFIED"
    elif v["a_main"]["ok"] and v["b_sign"]["ok"]:
        main = "CONFIRMED"
    else:
        main = "WEAK"
    v["main"] = {"ok": main == "CONFIRMED", "verdict": main}
    for k in ("g_range", "a_main", "b_sign", "d_align", "e_guard"):
        print(f"  [{'PASS' if v[k]['ok'] else 'FAIL'}] {k}")
    print(f"  Δt_BP(ρ=0.4): {v['b_sign']['detail']}")
    print("  ρ 曲线: " + " | ".join(f"r={x['r0']} ρ={x['rho']}: B {x['ret1_B']:.3f} P {x['ret1_P']:.3f} "
                                   f"Δ {x['d_BP']:+.3f}" for x in curve))
    print(f"  ρ^★: {star}")
    print(f"  ⇒ H26B 判定：**{main}**")
    return v


def pair_db(rows, r0):
    d = [r["cells"][f"B{R_MAIN:.1f}"]["ret1"] - r["cells"][f"R{R_MAIN:.1f}"]["ret1"]
         for r in rows if r["r0"] == r0]
    return (float(np.mean(d)) if d else float("nan"),
            float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"), d)


# ══════════════════════════════════════════════ 主
def main():
    global RHO_GRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26Br10")
    args = ap.parse_args()
    t0 = time.time()

    if args.stage == "stageA":
        stage_a()
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}"
              f"（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"], res["passed"] = "stageA", ok
    else:
        r0s = (8,) if args.stage == "smoke" else R0_GRID
        rhos = (R_MAIN,) if args.stage == "smoke" else RHO_GRID
        RHO_GRID = tuple(rhos)
        taus = {}
        for r0 in r0s:
            for rho in RHO_GRID:
                c = calibrate(r0, rho)
                taus[(r0, rho)] = c["c_tau"] if c["gate_ok"] else None
        seeds = [SEEDS[0]] if args.stage == "smoke" else SEEDS
        rows = []
        print(f"=== {args.stage}（{len(r0s) * len(seeds)} 格点） ===")
        for r0 in r0s:
            for s in seeds:
                o = run_cell(r0, s, taus)
                rows.append(o)
                if args.stage == "formal":
                    op = f"/home/ec2-user/sgmn-loop/dir26b/out/{args.tag}_rows.json"
                    os.makedirs(os.path.dirname(op), exist_ok=True)
                    json.dump(rows, open(op, "w"), ensure_ascii=False, indent=1, default=float)
                cs = o["cells"]
                print(f"  r={r0:2d} seed={s} 极差 {o['range']:.3f} | A {cs['A']['ret1']:.3f} "
                      f"D {cs['D']['ret1']:.3f} | B.4 {cs['B0.4']['ret1']:.3f} P.4 {cs['P0.4']['ret1']:.3f} "
                      f"R.4 {cs['R0.4']['ret1']:.3f} | k_B(T=560) "
                      + "/".join(f"{cs[f'B{t:.1f}']['issued']}" for t in RHO_GRID)
                      + f" ({o['wall']:.0f}s)")
        res["rows"] = rows
        if args.stage == "formal":
            outp0 = args.out or f"/home/ec2-user/sgmn-loop/dir26b/out/{args.tag}_{args.stage}.json"
            os.makedirs(os.path.dirname(outp0), exist_ok=True)
            json.dump(res, open(outp0, "w"), ensure_ascii=False, indent=1, default=float)   # ★ 先保 rows
            print(f"  [落盘] rows 已写入 {outp0}（判据前）")
            try:
                res["verdicts"] = criteria(rows)
                res["passed"] = res["verdicts"]["main"]["ok"]
            except Exception as e:                                   # 判据异常不得吞掉读数
                res["verdicts"] = {"error": f"{type(e).__name__}: {e}"}
                res["passed"] = False
                print(f"  ✗ 判据异常：{type(e).__name__}: {e}")

    res["meta"] = {"wall_seconds": round(time.time() - t0, 1), "device": "cpu", "d": D_MODEL,
                   "n_dist": N_DIST, "eta": ETA, "m": M, "s_key": S_KEY,
                   "r_grid": list(R0_GRID), "crit_r": list(CRIT_R), "rho_grid": list(RHO_GRID),
                   "R_pres": R_PRES, "theta_deg": round(math.degrees(THETA), 4),
                   "seeds": SEEDS, "seed_pilot": SEED_PILOT, "stage": args.stage,
                   "card": "DIR-26B rev1.0"}
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26b/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
