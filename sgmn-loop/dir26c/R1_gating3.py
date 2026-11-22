"""DIR-26-C · R1_gating3.py —— 内容收益在**跨情节查询**（读键≠写键）下是否仍成立？

卡片：`revolve/DIR-26C/DIRECTION_rev26c_v11.md`（rev 1.1 冻结；rev 1.0 存档）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，零改动）
★ 测量层/装置层/P 臂/ρ 定标全部 **import**（同一对象，非拷贝）：`dir26/R1_gating.py`、`dir26b/R1_gating2.py`

三读键：`clean` $k^{\\rm proto}_i$ · `Jin` 写用实例键 · **`Jout` 未参与写的独立抖动键（跨情节查询）**
聚合：`mean`（逐读命中率）· `maj`（$r_q{=}4$ 取 $\\ge3$）
臂：A 全写 · **B 内容门控** · **P 位置均衡（等发放数，content-blind）**，$\\rho\\in\\{.05,.10,.15,.20,.40\\}$
格点：$r\\in\\{6,8,12\\}$ × 5 种子 × 7 运行 = 105 次。
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26c")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26b")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")

import R1_gating as G                                   # 装置/测量层唯一来源（A26C-2）
import R1_gating2 as G2                                 # P 臂 + ρ 定标唯一来源（A26C-2）
from R1_gating import (D_MODEL, N_DIST, ETA, M, S_KEY, build_stream, drive, make_core,
                       metrics_plas, scalar_invariance, stream_idx, unit)
from R1_gating2 import calibrate, keep_positions

R0_GRID = (6, 8, 12)
RHO_MAIN = 0.40
RHO_SWEEP = (0.05, 0.10, 0.15, 0.20)
RHO_ALL = (0.40,) + RHO_SWEEP
R_PRES = 4
R_HO = 4
HO_OFFSET = 999
THETA = math.radians(25.8419)
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]
SEED_PILOT = 20261041
THR_MAIN = 0.10
THR_FALSIFY = 0.05
THR_DEGEN = 0.10                                        # (G) 退化闸门
THR_AGREE = 0.10                                        # (c) 聚合一致性
THR_QUOTA = 0.02                                        # (e) 免费额度容差
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
res = {"checks": [], "rows": [], "calib": []}


# ══════════════════════════════════════════════ 抖动读指标（卡片 §2）
def ret_jitter(P, Kq, vb, n, r_q=4):
    """读键 $Kq$（形状 $(r_q,n,d)$）。返回 mean / maj / 逐项。**不读 $W^0$ 以外的任何写侧状态。**"""
    hit = np.zeros((r_q, n), dtype=bool)
    vbU = unit(vb)
    for rho in range(r_q):
        Y = P @ Kq[rho].T
        S = unit(Y.T) @ vbU.T
        for i in range(n):
            hit[rho, i] = int(torch.argmax(S[i]).item()) == i
    maj_need = r_q // 2 + 1                             # r_q=4 ⇒ 3
    return {"ret1_mean": float(hit.mean()),
            "ret1_maj": float(np.mean([hit[:, i].sum() >= maj_need for i in range(n)])),
            "maj_need": int(maj_need),
            "n_hit": int(hit.sum()), "n_read": int(hit.size),
            "per_item": [int(x) for x in hit.sum(0)]}


def ret_jitter_bruteforce(P, Kq, vb, n, r_q=4):
    """暴力对照（A26C-3）：逐 (i,$\\rho$) 直接算 $\\cos(Pq,v_j)$，不经 $S$ 矩阵技巧。"""
    hit = np.zeros((r_q, n), dtype=bool)
    for rho in range(r_q):
        for i in range(n):
            q = P @ Kq[rho, i]
            y = q / q.norm().clamp_min(1e-30)
            scores = [(float(torch.dot(y, vb[j] / vb[j].norm().clamp_min(1e-30))), j) for j in range(n)]
            hit[rho, i] = max(scores)[1] == i
    maj_need = r_q // 2 + 1
    return {"ret1_mean": float(hit.mean()),
            "ret1_maj": float(np.mean([hit[:, i].sum() >= maj_need for i in range(n)])),
            "maj_need": int(maj_need), "n_hit": int(hit.sum()), "n_read": int(hit.size)}


def scalar_inv_jitter(P, Kq, vb, n, r_q=4, scales=(0.37, 1.0, 2.5)):
    """(f) 护栏：$\\cos(cPq,v)=\\cos(Pq,v)$ 对 $c>0$ 严格成立 ⇒ $\\mathrm{ret}_1$ 不变。"""
    return {c: ret_jitter(c * P, Kq, vb, n, r_q)["ret1_mean"] for c in scales}


# ══════════════════════════════════════════════ 一个格点
def run_cell(r0, seed, taus):
    D = build_stream(seed, D_MODEL, N_DIST, R_PRES, THETA)
    DH = build_stream(seed + HO_OFFSET, D_MODEL, N_DIST, R_HO, THETA)
    T = R_PRES * N_DIST * 2 * M
    out = {"r0": r0, "seed": seed, "T": T, "cells": {}}
    t0 = time.time()

    def record(tag, core, issued):
        W = core._W().detach(); P = W - core.W0
        m = metrics_plas(W, core.W0, D["proto"], D["vb"], N_DIST)
        row = {"issued": issued, "ret1": m["ret1"], "rowm": m["rowm"], "cosP": m["cosP"],
               "clean": m["ret1"],
               "Jin": ret_jitter(P, D["K"], D["vb"], N_DIST, R_PRES),
               "Jout": ret_jitter(P, DH["K"], D["vb"], N_DIST, R_HO)}
        out["cells"][tag] = row
        return row

    cA = make_core(D_MODEL, seed, r0); kA, _ = drive(cA, D, "A"); record("A", cA, kA)
    cB = make_core(D_MODEL, seed, r0); kB, _ = drive(cB, D, "B", tau=taus[(r0, RHO_MAIN)])
    record(f"B{RHO_MAIN:.2f}", cB, kB)
    keepP = keep_positions(N_DIST, R_PRES, kB)
    cP = make_core(D_MODEL, seed, r0); kP, _ = drive(cP, D, "R", keep=keepP)
    record(f"P{RHO_MAIN:.2f}", cP, kP)
    out["P_balance"] = {"k_P": kP, "k_B": kB}
    for rho in RHO_SWEEP:
        c = make_core(D_MODEL, seed, r0); k, _ = drive(c, D, "B", tau=taus[(r0, rho)])
        record(f"B{rho:.2f}", c, k)

    vals = [out["cells"][k]["Jout"]["ret1_mean"] for k in (f"B{RHO_MAIN:.2f}", f"P{RHO_MAIN:.2f}", "A")]
    out["spread_Jout"] = float(max(vals) - min(vals))
    si = scalar_inv_jitter(cB._W().detach() - cB.W0, DH["K"], D["vb"], N_DIST, R_HO)
    out["guard_f"] = {str(k_): v for k_, v in si.items()}
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
    print("=== Stage A（A26C-1 … A26C-10） ===")
    dev = "/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py"
    chk("A26C-1", md5(dev) == DEV_MD5, f"装置 md5 {md5(dev)} == {DEV_MD5}")

    bad = [n for n in ("unit", "make_core", "build_stream", "drive", "metrics_plas", "stream_idx")
           if getattr(G, n, None).__module__ != "R1_gating"]
    bad += [n for n in ("keep_positions", "calibrate") if getattr(G2, n, None).__module__ != "R1_gating2"]
    chk("A26C-2", not bad, f"装置/测量层/P 臂/定标均为 import 的同一对象：异常 {bad or '无'}")

    # A26C-3 抖动指标逐位正确（d=16 小装置）
    d16, n16 = 16, 8
    D16 = build_stream(SEEDS[0], d16, n16, R_PRES, THETA)
    D16H = build_stream(SEEDS[0] + HO_OFFSET, d16, n16, R_HO, THETA)
    c16 = make_core(d16, SEEDS[0], 4); drive(c16, D16, "A")
    P16 = c16._W().detach() - c16.W0
    a = ret_jitter(P16, D16H["K"], D16["vb"], n16, R_HO)
    b = ret_jitter_bruteforce(P16, D16H["K"], D16["vb"], n16, R_HO)
    same = (a["ret1_mean"] == b["ret1_mean"] and a["ret1_maj"] == b["ret1_maj"]
            and a["n_hit"] == b["n_hit"] and a["maj_need"] == b["maj_need"] == 3)
    chk("A26C-3", same, f"矩阵式与暴力逐项 cos 逐位相同：mean {a['ret1_mean']:.4f}={b['ret1_mean']:.4f}、"
                        f"maj {a['ret1_maj']:.4f}={b['ret1_maj']:.4f}、命中 {a['n_hit']}/{a['n_read']}、maj_need={a['maj_need']}")

    # A26C-4 跨情节键：同原型、独立抖动（在 d=896 上核）
    D = build_stream(SEEDS[0], D_MODEL, N_DIST, R_PRES, THETA)
    DH = build_stream(SEEDS[0] + HO_OFFSET, D_MODEL, N_DIST, R_HO, THETA)
    d_proto = float((D["proto"] - DH["proto"]).abs().max())
    Kw = D["K"][0]                                       # 写键（$\\rho{=}0$）
    same_item = float(torch.nn.functional.cosine_similarity(DH["K"][0], Kw).mean())
    diff_item = float(torch.nn.functional.cosine_similarity(DH["K"][0].unsqueeze(1),
                                                            Kw.unsqueeze(0), dim=-1)
                      .masked_fill(torch.eye(N_DIST, dtype=torch.bool), float("nan"))
                      .nanmean())
    c_ho_proto = float(torch.nn.functional.cosine_similarity(DH["K"][0], DH["proto"]).mean())
    ok4 = (d_proto == 0.0 and 0.78 <= same_item <= 0.84 and abs(diff_item) <= 0.10
           and abs(c_ho_proto - 0.9) <= 0.01)
    chk("A26C-4", ok4, f"跨情节键：proto 逐位同一(Δmax={d_proto:.1f})；同条目 cos={same_item:.4f}∈[0.78,0.84]；"
                       f"不同条目 cos={diff_item:+.4f}≤0.10；与原型 cos={c_ho_proto:.4f}=0.9±0.01")

    chk("A26C-5", torch.equal(D16["K"], build_stream(SEEDS[0], d16, n16, R_PRES, THETA)["K"]),
        "Jin 键逐位就是写键（同流重建 torch.equal=True）")

    taus = {}
    cals = []
    for r0 in R0_GRID:
        for rho in RHO_ALL:
            c = calibrate(r0, rho); cals.append(c); taus[(r0, rho)] = c["c_tau"] if c["gate_ok"] else None
    chk("A26C-7", all(c["gate_ok"] for c in cals),
        f"ρ 定标自洽（{len(cals)} 格）：跳过率 " + " ".join(f"{c['skip_rate']:.2f}" for c in cals[:5]) + " …")

    o = run_cell(8, SEEDS[0], taus)
    bal = o["P_balance"]
    chk("A26C-6", bal["k_P"] == bal["k_B"],
        f"P 臂等量：k_P={bal['k_P']} == k_B={bal['k_B']}（条目间差额另由 A26B-4 逐位保证）")

    Wb = c16._W().detach().clone()
    _ = ret_jitter(c16._W().detach() - c16.W0, D16H["K"], D16H["vb"], 8, R_HO)   # ★ d=16 的键配 d=16 的 P（DEF-26C-1）
    chk("A26C-8", o["T"] == 560 and o["cells"]["A"]["issued"] == 560
        and torch.equal(c16._W().detach(), Wb),
        f"T={o['T']}=560、A 臂实发 {o['cells']['A']['issued']}；三读键计算前后 W 逐位不变={torch.equal(c16._W().detach(), Wb)}")

    rho_stab = 1 - ETA * (1.0 * S_KEY + 0.05 / 2)
    cN = make_core(D_MODEL, SEEDS[0], 8); Wb2 = cN._W().detach().clone()
    kN, _ = drive(cN, D, "NONE")
    chk("A26C-9", rho_stab > 0 and kN == 0 and float((cN._W().detach() - Wb2).norm()) == 0.0,
        f"稳定性 ρ={rho_stab:.4f}>0；零发放臂发放 {kN}、‖ΔW‖=0")

    si = list(o["guard_f"].values())
    chk("A26C-10", max(si) - min(si) == 0.0,
        f"(f) 纯标量不变性（Jout 读）c∈{{0.37,1,2.5}} = {si}（逐位相同）")
    return o, taus


# ══════════════════════════════════════════════ 判据
def arms_diff(rows, r0, mode, agg, ta, tb):
    d = [r["cells"][ta][mode][agg] - r["cells"][tb][mode][agg] for r in rows if r["r0"] == r0]
    return float(np.mean(d)) if d else float("nan"), (float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")), d


def arm_mean(rows, r0, tag, mode=None, agg=None):
    if mode:
        v = [r["cells"][tag][mode][agg] for r in rows if r["r0"] == r0]
    else:
        v = [r["cells"][tag]["ret1"] for r in rows if r["r0"] == r0]
    return float(np.mean(v)) if v else float("nan")


def criteria(rows):
    print("=== 判据（rev 1.1 冻结阈值） ===")
    v = {}
    free, noir = [], []
    for r0 in R0_GRID:
        sp = float(np.mean([r["spread_Jout"] for r in rows if r["r0"] == r0]))
        (noir if sp < THR_DEGEN else free).append(r0)
    v["g_degen"] = {"ok": bool(free), "free": free, "no_range": noir, "threshold": THR_DEGEN,
                    "detail": {r0: float(np.mean([r["spread_Jout"] for r in rows if r["r0"] == r0]))
                               for r0 in R0_GRID}}

    tB, tP = f"B{RHO_MAIN:.2f}", f"P{RHO_MAIN:.2f}"
    a_ok, a_det = bool(free), []
    for r0 in free:
        m, sd, d = arms_diff(rows, r0, "Jout", "ret1_mean", tB, tP)
        mm, smm, _ = arms_diff(rows, r0, "Jout", "ret1_maj", tB, tP)
        ok = m >= THR_MAIN
        a_ok = a_ok and ok
        a_det.append({"r0": r0, "d_out_mean": m, "sd": sd, "d_out_maj": mm, "sd_maj": smm,
                      "n_pos": int(sum(x > 0 for x in d)), "n": len(d),
                      "ok": ok, "agree_ok": abs(m - mm) <= THR_AGREE and (m > 0) == (mm > 0),
                      "ret1": {k: arm_mean(rows, r0, t, "Jout", "ret1_mean") for k, t in
                               (("B", tB), ("P", tP), ("A", "A"))},
                      "clean": {k: arm_mean(rows, r0, t) for k, t in (("B", tB), ("P", tP), ("A", "A"))}})
    v["a_main"] = {"ok": a_ok, "cells": free, "details": a_det,
                   "note": "Δt=ret1_Jout,mean(B)−ret1_Jout,mean(P) ≥ 0.10 @ ρ=0.40，每个 free 格点"}
    v["b_sign"] = {"ok": all(x["n_pos"] == x["n"] for x in a_det),
                   "detail": {x["r0"]: f"{x['n_pos']}/{x['n']}" for x in a_det}}
    v["c_agree"] = {"ok": all(x["agree_ok"] for x in a_det),
                    "detail": {x["r0"]: {"mean": round(x["d_out_mean"], 4), "maj": round(x["d_out_maj"], 4)}
                               for x in a_det}}

    cost = []
    for r0 in free:
        e = {"r0": r0}
        for k, t in (("A", "A"), ("B", tB), ("P", tP)):
            cl = arm_mean(rows, r0, t)
            jm = arm_mean(rows, r0, t, "Jout", "ret1_mean")
            ji = arm_mean(rows, r0, t, "Jin", "ret1_mean")
            e[k] = {"clean": cl, "Jout": jm, "Jin": ji, "cost_out": cl - jm, "cost_in": cl - ji}
        cost.append(e)
    v["d_cost"] = {"ok": True, "detail": cost, "note": "抖动/跨情节代价（报告，不判）"}

    quota = []
    for r0 in free:
        aA = arm_mean(rows, r0, "A")
        row = {"r0": r0, "ret1_A": aA, "levels": {}}
        for rho in RHO_ALL:
            t = f"B{rho:.2f}"
            row["levels"][f"{rho:.2f}"] = {"ret1_B": arm_mean(rows, r0, t),
                                           "k_B": float(np.mean([r["cells"][t]["issued"] for r in rows
                                                                 if r["r0"] == r0]))}
        ok_rhos = [rho for rho in RHO_ALL if row["levels"][f"{rho:.2f}"]["ret1_B"] >= aA - THR_QUOTA]
        row["rho_max"] = (max(ok_rhos) if ok_rhos else 0.0)
        row["rho_max_str"] = (">=0.40" if row["rho_max"] == 0.40 else
                              ("0（无免费额度）" if not ok_rhos else f"{row['rho_max']:.2f}"))
        quota.append(row)
    v["e_quota"] = {"ok": True, "detail": quota, "tolerance": THR_QUOTA, "note": "ρ 免费额度上界（报告，不判）"}

    ok_f = all(max(r["guard_f"].values()) - min(r["guard_f"].values()) == 0.0 for r in rows)
    v["f_guard"] = {"ok": ok_f, "note": "纯标量不变性（Jout 读）逐格点逐位（护栏）"}

    fals = any(x["d_out_mean"] <= THR_FALSIFY for x in a_det)
    if not free or not (v["c_agree"]["ok"] and ok_f):
        main = "INDETERMINATE"
    elif fals:
        main = "FALSIFIED"
    elif v["a_main"]["ok"] and v["b_sign"]["ok"]:
        main = "CONFIRMED"
    else:
        main = "WEAK"
    v["main"] = {"ok": main == "CONFIRMED", "verdict": main}
    for k in ("g_degen", "a_main", "b_sign", "c_agree", "f_guard"):
        print(f"  [{'PASS' if v[k]['ok'] else 'FAIL'}] {k}  {v[k].get('detail', '')}")
    print(f"  ⇒ H26C 判定：**{main}**")
    return v


# ══════════════════════════════════════════════ 主
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26Cr11")
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
        taus = {}
        for r0 in r0s:
            for rho in RHO_ALL:
                c = calibrate(r0, rho)
                taus[(r0, rho)] = c["c_tau"] if c["gate_ok"] else None
        seeds = [SEEDS[0]] if args.stage == "smoke" else SEEDS
        rows = []
        print(f"=== {args.stage}（{len(r0s) * len(seeds)} 格点 × 7 运行） ===")
        for r0 in r0s:
            for s in seeds:
                o = run_cell(r0, s, taus)
                rows.append(o)
                if args.stage == "formal":
                    op = f"/home/ec2-user/sgmn-loop/dir26c/out/{args.tag}_rows.json"
                    os.makedirs(os.path.dirname(op), exist_ok=True)
                    json.dump(rows, open(op, "w"), ensure_ascii=False, indent=1, default=float)  # ★ 纪律 (xi)
                c = o["cells"]
                tB, tP = f"B{RHO_MAIN:.2f}", f"P{RHO_MAIN:.2f}"
                print(f"  r={r0:2d} seed={s} 极差(Jout) {o['spread_Jout']:.3f} | clean A {c['A']['clean']:.3f} "
                      f"B {c[tB]['clean']:.3f} P {c[tP]['clean']:.3f} | Jout mean A {c['A']['Jout']['ret1_mean']:.3f} "
                      f"B {c[tB]['Jout']['ret1_mean']:.3f} P {c[tP]['Jout']['ret1_mean']:.3f} | "
                      f"Δt.mean {c[tB]['Jout']['ret1_mean'] - c[tP]['Jout']['ret1_mean']:+.3f} "
                      f"maj {c[tB]['Jout']['ret1_maj'] - c[tP]['Jout']['ret1_maj']:+.3f} | "
                      f"k_B " + "/".join(f"{c[f'B{t:.2f}']['issued']}" for t in RHO_ALL) + f" ({o['wall']:.0f}s)")
        res["rows"] = rows
        if args.stage == "formal":
            outp0 = args.out or f"/home/ec2-user/sgmn-loop/dir26c/out/{args.tag}_{args.stage}.json"
            os.makedirs(os.path.dirname(outp0), exist_ok=True)
            json.dump(res, open(outp0, "w"), ensure_ascii=False, indent=1, default=float)      # ★ 先落盘
            print(f"  [落盘] rows 已写入 {outp0}（判据前）")
            try:
                res["verdicts"] = criteria(rows)
                res["passed"] = res["verdicts"]["main"]["ok"]
            except Exception as e:
                res["verdicts"] = {"error": f"{type(e).__name__}: {e}"}
                res["passed"] = False
                print(f"  ✗ 判据异常：{type(e).__name__}: {e}")

    res["meta"] = {"wall_seconds": round(time.time() - t0, 1), "device": "cpu", "d": D_MODEL,
                   "n_dist": N_DIST, "eta": ETA, "m": M, "s_key": S_KEY, "R_pres": R_PRES,
                   "R_ho": R_HO, "ho_offset": HO_OFFSET, "r_grid": list(R0_GRID),
                   "rho_main": RHO_MAIN, "rho_sweep": list(RHO_SWEEP),
                   "theta_deg": round(math.degrees(THETA), 4), "seeds": SEEDS,
                   "seed_pilot": SEED_PILOT, "stage": args.stage, "card": "DIR-26C rev1.1"}
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26c/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
