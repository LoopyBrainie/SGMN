"""DIR-26-D · R1_gating4.py —— 读侧错位阈值（内容收益在多大的查询错位下归零？）

卡片：`revolve/DIR-26D/DIRECTION_rev26d_v10.md`（rev 1.0 冻结）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，零改动）
★ 测量层/装置层/P 臂/ρ 定标/抖动指标全部 **import**：`dir26/R1_gating.py`、`dir26b/R1_gating2.py`、`dir26c/R1_gating3.py`

独立变量 = **读侧错位角 θ_r**（cosθ_r ∈ {0.9,0.7,0.5,0.3}），写侧固定 cosθ_w=0.9。
每组：主组 R_w=4（T=560）· 少写组 R_w=2（T=280）；臂 A / B(ρ=.4) / P。
格点：r ∈ {8,12} × 5 种子 × 6 次 = 60 次运行。
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("dir26d", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G                                    # 装置层/测量层唯一来源（A26D-2）
import R1_gating2 as G2                                  # P 臂 + ρ 定标唯一来源
import R1_gating3 as G3                                  # 抖动指标唯一来源（同一对象）
from R1_gating import (D_MODEL, N_DIST, ETA, M, S_KEY, build_stream, drive, make_core,
                       metrics_plas, unit)
from R1_gating2 import calibrate, keep_positions
from R1_gating3 import ret_jitter, ret_jitter_bruteforce

R0_GRID = (8, 12)
RW_MAIN, RW_THIN = 4, 2
R_MAIN = 0.40
COS_R = (0.9, 0.7, 0.5, 0.3)                             # ★ 唯一独立变量
R_HO4, R_HO8 = 4, 8
HO_OFFSET = 999
THETA_W = math.radians(25.8419)                          # cos = 0.9
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]
SEED_PILOT = 20261041
THR_MAIN = 0.10
THR_ZERO = 0.05
THR_DEGEN = 0.10
THR_MONO = 0.05
THR_AGREE = 0.10
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
D35_JSON = "/home/ec2-user/sgmn-loop/dir26c/out/DIR26Cr11_formal.json"
res = {"checks": [], "rows": [], "calib": [], "anchor_d35": {}}


def theta_of(cos_r):
    """cos=0.9 档必须**逐位**等于写侧/ D-35 的 `THETA_W`（`acos(0.9)` 与 `radians(25.8419)` 差 2.6e-6 ⇒ 键会不同 ⇒ A26D-4）。"""
    return THETA_W if abs(cos_r - 0.9) < 1e-9 else math.acos(cos_r)


# ══════════════════════════════════════════════ 定标（$R_{\\rm w}$ 参数化；A26D-7）
def calibrate_at(r0, rho, r_w):
    """与 `R1_gating2.calibrate` **同一逻辑**（同一 quantile 规则、同一 gate_ok 判据），只把 $R$ 参数化。"""
    D = build_stream(SEED_PILOT, D_MODEL, N_DIST, r_w, THETA_W)
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
    row = {"r0": r0, "rho": rho, "r_w": r_w, "c_tau": tau, "skip_rate": rate,
           "gap": m_skip - m_keep, "skip_frac_repeat": frac_rep, "gate_ok": bool(ok), "seed": SEED_PILOT}
    print(f"  [标定] r={r0:2d} ρ={rho:.1f} R_w={r_w} c_τ={tau:+.4f} 跳过率 {rate:.3f} "
          f"gap {row['gap']:+.4f} 重复占比 {frac_rep:.3f} {'OK' if ok else '★GATE-DEGENERATE'}")
    return row


# ══════════════════════════════════════════════ 读流（四条独立流 × 两条长度；A26D-6）
def read_streams(seed):
    out = {}
    for c in COS_R:
        key = f"{c:.1f}"
        out[key] = {"ho4": build_stream(seed + HO_OFFSET, D_MODEL, N_DIST, R_HO4, theta_of(c)),
                    "ho8": build_stream(seed + HO_OFFSET, D_MODEL, N_DIST, R_HO8, theta_of(c))}
    return out


def reads_at(P, vb, RS):
    """一条 $W$ 上的全部读键读数：{(θ_r, hoN): mean/maj4/maj8}。**读不改写。**"""
    out = {}
    for c in COS_R:
        key = f"{c:.1f}"
        m4 = ret_jitter(P, RS[key]["ho4"]["K"], vb, N_DIST, R_HO4)
        m8 = ret_jitter(P, RS[key]["ho8"]["K"], vb, N_DIST, R_HO8)
        out[key] = {"mean4": m4["ret1_mean"], "maj4": m4["ret1_maj"],
                    "mean8": m8["ret1_mean"], "maj8": m8["ret1_maj"],
                    "maj_need4": m4["maj_need"], "maj_need8": m8["maj_need"]}
    return out


# ══════════════════════════════════════════════ 一组（给定 $R_{\\rm w}$）
def run_group(r0, seed, r_w, tau, RS):
    D = build_stream(seed, D_MODEL, N_DIST, r_w, THETA_W)
    T = r_w * N_DIST * 2 * M
    out = {"r0": r0, "seed": seed, "r_w": r_w, "T": T, "arms": {}}

    def rec(tag, core, issued):
        P = core._W().detach() - core.W0
        w_before = core._W().detach().clone()
        rd = reads_at(P, D["vb"], RS)
        out["arms"][tag] = {"issued": issued, "clean": metrics_plas(core._W().detach(), core.W0,
                                                                   D["proto"], D["vb"], N_DIST)["ret1"],
                            "reads": rd,
                            "W_unchanged": bool(torch.equal(core._W().detach(), w_before))}
        return issued

    cA = make_core(D_MODEL, seed, r0); kA, _ = drive(cA, D, "A");       rec("A", cA, kA)
    cB = make_core(D_MODEL, seed, r0); kB, _ = drive(cB, D, "B", tau=tau); rec("B", cB, kB)
    keepP = keep_positions(N_DIST, r_w, kB)
    cP = make_core(D_MODEL, seed, r0); kP, _ = drive(cP, D, "R", keep=keepP); rec("P", cP, kP)
    out["kP_eq_kB"] = bool(kP == kB)
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
    print("=== Stage A（A26D-1 … A26D-11） ===")
    dev = "/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py"
    chk("A26D-1", md5(dev) == DEV_MD5, f"装置 md5 {md5(dev)} == {DEV_MD5}")

    bad = [n for n in ("unit", "make_core", "build_stream", "drive", "metrics_plas", "stream_idx")
           if getattr(G, n, None).__module__ != "R1_gating"]
    bad += [n for n in ("keep_positions", "calibrate") if getattr(G2, n, None).__module__ != "R1_gating2"]
    bad += [n for n in ("ret_jitter", "ret_jitter_bruteforce") if getattr(G3, n, None).__module__ != "R1_gating3"]
    chk("A26D-2", not bad, f"装置/测量/P 臂/定标/抖动指标 全为 import 的同一对象：异常 {bad or '无'}")

    RS = read_streams(SEEDS[0])
    D0 = build_stream(SEEDS[0], D_MODEL, N_DIST, RW_MAIN, THETA_W)
    al, cr = [], []
    for c in COS_R:
        K4 = RS[f"{c:.1f}"]["ho4"]["K"]
        # ★ dim=-1 必写：`cosine_similarity` 默认 dim=1（DEF-26D-1）
        al.append(float(torch.nn.functional.cosine_similarity(K4, D0["proto"], dim=-1).mean()))
        Kf = K4.reshape(-1, D_MODEL)
        cc = torch.nn.functional.cosine_similarity(Kf.unsqueeze(1), D0["proto"].unsqueeze(0), dim=-1)
        mask = torch.zeros_like(cc, dtype=torch.bool)
        for r_ in range(R_HO4):
            mask[r_ * N_DIST:(r_ + 1) * N_DIST] = torch.eye(N_DIST, dtype=torch.bool)
        cr.append(float(cc[~mask].abs().mean()))          # 平均，非最大值（rev 1.1 / DEF-26D-2）
    ok3 = all(abs(a - c) <= 0.01 for a, c in zip(al, COS_R)) and max(cr) <= 0.05
    chk("A26D-3", ok3, "读键对齐度：实测 cos(读键,原型)=" + " ".join(f"{a:.4f}" for a in al)
        + f"（目标 {list(COS_R)}）；异条目 |cos| 平均 {max(cr):.4f}≤0.05（随机基线 0.027）")

    D35 = json.load(open(D35_JSON))
    rs35 = [r for r in D35["rows"] if r["r0"] == 8 and r["seed"] == SEEDS[0]][0]
    DH4 = build_stream(SEEDS[0] + HO_OFFSET, D_MODEL, N_DIST, R_HO4, THETA_W)
    same_K = torch.equal(RS["0.9"]["ho4"]["K"], DH4["K"])
    g = run_group(8, SEEDS[0], RW_MAIN, calibrate_at(8, R_MAIN, RW_MAIN)["c_tau"], RS)
    ok_e = all(abs(g["arms"][a]["reads"]["0.9"]["mean4"] - rs35["cells"][t]["Jout"]["ret1_mean"]) == 0.0
               and abs(g["arms"][a]["reads"]["0.9"]["maj4"] - rs35["cells"][t]["Jout"]["ret1_maj"]) == 0.0
               for a, t in (("A", "A"), ("B", "B0.40"), ("P", "P0.40")))
    res["anchor_d35"] = {"B": g["arms"]["B"]["reads"]["0.9"], "P": g["arms"]["P"]["reads"]["0.9"],
                         "A": g["arms"]["A"]["reads"]["0.9"],
                         "d35": {k: rs35["cells"][k]["Jout"] for k in ("A", "B0.40", "P0.40")}}
    chk("A26D-4", same_K and ok_e,
        f"锚点流逐位同一 K={same_K}；θ_r=0.9 读数与 D-35 逐位相同={ok_e}"
        f"（B mean4 {g['arms']['B']['reads']['0.9']['mean4']:.4f} vs {rs35['cells']['B0.40']['Jout']['ret1_mean']:.4f}）")

    d16, n16 = 16, 8
    D16 = build_stream(SEEDS[0], d16, n16, R_HO4, THETA_W)
    c16 = make_core(d16, SEEDS[0], 4); drive(c16, D16, "A")
    P16 = c16._W().detach() - c16.W0
    K16 = build_stream(SEEDS[0] + HO_OFFSET, d16, n16, R_HO8, theta_of(0.5))["K"]
    a8 = ret_jitter(P16, K16, D16["vb"], n16, R_HO8)
    b8 = ret_jitter_bruteforce(P16, K16, D16["vb"], n16, R_HO8)
    chk("A26D-5", a8["ret1_mean"] == b8["ret1_mean"] and a8["ret1_maj"] == b8["ret1_maj"]
        and a8["maj_need"] == 5 and a8["n_read"] == n16 * R_HO8,
        f"抖动指标逐位正确：mean {a8['ret1_mean']:.4f}={b8['ret1_mean']:.4f}、maj {a8['ret1_maj']:.4f}="
        f"{b8['ret1_maj']:.4f}、maj_need={a8['maj_need']}（r_q=8）、读数 {a8['n_read']}")

    same_proto = all(torch.equal(RS[k]["ho4"]["proto"], RS["0.9"]["ho4"]["proto"]) for k in RS)
    same_val = all(torch.equal(RS[k]["ho4"]["va"], RS["0.9"]["ho4"]["va"])
                   and torch.equal(RS[k]["ho4"]["vb"], RS["0.9"]["ho4"]["vb"]) for k in RS)
    chk("A26D-6", same_proto and same_val,
        f"四条读流的 proto/va/vb 逐位相同={same_proto and same_val} ⇒ 读侧差异只来自 θ_r")

    cals = [calibrate_at(r0, R_MAIN, RW_MAIN) for r0 in R0_GRID]
    same_cal = all(abs(c["c_tau"] - G2cal["c_tau"]) == 0.0 and abs(c["skip_rate"] - G2cal["skip_rate"]) == 0.0
                   for c, G2cal in zip(cals, [calibrate(r0, R_MAIN) for r0 in R0_GRID]))
    cals_thin = [calibrate_at(r0, R_MAIN, RW_THIN) for r0 in R0_GRID]
    # rev 1.1：R_w=2 的「重复占比」基准率是 1-1/R=0.5 ⇒ 改用**重复富集**（DEF-26D-3）
    enrich_ok = all(abs(c["skip_rate"] - R_MAIN) <= 0.05 and c["gap"] >= 0.05
                    and (c["skip_frac_repeat"] - (1 - 1 / RW_THIN)) >= 0.15 for c in cals_thin)
    chk("A26D-7", same_cal and all(c["gate_ok"] for c in cals) and enrich_ok,
        f"calibrate_at(R_w=4) 与 R1_gating2.calibrate 逐位相同={same_cal}；R_w=2 跳过率 "
        + " ".join(f"{c['skip_rate']:.2f}" for c in cals_thin)
        + "、重复富集 " + " ".join(f"{c['skip_frac_repeat'] - 0.5:+.2f}" for c in cals_thin) + "≥0.15")

    g4, g2 = g, run_group(8, SEEDS[0], RW_THIN, cals_thin[0]["c_tau"], RS)
    chk("A26D-8", g4["T"] == 560 and g2["T"] == 280 and g4["arms"]["A"]["issued"] == 560
        and g2["arms"]["A"]["issued"] == 280
        and all(g4["arms"][a]["W_unchanged"] and g2["arms"][a]["W_unchanged"] for a in "ABP"),
        f"写步：主组 T={g4['T']}、少写组 T={g2['T']}；A 实发 {g4['arms']['A']['issued']}/{g2['arms']['A']['issued']}；"
        f"读不改写=True")

    rho_stab = 1 - ETA * (1.0 * S_KEY + 0.05 / 2)
    cN = make_core(D_MODEL, SEEDS[0], 8); W0n = cN._W().detach().clone()
    kN, _ = drive(cN, D0, "NONE")
    chk("A26D-9", rho_stab > 0 and kN == 0 and float((cN._W().detach() - W0n).norm()) == 0.0,
        f"稳定性 ρ={rho_stab:.4f}>0；零发放臂 {kN} 次、‖ΔW‖=0")

    chk("A26D-10", g4["kP_eq_kB"] and g2["kP_eq_kB"],
        f"P 臂等量：主组 k_P=k_B={g4['arms']['B']['issued']}；少写组 k_P=k_B={g2['arms']['B']['issued']}")

    cb = make_core(D_MODEL, SEEDS[0], 8); drive(cb, D0, "B", tau=cals[0]["c_tau"])
    Pb = cb._W().detach() - cb.W0
    si = [ret_jitter(s * Pb, RS["0.3"]["ho4"]["K"], D0["vb"], N_DIST, R_HO4)["ret1_mean"]
          for s in (0.37, 1.0, 2.5)]
    chk("A26D-11", max(si) - min(si) == 0.0, f"护栏（θ_r=0.3）：纯标量不变性 c∈{{0.37,1,2.5}} = {si}")
    return cals, cals_thin, RS


# ══════════════════════════════════════════════ 判据
def D_of(rows, r0, cos_r, agg="mean4"):
    ta, tb = {}, {}
    d = []
    for g in rows:
        if g["r0"] != r0:
            continue
        d.append(g["arms"]["B"]["reads"][f"{cos_r:.1f}"][agg] - g["arms"]["P"]["reads"][f"{cos_r:.1f}"][agg])
    return (float(np.mean(d)) if d else float("nan"),
            float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"), d)


def criteria(rows, thin):
    print("=== 判据（rev 1.0 冻结阈值） ===")
    v = {}
    spread = {}
    for r0 in R0_GRID:
        vals = []
        for g in rows:
            if g["r0"] == r0:
                vals.append([g["arms"][a]["reads"]["0.9"]["mean4"] for a in "ABP"])
        spread[r0] = float(np.mean([max(x) - min(x) for x in vals]))
    ok_g = spread[8] >= THR_DEGEN
    v["g_anchor"] = {"ok": ok_g, "detail": spread, "threshold": THR_DEGEN}

    d = {r0: {f"{c:.1f}": D_of(rows, r0, c) for c in COS_R} for r0 in R0_GRID}
    v["a_alive"] = {"ok": d[8]["0.7"][0] >= THR_MAIN, "d": d[8]["0.7"][0], "sd": d[8]["0.7"][1],
                    "threshold": THR_MAIN, "cell": "r=8, cosθ_r=0.7"}
    v["b_zero"] = {"ok": d[8]["0.3"][0] <= THR_ZERO, "d": d[8]["0.3"][0], "threshold": THR_ZERO,
                   "cell": "r=8, cosθ_r=0.3"}
    mono = {}
    for r0 in R0_GRID:
        seq = [d[r0][f"{c:.1f}"][0] for c in COS_R]
        bad = [(f"{COS_R[i]:.1f}→{COS_R[i+1]:.1f}", seq[i + 1] - seq[i]) for i in range(len(seq) - 1)
               if seq[i + 1] - seq[i] > THR_MONO]
        mono[r0] = {"seq": seq, "violations": bad, "ok": not bad}
    v["c_mono"] = {"ok": mono[8]["ok"] and mono[12]["ok"], "detail": mono, "tolerance": THR_MONO}
    npos = int(sum(x > 0 for x in d[8]["0.7"][2]))
    v["d_sign"] = {"ok": npos == len(d[8]["0.7"][2]), "detail": f"{npos}/{len(d[8]['0.7'][2])}"}

    D35 = json.load(open(D35_JSON))
    sp = [r for r in D35["rows"] if r["r0"] == 8]
    d35 = float(np.mean([r["cells"]["B0.40"]["Jout"]["ret1_mean"] - r["cells"]["P0.40"]["Jout"]["ret1_mean"]
                         for r in sp]))
    bit = abs(d[8]["0.9"][0] - d35) == 0.0
    v["e_anchor"] = {"ok": bit, "measured": d[8]["0.9"][0], "d35": d35,
                     "note": "同装置同读流 ⇒ 应逐位相同"}
    dm8 = D_of(rows, 8, 0.7, "maj8")[0]
    v["f_maj"] = {"ok": abs(dm8 - d[8]["0.7"][0]) <= THR_AGREE and (dm8 > 0) == (d[8]["0.7"][0] > 0),
                  "mean4": d[8]["0.7"][0], "maj8": dm8, "threshold": THR_AGREE}

    dt = []
    for g in thin:
        if g["r0"] == 8:
            dt.append(g["arms"]["B"]["reads"]["0.7"]["maj8"] - g["arms"]["P"]["reads"]["0.7"]["maj8"])
    v["g_thin"] = {"ok": float(np.mean(dt)) >= THR_MAIN, "d": float(np.mean(dt)),
                   "sd": float(np.std(dt, ddof=1)), "n": len(dt), "threshold": THR_MAIN,
                   "cell": "R_w=2, r_q=8(≥5), cosθ_r=0.7, r=8"}

    h = {}
    for r0 in R0_GRID:
        for c in (0.9, 0.3):
            h[f"{r0}@{c:.1f}"] = {a: float(np.mean([g["arms"][a]["reads"][f"{c:.1f}"]["mean4"]
                                                   for g in rows if g["r0"] == r0])) for a in "ABP"}
    v["h_chance"] = {"ok": True, "detail": h, "chance": 1.0 / N_DIST, "note": "报告：塌到随机 vs 持平但高于随机"}

    if not ok_g:
        main = "INDETERMINATE"
    elif v["a_alive"]["ok"] and v["b_zero"]["ok"] and v["c_mono"]["ok"] and v["d_sign"]["ok"] \
            and v["e_anchor"]["ok"] and v["f_maj"]["ok"] and v["g_thin"]["ok"]:
        main = "CONFIRMED"
    elif not v["a_alive"]["ok"] or not v["b_zero"]["ok"]:
        main = "FALSIFIED"
    else:
        main = "WEAK"
    v["main"] = {"ok": main == "CONFIRMED", "verdict": main,
                 "falsify_dir": ("方向一（归零更早）" if not v["a_alive"]["ok"] else
                                 ("方向二（归零更晚/不存在）" if not v["b_zero"]["ok"] else None))}
    for k in ("g_anchor", "a_alive", "b_zero", "c_mono", "d_sign", "e_anchor", "f_maj", "g_thin"):
        print(f"  [{'PASS' if v[k]['ok'] else 'FAIL'}] {k}  {v[k].get('detail', '')}")
    print(f"  ⇒ H26D 判定：**{main}**")
    return v


# ══════════════════════════════════════════════ 主
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26Dr10")
    args = ap.parse_args()
    t0 = time.time()
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26d/out/{args.tag}_{args.stage}.json"

    if args.stage == "stageA":
        stage_a()
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}"
              f"（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"], res["passed"] = "stageA", ok
    else:
        r0s = (8,) if args.stage == "smoke" else R0_GRID
        seeds = [SEEDS[0]] if args.stage == "smoke" else SEEDS
        cal_main = {r0: calibrate_at(r0, R_MAIN, RW_MAIN) for r0 in r0s}
        cal_thin = {r0: calibrate_at(r0, R_MAIN, RW_THIN) for r0 in r0s}
        rows, thin = [], []
        print(f"=== {args.stage}（{len(r0s) * len(seeds)} 格点 × 2 组） ===")
        for r0 in r0s:
            for s in seeds:
                RS = read_streams(s)
                g4 = run_group(r0, s, RW_MAIN, cal_main[r0]["c_tau"], RS)
                g2 = run_group(r0, s, RW_THIN, cal_thin[r0]["c_tau"], RS)
                rows.append(g4); thin.append(g2)
                if args.stage == "formal":
                    os.makedirs(os.path.dirname(outp), exist_ok=True)
                    json.dump({"rows": rows, "thin": thin}, open(outp, "w"),
                              ensure_ascii=False, indent=1, default=float)      # ★ 纪律 (xi)：先落盘
                dl = lambda g, c, agg="mean4": (g["arms"]["B"]["reads"][c][agg]
                                                - g["arms"]["P"]["reads"][c][agg])
                print(f"  r={r0:2d} seed={s} | Δ main " +
                      " ".join(f"{c:.1f}:{dl(g4, f'{c:.1f}'):+.3f}" for c in COS_R) +
                      f" | maj8 0.7:{dl(g4, '0.7', 'maj8'):+.3f}" +
                      f" | Δ thin(R_w=2,maj8) 0.9:{dl(g2, '0.9', 'maj8'):+.3f} 0.7:{dl(g2, '0.7', 'maj8'):+.3f}" +
                      f" | k_B {g4['arms']['B']['issued']}/{g2['arms']['B']['issued']}")
        res["rows"], res["thin"] = rows, thin
        if args.stage == "formal":
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
            print(f"  [落盘] rows/thin 已写入 {outp}（判据前）")
            try:
                res["verdicts"] = criteria(rows, thin)
                res["passed"] = res["verdicts"]["main"]["ok"]
            except Exception as e:
                res["verdicts"] = {"error": f"{type(e).__name__}: {e}"}
                res["passed"] = False
                print(f"  ✗ 判据异常：{type(e).__name__}: {e}")
            json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)

    res["meta"] = {"wall_seconds": round(time.time() - t0, 1), "device": "cpu", "d": D_MODEL,
                   "n_dist": N_DIST, "eta": ETA, "m": M, "s_key": S_KEY, "r_grid": list(R0_GRID),
                   "r_w_main": RW_MAIN, "r_w_thin": RW_THIN, "r_ho": [R_HO4, R_HO8],
                   "cos_read": list(COS_R), "rho_main": R_MAIN, "seeds": SEEDS,
                   "seed_pilot": SEED_PILOT, "stage": args.stage, "card": "DIR-26D rev1.0"}
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
