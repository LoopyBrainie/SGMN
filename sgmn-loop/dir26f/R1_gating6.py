"""DIR-26-F · R1_gating6.py —— 内容门控的收益与负债：来自「内容选择」还是「发放不均」？

卡片：`revolve/DIR-26F/DIRECTION_rev26f_v10.md`（rev 1.0 冻结）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，零改动）
★ 装置/测量层/定标/抖动指标/错位角 全部 import：
   `dir26/R1_gating.py`、`dir26b/R1_gating2.py`、`dir26c/R1_gating3.py`、`dir26e/R1_gating5.py`

2×2 四臂 + A（全写）：
  B = 内容门控（`drive(..., "B", tau)`）· E = B-even（内容排序 + 均分配额）
  U = P-uneven（B 的逐条目计数 + 位置选轮次）· P = 位置均衡（`keep_positions`）
判据量：(a) |δ_sel|even| 与 |δ_sel|uneven| ≤0.05（5 格点共 10 个）；(b) 记账恒等 + |δ_cnt|content − δ_cnt|pos| ≤0.10
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

for _p in ("dir26f", "dir26e", "dir26d", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating2 as G2
import R1_gating3 as G3
import R1_gating5 as G5
from R1_gating import (D_MODEL, N_DIST, ETA, M, S_KEY, build_stream, drive, make_core, metrics_plas)
from R1_gating2 import keep_positions, item_counts, calibrate
from R1_gating3 import ret_jitter, ret_jitter_bruteforce
from R1_gating5 import calibrate_w, theta_of

R_PRES = 4
R_HO4, R_HO8 = 4, 8
HO_OFFSET = 999
R_MAIN = 0.40
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]
ARMS = ["A", "B", "E", "P", "U"]
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
THR_SEL = 0.05          # (a) 选择主效应"不可分辨"阈值 = 2× 最大 SEM(0.016~0.035)
THR_FALS = 0.10         # 证伪阈值 = 健康档主效应量级
THR_CNT = 0.10          # (b2) 两个计数主效应之差
THR_PRE = 0.10          # (c)① 健康档 Δ(B−P) ≥ +0.10
THR_PEN = 0.03          # (c)② 惩罚子格 Δ(B−P) ≤ −0.03
D35 = "/home/ec2-user/sgmn-loop/dir26c/out/DIR26Cr11_formal.json"
D36 = "/home/ec2-user/sgmn-loop/dir26d/out/DIR26Dr10_formal.json"
D37 = "/home/ec2-user/sgmn-loop/dir26e/out/DIR26Er10_formal.json"

# 六格点：前 5 个是判据格点，NR12 只报告（无惩罚可解释）
CELLS = [
    {"tag": "H8",   "r0": 8,  "cr": 0.9, "cw": 0.9, "judge": True,  "ref": +0.1500},
    {"tag": "H12",  "r0": 12, "cr": 0.9, "cw": 0.9, "judge": True,  "ref": +0.1586},
    {"tag": "PR8",  "r0": 8,  "cr": 0.5, "cw": 0.9, "judge": True,  "ref": -0.0657},
    {"tag": "PW8",  "r0": 8,  "cr": 0.9, "cw": 0.3, "judge": True,  "ref": -0.0386},
    {"tag": "PW12", "r0": 12, "cr": 0.9, "cw": 0.3, "judge": True,  "ref": -0.0543},
    {"tag": "NR12", "r0": 12, "cr": 0.5, "cw": 0.9, "judge": False, "ref": +0.0086},
]
res = {"checks": [], "rows": [], "calib": []}


# ══════════════════════════════════════════════ 臂的构造（卡片 §3 冻结）
def _pos_idxs(c_i, per_item_tot):
    """与 `R1_gating2.keep_positions` **同款**的条目内位置规则（np.linspace 均匀取点）。"""
    c_i = max(int(c_i), 0)
    if c_i == 0:
        return []
    idxs = sorted({int(round(x)) for x in np.linspace(0, per_item_tot - 1, c_i)})
    j = 0
    while len(idxs) < c_i:
        if j not in idxs:
            idxs.append(j)
            idxs.sort()
        j += 1
    return idxs[:c_i]


def _idx_of(pos, i, n):
    from R1_gating import stream_idx
    rho, rest = divmod(pos, 2 * M)
    mbr, m = divmod(rest, M)
    return stream_idx(rho, i, mbr, m, n)


def build_even(n, r_pres, k_B, log_B):
    """E 臂 keep 集合：逐条目 $q_i\\in\\{base,base+1\\}$、$\\sum q_i=k_B$；条目内按 **B 的 $c_{now}$ 升序**取前 $q_i$。
    余数 $r_B$ 个 $+1$ 名额给"下一候选 $c_{now}$ 最低"的条目（内容决定谁多拿一个）。"""
    per_item_tot = r_pres * 2 * M
    by_item = {i: [] for i in range(n)}
    for t, e in enumerate(log_B):
        by_item[e["item"]].append((float(e["cos"]), t))
    for i in by_item:
        by_item[i].sort()
    base, rem = divmod(int(k_B), n)
    cand = sorted(range(n), key=lambda i: by_item[i][base][0] if base < len(by_item[i]) else -1e9)
    extra = set(cand[:rem])
    keep = set()
    for i in range(n):
        q_i = base + (1 if i in extra else 0)
        for _, t in by_item[i][:q_i]:
            keep.add(t)
    return keep


def build_uneven(n, r_pres, counts_B):
    """U 臂 keep 集合：逐条目计数 = B 的逐条目计数（逐位），条目内用**位置规则**选轮次。"""
    per_item_tot = r_pres * 2 * M
    keep = set()
    for i, c_i in enumerate(counts_B):
        for pos in _pos_idxs(c_i, per_item_tot):
            keep.add(_idx_of(pos, i, n))
    return keep


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


# ══════════════════════════════════════════════ 读数
def read_streams(seed, cos_r):
    th = theta_of(cos_r)
    return {"ho4": build_stream(seed + HO_OFFSET, D_MODEL, N_DIST, R_HO4, th),
            "ho8": build_stream(seed + HO_OFFSET, D_MODEL, N_DIST, R_HO8, th)}


def run_cell(cell, seed, tau, RS):
    """一个 (格点, 种子)：五臂同流同种子同 $W^0$，唯一差别是 keep/门控规则。"""
    th = theta_of(cell["cw"])
    D = build_stream(seed, D_MODEL, N_DIST, R_PRES, th)
    out = {"tag": cell["tag"], "r0": cell["r0"], "cr": cell["cr"], "cw": cell["cw"],
           "seed": seed, "T": R_PRES * N_DIST * 2 * M, "arms": {}}

    def rec(tag, core, issued):
        W = core._W().detach()
        W_before = W.clone()
        P = W - core.W0
        m4 = ret_jitter(P, RS["ho4"]["K"], D["vb"], N_DIST, R_HO4)
        m8 = ret_jitter(P, RS["ho8"]["K"], D["vb"], N_DIST, R_HO8)
        out["arms"][tag] = {"issued": issued, "clean": metrics_plas(W, core.W0, D["proto"], D["vb"], N_DIST)["ret1"],
                            "ho4_mean": m4["ret1_mean"], "ho4_maj": m4["ret1_maj"],
                            "ho8_mean": m8["ret1_mean"], "ho8_maj": m8["ret1_maj"],
                            "maj_need4": m4["maj_need"], "maj_need8": m8["maj_need"],
                            "W_unchanged": bool(torch.equal(core._W().detach(), W_before))}

    cA = make_core(D_MODEL, seed, cell["r0"]); kA, _ = drive(cA, D, "A"); rec("A", cA, kA)
    cB = make_core(D_MODEL, seed, cell["r0"]); kB, logB = drive(cB, D, "B", tau=tau); rec("B", cB, kB)
    keep_B = {t for t, e in enumerate(logB) if e["issued"]}
    cnt_B = item_counts(keep_B, N_DIST)

    keepE = build_even(N_DIST, R_PRES, kB, logB)
    cE = make_core(D_MODEL, seed, cell["r0"]); kE, _ = drive(cE, D, "R", keep=keepE); rec("E", cE, kE)

    keepP = keep_positions(N_DIST, R_PRES, kB)
    cP = make_core(D_MODEL, seed, cell["r0"]); kPu, _ = drive(cP, D, "R", keep=keepP); rec("P", cP, kPu)

    keepU = build_uneven(N_DIST, R_PRES, cnt_B)
    cU = make_core(D_MODEL, seed, cell["r0"]); kU, _ = drive(cU, D, "R", keep=keepU); rec("U", cU, kU)

    out["counts"] = {"B": cnt_B, "E": item_counts(keepE, N_DIST), "U": item_counts(keepU, N_DIST),
                     "P": item_counts(keepP, N_DIST)}
    out["jac"] = {"EP": jaccard(keepE, keepP), "BE": jaccard(keep_B, keepE),
                  "BU": jaccard(keep_B, keepU), "UP": jaccard(keepU, keepP)}
    out["k"] = {"A": kA, "B": kB, "E": kE, "P": kPu, "U": kU}
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


# ══════════════════════════════════════════════ Stage A
def stage_a():
    print("=== Stage A（A26F-1 … A26F-11） ===")
    dev = "/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py"
    chk("A26F-1", md5(dev) == DEV_MD5, f"装置 md5 {md5(dev)} == {DEV_MD5}")

    m = {}
    for n in ("unit", "make_core", "build_stream", "drive", "metrics_plas", "stream_idx"):
        m[n] = getattr(G, n, None).__module__
    for n in ("keep_positions", "item_counts", "calibrate"):
        m[n] = getattr(G2, n, None).__module__
    for n in ("ret_jitter", "ret_jitter_bruteforce"):
        m[n] = getattr(G3, n, None).__module__
    for n in ("calibrate_w", "theta_of"):
        m[n] = getattr(G5, n, None).__module__
    want = {"R1_gating": ("unit", "make_core", "build_stream", "drive", "metrics_plas", "stream_idx"),
            "R1_gating2": ("keep_positions", "item_counts", "calibrate"),
            "R1_gating3": ("ret_jitter", "ret_jitter_bruteforce"),
            "R1_gating5": ("calibrate_w", "theta_of")}
    bad = [n for mod, ns in want.items() for n in ns if m[n] != mod]
    chk("A26F-2", not bad, f"装置/测量/定标/抖动指标/错位角 全为 import 的同一对象：异常 {bad or '无'}")

    # A26F-3/4：四臂构造审计（最小实例 N=8 + 真实 k_B 比例）
    n8, seed8 = 8, SEEDS[0]
    D8 = build_stream(seed8, D_MODEL, n8, R_PRES, theta_of(0.9))
    c8 = make_core(D_MODEL, seed8, 8)
    kB8, logB8 = drive(c8, D8, "B", tau=calibrate(8, theta_of(0.9))["c_tau"])
    cnt8 = item_counts({t for t, e in enumerate(logB8) if e["issued"]}, n8)
    keepE8 = build_even(n8, R_PRES, kB8, logB8)
    keepU8 = build_uneven(n8, R_PRES, cnt8)
    keepP8 = keep_positions(n8, R_PRES, kB8)
    cntE8, cntU8, cntP8 = item_counts(keepE8, n8), item_counts(keepU8, n8), item_counts(keepP8, n8)
    ok3 = (sum(cntE8) == kB8 and max(cntE8) - min(cntE8) <= 1
           and cntU8 == cnt8 and sum(cntU8) == kB8 and max(cntP8) - min(cntP8) <= 1
           and jaccard(keepE8, keepP8) < 0.9)
    chk("A26F-3", ok3, f"k_B={kB8}：E 和={sum(cntE8)} 极差={max(cntE8) - min(cntE8)}；"
        f"U 与 B 逐条目相同={cntU8 == cnt8}；P 极差={max(cntP8) - min(cntP8)}；"
        f"Jaccard(E,P)={jaccard(keepE8, keepP8):.3f}（须 <0.9 ⇒ E 不是 P 的复制品）")

    # A26F-4：E 确实按内容升序取（保留轮次的 c_max ≤ 丢弃轮次的 c_min）
    viol, gaps = [], []
    for i in range(n8):
        items = sorted([(float(e["cos"]), t) for t, e in enumerate(logB8) if e["item"] == i])
        q_i = cntE8[i]
        if q_i == 0 or q_i >= len(items):
            continue
        if max(c for c, _ in items[:q_i]) > min(c for c, _ in items[q_i:]):
            viol.append(i)
        gaps.append(min(c for c, _ in items[q_i:]) - max(c for c, _ in items[:q_i]))
    chk("A26F-4", not viol, f"E 逐条目「保留的 c_max ≤ 丢弃的 c_min」：违反 {viol or '无'}；"
        f"平均 gap {float(np.mean(gaps)):.4f}")

    # A26F-10：读流与臂无关；对齐度
    RS8 = read_streams(SEEDS[0], 0.9)
    a9 = float(torch.nn.functional.cosine_similarity(RS8["ho4"]["K"], RS8["ho4"]["proto"], dim=-1).mean())
    RS5 = read_streams(SEEDS[0], 0.5)
    a5 = float(torch.nn.functional.cosine_similarity(RS5["ho4"]["K"], RS5["ho4"]["proto"], dim=-1).mean())
    chk("A26F-10", abs(a9 - 0.9) <= 0.01 and abs(a5 - 0.5) <= 0.01,
        f"读流对齐度 cos(ho4,原型)：0.9 档 {a9:.4f}、0.5 档 {a5:.4f}（目标 0.9000 / 0.5000）")

    # A26F-6：三锚点逐位复现（1 种子；DEF-26E-2 最小实例）
    s0 = SEEDS[0]
    cals = {(8, 0.9): calibrate_w(8, R_MAIN, R_PRES, theta_of(0.9))["c_tau"],
            (8, 0.3): calibrate_w(8, R_MAIN, R_PRES, theta_of(0.3))["c_tau"]}
    gH8 = run_cell(CELLS[0], s0, cals[(8, 0.9)], read_streams(s0, 0.9))
    gPR8 = run_cell(CELLS[2], s0, cals[(8, 0.9)], read_streams(s0, 0.5))
    gPW8 = run_cell(CELLS[3], s0, cals[(8, 0.3)], read_streams(s0, 0.9))
    d35 = [r for r in json.load(open(D35))["rows"] if r["r0"] == 8 and r["seed"] == s0][0]
    ok35 = all(gH8["arms"][a]["ho4_mean"] == d35["cells"][t]["Jout"]["ret1_mean"]
               and gH8["arms"][a]["ho4_maj"] == d35["cells"][t]["Jout"]["ret1_maj"]
               and gH8["arms"][a]["clean"] == d35["cells"][t]["clean"]
               for a, t in (("A", "A"), ("B", "B0.40"), ("P", "P0.40")))
    d36 = [r for r in json.load(open(D36))["rows"] if r["r0"] == 8 and r["seed"] == s0][0]
    ok36 = all(gPR8["arms"][a]["ho4_mean"] == d36["arms"][a]["reads"]["0.5"]["mean4"]
               and gPR8["arms"][a]["ho8_maj"] == d36["arms"][a]["reads"]["0.5"]["maj8"]
               for a in "BP")
    d37 = [r for r in json.load(open(D37))["rows"] if r["r0"] == 8 and r["seed"] == s0 and abs(r["cos_w"] - 0.3) < 1e-9][0]
    ok37w = all(gPW8["arms"][a]["ho4_mean"] == d37["arms"][a]["ho4_mean"]
                and gPW8["arms"][a]["ho8_maj"] == d37["arms"][a]["ho8_maj"] for a in "BP")
    gPW12 = run_cell(CELLS[4], s0, calibrate_w(12, R_MAIN, R_PRES, theta_of(0.3))["c_tau"], read_streams(s0, 0.9))
    d37b = [r for r in json.load(open(D37))["rows"] if r["r0"] == 12 and r["seed"] == s0 and abs(r["cos_w"] - 0.3) < 1e-9][0]
    ok37b = all(gPW12["arms"][a]["ho4_mean"] == d37b["arms"][a]["ho4_mean"]
                and gPW12["arms"][a]["ho8_maj"] == d37b["arms"][a]["ho8_maj"] for a in "BP")
    chk("A26F-6", ok35 and ok36 and ok37w and ok37b,
        f"三锚点逐位复现：H8 vs D-35={ok35}（B ho4 {gH8['arms']['B']['ho4_mean']:.4f}）、"
        f"PR8 vs D-36={ok36}、PW8 vs D-37={ok37w}（B {gPW8['arms']['B']['ho4_mean']:.4f}）、"
        f"PW12 vs D-37={ok37b}")

    # A26F-5：记账恒等（真实格点，1 种子）
    def dl(a, b):
        return gH8["arms"][a]["ho4_mean"] - gH8["arms"][b]["ho4_mean"]
    lhs = dl("B", "P")
    r1 = (dl("B", "E")) + (dl("E", "P"))
    r2 = (dl("B", "U")) + (dl("U", "P"))
    chk("A26F-5", abs(lhs - r1) <= 1e-12 and abs(lhs - r2) <= 1e-12,
        f"Δ(B−P)={lhs:.6f} = δ_cnt|content+δ_sel|even={r1:.6f} = δ_sel|uneven+δ_cnt|pos={r2:.6f}")

    # A26F-7：抖动指标逐位正确
    d16, n16 = 16, 8
    D16 = build_stream(SEEDS[0], d16, n16, R_PRES, theta_of(0.9))
    c16 = make_core(d16, SEEDS[0], 4); drive(c16, D16, "A")
    P16 = c16._W().detach() - c16.W0
    R16a = build_stream(SEEDS[0] + HO_OFFSET, d16, n16, R_HO4, theta_of(0.9))
    R16b = build_stream(SEEDS[0] + HO_OFFSET, d16, n16, R_HO8, theta_of(0.9))
    a4 = ret_jitter(P16, R16a["K"], D16["vb"], n16, R_HO4)
    b4 = ret_jitter_bruteforce(P16, R16a["K"], D16["vb"], n16, R_HO4)
    a8 = ret_jitter(P16, R16b["K"], D16["vb"], n16, R_HO8)
    b8 = ret_jitter_bruteforce(P16, R16b["K"], D16["vb"], n16, R_HO8)
    chk("A26F-7", a4["ret1_mean"] == b4["ret1_mean"] and a4["ret1_maj"] == b4["ret1_maj"]
        and a8["ret1_mean"] == b8["ret1_mean"] and a8["ret1_maj"] == b8["ret1_maj"]
        and a4["maj_need"] == 3 and a8["maj_need"] == 5,
        f"抖动指标逐位：r_q=4 mean {a4['ret1_mean']:.4f}/maj {a4['ret1_maj']:.4f}（need 3）；"
        f"r_q=8 mean {a8['ret1_mean']:.4f}/maj {a8['ret1_maj']:.4f}（need 5）")

    # A26F-8：护栏（U 臂上纯标量不变性）
    Dn = build_stream(s0, D_MODEL, N_DIST, R_PRES, theta_of(0.9))
    cU = make_core(D_MODEL, s0, 8)
    drive(cU, Dn, "R", keep=build_uneven(N_DIST, R_PRES, gH8["counts"]["B"]))
    PU = cU._W().detach() - cU.W0
    si = [ret_jitter(s * PU, RS8["ho4"]["K"], Dn["vb"], N_DIST, R_HO4)["ret1_mean"] for s in (0.37, 1.0, 2.5)]
    chk("A26F-8", max(si) - min(si) == 0.0, f"护栏（U 臂）：纯标量不变性 c∈{{0.37,1,2.5}} = {si}")

    # A26F-9：稳定性 + 零发放
    rho_stab = 1 - ETA * (1.0 * S_KEY + 0.05 / 2)
    cN = make_core(D_MODEL, s0, 8); W0n = cN._W().detach().clone()
    kN, _ = drive(cN, Dn, "NONE")
    chk("A26F-9", rho_stab > 0 and kN == 0 and float((cN._W().detach() - W0n).norm()) == 0.0,
        f"稳定性 ρ={rho_stab:.4f}>0；零发放臂 {kN} 次、‖ΔW‖=0")

    # A26F-11：A 臂实发 = T；读不改写
    chk("A26F-11", gH8["arms"]["A"]["issued"] == gH8["T"] == 560
        and all(gH8["arms"][a]["W_unchanged"] for a in ARMS),
        f"A 臂实发 {gH8['arms']['A']['issued']}=T={gH8['T']}；五臂读后 W 逐位不变="
        f"{all(gH8['arms'][a]['W_unchanged'] for a in ARMS)}")
    res["calib"] = [{"r0": r0, "cw": cw, "c_tau": t} for (r0, cw), t in cals.items()]
    return gH8


# ══════════════════════════════════════════════ 判据
def crit(rows):
    print("=== 判据（rev 1.0 冻结阈值） ===")

    def mean_of(tag, f, arr=None):
        xs = [f(g) for g in (arr if arr is not None else rows) if g["tag"] == tag]
        return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")), xs

    D = {}
    for c in CELLS:
        tg = c["tag"]
        dBP, sdBP, rawBP = mean_of(tg, lambda g: g["arms"]["B"]["ho4_mean"] - g["arms"]["P"]["ho4_mean"])
        dE_P, sdE_P, rawE = mean_of(tg, lambda g: g["arms"]["E"]["ho4_mean"] - g["arms"]["P"]["ho4_mean"])
        dB_U, sdB_U, rawU = mean_of(tg, lambda g: g["arms"]["B"]["ho4_mean"] - g["arms"]["U"]["ho4_mean"])
        dB_E, _, _ = mean_of(tg, lambda g: g["arms"]["B"]["ho4_mean"] - g["arms"]["E"]["ho4_mean"])
        dU_P, _, _ = mean_of(tg, lambda g: g["arms"]["U"]["ho4_mean"] - g["arms"]["P"]["ho4_mean"])
        iden = max(abs(dBP - ((dB_E) + dE_P)), abs(dBP - (dB_U + dU_P)))
        sel_even_maj, _, _ = mean_of(tg, lambda g: g["arms"]["E"]["ho8_maj"] - g["arms"]["P"]["ho8_maj"])
        sel_uneven_maj, _, _ = mean_of(tg, lambda g: g["arms"]["B"]["ho8_maj"] - g["arms"]["U"]["ho8_maj"])
        D[tg] = {"r0": c["r0"], "cr": c["cr"], "cw": c["cw"], "judge": c["judge"], "ref": c["ref"],
                 "dBP": dBP, "sdBP": sdBP, "n_pos_BP": int(sum(x > 0 for x in rawBP)),
                 "sel_even": dE_P, "sd_sel_even": sdE_P, "n_pos_sel_even": int(sum(x > 0 for x in rawE)),
                 "sel_uneven": dB_U, "sd_sel_uneven": sdB_U, "n_pos_sel_uneven": int(sum(x > 0 for x in rawU)),
                 "cnt_content": dB_E, "cnt_pos": dU_P, "identity_err": iden,
                 "sel_even_maj": sel_even_maj, "sel_uneven_maj": sel_uneven_maj}

    v = {}
    # (c) 前置复现 + 逐位锚点（锚点在 Stage A 已核 1 种子；此处核 5 种子）
    d35 = {(r["r0"], r["seed"]): r for r in json.load(open(D35))["rows"]}
    d36 = {(r["r0"], r["seed"]): r for r in json.load(open(D36))["rows"]}
    d37 = {(r["r0"], r["seed"], round(r["cos_w"], 1)): r for r in json.load(open(D37))["rows"]}
    ok_anchor, anc = True, {}
    for g in rows:
        s, r0 = g["seed"], g["r0"]
        if g["tag"] == "H8":
            ref = d35[(8, s)]["cells"]
            good = all(g["arms"][a]["ho4_mean"] == ref[t]["Jout"]["ret1_mean"]
                       and g["arms"][a]["ho4_maj"] == ref[t]["Jout"]["ret1_maj"] for a, t in (("A", "A"), ("B", "B0.40"), ("P", "P0.40")))
        elif g["tag"] == "H12":
            ref = d35[(12, s)]["cells"]
            good = all(g["arms"][a]["ho4_mean"] == ref[t]["Jout"]["ret1_mean"]
                       and g["arms"][a]["ho4_maj"] == ref[t]["Jout"]["ret1_maj"] for a, t in (("A", "A"), ("B", "B0.40"), ("P", "P0.40")))
        elif g["tag"] in ("PR8", "NR12"):
            ref = d36[(r0, s)]["arms"]
            good = all(g["arms"][a]["ho4_mean"] == ref[a]["reads"]["0.5"]["mean4"]
                       and g["arms"][a]["ho8_maj"] == ref[a]["reads"]["0.5"]["maj8"] for a in "BP")
        else:
            ref = d37[(r0, s, 0.3)]["arms"]
            good = all(g["arms"][a]["ho4_mean"] == ref[a]["ho4_mean"]
                       and g["arms"][a]["ho8_maj"] == ref[a]["ho8_maj"] for a in "BP")
        anc[f"{g['tag']}@{s}"] = good
        ok_anchor = ok_anchor and good
    pre_pos = all(D[t]["dBP"] >= THR_PRE for t in ("H8", "H12"))
    pre_neg = all(D[t]["dBP"] <= -THR_PEN for t in ("PR8", "PW8", "PW12"))
    v["c_pre"] = {"ok": bool(pre_pos and pre_neg and ok_anchor), "anchor": ok_anchor,
                  "healthy": {t: round(D[t]["dBP"], 4) for t in ("H8", "H12")},
                  "penalty": {t: round(D[t]["dBP"], 4) for t in ("PR8", "PW8", "PW12")}}
    v["c_anchor"] = {"ok": bool(ok_anchor), "detail": anc}

    judge = [c["tag"] for c in CELLS if c["judge"]]
    sel = [{"tag": t, "sel_even": round(D[t]["sel_even"], 4), "sel_uneven": round(D[t]["sel_uneven"], 4),
            "ok": abs(D[t]["sel_even"]) <= THR_SEL and abs(D[t]["sel_uneven"]) <= THR_SEL,
            "fals": abs(D[t]["sel_even"]) >= THR_FALS or abs(D[t]["sel_uneven"]) >= THR_FALS} for t in judge]
    v["a_sel"] = {"ok": all(x["ok"] for x in sel), "detail": sel, "threshold": THR_SEL,
                  "n_ok": sum(x["ok"] for x in sel), "n": len(sel)}
    v["a_fals"] = {"hit": any(x["fals"] for x in sel), "detail": [x["tag"] for x in sel if x["fals"]]}

    v["b1_identity"] = {"ok": all(D[t]["identity_err"] <= 1e-12 for t in D),
                        "max_err": max(D[t]["identity_err"] for t in D)}
    b2 = [{"tag": t, "cnt_content": round(D[t]["cnt_content"], 4), "cnt_pos": round(D[t]["cnt_pos"], 4),
           "diff": round(abs(D[t]["cnt_content"] - D[t]["cnt_pos"]), 4),
           "ok": abs(D[t]["cnt_content"] - D[t]["cnt_pos"]) <= THR_CNT} for t in judge]
    v["b2_cnt"] = {"ok": all(x["ok"] for x in b2), "detail": b2, "threshold": THR_CNT}

    fm = [{"tag": t, "mean4": round(D[t]["sel_even"], 4), "maj8": round(D[t]["sel_even_maj"], 4),
           "same_sign": (D[t]["sel_even"] > 0) == (D[t]["sel_even_maj"] > 0),
           "diff": round(abs(D[t]["sel_even"] - D[t]["sel_even_maj"]), 4),
           "ok": ((D[t]["sel_even"] > 0) == (D[t]["sel_even_maj"] > 0)) and abs(D[t]["sel_even"] - D[t]["sel_even_maj"]) <= 0.10}
          for t in judge]
    v["f_maj"] = {"ok": all(x["ok"] for x in fm), "detail": fm}

    v["e_reproduce"] = {"note": "位置不均臂是否复现 Δ(B−P)（报告）",
                        "detail": {t: {"cnt_pos_to_dBP": round(D[t]["cnt_pos"] - D[t]["dBP"], 4),
                                       "cnt_content_to_dBP": round(D[t]["cnt_content"] - D[t]["dBP"], 4)}
                                   for t in D}}

    if not v["c_anchor"]["ok"]:
        main = "INDETERMINATE"
    elif v["a_fals"]["hit"]:
        main = "FALSIFIED"
    elif v["a_sel"]["ok"] and v["b1_identity"]["ok"] and v["b2_cnt"]["ok"] and v["f_maj"]["ok"] and v["c_pre"]["ok"]:
        main = "CONFIRMED"
    else:
        main = "WEAK"
    v["main"] = {"ok": main == "CONFIRMED", "verdict": main}

    for k in ("c_pre", "c_anchor", "a_sel", "b1_identity", "b2_cnt", "f_maj"):
        print(f"  [{'PASS' if v[k]['ok'] else 'FAIL'}] {k}  {str(v[k].get('detail', v[k]))[:220]}")
    print(f"  (a) 选择主效应：{v['a_sel']['n_ok']}/{v['a_sel']['n']} 格点通过（阈值 {THR_SEL}）")
    for x in sel:
        print(f"    {x['tag']:5s} δ_sel|even={x['sel_even']:+.4f}  δ_sel|uneven={x['sel_uneven']:+.4f}  "
              f"{'ok' if x['ok'] else ('★FALS' if x['fals'] else '★边界')}")
    print(f"  ⇒ H26F 判定：**{main}**")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26Fr10")
    args = ap.parse_args()
    t0 = time.time()
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26f/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    if args.stage == "stageA":
        stage_a()
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}"
              f"（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"], res["passed"] = "stageA", ok
    else:
        cells = [CELLS[0]] if args.stage == "smoke" else CELLS
        seeds = [SEEDS[0]] if args.stage == "smoke" else SEEDS
        cals = {(r0, cw): calibrate_w(r0, R_MAIN, R_PRES, theta_of(cw))["c_tau"]
                for r0, cw in sorted({(c["r0"], c["cw"]) for c in cells})}
        res["calib"] = [{"r0": r0, "cw": cw, "c_tau": t} for (r0, cw), t in cals.items()]
        rows = []
        print(f"=== {args.stage}（{len(cells)} 格点 × {len(seeds)} 种子 × 5 臂） ===")
        for cell in cells:
            for s in seeds:
                RS = read_streams(s, cell["cr"])
                g = run_cell(cell, s, cals[(cell["r0"], cell["cw"])], RS)
                rows.append(g)
                json.dump({"rows": rows}, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
                a = g["arms"]
                print(f"  {cell['tag']:5s} seed={s} | ho4 A {a['A']['ho4_mean']:.3f} B {a['B']['ho4_mean']:.3f} "
                      f"E {a['E']['ho4_mean']:.3f} P {a['P']['ho4_mean']:.3f} U {a['U']['ho4_mean']:.3f} "
                      f"| δsel|even {a['E']['ho4_mean'] - a['P']['ho4_mean']:+.3f} "
                      f"δsel|uneven {a['B']['ho4_mean'] - a['U']['ho4_mean']:+.3f} "
                      f"| k {a['B']['issued']}/{a['E']['issued']}/{a['U']['issued']} "
                      f"| J(EP) {g['jac']['EP']:.2f}")
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
                   "n_dist": N_DIST, "eta": ETA, "m": M, "s_key": S_KEY, "r_pres": R_PRES,
                   "r_ho": [R_HO4, R_HO8], "rho_main": R_MAIN, "seeds": SEEDS, "arms": ARMS,
                   "cells": [{k: v for k, v in c.items()} for c in CELLS], "stage": args.stage,
                   "card": "DIR-26F rev1.0"}
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
