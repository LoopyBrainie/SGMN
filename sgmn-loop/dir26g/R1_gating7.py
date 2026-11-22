"""DIR-26-G · R1_gating7.py —— 写入剖面的形状：'偏重哪一轮'是独立因素，还是只与'计数不均'耦合才有意义？

卡片：`revolve/DIR-26G/DIRECTION_rev26g_v11.md`（rev 1.1 冻结）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，零改动）
★ 装置/测量层/定标/抖动指标/轮内位置规则 全部 import（同一对象）：
   `dir26/R1_gating.py`、`dir26b/R1_gating2.py`、`dir26c/R1_gating3.py`、`dir26e/R1_gating5.py`、`dir26f/R1_gating6.py`

六单元 = 剖面{前重 F, 均匀 U0, 后重 K} × 逐条目计数{不均 u, 均 e}，全部用 `drive(..., "R", keep)` 执行；
  轮内一律位置规则（关掉 D-38 的维度耦合，DEF-26F-2），⇒ 剖面成为唯一变量。
参照臂 A（全写）/ B（内容门控）/ P（`keep_positions`）只作逐位锚点，不进判据。
判据量：Δ(F−K) 的 4 个单元读数、(F−K) 的计数水平一致性、副口径一致。
"""
import argparse, hashlib, json, math, os, sys, time

import numpy as np
import torch

for _p in ("dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating2 as G2
import R1_gating3 as G3
import R1_gating5 as G5
import R1_gating6 as G6
from R1_gating import (D_MODEL, N_DIST, ETA, M, S_KEY, stream_idx,
                       build_stream, drive, make_core, metrics_plas)
from R1_gating2 import keep_positions, item_counts
from R1_gating3 import ret_jitter, ret_jitter_bruteforce
from R1_gating5 import calibrate_w, theta_of
from R1_gating6 import _pos_idxs

R_PRES = 4
R_HO4, R_HO8 = 4, 8
HO_OFFSET = 999
R_MAIN = 0.40
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"

# §3.2 剖面模板（写死；总和同为 10.64 ⇒ 只有形状不同）
W_F = (3.86, 3.46, 2.26, 1.06)
W_U = (2.66, 2.66, 2.66, 2.66)
W_K = (1.06, 2.26, 3.46, 3.86)
PROFILE = {"F": W_F, "U": W_U, "K": W_K}
JUDGE_COUNTS = ("u", "e")
JUDGE_R = (8, 12)
CAP_R = (6, 16)

THR_DIM = 0.10          # (a) Δ(F−K) 可分辨下限
THR_INERT = 0.05        # (F1) 惰性阈值
THR_INTER = 0.10        # (b) 两个计数水平上 Δ(F−K) 之差
THR_SUB = 0.10          # (e) 副口径差
THR_PRE = 0.10          # (c)② Δ(B−P) 复现下限
D38 = "/home/ec2-user/sgmn-loop/dir26f/out/DIR26Fr10_formal.json"
REF_BP = {8: 0.1500, 12: 0.1586}          # D-38 的 Δ(B−P) 冻结参考（(c)②）
TAG_OF_R = {8: "H8", 12: "H12"}          # ★ D-38 的 (r,seed) 键**不唯一**（H8/PR8/PW8 共享 r=8）⇒ 必须按 tag 取


def d38_rows():
    r"""D-38 只取健康档两格（H8/H12，$\cos\theta_r=\cos\theta_w=0.9$）作锚点。"""
    return {(r["tag"], r["seed"]): r for r in json.load(open(D38))["rows"] if r["tag"] in ("H8", "H12")}
res = {"checks": [], "rows": [], "calib": []}


# ══════════════════════════════════════════════ 构造（卡片 §3 冻结）
def round_quota(q, w, cap=2 * M):
    r"""逐轮配额：最大余数法 + **上限 cap=2M**（一个轮只有 2M 个机会）+ 溢出修复（rev 1.2）。

    (1) 比例目标 $a_\rho=q\,w_\rho/\sum w$，取 floor，余数给小数部分最大者；
        平局依次比 $|\rho-(R-1)/2|$ 小、再比 $\rho$ 小（**反演对称**）。
    (2) 若某轮超过 cap：截到 cap，把超出的名额按 $(-w_\rho,\ |\rho-c|,\ \rho)$ 顺序补给尚有余额的轮次。
        ⇒ 模板在 $q_i$ 大时**必然被压平**（$q_i{=}2RM$ 时三档全为 $(4,4,4,4)$），这是装置约束，不是选择。
    """
    q = int(q)
    L = len(w)
    if q <= 0:
        return [0] * L
    q = min(q, cap * L)
    tot = float(sum(w))
    a = [q * x / tot for x in w]
    n = [int(math.floor(x + 1e-12)) for x in a]
    D = q - sum(n)
    if D > 0:
        c = (L - 1) / 2.0
        order = sorted(range(L), key=lambda r: (-(a[r] - n[r]), abs(r - c), r))
        for j in range(D):
            n[order[j % L]] += 1
    excess = sum(max(n[r] - cap, 0) for r in range(L))
    if excess:
        n = [min(x, cap) for x in n]
        c = (L - 1) / 2.0
        for _ in range(excess):
            cand = [r for r in range(L) if n[r] < cap]
            if not cand:
                break
            r = min(cand, key=lambda r: (-w[r], abs(r - c), r))
            n[r] += 1
    return n


def even_counts(k_B, n):
    """均档：内容盲，与 `keep_positions` 的计数侧同规则。"""
    base, rem = divmod(int(k_B), n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def build_keep(n, r_pres, counts, w):
    """keep 集合 = 每 (条目, 轮) 取 `round_quota(q_i,w)[ρ]` 个机会，轮内用位置规则（与 `keep_positions` 同款）。"""
    keep, per_item = set(), []
    for i, q in enumerate(counts):
        n_rho = round_quota(q, w)
        per_item.append(n_rho)
        for rho, c in enumerate(n_rho):
            for pos in _pos_idxs(c, 2 * M):
                mbr, m = divmod(pos, M)
                keep.add(stream_idx(rho, i, mbr, m, n))
    return keep, per_item


def realized_profile(keep, n, r_pres):
    """每条目每轮的写入数（实测剖面）。"""
    h = [0] * r_pres
    for t in keep:
        h[(t // (2 * M)) // n] += 1
    return [x / n for x in h]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


# ══════════════════════════════════════════════ 读数
def read_streams(seed):
    th = theta_of(0.9)
    return {"ho4": build_stream(seed + HO_OFFSET, D_MODEL, N_DIST, R_HO4, th),
            "ho8": build_stream(seed + HO_OFFSET, D_MODEL, N_DIST, R_HO8, th)}


def run_unit(cell, seed, tau, RS):
    """一个 ($r$, 种子)：参照臂 A/B/P + 六单元（容量格点只跑 u 档三单元）。"""
    D = build_stream(seed, D_MODEL, N_DIST, R_PRES, theta_of(0.9))
    out = {"r0": cell["r0"], "seed": seed, "mode": cell["mode"],
           "T": R_PRES * N_DIST * 2 * M, "arms": {}, "prof": {}, "cnt": {}, "k": {}}

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
    cB_keep = {t for t, e in enumerate(logB) if e["issued"]}
    cnt_B = item_counts(cB_keep, N_DIST)
    out["k"] = {"A": kA, "B": kB}

    keepP = keep_positions(N_DIST, R_PRES, kB)
    cP = make_core(D_MODEL, seed, cell["r0"]); kP, _ = drive(cP, D, "R", keep=keepP); rec("P", cP, kP)

    counts = {"u": cnt_B, "e": even_counts(kB, N_DIST)}
    modes = JUDGE_COUNTS if cell["mode"] == "judge" else ("u",)
    for cm in modes:
        for pk in ("F", "U", "K"):
            keep, per_item = build_keep(N_DIST, R_PRES, counts[cm], PROFILE[pk])
            tag = f"{pk}{cm}"
            c = make_core(D_MODEL, seed, cell["r0"]); kk, _ = drive(c, D, "R", keep=keep); rec(tag, c, kk)
            got = item_counts(keep, N_DIST)
            out["cnt"][tag] = got
            out["prof"][tag] = realized_profile(keep, N_DIST, R_PRES)
            out["arms"][tag]["sum_ok"] = (kk == sum(got) == kB)
            out["arms"][tag]["quota_ok"] = all(sum(per_item[i]) == counts[cm][i] for i in range(N_DIST))
    out["counts_ref"] = {"u": cnt_B, "e": counts["e"]}
    out["jac"] = {f"{a}_{b}": jaccard(
        build_keep(N_DIST, R_PRES, counts["u"], PROFILE[a])[0],
        build_keep(N_DIST, R_PRES, counts["u"], PROFILE[b])[0]) for a, b in (("F", "U"), ("U", "K"), ("F", "K"))}
    return out


def chk(cid, ok, detail):
    res["checks"].append({"id": cid, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}  {detail}")
    return bool(ok)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ══════════════════════════════════════════════ Stage A
def stage_a():
    print("=== Stage A（A26G-1 … A26G-13） ===")
    dev = "/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py"
    chk("A26G-1", md5(dev) == DEV_MD5, f"装置 md5 {md5(dev)} == {DEV_MD5}")

    m = {}
    for n in ("unit", "make_core", "build_stream", "drive", "metrics_plas", "stream_idx"):
        m[n] = getattr(G, n, None).__module__
    for n in ("keep_positions", "item_counts"):
        m[n] = getattr(G2, n, None).__module__
    for n in ("ret_jitter", "ret_jitter_bruteforce"):
        m[n] = getattr(G3, n, None).__module__
    for n in ("calibrate_w", "theta_of"):
        m[n] = getattr(G5, n, None).__module__
    m["_pos_idxs"] = getattr(G6, "_pos_idxs", None).__module__
    want = {"R1_gating": ("unit", "make_core", "build_stream", "drive", "metrics_plas", "stream_idx"),
            "R1_gating2": ("keep_positions", "item_counts"),
            "R1_gating3": ("ret_jitter", "ret_jitter_bruteforce"),
            "R1_gating5": ("calibrate_w", "theta_of"),
            "R1_gating6": ("_pos_idxs",)}
    bad = [n for mod, ns in want.items() for n in ns if m[n] != mod]
    chk("A26G-2", not bad, f"装置/测量/定标/抖动指标/轮内位置规则 全为 import 的同一对象：异常 {bad or '无'}")

    # A26G-3：剖面构造审计（解析值 + 最小实例）
    q10 = {k: round_quota(10, w) for k, w in PROFILE.items()}
    q9 = {k: round_quota(9, w) for k, w in PROFILE.items()}
    q16 = {k: round_quota(16, w) for k, w in PROFILE.items()}
    chk("A26G-3", q10["F"] == [4, 3, 2, 1] and q10["U"] == [2, 3, 3, 2] and q10["K"] == [1, 2, 3, 4]
        and sum(q10["U"]) == 10 and sum(q10["F"]) == 10 and sum(q10["K"]) == 10
        and all(v == [4, 4, 4, 4] for v in q16.values()),
        f"q=10 逐轮配额 F={q10['F']} U0={q10['U']} K={q10['K']}；q=9 得 F={q9['F']} U0={q9['U']} K={q9['K']}；"
        f"q=16 三档同为 {q16['F']}（cap=4 生效）")

    # A26G-4：反演精确性 + keep 集合不同
    n8, seed8 = 8, SEEDS[0]
    D8 = build_stream(seed8, D_MODEL, n8, R_PRES, theta_of(0.9))
    c8 = make_core(D_MODEL, seed8, 8)
    tau8 = calibrate_w(8, R_MAIN, R_PRES, theta_of(0.9))["c_tau"]
    kB8, logB8 = drive(c8, D8, "B", tau=tau8)
    cnt8 = {"u": item_counts({t for t, e in enumerate(logB8) if e["issued"]}, n8), "e": even_counts(kB8, n8)}
    rev_ok, per_ok, n_cap = True, True, 0
    for cm in JUDGE_COUNTS:
        for i in range(n8):
            nf = round_quota(cnt8[cm][i], W_F); nk = round_quota(cnt8[cm][i], W_K)
            if nk != nf[::-1]:
                rev_ok = False
            if sum(nf) != cnt8[cm][i] or sum(nk) != cnt8[cm][i]:
                per_ok = False
            if cnt8[cm][i] > 2 * M:
                n_cap += 1
    kF = build_keep(n8, R_PRES, cnt8["u"], W_F)[0]
    kU = build_keep(n8, R_PRES, cnt8["u"], W_U)[0]
    kK = build_keep(n8, R_PRES, cnt8["u"], W_K)[0]
    sums_ok = len(kF) == len(kU) == len(kK) == kB8
    dom_ok = True
    for q in range(0, 2 * M * R_PRES + 1):
        cf, cu, ck = round_quota(q, W_F), round_quota(q, PROFILE["U"]), round_quota(q, W_K)
        for j in range(3):
            if not (sum(cf[:j + 1]) >= sum(cu[:j + 1]) >= sum(ck[:j + 1])):
                dom_ok = False
    chk("A26G-4", rev_ok and per_ok and sums_ok and dom_ok and jaccard(kF, kK) < 0.9,
        f"n_ρ(K)==n_(R−1−ρ)(F) 逐条目逐位={rev_ok}（含 {n_cap} 个受 cap 约束的条目）；逐条目 Σn=q_i={per_ok}；"
        f"cum_F≥cum_U0≥cum_K 对 q∈[0,16] 全成立={dom_ok}；三档 |keep|={len(kF)}/{len(kU)}/{len(kK)}=k_B={kB8}；"
        f"Jaccard(F,K)={jaccard(kF, kK):.3f}（须 <0.9）")

    # A26G-5：构造恒等（最小实例 + 真实 k_B）
    chk("A26G-5", sums_ok and item_counts(kF, n8) == cnt8["u"]
        and max(item_counts(build_keep(n8, R_PRES, cnt8["e"], W_F)[0], n8)) -
        min(item_counts(build_keep(n8, R_PRES, cnt8["e"], W_F)[0], n8)) <= 1,
        f"不均档逐条目计数 = c_i^B 逐位=True；均档极差 ≤1；Σ发放 = {kB8}")

    # A26G-12：实测剖面单调
    pF = realized_profile(kF, n8, R_PRES); pU = realized_profile(kU, n8, R_PRES); pK = realized_profile(kK, n8, R_PRES)
    chk("A26G-12", all(pF[i] > pF[i + 1] for i in range(3)) and all(pK[i] < pK[i + 1] for i in range(3))
        and abs(pU[0] - pU[3]) <= 0.3 and abs(pU[1] - pU[2]) <= 0.3,
        f"实测剖面 F={['%.2f' % x for x in pF]} U0={['%.2f' % x for x in pU]} K={['%.2f' % x for x in pK]}"
        f"（U0 只要求对称：|p0−p3|={abs(pU[0] - pU[3]):.2f}、|p1−p2|={abs(pU[1] - pU[2]):.2f}）")

    # A26G-6：三锚点逐位复现（A/B/P）
    RS8 = read_streams(SEEDS[0])
    a9 = float(torch.nn.functional.cosine_similarity(RS8["ho4"]["K"], RS8["ho4"]["proto"], dim=-1).mean())
    chk("A26G-10", abs(a9 - 0.9) <= 0.01, f"读流对齐度 cos(ho4,原型)={a9:.4f}（目标 0.9000）")

    d38 = d38_rows()
    okA, det = True, []
    for r0 in JUDGE_R:
        if r0 == 8:
            g = run_unit({"r0": 8, "mode": "judge"}, SEEDS[0], tau8, RS8)
        else:
            tau12 = calibrate_w(12, R_MAIN, R_PRES, theta_of(0.9))["c_tau"]
            g = run_unit({"r0": 12, "mode": "judge"}, SEEDS[0], tau12, RS8)
        ref = d38[(TAG_OF_R[r0], SEEDS[0])]["arms"]
        good = all(g["arms"][a]["ho4_mean"] == ref[a]["ho4_mean"]
                   and g["arms"][a]["ho4_maj"] == ref[a]["ho4_maj"]
                   and g["arms"][a]["clean"] == ref[a]["clean"] for a in ("A", "B", "P"))
        okA = okA and good
        det.append(f"r={r0}:{good}(B {g['arms']['B']['ho4_mean']:.4f} vs {ref['B']['ho4_mean']:.4f})")
        if r0 == 8:
            g8 = g
    chk("A26G-6", okA, f"三锚点（A/B/P 的 ho4_mean/ho4_maj/clean）与 D-38 逐位：{', '.join(det)}")

    # A26G-7：抖动指标逐位
    d16, n16 = 16, 8
    D16 = build_stream(SEEDS[0], d16, n16, R_PRES, theta_of(0.9))
    c16 = make_core(d16, SEEDS[0], 4); drive(c16, D16, "A")
    P16 = c16._W().detach() - c16.W0
    Ra = build_stream(SEEDS[0] + HO_OFFSET, d16, n16, R_HO4, theta_of(0.9))
    Rb = build_stream(SEEDS[0] + HO_OFFSET, d16, n16, R_HO8, theta_of(0.9))
    a4 = ret_jitter(P16, Ra["K"], D16["vb"], n16, R_HO4); b4 = ret_jitter_bruteforce(P16, Ra["K"], D16["vb"], n16, R_HO4)
    a8 = ret_jitter(P16, Rb["K"], D16["vb"], n16, R_HO8); b8 = ret_jitter_bruteforce(P16, Rb["K"], D16["vb"], n16, R_HO8)
    chk("A26G-7", a4["ret1_mean"] == b4["ret1_mean"] and a4["ret1_maj"] == b4["ret1_maj"]
        and a8["ret1_mean"] == b8["ret1_mean"] and a8["ret1_maj"] == b8["ret1_maj"]
        and a4["maj_need"] == 3 and a8["maj_need"] == 5,
        f"抖动指标逐位：4→mean {a4['ret1_mean']:.4f}/maj {a4['ret1_maj']:.4f}(need 3)；"
        f"8→mean {a8['ret1_mean']:.4f}/maj {a8['ret1_maj']:.4f}(need 5)")

    # A26G-8：护栏（Ku 臂纯标量不变性）+ 同构造复现 g8 的 Ku 读数（DEF-26G-1：不得混用 N=8 的流）
    keepK8 = build_keep(N_DIST, R_PRES, g8["counts_ref"]["u"], W_K)[0]
    Dn35 = build_stream(SEEDS[0], D_MODEL, N_DIST, R_PRES, theta_of(0.9))
    cK = make_core(D_MODEL, SEEDS[0], 8); drive(cK, Dn35, "R", keep=keepK8)
    PK = cK._W().detach() - cK.W0
    si = [ret_jitter(s * PK, RS8["ho4"]["K"], Dn35["vb"], N_DIST, R_HO4)["ret1_mean"] for s in (0.37, 1.0, 2.5)]
    chk("A26G-8", max(si) - min(si) == 0.0 and si[1] == g8["arms"]["Ku"]["ho4_mean"],
        f"护栏（Ku 臂，N=35 真流）：纯标量不变性 c∈{{0.37,1,2.5}} = {si}；"
        f"重跑复现 g8 的 Ku {g8['arms']['Ku']['ho4_mean']:.4f}")

    # A26G-9：稳定性 + 零发放
    rho_stab = 1 - ETA * (1.0 * S_KEY + 0.05 / 2)
    cN = make_core(D_MODEL, SEEDS[0], 8); W0n = cN._W().detach().clone()
    kN, _ = drive(cN, D8, "NONE")
    chk("A26G-9", rho_stab > 0 and kN == 0 and float((cN._W().detach() - W0n).norm()) == 0.0,
        f"稳定性 ρ={rho_stab:.4f}>0；零发放臂 {kN} 次、‖ΔW‖=0")

    # A26G-11：A 臂实发 = T；读不改写
    chk("A26G-11", g8["arms"]["A"]["issued"] == g8["T"] == 560
        and all(v["W_unchanged"] for v in g8["arms"].values()),
        f"A 臂实发 {g8['arms']['A']['issued']}=T={g8['T']}；九臂读后 W 逐位不变="
        f"{all(v['W_unchanged'] for v in g8['arms'].values())}")

    # A26G-13：定标
    cals = {}
    for r0 in (6, 8, 12, 16):
        c = calibrate_w(r0, R_MAIN, R_PRES, theta_of(0.9))
        cals[r0] = c
    chk("A26G-13", all(c["gate_ok"] and abs(c["skip_rate"] - R_MAIN) <= 0.05 for c in cals.values()),
        "四次标定跳过率：" + ", ".join(f"r={r0} {c['skip_rate']:.3f}/gate_ok={c['gate_ok']}"
                                    for r0, c in cals.items()))
    res["calib"] = [{"r0": r0, "c_tau": c["c_tau"], "skip_rate": c["skip_rate"], "gate_ok": c["gate_ok"]}
                    for r0, c in cals.items()]
    return g8


# ══════════════════════════════════════════════ 判据
def crit(rows):
    print("=== 判据（rev 1.1 冻结阈值） ===")
    jr = [g for g in rows if g["mode"] == "judge"]

    def arm(tag, key="ho4_mean"):
        d = {}
        for g in jr:
            d.setdefault((g["r0"], g["seed"]), {})[tag] = g["arms"][tag][key]
        return d

    U = {}
    for r0 in JUDGE_R:
        for cm in JUDGE_COUNTS:
            key = f"{r0}{cm}"
            seeds = sorted({g["seed"] for g in jr if g["r0"] == r0})
            d = {}
            for a in ("F", "U", "K"):
                xs = [g["arms"][f"{a}{cm}"]["ho4_mean"] for g in jr if g["r0"] == r0]
                ys = [g["arms"][f"{a}{cm}"]["ho8_maj"] for g in jr if g["r0"] == r0]
                d[a] = (float(np.mean(xs)), float(np.std(xs, ddof=1)), float(np.mean(ys)),
                        int(sum(x1 > x2 for x1, x2 in zip(xs, ys))))
            U[key] = {"r0": r0, "counts": cm, "n": len(seeds),
                      "F": d["F"][0], "U": d["U"][0], "K": d["K"][0],
                      "F_maj": d["F"][2], "U_maj": d["U"][2], "K_maj": d["K"][2],
                      "dFK": d["F"][0] - d["K"][0], "dFU": d["F"][0] - d["U"][0],
                      "dUK": d["U"][0] - d["K"][0],
                      "dFK_maj": d["F"][2] - d["K"][2],
                      "dFK_sd": float(np.std([g["arms"][f"F{cm}"]["ho4_mean"] - g["arms"][f"K{cm}"]["ho4_mean"]
                                              for g in jr if g["r0"] == r0], ddof=1)),
                      "pos": int(sum(g["arms"][f"F{cm}"]["ho4_mean"] > g["arms"][f"K{cm}"]["ho4_mean"]
                                     for g in jr if g["r0"] == r0))}

    v = {}
    # (c) 前置复现
    d38 = d38_rows()
    anc_ok, anc_det = True, []
    dBP_by_r = {}
    for g in jr:
        ref = d38[(TAG_OF_R[g["r0"]], g["seed"])]["arms"]
        good = all(g["arms"][a]["ho4_mean"] == ref[a]["ho4_mean"] and g["arms"][a]["ho4_maj"] == ref[a]["ho4_maj"]
                   for a in ("A", "B", "P"))
        anc_ok = anc_ok and good
        anc_det.append(f"{g['r0']}@{g['seed']}:{good}")
        dBP_by_r.setdefault(g["r0"], []).append(g["arms"]["B"]["ho4_mean"] - g["arms"]["P"]["ho4_mean"])
    # (c)② 按**冻结文字**：每个 $r$ 的 5 种子均值复现参考值，且 5/5 同号（DEF-26G-2：不得改成逐种子阈值）
    pre_ok, pre_det = True, {}
    for r0, xs in dBP_by_r.items():
        m = float(np.mean(xs))
        good = abs(m - REF_BP[r0]) <= 1e-3 and sum(x > 0 for x in xs) == len(xs)
        pre_ok = pre_ok and good
        pre_det[str(r0)] = {"mean": round(m, 4), "ref": REF_BP[r0],
                            "pos": f"{sum(x > 0 for x in xs)}/{len(xs)}", "min": round(min(xs), 4), "ok": good}
    v["c_pre"] = {"ok": bool(anc_ok and pre_ok), "anchor": anc_ok, "detail": anc_det, "dBP": pre_det}

    # (d) 构造恒等
    bad = []
    for g in rows:
        for t, a in g["arms"].items():
            if t in ("A", "B", "P"):
                continue
            if not (a.get("sum_ok") and a.get("quota_ok")):
                bad.append(f"{g['r0']}@{g['seed']}:{t}")
    v["d_identity"] = {"ok": not bad, "bad": bad[:6], "n_arms": sum(1 for g in rows for t in g["arms"] if t not in ("A", "B", "P"))}

    # (a) 剖面可分辨
    a_det = [{"cell": k, "dFK": round(U[k]["dFK"], 4), "sd": round(U[k]["dFK_sd"], 4), "pos": f"{U[k]['pos']}/{U[k]['n']}",
              "ok": U[k]["dFK"] >= THR_DIM, "inert": abs(U[k]["dFK"]) <= THR_INERT} for k in U]
    v["a_dim"] = {"ok": all(x["ok"] for x in a_det), "n_ok": sum(x["ok"] for x in a_det), "n": len(a_det),
                  "detail": a_det, "threshold": THR_DIM}

    # (b) 与计数水平无关（交互项）
    b_det = []
    for r0 in JUDGE_R:
        du, de = U[f"{r0}u"]["dFK"], U[f"{r0}e"]["dFK"]
        b_det.append({"r0": r0, "dFK_u": round(du, 4), "dFK_e": round(de, 4), "diff": round(abs(du - de), 4),
                      "same_sign": (du > 0) == (de > 0), "ok": abs(du - de) <= THR_INTER})
    signs = [(U[k]["dFK"] > 0) for k in U]
    v["b_inter"] = {"ok": all(x["ok"] and x["same_sign"] for x in b_det) and len(set(signs)) == 1,
                    "detail": b_det, "sign_uniform": len(set(signs)) == 1}

    # (e) 副口径
    e_det = [{"cell": k, "mean4": round(U[k]["dFK"], 4), "maj8": round(U[k]["dFK_maj"], 4),
              "same_sign": (U[k]["dFK"] > 0) == (U[k]["dFK_maj"] > 0),
              "diff": round(abs(U[k]["dFK"] - U[k]["dFK_maj"]), 4)} for k in U]
    v["e_maj"] = {"ok": all(x["same_sign"] and x["diff"] <= THR_SUB for x in e_det), "detail": e_det}

    # (F1)(F2) 证伪条款
    inert = [x["cell"] for x in a_det if x["inert"]]
    flip = [x["r0"] for x in b_det if not x["same_sign"] and max(abs(x["dFK_u"]), abs(x["dFK_e"])) >= THR_DIM]
    v["f1_inert"] = {"hit": len(inert) >= 2, "cells": inert}
    v["f2_flip"] = {"hit": bool(flip), "r0": flip}

    # (f) 报告项
    cap = [{"r0": g["r0"], "seed": g["seed"], "F": g["arms"]["Fu"]["ho4_mean"], "U": g["arms"]["Uu"]["ho4_mean"],
            "K": g["arms"]["Ku"]["ho4_mean"]} for g in rows if g["mode"] == "cap"]
    capm = {}
    for r0 in CAP_R:
        rs = [c for c in cap if c["r0"] == r0]
        capm[r0] = {k: round(float(np.mean([c[k] for c in rs])), 4) for k in ("F", "U", "K")}
        capm[r0]["dFK"] = round(capm[r0]["F"] - capm[r0]["K"], 4)
    v["f_report"] = {
        "capacity_dFK": capm,
        "profile_realized": {k: [round(x, 3) for x in g["prof"][k]] for k in ("Fu", "Uu", "Ku") for g in jr[:1]},
        "vs_D38_U": round(float(np.mean([g["arms"]["Uu"]["ho4_mean"] - d38[(TAG_OF_R[g["r0"]], g["seed"])]["arms"]["U"]["ho4_mean"]
                                         for g in jr])), 4),
        "vs_D38_E": round(float(np.mean([g["arms"]["Fe"]["ho4_mean"] - d38[(TAG_OF_R[g["r0"]], g["seed"])]["arms"]["E"]["ho4_mean"]
                                         for g in jr])), 4),
        "clean": {k: round(float(np.mean([g["arms"][k]["clean"] for g in jr])), 4)
                  for k in ("A", "B", "P", "Fu", "Uu", "Ku", "Fe", "Ue", "Ke")},
        "jac_u": {k: round(float(np.mean([g["jac"][k] for g in jr])), 3) for k in jr[0]["jac"]},
    }

    if not v["c_pre"]["anchor"] or not v["d_identity"]["ok"]:
        main = "INDETERMINATE"
    elif v["f1_inert"]["hit"] or v["f2_flip"]["hit"]:
        main = "FALSIFIED"
    elif v["a_dim"]["ok"] and v["b_inter"]["ok"] and v["e_maj"]["ok"] and v["c_pre"]["ok"]:
        main = "CONFIRMED"
    else:
        main = "WEAK"
    v["main"] = {"ok": main == "CONFIRMED", "verdict": main}

    for k in ("c_pre", "d_identity", "a_dim", "b_inter", "e_maj"):
        print(f"  [{'PASS' if v[k]['ok'] else 'FAIL'}] {k}")
    for k in U:
        x = U[k]
        print(f"    {k:4s} F {x['F']:.3f} U0 {x['U']:.3f} K {x['K']:.3f} | Δ(F−K) {x['dFK']:+.4f} "
              f"(±{x['dFK_sd']:.4f}, {x['pos']}/{x['n']}) | Δ(F−U0) {x['dFU']:+.4f} Δ(U0−K) {x['dUK']:+.4f}")
    print(f"  (b) 交互：{[(b['r0'], b['dFK_u'], b['dFK_e'], b['same_sign']) for b in b_det]}")
    print(f"  (F1) 惰性 {v['f1_inert']['cells']} ；(F2) 翻转 {v['f2_flip']['r0']}")
    print(f"  ⇒ H26G 判定：**{main}**")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26Gr11")
    args = ap.parse_args()
    t0 = time.time()
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26g/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    if args.stage == "stageA":
        stage_a()
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}"
              f"（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"], res["passed"] = "stageA", ok
        res["calib"] = res.get("calib", [])
    else:
        seeds = [SEEDS[0]] if args.stage == "smoke" else SEEDS
        cals = {r0: calibrate_w(r0, R_MAIN, R_PRES, theta_of(0.9))["c_tau"] for r0 in (6, 8, 12, 16)}
        res["calib"] = [{"r0": r0, "c_tau": t} for r0, t in cals.items()]
        plan = ([{"r0": 8, "mode": "judge"}] if args.stage == "smoke"
                else [{"r0": r0, "mode": "judge"} for r0 in JUDGE_R]
                + [{"r0": r0, "mode": "cap"} for r0 in CAP_R])
        rows = []
        print(f"=== {args.stage}（{len(plan)} 个 ($r$,模式) × {len(seeds)} 种子） ===")
        for cell in plan:
            for s in seeds:
                RS = read_streams(s)
                g = run_unit(cell, s, cals[cell["r0"]], RS)
                rows.append(g)
                json.dump({"rows": rows}, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
                a = g["arms"]
                kk = sorted(t for t in a if t not in ("A", "B", "P"))
                print(f"  r={cell['r0']:<2d} seed={s} | A {a['A']['ho4_mean']:.3f} B {a['B']['ho4_mean']:.3f} "
                      f"P {a['P']['ho4_mean']:.3f} | " +
                      " ".join(f"{t} {a[t]['ho4_mean']:.3f}" for t in kk) +
                      f" | k_B {g['k']['B']} | prof(F) {['%.1f' % x for x in g['prof'][kk[0]]]}")
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
                   "r_ho": [R_HO4, R_HO8], "rho_main": R_MAIN, "seeds": SEEDS,
                   "profile": {"F": W_F, "U": W_U, "K": W_K}, "judge_r": list(JUDGE_R),
                   "cap_r": list(CAP_R), "stage": args.stage, "card": "DIR-26G rev1.2"}
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
