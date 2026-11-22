"""DIR-26 · R1_gating.py —— 内容相关门控是否带来**独立于速率效应**的收益

卡片：`revolve/DIR-26/DIRECTION_rev26_v10.md`（rev 1.0 冻结）+ `_v11.md`（rev 1.1：三处**开跑前**结构订正，§C1 阈值按结构上界缩放 / §C2 A26-3 改写 / §C3 R=1 臂坍缩）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，**零改动**，DEC-02）

轴：$r\\in\\{8,35,896\\}$ × $\\theta\\in\\{0,25.8419^\\circ\\}$ × $R\\in\\{1,4\\}$ × 5 种子 × 5 臂（$R{=}1$ 只跑 A，§C3）+ (e) 副臂。

五臂（同一流、逐位相同的呈现顺序）：
  A $\\alpha\\equiv1$ · B 内容门控 $\\cos(W_tk_t,v_t)>c_\\tau$ 则跳过 · C 恒定 $\\alpha\\equiv\\bar\\alpha{=}k_B/T$（**等质量**）
  R 随机门控（**与 B 等发放数** $k_B$）· D 只写首现（位置选择）
测量（唯一判据口径，卡片 §2）：$\\mathrm{ret}_1(P)=\\frac1N\\sum_i \\mathbf1[\\arg\\max_j\\cos(P\\,k^{proto}_i,v_j)=i]$，$P=W-W^0$
  ★ `ret1` 对 $P\\to cP$ 严格不变 ⇒ 对速率效应免疫（工作算例 A≡C 在 $N{=}4/8/12$ 上逐位相同）。
"""
import argparse, ast, hashlib, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")
from sgmn_module_hebb_v7 import HebbianCore

D_MODEL = 896
ALPHA = 1.0                     # 逐臂重绑（make_core 读模块全局，保持逐字复用，同 DIR-25 的 LAM 做法）
LAM = 0.05                      # 逐臂重绑（仅 (e) 副臂用 0.10）
ETA = 0.1
M = 2
S_KEY = 5.0
N_SLOTS = 4096
N_DIST = 35
R_GRID = [1, 4]
THETA_GRID = [0.0, math.radians(25.8419)]        # 0 = 精确重复；25.8419° ⇒ cos θ = 0.9000
R0_GRID = [8, 35, 896]
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]
SEED_PILOT = 20261041                            # ★ 只用于 $c_\\tau$ 标定，与判据种子不相交
ARMS = ["A", "B", "C", "R", "D"]
RTOL = 1e-6
RTOL_P = 1e-6
RHO_SKIP = 0.40          # ★ rev 1.2 §C4 冻结：门控目标跳过率（按 pilot 的 cos 分位数标定 c_τ）
DTYPE = torch.float32
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
HERE = os.path.dirname(os.path.abspath(__file__))
res = {"checks": [], "rows": [], "calib": []}


# ══════════════════════════════════════════════ 复用层（与 dir25/R1_capacity.py 逐字相同，A26-2）
def unit(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-30)


def make_core(d, seed, r0):
    """`r0=0` ⇒ §3.2 **全矩阵参考步**（$W^0=0$ 且 $M_0\\equiv0$，唯一可表达的空记忆，A24-5）。
    `r0>=1` ⇒ §3.3 因子步 + **首 $r_0$ 列支撑**（不变子空间，DEC-15）。"""
    g = torch.Generator().manual_seed(seed)
    if r0 == 0:
        c = HebbianCore(d, d, ALPHA, LAM, ETA, g, "cpu", factorized=False,
                        n_entries=N_SLOTS, d_model_sig=8)
        c.W.zero_()
        c.W0.zero_()
        return c
    c = HebbianCore(d, d, ALPHA, LAM, ETA, g, "cpu", factorized=True,
                    n_entries=N_SLOTS, d_model_sig=8)
    U = (torch.randn(d, d, generator=g) / math.sqrt(d)).to("cpu")
    V = (torch.randn(d, d, generator=g) / math.sqrt(d)).to("cpu")
    U[:, r0:] = 0.0
    V[:, r0:] = 0.0
    c.U.copy_(U)
    c.V.copy_(V)
    return c


def best_q(core, kind, targets):
    """该臂信息集下的**最优**读：组内所有可能目标的均值 → 最小二乘解。

    ★ 为什么不用拟合（DEC-10）：六臂的差别是**信息集**（c 有没有情节、S 有没有进读取），
      不是参数估计能力。用 $20\times900$ 的岭回归去逼近，实测 margin 只有 $10^{-2}$ 量级，
      acc 由拟合噪声支配（空记忆下漂到 0.667/0.800）⇒ 那测的是回归器而不是假说。
      按信息分组取组内最优读，是"给定这些输入，最好能做到什么"的**精确**答案。
    """
    tgt = torch.stack(targets).mean(0) if isinstance(targets, list) else targets
    if kind == "hebbian":
        W = core._W().detach()
        try:
            return torch.linalg.lstsq(W, tgt.unsqueeze(1)).solution[:, 0]
        except Exception:
            return torch.linalg.pinv(W) @ tgt
    return tgt if tgt.dim() == 1 else tgt[0]


def write_A(core, D):
    """写序 $a,a,b,b$（$M=2$）；返回 5 个写端点的 $\\operatorname{rank}W$（A24-2）。"""
    n = D["n"]
    snaps, tot = [], n * 2 * M
    for i in range(n):
        for mbr, v in ((0, D["va"][i]), (1, D["vb"][i])):
            k = D["K"][i]
            for _ in range(M):
                core.write(k, v)
                if core.count in (1, n, tot // 2, tot - n, tot):
                    snaps.append(int(torch.linalg.matrix_rank(core._W().detach(), rtol=RTOL).item()))
    return {"rank_snaps": snaps, "delta_u": float(core.last_delta)}


# ══════════════════════════════════════════════ 独立变量：抖动冗余流（卡片 §3）
def build_stream(seed, d, n, r_pres, theta):
    """$N$ 个**不同**条目 × $R$ 次呈现，抖动键 $k_{i,\\rho}=\\sqrt5\\,\\mathrm{norm}(\\cos\\theta\\,e_i+\\sin\\theta\\,u_{i,\\rho})$（$u\\perp e_i$）。
    返回每轮的键、原型键、两成员值（$v^a\\perp v^b$）、目标位。"""
    g = torch.Generator().manual_seed(seed + 1)
    E = torch.zeros(n, d)
    for i in range(n):
        E[i, i] = 1.0
    U = unit(torch.randn(n, r_pres, d, generator=g))
    K = torch.zeros(r_pres, n, d, dtype=DTYPE)
    for rho in range(r_pres):
        for i in range(n):
            u = U[i, rho] - (U[i, rho] @ E[i]) * E[i]
            u = unit(u) if float(u.norm()) > 1e-6 else unit(torch.randn(d, generator=g))
            K[rho, i] = math.sqrt(S_KEY) * unit(math.cos(theta) * E[i] + math.sin(theta) * u)
    proto = math.sqrt(S_KEY) * E.clone()                       # 原型键（无抖动）
    va = unit(torch.randn(n, d, generator=g))
    r = torch.randn(n, d, generator=g)
    vb = unit(r - (r * va).sum(-1, keepdim=True) * va)
    return {"K": K, "proto": proto, "va": va, "vb": vb, "n": n, "r_pres": r_pres}


def stream_idx(rho, i, mbr, m, n):
    return ((rho * n + i) * 2 + mbr) * M + m


# ══════════════════════════════════════════════ 驱动（独立变量：写不写）
def drive(core, D, arm, tau=None, keep=None):
    """按呈现流写；返回发放数 $k$、逐写记录（供 $c_\\tau$ 标定与诊断）。"""
    n, R = D["n"], D["r_pres"]
    issued, log = 0, []
    for rho in range(R):
        for i in range(n):
            k = D["K"][rho, i]
            for mbr, v in ((0, D["va"][i]), (1, D["vb"][i])):
                for m in range(M):
                    t_idx = stream_idx(rho, i, mbr, m, n)
                    c_now = float(torch.nn.functional.cosine_similarity(
                        (core._W() @ k).unsqueeze(0), v.unsqueeze(0)).item())
                    if arm == "NONE":
                        go = False
                    elif arm in ("A", "C"):
                        go = True
                    elif arm == "D":
                        go = (rho == 0)
                    elif arm == "R":
                        go = (t_idx in keep)
                    else:                                        # B：内容门控
                        go = not (tau is not None and c_now > tau)
                    log.append({"round": rho, "item": i, "cos": c_now, "issued": bool(go)})
                    if go:
                        core.write(k, v)
                        issued += 1
    return issued, log


# ══════════════════════════════════════════════ 测量层（卡片 §2，唯一判据口径）
def metrics_plas(W, W0, proto, vb, n):
    P = W - W0
    Y = P @ proto.T                                            # (d, n)
    S = unit(Y.T) @ unit(vb).T
    n_ok = sum(int(torch.argmax(S[i]).item()) == i for i in range(n))
    off = []
    for i in range(n):
        off.append(float(torch.max(torch.cat([S[i, :i], S[i, i + 1:]]))))
    rowm = float(np.mean([(float(S[i, i]) - off[i]) / max(float((P @ proto[i]).norm()), 1e-9)
                          for i in range(n)]))
    return {"ret1": n_ok / n, "n_correct": int(n_ok), "rowm": rowm,
            "cosP": float(np.mean([float(S[i, i]) for i in range(n)])),
            "P_norm": float(P.norm()),
            "rank_P_1e6": int(torch.linalg.matrix_rank(P, rtol=1e-6).item()),
            "rank_P_1e4": int(torch.linalg.matrix_rank(P, rtol=1e-4).item()),
            "kappa_W": float(torch.linalg.cond(W).item())}


def scalar_invariance(W, W0, proto, vb, n, scales=(0.37, 1.0, 2.5)):
    """rev 1.3 §C7：对**固定 P** 乘标量，$\\arg\\max$ 不变 ⇒ $\\mathrm{ret}_1(cP)=\\mathrm{ret}_1(P)$。
    这是指标的**护栏**（数学性质），不是判别力来源；原 rev 1.0 的 A26-6「$A\\equiv C$」已撤回（§C7）。"""
    P = W - W0
    out = {}
    for c in scales:
        Y = (c * P) @ proto.T
        S = unit(Y.T) @ unit(vb).T
        out[c] = float(np.mean([int(torch.argmax(S[i]).item()) == i for i in range(n)]))
    return out


# ══════════════════════════════════════════════ $c_\\tau$ 标定（pilot 种子，冻结规则：中位中点）
def calibrate(r0, theta, seed=SEED_PILOT):
    D = build_stream(seed, D_MODEL, N_DIST, 4, theta)
    core = make_core(D_MODEL, seed, r0)
    _, log = drive(core, D, "A")
    first = [e["cos"] for e in log if e["round"] == 0]
    rep = [e["cos"] for e in log if e["round"] > 0]
    allc = [e["cos"] for e in log]
    tau = float(np.quantile(allc, 1.0 - RHO_SKIP))          # ★ rev 1.2 §C4：发放率定标（ρ 冻结 = 0.40）
    skip = [e for e in log if e["cos"] > tau]
    keep_ = [e for e in log if e["cos"] <= tau]
    skip_rate = len(skip) / len(allc)
    m_skip = float(np.median([e["cos"] for e in skip])) if skip else float("nan")
    m_keep = float(np.median([e["cos"] for e in keep_])) if keep_ else float("nan")
    gap = m_skip - m_keep
    frac_rep = float(np.mean([e["round"] > 0 for e in skip])) if skip else float("nan")
    ok = (0.35 <= skip_rate <= 0.45) and (gap >= 0.05) and (frac_rep >= 0.85)
    row = {"r0": r0, "theta_deg": round(math.degrees(theta), 4), "seed": seed,
           "cos_first_median": float(np.median(first)), "cos_repeat_median": float(np.median(rep)),
           "c_tau": tau, "gap": gap, "skip_rate": skip_rate, "skip_frac_repeat": frac_rep,
           "gate_ok": bool(ok), "n_first": len(first), "n_repeat": len(rep)}
    res["calib"].append(row)
    print(f"  [标定] r={r0:3d} θ={row['theta_deg']:7.4f}° c_τ={tau:+.4f} 跳过率 {skip_rate:.3f} "
          f"cos(跳/发)中位 {m_skip:+.4f}/{m_keep:+.4f} gap={gap:+.4f} 跳写中重复占比 {frac_rep:.3f} "
          f"{'OK' if ok else '★GATE-DEGENERATE'}")
    return row


# ══════════════════════════════════════════════ 一次格点（同流五臂）
def run_cell(r0, theta, r_pres, seed, tau=None, alpha_const=None, lam=None):
    global ALPHA, LAM
    D = build_stream(seed, D_MODEL, N_DIST, r_pres, theta)
    T = r_pres * N_DIST * 2 * M
    out = {"r0": r0, "theta_deg": round(math.degrees(theta), 4), "R": r_pres,
           "seed": seed, "T": T, "cells": {}}
    t0 = time.time()
    # B 先跑（拿 k_B）
    ALPHA, LAM = 1.0, (LAM if lam is None else lam)
    coreB = make_core(D_MODEL, seed, r0)
    k_B, logB = drive(coreB, D, "B", tau=tau)
    W = coreB._W().detach()
    out["cells"]["B"] = {**metrics_plas(W, coreB.W0, D["proto"], D["vb"], N_DIST),
                         "issued": k_B, "alpha_eff": k_B / T, "alpha": 1.0}
    # R：与 B 等发放数（随机保留 k_B 个位置）
    gk = torch.Generator().manual_seed(seed + 7)
    keep = set(torch.randperm(T, generator=gk)[:int(k_B)].tolist())
    coreR = make_core(D_MODEL, seed, r0)
    k_R, _ = drive(coreR, D, "R", keep=keep)
    out["cells"]["R"] = {**metrics_plas(coreR._W().detach(), coreR.W0, D["proto"], D["vb"], N_DIST),
                         "issued": k_R, "alpha_eff": k_R / T, "alpha": 1.0}
    # C：恒定 α，等质量
    a_bar = k_B / T
    ALPHA = a_bar
    coreC = make_core(D_MODEL, seed, r0)
    k_C, _ = drive(coreC, D, "C")
    out["cells"]["C"] = {**metrics_plas(coreC._W().detach(), coreC.W0, D["proto"], D["vb"], N_DIST),
                         "issued": k_C, "alpha_eff": a_bar, "alpha": a_bar}
    # A、D
    for arm in ("A", "D"):
        ALPHA = 1.0
        c = make_core(D_MODEL, seed, r0)
        k, _ = drive(c, D, arm)
        out["cells"][arm] = {**metrics_plas(c._W().detach(), c.W0, D["proto"], D["vb"], N_DIST),
                             "issued": k, "alpha_eff": k / T, "alpha": 1.0}
    out["wall"] = round(time.time() - t0, 1)
    out["k_B"], out["k_R"], out["T_out"] = k_B, k_R, T
    ALPHA = 1.0
    return out, logB


# ══════════════════════════════════════════════ Stage A
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


def func_src(path, names):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names}


def stage_a():
    print("=== Stage A（A26-1 … A26-10，卡片 rev 1.1） ===")
    dev = "/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py"
    chk("A26-1", md5(dev) == DEV_MD5, f"装置 md5 {md5(dev)} == dir22 {DEV_MD5}")

    names = ["unit", "make_core", "best_q", "write_A"]
    mine = func_src(os.path.join(HERE, "R1_gating.py"), names)
    other = func_src(os.path.join(HERE, "..", "dir25", "R1_capacity.py"), names)
    same = [k for k in names if mine.get(k) is not None and mine.get(k) == other.get(k)]
    chk("A26-2", len(same) == len(names), f"复用函数与 dir25 逐字相同 {len(same)}/{len(names)}：{same}")

    # A26-3（rev 1.1 §C2：对 P 断言，不对 W）
    D8 = build_stream(SEEDS[0], D_MODEL, N_DIST, 4, THETA_GRID[1])
    checks3 = {}
    for r0 in (8, 35):
        c = make_core(D_MODEL, SEEDS[0], r0)
        drive(c, D8, "A")
        W = c._W().detach()
        P = W - c.W0
        r6 = int(torch.linalg.matrix_rank(P, rtol=RTOL_P).item())
        r4 = int(torch.linalg.matrix_rank(P, rtol=1e-4).item())
        rW = int(torch.linalg.matrix_rank(W, rtol=1e-12).item())
        checks3[r0] = {"rank_P_1e6": r6, "rank_P_1e4": r4, "rank_W_1e12": rW}
    ok3a = all(v["rank_P_1e6"] <= r0 and v["rank_P_1e4"] == r0 for r0, v in checks3.items())
    chk("A26-3a", ok3a, f"塑性秩不变式 rank(P,1e-6)≤r 且 rank(P,1e-4)=r：{checks3}")
    chk("A26-3b", all(v["rank_W_1e12"] == D_MODEL for v in checks3.values()),
        f"W 恒满秩 rank(W,1e-12)=d={D_MODEL}：{[v['rank_W_1e12'] for v in checks3.values()]}")

    # A26-4：原型键尺度/正交 + 抖动余弦精确
    proto = D8["proto"]
    orth = float((proto @ proto.T - torch.eye(N_DIST) * S_KEY).abs().max())
    cth = [float(torch.nn.functional.cosine_similarity(D8["K"][0, i].unsqueeze(0), proto[i].unsqueeze(0)).item())
           for i in range(N_DIST)]
    chk("A26-4", orth <= 1e-4 and max(abs(x - math.cos(THETA_GRID[1])) for x in cth) <= 1e-6,
        f"max|KᵀK−5I|={orth:.2e} ≤1e-4；max|cos(k,proto)−cosθ|={max(abs(x - math.cos(THETA_GRID[1])) for x in cth):.2e} ≤1e-6")

    # 先标定 r=8/θ=0.9（§C4），再用真实 c_τ 跑下面的管道断言
    cal89 = calibrate(8, THETA_GRID[1])

    # A26-5：写步计数
    T_ = 4 * N_DIST * 2 * M
    o, _ = run_cell(8, THETA_GRID[1], 4, SEEDS[0], tau=cal89["c_tau"])
    chk("A26-5", o["T"] == T_ and o["cells"]["A"]["issued"] == T_,
        f"写步计数 T={o['T']} == R·N·2·m={T_}；A 臂实发 {o['cells']['A']['issued']}")

    # A26-6′（rev 1.3 §C7）：指标对**固定 P** 的纯标量缩放不变（护栏）
    Dx = build_stream(SEEDS[0], D_MODEL, N_DIST, 4, THETA_GRID[1])
    cc6 = make_core(D_MODEL, SEEDS[0], 8)
    drive(cc6, Dx, "A")
    si = scalar_invariance(cc6._W().detach(), cc6.W0, Dx["proto"], Dx["vb"], N_DIST)
    chk("A26-6p", max(si.values()) - min(si.values()) == 0.0,
        f"纯标量缩放不变性 ret1(cP) 对 c∈{{0.37,1,2.5}}：{si}（逐位相同）")
    dA_diag = abs(o["cells"]["A"]["ret1"] - o["cells"]["C"]["ret1"])
    res["delta_rate_diag"] = {"r0": 8, "theta_deg": round(math.degrees(THETA_GRID[1]), 4),
                              "ret1_A": o["cells"]["A"]["ret1"], "ret1_C": o["cells"]["C"]["ret1"],
                              "delta_rate": o["cells"]["C"]["ret1"] - o["cells"]["A"]["ret1"],
                              "note": "rev 1.3 §C8：报告项（等质量、恒定门控 vs 全写），不参与判定"}

    # A26-7：门控非退化（逐格点标定）
    cals = [cal89] + [calibrate(r0, th) for r0 in R0_GRID for th in THETA_GRID
                      if (r0, th) != (8, THETA_GRID[1])]
    chk("A26-7", all(c["gate_ok"] for c in cals),
        "c_τ 逐格点标定（rev 1.2 §C4：发放率定标 ρ=0.40）：跳过率 "
        + " ".join(f"{c['skip_rate']:.2f}" for c in cals)
        + " ｜ cos 分离 "
        + " ".join(f"{c['gap']:+.3f}" for c in cals)
        + " ｜ 跳写中重复占比 "
        + " ".join(f"{c['skip_frac_repeat']:.2f}" for c in cals)
        + "（阈值：跳过率 0.35–0.45，gap ≥0.05，重复占比 ≥0.85）")

    # A26-8：等发放数 / 等质量
    chk("A26-8", abs(o["k_B"] - o["k_R"]) <= 1 and abs(o["cells"]["C"]["alpha"] - o["k_B"] / T_) <= 1e-12,
        f"|k_B−k_R|={abs(o['k_B'] - o['k_R'])} ≤1；(k_B,k_R)=({o['k_B']},{o['k_R']})；ᾱ={o['cells']['C']['alpha']:.6f} == k_B/T")

    # A26-9：无写臂
    c0 = make_core(D_MODEL, SEEDS[0], 8)
    W_before = c0._W().detach().clone()
    k0, _ = drive(c0, Dx, "NONE")
    W_after = c0._W().detach()
    m0 = metrics_plas(W_after, c0.W0, Dx["proto"], Dx["vb"], N_DIST)
    chk("A26-9", k0 == 0 and float((W_after - W_before).norm()) == 0.0,
        f"零发放臂：发放数 {k0}，‖W_after−W_before‖={float((W_after - W_before).norm()):.1f}（逐位 0）"
        f"；未训练核的 ret1={m0['ret1']:.4f}（诊断：$P$ 的初值非零是 DEC-15 的支撑初始化，不是空记忆）")

    # A26-10：稳定性前置
    rho1 = 1 - ETA * (1.0 * S_KEY + LAM / 2)
    chk("A26-10", rho1 > 0, f"ρ(α=1,λ=.05)=1−η(αS+λ/2)={rho1:.4f} > 0")

    res["stage_a_points"] = {f"r{r0}_{k}": v for r0, v in checks3.items() for k in ("rank_P_1e6",)}
    return o


# ══════════════════════════════════════════════ 判据（rev 1.1 §C1 阈值）
def ceil_(r0):
    return min(r0, N_DIST) / N_DIST


def thr_content(r0):
    return 0.10 if ceil_(r0) >= 0.5 else 0.05


def mean_over(rows, r0, theta_deg, R, arm, key="ret1"):
    v = [r["cells"][arm][key] for r in rows
         if r["r0"] == r0 and abs(r["theta_deg"] - theta_deg) < 1e-6 and r["R"] == R]
    return float(np.mean(v)) if v else float("nan")


def paired(rows, r0, theta_deg, R, arm1, arm2, key="ret1"):
    v = [r["cells"][arm1][key] - r["cells"][arm2][key] for r in rows
         if r["r0"] == r0 and abs(r["theta_deg"] - theta_deg) < 1e-6 and r["R"] == R]
    return float(np.mean(v)) if v else float("nan"), (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))


def criteria(rows, extra):
    print("=== 判据（rev 1.1 冻结阈值） ===")
    th9 = round(math.degrees(THETA_GRID[1]), 4)
    th0 = 0.0
    v = {}

    # (a′) 抖动冗余消耗容量（同质量：R=4，θ=0 vs θ=0.9）
    a_details, ok_a = [], True
    for r0 in (8, 35):
        d_jit = mean_over(rows, r0, th0, 4, "A") - mean_over(rows, r0, th9, 4, "A")
        ok = d_jit >= 0.10
        ok_a = ok_a and ok
        a_details.append({"r0": r0, "d_jit": d_jit,
                          "ret1_A_theta0": mean_over(rows, r0, th0, 4, "A"),
                          "ret1_A_theta9": mean_over(rows, r0, th9, 4, "A"), "ok": ok})
    v["a_premise"] = {"ok": ok_a, "details": a_details,
                      "note": "Δ_jit = ret1(A;θ=0,R=4) − ret1(A;θ=0.9,R=4) ≥ 0.10（r=8 与 r=35 均需）"}

    # (b) 主判据：内容（等发放数）与集中（等质量）
    b_details, ok_b = [], True
    for r0 in (8, 35):
        dc, dc_sd = paired(rows, r0, th9, 4, "B", "R")
        dk, dk_sd = paired(rows, r0, th9, 4, "B", "C")
        th = thr_content(r0)
        ok = (dc >= th) and (dk >= 0.05)
        ok_b = ok_b and ok
        b_details.append({"r0": r0, "c_r": ceil_(r0), "thr_content": th,
                          "d_content_B_minus_R": dc, "d_content_sd": dc_sd,
                          "d_conc_B_minus_C": dk, "d_conc_sd": dk_sd,
                          "ret1": {a: mean_over(rows, r0, th9, 4, a) for a in ARMS},
                          "n_correct": {a: mean_over(rows, r0, th9, 4, a, "n_correct") for a in ARMS},
                          "ok": ok})
    v["b_main"] = {"ok": ok_b, "details": b_details,
                   "note": "Δ_content=ret1(B)−ret1(R) ≥ thr(r) 且 Δ_conc=ret1(B)−ret1(C) ≥ 0.05，r=8 与 r=35 均需"}

    # (c) 结构对照 r=896
    dc896, sd896 = paired(rows, 896, th9, 4, "B", "R")
    v["c_null"] = {"ok": abs(dc896) <= 0.10, "d_content_B_minus_R": dc896, "sd": sd896,
                   "ret1": {a: mean_over(rows, 896, th9, 4, a) for a in ARMS},
                   "note": "|ret1(B)−ret1(R)| ≤ 0.10（无塑性稀缺 ⇒ 预测 null）"}

    # (d′) 指标护栏（rev 1.3 §C7）：纯标量缩放不变性 —— 逐格点在 A 臂末态上重算
    d_det, ok_d = [], True
    for r0 in R0_GRID:
        for th in (th0, th9):
            for seed in SEEDS:
                D = build_stream(seed, D_MODEL, N_DIST, 4, th)
                c = make_core(D_MODEL, seed, r0)
                drive(c, D, "A")
                si = scalar_invariance(c._W().detach(), c.W0, D["proto"], D["vb"], N_DIST)
                ok = (max(si.values()) - min(si.values())) == 0.0
                ok_d = ok_d and ok
                d_det.append({"r0": r0, "theta_deg": round(math.degrees(th), 4), "seed": seed,
                              "ret1_by_scale": {str(k): x for k, x in si.items()}, "ok": ok})
    v["d_guard"] = {"ok": ok_d, "n_cells": len(d_det),
                    "details": [x for x in d_det if not x["ok"]][:5],
                    "note": "ret1(cP)=ret1(P) 对 c∈{0.37,1,2.5}，逐格点（护栏，非判别力）"}

    # §C8 报告项：恒定门控的速率效应量级（不参与判定）
    v["rate_effect_report"] = [{"r0": r0, "theta_deg": th,
                                "ret1_A": mean_over(rows, r0, th, 4, "A"),
                                "ret1_C": mean_over(rows, r0, th, 4, "C"),
                                "delta_rate_C_minus_A": mean_over(rows, r0, th, 4, "C")
                                - mean_over(rows, r0, th, 4, "A")}
                               for r0 in R0_GRID for th in (th0, th9)]

    # (e) 框架 §3.5 代数：α=.5,λ=.05 vs α=1,λ=.10（幅度敏感读 cosP）
    e_details = []
    for r0 in (8, 896):
        c1 = [x for x in extra if x["r0"] == r0 and x["arm"] == "half_lam05"]
        c2 = [x for x in extra if x["r0"] == r0 and x["arm"] == "full_lam10"]
        if c1 and c2:
            m1 = float(np.mean([x["cosP"] for x in c1]))
            m2 = float(np.mean([x["cosP"] for x in c2]))
            e_details.append({"r0": r0, "cosP_half_lam05": m1, "cosP_full_lam10": m2,
                              "abs_diff": abs(m1 - m2), "ok": abs(m1 - m2) <= 0.05})
    ok_e = bool(e_details) and all(x["ok"] for x in e_details)
    v["e_algebra"] = {"ok": ok_e, "details": e_details,
                      "note": "|cosP(α=.5,λ=.05) − cosP(α=1,λ=.10)| ≤ 0.05（r=8 与 896）"}

    # 合成（rev 1.1 §C1 末段）
    if not v["a_premise"]["ok"]:
        main = "PREMISE-ABSENT"
    elif not v["d_guard"]["ok"]:
        main = "INDETERMINATE"
    elif v["b_main"]["ok"] and v["c_null"]["ok"]:
        main = "CONFIRMED"
    else:
        main = "FALSIFIED"
    v["main"] = {"ok": main == "CONFIRMED", "verdict": main}
    for k in ("a_premise", "b_main", "c_null", "d_guard", "e_algebra"):
        print(f"  [{'PASS' if v[k]['ok'] else 'FAIL'}] {k}")
        if k in ("a_premise", "b_main"):
            for d in v[k]["details"]:
                print(f"        {d}")
    print(f"  ⇒ H26 判定：**{main}**")
    return v


# ══════════════════════════════════════════════ 主
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26r10")
    args = ap.parse_args()
    t0 = time.time()

    if args.stage == "stageA":
        stage_a()
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}"
              f"（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"], res["passed"] = "stageA", ok
    else:
        # 标定（formal 需要全部格点；smoke 只 1 格）
        cells = ([(8, THETA_GRID[1])] if args.stage == "smoke"
                 else [(r0, th) for r0 in R0_GRID for th in THETA_GRID])
        taus = {}
        for r0, th in cells:
            cal = calibrate(r0, th)
            taus[(r0, round(math.degrees(th), 4))] = cal["c_tau"] if cal["gate_ok"] else None
        seeds = [SEEDS[0]] if args.stage == "smoke" else SEEDS
        rows = []
        plan = [dict(r0=r0, th=th, R=R, seed=s) for r0, th in cells for R in R_GRID for s in seeds]
        print(f"=== {args.stage}（{len(plan)} 格点） ===")
        for p in plan:
            tau = taus[(p["r0"], round(math.degrees(p["th"]), 4))]
            o, _ = run_cell(p["r0"], p["th"], p["R"], p["seed"], tau=tau)
            if p["R"] == 1:                                    # §C3：臂坍缩，只跑 A
                o["collapsed_to_A"] = ["B", "C", "R", "D"]
                for a in ["B", "C", "R", "D"]:
                    o["cells"][a] = dict(o["cells"]["A"])
                    o["cells"][a]["collapsed"] = True
            rows.append(o)
            c = o["cells"]
            print(f"  r={p['r0']:3d} θ={o['theta_deg']:7.4f}° R={p['R']} seed={p['seed']} "
                  f"k=(B {o['k_B']:3d}/R {o['k_R']:3d}/T {o['T']:3d}) "
                  f"ret1 A={c['A']['ret1']:.3f} B={c['B']['ret1']:.3f} C={c['C']['ret1']:.3f} "
                  f"D={c['D']['ret1']:.3f} R={c['R']['ret1']:.3f} ({o['wall']:.0f}s)")
            if args.stage == "smoke":
                break
        extra = []
        if args.stage == "formal":                             # (e) 副臂
            for r0 in (8, 896):
                for s in SEEDS:
                    for arm, a_, l_ in (("half_lam05", 0.5, 0.05), ("full_lam10", 1.0, 0.10)):
                        o, _ = run_cell(r0, THETA_GRID[1], 4, s, tau=None, lam=l_)
                        # α 由 run_cell 内部置 1 ⇒ 用 A 臂 + 显式 α 重跑（等质量口径不适用，属框架代数核）
                        global ALPHA, LAM
                        ALPHA, LAM = a_, l_
                        cc = make_core(D_MODEL, s, r0)
                        D = build_stream(s, D_MODEL, N_DIST, 4, THETA_GRID[1])
                        drive(cc, D, "A")
                        m = metrics_plas(cc._W().detach(), cc.W0, D["proto"], D["vb"], N_DIST)
                        extra.append({"r0": r0, "seed": s, "arm": arm, "alpha": a_, "lam": l_, **m})
                        ALPHA, LAM = 1.0, 0.05
                        print(f"  (e) r={r0:3d} seed={s} {arm}: cosP={m['cosP']:.4f}")
        res["rows"], res["extra"] = rows, extra
        if args.stage == "formal":
            res["verdicts"] = criteria(rows, extra)
            res["passed"] = res["verdicts"]["main"]["ok"]

    res["meta"] = {"wall_seconds": round(time.time() - t0, 1), "device": "cpu", "d": D_MODEL,
                   "eta": ETA, "m": M, "s_key": S_KEY, "n_dist": N_DIST,
                   "r_grid": R0_GRID, "theta_grid_deg": [0.0, round(math.degrees(THETA_GRID[1]), 4)],
                   "R_grid": R_GRID, "arms": ARMS, "seeds": SEEDS, "seed_pilot": SEED_PILOT,
                   "stage": args.stage, "card": "rev1.0 + rev1.1(§C1–§C3)"}
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
