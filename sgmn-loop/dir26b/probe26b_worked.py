"""DIR-26-B · probe26b_worked.py —— 测量对象的**工作算例**（验证 P 臂与 $\\rho$ 曲线口径，不产生判据）

背景（D-33）：$r{=}8$ 上内容门控 B 比等发放数的随机门控 R 高 **+0.2914**（5/5 种子）。
但 D-33 留下 **未分离的混合量**：$\Delta_{\\text{conc}}=\\mathrm{ret}_1(B)-\\mathrm{ret}_1(C)=+0.2229$
里混着**衰减暴露**（C 衰减 560 次 vs B 372 次，因 $G=\\alpha\\cdot$数据项$+\\lambda(W-W^0)$ 中 $\\alpha$ **不缩放** $\\lambda$ 项，DEF-26-1）。

本算例要验的两件事：
  ① **P 臂（位置均衡写）**：与 B **逐位等发放数** $k_B$，但按**位置**（每条目轮转、条目内跨轮均匀）选取，
     **完全不看内容**。若 $\\mathrm{ret}_1(P)\\approx\\mathrm{ret}_1(B)$ ⇒ B 的收益只是"均衡覆盖"，与内容无关；
     若 $\\mathrm{ret}_1(P)\\ll\\mathrm{ret}_1(B)$ ⇒ 收益确实是内容驱动的。⇒ B vs P 才是 D-33 缺失的那个对照。
  ② **$\\rho$ 曲线**：跳过率 $\\rho\\in\\{0.2,0.4,0.6\\}$ 下 $\\mathrm{ret}_1(B_\\rho)$、$\\mathrm{ret}_1(P_\\rho)$ 与发放数的关系（"省写"曲线）。

手算预期（$d{=}16$、$r_0{=}4$、$N{=}4$、$R{=}4$、$k_B\\approx0.6T$）：
  · P 的每条目发放数只能是 $\\lfloor k_B/N\\rfloor$ 或 $+1$（差额 $\\le1$）⇒ **$P$ 不看内容但覆盖均匀**；
  · 若"均衡覆盖"足以解释收益，则 $P\\approx B$；D-33 的 $D$ 臂（只写首现、140 次、欠覆盖）为 0.337，说明覆盖**重要**，
    但覆盖的**方式**（均匀 vs 按需）是否等价，只有 P 臂能回答。
"""
import json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26")
from R1_gating import (build_stream, drive, metrics_plas, make_core, stream_idx,
                       unit, D_MODEL, N_DIST, M, S_KEY)

D16 = 16
R0 = 4
R_PRES = 4
THETA = math.radians(25.8419)
SEED = 20261028
RHOS = (0.2, 0.4, 0.6)


def keep_positions(n, r_pres, k):
    """P 臂的保留位置：按**位置**均匀分配（条目间差额 ≤1，条目内跨轮均匀），**不看内容**。"""
    per_item_tot = r_pres * 2 * M
    base, rem = divmod(int(k), n)
    keep = set()
    for i in range(n):
        c_i = base + (1 if i < rem else 0)
        idxs = sorted({int(round(x)) for x in np.linspace(0, per_item_tot - 1, max(c_i, 1))})
        j = 0
        while len(idxs) < c_i:                     # 去重后补齐（保持有序）
            if j not in idxs:
                idxs.append(j)
                idxs.sort()
            j += 1
        for pos in idxs[:c_i]:
            rho, rest = divmod(pos, 2 * M)
            mbr, m = divmod(rest, M)
            keep.add(stream_idx(rho, i, mbr, m, n))
    return keep


def per_item_counts(keep, n, r_pres):
    cnt = [0] * n
    for t in keep:
        cnt[(t // (2 * M)) % n] += 1
    return cnt


def tau_from_pilot(d_model, r0, n, rho, theta=THETA, seed=SEED):
    """$\rho$ 定标：pilot 上"若全写"的 cos 分布的 $1-\rho$ 分位（rev 1.2 §C4 的规则，pilot 种子独立）。"""
    D = build_stream(seed + 3, d_model, n, R_PRES, theta)
    c = make_core(d_model, seed + 3, r0)
    _, log = drive(c, D, "A")
    return float(np.quantile([e["cos"] for e in log], 1.0 - rho))


def cell(d_model, n, r0, rho):
    D = build_stream(SEED, d_model, n, R_PRES, THETA)
    T = R_PRES * n * 2 * M
    tau = tau_from_pilot(d_model, r0, n, rho)
    out = {"r0": r0, "N": n, "rho_target": rho, "T": T, "c_tau": tau, "arms": {}}

    cB = make_core(d_model, SEED, r0)
    kB, _ = drive(cB, D, "B", tau=tau)
    out["arms"]["B"] = {**metrics_plas(cB._W().detach(), cB.W0, D["proto"], D["vb"], n), "issued": kB}

    gk = torch.Generator().manual_seed(SEED + 7)
    keepR = set(torch.randperm(T, generator=gk)[:kB].tolist())
    cR = make_core(d_model, SEED, r0)
    kR, _ = drive(cR, D, "R", keep=keepR)
    out["arms"]["R"] = {**metrics_plas(cR._W().detach(), cR.W0, D["proto"], D["vb"], n), "issued": kR}

    keepP = keep_positions(n, R_PRES, kB)
    cP = make_core(d_model, SEED, r0)
    kP, _ = drive(cP, D, "R", keep=keepP)   # 同 keep 语义；标签在 JSON 里记 P
    out["arms"]["P"] = {**metrics_plas(cP._W().detach(), cP.W0, D["proto"], D["vb"], n), "issued": kP}
    out["P_item_counts"] = per_item_counts(keepP, n, R_PRES)

    import R1_gating
    R1_gating.ALPHA = kB / T
    cC = make_core(d_model, SEED, r0)
    kC, _ = drive(cC, D, "C")
    R1_gating.ALPHA = 1.0
    out["arms"]["C"] = {**metrics_plas(cC._W().detach(), cC.W0, D["proto"], D["vb"], n),
                        "issued": kC, "alpha": kB / T}

    cA = make_core(d_model, SEED, r0)
    kA, _ = drive(cA, D, "A")
    out["arms"]["A"] = {**metrics_plas(cA._W().detach(), cA.W0, D["proto"], D["vb"], n), "issued": kA}
    return out


if __name__ == "__main__":
    rows = []
    for d_model, n, r0s in ((D16, 4, (2, 4)), (D16, 8, (2, 4))):
        for r0 in r0s:
            for rho in RHOS:
                o = cell(d_model, n, r0, rho)
                a = o["arms"]
                print(f"d={d_model} N={n} r={r0} ρ={rho}: k_B={a['B']['issued']}/{o['T']} "
                      f"ret1 A={a['A']['ret1']:.3f} B={a['B']['ret1']:.3f} C={a['C']['ret1']:.3f} "
                      f"R={a['R']['ret1']:.3f} **P={a['P']['ret1']:.3f}** | P 每条目={o['P_item_counts']} "
                      f"| k_P={a['P']['issued']}")
                rows.append(o)
    # 判读：B 与 P 的差（本文件不产生判据，仅报口径是否可用）
    for o in rows:
        a = o["arms"]
        print(f"  ⇒ r={o['r0']} N={o['N']} ρ={o['rho_target']}: "
              f"B−P={a['B']['ret1'] - a['P']['ret1']:+.3f}  B−R={a['B']['ret1'] - a['R']['ret1']:+.3f}  "
              f"P−C={a['P']['ret1'] - a['C']['ret1']:+.3f}")
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "probe26b_worked.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rows, open(p, "w"), ensure_ascii=False, indent=1, default=float)
    print("→", p)
