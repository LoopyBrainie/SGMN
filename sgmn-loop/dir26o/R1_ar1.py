r"""DIR-26-O · R1_ar1.py —— 因子化装置在 **AR(1) 输入**下的稳定性与剂量（卡片 rev 1.1 冻结）

卡片：`revolve/DIR-26O/DIRECTION_rev26o_v11.md`（rev 1.0 → **rev 1.1**：只订正 A26O-4 的断言对象，阈值未动）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，**零改动**）

唯一改动（相对 `dir26n/R1_gating15.py`）——按纪律「只改测量层与独立变量」：
  ① **独立变量**：写流的**键**换成「边缘精确匹配的 AR(1) 输入」（§2.1）：抖动方向沿**轮次**按 $\varphi$ 做 AR(1) 过滤，
     $g_{i,\rho}=\varphi g_{i,\rho-1}+\sqrt{1-\varphi^2}U_{i,\rho}$；$g$ 的协方差恒为 $\sigma^2I$ ⇒
     **键的边缘分布与 $\varphi$ 无关**；$\varphi=0$ 走**原语句** ⇒ **位级锚点可用**；所有 $\varphi$ 档共用同一 $U$ ⇒ **逐种子配对**。
  ② **测量层**：新增 $\varphi$ 档、逐种子配对差、跨种子分位数（p50/p90/p99，D-02 的方法论：不得用均值平方）。
  **不改**：写流顺序、每条目写次数、值张量、读流、$\tau$ 的标定与判据算法、口径 O / 控制口径 X、种子表。
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

for _p in ("dir26o", "dir26n", "dir26m", "dir26l", "dir26k", "dir26j", "dir26i", "dir26h",
           "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating12 as K
import R1_gating14 as M14
import probe26n_feas as PR
from R1_gating import D_MODEL, DTYPE, M, S_KEY, drive, make_core, unit
from R1_gating5 import theta_of
from R1_gating12 import (R_HO4, R_HO8, R_PRES, SE_K, cal_at, measure_prec, read_stream)
from R1_gating7 import md5
from rerun_missing import measure_hit_bounded

N_READ = 20
N_KV = 60
PHIS = (0.0, 0.3, 0.6, 0.9)
SEEDS = K.SEEDS
D26N = "/home/ec2-user/sgmn-loop/dir26n/out/DIR26Nr11_formal.json"
TOL_RET1 = 0.025
TOL_MU = 0.004
SIGN_MIN = 27
T_UNIT = R_PRES * N_KV * 2 * G.M
OUT = "/home/ec2-user/sgmn-loop/dir26o/out"
RET1_E0_BAND = (0.70, 0.90)
RET1_TAU_BAND = (0.80, 0.95)
MOM_REL = 0.02
RHO_U_SPAN = 0.25
PROF = ("ret1", "mu", "issued", "P_norm")
ARMS = ("E0", "tau")


# ══════════════════════════════════════════════ 写流（唯一改动点：AR(1) 抖动）
def build_stream_ar1(b, seed, d, n, r_pres, theta, phi):
    r"""逐字派生自 `R1_gating.build_stream`；$\varphi=0$ 走**原语句**（位级锚点），$\varphi>0$ 走 AR(1) 分支。

    ★ 边缘精确匹配（卡片 §2.1）：$g_{i,\rho}$ 各向同性且平稳（$\mathrm{Cov}(g)=\sigma^2I$ 与 $\varphi$ 无关）
      ⇒ 投影归一化后方向均匀 ⇒ 键的边缘分布与 $\varphi$ 无关 ⇒ 唯一变化的是**时间相关结构**。
    ★ $\varphi=0$：$g=U$ ⇒ $K$ 与原装置**逐位**相同。
    ★ 配对：所有 $\varphi$ 档共用同一 `U`（同一 generator 抽取顺序）⇒ 逐种子差只来自输入的时间结构。
    """
    g = torch.Generator().manual_seed(seed + 1)
    E = torch.zeros(n, d)
    for i in range(n):
        E[i, i] = 1.0
    U = unit(torch.randn(n, r_pres, d, generator=g))          # ★ 与原装置**逐字**相同
    # ── 抖动方向：φ=0 用原路径；φ>0 用 AR(1) 过滤后的路径（兜底**不消耗 RNG**，保持 va/vb 跨 φ 逐位相同）
    Uc0 = torch.zeros(n, r_pres, d, dtype=DTYPE)
    for rho in range(r_pres):
        for i in range(n):
            u = U[i, rho] - (U[i, rho] @ E[i]) * E[i]
            u = unit(u) if float(u.norm()) > 1e-6 else unit(torch.randn(d, generator=g))
            Uc0[i, rho] = u
    Uc = Uc0
    g2 = float(torch.mean((Uc0 * Uc0).sum(-1)).item())
    if float(phi) != 0.0:
        sq = math.sqrt(1.0 - float(phi) ** 2)
        Gf = torch.empty_like(U)
        for i in range(n):
            prev = U[i, 0]
            Gf[i, 0] = prev
            for rho in range(1, r_pres):
                prev = float(phi) * prev + sq * U[i, rho]
                Gf[i, rho] = prev
        Uc = torch.zeros(n, r_pres, d, dtype=DTYPE)
        for rho in range(r_pres):
            for i in range(n):
                u = Gf[i, rho] - (Gf[i, rho] @ E[i]) * E[i]
                Uc[i, rho] = unit(u) if float(u.norm()) > 1e-6 else Uc0[i, rho]
        g2 = float(torch.mean((Gf * Gf).sum(-1)).item())
    # ── 键（与原装置逐字相同的构造式）
    K = torch.zeros(r_pres, n, d, dtype=DTYPE)
    for rho in range(r_pres):
        for i in range(n):
            K[rho, i] = math.sqrt(S_KEY) * unit(math.cos(theta) * E[i] + math.sin(theta) * Uc[i, rho])
    proto = math.sqrt(S_KEY) * E.clone()
    va = unit(torch.randn(n, d, generator=g))
    r = torch.randn(n, d, generator=g)
    vb = unit(r - (r * va).sum(-1, keepdim=True) * va)
    D = {"K": K, "proto": proto, "va": va, "vb": vb, "n": n, "r_pres": r_pres}
    if int(n) == N_KV and b < N_KV - N_READ:
        pat = M14.pat_of(b, n)
        D["va"] = D["va"][pat].clone()
        D["vb"] = D["vb"][pat].clone()
        D["pat"] = pat
    # ── 诊断：抖动相关 ρ_u、键相关 ρ_k、键矩（Stage A A26O-3/A26O-4 的输入）
    rho_u = rho_k = 0.0
    cnt = 0
    kn = torch.sqrt((K * K).sum(-1))                    # ★ DEF-26O-2：ρ_k 必须是**归一化**相关
    for i in range(n):
        for rho in range(r_pres - 1):
            rho_u += float(Uc[i, rho] @ Uc[i, rho + 1])
            rho_k += float(K[rho, i] @ K[rho + 1, i]) / float(kn[rho, i] * kn[rho + 1, i])
            cnt += 1
    km = {"k2": [], "cosp": [], "k4": [], "kd": []}
    for rho in range(r_pres):
        for i in range(n):
            k = K[rho, i]
            km["k2"].append(float(k @ k))
            km["cosp"].append(float(k @ proto[i]) / (float(k.norm()) * float(proto[i].norm())))
            km["k4"].append(float(k @ k) ** 2)
    for rho in range(r_pres):                           # ★ DEF-26O-3：kd 必须跨**全部轮**（第 0 轮按构造与 φ 恒等）
        for i in range(0, n - 1, 7):
            for j in range(i + 1, n, 7):
                km["kd"].append(float((K[rho, i] - K[rho, j]).norm()))
    D["diag"] = {"rho_u": rho_u / max(cnt, 1), "rho_k": rho_k / max(cnt, 1), "g2": g2,
                 "km": {k: float(np.mean(v)) for k, v in km.items()}}
    return D


# ══════════════════════════════════════════════ 单元
def run_unit(seed, phi, tau, b=N_KV):
    r"""一条 (seed, φ) 单元 = 2 条驱动：$E0$（无条件写）与 `tau`（内容门控）。"""
    D = build_stream_ar1(b, seed, D_MODEL, int(N_KV), R_PRES, theta_of(0.9), phi)
    assert int(D["n"]) == N_KV, f"写流 N 不符：{D['n']} != {N_KV}"
    RS8, RS4 = read_stream(seed, R_HO8), read_stream(seed, R_HO4)
    proto, vb = D["proto"][:N_READ], D["vb"][:N_READ]
    out = {"n_kv": N_KV, "n_read": N_READ, "seed": int(seed), "phi": float(phi), "b": int(b),
           "m": N_READ + int(b), "T": T_UNIT, "arms": {}, "diag": D["diag"]}

    def rec(tag, core, issued, extra=None):
        W = core._W().detach()
        P = W - core.W0
        m8 = measure_hit_bounded(P, RS8["K"], vb, N_READ, R_HO8)
        p4 = measure_prec(P, RS4["K"], vb, N_READ, R_HO4)
        mx = measure_hit_bounded(P, RS8["K"], RS8["vb"], N_READ, R_HO8)      # 控制口径 X（外来值）
        assert m8["ret1"] == measure_prec(P, RS8["K"], vb, N_READ, R_HO8)["ho_mean"], "μ/精度口径不符（同抖动档）"
        mp = G.metrics_plas(W, core.W0, proto, vb, N_READ)
        d = {"issued": int(issued), "wsig": K.wsig(W),
             "W_unchanged": bool(torch.equal(core._W().detach(), W)),
             "clean": mp["ret1"], "P_norm": mp["P_norm"], "cosP": mp["cosP"],
             "mu": m8["mu"], "mu_sd": m8["mu_sd"], "mu_read": m8["n_read"],
             "ret1": m8["ret1"], "ret1_maj": m8["ret1_maj"], "maj_need": m8["maj_need"],
             "ho8_mean": m8["ret1"], "ho8_maj": m8["ret1_maj"],
             "ho4_mean": p4["ho_mean"], "ho4_maj": p4["ho_maj"],
             "mu_ret1_consistent": m8["mu_ret1_consistent"],
             "X": {"mu": mx["mu"], "mu_sd": mx["mu_sd"], "ret1": mx["ret1"]}}
        if extra:
            d.update(extra)
        out["arms"][tag] = d

    c = make_core(D_MODEL, seed, K.R0)
    kA, _ = drive(c, D, "A")
    rec("E0", c, kA)
    c = make_core(D_MODEL, seed, K.R0)
    ki, lg, _dg = PR.drive_rule(c, D, "A", tau=tau)          # ★ 与 DIR-26-N 的 τ 臂**同代码路径**
    sk = [e for e in lg if not e["issued"]]
    rec("tau", c, ki, {"gate_rate": len(sk) / len(lg),
                       "skip_frac_repeat": (float(np.mean([e["round"] > 0 for e in sk])) if sk else float("nan")),
                       "qbar": ki / N_KV, "log_n": len(lg)})
    return out


# ══════════════════════════════════════════════ 判据骨架
def _pair(vals_a, vals_b):
    a, b = np.asarray(vals_a, float), np.asarray(vals_b, float)
    d = a - b
    n = len(d)
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    return {"mean": float(np.mean(d)), "sd": sd, "se": se, "k": SE_K * se, "n": n,
            "pos": int((d > 0).sum()), "neg": int((d < 0).sum()),
            "vals": [round(float(x), 6) for x in d]}


def _mono(m):
    r"""P3：三个 $\varphi$ 档的配对均值 —— 非零者同号 **且** $|m|$ 随 $\varphi$ 不减。"""
    nz = [x for x in m if x != 0.0]
    if not nz:
        return False, "全零"
    same = all(x > 0 for x in nz) or all(x < 0 for x in nz)
    nondec = abs(m[0]) <= abs(m[1]) + 1e-12 and abs(m[1]) <= abs(m[2]) + 1e-12
    return bool(same and nondec), f"m={['%+.4f' % x for x in m]} same_sign={same} nondec={nondec}"


def crit(rows):
    r"""判决（卡片 §1.1 / §2.2 冻结）。"""
    seeds = sorted({int(r["seed"]) for r in rows})
    by = {(int(r["seed"]), float(r["phi"])): r for r in rows}
    phis = sorted({float(r["phi"]) for r in rows})          # ★ DEF-26O-6：用行里**实际存在**的 φ 档
    assert phis[0] == 0.0, f"缺 φ=0 基准档：{phis}"
    for s in seeds:
        for p in phis:
            assert (s, p) in by, f"缺单元 seed={s} phi={p}"
    deltas = {}
    for arm in ARMS:
        for f in PROF:
            for p in phis[1:]:
                deltas[f"{arm}|{f}|{p}"] = _pair(
                    [by[(s, p)]["arms"][arm][f] for s in seeds],
                    [by[(s, 0.0)]["arms"][arm][f] for s in seeds])
    v = {"n_seed": len(seeds), "phis": phis, "seeds": seeds, "deltas": deltas}
    # ── P1（输入过程：边缘不变 + 相关性单调）—— 用**正式轮**的全部种子复核
    km = {}
    for f in ("k2", "cosp", "k4", "kd"):
        vals = {p: float(np.mean([by[(s, p)]["diag"]["km"][f] for s in seeds])) for p in phis}
        rel = (max(vals.values()) - min(vals.values())) / max(abs(np.mean(list(vals.values()))), 1e-12)
        km[f] = {"vals": vals, "rel_range": float(rel), "ok": bool(rel <= MOM_REL)}
    rho_u = {p: float(np.mean([by[(s, p)]["diag"]["rho_u"] for s in seeds])) for p in phis}
    rho_k = {p: float(np.mean([by[(s, p)]["diag"]["rho_k"] for s in seeds])) for p in phis}
    mono = all(rho_u[phis[i + 1]] > rho_u[phis[i]] for i in range(len(phis) - 1))
    span = rho_u[phis[-1]] - rho_u[phis[0]]
    v["P1"] = {"kmom": km, "rho_u": rho_u, "rho_k": rho_k,
               "mono_ok": bool(mono), "span": float(span),
               "ok": bool(all(x["ok"] for x in km.values()) and mono and span >= RHO_U_SPAN)}
    # ── P2（效应存在）：任一 (臂, 量) 在 φ=0.9 达标
    hits = []
    for arm in ARMS:
        for f, tol in (("ret1", TOL_RET1), ("mu", TOL_MU)):
            p = deltas[f"{arm}|{f}|{phis[-1]}"]
            ok = bool(abs(p["mean"]) >= tol and max(p["pos"], p["neg"]) >= SIGN_MIN)
            if ok:
                hits.append({"arm": arm, "metric": f, "tol": tol, "d": p})
    v["P2_hits"] = hits
    v["P2"] = bool(hits)
    # ── P3（单调性）：对每个命中的 (臂, 量) 判定
    p3 = {}
    for h in hits:
        if len(phis) < 4:                       # ★ Stage C 只有 2 档 ⇒ P3 不适用（非判决性）
            p3[f"{h['arm']}|{h['metric']}"] = {"m": [], "ok": None, "why": "φ 档不足，P3 不适用（Stage C 非判决）"}
            continue
        m = [deltas[f"{h['arm']}|{h['metric']}|{p}"]["mean"] for p in (0.3, 0.6, 0.9)]
        ok, why = _mono(m)
        p3[f"{h['arm']}|{h['metric']}"] = {"m": m, "ok": ok, "why": why}
    v["P3"] = p3
    v["P3_ok"] = bool(hits) and all(bool(x["ok"]) for x in p3.values())
    # ── 诊断（不判定）：跨种子分位数（D-02 的方法论）
    q = {}
    for arm in ARMS:
        for f in PROF:
            for p in phis:
                arr = np.array([by[(s, p)]["arms"][arm][f] for s in seeds], float)
                q[f"{arm}|{f}|{p}"] = {str(pq): float(np.percentile(arr, pq)) for pq in (50, 90, 99)}
                q[f"{arm}|{f}|{p}"]["min"] = float(arr.min())
                q[f"{arm}|{f}|{p}"]["max"] = float(arr.max())
    v["quantiles"] = q
    # ── 锚点（位级）：φ=0 的 E0/tau 臂 vs DIR-26-N 正式轮 b=40
    # ★ DEF-26O-7：锚点必须取**同 b** 的行（Stage C 判 b=4 时不能拿 b=40 的行比）
    b_rows = sorted({int(r["b"]) for r in rows})
    assert len(b_rows) == 1, f"一次 crit 只能判一个世界：{b_rows}"
    b_rows = b_rows[0]
    # ★ DEF-26O-9/10：主轮的行标签是 60（`run_unit` 的默认 `b=N_KV`），而**恒等世界**在档案里记作 40
    #   （`build_stream_ar1(60,...)` 与 `build_stream_ar1(40,...)` 写流**逐位相同**：折叠条件 `b<40` 均不触发）
    #   ⇒ 锚点查表按**世界**（40=恒等、4=冗余）而不是按标签。
    world = 40 if b_rows in (40, N_KV) else b_rows
    arch = {(int(r["n_kv"]), int(r["seed"]), int(r["b"])): r for r in json.load(open(D26N))["rows"]}
    AF_E0 = ("issued", "ret1", "mu", "clean", "wsig")
    AF_TAU = ("issued", "ret1", "wsig")
    ncmp, bads = 0, []
    for s in seeds:
        a = arch.get((N_KV, s, world))
        if not a:
            bads.append(f"{s}: DIR-26-N 缺该种子")
            continue
        for f in AF_E0:
            ncmp += 1
            if by[(s, 0.0)]["arms"]["E0"][f] != a["arms"]["E0"][f]:
                bads.append(f"{s}/E0.{f}")
        for f in AF_TAU:
            ncmp += 1
            if by[(s, 0.0)]["arms"]["tau"][f] != a["arms"]["tau"][f]:
                bads.append(f"{s}/tau.{f}")
    v["anchor"] = {"ok": not bads, "n_cmp": ncmp, "n_bad": len(bads), "bad": bads[:8]}
    # ── 正对照：**卡片 §2.2 的「正对照」= 与 DIR-26-N 的 E0/τ 逐位相等（即锚点）**；
    #    量程带是 **Stage A 断言 A26O-10**（只对主世界 b=40 标定）⇒ 这里只作**诊断**。
    # ★ DEF-26O-8：原实现把 b=40 的量程带当成正式轮的「正对照」⇒ Stage C（b=4）被误判 PREMISE-ABSENT
    #   （b=4 的历史 ret1(E0)=0.5652 本就不在 [0.70,0.90] 内）⇒ 现按卡片口径：正对照 = 锚点。
    e0m = float(np.mean([by[(s, 0.0)]["arms"]["E0"]["ret1"] for s in seeds]))
    tam = float(np.mean([by[(s, 0.0)]["arms"]["tau"]["ret1"] for s in seeds]))
    bands = (RET1_E0_BAND, RET1_TAU_BAND) if b_rows == 40 else None
    v["control"] = {"ret1_E0_phi0": e0m, "ret1_tau_phi0": tam, "b": b_rows,
                    "band_ok": (bool(bands[0][0] <= e0m <= bands[0][1] and bands[1][0] <= tam <= bands[1][1])
                                if bands else None),
                    "world": world, "ok": True}
    # ── 通道
    if not v["anchor"]["ok"] or not v["control"]["ok"] or not v["P1"]["ok"]:
        ch = "IND" if not v["anchor"]["ok"] else "PREMISE-ABSENT"
    elif not v["P2"]:
        ch = "CORR-ROBUST"
    elif len(phis) < 4:
        ch = "N/A-STAGE-C-NONDECISIVE"        # ★ P3 需要 3 个正 φ 档；Stage C 只 2 档 ⇒ 非判决性
    elif not v["P3_ok"]:
        ch = "CORR-NONMONO"
    else:
        ch = "CORR-SENSITIVE"
    v["channel"] = ch
    v["verdict_ok"] = bool(ch == "CORR-SENSITIVE")
    return v


def chk(checks, cid, ok, detail):
    r"""Stage A 断言（本地收集；纪律 ㊱：空断言集不得记 PASS）。"""
    checks.append({"id": cid, "ok": bool(ok), "msg": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}  {detail}")
    return bool(ok)


# ══════════════════════════════════════════════ Stage A
def stageA():
    v = {"checks": [], "rows": []}
    v["dev_md5"] = md5("/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py")
    chk(v["checks"], "A26O-1", v["dev_md5"] == M14.DEV_MD5, f"装置 md5 {v['dev_md5']}（期望 {M14.DEV_MD5}）")
    # ★ DEF-26O-4：rev 1.1 首跑把本断言写成**占位哈希 + 错误的长度判断** ⇒ 恒 FAIL（假失败）。
    #   正确口径：参照脚本的 md5 必须等于 **MANIFEST.md（cycle 27 的存档）** 记录的值。
    M15_MD5 = "a4de914d277df1ace1e52a78bae4d84d"
    m15 = md5("/home/ec2-user/sgmn-loop/dir26n/R1_gating15.py")
    v["ref_md5"] = m15
    chk(v["checks"], "A26O-2", m15 == M15_MD5,
        f"参照脚本 `dir26n/R1_gating15.py` md5={m15}（须等于 MANIFEST 存档 {M15_MD5}；导入链逐字沿用）")

    tau = cal_at(N_KV)["c_tau"]
    v["tau"] = tau
    s0 = SEEDS[0]

    # ── A26O-3 / A26O-4：边缘不变 + 相关性单调（pilot 种子）
    pilots = SEEDS[:5]
    km = {f: {} for f in ("k2", "cosp", "k4", "kd")}
    ru, rk = {p: [] for p in PHIS}, {p: [] for p in PHIS}
    for s in pilots:
        for p in PHIS:
            D = build_stream_ar1(40, s, D_MODEL, N_KV, R_PRES, theta_of(0.9), p)
            for f in km:
                km[f][p] = km[f].get(p, 0.0) + D["diag"]["km"][f] / len(pilots)
            ru[p].append(D["diag"]["rho_u"])
            rk[p].append(D["diag"]["rho_k"])
    worst = 0.0
    detail = []
    for f in km:
        vals = list(km[f].values())
        rel = (max(vals) - min(vals)) / max(abs(sum(vals) / len(vals)), 1e-12)
        worst = max(worst, rel)
        detail.append(f"{f}:{rel:.4f}")
    chk(v["checks"], "A26O-3", worst <= MOM_REL,
        f"边缘不变（{len(pilots)} 种子 × 4 档）：键矩相对极差最大 {worst:.4f} ≤ {MOM_REL}（{'; '.join(detail)}）")
    ru_m = {p: float(np.mean(ru[p])) for p in PHIS}
    rk_m = {p: float(np.mean(rk[p])) for p in PHIS}
    mono = all(ru_m[PHIS[i + 1]] > ru_m[PHIS[i]] for i in range(len(PHIS) - 1))
    span = ru_m[0.9] - ru_m[0.0]
    chk(v["checks"], "A26O-4", bool(mono and span >= RHO_U_SPAN),
        f"抖动相关 ρ_u（{'→'.join('%.3f' % ru_m[p] for p in PHIS)}）严格递增={mono}、跨度 {span:.3f} ≥ {RHO_U_SPAN}"
        f"；键相关 ρ_k（{'→'.join('%.3f' % rk_m[p] for p in PHIS)}）只作报告")
    v["rho_u"] = ru_m
    v["rho_k"] = rk_m

    # ── A26O-5：工作例 W1（AR(1) 过滤器的边缘方差不变）
    g2 = {}
    for p in PHIS:
        g2[p] = float(np.mean([build_stream_ar1(40, s, D_MODEL, N_KV, R_PRES, theta_of(0.9), p)["diag"]["g2"]
                               for s in pilots[:3]]))
    rel_g = abs(g2[0.9] - g2[0.0]) / max(abs(g2[0.0]), 1e-12)
    chk(v["checks"], "A26O-5", rel_g <= MOM_REL,
        f"工作例 W1：E‖g‖²（φ=0 {g2[0.0]:.6f} / φ=0.9 {g2[0.9]:.6f}）相对差 {rel_g:.4f} ≤ {MOM_REL}")

    # ── A26O-6：工作例 W2（配对：proto/va/vb 与 U 逐位相同；只有 K 不同）
    D0 = build_stream_ar1(40, s0, D_MODEL, N_KV, R_PRES, theta_of(0.9), 0.0)
    D9 = build_stream_ar1(40, s0, D_MODEL, N_KV, R_PRES, theta_of(0.9), 0.9)
    same_meta = (torch.equal(D0["proto"], D9["proto"]) and torch.equal(D0["va"], D9["va"])
                 and torch.equal(D0["vb"], D9["vb"]))
    diff_k = not torch.equal(D0["K"], D9["K"])
    chk(v["checks"], "A26O-6", bool(same_meta and diff_k),
        f"工作例 W2：proto/va/vb 逐位相同={same_meta}；K 不同={diff_k}")

    # ── A26O-7：工作例 W3（位级锚点）
    arch = {(int(r["n_kv"]), int(r["seed"])): r for r in json.load(open(D26N))["rows"] if r["b"] == 40}
    ncmp, bads = 0, []
    ANCH_SEEDS = SEEDS[:5]          # ★ Stage A 只跑 5 个种子（成本）；**全 30 种子的锚点复核在正式轮 `crit`**（卡片 A26O-7 的比较数在那里兑现）
    for s in ANCH_SEEDS:
        a = arch.get((N_KV, s))
        if not a:
            bads.append(f"{s}: DIR-26-N 缺种子")
            continue
        u = run_unit(s, 0.0, tau)
        for f in ("issued", "ret1", "mu", "clean", "wsig"):
            ncmp += 1
            if u["arms"]["E0"][f] != a["arms"]["E0"][f]:
                bads.append(f"{s}/E0.{f}: {u['arms']['E0'][f]} != {a['arms']['E0'][f]}")
        for f in ("issued", "ret1", "wsig"):
            ncmp += 1
            if u["arms"]["tau"][f] != a["arms"]["tau"][f]:
                bads.append(f"{s}/tau.{f}: {u['arms']['tau'][f]} != {a['arms']['tau'][f]}")
    v["anchor"] = {"n_cmp": ncmp, "n_bad": len(bads), "bad": bads[:8]}
    chk(v["checks"], "A26O-7", not bads,
        f"工作例 W3（位级锚点，Stage A 的 {len(ANCH_SEEDS)} 种子预演；全 30 种子复核在正式轮 `crit`）："
        f"φ=0 的 E0/tau 臂与 DIR-26-N 正式轮 b=40 比较 {ncmp} 项，坏 {len(bads)} 项 {bads[:4]}")

    # ── A26O-8：确定性（同 (seed, φ) 跑两次逐位相同）
    u1 = run_unit(s0, 0.6, tau)
    u2 = run_unit(s0, 0.6, tau)
    same = all(u1["arms"][a][f] == u2["arms"][a][f] for a in ARMS for f in ("issued", "ret1", "mu", "wsig"))
    chk(v["checks"], "A26O-8", same, "同 (seed, φ) 跑两次：两臂的 issued/ret1/mu/wsig 逐位相同")

    # ── A26O-9：口径 O 护栏
    ok9 = all(u1["arms"][a]["mu_ret1_consistent"] for a in ARMS)
    chk(v["checks"], "A26O-9", ok9, "每单元 μ/精度口径一致（mu_ret1_consistent）")

    # ── A26O-10：量程（φ=0 的两臂 ret1 均值）
    e0m = float(np.mean([arch[(N_KV, s)]["arms"]["E0"]["ret1"] for s in SEEDS if (N_KV, s) in arch]))
    tam = float(np.mean([arch[(N_KV, s)]["arms"]["tau"]["ret1"] for s in SEEDS if (N_KV, s) in arch]))
    chk(v["checks"], "A26O-10", bool(RET1_E0_BAND[0] <= e0m <= RET1_E0_BAND[1]
                                      and RET1_TAU_BAND[0] <= tam <= RET1_TAU_BAND[1]),
        f"量程：φ=0 的 ret1(E0)={e0m:.4f}∈{RET1_E0_BAND}、ret1(τ)={tam:.4f}∈{RET1_TAU_BAND}")

    v["passed"] = bool(v["checks"] and all(c["ok"] for c in v["checks"]))
    v["n_checks"] = len(v["checks"])
    return v


# ══════════════════════════════════════════════ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("stageA", "smoke", "formal"), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--phis", type=float, nargs="+", default=None)
    ap.add_argument("--bs", type=int, nargs="+", default=[N_KV])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if args.stage == "stageA":
        v = stageA()
        json.dump(v, open(f"{OUT}/DIR26Or11_stageA.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        for c in v["checks"]:
            print(("  ✓ " if c["ok"] else "  ✗ ") + c["id"] + " " + c["msg"])
        print(f"=== Stage A {'PASS' if v['passed'] else 'FAIL'}（{sum(c['ok'] for c in v['checks'])}/{len(v['checks'])}）"
              f" {time.time() - t0:.1f}s ===")
        sys.exit(0 if v["passed"] else 1)

    seeds = args.seeds or (list(SEEDS[:2]) if args.stage == "smoke" else list(SEEDS))
    phis = tuple(args.phis) if args.phis else PHIS
    bs = args.bs
    tau = cal_at(N_KV)["c_tau"]
    res = {"checks": [], "rows": [], "calib": {"c_tau": tau, "n_kv": N_KV},
           "stage": args.stage, "meta": {"PHIS": list(phis), "SEEDS": list(seeds), "BS": list(bs),
                                         "T": T_UNIT, "card": "DIRECTION_rev26o_v11.md",
                                         "thresholds": {"TOL_RET1": TOL_RET1, "TOL_MU": TOL_MU,
                                                        "SIGN_MIN": SIGN_MIN, "MOM_REL": MOM_REL,
                                                        "RHO_U_SPAN": RHO_U_SPAN}}}
    print(f"=== {args.stage}（φ∈{phis} × b∈{bs} × {len(seeds)} 种子 × 2 臂，tau={tau:.4f}）===")
    for b in bs:
        for s in seeds:
            for p in phis:
                tt = time.time()
                u = run_unit(s, p, tau, b=b)
                res["rows"].append(u)
                ar = u["arms"]
                print(f"  b={b:2d} seed={s:3d} φ={p:.1f} | E0 {ar['E0']['ret1']:.4f}/mu {ar['E0']['mu']:+.4f} | "
                      f"τ {ar['tau']['ret1']:.4f}/mu {ar['tau']['mu']:+.4f}/gate {ar['tau']['gate_rate']:.3f} | "
                      f"ρ_u {u['diag']['rho_u']:.3f} ρ_k {u['diag']['rho_k']:.3f} | {time.time() - tt:.0f}s", flush=True)
    if args.stage == "formal":
        res["verdicts"] = crit(res["rows"])
        res["passed"] = bool(res["verdicts"]["verdict_ok"])
        vv = res["verdicts"]
        print("\n===== 判决 =====")
        for key, d in vv["deltas"].items():
            if key.endswith("|0.9"):
                print(f"  {key:22s} mean {d['mean']:+.5f} (3se {d['k']:.5f}) 正 {d['pos']} 负 {d['neg']}")
        print(f"  P1 输入过程 ok={vv['P1']['ok']}（ρ_u {vv['P1']['rho_u']}、ρ_k {vv['P1']['rho_k']}、跨度 {vv['P1']['span']:.3f}）")
        print(f"  P2 命中：{[(h['arm'], h['metric']) for h in vv['P2_hits']]}")
        print(f"  P3：{vv['P3']}")
        print(f"  锚点：{vv['anchor']['n_cmp']} 项、坏 {vv['anchor']['n_bad']}；正对照 {vv['control']}")
        print(f"  ★ 通道 = {vv['channel']}")
    else:
        res["verdicts"] = {"note": "smoke/Stage A 不出判决（纪律 §5）"}
        res["passed"] = None
    p = f"{OUT}/DIR26Or11_{args.stage}{args.tag}.json"
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  → {p}  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
