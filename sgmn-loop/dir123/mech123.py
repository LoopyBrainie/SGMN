#!/usr/bin/env python3
"""`DIR-123` 测量层 —— 「每候选加成 `w_k` 与「见到几个」无关吗？」

★ 本模块是 `DIR-123` 的**唯一**数学实现：`precheck123` / `test_mech123` / `R1_task123`
  都 import 它，不得各自复制一份（`DIR-122` 的教训：两份实现会分叉）。

★ 三个核心量（全部由 `DIRECTION_rev123_v10.md` §0 冻结）：

  `p_k(u)`   —— 在 `k` 个见过干扰项、`3-k` 个未见过干扰项、每候选权重 `u`/`1` 时，
                逃逸（= 选中未见过）的概率。
  `w_k(u)`   —— 「每候选加成」：`w_k = S_k(3-k) / (U_k·k)`，代数上 **≡ u**（与 k 无关）。
  `H123`     —— `w_1 = w_2`；等价于 `OR = 4`（参数无关的精确预测）。

★ 判据 `J123-1` 用**精确条件检验**：给定实测 `m = e_1 + e_2`，`û` 被
  `N_1 p_1(û) + N_2 p_2(û) = m` 唯一确定 ⇒ `e_1 | m ~ Binomial(m, θ(m))`，
  没有剩余多余参数（不是 plug-in 近似）。
"""
import os
import sys
import json
import math
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT, os.path.join(ROOT, "dir122"), os.path.join(ROOT, "dir121"),
           os.path.join(ROOT, "dir120")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mech122 as M                                              # noqa: E402

# ---------------------------------------------------------------- 冻结常量
ALL_RNGS = (M.MAIN_RNG,) + tuple(M.C2_RNGS)
ALPHA_J1 = 0.025          # `J123-1`
ALPHA_J2 = 0.025          # `J123-2`（合取，不膨胀多重性）
B_BOOT = 20000            # report-only 用（本卡不用于判决）
BOOT_SEED = 20261125

__all__ = ["ALL_RNGS", "ALPHA_J1", "ALPHA_J2", "B_BOOT", "BOOT_SEED",
           "P1", "P2", "W1", "W2", "odds_ratio", "solve_u", "theta_of",
           "binom_pmf", "binom_two_sided_p", "accept_set",
           "load_cells", "counts_by_stratum", "correct_at_k", "pooled",
           "leave_one_out", "per_seed_table", "per_qset_table"]


# ================================================================ 机制层
def P1(u):
    """`k=1`：2 个未见过干扰项（权重 1）+ 1 个见过（权重 `u`）⇒ 逃逸概率。"""
    return 2.0 / (u + 2.0)


def P2(u):
    """`k=2`：1 个未见过（权重 1）+ 2 个见过（权重 `u`）。"""
    return 1.0 / (2.0 * u + 1.0)


def W1(u):
    """`w_1 = (1-P1)·(3-1) / (P1·1)` —— 代数上 ≡ `u`。"""
    p = P1(u)
    return (1.0 - p) * 2.0 / (p * 1.0)


def W2(u):
    """`w_2 = (1-P2)·(3-2) / (P2·2)` —— 代数上 ≡ `u`。"""
    p = P2(u)
    return (1.0 - p) * 1.0 / (p * 2.0)


def odds_ratio(u):
    """`OR = [p₁/(1-p₁)] / [p₂/(1-p₂)]`；恒等于 `4·w_2/w_1`。"""
    p1, p2 = P1(u), P2(u)
    return (p1 / (1.0 - p1)) / (p2 / (1.0 - p2))


# ================================================================ 条件检验
def solve_u(m, n1, n2):
    """解 `n1·P1(u) + n2·P2(u) = m`。

    左侧对 `u` **严格递减**（`dP1/du < 0`、`dP2/du < 0`）⇒
      `m ≥ n1·P1(1)+n2·P2(1)` 时把 `u` 夹到 `1.0`（`u<1` 无物理意义：加成不可能为负）；
      `m` 极小（→0）时 `u → ∞`，用对数二分避免溢出。
    """
    n1, n2 = float(n1), float(n2)
    m = float(m)
    hi_val = n1 * P1(1.0) + n2 * P2(1.0)
    if m >= hi_val:                      # 需要 u <= 1 ⇒ 取边界 1.0
        return 1.0
    if m <= 0.0:
        return float("inf")

    def f(u):
        return n1 * P1(u) + n2 * P2(u) - m

    lo, hi = 1.0, 2.0
    while f(hi) > 0.0:                   # 扩大上界直到 f(hi) <= 0
        hi *= 2.0
        if hi > 1e300:
            return float("inf")
    for _ in range(400):                 # 几何二分（跨越数量级）
        mid = math.sqrt(lo * hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def theta_of(m, n1, n2):
    """`θ(m) = n1·P1(û) / m`；`m <= 0` ⇒ `None`（判据不可执行）。"""
    if m <= 0:
        return None
    u = solve_u(m, n1, n2)
    if not (u == u) or u == float("inf"):
        return None
    th = n1 * P1(u) / float(m)
    return th if 0.0 < th < 1.0 else None


def binom_pmf(m, k, theta):
    if k < 0 or k > m:
        return 0.0
    return math.comb(m, k) * (theta ** k) * ((1.0 - theta) ** (m - k))


def binom_two_sided_p(m, e1, theta, rel_tol=1e-9):
    """精确双侧 p 值 = `Σ_{k: P(k) ≤ P(e1)·(1+rel_tol)} P(k)`（卡片 §2-1 第 3 步）。"""
    if m <= 0 or theta is None:
        return None
    obs = binom_pmf(m, e1, theta)
    cut = obs * (1.0 + rel_tol) + 1e-300
    return sum(binom_pmf(m, k, theta) for k in range(m + 1)
               if binom_pmf(m, k, theta) <= cut)


def accept_set(m, n1, n2, alpha=ALPHA_J1):
    """返回 `(接受域排序列表, 接受域概率总质量, θ)`；接受域 = `{e1 : p(e1) > alpha}`。"""
    th = theta_of(m, n1, n2)
    if th is None:
        return None, None, None
    acc = [e for e in range(m + 1)
           if binom_two_sided_p(m, e, th) > alpha + 1e-15]
    mass = sum(binom_pmf(m, e, th) for e in acc)
    return acc, mass, th


# ================================================================ 读档
def load_cells():
    """读 9 个 cell，抽出四条读数线的 `pick_argmin`（与 `DIR-122` 逐字一致）。"""
    files = sorted(glob.glob(os.path.join(ROOT, "dir119", "out", "cells",
                                         "COLOCATED_600_*.json")))
    out = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        picks = {M.MAIN_RNG: d["mech_by_mult"]["10000"]["pick_argmin"]}
        acc = {M.MAIN_RNG: d["mech_by_mult"]["10000"]["acc_argmin"]}
        for r in M.C2_RNGS:
            picks[r] = d["c2"]["10000"][str(r)]["pick_argmin"]
            acc[r] = d["c2"]["10000"][str(r)]["acc_argmin"]
        out.append({"seed": int(d["seed"]), "picks": picks, "acc": acc, "raw": d})
    return out


# ================================================================ 分层计数
def counts_by_stratum(items, ns, picks, tv):
    """返回 `{k: 逃逸数}`（`k = 0..3`）。`k=0` 恒 `0`（`n_seen>=1` 硬条件）。"""
    esc = M.escape_mask(items, ns, picks, tv)
    return {k: sum(1 for e, n in zip(esc, ns) if n == k and e) for k in range(M.N_CAND)}


def correct_at_k(items, ns, picks, k):
    """`n_seen == k` 的项里选对（`pick == tgt_i`）的计数。"""
    return sum(1 for it, n, p in zip(items, ns, picks)
               if n == k and int(p) == it["tgt_i"])


def pooled(cells, items, ns, tv):
    """汇总四题集 × 9 seed 的 `e_1 / e_2 / e_3` 与 `m`。"""
    e = {1: 0, 2: 0, 3: 0}
    per = {}
    for c in cells:
        for r in ALL_RNGS:
            ck_ = counts_by_stratum(items[r], ns[r], c["picks"][r], tv)
            per[(c["seed"], r)] = ck_
            for k in (1, 2, 3):
                e[k] += ck_[k]
    m = e[1] + e[2]
    return {"e1": e[1], "e2": e[2], "e3": e[3], "m": m,
            "total": e[1] + e[2] + e[3], "per": per}


def per_seed_table(cells, items, ns, tv):
    """逐 seed 的 `e_1 / e_2 / e_3`（report-only）。"""
    rows = []
    for c in cells:
        r1 = r2 = r3 = 0
        for r in ALL_RNGS:
            d = counts_by_stratum(items[r], ns[r], c["picks"][r], tv)
            r1 += d[1]
            r2 += d[2]
            r3 += d[3]
        rows.append({"seed": c["seed"], "e1": r1, "e2": r2, "e3": r3,
                     "m": r1 + r2})
    return rows


def per_qset_table(cells, items, ns, tv):
    """逐题集的 `e_1 / e_2 / e_3` 与机会数 `d_1/d_2/d_3`（report-only + 留一用）。"""
    rows = []
    for r in ALL_RNGS:
        d = [sum(1 for v in ns[r] if v == k) for k in range(M.N_CAND)]
        e1 = e2 = e3 = 0
        for c in cells:
            ck_ = counts_by_stratum(items[r], ns[r], c["picks"][r], tv)
            e1 += ck_[1]
            e2 += ck_[2]
            e3 += ck_[3]
        rows.append({"rng": r, "d1": d[1], "d2": d[2], "d3": d[3],
                     "e1": e1, "e2": e2, "e3": e3, "m": e1 + e2})
    return rows


def leave_one_out(pool, qtab, n_cells, alpha=ALPHA_J1):
    """留一题集：剔除题集 `j` 后重做 `J123-1`。返回逐题集的判决表。

    `N_1^{(-j)} = n_cells·d_1^{(j)}` 由结构决定（直方图逐 seed 相同 ⇒ × `n_cells`）。
    """
    n1_tot = n_cells * sum(q["d1"] for q in qtab)
    n2_tot = n_cells * sum(q["d2"] for q in qtab)
    out = []
    for q in qtab:
        n1 = n1_tot - n_cells * q["d1"]
        n2 = n2_tot - n_cells * q["d2"]
        m = pool["m"] - q["m"]
        e1 = pool["e1"] - q["e1"]
        th = theta_of(m, n1, n2)
        if th is None or not (0 < e1 <= m):
            out.append({"drop": q["rng"], "executable": False,
                        "m": m, "e1": e1, "theta": th, "p": None,
                        "reject": None, "dir": None})
            continue
        p = binom_two_sided_p(m, e1, th)
        exp1 = m * th
        out.append({"drop": q["rng"], "executable": True, "m": m, "e1": e1,
                    "theta": th, "p": p, "reject": bool(p <= alpha),
                    "dir": ("high" if e1 > exp1 else ("low" if e1 < exp1 else "eq"))})
    return out