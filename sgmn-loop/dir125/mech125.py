#!/usr/bin/env python3
"""`DIR-125` 测量层 —— 剂量扫描：超加性是**学习的产物**还是**先验**？

★ 本模块是 `DIR-125` 的**唯一**数学实现：`precheck125` / `test_mech125` /
  `R1_task125` 都 import 它，不得各自复制一份（`DIR-122` 的教训：两份实现会分叉）。

★ 与 `DIR-124` 的差异（**只有两处，其余逐字派生**）：

  ① **求解器定义域放宽**：`u ∈ (0, ∞)`，上限 `n₁+n₂`（旧版是 `u ≥ 1`、上限 `(2n₁+n₂)/3`）。
     理由见卡片 §2-4；`DEF-124-01` 已实测旧夹紧使检验 size 达 `0.9405`。
     ★ 本模块的 `solve_u` 返回 `(u, clamped)`，**夹紧必须可观测**（不得静默）。
  ② **cell 来源多一个目录**：`dir125/out/cells`（新剂量）与 `dir119/out/cells`（`dose ∈ {0,600}` 冻结只读）。

  其余（`N1/N2` 结构、`θ` 的定义、精确卷积零分布、双侧 `p`、功效/水平的算法）
  **逐字沿用** `DIR-124`。

★ `P1(u) = 2/(u+2)`、`P2(u) = 1/(2u+1)`、`W1(u) ≡ W2(u) ≡ u`（代数恒等）
  ⇒ **零假设就是 `w₂/w₁ = 1`**，观测 `ratio > 1` 才是超加性。
"""
import os
import sys
import json
import glob
import math

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT, os.path.join(ROOT, "dir124"), os.path.join(ROOT, "dir123"),
           os.path.join(ROOT, "dir122"), os.path.join(ROOT, "dir121"),
           os.path.join(ROOT, "dir120")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mech122 as M                                              # noqa: E402
import mech123 as M3                                             # noqa: E402

# ---------------------------------------------------------------- 冻结常量（卡片 §2）
ALL_RNGS = (M.MAIN_RNG,) + tuple(M.C2_RNGS)     # (555, 556, 557, 558)
N_QS = len(ALL_RNGS)

# 剂量网格（卡片 §2-2）
DOSE_ZERO = 0
DOSE_PILOT = 25                                  # report-only 过渡探针
DOSE_NEW = (150, 300, 450)                       # ★ J125-1/J125-2 的判据剂量
DOSE_HIGH = 600
DOSE_ALL = (0, 25, 150, 300, 450, 600)

SEEDS = (20260501, 20260502, 20260503, 20260504, 20260505,
         20260506, 20271126, 20271203, 20271210)

ALPHA = 0.05                    # `J125-1`（单检、不拆）
ALPHA_PLACEBO = 0.05            # `J125-3`（`dose=0` 重读；否证支 (b) 用）
SIDE = "two"                    # ★ 开卡前冻结：**双侧**
MC_REPS = 40000
MC_SEED = 20261250              # ★ 与 `DIR-124` 的 `20261240` 不同，避免复用同一批 MC 数
POWER_MIN = 0.80                # ★ `J125-1` 的功效门槛（开卡前）

# ★ 结构常量（卡片 §2-3；只由 `n_seen` 直方图决定，与训练无关）
STRUCT_N1_PER_SEED = 68
STRUCT_N2_PER_SEED = 99

# 归档目录（★ `dir119` 冻结只读；`dir125` 是本卡的新格）
DIR119_CELLS = os.path.join(ROOT, "dir119", "out", "cells")
DIR125_CELLS = os.path.join(HERE, "out", "cells")

__all__ = ["ALL_RNGS", "N_QS", "DOSE_ZERO", "DOSE_PILOT", "DOSE_NEW", "DOSE_HIGH",
           "DOSE_ALL", "SEEDS", "ALPHA", "ALPHA_PLACEBO", "SIDE", "MC_REPS", "MC_SEED",
           "POWER_MIN", "STRUCT_N1_PER_SEED", "STRUCT_N2_PER_SEED",
           "DIR119_CELLS", "DIR125_CELLS",
           "P1", "P2", "W1", "W2", "solve_u", "theta_of", "binom_pmf",
           "convolve_null", "tails", "strat_test", "w_hat", "ratio_ci", "build_ctx",
           "load_cells", "strata_of", "stratum_obs", "power_for_strata", "level_for_strata"]


# ================================================================ 模型（逐字沿用）
def P1(u):
    """`k=1`：2 个未见过干扰项（权重 1）+ 1 个见过（权重 `u`）⇒ 逃逸概率。"""
    return 2.0 / (u + 2.0)


def P2(u):
    """`k=2`：1 个未见过（权重 1）+ 2 个见过（权重 `u`）。"""
    return 1.0 / (2.0 * u + 1.0)


def W1(u):
    """`w_1 = (1-P1)·(3-1)/(P1·1)` —— 代数上 ≡ `u`。"""
    p = P1(u)
    return (1.0 - p) * 2.0 / p


def W2(u):
    """`w_2 = (1-P2)·(3-2)/(P2·2)` —— 代数上 ≡ `u`。"""
    p = P2(u)
    return (1.0 - p) * 1.0 / (2.0 * p)


# ================================================================ ★ 放宽域求解器
def solve_u(m, n1, n2, return_clamped=False):
    """解 `n1·P1(u) + n2·P2(u) = m`，**定义域 `u ∈ (0, ∞)`**。

    ★ 与 `mech123.solve_u` 的**唯一差别**：上限从 `(2n₁+n₂)/3`（`u = 1` 处）放宽到
      `n₁+n₂`（`u → 0⁺` 处）。理由见卡片 §2-4。

    左侧对 `u` 严格递减：`f(0⁺) = n1 + n2`，`f(∞) = 0`。
      - `m ≥ n1 + n2` ⇒ 无解（观测数超上限）⇒ 返回 `(0.0, True)` 并**标记夹紧**；
      - `m ≤ 0`      ⇒ `u → ∞` ⇒ 返回 `None`（判据不可执行）；
      - 否则在 `(0, ∞)` 上做对数二分。

    ★ `return_clamped=True` 时返回 `(u, clamped)`；**夹紧必须可观测**（`DEF-124-01`）。
    """
    n1, n2 = float(n1), float(n2)
    m = float(m)
    out = (None, True) if not return_clamped else None
    if m <= 0.0:
        return (None, False) if return_clamped else None
    hi_val = n1 + n2                      # ★ u → 0⁺ 的上限（旧版是 (2n1+n2)/3）
    if m >= hi_val:
        return (0.0, True) if return_clamped else 0.0
    lo, hi = 1e-12, 1e12
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        if n1 * P1(mid) + n2 * P2(mid) - m > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    u = math.sqrt(lo * hi)
    return (u, False) if return_clamped else u


def theta_of(m, n1, n2, return_clamped=False):
    """`θ(m) = n1·P1(û)/m`；`m <= 0` ⇒ `None`（判据不可执行）。"""
    if m <= 0:
        return (None, False) if return_clamped else None
    if return_clamped:
        u, cl = solve_u(m, n1, n2, return_clamped=True)
    else:
        u = solve_u(m, n1, n2)
    if u is None or not (u == u) or u == float("inf"):
        return (None, cl) if return_clamped else None
    th = n1 * float(P1(u)) / float(m)
    if not (0.0 < th < 1.0):
        return (None, cl) if return_clamped else None
    return (th, cl) if return_clamped else th


# ================================================================ 精确零分布（逐字沿用）
def binom_pmf(m, k, theta):
    if k < 0 or k > m:
        return 0.0
    return math.comb(m, k) * (theta ** k) * ((1.0 - theta) ** (m - k))


def convolve_null(ms, thetas):
    """`S = Σ_s e₁ₛ` 在 `e₁ₛ | mₛ ~ Binomial(mₛ, θₛ)` 独立下的**精确** pmf。"""
    pmf = np.array([1.0])
    for m, th in zip(ms, thetas):
        m = int(m)
        if m <= 0:
            continue
        ks = np.arange(m + 1)
        b = np.array([binom_pmf(m, int(k), th) for k in ks])
        pmf = np.convolve(pmf, b)
    return pmf


def tails(pmf, obs):
    """返回 `(上尾, 下尾, 双侧精确)`；双侧 = `Σ_{pmf(k) ≤ pmf(obs)(1+tol)}`。"""
    n = len(pmf)
    obs = int(obs)
    if n == 0 or obs < 0 or obs >= n:
        return None, None, None
    upper = float(pmf[obs:].sum())
    lower = float(pmf[:obs + 1].sum())
    cut = pmf[obs] * (1.0 + 1e-9) + 1e-300
    two = float(pmf[pmf <= cut].sum())
    # ★ 浮点归一（`DEF-124-02`）：三者数学上都不超过 1，但「全部纳入」时可得 1+1e-16。
    return min(1.0, upper), min(1.0, lower), min(1.0, two)


# ================================================================ 读档
def _load_dir(path, dose, tag):
    pat = os.path.join(path, "COLOCATED_%d_*.json" % dose)
    files = sorted(glob.glob(pat))
    out = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        picks = {M.MAIN_RNG: d["mech_by_mult"]["10000"]["pick_argmin"]}
        c2 = d.get("c2") or {}
        if "10000" in c2:
            for r in M.C2_RNGS:
                if str(r) in c2["10000"]:
                    picks[r] = c2["10000"][str(r)]["pick_argmin"]
        out.append({"seed": int(d["seed"]), "picks": picks, "dose": int(d["dose"]),
                    "raw": d, "src": tag})
    return out


def load_cells(dose):
    """读 `dose` 的 9 个 cell。

    ★ `dose ∈ {0, 600}` 从 `dir119/out/cells`（**冻结只读**）读；
      `dose ∈ {25,150,300,450}` 从 `dir125/out/cells`（本卡新格）读。
    ★ 两处都找不到 ⇒ 抛错（不得静默返回空）。
    """
    if dose in (DOSE_ZERO, DOSE_HIGH):
        cells = _load_dir(DIR119_CELLS, dose, "dir119")
        if not cells:
            raise RuntimeError("dir119 归档缺 dose=%d 的 cell" % dose)
        return cells
    cells = _load_dir(DIR125_CELLS, dose, "dir125")
    if not cells:
        raise RuntimeError("dir125 尚无 dose=%d 的 cell（先跑 --stage cell）" % dose)
    return cells


def build_ctx(dose, rngs=None):
    """建 `facts/items/ns/tv/cells`（与 `DIR-121/122/123/124` 逐字一致）。

    `rngs` 记录**实际可用**的题集；`strata_of` 只准用它们。
    ★ 若 `rngs` 给定，则强制截取（用于 `555`-only 的对照）。
    """
    facts = M.build_facts()
    tr, ev = M.partitions_colocated(facts)
    tv = {facts[i]["val"] for i in tr}
    items = {r: M.make_items(facts, ev, r) for r in ALL_RNGS}
    ns = {r: M.n_seen_vec(items[r], tv) for r in ALL_RNGS}
    cells = load_cells(dose)
    avail = tuple(r for r in ALL_RNGS if all(r in c["picks"] for c in cells))
    if rngs is not None:
        avail = tuple(r for r in rngs if r in avail)
    return {"facts": facts, "tr": tr, "ev": ev, "tv": tv,
            "items": items, "ns": ns, "cells": cells, "rngs": avail, "dose": dose}


# ================================================================ 分层
def strata_of(ctx, dose_label=None):
    """按 `seed` 切层（每层覆盖 `ctx["rngs"]` 里所有可用题集）。

    `N1`/`N2` **只由结构决定**（`n_seen` 直方图 × 层内题集数），与观测无关。
    `dose_label` 用于 `J125-1` 池化 27 层时的 key 前缀。
    """
    cells, ns, rngs = ctx["cells"], ctx["ns"], ctx["rngs"]

    def n_of(rng, k):
        return sum(1 for v in ns[rng] if v == k)

    out = []
    for si, c in enumerate(cells):
        pre = ("dose=%d," % (dose_label if dose_label is not None else ctx["dose"]))
        out.append({"key": pre + ("seed=%d" % c["seed"]),
                    "N1": sum(n_of(r, 1) for r in rngs),
                    "N2": sum(n_of(r, 2) for r in rngs),
                    "idx": [(si, r) for r in rngs]})
    return out


def stratum_obs(ctx, strata):
    """逐层观测 `e1 / e2`（用 `mech123.counts_by_stratum`，与 `D-154` 逐字同源）。"""
    cells, items, ns, tv = ctx["cells"], ctx["items"], ctx["ns"], ctx["tv"]
    out = []
    for s in strata:
        e1 = e2 = 0
        for si, r in s["idx"]:
            d = M3.counts_by_stratum(items[r], ns[r], cells[si]["picks"][r], tv)
            e1 += d[1]
            e2 += d[2]
        out.append({"key": s["key"], "N1": s["N1"], "N2": s["N2"],
                    "e1": e1, "e2": e2, "m": e1 + e2})
    return out


# ================================================================ ★ 核心：分层精确检验
def strat_test(rows, alpha=ALPHA, side=SIDE):
    """★ 分层精确条件检验（**放宽域**版）。

    逐层用**该层自己的** `m_s` 解 `û_s`（层内条件化，异质性不进零分布），
    再对 `S = Σ e₁ₛ` 做精确分布检验。

    返回 `{"executable","S","m","strata_used","clamped_layers","p_upper","p_lower",
           "p_two","p_main","side","reject","direction","expect","alpha"}`。
    ★ `clamped_layers` 是本卡的**可观测夹紧**：落在有效层里的夹紧层数。
    """
    ms, thetas, e1s = [], [], []
    clamped = 0
    for r in rows:
        m = int(r["e1"]) + int(r["e2"])
        if "m" in r and int(r["m"]) != m:
            raise ValueError("stratum %r: m=%r 与 e1+e2=%d 不一致"
                             % (r.get("key"), r["m"], m))
        if m <= 0:
            continue
        th, cl = theta_of(m, r["N1"], r["N2"], return_clamped=True)
        if cl:
            clamped += 1
        if th is None:
            return {"executable": False, "reason": "theta_none", "key": r.get("key"),
                    "S": None, "m": None, "strata_used": None, "clamped_layers": clamped,
                    "p_upper": None, "p_lower": None, "p_two": None, "p_main": None,
                    "side": side, "reject": None, "direction": None, "expect": None,
                    "alpha": alpha}
        ms.append(m)
        thetas.append(th)
        e1s.append(int(r["e1"]))
    if not ms:
        return {"executable": False, "reason": "no_stratum_with_m_gt_0",
                "S": 0, "m": 0, "strata_used": 0, "clamped_layers": clamped,
                "p_upper": None, "p_lower": None, "p_two": None, "p_main": None,
                "side": side, "reject": None, "direction": None, "expect": None,
                "alpha": alpha}
    pmf = convolve_null(ms, thetas)
    S = int(sum(e1s))
    m_tot = int(sum(ms))
    exp = sum(m * t for m, t in zip(ms, thetas))
    up, lo, two = tails(pmf, S)
    if up is None:
        return {"executable": False, "reason": "S_out_of_support", "S": S, "m": m_tot,
                "strata_used": len(ms), "clamped_layers": clamped,
                "p_upper": None, "p_lower": None, "p_two": None, "p_main": None,
                "side": side, "reject": None, "direction": None, "expect": exp,
                "alpha": alpha}
    p_main = two if side == "two" else up
    return {"executable": True, "S": S, "m": m_tot, "strata_used": len(ms),
            "clamped_layers": clamped, "p_upper": up, "p_lower": lo, "p_two": two,
            "p_main": p_main, "side": side, "reject": bool(p_main <= alpha),
            "direction": ("high" if S > exp else ("low" if S < exp else "eq")),
            "expect": exp, "alpha": alpha}


# ================================================================ 模型无关估计量
def w_hat(rows):
    """`p̂_k = e_k/N_k`、`ŵ_k = (1-p̂_k)(3-k)/(p̂_k·k)`、`ratio = ŵ₂/ŵ₁`。

    ★ **不用 `solve_u`** ⇒ 无定义域风险（`D-155` §5；卡片 §2-1 `J125-2`）。
    """
    N1 = sum(r["N1"] for r in rows)
    N2 = sum(r["N2"] for r in rows)
    e1 = sum(r["e1"] for r in rows)
    e2 = sum(r["e2"] for r in rows)
    if N1 == 0 or N2 == 0 or e1 == 0 or e2 == 0:
        return {"N1": N1, "N2": N2, "e1": e1, "e2": e2,
                "p1": (e1 / N1 if N1 else None), "p2": (e2 / N2 if N2 else None),
                "w1": None, "w2": None, "ratio": None}
    p1, p2 = e1 / float(N1), e2 / float(N2)
    w1 = (1.0 - p1) * 2.0 / p1
    w2 = (1.0 - p2) * 1.0 / (2.0 * p2)
    return {"N1": N1, "N2": N2, "e1": e1, "e2": e2, "p1": p1, "p2": p2,
            "w1": w1, "w2": w2, "ratio": w2 / w1}


# ================================================================ 功效 / 水平（结构驱动）
_theta_cache = {}


def _theta_cached(m, n1, n2):
    k = (int(m), int(n1), int(n2))
    if k not in _theta_cache:
        _theta_cache[k] = theta_of(*k)
    return _theta_cache[k]


def _draw_all(strata, u, ratio, reps, seed):
    """向量化按备择画 `(e1, e2)`：`k=1` 用 `P1(u)`、`k=2` 用 `P2(r·u)`。"""
    p1 = P1(u)
    p2 = P2(ratio * u)
    rng = np.random.default_rng(seed)
    E1 = np.empty((reps, len(strata)), dtype=np.int64)
    E2 = np.empty((reps, len(strata)), dtype=np.int64)
    for j, s in enumerate(strata):
        E1[:, j] = rng.binomial(int(s["N1"]), p1, size=reps)
        E2[:, j] = rng.binomial(int(s["N2"]), p2, size=reps)
    return E1, E2


def power_for_strata(strata, u=9.2, ratio=2.0, reps=MC_REPS, seed=MC_SEED,
                     alpha=ALPHA, side=SIDE):
    """★ 备择 `w₂/w₁ = ratio` 下的**拒绝率**（`p_main <= alpha`）。机器可验证。"""
    E1, E2 = _draw_all(strata, u, ratio, reps, seed)
    N1 = np.array([s["N1"] for s in strata], dtype=np.int64)
    N2 = np.array([s["N2"] for s in strata], dtype=np.int64)
    hit = 0
    for i in range(reps):
        e1 = E1[i]
        ms = e1 + E2[i]
        keep = ms > 0
        if not keep.any():
            continue
        mm = ms[keep]
        ths = [_theta_cached(mm[j], N1[keep][j], N2[keep][j]) for j in range(len(mm))]
        if any(t is None for t in ths):
            continue
        pmf = convolve_null(mm, ths)
        S = int(e1[keep].sum())
        up, lo, two = tails(pmf, S)
        p_main = two if side == "two" else up
        if p_main is not None and p_main <= alpha:
            hit += 1
    return hit / float(reps)


def level_for_strata(strata, u=9.2, reps=MC_REPS, seed=MC_SEED, alpha=ALPHA, side=SIDE):
    """`H125` 为真（`ratio = 1`）时的拒绝率 ⇒ 须 ≤ `alpha`（校准检验）。"""
    return power_for_strata(strata, u=u, ratio=1.0, reps=reps, seed=seed,
                            alpha=alpha, side=side)

# ================================================================ report-only：ratio 的抽样区间
def ratio_ci(rows, reps=20000, seed=MC_SEED + 7, alpha=0.05):
    """`ratio = ŵ₂/ŵ₁` 的**参数 bootstrap** 区间（report-only，**不改任何判据**）。

    ★ 为什么需要它：卡片 §2-1 的 `J125-2` 只写了「`ratio > 1`」，这是**点估计比较**、
      没有不确定性。本函数把区间报出来，供 `VERDICT` 判断该阈值是否够用 ——
      ★ **它不是判据**，事后也不得据此改 `J125-2` 的结论（那会是事后动判据）。
    """
    o = w_hat(rows)
    if o["ratio"] is None:
        return {"ratio": None, "lo": None, "hi": None, "reps": reps}
    rng = np.random.default_rng(seed)
    p1, p2 = o["p1"], o["p2"]
    N1, N2 = o["N1"], o["N2"]
    e1 = rng.binomial(N1, p1, size=reps).astype(np.float64)
    e2 = rng.binomial(N2, p2, size=reps).astype(np.float64)
    q1 = e1 / N1
    q2 = e2 / N2
    ok = (q1 > 0) & (q2 > 0) & (q1 < 1) & (q2 < 1)
    w1 = (1.0 - q1[ok]) * 2.0 / q1[ok]
    w2 = (1.0 - q2[ok]) * 1.0 / (2.0 * q2[ok])
    r = w2 / w1
    return {"ratio": o["ratio"],
            "lo": float(np.quantile(r, alpha / 2.0)),
            "hi": float(np.quantile(r, 1.0 - alpha / 2.0)),
            "reps": int(ok.sum())}
