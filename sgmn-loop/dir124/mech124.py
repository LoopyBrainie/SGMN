#!/usr/bin/env python3
"""`DIR-124` 测量层 —— 「每候选加成随见到个数增长」的**分层**精确检验 + 未训练负对照。

★ 本模块是 `DIR-124` 的**唯一**数学实现：`precheck124` / `test_mech124` / `R1_task124`
  都 import 它，不得各自复制一份（`DIR-122` 的教训：两份实现会分叉）。

★ 与 `DIR-123` 的**方法学差异**（不是阈值差异，`DEF-123-05` 的正面修法）：

  `DIR-123` 的 `J123-1` 把四题集 × 九 seed **全部合并**，只拟**一个** `û`
   ⇒ 隐含「所有层、所有 seed 共享同一个每候选加成」。
  `D-153` §2-4 已证 seed 级率异质（`20260506`：`Q = 11.91`、`p = 0.0026`），
  且三个 seed（`20260501/20260504/20271203`）逐层精确为 `0` ⇒ **合并是被错误指定的**。

  `DIR-124` 的 `J124-1` 改为**按 seed 分层**（`axis = "seed"`，9 层）：
  每层**各自**用该层的 `m_s` 拟定 `û_s` ⇒ 层内条件化，异质性不进零分布。
  层间独立 ⇒ `S = Σ_s e₁ₛ` 的零分布 = 各层二项分布的**精确卷积**（DP / `np.convolve`）。

★ 负对照 `J124-2`：同一统计量用在 **`dose = 0`（未训练 `W = W⁰`）** 的 9 个 cell 上。
  `dose=0` 从未被用于任何 `w₁/w₂` 问题 ⇒ **独立数据**。
  机制预测：无训练 ⇒ 无学到的加成；若仍出现**同向**不对称 ⇒ 归因被证伪。

★ 功效与水平**必须由本模块的 `power()` / `level()` 机器算出**（`precheck124` 断言），
  不得手算 —— `DIR-123` 的功效是手算的，正是 `DEF-123-05` 的根因。

★ 数值实现用 `numpy`（抽样向量化 + `np.convolve`）⇒ `40000` 次 MC 在秒级完成。
"""
import os
import sys
import json
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT, os.path.join(ROOT, "dir123"), os.path.join(ROOT, "dir122"),
           os.path.join(ROOT, "dir121"), os.path.join(ROOT, "dir120")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mech122 as M                                              # noqa: E402
import mech123 as M3                                             # noqa: E402

# ---------------------------------------------------------------- 冻结常量
ALL_RNGS = (M.MAIN_RNG,) + tuple(M.C2_RNGS)
N_QS = len(ALL_RNGS)                    # 4
ALPHA = 0.05                            # `J124-1`（单检、不拆）
ALPHA_PLACEBO = 0.05                    # `J124-2`（独立数据、对照）
AXIS = "seed"                           # ★ 开卡前按功效选定的分层轴
SIDE = "two"                            # ★ 开卡前冻结：**双侧**（消除方向选择偏倚）
AXES_CANDIDATES = ("both", "seed", "qset")
DOSE_MAIN = 600
DOSE_PLACEBO = 0
MC_REPS = 40000                         # 功效/水平 MC 次数（固定 seed ⇒ 可复现）
MC_SEED = 20261240
POWER_MIN = 0.80                        # ★ `J124-1` 的功效门槛（开卡前）

__all__ = ["ALL_RNGS", "N_QS", "ALPHA", "ALPHA_PLACEBO", "AXIS",
           "AXES_CANDIDATES", "SIDE", "DOSE_MAIN", "DOSE_PLACEBO", "MC_REPS", "MC_SEED",
           "POWER_MIN", "load_cells", "build_ctx", "strata_of", "stratum_obs",
           "convolve_null", "strat_test", "power", "level", "ctx_rngs"]


# ================================================================ 读档
def load_cells(dose):
    """读 `dose` 对应的 9 个 cell（`dose=600` 的取值与 `DIR-123` 逐字一致）。"""
    pat = os.path.join(ROOT, "dir119", "out", "cells", "COLOCATED_%d_*.json" % dose)
    files = sorted(glob.glob(pat))
    out = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        picks = {M.MAIN_RNG: d["mech_by_mult"]["10000"]["pick_argmin"]}
        c2 = d.get("c2") or {}
        if "10000" in c2:                       # ★ `dose=0` 归档**没有** `c2` 块
            for r in M.C2_RNGS:
                if str(r) in c2["10000"]:
                    picks[r] = c2["10000"][str(r)]["pick_argmin"]
        out.append({"seed": int(d["seed"]), "picks": picks,
                    "dose": int(d["dose"]), "raw": d})
    return out


def build_ctx(dose):
    """建 `items` / `ns` / `tv` / `cells`（与 `DIR-121/122/123` 逐字一致）。

    ★ 关键结构事实（本卡首次确立）：`dose = 0` 的归档 cell **没有 `c2` 块**
      ⇒ 未训练态**只有主题集 `555`**，而 `dose = 600` 有 `555/556/557/558` 四个。
      ⇒ 负对照的数据量只有主检的 **1/4**（`N1 = 9×23 = 207`、`N2 = 9×20 = 180`）。
      `rngs` 字段记录**实际可用**的题集，`strata_of` 只准用它们（不得假设 4 个）。
    """
    facts = M.build_facts()
    tr, ev = M.partitions_colocated(facts)
    tv = {facts[i]["val"] for i in tr}
    items = {r: M.make_items(facts, ev, r) for r in ALL_RNGS}
    ns = {r: M.n_seen_vec(items[r], tv) for r in ALL_RNGS}
    cells = load_cells(dose)
    rngs = tuple(r for r in ALL_RNGS
                 if all(r in c["picks"] for c in cells))
    return {"facts": facts, "tr": tr, "ev": ev, "tv": tv,
            "items": items, "ns": ns, "cells": cells, "rngs": rngs}


# ================================================================ 分层
def strata_of(ctx, axis):
    """按 `axis` 切层，返回 `[{"key", "N1", "N2", "idx"}]`。

    `N1` / `N2` **只由结构决定**（`n_seen` 直方图 × 层内 cell 数），与观测无关。
    """
    cells = ctx["cells"]
    ns = ctx["ns"]
    rngs = ctx["rngs"]
    n_seeds = len(cells)

    def n_of(rng, k):
        return sum(1 for v in ns[rng] if v == k)

    out = []
    if axis == "seed":
        for si, c in enumerate(cells):
            out.append({"key": "seed=%d" % c["seed"],
                        "N1": sum(n_of(r, 1) for r in rngs),
                        "N2": sum(n_of(r, 2) for r in rngs),
                        "idx": [(si, r) for r in rngs]})
    elif axis == "qset":
        # ★ 每层覆盖**全部** `n_seeds` 个 cell ⇒ 机会数须 × `n_seeds`
        #   （`S3` 曾抓到漏乘：ΣN1 得 68 而非 612）
        for r in rngs:
            out.append({"key": "qset=%d" % r,
                        "N1": n_seeds * n_of(r, 1),
                        "N2": n_seeds * n_of(r, 2),
                        "idx": [(si, r) for si in range(n_seeds)]})
    elif axis == "both":
        for si, c in enumerate(cells):
            for r in rngs:
                out.append({"key": "seed=%d,qset=%d" % (c["seed"], r),
                            "N1": n_of(r, 1), "N2": n_of(r, 2),
                            "idx": [(si, r)]})
    else:
        raise ValueError("unknown axis: %r" % (axis,))
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


# ================================================================ 精确卷积零分布
def convolve_null(ms, thetas):
    """`S = Σ_s e₁ₛ` 在 `e₁ₛ | mₛ ~ Binomial(mₛ, θₛ)` 独立下的**精确** pmf。

    零层（`m=0`）跳过；返回 `np.ndarray`，长度 `Σm + 1`，和为 `1`。
    """
    pmf = np.array([1.0])
    for m, th in zip(ms, thetas):
        m = int(m)
        if m <= 0:
            continue
        ks = np.arange(m + 1)
        b = np.array([M3.binom_pmf(m, int(k), th) for k in ks])
        pmf = np.convolve(pmf, b)
    return pmf


def _tails(pmf, obs):
    """返回 `(上尾, 下尾, 双侧精确)`；双侧 = `Σ_{pmf(k) ≤ pmf(obs)(1+tol)}`。"""
    n = len(pmf)
    obs = int(obs)
    if n == 0 or obs < 0 or obs >= n:
        return None, None, None
    upper = float(pmf[obs:].sum())
    lower = float(pmf[:obs + 1].sum())
    cut = pmf[obs] * (1.0 + 1e-9) + 1e-300
    two = float(pmf[pmf <= cut].sum())
    # ★ 浮点归一：三者数学上都不超过 1，但「全部纳入」时求和可得 1+1e-16
    #   （`A124-10` 曾因严格 `<= 1.0` 判 FAIL）。概率 > 1 无意义 ⇒ 夹到 1。
    return min(1.0, upper), min(1.0, lower), min(1.0, two)


def strat_test(rows, axis=AXIS, alpha=ALPHA, side=SIDE):
    """★ 核心：分层精确条件检验。

    逐层用**该层自己的** `m_s` 解 `û_s`（⇒ 层内条件化，异质性不进零分布），
    再对 `S = Σ_s e₁ₛ` 做精确分布检验。

    返回 `{"executable", "S", "m", "strata_used", "p_upper", "p_lower",
           "p_two", "reject", "direction", "expect"}`。
    `p_upper` 是**主判据**（`H124` 是方向性假设：`w₂ > w₁` ⇒ `e₁` 偏高）。
    """
    ms, thetas, e1s = [], [], []
    for r in rows:
        # `m` 可由 `e1 + e2` 推出；若同时给出则**必须一致**（接线错误的守卫）
        m = int(r["e1"]) + int(r["e2"])
        if "m" in r and int(r["m"]) != m:
            raise ValueError("stratum %r: m=%r 与 e1+e2=%d 不一致"
                             % (r.get("key"), r["m"], m))
        if m <= 0:
            continue
        th = M3.theta_of(m, r["N1"], r["N2"])
        if th is None:
            return {"executable": False, "reason": "theta_none",
                    "key": r.get("key"), "S": None, "m": None,
                    "strata_used": None, "p_upper": None, "p_lower": None,
                    "p_two": None, "reject": None, "direction": None,
                    "expect": None}
        ms.append(m)
        thetas.append(th)
        e1s.append(int(r["e1"]))
    if not ms:
        return {"executable": False, "reason": "no_stratum_with_m_gt_0",
                "S": 0, "m": 0, "strata_used": 0, "p_upper": None,
                "p_lower": None, "p_two": None, "reject": None,
                "direction": None, "expect": None}
    pmf = convolve_null(ms, thetas)
    S = int(sum(e1s))
    m_tot = int(sum(ms))
    exp = sum(m * t for m, t in zip(ms, thetas))
    up, lo, two = _tails(pmf, S)
    if up is None:
        return {"executable": False, "reason": "S_out_of_support", "S": S,
                "m": m_tot, "strata_used": len(ms), "p_upper": None,
                "p_lower": None, "p_two": None, "reject": None,
                "direction": None, "expect": exp}
    p_main = two if side == "two" else up
    return {"executable": True, "S": S, "m": m_tot, "strata_used": len(ms),
            "p_upper": up, "p_lower": lo, "p_two": two,
            "p_main": p_main, "side": side,
            "reject": bool(p_main <= alpha),
            "direction": ("high" if S > exp else ("low" if S < exp else "eq")),
            "expect": exp, "axis": axis}


# ================================================================ 功效 / 水平
_theta_cache = {}


def _theta_cached(m, n1, n2):
    k = (int(m), int(n1), int(n2))
    if k not in _theta_cache:
        _theta_cache[k] = M3.theta_of(*k)
    return _theta_cache[k]


def _draw_all(strata, u, ratio, reps, seed):
    """向量化按备择画 `(e1, e2)`：`k=1` 用 `p₁(u)`、`k=2` 用 `p₂(r·u)`。"""
    p1 = M3.P1(u)
    p2 = M3.P2(ratio * u)
    rng = np.random.default_rng(seed)
    E1 = np.empty((reps, len(strata)), dtype=np.int64)
    E2 = np.empty((reps, len(strata)), dtype=np.int64)
    for j, s in enumerate(strata):
        E1[:, j] = rng.binomial(int(s["N1"]), p1, size=reps)
        E2[:, j] = rng.binomial(int(s["N2"]), p2, size=reps)
    return E1, E2


def _reject_rate(strata, E1, E2, alpha, side=SIDE):
    """对每一 rep 做分层精确检验，返回拒绝率。"""
    reps = E1.shape[0]
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
        ths = [_theta_cached(mm[j], N1[keep][j], N2[keep][j])
               for j in range(len(mm))]
        if any(t is None for t in ths):
            continue
        pmf = convolve_null(mm, ths)
        S = int(e1[keep].sum())
        up, lo, two = _tails(pmf, S)
        p_main = two if side == "two" else up
        if p_main is not None and p_main <= alpha:
            hit += 1
    return hit / float(reps)


def power(axis=AXIS, u=9.2, ratio=2.0, reps=MC_REPS, seed=MC_SEED,
          alpha=ALPHA, side=SIDE, dose=DOSE_MAIN):
    """★ 备择 `w₂/w₁ = ratio` 下的**拒绝率**（`p_main <= alpha`）。机器可验证。

    ★ `dose` 决定**结构**（可用题集数、`N1/N2`）⇒ `J124-1` 与 `J124-2`
      的**功效不同**，必须分别计算（不得把主检的功效抄给负对照）。
    """
    ctx = build_ctx(dose)
    strata = strata_of(ctx, axis)
    E1, E2 = _draw_all(strata, u, ratio, reps, seed)
    return _reject_rate(strata, E1, E2, alpha, side=side)


def level(axis=AXIS, u=9.2, reps=MC_REPS, seed=MC_SEED, alpha=ALPHA, side=SIDE,
          dose=DOSE_MAIN):
    """`H124` 为真（`ratio = 1`）时的**拒绝率** ⇒ 须 ≈ `alpha`（校准检验）。"""
    return power(axis=axis, u=u, ratio=1.0, reps=reps, seed=seed,
                 alpha=alpha, side=side, dose=dose)