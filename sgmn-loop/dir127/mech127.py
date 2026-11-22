#!/usr/bin/env python3
"""`DIR-127` 测量层 —— 「**固定 `dose`、换题集子集**」结构上的读数。

★★★ 唯一的实现接缝（本卡的**全部**新增逻辑）：
    把 `ctx["rngs"]` 覆写为子集 `S`。
    因 `mech125.strata_of` 的 `N1/N2` 是 `sum(n_of(r,·) for r in ctx["rngs"])`、
    其 `idx` 是 `[(si, r) for r in ctx["rngs"]]` ⇒ **结构（闸门）与观测（e1/e2）
    同时、一致地**切到 `S`，不存在"结构切了观测没切"的错配。
    ⇒ 这使本卡成为**纯再分析**：不训练、不写 cell、不动任何归档。

★ 所有测量函数**逐字复用**，本模块**不重写**任何测量：
    `mech126.frac_low_u` / `mech126.sign_of_ratio` / `mech126.predict_sign`
    `mech125.w_hat` / `mech125.strata_of` / `mech125.stratum_obs`
  本模块只做三件事：① 枚举子集 ② 覆写 `rngs` ③ 分类判据结局。

★★ `ratio == 1.0` 的处置（**卡片缺口，登记 `DEF-127-01`**）：
    卡片 §1-3 的 (F-a) 要 `ratio < 1`、(F-b) 要 `ratio > 1`、(N-a) 要 `ratio` 不可算。
    `ratio == 1.0` **三者都不属**（可算，但符号为 0）。
    ⇒ 判为**第四类不可判 (E-1)**，**不计入**违例、**不计入**相符。
    理由：`H127` 的判据文字是 `frac=0 ⇒ ratio>1`，而 `ratio==1.0` 是 `obs=0 ≠ +1`
    ⇒ 既不能算"相符"（那是追认），也不能算"否证"（卡片没把它写成否证条件）。
    ★ 该情形在实测里**是否出现**由 `formal` 报数；若出现，`VERDICT` 必须单列。
"""
import itertools
import sys

sys.path.insert(0, "dir126")
sys.path.insert(0, ".")
import mech126 as M126  # noqa: E402

M = M126.M          # mech125（w_hat / strata_of / stratum_obs / solve_u / build_facts …）
ALL_RNGS = tuple(M126.ALL_RNGS)
N_QS = M126.N_QS
DOSE_ALL9 = tuple(M126.DOSE_ALL9)
DOSE_OLD = tuple(M126.DOSE_OLD)
DOSE_NEW2 = tuple(M126.DOSE_NEW2)

__all__ = ["M", "M126", "ALL_RNGS", "N_QS", "DOSE_ALL9", "DOSE_OLD", "DOSE_NEW2",
           "all_subsets", "available_rngs", "subsets_for", "gate_of", "reading",
           "all_readings", "verdict_of", "classify", "VERDICT_TAGS"]

#: 判据结局标签。★ `SAME` = 相符；`FA`/`FB` = 两类否证；`NA`/`NB`/`NC` = 卡片的三类不可判；
#: `E1` = `ratio == 1.0`（**本模块新增的第四类不可判**，见模块 docstring）。
VERDICT_TAGS = ("SAME", "FA", "FB", "NA", "NB", "NC", "E1")


def all_subsets():
    """15 个非空题集子集，按 `(k, 字典序)` 排列（与 `probe127_struct.py` 同一顺序）。"""
    out = []
    for k in range(1, len(ALL_RNGS) + 1):
        out.extend(itertools.combinations(ALL_RNGS, k))
    return out


def available_rngs(dose):
    """该剂量**实际**可用的题集（只读 `picks` 的**键**）与 cell 列表。"""
    cells = M126.load_cells(dose)
    avail = tuple(r for r in ALL_RNGS if all(r in c["picks"] for c in cells))
    return avail, cells


def subsets_for(dose):
    """该剂量可用的非空子集（**只保留全部题集都在 `avail` 里的**）。"""
    avail, cells = available_rngs(dose)
    subs = [s for s in all_subsets() if all(r in avail for r in s)]
    return subs, avail, cells


def gate_of(n1, n2):
    """闸门 `(2N₁+N₂)/3` —— 逐结构量，与观测无关。"""
    return (2.0 * n1 + n2) / 3.0


def _supports(dose):
    """构造一次（缓存），避免每个子集重跑 `build_facts`。"""
    key = ("_supports", dose)
    cache = _supports.cache
    if key in cache:
        return cache[key]
    ctx = M126.build_ctx(dose)
    cache[key] = ctx
    return ctx


_supports.cache = {}


def reading(dose, sub):
    """`R(dose, sub)` —— 全卡**唯一**的读数入口。

    ★ 实现只有一行是新的：`ctx["rngs"] = tuple(sub)`。
      其余全是复用（`strata_of` / `stratum_obs` / `frac_low_u` / `w_hat`）。
    """
    base = _supports(dose)
    ctx = dict(base)                     # 浅拷贝：只改 rngs，不动缓存本体
    ctx["rngs"] = tuple(sub)
    cells = ctx["cells"]
    st = M.strata_of(ctx)
    obs = M.stratum_obs(ctx, st)
    fr = M126.frac_low_u(obs)
    wh = M.w_hat(obs)

    n1 = int(st[0]["N1"]) if st else 0
    n2 = int(st[0]["N2"]) if st else 0
    ms = [int(r["e1"]) + int(r["e2"]) for r in obs]
    return {
        "dose": int(dose),
        "sub": tuple(sub),
        "k": len(sub),
        "n_cells": len(cells),
        "N1": n1,
        "N2": n2,
        "gate": gate_of(n1, n2),
        "m_sum": int(sum(ms)),
        "m_max": (max(ms) if ms else 0),
        "m_lo": int(fr["m_lo"]),
        "m_tot": int(fr["m_tot"]),
        "frac": fr["frac"],
        "n_low": int(fr["n_low"]),
        "n_clamped": int(fr["n_clamped"]),
        "e1": int(wh["e1"]),
        "e2": int(wh["e2"]),
        "p1": wh["p1"],
        "p2": wh["p2"],
        "w1": wh["w1"],
        "w2": wh["w2"],
        "ratio": wh["ratio"],
        "detail": fr["detail"],
    }


def classify(r):
    """把一个读数判成 `VERDICT_TAGS` 之一。**判据文字逐字来自卡片 §1-3 / 模块 docstring。**"""
    if r["n_cells"] != 9:
        return "NC"
    if r["ratio"] is None:
        return "NA"
    if r["m_tot"] == 0 or r["frac"] is None:
        return "NB"
    if r["ratio"] == 1.0:
        return "E1"
    if r["frac"] == 0.0:
        return "SAME" if r["ratio"] > 1.0 else "FA"
    return "SAME" if r["ratio"] < 1.0 else "FB"


def verdict_of(r):
    """给读数补上 `pred` / `obs` / `verdict`（不修改入参）。"""
    r = dict(r)
    r["pred"] = M126.predict_sign(r["frac"])
    r["obs"] = M126.sign_of_ratio(r["ratio"])
    r["verdict"] = classify(r)
    return r


def all_readings(doses=None, include_full_only=False):
    """全部 `(dose, 子集)` 读数（`verdict_of` 之后）。默认 = 卡片 §2-1 的 121 个结构。"""
    out = []
    for d in (DOSE_ALL9 if doses is None else doses):
        subs, avail, _cells = subsets_for(d)
        full = tuple(sorted(avail))
        for s in subs:
            if include_full_only and tuple(s) != full:
                continue
            out.append(verdict_of(reading(d, s)))
    return out
