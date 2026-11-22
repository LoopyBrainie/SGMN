#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`DIR-129` 的测量层：对 `dir127/out/readings.json` 的 **121 个既有结构**做纯再分析。

## 本模块要回答的事（卡片 §1）

`P128-1` 的提议是「**改用 rng 配对**（固定题集、只变剂量）来分离
`H126`（`u` 制度：`sign(ratio-1) = -sign(frac)`）与「纯剂量阈值族 `T_{D*}`」」。

本模块**不重跑任何实验**，只从已发表的读数文件里机械地判定：
**该提议在原理上是否可行**。

## 三条结构事实（卡片 §2，全部可机械核验）

1. **共线**：在 121 个结构上 `frac == 0  <=>  dose >= 350`（`D-158` 已发表；本模块独立复现）。
2. **网格无内点**：剂量集合 `{0,25,150,300,350,...}` 在 `(300,350)` 内**没有测点**。
3. ⇒ 在**任一固定题集**上，`frac` 是 `dose` 的**确定性阶跃**，转折点处处相同。

⇒ 由 1+2：`H126` 的预测向量与 `T_{350}` 的预测向量**在 121 个结构上逐结构相同**；
且 `D*` 的整个区间 `(300,350]` 给出**同一个**向量 ⇒ **配对无判别力**。

## 本模块的第二个（真正未发表的）测量

`D-158` 只报了 `dose=450` 的**横截面**（13 正 / 2 负）。本模块报
**15 条题集的纵向轨迹**分类 —— 这是 `D-158` 没有过的量。

★ 纯标准库；不依赖 torch/tokenizer；可在本地跑（权威值仍在 trainer 产出）。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
READINGS = os.path.join(HERE, os.pardir, "dir127", "out", "readings.json")

#: 卡片 §2-2 冻结：`D-158` 的转折剂量。
TRANSITION_DOSE = 350
#: 卡片 §2-1 冻结：剂量网格（9 个）。
DOSE_ALL9 = (0, 25, 150, 300, 350, 375, 400, 450, 600)
#: 卡片 §2-1 冻结：4 个 rng 名。
RNG_ALL = (555, 556, 557, 558)

#: 轨迹标签（卡片 §2-4 冻结）
TAG_CLEAN = "CLEAN"      # 存在转折窗 (max_neg, min_pos]
TAG_NONMONO = "NONMONO"  # 交错 ⇒ 不是单调阈值
TAG_ALL_NEG = "ALL_NEG"  # 全负 ⇒ 无转折
TAG_ALL_POS = "ALL_POS"  # 全正 ⇒ 无转折
TAG_UNDET = "UNDET"      # 含 0/None ⇒ 不可判


def load_readings(path=None):
    """读 `dir127/out/readings.json`（`mech127.all_readings()` 的落盘结果，121 条）。"""
    with open(path or READINGS, "r", encoding="utf-8") as fh:
        return json.load(fh)


def cohort(rows):
    """按 `(dose, sub)` 建索引；`sub` 转成 `tuple`。返回 `dict[(dose, sub)] -> row`。"""
    return {(r["dose"], tuple(r["sub"])): r for r in rows}


def doses_of(rows):
    """结构里出现的剂量，升序。"""
    return tuple(sorted({r["dose"] for r in rows}))


def subsets_of(rows):
    """结构里出现的题集（`tuple`），按 (大小, 字典序) 升序。"""
    return tuple(sorted({tuple(r["sub"]) for r in rows}, key=lambda s: (len(s), s)))


def dose_span(rows, sub):
    """某题集出现的剂量列表（升序）。"""
    sub = tuple(sub)
    return tuple(sorted(r["dose"] for r in rows if tuple(r["sub"]) == sub))


# ---------------------------------------------------------------- 判据 1：共线
def collinearity_census(rows, thr=None):
    """**判据 `C1`**：`frac == 0  <=>  dose >= thr` 是否在全部结构上成立。

    返回的四个计数就是判据读数（卡片 §3 冻结）：
      * `n_frac0_below`   = `frac==0` 且 `dose < thr` 的结构数  → 必须为 **0**
      * `n_fracpos_above` = `frac >0` 且 `dose >= thr` 的结构数 → 必须为 **0**
      * `n_frac0` / `n_fracpos` 为边际计数（供 `D-158` 复核：75 / 46）
    """
    thr = TRANSITION_DOSE if thr is None else thr
    n0_below = npos_above = n0 = npos = 0
    wit_below = []   # 见证（若有违例，逐条列出 —— 不许只报计数）
    wit_above = []
    for r in rows:
        fr = r["frac"]
        if fr is None:
            continue
        is0 = (fr == 0.0)          # ★ 严格 == 0，与 mech126.predict_sign 同口径
        if is0:
            n0 += 1
        else:
            npos += 1
        if is0 and r["dose"] < thr:
            n0_below += 1
            wit_below.append({"dose": r["dose"], "sub": list(r["sub"]), "frac": fr})
        if (not is0) and r["dose"] >= thr:
            npos_above += 1
            wit_above.append({"dose": r["dose"], "sub": list(r["sub"]), "frac": fr})
    return {
        "thr": thr,
        "n_total": len(rows),
        "n_frac0": n0,
        "n_fracpos": npos,
        "n_frac0_below": n0_below,
        "n_fracpos_above": npos_above,
        "holds": (n0_below == 0 and npos_above == 0),
        "witness_below": wit_below,
        "witness_above": wit_above,
    }


# ------------------------------------------------- 判据 2：阈值族与 H126 等价
def pred_h126(frac):
    """`H126` 的预测：`frac == 0 ⇒ +1`；`frac > 0 ⇒ -1`（逐字复用 `mech126.predict_sign`）。"""
    if frac is None:
        return None
    return 1 if frac == 0.0 else -1


def pred_threshold(dose, dstar):
    """纯剂量阈值族 `T_{D*}` 的预测：`dose >= D* ⇒ +1`，否则 `-1`。"""
    return 1 if dose >= dstar else -1


def vector_h126(rows):
    """`H126` 在全部结构上的预测向量（按 `(dose, sub)` 排序，保证可复现）。"""
    return tuple(pred_h126(r["frac"]) for r in sorted(rows, key=lambda r: (r["dose"], tuple(r["sub"]))))


def vector_threshold(rows, dstar):
    """`T_{D*}` 在全部结构上的预测向量（同序）。"""
    return tuple(pred_threshold(r["dose"], dstar) for r in sorted(rows, key=lambda r: (r["dose"], tuple(r["sub"]))))


def breakpoint_intervals(doses):
    """把 `D*` 的实轴按**观测剂量**切成区间；同一区间内 `T_{D*}` 的向量必然相同。

    返回 `[(lo, hi, rep), ...]`，区间为 `(lo, hi]` 半开（`lo=-inf` 用 `None`，
    `hi=+inf` 用 `None`），`rep` 是一个落在该区间内的代表点。
    """
    ds = sorted(doses)
    out = []
    lo = None
    for d in ds:
        # 区间 (lo, d]：rep 取 d（落在区间内，且 dose>=d 才为 +1）
        out.append((lo, d, float(d)))
        lo = d
    out.append((lo, None, float(ds[-1]) + 1.0))
    # 最左区间：(-inf, ds[0]]，rep 取 ds[0]
    return out


def equivalent_thresholds(rows):
    """**判据 `C2`**：哪些 `D*` 使 `T_{D*}` 的预测向量与 `H126` **逐结构相同**。

    对每个 breakpoint 区间取代表点，比较向量；返回等价区间的清单 + 是否为空。
    """
    vh = vector_h126(rows)
    eq = []
    for lo, hi, rep in breakpoint_intervals(doses_of(rows)):
        if vector_threshold(rows, rep) == vh:
            eq.append({"lo": lo, "hi": hi, "rep": rep})
    return {"vec_h126": vh, "n_structures": len(vh), "equivalent": eq, "is_empty": (len(eq) == 0)}


# ------------------------------------------------------- 判据 3：纵向轨迹分类
def trajectory(rows, sub):
    """某题集的纵向轨迹：按剂量升序的 `(dose, frac, ratio, obs, verdict)`。"""
    sub = tuple(sub)
    ts = [r for r in rows if tuple(r["sub"]) == sub]
    ts.sort(key=lambda r: r["dose"])
    return [(r["dose"], r["frac"], r["ratio"], r["obs"], r["verdict"]) for r in ts]


def classify_trajectory(traj):
    """**判据 `C3`** 的分类器（卡片 §2-4 **开卡前冻结**，不看结局设计）。

    规则（逐条、无自由参数）：
      1. 若任一 `obs` 不属于 `{-1,+1}` ⇒ `UNDET`（不可判，不算失败）。
      2. 若全为 `-1` ⇒ `ALL_NEG`；全为 `+1` ⇒ `ALL_POS`（无转折）。
      3. 若 `max(负集剂量) < min(正集剂量)` ⇒ `CLEAN`，转折窗 `(max_neg, min_pos]`。
      4. 否则（交错）⇒ `NONMONO`。
    返回 `(tag, window)`，`window = (lo, hi]` 或 `None`。
    """
    neg = [t[0] for t in traj if t[3] == -1]
    pos = [t[0] for t in traj if t[3] == 1]
    bad = [t[0] for t in traj if t[3] not in (-1, 1)]
    if bad or (not neg and not pos):
        return TAG_UNDET, None
    if not pos:
        return TAG_ALL_NEG, None
    if not neg:
        return TAG_ALL_POS, None
    lo, hi = max(neg), min(pos)
    if lo < hi:
        return TAG_CLEAN, (lo, hi)
    return TAG_NONMONO, None


def trajectory_census(rows):
    """对**全部 15 个题集**跑 `classify_trajectory`，并给出按大小分层的计数。"""
    out = []
    for s in subsets_of(rows):
        tr = trajectory(rows, s)
        tag, win = classify_trajectory(tr)
        out.append({
            "sub": list(s), "k": len(s), "n_doses": len(tr),
            "doses": [t[0] for t in tr],
            "obs": [t[3] for t in tr],
            "tag": tag, "window": list(win) if win else None,
        })
    counts = {}
    for rec in out:
        counts[rec["tag"]] = counts.get(rec["tag"], 0) + 1
    by_k = {}
    for rec in out:
        by_k.setdefault(rec["k"], {}).setdefault(rec["tag"], 0)
        by_k[rec["k"]][rec["tag"]] += 1
    windows = sorted({(rec["window"][0], rec["window"][1]) for rec in out if rec["window"]})
    return {"records": out, "counts": counts, "by_k": by_k, "distinct_windows": [list(w) for w in windows]}


# ---------------------------------------------------------------- 判据 4：功效
def effective_units(rows):
    """**判据 `C4`**：有效独立单元数。

    15 个子集是 4 个 rng 的**全部非空子集**（`2^4-1`） ⇒ **嵌套、非独立**。
    唯一不被别的子集"包含"的独立单元是 4 个 singleton。膨胀因子 = `15/4`。
    """
    subs = subsets_of(rows)
    singl = [s for s in subs if len(s) == 1]
    return {
        "n_subsets": len(subs),
        "n_singleton": len(singl),
        "inflation": len(subs) / float(len(singl)),
        "is_full_lattice": len(subs) == 2 ** len(RNG_ALL) - 1,
    }


def verdict_census(rows):
    """复核 `D-158` 的 `verdict` 计数（`SAME` / `FA` / `FB` / `E1` / `NC` / `NA` / `NB`）。"""
    c = {}
    for r in rows:
        c[r["verdict"]] = c.get(r["verdict"], 0) + 1
    return c


def analyse(rows):
    """把四条判据的读数打包（`R1_task129 --stage formal` 全部基于这个函数）。"""
    return {
        "c1_collinearity": collinearity_census(rows),
        "c2_equivalence": equivalent_thresholds(rows),
        "c3_trajectories": trajectory_census(rows),
        "c4_units": effective_units(rows),
        "verdict_counts": verdict_census(rows),
        "doses": list(doses_of(rows)),
        "subsets": [list(s) for s in subsets_of(rows)],
    }
