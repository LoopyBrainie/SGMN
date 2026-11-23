"""`DIR-130` 机制层 —— `frac` 的 **seed 数稳定性**（纯再分析，零重训、零新 cell）。

本模块只有**一个**新动作：把 `ctx["cells"]` 换成它的**非空子集** `C`。
其余全部复用 `mech127.reading` 的同一条路径（`strata_of` / `stratum_obs` /
`mech126.frac_low_u`），**不新增任何测量定义**。

★ 合法性的代数和依据（卡片 §2-3）：
  (F1) `N1`/`N2` 只由 `ns` 决定，与 `cells` **无关** ⇒ 削 seed 数不改 estimand 的总体；
  (F2) `m_s, e1_s, e2_s` 逐 cell **求和** ⇒ `frac(dose, sub, C)` 只由 `C` 里的 cell 决定；
  (F3) `û_s` 只由 `(m_s, N1, N2)` 决定 ⇒ 削 `C` 改的是**证据量**，不是总体大小。
  ⇒ `frac(dose, sub, C)` **恰好等于**"只有 `C` 里这些 seed 参与的研究"会算出的**同一个**统计量。
    这是**精确反事实**，不是重采样近似。

★ 只读：本模块**不写任何文件**（`open(` 只以只读模式出现；`S6` 机械核验）。
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _d in (os.path.join(ROOT, "dir126"), os.path.join(ROOT, "dir127"), ROOT):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import mech126 as M126                                            # noqa: E402
import mech127 as M127                                            # noqa: E402

M = M126.M                                                        # mech125

#: `DIR-127` 的落盘读数（`mech127.all_readings()` 的结果，121 条）；本卡只做**复现核**
READINGS = os.path.join(ROOT, "dir127", "out", "readings.json")

#: 卡号（**结构常量**：出现在输出标题行 `DIR-130 …` 里；`v1.3` 的 `S10` 数字来源审计须把它计入白名单）
CARD_ID = 130

# ---------------------------------------------------------------- 冻结常量（卡片 §2/§3）
DOSE_A = 300                     # 正侧（`frac > 0`）
DOSE_B = 350                     # 零侧（`frac == 0`）
#: ★ 归档里实际的 9 个 seed（**不是**连续的 20260501..09；由代码核对，`S1` 机械核）
SEEDS_ARCHIVED = (20260501, 20260502, 20260503, 20260504, 20260505, 20260506,
                  20271126, 20271203, 20271210)
N_CELLS = 9
N_SUBS = 15                      # `{555,556,557,558}` 的全部非空子集
N_CELL_SUBSETS = 2 ** N_CELLS - 1            # 511
NOMINAL_PER_DOSE = N_CELL_SUBSETS * N_SUBS   # 7665
NOMINAL_TOTAL = 2 * NOMINAL_PER_DOSE         # 15330
P350_MAX = 0.05                  # `C2` 阈值（冻结）
P300_MIN = 0.95                  # `C3` 阈值（冻结）
MIN_COMPUTABLE = 0.50            # `C0-2`：可算基数 ≥ 50% 标称

VERDICTS = ("C130-PROTOCOL-INVALID",
            "C130-FRAC-STABLE-BOTH-SIDES",
            "C130-FRAC-FRAGILE-POSITIVE",
            "C130-FRAC-NOISE-DOMINATED-ZERO",
            "C130-FRAC-NOISE-DOMINATED-BOTH")

# ---------------------------------------------------------------- 判据量载体（卡片 `v1.3` §6）
#: 当前阶段。`"precheck"` 下**真实读数的跨读数聚合量不可渲染**；`"smoke"`/`"formal"` 下无限制。
_STAGE = "formal"


def set_stage(stage):
    """设置阶段（`"precheck"` / `"smoke"` / `"formal"`）。`precheck130.py` 在入口调用。"""
    global _STAGE
    if stage not in ("precheck", "smoke", "formal"):
        raise ValueError("stage 须为 precheck/smoke/formal")
    _STAGE = stage
    return _STAGE


def get_stage():
    return _STAGE


class AggPrintBlocked(RuntimeError):
    """★ 在 `precheck` 阶段试图**渲染**一个真实读数的跨读数聚合量（`v1.3` §6 纪律）。"""


class Agg(float):
    """带身份的**跨读数聚合量**载体（`v1.3` §6 的**结构性守卫**）。

    ★ `origin="reading"`（真实读数的聚合）在 `precheck` 阶段**渲染即抛异常**
      ⇒ 被禁量**进不了文本** —— 不是「没打印」，是**打印不出来**。
    ★ `origin="fixture"`（合成夹具）**可渲染** —— 自测的期望值必须可打印（`T1`/`T10`/`T12`）。
    ★ `_isint` 记住它原本是 `int` 还是 `float`，使渲染与**冻结前**逐字一致
      （`n_comp` 印 `7440` 不是 `7440.0`；`p` 印 `0.0` 不是 `0`）。
    """

    __slots__ = ("origin", "tag", "_isint")

    def __new__(cls, value, origin="reading", tag="", isint=None):
        obj = super().__new__(cls, value)
        obj.origin = origin
        obj.tag = tag
        obj._isint = (isint if isint is not None else isinstance(value, int)) and not isinstance(value, bool)
        return obj

    def _blocked(self):
        return self.origin == "reading" and _STAGE == "precheck"

    def _txt(self, spec=""):
        v = float(self)
        if spec:
            # int 型格式码（d/x/o/b/n/c）对 float 不合法 ⇒ 先转 int
            if any(ch in spec for ch in "dxXobnc") and self._isint:
                return format(int(v), spec)
            return format(v, spec)
        return str(int(v)) if self._isint else repr(v)

    def __repr__(self):
        if self._blocked():
            raise AggPrintBlocked(self.tag)
        return self._txt()

    def __str__(self):
        if self._blocked():
            raise AggPrintBlocked(self.tag)
        return self._txt()

    def __format__(self, spec):
        if self._blocked():
            raise AggPrintBlocked(self.tag)
        return self._txt(spec)


def _origin_of(rows):
    """聚合的来源身份：只要含**真实读数**行 ⇒ `"reading"`（保守）。"""
    for r in rows:
        if isinstance(r, dict) and r.get("origin") == "reading":
            return "reading"
    return "fixture"


def plainify(obj):
    """把 `Agg` 还原成普通 `int`/`float`（用于 `json.dump` —— `json` 会把 float 子类当 float 写，
    **绕过** `__repr__` 的守卫；且 `int` 会被写成 `0.0` ⇒ 与冻结前的 JSON 不逐字相同）。"""
    if isinstance(obj, Agg):
        v = float(obj)
        return int(v) if obj._isint else v
    if isinstance(obj, dict):
        return {k: plainify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plainify(v) for v in obj]
    return obj


# ---------------------------------------------------------------- 结构层
def all_subsets():
    """`{555,556,557,558}` 的全部非空子集（**与 `DIR-127`/`DIR-129` 逐字相同**）。"""
    out = []
    for n in range(1, len(M.ALL_RNGS) + 1):
        for c in itertools.combinations(M.ALL_RNGS, n):
            out.append(tuple(c))
    return out


def k_counts():
    """`C(9,k)`，`k=1..9` —— 卡片 §2-2 的 `9,36,84,126,126,84,36,9,1`。"""
    return [len(list(itertools.combinations(range(N_CELLS), k))) for k in range(1, N_CELLS + 1)]


def cell_subsets():
    """遍历全部**非空** cell 子集，`yield (k, idxs)`；`Σ C(9,k) = 511`。"""
    for k in range(1, N_CELLS + 1):
        for idxs in itertools.combinations(range(N_CELLS), k):
            yield k, idxs


def _base_ctx(dose):
    """一次构造（`mech127` 已按 dose 缓存），避免每个子集重建 `build_facts`。"""
    return M127._supports(dose)


def reading_cells(dose, sub, idxs):
    """★ 全卡**唯一**的读数入口。

    `idxs` 是 cell 的下标子集；`sub` 是 rng 的元组。
    ★ 实现上只改两处：`ctx["rngs"]=sub`、`ctx["cells"]=子集`。
    """
    base = _base_ctx(dose)
    ctx = dict(base)                                   # 浅拷贝：不动缓存本体
    ctx["rngs"] = tuple(sub)
    ctx["cells"] = [base["cells"][i] for i in idxs]
    st = M.strata_of(ctx)
    obs = M.stratum_obs(ctx, st)
    fr = M126.frac_low_u(obs)
    n1 = int(st[0]["N1"]) if st else 0
    n2 = int(st[0]["N2"]) if st else 0
    return {
        "dose": int(dose), "sub": tuple(sub), "idxs": tuple(idxs), "k": len(idxs),
        "n_cells": len(ctx["cells"]), "N1": n1, "N2": n2,
        "m_tot": int(fr["m_tot"]), "m_lo": int(fr["m_lo"]),
        "frac": fr["frac"], "n_low": int(fr["n_low"]), "n_clamped": int(fr["n_clamped"]),
        "n_valid": int(fr["n_valid"]), "umin": fr["umin"], "umax": fr["umax"],
        "detail": fr["detail"],
        "origin": "reading",           # ★ `v1.3`：来源标记 —— 一切跨读数聚合据此判身份
    }


# ---------------------------------------------------------------- 全域枚举（`C5`/`C2`/`C3` 的原料）
def sweep(dose, subs=None):
    """遍历 `dose` 上 **全部** `(sub, C)` 对。返回 `(rows, meta)`；`rows` 有 `7665` 条。"""
    subs = all_subsets() if subs is None else list(subs)
    rows = []
    for k, idxs in cell_subsets():
        for sub in subs:
            rows.append(reading_cells(dose, sub, idxs))
    meta = {"dose": int(dose), "n_sub": len(subs), "n_cell_subsets": N_CELL_SUBSETS,
            "n_rows": len(rows)}
    return rows, meta


def p_flip(rows):
    """`p` = 在**可算**（`frac is not None`）的读数里，`frac > 0` 的占比（`C2`/`C3` 的估计量）。

    ★ `frac` **严格** `> 0`，无容差（卡片 §3）。
    ★ `frac is None`（`m_tot == 0`，不可判）**不进分母**，但计入 `n_uncomputable`。
    ★ `v1.3`：全部返回**跨读数聚合量** ⇒ 一律用带身份的 `Agg` 包装（`origin` 由行来源决定）。
    """
    comp = [r for r in rows if r["frac"] is not None]
    pos = sum(1 for r in comp if r["frac"] > 0.0)
    org = _origin_of(rows)
    n = len(rows)
    nc = len(comp)
    return {"p": (Agg(pos / float(nc), origin=org, tag="p") if nc else None),
            "n_pos": Agg(pos, origin=org, tag="n_pos", isint=True),
            "n_comp": Agg(nc, origin=org, tag="n_comp", isint=True),
            "n_uncomp": Agg(n - nc, origin=org, tag="n_uncomp", isint=True),
            "nominal": Agg(n, origin=org, tag="nominal", isint=True),
            "comp_frac": (Agg(nc / float(n), origin=org, tag="comp_frac") if n else None)}


def count_over(rows, pred, tag):
    """★ `v1.3`：**跨读数计数**的唯一出口 —— 返回带身份的 `Agg`。

    ★ 为什么需要它：跨读数的**计数**（如「多少条读数 `frac is None`」）与 `p_flip` 的
      返回值**同性质**（都是跨读数聚合），但它**不经过** `p_flip` ⇒ 若不收口，
      它就是一个可以随手 `print` 的裸整数 ⇒ 泄漏面重现（`P133-8` 实测的 `None=270`/`零=11092`）。
    ★ 因此：**一切跨读数计数必须走这里或 `p_flip`**，两者都返回 `Agg` ⇒ precheck 渲染即失败。
    """
    n = sum(1 for r in rows if pred(r))
    return Agg(n, origin=_origin_of(rows), tag=tag, isint=True)


def per_k(rows):
    """`C5`：逐 `k` 的占比曲线（**报告**）。"""
    out = {}
    for k in range(1, N_CELLS + 1):
        sub = [r for r in rows if r["k"] == k]
        d = p_flip(sub)
        d["n_rows"] = len(sub)
        out[k] = d
    return out


def u_margin(rows, k9_only=False):
    """`C4`：`û` 余量（**报告**）。只在 `m_s>0` 的层上取，`û` 被夹紧者单独计数。"""
    us = []
    n_cl = 0
    for r in rows:
        if k9_only and r["k"] != N_CELLS:
            continue
        for d in r["detail"]:
            if d["m"] <= 0:
                continue
            if d["clamped"]:
                n_cl += 1
            if d["u"] is not None:
                us.append(d["u"])
    lows = [u for u in us if u < 1.0]
    org = _origin_of(rows)

    def _f(x):
        return None if x is None else Agg(x, origin=org, tag="u_margin")

    return {"umin": _f((min(us) if us else None)), "umax": _f((max(us) if us else None)),
            "u_maxlow": _f((max(lows) if lows else None)),
            "margin_above": _f((min(us) - 1.0) if us else None),
            "margin_below": _f((1.0 - max(lows)) if lows else None),
            "n_layers": Agg(len(us), origin=org, tag="n_layers", isint=True),
            "n_low_layers": Agg(len(lows), origin=org, tag="n_low_layers", isint=True),
            "n_clamped": Agg(n_cl, origin=org, tag="n_clamped", isint=True)}


# ---------------------------------------------------------------- 复现核（`C0-1` / `S3`）
def load_readings(path=None):
    """读 `dir127/out/readings.json`（`mech127.all_readings()` 的落盘结果，121 条）。"""
    with open(path or READINGS, "r", encoding="utf-8") as fh:
        return json.load(fh)


def reproduction(rows, readings=None):
    """`k=9` 的 30 个结构是否与 `readings.json` **逐位相同**（`frac` 用 `repr` 相等）。

    ★ 返回逐条比对明细，供 `S3` 与 `VERDICT` 直接引用。
    """
    if readings is None:
        readings = load_readings()
    ref = {}
    for r in readings or []:
        ref[(int(r["dose"]), tuple(r["sub"]))] = r
    bad = []
    n = 0
    for r in rows:
        if r["k"] != N_CELLS:
            continue
        key = (int(r["dose"]), tuple(r["sub"]))
        if key not in ref:
            bad.append({"key": key, "why": "readings 缺该结构"})
            continue
        e = ref[key]
        n += 1
        if repr(r["frac"]) != repr(e["frac"]):
            bad.append({"key": key, "field": "frac", "got": repr(r["frac"]), "exp": repr(e["frac"])})
        if int(r["m_tot"]) != int(e["m_tot"]):
            bad.append({"key": key, "field": "m_tot", "got": r["m_tot"], "exp": e["m_tot"]})
        if int(r["n_low"]) != int(e["n_low"]):
            bad.append({"key": key, "field": "n_low", "got": r["n_low"], "exp": e["n_low"]})
    return {"n_compared": n, "n_bad": len(bad), "bad": bad[:20], "ok": (n > 0 and not bad)}


# ---------------------------------------------------------------- 判决（`C0`/`C2`/`C3`）
def decide(c0_ok, p350, p300):
    """判决表逐行（卡片 §3）。**顺序即优先级**：`C0` 不过 ⇒ 不产出判决。"""
    if not c0_ok:
        return VERDICTS[0]
    c2 = (p350 is not None) and (p350 <= P350_MAX)
    c3 = (p300 is not None) and (p300 >= P300_MIN)
    if c2 and c3:
        return VERDICTS[1]
    if c2 and not c3:
        return VERDICTS[2]
    if (not c2) and c3:
        return VERDICTS[3]
    return VERDICTS[4]


def analyse(readings=None, rows_a=None, rows_b=None):
    """全域枚举 + 判据。**只在 `--stage formal` 调用**（产出 C2/C3 读数）。

    ★ `rows_a`/`rows_b` 可由调用方传入（`Stage A` 的 `S5`/`S8` 已做同一遍全域枚举）
      ⇒ **不重复枚举**，且保证 Stage A 与 formal 看的是**同一批读数**。
    """
    subs = all_subsets()
    if rows_a is None:
        rows_a, meta_a = sweep(DOSE_A, subs)
    else:
        meta_a = {"dose": DOSE_A, "n_sub": len(subs), "n_cell_subsets": N_CELL_SUBSETS,
                  "n_rows": len(rows_a)}
    if rows_b is None:
        rows_b, meta_b = sweep(DOSE_B, subs)
    else:
        meta_b = {"dose": DOSE_B, "n_sub": len(subs), "n_cell_subsets": N_CELL_SUBSETS,
                  "n_rows": len(rows_b)}
    if readings is None:
        readings = load_readings()
    rep = reproduction(rows_a + rows_b, readings)
    pa, pb = p_flip(rows_a), p_flip(rows_b)
    c0 = {"reproduction_ok": rep["ok"], "n_compared": rep["n_compared"],
          "comp_frac_a": pa["comp_frac"], "comp_frac_b": pb["comp_frac"],
          "min_computable": MIN_COMPUTABLE}
    c0_ok = (rep["ok"] and pa["comp_frac"] is not None and pb["comp_frac"] is not None
             and pa["comp_frac"] >= MIN_COMPUTABLE and pb["comp_frac"] >= MIN_COMPUTABLE)
    v = decide(c0_ok, pb["p"], pa["p"])
    return {"meta_a": meta_a, "meta_b": meta_b,
            "c0": dict(c0, ok=c0_ok, reproduction=rep),
            "c2_p350": dict(pb, thr_max=P350_MAX, holds=(pb["p"] is not None and pb["p"] <= P350_MAX)),
            "c3_p300": dict(pa, thr_min=P300_MIN, holds=(pa["p"] is not None and pa["p"] >= P300_MIN)),
            "c4_u350": u_margin(rows_b), "c4_u350_k9": u_margin(rows_b, k9_only=True),
            "c4_u300": u_margin(rows_a), "c4_u300_k9": u_margin(rows_a, k9_only=True),
            "c5_per_k_350": per_k(rows_b), "c5_per_k_300": per_k(rows_a),
            "n_rows_total": len(rows_a) + len(rows_b),
            "verdict": v}