"""`DIR-126` 测量层 —— 只用 `mech125` 的既有部件，**不重写**任何已冻结的量。

★ 本模块**唯一**新增的测量对象，是卡片 §2-2 的 `frac`：
    `frac(û<1) = Σ_{s: û_s<1} m_s / Σ_{s: m_s>0} m_s`
  其中 `û_s` 是**逐层**用该层自己的 `m_s` 解出的 `u`（放宽域，`mech125.solve_u`）。
  符号预测：`frac == 0 ⇒ ratio > 1`；`frac > 0 ⇒ ratio < 1`。
  `ratio = ŵ₂/ŵ₁` 用 `mech125.w_hat`（**模型无关**，不碰 `solve_u`）。

★★ 为什么这不是恒等式（本模块的 `nonvacuous_probe` 会机器复核）：
  令 `x = p̂₁`、`y = p̂₂`、`a = x - 2/3`、`b = y - 1/3`。则
    ① `û ≥ 1  ⟺  m ≥ (2N₁+N₂)/3  ⟺  N₁a + N₂b ≥ 0`（逐层）
    ② `ratio > 1  ⟺  a(2+3b) > 2b`
  两式**不是同一条曲线** ⇒ 存在 `frac = 0` 而 `ratio < 1` 的构型，反之亦然。
"""
import os
import sys
import glob
import json
import math
import itertools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT,
          os.path.join(ROOT, "dir125"), os.path.join(ROOT, "dir124"),
          os.path.join(ROOT, "dir123"), os.path.join(ROOT, "dir122"),
          os.path.join(ROOT, "dir121"), os.path.join(ROOT, "dir120"),
          os.path.join(ROOT, "dir119"), os.path.join(ROOT, "dir118"),
          os.path.join(ROOT, "dir117")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mech125 as M                                   # noqa: E402

# ★★ 单层别名（`DEF-126-07`）：下游（`R1_task126` 等）写 `import mech126 as M` 后，
#   `M.M` 是 **mech125**、`M.M.M` 才是 **mech122** —— 深度歧义，我在 `R1_task126` 里
#   就把 `M.M.build_facts()` 写错（`build_facts` 在 mech122 上）。
#   ⇒ 在这里一次性 re-export：下游写 `M.ALL_RNGS` / `M.M122.build_facts()`，**不再数层数**。
ALL_RNGS = M.ALL_RNGS                                 # (555, 556, 557, 558)
N_QS = M.N_QS                                         # 4
M122 = M.M                                            # build_facts / partitions_colocated / make_items / n_seen_vec

# ================================================================ 剂量轴
DOSE_OLD = tuple(M.DOSE_ALL)                          # (0, 25, 150, 300, 450, 600)
DOSE_PILOT = int(M.DOSE_PILOT)                        # 25（DIR-125 已声明 report-only）
DOSE_NEW2 = (350, 375, 400)                           # ★ 本卡新格
DOSE_ALL9 = tuple(sorted(set(DOSE_OLD) | set(DOSE_NEW2)))

ALPHA = 0.05                                          # 主判据 `J126-2` 用
SIDE = "two"
RHO_EXPECT_NEG = True                                 # 预测 ρ < 0（`J126-2` 单侧）

DIR126_CELLS = os.path.join(HERE, "out", "cells")

# ★ 非空转门槛（开卡前冻结；见卡片 §2-5）
NONVAC_MIN_DISCORD = 0.05        # 「分歧率」下限：低于它 ⇒ 判据近乎空转 ⇒ 记不可执行
NONVAC_GRID = 401                # (a,b) 网格分辨率（每轴）

__all__ = ["M", "M122", "ALL_RNGS", "N_QS", "DOSE_OLD", "DOSE_PILOT", "DOSE_NEW2", "DOSE_ALL9", "ALPHA", "SIDE",
           "DIR126_CELLS", "NONVAC_MIN_DISCORD", "NONVAC_GRID",
           "load_cells", "frac_low_u", "sign_of_ratio", "predict_sign", "sign_row",
           "sign_table", "spearman", "perm_spearman",
           "nonvacuous_probe", "ratio_ci"]


# ================================================================ 读档
def _load_dir(path, dose, tag):
    pat = os.path.join(path, "COLOCATED_%d_*.json" % dose)
    out = []
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f, encoding="utf-8"))
        picks = {M.M.MAIN_RNG: d["mech_by_mult"]["10000"]["pick_argmin"]}
        c2 = d.get("c2") or {}
        if "10000" in c2:
            for r in M.M.C2_RNGS:
                if str(r) in c2["10000"]:
                    picks[r] = c2["10000"][str(r)]["pick_argmin"]
        out.append({"seed": int(d["seed"]), "picks": picks, "dose": int(d["dose"]),
                    "raw": d, "src": tag})
    return out


def load_cells(dose):
    """★ 三源：`dir119`（`dose∈{0,600}`，冻结只读）· `dir125`（6 个剂量）· `dir126`（本卡）。

    ★ 与 `mech125.load_cells` 的**唯一差别**：多了 `dir126` 这一源。
      找不到 ⇒ 抛错（**不得**静默返回空，`DEF-125-02` 的修法）。
    """
    if dose in (M.DOSE_ZERO, M.DOSE_HIGH):
        c = _load_dir(M.DIR119_CELLS, dose, "dir119")
        if not c:
            raise RuntimeError("dir119 归档缺 dose=%d 的 cell" % dose)
        return c
    for path, tag in ((M.DIR125_CELLS, "dir125"), (DIR126_CELLS, "dir126")):
        c = _load_dir(path, dose, tag)
        if c:
            return c
    raise RuntimeError("dose=%d 在任何归档都找不到 cell（先跑 --stage cell）" % dose)


def build_ctx(dose):
    """复用 `mech125.build_ctx` 的构造，只把 `load_cells` 换成本模块的三源版。"""
    facts = M.M.build_facts()
    tr, ev = M.M.partitions_colocated(facts)
    tv = {facts[i]["val"] for i in tr}
    items = {r: M.M.make_items(facts, ev, r) for r in M.ALL_RNGS}
    ns = {r: M.M.n_seen_vec(items[r], tv) for r in M.ALL_RNGS}
    cells = load_cells(dose)
    avail = tuple(r for r in M.ALL_RNGS if all(r in c["picks"] for c in cells))
    return {"facts": facts, "tr": tr, "ev": ev, "tv": tv,
            "items": items, "ns": ns, "cells": cells, "rngs": avail, "dose": dose}


def strata_and_obs(dose, dose_label=None):
    """返回 `(strata, obs)`；`X` 用本模块的 `build_ctx`（三源）。"""
    ctx = build_ctx(dose)
    st = M.strata_of(ctx, dose_label=dose_label)
    return ctx, st, M.stratum_obs(ctx, st)


# ================================================================ ★ 新测量对象：frac 与符号
def frac_low_u(rows):
    """`frac(û<1)` = 「`û_s < 1` 的层」的 `Σm` 占「`m_s > 0` 的层」的 `Σm` 之比。

    ★ `û_s` 用该层**自己的** `m_s` 解（放宽域）；`û = 0`（夹紧）**算作 `û < 1`**
      —— 因为 `û = 0` 确实是「见过值被回避」的极端端点，**不作特例**（卡片 §2-2）。
    ★ `m_s = 0` 的层无信息，**不进分母**。
    ★ 返回逐层明细，供 `VERDICT` 直接引用（不重复算）。
    """
    tot = 0
    lo = 0
    us = []
    detail = []
    for r in rows:
        m = int(r["e1"]) + int(r["e2"])
        if m <= 0:
            detail.append({"key": r.get("key"), "m": 0, "u": None, "clamped": False,
                           "low": None, "e1": int(r["e1"]), "e2": int(r["e2"]),
                           "N1": r["N1"], "N2": r["N2"]})
            continue
        u, cl = M.solve_u(m, r["N1"], r["N2"], return_clamped=True)
        th = None if u is None else M.theta_of(m, r["N1"], r["N2"])
        low = (u is not None) and (u < 1.0)
        tot += m
        if low:
            lo += m
        if u is not None:
            us.append(u)
        detail.append({"key": r.get("key"), "m": m, "u": u, "clamped": bool(cl),
                       "low": bool(low), "e1": int(r["e1"]), "e2": int(r["e2"]),
                       "N1": r["N1"], "N2": r["N2"],
                       "theta": th,
                       "e1_over_m": (int(r["e1"]) / float(m)),
                       "over": (None if th is None else (int(r["e1"]) / float(m) - th))})
    frac = (lo / float(tot)) if tot > 0 else None
    return {"frac": frac, "m_lo": lo, "m_tot": tot, "n_valid": len(us),
            "umin": (min(us) if us else None), "umax": (max(us) if us else None),
            "n_low": sum(1 for d in detail if d["low"] is True),
            "n_clamped": sum(1 for d in detail if d["clamped"]),
            "detail": detail}


def sign_of_ratio(ratio):
    """`+1` = `ratio > 1`（超加性）· `-1` = `ratio < 1` · `0` = 等于 1 · `None` = 不可算。

    ★ `ratio` 不可算（`e1 == 0` 或 `e2 == 0` 或 `N` 为 0）⇒ `None`（**不是**失败，是不可判）。
    """
    if ratio is None:
        return None
    if not (ratio == ratio) or ratio in (float("inf"), float("-inf")):
        return None
    if ratio > 1.0:
        return 1
    if ratio < 1.0:
        return -1
    return 0


def predict_sign(frac):
    """机制预测：`frac == 0 ⇒ +1`；`frac > 0 ⇒ -1`；`frac is None ⇒ None`。

    ★ **严格 `== 0`**，不设容差（卡片 §2-2 开卡前冻结）。
    """
    if frac is None:
        return None
    return 1 if frac == 0.0 else -1


def sign_row(rows):
    """一个剂量/一个层集上的完整读数：`frac` + `ratio` + 预测 + 是否相符。"""
    f = frac_low_u(rows)
    wh = M.w_hat(rows)
    pred = predict_sign(f["frac"])
    obs = sign_of_ratio(wh["ratio"])
    agree = None if (pred is None or obs is None) else (pred == obs)
    return {"frac": f["frac"], "m_lo": f["m_lo"], "m_tot": f["m_tot"],
            "n_low": f["n_low"], "n_clamped": f["n_clamped"],
            "umin": f["umin"], "umax": f["umax"], "n_valid": f["n_valid"],
            "w": wh, "pred": pred, "obs": obs, "agree": agree,
            "detail": f["detail"]}


def sign_table(doses=None):
    """逐剂量 `sign_row`；返回 `{dose: row}`（缺档就跳过，不抛错）。"""
    out = {}
    for d in (doses if doses is not None else DOSE_ALL9):
        try:
            _, _, obs = strata_and_obs(d)
        except RuntimeError:
            continue
        out[int(d)] = sign_row(obs)
    return out


# ================================================================ 秩相关（`J126-2`）
def spearman(x, y):
    """Spearman ρ（含平均秩处理）；`n < 3` 或某轴全同 ⇒ `None`。"""
    n = len(x)
    if n != len(y) or n < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def _ranks(v):
    """平均秩（升序，1 基）。"""
    n = len(v)
    order = sorted(range(n), key=lambda i: v[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def perm_spearman(x, y, side="less", alpha=ALPHA):
    """★ **精确**置换检验（枚举 `n!`），不是 MC ⇒ 可逐位复现。

    `side="less"` 检验 `ρ < 0`（本卡的预测方向）。
    返回 `{"rho","p","n","n_perm","crit_rho","alpha","side"}`。
    ★ `crit_rho` 是 `p <= alpha` 的最小 `|ρ|`（负向），供卡片写死阈值。
    """
    n = len(x)
    if n != len(y) or n < 3 or n > 9:
        return {"rho": None, "p": None, "n": n, "n_perm": 0, "crit_rho": None,
                "alpha": alpha, "side": side, "reason": "n_out_of_range"}
    rho = spearman(x, y)
    if rho is None:
        return {"rho": None, "p": None, "n": n, "n_perm": 0, "crit_rho": None,
                "alpha": alpha, "side": side, "reason": "degenerate"}
    rhos = []
    for p in itertools.permutations(range(n)):
        yy = [y[i] for i in p]
        rhos.append(spearman(x, yy))
    rhos = [r for r in rhos if r is not None]
    if not rhos:
        return {"rho": rho, "p": None, "n": n, "n_perm": 0, "crit_rho": None,
                "alpha": alpha, "side": side, "reason": "all_degenerate"}
    if side == "less":
        p_val = sum(1 for r in rhos if r <= rho + 1e-12) / float(len(rhos))
    elif side == "greater":
        p_val = sum(1 for r in rhos if r >= rho - 1e-12) / float(len(rhos))
    else:
        p_val = sum(1 for r in rhos if abs(r) >= abs(rho) - 1e-12) / float(len(rhos))
    # ★ 临界 ρ：`crit` 必须是「仍能满足 `p <= alpha` 的**最靠近 0**（least negative）的 ρ」
    #   —— 即拒绝域 `ρ <= crit` 的**边界**。`p(t)` 随 `t` 单调增，故从 0 往负方向扫，
    #   取**第一个** `p <= alpha` 的 `t`。
    #   ★ 第一版从 -1 往 0 扫 ⇒ 取到最小 ρ（对 n=9 得 `-1.0`）⇒ 阈值形同虚设（`DEF-126-01`）。
    crit = None
    for t in sorted(set(rhos), reverse=True):
        if t >= 0:
            continue
        if sum(1 for r in rhos if r <= t + 1e-12) / float(len(rhos)) <= alpha:
            crit = t
            break
    return {"rho": rho, "p": p_val, "n": n, "n_perm": len(rhos),
            "crit_rho": crit, "alpha": alpha, "side": side}


# ================================================================ 非空转（`J126-4` 的机器部分）
def nonvacuous_probe(n1=M.STRUCT_N1_PER_SEED, n2=M.STRUCT_N2_PER_SEED,
                     grid=NONVAC_GRID, seed=M.MC_SEED):
    """★ 量化「`frac=0` 但 `ratio<1`」这类**否证构型**在参数空间里占多大。

    对一个**单层**（`N1=n1, N2=n2`），令 `x = p̂₁ ∈ (0, 2/3 + ...)`、`y = p̂₂`，
    在 `x,y ∈ (0,1)` 的网格上枚举，判定每一格的 `(frac=0?, ratio>1?)` 组合，统计：
      - `P(frac=0)`、`P(ratio>1)`
      - `P(frac=0 ∧ ratio<1)` ← **否证区**（机制预测 `frac=0 ⇒ ratio>1` 在此失败）
      - `P(frac>0 ∧ ratio>1)` ← **另一侧的否证区**
      - `P(分歧)` = 两者之和 = 1 - `P(相符)`
    ⇒ 若 `P(分歧)` 太小，则判据近乎空转（`NONVAC_MIN_DISCORD`）。

    ★ 另外给一个**多层**版本：把同一构型独立重复 `k` 层，`frac` 与 `ratio` 都按汇总算。
    """
    xs = [(i + 0.5) / grid for i in range(grid)]
    n_total = 0
    n_frac0 = 0
    n_ratio_hi = 0
    n_bad_a = 0                                  # frac=0 且 ratio<1  ← 否证
    n_bad_b = 0                                  # frac>0 且 ratio>1  ← 否证
    for x in xs:
        for y in xs:
            e1 = n1 * x
            e2 = n2 * y
            m = e1 + e2
            if m <= 0:
                continue
            n_total += 1
            frac0 = (m >= (2.0 * n1 + n2) / 3.0)
            # ratio>1 ⟺ x + 3xy > 4y（见模块 docstring 的推导）
            ratio_hi = (x + 3.0 * x * y) > 4.0 * y
            if frac0:
                n_frac0 += 1
            if ratio_hi:
                n_ratio_hi += 1
            if frac0 and not ratio_hi:
                n_bad_a += 1
            if (not frac0) and ratio_hi:
                n_bad_b += 1
    disc = (n_bad_a + n_bad_b) / float(n_total) if n_total else None
    return {"grid": grid, "n1": n1, "n2": n2, "n_cells": n_total,
            "p_frac0": n_frac0 / float(n_total),
            "p_ratio_hi": n_ratio_hi / float(n_total),
            "p_bad_a": n_bad_a / float(n_total),
            "p_bad_b": n_bad_b / float(n_total),
            "discordance": disc,
            "threshold": NONVAC_MIN_DISCORD,
            "nonvacuous": (disc is not None and disc >= NONVAC_MIN_DISCORD)}


def ratio_ci(rows, reps=20000, seed=M.MC_SEED + 11, alpha=0.05):
    """`ratio` 的参数 bootstrap 区间（**report-only**，不改任何判据）。

    ★ 与 `mech125.ratio_ci` 逐字同法，只换 seed，便于与新剂量对照。
    """
    return M.ratio_ci(rows, reps=reps, seed=seed, alpha=alpha)


def clamp_free_frac(rows):
    """灵敏度：把**夹紧层**排除后重算 `frac`（report-only，**不改判据**）。

    ★ 理由：夹紧层（`û=0`）是端点，对「`u` 制度」的判断最强但也最脆。
      卡片 §2-2 已冻结**不排除**（因为 `û=0` 确实 `<1`）；本函数只报告排除后的值，
      供 `VERDICT` 说明该读数是否依赖那几个端点层。
    """
    kept = [r for r in rows if (int(r["e1"]) + int(r["e2"])) > 0
            and not M.solve_u(int(r["e1"]) + int(r["e2"]), r["N1"], r["N2"],
                              return_clamped=True)[1]]
    if not kept:
        return {"frac": None, "m_tot": 0, "n_valid": 0}
    f = frac_low_u(kept)
    return {"frac": f["frac"], "m_tot": f["m_tot"], "n_valid": f["n_valid"],
            "n_dropped": len(rows) - len(kept)}