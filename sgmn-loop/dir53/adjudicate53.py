#!/opt/pytorch/bin/python
r"""DIR-53 · **独立裁决器**（卡 §3.3 / §4，冻结）。

★ 独立性：**不 import** `R1_task53`（也不 import `T40`）—— 只读 15 个 `DIR53r12_*.json`
  的**原始逐实例 `L_series`**，用 `statistics` 自己重算 `mean / 配对差 / se / 3se`，
  再按卡 §3.3 的三谓词与 §4 的六支判定表裁决。⇒ 与 `R1_task53.merge` 的结果一致才是**真**一致
  （同一个函数算两遍不算独立）。

★ 有效性闸（**不是判据**，与 `merge` 同口径）：
   n 不符 / 缺臂 / `OFF` 臂 `rel_dPk=0` 或行范数投影失效 / 同格 `FIX` 与 `ORACLE` 的 `pk1_sha1` 不等
   ⇒ `INCOMPLETE`（判据**从未被计算** ≠ FAIL）。

用法：`/opt/pytorch/bin/python dir53/adjudicate53.py`
"""
import json
import math
import os
import statistics
import sys

# ★ 演练用：`DIR53_OUT` 可把读数目录指向临时树（`rehearse53.py` 用）；默认为权威 `out/`。
OUT = os.environ.get("DIR53_OUT", "/home/ec2-user/sgmn-loop/dir53/out")
CELLS = ("P1", "P2", "P3", "P4", "P5")
ARMS = ("FIX", "OFF", "ORACLE")
LEVELS = ("FIX", "OFF", "OFF_rc", "ORACLE")
N_TRAIN, N_EVAL, N_REP, STEPS = 96, 400, 16, 250
SIG_K, FRAC, RATE_MIN = 3.0, 0.20, 4


def _all_finite(v):
    return all(math.isfinite(x) for x in v)


def mean(v):
    # ★ `DEF-53-5`：`statistics.stdev` 遇到 NaN/inf 会**抛 ValueError** ⇒ 裁决器会崩溃、
    #   连 `DIVERGED` 都报不出来。故：非有限一律返回 nan，由 `nonfinite`/`finite` 分支处置。
    return statistics.fmean(v) if _all_finite(v) else float("nan")


def stdev(v):
    if len(v) < 2 or not _all_finite(v):
        return float("nan")
    return statistics.stdev(v)


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    se = stdev(d) / math.sqrt(len(d)) if d else float("nan")
    return {"mean": mean(d) if d else float("nan"), "sd": stdev(d),
            "se": se, "3se": SIG_K * se, "n": len(d)}


def main():
    rows, missing = {}, []
    for c in CELLS:
        for a in ARMS:
            p = os.path.join(OUT, f"DIR53r12_{c}_{a}.json")
            if not os.path.exists(p):
                missing.append(f"{c}/{a}")
                continue
            rows[(c, a)] = json.load(open(p, encoding="utf-8"))

    bad_n, inert, sha_bad, nonfinite = [], [], [], []
    for (c, a), d in rows.items():
        for lv, v in d.get("L_series", {}).items():
            if not _all_finite(v):
                nonfinite.append(f"{c}/{a}:{lv}")
    for (c, a), d in rows.items():
        if (d.get("n_train") != N_TRAIN or d.get("n_eval") != N_EVAL
                or d.get("n_rep") != N_REP or d.get("steps") != STEPS):
            bad_n.append(f"{c}/{a}")
        if a == "OFF":
            om = d.get("off_meta") or {}
            if not (float(d.get("rel_dPk_arm", 0.0)) > 0.0
                    and float(om.get("row_norm_dev_max", 1.0)) < 1e-12):
                inert.append(f"{c}/{a}")
    for c in CELLS:
        f, o, off = rows.get((c, "FIX")), rows.get((c, "ORACLE")), rows.get((c, "OFF"))
        if f and o and f.get("pk1_sha1") != o.get("pk1_sha1"):
            sha_bad.append(f"{c}:FIX!=ORACLE")
        if f and off:
            p0 = (off.get("off_meta") or {}).get("pk0_sha1")
            if p0 and p0 != f.get("pk1_sha1"):
                sha_bad.append(f"{c}:OFF.pk0!=FIX")

    cells = {}
    # ★ 层 → 臂 的映射（`OFF_rc` 是 `OFF` 文件里的**层**，不是独立臂）——`DEF-53-4`：
    #   初稿按层名去索引文件，`("OFF_rc")` 无文件 ⇒ `any(...)` 恒真 ⇒ **每格都被跳过** ⇒
    #   `cells` 空 ⇒ 结局被静默洗成 `FIXED-SUFFICES`（"判据从未被计算"冒充结论）。
    LEVEL_ARM = {"FIX": "FIX", "OFF": "OFF", "OFF_rc": "OFF", "ORACLE": "ORACLE"}
    for c in CELLS:
        if any((c, LEVEL_ARM[lv]) not in rows
               or lv not in rows[(c, LEVEL_ARM[lv])]["L_series"] for lv in LEVELS):
            continue
        s = {lv: rows[(c, LEVEL_ARM[lv])]["L_series"][lv] for lv in LEVELS}
        L = {lv: mean(s[lv]) for lv in LEVELS}
        R = L["FIX"] - L["ORACLE"]
        floor = FRAC * R
        delta = L["FIX"] - L["OFF"]
        pd = paired(s["FIX"], s["OFF"])
        pc = paired(s["OFF_rc"], s["OFF"])
        p1 = delta >= SIG_K * pd["se"]
        p2 = delta >= floor
        p3 = pc["mean"] >= SIG_K * pc["se"]
        om = (rows[(c, "OFF")].get("off_meta") or {})
        train_improved = ((om.get("L_off_first20", 0.0) - om.get("L_off_last20", 0.0))
                          >= SIG_K * om.get("L_off_se_window", 0.0))
        cells[c] = {"L": L, "R": R, "floor": floor, "delta": delta, "pd": pd, "pc": pc,
                    "P1": bool(p1), "P2": bool(p2), "P3": bool(p3),
                    "pass": bool(p1 and p2 and p3),
                    "off_train_improved": bool(train_improved),
                    "off_rel_dPk": om.get("rel_dPk"),
                    "off_row_norm_dev_max": om.get("row_norm_dev_max"),
                    "off_L_first20": om.get("L_off_first20"),
                    "off_L_last20": om.get("L_off_last20")}

    passed = [c for c in CELLS if cells.get(c, {}).get("pass")]
    overfit = [c for c in CELLS if cells.get(c, {}).get("off_train_improved")
               and not cells.get(c, {}).get("pass")]
    finite = all(math.isfinite(v) for c in cells.values()
                 for v in [c["L"]["FIX"], c["L"]["OFF"], c["L"]["ORACLE"], c["L"]["OFF_rc"],
                           c["pd"]["se"], c["pc"]["se"]])

    if missing or bad_n or inert or sha_bad:
        outcome, counted = "INCOMPLETE", False
    elif nonfinite or not finite:
        outcome, counted = "DIVERGED", False
    elif any(c["R"] <= 0 for c in cells.values()):
        outcome, counted = "INDETERMINATE", False
    elif overfit and len(passed) < RATE_MIN:
        outcome, counted = "OVERFIT-ONLY", False
    elif len(passed) < RATE_MIN:
        outcome, counted = "FIXED-SUFFICES", False
    else:
        outcome, counted = "PASS", True

    res = {"n_cells_read": len(cells), "missing": missing, "bad_n": bad_n,
           "inert_arms": inert, "sha_mismatch": sha_bad, "nonfinite": nonfinite,
           "pass_count": len(passed), "pass_cells": passed, "overfit_cells": overfit,
           "outcome": outcome, "counted": counted, "cells": cells}

    # ── 与 `merge` 的交叉核对（同日两份读数必须一致；不一致 ⇒ 必须查）
    mp = os.path.join(OUT, "DIR53r12_formal.json")
    if os.path.exists(mp):
        m = json.load(open(mp, encoding="utf-8"))
        diffs = []
        for c in CELLS:
            if c in cells and c not in m.get("cells", {}):
                diffs.append(f"{c}: 在独立裁决中存在、但 merge 文件里缺格")
                continue
            if c in cells and c in m.get("cells", {}):
                a, b = cells[c], m["cells"][c]
                for k in ("R", "delta", "floor"):
                    av = a["floor"] if k == "floor" else a[k]
                    bv = b["floor_0.20R"] if k == "floor" else b[k]
                    if abs(av - bv) > 1e-9:
                        diffs.append(f"{c}.{k}: {av!r} vs {bv!r}")
                if a["pass"] != b["pass"]:
                    diffs.append(f"{c}.pass: {a['pass']} vs {b['pass']}")
        res["merge_agreement"] = {"outcome_equal": m.get("outcome") == outcome,
                                  "counted_equal": m.get("counted") == counted,
                                  "numeric_diffs": diffs,
                                  "agree": bool(m.get("outcome") == outcome
                                                and m.get("counted") == counted
                                                and not diffs)}

    json.dump(res, open(os.path.join(OUT, "DIR53r12_adjudication.json"), "w"),
              ensure_ascii=False, indent=1)

    print("══ DIR-53 独立裁决（不复用 merge 的算术） ══")
    for c in CELLS:
        r = cells.get(c)
        if not r:
            print(f"  [{c}] 缺读数")
            continue
        print(f"\n── {c} ──  L: FIX={r['L']['FIX']:.6f}  OFF={r['L']['OFF']:.6f}  "
              f"OFF_rc={r['L']['OFF_rc']:.6f}  ORACLE={r['L']['ORACLE']:.6f}")
        print(f"   R={r['R']:.6f}  地板={r['floor']:.6f}  Δ={r['delta']:.6f} "
              f"3se={r['pd']['3se']:.6f}  C={r['pc']['mean']:.6f} 3se={r['pc']['3se']:.6f}")
        print(f"   P1={r['P1']} P2={r['P2']} P3={r['P3']} ⇒ {'PASS' if r['pass'] else '不 PASS'}"
              f"   (OFF rel_dPk={r['off_rel_dPk']:.4f})")
    print(f"\n  PASS {len(passed)}/5 = {passed}   过拟合候选 = {overfit}")
    if "merge_agreement" in res:
        print(f"  与 merge 一致：{res['merge_agreement']['agree']}  "
              f"{res['merge_agreement']['numeric_diffs']}")
    print(f"  ★ 结局 = {outcome} · 计入 = {counted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())