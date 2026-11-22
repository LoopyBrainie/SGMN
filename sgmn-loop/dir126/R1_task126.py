#!/usr/bin/env python3
"""`R1_task126.py` —— `DIR-126` rev 1.0（冻结）实现。

★ 逐字派生：**直接 import** `dir125/R1_task125.py`（它 → `dir119` → `dir118` → … → `dir90`）。
★ 卡片 §7-1 声明的改动**只有两条**：
    ① 独立变量：加 `dose ∈ {350, 375, 400}`（`M.DOSE_NEW2`）
    ② 输出目录：`OUT` 改指 `dir126/out`（`DEF-106-06` 的正面预防）
  ★ **测量层**（`frac_low_u` 等）在 `mech126.py`，不在本文件。
  ★ **训练 / 增益网格 / 超参 / 题集 / seed / `N_CAND` / `rng_seed` /
    `score_and_mech` 调用序列 / `run_cell` / `build_c2` 全部复用 `R1_task125`**，
    **一字不动**（`DEF-118-17`：禁止重写训练部分）。

用法：
  --stage a        Stage A 硬门（不需要新剂量数据）
  --stage smoke    冒烟：跑 1 个新剂量格，验 `c2` 非空 + 结构常量 + 自确定性
  --stage cell --dose D --seed S   单格（一格一进程，`DEF-105-02`）
  --stage formal   汇总 + 裁决
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_R = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_R / "dir125"))
sys.path.insert(0, str(_R / "dir119"))

import numpy as np                                     # noqa: E402

import mech126 as M                                    # noqa: E402
import R1_task125 as R125                              # noqa: E402

P119 = R125.P119
P118 = R125.P118
D90 = R125.D90
SEEDS = R125.SEEDS
ARM = R125.ARM
MULT = R125.MULT

# ★ 卡片 §7-1 ①：新剂量轴（只加这三个）
DOSES = tuple(M.DOSE_NEW2)                             # (350, 375, 400)
SMOKE_DOSE, SMOKE_SEED = DOSES[0], SEEDS[0]

# ★ 卡片 §7-1 ②：OUT 改指 `dir126/out`
OUT = _HERE / "out"
for _d in ("cells", "det", "c2"):
    (OUT / _d).mkdir(parents=True, exist_ok=True)
D90.OUT = OUT
P118.OUT = OUT
P119.OUT = OUT
R125.OUT = OUT                                         # ★ `R125.run_cell` 用模块级 OUT 写 cell

RES = {"stageA": [], "cells": {}, "smoke": {}, "verdict": {}}


def rec(name, ok, detail, blocking=True):
    RES.setdefault("stageA", []).append(
        {"name": name, "ok": bool(ok), "detail": str(detail), "blocking": bool(blocking)})
    print("  [%s] %s: %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    return bool(ok)


def dump(o, n):
    p = OUT / n
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")


def md5_cells(dirpath):
    out = {}
    p = pathlib.Path(dirpath)
    if not p.is_dir():
        return out
    for f in sorted(p.glob("*.json")):
        out[f.name] = hashlib.md5(f.read_bytes()).hexdigest()
    return out


def close(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


# ================================================================ 单格
def cells_main(seed, dose):
    """★ 复用 `R125.cells_main` 的**全部**逻辑，只把写盘目标换到 `dir126/out`。"""
    assert int(dose) in DOSES, "dose=%s 不在本卡的剂量轴 %s 里" % (dose, DOSES)
    return R125.cells_main(int(seed), int(dose))


# ================================================================ Stage A
def stage_a():
    print("=" * 104)
    print("Stage A —— 硬门（`A126-1…10`，不需要新剂量数据）")
    print("=" * 104)

    sa = {}

    # ---- A126-1：D-156 冻结读数逐位不变
    _, _, obs600 = M.strata_and_obs(600)
    r600 = M.M.strat_test(obs600)
    ok1 = (r600["S"] == 109 and r600["m"] == 133
           and close(r600["expect"], 90.1568, 1e-3)
           and close(r600["p_two"], 3.66782e-04, 1e-8)
           and r600["direction"] == "high" and r600["clamped_layers"] == 0)
    sa["A126-1"] = {"ok": ok1, "S": r600["S"], "m": r600["m"],
                    "expect": r600["expect"], "p_two": r600["p_two"],
                    "dir": r600["direction"], "clamped": r600["clamped_layers"]}
    rec("A126-1", ok1, "dose=600 S=%s m=%s E[S]=%.4f p_two=%.9g dir=%s clamped=%s"
        % (r600["S"], r600["m"], r600["expect"], r600["p_two"],
           r600["direction"], r600["clamped_layers"]))

    # ---- A126-2：D-156 的 6 个剂量 (frac, ratio) 逐位
    PUB = {0: (0.8589, 0.9353), 25: (0.6422, 0.7262), 150: (0.7376, 0.4596),
           300: (0.3389, 0.7587), 450: (0.0000, 1.4336), 600: (0.0000, 1.9571)}
    ok2 = True
    got2 = {}
    for d, (pf, pr) in PUB.items():
        _, _, ob = M.strata_and_obs(d)
        rr = M.sign_row(ob)
        good = (abs(rr["frac"] - pf) <= 1e-4 and close(rr["w"]["ratio"], pr, 1e-3))
        ok2 = ok2 and good
        got2[str(d)] = {"frac": rr["frac"], "ratio": rr["w"]["ratio"], "ok": good}
    sa["A126-2"] = {"ok": ok2, "doses": got2}
    rec("A126-2", ok2, "6 个剂量的 (frac, ratio) 逐位复现：%s"
        % " ".join("d%s:%.4f/%.4f" % (d, v["frac"], v["ratio"]) for d, v in got2.items()))

    # ---- A126-3 / A126-4：归档 md5 前后不变（基线在 precheck）
    before119 = md5_cells(M.M.DIR119_CELLS)
    before125 = md5_cells(M.M.DIR125_CELLS)
    sa["A126-3"] = {"ok": len(before119) == 18, "n": len(before119)}
    sa["A126-4"] = {"ok": len(before125) == 36, "n": len(before125)}
    rec("A126-3", len(before119) == 18, "dir119 = %d 个 cell（基线，formal 里复核不变）"
        % len(before119))
    rec("A126-4", len(before125) == 36, "dir125 = %d 个 cell（★ 本卡不得改动，formal 里复核）"
        % len(before125))

    # ---- A126-5：非空转硬门
    nv = M.nonvacuous_probe()
    ok5 = bool(nv["nonvacuous"] and nv["discordance"] >= M.NONVAC_MIN_DISCORD)
    sa["A126-5"] = {"ok": ok5, **{k: nv[k] for k in ("discordance", "p_bad_a", "p_bad_b")}}
    rec("A126-5", ok5, "非空转：discordance=%.6f ≥ 门槛 %.2f（bad_a=%.4f, bad_b=%.4f）"
        % (nv["discordance"], M.NONVAC_MIN_DISCORD, nv["p_bad_a"], nv["p_bad_b"]))

    # ---- A126-6：工作例 W1/W2/W4/W5 逐位
    def one(e1, e2):
        rr = M.sign_row([{"key": "w", "N1": 68, "N2": 99, "e1": int(e1), "e2": int(e2),
                          "m": int(e1) + int(e2)}])
        return {"frac": rr["frac"], "ratio": rr["w"]["ratio"], "agree": rr["agree"]}

    W = {"W1": one(46, 35), "W2": one(42, 37), "W4": one(41, 30), "W5": one(49, 38)}
    ok6 = (close(W["W1"]["ratio"], 0.955844, 1e-5) and W["W1"]["agree"] is True
           and close(W["W2"]["ratio"], 0.676715, 1e-5) and W["W2"]["agree"] is True
           and W["W4"]["frac"] == 0.0 and close(W["W4"]["ratio"], 0.873148, 1e-5)
           and W["W4"]["agree"] is False
           and W["W5"]["frac"] == 1.0 and close(W["W5"]["ratio"], 1.034972, 1e-5)
           and W["W5"]["agree"] is False)
    sa["A126-6"] = {"ok": ok6, "worked": W}
    rec("A126-6", ok6, "W1=%.6f/符%s W2=%.6f/符%s W4=%.6f/符%s(F-a) W5=%.6f/符%s(F-b)"
        % (W["W1"]["ratio"], W["W1"]["agree"], W["W2"]["ratio"], W["W2"]["agree"],
           W["W4"]["ratio"], W["W4"]["agree"], W["W5"]["ratio"], W["W5"]["agree"]))

    # ---- A126-7：精确置换全枚举
    x9 = list(range(9))
    ps = M.perm_spearman(x9, list(reversed(x9)), side="less")
    ok7 = (ps["n_perm"] == 362880 and close(ps["crit_rho"], -0.6, 1e-9))
    sa["A126-7"] = {"ok": ok7, "n_perm": ps["n_perm"], "crit_rho": ps["crit_rho"]}
    rec("A126-7", ok7, "n_perm=%d（须 362880）crit_rho=%.6f（须 -0.6）"
        % (ps["n_perm"], ps["crit_rho"]))

    # ---- A126-8：三个新剂量的 cell 齐全（formal 前可跳过；formal 里必查）
    got8 = {}
    all9 = True
    for d in DOSES:
        try:
            c = M.load_cells(d)
        except RuntimeError:
            c = []
        got8[str(d)] = len(c)
        if len(c) != 9:
            all9 = False
    sa["A126-8"] = {"ok": all9, "n_by_dose": got8, "deferred_to_formal": True}
    rec("A126-8", all9, "新剂量 cell 数 = %s（须各 9；stage a 阶段允许未跑）"
        % got8, blocking=False)

    # ---- A126-9：m=0 层不进分母
    f9 = M.frac_low_u([{"key": "z", "N1": 68, "N2": 99, "e1": 0, "e2": 0, "m": 0},
                       {"key": "a", "N1": 68, "N2": 99, "e1": 41, "e2": 30, "m": 71}])
    ok9 = (f9["m_tot"] == 71 and f9["n_valid"] == 1 and f9["frac"] == 0.0)
    sa["A126-9"] = {"ok": ok9, "m_tot": f9["m_tot"], "n_valid": f9["n_valid"]}
    rec("A126-9", ok9, "m_tot=%d(须71) n_valid=%d(须1) frac=%.1f"
        % (f9["m_tot"], f9["n_valid"], f9["frac"]))

    # ---- A126-10：两支在 6 个旧剂量上一致 ⇒ 新剂量是唯一判别窗口
    alt = {d: (1 if d >= 450 else -1) for d in M.DOSE_ALL9}
    ok10 = True
    mism = {}
    for d in M.DOSE_OLD:
        _, _, ob = M.strata_and_obs(d)
        pr = M.sign_row(ob)["pred"]
        if pr is not None and pr != alt[d]:
            ok10 = False
            mism[str(d)] = pr
    sa["A126-10"] = {"ok": ok10, "mismatch": mism}
    rec("A126-10", ok10, "两支在 6 个旧剂量上预测一致=%s ⇒ 三个新剂量是唯一判别窗口" % ok10)

    dump(sa, "stage_stageA.json")
    RES["stageA"] = sa
    RES["d119_md5_before"] = before119
    RES["d125_md5_before"] = before125
    dump(RES, "stage_stageA_ctx.json")

    blk = [k for k, v in sa.items() if not v["ok"]
           and k not in ("A126-8",)]
    print("\n★ Stage A：%d 条断言，阻断失败 = %s"
          % (len(sa), ("无" if not blk else blk)))
    return 0 if not blk else 1


# ================================================================ Smoke
def stage_smoke():
    print("=" * 104)
    print("Smoke —— 协议可执行（`S126-1…3`；**不给判据判决**）")
    print("=" * 104)
    # ★ `DEF-126-05`：第一版在这里直接 `run_cell(..., write_cell=True)` 再一次
    #   `write_cell=False` 做自确定性对比 —— 第二次会被 `P118.run_cell` 的
    #   **skip-if-exists** 命中（文件已在 `OUT/cells` 里）⇒ `S_all` 空 ⇒ 断言炸。
    #   修法：两次都用**各自独立**的 `scratch_out` + `write_cell=False`，
    #   **完全不写** `OUT/cells`；正式的 27 格由驱动统一产出（格式一致）。
    tok, model = D90.model_and_tok()
    facts, parts, csf, IT = P118.ground(tok, ARM)
    items, meta = IT["items_ev"], IT["meta_ev"]
    tv, tr, ev = IT["tv"], IT["tr"], IT["ev"]
    c2i, c2m, c2t = R125.build_c2(tok, facts, ev)

    def _run(tag):
        sc = OUT / "det" / ("smoke_%s" % tag)
        sc.mkdir(parents=True, exist_ok=True)
        for f in (sc / "cells").glob("COLOCATED_*.json"):
            f.unlink()                       # 清除上一次残留，否则 skip-if-exists 命中
        return R125.run_cell(int(SMOKE_SEED), int(SMOKE_DOSE), tok, model, facts, tr, ev,
                             items, meta, tv, c2i, c2m, c2t,
                             write_cell=False, scratch_out=sc)

    da = _run("a")
    db = _run("b")

    # ---- S126-1：c2 对新剂量非空（4 题集）—— DEF125-02 未复发
    c2 = da.get("c2") or {}
    inner = c2.get("10000") or {}
    ok1 = (set(str(r) for r in M.M.M.C2_RNGS) <= set(inner))
    rec("S126-1", ok1, "c2['10000'] 的题集键 = %s（须含 %s）"
        % (sorted(inner), sorted(str(r) for r in M.M.M.C2_RNGS)))

    # ---- S126-2：结构常量 (68,99)（**纯结构**，不需要任何 cell）
    #   ★ `DEF-126-04`（第 1 版）：写成 `len(stn) == 9` —— 但 smoke 阶段只有 1 个 cell ⇒ 假 FAIL。
    #   ★ `DEF-126-06`（第 2 版）：改成读 `strata_and_obs(350)` —— 但 smoke 现在**不写** cell
    #     ⇒ `load_cells(350)` 抛 RuntimeException。
    #   正解：`N₁/N₂` **只由 `n_seen` 直方图决定**（卡片 §3-1），直接从 facts 算，不碰 cell。
    _facts = M.M122.build_facts()
    _tr, _ev = M.M122.partitions_colocated(_facts)
    _tv = {_facts[i]["val"] for i in _tr}
    _c1 = _c2 = 0
    for r in M.ALL_RNGS:
        ns = M.M122.n_seen_vec(M.M122.make_items(_facts, _ev, r), _tv)
        _c1 += sum(1 for v in ns if v == 1)
        _c2 += sum(1 for v in ns if v == 2)
    _, st6, _ = M.strata_and_obs(600)                  # dose=600 的归档格可读
    ok2 = (_c1 == 68 and _c2 == 99 and M.N_QS == 4
           and all(x["N1"] == 68 and x["N2"] == 99 for x in st6))
    rec("S126-2", ok2, "4 题集的 n_seen 直方图 ⇒ N₁=%d（须68）N₂=%d（须99）；dose=600 逐层（68,99）"
        % (_c1, _c2))

    # ---- S126-3：自确定性（两个独立 scratch 进程逐位相同）
    same = (json.dumps(da["acc_by_mult"], sort_keys=True)
            == json.dumps(db["acc_by_mult"], sort_keys=True))
    picks_same = all((da["mech_by_mult"].get(k) or {}).get("pick_argmin")
                     == (db["mech_by_mult"].get(k) or {}).get("pick_argmin")
                     for k in da["mech_by_mult"])
    ok3 = bool(same and picks_same)
    rec("S126-3", ok3, "两个独立 scratch 的 acc_by_mult 逐位相同=%s、9 个 mult 的 "
                        "pick_argmin 逐位相同=%s" % (same, picks_same))

    dump({"seed": int(SMOKE_SEED), "dose": int(SMOKE_DOSE),
          "acc_by_mult": da["acc_by_mult"],
          "c2_keys": sorted(da.get("c2") or {}),
          "mean_acc_over_mults": float(sum(da["acc_by_mult"].values())
                                       / len(da["acc_by_mult"])),
          "self_det": {"acc_same": bool(same), "picks_same": bool(picks_same)},
          "card": "DIR-126/v10"}, "stage_smoke.json")

    RES["stageA"] = [{"name": k, "ok": v, "blocking": True}
                     for k, v in (("S126-1", ok1), ("S126-2", ok2), ("S126-3", ok3))]
    print("\n★ Smoke：%s" % ("全绿" if (ok1 and ok2 and ok3) else "有红"))
    return 0 if (ok1 and ok2 and ok3) else 1


# ================================================================ Formal
def stage_formal():
    print("=" * 104)
    print("Formal —— 汇总 + 裁决（`DIR-126`）")
    print("=" * 104)

    # ---- A126-8 正式查（cell 必须 9/9）
    n_by = {}
    complete = True
    for d in DOSES:
        try:
            c = M.load_cells(d)
        except RuntimeError:
            c = []
        n_by[str(d)] = len(c)
        if len(c) != 9:
            complete = False
    rec("A126-8", complete, "新剂量 cell 数 = %s（须各 9）" % n_by)

    # ---- 逐剂量读数（9 个剂量）
    per = {}
    for d in M.DOSE_ALL9:
        try:
            _, _, ob = M.strata_and_obs(d)
        except RuntimeError as e:
            print("  ⚠ dose=%d 不可读：%s" % (d, e))
            continue
        rr = M.sign_row(ob)
        ci = M.ratio_ci(ob)
        cf = M.clamp_free_frac(ob)
        ms = [o["m"] for o in ob]
        gate = (2.0 * ob[0]["N1"] + ob[0]["N2"]) / 3.0 if ob else None
        per[int(d)] = {"frac": rr["frac"], "ratio": rr["w"]["ratio"], "ci": ci,
                       "pred": rr["pred"], "obs": rr["obs"], "agree": rr["agree"],
                       "n_low": rr["n_low"], "n_valid": rr["n_valid"],
                       "n_clamped": rr["n_clamped"], "m_tot": rr["m_tot"],
                       "m_lo": rr["m_lo"], "umin": rr["umin"], "umax": rr["umax"],
                       "max_m": (max(ms) if ms else None), "gate": gate,
                       "clamp_free_frac": cf,
                       "w": {k: rr["w"][k] for k in ("N1", "N2", "e1", "e2", "p1", "p2",
                                                     "w1", "w2", "ratio")},
                       "detail": rr["detail"]}

    print("\n  %-6s %-7s %-9s %-9s %-8s %-8s %-9s %-13s %s"
          % ("dose", "Σm", "frac", "ratio", "pred", "obs", "max m_s", "闸门", "符?"))
    print("  " + "-" * 96)
    for d in M.DOSE_ALL9:
        if d not in per:
            continue
        p = per[d]
        print("  %-6d %-7d %-9.4f %-9.4f %-8s %-8s %-9s %-13s %s"
              % (d, p["m_tot"], p["frac"], p["ratio"], p["pred"], p["obs"],
                 p["max_m"], ("%.3f" % p["gate"]) if p["gate"] else "-",
                 {True: "✓", False: "✗否证", None: "—"}[p["agree"]]))

    # ---- J126-1：三个新剂量的符号预测
    j1 = {}
    for d in DOSES:
        if d not in per:
            j1[str(d)] = {"frac": None, "ratio": None, "pred": None, "obs": None,
                          "agree": None, "status": "不可判(MISSING)"}
            continue
        p = per[d]
        if p["obs"] is None or p["pred"] is None:
            st = "不可判(N-c)"
        elif p["agree"] is True:
            st = "相符"
        elif p["frac"] == 0.0:
            st = "否证(F-a)"
        else:
            st = "否证(F-b)"
        j1[str(d)] = {"frac": p["frac"], "ratio": p["ratio"], "pred": p["pred"],
                      "obs": p["obs"], "agree": p["agree"], "status": st,
                      "n_low": p["n_low"], "n_valid": p["n_valid"],
                      "m_lo": p["m_lo"], "m_tot": p["m_tot"],
                      "max_m": p["max_m"], "gate": p["gate"]}
        print("  J126-1 dose=%-4d frac=%-8.4f ratio=%-9.4f pred=%+d obs=%s ⇒ %s"
              % (d, p["frac"], p["ratio"], p["pred"], p["obs"], st))

    n_fa = sum(1 for v in j1.values() if v["status"] == "否证(F-a)")
    n_fb = sum(1 for v in j1.values() if v["status"] == "否证(F-b)")
    n_agree = sum(1 for v in j1.values() if v["status"] == "相符")
    n_und = sum(1 for v in j1.values() if v["status"].startswith("不可判"))
    any_frac0 = any(v["frac"] == 0.0 for v in j1.values() if v["frac"] is not None)

    # ---- J126-2：全 9 个剂量的秩一致（精确置换，单侧）
    xs, ys, used = [], [], []
    for d in M.DOSE_ALL9:
        if d in per and per[d]["frac"] is not None and per[d]["ratio"] is not None:
            xs.append(per[d]["frac"])
            ys.append(per[d]["ratio"])
            used.append(d)
    j2 = M.perm_spearman(xs, ys, side="less")
    j2["doses_used"] = used
    print("\n  J126-2（n=%d，单侧 less，精确置换）：rho=%.6f  crit_rho=%s  p=%.6f ⇒ %s"
          % (j2["n"], j2["rho"], j2["crit_rho"], j2["p"],
             "拒绝（ρ 显著为负）" if (j2["p"] is not None and j2["p"] <= M.ALPHA) else "未拒绝"))

    # ---- 稳健性（report-only）：排除 dose=0（结构不可比，边界 11）
    xs2, ys2, used2 = [], [], []
    for d in M.DOSE_ALL9:
        if d == 0:
            continue
        if d in per and per[d]["frac"] is not None and per[d]["ratio"] is not None:
            xs2.append(per[d]["frac"])
            ys2.append(per[d]["ratio"])
            used2.append(d)
    j2b = M.perm_spearman(xs2, ys2, side="less")
    j2b["doses_used"] = used2
    print("  J126-2 稳健性（排除 dose=0，n=%d）：rho=%.6f  p=%.6f（report-only，不改判据）"
          % (j2b["n"], j2b["rho"], j2b["p"]))

    # ---- J126-4 非空转
    nv = M.nonvacuous_probe()

    # ---- 归档只读复核
    after119 = md5_cells(M.M.DIR119_CELLS)
    after125 = md5_cells(M.M.DIR125_CELLS)
    before119 = RES.get("d119_md5_before") or {}
    before125 = RES.get("d125_md5_before") or {}
    if not before119 or not before125:
        p = OUT / "stage_stageA_ctx.json"
        if p.is_file():
            ctx = json.loads(p.read_text(encoding="utf-8"))
            before119 = before119 or ctx.get("d119_md5_before") or {}
            before125 = before125 or ctx.get("d125_md5_before") or {}
    ok119 = bool(before119) and before119 == after119
    ok125 = bool(before125) and before125 == after125
    rec("A126-3", ok119, "dir119 逐位不变=%s（%d 个）" % (ok119, len(after119)))
    rec("A126-4", ok125, "dir125 逐位不变=%s（%d 个）" % (ok125, len(after125)))

    # ---- 判决表（卡片 §5，5 行）
    nonvac_ok = bool(nv["nonvacuous"] and nv["discordance"] >= M.NONVAC_MIN_DISCORD)
    j1_ok = (n_agree == len(DOSES)) and any_frac0
    j1_agree_all = (n_agree == len(DOSES))
    j2_ok = (j2["p"] is not None and j2["p"] <= M.ALPHA)

    if not complete or not nonvac_ok or n_und == len(DOSES):
        row, verdict = 5, "INEXECUTABLE"
    elif n_fa + n_fb > 0:
        row, verdict = 3, "H126_FALSIFIED"
    elif j1_ok and j2_ok:
        row, verdict = 1, "H126_CONFIRMED_MEDIATED"
    elif j1_agree_all:
        row, verdict = 2, "UNDECIDABLE_BOTH_ALIVE"
    else:
        row, verdict = 4, "SIGN_OK_RANK_FAILS"

    print("\n" + "=" * 104)
    print("★★ 判决：第 %d 行 → %s" % (row, verdict))
    print("   J126-1：相符 %d/%d、否证(F-a) %d、否证(F-b) %d、不可判 %d；任一 frac=0 = %s"
          % (n_agree, len(DOSES), n_fa, n_fb, n_und, any_frac0))
    print("   J126-2：ρ=%.6f ≤ crit %.6f ? %s" % (j2["rho"], j2["crit_rho"], j2_ok))
    print("=" * 104)

    RES["verdict"] = {
        "row": row, "verdict": verdict, "complete": complete,
        "n_cells_by_dose": n_by, "doses": list(DOSES),
        "J126_1": j1, "J126_1_agree": n_agree, "J126_1_any_frac0": any_frac0,
        "J126_1_n_fa": n_fa, "J126_1_n_fb": n_fb, "J126_1_n_und": n_und,
        "J126_2": {k: j2[k] for k in ("rho", "p", "n", "n_perm", "crit_rho",
                                      "alpha", "side")},
        "J126_2_doses": used, "J126_2_ok": j2_ok,
        "J126_2_sensitivity_no_dose0": {k: j2b[k] for k in ("rho", "p", "n",
                                                            "crit_rho")},
        "J126_4": {"discordance": nv["discordance"], "p_bad_a": nv["p_bad_a"],
                   "p_bad_b": nv["p_bad_b"], "nonvacuous": nv["nonvacuous"],
                   "threshold": M.NONVAC_MIN_DISCORD},
        "per_dose": {str(d): {k: v[k] for k in ("frac", "ratio", "pred", "obs", "agree",
                                                "n_low", "n_valid", "n_clamped",
                                                "m_tot", "m_lo", "umin", "umax",
                                                "max_m", "gate", "clamp_free_frac",
                                                "ci")}
                     for d, v in per.items()},
        "archive_unchanged": {"dir119": ok119, "dir125": ok125},
        "stageA": RES.get("stageA"),
    }
    dump(RES["verdict"], "formal.json")
    print("产物：dir126/out/formal.json")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["a", "smoke", "cell", "formal"])
    ap.add_argument("--dose", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    t0 = time.time()
    if a.stage == "a":
        rc = stage_a()
    elif a.stage == "smoke":
        rc = stage_smoke()
    elif a.stage == "cell":
        assert a.dose in DOSES, "--dose 必须是 %s 之一" % (DOSES,)
        assert a.seed in SEEDS, "--seed 必须是 %s 之一" % (SEEDS,)
        o = cells_main(a.seed, a.dose)
        rc = 0 if o else 1
    else:
        rc = stage_formal()
    print("\n[rc=%d, %.1f s]" % (rc, time.time() - t0))
    return rc


if __name__ == "__main__":
    sys.exit(main())