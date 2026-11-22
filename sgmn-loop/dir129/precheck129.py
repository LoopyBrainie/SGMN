#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`DIR-129` 的 **Stage A 守卫核** + 自测。

★ 纪律：本脚本**只**打 ① 结构事实 ② 已发表常数（`D-158` 的 `T1` 与 verdict 计数）
  ③ 逻辑层 level；**不打印** `C3`（纵向轨迹分类）与 `C4` 的读数 —— 那是 `--stage formal` 的事。

用法：
    python3 precheck129.py --stage precheck        # Stage A 硬门 + 自测 + 工作例
    python3 precheck129.py --stage formal          # 上面全过之后再跑，打印全部判据读数
    python3 precheck129.py --inject K              # 让第 K 条自测必定变红（反空转检查）
"""
import argparse
import json
import os
import sys

import mech129 as M

# ------------------------------------------------------------------ 合成夹具
def _row(dose, sub, frac, ratio, verdict):
    obs = 1 if ratio > 1 else (-1 if ratio < 1 else 0)
    return {
        "dose": dose, "sub": list(sub), "k": len(sub), "n_cells": 9,
        "N1": 1, "N2": 1, "gate": 1.0, "m_sum": 1, "m_max": 1, "m_lo": 1, "m_tot": 1,
        "frac": frac, "n_low": 0, "n_clamped": 0,
        "e1": 1, "e2": 1, "p1": 0.5, "p2": 0.5, "w1": 1.0, "w2": ratio,
        "ratio": ratio, "pred": M.pred_h126(frac), "obs": obs, "verdict": verdict,
    }


def fx_collinear():
    """共线成立的最小夹具：`frac=0 ⟺ dose>=350`。"""
    return [_row(0, (555,), 0.90, 0.50, "SAME"),
            _row(300, (555,), 0.40, 0.80, "SAME"),
            _row(350, (555,), 0.00, 1.50, "SAME")]


def fx_broken_below():
    """违例：`frac==0` 但 `dose < 350`。"""
    r = fx_collinear()
    r[1]["frac"] = 0.0
    return r


def fx_broken_above():
    """违例：`frac > 0` 但 `dose >= 350`。"""
    r = fx_collinear()
    r[2]["frac"] = 0.25
    return r


# ------------------------------------------------------------------ 自测 T1..T12
def _tests(inject=0):
    """返回 `[(name, fn), ...]`；`fn()` 返回 `(ok, msg)`。`inject==k` 时把第 k 条的期望翻反。"""
    T = []

    def t1():
        c = M.collinearity_census(fx_broken_below())
        exp_holds, exp_wit = False, 1
        if inject == 1:
            exp_holds, exp_wit = True, 0
        return (c["holds"] == exp_holds and len(c["witness_below"]) == exp_wit,
                f"holds={c['holds']} n_below={c['n_frac0_below']} 见证={len(c['witness_below'])}")
    T.append(("T1 共线核能抓 frac=0 在 dose<350", t1))

    def t2():
        c = M.collinearity_census(fx_broken_above())
        exp_holds, exp_wit = False, 1
        if inject == 2:
            exp_holds, exp_wit = True, 0
        return (c["holds"] == exp_holds and len(c["witness_above"]) == exp_wit,
                f"holds={c['holds']} n_above={c['n_fracpos_above']} 见证={len(c['witness_above'])}")
    T.append(("T2 共线核能抓 frac>0 在 dose>=350", t2))

    def t3():
        rows = fx_collinear()
        same = (M.vector_h126(rows) == M.vector_threshold(rows, M.TRANSITION_DOSE))
        exp = True
        if inject == 3:
            exp = False
        return (same == exp, f"V126==V350 ? {same}")
    T.append(("T3 共线夹具上 V126 与 T350 逐结构相同", t3))

    def t4():
        e = M.equivalent_thresholds(fx_collinear())
        got = [(x["lo"], x["hi"]) for x in e["equivalent"]]
        exp = [(300, 350)]
        if inject == 4:
            exp = [(300, 375)]
        return (got == exp, f"E={got} 期望={exp}")
    T.append(("T4 等价区间恰为 (300,350] 唯一一个", t4))

    def t5():
        tr = [(0, 0.9, 0.5, -1, "SAME"), (300, 0.4, 0.8, -1, "SAME"), (350, 0.0, 1.5, 1, "SAME")]
        tag, win = M.classify_trajectory(tr)
        exp = (M.TAG_CLEAN, (300, 350))
        if inject == 5:
            exp = (M.TAG_NONMONO, None)
        return ((tag, win) == exp, f"tag={tag} win={win} 期望={exp}")
    T.append(("T5 CLEAN 分类与转折窗", t5))

    def t6():
        tr = [(0, 0.9, 0.5, -1, "SAME"), (300, 0.4, 1.5, 1, "SAME"),
              (350, 0.0, 0.8, -1, "FA"), (400, 0.0, 1.5, 1, "SAME")]
        tag, _ = M.classify_trajectory(tr)
        exp = M.TAG_NONMONO
        if inject == 6:
            exp = M.TAG_CLEAN
        return (tag == exp, f"tag={tag} 期望={exp}")
    T.append(("T6 交错轨迹 ⇒ NONMONO", t6))

    def t7():
        a, _ = M.classify_trajectory([(0, 0.9, 0.5, -1, "SAME"), (300, 0.4, 0.8, -1, "SAME")])
        b, _ = M.classify_trajectory([(350, 0.0, 1.5, 1, "SAME"), (400, 0.0, 1.9, 1, "SAME")])
        exp = (M.TAG_ALL_NEG, M.TAG_ALL_POS)
        if inject == 7:
            exp = (M.TAG_ALL_POS, M.TAG_ALL_NEG)
        return ((a, b) == exp, f"all-neg={a} all-pos={b} 期望={exp}")
    T.append(("T7 单调无转折 ⇒ ALL_NEG / ALL_POS", t7))

    def t8():
        c, _ = M.classify_trajectory([(0, 0.9, 1.0, 0, "E1"), (350, 0.0, 1.5, 1, "SAME")])
        d, _ = M.classify_trajectory([(0, 0.9, None, None, "NA"), (350, 0.0, 1.5, 1, "SAME")])
        exp = (M.TAG_UNDET, M.TAG_UNDET)
        if inject == 8:
            exp = (M.TAG_CLEAN, M.TAG_CLEAN)
        return ((c, d) == exp, f"ratio==1 ⇒ {c}; ratio=None ⇒ {d} 期望={exp}")
    T.append(("T8 obs 非 ±1 ⇒ UNDET（不算失败）", t8))

    def t9():
        got = (M.pred_h126(0.0), M.pred_h126(1e-300), M.pred_h126(0.5), M.pred_h126(None))
        exp = (1, -1, -1, None)
        if inject == 9:
            exp = (1, 1, -1, None)
        return (got == exp, f"strict==0: {got} 期望={exp}")
    T.append(("T9 pred_h126 严格 ==0、无容差", t9))

    def t10():
        u = M.effective_units(fx_collinear() + [_row(350, (556,), 0.0, 1.2, "SAME")])
        exp = (2, 1.0, False)          # 2 子集 / 2 singleton ⇒ inflation = 1.0
        if inject == 10:
            exp = (2, 3.75, True)
        return ((u["n_singleton"], u["inflation"], u["is_full_lattice"]) == exp,
                f"singleton={u['n_singleton']} infl={u['inflation']} lattice={u['is_full_lattice']} 期望={exp}")
    T.append(("T10 非全格时 is_full_lattice=False", t10))

    def t11():
        iv = M.breakpoint_intervals(M.DOSE_ALL9)
        lobes = [x[1] for x in iv]
        exp = (10, [0, 25, 150, 300, 350, 375, 400, 450, 600, None])
        if inject == 11:
            exp = (9, [0, 25, 150, 300, 350, 375, 400, 450, 600, None])
        return ((len(iv), lobes) == exp, f"区间数={len(iv)} 右端={lobes} 期望={exp}")
    T.append(("T11 9 个剂量 ⇒ 10 个 breakpoint 区间", t11))

    def t12():
        rows = M.load_readings()
        c = M.cohort(rows)
        ok = (len(rows) == 121 and len(c) == 121 and (0, (555,)) in c)
        exp = True
        if inject == 12:
            exp = False
        return (ok == exp, f"len={len(rows)} cohort={len(c)} 含(0,{555})={(0, (555,)) in c}")
    T.append(("T12 readings.json 121 条且可索引", t12))

    return T


def run_selftests(inject=0):
    T = _tests(inject)
    bad = []
    for name, fn in T:
        try:
            ok, msg = fn()
        except Exception as exc:                      # noqa: BLE001
            ok, msg = False, f"异常 {type(exc).__name__}: {exc}"
        print(f"  {'✓' if ok else '✗'} {name}: {msg}")
        if not ok:
            bad.append(name)
    return bad


# ------------------------------------------------------------------ Stage A 守卫
def stage_a(rows):
    """`S1`..`S8`（卡片 §5）。返回 `[(id, ok, detail), ...]`。"""
    import collections
    out = []
    doses = M.doses_of(rows)
    subs = M.subsets_of(rows)
    by_k = collections.Counter(len(s) for s in subs)
    span = {s: len(M.dose_span(rows, s)) for s in subs}
    ok_S1 = (len(rows) == 121 and doses == M.DOSE_ALL9 and len(subs) == 15
             and (by_k[1], by_k[2], by_k[3], by_k[4]) == (4, 6, 4, 1)
             and span[(555,)] == 9 and all(span[s] == 8 for s in subs if s != (555,)))
    out.append(("S1", ok_S1, f"n={len(rows)} doses={len(doses)} subs={len(subs)} k={dict(by_k)} span555={span[(555,)]}"))

    keys = {"dose", "sub", "k", "n_cells", "N1", "N2", "gate", "m_sum", "m_max", "m_lo",
            "m_tot", "frac", "n_low", "n_clamped", "e1", "e2", "p1", "p2", "w1", "w2",
            "ratio", "pred", "obs", "verdict"}
    miss = [i for i, r in enumerate(rows) if not keys <= set(r)]
    ncell = sum(1 for r in rows if r.get("n_cells") != 9)
    nones = [r for r in rows if r.get("frac") is None or r.get("ratio") is None or r.get("obs") is None]
    out.append(("S2", not miss and ncell == 0 and not nones,
                f"缺键={len(miss)} n_cells!=9={ncell} None 读数={len(nones)}"))

    c1 = M.collinearity_census(rows)
    ok_S3 = (c1["n_frac0_below"] == 0 and c1["n_fracpos_above"] == 0
             and c1["n_frac0"] == 75 and c1["n_fracpos"] == 46)
    out.append(("S3", ok_S3,
                f"T1 复现: frac=0 在 dose<350 的条数={c1['n_frac0_below']} / frac>0 在 dose>=350 的条数={c1['n_fracpos_above']} "
                f"| 边际 frac0={c1['n_frac0']} fracpos={c1['n_fracpos']}"))

    vc = M.verdict_census(rows)
    ok_S4 = (vc.get("FA", 0) == 2 and vc.get("FB", 0) == 0 and vc.get("E1", 0) == 0
             and vc.get("NC", 0) == 0 and vc.get("NA", 0) == 0 and vc.get("NB", 0) == 0
             and vc.get("SAME", 0) == 119)
    out.append(("S4", ok_S4, f"D-158 计数复核: {dict(sorted(vc.items()))}"))

    same = sum(1 for a, b in zip(M.vector_h126(rows), M.vector_threshold(rows, M.TRANSITION_DOSE)) if a == b)
    out.append(("S5", same == 121, f"V126 与 T350 逐结构相同 {same}/121"))

    ok_S6 = (M.pred_h126(0.0) == 1 and M.pred_h126(1e-300) == -1 and M.pred_h126(None) is None)
    out.append(("S6", ok_S6, f"pred(0.0)={M.pred_h126(0.0)} pred(1e-300)={M.pred_h126(1e-300)} pred(None)={M.pred_h126(None)}"))

    from itertools import combinations
    want = set()
    for k in (1, 2, 3, 4):
        for c in combinations(M.RNG_ALL, k):
            want.add(tuple(sorted(c)))
    out.append(("S7", len(subs) == 2 ** len(M.RNG_ALL) - 1 and set(subs) == want,
                f"15==2^4-1? {len(subs) == 2 ** len(M.RNG_ALL) - 1} 格完整? {set(subs) == want}"))

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mech129.py"),
               "r", encoding="utf-8").read()
    writes = [ln.strip() for ln in src.splitlines()
              if ('open(' in ln and '"w"' in ln) or "'w'" in ln]
    out.append(("S8", not writes, f"mech129 内的写模式 open 调用数={len(writes)}"))
    return out


# ------------------------------------------------------------------ 工作例
def worked_example(rows):
    """卡片 §6 的工作例：`(dose=0, sub={555})`。"""
    r = M.cohort(rows)[(0, (555,))]
    print("\n【工作例 §6】结构 (dose=0, sub={555}) —— readings.json 第 0 条")
    print(f"  frac = {r['frac']!r}  ⇒ pred_h126 = {M.pred_h126(r['frac'])}")
    print(f"  ratio = {r['ratio']!r} ⇒ obs = {r['obs']}")
    print(f"  verdict = {r['verdict']}   （pred 与 obs 一致 ⇒ SAME）")
    print(f"  该结构属 {555} 的负集（obs=-1）：{r['obs'] == -1}")
    print("  ★ 轨迹与分类属 C3，按纪律**只在 --stage formal 打印**")
    return r['frac'], r['ratio']


# ------------------------------------------------------------------ 主流程
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="precheck", choices=["precheck", "smoke", "formal"])
    ap.add_argument("--inject", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    rows = M.load_readings()

    print("=" * 78)
    print("DIR-129 Stage A 守卫核（只打结构事实与已发表常数，不打 C3/C4 读数）")
    print("=" * 78)
    for sid, ok, detail in stage_a(rows):
        print(f"  {'✓' if ok else '✗'} {sid}: {detail}")

    print("\n自测 T1..T12" + ("（★ 注入模式，期望恰好 1 红）" if a.inject else ""))
    bad = run_selftests(a.inject)
    print(f"  ⇒ 自测 {len(_tests(a.inject)) - len(bad)}/{len(_tests(a.inject))} 过，红 {len(bad)} 条: {bad}")

    worked_example(rows)

    sbad = [sid for sid, ok, _ in stage_a(rows) if not ok]
    if a.inject:
        # 注入模式：恰好 1 条红 ⇒ 反空转通过
        ok = (len(bad) == 1)
        print(f"\n★ 注入 #{a.inject}: 红 {len(bad)} 条 ⇒ {'通过（该测试非空转）' if ok else '失败（非恰好 1 红）'}")
        return 0 if ok else 2
    if sbad:
        print(f"\n★ Stage A 失败: {sbad} ⇒ 阻断 formal")
        return 1
    print("\n★ Stage A 全过")

    if a.stage == "smoke":
        # ★ smoke：证明管线**可执行**，**不产出判据判决**（不打印 C1/C2 读数、不打印判决行）。
        res = M.analyse(rows)
        need = {"c1_collinearity", "c2_equivalence", "c3_trajectories", "c4_units", "verdict_counts"}
        shape = {
            "有五个顶层键": need <= set(res),
            "C1 有 holds": "holds" in res["c1_collinearity"],
            "C2 有 equivalent/is_empty": {"equivalent", "is_empty"} <= set(res["c2_equivalence"]),
            "C3 有 15 条 records": len(res["c3_trajectories"]["records"]) == 15,
            "C4 有 inflation": "inflation" in res["c4_units"],
            "向量长度 121": len(res["c2_equivalence"]["vec_h126"]) == 121,
        }
        print("\n【SMOKE —— 只验管线形状，不产出判决】")
        for k, v in shape.items():
            print(f"  {'✓' if v else '✗'} {k}")
        bad_shape = [k for k, v in shape.items() if not v]
        print(f"  ⇒ smoke {'通过' if not bad_shape else '失败: ' + str(bad_shape)}（★ 不出判决）")
        if a.out:
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump({"stage": "smoke", "shape": shape}, fh, ensure_ascii=False, indent=1)
        return 0 if not bad_shape else 3

    if a.stage == "formal":
        print("\n" + "=" * 78)
        print("FORMAL —— 全部判据读数")
        print("=" * 78)
        res = M.analyse(rows)
        c1, c2 = res["c1_collinearity"], res["c2_equivalence"]
        print(f"\n[C1 共线普查] holds={c1['holds']}  frac0_below={c1['n_frac0_below']}  fracpos_above={c1['n_fracpos_above']}"
              f"  |  frac0={c1['n_frac0']} fracpos={c1['n_fracpos']} 总={c1['n_total']}")
        print(f"[C2 等价阈值] E 是否为空={c2['is_empty']}  等价区间={[(x['lo'], x['hi']) for x in c2['equivalent']]}")
        c3, c4 = res["c3_trajectories"], res["c4_units"]
        print(f"\n[C3 纵向轨迹 · 报告] 计数={c3['counts']}")
        print(f"   按 k 分层={c3['by_k']}")
        print(f"   不同转折窗={c3['distinct_windows']}")
        print("   逐题集:")
        for rec in c3["records"]:
            print(f"     {str(tuple(rec['sub'])):<22} k={rec['k']} tag={rec['tag']:<9} "
                  f"win={rec['window']} obs={rec['obs']}")
        print(f"\n[C4 有效单元 · 报告] {c4}")
        print(f"\n[verdict 计数] {res['verdict_counts']}")

        # 判决表（卡片 §4）
        if not c1["holds"]:
            row = 1
        elif not c2["is_empty"]:
            row = 2
        else:
            row = 3
        verdict = {1: "SURVIVES（H129 存活，配对有判别力）",
                   2: "C129-PAIRING-CANNOT-BREAK-CONFOUND（H129 被否证）",
                   3: "UNDETERMINED（内部矛盾 ⇒ 记 DEF）"}[row]
        print(f"\n★ 判决表第 {row} 行 ⇒ {verdict}")
        print("★ SUCCEEDED 不增（保持 57）" if row in (2, 3) else "★ SUCCEEDED 不增（未结题）")
        payload = {"stage": "formal", "row": row, "verdict": verdict, "result": res}
    else:
        payload = {"stage": "precheck", "stage_a": [{"id": s, "ok": o, "detail": d} for s, o, d in stage_a(rows)]}

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print(f"\n读数已写 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
