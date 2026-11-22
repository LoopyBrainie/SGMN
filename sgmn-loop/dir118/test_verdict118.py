"""`test_verdict118.py` —— `DIR-118` 判据**纯逻辑**单元测试（**不加载模型、不碰 GPU**）。

★★ 纪律（`DEF-117-01`）：**本文件一个阈值都用字面量快照，绝不引用 `R.MIN_EFFECT` 等**。
   否则注入（改那个常量）时用例会**跟着漂移**，检测不到被注入的错误。
   冻结值快照：`MIN_EFFECT=3.0` `T_MIN=2.0` `EQUIV_MARGIN=2.0` `SPEC_MARGIN=3.0`。
"""
import sys
import pathlib
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import R1_task118 as R    # noqa: E402

# ★ 字面量快照（不得改成 R.MIN_EFFECT …）
MIN_EFFECT, T_MIN, EQUIV_MARGIN, SPEC_MARGIN = 3.0, 2.0, 2.0, 3.0
JUDGE, REF = "10000", "1"
ALLM = [f"{m:g}" for m in (1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000)]

# 合成基线：random 臂 acc(1)=16/48、acc(1e4)=11/48、其余 12/48；Δ(1) = −2
R_REF1, R_REFJ, R_MID = 16.0, 11.0, 12.0
D_REF = -2.0


def _acc_vec(vals):
    d = {m: R_MID / 48 for m in ALLM}
    d[REF] = vals[0] / 48
    d[JUDGE] = vals[1] / 48
    return d


def mk(R_S, R_C):
    """`R_S`/`R_C` 可为常数或 9 个逐 seed 的 `R` 值。造出完整 36 格。"""
    if not isinstance(R_S, (list, tuple)):
        R_S = [R_S] * len(R.SEEDS)
    if not isinstance(R_C, (list, tuple)):
        R_C = [R_C] * len(R.SEEDS)
    cells = {}
    for arm, Rv in (("SPLIT", R_S), ("COLOCATED", R_C)):
        for i, s in enumerate(R.SEEDS):
            d_judge = Rv[i] + D_REF
            cells[f"{arm}|0|{s}"] = {
                "arm": arm, "seed": s, "dose": 0, "prior_n": 17 if arm == "SPLIT" else 18,
                "acc_by_mult": _acc_vec([R_REF1, R_REFJ]),
                "finite_by_mult": {m: True for m in ALLM},
                "ok_by_mult": {m: None for m in ALLM}}
            cells[f"{arm}|600|{s}"] = {
                "arm": arm, "seed": s, "dose": 600, "prior_n": 17 if arm == "SPLIT" else 18,
                "acc_by_mult": _acc_vec([R_REF1 + D_REF, R_REFJ + d_judge]),
                "finite_by_mult": {m: True for m in ALLM},
                "ok_by_mult": {m: None for m in ALLM}}
    return cells


def run(cells, stageA=None):
    R.RES = {"cells": cells, "stageA": stageA or [], "plan_n": len(cells)}
    return R.verdict()


# ---------------------------------------------------------------- 用例
# ★★ 逐 seed 带**真实离散**的 `R` 模板：mean=6.33、sd≈2.96、**t≈6.4（有限）**。
#    ★ 必须用有限 `t`：若 `R` 恒定 ⇒ sd=0 ⇒ t=inf ⇒ 任何 `T_MIN` 都满足 ⇒
#      `T_MIN` 这条门变成**不可测的死代码**（注入 D 实测：全部仍绿 ⇒ 无鉴别力）。
R_VAR = [10, 8, 6, 4, 8, 10, 6, 4, 1]        # mean 6.33 / sd 2.96 / t 6.4


def case_generic():
    """R_S≈+6.3、R_C≈+5.3（都带离散）⇒ |gap|≈1.0 ≤ 2.0 ⇒ GENERIC（H118 证实）。"""
    return mk(R_VAR, [x - 1.0 for x in R_VAR]), {"code": "C118-REVERSAL-GENERIC",
                                                 "success": True}


def case_value_specific():
    """R_S≈+6.3、R_C≈+1.0 ⇒ gap≈−5.3 ≤ −3.0 ⇒ VALUE-SPECIFIC（H118 否证）。"""
    return mk(R_VAR, [x - 5.3 for x in R_VAR]), {"code": "C118-REVERSAL-VALUE-SPECIFIC",
                                                 "success": False}


def case_band_pos():
    """★ clause 6 的**正向**可证伪性：gap=+5.0（远大于 +2.0）必**不得**判 GENERIC。
    合成：R_S≈+1.0、R_C≈+6.0 ⇒ gap=+5.0。此时既不满足第 5 条（gap ≤ −3.0），
    也不满足第 6 条（|gap| ≤ 2.0）⇒ 必须落 BAND，而不是被当成"通用"。
    ★ 这一条专门证明：**clause 6 不是恒真门**（`DEF-118-05` 的教训用在决定性的那条门上）。"""
    # ★ R_S 必须 >= 3.0（否则第 4 条先跳，用例到不了第 6 条）；且 gap >= +2.0。
    #   R_VAR 均值 6.333 ⇒ [-2.33]→R_S=4.0（过第 4 条）、[+2.67]→R_C=9.0 ⇒ gap=+5.0
    return mk([x - 2.33 for x in R_VAR], [x + 2.67 for x in R_VAR]), \
           {"code": "C118-INCONCLUSIVE-BAND", "success": False}


def case_band():
    """R_S≈+6.3、R_C≈+3.8 ⇒ gap≈−2.5 ∈ (−3,−2) ⇒ BAND（不得择一）。"""
    return mk(R_VAR, [x - 2.5 for x in R_VAR]), {"code": "C118-INCONCLUSIVE-BAND",
                                                 "success": False}


def case_absent_small():
    """R_S=+2.0 < 3.0 ⇒ REVERSAL-ABSENT（即使 R_C 也很小）。"""
    return mk([x - 4.33 for x in R_VAR], 1.0), {"code": "C118-REVERSAL-ABSENT", "success": False}


def case_absent_t():
    """★ R_S 均值 = 3.0 ≥ 3.0，但逐 seed 抖动大 ⇒ t < 2.0 ⇒ REVERSAL-ABSENT。
    值 [20,0,0,0,0,0,0,3,4]：mean=3.0、sd≈6.5 ⇒ t≈1.38。"""
    return mk([20, 0, 0, 0, 0, 0, 0, 3, 4], 1.0), {"code": "C118-REVERSAL-ABSENT",
                                                    "success": False}


def case_missing():
    """缺一格 ⇒ INCOMPLETE（优先于全部实质支）。"""
    c = mk(R_VAR, [x - 1.0 for x in R_VAR])
    c.pop(f"COLOCATED|600|{R.SEEDS[0]}")
    return c, {"code": "C118-INCOMPLETE", "success": False}


def case_drift():
    """阻断守卫红 ⇒ DEVICE-DRIFT（即使读数完美、本可判 GENERIC）。"""
    bad = [{"name": "A118-3", "ok": False, "detail": "设备接续不符", "blocking": True}]
    return mk(R_VAR, [x - 1.0 for x in R_VAR]), {"code": "C118-DEVICE-DRIFT", "success": False}, bad


def case_censored():
    """主判点删失 ⇒ CENSORED。"""
    c = mk(R_VAR, [x - 1.0 for x in R_VAR])
    k = f"SPLIT|600|{R.SEEDS[0]}"
    c[k]["finite_by_mult"][JUDGE] = False
    c[k]["acc_by_mult"][JUDGE] = None
    return c, {"code": "C118-CENSORED", "success": False}


CASES = [
    ("R_S=+6.3 R_C=+5.3 ⇒ GENERIC（H118 证实）", case_generic),
    ("R_S=+6.3 R_C=+1.0 ⇒ VALUE-SPECIFIC（H118 否证）", case_value_specific),
    ("gap=−2.5 落带状区 ⇒ INCONCLUSIVE-BAND", case_band),
    ("★ gap=+5.0（正向）⇒ 必不得判 GENERIC（clause 6 可证伪）", case_band_pos),
    ("R_S=+2.0 < 3.0 ⇒ REVERSAL-ABSENT", case_absent_small),
    ("★ R_S 均值=3.0 但 t≈1.38 < 2.0 ⇒ REVERSAL-ABSENT", case_absent_t),
    ("缺一格 ⇒ INCOMPLETE（优先）", case_missing),
    ("阻断守卫红 ⇒ DEVICE-DRIFT（优先于数值）", case_drift),
    ("主判点删失 ⇒ CENSORED", case_censored),
]

INJ = {
    "A": ("MIN_EFFECT", 3.0, 0.5),        # 门槛放松 ⇒ 本该 ABSENT 的变 GENERIC-ish
    "B": ("EQUIV_MARGIN", 2.0, 6.0),      # 等价带放宽 ⇒ band 被吞成 GENERIC
    "C": ("SPEC_MARGIN", 3.0, 0.4),       # 特异门收紧 ⇒ generic 被误判成 value-specific
    "D": ("T_MIN", 2.0, 100.0),           # t 门槛荒谬 ⇒ 全部 ABSENT
}


def main():
    inj = None
    if "--inject" in sys.argv:
        inj = sys.argv[sys.argv.index("--inject") + 1]
        name, old, new = INJ[inj]
        assert abs(getattr(R, name) - old) < 1e-12, f"{name} 现值不是快照 {old}"
        setattr(R, name, new)
        print(f"★★ 注入 {inj}：{name} {old} -> {new}\n")

    ok = 0
    fails = []
    for nm, fn in CASES:
        got_in = fn()
        if len(got_in) == 3:
            cells, exp, stageA = got_in
        else:
            (cells, exp), stageA = got_in, None
        v = run(cells, stageA=stageA)
        good = (v["code"] == exp["code"] and v["success"] == exp["success"])
        ok += good
        tag = "绿" if good else "红"
        print(f"  [{tag}] {nm}\n        ⇒ {v['code']} su={v['success']}"
              + ("" if good else f"   ✗ 期望 {exp['code']} su={exp['success']}"))
        if not good:
            fails.append((nm, v["code"], exp["code"]))

    # A118-16：完整数据下 7 条 clause 全部求值
    v = run(mk(R_VAR, [x - 1.0 for x in R_VAR]))
    got_c = set(v.get("clauses", {}).keys())
    want_c = {"1_complete", "2_drift", "3_censored", "4_reversal_exists",
              "5_value_specific", "6_generic", "7_band"}
    full = want_c <= got_c
    ok += full
    print(f"  [{'绿' if full else '红'}] 7 条判据逐条求值（缺 {sorted(want_c - got_c)}）")

    n = len(CASES) + 1
    print(f"\n★ {ok}/{n} 通过")
    if inj:
        if ok < n:
            print(f"✓ 注入 {inj} 后有 {n - ok} 条变红 ⇒ 自测有鉴别力")
        else:
            print(f"✗ 注入 {inj} 后全部仍绿 ⇒ 自测无鉴别力，必须修")
    return 0 if (ok == n or inj) else 1


if __name__ == "__main__":
    sys.exit(main())
