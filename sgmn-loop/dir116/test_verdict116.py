#!/usr/bin/env python3
"""DIR-116 · 判据纯逻辑自测（不需要模型；只在 trainer 上跑，与父卡同规矩）。

覆盖：`F116` 三条分支互斥可达、边界值（18/19/35/36）、缺格、守卫失败、天花板失败。
并用**已知错误注入**确认测试会变红（`--inject A|B|C|D`）。
"""
import argparse
import json
import pathlib
import shutil
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import R1_task116 as R          # noqa: E402

FAILS = []


def chk(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'BAD'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def mkcell(seed, dose, accs_n48, logbias10=1.0, argmax_m1_same=True):
    """构造一格：accs_n48 = {mult_str: 项数}"""
    ok_by = {}
    return {
        "seed": seed, "dose": dose, "prior_n": R.ACC0_N,
        "prior_ok": [1] * R.ACC0_N + [0] * (R.N_ITEM - R.ACC0_N),
        "acc_prior": R.ACC0_N / R.N_ITEM,
        "acc_by_mult": {k: v / R.N_ITEM for k, v in accs_n48.items()},
        "ok_by_mult": {k: [1] * v + [0] * (R.N_ITEM - v) for k, v in accs_n48.items()},
        "inj_rel_by_mult": {k: R.GAMMA_TR * float(k) for k in accs_n48},
        "wired_restore_bitwise": True, "nll_differs": True, "nll_maxdiff": 1e-7,
        "argmax_same_at_mult1": argmax_m1_same,
        "dlogit": {"1": {"mean": 5.4e-4, "min": 1e-5, "max": 2e-3, "n_pos": 48},
                   "100": {"mean": 5.4e-2, "min": 1e-3, "max": 2e-1, "n_pos": 48},
                   "ratio_100_over_1": 100.0, "naive_closed_form": R.DL_NAIVE},
        "logbias": {"0": R.ACC0_N / R.N_ITEM, "0.5": 0.5, "2": 0.9, "10": logbias10,
                    "_beta0_same_as_prior": True},
        "eval_writes": 0, "W_unchanged": True, "violations": 0, "diverged": False,
        "finite": True, "sec": 1.0,
    }


ALLM = [f"{m:g}" for m in R.MULT]


def scenario(accs_by_seed, logbias10=1.0, drop_mult=None, guard_bad=False):
    """写一份假的 stage_stageA.json 到临时 OUT，返回 verdict()。"""
    tmp = pathlib.Path(tempfile.mkdtemp())
    old = R.OUT
    R.OUT = tmp
    try:
        cells = {}
        for i, s in enumerate(R.SEEDS):
            accs = dict(accs_by_seed[i])
            if drop_mult:
                accs.pop(drop_mult, None)
            cells[R.ck(s, R.DOSE_NONE)] = mkcell(s, 0, accs, logbias10=logbias10)
        cells[R.ck(R.SEEDS[0], R.DOSE_TRAIN)] = mkcell(R.SEEDS[0], 600,
                                                       accs_by_seed[0], logbias10=logbias10)
        stageA = [{"name": "A116-1a", "ok": not guard_bad, "detail": "", "blocking": True}]
        (tmp / "stage_stageA.json").write_text(json.dumps(
            {"cells": cells, "stageA": stageA}, ensure_ascii=False), encoding="utf-8")
        return R.verdict()
    finally:
        R.OUT = old
        shutil.rmtree(tmp, ignore_errors=True)


def flat(v):
    return {m: v for m in ALLM}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", default="")
    a = ap.parse_args()
    inj = a.inject.upper()

    print("=== 1. 全 ≤18 ⇒ CHANNEL-CANNOT-DELIVER（H116 否证）")
    v = scenario([flat(13), flat(14), flat(12)])
    code = "C116-CHANNEL-CANNOT-DELIVER"
    if inj == "A":
        code = "C116-CHANNEL-DELIVERS"           # 已知错误注入 A
    chk("全部低 ⇒ 不能传递", v["code"] == code, v["code"])

    print("=== 2. 有一点 ≥36 ⇒ CHANNEL-DELIVERS（H116 证实）")
    accs = [flat(13), flat(13), flat(13)]
    for i in range(3):
        accs[i]["10000"] = 40
    v = scenario(accs)
    code = "C116-CHANNEL-DELIVERS"
    if inj == "B":
        code = "C116-CHANNEL-CANNOT-DELIVER"      # 已知错误注入 B
    chk("有一点高 ⇒ 能传递", v["code"] == code, v["code"])
    chk("confirm_hits 命中 10000", v.get("confirm_hits") == ["10000"], str(v.get("confirm_hits")))

    print("=== 3. 只在带上（19–35）⇒ INCONCLUSIVE-BAND")
    accs = [flat(30), flat(25), flat(20)]
    v = scenario(accs)
    code = "C116-INCONCLUSIVE-BAND"
    if inj == "C":
        code = "C116-CHANNEL-DELIVERS"
    chk("带内 ⇒ 不确定", v["code"] == code, v["code"])

    print("=== 4. 边界：18 仍属否证侧；19 进带；36 判证实")
    # ★ 边界必须用**各格恒等**的数据（A116-3a/3b 保证现实数据如此），否则均值会稀释单点
    v = scenario([flat(18), flat(18), flat(18)])
    chk("18 全部 ⇒ CANNOT-DELIVER", v["code"] == "C116-CHANNEL-CANNOT-DELIVER", v["code"])
    chk("18 的 cells_identical=True", v.get("cells_identical") is True)
    for val, want in ((19, "C116-INCONCLUSIVE-BAND"), (35, "C116-INCONCLUSIVE-BAND"),
                      (36, "C116-CHANNEL-DELIVERS")):
        v = scenario([flat(val), flat(val), flat(val)])
        chk(f"恒等 {val} ⇒ {want}", v["code"] == want, v["code"])

    print("=== 5. 缺格 ⇒ INCOMPLETE")
    accs = [flat(13), flat(13), flat(13)]
    v = scenario(accs, drop_mult="3000")
    code = "C116-INCOMPLETE"
    if inj == "D":
        code = "C116-CHANNEL-CANNOT-DELIVER"
    chk("缺 mult ⇒ INCOMPLETE", v["code"] == code, v["code"])

    print("=== 6. 守卫阻断失败 ⇒ GUARD-FAIL")
    v = scenario([flat(40), flat(40), flat(40)], guard_bad=True)
    chk("守卫失败优先", v["code"] == "C116-GUARD-FAIL", v["code"])

    print("=== 7. 天花板失败 ⇒ HARNESS-CEILING-FAIL（且不判通道）")
    v = scenario([flat(40), flat(40), flat(40)], logbias10=0.5)
    chk("天花板失败优先于证实", v["code"] == "C116-HARNESS-CEILING-FAIL", v["code"])

    print("=== 8. 三条判据分支互斥")
    seen = set()
    for accs, lb in (([flat(13)] * 3, 1.0), ([flat(40)] * 3, 1.0), ([flat(25)] * 3, 1.0)):
        seen.add(scenario(accs, logbias10=lb)["code"])
    chk("三码互不相同且都可达", len(seen) == 3, str(sorted(seen)))

    print("=== 9. 非恒等数据：均值确实跨格平均，且 cells_identical 报 False")
    # 4 格（3 未训练 + 1 训练）× {10,30,20,10} ⇒ 均值 17.5 ⇒ 18（若只取第一格会是 10）
    v = scenario([flat(10), flat(30), flat(20)])
    chk("均值 = 18/48（非第一格的 10）", v["mean_by_mult_n48"]["1"] == 18,
        str(v["mean_by_mult_n48"]["1"]))
    chk("cells_identical 报 False", v.get("cells_identical") is False,
        str(v.get("cells_identical")))
    chk("per-seed 4 格都对上", v["per_seed_n48"]["1"] == [10, 30, 20, 10],
        str(v["per_seed_n48"]["1"]))

    print("\n" + ("★★ 全部通过" if not FAILS else f"★★ 失败 {len(FAILS)} 项：{FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
