#!/usr/bin/env python3
"""DIR-117 · 判据纯逻辑自测（不训练、不加载模型；只 import 判据模块）。

用法：
  python test_verdict117.py                # 全部用例须通过
  python test_verdict117.py --inject A|B|C|D   # 注入**已知错误**，须至少一条用例变红

★ 纪律（`handoff` ⑥）：写自检脚本后**必须注入已知错误值确认它会变红**，
  且每次只红在**该**红的那一项。
"""
import argparse
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _ROOT / "dir90", _ROOT / "dir104", _ROOT / "dir114", _ROOT / "dir115"):
    sys.path.insert(0, str(p))

import R1_task117 as R          # noqa: E402

# ★ `DEF-117-01`：用例若用 `R.JUDGE_MULT` 建键，注入 D（改该常量）会让用例**随之漂移** ⇒
#   检测不到被注入的错误。故在 import 时**冻结**一份快照，用例一律用快照值。
JUDGE0 = R.JUDGE_MULT


def mk(train_acc, rand_acc=None, prior=17, cens_train=(), cens_rand=()):
    """train_acc/rand_acc: {mult_str: 正确项数(int)}；cens_*: 要删失的 mult_str 集合。"""
    cells = {}
    for dose in R.DOSES:
        for seed in R.SEEDS:
            src = train_acc if int(dose) == 600 else (rand_acc if rand_acc is not None else train_acc)
            cens = cens_train if int(dose) == 600 else cens_rand
            ab, fb = {}, {}
            for m in R.MULT:
                k = f"{m:g}"
                if k in cens:
                    ab[k], fb[k] = None, False
                else:
                    ab[k], fb[k] = src.get(k, prior) / 48.0, True
            cells[R.ck(dose, seed)] = {"prior_n": prior, "acc_by_mult": ab, "finite_by_mult": fb}
    return cells


def run(cells, plan_n=6, stageA=None):
    R.RES = {"cells": cells, "stageA": stageA or [], "plan_n": plan_n}
    return R.verdict()


FLAT17 = {f"{m:g}": 17 for m in R.MULT}


def case_flat():
    return mk(FLAT17), {"code": "C117-GAIN-CANNOT-RESCUE", "success": True}


def case_rescue():
    a = dict(FLAT17); a[f"{JUDGE0:g}"] = 48
    return mk(a), {"code": "C117-GAIN-RESCUES", "success": False}


def case_band():
    a = dict(FLAT17); a[f"{JUDGE0:g}"] = 25
    return mk(a), {"code": "C117-INCONCLUSIVE-BAND", "success": False}


def case_degraded():
    a = dict(FLAT17); a["3000"] = 17
    cens = {f"{m:g}" for m in (10000, 30000, 100000)}
    return (mk(a, cens_train=cens),
            {"code": "C117-GAIN-CANNOT-RESCUE", "success": True,
             "judge_degraded": True, "judge_mult": 3000})


def case_censored():
    cens = {f"{m:g}" for m in R.MULT}
    return mk(FLAT17, cens_train=cens), {"code": "C117-CENSORED", "success": False}


def case_incomplete():
    cells = mk(FLAT17)
    cells.pop(R.ck(600, R.SEEDS[0]))
    return cells, {"code": "C117-INCOMPLETE", "success": False}


def case_seed19():
    """均值 ≤17 但有 seed ≥19 ⇒ 必须归入带，不得算证实（卡片 §5-7）。"""
    cells = mk(FLAT17)
    # 三个训练 seed 分别 20 / 14 / 14 ⇒ 均值 16.0
    for seed, n in zip(R.SEEDS, (20, 14, 14)):
        k = R.ck(600, seed)
        cells[k]["acc_by_mult"][f"{JUDGE0:g}"] = n / 48.0
    return cells, {"code": "C117-INCONCLUSIVE-BAND", "success": False}


def case_drift():
    """§5-2：阻断守卫失败 ⇒ 即使读数齐全也必须报 DEVICE-DRIFT（不得给实质判决）。"""
    bad = [{"name": "A117-1", "ok": False, "detail": "gamma=0 先验 = 13（须 17）", "blocking": True}]
    cells, exp = case_flat()
    exp = dict(exp); exp["code"] = "C117-DEVICE-DRIFT"; exp["success"] = False
    return (cells, bad), exp


CASES = [
    ("flat17（全点低于先验）", case_flat),
    ("judge=48（增益可救）", case_rescue),
    ("judge=25（带）", case_band),
    ("judge 删失 ⇒ 预先指定降级", case_degraded),
    ("全点删失 ⇒ CENSORED", case_censored),
    ("缺一格 ⇒ INCOMPLETE", case_incomplete),
    ("均值≤17 但单 seed≥19 ⇒ 带（§5-7）", case_seed19),
    ("阻断守卫失败 ⇒ DEVICE-DRIFT（§5-2）", case_drift),
]

INJECT = {
    "A": ("TH_RESCUE 17（把先验本身当可用门槛）", "TH_RESCUE", 17),
    "B": ("TH_NOSIGNAL 47（容差被放大到吞掉带）", "TH_NOSIGNAL", 47),
    "C": ("ACC0_N 0（先验被抹成 0）", "ACC0_N", 0),
    "D": ("JUDGE_MULT 1（主判点挪走，救不回）", "JUDGE_MULT", 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", default=None, choices=tuple(INJECT))
    a = ap.parse_args()

    if a.inject:
        desc, attr, val = INJECT[a.inject]
        old = getattr(R, attr)
        setattr(R, attr, val)
        print(f"★ 注入已知错误 {a.inject}：{desc}（{attr} {old} → {val}）\n", flush=True)

    fails = 0
    for name, fn in CASES:
        got_in = fn()
        if name.startswith("阻断守卫失败"):
            (cells, stageA), exp = got_in
            got = run(cells, stageA=stageA)
        else:
            cells, exp = got_in
            got = run(cells)
        bad = []
        for k, v in exp.items():
            if got.get(k) != v:
                bad.append(f"{k}: 期望 {v!r} 实得 {got.get(k)!r}")
        if bad:
            fails += 1
            print(f"  [红] {name}\n       " + "\n       ".join(bad))
        else:
            print(f"  [绿] {name} ⇒ {got['code']} su={got['success']}"
                  f"{' 降级@' + str(got.get('judge_mult')) if got.get('judge_degraded') else ''}")

    tot = len(CASES)
    print(f"\n★ {tot - fails}/{tot} 通过", flush=True)
    if a.inject:
        if fails == 0:
            print(f"✗ 注入 {a.inject} 后**没有任何用例变红** ⇒ 自测无鉴别力", flush=True)
            return 1
        print(f"✓ 注入 {a.inject} 后有 {fails} 条变红 ⇒ 自测有鉴别力", flush=True)
        return 0
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())