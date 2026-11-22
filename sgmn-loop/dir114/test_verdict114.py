#!/usr/bin/env python3
"""DIR-114 · 裁决逻辑自测（**纯逻辑，不加载模型**）。

卡片 §10-2 要求必含：`INCOMPLETE`、四支判决、`A114-9` 双向、`A114-6` 顺序优先。
★ 纪律：写自测后必须**注入已知错误值确认它会变红**（连跑两次：一次应全绿，一次应挂）。
"""
import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import R1_task114 as R                                               # noqa: E402

FAIL = []


def chk(name, got, want):
    ok = (got == want)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got} want={want}")
    if not ok:
        FAIL.append(name)


def cell(acc_held, acc_train):
    return {"acc_held": float(acc_held), "acc_train": float(acc_train),
            "acc_held_q": float(acc_held), "dose": None, "seed": None}


def build(col_dmg, spl_dmg, spl_train, *, full=True, drop=None):
    """合成 54 格。`col_dmg`/`spl_dmg` = 每 seed 的 damage（acc_held(600)−acc_held(0)）。

    `acc_held(0)` 固定 0.2917（D-144 的未训练基线量级）；`acc_held(600) = 0.2917 + dmg`。
    ★ 必须凑满 `plan_keys()` 的 9 格/seed（含 d=100 两格）—— 漏了就永远 INCOMPLETE。
    """
    base = 0.2917
    mid = base + (col_dmg + spl_dmg) / 2.0          # d=100 的中间值（不进判据，只为占位）
    cells = {}
    for s in R.SEEDS:
        cells[R.ck("COLOCATED", 0, s)] = cell(base, 0.10)
        cells[R.ck("COLOCATED", 100, s)] = cell(mid, 0.50)
        cells[R.ck("COLOCATED", 600, s)] = cell(base + col_dmg, 0.95)
        cells[R.ck("SPLIT", 0, s)] = cell(base, 0.10)
        cells[R.ck("SPLIT", 100, s)] = cell(mid, 0.50)
        cells[R.ck("SPLIT", 600, s)] = cell(base + spl_dmg, spl_train)
        for d in R.LEGACY_DOSES:
            cells[R.ck("LEGACY", d, s)] = cell(base, 0.10)
        cells[R.ck("INJ-OFF", 0, s)] = cell(base, 0.10)
    if drop:
        for k in drop:
            cells.pop(k, None)
    return cells


def run(name, cells, want_code):
    R.RES["cells"] = cells
    out = R.verdict()
    chk(name, out.get("code"), want_code)
    return out


print("=== DIR-114 裁决逻辑自测 ===")

# ---- 1. A114-9 双向：缺格 ⇒ INCOMPLETE ----
print("\n[1] A114-9 双向")
pk = R.plan_keys()
chk("plan_n == 54", len(pk), 54)
run("缺 1 格 ⇒ INCOMPLETE", build(-0.19, 0.0, 0.95,
                                  drop=[R.ck("SPLIT", 600, R.SEEDS[3])]), "INCOMPLETE")
o = run("满 54 格 ⇒ 不进 INCOMPLETE", build(-0.19, 0.0, 0.95), "C114-COVERAGE-CAUSAL")
chk("满格时 n_missing == 0", o.get("n_missing"), 0)
chk("合成格数 == plan_n", len(build(-0.19, 0.0, 0.95)), 54)

# ---- 2. C114-DEVICE-DRIFT：COLOCATED 没再现现象 ----
print("\n[2] C114-DEVICE-DRIFT（COLOCATED 未再现损伤）")
run("COLOCATED damage = 0（无损伤）⇒ DRIFT", build(0.0, 0.0, 0.95), "C114-DEVICE-DRIFT")
run("COLOCATED damage = −1/48（未过 −2/48）⇒ DRIFT",
    build(-1.0 / 48.0, 0.0, 0.95), "C114-DEVICE-DRIFT")

# ---- 3. C114-COVERAGE-CAUSAL ----
print("\n[3] C114-COVERAGE-CAUSAL（H114 成立）")
o = run("COLOCATED −0.19，SPLIT 0.0，acc_train 0.95 ⇒ CAUSAL",
        build(-0.19, 0.0, 0.95), "C114-COVERAGE-CAUSAL")
chk("CAUSAL 记 success=True", o.get("success"), True)
chk("CAUSAL 的 SPLIT_fit 6/6", o["SPLIT_fit"]["n"], 6)
chk("CAUSAL 的学习门槛 == 1/16", o["SPLIT_fit"]["gate_rel"], 1.0 / 16.0)
chk("CAUSAL 的 6/6 delta ≥ 1/16",
    all(v >= 1.0 / 16.0 for v in o["SPLIT_fit"]["delta_per_seed"].values()), True)
chk("CAUSAL 的 SPLIT_damage_gone 6/6", o["SPLIT_damage_gone"]["n"], 6)

# ---- 4. C114-COVERAGE-NOT-CAUSAL（否证 F1）----
print("\n[4] C114-COVERAGE-NOT-CAUSAL（否证 F1：SPLIT 仍损伤）")
o = run("COLOCATED −0.19，SPLIT −0.19 ⇒ NOT-CAUSAL",
        build(-0.19, -0.19, 0.95), "C114-COVERAGE-NOT-CAUSAL")
chk("NOT-CAUSAL 不记 success", o.get("success"), False)

# 只有 1 个 seed 仍损伤也须翻红（不能只看均值）
cells = build(-0.19, 0.0, 0.95)
cards = dict(cells[R.ck("SPLIT", 600, R.SEEDS[2])])
cards["acc_held"] = 0.2917 - 3.0 / 48.0
cells[R.ck("SPLIT", 600, R.SEEDS[2])] = cards
run("仅 1/6 seed 的 SPLIT 仍损伤 ⇒ 仍 NOT-CAUSAL", cells, "C114-COVERAGE-NOT-CAUSAL")

# ---- 5. C114-FIX-DEGENERATES（否证 F2）----
print("\n[5] C114-FIX-DEGENERATES（否证 F2：不再学）")
run("SPLIT damage 消失但 acc_train 0.14 < 0.10+1/16 ⇒ DEGENERATES",
    build(-0.19, 0.0, 0.14), "C114-FIX-DEGENERATES")
run("SPLIT acc_train 恰好 0.10+1/16 ⇒ 算「在学」",
    build(-0.19, 0.0, 0.10 + 1.0 / 16.0), "C114-COVERAGE-CAUSAL")
run("SPLIT acc_train 0.10+1/16 − ε ⇒ 不算「在学」",
    build(-0.19, 0.0, 0.10 + 1.0 / 16.0 - 1e-3), "C114-FIX-DEGENERATES")
cells = build(-0.19, 0.0, 0.95)
c1 = dict(cells[R.ck("SPLIT", 600, R.SEEDS[4])])
c1["acc_train"] = 0.15
cells[R.ck("SPLIT", 600, R.SEEDS[4])] = c1
run("仅 1/6 seed 的 acc_train 相对提升不足 ⇒ 仍 DEGENERATES", cells, "C114-FIX-DEGENERATES")

# ---- 6. 顺序优先：DEVICE-DRIFT 压过 CAUSAL ----
print("\n[6] 顺序优先（A114-6 先于主判据）")
run("COLOCATED 未再现 + SPLIT 完美 ⇒ 仍 DRIFT（不得判 CAUSAL）",
    build(0.0, 0.0, 0.95), "C114-DEVICE-DRIFT")
run("缺格 + 其它都完美 ⇒ INCOMPLETE 压过 DRIFT",
    build(0.0, 0.0, 0.95, drop=[R.ck("COLOCATED", 0, R.SEEDS[0])]), "INCOMPLETE")

# ---- 7. 阈值边界（±1 项，卡片 §2.3）----
print("\n[7] 阈值边界（TOL=1/48, FALSIFY=2/48）")
run("SPLIT damage = −2/48 恰好 ⇒ 仍算「损伤」（≤）",
    build(-0.19, -2.0 / 48.0, 0.95), "C114-COVERAGE-NOT-CAUSAL")
run("SPLIT damage = −1.9/48 ⇒ 算「消失」",
    build(-0.19, -1.9 / 48.0, 0.95), "C114-COVERAGE-CAUSAL")
run("COLOCATED damage = −2/48 恰好 ⇒ 再现现象（≤）",
    build(-2.0 / 48.0, 0.0, 0.95), "C114-COVERAGE-CAUSAL")
run("COLOCATED damage = −1.9/48 ⇒ 未再现 ⇒ DRIFT",
    build(-1.9 / 48.0, 0.0, 0.95), "C114-DEVICE-DRIFT")

print("\n=== 结果 ===")
if FAIL:
    print(f"★ {len(FAIL)} 项失败：{FAIL}")
    sys.exit(1)
print("★ 全部通过")
sys.exit(0)