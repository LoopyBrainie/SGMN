#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-106 裁决逻辑的**单元测试**（`DEF-106-07`）。

为什么需要它：`verdict_main` 的**分支顺序**决定了「缺格」会不会退化成一个实质结论。
若缺格 ⇒ `all_at_chance=False` ⇒ 落到 `C106-PARTIAL`，而 `PARTIAL` 的语义是
「测了、但不是全贴地板」——**那是科学结论，不是「还没跑完」**。一个中途崩掉的驱动
会因此伪装成 `C106-PARTIAL`，而 `PARTIAL` 恰好是**禁止**把 `§7.2-7b` 记 `已足够` 的分支
⇒ 错误方向上是"安全"的，但会**误导**后续 cycle 以为数据已齐。

本测试**不加载模型、不碰 GPU**（驱动仍在跑，避免任何竞争），
只校验：完备性守卫、四支判决、以及池化算术。

用法：`/opt/pytorch/bin/python test_verdict106.py`
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import R1_task106 as T

FAILS = []


def ck(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{' · ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


def mk_cells(arm, n_list, acc_held_by_n, scale=1):
    """造某一 arm 的格。`scale` = 该 (arm,n) 的格数。"""
    out = []
    for n in n_list:
        for k in range(scale):
            acc = acc_held_by_n[n]
            n_held = T.N_FACTS - n
            out.append({"arm": arm, "n_train": n, "split": k % 4, "seed": T.SEEDS[k % 3],
                        "acc_held": acc, "n_held": n_held,
                        "acc_train": 0.9375, "W_drift": 0.0})
    return out


def full_set(acc_main, guards_ok=True):
    """造一份**完备**的 72 格数据集。"""
    cells = []
    # main: 4 划分 × 3 seed = 12 格/n
    for n in T.N_TRAIN_SET:
        for sp in range(4):
            for sd in T.SEEDS:
                cells.append({"arm": "main", "n_train": n, "split": sp, "seed": sd,
                              "acc_held": acc_main[n], "n_held": T.N_FACTS - n,
                              "acc_train": 0.9375, "W_drift": 0.0})
    # shuf / w0: 1 划分 × 3 seed = 3 格/n
    for arm in ("shuf", "w0"):
        for n in T.N_TRAIN_SET:
            for sd in T.SEEDS:
                cells.append({"arm": arm, "n_train": n, "split": 0, "seed": sd,
                              "acc_held": 0.0625, "n_held": T.N_FACTS - n,
                              "acc_train": 0.9375, "W_drift": 0.0})
    return {"cells": cells,
            "guards": {"A106-2": {"ok": guards_ok, "detail": "synthetic"}}}


def run(d, tag):
    p = pathlib.Path(f"/tmp/t106_{tag}.json")
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    T.verdict_main(str(p))
    d = json.loads(pathlib.Path(f"/tmp/t106_{tag}_verdict.json").read_text(encoding="utf-8"))
    # ★ `d["verdict"]` 是 **dict**（内含 "verdict" 键）⇒ 必须用扁平键 `verdict_str`（`DEF-106-09`）
    #   ⇒ 三元组：判决字符串 / 判决明细 / **完整文档**（`guards` 在顶层，不在 `RESv` 里）
    return d["verdict_str"], d["verdict"], d


print("=" * 70)
print("T-1 池化算术与题数账目（合并后应为 192/144/96/48）")
print("=" * 70)
ck("EXPECT_NQ_MAIN", T.EXPECT_NQ_MAIN == {16: 192, 20: 144, 24: 96, 28: 48},
   f"{T.EXPECT_NQ_MAIN}")
ck("EXPECT_CELLS", T.EXPECT_CELLS == {"main": 48, "shuf": 12, "w0": 12}, f"{T.EXPECT_CELLS}")
ck("格总数 72", sum(T.EXPECT_CELLS.values()) == 72, str(sum(T.EXPECT_CELLS.values())))

print()
print("=" * 70)
print("T-2 四支判决：每个场景必须命中**唯一**分支")
print("=" * 70)

# --- C106-REACHABLE：n=28 池化 0.5625 ≥ 0.50，且严格递增
arc = {16: 0.0625, 20: 0.1250, 24: 0.3125, 28: 0.5625}
v, d, doc = run(full_set(arc), "reach")
ck("C106-REACHABLE", v == "C106-REACHABLE", f"实测 {v}；M1={d.get('M106-1')} M2={d.get('M106-2')}")

# --- C106-NO-GENERALIZATION：四点全在 chance+2SE 内（0.0625 远低于 0.25）
arc2 = {16: 0.0625, 20: 0.0625, 24: 0.0625, 28: 0.0625}
v, d, doc = run(full_set(arc2), "nogen")
ck("C106-NO-GENERALIZATION", v == "C106-NO-GENERALIZATION",
   f"实测 {v}；all_at_chance={d.get('all_at_chance')}")

# --- C106-PARTIAL：严格递增但 n=28 未到 0.50
arc3 = {16: 0.0625, 20: 0.1250, 24: 0.2500, 28: 0.3750}
v, d, doc = run(full_set(arc3), "partial")
ck("C106-PARTIAL", v == "C106-PARTIAL", f"实测 {v}；M1={d.get('M106-1')} M2={d.get('M106-2')}")

# --- UNTESTABLE：A106-2 不 ok
v, d, doc = run(full_set(arc, guards_ok=False), "untest")
ck("UNTESTABLE", v == "UNTESTABLE", f"实测 {v}")

print()
print("=" * 70)
print("T-3 ★ 缺格必须判 INCOMPLETE，**不得**退化成 C106-PARTIAL（DEF-106-07 核心）")
print("=" * 70)

# --- 缺 1 格（删掉一个 main 格）⇒ 题数少 16 ⇒ INCOMPLETE
d = full_set(arc3)
d["cells"] = [c for c in d["cells"] if not (c["arm"] == "main" and c["n_train"] == 16
                                            and c["split"] == 0 and c["seed"] == T.SEEDS[0])]
v, dd, doc = run(d, "miss1")
ck("缺 1 格 ⇒ INCOMPLETE", v == "INCOMPLETE",
   f"实测 {v}（若为 PARTIAL 即为**假结论**）")
ck("缺 1 格时 A106-14 为 FAIL", doc["guards"]["A106-14"]["ok"] is False,
   str(doc["guards"]["A106-14"]["value"]))

# --- 只有 main 臂（shuf/w0 全缺）⇒ INCOMPLETE
d = full_set(arc3)
d["cells"] = [c for c in d["cells"] if c["arm"] == "main"]
v, dd, doc = run(d, "missing_arms")
ck("缺 shuf/w0 臂 ⇒ INCOMPLETE", v == "INCOMPLETE", f"实测 {v}")

# --- 空 cells ⇒ INCOMPLETE（不得崩溃、不得 None）
v, dd, doc = run({"cells": [], "guards": {"A106-2": {"ok": True}}}, "empty")
ck("空数据 ⇒ INCOMPLETE", v == "INCOMPLETE", f"实测 {v}")

# --- 全部 acc_held 为 null（n=32 那种情形误入 main）⇒ INCOMPLETE
d = full_set(arc3)
for c in d["cells"]:
    c["acc_held"] = None
v, dd, doc = run(d, "allnull")
ck("acc_held 全 null ⇒ INCOMPLETE", v == "INCOMPLETE", f"实测 {v}")

print()
print("=" * 70)
print("T-4 完备性守卫不得把**正确**数据判成缺格（反向保护）")
print("=" * 70)
v, dd, doc = run(full_set(arc), "okcase")
ck("完备数据 A106-14 为 PASS", doc["guards"]["A106-14"]["ok"] is True,
   str(doc["guards"]["A106-14"]["value"]))
ck("完备数据题数 192/144/96/48",
   {k: v2["n_q"] for k, v2 in dd["rows"].items()} == {"16": 192, "20": 144, "24": 96, "28": 48},
   str({k: v2["n_q"] for k, v2 in dd["rows"].items()}))

print()
print("=" * 70)
print(f"★ 单元测试 {'全部通过' if not FAILS else '有 FAIL：' + str(FAILS)}")
print("=" * 70)
sys.exit(0 if not FAILS else 1)