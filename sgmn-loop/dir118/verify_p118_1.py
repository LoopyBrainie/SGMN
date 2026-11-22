#!/usr/bin/env python
"""`P118-1` 交付前的**数字复核**（纪律 ⑱：判决/报告里每个定量断言，落盘前必须由脚本重算并打印对照）。

把 `revolve/DIR-118/DIAG-P118-1.md` 里出现的每个数字重新算一遍，逐条打印 PASS/FAIL。
只在**存档数据**上运算，不加载模型、不跑实验。
"""
import json
import pathlib
import itertools

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
CELLS = OUT / "cells"
N_ITEM = 48
SEEDS = (20271126, 20271203, 20271210, 20260501, 20260502, 20260503,
         20260504, 20260505, 20260506)

FAILS = []


def chk(tag, got, want, tol=1e-9):
    if isinstance(want, float):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: 实测={got!r} 报告={want!r}")
    if not ok:
        FAILS.append(tag)


def load(arm, dose, seed):
    f = CELLS / f"{arm}_{int(dose)}_{int(seed)}.json"
    return json.loads(f.read_text(encoding="utf-8"))


def okvec(d, mult):
    return [int(x) for x in d["ok_by_mult"][f"{mult:g}"]]


print("=" * 74)
print("§1/§2 前提核的数字（来自 probe118_p1_premise.py 的落盘读数）")
print("=" * 74)
pre = json.loads((OUT / "premise_p118_1.json").read_text(encoding="utf-8"))
pos = {int(k): v for k, v in pre["COLOCATED"]["pos_hist"].items()}
chk("正确位置直方图", pos, {0: 15, 1: 16, 2: 6, 3: 11})
accs = [pos.get(j, 0) / N_ITEM for j in range(4)]
print(f"   固定位置策略准确率 = {[f'{a*100:.2f}%' for a in accs]}")
chk("固定位置下限（j=2）", round(min(accs), 6), 0.125, 1e-6)
chk("P2 COLOCATED correct_is_seen", pre["COLOCATED"]["correct_is_seen"], 0)
chk("P3 SPLIT correct_is_seen", pre["SPLIT"]["correct_is_seen"], 48)
sc = {int(k): v for k, v in pre["COLOCATED"]["n_seen_cand_hist"].items()}
chk("COLOCATED 每题见过值个数直方图", sc, {0: 2, 1: 23, 2: 20, 3: 3})
chk("SPLIT 每题见过值个数直方图(=DEF-118-01 {4:48})",
    {int(k): v for k, v in pre["SPLIT"]["n_seen_cand_hist"].items()}, {4: 48})

print()
print("=" * 74)
print("§3 跨 seed 结构（诊断 A）")
print("=" * 74)
EXPECT = {
    ("COLOCATED", 1): (23, [2, 1, 1, 0, 1, 4, 1, 3, 10], 32, 0.087),
    ("COLOCATED", 10000): (20, [1, 1, 3, 0, 2, 7, 1, 0, 5], 34, 0.076),
    ("SPLIT", 1): (115, [13, 15, 11, 10, 20, 15, 10, 8, 13], 5, 0.185),
    ("SPLIT", 10000): (113, [13, 15, 13, 11, 6, 15, 16, 11, 13], 3, 0.142),
}
for (arm, m), (tot_w, seeds_w, never_w, jac_w) in EXPECT.items():
    M = {s: okvec(load(arm, 600, s), m) for s in SEEDS}
    tot = sum(sum(M[s]) for s in SEEDS)
    per = [sum(M[s][i] for s in SEEDS) for i in range(N_ITEM)]
    sizes = [sum(M[s]) for s in SEEDS]
    never = sum(1 for v in per if v == 0)
    all9 = sum(1 for v in per if v == len(SEEDS))
    cs = [set(i for i in range(N_ITEM) if M[s][i]) for s in SEEDS]
    js = [len(a & b) / len(a | b) if (a | b) else 0.0
          for a, b in itertools.combinations(cs, 2)]
    jm = sum(js) / len(js)
    print(f"\n-- {arm} mult={m:g} --")
    chk("总正确 token", tot, tot_w)
    chk("各 seed 正确数", sizes, seeds_w)
    chk("从不正确的 item 数", never, never_w)
    chk("跨 seed Jaccard 均值", round(jm, 3), jac_w, 1e-3)
    chk("9/9 全对 item 数（必须为 0）", all9, 0)

print()
print("=" * 74)
print("§4 未训练 → 训练后的落差（改写自 stage_formal.json）")
print("=" * 74)
fm = json.loads((HERE.parent / "dir118" / "out" / "stage_formal.json").read_text(encoding="utf-8"))
cells = fm["cells"]
for arm, unt_w, tr_w, d_w, t_w in (
        ("COLOCATED", 37.5, 5.32, -15.44, -15.21),
        ("SPLIT", 34.72, 26.6, -3.89, -3.08)):
    u = [cells[f"{arm}|0|{s}"]["acc_by_mult"]["1"] * 48 for s in SEEDS]
    tr = [cells[f"{arm}|600|{s}"]["acc_by_mult"]["1"] * 48 for s in SEEDS]
    um, tm = sum(u) / len(u), sum(tr) / len(tr)
    dif = [b - a for a, b in zip(u, tr)]
    dm = sum(dif) / len(dif)
    sd = (sum((x - dm) ** 2 for x in dif) / (len(dif) - 1)) ** 0.5
    tv = dm / (sd / len(dif) ** 0.5)
    print(f"\n-- {arm} --")
    chk("未训练 acc%", round(um / 48 * 100, 2), unt_w, 0.02)
    chk("训练后 acc%", round(tm / 48 * 100, 2), tr_w, 0.02)
    chk("落差 Δ(1) 项", round(dm, 2), d_w, 0.02)
    chk("落差 t", round(tv, 2), t_w, 0.02)
    chk("★ 观测值 < 固定位置下限 12.5%？",
        (tm / 48 * 100) < 12.5, arm == "COLOCATED")

print()
print("=" * 74)
print(f"独立基准：25% 正确率下的期望 Jaccard = {0.25*0.25/(2*0.25-0.25*0.25):.3f}")
print(f"{'全部 PASS' if not FAILS else 'FAIL 项：' + str(FAILS)}")
print("=" * 74)
if FAILS:
    raise SystemExit(1)