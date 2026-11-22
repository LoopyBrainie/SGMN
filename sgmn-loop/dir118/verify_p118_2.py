#!/usr/bin/env python
"""`P118-2` 交付前的数字复核（纪律 ⑱ / ㊲：报告里每个定量断言都要被重算，且检查必须**能**报 FAIL）。

两件事：
① 独立重算所有算术（`Σm_v²/N`、比例、功效的 `z` 与 3se 所需 `n`），与 `METHOD-P118-2.md` 的断言逐条对照；
② ★ **跨文件不变量**（真正会 FAIL 的那条）：`SPLIT` 的「未见值个数 == 0」必须与 `P118-1` 的
   `correct_is_seen == 48/48` **等价**；`COLOCATED` 的「未见值 8 个」必须与 `correct_is_seen == 0/48` 相容。
   若哪天划分变了而这两条不一起变 ⇒ 本卡的结构论证就塌了，必须报红。
"""
import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
FAILS = []


def chk(tag, got, want, tol=1e-9):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: 实测={got!r} 报告={want!r}")
    if not ok:
        FAILS.append(tag)


p2 = json.loads((OUT / "premise_p118_2.json").read_text(encoding="utf-8"))
p1 = json.loads((OUT / "premise_p118_1.json").read_text(encoding="utf-8"))

print("=" * 76)
print("§1 装置值结构（独立重算 Σm_v²/N）")
print("=" * 76)
n_facts, n_vals = p2["n_facts"], p2["n_distinct_vals"]
chk("facts", n_facts, 32)
chk("distinct_vals", n_vals, 16)
mh = {int(k): v for k, v in p2["mult_hist"].items()}
chk("重数分布 m_v 的直方图", mh, {2: 16})
assert n_vals * 2 == n_facts, "每个值恰好两次 ⇒ 值总数须等于事实数"
sum_sq = sum(int(m) ** 2 * cnt for m, cnt in mh.items())
e_rand = sum_sq / n_facts
chk("E_random 分子 Σm_v²", sum_sq, 64)
chk("E_random(N_hit^v)", e_rand, 2.0)
chk("E_random 比例", e_rand / n_facts, 0.0625)
chk("常数 argmax 天花板 max m_v", max(mh), 2)
chk("★ 不得照抄 D-127 的 1.3125", e_rand != 1.3125, True)

print()
print("=" * 76)
print("§2 逐臂未见值（结构核）")
print("=" * 76)
for arm, want_tv, want_unseen, want_all, want_ge1 in (
        ("SPLIT", 16, 0, 0, 0),
        ("COLOCATED", 8, 8, 16, 16)):
    a = p2["arms"][arm]
    print(f"\n-- {arm} --")
    chk("训练覆盖的不同值", a["n_train_vals"], want_tv)
    chk("未见值个数", a["n_unseen_vals"], want_unseen)
    chk("未见值个数 == 装置字段", a["n_unseen_vals"], a["n_unseen_dev_field"])
    chk("可全未见干扰的 item 数", a["items_can_have_all_unseen_distractors"], want_all)
    chk("可≥1 未见干扰的 item 数", a["items_can_have_ge1_unseen_distractor"], want_ge1)
    chk("pool 大小 min", a["pool_size_min"], 15)
    chk("pool 大小 max", a["pool_size_max"], 15)
    chk("评测 item 数", a["n_ev_items"], 16)

print()
print("=" * 76)
print("★ 跨文件不变量（与 P118-1 的 correct_is_seen 必须相容）")
print("=" * 76)
seen_frac = {"SPLIT": p1["SPLIT"]["correct_is_seen"], "COLOCATED": p1["COLOCATED"]["correct_is_seen"]}
for arm in ("SPLIT", "COLOCATED"):
    nu = p2["arms"][arm]["n_unseen_vals"]
    cs = seen_frac[arm]
    if nu == 0:
        want_cs = 48          # 无未见值 ⇒ 正确值必然见过 ⇒ 48/48
    else:
        want_cs = 0           # 正确值取自未见值层 ⇒ 0/48
    chk(f"{arm}: 未见值={nu} ⇒ correct_is_seen 应为 {want_cs}/48", cs, want_cs)

print()
print("=" * 76)
print("§5 功效（独立重算 z 与 3se 所需 n）")
print("=" * 76)
p0, p1_, n_item = 0.25, 0.40, 48
z = (p1_ - p0) / math.sqrt(p1_ * (1 - p1_) / n_item)
chk("n=48 的 z", round(z, 2), 2.12, 0.005)
chk("z < 3 ⇒ 未达 3se", z < 3.0, True)
n3 = next(n for n in range(2, 5000) if (p1_ - p0) / math.sqrt(p1_ * (1 - p1_) / n) >= 3.0)
chk("达 3se 所需 n", n3, 96)

print()
print("=" * 76)
print(f"{'全部 PASS' if not FAILS else 'FAIL 项：' + str(FAILS)}")
print("=" * 76)
if FAILS:
    raise SystemExit(1)