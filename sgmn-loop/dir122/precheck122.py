"""`precheck122.py` —— `DIR-122` 的**开卡前结构核**（本地纯 Python，无 torch / 无 GPU）。

★ 纪律（`handoff` Cycle 233 新增）：**开卡前必须先跑结构核**，且它**只打印
  ① 结构事实 ② 已发表常数** —— **不打印任何新判据观测值**（否则阈值可能被结果倒推）。

★★ 本文件**不自己实现重建**：全部 import 自 `mech122`（唯一一套逻辑）。
   因此本核同时是 `mech122` 重建路径的验证：它在**已发表常数**上必须逐位对齐。

== 本核要回答的三个结构问题 ==
`S1` **能否脱离 tokenizer 重建** —— `facts`/`partitions`/`cand_val` 只由固定 `random.Random` 驱动。
`S2` **重建是否逐项对齐**（不只是多重集对齐）—— 用**已发表**的顺序敏感常数
     逐 seed 逃逸数 `[0,3,18,0,2,9,5,0,5]`、合计 `42`（`D-152`）+ 主读物 `n_seen_hist`（`D-151`）。
`S3` **c2 的三个题集是什么** —— 同一批 16 key、只换干扰项抽样；与训练 seed 无关（配对）。
`S4` **独立单位** —— `48 项 = 16 key × 3 draw` ⇒ key 才是单位。

跑法：本地 `/usr/bin/python3 precheck122.py`；trainer 上亦可。
"""
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)                                    # 所有相对路径相对仓库根
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mech122 as M  # noqa: E402

# ---- 已发表常数（护栏用；来源见 docstring）----
PUB_MAIN_NSH = {"0": 2, "1": 23, "2": 20, "3": 3}      # D-151
PUB_ESCAPE = [0, 3, 18, 0, 2, 9, 5, 0, 5]              # D-152（顺序敏感）
PUB_ESCAPE_TOTAL = 42                                  # D-152
PUB_D_ROW = 58.28571428571427                          # D-152（主题集）

CELLS = "dir119/out/cells/COLOCATED_600_*.json"


def rec(tag, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}  {msg}", flush=True)
    return bool(ok)


def main():
    ok = True

    # ---------- S0 常量来源核对 ----------
    src90 = open("dir90/R1_task90.py", encoding="utf-8").read()
    src114 = open("dir114/R1_task114.py", encoding="utf-8").read()
    ok &= rec("S0-a", "N_CAND = 4" in src90, "D90 的 N_CAND = 4（源文件核对）")
    ok &= rec("S0-b", "N_DRAWS = 3" in src90, "D90 的 N_DRAWS = 3")
    ok &= rec("S0-c", "N_VAL = 16" in src114 and "N_FACT = 2 * N_VAL" in src114,
              "dir114 的 N_VAL=16 / N_FACT=32")
    ok &= rec("S0-d", "def build_facts_pairs(tok, seed=9091)" in src114,
              "build_facts_pairs 默认 seed = 9091")
    ok &= rec("S0-e", "def partitions(facts, legacy_seed=777, shuffle_seed=777)" in src114,
              "partitions 默认 shuffle_seed = 777")
    ok &= rec("S0-f", "random.Random(rng_seed)" in src114,
              "make_items_with_vals 用 random.Random(rng_seed)")

    # ---------- S1 无 tokenizer 重建 ----------
    facts, tr, ev, tv, items, ns = M.build_all()
    ok &= rec("S1-a", len(facts) == M.N_FACT, f"facts = {len(facts)} 条（须 {M.N_FACT}）")
    ok &= rec("S1-b", len(tv) == 8, f"COLOCATED 训练侧值数 = {len(tv)}（须 8）")
    ok &= rec("S1-c", len(ev) == 16, f"COLOCATED 留出侧 key 数 = {len(ev)}（须 16）")
    ok &= rec("S1-d", len(set(facts[i]["val"] for i in ev)) == 8,
              f"留出侧值数 = {len(set(facts[i]['val'] for i in ev))}（须 8）")

    # ---------- S2 逐项对齐（用已发表常数）----------
    ok &= rec("S2-a", len(items[M.MAIN_RNG]) == 48,
              f"rng={M.MAIN_RNG} 项数 = {len(items[M.MAIN_RNG])}（= 16 key × {M.N_DRAWS}）")
    h_main = M.hist_of(ns[M.MAIN_RNG])
    ok &= rec("S2-b", h_main == PUB_MAIN_NSH,
              f"rng={M.MAIN_RNG} 的 n_seen_hist = {h_main}（须 == 已发表 {PUB_MAIN_NSH}，D-151）")

    cells = sorted(glob.glob(CELLS))
    ok &= rec("S2-c", len(cells) == 9, f"cell 文件 = {len(cells)}（须 9）")
    picks_main = [json.load(open(f, encoding="utf-8"))["mech_by_mult"]["10000"]["pick_argmin"]
                  for f in cells]
    esc = [int(sum(M.escape_mask(items[M.MAIN_RNG], ns[M.MAIN_RNG], p, tv))) for p in picks_main]
    ok &= rec("S2-d", esc == PUB_ESCAPE,
              f"重建后逐 seed 逃逸数 = {esc}（须 == 已发表 {PUB_ESCAPE}，D-152）★ 顺序敏感")
    ok &= rec("S2-e", sum(esc) == PUB_ESCAPE_TOTAL,
              f"逃逸合计 = {sum(esc)}（须 == 已发表 {PUB_ESCAPE_TOTAL}）")

    # ---------- S3 c2 三个题集的结构 ----------
    print("\n--- S3 c2 题集结构（只报结构与已发表直方图，不报新读数）---")
    d0 = json.load(open(cells[0], encoding="utf-8"))
    for r in M.C2_RNGS:
        h = M.hist_of(ns[r])
        pub = M.norm_hist(d0["c2"]["10000"][str(r)]["n_seen_hist"])
        ok &= rec(f"S3-{r}a", h == pub, f"rng={r} 的 n_seen_hist = {h}（须 == 存档 c2 {pub}）")
        elig = sum(1 for v in ns[r] if v >= 1)
        print(f"      rng={r}: 项数 {len(items[r])}、可逃逸项 = {elig}、key 数 = {len(ev)}")
        same = all(a["cand_val"] == b["cand_val"]
                   for a, b in zip(items[M.MAIN_RNG], items[r]))
        ok &= rec(f"S3-{r}b", not same, f"rng={r} 的候选值组合与主读物（{M.MAIN_RNG}）不同")

    hists = []
    for f in cells:
        d = json.load(open(f, encoding="utf-8"))
        for r in M.C2_RNGS:
            hists.append(json.dumps(d["c2"]["10000"][str(r)]["n_seen_hist"], sort_keys=True))
    per_r = [sorted(set(hists[i::len(M.C2_RNGS)])) for i in range(len(M.C2_RNGS))]
    ok &= rec("S3-d", all(len(s) == 1 for s in per_r),
              "同一 rng 的 n_seen_hist 在 9 个 seed 上完全相同 ⇒ 题集与训练 seed 无关（配对设计）")

    # ---------- S4 key 聚类 ----------
    per_key = {}
    for i, it in enumerate(items[M.MAIN_RNG]):
        per_key.setdefault(it["key_idx"], []).append(i)
    ok &= rec("S4-a", len(per_key) == 16 and all(len(v) == M.N_DRAWS for v in per_key.values()),
              f"每 key 的项数 = {sorted(set(len(v) for v in per_key.values()))}、key 数 = {len(per_key)}"
              f" ⇒ ★ 48 项不是独立单位，key 才是（16 个独立单位/qset）")

    # ---------- S5 已发表 D_row 回归（主题集）----------
    E_main = [[int(sum(M.escape_mask(items[M.MAIN_RNG], ns[M.MAIN_RNG], p, tv)))]
              for p in picks_main]
    dr = M.d_row_per_qset(E_main)[0]
    ok &= rec("S5-a", abs(dr - PUB_D_ROW) < 5e-9,
              f"主题集 D_row = {dr!r}（须 == 已发表 {PUB_D_ROW!r}，D-152）")

    print(f"\n[precheck122] {'全部 PASS' if ok else '★ 有 FAIL'}")
    print("[precheck122] ★ 本核未计算、未打印 556/557/558 上的任何逃逸读数（判据观测值保持未见）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())