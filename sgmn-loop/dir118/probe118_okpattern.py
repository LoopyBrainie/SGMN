#!/usr/bin/env python
"""`P118-1` 诊断 A（**纯存档分析，不加载模型、不新跑实验**）。

问题：`COLOCATED` 训练后 `mult=1` = 2.56/48 (5.3%)，远**低于**随机 25.0%。
两种可能必须先分开：
  (i) **构造/位置工件** —— 若"哪些 item 正确"由 **item 位置**决定（与模型状态无关），
      则**不同 seed**（不同训练出的 W）会得到**同一批**正确 item。
  (ii) **模型状态依赖** —— 若正确 item 随 seed 变化，则读出偏好由**训练出的状态**决定。

`cells` 只存 `ok_by_mult`（48 布尔），**没存所选下标** ⇒ 本脚本只能做 (i)/(ii) 的判别，
**不能**直接给名次直方图（那需要新测量层）。
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


def load(arm, dose, seed):
    f = CELLS / f"{arm}_{int(dose)}_{int(seed)}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def okvec(d, mult):
    o = d["ok_by_mult"][f"{mult:g}"]
    assert len(o) == N_ITEM, (len(o), N_ITEM)
    return [int(x) for x in o]


def analyse(arm, dose, mults):
    print(f"\n{'='*72}\n### {arm}  dose={dose}\n{'='*72}")
    docs = {}
    for s in SEEDS:
        d = load(arm, dose, s)
        if d is None:
            print(f"  缺 ! {arm}|{dose}|{s}")
            return
        docs[s] = d

    for m in mults:
        M = {s: okvec(docs[s], m) for s in SEEDS}
        # 每个 item 在 9 个 seed 里正确的次数
        per_item = [sum(M[s][i] for s in SEEDS) for i in range(N_ITEM)]
        hist = {}
        for v in per_item:
            hist[v] = hist.get(v, 0) + 1
        total = sum(sum(M[s]) for s in SEEDS)
        print(f"\n-- mult={m:g} -- 总正确 token = {total} / {N_ITEM*len(SEEDS)}"
              f" = {total/(N_ITEM*len(SEEDS)):.3f}")
        print(f"   『每 item 在几个 seed 上正确』直方图：{dict(sorted(hist.items()))}")
        all9 = [i for i in range(N_ITEM) if per_item[i] == len(SEEDS)]
        none = [i for i in range(N_ITEM) if per_item[i] == 0]
        print(f"   9/9 全对的 item：{len(all9)} 个 {all9[:12]}")
        print(f"   0/9 全错的 item：{len(none)} 个")

        # 两两 seed 的 Jaccard（正确集合的重合度）—— 若接近 1 ⇒ item 位置决定
        cs = [set(i for i in range(N_ITEM) if M[s][i]) for s in SEEDS]
        js = [len(a & b) / len(a | b) if (a | b) else 0.0
              for a, b in itertools.combinations(cs, 2)]
        print(f"   跨 seed 正确集合 Jaccard：mean={sum(js)/len(js):.3f} "
              f"min={min(js):.3f} max={max(js):.3f}（n={len(js)} 对）")
        sizes = [len(c) for c in cs]
        print(f"   各 seed 正确数：{sizes}")
        # 若所有 seed 的正确集合几乎不相交，Jaccard≈0；若完全相同，=1
        print(f"   期望随机 Jaccard（4/48 口径 ≈ 25%）≈ "
              f"{0.25*0.25/(2*0.25-0.25*0.25):.3f}")

    # 跨 mult（同一 seed 内）：正确集合是否随增益稳定
    print(f"\n-- 跨 mult 稳定性（同 seed 内）--")
    for s in SEEDS[:3]:
        row = []
        for m in mults:
            row.append(sum(okvec(docs[s], m)))
        print(f"   seed {s}: " + " ".join(f"{m:g}:{v}" for m, v in zip(mults, row)))


if __name__ == "__main__":
    MULTS = (1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000)
    for arm in ("COLOCATED", "SPLIT"):
        analyse(arm, 600, (1, 10000))