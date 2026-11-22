#!/usr/bin/env python3
"""`DIR-127` 结构核（**盲态**）：只打结构量 —— `N1/N2/闸门/子集可用性`。

★★ 纪律（`P126-1` 派生的开卡纪律 + 本 loop 已积累的十一条）：
   本脚本**绝不读** `e1/e2`、**绝不计算** `frac` / `ratio` / 任何判据观测值。
   它只回答一个问题：**「固定 dose、换题集子集」能不能给出一条不相交的结构轴？**
   ⇒ 若闸门随子集单调下降且覆盖观测到的 `m_s` 范围，则设计成立；否则整卡不可执行。

★ 为什么必须盲态：若先看到 `(frac,ratio)` 再定判据，判据就成了对数据的复述
   （`DEF-126-09` 的同族错误：备择族没被真正枚举）。
"""
import itertools
import sys

sys.path.insert(0, "dir126")
sys.path.insert(0, ".")
import mech126 as M  # noqa: E402

ALL_RNGS = tuple(M.ALL_RNGS)
DOSE_OLD = tuple(M.DOSE_OLD)
DOSE_NEW2 = tuple(M.DOSE_NEW2)
DOSE_ALL9 = tuple(M.DOSE_ALL9)


def subsets():
    """15 个非空题集子集，按 (k, 字典序)。"""
    out = []
    for k in range(1, len(ALL_RNGS) + 1):
        out.extend(itertools.combinations(ALL_RNGS, k))
    return out


def main():
    facts = M.M122.build_facts()
    tr, ev = M.M122.partitions_colocated(facts)
    tv = {facts[i]["val"] for i in tr}
    items = {r: M.M122.make_items(facts, ev, r) for r in ALL_RNGS}
    ns = {r: M.M122.n_seen_vec(items[r], tv) for r in ALL_RNGS}

    def n_of(r, k):
        return sum(1 for v in ns[r] if v == k)

    print("=" * 92)
    print("STRUCT-1 · 各题集**单独**的结构（与训练无关，只看 n_seen 直方图）")
    print("=" * 92)
    print("%-8s %-6s %-6s %-6s %-10s" % ("rng", "N1", "N2", "n_items", "单题集闸门"))
    for r in ALL_RNGS:
        n1, n2 = n_of(r, 1), n_of(r, 2)
        print("%-8d %-6d %-6d %-6d %-10.4f"
              % (r, n1, n2, len(ns[r]), (2 * n1 + n2) / 3.0))

    print()
    print("=" * 92)
    print("STRUCT-2 · 15 个非空子集的结构轴（★ 这是本卡唯一的自变量）")
    print("=" * 92)
    rows = []
    for s in subsets():
        n1 = sum(n_of(r, 1) for r in s)
        n2 = sum(n_of(r, 2) for r in s)
        gate = (2 * n1 + n2) / 3.0
        rows.append({"sub": s, "k": len(s), "N1": n1, "N2": n2, "gate": gate})
    print("%-4s %-22s %-6s %-6s %-10s" % ("k", "子集", "N1", "N2", "闸门"))
    for x in rows:
        print("%-4d %-22s %-6d %-6d %-10.4f"
              % (x["k"], str(x["sub"]), x["N1"], x["N2"], x["gate"]))
    gates = sorted(x["gate"] for x in rows)
    print()
    print("★ 闸门范围：%.4f … %.4f（跨度 %.3f×）；去重后 %d 个不同值"
          % (gates[0], gates[-1], gates[-1] / gates[0], len(set(gates))))
    print("★ 15 个子集**不独立**（嵌套/重叠）：k=1 有 4 个、k=2 有 6 个、k=3 有 4 个、k=4 有 1 个")

    print()
    print("=" * 92)
    print("STRUCT-3 · 每个剂量的**可用子集数**（只读 picks 的**键**，不读值）")
    print("=" * 92)
    print("%-8s %-10s %-14s %-10s" % ("dose", "可用题集", "可用子集数", "9/9 cell？"))
    total = 0
    for d in DOSE_ALL9:
        try:
            cells = M.load_cells(d)
        except Exception as e:  # noqa: BLE001
            print("%-8d 加载失败: %s" % (d, e))
            continue
        avail = tuple(r for r in ALL_RNGS if all(r in c["picks"] for c in cells))
        ncell = len(cells)
        # 只有全部 qset 都在 avail 里的子集才可用
        nsub = sum(1 for s in subsets() if all(r in avail for r in s))
        total += nsub
        print("%-8d %-10s %-14d %-10s"
              % (d, str(avail), nsub, "是" if ncell == 9 else "否(%d)" % ncell))
    print()
    print("★ 全部 (dose, 子集) 组合总数 = %d" % total)
    print("★★ 判据只能在这 %d 个组合上定义；`dose=0` 只有 1 题集 ⇒ 只有 1 个子集 ⇒ **无子集自由度**" % total)

    print()
    print("=" * 92)
    print("STRUCT-4 · 结构判定（设计是否成立）")
    print("=" * 92)
    print("① 闸门随 |子集| 单调上升？ ", end="")
    byk = {}
    for x in rows:
        byk.setdefault(x["k"], []).append(x["gate"])
    mono = all(max(byk[k]) < min(byk[k + 1]) for k in range(1, 4))
    print("是（各 k 内闸门唯一：%s）" % {k: sorted(set(v)) for k, v in byk.items()} if mono
          else "否 ⇒ 设计不成立")
    print("② 闸门跨度 ≥ 2×？ ", "是（%.3f×）" % (gates[-1] / gates[0]) if gates[-1] / gates[0] >= 2 else "否")
    print("③ 每个子集 N1,N2 都 > 0？ ",
          "是" if all(x["N1"] > 0 and x["N2"] > 0 for x in rows) else "否 ⇒ 有 (N-a)")
    print()
    print("★ 结论：**结构轴与剂量轴正交** —— 固定 dose、换子集，闸门单独从 %.3f 扫到 %.3f"
          % (gates[0], gates[-1]))
    print("        而 dose 只通过 `Σm` 与 `e1/e2` 进入，不进闸门。")
    print("        ⇒ 「让 frac 独立于 dose 变化」这一设计目标**结构上可达**。")


if __name__ == "__main__":
    main()
