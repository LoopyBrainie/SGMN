#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`DIR-129` 的 §9 **探索性**读数（`exploratory`、**非判据**）：临界剂量的余量。

问：`dose=350` 上 `frac=0`（`n_low=0`）离"被推翻"有多远？
用已存字段 `n_low`（该结构里 `û_s < 1` 的层数）与 `m_tot`（有质量的层数）看。
★ 若 350 上 `n_low` 恒为 0 而 300 上 `n_low` 明显 >0，则"350 是勉强越过"或"300 是勉强未越过"
  可由两端的 `n_low/m_tot` 比例读出，从而判断缩 `n_cells` 是否有可能造出 `frac>0 @ 350`
  （`P129-1` 的可行性）。
"""
import os
import sys
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mech129 as M


def main():
    rows = M.load_readings()
    print("【exploratory · 非判据】各剂量上 n_low / m_tot 的分布（按题集×结构的 15 条）")
    hdr = f"{'dose':>5} {'#结构':>5} {'n_low 取值':>22} {'m_tot 取值':>16} {'frac 最小':>12} {'frac 最大':>12}"
    print(hdr)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["dose"]].append(r)
    for d in sorted(by):
        rs = by[d]
        nl = sorted({r["n_low"] for r in rs})
        mt = sorted({r["m_tot"] for r in rs})
        fr = [r["frac"] for r in rs]
        print(f"{d:>5} {len(rs):>5} {str(nl):>22} {str(mt):>16} {min(fr):>12.6f} {max(fr):>12.6f}")
    print()
    print("【exploratory】临界两侧逐结构：")
    for d in (300, 350, 375):
        rs = sorted([r for r in rows if r["dose"] == d], key=lambda r: tuple(r["sub"]))
        print(f"  dose={d}: " + ", ".join(f"{tuple(r['sub'])}{{n_low={r['n_low']},m_tot={r['m_tot']},frac={r['frac']:.4f}}}"
                                         for r in rs[:4]) + (" ..." if len(rs) > 4 else ""))
        print(f"           n_low 全为 {sorted({r['n_low'] for r in rs})}，frac 全为 0 ? {all(r['frac']==0.0 for r in rs)}")


if __name__ == "__main__":
    main()
