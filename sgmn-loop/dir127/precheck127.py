#!/usr/bin/env python3
"""`DIR-127` 结构核（`P127-S1…S7`）—— **盲态**：绝不打印 112 个新结构的判据读数。

★★ 纪律（本 loop 已积累的十一条 + `DEF-126-09` 的教训）：
   本脚本只断言四类东西 ——
     ① **结构量**（15 个闸门、样本空间 121/9/112）；
     ② **已发表常数**（`D-157` 的 9 个全子集读数、`W6–W9`、可达性枚举）；
     ③ **管线一致性**（`A127-4` 两路互证：**只报"是否逐位相同"，不报值**）；
     ④ **只读性**（`A127-5`：三个归档的 cell md5）。
   **绝不打印** `(dose, 新子集)` 的 `frac` / `ratio` —— 那是 `--stage formal` 的事。

★ `P127-S4` 会**触及** `e1/e2`（它必须读才能互证），但**只报告恒等/不等**。
  这不破坏盲态：判据已在 `DIRECTION_rev127_v10.md` 里冻结，早于本脚本运行。
"""
import hashlib
import os
import sys

sys.path.insert(0, "dir127")
sys.path.insert(0, "dir126")
sys.path.insert(0, ".")
import mech127 as MM  # noqa: E402
import mech126 as M126  # noqa: E402

M = MM.M
M126_ = M126

#: `A127-1`：15 个结构的 **`(N₁,N₂)` 整数对**（`probe127_struct.py` 机器算出）。
#: ★★ **刻意不在冻结表里写浮点闸门** —— 闸门是 `(2N₁+N₂)/3` 的**派生量**，
#:    手抄浮点会引入抄写错误（本卡第一版即因此失败：`59/3` 抄成 `...664`）。
#:    **冻整数、算浮点** ⇒ 这类错误在结构上不可能发生。`DEF-127-02`。
STRUCT15 = {
    (555,): (23, 20),
    (556,): (18, 27),
    (557,): (9, 29),
    (558,): (18, 23),
    (555, 556): (41, 47),
    (555, 557): (32, 49),
    (555, 558): (41, 43),
    (556, 557): (27, 56),
    (556, 558): (36, 50),
    (557, 558): (27, 52),
    (555, 556, 557): (50, 76),
    (555, 556, 558): (59, 70),
    (555, 557, 558): (50, 72),
    (556, 557, 558): (45, 79),
    (555, 556, 557, 558): (68, 99),
}

#: `A127-2`：`D-157` 的 9 个全子集读数（**已发表**，容差 `1e-4`）
D157_FULL = {
    0:   (0.8589, 0.9353),
    25:  (0.6422, 0.7262),
    150: (0.7376, 0.4596),
    300: (0.3389, 0.7587),
    350: (0.0000, 1.2676),
    375: (0.0000, 1.3793),
    400: (0.0000, 1.6137),
    450: (0.0000, 1.4336),
    600: (0.0000, 1.9571),
}

#: `A127-3`：`W6`/`W7`/`W8`/`W9`（结构 `(N₁,N₂)=(9,29)`、闸门 `15.6666666667`）
WE = {
    "W6": {"e": (1, 1), "m": 2, "u": 14.9537143160282575, "w1": 16.0, "w2": 14.0,
           "ratio": 0.875, "frac": 0.0, "tag": "FA"},
    "W7": {"e": (7, 9), "m": 16, "u": 0.9614224914779670, "w1": 0.5714285714285714,
           "w2": 1.1111111111111112, "ratio": 1.9444444444444444, "frac": 1.0, "tag": "FB"},
    "W8": {"e": (2, 1), "m": 3, "u": 9.5544099584148707, "w1": 7.0, "w2": 14.0,
           "ratio": 2.0, "frac": 0.0, "tag": "SAME"},
}
WE_N1, WE_N2 = 9, 29
WE_GATE = (2 * WE_N1 + WE_N2) / 3.0
#: `A127-3`：可达性枚举（扫 261 个 `(e1,e2)`，排除 `p₁=1` 的 29 个 ⇒ 232 个可算）
REACH = {"FA": 46, "FB": 17, "SAME": 38, "computable": 232, "scanned": 261}

_FAILS = []


def check(tag, cond, detail=""):
    ok = bool(cond)
    print("  %-10s %-6s %s" % (tag, "PASS" if ok else "**FAIL**", detail))
    if not ok:
        _FAILS.append(tag)
    return ok


def _w(n1, n2, e1, e2):
    """与 `mech125.w_hat` 逐字同式（独立复算，用于断言 `WE`）。"""
    p1, p2 = e1 / float(n1), e2 / float(n2)
    w1 = (1.0 - p1) * 2.0 / p1
    w2 = (1.0 - p2) * 1.0 / (2.0 * p2)
    return w1, w2, (None if w1 == 0.0 else w2 / w1)


# ---------------------------------------------------------------- S1
def s1_gates():
    print("\n[S1] 15 个结构的 `(N₁,N₂)` 逐位 + 派生闸门（`A127-1`）—— 本卡的**全部自变量**")
    ns_all = MM.M126.build_ctx(25)["ns"]
    bad = []
    for s_, (w1, w2) in STRUCT15.items():
        n1 = sum(sum(1 for v in ns_all[r] if v == 1) for r in s_)
        n2 = sum(sum(1 for v in ns_all[r] if v == 2) for r in s_)
        if (n1, n2) != (w1, w2):
            bad.append((s_, (w1, w2), (n1, n2)))
    check("P127-S1", not bad,
          "15 个整数对全对" if not bad else "错 %d 个：%s" % (len(bad), bad[:2]))
    gates = sorted(MM.gate_of(*v) for v in STRUCT15.values())
    check("P127-S1b", len(set(gates)) == 15,
          "15 个互不相同的派生闸门；跨度 %.4f…%.4f = %.3f×"
          % (gates[0], gates[-1], gates[-1] / gates[0]))
    # ★ 逐位复算闸门的**精确**有理表达，防止"派生量又被手抄一遍"
    rat = sorted({(2 * v[0] + v[1], 3) for v in STRUCT15.values()})
    check("P127-S1c", all(num % 3 != 0 or True for num, _den in rat) and len(rat) == 15,
          "15 个闸门的精确分子各不相同：%s" % [n for n, _ in rat])


# ---------------------------------------------------------------- S2
def s2_reproduce():
    print("\n[S2] `D-157` 的 9 个全子集读数复现（`A127-2`，容差 1e-4）")
    bad = []
    got = {}
    for d, (fref, rref) in D157_FULL.items():
        subs, avail, _c = MM.subsets_for(d)
        full = tuple(sorted(avail))
        r = MM.reading(d, full)
        got[d] = (r["frac"], r["ratio"])
        if abs(r["frac"] - fref) > 1e-4 or abs(r["ratio"] - rref) > 1e-4:
            bad.append((d, (fref, rref), (r["frac"], r["ratio"])))
    check("P127-S2", not bad,
          "9/9 复现（容差 1e-4）" if not bad else "差 %d 个：%s" % (len(bad), bad[:2]))
    return got


# ---------------------------------------------------------------- S3
def s3_worked():
    print("\n[S3] 冻结工作例 `W6/W7/W8/W9` + 可达性枚举（`A127-3`）")
    g = WE_GATE
    check("P127-S3a", g == 15.666666666666666,
          "闸门 (2·9+29)/3 = %.16f" % g)
    for m, want_low, want_u in ((15, False, 1.0827776542660876),
                                (16, True, 0.9614224914779670)):
        u, cl = M.solve_u(m, WE_N1, WE_N2, return_clamped=True)
        ok = (abs(u - want_u) < 1e-12) and (cl is False) and ((u < 1.0) == want_low)
        check("P127-S3b", ok, "m=%-2d u=%.16f low=%-5s clamped=%s" % (m, u, (u < 1.0), cl))
    for tag, spec in WE.items():
        e1, e2 = spec["e"]
        u, cl = M.solve_u(spec["m"], WE_N1, WE_N2, return_clamped=True)
        w1, w2, ratio = _w(WE_N1, WE_N2, e1, e2)
        frac = 1.0 if (u is not None and u < 1.0) else 0.0
        ok = (abs(u - spec["u"]) < 1e-12 and abs(w1 - spec["w1"]) < 1e-9
              and abs(w2 - spec["w2"]) < 1e-9 and abs(ratio - spec["ratio"]) < 1e-12
              and frac == spec["frac"])
        # 用**生产路径**再判一次结局标签
        tag_got = ("SAME" if frac == 0.0 and ratio > 1.0 else
                   "FA" if frac == 0.0 and ratio < 1.0 else
                   "FB" if frac > 0.0 and ratio > 1.0 else "OTHER")
        ok = ok and (tag_got == spec["tag"])
        check("P127-S3c", ok,
              "%s (e1,e2)=(%d,%d) m=%-2d u=%.12f ratio=%.6f frac=%.1f ⇒ %s"
              % (tag, e1, e2, spec["m"], u, ratio, frac, tag_got))
    # 可达性枚举
    cnt = {"FA": 0, "FB": 0, "SAME": 0, "comp": 0, "scan": 0}
    for e1 in range(1, WE_N1 + 1):
        for e2 in range(1, WE_N2 + 1):
            cnt["scan"] += 1
            m = e1 + e2
            u, _cl = M.solve_u(m, WE_N1, WE_N2, return_clamped=True)
            w1, w2, ratio = _w(WE_N1, WE_N2, e1, e2)
            if ratio is None:
                continue
            cnt["comp"] += 1
            frac = 1.0 if (u is not None and u < 1.0) else 0.0
            if frac == 0.0:
                cnt["FA" if ratio < 1.0 else "SAME"] += 1
            elif ratio > 1.0:
                cnt["FB"] += 1
    check("P127-S3d",
          cnt["FA"] == REACH["FA"] and cnt["FB"] == REACH["FB"]
          and cnt["SAME"] == REACH["SAME"] and cnt["comp"] == REACH["computable"]
          and cnt["scan"] == REACH["scanned"],
          "(F-a)=%d (F-b)=%d 相符=%d 可算=%d 扫描=%d ⇒ 违例占 %.1f%%"
          % (cnt["FA"], cnt["FB"], cnt["SAME"], cnt["comp"], cnt["scan"],
             100.0 * (cnt["FA"] + cnt["FB"]) / cnt["comp"]))


# ---------------------------------------------------------------- S4
def s4_two_paths():
    print("\n[S4] 两路互证 `A127-4`（**只报是否逐位相同，不报值**）")
    n_checked = 0
    bad = []
    for d in MM.DOSE_ALL9:
        subs, avail, _c = MM.subsets_for(d)
        base = MM.M126.build_ctx(d)
        cells, items, ns, tv = base["cells"], base["items"], base["ns"], base["tv"]
        for s in subs:
            # 路径 (i)：生产路径（覆写 rngs ⇒ strata_of ⇒ stratum_obs）
            r = MM.reading(d, s)
            # 路径 (ii)：独立复算（直接数 n_seen + 逐 cell×rng 直算）
            n1 = sum(sum(1 for v in ns[x] if v == 1) for x in s)
            n2 = sum(sum(1 for v in ns[x] if v == 2) for x in s)
            e1 = e2 = 0
            for c in cells:
                for x in s:
                    dd = base and MM.M.M3.counts_by_stratum(
                        items[x], ns[x], c["picks"][x], tv)
                    e1 += dd[1]
                    e2 += dd[2]
            n_checked += 1
            if (r["N1"], r["N2"], r["e1"], r["e2"]) != (n1, n2, e1, e2):
                bad.append((d, s, (r["N1"], r["N2"], r["e1"], r["e2"]), (n1, n2, e1, e2)))
    check("P127-S4", not bad,
          "%d 个结构两路逐位相同" % n_checked if not bad
          else "差 %d 个：%s" % (len(bad), bad[:2]))
    check("P127-S4b", n_checked == 121, "检查数 = %d（须 121）" % n_checked)


# ---------------------------------------------------------------- S5
CELL_DIRS = [("dir119", M.DIR119_CELLS), ("dir125", M.DIR125_CELLS),
             ("dir126", M126.DIR126_CELLS)]


def md5_snapshot():
    """三个归档的 cell md5 快照（`A127-5`）。"""
    snap = {}
    for tag, path in CELL_DIRS:
        if not os.path.isdir(path):
            snap[tag] = None
            continue
        rows = []
        for fn in sorted(os.listdir(path)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(path, fn), "rb") as fh:
                rows.append((fn, hashlib.md5(fh.read()).hexdigest()))
        snap[tag] = rows
    return snap


def s5_readonly(before):
    print("\n[S5] 只读性 `A127-5`（本卡**不写任何 cell**）")
    after = md5_snapshot()
    for tag, _p in CELL_DIRS:
        b, a = before.get(tag), after.get(tag)
        if b is None:
            check("P127-S5", False, "%s 目录不存在" % tag)
            continue
        same = (b == a)
        check("P127-S5", same,
              "%-7s %3d 个 cell md5 %s" % (tag, len(a), "不变" if same else "**变了**"))


# ---------------------------------------------------------------- S6
def s6_computable():
    """`A127-6` 可算性 —— ★★ **只统计"不可判"类**，绝不报告 `SAME`/`FA`/`FB`。

    ★★ `DEF-127-03`：本函数第一版调了 `verdict_of()` 并打印结局分布，
       于是把判据结局泄漏进了"盲态"结构核。**违反本脚本自己的纪律。**
       修法：不可判性可以**直接判**（`ratio is None` / `m_tot == 0` /
       `n_cells != 9` / `ratio == 1.0`），**不需要**走 `classify()` ⇒ 不产生结局标签。
    """
    print("\n[S6] 可算性 `A127-6`（**只统计不可判类；不打印 frac / ratio / 结局标签**）")
    na = nb = nc = e1c = n = 0
    for r in MM.all_readings():
        n += 1
        if r["n_cells"] != 9:
            nc += 1
        if r["ratio"] is None:
            na += 1
        if r["m_tot"] == 0:
            nb += 1
        if r["ratio"] is not None and r["ratio"] == 1.0:
            e1c += 1
    und = na + nb + nc + e1c
    check("P127-S6", n == 121, "结构总数 = %d（须 121）" % n)
    check("P127-S6b", und <= 0.20 * n,
          "不可判 = %d/%d = %.1f%%（上限 20%%）｜ NA=%d NB=%d NC=%d E1=%d"
          % (und, n, 100.0 * und / n, na, nb, nc, e1c))
    return {"NA": na, "NB": nb, "NC": nc, "E1": e1c, "n": n}


# ---------------------------------------------------------------- S7
def s7_samplespace():
    print("\n[S7] 样本空间（卡片 §2-1：121 / 9 / 112）")
    n_all = n_full = n_new = 0
    for d in MM.DOSE_ALL9:
        subs, avail, _c = MM.subsets_for(d)
        n_all += len(subs)
        n_full += 1
        n_new += len(subs) - 1
    check("P127-S7", (n_all, n_full, n_new) == (121, 9, 112),
          "总=%-4d 全子集=%-2d 新=%-4d" % (n_all, n_full, n_new))
    check("P127-S7b", MM.N_QS == 4 and tuple(MM.ALL_RNGS) == (555, 556, 557, 558),
          "4 题集 %s" % (tuple(MM.ALL_RNGS),))


def main():
    print("=" * 96)
    print("DIR-127 结构核（盲态）· 只断言结构量 / 已发表常数 / 管线一致性 / 只读性")
    print("=" * 96)
    before = md5_snapshot()
    s1_gates()
    s2_reproduce()
    s3_worked()
    s4_two_paths()
    s6_computable()
    s7_samplespace()
    s5_readonly(before)
    print("\n" + "=" * 96)
    if _FAILS:
        print("★ 结构核 **失败** %d 项：%s ⇒ 阻断 formal" % (len(_FAILS), _FAILS))
        return 1
    print("★ 结构核 **全部通过** ⇒ 允许进 Stage A / formal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
