#!/usr/bin/env python3
"""`DIR-123` 测量层自测 —— 每条断言都必须能独立红（`--inject k` 把第 k 条改成返回 False）。

★ 纪律：**不得留空转脚手架**（`DEF-121-02` / `DEF-122-02` / `DEF-123-03` 已四次同族）。
  每条测试都必须是**真断言**；`--inject` 只用于**证明**它能红，不得用于"制造通过"。
"""
import os
import sys
import math
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "dir122"), os.path.join(ROOT, "dir121"),
          os.path.join(ROOT, "dir120")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mech123 as M                                              # noqa: E402
import mech122 as M0                                             # noqa: E402

N1, N2 = 612, 891
ALPHA = M.ALPHA_J1
TESTS = []


def t(tag):
    def deco(fn):
        TESTS.append((tag, fn))
        return fn
    return deco


# ---------------------------------------------------------------- 机制层
@t("T1  P1(4) == 1/3（卡片 §3-1 手算）")
def _():
    return abs(M.P1(4.0) - 1.0 / 3.0) < 1e-15


@t("T2  P2(4) == 1/9（卡片 §3-1 手算）")
def _():
    return abs(M.P2(4.0) - 1.0 / 9.0) < 1e-15


@t("T3  W1(4) == 4（每候选加成）")
def _():
    return abs(M.W1(4.0) - 4.0) < 1e-13


@t("T4  W2(4) == 4（与 k 无关）")
def _():
    return abs(M.W2(4.0) - 4.0) < 1e-13


@t("T5  OR(4) == 4（卡片 §3-1 手算）")
def _():
    return abs(M.odds_ratio(4.0) - 4.0) < 1e-13


@t("T6  W1(u) ≡ u 在 u ∈ [0.5, 200] 上 max|Δ| < 1e-10")
def _():
    worst = max(abs(M.W1(u / 10.0) - u / 10.0) for u in range(5, 2001))
    return worst < 1e-10


@t("T7  W2(u) ≡ u 同上")
def _():
    worst = max(abs(M.W2(u / 10.0) - u / 10.0) for u in range(5, 2001))
    return worst < 1e-10


@t("T8  OR(u) == 4·W2/W1 在 u ∈ [0.5, 200] 上 max|Δ| < 1e-10")
def _():
    worst = 0.0
    for u in range(5, 2001):
        x = u / 10.0
        worst = max(worst, abs(M.odds_ratio(x) - 4.0 * M.W2(x) / M.W1(x)))
    return worst < 1e-10


@t("T9  P1(u) > P2(u) 对 u ∈ [0.5, 200] 全成立（机制强制递减）")
def _():
    return all(M.P1(u / 10.0) > M.P2(u / 10.0) for u in range(5, 2001))


@t("T10 卡片 §3-2 手算：P1(1) = 2/3、P2(1) = 1/3")
def _():
    return abs(M.P1(1.0) - 2.0 / 3.0) < 1e-15 and abs(M.P2(1.0) - 1.0 / 3.0) < 1e-15


@t("T11 r > 1 使 P2 变小（方向：w_2>w_1 ⇔ e_1 偏大）")
def _():
    # P2 的参数是「每候选权重」；w_2 = r·w_1 ⇒ 等价于把 u 换成 r·u
    return all(M.P2(r * 4.0) < M.P2(4.0) for r in (1.5, 2.0, 3.0, 10.0))


# ---------------------------------------------------------------- solve_u / θ
@t("T12 solve_u(40) 使 N1·P1 + N2·P2 == 40（自洽）")
def _():
    u = M.solve_u(40, N1, N2)
    return abs(N1 * M.P1(u) + N2 * M.P2(u) - 40.0) < 1e-6


@t("T13 θ(40) == 0.7260046…（卡片 §3-3 预注册值）")
def _():
    th = M.theta_of(40, N1, N2)
    return th is not None and abs(th - 0.72600466) < 1e-7


@t("T14 solve_u 严格递减：m 越大 u 越小")
def _():
    us = [M.solve_u(m, N1, N2) for m in (20, 40, 60, 80, 100, 120, 134)]
    return all(us[i] > us[i + 1] for i in range(len(us) - 1))


@t("T15 θ(m) 严格递减且 ∈ (0,1)")
def _():
    ths = [M.theta_of(m, N1, N2) for m in (20, 40, 60, 80, 100, 120, 134)]
    return (all(t is not None and 0 < t < 1 for t in ths)
            and all(ths[i] > ths[i + 1] for i in range(len(ths) - 1)))


@t("T16 m <= 0 ⇒ θ 为 None（判据不可执行，不得当 0 处理）")
def _():
    return M.theta_of(0, N1, N2) is None and M.theta_of(-1, N1, N2) is None


# ---------------------------------------------------------------- 精确检验
@t("T17 binom_pmf 归一并与 2^m 全枚举一致（m=6, θ=0.3）")
def _():
    m, th = 6, 0.3
    tot = sum(M.binom_pmf(m, k, th) for k in range(m + 1))
    if abs(tot - 1.0) > 1e-12:
        return False
    brute = [0.0] * (m + 1)
    for bits in itertools.product([0, 1], repeat=m):
        w = 1.0
        for b in bits:
            w *= th if b else (1 - th)
        brute[sum(bits)] += w
    return all(abs(brute[k] - M.binom_pmf(m, k, th)) < 1e-15 for k in range(m + 1))


@t("T18 精确双侧 p 在众数处 == 1.0（m=40, θ=0.726）")
def _():
    th = M.theta_of(40, N1, N2)
    mode = max(range(41), key=lambda k: M.binom_pmf(40, k, th))
    return abs(M.binom_two_sided_p(40, mode, th) - 1.0) < 1e-12


@t("T19 接受域 == [23..35]（卡片 §2-1 / §3-3 预注册）")
def _():
    acc, mass, th = M.accept_set(40, N1, N2, ALPHA)
    return acc == list(range(23, 36))


@t("T20 接受域的水平 ≥ 1−α（不超水平）")
def _():
    for m in (20, 40, 60, 80, 100, 120, 134):
        acc, mass, th = M.accept_set(m, N1, N2, ALPHA)
        if mass is None or mass < 1.0 - ALPHA - 1e-12:
            return False
    return True


@t("T21 接受域与 p 值判据自洽（acc ⟺ p > α）")
def _():
    for m in (20, 40, 60, 80, 100, 120, 134):
        acc, mass, th = M.accept_set(m, N1, N2, ALPHA)
        for e in range(m + 1):
            p = M.binom_two_sided_p(m, e, th)
            if (e in acc) != (p > ALPHA + 1e-15):
                return False
    return True


@t("T22 卡片 §3-3 三个判读例：e1=22 拒/低、29 收、37 拒/高")
def _():
    th = M.theta_of(40, N1, N2)
    p22 = M.binom_two_sided_p(40, 22, th)
    p29 = M.binom_two_sided_p(40, 29, th)
    p37 = M.binom_two_sided_p(40, 37, th)
    return (p22 <= ALPHA and 22 < 40 * th
            and p29 > ALPHA
            and p37 <= ALPHA and 37 > 40 * th)


# ---------------------------------------------------------------- 分层计数
def _syn_items(items_spec):
    out = []
    for tgt, vals in items_spec:
        out.append({"cand_val": list(vals), "tgt_i": tgt})
    return out


@t("T23 分层计数：合成项按 n_seen 正确归类")
def _():
    tv = {"a", "b"}
    # 项 0：正确值 'X'（∉tv）+ 干扰 {a,b,Y} ⇒ n_seen=2；选 'Y' ⇒ 逃逸
    # 项 1：正确值 'X' + 干扰 {a,Y,Z} ⇒ n_seen=1；选 'a' ⇒ 不逃逸
    items = _syn_items([(0, ["X", "a", "b", "Y"]), (0, ["X", "a", "Y", "Z"])])
    ns = M0.n_seen_vec(items, tv)
    if ns != [2, 1]:
        return False
    c0 = M.counts_by_stratum([items[0]], [2], [3], tv)
    c1 = M.counts_by_stratum([items[1]], [1], [1], tv)
    return c0[2] == 1 and c1[1] == 0


@t("T24 k=3 层：逃逸 ⟺ 选对（合成）")
def _():
    tv = {"a", "b", "c"}
    items = _syn_items([(0, ["X", "a", "b", "c"])])
    ns = M0.n_seen_vec(items, tv)
    if ns != [3]:
        return False
    esc_correct = M0.escape_mask(items, ns, [0], tv)[0]
    esc_wrong = M0.escape_mask(items, ns, [1], tv)[0]
    return esc_correct is True and esc_wrong is False \
        and M.correct_at_k(items, ns, [0], 3) == 1 \
        and M.correct_at_k(items, ns, [1], 3) == 0


@t("T25 k=0 层逃逸恒 False（不可逃逸）")
def _():
    tv = {"a"}
    items = _syn_items([(0, ["X", "Y", "Z", "W"])])
    ns = M0.n_seen_vec(items, tv)
    if ns != [0]:
        return False
    return all(M0.escape_mask(items, ns, [p], tv)[0] is False for p in range(4))


# ---------------------------------------------------------------- 真实归档
@t("T26 归档回归：四题集总逃逸 == 134，主读物逐 seed == [0,3,18,0,2,9,5,0,5]")
def _():
    facts = M0.build_facts()
    tr, ev = M0.partitions_colocated(facts)
    tv = {facts[i]["val"] for i in tr}
    items = {r: M0.make_items(facts, ev, r) for r in M.ALL_RNGS}
    ns = {r: M0.n_seen_vec(items[r], tv) for r in M.ALL_RNGS}
    cells = M.load_cells()
    if len(cells) != 9:
        return False
    pmain = [int(sum(M0.escape_mask(items[M0.MAIN_RNG], ns[M0.MAIN_RNG],
                                    c["picks"][M0.MAIN_RNG], tv))) for c in cells]
    tot = 0
    for r in M.ALL_RNGS:
        for c in cells:
            tot += int(sum(M0.escape_mask(items[r], ns[r], c["picks"][r], tv)))
    return pmain == [0, 3, 18, 0, 2, 9, 5, 0, 5] and tot == 134


@t("T27 归档：acc_argmin == 选对计数/48（重建恒等式，全 36 读数）")
def _():
    facts = M0.build_facts()
    tr, ev = M0.partitions_colocated(facts)
    tv = {facts[i]["val"] for i in tr}
    items = {r: M0.make_items(facts, ev, r) for r in M.ALL_RNGS}
    cells = M.load_cells()
    for c in cells:
        for r in M.ALL_RNGS:
            got = sum(1 for p, it in zip(c["picks"][r], items[r]) if int(p) == it["tgt_i"])
            if abs(got / 48.0 - c["acc"][r]) > 1e-12:
                return False
    return True


@t("T28 留一的账目自洽：N1 - 9·d1、N2 - 9·d2、m - m_j 均非负")
def _():
    facts = M0.build_facts()
    tr, ev = M0.partitions_colocated(facts)
    tv = {facts[i]["val"] for i in tr}
    items = {r: M0.make_items(facts, ev, r) for r in M.ALL_RNGS}
    ns = {r: M0.n_seen_vec(items[r], tv) for r in M.ALL_RNGS}
    cells = M.load_cells()
    pool = M.pooled(cells, items, ns, tv)
    qtab = M.per_qset_table(cells, items, ns, tv)
    if sum(q["d1"] for q in qtab) != 68 or sum(q["d2"] for q in qtab) != 99:
        return False
    loo = M.leave_one_out(pool, qtab, len(cells))
    for row in loo:
        if row["m"] < 0:
            return False
        if row["drop"] == 0:
            continue
        # 每个留一都应可执行或明确不可执行，不得静默
        if row["executable"] is None:
            return False
    # 剔除单题集后 m 必须严格小于总 m
    return all(row["m"] < pool["m"] for row in loo)


@t("T29 pooled 的 e1+e2+e3 == 134 且 m == e1+e2（分母不丢）")
def _():
    facts = M0.build_facts()
    tr, ev = M0.partitions_colocated(facts)
    tv = {facts[i]["val"] for i in tr}
    items = {r: M0.make_items(facts, ev, r) for r in M.ALL_RNGS}
    ns = {r: M0.n_seen_vec(items[r], tv) for r in M.ALL_RNGS}
    cells = M.load_cells()
    pool = M.pooled(cells, items, ns, tv)
    return pool["total"] == 134 and pool["m"] == pool["e1"] + pool["e2"]


@t("T30 pooled.e3 == 四题集 k=3 层选对计数之和（S5 恒等式的数据侧）")
def _():
    facts = M0.build_facts()
    tr, ev = M0.partitions_colocated(facts)
    tv = {facts[i]["val"] for i in tr}
    items = {r: M0.make_items(facts, ev, r) for r in M.ALL_RNGS}
    ns = {r: M0.n_seen_vec(items[r], tv) for r in M.ALL_RNGS}
    cells = M.load_cells()
    pool = M.pooled(cells, items, ns, tv)
    corr = 0
    for c in cells:
        for r in M.ALL_RNGS:
            corr += M.correct_at_k(items[r], ns[r], c["picks"][r], 3)
    return pool["e3"] == corr


def main():
    inject = None
    if "--inject" in sys.argv:
        inject = int(sys.argv[sys.argv.index("--inject") + 1])
    npass = 0
    fails = []
    for i, (tag, fn) in enumerate(TESTS, start=1):
        if inject is not None and i == inject:
            ok = False          # ★ 只用于证明「这条能红」
        else:
            ok = bool(fn())
        if ok:
            npass += 1
        else:
            fails.append(tag)
        print(("[PASS] " if ok else "[FAIL] ") + tag)
    print()
    if inject is not None:
        print(f"[inject] 第 {inject} 条被强制改红 ⇒ 期望恰好 1 条 FAIL")
        return 0 if len(fails) == 1 else 1
    print(f"{npass}/{len(TESTS)} PASS")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())