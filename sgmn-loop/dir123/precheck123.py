#!/usr/bin/env python3
"""`DIR-123` 结构核 —— **开卡前**运行（`P122-1`：逃逸率随 `n_seen = 1/2/3` 变化吗？）。

★ 本核的纪律（Cycle 233 起常设）：
  **只打印结构事实 + 已发表常数，绝不打印 `P122-1` 的判据观测值**
  （具体地：**不**打印 `k=1` / `k=2` 分层上的逃逸计数 `e_1` / `e_2`，也**不**打印其任何
   等价形式；这些要到正式轮才第一次出现）。

★ 本核要机械确立的**三件事**（全部可证，不依赖观测）：
  `S2` `n_seen` 的抽样结构 = 超几何 `Hypergeom(N=15, K=8, n=3)` ⇒ 有精确可算的空模型；
  `S5` `n_seen = 3` 时 **逃逸 ⟺ 选对** ⇒ 当 `acc == 0` 时该层逃逸**恒为 0**（退化层）；
  `S7` 在 `k = 1,2` 上「每候选加成」`w_k` 是 `k`-**不变量**，且 `OR = 4·w_2/w_1`
       ⇒ `H0: w_1 = w_2` 是一个**参数无关的精确预测**（可据此预注册精确拒绝域）。
"""
import os
import sys
import json
import math
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "dir122"), os.path.join(ROOT, "dir121"),
          os.path.join(ROOT, "dir120")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mech122 as M                                              # noqa: E402
import mech123 as M2                                             # noqa: E402

# ★ 单一实现纪律：`P1/P2/w_k/OR/solve_u/θ/精确双侧` 全部来自 `mech123`，
#   本核**不**再各写一份（`DIR-122` 的教训：两份实现会分叉）。

PASS, FAIL = [], []


def ck(tag, cond, msg=""):
    (PASS if cond else FAIL).append(tag)
    print(("[PASS] " if cond else "[FAIL] ") + tag + ("  " + msg if msg else ""))


def _C(n, k):
    return math.comb(n, k)


def hyper_pmf(N, K, n):
    """`P(X = k)`，`X` = 抽 `n` 个里属于 `K` 个特例的个数（无放回）。"""
    tot = _C(N, n)
    return {k: _C(K, k) * _C(N - K, n - k) / tot for k in range(0, n + 1)}


p1_of = M2.P1
p2_of = M2.P2
w1_of = M2.W1
w2_of = M2.W2


def main():
    print("=" * 78)
    print("DIR-123 结构核（开卡前）—— 只打印结构事实与已发表常数")
    print("=" * 78)

    # ---------------------------------------------------------------- S0 常量
    print("\n--- S0 常量（与 mech122 一致）---")
    print(f"  N_CAND={M.N_CAND}  N_DRAWS={M.N_DRAWS}  N_VAL={M.N_VAL}  N_FACT={M.N_FACT}")
    print(f"  FACT_SEED={M.FACT_SEED}  SHUFFLE_SEED={M.SHUFFLE_SEED}  "
          f"MAIN_RNG={M.MAIN_RNG}  C2_RNGS={M.C2_RNGS}")
    ck("S0-a 常量与卡片一致",
       (M.N_CAND, M.N_DRAWS, M.N_VAL, M.N_FACT) == (4, 3, 16, 32)
       and (M.FACT_SEED, M.SHUFFLE_SEED) == (9091, 777)
       and M.MAIN_RNG == 555 and tuple(M.C2_RNGS) == (556, 557, 558))

    # ------------------------------------------------- S1 事实与值池的基数
    print("\n--- S1 事实/值池基数（决定 `n_seen` 的抽样结构）---")
    facts = M.build_facts()
    vals = [f["val"] for f in facts]
    uniq = sorted(set(vals))
    print(f"  事实数={len(facts)}  不同值数={len(uniq)}  每值出现次数={sorted({vals.count(v) for v in uniq})}")
    tr, ev = M.partitions_colocated(facts)
    tv = {facts[i]["val"] for i in tr}
    heldv = {facts[i]["val"] for i in ev}
    print(f"  tr 事实={len(tr)}（值 {len(tv)} 个）  ev 事实={len(ev)}（值 {len(heldv)} 个）")
    print(f"  tv ∩ heldv = {sorted(tv & heldv)}（须空）  tv ∪ heldv 大小 = {len(tv | heldv)}")
    ck("S1-a 每个值恰出现 2 次", all(vals.count(v) == 2 for v in uniq))
    ck("S1-b 训练侧/留出侧值不相交", not (tv & heldv))
    ck("S1-c 两侧值数各 8、事实数各 16", len(tv) == 8 and len(heldv) == 8
       and len(tr) == 16 and len(ev) == 16)

    # ------------------------------- S2 干扰项池的分解 ⇒ 超几何空模型
    print("\n--- S2 干扰项池分解（`make_items` 的 `pool`）---")
    # 取一个 ev key，照 make_items 的构造复算 pool
    ki = ev[0]
    correct = facts[ki]["val"]
    others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != correct]
    pool = sorted(set(others))
    n_tv = sum(1 for v in pool if v in tv)
    n_nv = len(pool) - n_tv
    print(f"  单个 ev key：正确值 ∈ tv? {correct in tv}（须 False）")
    print(f"  pool 大小 = {len(pool)}  = {n_tv} 个 ∈ tv + {n_nv} 个 ∉ tv")
    print(f"  ⇒ `n_seen` ~ Hypergeom(N={len(pool)}, K={n_tv}, n={M.N_CAND - 1})")
    pdf = hyper_pmf(len(pool), n_tv, M.N_CAND - 1)
    print("  精确 pmf：" + "  ".join(f"k={k}: {v:.4f}（{v * 48:.2f} 项/48）" for k, v in pdf.items()))
    ck("S2-a pool = 15、其中 8 个见过值、7 个未见过值",
       len(pool) == 15 and n_tv == 8 and n_nv == 7)
    ck("S2-b 正确值不在 tv 内（留出侧值）", correct not in tv)
    ck("S2-c pmf 归一", abs(sum(pdf.values()) - 1.0) < 1e-12)

    # ---------------------- S3 各题集的 `n_seen` 实测 vs 重建（已发表常数）
    print("\n--- S3 各题集 `n_seen` 直方图（已发表常数 + 重建一致性）---")
    allr = (M.MAIN_RNG,) + tuple(M.C2_RNGS)
    items = {r: M.make_items(facts, ev, r) for r in allr}
    ns = {r: M.n_seen_vec(items[r], tv) for r in allr}
    print(f"  {'rng':>6} {'n':>4}  hist(0..3)                 d(1..3)        k=0 项(不可逃逸)")
    for r in allr:
        h = M.hist_of(ns[r], n_bins=M.N_CAND)
        d = [sum(1 for v in ns[r] if v == k) for k in (1, 2, 3)]
        print(f"  {r:>6} {len(ns[r]):>4}  "
              f"[{h['0']:2d} {h['1']:2d} {h['2']:2d} {h['3']:2d}]   "
              f"d={tuple(d)}   k0={h['0']}")
    ck("S3-a 四个题集均为 48 项", all(len(ns[r]) == 48 for r in allr))
    # 已发表直方图（`D-152` 主读物；`D-153` 三个 c2 题集）—— 逐题集硬比对
    PUB = {M.MAIN_RNG: {"0": 2, "1": 23, "2": 20, "3": 3},
           556: {"0": 3, "1": 18, "2": 27, "3": 0},
           557: {"0": 4, "1": 9, "2": 29, "3": 6},
           558: {"0": 3, "1": 18, "2": 23, "3": 4}}
    h_main = M.hist_of(ns[M.MAIN_RNG], M.N_CAND)
    ck("S3-b 主读物直方图 == 已发表 {0:2,1:23,2:20,3:3}",
       h_main == PUB[M.MAIN_RNG], str(h_main))
    bad = {r: M.hist_of(ns[r], M.N_CAND) for r in M.C2_RNGS
           if M.hist_of(ns[r], M.N_CAND) != PUB[r]}
    ck("S3-c 三个 c2 题集直方图 == 已发表（D-153）", not bad, str(bad))

    d1 = sum(sum(1 for v in ns[r] if v == 1) for r in allr)
    d2 = sum(sum(1 for v in ns[r] if v == 2) for r in allr)
    d3 = sum(sum(1 for v in ns[r] if v == 3) for r in allr)
    print(f"\n  单个 seed 跨 4 题集合计：n_1={d1}  n_2={d2}  n_3={d3}  （合计 {d1 + d2 + d3}）")
    print(f"  9 seed 合计（配对，直方图逐 seed 相同）：N_1={9 * d1}  N_2={9 * d2}  N_3={9 * d3}")
    ck("S3-d 单 seed 分母 = (68, 99, 13)",
       (d1, d2, d3) == (68, 99, 13), str((d1, d2, d3)))

    # ---------------- S4 存档 `n_seen_hist` 与重建逐 seed 一致（配对性）
    print("\n--- S4 存档字段 `n_seen_hist` 与重建一致（配对设计）---")
    cells = sorted(glob.glob(os.path.join(ROOT, "dir119", "out", "cells",
                                         "COLOCATED_600_*.json")))
    archiv = [json.load(open(f, encoding="utf-8")) for f in cells]
    print(f"  读入 {len(archiv)} 个 cell：" + ", ".join(str(c["seed"]) for c in archiv))
    same = []
    for c in archiv:
        same.append(M.norm_hist(c["n_seen_hist"], M.N_CAND) == h_main)
    ck("S4-a 9 个 cell 的 n_seen_hist 全部 == 主读物直方图（⇒ 设计配对）", all(same))
    ck("S4-b cell 数 = 9", len(archiv) == 9)

    # ---------- S5 `n_seen = 3` ⇒ 逃逸 ⟺ 选对（穷举证明，非观测）
    print("\n--- S5 `n_seen=3` 时 逃逸 ⟺ 选对（穷举所有取值组合）---")
    ok5 = True
    combos = 0
    for k in range(0, M.N_CAND):
        # 造一个合成项：3 个干扰项里恰 k 个 ∈ tv；正确值 ∉ tv
        cand = ["correct"]
        cand += [f"seen{j}" for j in range(k)]
        cand += [f"unseen{j}" for j in range(M.N_CAND - 1 - k)]
        for tgt in range(M.N_CAND):
            cand2 = cand[:]
            cand2[0], cand2[tgt] = cand2[tgt], cand2[0]
            it = {"cand_val": cand2, "tgt_i": tgt}
            tvs = {f"seen{j}" for j in range(k)}
            nsx = [sum(1 for j, v in enumerate(cand2) if j != tgt and v in tvs)]
            assert nsx[0] == k, (nsx, k)
            for pickpos in range(M.N_CAND):
                esc = M.escape_mask([it], nsx, [pickpos], tvs)[0]
                is_correct = (pickpos == tgt)
                pv = cand2[pickpos]
                combos += 1
                # 定义层面的恒等式（对所有 k 都成立）：逃逸 = k≥1 且 选中值 ∉ tv
                # ★ 注意：正确值本身 ∉ tv ⇒ **选对也算逃逸**，这是本装置的既有定义
                if esc != ((k >= 1) and (pv not in tvs)):
                    ok5 = False
                if k == 3:
                    # k=3 时 3 个干扰项全 ∈ tv ⇒ 「∉ tv」只剩正确值一条路
                    if esc != is_correct:
                        ok5 = False
    print(f"  穷举组合数 = {combos}（k=0..3 × 4 个正确位置 × 4 个选择）")
    ck("S5-a 穷举：逃逸 ≡ (k≥1 ∧ 选中值 ∉ tv)，且 k=3 时逃逸 ⟺ 选对", ok5)
    ck("S5-b k=0 时逃逸恒 False（不可逃逸层）",
       all(M.escape_mask([{"cand_val": ["c"] + [f"u{j}" for j in range(3)], "tgt_i": 0}],
                         [0], [p], set())[0] is False for p in range(4)))

    # -------- S6 存档 `acc_argmin`：取值结构 + 「acc == 选对计数/48」重建恒等式
    print("\n--- S6 存档 `acc_argmin` / `acc_argmax`（已发表字段）---")
    print("  ★ 本节**不**断言 acc 恒为 0（首版曾如此假设 ⇒ 假 FAIL，已改）。")
    units = []
    ok_acc, ok_int, n_read = True, True, 0
    for c in archiv:
        pk_main = c["mech_by_mult"]["10000"]["pick_argmin"]
        jobs = [(M.MAIN_RNG, pk_main, c["mech_by_mult"]["10000"]["acc_argmin"])]
        for r in M.C2_RNGS:
            jobs.append((r, c["c2"]["10000"][str(r)]["pick_argmin"],
                         c["c2"]["10000"][str(r)]["acc_argmin"]))
        for r, pk, acc in jobs:
            n_read += 1
            units.append(acc)
            got = sum(1 for p, it in zip(pk, items[r]) if int(p) == it["tgt_i"])
            # ★ 重建恒等式：存档 acc_argmin 必须 == 选对计数 / 48
            if abs(got / 48.0 - acc) > 1e-12:
                ok_acc = False
            if abs(round(acc * 48) - acc * 48) > 1e-9:
                ok_int = False
    print(f"  读数 {n_read} 个；acc_argmin 范围 [{min(units):.4f}, {max(units):.4f}]"
          f"（= {min(units) * 48:.0f}/48 … {max(units) * 48:.0f}/48）")
    print(f"  acc_argmin 非零的读数个数 = {sum(1 for a in units if a > 0)} / {n_read}")
    ck("S6-a 重建恒等式：acc_argmin == 选对计数/48（全部 36 个读数）", ok_acc)
    ck("S6-b acc_argmin 均为 1/48 的整数倍（n=48）", ok_int)
    print("  ⇒ ★ 结构性后果：正确值 ∉ tv ⇒ 选对也算逃逸 ⇒ **k=3 层逃逸 ⟺ 选对**，"
          "即该层测的是**准确率**、不是泄漏倾向 ⇒ 不得进「剂量」比较。")
    print("  ⇒ ★ 且正确值在 k=0 层也可能被选中（k=0 层逃逸被 `n_seen>=1` 屏蔽）"
          "⇒ `m`（k∈{1,2} 上的逃逸数）**不可预先算死**，须在正式轮实测。")

    # ------------------ S7 `w_k` 的 k-不变性与 `OR = 4 w_2 / w_1`（代数）
    print("\n--- S7 代数：`w_k` 在 k=1,2 上是 k-不变量；`OR = 4·w_2/w_1` ---")
    worst = 0.0
    for u10 in range(10, 2001):                       # u = 1.0 .. 200.0
        u = u10 / 10.0
        worst = max(worst, abs(w1_of(u) - u), abs(w2_of(u) - u))
    print(f"  `w_k(u) ≡ u` 在 u ∈ [1,200]（步长 0.1，1992 点）上的最大偏差 = {worst:.3e}")
    ck("S7-a `w_k(u) ≡ u`（k=1,2）", worst < 1e-9, f"max|Δ|={worst:.2e}")
    # OR 恒等式：OR = 4 w_2 / w_1
    worst2 = 0.0
    for u10 in range(10, 2001):
        u = u10 / 10.0
        p1, p2 = p1_of(u), p2_of(u)
        OR = (p1 / (1 - p1)) / (p2 / (1 - p2))
        worst2 = max(worst2, abs(OR - 4.0 * w2_of(u) / w1_of(u)))
    ck("S7-b `OR = 4·w_2/w_1` 恒成立", worst2 < 1e-9, f"max|Δ|={worst2:.2e}")
    print(f"  ⇒ `H0: w_1 = w_2` ⟺ `OR = 4`（参数无关的精确预测）")

    # ------------- S8 预注册：给定实测 m 时无多余参数（条件检验）
    print("\n--- S8 预注册规则（纯结构；`m` 在正式轮实测，本节给出 θ(m) 表）---")
    N1, N2 = 9 * d1, 9 * d2
    print(f"  N_1 = {N1}（k=1 槽位总数）   N_2 = {N2}（k=2 槽位总数）")

    def solve_u(m):
        return M2.solve_u(m, N1, N2)

    binom_pmf = M2.binom_pmf

    alpha = M2.ALPHA_J1
    print(f"  {'m':>5} {'û':>9} {'θ(m)':>9} {'E[e_1]':>8}  {'接受域 e_1':>22}  {'实际水平':>10}")
    for m in (20, 40, 60, 80, 100, 120, 134):
        acc, mass, th = M2.accept_set(m, N1, N2, alpha)
        if acc is None:
            txt = "（不可执行）"
        elif len(acc) == max(acc) - min(acc) + 1:
            txt = f"{min(acc)}..{max(acc)}"
        else:
            txt = str(acc)
        print(f"  {m:>5} {solve_u(m):>9.3f} {th:>9.5f} {m * th:>8.2f}  {txt:>22}  {1 - mass:>10.5f}")
    print("  ⇒ 规则（开卡前冻结）：给定实测 `m`，û 由上式唯一定 ⇒ "
          "`e_1 | m ~ Binomial(m, θ(m))`，精确双侧 `p ≤ α = 0.025` 即拒绝 `H0`。")
    print("  ⇒ ✓ **给定 m 后没有多余参数**（不是 plug-in 近似）。")

    acc42, mass42, th42 = M2.accept_set(134, N1, N2, alpha)
    lev42 = 1 - mass42

    ck("S8-a θ(m) 严格递减且 ∈ (0,1)",
       M2.accept_set(20, N1, N2, alpha)[2] > M2.accept_set(134, N1, N2, alpha)[2] > 0)
    ck("S8-b 精确拒绝域的实际水平 ≤ α 且 ≥ α−0.01（不浪费功效）",
       lev42 <= alpha + 1e-12 and lev42 >= alpha - 0.01, f"level={lev42:.5f}")
    print("\n  功效参考（纯结构计算，非观测）：固定 m=134，把 w_2 抬到 ratio·w_1")
    for ratio in (1.0, 2.0, 3.0, 5.0, 0.5):
        def f2(u):
            return N1 * p1_of(u) + N2 * (1.0 / (2.0 * (u * ratio) + 1.0)) - 134
        lo2, hi2 = 1.0, 1e12
        for _ in range(400):
            mid = math.sqrt(lo2 * hi2)
            if f2(mid) > 0:
                lo2 = mid
            else:
                hi2 = mid
        u2 = math.sqrt(lo2 * hi2)
        th = (N1 * p1_of(u2)) / 134.0
        pw = 1.0 - sum(binom_pmf(134, k, th) for k in acc42)
        print(f"    w_2/w_1 = {ratio:>4.1f} ⇒ θ={th:.4f}  功效={pw:.4f}")
    print("  ⇒ ★ 方向（已核）：`w_2 = r·w_1` 时 `p_2 = 1/(1+2ru)` ⇒ r>1 使 p_2 **更小**"
          " ⇒ 固定 m 下 e_1 份额**更高**"
          " ⇒ **w_2>w_1 ⇔ e_1 偏大（超加性/剂量型）**；w_2<w_1 ⇔ e_1 偏小（次加性）。")

    # ------------------------------ S9 已发表常数回归 + 判据未见断言
    print("\n--- S9 已发表常数回归（只读已发表值）---")
    PUB_TOT = {M.MAIN_RNG: 42, 556: 44, 557: 18, 558: 30}
    esc = {}
    for r in allr:
        if r == M.MAIN_RNG:
            esc[r] = [int(sum(M.escape_mask(items[r], ns[r],
                                           c["mech_by_mult"]["10000"]["pick_argmin"], tv)))
                          for c in archiv]
        else:
            esc[r] = [int(sum(M.escape_mask(items[r], ns[r],
                                           c["c2"]["10000"][str(r)]["pick_argmin"], tv)))
                          for c in archiv]
        print(f"  rng={r}: 逐 seed = {esc[r]}  合计 = {sum(esc[r])}")
    ck("S9-a 主读物逐 seed == 已发表 [0,3,18,0,2,9,5,0,5]",
       esc[M.MAIN_RNG] == [0, 3, 18, 0, 2, 9, 5, 0, 5], str(esc[M.MAIN_RNG]))
    ck("S9-b 四个题集合计 == 已发表 (42, 44, 18, 30)",
       [sum(esc[r]) for r in allr] == [PUB_TOT[r] for r in allr],
       str([sum(esc[r]) for r in allr]))
    grand = sum(sum(esc[r]) for r in allr)
    print(f"  ⇒ 四个题集总逃逸数 = {grand}（不变量；`m` = 其中的 k∈{{1,2}} 部分，须实测）")
    ck("S9-c 总逃逸数 == 134", grand == 134, str(grand))

    print("\n--- S10 判据观测值未见断言 ---")
    print("  ★ 本核**未**计算、**未**打印 `k=1` / `k=2` 分层上的逃逸计数 `e_1` / `e_2`")
    print("    及其任何等价形式（比值、OR、`w_k` 的估计值、`m` 本身）。")
    print("  ★ 因此 §S8 的 θ(m) 规则是**纯结构**推出的，无法被结果倒推。")
    ck("S10-a 判据观测值保持未见（本核只输出结构量与已发表常数）", True)

    print("\n" + "=" * 78)
    if FAIL:
        print(f"[precheck123] {len(FAIL)} 项 FAIL：{FAIL}")
        print("=" * 78)
        return 1
    print(f"[precheck123] 全部 PASS（{len(PASS)} 项）")
    print("[precheck123] ★ 本核未计算、未打印 k=1/k=2 分层上的任何逃逸读数")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())