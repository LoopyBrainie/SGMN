#!/usr/bin/env python3
"""`DIR-124` 测量层自测 —— 每条断言都必须能独立红（`--inject k` 把第 k 条改成返回 False）。

★ 纪律：**不得留空转脚手架**（`DEF-121-02` / `DEF-122-02` / `DEF-123-03` 已四次同族）。
  每条测试都必须是**真断言**；`--inject` 只用于**证明**它能红，不得用于"制造通过"。

★ 本卡新增的两条纪律（`DEF-123-05` 的正面修法）：
  `T-P*` 组**真的调用** `power()`/`level()`（小 `reps`）并断言其数值性质 ——
  功效不是"手算声明"，而是**被测试的对象**。
"""
import os
import sys
import math
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "dir123"), os.path.join(ROOT, "dir122")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mech124 as M4                                             # noqa: E402
import mech123 as M3                                             # noqa: E402
import mech122 as M0                                             # noqa: E402

TESTS = []


def t(tag):
    def deco(fn):
        TESTS.append((tag, fn))
        return fn
    return deco


# ================================================================ 机制层（继承，不改）
@t("T1  P1(2) == 1/2、P2(2) == 1/5（DIR-123 同一实现）")
def _():
    return abs(M3.P1(2.0) - 0.5) < 1e-15 and abs(M3.P2(2.0) - 0.2) < 1e-15


@t("T2  W1(u) ≡ W2(u) ≡ u（每候选加成与 k 无关）在 u ∈ [1,200] 上 < 1e-10")
def _():
    return all(abs(M3.W1(u) - u) < 1e-10 and abs(M3.W2(u) - u) < 1e-10
               for u in [1.0, 1.5, 2.0, 4.0, 9.2, 18.0, 50.0, 200.0])


@t("T3  OR(u) ≡ 4 在 u ∈ [1,200] 上 < 1e-10（参数无关的精确预测）")
def _():
    return all(abs(M3.odds_ratio(u) - 4.0) < 1e-10
               for u in [1.0, 2.0, 9.2, 18.0, 200.0])


@t("T4  P2(r·u) < P2(u) 当 r > 1（备择方向由机制决定，不是约定）")
def _():
    return all(M3.P2(2.0 * u) < M3.P2(u) for u in (1.5, 5.0, 9.2, 18.0))


# ================================================================ 分层结构
@t("T5  strata_of 三轴的 ΣN1/ΣN2 全 = (612, 891)")
def _():
    ctx = M4.build_ctx(M4.DOSE_MAIN)
    for axis in M4.AXES_CANDIDATES:
        st = M4.strata_of(ctx, axis)
        if (sum(s["N1"] for s in st), sum(s["N2"] for s in st)) != (612, 891):
            return False
    return True


@t("T6  strata_of 层数：both=36、seed=9、qset=4")
def _():
    ctx = M4.build_ctx(M4.DOSE_MAIN)
    return all(len(M4.strata_of(ctx, a)) == n
               for a, n in (("both", 36), ("seed", 9), ("qset", 4)))


@t("T7  qset 轴的每层 N1 = 9 × 该题集单 cell 的 d1（×n_seeds，S3 曾抓到漏乘）")
def _():
    ctx = M4.build_ctx(M4.DOSE_MAIN)
    st_q = {s["key"]: s for s in M4.strata_of(ctx, "qset")}
    st_b = {s["key"]: s for s in M4.strata_of(ctx, "both")}
    ns = ctx["ns"]
    n_seeds = len(ctx["cells"])
    for r in M4.ALL_RNGS:
        d1 = sum(1 for v in ns[r] if v == 1)
        if st_q["qset=%d" % r]["N1"] != n_seeds * d1:
            return False
        # both 轴的每层 = 单 cell ⇒ 不乘
        if st_b["seed=%d,qset=%d" % (ctx["cells"][0]["seed"], r)]["N1"] != d1:
            return False
    return True


@t("T8  分层不重叠且不漏：所有 idx 的并集大小 = n_seeds × 4（每轴）")
def _():
    ctx = M4.build_ctx(M4.DOSE_MAIN)
    n_seeds = len(ctx["cells"])
    want = n_seeds * len(M4.ALL_RNGS)
    for axis in M4.AXES_CANDIDATES:
        seen = set()
        for s in M4.strata_of(ctx, axis):
            for idx in s["idx"]:
                seen.add(idx)
        if len(seen) != want:
            return False
    return True


# ================================================================ 卷积零分布
@t("T9  convolve_null 归一为 1（合成 m 向量）")
def _():
    ms = [3, 4, 2, 1]
    ths = [0.7, 0.55, 0.4, 0.9]
    pmf = M4.convolve_null(ms, ths)
    return abs(float(pmf.sum()) - 1.0) < 1e-12 and len(pmf) == sum(ms) + 1


@t("T10 convolve_null vs 暴力枚举（4 层、60 项）max|Δ| < 1e-15")
def _():
    ms, ths = [3, 4, 2], [0.7, 0.55, 0.4]
    dp = M4.convolve_null(ms, ths)
    bf = [0.0] * (sum(ms) + 1)
    for ks in itertools.product(*[range(m + 1) for m in ms]):
        p = 1.0
        for k, m, th in zip(ks, ms, ths):
            p *= M3.binom_pmf(m, k, th)
        bf[sum(ks)] += p
    return max(abs(a - b) for a, b in zip(dp, bf)) < 1e-15


@t("T11 convolve_null 跳过 m == 0 层（不改变分布）")
def _():
    a = M4.convolve_null([3, 0, 2], [2.0 / 3.0, 0.5, 0.4])
    b = M4.convolve_null([3, 2], [2.0 / 3.0, 0.4])
    return len(a) == len(b) and max(abs(x - y) for x, y in zip(a, b)) < 1e-15


@t("T12 单层时 上尾 = P(X >= e)、且 上尾 + 下尾 = 1 + pmf(e)")
def _():
    m, th = 8, 0.62
    pmf = M4.convolve_null([m], [th])
    ok = True
    for e in (0, 2, 5, 8):
        up, lo, _ = M4._tails(pmf, e)
        direct = sum(M3.binom_pmf(m, k, th) for k in range(e, m + 1))
        ok &= abs(up - direct) < 1e-12
        ok &= abs(up + lo - 1.0 - pmf[e]) < 1e-12
    return ok


@t("T13 双侧 p：∈ [pmf(obs), 1]、众数处 = 1（★ 不 ≥ 上尾 —— 上尾含 pmf 更大的点）")
def _():
    pmf = M4.convolve_null([3, 1], [2.0 / 3.0, 0.8768943744])
    ok = True
    for e in range(len(pmf)):
        up, lo, two = M4._tails(pmf, e)
        # 三条恒真性质：① 必含 obs ② ≤ 1 ③ 上尾/下尾各自必含 obs
        ok &= two >= pmf[e] - 1e-12 and two <= 1.0 + 1e-12
        ok &= up >= pmf[e] - 1e-12 and lo >= pmf[e] - 1e-12
    mode = int(max(range(len(pmf)), key=lambda i: pmf[i]))
    ok &= abs(M4._tails(pmf, mode)[2] - 1.0) < 1e-12
    return ok


# ================================================================ 工作例（卡片 §3-2 逐位）
@t("T14 工作例 层A：û = 2 精确（4·p1+5·p2 = 3）")
def _():
    return (abs(M3.solve_u(3, 4, 5) - 2.0) < 1e-12
            and abs(M3.P1(2.0) - 0.5) < 1e-15
            and abs(M3.P2(2.0) - 0.2) < 1e-15)


@t("T15 工作例 层A：θ = 2/3、Binomial(3,2/3) = [1,6,12,8]/27")
def _():
    th = M3.theta_of(3, 4, 5)
    want = [1.0 / 27, 6.0 / 27, 12.0 / 27, 8.0 / 27]
    return (abs(th - 2.0 / 3.0) < 1e-12
            and max(abs(M3.binom_pmf(3, k, th) - want[k]) for k in range(4)) < 1e-12)


@t("T16 工作例 层B：û = 11.6846584384、θ = 0.8768943744")
def _():
    u = M3.solve_u(1, 6, 3)
    th = M3.theta_of(1, 6, 3)
    return abs(u - 11.6846584384) < 1e-9 and abs(th - 0.8768943744) < 1e-9


@t("T17 工作例 卷积 pmf 逐位 = 卡片 §3-2 表")
def _():
    want = [0.00455947, 0.05983438, 0.24957903, 0.42620657, 0.25982056]
    pmf = M4.convolve_null([3, 1], [2.0 / 3.0, M3.theta_of(1, 6, 3)])
    return len(pmf) == 5 and max(abs(pmf[i] - want[i]) for i in range(5)) < 5e-9


@t("T18 工作例 strat_test：S=4、p_two=0.5737934260、p_upper=0.2598205554、dir=high")
def _():
    rows = [{"key": "A", "N1": 4, "N2": 5, "e1": 3, "e2": 0},
            {"key": "B", "N1": 6, "N2": 3, "e1": 1, "e2": 0}]
    r = M4.strat_test(rows)
    return (r["executable"] and r["S"] == 4 and r["m"] == 4
            and abs(r["p_two"] - 0.5737934260) < 1e-9
            and abs(r["p_upper"] - 0.2598205554) < 1e-9
            and r["direction"] == "high" and r["reject"] is False
            and abs(r["expect"] - 2.8768943744) < 1e-9)


@t("T19 工作例 m 一致：显式给出错误 m ⇒ 抛 ValueError")
def _():
    rows = [{"key": "A", "N1": 4, "N2": 5, "e1": 3, "e2": 0, "m": 99}]
    try:
        M4.strat_test(rows)
    except ValueError:
        return True
    return False


# ================================================================ 判据行为
@t("T20 全 e1 = 0 ⇒ S=0、上尾=1、双侧=pmf[0]（下尾显著）⇒ 双侧下 reject=True、dir=low")
def _():
    rows = [{"key": "A", "N1": 4, "N2": 5, "e1": 0, "e2": 3},
            {"key": "B", "N1": 6, "N2": 3, "e1": 0, "e2": 1}]
    r = M4.strat_test(rows)
    pmf0 = M4.convolve_null([3, 1], [2.0 / 3.0, M3.theta_of(1, 6, 3)])[0]
    return (r["executable"] and r["S"] == 0
            and abs(r["p_upper"] - 1.0) < 1e-12
            and abs(r["p_two"] - float(pmf0)) < 1e-12
            and r["direction"] == "low")


@t("T21 e1 全部推到上界（e1 = N1）⇒ 必拒绝、方向 high")
def _():
    rows = [{"key": "A", "N1": 4, "N2": 5, "e1": 4, "e2": 0},
            {"key": "B", "N1": 6, "N2": 3, "e1": 6, "e2": 0}]
    r = M4.strat_test(rows)
    return (r["executable"] and r["reject"] is True
            and r["direction"] == "high" and r["p_two"] <= 0.05)


@t("T22 全部 m = 0 ⇒ 不可执行（reason = no_stratum_with_m_gt_0，p = None）")
def _():
    rows = [{"key": "A", "N1": 4, "N2": 5, "e1": 0, "e2": 0}]
    r = M4.strat_test(rows)
    return (r["executable"] is False
            and r["reason"] == "no_stratum_with_m_gt_0"
            and r["p_two"] is None and r["reject"] is None)


@t("T23 m = 0 的层被跳过但不改变判决（strata_used 减少 1，S/p 不变）")
def _():
    a = M4.strat_test([{"key": "A", "N1": 4, "N2": 5, "e1": 3, "e2": 0}])
    b = M4.strat_test([{"key": "A", "N1": 4, "N2": 5, "e1": 3, "e2": 0},
                       {"key": "Z", "N1": 9, "N2": 9, "e1": 0, "e2": 0}])
    return (a["strata_used"] == 1 and b["strata_used"] == 1
            and a["S"] == b["S"] and abs(a["p_two"] - b["p_two"]) < 1e-15)


@t("T24 判据只吃 k=1/2：stratum_obs 的键里没有 e3（CORR-35 ① 结构守卫）")
def _():
    ctx = M4.build_ctx(M4.DOSE_MAIN)
    rows = M4.stratum_obs(ctx, M4.strata_of(ctx, "seed"))
    return all(set(r.keys()) == {"key", "N1", "N2", "e1", "e2", "m"}
               for r in rows)


@t("T25 dose=600 观测读数 = D-154 已发表（e1=109、e2=24、m=133）")
def _():
    ctx = M4.build_ctx(M4.DOSE_MAIN)
    rows = M4.stratum_obs(ctx, M4.strata_of(ctx, "seed"))
    e1 = sum(r["e1"] for r in rows)
    e2 = sum(r["e2"] for r in rows)
    return (e1, e2, e1 + e2) == (109, 24, 133)


@t("T26 dose=0：9 cell 全是 dose=0、W_bitwise_same 真、**只有题集 555**（无 c2）")
def _():
    ctx = M4.build_ctx(M4.DOSE_PLACEBO)
    ok = len(ctx["cells"]) == 9
    for c in ctx["cells"]:
        ok &= c["dose"] == 0
        ok &= bool(c["raw"].get("W_bitwise_same")) is True
        ok &= (c["raw"].get("c2") == {} or not c["raw"].get("c2"))
    ok &= ctx["rngs"] == (M0.MAIN_RNG,)
    return ok


@t("T26b dose=0 的机会数 ΣN1/ΣN2 = (207,180)（= 主检的 0.338 / 0.202）")
def _():
    st = M4.strata_of(M4.build_ctx(M4.DOSE_PLACEBO), "seed")
    n1 = sum(s["N1"] for s in st)
    n2 = sum(s["N2"] for s in st)
    return (n1, n2) == (207, 180) and abs(n1 / 612.0 - 0.338) < 5e-4


@t("T26c dose=600 的 rngs = (555,556,557,558)（主检用全四题集）")
def _():
    return M4.build_ctx(M4.DOSE_MAIN)["rngs"] == (555, 556, 557, 558)


@t("T27 两批数据的 pick_argmin 不完全相同（负对照不是训练态副本）")
def _():
    a = M4.build_ctx(M4.DOSE_MAIN)["cells"]
    b = M4.build_ctx(M4.DOSE_PLACEBO)["cells"]
    same = all(a[i]["picks"][r] == b[i]["picks"][r]
               for i in range(len(a)) for r in M4.ALL_RNGS)
    return not same


# ================================================================ 功效（被测试的对象，不是声明）
@t("T28 power(ratio=1) < power(ratio=2)（小 reps）")
def _():
    p1 = M4.power(axis="seed", u=9.2, ratio=1.0, reps=1500)
    p2 = M4.power(axis="seed", u=9.2, ratio=2.0, reps=1500)
    return p1 < p2


@t("T29 power(ratio=3) > power(ratio=2)（小 reps，单调性）")
def _():
    p2 = M4.power(axis="seed", u=9.2, ratio=2.0, reps=1500)
    p3 = M4.power(axis="seed", u=9.2, ratio=3.0, reps=1500)
    return p3 > p2


@t("T30 level(ratio=1) ≤ α + 0.02（有效性，小 reps）")
def _():
    lv = M4.level(axis="seed", u=9.2, reps=1500)
    return lv <= M4.ALPHA + 0.02


@t("T31 power 对同一 seed 可复现（固定 MC_SEED ⇒ 逐位相同）")
def _():
    a = M4.power(axis="seed", u=9.2, ratio=2.0, reps=800)
    b = M4.power(axis="seed", u=9.2, ratio=2.0, reps=800)
    return a == b


@t("T32 power 在小 u 下高于大 u（m 更大 ⇒ 信息更多，结构性质）")
def _():
    lo = M4.power(axis="seed", u=1.5, ratio=2.0, reps=1500)
    hi = M4.power(axis="seed", u=18.0, ratio=2.0, reps=1500)
    return lo > hi


@t("T33 冻结量自检：SIDE/AXIS/ALPHA/DOSE 与卡片 §9 一致")
def _():
    return (M4.SIDE == "two" and M4.AXIS == "seed"
            and M4.ALPHA == 0.05 and M4.ALPHA_PLACEBO == 0.05
            and M4.DOSE_MAIN == 600 and M4.DOSE_PLACEBO == 0
            and M4.POWER_MIN == 0.80)


@t("T34 判据不是恒真也不是恒假：合成池上 reject 真出现过 True 与 False")
def _():
    seen = set()
    for e1a in range(0, 5):
        for e1b in range(0, 7):
            rows = [{"key": "A", "N1": 4, "N2": 5, "e1": e1a, "e2": 2},
                    {"key": "B", "N1": 6, "N2": 3, "e1": e1b, "e2": 1}]
            r = M4.strat_test(rows)
            if r["executable"]:
                seen.add(r["reject"])
    return seen == {True, False}


# ================================================================ main
def main():
    inject = None
    if "--inject" in sys.argv:
        inject = int(sys.argv[sys.argv.index("--inject") + 1])

    n_ok = 0
    print("=" * 78)
    print("DIR-124 测量层自测（%d 条）" % len(TESTS))
    print("=" * 78)
    for i, (tag, fn) in enumerate(TESTS):
        try:
            ok = bool(fn())
        except Exception as exc:                                  # noqa: BLE001
            ok = False
            tag = "%s  [EXC: %s]" % (tag, exc)
        if inject is not None and i == inject:
            ok = not ok                    # 强制翻转 ⇒ 必须恰好 1 条 FAIL
        print("%-4s %s" % ("PASS" if ok else "FAIL", tag))
        n_ok += 1 if ok else 0

    print("=" * 78)
    print("%d / %d PASS" % (n_ok, len(TESTS)))
    if inject is not None:
        print("[inject] 第 %d 条被强制翻转 ⇒ 期望恰好 1 条 FAIL" % inject)
    print("=" * 78)
    return 0 if n_ok == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())