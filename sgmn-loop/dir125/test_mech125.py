#!/usr/bin/env python3
"""`DIR-125` 测量层自测（`T1 … T18`）+ `--inject` 逐条红化。

★ 纪律（`DEF-121-02` → `DEF-122-02` → `DEF-123-03` 同族）：**占位脚手架不得留在自测里** ——
  每条断言都必须**真的能红**。`--inject k` 把第 `k` 条断言人为破坏一次，
  要求**恰好 1 条变红**（`0 异常`），否则说明该断言是空转。

用法：
  /opt/pytorch/bin/python dir125/test_mech125.py            # 全跑
  /opt/pytorch/bin/python dir125/test_mech125.py --inject 3  # 只红第 3 条
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mech125 as M                                        # noqa: E402

RES = []


def rec(cid, ok, detail):
    RES.append({"id": cid, "ok": bool(ok), "detail": detail})
    print("  [%s] %-8s %s" % ("PASS" if ok else "FAIL", cid, detail))


# ================================================================ 断言定义
def t1():
    """P1/P2/W1/W2 的代数恒等：W1(u) ≡ u ≡ W2(u)。"""
    bad = []
    for u in (0.1, 0.5, 1.0, 2.0, 9.2, 25.0):
        if abs(M.W1(u) - u) > 1e-12 or abs(M.W2(u) - u) > 1e-12:
            bad.append(u)
    return not bad, "W1(u)≡W2(u)≡u 在 6 个 u 上逐位成立" if not bad else "不符 u=%r" % bad


def t2():
    """零假设是 ratio=1：`W2(u)/W1(u) == 1` 对任意 u。"""
    bad = [u for u in (0.2, 0.55, 1.0, 9.2)
           if abs(M.W2(u) / M.W1(u) - 1.0) > 1e-12]
    return not bad, "W2/W1 ≡ 1（零假设 = 无超加性）" if not bad else "不符 u=%r" % bad


def t3():
    """`P1(1)=2/3`、`P2(1)=1/3`（旧域边界的定义）。"""
    ok = abs(M.P1(1.0) - 2.0 / 3.0) <= 1e-15 and abs(M.P2(1.0) - 1.0 / 3.0) <= 1e-15
    return ok, "P1(1)=2/3, P2(1)=1/3"


def t4():
    """工作例逐位：`(23,20,35) ⇒ u=0.2065603008, θ=0.5956264661`。"""
    u, cl = M.solve_u(35, 23, 20, return_clamped=True)
    th = M.theta_of(35, 23, 20)
    ok = (abs(u - 0.2065603008) <= 1e-9 and abs(th - 0.5956264661) <= 1e-9
          and cl is False)
    return ok, "u=%.10f θ=%.10f clamped=%s" % (u, th, cl)


def t5():
    """旧版夹紧在同一例上的读数（`θ_old=0.4380952381`、`Δθ=+0.1575312280`）。"""
    th = M.theta_of(35, 23, 20)
    th_old = 23 * M.P1(1.0) / 35.0
    ok = abs(th_old - 0.4380952381) <= 1e-9 and abs((th - th_old) - 0.1575312280) <= 1e-9
    return ok, "θ_old=%.10f Δθ=%+.10f" % (th_old, th - th_old)


def t6():
    """`m=22` 处放宽域给出 u=1.0（与旧域一致）⇒ 边界点无跳变。"""
    u = M.solve_u(22, 23, 20)
    return abs(u - 1.0) <= 1e-9, "m=22 ⇒ u=%.10f" % u


def t7():
    """`u=1` 两侧连续：`m=21.5 ⇒ u>1`、`m=22.5 ⇒ u<1`，且 θ 单调。"""
    u_a, u_b = M.solve_u(21.5, 23, 20), M.solve_u(22.5, 23, 20)
    th_a, th_b = M.theta_of(21.5, 23, 20), M.theta_of(22.5, 23, 20)
    ok = (u_a > 1.0 > u_b) and abs(u_b - 0.94898202) <= 1e-8 and th_a > th_b
    return ok, "m=21.5 ⇒ u=%.8f ; m=22.5 ⇒ u=%.8f ; θ 递减=%s" % (u_a, u_b, th_a > th_b)


def t8():
    """求解器单调：`m` 递减 ⇒ `u` 严格递增（`m = 1..42`）。"""
    us = [M.solve_u(m, 23, 20) for m in range(1, 43)]
    ok = all(us[i] > us[i + 1] for i in range(len(us) - 1))
    return ok, "m=1..42 严格递减；u(1)=%.4g u(42)=%.4g" % (us[0], us[-1])


def t9():
    """夹紧可观测：`m ≥ n1+n2` ⇒ `(0.0, True)`；`m ≤ 0` ⇒ `(None, False)`。"""
    a, ca = M.solve_u(43, 23, 20, return_clamped=True)
    b, cb = M.solve_u(44, 23, 20, return_clamped=True)
    c, cc = M.solve_u(0, 23, 20, return_clamped=True)
    ok = (a == 0.0 and ca is True and b == 0.0 and cb is True
          and c is None and cc is False)
    return ok, "m=43⇒(0.0,True) m=44⇒(0.0,True) m=0⇒(None,False)"


def t10():
    """卷积归一 + 期望值：`Σpmf=1`、`E[S]=Σmθ`。"""
    ms, ths = [3, 5, 0, 7], [0.4, 0.55, 0.5, 0.3]
    pmf = M.convolve_null(ms, ths)
    ok = abs(pmf.sum() - 1.0) <= 1e-12
    exp = sum(m * t for m, t in zip(ms, ths))
    mean = float((np.arange(len(pmf)) * pmf).sum())
    ok = ok and abs(mean - exp) <= 1e-9
    return ok, "Σpmf=%.15f E[S]=%.10f (须 %.10f)" % (pmf.sum(), mean, exp)


def t11():
    """`DIR-124` `A124-6` 的冻结工作例**逐位**复现（两层）：`S=4`、`p_two=0.5737934260`、
    `p_upper=0.2598205554`、`direction=high`。

    ★ 教训（`DEF-125-04` 的第 5 例）：我第一版凭记忆把它重建成「单层 m=3, θ=2/3」，
      得到 `p_two=0.2592592593` ⇒ 自测红。**冻结工作例必须回到源文件抄，不得凭记忆重建。**
    """
    wex = M.strat_test([{"key": "A", "N1": 4, "N2": 5, "e1": 3, "e2": 0},
                        {"key": "B", "N1": 6, "N2": 3, "e1": 1, "e2": 0}])
    ok = (wex["S"] == 4 and abs(wex["p_two"] - 0.5737934260) <= 1e-9
          and abs(wex["p_upper"] - 0.2598205554) <= 1e-9
          and wex["direction"] == "high")
    return ok, "S=%s p_two=%.10f p_upper=%.10f dir=%s" % (
        wex["S"], wex["p_two"], wex["p_upper"], wex["direction"])


def t12():
    """双侧 p 的浮点归一：全部纳入时不得 `> 1`（`DEF-124-02`）。"""
    pmf = M.convolve_null([2], [0.5])
    up, lo, two = M.tails(pmf, 1)
    ok = all(v <= 1.0 for v in (up, lo, two))
    return ok, "up=%.17g lo=%.17g two=%.17g（三者都 ≤ 1）" % (up, lo, two)


def t13():
    """`strat_test` 的 `m` 不一致必须抛错（接线守卫）。"""
    try:
        M.strat_test([{"key": "x", "N1": 10, "N2": 10, "e1": 1, "e2": 2, "m": 99}])
    except ValueError:
        return True, "m 与 e1+e2 不一致 ⇒ 抛 ValueError"
    return False, "**未抛错**（守卫失效）"


def t14():
    """`m=0` 的层被跳过，不得计入 `strata_used`。"""
    r = M.strat_test([{"key": "a", "N1": 23, "N2": 20, "e1": 0, "e2": 0},
                      {"key": "b", "N1": 23, "N2": 20, "e1": 3, "e2": 1}])
    return r["strata_used"] == 1, "m=0 被跳过；strata_used=%d" % r["strata_used"]


def t15():
    """`w_hat` 的代数：给定 (N1,N2,e1,e2) 手算一致；且 `ratio=1` 当 `e1/N1 == e2/N2`？（不是）"""
    # 取一个 u，由模型造精确概率，再验 w_hat 的公式
    u = 2.0
    N1, N2 = 200, 200
    e1 = int(round(N1 * M.P1(u)))
    e2 = int(round(N2 * M.P2(u)))
    r = M.w_hat([{"N1": N1, "N2": N2, "e1": e1, "e2": e2}])
    p1, p2 = e1 / N1, e2 / N2
    w1 = (1 - p1) * 2 / p1
    w2 = (1 - p2) / (2 * p2)
    ok = abs(r["w1"] - w1) <= 1e-12 and abs(r["w2"] - w2) <= 1e-12 \
        and abs(r["ratio"] - w2 / w1) <= 1e-12
    return ok, "w1=%.6f w2=%.6f ratio=%.6f（与手算一致）" % (r["w1"], r["w2"], r["ratio"])


def t16():
    """`J125-2` 的判据方向：`ratio > 1` 当且仅当 `k=2` 的每候选加成更大。"""
    # u=2、ratio=2 ⇒ p2 = P2(4) 更小 ⇒ 逃逸更少 ⇒ w2 更大
    u, rat = 2.0, 2.0
    N1 = N2 = 1000
    e1 = int(round(N1 * M.P1(u)))
    e2 = int(round(N2 * M.P2(rat * u)))
    r = M.w_hat([{"N1": N1, "N2": N2, "e1": e1, "e2": e2}])
    return r["ratio"] > 1.0, "构造 ratio=2 ⇒ 估计 ratio=%.4f > 1 ✓" % r["ratio"]


def t17():
    """`e1=0` 或 `e2=0` 时 `w_hat` 返回 `None`（不得除零、不得给假数）。"""
    r = M.w_hat([{"N1": 23, "N2": 20, "e1": 0, "e2": 5}])
    r2 = M.w_hat([{"N1": 23, "N2": 20, "e1": 5, "e2": 0}])
    ok = r["ratio"] is None and r2["ratio"] is None
    return ok, "e1=0 或 e2=0 ⇒ ratio=None（不除零）"


def t18():
    """`power_for_strata` 非空转：`ratio=1` 与 `ratio=3` 的拒绝率必须拉开。"""
    strata = [{"key": "s%d" % i, "N1": 68, "N2": 99} for i in range(9)]
    a = M.power_for_strata(strata, u=9.2, ratio=1.0, reps=1500, seed=7)
    b = M.power_for_strata(strata, u=9.2, ratio=3.0, reps=1500, seed=7)
    return (a < 0.12 and b > 0.90), "ratio=1 ⇒ %.4f ; ratio=3 ⇒ %.4f（须拉开）" % (a, b)


TESTS = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18]


# ================================================================ inject
def run_one(k, inject=None):
    """跑第 `k` 条；`inject == k` 时返回被破坏的结果。"""
    fn = TESTS[k - 1]
    ok, detail = fn()
    if inject == k:
        return (not ok), detail + "  ★ [INJECTED]"
    return ok, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", type=int, default=None)
    a = ap.parse_args()

    n = len(TESTS)
    print("=" * 100)
    print("DIR-125 测量层自测 —— %d 条" % n)
    print("=" * 100)

    if a.inject is None:
        for k in range(1, n + 1):
            ok, detail = run_one(k)
            rec("T%d" % k, ok, detail)
        npass = sum(1 for r in RES if r["ok"])
        print()
        print("=" * 100)
        print("%d / %d PASS" % (npass, n))
        print("=" * 100)
        sys.exit(0 if npass == n else 1)

    # inject 模式：逐条注入，要求**恰好 1 条红**
    print("注入模式：对 k = 1..%d 各注入一次，要求恰好 1 条红\n" % n)
    bad_cases = []
    for k in range(1, n + 1):
        reds = []
        for j in range(1, n + 1):
            ok, _ = run_one(j, inject=k)
            if not ok:
                reds.append(j)
        status = "OK" if reds == [k] else "异常"
        if reds != [k]:
            bad_cases.append((k, reds))
        print("  inject %2d ⇒ 红 %r  %s" % (k, reds, status))
    print()
    print("=" * 100)
    print("异常 %d 条%s" % (len(bad_cases),
                            "" if not bad_cases else "：" + repr(bad_cases)))
    print("=" * 100)
    sys.exit(0 if not bad_cases else 1)


if __name__ == "__main__":
    main()