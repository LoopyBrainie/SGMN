#!/usr/bin/env python3
"""`DIR-124` **开卡前**结构核 —— 只打印结构事实 + 已发表常数 + 功效/水平。

★★ 本脚本的**存在理由**（`DEF-123-05` 的正面修法）：
   `DIR-123` 的 `J123-2` 用「四个留一**都**须在 `0.025` 下拒绝」的**合取**，
   而该合取的**功效从未被计算** ⇒ 一个方向四题集全一致、汇总 `p = 0.0041` 的效应
   被判成「行 4 不可判」。根因是**功效手算**（甚至根本没算）。
   ⇒ `DIR-124` 把功效做成 `mech124.power()` **函数**，本脚本断言它 `≥ 0.80`。

★★ 纪律：本脚本**不得**计算/打印 `dose=600` 或 `dose=0` 的任何逃逸读数
   （`S`、`e₁`、`e₂`、`m`、`p` 观测值）。`S12` 明文断言这一点。
   功效计算**只**用 `strata_of()` 的 `N1/N2`（纯结构）与备择 `(u, ratio)`，
   **不读**任何 `picks` 观测值 ⇒ 不构成"看过数据"。

用法：`/opt/pytorch/bin/python dir124/precheck124.py`
"""
import os
import sys
import json
import math

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT, os.path.join(ROOT, "dir123"), os.path.join(ROOT, "dir122")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mech122 as M                                              # noqa: E402
import mech123 as M3                                             # noqa: E402
import mech124 as M4                                             # noqa: E402

OUT = os.path.join(HERE, "out")
CHECKS = []


def rec(tag, ok, msg=""):
    CHECKS.append({"tag": tag, "ok": bool(ok), "msg": msg})
    print("%-8s %s  %s" % (tag, "PASS" if ok else "FAIL", msg))
    return ok


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 78)
    print("DIR-124 开卡前结构核（只打印结构事实 + 已发表常数 + 功效/水平）")
    print("=" * 78)

    ctx = M4.build_ctx(M4.DOSE_MAIN)

    # ---------------------------------------------------------------- S1 结构
    hs = {r: M.hist_of(ctx["ns"][r], M.N_CAND) for r in M4.ALL_RNGS}
    # ★ 已发表 `d`（`D-154`）只给了 `(d₁,d₂,d₃)`；`d₀` 由本卡首次确立（结构事实）
    published = {555: [2, 23, 20, 3], 556: [3, 18, 27, 0],
                 557: [4, 9, 29, 6], 558: [3, 18, 23, 4]}
    as_list = {r: [hs[r][str(k)] for k in range(M.N_CAND)] for r in M4.ALL_RNGS}
    rec("S1", all(as_list[r] == published[r] for r in M4.ALL_RNGS),
        "n_seen 直方图逐题集 = 已发表 d（D-154）：%s" % (as_list,))

    n1 = sum(h["1"] for h in hs.values())
    n2 = sum(h["2"] for h in hs.values())
    rec("S1b", (n1, n2) == (68, 99),
        "单 seed 合计 (N1,N2) = (%d,%d)（已发表 (68,99)）" % (n1, n2))

    # ---------------------------------------------------------------- S2 池化分母
    n_cells = len(ctx["cells"])
    rec("S2", (n_cells * n1, n_cells * n2) == (612, 891),
        "池化分母 = %d×%d / %d×%d = (%d,%d)（已发表 612/891）"
        % (n_cells, n1, n_cells, n2, n_cells * n1, n_cells * n2))

    # ---------------------------------------------------------------- S3 分层完整性
    ok3 = True
    detail = {}
    for axis in M4.AXES_CANDIDATES:
        st = M4.strata_of(ctx, axis)
        s1 = sum(s["N1"] for s in st)
        s2 = sum(s["N2"] for s in st)
        detail[axis] = (len(st), s1, s2)
        if (s1, s2) != (n_cells * n1, n_cells * n2):
            ok3 = False
    rec("S3", ok3,
        "三种分层的 ΣN1/ΣN2 全等于池化分母：%s" % (detail,))

    # ---------------------------------------------------------------- S4 卷积零分布归一
    st_seed = M4.strata_of(ctx, "seed")
    # 用**结构**构造一个合成 m 向量（不是观测值）：令每层 m_s = N1_s//4
    ms_fake = [max(1, s["N1"] // 4) for s in st_seed]
    ths_fake = [M3.theta_of(m, s["N1"], s["N2"]) for m, s in zip(ms_fake, st_seed)]
    ok4 = all(t is not None for t in ths_fake)
    pmf = M4.convolve_null(ms_fake, ths_fake) if ok4 else np.zeros(1)
    tot = float(pmf.sum())
    rec("S4", ok4 and abs(tot - 1.0) < 1e-9 and len(pmf) == sum(ms_fake) + 1,
        "合成 m 向量下卷积 pmf 归一 = %.12f（长度 %d = Σm+1 = %d）"
        % (tot, len(pmf), sum(ms_fake) + 1))

    # ---------------------------------------------------------------- S5 卷积 vs 暴力枚举
    # 3 层小规模：把 DP 结果与「逐项枚举笛卡尔积」对齐
    import itertools
    ms_small = [3, 4, 2]
    ths_small = [0.7, 0.55, 0.4]
    dp = M4.convolve_null(ms_small, ths_small)
    bf = [0.0] * (sum(ms_small) + 1)
    for ks in itertools.product(*[range(m + 1) for m in ms_small]):
        p = 1.0
        for k, m, t in zip(ks, ms_small, ths_small):
            p *= M3.binom_pmf(m, k, t)
        bf[sum(ks)] += p
    maxdiff = max(abs(a - b) for a, b in zip(dp, bf))
    rec("S5", maxdiff < 1e-14,
        "DP 卷积 vs 暴力枚举（3 层，%d 项）max|Δ| = %.2e"
        % (math.prod(m + 1 for m in ms_small), maxdiff))

    # ---------------------------------------------------------------- S6 尾概率语义
    # 单层时 S == e1 ⇒ 上尾必须等于 binom_two_sided 之外的单尾值
    m1, Nn1, Nn2 = 8, 23, 20
    th1 = M3.theta_of(m1, Nn1, Nn2)
    pm1 = M4.convolve_null([m1], [th1])
    ok6 = True
    for e in (0, 3, 8):
        up, lo, two = M4._tails(pm1, e)
        # 单层 §上尾 = P(X >= e)；用 binom pmf 直算对照
        up_direct = sum(M3.binom_pmf(m1, k, th1) for k in range(e, m1 + 1))
        if abs(up - up_direct) > 1e-12:
            ok6 = False
        if abs(up + lo - pm1[e] - 1.0) > 1e-9:      # up+lo = 1 + P(obs)
            ok6 = False
    rec("S6", ok6, "单层时上尾 = P(X>=e)、且 上尾+下尾 = 1+P(obs)（e ∈ {0,3,8}）")

    # ---------------------------------------------------------------- S7 θ 语义
    # θ_s = N1·P1(û_s)/m_s，且必须满足 m_s = N1 P1 + N2 P2（解的定义）
    ok7 = True
    for s in st_seed[:4]:
        for m in (1, 3, 7):
            th = M3.theta_of(m, s["N1"], s["N2"])
            if th is None:
                ok7 = False
                continue
            u = M3.solve_u(m, s["N1"], s["N2"])
            lhs = s["N1"] * M3.P1(u) + s["N2"] * M3.P2(u)
            if abs(lhs - m) > 1e-8 or not (0.0 < th < 1.0):
                ok7 = False
    rec("S7", ok7,
        "θ_s 语义：solve_u 反代 N1·P1(û)+N2·P2(û) = m（误差 < 1e-8）、且 θ ∈ (0,1)")

    # ---------------------------------------------------------------- S8 备择可辨识
    # 在备择下 p₂ 必须**变小**（⇒ e1 相对偏高）；检查 ratio>1 的方向
    ok8 = all(M3.P2(2.0 * u) < M3.P2(u) for u in (1.5, 5.0, 9.2, 18.0))
    rec("S8", ok8,
        "备择方向：ratio = 2 ⇒ P2(2u) < P2(u)，u ∈ {1.5,5,9.2,18} 全部成立")

    # ---------------------------------------------------------------- S9 功效：三轴
    print("\n--- S9 功效（side=%s，备择 ratio = 2，u = 9.2，%d 次 MC）---"
          % (M4.SIDE, M4.MC_REPS))
    pw = {}
    for axis in M4.AXES_CANDIDATES:
        pw[axis] = M4.power(axis=axis, u=9.2, ratio=2.0, reps=M4.MC_REPS)
        print("    axis=%-5s  功效 = %.4f" % (axis, pw[axis]))
    chosen = M4.AXIS
    rec("S9", pw[chosen] >= M4.POWER_MIN,
        "选定轴 %r 的功效 = %.4f ≥ %.2f" % (chosen, pw[chosen], M4.POWER_MIN))
    rec("S9b", all(pw[a] < 0.999 for a in pw),
        "三轴功效都不是 1.0 ⇒ 检验不是恒拒绝（最低 %.4f）" % min(pw.values()))

    # ---------------------------------------------------------------- S10 功效随 ratio 单调
    print("\n--- S10 功效单调性（axis = %r，u = 9.2）---" % chosen)
    prof = {}
    for ratio in (1.0, 1.5, 2.0, 3.0):
        prof[ratio] = M4.power(axis=chosen, u=9.2, ratio=ratio,
                               reps=M4.MC_REPS // 2)
        print("    ratio=%.1f  拒绝率 = %.4f" % (ratio, prof[ratio]))
    rec("S10", prof[1.0] < prof[1.5] < prof[2.0] < prof[3.0],
        "拒绝率随 ratio 严格递增 ⇒ 检验对备择有方向性")

    # ---------------------------------------------------------------- S10b 功效 vs u
    # `u` 是未知参数 ⇒ 功效须在**合理区间**都够，不能只在 `u = 9.2` 成立
    print("\n--- S10b 功效对 u 的稳健性（axis = %r，ratio = 2）---" % chosen)
    pw_u = {}
    for u in (1.5, 5.0, 9.2, 18.0):
        pw_u[u] = M4.power(axis=chosen, u=u, ratio=2.0, reps=M4.MC_REPS // 2)
        print("    u=%-5.1f  功效 = %.4f" % (u, pw_u[u]))
    u_ok = [u for u, v in pw_u.items() if v >= M4.POWER_MIN]
    rec("S10b", len(u_ok) >= 3,
        "u ∈ {1.5,5,9.2,18} 中 %d/4 个点的功效 ≥ %.2f（达标 u = %s）"
        % (len(u_ok), M4.POWER_MIN, u_ok))

    # ---------------------------------------------------------------- S11 水平校准
    print("\n--- S11 水平校准（ratio = 1，u 扫三点）---")
    lv = {}
    for u in (1.5, 9.2, 18.0):
        lv[u] = M4.level(axis=chosen, u=u, reps=M4.MC_REPS)
        print("    u=%-5.1f  拒绝率 = %.4f（名义 α = %.2f）" % (u, lv[u], M4.ALPHA))
    # ★ 精确条件检验在离散小样本下**保守**（拒绝率 < α）—— 保守是**安全**的；
    #   断言只要求 ① **有效性**（不超过 α + MC 误差）② **不退化**（不是恒不拒绝）。
    tol_up, floor = 0.004, 0.005
    valid = all(v <= M4.ALPHA + tol_up for v in lv.values())
    notdeg = all(v >= floor for v in lv.values())
    rec("S11", valid and notdeg,
        "有效性：三 u 拒绝率全部 ≤ α+%.3f = %.3f；不退化：全部 ≥ %.3f"
        "（实际 %s ⇒ 保守但有效）"
        % (tol_up, M4.ALPHA + tol_up, floor,
           {k: round(v, 4) for k, v in lv.items()}))

    # ---------------------------------------------------------------- S12 ★ 无观测泄漏
    # 本脚本**不**调用 stratum_obs / strat_test(观测)；断言源码里没有对它们的调用
    # ★ 禁用词用**拼接构造**，否则禁用词表自身的字面量会造成假 FAIL（上一版就是这样）
    banned = ["stratum" + "_obs(", "strat" + "_test(ctx", "counts_by" + "_stratum("]
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    body = [ln for ln in src.splitlines()
            if not ln.lstrip().startswith("#") and "banned =" not in ln]
    hits = [b for b in banned if any(b in ln for ln in body)]
    rec("S12", not hits,
        "本脚本未调用任何观测读数函数（禁用词命中 = %s）" % (hits or "无"))

    # ---------------------------------------------------------------- S13 冻结量自检
    rec("S13", M4.SIDE in ("two", "upper") and M4.AXIS in M4.AXES_CANDIDATES
        and M4.ALPHA == M4.ALPHA_PLACEBO == 0.05,
        "冻结量自检：SIDE=%r、AXIS=%r、ALPHA=%s、ALPHA_PLACEBO=%s"
        % (M4.SIDE, M4.AXIS, M4.ALPHA, M4.ALPHA_PLACEBO))

    # ---------------------------------------------------------------- 汇总
    n_ok = sum(1 for c in CHECKS if c["ok"])
    print("\n" + "=" * 78)
    print("%d / %d PASS" % (n_ok, len(CHECKS)))
    print("=" * 78)
    json.dump({"checks": CHECKS, "all_ok": n_ok == len(CHECKS),
               "power": {k: v for k, v in pw.items()},
               "power_profile": {str(k): v for k, v in prof.items()},
               "level": {str(k): v for k, v in lv.items()},
               "power_vs_u": {str(k): v for k, v in pw_u.items()},
               "axis_chosen": chosen, "alpha": M4.ALPHA, "side": M4.SIDE,
               "mc_reps": M4.MC_REPS, "mc_seed": M4.MC_SEED},
              open(os.path.join(OUT, "precheck124.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("→ out/precheck124.json")
    return 0 if n_ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())