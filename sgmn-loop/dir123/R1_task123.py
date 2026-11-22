#!/usr/bin/env python3
"""`DIR-123` 执行器：`--stage a`（14 条守卫）与 `--stage formal`（正式轮）。

判据见 `revolve/DIR-123/DIRECTION_rev123_v10.md` §2（**开卡前冻结**）：
  `J123-1`  精确条件检验：给定实测 `m`，`e_1 | m ~ Binomial(m, θ(m))`，`p ≤ 0.025` ⇒ 拒绝。
  `J123-2`  留一题集：四个留一重算**全部**在 `0.025` 下拒绝且方向一致。
  `J123-3`  report-only。
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "dir122"), os.path.join(ROOT, "dir121"),
          os.path.join(ROOT, "dir120")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mech122 as M                                              # noqa: E402
import mech123 as M2                                             # noqa: E402

OUT = os.path.join(HERE, "out")
ALPHA1 = M2.ALPHA_J1
ALPHA2 = M2.ALPHA_J2
N_CELLS = 9


def _ctx():
    facts = M.build_facts()
    tr, ev = M.partitions_colocated(facts)
    tv = {facts[i]["val"] for i in tr}
    items = {r: M.make_items(facts, ev, r) for r in M2.ALL_RNGS}
    ns = {r: M.n_seen_vec(items[r], tv) for r in M2.ALL_RNGS}
    cells = M2.load_cells()
    return {"facts": facts, "tr": tr, "ev": ev, "tv": tv,
            "items": items, "ns": ns, "cells": cells}


# ================================================================ 判据
def judge(pool, qtab):
    """返回 `J123-1` 全量读数（不打印观测值以外的东西）。"""
    m, e1 = pool["m"], pool["e1"]
    th = M2.theta_of(m, 612, 891)
    if th is None:
        return {"executable": False, "m": m, "e1": e1, "theta": None,
                "p": None, "reject": None, "expect": None, "direction": None,
                "accept": None}
    acc, mass, _ = M2.accept_set(m, 612, 891, ALPHA1)
    p = M2.binom_two_sided_p(m, e1, th)
    reject = bool(p <= ALPHA1)
    expect = m * th
    direction = "high" if e1 > expect else ("low" if e1 < expect else "eq")
    return {"executable": True, "m": m, "e1": e1, "e2": pool["e2"],
            "theta": th, "p": p, "reject": reject, "expect": expect,
            "direction": direction, "accept": acc, "accept_mass": mass}


def judge_loo(pool, qtab):
    rows = M2.leave_one_out(pool, qtab, N_CELLS, ALPHA1)
    ok = all(r["executable"] and r["reject"] for r in rows)
    ref = judge(pool, qtab)["direction"]
    same = all(r["dir"] == ref for r in rows if r["executable"])
    return {"rows": rows, "all_reject": ok, "same_direction": same,
            "passed": bool(ok and same)}


def decide(j1, j2):
    """卡片 §2-4 冻结判决表（五行）。"""
    if not j1["executable"]:
        return 5, "H123_UNJUDGEABLE", "判据不可执行"
    if not j1["reject"]:
        if j2["passed"]:
            return 1, "H123_CONFIRMED", "每候选加成与「见到几个」无关"
        return 4, "H123_NOT_ROBUST", "留一不稳（J123-1 未拒绝，故按行 4 记不可判）"
    if not j2["passed"]:
        return 4, "H123_NOT_ROBUST", "结论依赖单一题集 ⇒ 降级描述性"
    if j1["direction"] == "high":
        return 2, "H123_FALSIFIED_SUPERADDITIVE", "w_2 > w_1（超加性/剂量型）"
    if j1["direction"] == "low":
        return 3, "H123_FALSIFIED_SUBADDITIVE", "w_2 < w_1（次加性）"
    return 4, "H123_NOT_ROBUST", "拒绝但方向为 eq（不应发生）"


# ================================================================ Stage A
def stage_a():
    C = _ctx()
    R, ok_all = [], True

    def rec(tag, ok, msg=""):
        nonlocal ok_all
        R.append({"tag": tag, "ok": bool(ok), "msg": msg})
        print(("[PASS] " if ok else "[FAIL] ") + tag + ("  " + msg if msg else ""))
        if not ok:
            ok_all = False

    PUB = {M.MAIN_RNG: {"0": 2, "1": 23, "2": 20, "3": 3},
           556: {"0": 3, "1": 18, "2": 27, "3": 0},
           557: {"0": 4, "1": 9, "2": 29, "3": 6},
           558: {"0": 3, "1": 18, "2": 23, "3": 4}}
    hs = {r: M.hist_of(C["ns"][r], M.N_CAND) for r in M2.ALL_RNGS}

    rec("A123-1 四题集 n_seen 直方图 == 已发表值",
        all(hs[r] == PUB[r] for r in M2.ALL_RNGS), str(hs))

    d = {r: tuple(sum(1 for v in C["ns"][r] if v == k) for k in (1, 2, 3))
         for r in M2.ALL_RNGS}
    want_d = {M.MAIN_RNG: (23, 20, 3), 556: (18, 27, 0),
              557: (9, 29, 6), 558: (18, 23, 4)}
    rec("A123-2 逐题集 (d1,d2,d3) == (23,20,3)/(18,27,0)/(9,29,6)/(18,23,4)",
        all(d[r] == want_d[r] for r in M2.ALL_RNGS), str(d))

    s1 = sum(d[r][0] for r in M2.ALL_RNGS)
    s2 = sum(d[r][1] for r in M2.ALL_RNGS)
    s3 = sum(d[r][2] for r in M2.ALL_RNGS)
    rec("A123-3 单 seed 合计 (68,99,13) ⇒ N1=612 N2=891 N3=117",
        (s1, s2, s3) == (68, 99, 13),
        f"({s1},{s2},{s3}) ⇒ ({9 * s1},{9 * s2},{9 * s3})")

    esc = {}
    for r in M2.ALL_RNGS:
        esc[r] = [int(sum(M.escape_mask(C["items"][r], C["ns"][r],
                                       c["picks"][r], C["tv"]))) for c in C["cells"]]
    rec("A123-4 已发表回归：主读物逐 seed == [0,3,18,0,2,9,5,0,5]",
        esc[M.MAIN_RNG] == [0, 3, 18, 0, 2, 9, 5, 0, 5], str(esc[M.MAIN_RNG]))
    tots = [sum(esc[r]) for r in M2.ALL_RNGS]
    rec("A123-4b 四题集合计 == (42,44,18,30) ⇒ 总 134",
        tots == [42, 44, 18, 30] and sum(tots) == 134, str(tots))

    pool = M2.pooled(C["cells"], C["items"], C["ns"], C["tv"])
    corr3 = sum(M2.correct_at_k(C["items"][r], C["ns"][r], c["picks"][r], 3)
                for c in C["cells"] for r in M2.ALL_RNGS)
    rec("A123-5 k=3 层逃逸数 == k=3 项上的选对计数",
        pool["e3"] == corr3, f"e3={pool['e3']} corr3={corr3}")

    e0 = 0
    for r in M2.ALL_RNGS:
        for c in C["cells"]:
            e0 += M2.counts_by_stratum(C["items"][r], C["ns"][r],
                                       c["picks"][r], C["tv"])[0]
    rec("A123-6 k=0 层逃逸数恒 0", e0 == 0, f"e0={e0}")

    rec("A123-7 e1+e2+e3 == 134（分母不静默丢失）",
        pool["total"] == 134, f"e1={pool['e1']} e2={pool['e2']} "
                             f"e3={pool['e3']} total={pool['total']}")

    grid = (0.5, 1.0, 4.0, 10.0, 100.0, 1e4)
    rec("A123-8 P1(u) > P2(u) 对 u ∈ {0.5,1,4,10,100,1e4}（机制强制递减）",
        all(M2.P1(u) > M2.P2(u) for u in grid))

    worst9 = max(abs(M2.W1(k / 10.0) - k / 10.0) for k in range(10, 2001))
    worst9b = max(abs(M2.W2(k / 10.0) - k / 10.0) for k in range(10, 2001))
    rec("A123-9 w_k(u) ≡ u（k=1,2）在 u∈[1,200] 上 max|Δ| < 1e-9",
        max(worst9, worst9b) < 1e-9, f"{max(worst9, worst9b):.2e}")

    worst10 = 0.0
    for k in range(10, 2001):
        u = k / 10.0
        worst10 = max(worst10, abs(M2.odds_ratio(u) - 4.0 * M2.W2(u) / M2.W1(u)))
    rec("A123-10 OR = 4·w2/w1 在同网格 max|Δ| < 1e-9", worst10 < 1e-9,
        f"{worst10:.2e}")

    uu = M2.solve_u(40, 612, 891)
    rec("A123-11 solve_u(40) 唯一可解且在 (1e-9, 1e9)",
        1e-9 < uu < 1e9, f"u={uu:.6f}")

    th40 = M2.theta_of(40, 612, 891)
    acc40, mass40, _ = M2.accept_set(40, 612, 891, ALPHA1)
    rec("A123-12 m=40 ⇒ θ=0.72600、接受域 23..35（复现卡片 §3-3）",
        abs(th40 - 0.72600466) < 1e-7 and acc40 == list(range(23, 36)),
        f"θ={th40:.8f} acc={acc40[0]}..{acc40[-1]}")

    lev_ok = True
    for m in (20, 60, 100, 134):
        acc, mass, th = M2.accept_set(m, 612, 891, ALPHA1)
        if mass is None or mass < 1.0 - ALPHA1 - 1e-12:
            lev_ok = False
    rec("A123-13 精确检验水平自检：接受域质量 ≥ 1−α（不超水平）", lev_ok)

    # ★ 在**合成**池上跑完整判据链：只断言「非 None / m>0 / 分支互斥」，不泄露真实 p
    syn_pool = {"e1": 30, "e2": 12, "e3": 8, "m": 42, "total": 50}
    syn_q = [{"rng": 555, "d1": 23, "d2": 20, "d3": 3, "e1": 8, "e2": 3, "e3": 1, "m": 11},
             {"rng": 556, "d1": 18, "d2": 27, "d3": 0, "e1": 7, "e2": 4, "e3": 0, "m": 11},
             {"rng": 557, "d1": 9, "d2": 29, "d3": 6, "e1": 6, "e2": 3, "e3": 4, "m": 9},
             {"rng": 558, "d1": 18, "d2": 23, "d3": 4, "e1": 9, "e2": 2, "e3": 3, "m": 11}]
    j1s = judge(syn_pool, syn_q)
    j2s = judge_loo(syn_pool, syn_q)
    row_s, tag_s, _ = decide(j1s, j2s)
    rec("A123-14 合成池上判据真的被计算过（p 非 None、m>0、行 ∈ 1..5）",
        j1s["p"] is not None and j1s["m"] > 0 and row_s in (1, 2, 3, 4, 5)
        and j2s["rows"] and all(r["executable"] is not None for r in j2s["rows"]),
        f"syn row={row_s}")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"checks": R, "all_ok": ok_all},
              open(os.path.join(OUT, "stage_stageA.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n★ Stage A {'全部通过' if ok_all else '有 FAIL'}"
          f"（{sum(1 for r in R if r['ok'])}/{len(R)}）")
    return 0 if ok_all else 1


# ================================================================ 正式轮
def formal():
    C = _ctx()
    pool = M2.pooled(C["cells"], C["items"], C["ns"], C["tv"])
    qtab = M2.per_qset_table(C["cells"], C["items"], C["ns"], C["tv"])
    j1 = judge(pool, qtab)
    j2 = judge_loo(pool, qtab)
    row, verdict, desc = decide(j1, j2)

    print("=" * 78)
    print("DIR-123 正式轮 —— 每候选加成 w_k 与「见到几个」无关吗？")
    print("=" * 78)
    print(f"\n分层逃逸（四题集 × {N_CELLS} seed 汇总）：")
    print(f"  e1 (k=1) = {pool['e1']}    e2 (k=2) = {pool['e2']}    "
          f"e3 (k=3, ⟺选对) = {pool['e3']}")
    print(f"  m := e1+e2 = {pool['m']}    总 = {pool['total']} (须 134)")

    print(f"\nJ123-1（主，α={ALPHA1}）：")
    if not j1["executable"]:
        print(f"  ✗ 不可执行（m={j1['m']}）")
    else:
        print(f"  θ(m) = {j1['theta']:.6f}   E[e1] = {j1['expect']:.3f}")
        print(f"  接受域 = {j1['accept'][0]}..{j1['accept'][-1]}"
              f"（质量 {j1['accept_mass']:.5f}）")
        print(f"  e1 = {j1['e1']}   精确双侧 p = {j1['p']:.6g}   "
              f"拒绝 H123 = {j1['reject']}   方向 = {j1['direction']}")

    print(f"\nJ123-2（留一题集，α={ALPHA2}）：")
    for r in j2["rows"]:
        if r["executable"]:
            print(f"  剔除 {r['drop']}: m={r['m']:>3} e1={r['e1']:>3} "
                  f"θ={r['theta']:.5f} p={r['p']:.4g} 拒绝={r['reject']} 方向={r['dir']}")
        else:
            print(f"  剔除 {r['drop']}: 不可执行（m={r['m']}, e1={r['e1']}）")
    print(f"  四个留一全部拒绝 = {j2['all_reject']}；方向一致 = {j2['same_direction']}"
          f" ⇒ J123-2 {'过' if j2['passed'] else '未过'}")

    # ---------------- J123-3 report-only
    m, e1 = pool["m"], pool["e1"]
    p1h = e1 / 612.0 if 612 else None
    p2h = pool["e2"] / 891.0 if 891 else None
    print(f"\nJ123-3（report-only）：")
    print(f"  p1 = e1/N1 = {e1}/612 = {p1h:.5f}    p2 = e2/N2 = {pool['e2']}/891 = {p2h:.5f}")
    if p1h and p2h and p1h < 1 and p2h < 1:
        w1 = (1 - p1h) * 2.0 / (p1h * 1.0)
        w2 = (1 - p2h) * 1.0 / (p2h * 2.0)
        OR = (p1h / (1 - p1h)) / (p2h / (1 - p2h))
        print(f"  ŵ1 = {w1:.4f}   ŵ2 = {w2:.4f}   ŵ2/ŵ1 = {w2 / w1:.4f}")
        print(f"  OR 估计 = {OR:.4f}（H123 预测 = 4.0000）  "
              f"隐含 b̂ 在 k=1/2 上分别为 {w1 - 1:.4f} / {w2 - 1:.4f}")
    print("\n  逐 seed：")
    for r in M2.per_seed_table(C["cells"], C["items"], C["ns"], C["tv"]):
        print(f"    {r['seed']}: e1={r['e1']:>2} e2={r['e2']:>2} e3={r['e3']:>2} m={r['m']:>2}")
    print("\n  逐题集：")
    for r in qtab:
        print(f"    rng={r['rng']}: d=({r['d1']},{r['d2']},{r['d3']}) "
              f"e=({r['e1']},{r['e2']},{r['e3']}) m={r['m']}")
    # k=0 / k=3 的准确率探针
    acc0 = sum(M2.correct_at_k(C["items"][r], C["ns"][r], c["picks"][r], 0)
               for c in C["cells"] for r in M2.ALL_RNGS)
    n0 = 9 * sum(1 for r in M2.ALL_RNGS for v in C["ns"][r] if v == 0)
    n3 = 9 * sum(1 for r in M2.ALL_RNGS for v in C["ns"][r] if v == 3)
    print(f"\n  准确率探针：k=0 层 选对 {acc0}/{n0}"
          f"（{acc0 / n0:.4f}）   k=3 层 选对 {pool['e3']}/{n3}"
          f"（{pool['e3'] / n3:.4f}）")

    print(f"\n{'=' * 78}")
    print(f"★★ 判决：第 {row} 行 → {verdict}　（{desc}）")
    print("=" * 78)

    os.makedirs(OUT, exist_ok=True)
    json.dump({"row": row, "verdict": verdict, "desc": desc,
               "J123_1": j1, "J123_2": j2,
               "J123_3": {"pool": {k: v for k, v in pool.items() if k != "per"},
                          "per_qset": qtab,
                          "per_seed": M2.per_seed_table(C["cells"], C["items"],
                                                        C["ns"], C["tv"]),
                          "p1_hat": p1h, "p2_hat": p2h,
                          "acc_k0": [acc0, n0], "acc_k3": [pool["e3"], n3]},
               "alphas": {"j1": ALPHA1, "j2": ALPHA2}},
              open(os.path.join(OUT, "formal.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0


def main():
    stage = "formal"
    if "--stage" in sys.argv:
        stage = sys.argv[sys.argv.index("--stage") + 1]
    if stage == "a":
        return stage_a()
    return formal()


if __name__ == "__main__":
    sys.exit(main())