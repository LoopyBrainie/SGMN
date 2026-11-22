#!/usr/bin/env python3
"""`DIR-124` 诊断 —— `solve_u` 的 `u >= 1` 夹紧是否在 `dose = 0` 上被违反？

★★ 本脚本是**诊断**，不是判据。它的全部结论都必须标注「非判据」。
   触发原因：正式轮 `J124-2`（`dose = 0`）给出 `m = 248`，
   而 `u >= 1` 下的**可达上限**是 `207·(2/3) + 180·(1/3) = 198` ⇒ `m > 上限`。
   ⇒ `solve_u` 被 `mech123.solve_u` 的 `if m >= hi_val: return 1.0` **夹到边界**，
     零分布 = `θ(m)` 在 `u = 1` 处取值 ⇒ **零模型不是产生数据的模型**。

做四件事（全部可机械核）：
  D1  逐层列出 `m_s` 与 `u = 1` 的可达上限 `hi_s`，判定"边界是否被激活"。
  D2  **模型无关**地算 `ŵ₁ / ŵ₂`（`DIR-123` §4 的描述性估计量，**不用 `solve_u`**），
      比较两个 `dose`。★ 这是本诊断唯一能回答「有没有同向不对称」的量。
  D3  在 `u < 1` 上做**合成水平检验**：若零模型用 `u = 1` 边界而数据由 `u = 0.55` 产生，
      拒绝率是否远大于 `α` ⇒ 证明 `J124-2` 的拒绝是**夹紧假象**。
  D4  在 `u >= 1` 上做同样的水平检验（`dose = 600` 的情形）⇒ 应当 ≈ `α`（校准正常）。
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "dir123"), os.path.join(ROOT, "dir122")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mech124 as M4                                             # noqa: E402
import mech123 as M3                                             # noqa: E402

OUT = os.path.join(HERE, "out")


def hi_val(n1, n2):
    """`u = 1` 处的可达上限：`n1·P1(1) + n2·P2(1)`。"""
    return n1 * M3.P1(1.0) + n2 * M3.P2(1.0)


def w_estimates(rows):
    """**模型无关**的每候选加成估计（`DIR-123` §4 的描述性量，不用 `solve_u`）。

    `p̂_k = e_k / N_k`；`ŵ_k = (1-p̂_k)(3-k)/(p̂_k·k)`。
    """
    n1 = sum(r["N1"] for r in rows)
    n2 = sum(r["N2"] for r in rows)
    e1 = sum(r["e1"] for r in rows)
    e2 = sum(r["e2"] for r in rows)
    p1 = e1 / float(n1) if n1 else float("nan")
    p2 = e2 / float(n2) if n2 else float("nan")
    w1 = (1.0 - p1) * 2.0 / (p1 * 1.0) if p1 > 0 else float("inf")
    w2 = (1.0 - p2) * 1.0 / (p2 * 2.0) if p2 > 0 else float("inf")
    return {"N1": n1, "N2": n2, "e1": e1, "e2": e2,
            "p1": p1, "p2": p2, "w1": w1, "w2": w2,
            "ratio": (w2 / w1 if w1 else float("inf")),
            "OR": ((p1 / (1 - p1)) / (p2 / (1 - p2))) if 0 < p2 < 1 and 0 < p1 < 1 else None}


def d1_boundary():
    print("\n=== D1 逐层 `m_s` vs `u = 1` 可达上限 `hi_s` ===")
    out = {}
    for dose in (M4.DOSE_MAIN, M4.DOSE_PLACEBO):
        ctx = M4.build_ctx(dose)
        rows = M4.stratum_obs(ctx, M4.strata_of(ctx, M4.AXIS))
        act = []
        print("  dose=%d：" % dose)
        for r in rows:
            h = hi_val(r["N1"], r["N2"])
            hit = r["m"] >= h
            act.append(hit)
            print("    %-22s N1=%3d N2=%3d m=%3d  hi=%7.3f  边界激活=%s"
                  % (r["key"], r["N1"], r["N2"], r["m"], h, hit))
        out[dose] = {"active": act, "n_active": sum(act), "n": len(act),
                     "m_tot": sum(r["m"] for r in rows),
                     "hi_tot": sum(hi_val(r["N1"], r["N2"]) for r in rows)}
        print("    ⇒ 边界激活层数 %d/%d，Σm = %d，Σhi = %.2f"
              % (out[dose]["n_active"], out[dose]["n"], out[dose]["m_tot"],
                 out[dose]["hi_tot"]))
    return out


def d2_model_free():
    print("\n=== D2 **模型无关**的每候选加成估计（不用 solve_u）===")
    out = {}
    for dose in (M4.DOSE_MAIN, M4.DOSE_PLACEBO):
        ctx = M4.build_ctx(dose)
        rows = M4.stratum_obs(ctx, M4.strata_of(ctx, M4.AXIS))
        e = w_estimates(rows)
        out[dose] = e
        print("  dose=%d：N1=%d N2=%d e1=%d e2=%d"
              % (dose, e["N1"], e["N2"], e["e1"], e["e2"]))
        print("           p̂1=%.6f p̂2=%.6f  ŵ1=%.6f ŵ2=%.6f  ŵ2/ŵ1=%.6f  OR=%s"
              % (e["p1"], e["p2"], e["w1"], e["w2"], e["ratio"],
                 "None" if e["OR"] is None else "%.6f" % e["OR"]))
    return out


def d3_level_under_u_below_1(u_true, reps, seed):
    """在 `u_true < 1`（共同 `u`，即 `H124` **为真**）下跑 `strat_test` 的拒绝率。

    ★ 若零模型在 `u = 1` 边界上被夹紧，则此处拒绝率应远大于 `alpha` ⇒ 假象证明。
    """
    ctx = M4.build_ctx(M4.DOSE_PLACEBO)
    strata = M4.strata_of(ctx, M4.AXIS)
    rng = np.random.default_rng(seed)
    p1 = M3.P1(u_true)
    p2 = M3.P2(u_true)                 # ratio = 1 ⇒ 共同 u ⇒ `H124` 为真
    E1 = np.empty((reps, len(strata)), dtype=np.int64)
    E2 = np.empty((reps, len(strata)), dtype=np.int64)
    for j, s in enumerate(strata):
        E1[:, j] = rng.binomial(int(s["N1"]), p1, size=reps)
        E2[:, j] = rng.binomial(int(s["N2"]), p2, size=reps)
    hit = 0
    clamped = 0
    for i in range(reps):
        rows = []
        for j, s in enumerate(strata):
            e1 = int(E1[i, j])
            e2 = int(E2[i, j])
            rows.append({"key": s["key"], "N1": s["N1"], "N2": s["N2"],
                         "e1": e1, "e2": e2, "m": e1 + e2})
        # 统计"边界被激活"的层
        clamped += sum(1 for r in rows
                       if r["m"] > 0 and r["m"] >= hi_val(r["N1"], r["N2"]))
        r = M4.strat_test(rows, alpha=M4.ALPHA, side=M4.SIDE)
        if r["executable"] and r["reject"]:
            hit += 1
    return {"u_true": u_true, "reject_rate": hit / float(reps),
            "clamped_strata_frac": clamped / float(reps * len(strata))}


def d4_level_under_u_above_1(u_true, reps, seed):
    """`dose = 600` 结构、`u >= 1`、`ratio = 1` ⇒ 拒绝率应 ≈ `alpha`（校准正常）。"""
    ctx = M4.build_ctx(M4.DOSE_MAIN)
    strata = M4.strata_of(ctx, M4.AXIS)
    rng = np.random.default_rng(seed)
    p1 = M3.P1(u_true)
    p2 = M3.P2(u_true)
    E1 = np.empty((reps, len(strata)), dtype=np.int64)
    E2 = np.empty((reps, len(strata)), dtype=np.int64)
    for j, s in enumerate(strata):
        E1[:, j] = rng.binomial(int(s["N1"]), p1, size=reps)
        E2[:, j] = rng.binomial(int(s["N2"]), p2, size=reps)
    hit = 0
    for i in range(reps):
        rows = [{"key": s["key"], "N1": s["N1"], "N2": s["N2"],
                 "e1": int(E1[i, j]), "e2": int(E2[i, j]),
                 "m": int(E1[i, j]) + int(E2[i, j])}
                for j, s in enumerate(strata)]
        r = M4.strat_test(rows, alpha=M4.ALPHA, side=M4.SIDE)
        if r["executable"] and r["reject"]:
            hit += 1
    return {"u_true": u_true, "reject_rate": hit / float(reps)}


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 78)
    print("DIR-124 诊断：solve_u 的 u >= 1 夹紧是否在 dose = 0 上被违反？")
    print("★ 本脚本是**诊断**，不是判据。结论一律标「非判据」。")
    print("=" * 78)

    b = d1_boundary()
    w = d2_model_free()

    print("\n=== D3 合成水平：`u_true < 1`（`dose = 0` 结构，`ratio = 1` ⇒ `H124` 为真）===")
    print("    ★ 若拒绝率 ≫ alpha = 0.05 ⇒ `J124-2` 的拒绝是**夹紧假象**")
    d3 = {}
    for u in (0.55, 0.8):
        res = d3_level_under_u_below_1(u, reps=2000, seed=20261241)
        d3[u] = res
        print("    u_true=%.2f  拒绝率=%.4f  边界激活层占比=%.4f"
              % (u, res["reject_rate"], res["clamped_strata_frac"]))

    print("\n=== D4 合成水平：`u_true >= 1`（`dose = 600` 结构，`ratio = 1`）===")
    print("    ★ 此处边界不激活 ⇒ 拒绝率应 ≈ alpha（校准正常）")
    d4 = {}
    for u in (2.0, 9.2):
        res = d4_level_under_u_above_1(u, reps=2000, seed=20261242)
        d4[u] = res
        print("    u_true=%.2f  拒绝率=%.4f" % (u, res["reject_rate"]))

    # ---------------------------------------------------------------- 判定
    print("\n" + "=" * 78)
    print("★★ 诊断结论（非判据）")
    print("=" * 78)
    v1 = b[M4.DOSE_PLACEBO]["n_active"] == b[M4.DOSE_PLACEBO]["n"]
    v2 = b[M4.DOSE_MAIN]["n_active"] == 0
    v3 = d3[0.55]["reject_rate"] > 3 * M4.ALPHA
    v4 = all(abs(r["reject_rate"] - M4.ALPHA) < 0.03 for r in d4.values())
    print("  D1  dose=0 全部 9 层边界激活：%s（%d/%d）"
          % (v1, b[M4.DOSE_PLACEBO]["n_active"], b[M4.DOSE_PLACEBO]["n"]))
    print("  D1  dose=600 无层边界激活：%s（%d/%d）"
          % (v2, b[M4.DOSE_MAIN]["n_active"], b[M4.DOSE_MAIN]["n"]))
    print("  D3  合成 u_true=0.55 时拒绝率 %.4f ≫ α=%.2f：%s"
          % (d3[0.55]["reject_rate"], M4.ALPHA, v3))
    print("  D4  u_true >= 1 时拒绝率 ≈ α：%s（%s）"
          % (v4, {k: round(r["reject_rate"], 4) for k, r in d4.items()}))
    print()
    print("  ŵ 比较（D2，模型无关）：")
    for dose in (M4.DOSE_MAIN, M4.DOSE_PLACEBO):
        e = w[dose]
        print("    dose=%-4d ŵ1=%.4f  ŵ2=%.4f  ŵ2/ŵ1=%.4f"
              % (dose, e["w1"], e["w2"], e["ratio"]))
    print()
    print("  ⇒ 若 D1(dose=0 全激活) ∧ D3(假象证实) 同时成立：")
    print("     `J124-2` 的拒绝**不可解释为混杂**（零模型被违反）。")
    print("     `dose=0` 的 ŵ2/ŵ1 ≈ %.2f ⇒ **无同向不对称**（同向需 ≫ 1）。"
          % w[M4.DOSE_PLACEBO]["ratio"])

    json.dump({"boundary": {str(k): v for k, v in b.items()},
               "w_model_free": {str(k): v for k, v in w.items()},
               "D3_u_below_1": {str(k): v for k, v in d3.items()},
               "D4_u_at_least_1": {str(k): v for k, v in d4.items()},
               "verdict_flags": {"placebo_all_clamped": v1,
                                 "main_none_clamped": v2,
                                 "artifact_confirmed": v3,
                                 "calibrated_above_1": v4}},
              open(os.path.join(OUT, "diag_clamp124.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n→ out/diag_clamp124.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())