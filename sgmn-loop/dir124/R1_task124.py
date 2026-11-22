#!/usr/bin/env python3
"""`DIR-124` 主脚本 —— Stage A 硬门 + 正式轮。

用法：
  /opt/pytorch/bin/python dir124/R1_task124.py --stage a        # Stage A（12 条守卫）
  /opt/pytorch/bin/python dir124/R1_task124.py --stage formal   # 正式轮（须先过 Stage A）

★ 本卡的两个判据（卡片 rev 1.1 冻结）：
  `J124-1` 主检：`dose = 600`（四题集）上的**按 seed 分层**精确条件检验，双侧，`α = 0.05`
  `J124-2` 负对照：`dose = 0`（**仅题集 555**）上的**同一统计量**，双侧，`α = 0.05`
★ 判决表 5 行见卡片 §2-3。★ 第 1 行的「归因」**一律不可判**（负对照功效 0.2487，卡片 §2-5b）。
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "dir123"), os.path.join(ROOT, "dir122")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mech124 as M4                                             # noqa: E402
import mech123 as M3                                             # noqa: E402
import mech122 as M0                                             # noqa: E402

OUT = os.path.join(HERE, "out")
CHECKS = []


def rec(tag, ok, msg=""):
    CHECKS.append({"tag": tag, "ok": bool(ok), "msg": msg})
    print("%-10s %s  %s" % (tag, "PASS" if ok else "FAIL", msg))
    return ok


# ================================================================ 判据封装
def judge(dose, axis=M4.AXIS):
    """跑一次判据，返回全量读数（含 report-only 的另两轴留给调用方）。"""
    ctx = M4.build_ctx(dose)
    rows = M4.stratum_obs(ctx, M4.strata_of(ctx, axis))
    r = M4.strat_test(rows, axis=axis, alpha=M4.ALPHA, side=M4.SIDE)
    r["dose"] = dose
    r["rngs"] = list(ctx["rngs"])
    r["rows"] = rows
    return r


def sensitivity(dose):
    """report-only：另两轴的结果（**不参与判决**，避免多重性）。"""
    ctx = M4.build_ctx(dose)
    out = {}
    for axis in M4.AXES_CANDIDATES:
        rows = M4.stratum_obs(ctx, M4.strata_of(ctx, axis))
        r = M4.strat_test(rows, axis=axis, alpha=M4.ALPHA, side=M4.SIDE)
        out[axis] = {k: r[k] for k in ("executable", "S", "m", "strata_used",
                                       "p_two", "p_upper", "reject",
                                       "direction", "expect")}
    return out


def decide(j1, j2):
    """★ 判决表（卡片 rev 1.1 §2-3）。返回 `(row, verdict, attribution)`。"""
    if not j1.get("executable"):
        return 4, "H124_INEXECUTABLE", "不可判"
    if j1["direction"] == "low" and j1["reject"]:
        return 5, "H124_DIRECTION_FALSIFIED", "不可判"
    if not j1["reject"]:
        return 3, "H124_NOT_SUPPORTED", "不可判"
    # `J124-1` 拒绝且方向 high
    if not j2.get("executable"):
        return 1, "H124_CONFIRMED_PLACEBO_INEXECUTABLE", "不可判"
    if j2["reject"] and j2["direction"] == "high":
        return 2, "H124_CONFOUNDED", "不可判"
    # ★ 第 1 行：`H124` 成立，但**归因一律不可判**（负对照功效 0.2487，卡片 §2-5b）
    return 1, "H124_CONFIRMED_ATTRIBUTION_UNDECIDABLE", "不可判"


# ================================================================ Stage A
def stage_a():
    print("=" * 78)
    print("DIR-124 Stage A —— 12 条守卫（任一条 FAIL 即阻止正式轮）")
    print("=" * 78)

    ctx6 = M4.build_ctx(M4.DOSE_MAIN)
    ctx0 = M4.build_ctx(M4.DOSE_PLACEBO)

    # A124-1 数据完整性
    seeds = [c["seed"] for c in ctx6["cells"]]
    rec("A124-1", len(ctx6["cells"]) == 9 and len(set(seeds)) == 9
        and all(c["dose"] == 600 for c in ctx6["cells"]),
        "dose=600：9 cell、seed 唯一、dose 全 600（%s）" % (sorted(seeds),))

    # A124-2 与 D-154 回归一致
    rows6 = M4.stratum_obs(ctx6, M4.strata_of(ctx6, "seed"))
    e1 = sum(r["e1"] for r in rows6)
    e2 = sum(r["e2"] for r in rows6)
    rec("A124-2", (e1, e2, e1 + e2) == (109, 24, 133),
        "dose=600 观测 = D-154 已发表：e1=%d、e2=%d、m=%d（须 109/24/133）"
        % (e1, e2, e1 + e2))

    # A124-3 负对照数据完整
    ok3 = (len(ctx0["cells"]) == 9 and all(c["dose"] == 0 for c in ctx0["cells"])
           and all(all(len(c["picks"][r]) == len(ctx0["items"][r])
                       for r in ctx0["rngs"]) for c in ctx0["cells"]))
    rec("A124-3", ok3,
        "dose=0：9 cell、dose 全 0、pick_argmin 长度对（可用题集 %s）"
        % (list(ctx0["rngs"]),))

    # A124-4 分层机会数不丢
    ok4 = True
    det = {}
    for ctx, want in ((ctx6, (612, 891)), (ctx0, (207, 180))):
        for axis in M4.AXES_CANDIDATES:
            st = M4.strata_of(ctx, axis)
            got = (sum(s["N1"] for s in st), sum(s["N2"] for s in st))
            det[(ctx["cells"][0]["dose"], axis)] = got
            if got != want:
                ok4 = False
    rec("A124-4", ok4,
        "分层不丢机会数：dose=600 须 (612,891)、dose=0 须 (207,180) ⇒ %s" % (det,))

    # A124-5 卷积归一
    st = M4.strata_of(ctx6, "seed")
    ms = [max(1, s["N1"] // 4) for s in st]
    ths = [M3.theta_of(m, s["N1"], s["N2"]) for m, s in zip(ms, st)]
    ok5 = all(t is not None for t in ths)
    pmf = M4.convolve_null(ms, ths) if ok5 else None
    rec("A124-5", ok5 and pmf is not None
        and abs(float(pmf.sum()) - 1.0) < 1e-12
        and len(pmf) == sum(ms) + 1,
        "卷积零分布归一 = %.15f（长度 %d = Σm+1）"
        % (float(pmf.sum()) if pmf is not None else -1, len(pmf) if pmf is not None else -1))

    # A124-6 工作例逐位复现
    wex = M4.strat_test([{"key": "A", "N1": 4, "N2": 5, "e1": 3, "e2": 0},
                         {"key": "B", "N1": 6, "N2": 3, "e1": 1, "e2": 0}])
    rec("A124-6", wex["S"] == 4 and abs(wex["p_two"] - 0.5737934260) < 1e-9
        and abs(wex["p_upper"] - 0.2598205554) < 1e-9
        and wex["direction"] == "high",
        "工作例复现：S=4、p_two=%.10f、p_upper=%.10f、dir=%s"
        % (wex["p_two"], wex["p_upper"], wex["direction"]))

    # A124-7 k=3 层完全未使用（结构守卫）
    ok7 = all(set(r.keys()) == {"key", "N1", "N2", "e1", "e2", "m"} for r in rows6)
    rec("A124-7", ok7,
        "判据输入只有 e1/e2（无 e3）⇒ k=3 层未被使用（CORR-35 ① 守卫）")

    # A124-8 m 不一致 ⇒ 抛错
    try:
        M4.strat_test([{"key": "Z", "N1": 4, "N2": 5, "e1": 3, "e2": 0, "m": 99}])
        ok8 = False
    except ValueError:
        ok8 = True
    rec("A124-8", ok8, "m 与 e1+e2 不一致时抛 ValueError（接线错误守卫）")

    # A124-9 冻结量自检
    rec("A124-9", M4.SIDE == "two" and M4.AXIS == "seed"
        and M4.ALPHA == 0.05 and M4.ALPHA_PLACEBO == 0.05
        and M4.DOSE_MAIN == 600 and M4.DOSE_PLACEBO == 0,
        "冻结量：SIDE=%r AXIS=%r ALPHA=%s ALPHA_PLACEBO=%s DOSE=%d/%d"
        % (M4.SIDE, M4.AXIS, M4.ALPHA, M4.ALPHA_PLACEBO,
           M4.DOSE_MAIN, M4.DOSE_PLACEBO))

    # A124-10 合成池上判据真被计算过（非 None、非恒真恒假）
    oks = set()
    p_ok = True
    for e1a in range(0, 5):
        for e1b in range(0, 7):
            r = M4.strat_test([{"key": "A", "N1": 4, "N2": 5, "e1": e1a, "e2": 2},
                               {"key": "B", "N1": 6, "N2": 3, "e1": e1b, "e2": 1}])
            if r["executable"]:
                oks.add(r["reject"])
                p_ok &= r["p_two"] is not None and 0.0 <= r["p_two"] <= 1.0
    rec("A124-10", p_ok and oks == {True, False},
        "合成池 35 组：reject 出现过 %s（须 True 与 False 都有）、p 全部有效"
        % (sorted(oks),))

    # A124-11 负对照确为未训练态
    rec("A124-11", all(bool(c["raw"].get("W_bitwise_same")) for c in ctx0["cells"])
        and len(ctx0["rngs"]) == 1,
        "dose=0 全部 W_bitwise_same=True（确为 W=W⁰）、可用题集 = %s"
        % (list(ctx0["rngs"]),))

    # A124-12 功效在 Stage A 里再算一次（小 reps，可复核）
    pw6 = M4.power(axis="seed", u=9.2, ratio=2.0, reps=6000)
    pw0 = M4.power(axis="seed", u=9.2, ratio=2.0, reps=6000, dose=0)
    rec("A124-12", pw6 >= M4.POWER_MIN and pw0 < pw6,
        "功效复算（6000 次 MC）：主检 %.4f ≥ %.2f；负对照 %.4f < 主检（结构决定）"
        % (pw6, M4.POWER_MIN, pw0))

    n_ok = sum(1 for c in CHECKS if c["ok"])
    print("=" * 78)
    print("Stage A：%d / %d PASS%s"
          % (n_ok, len(CHECKS), "" if n_ok == len(CHECKS) else "  ★ 有 FAIL ⇒ 禁止正式轮"))
    print("=" * 78)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"checks": CHECKS, "all_ok": n_ok == len(CHECKS),
               "power_recheck": {"dose600": pw6, "dose0": pw0}},
              open(os.path.join(OUT, "stage_stageA.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0 if n_ok == len(CHECKS) else 1


# ================================================================ 正式轮
def formal():
    print("=" * 78)
    print("DIR-124 正式轮 —— 每候选加成随见到个数增长吗？（分层检验 + 未训练负对照）")
    print("=" * 78)

    j1 = judge(M4.DOSE_MAIN)
    j2 = judge(M4.DOSE_PLACEBO)
    row, verdict, attribution = decide(j1, j2)

    print("\n--- 结构 ---")
    for tag, j in (("J124-1", j1), ("J124-2", j2)):
        print("  %s  dose=%d  题集=%s  层数(有效)=%s"
              % (tag, j["dose"], j["rngs"],
                 j.get("strata_used")))
    print("\n--- 分层读数（seed 轴，逐层）---")
    for tag, j in (("J124-1 dose=600", j1), ("J124-2 dose=0", j2)):
        print("  %s:" % tag)
        print("    %s" % ("  ".join("%s:N1=%d,N2=%d,e1=%d,e2=%d"
                                    % (r["key"].split("=")[-1], r["N1"], r["N2"],
                                       r["e1"], r["e2"]) for r in j["rows"])))
        print("    S=%s  m=%s  E[S]=%.4f  p_two=%s  p_upper=%s  reject=%s  dir=%s"
              % (j["S"], j["m"], j["expect"],
                 "None" if j["p_two"] is None else "%.10g" % j["p_two"],
                 "None" if j["p_upper"] is None else "%.10g" % j["p_upper"],
                 j["reject"], j["direction"]))

    print("\n--- report-only：另两轴（不参与判决）---")
    sens = {"600": sensitivity(M4.DOSE_MAIN), "0": sensitivity(M4.DOSE_PLACEBO)}
    for dk, sv in sens.items():
        for axis, r in sv.items():
            print("  dose=%s axis=%-5s p_two=%s reject=%s dir=%s"
                  % (dk, axis,
                     "None" if r["p_two"] is None else "%.10g" % r["p_two"],
                     r["reject"], r["direction"]))

    print("\n" + "=" * 78)
    print("★★ 判决：第 %d 行 → %s（归因：%s）" % (row, verdict, attribution))
    print("=" * 78)

    os.makedirs(OUT, exist_ok=True)
    json.dump({"row": row, "verdict": verdict, "attribution": attribution,
               "J124_1": {k: v for k, v in j1.items() if k != "rows"},
               "J124_2": {k: v for k, v in j2.items() if k != "rows"},
               "rows_main": j1["rows"], "rows_placebo": j2["rows"],
               "sensitivity": sens,
               "frozen": {"axis": M4.AXIS, "side": M4.SIDE, "alpha": M4.ALPHA,
                          "alpha_placebo": M4.ALPHA_PLACEBO,
                          "dose_main": M4.DOSE_MAIN,
                          "dose_placebo": M4.DOSE_PLACEBO,
                          "mc_reps": M4.MC_REPS, "mc_seed": M4.MC_SEED,
                          "power_min": M4.POWER_MIN}},
              open(os.path.join(OUT, "formal.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("→ out/formal.json")
    return 0


def main():
    stage = "a"
    if "--stage" in sys.argv:
        stage = sys.argv[sys.argv.index("--stage") + 1].lower()
    if stage in ("a", "stagea"):
        return stage_a()
    if stage == "formal":
        # ★ 硬门：Stage A 必须已有全绿记录
        f = os.path.join(OUT, "stage_stageA.json")
        if not os.path.exists(f):
            print("★ 拒绝：Stage A 尚未运行（缺 out/stage_stageA.json）")
            return 2
        sa = json.load(open(f, encoding="utf-8"))
        if not sa.get("all_ok"):
            print("★ 拒绝：Stage A 未全绿 ⇒ 禁止正式轮")
            return 2
        return formal()
    print("未知 --stage %r" % (stage,))
    return 2


if __name__ == "__main__":
    sys.exit(main())