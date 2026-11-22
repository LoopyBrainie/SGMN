#!/usr/bin/env python
"""DIR-110 : 离线重判 DIR-109 的 24 格 —— 两通路复合是「串联最弱环」还是「可加求和」？

★ 零新格：只读 ../dir109/out/stage_verdict.json（md5 a20a7222450217bc1eb6717368fc5036）。
★ 两套裁决并存：
    rev 1.0（卡 DIRECTION_rev110_v10.md）—— 主判据用 TESTABLE（TAU_SAT 硬门）
    rev 1.1（卡 DIRECTION_rev110_v11.md §9）—— 主判据用 ACTIVE（两臂都作用），新增「判据不稳」码
★ 阈值与判据文字按卡冻结；本文件不得改阈值。
★ 只 import 标准库（无新格要跑）。
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "dir109" / "out" / "stage_verdict.json"
OUT = HERE / "out"

ARMS = ("T-NONE", "T-Q", "T-O", "T-BOTH")
GAMS = (0.005, 0.05)
SEEDS_WANT = [20261028, 20261104, 7]
JUDGE_LAYERS = ("8", "16")
OBJS = ("q", "m")
METS = ("rank_eff", "cos_mean")
TAU_ARM = 0.10
TAU_SAT_MAIN = 0.05
TAU_SAT_SWEEP = (0.02, 0.05, 0.10, 0.20)
TAU_ADD_SWEEP = (0.10, 0.15, 0.20)
EPS = 1e-12

G = []


def guard(gid, ok, detail):
    G.append({"id": gid, "ok": bool(ok), "detail": detail})
    print(("  PASS  " if ok else "  **FAIL** ") + gid + "  " + str(detail)[:190])


def main():
    print("=" * 78)
    print("DIR-110 Stage A —— 复用 DIR-109 的 24 格前，先证明它完好、且我读的是被判的那份")
    print("=" * 78)
    if not SRC.exists():
        print("  源文件不存在:", SRC)
        return 2
    d = json.loads(SRC.read_text())
    _raw = d["cells"]
    cells = list(_raw.values()) if isinstance(_raw, dict) else list(_raw)
    idx = {(c["arm"], float(c["gamma0"]), int(c["seed"])): c for c in cells}

    # ---------------- A110-1 ----------------
    ok1 = (len(cells) == 24 and d.get("plan_n") == 24
           and all(bool(c.get("finite_params")) for c in cells)
           and all(int(c.get("eval_writes", -1)) == 0 for c in cells))
    guard("A110-1", ok1, "cells=%d plan_n=%s finite_params 全真=%s eval_writes 全0=%s" % (
        len(cells), d.get("plan_n"),
        all(bool(c.get("finite_params")) for c in cells),
        all(int(c.get("eval_writes", -1)) == 0 for c in cells)))

    # ---------------- A110-3 ----------------
    seeds_ok = sorted(d.get("seeds", [])) == sorted(SEEDS_WANT)
    complete = all((a, g, s) in idx for a in ARMS for g in GAMS for s in SEEDS_WANT)
    guard("A110-3", seeds_ok and complete,
          "seeds=%s 三 seed 齐全=%s" % (d.get("seeds"), complete))

    # ---------------- A110-4 ----------------
    tag = str(d.get("dir", "")) + "|" + str(d.get("card", ""))
    guard("A110-4", ("DIR-109" in tag) and bool(d.get("device_md5")),
          "指向 DIR-109=%s device_md5=%s" % ("DIR-109" in tag, str(d.get("device_md5"))[:16]))

    # ---------------- A110-5 ----------------
    guard("A110-5", abs(float(d.get("read_lr", -1)) - 0.01) < 1e-12,
          "read_lr=%s（dir90 系必须 0.01）" % d.get("read_lr"))

    # ---------------- A110-2 : 重算 DIR-109 的 acc 2x2 ----------------
    crit = d.get("crit", {})
    rows = crit.get("rows", {})
    worst = 0.0
    detail2 = []
    for g in GAMS:
        acc = {a: st.mean([float(idx[(a, g, s)]["acc_item"]) for s in SEEDS_WANT]) for a in ARMS}
        D_Q = acc["T-NONE"] - acc["T-Q"]
        D_O = acc["T-NONE"] - acc["T-O"]
        D_B = acc["T-NONE"] - acc["T-BOTH"]
        I = D_B - D_Q - D_O
        r = rows.get(str(g), {})
        pairs = [("acc/NONE", acc["T-NONE"], r.get("acc", {}).get("T-NONE")),
                 ("acc/Q", acc["T-Q"], r.get("acc", {}).get("T-Q")),
                 ("acc/O", acc["T-O"], r.get("acc", {}).get("T-O")),
                 ("acc/BOTH", acc["T-BOTH"], r.get("acc", {}).get("T-BOTH")),
                 ("D_Q", D_Q, r.get("D_Q")), ("D_O", D_O, r.get("D_O")),
                 ("D_B", D_B, r.get("D_B")), ("I", I, r.get("I"))]
        for nm, mine, frozen in pairs:
            if frozen is None:
                worst = float("inf")
                detail2.append("%s/%s 冻结缺" % (g, nm))
            else:
                dd = abs(mine - float(frozen))
                worst = max(worst, dd)
                if dd > EPS:
                    detail2.append("%s/%s 差 %.3e" % (g, nm, dd))
    guard("A110-2", worst <= EPS, "重算 vs 冻结 crit 最大绝对差 = %.3e（阈值 1e-12）%s" % (
        worst, "；" + "；".join(detail2[:4]) if detail2 else "；逐位一致"))
    print("   冻结判决 = %s ；逐 γ = %s" % (crit.get("verdict"), crit.get("codes_per_gamma")))

    # ---------------- A110-6 ----------------
    bad = []
    for (a, g, s), c in idx.items():
        for L in ("8", "12", "16"):
            for o in ("q", "m", "c"):
                node = c["state"][L][o]
                for m in ("rank_eff", "cos_mean", "rho"):
                    v = float(node[m])
                    if not math.isfinite(v):
                        bad.append((a, g, s, L, o, m, "nan"))
                if float(node["rank_eff"]) < 1.0 - 1e-6:
                    bad.append((a, g, s, L, o, "rank<1", node["rank_eff"]))
                if float(node["cos_mean"]) > 1.0 + 1e-6:
                    bad.append((a, g, s, L, o, "cos>1", node["cos_mean"]))
    guard("A110-6", not bad, "27 个 state 字段有限且方向正确；违例 %d %s" % (len(bad), bad[:3]))

    # ---------------- A110-7 ----------------
    worst7 = 0.0
    for g in GAMS:
        for m in ("rank_eff", "cos_mean", "rho"):
            vs = [float(idx[(a, g, s)]["state"]["8"]["c"][m]) for a in ARMS for s in SEEDS_WANT]
            worst7 = max(worst7, max(vs) - min(vs))
    guard("A110-7", worst7 < 1e-6,
          "c@L8 跨四臂×三 seed 的 max−min = %.3e < 1e-6 ⇒ L8 在任何注入之前" % worst7)

    # ==================== 点与量 ====================
    def X(a, g, L, o, m, s=None):
        if s is None:
            return st.mean([float(idx[(a, g, ss)]["state"][L][o][m]) for ss in SEEDS_WANT])
        return float(idx[(a, g, s)]["state"][L][o][m])

    def point(g, L, o, m):
        xn = X("T-NONE", g, L, o, m)
        span = 1.0 - xn
        if abs(span) < 1e-9:
            return None
        dq = (X("T-Q", g, L, o, m) - xn) / span
        do = (X("T-O", g, L, o, m) - xn) / span
        db = (X("T-BOTH", g, L, o, m) - xn) / span
        xb = X("T-BOTH", g, L, o, m)
        sat = abs(xb - 1.0) / abs(span)
        per = []
        for s in SEEDS_WANT:
            sn = X("T-NONE", g, L, o, m, s)
            sp = 1.0 - sn
            per.append((X("T-BOTH", g, L, o, m, s) - sn) / sp
                       - (X("T-Q", g, L, o, m, s) - sn) / sp
                       - (X("T-O", g, L, o, m, s) - sn) / sp)
        return {"gamma": g, "layer": L, "obj": o, "met": m,
                "X_NONE": xn, "X_BOTH": xb, "span": span,
                "dQ": dq, "dO": do, "dB": db, "I": db - dq - do,
                "sat": sat, "I_per_seed": per,
                "I_sd": st.stdev(per) if len(per) > 1 else 0.0,
                "max_err": abs(db - max(dq, do)), "sum_err": abs(db - (dq + do))}

    def classify(p, tau_sat):
        if abs(p["dQ"]) < TAU_ARM or abs(p["dO"]) < TAU_ARM:
            return "DEGENERATE"
        if p["sat"] < tau_sat:
            return "SATURATED"
        return "TESTABLE"

    def active(p):
        return abs(p["dQ"]) >= TAU_ARM and abs(p["dO"]) >= TAU_ARM

    points = [(o, m, L) for o in OBJS for m in METS for L in JUDGE_LAYERS]
    guard("A110-8", len(points) == 8,
          "被测点 = 4 量 × 2 层 = %d 个，每个在 2 个 γ 上求值（16 行），无按结果筛除" % len(points))

    tab = []
    for (o, m, L) in points:
        for g in GAMS:
            p = point(g, L, o, m)
            if p is None:
                tab.append({"gamma": g, "layer": L, "obj": o, "met": m, "skip": True})
            else:
                p["cls"] = classify(p, TAU_SAT_MAIN)
                p["active"] = active(p)
                tab.append(p)

    print("\n" + "=" * 126)
    print("§2.3 点表（8 点 × 2 γ = 16 行；q/m × rank_eff/cos_mean × L8/L16）")
    print("=" * 126)
    hdr = "%-5s %-9s %-6s %-8s %-8s %-7s %-7s %-7s %-7s %-6s %-11s %-6s %-8s %-8s %-7s"
    print(hdr % ("obj", "met", "γ", "X_NONE", "X_BOTH", "δ_Q", "δ_O", "δ_B", "I_rel",
                 "sat", "rev1.0分类", "ACTIVE", "MAXerr", "SUMerr", "MAX/SUM"))
    for p in tab:
        if p.get("skip"):
            print("%-5s %-9s %-6s (span=0 跳过)" % (p["obj"], p["met"], p["gamma"]))
            continue
        ratio = (p["max_err"] / p["sum_err"]) if p["sum_err"] > 1e-12 else float("nan")
        print(hdr % (p["obj"], p["met"], p["gamma"], "%.3f" % p["X_NONE"], "%.3f" % p["X_BOTH"],
                     "%.3f" % p["dQ"], "%.3f" % p["dO"], "%.3f" % p["dB"], "%.3f" % p["I"],
                     "%.3f" % p["sat"], classify(p, TAU_SAT_MAIN), str(p["active"]),
                     "%.3f" % p["max_err"], "%.3f" % p["sum_err"], "%.3f" % ratio))

    # ==================== rev 1.0 裁决（TAU_SAT 硬门） ====================
    sds_all = [p["I_sd"] for p in tab if not p.get("skip")]
    sd_all = math.sqrt(sum(s * s for s in sds_all) / len(sds_all)) if sds_all else 0.0
    tau_add_v10 = max(0.10, 3.0 * sd_all)

    def verdict_testable(tau_sat, tau_add):
        per_g = {}
        for g in GAMS:
            ts = [p for p in tab if (not p.get("skip")) and p["gamma"] == g and classify(p, tau_sat) == "TESTABLE"]
            if not ts:
                per_g[g] = {"code": "C110-NO-TESTABLE-POINT", "n": 0}
            elif any(abs(p["I"]) < tau_add for p in ts):
                hits = ["%s.%s@L%s I=%.3f" % (p["obj"], p["met"], p["layer"], p["I"])
                        for p in ts if abs(p["I"]) < tau_add]
                per_g[g] = {"code": "C110-ADDITIVE-HOLDS", "n": len(ts), "at": hits}
            elif all(p["max_err"] < p["sum_err"] for p in ts):
                per_g[g] = {"code": "C110-SERIES-WEAKEST-LINK", "n": len(ts)}
            else:
                per_g[g] = {"code": "C110-MIXED", "n": len(ts)}
        codes = {v["code"] for v in per_g.values()}
        return per_g, (codes.pop() if len(codes) == 1 else "C110-MIXED")

    per_g10, ov10 = verdict_testable(TAU_SAT_MAIN, tau_add_v10)
    sweep10 = []
    for ts in TAU_SAT_SWEEP:
        for ta in TAU_ADD_SWEEP:
            pg, ov = verdict_testable(ts, ta)
            sweep10.append({"tau_sat": ts, "tau_add": ta, "overall": ov,
                            "n": {str(g): pg[g]["n"] for g in GAMS}})
    uniq10 = len({s["overall"] for s in sweep10}) == 1
    guard("A110-9", uniq10 and len(sweep10) == 12,
          "rev1.0：12 组合全产出；总体码唯一=%s ⇒ %s（%s）" % (
              uniq10, ov10, "唯一" if uniq10 else "**不唯一 ⇒ 按 §4 阻断 rev1.0**"))

    print("\n" + "=" * 78)
    print("【rev 1.0】TAU_SAT=%.2f  TAU_ADD=%.4f（SD_pooled 取全 16 行 = %.4f）" % (
        TAU_SAT_MAIN, tau_add_v10, sd_all))
    print("=" * 78)
    for g in GAMS:
        v = per_g10[g]
        print("  γ=%-6s %-28s n=%d %s" % (g, v["code"], v["n"], v.get("at") or ""))
    print("  ⇒ rev1.0 总体：%s ；扫描唯一=%s ⇒ %s" % (
        ov10, uniq10, "可裁决" if uniq10 else "**阻断，rev1.0 无判决**"))

    # ==================== rev 1.1 裁决（ACTIVE 集） ====================
    act_rows = [p for p in tab if (not p.get("skip")) and p["active"]]
    sds_act = [p["I_sd"] for p in act_rows]
    sd_act = math.sqrt(sum(s * s for s in sds_act) / len(sds_act)) if sds_act else 0.0
    tau_add_v11 = max(0.10, 3.0 * sd_act)

    def verdict_active(tau_add):
        per_g = {}
        for g in GAMS:
            ar = [p for p in act_rows if p["gamma"] == g]
            if not ar:
                per_g[g] = {"code": "C110-NO-TESTABLE-POINT", "n": 0}
            elif any(abs(p["I"]) < tau_add for p in ar):
                hits = ["%s.%s@L%s I=%.3f" % (p["obj"], p["met"], p["layer"], p["I"])
                        for p in ar if abs(p["I"]) < tau_add]
                per_g[g] = {"code": "C110-ADDITIVE-HOLDS", "n": len(ar), "at": hits}
            elif all(p["max_err"] < p["sum_err"] for p in ar):
                per_g[g] = {"code": "C110-SERIES-WEAKEST-LINK", "n": len(ar)}
            else:
                per_g[g] = {"code": "C110-MIXED", "n": len(ar),
                            "why": ["%s.%s@L%s %.3f vs %.3f" % (p["obj"], p["met"], p["layer"],
                                                                p["max_err"], p["sum_err"])
                                    for p in ar if not (p["max_err"] < p["sum_err"])]}
        codes = {v["code"] for v in per_g.values()}
        return per_g, (codes.pop() if len(codes) == 1 else "C110-MIXED")

    per_g11, ov11 = verdict_active(tau_add_v11)
    sweep11 = []
    for ta in (tau_add_v11,) + TAU_ADD_SWEEP:
        pg, ov = verdict_active(ta)
        sweep11.append({"tau_add": ta, "overall": ov,
                        "n": {str(g): pg[g]["n"] for g in GAMS}})
    uniq11 = len({s["overall"] for s in sweep11}) == 1
    final11 = "C110-CRITERION-UNSTABLE" if not uniq11 else ov11
    guard("A110-10", len(sweep11) == 1 + len(TAU_ADD_SWEEP),
          "rev1.1：%d 个 TAU_ADD 组合全产出；总体码唯一=%s ⇒ 判据%s" % (
              len(sweep11), uniq11, "稳" if uniq11 else "**不稳（记 C110-CRITERION-UNSTABLE）**"))

    print("\n" + "=" * 96)
    print("【rev 1.1】ACTIVE 集（两臂都作用：|δ_Q|≥%.2f 且 |δ_O|≥%.2f）—— 饱和不再筛点" % (TAU_ARM, TAU_ARM))
    print("=" * 96)
    print("%-5s %-9s %-6s %-7s %-7s %-7s %-8s %-8s %-8s %-6s" % (
        "obj", "met", "γ", "δ_Q", "δ_O", "δ_B", "I_rel", "MAXerr", "SUMerr", "sat"))
    for p in sorted(act_rows, key=lambda x: (x["obj"], x["met"], x["layer"], x["gamma"])):
        print("%-5s %-9s %-6s %-7.3f %-7.3f %-7.3f %-8.3f %-8.3f %-8.3f %-6.3f" % (
            p["obj"], p["met"], p["gamma"], p["dQ"], p["dO"], p["dB"], p["I"],
            p["max_err"], p["sum_err"], p["sat"]))
    print("  ACTIVE 行数 = %d（每 γ %d 个点）; SD_pooled(ACTIVE) = %.4f ⇒ TAU_ADD = %.4f"
          % (len(act_rows), len(act_rows) // 2, sd_act, tau_add_v11))
    print("  MAXerr < SUMerr 的行数 = %d / %d" % (
        sum(1 for p in act_rows if p["max_err"] < p["sum_err"]), len(act_rows)))
    for g in GAMS:
        v = per_g11[g]
        print("  γ=%-6s %-28s n=%d %s" % (g, v["code"], v["n"], v.get("at") or v.get("why") or ""))
    print("\n  rev1.1 扫描（TAU_ADD 变化；TAU_SAT 已不参与）：")
    for s in sweep11:
        print("    TAU_ADD=%.4f ⇒ %-28s n=%s" % (s["tau_add"], s["overall"], s["n"]))
    print("  ⇒ rev1.1 总体：%s（唯一=%s）" % (final11, uniq11))

    # ---- rev1.0 扫描表（留档）----
    print("\n  rev1.0 扫描（TAU_SAT × TAU_ADD，12 组合）—— 留档，用于说明为何 rev1.0 被阻断：")
    for s in sweep10:
        print("    TAU_SAT=%.2f TAU_ADD=%.2f ⇒ %-28s n=%s" % (
            s["tau_sat"], s["tau_add"], s["overall"], s["n"]))

    # ---- L12 报告块 ----
    l12 = []
    for o in OBJS:
        for m in METS:
            for g in GAMS:
                p = point(g, "12", o, m)
                if p:
                    p["cls"] = classify(p, TAU_SAT_MAIN)
                    p["active"] = active(p)
                    l12.append(p)
    print("\nL12 报告块（§5.2：不参与判决）")
    for p in l12:
        print("  %s.%-9s γ=%-6s δ_Q=%-7.3f δ_O=%-7.3f δ_B=%-7.3f I=%-7.3f sat=%-6.3f %s" % (
            p["obj"], p["met"], p["gamma"], p["dQ"], p["dO"], p["dB"], p["I"], p["sat"], p["cls"]))

    # ---- 写盘 ----
    OUT.mkdir(parents=True, exist_ok=True)
    core = [g for g in G if g["id"] != "A110-9"]
    all_pass = all(x["ok"] for x in core) and all(x["ok"] for x in G if x["id"] == "A110-10")
    stageA = {"dir": "DIR-110", "cards": ["rev110_v10", "rev110_v11"],
              "source": str(SRC), "src_md5": "a20a7222450217bc1eb6717368fc5036",
              "guards": G, "n_pass": sum(1 for x in G if x["ok"]), "n_total": len(G),
              "rev10_blocked_by_A110_9": not uniq10,
              "all_pass_under_rev11": all_pass,
              "c_L8_control_max_minus_min": worst7, "crit_identity_max_abs_diff": worst}
    (OUT / "stageA110.json").write_text(json.dumps(stageA, ensure_ascii=False, indent=1))
    print("\nStage A: %d/%d PASS" % (stageA["n_pass"], stageA["n_total"]))
    print("  rev1.0 主判据 ⇒ %s" % ("**被 A110-9 阻断，无判决**" if not uniq10 else "可裁决"))
    print("  rev1.1 主判据 ⇒ %s" % ("可裁决" if all_pass else "**阻断**"))

    uncertified = final11 == "C110-SERIES-WEAKEST-LINK"
    verdict = {
        "dir": "DIR-110", "zero_new_cells": True, "source": str(SRC),
        "tau_arm": TAU_ARM,
        "rev10": {"card": "rev110_v10", "gated_by": "TAU_SAT",
                  "tau_sat": TAU_SAT_MAIN, "tau_add": tau_add_v10, "sd_pooled": sd_all,
                  "per_gamma": {str(g): per_g10[g] for g in GAMS},
                  "overall": ov10, "sweep_unique": uniq10,
                  "adjudicable": uniq10, "sweep": sweep10,
                  "note": "A110-9 FAIL ⇒ 按 rev1.0 §4 阻断，不得据此下判决"},
        "rev11": {"card": "rev110_v11", "gated_by": "ACTIVE(两臂都作用)",
                  "tau_add": tau_add_v11, "sd_pooled_active": sd_act,
                  "n_active_rows": len(act_rows),
                  "n_maxerr_lt_sum": sum(1 for p in act_rows if p["max_err"] < p["sum_err"]),
                  "per_gamma": {str(g): per_g11[g] for g in GAMS},
                  "overall": final11, "sweep_unique": uniq11, "sweep": sweep11},
        "verdict_rev11": final11,
        "final_uncertified": uncertified,
        "final_code": "C110-POSTHOC-WEAKEST-LINK-UNCERTIFIED" if uncertified else final11,
        "posthoc_disclosure": ("rev 1.1 是在看到 rev 1.0 的 Stage A 读数（含全部 8 点 MAXerr/SUMerr）"
                               "之后写的；故其 SERIES-WEAKEST-LINK 不是预注册确认，"
                               "SUCCEEDED 不增，须由 P110-1 用新 seed/新 γ 预注册复现"),
        "points": [{k: v for k, v in p.items() if k != "I_per_seed"} for p in tab],
        "l12_report_only": [{k: v for k, v in p.items() if k != "I_per_seed"} for p in l12],
        "hypothesis": ("H110: 两通路串联 ⇒ 复合损伤服从最弱环 max(δ_Q,δ_O) 而非求和；"
                       "FALSIFIED iff ∃裁决点 |I_rel| < TAU_ADD"),
    }
    (OUT / "verdict110.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1))
    print("已写 stageA110.json 与 verdict110.json")
    print("\n★ 最终码（含事后条款）：%s" % verdict["final_code"])
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())