#!/usr/bin/env python3
"""`DIR-128` 正式轮驱动 —— 按**冻结卡 rev 1.1** 逐条算 `J128-1..3` + 报告项。

用法：
    python R1_task128.py --stage precheck      # 只跑结构核（转发）
    python R1_task128.py --stage smoke         # 10 个结构抽检（**不出判据**）
    python R1_task128.py --stage formal        # 全 121 结构 + 判据 + 报告 + 功效

★ 纪律：判据**只在 formal 算一次**；smoke 只证明协议可执行，**不产判决**。
★ 功效（`P_binom`/`P_cluster`）**与判据分开报**，不得互相追认（卡片 §4 / §5）。
"""
import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mech128 as M  # noqa: E402

OUT = os.path.join(HERE, "out")
TOL = 1.10          # ★ 卡片 §1-4 冻结
Z = 1.959963984540054

T = M.load_tensor()
DOSES = M.all_doses(T)


def canonical(d):
    v = M.dose_view(T, d)
    return v, tuple(sorted(v["avail"]))


# ================================================================ 判据
def j128_1():
    """`H128`：轴 1 的**值**不变性（规范读数，`A0..A4` 的 max/min ≤ 1.10×，每剂量）。"""
    rows, fail, undet = [], [], []
    for d in DOSES:
        v, S = canonical(d)
        o = M.rules(v, S)
        got = [(o[k], k) for k in M.AXIS1 if o.get(k) is not None and o[k] > 0]
        if len(got) < 2:
            undet.append(d)
            rows.append({"dose": d, "tension": None, "n_use": len(got), "state": "UNDETERMINED"})
            continue
        hi, lo = max(got), min(got)
        t = hi[0] / lo[0]
        st = "PASS" if t <= TOL else "FAIL"
        if st == "FAIL":
            fail.append(d)
        rows.append({"dose": d, "tension": t, "argmax": hi[1], "argmin": lo[1],
                     "n_use": len(got), "state": st,
                     "log_sys": abs(math.log(t)),
                     "log_ratio": abs(math.log(o["A0"])) if o.get("A0") else None})
    verdict = "UNDETERMINED" if (undet and not fail) else ("FAIL" if fail else "PASS")
    return {"id": "J128-1", "verdict": verdict, "rows": rows,
            "fail_doses": fail, "undet_doses": undet,
            "note": "H128 主假设；FAIL ⇒ SUCCEEDED 不增（卡片 §5）"}


def j128_2():
    """`H128-S`：轴 1 的**符号**一致性（规范读数，每剂量 5 条规则同号）。"""
    rows, bad = [], []
    for d in DOSES:
        v, S = canonical(d)
        o = M.rules(v, S)
        sg = {M.sign_of(o[k]): k for k in M.AXIS1 if o.get(k) is not None}
        signs = set(sg)
        st = "PASS" if len(signs) == 1 else "FAIL"
        if st == "FAIL":
            bad.append(d)
        rows.append({"dose": d, "signs": sorted(x for x in signs if x is not None),
                     "has_zero": 0 in signs, "state": st})
    return {"id": "J128-2", "verdict": "FAIL" if bad else "PASS", "rows": rows, "fail_doses": bad}


def j128_3():
    """`H128-Q`：子集轴的**符号**一致性（15 个子集同号；`dose=0` 只有 1 个子集 ⇒ 不可判）。"""
    rows, bad, undet = [], [], []
    for d in DOSES:
        v = M.dose_view(T, d)
        subs = M.all_subsets(v["avail"])
        if len(subs) < 2:
            undet.append(d)
            rows.append({"dose": d, "n_sub": 1, "state": "UNDETERMINED"})
            continue
        signs, lo, hi, lo_s, hi_s = set(), None, None, None, None
        for s in subs:
            x = M.rules(v, s)["A0"]
            if x is None:
                continue
            signs.add(M.sign_of(x))
            if lo is None or x < lo:
                lo, lo_s = x, s
            if hi is None or x > hi:
                hi, hi_s = x, s
        st = "PASS" if len(signs - {0}) <= 1 else "FAIL"
        if st == "FAIL":
            bad.append(d)
        rows.append({"dose": d, "n_sub": len(subs), "signs": sorted(x for x in signs if x is not None),
                     "min_ratio": lo, "max_ratio": hi, "argmin": list(lo_s or ()),
                     "argmax": list(hi_s or ()), "state": st,
                     "flip": st == "FAIL"})
    return {"id": "J128-3", "verdict": "FAIL" if bad else ("UNDETERMINED" if undet and len(undet) == len(DOSES) else "PASS"),
            "rows": rows, "fail_doses": bad, "undet_doses": undet}


# ================================================================ 报告项（非判据）
def report_axes():
    """轴 2（平滑）/ 轴 3（logit 层）/ pair 级率族 —— 与 `A0` 的偏离倍数（**只报不判**）。"""
    rows = []
    for d in DOSES:
        v, S = canonical(d)
        o = M.rules(v, S)
        base = o.get("A0")
        if base is None or base <= 0:
            continue
        r = {"dose": d, "A0": base}
        for k in M.AXIS2 + M.AXIS3 + M.PAIRF:
            r[k] = o.get(k)
            r[k + "_dev"] = (o[k] / base) if (o.get(k) and o[k] > 0) else None
        rows.append(r)
    return rows


def report_pair_family_tension():
    """pair 级率族的族内 max/min（分辨率下限的直接证据，**只报**）。"""
    rows = []
    for d in DOSES:
        v, S = canonical(d)
        t, amx, amn, n = M.tension(v, S, M.PAIRF)
        rows.append({"dose": d, "tension": t, "argmax": amx, "argmin": amn, "n_use": n})
    return rows


def report_budget():
    """★ 核心可带走量：轴 1 系统量 `|log(max/min)|` vs 判符号余量 `|log ratio|`。"""
    out = []
    for d in DOSES:
        v, S = canonical(d)
        o = M.rules(v, S)
        got = [(o[k], k) for k in M.AXIS1 if o.get(k) is not None and o[k] > 0]
        base = o.get("A0")
        if len(got) < 2 or base is None:
            out.append({"dose": d, "state": "UNDETERMINED"})
            continue
        t = max(x for x, _ in got) / min(x for x, _ in got)
        sys_log = abs(math.log(t))
        mar_log = abs(math.log(base))
        out.append({"dose": d, "tension": t, "log_sys": sys_log, "log_ratio": mar_log,
                    "frac_of_margin": (sys_log / mar_log) if mar_log > 0 else None,
                    "sign_undetermined_by_axis1": sys_log >= mar_log})
    return out


def _mc_binom_power(p1, p2, n1, n2, k, reps, seed):
    """`P_binom`：把 (e1,e2) 放大 k 倍后，log(ratio) 符号判对的概率（二项重采样）。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    N1 = max(int(round(k * n1)), 1)
    N2 = max(int(round(k * n2)), 1)
    E1 = rng.binomial(N1, p1, size=reps)
    E2 = rng.binomial(N2, p2, size=reps)
    ok = (E1 > 0) & (E2 > 0) & (E1 < N1) & (E2 < N2)
    if ok.sum() == 0:
        return None
    q1 = E1[ok] / float(N1)
    q2 = E2[ok] / float(N2)
    g = np.log(q1 / (1 - q1)) - np.log(q2 / (1 - q2))
    truth = math.log(p1 / (1 - p1)) - math.log(p2 / (1 - p2))
    tgt = 1 if truth > M.LOG4 else -1
    return float((np.sign(g - M.LOG4) == tgt).mean())


def power_report(reps=20000, seed=20261124):
    """功效：`P_binom`（乐观上界，含 MC 极差）+ `P_cluster`（9 簇 bootstrap）。"""
    import numpy as np
    out = []
    for d in DOSES:
        v, S = canonical(d)
        per, nc = v["per_rng"], v["nc"]
        n1 = sum(per[r]["n1"] for r in S)
        n2 = sum(per[r]["n2"] for r in S)
        e1 = sum(sum(per[r]["e1"]) for r in S)
        e2 = sum(sum(per[r]["e2"]) for r in S)
        N1, N2 = nc * n1, nc * n2
        p1, p2 = e1 / float(N1), e2 / float(N2)
        rec = {"dose": d, "N1": N1, "N2": N2, "e1": e1, "e2": e2, "p1": p1, "p2": p2}
        # --- P_binom: 找 k 使 power>=0.80 / 0.95（CRN：同一 k 的多次重复共享 seed 序列）
        def pw(k, sd):
            return _mc_binom_power(p1, p2, N1, N2, k, reps, sd)

        def find(target):
            lo, hi = 1.0, 4096.0
            if pw(hi, seed) is None or pw(hi, seed) < target:
                return None
            for _ in range(24):
                mid = (lo * hi) ** 0.5
                p = pw(mid, seed)
                if p is None:
                    lo = mid
                    continue
                if p >= target:
                    hi = mid
                else:
                    lo = mid
            return hi
        k80, k95 = find(0.80), find(0.95)
        rec["k80"] = k80
        rec["k95"] = k95
        rec["e2_80"] = (k80 * e2) if k80 else None
        rec["e2_95"] = (k95 * e2) if k95 else None
        # MC 极差（3 个独立 seed，同一个 k80）
        if k80:
            ps = [pw(k80, seed + j) for j in range(3)]
            ps = [x for x in ps if x is not None]
            rec["k80_mc_range"] = [min(ps), max(ps)] if ps else None
        # --- P_cluster: 9 个 cell 有放回重采样
        rng = np.random.default_rng(seed)
        B = 4000
        idx = rng.integers(0, nc, size=(B, nc))
        # 逐 cell 的 e/N 已由结构决定 N（与 cell 无关）⇒ 只需重采样 e1_c/e2_c
        E1c = [sum(per[r]["e1"][c] for r in S) for c in range(nc)]
        E2c = [sum(per[r]["e2"][c] for r in S) for c in range(nc)]
        a1 = np.array(E1c)[idx].sum(axis=1) / float(nc * n1)
        a2 = np.array(E2c)[idx].sum(axis=1) / float(nc * n2)
        good = (a1 > 0) & (a1 < 1) & (a2 > 0) & (a2 < 1)
        if good.sum() > 10:
            g = np.log(a1[good] / (1 - a1[good])) - np.log(a2[good] / (1 - a2[good]))
            lr = g - M.LOG4
            rec["cluster_lo"] = float(np.percentile(lr, 2.5))
            rec["cluster_hi"] = float(np.percentile(lr, 97.5))
            rec["cluster_crosses_0"] = bool(rec["cluster_lo"] <= 0.0 <= rec["cluster_hi"])
            rec["cluster_p_pos"] = float((lr > 0).mean())
            rec["cluster_n_eff"] = int(good.sum())
        out.append(rec)
    return out


# ================================================================ stage 驱动
def stage_precheck():
    return subprocess.call([sys.executable, os.path.join(HERE, "precheck128.py")])


def stage_smoke():
    """10 个结构抽检 —— **只证明协议可执行，不出判据**。"""
    picks = [(0, ("555",)), (25, ("555",)), (300, ("555", "556")), (350, ("556", "557")),
             (375, ("557", "558")), (400, ("558",)), (450, ("557",)), (450, ("555",)),
             (600, ("555", "558")), (600, ("555", "556", "557", "558"))]
    rows = []
    for d, s in picks:
        v = M.dose_view(T, d)
        if not set(s) <= set(v["avail"]):
            continue
        o = M.rules(v, s)
        rows.append({"dose": d, "sub": list(s), "A0": o["A0"],
                     "keys_present": sum(1 for k in (M.AXIS1 + M.AXIS2 + M.AXIS3 + M.PAIRF)
                                         if o.get(k) is not None)})
    os.makedirs(OUT, exist_ok=True)
    json.dump({"n": len(rows), "rows": rows}, open(os.path.join(OUT, "stage_smoke.json"), "w"),
              indent=1, ensure_ascii=False)
    NK = len(M.AXIS1) + len(M.AXIS2) + len(M.AXIS3) + len(M.PAIRF)
    print("smoke: %d 个结构完成（**不产判决**）；%d 键全有的结构数 = %d"
          % (len(rows), NK, sum(1 for r in rows if r["keys_present"] == NK)))
    return 0


def stage_formal():
    os.makedirs(OUT, exist_ok=True)
    J1, J2, J3 = j128_1(), j128_2(), j128_3()
    budget = report_budget()
    ax = report_axes()
    pf = report_pair_family_tension()
    print("=" * 104)
    print("`J128-1`（`H128` · 轴 1 的**值**不变性，规范读数，阈值 %.2f×）" % TOL)
    print("=" * 104)
    print("%-6s %-10s %-9s %-9s %-8s %s" % ("dose", "max/min", "argmax", "argmin", "n_use", "判"))
    for r in J1["rows"]:
        print("%-6d %-10s %-9s %-9s %-8d %s" % (
            r["dose"], ("%.4f×" % r["tension"]) if r["tension"] else "—",
            r.get("argmax", "—"), r.get("argmin", "—"), r["n_use"], r["state"]))
    print("⇒ `J128-1` = **%s**%s" % (J1["verdict"],
          ("  （FAIL 的剂量：%s）" % J1["fail_doses"]) if J1["fail_doses"] else ""))
    print()
    print("=" * 104)
    print("`J128-2`（`H128-S` · 轴 1 的**符号**一致性）  /  `J128-3`（`H128-Q` · **子集轴**的符号一致性）")
    print("=" * 104)
    print("%-6s %-14s %-8s | %-7s %-12s %-10s %-10s %s" % (
        "dose", "规则符号", "J128-2", "n_sub", "子集符号", "min ratio", "max ratio", "J128-3"))
    r2 = {r["dose"]: r for r in J2["rows"]}
    for r in J3["rows"]:
        a = r2.get(r["dose"], {})
        print("%-6d %-14s %-8s | %-7d %-12s %-10s %-10s %s%s" % (
            r["dose"], str(a.get("signs")), a.get("state", "—"),
            r["n_sub"], str(r.get("signs", "—")),
            ("%.4f" % r["min_ratio"]) if r.get("min_ratio") else "—",
            ("%.4f" % r["max_ratio"]) if r.get("max_ratio") else "—",
            r["state"], "  ★翻符号" if r.get("flip") else ""))
    print("⇒ `J128-2` = **%s**   `J128-3` = **%s**%s" % (
        J2["verdict"], J3["verdict"],
        ("  （翻符号剂量：%s）" % J3["fail_doses"]) if J3["fail_doses"] else ""))
    print()
    print("=" * 104)
    print("★ 报告项（**非判据**）：系统量 vs 余量")
    print("=" * 104)
    print("%-6s %-10s %-11s %-11s %-10s %s" % ("dose", "max/min", "|log sys|", "|log ratio|", "占余量", "⇒ 符号是否被轴1定为未定"))
    for r in budget:
        if r.get("state") == "UNDETERMINED":
            print("%-6d —" % r["dose"])
            continue
        print("%-6d %-10.4f %-11.5f %-11.5f %-10s %s" % (
            r["dose"], r["tension"], r["log_sys"], r["log_ratio"],
            ("%.1f%%" % (100 * r["frac_of_margin"])) if r["frac_of_margin"] is not None else "—",
            "★ 是（Δ_sys ≥ 余量）" if r["sign_undetermined_by_axis1"] else "否"))
    print()
    print("★ pair 级率族（分辨率下限，只报）：族内 max/min 最差 = %.4f× @ dose=%s（n_use=%s）" % (
        max([r["tension"] for r in pf if r["tension"]] or [float("nan")]),
        max([r for r in pf if r["tension"]], key=lambda x: x["tension"])["dose"],
        max([r for r in pf if r["tension"]], key=lambda x: x["tension"])["n_use"]))
    print()
    print("★ 计算功效（`P_binom` 乐观上界 + `P_cluster` 9 簇 bootstrap）…")
    pw = power_report()
    print("%-6s %-8s %-8s %-8s %-9s %-9s %-22s %s" % (
        "dose", "p1", "p2", "e2", "k80", "e2*80", "cluster 95%CI(log ratio)", "跨0?"))
    for r in pw:
        ci = ("[%+.3f, %+.3f]" % (r["cluster_lo"], r["cluster_hi"])) if "cluster_lo" in r else "—"
        print("%-6d %-8.4f %-8.4f %-8d %-9s %-9s %-22s %s" % (
            r["dose"], r["p1"], r["p2"], r["e2"],
            ("%.1f" % r["k80"]) if r.get("k80") else ">4096",
            ("%.0f" % r["e2_80"]) if r.get("e2_80") else "—",
            ci, ("★ 是" if r.get("cluster_crosses_0") else "否")))
    # --- 判决表
    table = {
        ("PASS", "PASS", "PASS"): ("C128-INSTRUMENT-EXISTS", True),
        ("FAIL", "PASS", "PASS"): ("C128-VALUE-NOT-INVARIANT-SIGN-IS", False),
        ("FAIL", "PASS", "FAIL"): ("C128-SIGN-IS-QUESTION-SET-PROPERTY", False),
        ("PASS", "FAIL", "PASS"): ("C128-VALUE-INVARIANT-SIGN-NOT", False),
        ("PASS", "PASS", "FAIL"): ("C128-VALUE-INVARIANT-SIGN-NOT", False),
        ("FAIL", "FAIL", "PASS"): ("C128-VALUE-VS-SIGN-SPLIT", False),
        ("FAIL", "FAIL", "FAIL"): ("C128-NOTHING-INVARIANT", False),
        ("PASS", "FAIL", "FAIL"): ("C128-VALUE-INVARIANT-SIGN-NOT", False),
    }
    key = (J1["verdict"], J2["verdict"], J3["verdict"])
    tag, inc = table.get(key, ("C128-UNMAPPED", False))
    print()
    print("=" * 104)
    print("★ 判决 = **%s**   （J128-1=%s, J128-2=%s, J128-3=%s）" % (tag, *key))
    print("★ `SUCCEEDED` %s（主判据 `J128-1` %s）" % ("+1" if inc else "不增", key[0]))
    print("★ 声明：`H128` 的否证是**族相对**的（`A0..A4` 里没有一条满足）；**不得**写成「不存在任何加权规则」。")
    print("=" * 104)
    res = {"card": "DIRECTION_rev128_v11.md", "tensor_md5_note": "6794994174c9218d0a166ce75eea45ce",
           "tol": TOL, "J128_1": J1, "J128_2": J2, "J128_3": J3,
           "budget": budget, "axes": ax, "pair_family": pf, "power": pw,
           "verdict": tag, "succeeded_increment": bool(inc)}
    json.dump(res, open(os.path.join(OUT, "stage_formal.json"), "w"), indent=1, ensure_ascii=False)
    print("已写 dir128/out/stage_formal.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="formal",
                    choices=["precheck", "smoke", "formal"])
    a = ap.parse_args()
    if a.stage == "precheck":
        sys.exit(stage_precheck())
    if a.stage == "smoke":
        sys.exit(stage_smoke())
    sys.exit(stage_formal())