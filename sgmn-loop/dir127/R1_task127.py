#!/usr/bin/env python3
"""`DIR-127` 正式轮驱动。三段分离：`--stage a`（硬门）· `--stage formal`（主判据）· `--stage all`。

★★ 本卡**不训练、不写 cell** —— 纯再分析。唯一新逻辑是 `ctx["rngs"]` 覆写。
★ `--stage a` 复用 `precheck127.py` 的全部断言（任一红即阻断 formal）。
★ `--stage formal` 才计算 `J127-0…J127-4`，并按卡片 §5 的**判决表**给出结局行。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dir126"))
sys.path.insert(0, os.path.join(HERE, ".."))

import mech127 as MM  # noqa: E402
import precheck127 as PC  # noqa: E402

OUT = os.path.join(HERE, "out")
M = MM.M
DOSE_ALL9 = MM.DOSE_ALL9


def _mkout():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)


# ---------------------------------------------------------------- Stage A
def stage_a():
    print("#" * 96)
    print("# DIR-127 · STAGE A（硬门）—— 复用 precheck127 的全部断言")
    print("#" * 96)
    rc = PC.main()
    _mkout()
    with open(os.path.join(OUT, "stage_a.log"), "w", encoding="utf-8") as fh:
        fh.write("rc=%d fails=%s\n" % (rc, PC._FAILS))
    return rc


# ---------------------------------------------------------------- formal
def _split():
    """(全子集读数, 新结构读数)。"""
    full, new = [], []
    for d in DOSE_ALL9:
        subs, avail, _c = MM.subsets_for(d)
        f = tuple(sorted(avail))
        for s in subs:
            r = MM.verdict_of(MM.reading(d, s))
            (full if tuple(s) == f else new).append(r)
    return full, new


def j0_nonvacuous(allr):
    """`J127-0`：四种取值每种至少 1 次（在 121 个结构上）。"""
    has_f0 = sum(1 for r in allr if r["frac"] is not None and r["frac"] == 0.0)
    has_fp = sum(1 for r in allr if r["frac"] is not None and r["frac"] > 0.0)
    has_rp = sum(1 for r in allr if r["ratio"] is not None and r["ratio"] > 1.0)
    has_rm = sum(1 for r in allr if r["ratio"] is not None and r["ratio"] < 1.0)
    ok = min(has_f0, has_fp, has_rp, has_rm) >= 1
    return ok, {"n_frac_zero": has_f0, "n_frac_pos": has_fp,
                "n_ratio_gt1": has_rp, "n_ratio_lt1": has_rm}


def j1_mixed(allr):
    """`J127-1`：逐剂量内 15 个子集的符号是否出现不一致。"""
    per = {}
    for r in allr:
        if r["ratio"] is None or r["ratio"] == 1.0:
            continue
        per.setdefault(r["dose"], []).append((r["sub"], MM.M126.sign_of_ratio(r["ratio"]), r))
    mixed = []
    detail = {}
    for d, rs in sorted(per.items()):
        signs = {s for _sub, s, _r in rs}
        detail[d] = {"n": len(rs), "signs": sorted(signs)}
        if len(rs) >= 2 and len(signs) > 1:
            pos = [(sub, r) for sub, s, r in rs if s > 0]
            neg = [(sub, r) for sub, s, r in rs if s < 0]
            mixed.append({"dose": d,
                          "example_pos": {"sub": pos[0][0], "ratio": pos[0][1]["ratio"],
                                          "frac": pos[0][1]["frac"]} if pos else None,
                          "example_neg": {"sub": neg[0][0], "ratio": neg[0][1]["ratio"],
                                          "frac": neg[0][1]["frac"]} if neg else None,
                          "n_pos": len(pos), "n_neg": len(neg)})
    return len(mixed), mixed, detail


def j2_main(newr):
    """`J127-2`：112 个新结构上的违例计数。"""
    fa = [r for r in newr if r["verdict"] == "FA"]
    fb = [r for r in newr if r["verdict"] == "FB"]
    return len(fa), len(fb), fa, fb


def _fit_threshold(rows, key):
    """全枚举 `M*`：`ratio > 1 ⟺ key ≤ M*`，返回 (最小违例数, 最优 M* 列表)。"""
    cand = sorted({r[key] for r in rows})
    if not cand:
        return None, []
    cand = [cand[0] - 1] + cand + [cand[-1] + 1]
    best, argbest = None, []
    for m in cand:
        v = 0
        for r in rows:
            if r["ratio"] is None or r["ratio"] == 1.0:
                continue
            pred = (r[key] <= m)
            obs = (r["ratio"] > 1.0)
            if pred != obs:
                v += 1
        if best is None or v < best:
            best, argbest = v, [m]
        elif v == best:
            argbest.append(m)
    return best, argbest


def j3_alt2(newr):
    """`J127-3`：写量阈值族（`Σm` 与 `max m_s`）的可分性。"""
    v_sum, arg_sum = _fit_threshold(newr, "m_sum")
    v_max, arg_max = _fit_threshold(newr, "m_max")
    return {"v_m_sum": v_sum, "arg_m_sum": arg_sum,
            "v_m_max": v_max, "arg_m_max": arg_max}


def j4_report(allr, newr):
    """`J127-4`：report-only 的表。"""
    per_dose = {}
    for r in allr:
        per_dose.setdefault(r["dose"], []).append(
            {"sub": list(r["sub"]), "k": r["k"], "N1": r["N1"], "N2": r["N2"],
             "gate": r["gate"], "m_sum": r["m_sum"], "m_max": r["m_max"],
             "frac": r["frac"], "ratio": r["ratio"], "verdict": r["verdict"]})
    for d in per_dose:
        per_dose[d].sort(key=lambda x: (x["k"], x["sub"]))
    # 逐剂量「闸门下降速度 vs Σm 下降速度」：端点比
    speed = {}
    for d, rows in per_dose.items():
        if len(rows) < 2:
            speed[d] = None
            continue
        lo = min(rows, key=lambda x: x["gate"])
        hi = max(rows, key=lambda x: x["gate"])
        speed[d] = {"gate_ratio": (hi["gate"] / lo["gate"]) if lo["gate"] else None,
                    "m_ratio": (hi["m_sum"] / lo["m_sum"]) if lo["m_sum"] else None,
                    "min_gate": lo["gate"], "max_gate": hi["gate"],
                    "m_at_min_gate": lo["m_sum"], "m_at_max_gate": hi["m_sum"]}
    return {"per_dose": per_dose, "speed": speed}


def _spearman(x, y):
    n = len(x)
    if n < 3:
        return None

    def rk(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rk(x), rk(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if dx * dy == 0 else num / (dx * dy)


def stage_formal():
    print("#" * 96)
    print("# DIR-127 · FORMAL —— 「固定 dose、换题集子集」的 121 个结构")
    print("#" * 96)
    _mkout()
    allr = [MM.verdict_of(MM.reading(d, s))
            for d in DOSE_ALL9 for s in MM.subsets_for(d)[0]]
    full, new = _split()
    tally = {t: 0 for t in MM.VERDICT_TAGS}
    for r in allr:
        tally[r["verdict"]] += 1
    print("\n★ 结构总数 = %d（全子集 %d + 新结构 %d）；结局分布 = %s"
          % (len(allr), len(full), len(new), tally))

    # ---- J127-0
    ok0, d0 = j0_nonvacuous(allr)
    print("\n[J127-0] 非空转门 —— %s" % ("PASS" if ok0 else "**FAIL**"))
    for k, v in d0.items():
        print("    %-14s = %d" % (k, v))

    # ---- 逐剂量表（report）
    rep = j4_report(allr, new)
    print("\n[J127-4 / report] 逐剂量 · 15 个子集的结构与读数")
    for d in DOSE_ALL9:
        rows = rep["per_dose"][d]
        sp = rep["speed"][d]
        print("\n  ── dose=%d（%d 个子集）──" % (d, len(rows)))
        if sp:
            print("     闸门 %.4f → %.4f（×%.3f）；Σm %d → %d（×%.3f）"
                  % (sp["min_gate"], sp["max_gate"], sp["gate_ratio"] or 0,
                     sp["m_at_max_gate"], sp["m_at_min_gate"], sp["m_ratio"] or 0))
        print("     %-20s %-3s %-6s %-6s %-9s %-7s %-7s %-10s %-10s %s"
              % ("子集", "k", "N1", "N2", "闸门", "Σm", "maxm", "frac", "ratio", "结局"))
        for x in rows:
            print("     %-20s %-3d %-6d %-6d %-9.4f %-7d %-7d %-10.6f %-10.6f %s"
                  % (str(tuple(x["sub"])), x["k"], x["N1"], x["N2"], x["gate"],
                     x["m_sum"], x["m_max"],
                     (x["frac"] if x["frac"] is not None else float("nan")),
                     (x["ratio"] if x["ratio"] is not None else float("nan")),
                     x["verdict"]))

    # ---- J127-1
    n_mixed, mixed, per_dose_sign = j1_mixed(allr)
    print("\n[J127-1] 决定性判别（`H127-alt-1` 纯剂量阈值）—— n_mixed = %d" % n_mixed)
    for d, v in sorted(per_dose_sign.items()):
        print("    dose=%-4d 可算 %-3d 个 · 符号集合 %s" % (d, v["n"], v["signs"]))
    if n_mixed >= 1:
        print("    ⇒ **`H127-alt-1` 被否证**（`D* ∈ (0,∞)` 整族）：")
        for m in mixed:
            print("        dose=%d：%d 个子集 `ratio>1`、%d 个 `ratio<1`"
                  % (m["dose"], m["n_pos"], m["n_neg"]))
            if m["example_pos"]:
                print("          正例 子集=%s frac=%.6f ratio=%.6f"
                      % (m["example_pos"]["sub"], m["example_pos"]["frac"],
                         m["example_pos"]["ratio"]))
            if m["example_neg"]:
                print("          负例 子集=%s frac=%.6f ratio=%.6f"
                      % (m["example_neg"]["sub"], m["example_neg"]["frac"],
                         m["example_neg"]["ratio"]))
    else:
        print("    ⇒ **`H127-alt-1` 存活** ⇒ 本设计未能分开两支")

    # ---- J127-2
    n_fa, n_fb, fa, fb = j2_main(new)
    n_viol = n_fa + n_fb
    print("\n[J127-2] 主判据（`H127`，仅 112 个**新结构**）—— 违例 = %d（(F-a) %d + (F-b) %d）"
          % (n_viol, n_fa, n_fb))
    if n_viol == 0:
        print("    ⇒ **`H127` 成立**（0 违例）")
    else:
        print("    ⇒ **`H127` 被否证** —— 反例逐字列出：")
        for tag, lst in (("(F-a) frac=0 ∧ ratio<1", fa), ("(F-b) frac>0 ∧ ratio>1", fb)):
            for r in lst:
                print("       %s ：dose=%-4d 子集=%-20s frac=%.6f ratio=%.6f"
                      % (tag, r["dose"], str(r["sub"]), r["frac"], r["ratio"]))

    # ---- J127-3
    j3 = j3_alt2(new)
    print("\n[J127-3] 写量阈值族（`H127-alt-2`）可分性 —— 强制报告")
    print("    `Σm ≤ M*`   ：最小违例 = %-3s（最优 M* ∈ %s）"
          % (j3["v_m_sum"], j3["arg_m_sum"][:4]))
    print("    `max m_s ≤ M*`：最小违例 = %-3s（最优 M* ∈ %s）"
          % (j3["v_m_max"], j3["arg_m_max"][:4]))
    if j3["v_m_sum"] == 0 or j3["v_m_max"] == 0:
        print("    ⇒ **本设计不能把 `u` 制度与写量阈值分开**（`H127` 的成立不得读成「中介」）")
    elif (j3["v_m_sum"] or 10 ** 9) > n_viol:
        print("    ⇒ 写量阈值**较弱**（%s/%s > %d）★ 仍**不得**称「已确立中介」"
              % (j3["v_m_sum"], j3["v_m_max"], n_viol))
    else:
        print("    ⇒ 写量阈值与 `H127` **同强或更强** ⇒ 无法分开")

    # ---- 逐剂量 frac↔ratio 秩（report-only）
    print("\n[J127-4 / report] 逐剂量**子集内** `ρ(frac, ratio)`（15 点，不独立 ⇒ 只报不判）")
    for d in DOSE_ALL9:
        rows = [r for r in allr if r["dose"] == d and r["ratio"] is not None
                and r["frac"] is not None]
        if len(rows) < 3:
            print("    dose=%-4d n=%d（不足 3 点）" % (d, len(rows)))
            continue
        rho = _spearman([r["frac"] for r in rows], [r["ratio"] for r in rows])
        print("    dose=%-4d n=%-3d ρ = %s" % (d, len(rows), ("%.6f" % rho) if rho is not None else "None"))

    # ---- 判决行
    if not ok0:
        row, verdict = 5, "整卡不可执行（J127-0 未过）"
    elif n_mixed >= 1 and n_viol == 0:
        row, verdict = 1, "H127 成立 且 H127-alt-1 被否证"
    elif n_mixed >= 1 and n_viol >= 1:
        row, verdict = 2, "H127 被否证（alt-1 已死）"
    elif n_mixed == 0 and n_viol == 0:
        row, verdict = 3, "两支都活 —— u 制度未获增量证据"
    else:
        row, verdict = 4, "H127 被否证 且 alt-1 存活"
    succ = 1 if row == 1 else 0
    print("\n" + "=" * 96)
    print("★ 判决表第 %d 行：%s" % (row, verdict))
    print("★ `SUCCEEDED` %s（57 → %d）" % ("+1" if succ else "不增", 57 + succ))
    print("=" * 96)

    res = {"tally": tally, "n_struct": len(allr), "n_full": len(full), "n_new": len(new),
           "J127_0": {"ok": ok0, **d0}, "J127_1": {"n_mixed": n_mixed, "mixed": mixed,
                                                  "per_dose_signs": {str(k): v for k, v in per_dose_sign.items()}},
           "J127_2": {"n_fa": n_fa, "n_fb": n_fb, "n_viol": n_viol,
                      "examples": [{"tag": r["verdict"], "dose": r["dose"], "sub": list(r["sub"]),
                                    "frac": r["frac"], "ratio": r["ratio"]} for r in fa + fb]},
           "J127_3": j3, "verdict_row": row, "verdict_text": verdict, "succeeded_inc": succ}
    with open(os.path.join(OUT, "formal.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "readings.json"), "w", encoding="utf-8") as fh:
        json.dump([{k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in r.items() if k != "detail"} for r in allr],
                  fh, ensure_ascii=False, indent=1)
    return 0


def main():
    stage = "all"
    if "--stage" in sys.argv:
        stage = sys.argv[sys.argv.index("--stage") + 1]
    if stage == "a":
        return stage_a()
    if stage == "formal":
        return stage_formal()
    rc = stage_a()
    if rc != 0:
        print("\n★ Stage A 未过 ⇒ 阻断 formal")
        return rc
    return stage_formal()


if __name__ == "__main__":
    sys.exit(main())
