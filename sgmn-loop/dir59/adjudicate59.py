#!/opt/pytorch/bin/python
r"""DIR-59 · 独立裁决器（`DIRECTION_rev59_v10.md §3.3/§4` 的逐字实现）

`--examples`：用卡 §5 的三个工作例自检判定逻辑（必须复现 `FALS/EQV/GREY`）。
`--run`     ：读 20 个 `DIR59r10_c{j}_{AA,AB,BA,BB}.json`，算每格量与判定，出总判定表。
"""
import argparse
import json
import math
import pathlib
import statistics
import sys

SIG_K, FRAC_MAT, FRAC_EQ, PASS_CELLS_MIN, N_EVAL = 3.0, 0.50, 0.20, 4, 400
OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir59/out")
CELLS, ARMS = (1, 2, 3, 4, 5), ("AA", "AB", "BA", "BB")


def se(v):
    return statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")


def paired(x, y):
    d = [a - b for a, b in zip(x, y)]
    return statistics.fmean(d), se(d)


def classify(co, srcA, seA, srcB, seB, C, seC):
    r"""§3.3 的逐字实现。返回 (类别, 量)。"""
    dA = max(SIG_K * seA, FRAC_MAT * co)
    dB = max(SIG_K * seB, FRAC_MAT * co)
    deq = FRAC_EQ * co
    q = {"co": co, "srcA": srcA, "seA": seA, "srcB": srcB, "seB": seB,
         "dA": dA, "dB": dB, "deq": deq, "C": C, "seC": seC}
    if not (co > 0) or not math.isfinite(co):
        return "UNUSABLE", q
    if not (C >= SIG_K * seC):
        return "NO-POWER", q
    if srcA >= dA or srcB >= dB:
        return "FALS", q
    if abs(srcA) < deq and abs(srcB) < deq:
        return "EQV", q
    return "GREY", q


def cell_from_files(j):
    recs = {}
    for arm in ARMS:
        p = OUT / f"DIR59r10_c{j}_{arm}.json"
        if not p.exists():
            return None
        recs[arm] = json.loads(p.read_text(encoding="utf-8"))
    L = {a: recs[a]["mean_L"] for a in ARMS}
    sA, seA = paired(recs["BA"]["L_series"], recs["AA"]["L_series"])
    sB, seB = paired(recs["AB"]["L_series"], recs["BB"]["L_series"])
    co = (L["AA"] + L["BB"]) / 2.0
    mis = (L["AB"] + L["BA"]) / 2.0
    if "rc" not in recs["AA"]:
        C, sC = float("nan"), float("nan")
    else:
        C, sC = paired(recs["AA"]["rc"]["L_series"], recs["AA"]["L_series"])
    cls, q = classify(co, sA, seA, sB, seB, C, sC)
    q.update({"L_AA": L["AA"], "L_AB": L["AB"], "L_BA": L["BA"], "L_BB": L["BB"],
              "L_co": co, "L_mis": mis, "d_mis_co": mis - co,
              "n_eval": {a: len(recs[a]["L_series"]) for a in ARMS},
              "pk1_sha1": {a: recs[a].get("sha1_pk1", "")[:12] for a in ARMS},
              "class": cls})
    return q


def run():
    rows, missing = [], []
    for j in CELLS:
        q = cell_from_files(j)
        if q is None:
            missing.append(j)
            continue
        rows.append(q)
    usable = [r for r in rows if r["class"] not in ("UNUSABLE", "NO-POWER")]
    n_fals = sum(r["class"] == "FALS" for r in usable)
    n_eqv = sum(r["class"] == "EQV" for r in usable)
    n_grey = sum(r["class"] == "GREY" for r in usable)
    neg = [r for r in usable if r["srcA"] < 0 and abs(r["srcA"]) >= r["dA"]
           or r["srcB"] < 0 and abs(r["srcB"]) >= r["dB"]]
    print(f"{'j':>2s} {'L_AA':>8s} {'L_AB':>8s} {'L_BA':>8s} {'L_BB':>8s} {'L_co':>7s} "
          f"{'srcA':>8s} {'±3se':>7s} {'srcB':>8s} {'±3se':>7s} {'Δ*':>6s} {'δeq':>6s} "
          f"{'C':>7s} {'class':>8s}")
    for r in rows:
        print(f"{r.get('j', 0):2d} {r['L_AA']:8.4f} {r['L_AB']:8.4f} {r['L_BA']:8.4f} "
              f"{r['L_BB']:8.4f} {r['L_co']:7.4f} {r['srcA']:+8.4f} {3*r['seA']:7.4f} "
              f"{r['srcB']:+8.4f} {3*r['seB']:7.4f} {r['dA']:6.4f} {r['deq']:6.4f} "
              f"{r['C']:7.4f} {r['class']:>8s}")
    if missing:
        verdict = "INCOMPLETE"
    elif any(not all(math.isfinite(r[k]) for k in ("L_AA", "srcA", "srcB", "seA", "seB"))
             for r in rows):
        verdict = "DIVERGED"
    elif len(usable) < 4:
        verdict = "INCONCLUSIVE-NO-POWER"
    elif n_fals >= PASS_CELLS_MIN:
        verdict = "SOURCE-IS-LOAD"
    elif n_eqv >= PASS_CELLS_MIN:
        verdict = "PASS"
    else:
        verdict = "INDETERMINATE"
    print(f"\n可判格 {len(usable)}/5 · FALS {n_fals} · EQV {n_eqv} · GREY {n_grey} "
          f"· 负向显著 {len(neg)}")
    print(f"★★ 总判定 = {verdict}")
    (OUT / "DIR59r10_verdict.json").write_text(json.dumps(
        {"verdict": verdict, "missing_cells": missing, "n_usable": len(usable),
         "n_fals": n_fals, "n_eqv": n_eqv, "n_grey": n_grey,
         "negative_direction_cells": [r.get("j") for r in neg],
         "constants": {"SIG_K": SIG_K, "FRAC_MAT": FRAC_MAT, "FRAC_EQ": FRAC_EQ,
                       "PASS_CELLS_MIN": PASS_CELLS_MIN, "N_EVAL": N_EVAL},
         "cells": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return verdict


def examples():
    ok = True
    # 例 A：否证（探针同型量级）
    c, q = classify(1.63, 11.63, 1.04 / 3, 9.00, 0.90 / 3, 15.15, 0.90 / 3)
    print(f"例A co=1.63 srcA=+11.63 3se=1.04 C=15.15 ⇒ {c}（期望 FALS）"); ok &= c == "FALS"
    # 例 B：确认（反事实）
    c, q = classify(1.63, 0.05, 1.04 / 3, -0.06, 1.04 / 3, 15.15, 0.90 / 3)
    print(f"例B srcA=+0.05 srcB=-0.06 δeq={q['deq']:.3f} ⇒ {c}（期望 EQV）"); ok &= c == "EQV"
    # 例 C：灰区
    c, q = classify(1.63, 0.60, 1.04 / 3, 0.10, 1.04 / 3, 15.15, 0.90 / 3)
    print(f"例C srcA=+0.60 Δ*={q['dA']:.3f} δeq={q['deq']:.3f} ⇒ {c}（期望 GREY）"); ok &= c == "GREY"
    # 附加：阳性对照失败 ⇒ NO-POWER（不是 FAIL）
    c, _ = classify(1.63, 11.63, 0.34, 9.0, 0.30, 0.2, 0.5)
    print(f"对照失效 C=0.2 3se=1.5 ⇒ {c}（期望 NO-POWER）"); ok &= c == "NO-POWER"
    print("★ 判定逻辑自检通过" if ok else "★★ 判定逻辑自检失败")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true"); ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.examples:
        sys.exit(examples())
    elif a.run:
        run()
    else:
        ap.print_help()
