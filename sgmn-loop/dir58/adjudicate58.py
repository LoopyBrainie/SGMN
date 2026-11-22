#!/opt/pytorch/bin/python
r"""DIR-58 · **独立裁决器**（**不 import** `R1_task58`；只读两臂 JSON + Stage A JSON，按卡 §5 冻结表**按序命中**）。

卡：`revolve/DIR-58/DIRECTION_rev58_v10.md`（rev 1.0，md5 `efc4fce81837752b236d7e68014a9a16`）。
判据量：卡 §3；设备门：卡 §4；判定表：卡 §5（6 支，只有最后一支 `counted=true`）。
用法：`/opt/pytorch/bin/python dir58/adjudicate58.py` → `dir58/out/DIR58r10_adjudication.json`
"""
import hashlib
import json
import os
import statistics
import sys

OUT = "/home/ec2-user/sgmn-loop/dir58/out"
CARD = "/home/ec2-user/sgmn-loop/revolve/DIR-58/DIRECTION_rev58_v10.md"
CARD_MD5 = "efc4fce81837752b236d7e68014a9a16"
ARMS = ("D", "W")
QS = {"D": (0, 1, 2, 4, 8), "W": (1, 2, 4, 8)}
QSTAR = (1, 2, 4)

# ── 冻结阈值（卡 §1.1 / §1.2 / §4）──
G1_MIN_S0 = 0.20
G2_MIN_POS = 0.90
G3_MIN_ENULL = 0.70
G4_MIN_DR = 0.80
R_PASS, R_FALSIFY, R_SE_MAX = 2.00, 1.30, 0.50
DQS_PASS, DQS_FALSIFY, DQS_SE_MAX = 2.00, 0.50, 1.00
COV_MIN = 0.75
D_EPS = {1: 0.05, 2: 0.05, 4: 0.02}
D_POS_MIN = 0.85


def load(arm):
    with open(os.path.join(OUT, "DIR58r10_%s.json" % arm)) as f:
        return json.load(f)


def qhalf(curve, qs, i0):
    r"""从 qs[i0] 的水平起线性内插到一半处；不足则 None。"""
    tgt = curve[i0] / 2
    for i in range(i0, len(curve) - 1):
        if curve[i] >= tgt >= curve[i + 1]:
            return qs[i] + (qs[i + 1] - qs[i]) * (curve[i] - tgt) / (curve[i] - curve[i + 1])
    return None


def qstar(curve, qs, target):
    r"""在只衰减曲线上内插出 level=target 的时间；落在窗外则 None。"""
    for i in range(len(curve) - 1):
        if curve[i] >= target >= curve[i + 1]:
            return qs[i] + (qs[i + 1] - qs[i]) * (curve[i] - target) / (curve[i] - curve[i + 1])
    return None


def med3(v):
    v = [x for x in v if x is not None]
    if not v:
        return {"n": 0, "med": None, "se": None, "3se": None}
    m = statistics.median(v)
    se = statistics.stdev(v) / (len(v) ** 0.5) if len(v) > 1 else 0.0
    return {"n": len(v), "med": m, "se": se, "3se": 3 * se,
            "min": min(v), "max": max(v),
            "q05": sorted(v)[max(0, int(round(0.05 * len(v))) - 1)]}


def main():
    with open(CARD, "rb") as f:
        card_md5 = hashlib.md5(f.read()).hexdigest()
    D, W = load("D"), load("W")
    with open(os.path.join(OUT, "DIR58r10_stageA.json")) as f:
        SA = json.load(f)
    rows_d = {r["seed"]: r["cells"] for r in D["rows"]}
    rows_w = {r["seed"]: r["cells"] for r in W["rows"]}
    seeds = sorted(set(rows_d) & set(rows_w))

    SD = [[rows_d[s]["q%d" % q]["S_old"] for q in QS["D"]] for s in seeds]
    CTRL = [[rows_d[s]["q%d" % q]["S_new"] for q in QS["D"]] for s in seeds]
    SW = [[rows_w[s]["q%d" % q]["S_old"] for q in QS["W"]] for s in seeds]
    SN = [[rows_w[s]["q%d" % q]["S_new"] for q in QS["W"]] for s in seeds]
    EN = [rows_d[s]["q0"]["E_null"] for s in seeds]

    # ── 判据量 ──
    R = []
    for c in SD:
        a, b = qhalf(c, QS["D"], 0), qhalf(c, QS["D"], 2)
        R.append(b / a if (a and b and a > 0) else None)
    qs_curve = []
    for i in range(len(seeds)):
        qs_curve.append([qstar(SD[i], QS["D"], SW[i][j]) for j, q in enumerate(QS["W"])])
    qstar_med = {q: med3([r[j] for r in qs_curve]) for j, q in enumerate(QS["W"])}
    cov1 = sum(1 for r in qs_curve if r[0] is not None) / len(seeds)
    cov4 = sum(1 for r in qs_curve if r[2] is not None) / len(seeds)
    both = [(r[2], r[0]) for r in qs_curve if r[2] is not None and r[0] is not None]
    dqstar = [b - a for b, a in both]
    delta = {q: med3([SN[i][j] - CTRL[i][QS["D"].index(q)] for i in range(len(seeds))])
             for j, q in enumerate(QS["W"])}
    pos_frac = {q: sum(1 for i in range(len(seeds))
                       if SN[i][QS["W"].index(q)] - CTRL[i][QS["D"].index(q)] > 0) / len(seeds)
                for q in QS["W"]}

    # ── 设备门 ──
    s0 = [c[0] for c in SD]
    G = {"G1": statistics.median(s0) >= G1_MIN_S0,
         "G2": sum(1 for x in s0 if x > 0) / len(s0) >= G2_MIN_POS,
         "G3": statistics.median(EN) >= G3_MIN_ENULL,
         "G4": None, "G5": None}
    med_curve = [statistics.median([c[i] for c in SD]) for i in range(len(QS["D"]))]
    G["G4"] = (med_curve[0] - med_curve[-1]) / med_curve[0] >= G4_MIN_DR
    G["G5"] = all(med_curve[i + 1] <= med_curve[i] + 1e-12 for i in range(len(med_curve) - 1))

    # ── 判定表（按序命中）──
    r_m = med3(R)
    dq_m = med3(dqstar)
    fin = all(x == x and abs(x) != float("inf")
              for name, r in (("D", D), ("W", W)) for row in r["rows"]
              for c in row["cells"].values() for x in
              (c["E_old"], c["E_new"], c["E_null"], c["S_old"], c["S_new"]))
    f3_bad = []
    for q in QSTAR:
        d = delta[q]
        if not (d["med"] is not None and d["med"] > d["3se"] and d["med"] >= D_EPS[q]
                and pos_frac[q] >= D_POS_MIN):
            f3_bad.append(q)

    table = []
    if card_md5 != CARD_MD5:
        table.append({"n": 1, "status": "INVALID-DEVICE", "counted": False,
                      "why": "卡片 md5 变了（%s）" % card_md5})
    elif not SA.get("all_pass"):
        table.append({"n": 1, "status": "INVALID-DEVICE", "counted": False, "why": "Stage A 未全过"})
    elif not all(G.values()):
        table.append({"n": 1, "status": "INVALID-DEVICE", "counted": False,
                      "why": "设备门失败：%s" % [k for k, v in G.items() if not v]})
    elif not fin:
        table.append({"n": 1, "status": "INVALID-DEVICE", "counted": False, "why": "读数非有限"})
    elif (r_m["3se"] or 0) > R_SE_MAX or (dq_m["3se"] or 0) > DQS_SE_MAX \
            or min(cov1, cov4) < COV_MIN:
        table.append({"n": 2, "status": "INFEASIBLE-POWER", "counted": False,
                      "why": "3se(R)=%.3f / 3se(Δq*)=%.3f / 覆盖率 %.2f/%.2f"
                             % (r_m["3se"] or -1, dq_m["3se"] or -1, cov1, cov4)})
    elif r_m["med"] is not None and r_m["med"] <= R_FALSIFY:
        table.append({"n": 3, "status": "LIFETIME-IS-CONSTANT", "counted": False,
                      "why": "median R=%.3f ≤ %.2f" % (r_m["med"], R_FALSIFY)})
    elif dq_m["med"] is not None and dq_m["med"] <= DQS_FALSIFY:
        table.append({"n": 4, "status": "EVICTION-IS-TIME-SHIFT", "counted": False,
                      "why": "Δq*=%.3f ≤ %.2f" % (dq_m["med"], DQS_FALSIFY)})
    elif f3_bad:
        table.append({"n": 5, "status": "NO-NEW-ASSOCIATION", "counted": False,
                      "why": "Δ 不达标的格：q=%s" % f3_bad})
    else:
        table.append({"n": 6, "status": "TWO-CHANNEL-LIFETIME-NO-CONSTANT-HALF-LIFE",
                      "counted": True, "why": "三条子句全部成立"})

    res = {"card_md5": card_md5, "n_seeds": len(seeds),
           "median_S_old_D": {str(q): med_curve[i] for i, q in enumerate(QS["D"])},
           "median_S_old_W": {str(q): med3([c[j] for c in SW])["med"]
                              for j, q in enumerate(QS["W"])},
           "gates": G, "gate_values": {"G1_S0": statistics.median(s0), "G2_pos": G["G2"],
                                       "G3_E_null": statistics.median(EN),
                                       "G4_DR": (med_curve[0] - med_curve[-1]) / med_curve[0]},
           "R": r_m, "R_range": [min(x for x in R if x), max(x for x in R if x)],
           "R_pos_frac": sum(1 for x in R if x and x > 1) / max(1, sum(1 for x in R if x)),
           "qstar": {str(q): qstar_med[q] for q in QS["W"]},
           "qstar_coverage": {str(q): sum(1 for r in qs_curve if r[j] is not None) / len(seeds)
                              for j, q in enumerate(QS["W"])},
           "delta_qstar": dq_m, "delta": {str(q): delta[q] for q in QS["W"]},
           "delta_pos_frac": {str(q): pos_frac[q] for q in QS["W"]},
           "f3_bad": f3_bad, "table": table, "counted": table[-1]["counted"],
           "status": table[-1]["status"], "prior_succeeded": 39,
           "succeeded_after": 39 + (1 if table[-1]["counted"] else 0)}
    with open(os.path.join(OUT, "DIR58r10_adjudication.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("种子 %d · 卡片 md5 %s %s" % (len(seeds), card_md5[:8],
                                     "= 冻结" if card_md5 == CARD_MD5 else "≠ 冻结"))
    print("设备门：%s（S0=%.4f · DR=%.3f · E_null=%.3f）"
          % (G, statistics.median(s0), res["gate_values"]["G4_DR"], statistics.median(EN)))
    print("R = %.3f ± %.3f（3se）· 逐种子范围 [%.2f, %.2f] · Pr(R>1)=%.2f"
          % (r_m["med"], r_m["3se"], res["R_range"][0], res["R_range"][1], res["R_pos_frac"]))
    print("q*（对）：%s · 覆盖率 %.2f/%.2f · Δq* = %.3f ± %.3f"
          % ({q: round(qstar_med[q]["med"], 2) if qstar_med[q]["med"] is not None else None
              for q in QS["W"]}, cov1, cov4, dq_m["med"], dq_m["3se"]))
    for q in QS["W"]:
        d = delta[q]
        aux = "（辅助，不进判据）" if q not in D_EPS else ""
        print("  Δ(q=%d) = %+.4f ± %.4f（3se）· Pr>0 = %.2f · 下限 %s ⇒ %s%s"
              % (q, d["med"], d["3se"], pos_frac[q], D_EPS.get(q, "—"),
                 "过" if q not in f3_bad else "**不达标**", aux))
    print("\n★ 判定 **%s**（counted=%s）⇒ SUCCEEDED %d/%d"
          % (table[-1]["status"], table[-1]["counted"], res["succeeded_after"], res["succeeded_after"]))
    sys.exit(0 if table[-1]["counted"] else 2)


if __name__ == "__main__":
    main()