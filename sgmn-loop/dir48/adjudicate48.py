#!/opt/pytorch/bin/python
r"""DIR-48 · 独立裁决器：**不 import 实现**，只读 9 个分片 jsonl，**自己**从 `c(d)` 重算 `L`、`p`、KM 中位数、发散比，逐条转写卡 rev 1.1 §5 判定表。

用法：`adjudicate48.py --out dir48/out/DIR48r11_adjudication.json`
"""
import argparse, glob, json, math, os, sys

ROOT = "/home/ec2-user/sgmn-loop"
RHO, H_CMP = 0.5, 7
P1A_MAX, P1A_FALSIFY = 0.30, 0.50
P2A_MIN, MED_BUSY_MAX, N_DIV_MAX = 0.90, 3, 0.20
ANCHOR_C0_MIN, ANCHOR_FRAC = 0.30, 0.90
SIG_K = 3
N_INST, SEED_LO, SEED_HI = 1000, 27400, 28399
W_EXP = {"quiet": 16, "decay": 32, "busy": 8}
ARMS = ("quiet", "decay", "busy")


def load(arm):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, f"dir48/out/DIR48r11_arm_{arm}_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def recompute(r):
    r"""从 `c(d)` **独立**重算：`L`（首个跌破 `RHO·c(0)` 的 d）、`censored`、`undef`（非有限）。"""
    c, W = r["c"], r["W"]
    errs = []
    if len(c) != W or W != W_EXP[r["arm"]]:
        errs.append(f"len(c)={len(c)} W={W}")
    if not all(math.isfinite(x) for x in c):
        return {"undef": True, "L": None, "cens": None, "c0": c[0], "errs": errs}
    c0 = c[0]
    if not (c0 > 0):
        return {"undef": True, "L": None, "cens": None, "c0": c0, "errs": errs}
    thr, L, cens = RHO * c0, W, True
    for d in range(1, W):
        if c[d] < thr:
            L, cens = d, False
            break
    if (L, cens) != (r["L"], r["censored"]):
        errs.append(f"L/cens 不一致 mine=({L},{cens}) got=({r['L']},{r['censored']})")
    return {"undef": False, "L": L, "cens": cens, "c0": c0, "errs": errs}


def km_median(Ls, cens, W):
    pairs = sorted(zip(Ls, cens), key=lambda x: x[0])
    n_at, surv, med, i = len(pairs), 1.0, None, 0
    while i < len(pairs):
        t = pairs[i][0]
        d = c = 0
        while i < len(pairs) and pairs[i][0] == t:
            if pairs[i][1]:
                c += 1
            else:
                d += 1
            i += 1
        if d and n_at > 0:
            surv *= (1 - d / n_at)
        if surv <= 0.5 and med is None:
            med = t
        n_at -= (d + c)
        if n_at <= 0:
            break
    return med, (med is not None)


def se3(p, n):
    return SIG_K * math.sqrt(max(p * (1 - p), 0.0) / n) if n else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {arm: load(arm) for arm in ARMS}
    by = {arm: {int(r["seed"]): r for r in R[arm]} for arm in ARMS}
    viol, stat = [], {}
    for arm in ARMS:
        Ls, cs, c0s, nd, errs = [], [], [], 0, 0
        for r in R[arm]:
            o = recompute(r)
            errs += len(o["errs"])
            if o["errs"]:
                viol.append({"arm": arm, "seed": r["seed"], "errs": o["errs"]})
            if o["undef"]:
                nd += 1
                continue
            Ls.append(o["L"]); cs.append(o["cens"]); c0s.append(o["c0"])
        n_use = len(Ls)
        n_cens = sum(1 for x in cs if x)
        med, reached = km_median(Ls, cs, W_EXP[arm])
        p_cmp = sum(1 for L in Ls if L <= H_CMP) / n_use if n_use else None
        stat[arm] = {"n": len(R[arm]), "n_div": nd, "div_frac": nd / len(R[arm]),
                     "n_use": n_use, "cens_frac": n_cens / n_use if n_use else None,
                     "p_HCMP": p_cmp, "p_HCMP_3se": se3(p_cmp, n_use) if p_cmp is not None else None,
                     "km_median": med, "km_reached": reached,
                     "mean_c0": sum(c0s) / len(c0s) if c0s else None,
                     "c0_anchor_frac": sum(1 for x in c0s if x >= ANCHOR_C0_MIN) / n_use if n_use else None,
                     "mean_L": sum(Ls) / n_use if n_use else None}
    tab = {}

    # ①INCOMPLETE
    inc = []
    for arm in ARMS:
        if len(R[arm]) != N_INST:
            inc.append(f"{arm} 行数 {len(R[arm])} != {N_INST}")
        if sorted(by[arm]) != list(range(SEED_LO, SEED_HI + 1)):
            inc.append(f"{arm} 种子集合不符")
    # quiet 逐位不变（NaN 感知）
    qbad = 0
    for r in R["quiet"]:
        c = r["c"]
        same = lambda x, y: (x == y) or (math.isnan(x) and math.isnan(y))
        if not all(same(x, c[0]) for x in c):
            qbad += 1
    if qbad:
        inc.append(f"quiet 读数改状态 {qbad} 行")
    if viol:
        inc.append(f"L/cens 重算不一致或长度不符 {len(viol)} 行")
    tab["1_INCOMPLETE"] = {"hit": bool(inc), "detail": inc}

    # ②DIVERGED
    div_hit = stat["busy"]["div_frac"] > N_DIV_MAX
    tab["2_DIVERGED"] = {"hit": div_hit, "div": {k: stat[k]["div_frac"] for k in ARMS}, "N_DIV_MAX": N_DIV_MAX}

    # ③EMPTY-CRITERION
    anchor = stat["busy"]["c0_anchor_frac"]
    tab["3_EMPTY-CRITERION"] = {"hit": bool(anchor is None or anchor < ANCHOR_FRAC),
                                "c0_anchor_frac": anchor, "ANCHOR_FRAC": ANCHOR_FRAC}

    p_dec, p_bus = stat["decay"]["p_HCMP"], stat["busy"]["p_HCMP"]
    med_bus, med_dec = stat["busy"]["km_median"], stat["decay"]["km_median"]
    dec_reached = stat["decay"]["km_reached"]

    tab["4_DECAY-DOMINANT"] = {"hit": bool(p_dec is not None and p_dec >= P1A_FALSIFY),
                               "p_decay": p_dec, "FALSIFY": P1A_FALSIFY}
    p1a_ok = p_dec is not None and p_dec <= P1A_MAX
    p1b_ok = not dec_reached            # KM 中位未达 ⇒ L_decay > 32
    p2a_ok = p_bus is not None and p_bus >= P2A_MIN
    p2b_ok = (med_bus is not None) and med_bus <= MED_BUSY_MAX
    tab["5_INTERFERENCE-WEAK"] = {"hit": not (p1b_ok and p2a_ok and p2b_ok),
                                 "p1b_med_decay_gt_32": p1b_ok, "p2a": p2a_ok, "p2b": p2b_ok,
                                 "p_busy": p_bus, "med_busy": med_bus, "med_decay": med_dec,
                                 "MED_BUSY_MAX": MED_BUSY_MAX}
    tab["6_PARTIAL-DECAY"] = {"hit": bool(p_dec is not None and P1A_MAX <= p_dec < P1A_FALSIFY),
                              "p_decay": p_dec}
    # ⑦MARGINAL
    m_pdec = p_dec is not None and min(abs(p_dec - P1A_MAX), abs(p_dec - P1A_FALSIFY)) < se3(p_dec, stat["decay"]["n_use"])
    m_pbus = p_bus is not None and abs(p_bus - P2A_MIN) < se3(p_bus, stat["busy"]["n_use"])
    tab["7_MARGINAL"] = {"hit": bool(m_pdec or m_pbus), "near_p_decay": m_pdec, "near_p_busy": m_pbus}

    order = ["1_INCOMPLETE", "2_DIVERGED", "3_EMPTY-CRITERION", "4_DECAY-DOMINANT",
             "5_INTERFERENCE-WEAK", "6_PARTIAL-DECAY", "7_MARGINAL"]
    verdict, rn = "PASS", 8
    for i, k in enumerate(order):
        if tab[k]["hit"]:
            verdict, rn = k.split("_", 1)[1], i + 1
            break

    # P-3 正控：p_busy 必须显著 > p_decay
    p3_ok = (p_bus is not None and p_dec is not None
             and (p_bus - p_dec) > math.sqrt(se3(p_bus, stat["busy"]["n_use"]) ** 2 / 9 + se3(p_dec, stat["decay"]["n_use"]) ** 2 / 9) * 3)
    out = {"card": "DIR-48 rev 1.1 (md5 62022454dcefd936b000a347f709b18b)",
           "verdict": verdict, "rule_hit": {"n": rn, "name": verdict},
           "H48": {"P1a_p_decay_le_0.30": p1a_ok, "P1b_med_decay_gt_32": p1b_ok,
                   "P2a_p_busy_ge_0.90": p2a_ok, "P2b_med_busy_le_3": p2b_ok},
           "stats": stat, "P3_positive_control_ok": bool(p3_ok),
           "decomposition": {"med_busy_pairs": med_bus, "med_decay_lower_bound_pairs": W_EXP["decay"] if not dec_reached else med_dec,
                             "ratio_lower_bound": (W_EXP["decay"] / med_bus) if med_bus else None,
                             "tau_ref_pairs": 100 / 16,
                             "med_busy_in_tau_ref": (med_bus * 16) / 100 if med_bus else None},
           "table": tab, "n_violations": len(viol), "violations": viol[:8],
           "frozen": {"RHO": RHO, "H_CMP": H_CMP, "P1A_MAX": P1A_MAX, "P1A_FALSIFY": P1A_FALSIFY,
                      "P2A_MIN": P2A_MIN, "MED_BUSY_MAX": MED_BUSY_MAX, "N_DIV_MAX": N_DIV_MAX,
                      "ANCHOR": [ANCHOR_C0_MIN, ANCHOR_FRAC], "SIG_K": SIG_K,
                      "N_INST": N_INST, "SEED": [SEED_LO, SEED_HI], "W": W_EXP}}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
