#!/opt/pytorch/bin/python
r"""DIR-47 · 独立裁决器：**不 import 实现**，只读 4 个分片 jsonl，**自己**从逐槽位 `c_s` 重算，逐条转写卡 §5 判定表。

用法：`adjudicate47.py --out dir47/out/DIR47r10_adjudication.json`
"""
import argparse, glob, json, math, os, sys

ROOT = "/home/ec2-user/sgmn-loop"
TAU_REPAIR, DELTA_EQ, CENS_MAX, ANCHOR_MIN, SIG_K = 0.5, 0.15, 0.20, 0.02, 3
N_INST, SEED_LO, SEED_HI, N_SLOTS, M_TOT = 500, 25200, 25699, 7, 13
ARMS = ("nor", "rep_on")


def load(arm):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, f"dir47/out/DIR47r10_arm_{arm}_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def mean(v):
    return sum(v) / len(v) if v else None


def sd(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def check_row(r):
    """逐行独立校验：计数恒等式 + 分组与逐槽位一致（**不 import 实现**）。"""
    errs = []
    slots = {int(k): float(v) for k, v in r["slots"].items()}
    hit = sorted({int(p) for p in r["picks"]})
    rest = [s for s in range(1, 8) if s not in hit]
    if len(hit) + len(rest) != N_SLOTS:
        errs.append("len(hit)+len(rest)!=7")
    if len(slots) + int(r["n_skip"]) != N_SLOTS:
        errs.append("n_read+n_skip!=7")
    if int(r["n_writes"]) != M_TOT:
        errs.append("n_writes!=13")
    for s in range(1, 8):
        if s in slots and not math.isfinite(slots[s]):
            errs.append(f"slot {s} 非有限却计入")
    if r["finite"] is False and slots:
        errs.append("删失却有可读槽")
    if not all(1 <= p <= 7 for p in hit):
        errs.append("pick 越界")
    # 分组均值须与逐槽位重算一致
    ch = mean([slots[s] for s in hit if s in slots])
    cr = mean([slots[s] for s in rest if s in slots])
    ca = mean([slots[s] for s in range(1, 8) if s in slots])
    for name, mine, got in (("C_hit", ch, r["C_hit"]), ("C_rest", cr, r["C_rest"]), ("C_all", ca, r["C_all"])):
        if (mine is None) != (got is None) or (mine is not None and abs(mine - got) > 1e-12):
            errs.append(f"{name} 不一致 mine={mine} got={got}")
    return errs, hit, slots


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args()

    R = {a: load(a) for a in ARMS}
    by = {a: {int(r["seed"]): r for r in R[a]} for a in ARMS}
    viol, bad = [], {}
    for a in ARMS:
        for r in R[a]:
            errs, _, _ = check_row(r)
            if errs:
                viol.append({"arm": a, "seed": r["seed"], "errs": errs})
    tab = {}

    # ①INCOMPLETE
    inc = []
    for a in ARMS:
        if len(R[a]) != N_INST:
            inc.append(f"{a} 行数 {len(R[a])} != {N_INST}")
        if sorted(by[a]) != list(range(SEED_LO, SEED_HI + 1)):
            inc.append(f"{a} 种子集合不符")
    pk_mismatch = [sd for sd in by["nor"] if by["nor"][sd]["picks"] != by["rep_on"].get(sd, {}).get("picks")]
    if pk_mismatch:
        inc.append(f"两臂 picks 不同 {len(pk_mismatch)} 例")
    if viol:
        inc.append(f"结构违规 {len(viol)} 行")
    tab["1_INCOMPLETE"] = {"hit": bool(inc), "detail": inc}

    # 配对集合：两臂都非删失
    pairs = [sd for sd in sorted(by["nor"])
             if by["nor"][sd]["C_hit"] is not None and by["rep_on"][sd]["C_hit"] is not None]
    n_div = {a: sum(1 for r in R[a] if r["C_hit"] is None) for a in ARMS}
    cens = {a: n_div[a] / N_INST for a in ARMS}

    # ②DIVERGED
    div_hit = any(cens[a] > CENS_MAX for a in ARMS)
    tab["2_DIVERGED"] = {"hit": bool(div_hit), "cens": cens, "CENS_MAX": CENS_MAX}

    c_nor_all = mean([by["nor"][sd]["C_all"] for sd in pairs if by["nor"][sd]["C_all"] is not None])
    c_on_hit = [by["rep_on"][sd]["C_hit"] for sd in pairs]
    c_nor_hit = [by["nor"][sd]["C_hit"] for sd in pairs]
    d_hit = [x - y for x, y in zip(c_on_hit, c_nor_hit)]
    d_rest = [by["rep_on"][sd]["C_rest"] - by["nor"][sd]["C_rest"] for sd in pairs
              if by["nor"][sd]["C_rest"] is not None and by["rep_on"][sd]["C_rest"] is not None]
    st = lambda v: {"mean": mean(v), "sd": sd(v), "n": len(v),
                    "3se": SIG_K * sd(v) / math.sqrt(len(v)) if len(v) else None}
    P1, Drest, Dhit = st(c_on_hit), st(d_rest), st(d_hit)

    # ③EMPTY-CRITERION
    tab["3_EMPTY-CRITERION"] = {"hit": bool(c_nor_all is None or c_nor_all < ANCHOR_MIN),
                                "C_nor_all": c_nor_all, "ANCHOR_MIN": ANCHOR_MIN}
    # ④..⑥
    tab["4_NO-REPAIR"] = {"hit": bool(P1["mean"] is not None and P1["mean"] < TAU_REPAIR),
                          "C_on_hit": P1, "TAU_REPAIR": TAU_REPAIR}
    tab["5_GLOBAL-COST"] = {"hit": bool(Drest["mean"] is not None and Drest["mean"] <= -DELTA_EQ),
                            "d_rest": Drest, "DELTA_EQ": DELTA_EQ}
    tab["6_GLOBAL-GAIN"] = {"hit": bool(Drest["mean"] is not None and Drest["mean"] >= DELTA_EQ),
                            "d_rest": Drest, "DELTA_EQ": DELTA_EQ}
    # ⑦MARGINAL
    m1 = P1["mean"] is not None and abs(P1["mean"] - TAU_REPAIR) < P1["3se"]
    m2 = Drest["mean"] is not None and abs(abs(Drest["mean"]) - DELTA_EQ) < Drest["3se"]
    tab["7_MARGINAL"] = {"hit": bool(m1 or m2), "near_TAU": m1, "near_DELTA": m2}

    order = ["1_INCOMPLETE", "2_DIVERGED", "3_EMPTY-CRITERION", "4_NO-REPAIR",
             "5_GLOBAL-COST", "6_GLOBAL-GAIN", "7_MARGINAL"]
    verdict = "PASS"
    for k in order:
        if tab[k]["hit"]:
            verdict = k.split("_", 1)[1]
            break

    # P-3 正控：Δ_hit 必须显著（否则 MEASUREMENT-DEAD）
    p3_ok = Dhit["mean"] is not None and Dhit["mean"] > Dhit["3se"]
    out = {"verdict": verdict, "rule_hit": {"n": order.index(
        [k for k in order if tab[k]["hit"]][0]) + 1 if any(tab[k]["hit"] for k in order) else 8,
        "name": verdict},
        "n_pairs": len(pairs), "n_div": n_div, "cens": cens,
        "C_nor_all": c_nor_all,
        "C_on_hit": P1, "C_nor_hit": st(c_nor_hit), "d_hit": Dhit,
        "C_on_rest": st([by["rep_on"][sd]["C_rest"] for sd in pairs]),
        "C_nor_rest": st([by["nor"][sd]["C_rest"] for sd in pairs]),
        "d_rest": Drest,
        "levels_all": {"nor": st([by["nor"][sd]["C_all"] for sd in pairs]),
                       "rep_on": st([by["rep_on"][sd]["C_all"] for sd in pairs])},
        "P3_positive_control_ok": bool(p3_ok),
        "table": tab, "violations": viol[:10], "n_violations": len(viol),
        "frozen": {"TAU_REPAIR": TAU_REPAIR, "DELTA_EQ": DELTA_EQ, "CENS_MAX": CENS_MAX,
                   "ANCHOR_MIN": ANCHOR_MIN, "SIG_K": SIG_K, "N_INST": N_INST,
                   "SEED": [SEED_LO, SEED_HI]}}
    print(json.dumps({k: out[k] for k in ("verdict", "rule_hit", "n_pairs", "cens",
                                          "C_nor_all", "C_on_hit", "C_nor_hit", "d_hit",
                                          "C_on_rest", "C_nor_rest", "d_rest",
                                          "levels_all", "P3_positive_control_ok", "n_violations")},
                     ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
