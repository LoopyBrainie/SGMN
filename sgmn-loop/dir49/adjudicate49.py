#!/opt/pytorch/bin/python
r"""DIR-49 · 独立裁决器：**不 import 实现**，只读 4 臂 jsonl，**自己**从 `c0/c1/cN` 重算全部判据量，逐条转写卡 rev 1.2 §5 判定表。

用法：`adjudicate49.py --out dir49/out/DIR49r10_adjudication.json`
"""
import argparse, glob, json, math, os, sys

ROOT = "/home/ec2-user/sgmn-loop"
ADJ_MIN, DELTA_END_MIN, REL_ORTHO_MAX, N_BAD_MAX, SIG_K = 0.10, 0.30, 0.95, 0.10, 3
N_INST, SEED_LO, SEED_HI = 1000, 29400, 30399
ARMS = ("ortho", "d0.0", "d0.5", "d1.0")
ORDER = ("ortho", "d0.0", "d0.5", "d1.0")      # ov_k 递增序


def load(arm):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, f"dir49/out/DIR49r10_arm_{arm}_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def sd3(v):
    n = len(v)
    if n < 2:
        return None
    m = sum(v) / len(v)
    return SIG_K * ((sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5) / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {arm: load(arm) for arm in ARMS}
    by = {arm: {int(r["seed"]): r for r in R[arm]} for arm in ARMS}
    viol = []
    stats = {}
    for arm in ARMS:
        n_bad = 0; rel1s = []; relNs = []; ovks = []; ovvs = []
        for r in R[arm]:
            c0, c1, cN = r["c0"], r["c1"], r["cN"]
            # ★ 独立重算 rel1 / relN（`DEF-49-4`：rel1 只依赖 (c0,c1)）
            ok0 = math.isfinite(c0) and c0 > 0
            rel1 = (c1 / c0) if (ok0 and math.isfinite(c1)) else None
            relN = (cN / c0) if (ok0 and math.isfinite(cN)) else None
            if rel1 is None:
                n_bad += 1
            else:
                rel1s.append(rel1)
                if abs(rel1 - r["rel1"]) > 1e-12:
                    viol.append({"arm": arm, "seed": r["seed"], "err": "rel1 重算不一致"})
            if relN is not None and abs(relN - (r["relN"] if r["relN"] is not None else relN)) > 1e-12:
                viol.append({"arm": arm, "seed": r["seed"], "err": "relN 重算不一致"})
            if relN is not None:
                relNs.append(relN)
            ovks.append(r["ov_k"]); ovvs.append(r["ov_v"])
        stats[arm] = {"n": len(R[arm]), "n_bad": n_bad, "n_bad_frac": n_bad / len(R[arm]),
                      "n_use": len(rel1s),
                      "rel1_mean": sum(rel1s) / len(rel1s) if rel1s else None,
                      "rel1_3se": sd3(rel1s),
                      "relN_mean": sum(relNs) / len(relNs) if relNs else None,
                      "relN_3se": sd3(relNs),
                      "ov_k_mean": sum(ovks) / len(ovks), "ov_k_max": max(ovks),
                      "ov_v_mean": sum(ovvs) / len(ovvs)}
    tab = {}

    # ①INCOMPLETE
    inc = []
    for arm in ARMS:
        if len(R[arm]) != N_INST:
            inc.append(f"{arm} 行数 {len(R[arm])} != {N_INST}")
        if sorted(by[arm]) != list(range(SEED_LO, SEED_HI + 1)):
            inc.append(f"{arm} 种子集合不符")
    # 四臂 ov_v / c0 逐位相同
    ovv_bad = c0_bad = 0
    for sd in range(SEED_LO, SEED_HI + 1):
        if sd in by["ortho"]:
            ovvs = [by[arm][sd]["ov_v"] for arm in ARMS if sd in by[arm]]
            c0s = [by[arm][sd]["c0"] for arm in ARMS if sd in by[arm]]
            if len(set(ovvs)) != 1:
                ovv_bad += 1
            if len(set(c0s)) != 1:
                c0_bad += 1
    if ovv_bad:
        inc.append(f"四臂 ov_v 不逐位相同 {ovv_bad} 个种子")
    if c0_bad:
        inc.append(f"四臂 c0 不逐位相同 {c0_bad} 个种子")
    if viol:
        inc.append(f"重算不一致 {len(viol)} 行")
    tab["1_INCOMPLETE"] = {"hit": bool(inc), "detail": inc}

    # ②DIVERGED（逐臂取最坏臂）
    nbad_max = max(stats[arm]["n_bad_frac"] for arm in ARMS)
    tab["2_DIVERGED"] = {"hit": bool(nbad_max > N_BAD_MAX), "n_bad_max": nbad_max,
                         "N_BAD_MAX": N_BAD_MAX, "per_arm": {arm: stats[arm]["n_bad_frac"] for arm in ARMS}}

    # 配对集合：四臂 rel1 都可算的实例
    common = [sd for sd in range(SEED_LO, SEED_HI + 1)
              if all(sd in by[arm] and by[arm][sd]["rel1"] is not None for arm in ARMS)]
    def rel(arm, sd):
        return by[arm][sd]["rel1"]
    # P2 端点差（配对）
    d_end = [rel("ortho", sd) - rel("d1.0", sd) for sd in common]
    # P1 相邻配对差
    adj = []
    for sd in common:
        for i in range(len(ORDER) - 1):
            adj.append(rel(ORDER[i], sd) - rel(ORDER[i + 1], sd))
    adj_mean = sum(adj) / len(adj) if adj else None
    adj_3se = sd3(adj)
    d_end_mean = sum(d_end) / len(d_end) if d_end else None
    d_end_3se = sd3(d_end)
    rel_ortho = sum(rel("ortho", sd) for sd in common) / len(common) if common else None
    rel_ortho_3se = sd3([rel("ortho", sd) for sd in common])
    mono_inst = sum(1 for sd in common
                    if all(rel(ORDER[i], sd) > rel(ORDER[i + 1], sd) for i in range(len(ORDER) - 1)))
    p1_ok = adj_mean is not None and adj_mean >= ADJ_MIN and adj_3se is not None and adj_mean >= adj_3se
    p2_ok = d_end_mean is not None and d_end_mean >= DELTA_END_MIN and d_end_3se is not None and d_end_mean >= d_end_3se
    p3_ok = rel_ortho is not None and rel_ortho <= REL_ORTHO_MAX

    tab["3_NO-KEY-MODULATION"] = {"hit": not p2_ok, "DELTA_END": d_end_mean, "3se": d_end_3se,
                                  "DELTA_END_MIN": DELTA_END_MIN}
    tab["4_PURE-KEY-SELECTIVE"] = {"hit": not p3_ok, "rel_ortho": rel_ortho, "3se": rel_ortho_3se,
                                   "REL_ORTHO_MAX": REL_ORTHO_MAX}
    tab["5_PARTIAL-MONOTONE"] = {"hit": (p2_ok and not p1_ok), "ADJ": adj_mean, "3se": adj_3se,
                                  "ADJ_MIN": ADJ_MIN}
    # ⑦MARGINAL
    m_adj = adj_mean is not None and adj_3se is not None and abs(adj_mean - ADJ_MIN) < adj_3se
    m_end = d_end_mean is not None and d_end_3se is not None and abs(d_end_mean - DELTA_END_MIN) < d_end_3se
    m_ortho = rel_ortho is not None and rel_ortho_3se is not None and abs(rel_ortho - REL_ORTHO_MAX) < rel_ortho_3se
    tab["6_MARGINAL"] = {"hit": bool(m_adj or m_end or m_ortho), "near_adj": m_adj, "near_end": m_end, "near_ortho": m_ortho}

    order = ["1_INCOMPLETE", "2_DIVERGED", "3_NO-KEY-MODULATION", "4_PURE-KEY-SELECTIVE",
             "5_PARTIAL-MONOTONE", "6_MARGINAL"]
    verdict, rn = "PASS", 7
    for i, k in enumerate(order):
        if tab[k]["hit"]:
            verdict, rn = k.split("_", 1)[1], i + 1
            break

    out = {"card": "DIR-49 rev 1.2 (md5 3977442d90524aa85875e5dd66b476fb)",
           "verdict": verdict, "rule_hit": {"n": rn, "name": verdict},
           "H49": {"P1_adj_ge_0.10_and_3se": p1_ok, "P2_end_ge_0.30_and_3se": p2_ok, "P3_rel_ortho_le_0.95": p3_ok},
           "n_common": len(common),
           "ADJ": {"mean": adj_mean, "3se": adj_3se, "n": len(adj)},
           "DELTA_END": {"mean": d_end_mean, "3se": d_end_3se, "n": len(d_end)},
           "rel_ortho": {"mean": rel_ortho, "3se": rel_ortho_3se},
           "mono_inst_frac": mono_inst / len(common) if common else None,
           "per_arm_rel1": {arm: {"mean": stats[arm]["rel1_mean"], "3se": stats[arm]["rel1_3se"],
                                  "n_use": stats[arm]["n_use"], "n_bad": stats[arm]["n_bad"],
                                  "ov_k_mean": stats[arm]["ov_k_mean"], "ov_k_max": stats[arm]["ov_k_max"],
                                  "ov_v_mean": stats[arm]["ov_v_mean"]} for arm in ARMS},
           "per_arm_relN": {arm: {"mean": stats[arm]["relN_mean"], "3se": stats[arm]["relN_3se"]} for arm in ARMS},
           "table": tab, "n_violations": len(viol), "violations": viol[:8],
           "frozen": {"ADJ_MIN": ADJ_MIN, "DELTA_END_MIN": DELTA_END_MIN, "REL_ORTHO_MAX": REL_ORTHO_MAX,
                      "N_BAD_MAX": N_BAD_MAX, "SIG_K": SIG_K, "N_INST": N_INST, "SEED": [SEED_LO, SEED_HI]}}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
