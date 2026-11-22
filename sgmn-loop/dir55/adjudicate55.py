#!/opt/pytorch/bin/python
r"""DIR-55 · 独立裁决器（**不 import** `R1_task55`；只读它产出的 JSON，按卡 rev 1.2 §3/§4 重算）。

冻结常数**逐字**来自卡片（`A55-12` 由 `stageA55.py` 从卡片正文解析后与本文件比对）：
`T_A=0.20`、`T_B1=0.50`、`T_B2=0.90`、`T_C=0.40`、`T_C_ABS=0.90`、`SURV=0.50`、`STORE=64`、`R_NAT=8`、`K_MIN=0.5`。
"""
import argparse
import json
import math
import os
import statistics
import sys

T_A, T_B1, T_B2, T_C, T_C_ABS = 0.20, 0.50, 0.90, 0.40, 0.90
SURV, STORE, R_NAT, K_MIN = 0.50, 64, 8, 0.25   # DEF-55-10 修正：rev1.4 时此处被误写成 SURV=0.25
CELLS = (("none", 2), ("natural_r8", 2), ("none", 3), ("natural_m1", 3),
         ("reinforce_m1", 3), ("order012", 3), ("order210", 3))


def files(prefix):
    return {k: f"{prefix}_{k[0]}_n{k[1]}.json" for k in CELLS}


def survivors(diag, thr=SURV):
    return [i for i, x in enumerate(diag) if x >= thr]


def load(outdir, prefix="DIR55r13"):
    data = {}
    for key, fn in files(prefix).items():
        path = os.path.join(outdir, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        d = json.load(open(path))
        assert d["n"] == key[1] and d["arm"] == key[0], (d["n"], d["arm"], key)
        data[key] = d
    return data


def compute(data):
    r = {}
    # ── (a) \bar M_8：n=2、8 条独立写入的平均同损差分
    M8 = []
    for row in data[("natural_r8", 2)]["rows"]:
        base = row["diag"]
        assert len(row["reps"]) == R_NAT, len(row["reps"])
        d0 = statistics.fmean([x["diag"][0] - base[0] for x in row["reps"]])
        d1 = statistics.fmean([x["diag"][1] - base[1] for x in row["reps"]])
        M8.append(d0 - d1)
    sd = statistics.stdev(M8)
    cbar = [statistics.fmean([x["diag"][i] for row in data[("natural_r8", 2)]["rows"] for x in row["reps"]])
            for i in (0, 1)]
    r["a"] = {"n_seeds": len(M8), "M8_mean": statistics.fmean(M8), "M8_sd": sd,
              "M8_3se": 3 * sd / math.sqrt(len(M8)), "cbar_after": cbar}
    # ── (b) n=3：唯一幸存率 + 近因率（分母 = 唯一幸存格）
    one, denom, hit = 0, 0, 0
    for arm in ("order012", "order210"):
        for row in data[(arm, 3)]["rows"]:
            S = row["S_after"]
            one += int(len(S) == 1)
            if len(S) == 1:
                denom += 1
                hit += int(row["idx"][S[0]] == row["idx"][-1])
    n_cells = sum(len(data[(a, 3)]["rows"]) for a in ("order012", "order210"))
    r["b"] = {"n_cells": n_cells, "P_one_survivor": one / n_cells,
              "n_unique": denom, "n_recency": hit,
              "P_recency": (hit / denom) if denom else None}
    # ── (c) 定向 vs 非定向（各 1 次追加写入）
    def p0(arm):
        rows = data[(arm, 3)]["rows"]
        return sum(int(0 in row["S_after"]) for row in rows) / len(rows)
    p_nat, p_dir = p0("natural_m1"), p0("reinforce_m1")
    r["c"] = {"P_dir": p_dir, "P_nat": p_nat, "D_c": p_dir - p_nat}
    # ── 有效性读数（#1）
    base2 = data[("none", 2)]["rows"]
    dg = [min(row["diag_store"]) for row in base2]
    of = [max(abs(row["C_store"][0][1]), abs(row["C_store"][1][0])) for row in base2]
    r["validity"] = {"n_seeds": len(base2),
                     "n2_diag_min": min(dg), "n2_off_max": max(of),
                     "n2_diag_median": statistics.median(dg), "n2_off_median": statistics.median(of)}
    return r


def decide(r, stage_ok=True, gate="median"):
    """按卡 §4 六支**按序命中**。`gate`：`median` = rev 2.0 分位口径；`extreme` = rev 1.4 最坏种子口径。"""
    hits = []
    if not stage_ok:
        hits.append(("INVALID-DEVICE", "Stage A 未全过", False))
    v = r["validity"]
    if gate == "median":
        bad = bool(v["n2_diag_median"] < 0.95 or v["n2_off_median"] > 0.30)
        why = f"median diag={v['n2_diag_median']:.4f} / median off={v['n2_off_median']:.4f}"
    else:
        bad = bool(v["n2_diag_min"] < 0.95 or v["n2_off_max"] > 0.15)
        why = f"min diag={v['n2_diag_min']:.4f} / max off={v['n2_off_max']:.4f}"
    if bad:
        hits.append(("INVALID-DEVICE", f"n=2 基线有效性不成立（{why}）", False))
    if not hits and r["a"]["M8_3se"] >= T_A:
        hits.append(("INFEASIBLE-POWER", f"3se(M8)={r['a']['M8_3se']:.4f} >= {T_A}", False))
    if not hits and (abs(r["a"]["M8_mean"]) > T_A or min(r["a"]["cbar_after"]) < 0.30):
        hits.append(("SPLIT-IN-CAPACITY", f"M8={r['a']['M8_mean']:.4f}", False))
    if not hits and (r["b"]["P_one_survivor"] < T_B1 or (r["b"]["P_recency"] or 0.0) < T_B2):
        hits.append(("EVICTION-NOT-RECENCY", f"P1={r['b']['P_one_survivor']:.3f} Prec={r['b']['P_recency']}", False))
    if not hits and (r["c"]["D_c"] < T_C or r["c"]["P_dir"] < T_C_ABS):
        hits.append(("NO-DIRECTED-CONTROL", f"Dc={r['c']['D_c']:.3f} Pdir={r['c']['P_dir']:.3f}", False))
    if not hits:
        hits.append(("PERSONA-SET-NOT-CONSERVED", "三段全部成立", True))
    return {"row": hits[0][0], "reason": hits[0][1], "counted": hits[0][2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/ec2-user/sgmn-loop/dir55/out")
    ap.add_argument("--stage-a", default=None, help="stageA JSON 路径（缺省则读 DIR55r12_stageA.json）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default="DIR55r13")
    ap.add_argument("--gate", choices=("median", "extreme"), default="extreme")
    ap.add_argument("--card", default="rev1.4")
    a = ap.parse_args()
    data = load(a.out_dir, a.prefix)
    r = compute(data)
    stage_ok = True
    if a.stage_a:
        sa = json.load(open(a.stage_a))
        stage_ok = bool(sa.get("all_ok"))
        r["stage_a_all_ok"] = stage_ok
    res = {"dir": "DIR-55", "card": a.card, "gate": a.gate, "prefix": a.prefix,
           "readings": r, "decision": decide(r, stage_ok, a.gate)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()