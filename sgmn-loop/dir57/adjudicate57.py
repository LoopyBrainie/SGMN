#!/opt/pytorch/bin/python
r"""DIR-57 · **独立裁决器**（**不 import** `R1_task57`；只读 four-arm JSON + Stage A JSON，按卡 §4 冻结表**按序命中**）。

卡：`revolve/DIR-57/DIRECTION_rev57_v10.md`（rev 1.0）。判据量：§3；判定表：§4（8 支，只有最后一支 `counted=true`）。
用法：`/opt/pytorch/bin/python dir57/adjudicate57.py` → `dir57/out/DIR57r10_adjudication.json`
"""
import json
import os
import statistics
import sys

OUT = "/home/ec2-user/sgmn-loop/dir57/out"
ARMS = ("m1", "m2", "m4", "m4_int8")
SEED0, N = 59000, 96
T_GRAD = 0.20      # F1 / F0
T_ERASE = 0.10     # F3
T_DISP = 0.20      # F4
T_MIN = 0.50       # F5


def load(arm):
    with open(os.path.join(OUT, "DIR57r10_%s.json" % arm)) as f:
        return json.load(f)


def median(xs):
    return statistics.median(xs)


def main():
    stage = json.load(open(os.path.join(OUT, "DIR57r10_stageA.json")))
    A = {a: load(a) for a in ARMS}
    # ── 结构与来源一致性
    seeds = {a: [r["seed"] for r in A[a]["rows"]] for a in ARMS}
    same_seeds = all(seeds[a] == seeds["m1"] for a in ARMS)
    n_ok = all(len(seeds[a]) == N for a in ARMS)
    range_ok = all(A[a]["seeds"] == [SEED0, SEED0 + N - 1] for a in ARMS)
    rows4 = A["m4"]["rows"]
    rows8 = A["m4_int8"]["rows"]
    m1 = A["m1"]["rows"]
    # ── 独立重算（只从 raw E / E_null 出发）
    S4 = [[r["E_null"] - e for e in r["E"]] for r in rows4]
    # ★ DEF-57-5：int8 臂必须读 `E_int8` / `E_null_int8`（首版误读 `E` ⇒ 把"干扰前"当成"干扰后"）
    S8 = [[r["E_null_int8"] - e for e in r["E_int8"]] for r in rows8]
    E8 = [[e for e in r["E_int8"]] for r in rows8]
    S1 = [s[0] for s in S4]
    S4last = [s[3] for s in S4]
    G = [S4last[i] - S1[i] for i in range(N)]
    pi_min = sum(1 for s in S4 if min(s) > 0) / N
    med8 = [median(s) for s in S8]
    disp8 = [max(e) - min(e) for e in E8]
    S1_bar = median(S1)
    G_bar = median(G)
    M8_bar = median(med8)
    D8_bar = median(disp8)
    sdG = statistics.stdev(G)
    three_se_G = 3 * sdG / (N ** 0.5)
    # 驱动自带字段与独立重算的一致性（merge agreement）
    agree_S4 = max(abs(rows4[i]["S"][j] - S4[i][j]) for i in range(N) for j in range(4))
    agree_S8 = max(abs(rows8[i]["S_int8"][j] - S8[i][j]) for i in range(N) for j in range(4))
    assert rows8[0].get("n_int") == 8, "int8 臂结构缺失"
    # ── 装置门（卡 §4 第 1 行）
    m1_pos = sum(1 for r in m1 if r["S"][0] > 0) / N
    m1_null = sum(1 for r in m1 if r["E_null"] >= 0.70) / N
    finite_ok = all(r["finite"] for r in m1 + rows4) and all(r.get("finite_int8") for r in rows8)
    gate_fail = (not stage["_summary"]["all_pass"]) or (m1_pos < 0.90) or (m1_null < 0.90) or (not finite_ok)
    struct_ok = same_seeds and n_ok and range_ok
    # ── 判定表（按序命中）
    branch, counted, cond = None, False, None
    if gate_fail or not struct_ok:
        branch, counted, cond = "INVALID-DEVICE", False, "gate/结构"
    elif three_se_G >= T_GRAD:
        branch, counted, cond = "INFEASIBLE-POWER", False, "3se(G)>=0.20"
    elif S1_bar <= 0:
        branch, counted, cond = "EVICTED-FIRST-WRITER", False, "F2"
    elif G_bar >= T_GRAD:
        branch, counted, cond = "ORDER-DOMINATED", False, "F1"
    elif D8_bar > T_DISP:
        branch, counted, cond = "SPLIT-UNDER-PRESSURE", False, "F4"
    elif M8_bar <= T_ERASE:
        branch, counted, cond = "ERASED-BY-CONTENT", False, "F3"
    elif pi_min < T_MIN:
        branch, counted, cond = "MINORITY-KEYS-UNREADABLE", False, "F5"
    else:
        branch, counted, cond = "CROSS-KEY-CONSISTENT-AND-CONSERVED", True, "全过"

    res = {"card": "DIR-57 rev 1.0", "n": N, "seed_range": [SEED0, SEED0 + N - 1],
           "structure": {"same_seeds": same_seeds, "n_ok": n_ok, "range_ok": range_ok},
           "merge_agreement": {"agree": bool(agree_S4 == 0.0 and agree_S8 == 0.0),
                               "max_abs_diff_m4_S": agree_S4, "max_abs_diff_m4_int8_S": agree_S8},
           "rule": {"T_GRAD": T_GRAD, "T_ERASE": T_ERASE, "T_DISP": T_DISP, "T_MIN": T_MIN},
           "readings": {"S1_bar": S1_bar, "G_bar": G_bar, "three_se_G": three_se_G,
                        "pi_min": pi_min, "M8_bar": M8_bar, "D8_bar": D8_bar,
                        "m1_pos_frac": m1_pos, "m1_null_frac": m1_null,
                        "S4_by_position_median": [median([s[i] for s in S4]) for i in range(4)],
                        "S8_by_position_median": [median([s[i] for s in S8]) for i in range(4)],
                        "dS_by_position_median": [median([rows8[i]["S"][j] - S8[i][j] for i in range(N)])
                                                  for j in range(4)],
                        "frac_all_S8_positive": sum(1 for s in S8 if min(s) > 0) / N},
           "gate": {"fail": bool(gate_fail), "finite_ok": finite_ok},
           "branch": branch, "hit": cond, "counted": counted}
    with open(os.path.join(OUT, "DIR57r10_adjudication.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    r = res["readings"]
    print("── 结构：same_seeds=%s n=%s range=%s · merge_agreement=%s（Δ=%.1e / %.1e）"
          % (same_seeds, n_ok, range_ok, res["merge_agreement"]["agree"],
             agree_S4, agree_S8))
    print("── 装置门：m1 Pr(S>0)=%.3f · Pr(E_null>=0.70)=%.3f · finite=%s"
          % (m1_pos, m1_null, finite_ok))
    print("── 判据量：S̄1=%+.4f · Ḡ(末−首)=%+.4f (3se=%.4f) · π_min=%.3f · M̄8=%+.4f · D̄8=%.4f"
          % (r["S1_bar"], r["G_bar"], r["three_se_G"], r["pi_min"], r["M8_bar"], r["D8_bar"]))
    print("── m4 逐位置 S 中位：%s" % ["%+.3f" % v for v in r["S4_by_position_median"]])
    print("── int8 逐位置 S 中位：%s · 全键 S>0 的实例 %.3f"
          % (["%+.3f" % v for v in r["S8_by_position_median"]], r["frac_all_S8_positive"]))
    print("★ 命中 %s ⇒ **%s**（counted=%s）" % (cond, branch, counted))


if __name__ == "__main__":
    main()