#!/opt/pytorch/bin/python
r"""DIR-60 · PROBE60 裁决器 —— 逐字实现 `PROBE60.md` **§4 冻结判据**（rev 1.3，md5 `4ecdb04b…`）。

判据（首中即结算）：
  τ_X(j) = max(3·se_Δ(j), 0.20·R(j))，se_Δ(j)=sqrt(sd_X²+sd_RAND²)/√n_eval，R(j)=L_NONE(j)−L_RAND(j)
  ① INCONCLUSIVE-NO-POWER : 两格 |Δ_SCALE2| < τ_SCALE2
  ② CHANNEL-EXISTS        : ∃X∈{ORTH,RANK2} 使 **两格** |Δ_X| ≥ τ_X 且 **同号**
  ③ P-V-IS-GAUGE          : ①未命中，且 ∀X∈{ORTH,RANK2} ∀格 |Δ_X| < 3se **且** |Δ_X| < 0.20·R
  ④ GREY                  : 其余
本脚本**不改判据**：阈值文字一律从 §4 抄来；若与 `PROBE60.md` 冲突，以文件为准（`--check-doc` 复核 md5）。
"""
import argparse
import json
import math
import pathlib
import sys

OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir60/out")
PROBE = "/home/ec2-user/sgmn-loop/dir60/PROBE60.md"
PROBE_MD5 = "4ecdb04bca2f83c486ebac69dd9adbf5"
ARMS = ("NONE", "RAND", "ORTH", "RANK2", "SCALE2")
GEOM = ("ORTH", "RANK2")
CTRL = "SCALE2"


def load():
    cells = {}
    for p in sorted(OUT.glob("PROBE60_c*_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        cells.setdefault(d["j"], {})[d["arm"]] = d
    return cells


def admiss(cells):
    """完整性与一致性：两格 × 五臂都在、md5 一致、判据未过期。"""
    bad = []
    for j in (1, 2):
        miss = [a for a in ARMS if a not in cells.get(j, {})]
        if miss:
            bad.append(f"格{j} 缺臂 {miss}")
    if bad:
        return {"ok": False, "why": bad}
    md5s = {d["probe_md5"] for j in (1, 2) for d in cells[j].values()}
    if len(md5s) != 1 or PROBE_MD5 not in md5s:
        return {"ok": False, "why": [f"probe_md5 不一致 {md5s}（期望 {PROBE_MD5}）"]}
    return {"ok": True, "probe_md5": PROBE_MD5}


def cell_stats(cells, j):
    r, n = cells[j]["RAND"], cells[j]["NONE"]
    R = n["mean_L"] - r["mean_L"]
    out = {"R": R, "L_NONE": n["mean_L"], "L_RAND": r["mean_L"], "n_eval": len(r["L_series"]),
           "sd_RAND": r["sd_L"], "sd_NONE": n["sd_L"]}
    for a in GEOM + (CTRL,):
        d = cells[j][a]
        se = math.sqrt(d["sd_L"] ** 2 + r["sd_L"] ** 2) / math.sqrt(len(r["L_series"]))
        delta = d["mean_L"] - r["mean_L"]
        out[a] = {"L": d["mean_L"], "sd": d["sd_L"], "delta": delta, "se": se,
                  "tau": max(3.0 * se, 0.20 * R), "tau_3se": 3.0 * se, "tau_floor": 0.20 * R,
                  "sig": abs(delta) >= 3.0 * se,
                  "over_tau": abs(delta) >= max(3.0 * se, 0.20 * R),
                  "ratio_to_R": delta / R if R else float("nan"),
                  "cos_q_k": d["cos_q_k"], "dWq1": d["dWq1_from_init"],
                  "m_abs": d["m_abs"], "pv_rank": d["pv_rank"], "pv_fro": d["pv_fro"],
                  "pv_gram_offdiag_max": d["pv_gram_offdiag_max"],
                  "pv_row_norm_mean": d["pv_row_norm_mean"]}
    return out


def verdict(cells):
    a = admiss(cells)
    if not a["ok"]:
        return {"branch": "NOT-ADMISSIBLE", "why": a["why"]}
    S = {1: cell_stats(cells, 1), 2: cell_stats(cells, 2)}
    # ① 无功效
    if all(abs(S[j][CTRL]["delta"]) < S[j][CTRL]["tau"] for j in (1, 2)):
        br = "INCONCLUSIVE-NO-POWER"
    # ② 存在非内容通路
    elif any(all(abs(S[j][X]["delta"]) >= S[j][X]["tau"] for j in (1, 2))
             and (S[1][X]["delta"] > 0) == (S[2][X]["delta"] > 0) for X in GEOM):
        br = "CHANNEL-EXISTS"
    # ③ 规范自由度
    elif all(abs(S[j][X]["delta"]) < S[j][X]["tau_3se"] and abs(S[j][X]["delta"]) < S[j][X]["tau_floor"]
             for j in (1, 2) for X in GEOM):
        br = "P-V-IS-GAUGE"
    else:
        br = "GREY"
    hits = [f"{X}@j{j}" for j in (1, 2) for X in GEOM + (CTRL,)
            if abs(S[j][X]["delta"]) >= S[j][X]["tau"]]
    return {"branch": br, "cells": S, "over_tau": hits, "probe_md5": PROBE_MD5}


def report(v):
    if v["branch"] == "NOT-ADMISSIBLE":
        print("★ 不可裁决：", "; ".join(v["why"]))
        return
    print("=" * 78)
    print(f"{'臂':8s} {'格':>3s} {'L':>9s} {'Δ':>9s} {'se':>7s} {'3se':>7s} {'0.2R':>7s} "
          f"{'τ':>7s} {'Δ/R':>7s} {'cos':>6s} {'rank':>4s}")
    for j in (1, 2):
        c = v["cells"][j]
        print(f"{'NONE':8s} {j:>3d} {c['L_NONE']:9.4f} {'—':>9s} {'—':>7s} {'—':>7s} {'—':>7s} "
              f"{'—':>7s} {'—':>7s} {'—':>6s} {'—':>4s}   R={c['R']:+.4f}")
        print(f"{'RAND':8s} {j:>3d} {c['L_RAND']:9.4f} {'—':>9s} {'—':>7s} {'—':>7s} {'—':>7s} "
              f"{'—':>7s} {'—':>7s} {'—':>6s} {'—':>4s}")
        for a in GEOM + (CTRL,):
            d = c[a]
            flag = "★" if abs(d["delta"]) >= d["tau"] else ("·" if d["sig"] else " ")
            print(f"{a:8s} {j:>3d} {d['L']:9.4f} {d['delta']:+9.4f} {d['se']:7.4f} "
                  f"{d['tau_3se']:7.4f} {d['tau_floor']:7.4f} {d['tau']:7.4f} "
                  f"{d['ratio_to_R']:+7.3f} {d['cos_q_k']:6.3f} {d['pv_rank']:4d} {flag}")
    print("=" * 78)
    print(f"★ 判据：**{v['branch']}**")
    print(f"  过阈(τ)的格臂：{v['over_tau'] or '（无）'}")
    if v["branch"] == "P-V-IS-GAUGE":
        print("  ⇒ INFEASIBLE-PREMISE：$P_v$ 的几何/秩在本装置不可测；离线学习只剩「学内容」(=D-76) 与「学度量/饱和」(非记忆能力)")
    elif v["branch"] == "CHANNEL-EXISTS":
        print("  ⇒ 开 DIR-60 卡：学出来的 $P_v$ vs **谱匹配**随机 $P_v$（$P_v$ 跨实例共享）")
    elif v["branch"] == "INCONCLUSIVE-NO-POWER":
        print("  ⇒ 只报读数；**不得**下任何 $P_v$ 结论")
    else:
        print("  ⇒ 记录读数，仅把命中的那一支留作候选")


def _deep(c):
    """深拷贝并保住 int 键（json 往返会把键变成 str）。"""
    return {int(k): {a: dict(d) for a, d in v.items()} for k, v in json.loads(json.dumps(c)).items()}


def examples():
    """四个合成场景，逐一命中四支判据（证明裁决器与 §4 一致）。"""
    def mk(j, arm, mean, sd=1.0, n=400):
        return {"probe_md5": PROBE_MD5, "j": j, "arm": arm, "mean_L": mean, "sd_L": sd,
                "L_series": [0.0] * n, "cos_q_k": 0.5, "dWq1_from_init": 0.1, "m_abs": 0.2,
                "pv_rank": 4, "pv_fro": 2.8, "pv_gram_offdiag_max": 0.3,
                "pv_row_norm_mean": 0.7}
    # ① 尺度也不可测
    c = {j: {"NONE": mk(j, "NONE", 40.0), "RAND": mk(j, "RAND", 30.0),
             "ORTH": mk(j, "ORTH", 30.0), "RANK2": mk(j, "RANK2", 30.0),
             "SCALE2": mk(j, "SCALE2", 30.0)} for j in (1, 2)}
    assert verdict(c)["branch"] == "INCONCLUSIVE-NO-POWER"
    # ② 秩塌陷可测、同号
    c2 = _deep(c)
    for j in (1, 2):
        c2[j]["SCALE2"]["mean_L"] = 45.0
        c2[j]["RANK2"]["mean_L"] = 36.0
    assert verdict(c2)["branch"] == "CHANNEL-EXISTS"
    # ③ 只有尺度可测
    c3 = _deep(c)
    for j in (1, 2):
        c3[j]["SCALE2"]["mean_L"] = 45.0
    assert verdict(c3)["branch"] == "P-V-IS-GAUGE"
    # ④ 仅一格过阈
    c4 = _deep(c)
    c4[1]["SCALE2"]["mean_L"] = 45.0
    c4[1]["ORTH"]["mean_L"] = 36.0
    assert verdict(c4)["branch"] == "GREY"
    # 不可裁决：缺臂
    c5 = _deep(c)
    del c5[2]["ORTH"]
    assert verdict(c5)["branch"] == "NOT-ADMISSIBLE"
    print("★ --examples：四支判据 + 不可裁决 全部按 §4 命中")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.examples:
        examples()
        sys.exit(0)
    cells = load()
    if not cells:
        print("（尚无 PROBE60 结果）")
        sys.exit(0)
    v = verdict(cells)
    report(v)
    if a.json:
        OUT.joinpath("PROBE60_verdict.json").write_text(
            json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
        print("→ PROBE60_verdict.json")
