#!/opt/pytorch/bin/python
r"""PROBE60B 裁决器 —— 逐字实现 `PROBE60B.md` **rev 1.0 §4** 的冻结判据（md5 `539a84d9…`）。

判据（按序命中，首中即结算）：
  1 NO-TWO-CELL-POWER        : ∀s∈{0.5,1.5,3,4} 无剂量在两格同时 |Δ_s| ≥ τ_s
  2 NO-PLATEAU               : 有两格功效，但 s∈{0.5,1.5} 某格 |Δ| > 0.20R
  3 TWO-CELL-POWER-NO-CLIFF  : 功效+平台，但 s∈{3,4} 无任何格 |Δ| ≥ 0.20R
  4 CLIFF-AND-PLATEAU        : 功效+平台+某格在 s∈{3,4} |Δ| ≥ 0.20R   ⇒ H60B 成立
  5 GREY-B                   : 其余
基线复用：`PROBE60_c{j}_{RAND,NONE,SCALE2}.json`；本探针只提供 s∈{0.5,1.5,3,4}。
"""
import argparse
import json
import math
import pathlib
import sys

OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir60/out")
MD5 = "539a84d991381e6babfcc56ff8d1a88c"
CELLS = (1, 2)
NEW = (0.5, 1.5, 3.0, 4.0)          # 判据 1 只认这四个新剂量
PLATEAU = (0.5, 1.5)
CLIFF = (3.0, 4.0)


def _load():
    cells = {}
    for j in CELLS:
        d = {}
        for arm in ("RAND", "NONE", "SCALE2"):
            p = OUT / f"PROBE60_c{j}_{arm}.json"
            if p.exists():
                d[{1: 1.0, 2: 2.0}[j] if False else {"RAND": 1.0, "NONE": 1.0, "SCALE2": 2.0}[arm]] = \
                    json.loads(p.read_text(encoding="utf-8")) if arm != "NONE" else None
                d.setdefault("NONE", None)
                if arm == "NONE":
                    d["NONE"] = json.loads(p.read_text(encoding="utf-8"))
        for s in NEW:
            p = OUT / f"PROBE60B_c{j}_s{s:g}.json"
            if p.exists():
                d[s] = json.loads(p.read_text(encoding="utf-8"))
        cells[j] = d
    return cells


def load():
    cells = {}
    for j in CELLS:
        d = {"NONE": json.loads((OUT / f"PROBE60_c{j}_NONE.json").read_text(encoding="utf-8"))
             if (OUT / f"PROBE60_c{j}_NONE.json").exists() else None}
        for s, arm in ((1.0, "RAND"), (2.0, "SCALE2")):
            p = OUT / f"PROBE60_c{j}_{arm}.json"
            if p.exists():
                d[s] = json.loads(p.read_text(encoding="utf-8"))
        for s in NEW:
            p = OUT / f"PROBE60B_c{j}_s{s:g}.json"
            if p.exists():
                d[s] = json.loads(p.read_text(encoding="utf-8"))
        cells[j] = d
    return cells


def admiss(cells):
    bad = []
    for j in CELLS:
        for s in NEW:
            if s not in cells[j]:
                bad.append(f"格{j} 缺剂量 s={s:g}")
        for s in (1.0, 2.0):
            if s not in cells[j]:
                bad.append(f"格{j} 缺基线 s={s:g}")
    if bad:
        return {"ok": False, "why": bad}
    md5s = {cells[j][s]["probe_md5"] for j in CELLS for s in NEW}
    if md5s != {MD5}:
        return {"ok": False, "why": [f"probe_md5 不一致 {md5s}（期望 {MD5}）"]}
    for j in CELLS:
        for s in NEW:
            d = cells[j][s]
            if d["pv_seed"] != 20263431 or d["pk1_seed"] != (20263351 if j == 1 else 20263352):
                return {"ok": False, "why": [f"格{j} s={s:g} 种子与基线不符"]}
    return {"ok": True}


def table(cells, j):
    r = cells[j][1.0]
    n = 400
    rows = {}
    for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        d = cells[j][s]
        delta = d["mean_L"] - r["mean_L"]
        se = math.sqrt(d["sd_L"] ** 2 + r["sd_L"] ** 2) / math.sqrt(n)
        rows[s] = {"L": d["mean_L"], "sd": d["sd_L"], "delta": delta, "se": se,
                   "tau": max(3 * se, 0.20 * r and 0.0) if False else None,
                   "cos_q_k": d.get("cos_q_k"), "fro": d.get("pv_fro"),
                   "gram": d.get("pv_gram_offdiag_max")}
    R = cells[j]["NONE"]["mean_L"] - r["mean_L"]
    for s, row in rows.items():
        row["tau"] = max(3 * row["se"], 0.20 * R)
        row["ratio_to_R"] = row["delta"] / R if R else float("nan")
        row["over_tau"] = abs(row["delta"]) >= row["tau"]
        row["over_floor"] = abs(row["delta"]) >= 0.20 * R
    return {"R": R, "rows": rows}


def verdict(cells):
    a = admiss(cells)
    if not a["ok"]:
        return {"branch": "NOT-ADMISSIBLE", "why": a["why"]}
    T = {j: table(cells, j) for j in CELLS}
    power = [s for s in NEW if all(T[j]["rows"][s]["over_tau"] for j in CELLS)]
    if not power:
        br = "NO-TWO-CELL-POWER"
    elif any(T[j]["rows"][s]["over_floor"] for s in PLATEAU for j in CELLS):
        br = "NO-PLATEAU"
    elif not any(T[j]["rows"][s]["over_floor"] for s in CLIFF for j in CELLS):
        br = "TWO-CELL-POWER-NO-CLIFF"
    else:
        br = "CLIFF-AND-PLATEAU"
    cliff = {j: {"last_plateau_s": max([s for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
                                       if not T[j]["rows"][s]["over_floor"]], default=None),
                 "first_cliff_s": min([s for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
                                       if T[j]["rows"][s]["over_floor"]], default=None)}
             for j in CELLS}
    return {"branch": br, "cells": T, "two_cell_power_doses": power, "cliff": cliff,
            "diag": diagnostics(cells), "probe_md5": MD5}


def report(v):
    if v["branch"] == "NOT-ADMISSIBLE":
        print("★ 不可裁决：", "; ".join(v["why"]))
        return
    for j in CELLS:
        t = v["cells"][j]
        print(f"— 格{j}（R={t['R']:+.4f}）" + "-" * 52)
        print(f"{'s':>5} {'L':>9} {'Δ':>9} {'Δ/R':>7} {'se':>6} {'τ':>7} {'cos':>7} {'fro':>6} "
              f"{'gram':>6} {'τ?':>3} {'0.2R?':>5}")
        for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            r = t["rows"][s]
            star = "★" if r["over_tau"] else ("·" if r["over_floor"] else " ")
            print(f"{s:>5.1f} {r['L']:9.4f} {r['delta']:+9.4f} {r['ratio_to_R']:+7.3f} "
                  f"{r['se']:6.4f} {r['tau']:7.4f} {r['cos_q_k']:+7.3f} {r['fro']:6.3f} "
                  f"{r['gram']:6.3f} {'Y' if r['over_tau'] else '.':>3} "
                  f"{'Y' if r['over_floor'] else '.':>5} {star}")
    if "diag" in v:
        for j in CELLS:
            g = v["diag"][j]
            print(f"— 格{j} 诊断：斜率 dL/ds = {g['slope_per_unit_s']:+.4f} ± {g['slope_se']:.4f}"
                  f"（{g['slope_sigmas']:.1f}σ，{g['slope_over_R']*100:+.2f}%R / 单位 s）")
            print("    逐点 |Δ|/se ：" + "  ".join(
                f"s={s:g}:{g['rows'][s]['sigmas']:.1f}σ" for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)))
        ag = v["diag"]["slope_agree"]
        print(f"    两格斜率同号={ag['same_sign']} 相对差={ag['rel_diff']:.3f}"
              f"（s1={ag['s1']:+.4f} s2={ag['s2']:+.4f}）")
    print("=" * 78)
    print(f"★ 判据：**{v['branch']}**")
    print(f"  两格过 τ 的剂量：{v['two_cell_power_doses'] or '（无）'}")
    print(f"  悬崖区间：{v['cliff']}")
    if v["branch"] == "CLIFF-AND-PLATEAU":
        print("  ⇒ H60B 成立：PROBE60 的 ④ 归因为 DEF-60-6 子句不对称（而非无功效）")
    elif v["branch"] == "NO-TWO-CELL-POWER":
        print("  ⇒ 尺度支收束：$P_v$ 只能单格测；PROBE60 ④ 永久 GREY；转下一候选")


def diagnostics(cells):
    """★ 只作**读数**，不参与任何判定：把「统计可检性」与「材料阈值」分开，
    并估计平台段斜率 dL/ds（三点 0.5/1.0/1.5 的最小二乘 + 逐点 se）。"""
    out = {}
    for j in CELLS:
        r = cells[j][1.0]
        rows = {}
        for s in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            d = cells[j][s]
            delta = d["mean_L"] - r["mean_L"]
            se = math.sqrt(d["sd_L"] ** 2 + r["sd_L"] ** 2) / math.sqrt(400)
            rows[s] = {"delta": delta, "se": se, "sigmas": abs(delta) / se if se else None}
        R = cells[j]["NONE"]["mean_L"] - r["mean_L"]
        xs = [0.5, 1.0, 1.5]
        ys = [rows[s]["delta"] for s in xs]
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        # 斜率的标准误：用各点 se 的均方根当作 y 的噪声
        se_y = math.sqrt(sum(rows[s]["se"] ** 2 for s in xs) / n)
        se_slope = se_y / math.sqrt(sxx)
        out[j] = {"R": R, "rows": rows, "slope_per_unit_s": slope,
                  "slope_se": se_slope, "slope_over_R": slope / R,
                  "slope_sigmas": abs(slope) / se_slope if se_slope else None}
    # 两格一致性：斜率是否同号、量级是否接近
    s1, s2 = out[1]["slope_per_unit_s"], out[2]["slope_per_unit_s"]
    out["slope_agree"] = {"s1": s1, "s2": s2, "same_sign": (s1 > 0) == (s2 > 0),
                          "rel_diff": abs(s1 - s2) / max(abs(s1), abs(s2)) if s1 or s2 else None}
    return out


def examples():
    """自检：**可达支**逐支命中 + 证明 ③/⑤ 在冻结阈值下不可达（`DEF-60B-2`）。"""
    def mk(j, s, L, sd=1.0):
        return {"probe_md5": MD5, "j": j, "s": s, "mean_L": L, "sd_L": sd,
                "pv_seed": 20263431, "pk1_seed": 20263351 if j == 1 else 20263352,
                "cos_q_k": 0.5, "pv_fro": 2.2338 * max(s, 1e-9),
                "pv_gram_offdiag_max": 0.999 * max(s, 1e-9) ** 2}

    def frame(deltas, R=24.0):
        """deltas[j][s] ⇒ 构造两格数据（s=1 基线 L=2.0，NONE=2.0+R）。"""
        c = {}
        for j in (1, 2):
            cj = {1.0: mk(j, 1.0, 2.0), 2.0: mk(j, 2.0, 2.0 + deltas[j].get(2.0, 0.0)),
                  "NONE": mk(j, 0, 2.0 + R)}
            for s in NEW:
                cj[s] = mk(j, s, 2.0 + deltas[j].get(s, 0.0))
            c[j] = cj
        return c

    # ① 无两格功效：所有剂量都远低于 0.2R=4.8
    d1 = {j: {s: (1.0 if j == 1 else 0.5) for s in NEW} for j in (1, 2)}
    assert verdict(frame(d1))["branch"] == "NO-TWO-CELL-POWER"
    # ② 功效在平台剂量上（1.5× 超 0.2R）
    d2 = {j: {**{s: 1.0 for s in NEW}, 1.5: 6.0} for j in (1, 2)}
    assert verdict(frame(d2))["branch"] == "NO-PLATEAU", verdict(frame(d2))["branch"]
    # ④ 功效 + 平台 + 悬崖（s=3 过 0.2R）
    d4 = {j: {**{s: 1.0 for s in NEW}, 3.0: 6.0} for j in (1, 2)}
    assert verdict(frame(d4))["branch"] == "CLIFF-AND-PLATEAU", verdict(frame(d4))["branch"]
    # 不可裁决：缺剂量
    c5 = frame(d4)
    del c5[2][3.0]
    assert verdict(c5)["branch"] == "NOT-ADMISSIBLE"
    # ★ DEF-60B-2：穷举 ⇒ ③/⑤ 不可达（τ≥0.2R ⇒ over_tau ⊂ over_floor）
    grid = (-30.0, -6.0, -4.9, -4.0, 0.0, 4.0, 4.9, 6.0, 30.0)
    seen, bad = set(), []
    for x in grid:                     # 单剂量强力
        for s0 in NEW:
            for j2 in (1.0, 6.0):
                d = {j: {s: (x if (j == 1 and s == s0) else (j2 if (j == 2 and s == s0) else 0.0))
                         for s in NEW} for j in (1, 2)}
                seen.add(verdict(frame(d))["branch"])
    for x in grid:                     # 全部剂量同值
        d = {j: {s: x for s in NEW} for j in (1, 2)}
        seen.add(verdict(frame(d))["branch"])
    bad = sorted(seen - {"NO-TWO-CELL-POWER", "NO-PLATEAU", "CLIFF-AND-PLATEAU"})
    assert not bad, f"出现了本应不可达的支：{bad}"
    print(f"★ --examples：可达三支全命中；穷举 {len(grid)}×{len(NEW)}×2 组合只产出 {sorted(seen)}"
          f" ⇒ **③/⑤ 不可达**（τ≥0.2R）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.examples:
        examples()
        sys.exit(0)
    cells = load()
    v = verdict(cells)
    report(v)
    if a.json:
        OUT.joinpath("PROBE60B_verdict.json").write_text(
            json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
        print("→ PROBE60B_verdict.json")
