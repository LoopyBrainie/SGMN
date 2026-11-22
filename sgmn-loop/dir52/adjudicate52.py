#!/opt/pytorch/bin/python
r"""DIR-52 · **独立裁决器**（卡 rev 1.0 §4 判定表 + rev 1.1 的成本订正注释）。

★ 不 import 实现、不 import 装置：只读落盘 `jsonl`，自己重算 `EARLY/LATE/TOTAL/SHAPE`、`se`、`t`。
★ 阈值、`n`、`δ*`、分辨率闸门**全部从命令行传入**（卡冻结值），文件内**无魔数阈值**。

判定分支（互斥、按序）：
  ① P3 不过 ⇒ `INVALID`
  ② P4 不过（`3se > 0.000861`）⇒ `INFEASIBLE-POWER`
  ③ P1 过（`mean(SHAPE) < −max(δ*, 3se)`）⇒ `PASS`
  ④ P2 过（`mean(SHAPE) > +max(δ*, 3se)`）⇒ `EARLY-LOCALIZED`
  ⑤ `3se ≤ δ*` 且两侧都不过 ⇒ `NOT-LOCALIZED`
  ⑥ 否则 ⇒ `INCONCLUSIVE-BELOW-RESOLUTION`
"""
import argparse
import json
import math
import os
from statistics import NormalDist


def mean(v):
    return float(sum(v) / len(v)) if v else float("nan")


def sd(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return float(math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)))


def t975(nu):
    if nu <= 0:
        return float("nan")
    z = NormalDist().inv_cdf(0.975)
    v = float(nu)
    return (z + (z ** 3 + z) / (4 * v) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * v * v)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * v ** 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--delta-star", type=float, required=True)
    ap.add_argument("--power-gate", type=float, required=True, help="P4：3se 必须 ≤ 该值")
    ap.add_argument("--n-expected", type=int, required=True)
    ap.add_argument("--max-indet-frac", type=float, required=True)
    ap.add_argument("--min-n-eff", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-segment", nargs=2, type=int, default=None,
                    help="可选：冻结种子段 [lo hi]；只作**越界报告**，不改判据")
    a = ap.parse_args()

    # ★★ 加固（2026-11-16，Cycle 77）—— 三个**纯健壮性/报告**补正，**均不改判据**：
    #   ① `json.loads` 包 try/except：8 小时 4 进程的落盘若被中途截断（末行半截），
    #      原实现会**整场崩溃**；现改为**跳过并计数**（该行不成为样本，由
    #      `n_missing_vs_expected` 如实反映缺失）。
    #   ② 重复种子：原实现 `rows[seed]=r` 会**静默覆盖**（重跑/错 seed0 时会悄悄换掉样本）；
    #      现改为**首次为准 + 计数**（确定性、可审计）。
    #   ③ 可选 `--seed-segment`：报告**越界种子数**（冻结种子段是完整性约束，原实现不检查）。
    #   ★ 三者都**不**进入任何判据谓词；`P1..P4` 与分支序一字未动。
    rows, n_lines, unparsable, dup_seeds = {}, 0, [], []
    for p in a.shards:
        with open(p, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                try:
                    r = json.loads(line)
                    sd_ = int(r["seed"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    unparsable.append({"file": os.path.basename(p), "line": ln, "err": type(e).__name__})
                    continue
                if sd_ in rows:
                    dup_seeds.append(sd_)
                    continue
                rows[sd_] = r

    # ★ 行自洽复算（数据完整性，**不是判据**）：若行内含 `phi` 的逐轮 μ，
    #   则从 μ 独立重算 EARLY/LATE/TOTAL/SHAPE，与行内派生字段比对。
    #   ★ 只在**能复算**时执行（合成/旧行无 `phi` ⇒ 跳过）；不一致的行按「无效」剔除，
    #     与 `identity_err` 同属有效性检查（阈值 0、不改任何判据）。
    derive_bad = {}
    for s, r in list(rows.items()):
        ph = r.get("phi")
        if not ph or "0.0" not in ph or "0.9" not in ph:
            continue
        try:
            def Gf(p_, rho):
                return ph[p_]["mu"]["K"][str(rho)] - ph[p_]["mu"]["F"][str(rho)]
            e = Gf("0.9", 2) - Gf("0.0", 2)
            t_ = Gf("0.9", 4) - Gf("0.0", 4)
            d = max(abs(e - r["EARLY"]), abs(t_ - r["TOTAL"]),
                    abs((2 * e - t_) - r["SHAPE"]),
                    abs((t_ - e) - r["LATE"]) if "LATE" in r else 0.0)
        except (KeyError, TypeError):
            derive_bad[s] = "unreadable_phi"
            continue
        if d > 1e-15:
            derive_bad[s] = d

    seeds = sorted(rows)
    # P3a：有效性（数值有限 + 恒等式成立）
    eff, bad = [], {"nonfinite": 0, "identity": 0}
    for s in seeds:
        r = rows[s]
        if not r.get("finite", True) or not all(math.isfinite(r[k]) for k in ("EARLY", "TOTAL", "SHAPE")):
            bad["nonfinite"] += 1
            continue
        if r["identity_err"] > 1e-12:
            bad["identity"] += 1
            continue
        if s in derive_bad:
            bad["derived_fields"] = bad.get("derived_fields", 0) + 1
            continue
        eff.append(s)
    n_eff = len(eff)
    indet = 1 - n_eff / max(len(seeds), 1)

    # ★ 报告口径补正（2026-11-16，Cycle 76）：`indet` 的分母是**已落盘行数**，
    #   若某分片中途退出（如 190/200 行），缺失的 10 行**不**计入 indet ⇒ 会报 0%。
    #   ⇒ 另报一个**纯报告**字段 `indet_vs_expected`（分母 = `--n-expected`）。
    #   ★★ **不动任何判据输入**：P3 仍只用 `indet` 与 `n_eff`（冻结口径）。
    #   注：在本卡冻结值（`n_expected=200`、`min_n_eff=160=0.8·200`）下，
    #   `indet ≤ 0.20 ⟺ n_eff ≥ 160` ⇒ 两口径对 P3 **恒等价**；缺失行只在
    #   `n_eff < 160` 时才有影响，而那时两种口径**都**判 INVALID。故本字段不影响任何分支。
    n_missing = max(a.n_expected - len(seeds), 0)
    indet_vs_expected = 1 - n_eff / max(a.n_expected, 1)

    shape = [rows[s]["SHAPE"] for s in eff]
    early = [rows[s]["EARLY"] for s in eff]
    late = [rows[s]["LATE"] for s in eff]
    total = [rows[s]["TOTAL"] for s in eff]

    m_sh, s_sh = mean(shape), sd(shape)
    se_sh = s_sh / math.sqrt(n_eff) if n_eff > 1 else float("nan")
    t = t975(n_eff - 1) if n_eff > 1 else float("nan")
    thr = max(a.delta_star, 3 * se_sh)

    p3 = (indet <= a.max_indet_frac) and (n_eff >= a.min_n_eff)
    p4 = (3 * se_sh) <= a.power_gate
    p1 = m_sh < -thr
    p2 = m_sh > thr
    if not p3:
        outcome = "INVALID"
    elif not p4:
        outcome = "INFEASIBLE-POWER"
    elif p1:
        outcome = "PASS"
    elif p2:
        outcome = "EARLY-LOCALIZED"
    elif 3 * se_sh <= a.delta_star:
        outcome = "NOT-LOCALIZED"
    else:
        outcome = "INCONCLUSIVE-BELOW-RESOLUTION"

    out = {
        "n_rows": len(seeds), "n_eff": n_eff, "indeterminate_frac": indet,
        "n_missing_vs_expected": n_missing, "indet_vs_expected": indet_vs_expected,
        "integrity": {"n_lines": n_lines, "n_unique_seeds": len(rows),
                      "n_unparsable": len(unparsable), "unparsable_detail": unparsable[:10],
                      "n_duplicate_seeds": len(dup_seeds), "duplicate_seeds": sorted(set(dup_seeds))[:10],
                      "seed_segment": a.seed_segment,
                      "n_outside_segment": (
                          sum(1 for s in rows if not (a.seed_segment[0] <= s <= a.seed_segment[1]))
                          if a.seed_segment else None),
                      "n_lines_ge_expected": n_lines >= a.n_expected,
                      "note": "完整性/健壮性报告；不参与 P1..P4 任何谓词"},
        "invalid_reasons": bad,
        "derive_mismatch_detail": {str(k): v for k, v in list(derive_bad.items())[:10]},
        "derive_checked": sum(1 for s in rows if rows[s].get("phi")),
        "seed_min": min(seeds) if seeds else None, "seed_max": max(seeds) if seeds else None,
        "n_expected": a.n_expected,
        "frozen": {"delta_star": a.delta_star, "power_gate": a.power_gate,
                   "max_indet_frac": a.max_indet_frac, "min_n_eff": a.min_n_eff},
        "SHAPE": {"mean": m_sh, "sd": s_sh, "se": se_sh, "t975": t, "threshold": thr,
                  "n_pos": sum(1 for x in shape if x > 0), "n_neg": sum(1 for x in shape if x < 0)},
        "EARLY": {"mean": mean(early), "sd": sd(early), "se": sd(early) / math.sqrt(n_eff) if n_eff > 1 else None},
        "LATE": {"mean": mean(late), "sd": sd(late)},
        "TOTAL": {"mean": mean(total), "sd": sd(total),
                  "se": sd(total) / math.sqrt(n_eff) if n_eff > 1 else None,
                  "n_pos": sum(1 for x in total if x > 0)},
        "D67_cross_check": {"D67_dG": 0.000860862, "D67_3se": 0.000810078,
                            "this_TOTAL_mean": mean(total),
                            "note": "只作一致性报告，不作判据/不改 D-67"},
        "decision": {"P1_late_localized": p1, "P2_early_localized": p2,
                     "P3_valid": p3, "P4_resolvable": p4, "outcome": outcome},
        "rows": {str(s): {"SHAPE": rows[s]["SHAPE"], "EARLY": rows[s]["EARLY"],
                          "LATE": rows[s]["LATE"], "TOTAL": rows[s]["TOTAL"]} for s in eff},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== DIR-52 裁决 ===")
    print(f"  行数 {len(seeds)}  有效 n_eff={n_eff}  剔除率={indet:.4f}  无效原因={bad}")
    if n_missing:
        print(f"  ★ 缺失 {n_missing} 行（期望 {a.n_expected}）⇒ 按期望计剔除率={indet_vs_expected:.4f}"
              f"（纯报告；P3 仍用冻结口径）")
    print(f"  种子段 {out['seed_min']}..{out['seed_max']}（期望 {a.n_expected}）")
    _ig = out["integrity"]
    if _ig["n_unparsable"] or _ig["n_duplicate_seeds"] or _ig["n_outside_segment"]:
        print(f"  ★★ 完整性告警：不可解析行 {_ig['n_unparsable']} · 重复种子 {_ig['n_duplicate_seeds']}"
              f" · 越界种子 {_ig['n_outside_segment']}（详见 integrity）")
    else:
        print(f"  完整性：行 {_ig['n_lines']} 全部可解析、无重复种子"
              + (f"、无越界种子（段 {a.seed_segment}）" if a.seed_segment else ""))
    print(f"  SHAPE: mean={m_sh:+.6f} sd={s_sh:.6f} se={se_sh:.6f} 3se={3*se_sh:.6f} "
          f"阈值=max(δ*,3se)={thr:.6f}  正/负 {out['SHAPE']['n_pos']}/{out['SHAPE']['n_neg']}")
    print(f"  EARLY={mean(early):+.6f}  LATE={mean(late):+.6f}  TOTAL={mean(total):+.6f} "
          f"(D-67 参考 {0.000860862:+.6f})")
    print(f"  P1(晚段)={p1}  P2(早段)={p2}  P3(有效)={p3}  P4(可分辨 {3*se_sh:.6f}≤{a.power_gate})={p4}")
    print(f"  ★ 结局：**{outcome}**")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
