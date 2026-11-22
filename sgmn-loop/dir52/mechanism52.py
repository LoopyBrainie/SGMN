#!/opt/pytorch/bin/python
r"""DIR-52 机制分析：把 `SHAPE` 逐档、逐臂（F/K）拆开。

★ 这不是判据，是**机制**：判据只由 `adjudicate52.py` 的六支给出。
★ 恒等式（算术，非发现）：`SHAPE = EARLY − LATE = 2ΔG(2) − ΔG(4)`，而
  `ΔG(ρ) = ΔK(ρ) − ΔF(ρ)` ⇒ `SHAPE = −(2ΔF₂ − ΔF₄) + (2ΔK₂ − ΔK₄) ≡ SHAPE_K + SHAPE_F`。
  该分解**必然**精确成立，故它只回答「哪条臂承载了形状」，**不**提供额外证据。
★ 第 1 轮 `ΔδG₁ ≡ 0`（F-52-a）⇒ 4 档分解实际只有 3 个有信息增量；`EARLY` 覆盖轮 1–2，`LATE` 覆盖轮 3–4。
★ `keep_n` / `wnorm` 属**探索性**读数（不预注册、不进判据），只用于解释「为什么」。
用法：mechanism52.py --shards a.jsonl b.jsonl ... [--out FILE]
"""
import argparse
import json
import math


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    # 同一种子只取第一次出现（分片不重叠，纯防御）
    seen, out = set(), []
    for r in rows:
        if r["seed"] in seen:
            continue
        seen.add(r["seed"])
        out.append(r)
    return sorted(out, key=lambda r: r["seed"])


def agg(vals):
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1) if n > 1 else float("nan")
    se = math.sqrt(var / n)
    return {"n": n, "mean": m, "sd": math.sqrt(var), "se": se, "t": m / se if se > 0 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = load(a.shards)
    per = {"dF2": [], "dF4": [], "dK2": [], "dK4": [],
           "shape_F": [], "shape_K": [], "shape": [], "early": [], "late": [],
           "keepF2": [], "keepF4": [], "keepK2": [], "keepK4": []}

    for r in rows:
        if not r.get("finite", True) or r.get("identity_err", 0.0) > 1e-12:
            continue
        ph = r["phi"]
        for rho, key in (("2", "2"), ("4", "4")):
            df = ph["0.9"]["mu"]["F"][rho] - ph["0.0"]["mu"]["F"][rho]
            dk = ph["0.9"]["mu"]["K"][rho] - ph["0.0"]["mu"]["K"][rho]
            if rho == "2":
                per["dF2"].append(df); per["dK2"].append(dk)
            else:
                per["dF4"].append(df); per["dK4"].append(dk)
            per[f"keepF{rho}"].append(ph["0.9"]["keep_n"]["F"][rho])
            per[f"keepK{rho}"].append(ph["0.9"]["keep_n"]["K"][rho])
        dG2 = per["dK2"][-1] - per["dF2"][-1]
        dG4 = per["dK4"][-1] - per["dF4"][-1]
        shF = -(2 * per["dF2"][-1] - per["dF4"][-1])
        shK = +(2 * per["dK2"][-1] - per["dK4"][-1])
        per["shape_F"].append(shF); per["shape_K"].append(shK)
        per["shape"].append(shF + shK)
        per["early"].append(dG2); per["late"].append(dG4 - dG2)

    res = {"n_rows": len(rows), "n_used": len(per["shape"]),
           "note": "机制分析，非判据；分解恒等成立不构成发现"}
    for k, v in per.items():
        res[k] = agg(v)

    # ★ 分解的自洽性（必须精确）：shape_F + shape_K == shape
    res["decomp_err_max"] = max(abs(per["shape_F"][i] + per["shape_K"][i] - per["shape"][i])
                                for i in range(len(per["shape"]))) if per["shape"] else None

    # ★ 探索性：形状与 "后重档保留量之差" 的关系（只作机制线索，绝不进判据）
    if len(per["shape"]) >= 3:
        dk_keep = [per["keepK2"][i] - per["keepK4"][i] for i in range(len(per["shape"]))]
        df_keep = [per["keepF2"][i] - per["keepF4"][i] for i in range(len(per["shape"]))]
        for name, x in (("keepK", dk_keep), ("keepF", df_keep)):
            mx = sum(x) / len(x); my = res["shape"]["mean"]
            sxy = sum((x[i] - mx) * (per["shape"][i] - my) for i in range(len(x)))
            sxx = sum((v - mx) ** 2 for v in x)
            syy = sum((v - my) ** 2 for v in per["shape"])
            res[f"expl_corr_{name}_shape"] = (sxy / math.sqrt(sxx * syy)) if sxx > 0 and syy > 0 else None

    txt = json.dumps(res, indent=1, ensure_ascii=False)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        print(f"WROTE {a.out}")

    print(f"=== DIR-52 机制分解（n_used={res['n_used']}）★ 非判据 ===")
    for lab, k in (("ΔF(2)", "dF2"), ("ΔF(4)", "dF4"), ("ΔK(2)", "dK2"), ("ΔK(4)", "dK4"),
                   ("EARLY=ΔG(2)", "early"), ("LATE=ΔG(4)−ΔG(2)", "late"),
                   ("SHAPE", "shape"), ("  └ F 臂", "shape_F"), ("  └ K 臂", "shape_K")):
        s = res[k]
        print(f"  {lab:20s} {s['mean']:+.6f}  3se={3*s['se']:.6f}  t={s['t']:+.2f}")
    print(f"  分解残差 max|F+K−SHAPE| = {res['decomp_err_max']:.3e}")
    print("  保留量（均值，探索性）: " + ", ".join(
        f"{k}={res[k]['mean']:.1f}" for k in ("keepF2", "keepF4", "keepK2", "keepK4")))
    for k in ("expl_corr_keepK_shape", "expl_corr_keepF_shape"):
        if res.get(k) is not None:
            print(f"  {k} = {res[k]:+.3f}")


if __name__ == "__main__":
    main()
