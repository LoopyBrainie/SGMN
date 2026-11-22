#!/opt/pytorch/bin/python
r"""DIR-52 · **功效分析**（`power52.py`）—— 读探针 JSON，算判据量所需的种子数与 Go 门。

★ 不进判据。用途：把 `probe52a.py` 的逐轮增量换算成「**两分割**（早段 = 轮 1–2 / 晚段 = 轮 3–4）」
  判据量的跨种子 `sd` 与所需 `n`，从而决定「开卡」还是「`INFEASIBLE-POWER`」。

判据量（2 分割，最强信噪比的分解）：
  `ΔG(ρ) := G_{0.9}(ρ) − G_{0}(ρ)`，`G_φ(ρ) = μ_{K,φ}(ρ) − μ_{F,φ}(ρ)`（累计：前 ρ 轮写完后）
  `EARLY := ΔG(2)`，`LATE := ΔG(4) − ΔG(2)`
  `SHAPE := EARLY − LATE`   （M1「日程累积」预测 SHAPE < 0；M2「逐次伤害」预测 SHAPE > 0）

输出：每量的 mean/sd/se@n、所需 n（3se 达 |mean|）、以及「给定预算下的可分辨性」。
"""
import argparse
import json
import math
from statistics import NormalDist


def mean(v):
    return float(sum(v) / len(v)) if v else float("nan")


def sd(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return float(math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)))


def se(v):
    return sd(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def n_needed(sd_v, eff_abs, k=3.0):
    r"""使 `k·se(n) = |eff|` 所需的 `n`（`eff` 用探针的均值估计）。"""
    if not (sd_v > 0) or not (eff_abs > 0):
        return None
    return int(math.ceil((k * sd_v / eff_abs) ** 2))


def t975(nu):
    if nu <= 0:
        return float("nan")
    z = NormalDist().inv_cdf(0.975)
    v = float(nu)
    return z + (z ** 3 + z) / (4 * v) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * v * v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="/home/ec2-user/sgmn-loop/dir52/out/DIR52_probe_a.json")
    ap.add_argument("--budget-h", type=float, default=12.0, help="可接受墙钟（小时）")
    ap.add_argument("--sec-per-seed", type=float, default=184.0, help="2 分割实现下每种子墙钟（秒）")
    ap.add_argument("--out", default="/home/ec2-user/sgmn-loop/dir52/out/DIR52_power.json")
    a = ap.parse_args()
    d = json.load(open(a.probe, encoding="utf-8"))
    per = d["per_seed"]
    n = len(per)

    early, late, total, shape = [], [], [], []
    for x in per:
        dg = x["dG"]
        e = float(sum(dg[:2]))
        l = float(sum(dg[2:]))
        early.append(e)
        late.append(l)
        total.append(e + l)
        shape.append(e - l)

    def rep(name, v):
        m, s, e = mean(v), sd(v), se(v)
        nn = n_needed(s, abs(m))
        return {"name": name, "n": n, "mean": m, "sd": s, "se": e, "3se": 3 * e,
                "ratio_mean_over_3se": abs(m) / (3 * e) if e and e > 0 else None,
                "n_needed_3se": nn,
                "wall_h_at_needed": (nn * a.sec_per_seed / 3600.0) if nn else None,
                "n_pos": sum(1 for y in v if y > 0), "vals": v}

    rows = [rep("EARLY=ΔG(2)", early), rep("LATE=ΔG(4)−ΔG(2)", late),
            rep("TOTAL=ΔG(4)", total), rep("SHAPE=EARLY−LATE", shape)]

    # 4 分割（原计划）也一并报告，用于对比成本
    four = []
    for i in range(4):
        v = [x["dG"][i] for x in per]
        four.append(rep(f"ΔδG_{i+1}", v))

    budget_n = int(a.budget_h * 3600.0 / a.sec_per_seed)
    go = {"budget_h": a.budget_h, "sec_per_seed": a.sec_per_seed, "budget_n_seeds": budget_n}
    for r in rows:
        nn = r["n_needed_3se"]
        r["feasible_in_budget"] = (nn is not None and nn <= budget_n)

    # 决策：SHAPE 是最能区分 M1/M2 的量；只在它可分辨时才开卡
    shape_r = [r for r in rows if r["name"].startswith("SHAPE")][0]
    dec = ("CARD-2WAY" if (shape_r["n_needed_3se"] is not None
                          and shape_r["n_needed_3se"] <= budget_n) else "INFEASIBLE-POWER")
    out = {"probe_wall_s": d.get("wall_s"), "probe_summary": d.get("summary"),
           "rows_2way": rows, "rows_4way": four, "budget": go,
           "decision_hint": dec,
           "aggregate_check": {"probe_dG_total_mean": d["summary"]["dG_total_mean"],
                               "probe_dG_total_3se": d["summary"]["dG_total_3se"],
                               "D67_dG": 0.000860862, "D67_3se": 0.000810078,
                               "D67_sd": 0.001947184}}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== DIR-52 功效分析 ===")
    print(f"  探针 n={n}  聚合 ΔG(4) mean={mean(total):+.6f} 3se={3*se(total):.6f}   "
          f"(D-67: +0.000860862 / 0.000810078)")
    for r in rows + four:
        print(f"  {r['name']:<18s} mean={r['mean']:+.6f} sd={r['sd']:.6f} 3se@n={r['3se']:.6f} "
              f"ratio={('%.2f' % r['ratio_mean_over_3se']) if r['ratio_mean_over_3se'] else 'NA':>5s} "
              f"n_need={r['n_needed_3se']} 墙钟={('%.1f h' % r['wall_h_at_needed']) if r['wall_h_at_needed'] else 'NA'}")
    print(f"  预算 {a.budget_h} h ⇒ {budget_n} 种子")
    print(f"  ★ 决策提示：{dec}")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
