#!/opt/pytorch/bin/python
r"""DIR-43 · **机制深化分析**（**探索性；不是判据、不产生新裁决、不计入 `SUCCEEDED`**）。

★ **为什么做**：DIR-43 判出 `NOT-ADDRESSING`（$\rho\approx0$）⇒ 但「$\rho$ **为什么**是 0」还有多种形状：
  (a) **均匀地**没有 ⇒ 梯度里根本没有地址信息；
  (b) **集中在某个子群**里没有 ⇒ 只在某些实例/某个量级下才被淹没 ⇒ ⇒ 下一轮（DIR-44：可分性更强的基座）
  该动哪个旋钮，**取决于这个答案**。
★ **纪律**：本文件**只读**已落盘的 900 行，**不引入任何新的判据/阈值**，**不碰** `decide()` ⇒
  ⇒ 它的输出**只能**写成「**探索性观察**」，**不得**写进任何 `PASS/FAIL`。
★ **零算力**：不训练、不跑 rollout，只做统计。

用法（**在 trainer 上**）：`/opt/pytorch/bin/python dir43/analyze43.py`
"""
import glob
import json
import math
import os
import statistics

ROOT = "/home/ec2-user/sgmn-loop"
OUT = f"{ROOT}/dir43/out"


def read_arm(arm):
    rows = []
    for f in sorted(glob.glob(f"{OUT}/DIR43r10_arm_{arm}_shard*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: r["idx"]):
        if r["idx"] in seen:
            continue
        seen.add(r["idx"])
        uniq.append(r)
    return uniq


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    return pearson(rank(x), rank(y))


def quartiles(v):
    """按**秩**分四分位，返回「每段的下标列表」。★ 用秩而不是**成员判定**分组：
    行里只要有一个 NaN，`dict.__eq__` 就恒为 False ⇒ 成员判定会把所有行判为「不在任何段里」。"""
    order = sorted(range(len(v)), key=lambda i: v[i])
    q = len(order) // 4
    return [order[:q], order[q:2 * q], order[2 * q:3 * q], order[3 * q:]] or [[] for _ in range(4)]


def main():
    L = read_arm("learned")
    R = {r["idx"]: r for r in read_arm("rand")}
    D = read_arm("direct")
    out = {"note": "探索性观察；不是判据；不产生新裁决；不计入 SUCCEEDED",
           "n_learned": len(L), "n_rand": len(R), "n_direct": len(D)}

    rho = [r["rho_final"] for r in L]
    rho0 = [r["rho0"] for r in L]
    rhor = [R[r["idx"]]["rho_rand_final"] for r in L if r["idx"] in R]
    rho_paired = [r["rho_final"] for r in L if r["idx"] in R]

    # ── ① ρ 与逐实例量的关系（相关；**不判**）──
    feats = {"L": [r["L"] for r in L], "cos_q_k": [r["cos"] for r in L],
             "q_norm": [r["q_norm"] for r in L], "y_norm": [r["y_norm"] for r in L],
             "m_abs": [r["m_abs"] for r in L], "dWq1": [r["dWq1"] for r in L],
             "gn_Wq1": [r["gn_Wq1"] for r in L], "rho0": rho0,
             "L_full": [r["L_full"] for r in L], "Wmax": [r["Wmax"] for r in L]}
    corr = {}
    print("=== ① ρ(终态) 与逐实例量的相关（探索性）===")
    for k, v in feats.items():
        c = pearson(rho, v)
        s = spearman(rho, v)
        corr[k] = {"pearson": c, "spearman": s}
        print(f"  {k:10s} pearson={c:+.4f}  spearman={s:+.4f}")
    out["corr_rho_final"] = corr

    # ── ② 配对差（learned − rand）与逐实例量的相关 ──
    paired_keys = [r["idx"] for r in L if r["idx"] in R]
    diff = [L[i]["rho_final"] - R[i]["rho_rand_final"] for i in paired_keys]
    sub = [r for r in L if r["idx"] in R]
    corr2 = {}
    print("\n=== ② (ρ_learned − ρ_rand) 与逐实例量的相关 ===")
    for k in ("L", "cos", "q_norm", "dWq1", "gn_Wq1", "rho0"):
        c = pearson(diff, [r[k] for r in sub])
        corr2[k] = c
        print(f"  {k:10s} pearson={c:+.4f}")
    out["corr_diff"] = corr2

    # ── ③ 分层：ρ 是否在某��子群里更大？（按 |L| 与 `cos` 各分四分位）──
    strata = {}
    print("\n=== ③ 分层（探索性）：|L| 四分位 ===")
    qs = quartiles([abs(r["L"]) for r in L])
    idx_by_q = {qi: [L[i] for i in seg] for qi, seg in enumerate(qs) if seg}
    for qi, group in sorted(idx_by_q.items()):
        if len(group) < 5:
            continue
        g = [r["rho_final"] for r in group]
        m = statistics.fmean(g)
        sd = statistics.stdev(g) if len(g) > 1 else float("nan")
        se = sd / math.sqrt(len(g))
        strata[f"L_q{qi+1}"] = {"n": len(group), "rho_mean": m, "rho_3se": 3 * se,
                                "absL_range": [min(abs(r["L"]) for r in group),
                                               max(abs(r["L"]) for r in group)]}
        print(f"  |L| 第{qi+1}分位  n={len(group):3d}  ρ̄={m:+.4f}  3se={3*se:.4f}  "
              f"|L|∈[{min(abs(r['L']) for r in group):.3f},{max(abs(r['L']) for r in group):.3f}]")
    print("\n=== ③ 分层（探索性）：cos_q_k 四分位 ===")
    qs2 = quartiles([r["cos"] for r in L])
    idx2 = {qi: [L[i] for i in seg] for qi, seg in enumerate(qs2) if seg}
    for qi, group in sorted(idx2.items()):
        if len(group) < 5:
            continue
        g = [r["rho_final"] for r in group]
        m = statistics.fmean(g)
        sd = statistics.stdev(g) if len(g) > 1 else float("nan")
        se = sd / math.sqrt(len(g))
        strata[f"cos_q{qi+1}"] = {"n": len(group), "rho_mean": m, "rho_3se": 3 * se,
                                  "cos_range": [min(r["cos"] for r in group),
                                                max(r["cos"] for r in group)]}
        print(f"  cos 第{qi+1}分位  n={len(group):3d}  ρ̄={m:+.4f}  3se={3*se:.4f}  "
              f"cos∈[{min(r['cos'] for r in group):+.4f},{max(r['cos'] for r in group):+.4f}]")
    out["strata"] = strata

    # ── ④ 轨迹的噪声底：ρ(t) 的 sd 随 t 怎么走 ──
    curve = {}
    print("\n=== ④ ρ(t) 的**噪声底**（逐实例 sd）===")
    for t in ("0", "1", "2", "5", "10", "25", "50", "100", "250"):
        vals = [r["rho_curve"][t] for r in L if t in r.get("rho_curve", {})]
        if len(vals) > 2:
            sd = statistics.stdev(vals)
            curve[t] = {"mean": statistics.fmean(vals), "sd": sd,
                        "sem": sd / math.sqrt(len(vals))}
            print(f"  t={t:>3s}  n={len(vals)}  ρ̄={curve[t]['mean']:+.4f}  "
                  f"sd={sd:.4f}  sem={curve[t]['sem']:.4f}  3sem={3*curve[t]['sem']:.4f}")
    out["rho_curve_noise"] = curve

    # ── ⑤ ρ 的分布形状（零附近的对称性）──
    q = quartiles(rho)
    out["rho_quartiles"] = [{"q": i + 1, "lo": min(rho[i] for i in seg),
                             "hi": max(rho[i] for i in seg), "n": len(seg)}
                             for i, seg in enumerate(q) if seg]
    print("\n=== ⑤ ρ(终态) 的四分位 ===")
    for e in out["rho_quartiles"]:
        print(f"  Q{e['q']}  [{e['lo']:+.4f}, {e['hi']:+.4f}]  n={e['n']}")
    print(f"  负值占比 = {sum(1 for x in rho if x < 0)/len(rho):.3f}")

    # ── ⑥ `direct` 臂：可达性（判据非空的佐证）──
    dc = [r["direct_cos_mean"] for r in D]
    dm = [r["direct_cos_min"] for r in D]
    out["direct"] = {"cos_mean_avg": statistics.fmean(dc), "cos_min_avg": statistics.fmean(dm),
                     "cos_min_worst": min(dm), "cos_mean_worst": min(dc),
                     "Wdist_Pk1_mean": statistics.fmean([r["direct_Wdist_Pk1"] for r in D])}
    print("\n=== ⑥ direct 正控（可达性）===")
    print(f"  cos_mean 平均 {out['direct']['cos_mean_avg']:.6f}（最差实例 {out['direct']['cos_mean_worst']:.6f}）")
    print(f"  cos_min  平均 {out['direct']['cos_min_avg']:.6f}（最差实例 {out['direct']['cos_min_worst']:.6f}）")
    print(f"  ‖W − Pk1‖ 平均 {out['direct']['Wdist_Pk1_mean']:.6f}")

    json.dump(out, open(f"{OUT}/DIR43r10_mechanism_analysis.json", "w"), indent=1)
    print(f"\n已写 {OUT}/DIR43r10_mechanism_analysis.json")
    print("★ **探索性**：以上不构成任何判据、不改变 DIR-43 的裁决（`NOT-ADDRESSING`）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
