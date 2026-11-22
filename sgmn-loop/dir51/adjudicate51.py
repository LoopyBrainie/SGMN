#!/opt/pytorch/bin/python
r"""DIR-51 · **独立裁决器**（卡 `rev51_v12.md` §4 判据表）。

读正式轮 5 臂 jsonl ⇒ 配对计算 `D_P^{(a)}(N)` ⇒ 判据表 ⇒ 分支裁决 ⇒ 机制分解 ⇒ 落盘 JSON。
★ 阈值、终点、样本量、门槛**全部从命令行传入**（卡冻结值），本文件**不含任何魔数阈值**。

用法：
  adjudicate51.py --arms dir51/out/DIR51r12_arm_{alpha0,random,ortho,orthoP,replay}.jsonl \
                  --tau 0.xxx --tau-q95 0.yyy --n-expected 1000 --out dir51/out/DIR51r12_adjudication.json
"""
import argparse
import json
import math
import os
from statistics import NormalDist


def t975(nu):
    r"""双侧 95% 的 t 分位数（Cornish–Fisher 展开，ν 大时误差 <1e-5；无 scipy 依赖）。"""
    if nu <= 0:
        return float("nan")
    z = NormalDist().inv_cdf(0.975)
    v = float(nu)
    return (z + (z ** 3 + z) / (4 * v)
            + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * v * v)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * v ** 3))


def mean(v):
    return float(sum(v) / len(v)) if v else float("nan")


def sd(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return float(math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)))


def se(v):
    return sd(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["seed"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    for arm in ("alpha0", "random", "ortho", "orthoP", "replay"):
        ap.add_argument(f"--{arm}", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--tau-q95", type=float, required=True)
    ap.add_argument("--n-expected", type=int, required=True)
    ap.add_argument("--max-indeterminate-frac", type=float, required=True)
    ap.add_argument("--min-n-eff", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = {arm: load(getattr(a, arm)) for arm in ("alpha0", "random", "ortho", "orthoP", "replay")}
    seeds = sorted(set.intersection(*[set(v) for v in arms.values()]))
    if len(seeds) != a.n_expected:
        print(f"!! 警告：种子数 {len(seeds)} ≠ 期望 {a.n_expected}", flush=True)

    # ── 有效性（E1/E2/数值）**配对**判定：同一 seed 的五臂必须同时有效
    eff, reasons = [], {"undef": 0, "e1": 0, "e2": 0}
    for s in seeds:
        r0 = arms["alpha0"][s]
        bad = None
        if any(arms[x][s]["undef"] for x in arms):
            bad = "undef"
        elif r0["c0"] <= r0["c0_rand_q95"]:
            bad = "e1"
        elif r0["n0"] <= r0["n_floor"]:
            bad = "e2"
        if bad:
            reasons[bad] += 1
        else:
            eff.append(s)
    n_eff = len(eff)
    indet_frac = 1 - n_eff / max(len(seeds), 1)

    # ── D_P（配对）
    D = {}
    for arm in ("random", "ortho", "orthoP", "replay"):
        D[arm] = [arms["alpha0"][s]["cs"][-1] - arms[arm][s]["cs"][-1] for s in eff]
    dc_a0 = [arms["alpha0"][s]["cs"][-1] - arms["alpha0"][s]["c0"] for s in eff]   # 无写入对照自然漂移
    # 内容特异性（配对）：D^ortho − D^orthoP
    d_spec = [arms["orthoP"][s]["cs"][-1] - arms["ortho"][s]["cs"][-1] for s in eff]

    m_o, s_o, se_o = mean(D["ortho"]), sd(D["ortho"]), se(D["ortho"])
    m_r, se_r = mean(D["random"]), se(D["random"])
    m_p, se_p = mean(D["orthoP"]), se(D["orthoP"])
    t = t975(n_eff - 1) if n_eff > 1 else float("nan")

    def branch(tau):
        p1 = (m_o > tau) and (m_o > 3 * se_o)
        p2 = m_r > tau
        p3 = (m_o + t * se_o) < tau
        p4 = (indet_frac <= a.max_indeterminate_frac) and (n_eff >= a.min_n_eff)
        if not p4:
            out = "NOT-ADDRESSING"
        elif not p2:
            out = "NOT-ADDRESSING"
        elif p1 and p2 and p4:
            out = "PASS"
        elif (not p1) and p2 and p3:
            out = "FALSIFIED"
        else:
            out = "INCONCLUSIVE"
        return {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "outcome": out,
                "mean_D_ortho": m_o, "se_D_ortho": se_o, "upper_bound": m_o + t * se_o}

    main_b = branch(a.tau)
    sens_b = branch(a.tau_q95)

    # ── 机制分解（P5）
    anchor_last = {arm: mean([arms[arm][s]["c_anchor_last"] for s in eff]) for arm in arms}
    anchor_first = {arm: mean([arms[arm][s]["c_anchor_first"] for s in eff]) for arm in arms}
    key_max = max(max((abs(x) for x in arms[arm][s]["ks"]), default=0.0)
                  for arm in ("ortho", "orthoP") for s in eff)
    vrand_max = max(max((abs(x) for x in arms["random"][s]["vrand_cos"]), default=0.0) for s in eff)

    out = {
        "n_seeds": len(seeds), "n_eff": n_eff, "indeterminate_frac": indet_frac,
        "indeterminate_reasons": reasons,
        "tau_main": a.tau, "tau_q95_sensitivity": a.tau_q95,
        "t975": t, "n_expected": a.n_expected,
        "D_mean": {k: mean(v) for k, v in D.items()},
        "D_sd": {k: sd(v) for k, v in D.items()},
        "D_se": {k: se(v) for k, v in D.items()},
        "dc_alpha0": {"mean": mean(dc_a0), "sd": sd(dc_a0),
                      "median_abs": sorted(abs(x) for x in dc_a0)[n_eff // 2] if n_eff else float("nan"),
                      "q95_abs": sorted(abs(x) for x in dc_a0)[int(0.95 * (n_eff - 1))] if n_eff else float("nan")},
        "content_specificity": {"mean_diff_ortho_minus_orthoP": mean(d_spec), "se": se(d_spec),
                                "t_stat": (mean(d_spec) / se(d_spec)) if se(d_spec) else float("nan")},
        "mechanism": {"c_anchor_last_mean": anchor_last, "c_anchor_first_mean": anchor_first,
                      "key_orthogonality_max_abs_cos": key_max, "vrand_max_abs_cos": vrand_max,
                      "reg_decay_mean": mean(dc_a0)},
        "main": main_b, "sensitivity_q95": sens_b,
        "sensitivity_flips": main_b["outcome"] != sens_b["outcome"],
        "rows": {s: {arm: arms[arm][s] for arm in arms} for s in eff},
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== DIR-51 裁决 ===")
    print(f"  n={len(seeds)}  有效 n_eff={n_eff}  剔除率={indet_frac:.4f}  原因={reasons}")
    for k in ("random", "ortho", "orthoP", "replay"):
        print(f"  D^{k:7s} mean={mean(D[k]):+.6f} sd={sd(D[k]):.6f} se={se(D[k]):.6f}")
    print(f"  无写入对照自然漂移 |Δc^α0| mean={out['dc_alpha0']['mean']:+.6f} median_abs={out['dc_alpha0']['median_abs']:.6f}")
    print(f"  内容特异性差 (orthoP−ortho) mean={out['content_specificity']['mean_diff_ortho_minus_orthoP']:+.6f} t={out['content_specificity']['t_stat']:+.3f}")
    print(f"  机制: c_anchor_last={ {k: round(v,4) for k,v in anchor_last.items()} }  键正交max={key_max:.3e}")
    print(f"  主判据(τ={a.tau:.6f}): P1={main_b['P1']} P2={main_b['P2']} P3={main_b['P3']} P4={main_b['P4']} ⇒ **{main_b['outcome']}**")
    print(f"  敏感性(τ_q95={a.tau_q95:.6f}): ⇒ **{sens_b['outcome']}**   翻转={out['sensitivity_flips']}")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
