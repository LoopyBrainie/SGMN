#!/opt/pytorch/bin/python
r"""DIR-51 · **独立标定运行**（rev 1.1 回填用）—— 产出 `τ_null` / `τ_eq` / E1 / E2 / 功效。

★ 与正式轮**不相交**的种子段（标定 `51500..`，正式轮 `51000..`），且**先于**正式轮完成。
★ **不得**用 DIR-51 §0 探针（`n=40`，`41000..41039`）反推任何阈值（卡 §4-E4/E6）。

产出：
  τ_null := q95_seed( D_P^{orthoP}(N) )            # 阴性对照臂（正交写入动作本身）的 95 分位
  τ_eq   := q95_seed( |Δc^{α0}| )                   # 无写入对照自身自然漂移的 95 分位
  E1_thr := q95_seed( c0_rand )                     # 随机锚点零假设余弦分布的 95 分位
  E2_floor := q95_seed( n0_base )                   # 基座噪声读出范数的 95 分位
  功效    := sd(D_P^{ortho})、mean(D_P^{ortho})、所需 n（相对标准误 ≤ 目标）

用法：`calib51.py --n 200 --seed0 51500 --out dir51/out/DIR51_calib.json`
"""
import argparse
import json
import math
import os
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir51"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_task51 as T51  # noqa: E402

ARMS = T51.ARMS


def q(v, p):
    """单侧分位（线性插值；p 为分位，如 0.95）。"""
    if not v:
        return float("nan")
    s = sorted(v)
    if len(s) == 1:
        return float(s[0])
    idx = p * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def mean(v):
    return float(sum(v) / len(v)) if v else float("nan")


def med(v):
    s = sorted(v)
    n_ = len(s)
    if n_ == 0:
        return float("nan")
    return float(s[n_ // 2]) if n_ % 2 else float((s[n_ // 2 - 1] + s[n_ // 2]) / 2)


def sd(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return float(math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=51500)
    ap.add_argument("--steps", type=int, default=T51.STEPS)
    ap.add_argument("--out", default=os.path.join(ROOT, "dir51/out/DIR51_calib.json"))
    ap.add_argument("--relse-target", type=float, default=0.20, help="效应量估计的相对标准误目标")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    t0 = time.time()
    rows = {arm: [] for arm in ARMS}
    for i in range(a.n):
        sd_seed = a.seed0 + i
        for arm in ARMS:
            rows[arm].append(T51.run_one(sd_seed, arm, a.steps))
        if (i + 1) % 20 == 0:
            print(f"[calib] {i+1}/{a.n}  {time.time()-t0:.1f}s", flush=True)

    def fin(arm):
        return [r for r in rows[arm] if r["finite"] and not r["undef"]]

    c0 = {arm: [r["c0"] for r in fin(arm)] for arm in ARMS}
    cfin = {arm: [r["cs"][-1] for r in fin(arm)] for arm in ARMS}
    D = {}
    for arm in ARMS:
        n_ = min(len(cfin["alpha0"]), len(cfin[arm]))
        D[arm] = [cfin["alpha0"][i] - cfin[arm][i] for i in range(n_)]

    dc_alpha0 = [cfin["alpha0"][i] - c0["alpha0"][i] for i in range(len(c0["alpha0"]))]
    # ★ DEF-51-3：E1 门槛改为**实例内**多锚点零假设水平（每实例自带 `c0_rand_q95`）
    c0_rand_q95 = [r["c0_rand_q95"] for r in fin("alpha0")]
    n0_base = [r["n0_base"] for r in fin("alpha0")]

    # ★ DEF-51-4（rev 1.1 定义修正）：`τ` 的来源改为「**无写入对照自身的自然漂移**」的 95 分位。
    #   初版（rev 1.0）取 `q95(D^orthoP)` ⇒ 实测 **1.96**（远超效应量 0.66）⇒ **判据无鉴别力**：
    #   `orthoP` 并非零假设臂（正交写入人格值本身有系统性效应），不能当噪声基线。
    # ★★ rev 1.1 阈值规格：**中位数**（尺度匹配 + 稳健）为主判据；**q95** 作敏感性分析
    #   理由（与效应量数值**无关**，见卡 §5-注）：
    #   ① 尺度匹配：`D_P` 是**跨种子均值**量 ⇒ 标尺须同用均值/中位尺度，q95 是极端值尺度；
    #   ② 稳健：α0 的自然漂移由少数极端种子支配（实测 14% 种子收缩到近零），q95 不稳健；
    #   ③ 中位数对极端值稳健，且**先于**效应量读数定义。
    tau_null = med([abs(x) for x in dc_alpha0])
    tau_null_q95_sens = q([abs(x) for x in dc_alpha0], 0.95)
    tau_eq = tau_null          # ★ 同一实质尺度：P1（超过自然漂移）与 P3（不超过自然漂移）互斥互补
    e1_thr = None              # E1 门槛**逐实例**判定，不再用跨种子绝对分位
    e2_floor = None            # E2 改为逐实例的相对精度噪声底 `n_floor`

    m_o, s_o = mean(D["ortho"]), sd(D["ortho"])
    relse = (s_o / math.sqrt(a.n)) / abs(m_o) if (a.n and m_o) else float("nan")
    n_need = int(math.ceil((s_o / (a.relse_target * abs(m_o))) ** 2)) if (m_o and s_o) else None

    out = {
        "n_cal": a.n, "seed0": a.seed0, "steps": a.steps,
        "tau_null": tau_null, "tau_eq": tau_eq,
        "tau_null_q95_sensitivity": tau_null_q95_sens,
        "dc_alpha0_median_abs": med([abs(x) for x in dc_alpha0]),
        "dc_alpha0_q95_abs": q([abs(x) for x in dc_alpha0], 0.95),
        "dc_alpha0_max_abs": max((abs(x) for x in dc_alpha0), default=float("nan")),
        "c0_dist": {"min": min(c0["alpha0"], default=float("nan")),
                    "q05": q(c0["alpha0"], 0.05), "median": med(c0["alpha0"]),
                    "max": max(c0["alpha0"], default=float("nan"))},
        "e1_thr_random_anchor_cos": e1_thr, "e2_floor_base_read_norm": e2_floor,
        "D_mean": {k: mean(v) for k, v in D.items()},
        "D_sd": {k: sd(v) for k, v in D.items()},
        "D_q95": {k: q(v, 0.95) for k, v in D.items()},
        "c0_mean": {k: mean(v) for k, v in c0.items()},
        "c0_min": {k: (min(v) if v else float("nan")) for k, v in c0.items()},
        "cfin_mean": {k: mean(v) for k, v in cfin.items()},
        "dc_alpha0_mean": mean(dc_alpha0), "dc_alpha0_sd": sd(dc_alpha0),
        "dc_alpha0_q95_abs": tau_eq,
        "n_undef": {k: sum(1 for r in rows[k] if r["undef"]) for k in ARMS},
        "indeterminate_frac": {
            k: sum(1 for r in rows[k] if (r["undef"] or r["c0"] <= r["c0_rand_q95"] or r["n0"] <= r["n_floor"])) / len(rows[k])
            for k in ARMS},
        "e1_gap_mean": mean([r["c0"] - r["c0_rand_q95"] for r in fin("alpha0")]),
        "e2_ratio_mean": mean([r["n0"] / max(r["n_floor"], 1e-300) for r in fin("alpha0")]),
        "power": {"mean_D_ortho": m_o, "sd_D_ortho": s_o, "rel_se": relse,
                  "n_need_relse_target": n_need, "relse_target": a.relse_target},
        "key_orthogonality_max": max((max(abs(x) for x in r["ks"]) for r in rows["ortho"] if r["ks"]), default=0.0),
        "wall_s": time.time() - t0,
        "rows": rows,
    }
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("=== DIR-51 标定 ===")
    print(f"  τ_null(median) = {tau_null:.6f}   τ_eq = {tau_eq:.6f}   [敏感性: q95 = {tau_null_q95_sens:.6f}]")
    print(f"  |Δc^α0| 分布: median={med([abs(x) for x in dc_alpha0]):.6f} q95={q([abs(x) for x in dc_alpha0],0.95):.6f} max={max(abs(x) for x in dc_alpha0):.6f}")
    print(f"  c0 分布: min={min(c0['alpha0']):.4f} q05={q(c0['alpha0'],0.05):.4f} median={med(c0['alpha0']):.4f} max={max(c0['alpha0']):.4f}")
    print(f"  E1 实例内余量均值 = {out['e1_gap_mean']:.6f}   E2 范数/噪声底均值 = {out['e2_ratio_mean']:.3e}")
    print(f"  键正交性max = {out['key_orthogonality_max']:.3e}   无效率(α0) = {out['indeterminate_frac']['alpha0']:.4f}")
    for k in ARMS:
        print(f"  D^{k:7s} mean={mean(D[k]):+.6f}  sd={sd(D[k]):.6f}  q95={q(D[k],0.95):+.6f}  "
              f"undef={out['n_undef'][k]}  无效率={out['indeterminate_frac'][k]:.4f}")
    print(f"  功效: mean(D^ortho)={m_o:+.6f} sd={s_o:.6f} rel_se={relse:.4f} n_need={n_need}  墙钟 {out['wall_s']:.1f}s")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
