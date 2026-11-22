#!/opt/pytorch/bin/python
r"""DIR-52 · **可行性/功效探针**（`probe52a.py`）—— AR(1) 后重优势的**逐轮分解**是否在本装置可分辨。

★ **不进判据**。本探针只回答两件事：
  ① `μ` 的**逐轮（按轮截断）**分解是否非平凡、且可复现 D-67 的总 `ΔG`；
  ② 逐轮增量 `ΔδG_ρ` 的**跨种子 sd** 有多大 ⇒ 每个 ρ 的效应能否在**任何**预算内被 3se 分辨（功效）。

★ **装置逐字复用**（零改动）：`dir26s/R1_dose_ar1_v3.py` 的 `build_write` / `run_unit_dose`（拿 frozen cnt0）、
  `dir26g/R1_gating7.build_keep` + `PROFILE`、`dir26/R1_gating.drive`、`dir26i/R1_gating10.ret_jitter_margin`。
  **唯一增量** = 把 `keep` **按轮截断**到 `ρ`，再 `drive` 一次并读数 ⇒ `μ(ρ)`。

★★ **为什么不用 DIR-26-T 探针的 `i//5` 切片**：`ret_jitter_margin` 的 `i` 轴是**条目索引**（`N_READ=20`），
  而写流的顺序轴是 `stream_idx = ((ρ·n + i)·2 + mbr)·M + m`（**轮优先**）⇒ `i//5` 切的是**条目**、不是**轮**。
  ⇒ 逐轮语义只能靠**截断 `keep` 到轮边界**得到。本文件用后者。

用法：`probe52a.py --n 8 --seed0 61000 --out dir52/out/DIR52_probe_a.json`
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

for _p in ("dir26s", "dir26r", "dir26o", "dir26k", "dir26j", "dir26i", "dir26h", "dir26g",
           "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G                      # noqa: E402
import R1_gating2 as G2                    # noqa: E402
import R1_gating7 as G7                    # noqa: E402
import R1_gating12 as K12                  # noqa: E402
import R1_dose_ar1_v3 as V3                # noqa: E402
from R1_gating import D_MODEL, M, drive, make_core                  # noqa: E402
from R1_gating7 import build_keep, PROFILE, realized_profile        # noqa: E402
from R1_gating12 import (N_READ, R_HO8, R_PRES, SEEDS, cal_at, read_stream,
                         ret_jitter_margin, wsig)                   # noqa: E402

N_KV = V3.N_KV
PHIS = (0.0, 0.9)
ROUND_END = [((rho + 1) * N_KV) * 2 * M for rho in range(R_PRES)]   # 每轮末的 stream 上界


def trunc_keep(keep, ub):
    r"""把 `keep` 截到 `stream_idx < ub` ⇒ 「只呈现前 ρ 轮」。"""
    return {t for t in keep if t < ub}


def mu_of(seed, D, keep, vb, RS8):
    c = make_core(D_MODEL, seed, K12.R0)
    drive(c, D, "R", keep=keep)
    W = c._W().detach()
    P = W - c.W0
    return ret_jitter_margin(P, RS8["K"], vb, N_READ, R_HO8), W


def run_seed(seed, phi, tau, cnt0):
    """返回 {profile: [μ(0), μ(1), ..., μ(4)]}。μ(0) = 未写（W0），μ(ρ) = 前 ρ 轮写完后。"""
    D = V3.build_write(seed, phi, N_KV)
    RS8 = read_stream(seed, R_HO8)
    vb = D["vb"][:N_READ]
    out = {"seed": seed, "phi": float(phi), "prof": {}}

    # cnt0：φ=0 由 B 臂给出；φ=0.9 用传入的 frozen cnt0
    if cnt0 is None:
        full = V3.run_unit_dose(seed, phi, tau, cnt0=None)
        cnt = [int(x) for x in full["counts"]]
        src = full["cnt_source"]
    else:
        cnt = [int(x) for x in cnt0]
        src = "frozen_cnt0"
    out["cnt_source"] = src
    out["cnt"] = cnt

    # μ(0)：基座未写
    c0 = make_core(D_MODEL, seed, K12.R0)
    m0 = ret_jitter_margin(c0._W().detach() - c0.W0, RS8["K"], vb, N_READ, R_HO8)
    out["mu0"] = m0["mu"]

    for pk in ("F", "K"):
        keep, per_item = build_keep(N_KV, R_PRES, cnt, PROFILE[pk])
        assert len(keep) == sum(cnt), (len(keep), sum(cnt))
        mus = [m0["mu"]]
        for rho, ub in enumerate(ROUND_END, start=1):
            kt = trunc_keep(keep, ub)
            m, W = mu_of(seed, D, kt, vb, RS8)
            mus.append(m["mu"])
            if rho == R_PRES:
                out.setdefault("final", {})[pk] = {"mu": m["mu"], "wsig": wsig(W), "n_keep": len(kt)}
        out["prof"][pk] = {"mu": mus, "per_item": per_item, "realized": realized_profile(keep, N_KV, R_PRES)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=61000)
    ap.add_argument("--out", default="/home/ec2-user/sgmn-loop/dir52/out/DIR52_probe_a.json")
    ap.add_argument("--rows", default="/home/ec2-user/sgmn-loop/dir52/out/DIR52_probe_a_rows.jsonl",
                    help="逐 seed 增量落盘（★ DEF-52-2：算完即弃是 D-67 纪律 108 明令禁止的）")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    tau = cal_at(N_KV)
    print("c_tau =", float(tau["c_tau"]), flush=True)

    seeds = [a.seed0 + i for i in range(a.n)]
    rows, cnt0_cache = [], {}
    done = set()
    if a.resume and os.path.exists(a.rows):
        for line in open(a.rows, encoding="utf-8"):
            line = line.strip()
            if line:
                rec = json.loads(line)
                rows.append(rec); done.add(rec["seed"])
                cnt0_cache[rec["seed"]] = rec["phi0"]["cnt"]
        print(f"[resume] 已有 {len(done)} 个种子：{sorted(done)}", flush=True)
    for s in seeds:
        if s in done:
            continue
        rec = {"seed": s}
        r0 = run_seed(s, 0.0, tau, None)
        cnt0_cache[s] = r0["cnt"]
        r9 = run_seed(s, 0.9, tau, cnt0_cache[s])
        rec["phi0"], rec["phi9"] = r0, r9
        rows.append(rec)
        with open(a.rows, "a", encoding="utf-8") as f:      # ★ 增量落盘
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  seed {s} done  {time.time()-t0:.1f}s", flush=True)

    # ── 逐轮增量 ΔδG_ρ = δG_ρ(0.9) − δG_ρ(0)，δG_ρ = δμ_K,ρ − δμ_F,ρ
    per = []
    for rec in rows:
        d = {"seed": rec["seed"], "dG": [], "dF": [], "dK": []}
        def inc(r, pk, rho_=None):
            # ★ `run_seed` 返回的是容器 ⇒ 必须走 ["prof"][pk]（DEF-52-2b：初版漏了这一层）
            mu = r["prof"][pk]["mu"]
            return mu[rho] - mu[rho - 1]
        for rho in range(1, R_PRES + 1):
            f0, f9 = inc(rec["phi0"], "F", rho), inc(rec["phi9"], "F", rho)
            k0, k9 = inc(rec["phi0"], "K", rho), inc(rec["phi9"], "K", rho)
            d["dF"].append(f9 - f0)
            d["dK"].append(k9 - k0)
            d["dG"].append((k9 - k0) - (f9 - f0))
        d["G0"] = rec["phi0"]["final"]["K"]["mu"] - rec["phi0"]["final"]["F"]["mu"]
        d["G9"] = rec["phi9"]["final"]["K"]["mu"] - rec["phi9"]["final"]["F"]["mu"]
        d["mu_curve"] = {ph: {pk: rec[ph]["prof"][pk]["mu"] for pk in ("F", "K")}
                         for ph in ("phi0", "phi9")}
        d["dG_total"] = d["G9"] - d["G0"]
        d["sum_dG_rounds"] = float(np.sum(d["dG"]))
        per.append(d)

    def col(key, i=None):
        return [x[key][i] if i is not None else x[key] for x in per]

    n = len(per)
    summary = {"n": n, "seeds": seeds, "c_tau": float(tau["c_tau"])}
    summary["dG_total_mean"] = float(np.mean(col("dG_total")))
    summary["dG_total_sd"] = float(np.std(col("dG_total"), ddof=1))
    summary["dG_total_3se"] = 3 * summary["dG_total_sd"] / math.sqrt(n)
    summary["identity_max_abs_err"] = float(max(abs(x["sum_dG_rounds"] - x["dG_total"]) for x in per))
    summary["rounds"] = []
    for i in range(R_PRES):
        v = col("dG", i)
        summary["rounds"].append({
            "rho": i + 1, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
            "se": float(np.std(v, ddof=1) / math.sqrt(n)),
            "n_pos": int(sum(1 for x in v if x > 0)),
            "vals": v,
        })
    # 集中度（均匀基准 1/R_PRES）
    absv = [abs(r["mean"]) for r in summary["rounds"]]
    tot = sum(absv)
    summary["concentration_max_over_sum"] = float(max(absv) / tot) if tot > 0 else None
    summary["uniform_reference"] = 1.0 / R_PRES
    # 功效：把每轮效应当 1/4 的目标时需要的 n
    need = []
    for r in summary["rounds"]:
        if r["sd"] > 0 and r["mean"] != 0:
            need.append({"rho": r["rho"], "ratio_mean_over_3se": abs(r["mean"]) / (3 * r["se"]) if r["se"] > 0 else None,
                         "n_for_3se": int(math.ceil((3 * r["sd"] / abs(r["mean"])) ** 2)) if abs(r["mean"]) > 0 else None})
    summary["power_per_round"] = need
    verdict_hint = ("RESOLVABLE" if any((x.get("ratio_mean_over_3se") or 0) >= 1.0 for x in need)
                    else "BELOW-RESOLUTION")
    summary["probe_hint"] = verdict_hint

    out = {"summary": summary, "rows": rows, "per_seed": per, "wall_s": time.time() - t0}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== DIR-52 可行性/功效探针 ===")
    print(f"  n={n}  ΔG_total mean={summary['dG_total_mean']:+.6f} sd={summary['dG_total_sd']:.6f} "
          f"3se={summary['dG_total_3se']:.6f}   (D-67 参考 +0.000861, 3se 0.000810)")
    print(f"  恒等式 max|Σ_ρ ΔδG_ρ − ΔG_total| = {summary['identity_max_abs_err']:.3e}")
    for r in summary["rounds"]:
        print(f"  ρ={r['rho']}: ΔδG={r['mean']:+.6f}  sd={r['sd']:.6f}  3se={3*r['se']:.6f}  "
              f"正号 {r['n_pos']}/{n}  效应/3se={abs(r['mean'])/(3*r['se']) if r['se']>0 else float('nan'):.2f}")
    print(f"  集中度 max/Σ|·| = {summary['concentration_max_over_sum']}  (均匀基准 {summary['uniform_reference']})")
    print(f"  每轮所需 n（3se 分辨）：{[(x['rho'], x['n_for_3se']) for x in need]}")
    print(f"  ★ 探针提示：{verdict_hint}   墙钟 {out['wall_s']:.1f}s")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
