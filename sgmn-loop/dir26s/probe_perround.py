#!/opt/pytorch/bin/python
r"""DIR-26-T（AR(1) 后重优势机制分解）· §0 可行性探针 —— per-round μ_t 分解 + 聚合复现 D-67。

★ **不进判据**。
★ ★ **逐字复刻 D-67（`dir26s/R1_dose_ar1_v3.py::run_unit_dose`）**：写流 AR(1) 分派（DEF-26S-9）、
   τ 由 `cal_at(60)` 运行时取得、**frozen cnt0**（φ=0 的 B 臂配额，两档共用）、六臂 A/B/P/Fu/Uu/Ku。
   **唯一增量** = 对 `ret_jitter_margin` 的 `m_{i,ρ}` 按轮 t=i//(N_READ//R_PRES) 切片 ⇒ per-round μ_t。
验证：(1) 重算聚合 μ 与 D-67 落盘 μ **逐位一致**（证明切片/复刻正确、装置确定）；
      (2) per-round Δμ_t 给出非平凡轮次结构（预判可分解性）。
"""
import argparse, json, math, os, sys
import numpy as np
import torch
for _p in ("dir26s","dir26r","dir26o","dir26k","dir26j","dir26i","dir26h","dir26g",
           "dir26f","dir26e","dir26c","dir26b","dir26","dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")
import R1_gating as G
import R1_gating2 as G2
import R1_gating5 as G5
import R1_gating7 as G7
import R1_gating12 as K12
import R1_dose_ar1_v3 as V3      # ★ 逐字复刻 D-67 的 run_unit_dose / build_write / set_phi
import R1_ar1 as RA
from R1_gating import D_MODEL, M, drive, make_core, unit
from R1_gating2 import item_counts
from R1_gating5 import theta_of
from R1_gating7 import build_keep, realized_profile, PROFILE
from R1_gating12 import (N_READ, QMAX_CAP, R0, R_HO8, R_HO4, R_PRES, SEEDS, cal_at,
                         measure_hit, measure_prec, read_stream, wsig, ret_jitter_margin)

N_KV = V3.N_KV
PHIS = (0.0, 0.9)

def per_round_mu(P, Kq, vb, n_read, r_q, r_pres):
    """`ret_jitter_margin` 的 m_{i,ρ} 按轮 t=i//(n_read//r_pres) 切片 ⇒ per-round μ_t + 整体 μ（应=ret_jitter_margin 的 μ）。"""
    vbU = unit(vb)
    m_all = []; per_round = [[] for _ in range(r_pres)]
    block = n_read // r_pres
    assert block * r_pres == n_read, (n_read, r_pres)
    for rho in range(r_q):
        Y = P @ Kq[rho].T
        S = unit(Y.T) @ vbU.T
        for i in range(n_read):
            row = S[i]
            other = torch.cat([row[:i], row[i + 1:]]) if i + 1 < n_read else row[:i]
            m = float(row[i] - other.max())
            m_all.append(m); per_round[i // block].append(m)
    mu = float(np.mean(m_all))
    mu_t = [float(np.mean(x)) for x in per_round]
    # 与 ret_jitter_margin 的 μ 逐位一致（证明切片复刻正确）
    m_ref = ret_jitter_margin(P, Kq, vb, n_read, r_q)
    assert abs(mu - m_ref["mu"]) < 1e-12, (mu, m_ref["mu"])
    return mu, mu_t

def run_one(seed, phi, tau, cnt0):
    """**逐字调用 V3.run_unit_dose**（D-67 实现，frozen cnt0）拿完整六臂单元；
    对 Fu/Ku 的 W 用 per_round_mu 切片。装置完全交给 V3，避免复刻偏差。"""
    r = V3.run_unit_dose(seed, phi, tau, cnt0=cnt0)
    D = V3.build_write(seed, phi, N_KV)
    RS8 = read_stream(seed, R_HO8)
    vb = D["vb"][:N_READ]
    # 用落盘的 wsig 无法反推 W；改为： Fu/Ku 重跑用 cnt（已 frozen）按 K12 驱动 + per_round_mu
    cnt = [int(x) for x in r["counts"]]
    out = {"cnt": cnt}
    for pk in ("F", "K"):
        keep, _ = build_keep(N_KV, R_PRES, cnt, PROFILE[pk])
        c = make_core(D_MODEL, seed, R0)
        drive(c, D, "R", keep=keep)
        W = c._W().detach(); P = W - c.W0
        mu, mu_t = per_round_mu(P, RS8["K"], vb, N_READ, R_HO8, R_PRES)
        assert abs(mu - r["arms"][pk + "u"]["mu"]) < 1e-12, (pk, mu, r["arms"][pk + "u"]["mu"])
        out[pk] = {"mu": mu, "mu_t": mu_t, "wsig": wsig(W)}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default="/home/ec2-user/sgmn-loop/dir26s/out/DIR26T_probe_perround.json")
    A = ap.parse_args()
    seeds = SEEDS[:A.n]
    disk = json.load(open("/home/ec2-user/sgmn-loop/dir26s/out/DIR26S_rows.json"))["rows"]
    dmap = {(r["phi"], r["seed"]): r for r in disk}
    tau = cal_at(N_KV)
    res = {"phi": {}, "agg_repro": {}, "n": A.n, "seeds": seeds, "c_tau": float(tau["c_tau"])}
    maxdiff = 0.0
    cnt0_cache = {}
    for phi in PHIS:
        per_arm = {"F": [], "K": []}
        for sd in seeds:
            cnt0 = None if phi == 0.0 else cnt0_cache.get(sd)   # φ=0.9 用 φ=0 的 frozen cnt0
            r = run_one(sd, phi, tau, cnt0)
            if phi == 0.0:
                cnt0_cache[sd] = r["cnt"]        # 存下 φ=0 的 B 臂配额，供 φ=0.9 复用
            for pk in ("F", "K"):
                per_arm[pk].append({"mu": r[pk]["mu"], "mu_t": r[pk]["mu_t"], "seed": sd})
            row = dmap.get((phi, sd))
            if row:
                for pk in ("F", "K"):
                    maxdiff = max(maxdiff, abs(r[pk]["mu"] - row["arms"][pk + "u"]["mu"]))
        res["phi"][phi] = per_arm
    res["agg_repro"]["max_abs_mu_diff_vs_disk"] = maxdiff
    dmu_t = []
    for t in range(R_PRES):
        f0 = np.mean([x["mu_t"][t] for x in res["phi"][0.0]["F"]]); f9 = np.mean([x["mu_t"][t] for x in res["phi"][0.9]["F"]])
        k0 = np.mean([x["mu_t"][t] for x in res["phi"][0.0]["K"]]); k9 = np.mean([x["mu_t"][t] for x in res["phi"][0.9]["K"]])
        dmu_t.append({"t": t, "dF": f9 - f0, "dK": k9 - k0, "dG_t": (k9 - k0) - (f9 - f0)})
    res["dmu_t_probe"] = dmu_t
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("n =", A.n, "seeds =", seeds, "c_tau =", round(float(tau["c_tau"]), 4))
    print("★ 复现校验：max|重算μ − 落盘μ| =", maxdiff)
    for d in dmu_t:
        print(f"  t={d['t']}:  ΔF={d['dF']:+.6f}  ΔK={d['dK']:+.6f}  ΔG_t={d['dG_t']:+.6f}")
    print("WROTE", A.out)

if __name__ == "__main__":
    main()
