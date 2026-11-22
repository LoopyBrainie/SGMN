r"""DIR-26-K · probe26k_feas.py —— $n_{\rm kv}{=}60$ 的**可行性探针**（冻结前，单种子，只定位不作判据）

回答四件事，全部是"能不能跑"而不是"结论是什么"：
  ① $n_{\rm kv}{=}60$ 上 `calibrate_w` 是否收敛（skip_rate、gap、skip_frac_repeat）；
  ② 单元耗时（决定正式轮种子数预算）；
  ③ $\bar q$ / $\max q_i$ / 读集**触顶比例** $\nu_{\rm cap}$（$q_i{=}4R$ 的条目在被顶平后日程无差别 ⇒ 若 $\nu_{\rm cap}$ 随 $n_{\rm kv}$ 变大，剂量会被稀释，那是装置伪影）；
  ④ 读集几何的 $n_{\rm kv}$-**无关性**（D-42 配对相关 $\hat\rho{=}0.705$ 的结构原因 ⇒ 本轮断言 A26K-9 的基础）。

★ 探针读数**不进判据、不定阈值**（纪律 (i)）。
"""
import sys, time
import numpy as np
import torch

for _p in ("dir26k", "dir26j", "dir26i", "dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating11 as G11
from R1_gating11 import cal_at, run_unit, DEV_MD5, SEEDS
from R1_gating import D_MODEL, N_DIST, build_stream, make_core, unit
from R1_gating5 import theta_of
from R1_gating7 import md5

PROBE_SEED = SEEDS[0]
CELLS = ({"n_kv": 20, "n_read": 20}, {"n_kv": 35, "n_read": 20}, {"n_kv": 60, "n_read": 20})


def main():
    print("=== ① 装置与标定（逐 n_kv）===")
    print(f"  装置 md5 OK ? {md5('/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py') == DEV_MD5}")
    taus = {}
    for cell in CELLS:
        n = cell["n_kv"]
        t0 = time.time()
        t = cal_at(n)
        taus[n] = t["c_tau"]
        print(f"  n_kv={n:>2}  c_tau={t['c_tau']:.4f}  skip={t['skip_rate']:.3f}  gap={t['gap']:+.3f}  "
              f"rep={t['skip_frac_repeat']:.3f}  gate_ok={t['gate_ok']}  ({time.time()-t0:.1f}s)")
    print(f"  标定后复位 R1_gating5.N_DIST == {N_DIST} ? {G11.G5.N_DIST == N_DIST}")

    print("\n=== ② 单元耗时 + ③ 计数分布 ===")
    for cell in CELLS:
        n = cell["n_kv"]
        t0 = time.time()
        g = run_unit(cell, PROBE_SEED, taus[n])
        dt = time.time() - t0
        cnt = np.array(g["counts"])
        rd = cnt[:cell["n_read"]]
        nu = float(np.mean(rd >= 4 * G11.R_PRES))
        a = g["arms"]
        print(f"  n_kv={n:>2}  {dt:6.1f}s  k_B={g['k']['B']:>4}  q̄={g['qbar']:.3f}  q_max={g['qmax']:>2}  "
              f"ν_cap={nu:.3f}  |  D_abs={g['D_abs']:+.5f}  U0={g['U0']:+.5f}  R_mu={g['R_mu']:+.4f}")
        print(f"           μ: A {a['A']['mu']:+.4f} B {a['B']['mu']:+.4f} P {a['P']['mu']:+.4f} "
              f"F {a['Fu']['mu']:+.4f} U0 {a['Uu']['mu']:+.4f} K {a['Ku']['mu']:+.4f}")
        print(f"           ho8: A {a['A']['ho8_mean']:.3f} B {a['B']['ho8_mean']:.3f} "
              f"F {a['Fu']['ho8_mean']:.3f} U0 {a['Uu']['ho8_mean']:.3f} K {a['Ku']['ho8_mean']:.3f}")
        print(f"           写集计数 min/median/max = {cnt.min()}/{int(np.median(cnt))}/{cnt.max()}；"
              f"读集(前 {cell['n_read']}) 触顶 {int((rd >= 4*G11.R_PRES).sum())} 条；镜像 {g['mirror_ok']}")

    print("\n=== ④ 读集几何的 n_kv-无关性（断言 A26K-9 的基础）===")
    th = theta_of(0.9)
    base = build_stream(PROBE_SEED, D_MODEL, 20, G11.R_PRES, th)
    for n in (35, 60):
        s = build_stream(PROBE_SEED, D_MODEL, n, G11.R_PRES, th)
        eq = {
            "K": torch.equal(s["K"][:, :20], base["K"]),
            "proto": torch.equal(s["proto"][:20], base["proto"]),
            "va": torch.equal(s["va"][:20], base["va"]),
            "vb": torch.equal(s["vb"][:20], base["vb"]),
        }
        print(f"  n_kv={n:>2} 前 20 条目的键/原型/值与前 20 条流逐位相同：{eq}")

    print("\n=== 预测量（供冻结时的功效核算，非判据）===")
    print("  已测 D-42：G20=0.00637、G35=0.00799 ⇒ Δ1=+0.00162（3se 0.00141, n=10, 配对 ρ̂=0.71）")
    for g_ in (1 / 3, 0.4, 0.5, 1.0):
        d60 = 0.00637 * 3 ** g_
        print(f"  若幂律 γ={g_:.2f} ⇒ G60={d60:.5f}, Δ2={d60-0.00799:+.5f}")


if __name__ == "__main__":
    main()
