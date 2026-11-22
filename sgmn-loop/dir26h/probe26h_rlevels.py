"""DIR-26-H · probe26h_rlevels.py —— 支撑秩 r 上的剖面效应电平（卡片冻结前的**探针**，不是判据读数）。

只回答一个问题：r ∈ {8,16,32,64} 上，$Fu/Uu/Ku$ 与 $B/P$ 的 `ho4`/`ho8` 电平各是多少？
  ⇒ 目的：给卡片 §3 的判据阈值与"天花板"条款定位；并检查 r=32/64 是否已被天花板压缩。
★ 构造逐字复用 `dir26g/R1_gating7.py`（`round_quota`/`build_keep`/`realized_profile`/`even_counts`），
  装置 `dir22/sgmn_module_hebb_v7.py` 零改动。
"""
import sys

import numpy as np
import torch

for _p in ("dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
from R1_gating import (D_MODEL, N_DIST, M, build_stream, drive, make_core, metrics_plas)
from R1_gating2 import keep_positions, item_counts
from R1_gating3 import ret_jitter
from R1_gating5 import calibrate_w, theta_of
from R1_gating6 import _pos_idxs                     # noqa: F401  (build_keep 依赖)
import R1_gating7 as G7

R_PRES, R_HO4, R_HO8, HO_OFFSET, R_MAIN = 4, 4, 8, 999, 0.40
SEED = 20261028


def main():
    RS = {"ho4": build_stream(SEED + HO_OFFSET, D_MODEL, N_DIST, R_HO4, theta_of(0.9)),
          "ho8": build_stream(SEED + HO_OFFSET, D_MODEL, N_DIST, R_HO8, theta_of(0.9))}
    D = build_stream(SEED, D_MODEL, N_DIST, R_PRES, theta_of(0.9))
    print(f"seed={SEED}  n={N_DIST}  T={R_PRES * N_DIST * 2 * M}")
    print(f"{'r':>3} {'c_tau':>8} {'k_B':>5} | " + " | ".join(f"{t:>13}" for t in ("A", "B", "P", "Fu", "Uu", "Ku"))
          + " | dFK(ho4) dFK(ho8)  prof(F)")

    for r0 in (8, 16, 32, 64):
        tau = calibrate_w(r0, R_MAIN, R_PRES, theta_of(0.9))["c_tau"]
        out = {}

        def rec(tag, core):
            W = core._W().detach()
            P = W - core.W0
            m4 = ret_jitter(P, RS["ho4"]["K"], D["vb"], N_DIST, R_HO4)
            m8 = ret_jitter(P, RS["ho8"]["K"], D["vb"], N_DIST, R_HO8)
            out[tag] = (m4["ret1_mean"], m8["ret1_mean"], metrics_plas(W, core.W0, D["proto"], D["vb"], N_DIST)["ret1"])

        c = make_core(D_MODEL, SEED, r0); kA, _ = drive(c, D, "A"); rec("A", c)
        c = make_core(D_MODEL, SEED, r0); kB, logB = drive(c, D, "B", tau=tau); rec("B", c)
        cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, N_DIST)
        c = make_core(D_MODEL, SEED, r0)
        kP, _ = drive(c, D, "R", keep=keep_positions(N_DIST, R_PRES, kB)); rec("P", c)
        prof = None
        for pk in ("F", "U", "K"):
            keep, _ = G7.build_keep(N_DIST, R_PRES, cnt, G7.PROFILE[pk])
            c = make_core(D_MODEL, SEED, r0)
            kk, _ = drive(c, D, "R", keep=keep); rec(f"{pk}u", c)
            assert kk == kB, (r0, pk, kk, kB)
            if pk == "F":
                prof = G7.realized_profile(keep, N_DIST, R_PRES)
        d4 = out["Fu"][0] - out["Ku"][0]
        d8 = out["Fu"][1] - out["Ku"][1]
        cells = " | ".join(f"{out[t][0]:.4f}/{out[t][1]:.4f}" for t in ("A", "B", "P", "Fu", "Uu", "Ku"))
        print(f"{r0:>3} {tau:>8.4f} {kB:>5} | {cells} | {d4:+.4f} {d8:+.4f}  {[round(x, 2) for x in prof]}")
        print(f"      clean: " + " ".join(f"{t} {out[t][2]:.3f}" for t in ("A", "B", "P", "Fu", "Uu", "Ku")))


if __name__ == "__main__":
    main()
