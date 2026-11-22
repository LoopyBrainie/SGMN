"""DIR-26-H · probe26h_load.py —— 载荷 $N$（条目数）上的剖面效应电平（卡片冻结前的**探针**）。

为什么是 $N$：r 轴已被探到天花板（`P26H_levels.log`：$r{=}32$ **六臂全 1.000**）⇒ "加大支撑秩看效应是否消失"无法分离机制。
  而 $N$ 只改**共享支撑里的条目数**（干扰），不改每条目的写入结构：$\\rho{=}0.4$ 的标定使
  $k_B=0.6T=0.6\\cdot R\\cdot2M\\cdot N$ ⇒ 每条目 $\\bar q=9.6$ **与 $N$ 无关**。
  ★ 装置约束：原型键 $E=\\mathrm{diag}$ 在 $d{=}64$ ⇒ $N\\le64$。
"""
import sys

import numpy as np

for _p in ("dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

from R1_gating import D_MODEL, N_DIST, M, build_stream, drive, make_core, metrics_plas
from R1_gating2 import keep_positions, item_counts
from R1_gating3 import ret_jitter
import R1_gating5 as G5
from R1_gating5 import calibrate_w, theta_of
from R1_gating6 import _pos_idxs                     # noqa: F401
import R1_gating7 as G7

R_PRES, R_HO4, R_HO8, HO_OFFSET, R_MAIN, R0 = 4, 4, 8, 999, 0.40, 8
SEED = 20261028


def main():
    print(f"seed={SEED}  r={R0}  R={R_PRES}  M={M}")
    print(f"{'N':>3} {'c_tau':>7} {'k_B':>5} {'q_mean':>7} | " +
          " | ".join(f"{t:>13}" for t in ("B", "P", "Fu", "Uu", "Ku")) + " | dFK(ho4) dFK(ho8)  prof(F)  maxho8")
    for n in (10, 20, 35, 60):
        RS = {"ho4": build_stream(SEED + HO_OFFSET, D_MODEL, n, R_HO4, theta_of(0.9)),
              "ho8": build_stream(SEED + HO_OFFSET, D_MODEL, n, R_HO8, theta_of(0.9))}
        D = build_stream(SEED, D_MODEL, n, R_PRES, theta_of(0.9))
        G5.N_DIST = n        # ★ 模块级重绑定（与 DIR-25/26 重绑 LAM 同一做法）：标定必须在**同一个 n** 上取分位数
        tau = calibrate_w(R0, R_MAIN, R_PRES, theta_of(0.9))["c_tau"]
        G5.N_DIST = N_DIST   # 复位
        out = {}

        def rec(tag, core):
            W = core._W().detach()
            P = W - core.W0
            m4 = ret_jitter(P, RS["ho4"]["K"], D["vb"], n, R_HO4)
            m8 = ret_jitter(P, RS["ho8"]["K"], D["vb"], n, R_HO8)
            out[tag] = (m4["ret1_mean"], m8["ret1_mean"], metrics_plas(W, core.W0, D["proto"], D["vb"], n)["ret1"])

        c = make_core(D_MODEL, SEED, R0); kB, logB = drive(c, D, "B", tau=tau); rec("B", c)
        cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, n)
        c = make_core(D_MODEL, SEED, R0)
        drive(c, D, "R", keep=keep_positions(n, R_PRES, kB)); rec("P", c)
        prof = None
        for pk in ("F", "U", "K"):
            keep, _ = G7.build_keep(n, R_PRES, cnt, G7.PROFILE[pk])
            c = make_core(D_MODEL, SEED, R0)
            kk, _ = drive(c, D, "R", keep=keep); rec(f"{pk}u", c)
            assert kk == kB, (n, pk, kk, kB)
            if pk == "F":
                prof = G7.realized_profile(keep, n, R_PRES)
        d4 = out["Fu"][0] - out["Ku"][0]
        d8 = out["Fu"][1] - out["Ku"][1]
        cells = " | ".join(f"{out[t][0]:.4f}/{out[t][1]:.4f}" for t in ("B", "P", "Fu", "Uu", "Ku"))
        mx = max(out[t][1] for t in ("B", "P", "Fu", "Uu", "Ku"))
        print(f"{n:>3} {tau:>7.4f} {kB:>5} {kB / n:>7.2f} | {cells} | {d4:+.4f} {d8:+.4f}  "
              f"{[round(x, 2) for x in prof]}  {mx:.3f}")


if __name__ == "__main__":
    main()
