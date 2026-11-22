"""DIR-26-I · probe26i_rlevels.py —— **支撑秩向下**（$r\in\{3,4,6,8,12\}$）× 载荷 $N\in\{10,35\}$ 的电平与串扰比

卡片冻结前的**探针**（不是判据读数）。只回答三件事：
  ① 每个 $(r,N)$ 格点上六臂的 `ho8` 电平 —— 判断是否有**地板**（近随机）或**天花板**（近 1.000）；
  ② $\bar q$、$c_\tau$、$k_B$ 是否仍可标定（$\rho{=}0.4$ 的 skip_rate 与 gap）；
  ③ **串扰比** $\chi$（测量层新量，见下）—— 用于把"干扰强度"从口号变成一个可测的数。

$\chi$：对 $K$ 臂的 $W$，用 `ho8` 的抖动键
  $$s=\overline{\cos(Wk_{i,\rho},\,v^{a}_i)},\qquad x=\overline{|\cos(Wk_{i,\rho},\,v_j)|}_{j\ne i},\qquad \chi=x/s$$
  $\chi$ 只由 $W$ 与流决定，**不含任何自由参数**；它把"条目间干扰有多强"变成逐格点的读数。
★ 为什么向**下**扫 $r$：$r{=}32/64$ 上六臂全 1.0000（`P26H_levels.log`）⇒ 向上只会撞天花板；
  缩小共享支撑的维数才是**加强**条目间干扰的方向（$N$ 个条目挤在 $r$ 维里）。
★ 构造/测量/定标/抖动/轮内位置规则全部 import 同一对象（`dir26g/R1_gating7.py`）；装置零改动。
"""
import sys

import numpy as np
import torch

for _p in ("dir26i", "dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

from R1_gating import D_MODEL, N_DIST, M, build_stream, drive, make_core, metrics_plas
from R1_gating2 import keep_positions, item_counts
from R1_gating3 import ret_jitter
import R1_gating5 as G5
from R1_gating5 import calibrate_w, theta_of
from R1_gating6 import _pos_idxs                     # noqa: F401
import R1_gating7 as G7

R_PRES, R_HO4, R_HO8, HO_OFFSET, R_MAIN = 4, 4, 8, 999, 0.40
SEED = 20261028
R_GRID = (3, 4, 6, 8, 12)
N_GRID = (10, 35)


def crosstalk(W, K, va, vb, n, r_pres):
    r"""$\chi=x/s$：$\cos(Wk_i,v_i)$ 的均值（信号）与 $|\cos(Wk_i,v_j)|_{j\ne i}$ 的均值（串扰）之比。"""
    s, x = [], []
    for rho in range(r_pres):
        Y = (W @ K[rho].T).T                       # (n, d_v)：每个键的读出
        Yn = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-30)
        for v in (va, vb):
            C = Yn @ v.T                           # (n, n)：C[i,j] = cos(W k_i, v_j)
            s.append(torch.diagonal(C).mean().item())
            off = C.clone()
            off.fill_diagonal_(0.0)
            x.append(off.abs().sum().item() / (n * (n - 1)))
    return float(np.mean(x) / np.mean(s)), float(np.mean(s)), float(np.mean(x))


def main():
    print(f"seed={SEED}  R={R_PRES}  M={M}  theta=0.9  rho={R_MAIN}")
    print(f"{'N':>3} {'r':>3} {'c_tau':>7} {'k_B':>5} {'q̄':>6} | " +
          " | ".join(f"{t:>9}" for t in ("A", "B", "P", "Fu", "Uu", "Ku")) +
          " | dFK(ho8)  dFK(ho4)   chi   sig   prof(F)")
    for n in N_GRID:
        RS = {"ho4": build_stream(SEED + HO_OFFSET, D_MODEL, n, R_HO4, theta_of(0.9)),
              "ho8": build_stream(SEED + HO_OFFSET, D_MODEL, n, R_HO8, theta_of(0.9))}
        D = build_stream(SEED, D_MODEL, n, R_PRES, theta_of(0.9))
        assert int(D["n"]) == n and int(RS["ho8"]["n"]) == n, "跨载荷流混用（DEF-26H-2）"
        for r0 in R_GRID:
            G5.N_DIST = n            # ★ 标定必须在同一个 n 上取分位数
            try:
                cal = calibrate_w(r0, R_MAIN, R_PRES, theta_of(0.9))
            finally:
                G5.N_DIST = N_DIST   # 复位（A26H-3）
            tau = cal["c_tau"]
            out = {}

            def rec(tag, core, kB_arm):
                W = core._W().detach()
                P = W - core.W0
                m4 = ret_jitter(P, RS["ho4"]["K"], D["vb"], n, R_HO4)
                m8 = ret_jitter(P, RS["ho8"]["K"], D["vb"], n, R_HO8)
                out[tag] = (m4["ret1_mean"], m8["ret1_mean"],
                            metrics_plas(W, core.W0, D["proto"], D["vb"], n)["ret1"])

            c = make_core(D_MODEL, SEED, r0); kA, _ = drive(c, D, "A"); rec("A", c, None)
            c = make_core(D_MODEL, SEED, r0); kB, logB = drive(c, D, "B", tau=tau); rec("B", c, None)
            W_B = c._W().detach()
            cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, n)
            c = make_core(D_MODEL, SEED, r0)
            drive(c, D, "R", keep=keep_positions(n, R_PRES, kB)); rec("P", c, None)
            prof, chi, sig = None, float("nan"), float("nan")
            for pk in ("F", "U", "K"):
                keep, _ = G7.build_keep(n, R_PRES, cnt, G7.PROFILE[pk])
                c = make_core(D_MODEL, SEED, r0)
                kk, _ = drive(c, D, "R", keep=keep); rec(f"{pk}u", c, None)
                assert kk == kB, (n, r0, pk, kk, kB)
                if pk == "F":
                    prof = G7.realized_profile(keep, n, R_PRES)
                chi, sig, _ = crosstalk(c._W().detach(), RS["ho8"]["K"], D["va"], D["vb"], n, R_HO8)
            d8 = out["Fu"][1] - out["Ku"][1]
            d4 = out["Fu"][0] - out["Ku"][0]
            cells = " | ".join(f"{out[t][1]:.4f}" for t in ("A", "B", "P", "Fu", "Uu", "Ku"))
            print(f"{n:>3} {r0:>3} {tau:>7.4f} {kB:>5} {kB / n:>6.2f} | {cells} | {d8:+.4f}  {d4:+.4f}  "
                  f"{chi:>5.3f} {sig:>5.3f}  {[round(x, 2) for x in prof]}")
            print(f"        skip_rate {cal['skip_rate']:.3f} gap {cal['gap']:+.4f} rep {cal['skip_frac_repeat']:.3f} "
                  f"gate {cal['gate_ok']} | clean " +
                  " ".join(f"{t} {out[t][2]:.3f}" for t in ("A", "B", "P", "Fu", "Uu", "Ku")))


if __name__ == "__main__":
    main()
