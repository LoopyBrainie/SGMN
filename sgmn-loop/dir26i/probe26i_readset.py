r"""DIR-26-I · probe26i_readset.py —— **写侧固定、读集可变**：headroom 与单种子电平

卡片冻结前的**探针**（不是判据读数）。固定写侧 $n_{\rm kv}=35$，把**读集**改成 $N_{\rm read}\in\{10,20,28,35\}$：
  * 写侧条目数（共享支撑里的干扰源数）不变 ⇒ 若剖面效应由**写侧**决定，$\Delta$ 应基本不动；
  * 读侧竞争者数变了 ⇒ 若由**读侧竞争**决定，$\Delta$ 应随 $N_{\rm read}$ 变化。
★ 只回答：① 每个 $N_{\rm read}$ 的六臂电平（判断地板/天花板）；② 单种子 $\Delta$ 的走向；③ $\chi$ 与 $\bar q$。

★ 前缀恒等（地基）：写流按 $n_{\rm kv}$ 建，读流按 $N_{\rm read}$ **另建**（不同 seed，独立抖动键），
  读的目标值取**写流**的前 $N_{\rm read}$ 个 $v^b$（`D["vb"][:N_read]`）——D-40 的做法。
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

R_PRES, R_HO4, R_HO8, HO_OFFSET, R_MAIN, R0 = 4, 4, 8, 999, 0.40, 8
SEED = 20261028
N_KV = 35
N_READ_GRID = (10, 20, 28, 35)


def crosstalk(W, K, vb, n, r_pres):
    s, x = [], []
    for rho in range(r_pres):
        Y = (W @ K[rho].T).T
        Yn = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-30)
        C = Yn @ vb.T
        s.append(torch.diagonal(C).mean().item())
        off = C.clone(); off.fill_diagonal_(0.0)
        x.append(off.abs().sum().item() / (n * (n - 1)))
    return float(np.mean(x) / np.mean(s)), float(np.mean(s))


def main():
    print(f"seed={SEED}  n_kv={N_KV}  r={R0}  R={R_PRES}  rho={R_MAIN}")
    print(f"{'Nr':>3} {'c_tau':>7} {'k_B':>5} {'q̄':>6} | " +
          " | ".join(f"{t:>9}" for t in ("A", "B", "P", "Fu", "Uu", "Ku")) +
          " | dFK(ho8)  dFK(ho4)   chi   sig  cleanA")
    D = build_stream(SEED, D_MODEL, N_KV, R_PRES, theta_of(0.9))
    assert int(D["n"]) == N_KV
    G5.N_DIST = N_KV
    try:
        cal = calibrate_w(R0, R_MAIN, R_PRES, theta_of(0.9))
    finally:
        G5.N_DIST = N_DIST
    tau = cal["c_tau"]
    for n_read in N_READ_GRID:
        RS = {"ho4": build_stream(SEED + HO_OFFSET, D_MODEL, n_read, R_HO4, theta_of(0.9)),
              "ho8": build_stream(SEED + HO_OFFSET, D_MODEL, n_read, R_HO8, theta_of(0.9))}
        assert int(RS["ho8"]["n"]) == n_read
        out = {}

        def rec(tag, core):
            W = core._W().detach()
            P = W - core.W0
            m4 = ret_jitter(P, RS["ho4"]["K"], D["vb"][:n_read], n_read, R_HO4)
            m8 = ret_jitter(P, RS["ho8"]["K"], D["vb"][:n_read], n_read, R_HO8)
            out[tag] = (m4["ret1_mean"], m8["ret1_mean"],
                        metrics_plas(W, core.W0, D["proto"][:n_read], D["vb"][:n_read], n_read)["ret1"])

        c = make_core(D_MODEL, SEED, R0); kA, _ = drive(c, D, "A"); rec("A", c)
        c = make_core(D_MODEL, SEED, R0); kB, logB = drive(c, D, "B", tau=tau); rec("B", c)
        cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, N_KV)
        c = make_core(D_MODEL, SEED, R0)
        drive(c, D, "R", keep=keep_positions(N_KV, R_PRES, kB)); rec("P", c)
        chi, sig = float("nan"), float("nan")
        for pk in ("F", "U", "K"):
            keep, _ = G7.build_keep(N_KV, R_PRES, cnt, G7.PROFILE[pk])
            c = make_core(D_MODEL, SEED, R0)
            kk, _ = drive(c, D, "R", keep=keep); rec(f"{pk}u", c)
            assert kk == kB, (n_read, pk, kk, kB)
            if pk == "K":
                chi, sig = crosstalk(c._W().detach(), RS["ho8"]["K"], D["vb"][:n_read], n_read, R_HO8)
        d8 = out["Fu"][1] - out["Ku"][1]
        d4 = out["Fu"][0] - out["Ku"][0]
        cells = " | ".join(f"{out[t][1]:.4f}" for t in ("A", "B", "P", "Fu", "Uu", "Ku"))
        print(f"{n_read:>3} {tau:>7.4f} {kB:>5} {kB / N_KV:>6.2f} | {cells} | {d8:+.4f}  {d4:+.4f}  "
              f"{chi:>5.3f} {sig:>5.3f}  {out['A'][2]:.3f}")


if __name__ == "__main__":
    main()
