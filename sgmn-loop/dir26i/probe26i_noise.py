r"""DIR-26-I · probe26i_noise.py —— **纯干扰注入**：写 $N_{\rm real}{+}M$ 个条目，只读其中 $N_{\rm real}$ 个

卡片冻结前的**探针**（不是判据读数）。只回答三件事：
  ① $M\in\{0,10,20,40\}$ 时六臂的 `ho8` 电平 —— 判断地板/天花板；
  ② $\bar q$、$c_\tau$、$k_B$ 在 $n_{\rm kv}=N_{\rm real}{+}M$ 上是否仍可标定；
  ③ **串扰比** $\chi$ 随 $M$ 的走向 —— 检验"$\chi$ 是否就是载荷效应的载体"。

★ 为什么注入噪声条目：D-40 里 $N$ 同时改了三件事——(a) 共享支撑里的条目数（**写侧干扰**）、
  (b) `argmax` 的竞争者数（**读侧竞争**）、(c) 任务难度。三者被绑在一个旋钮上。
  注入 $M$ 个**只写不读**的条目，可以只动 (a)：读集始终是那 $N_{\rm real}$ 个真实条目。
  $M$ 增大而 $\lvert\Delta\rvert$ 不增大 ⇒ 干扰不是载体、读侧竞争才是；反之 ⇒ 干扰是载体。

★ 前缀恒等（本设计的地基）：`build_stream` 用同一个 torch 生成器按行主序抽取，
  $n_{\rm kv}$ 流的**前 $N_{\rm real}$ 行**与 $N_{\rm real}$ 流的行**逐位相同**（$E$ 是单位阵、$U$/$v^a$/$v^b$ 只按行取）。
  Stage A 必须逐位断言这一点，否则"只加噪声"不成立。
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
N_REAL = 10
M_GRID = (0, 10, 20, 40)


def crosstalk(W, K, va, vb, n, r_pres):
    r"""$\chi=x/s$（只在**真实条目**上算；噪声条目不进读集，但它们的写入就在 $W$ 里）。"""
    s, x = [], []
    for rho in range(r_pres):
        Y = (W @ K[rho].T).T
        Yn = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-30)
        for v in (va, vb):
            C = Yn @ v.T
            s.append(torch.diagonal(C).mean().item())
            off = C.clone()
            off.fill_diagonal_(0.0)
            x.append(off.abs().sum().item() / (n * (n - 1)))
    return float(np.mean(x) / np.mean(s)), float(np.mean(s)), float(np.mean(x))


def main():
    print(f"seed={SEED}  N_real={N_REAL}  r={R0}  R={R_PRES}  M={M}  rho={R_MAIN}")
    print(f"{'M':>3} {'n_kv':>5} {'c_tau':>7} {'k_B':>5} {'q̄':>6} | " +
          " | ".join(f"{t:>9}" for t in ("A", "B", "P", "Fu", "Uu", "Ku")) +
          " | dFK(ho8)  dFK(ho4)   chi   sig")
    RS = {"ho4": build_stream(SEED + HO_OFFSET, D_MODEL, N_REAL, R_HO4, theta_of(0.9)),
          "ho8": build_stream(SEED + HO_OFFSET, D_MODEL, N_REAL, R_HO8, theta_of(0.9))}
    for m_noise in M_GRID:
        n_kv = N_REAL + m_noise
        D = build_stream(SEED, D_MODEL, n_kv, R_PRES, theta_of(0.9))
        assert int(D["n"]) == n_kv and int(RS["ho8"]["n"]) == N_REAL
        G5.N_DIST = n_kv
        try:
            cal = calibrate_w(R0, R_MAIN, R_PRES, theta_of(0.9))
        finally:
            G5.N_DIST = N_DIST
        tau = cal["c_tau"]
        out = {}

        def rec(tag, core):
            W = core._W().detach()
            P = W - core.W0
            # ★ 读侧：抖动键取自独立读流 RS，**目标值必须取写流的 $v^b$**（D-40 的做法）
            m4 = ret_jitter(P, RS["ho4"]["K"], D["vb"][:N_REAL], N_REAL, R_HO4)
            m8 = ret_jitter(P, RS["ho8"]["K"], D["vb"][:N_REAL], N_REAL, R_HO8)
            out[tag] = (m4["ret1_mean"], m8["ret1_mean"],
                        metrics_plas(W, core.W0, D["proto"][:N_REAL], D["vb"][:N_REAL], N_REAL)["ret1"])

        c = make_core(D_MODEL, SEED, R0); kA, _ = drive(c, D, "A"); rec("A", c)
        c = make_core(D_MODEL, SEED, R0); kB, logB = drive(c, D, "B", tau=tau); rec("B", c)
        cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, n_kv)
        c = make_core(D_MODEL, SEED, R0)
        drive(c, D, "R", keep=keep_positions(n_kv, R_PRES, kB)); rec("P", c)
        chi, sig = float("nan"), float("nan")
        for pk in ("F", "U", "K"):
            keep, _ = G7.build_keep(n_kv, R_PRES, cnt, G7.PROFILE[pk])
            c = make_core(D_MODEL, SEED, R0)
            kk, _ = drive(c, D, "R", keep=keep); rec(f"{pk}u", c)
            assert kk == kB, (m_noise, pk, kk, kB)
            if pk == "K":
                chi, sig, _ = crosstalk(c._W().detach(), RS["ho8"]["K"][:, :N_REAL], D["va"][:N_REAL],
                                        D["vb"][:N_REAL], N_REAL, R_HO8)
        d8 = out["Fu"][1] - out["Ku"][1]
        d4 = out["Fu"][0] - out["Ku"][0]
        if m_noise == 0:      # 自检：M=0 必须复现 D-40 的 (N=10, seed 20261028) 电平
            want = {"A": 0.8500, "B": 0.8625, "Fu": 0.8625, "Uu": 0.8000, "Ku": 0.8000}
            bad = {t2: (out[t2][1], v) for t2, v in want.items() if abs(out[t2][1] - v) > 1e-9}
            print(f"        ★ M=0 自检 vs D-40：{'逐位一致' if not bad else f'不符 {bad}'}")
        cells = " | ".join(f"{out[t][1]:.4f}" for t in ("A", "B", "P", "Fu", "Uu", "Ku"))
        print(f"{m_noise:>3} {n_kv:>5} {tau:>7.4f} {kB:>5} {kB / n_kv:>6.2f} | {cells} | {d8:+.4f}  {d4:+.4f}  "
              f"{chi:>5.3f} {sig:>5.3f}")
        print(f"        skip {cal['skip_rate']:.3f} gap {cal['gap']:+.4f} rep {cal['skip_frac_repeat']:.3f} "
              f"gate {cal['gate_ok']} | clean " +
              " ".join(f"{t} {out[t][2]:.3f}" for t in ("A", "B", "P", "Fu", "Uu", "Ku")))


if __name__ == "__main__":
    main()
