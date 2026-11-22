r"""DIR-26-I · probe26i_margin.py —— **边际读数** $\mu$ 的定义、自检与预读

卡片冻结前的**探针**。只回答三件事：
  ① 边际读数 $\mu$ 的定义是否与既有精度读数**逐位兼容**（$\mu>0\iff$ 命中）；
  ② 三个关键格点上 $\mu$ 与相对边际对比 $R_\mu=[\mu(F)-\mu(K)]/\mu(U0)$ 的电平（是否退化/变号）；
  ③ 精度口径与边际口径是否给出**同一个结论**（D-40 的 $\Delta$ 排序在 $\mu$ 上是否复现）。

$\mu$（每个 (轮, 条目) 的**有符号分离度**）：
    $$m_{i,\rho}=\cos(y_{i,\rho},v_i)-\max_{j\ne i}\cos(y_{i,\rho},v_j),\qquad y_{i,\rho}=P\,k_{i,\rho}$$
  $\mu=\overline{m}$。$\mu>0$ 当且仅当 $\arg\max$ 命中 ⇒ 与 `ret_jitter` 的 `ret1` **逐位一致**（本探针逐条断言）。
  ★ 为什么需要它：精度是**有界**读数（天花板 1.0、地板 1/n），当不同格点的容量差很多时，
    "精度差"会被天花板/地板压缩，$|\Delta|$ 无法跨格点比较（D-40 的 $r$ 轴探针就是这样被污染的）。
    $\mu$ 无界、随状态连续变化 ⇒ 用**相对**量 $R_\mu=\Delta\mu/\mu(U0)$ 可做跨格点比较（无量纲、无天花板）。
★ 构造/测量/定标/抖动/轮内位置规则全部 import 同一对象（`dir26g/R1_gating7.py`）；装置零改动。
"""
import sys

import numpy as np
import torch

for _p in ("dir26i", "dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

from R1_gating import D_MODEL, N_DIST, M, build_stream, drive, make_core, metrics_plas, unit
from R1_gating2 import keep_positions, item_counts
from R1_gating3 import ret_jitter, ret_jitter_bruteforce
import R1_gating5 as G5
from R1_gating5 import calibrate_w, theta_of
from R1_gating6 import _pos_idxs                     # noqa: F401
import R1_gating7 as G7

R_PRES, R_HO8, HO_OFFSET, R_MAIN, R0 = 4, 8, 999, 0.40, 8
SEED = 20261028
CELLS = ((35, 35), (20, 20), (35, 20))


def ret_jitter_margin(P, Kq, vb, n, r_q=8):
    r"""有符号分离度 $\mu$；同时返回逐位兼容的 `ret1` 与命中一致性计数。"""
    vbU = unit(vb)
    m_all, hit = [], np.zeros((r_q, n), dtype=bool)
    for rho in range(r_q):
        Y = P @ Kq[rho].T                       # (n, d_v)
        S = unit(Y.T) @ vbU.T                   # (n, n)：S[i,j] = cos(y_i, v_j)
        for i in range(n):
            row = S[i]
            j = int(torch.argmax(row).item())
            hit[rho, i] = j == i
            other = torch.cat([row[:i], row[i + 1:]])
            m_all.append(float(row[i] - other.max()))
    return {"mu": float(np.mean(m_all)), "mu_sd": float(np.std(m_all, ddof=1)),
            "ret1": float(hit.mean()), "per_round_mu": [],
            "n_hit": int(hit.sum()), "n_read": int(hit.size),
            "maj_need": r_q // 2 + 1,
            "ret1_maj": float(np.mean([hit[:, i].sum() >= r_q // 2 + 1 for i in range(n)]))}


def main():
    print(f"seed={SEED}  r={R0}  R={R_PRES}  rho={R_MAIN}")
    for n_kv, n_read in CELLS:
        D = build_stream(SEED, D_MODEL, n_kv, R_PRES, theta_of(0.9))
        RS = build_stream(SEED + HO_OFFSET, D_MODEL, n_read, R_HO8, theta_of(0.9))
        assert int(D["n"]) == n_kv and int(RS["n"]) == n_read, "流-N 不一致"
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
            mm = ret_jitter_margin(P, RS["K"], D["vb"][:n_read], n_read, R_HO8)
            acc = ret_jitter(P, RS["K"], D["vb"][:n_read], n_read, R_HO8)
            bf = ret_jitter_bruteforce(P, RS["K"], D["vb"][:n_read], n_read, R_HO8)
            assert mm["ret1"] == acc["ret1_mean"] == bf["ret1_mean"], (tag, mm["ret1"], acc, bf["ret1_mean"])
            assert (mm["mu"] > 0) == (mm["ret1"] > 0.5), (tag, mm["mu"], mm["ret1"])
            out[tag] = (mm["mu"], mm["ret1"], mm["ret1_maj"])

        c = make_core(D_MODEL, SEED, R0); kA, _ = drive(c, D, "A"); rec("A", c)
        c = make_core(D_MODEL, SEED, R0); kB, logB = drive(c, D, "B", tau=tau); rec("B", c)
        cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, n_kv)
        c = make_core(D_MODEL, SEED, R0)
        drive(c, D, "R", keep=keep_positions(n_kv, R_PRES, kB)); rec("P", c)
        for pk in ("F", "U", "K"):
            keep, _ = G7.build_keep(n_kv, R_PRES, cnt, G7.PROFILE[pk])
            c = make_core(D_MODEL, SEED, R0)
            kk, _ = drive(c, D, "R", keep=keep); rec(f"{pk}u", c)
            assert kk == kB, (n_kv, n_read, pk, kk, kB)
        rm = (out["Fu"][0] - out["Ku"][0]) / out["Uu"][0]
        da = out["Fu"][1] - out["Ku"][1]
        print(f"  n_kv={n_kv:>2} N_read={n_read:>2} | mu: " +
              " ".join(f"{t} {out[t][0]:+.4f}" for t in ("A", "B", "P", "Fu", "Uu", "Ku")) +
              f" | R_mu {rm:+.4f} | dAcc {da:+.4f} | acc: " +
              " ".join(f"{t} {out[t][1]:.4f}" for t in ("A", "B", "P", "Fu", "Uu", "Ku")))


if __name__ == "__main__":
    main()
