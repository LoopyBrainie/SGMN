#!/opt/pytorch/bin/python
r"""DIR-56 · **前提探针 B**：$P_v$ 的**可寻址余量** $R_v$ 还在吗？

★ 探针纪律：**不设阈值、不写判据、不产出裁决读数**。
★ 关键设计（**首版在这里错过，已留档**）：`Pv1` **逐实例独立** ⇒ 变体必须**逐实例**相对该实例自己的 `Pv1`
   定义（`pv_sd → φ(pv_sd)`），不能用某一个实例的 `Pv1` 去套所有实例的 `h`（那会把"真映射"算成噪声）。
★ 两个口径（同一批 96 实例 × 16 步 = 1536 对）：
   `L_off  = mean‖Y − V‖²`（自检索，DIR-53 冻结同式；`W = VᵀK`、`Y = (W Kᵀ)ᵀ`）
   `L_ideal= min_A mean‖Y A − Vm‖²`（**理想线性读出**；`Vm` = 该实例真值 `tanh(h·Pv1)`）
   ★ 为什么必须用 `L_ideal`：裸 `‖Y−Vm‖²` 被 `W` 的整体尺度支配，`Y→0` 时反而变小 ⇒ 会把"塌缩"误报成改善。

用法：`/opt/pytorch/bin/python dir56/probe56b.py`
输出：`dir56/out/PROBE56B.json`
"""
import json
import math
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
for _d in ("dir31", "dir33", "dir37", "dir38", "dir39", "dir40"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_d}")
import R1_task33 as R
import R1_task37 as T37
import R1_task40 as T40

OUT = "/home/ec2-user/sgmn-loop/dir56/out"
SEED0 = 20261111
TRAIN_SEEDS = [SEED0 + 21000 + i for i in range(96)]


def l_ideal(Y, Tgt):
    A = torch.linalg.lstsq(Y, Tgt).solution
    return float((((Y @ A) - Tgt) ** 2).mean()), float(A.norm())


def effrank(Mx):
    s = torch.linalg.svdvals(Mx.to(torch.float64))
    if float(s.sum()) == 0.0:
        return 0.0
    return float(s.sum() ** 2 / (s ** 2).sum())


def assemble(pairs, map_fn):
    r"""`map_fn(pv_sd)` ⇒ 写入侧映射（**逐实例**）；返回拼接后的 `Y`（读出）、`V`（写入值）、`Vm`（真值）。"""
    Ys, Vs, Vms = [], [], []
    for H, pk, pv in pairs:
        K = torch.tanh(H @ pk)
        Vm = torch.tanh(H @ pv)
        V = torch.tanh(H @ map_fn(pv))
        W = V.T @ K
        Ys.append((W @ K.T).T)
        Vs.append(V)
        Vms.append(Vm)
    return torch.cat(Ys), torch.cat(Vs), torch.cat(Vms)


def main():
    os.makedirs(OUT, exist_ok=True)
    phi0 = T40.init_params()
    pairs = []
    for sd in TRAIN_SEEDS:
        P = T37._mk(sd, phi0, train=True)
        pairs.append((torch.stack(R.hs_of(P)), P["Pk1"], P["Pv1"]))
    n_pairs = sum(H.shape[0] for H, _, _ in pairs)

    g = torch.Generator().manual_seed(56_002)
    orth = {}
    for i in range(5):
        A = torch.randn(R.D_V, R.D_V, generator=g)
        Q, _ = torch.linalg.qr(A)
        orth[f"orth_Q{i}"] = Q
    rnd = {}
    for i in range(3):
        rnd[f"random_same_scale_{i}"] = torch.randn(R.D_H, R.D_V, generator=g)
    u_dict = {}
    for j, (H, pk, pv) in enumerate(pairs):
        _, S, Vh = torch.linalg.svd(pv)
        u = Vh[0].reshape(-1, 1)
        u_dict[j] = (u @ u.T, float(pv.norm() / math.sqrt(pv.numel())))

    variants = {"true_Pv1": lambda pv, j=None: pv,
                "zero": lambda pv, j=None: torch.zeros_like(pv),
                "s0.25": lambda pv, j=None: 0.25 * pv,
                "s0.50": lambda pv, j=None: 0.50 * pv,
                "s1.50": lambda pv, j=None: 1.50 * pv,
                "s4.00": lambda pv, j=None: 4.00 * pv,
                "rank1_proj": lambda pv, j=None: pv @ u_dict[j][0]}

    tab = {}
    for name, fn in variants.items():
        Ys, Vs, Vms = [], [], []
        for j, (H, pk, pv) in enumerate(pairs):
            K = torch.tanh(H @ pk)
            Vm = torch.tanh(H @ pv)
            V = torch.tanh(H @ fn(pv, j))
            W = V.T @ K
            Ys.append((W @ K.T).T)
            Vs.append(V)
            Vms.append(Vm)
        Y, V, Vm = torch.cat(Ys), torch.cat(Vs), torch.cat(Vms)
        li, an = l_ideal(Y, Vm)
        tab[name] = {"L_off": float(((Y - V) ** 2).mean()), "L_ideal": li, "readout_norm": an,
                     "L_raw": float(((Y - Vm) ** 2).mean()),
                     "v_norm_mean": float(V.norm(dim=1).mean()), "effrank_v": effrank(V)}
    for name, Q in orth.items():
        Ys, Vs, Vms = [], [], []
        for H, pk, pv in pairs:
            K = torch.tanh(H @ pk)
            Vm = torch.tanh(H @ pv)
            V = torch.tanh(H @ (pv @ Q))
            W = V.T @ K
            Ys.append((W @ K.T).T)
            Vs.append(V)
            Vms.append(Vm)
        Y, V, Vm = torch.cat(Ys), torch.cat(Vs), torch.cat(Vms)
        li, an = l_ideal(Y, Vm)
        tab[name] = {"L_off": float(((Y - V) ** 2).mean()), "L_ideal": li, "readout_norm": an,
                     "L_raw": float(((Y - Vm) ** 2).mean()),
                     "v_norm_mean": float(V.norm(dim=1).mean()), "effrank_v": effrank(V)}
    for name, base in rnd.items():
        Ys, Vs, Vms = [], [], []
        for H, pk, pv in pairs:
            K = torch.tanh(H @ pk)
            Vm = torch.tanh(H @ pv)
            pv2 = base * (pv.norm() / base.norm())          # 同尺度的**别的**映射
            V = torch.tanh(H @ pv2)
            W = V.T @ K
            Ys.append((W @ K.T).T)
            Vs.append(V)
            Vms.append(Vm)
        Y, V, Vm = torch.cat(Ys), torch.cat(Vs), torch.cat(Vms)
        li, an = l_ideal(Y, Vm)
        tab[name] = {"L_off": float(((Y - V) ** 2).mean()), "L_ideal": li, "readout_norm": an,
                     "L_raw": float(((Y - Vm) ** 2).mean()),
                     "v_norm_mean": float(V.norm(dim=1).mean()), "effrank_v": effrank(V)}

    t_true = tab["true_Pv1"]
    nc = [k for k in tab if tab[k]["v_norm_mean"] >= 0.5 * max(t_true["v_norm_mean"], 1e-12)]
    best_off_all = min(tab, key=lambda k: tab[k]["L_off"])
    best_off_nc = min(nc, key=lambda k: tab[k]["L_off"])
    best_task_all = min(tab, key=lambda k: tab[k]["L_ideal"])
    best_task_nc = min(nc, key=lambda k: tab[k]["L_ideal"])
    mis = [k for k in tab if tab[k]["L_off"] < t_true["L_off"] - 1e-12
           and tab[k]["L_ideal"] > t_true["L_ideal"] + 1e-12]

    res = {"probe": "56b", "n_pairs": n_pairs, "n_instances": len(pairs),
           "train_seeds": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
           "variants_defined": "逐实例 φ(Pv1_sd)（★ 首版用单实例 Pv1 套所有实例 ⇒ 已作废）",
           "table": tab,
           "summary": {"L_off_true": t_true["L_off"], "L_ideal_true": t_true["L_ideal"],
                       "best_L_off_all": best_off_all, "best_L_off_noncollapse": best_off_nc,
                       "best_L_ideal_all": best_task_all, "best_L_ideal_noncollapse": best_task_nc,
                       "R_v_off_noncollapse": t_true["L_off"] - tab[best_off_nc]["L_off"],
                       "R_v_ideal_noncollapse": t_true["L_ideal"] - tab[best_task_nc]["L_ideal"],
                       "L_ideal_zero": tab["zero"]["L_ideal"],
                       "misaligned_variants": mis, "n_variants": len(tab)}}
    with open(os.path.join(OUT, "PROBE56B.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print("%-22s %12s %10s %9s %8s" % ("variant", "L_off", "L_ideal", "|v|", "effrank"))
    for name in ["true_Pv1", "zero", "s0.25", "s0.50", "s1.50", "s4.00",
                 "rank1_proj", "orth_Q0", "orth_Q1", "random_same_scale_0"]:
        t = tab[name]
        print("%-22s %12.2f %10.5f %9.4f %8.4f"
              % (name, t["L_off"], t["L_ideal"], t["v_norm_mean"], t["effrank_v"]))
    print("★ L_ideal(true) = %.6f（=0 表示「写入值就是目标值」）· L_ideal(zero) = %.6f"
          % (t_true["L_ideal"], tab["zero"]["L_ideal"]))
    print("★ 非塌缩最优：L_off→%s（R_v_off=%.2f）· L_ideal→%s（R_v_ideal=%.6f）"
          % (best_off_nc, res["summary"]["R_v_off_noncollapse"],
             best_task_nc, res["summary"]["R_v_ideal_noncollapse"]))
    print("★ 目标错位（L_off 更好而 L_ideal 更差）：%s" % (mis if mis else "无"))


if __name__ == "__main__":
    main()