#!/opt/pytorch/bin/python
r"""DIR-57 · **前提探针 B**：容量——`r = 2, D_V = 4, D_K = 4` 时能挂几个键？

★ 探针纪律：**不设阈值、不写判据、不产出裁决读数**。
★ 两族（每族都**逐字**用 `dir48.step_pair`，α=1，16 内步）：
  ① **同值族**：`m` 个互不平行（两两 `|cos| ≤ 0.3`）的键**全部绑定同一个 `v_P`**，按 `k_1..k_m` 顺序写入
     ⇒ 逐步记录 `cos(W k_i, v_P)`（i ≤ 当前步数）⇒ 得到**干扰曲线**（最早那个键被后续写入推掉多少）
  ② **异值族**：`m` 个键各绑定**互不平行**的值 `v_1..v_m` ⇒ 检验"赢者通吃"（是否只剩最后一个）
★ 容量事实：`W = U Vᵀ`，`U∈R^{8×2}`、`V∈R^{4×2}` ⇒ `rank(W) ≤ 2`，值空间 `D_V = 4`。
用法：`/opt/pytorch/bin/python dir57/probe57b.py` → `dir57/out/PROBE57B.json`
"""
import json
import os
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir51"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task48 as T48      # noqa: E402
import R1_task51 as T51      # noqa: E402

OUT = os.path.join(ROOT, "dir57", "out")
SEED0 = 20261117
SEEDS = [SEED0 + 58100 + i for i in range(48)]
M_LIST = (1, 2, 3, 4)
OFF_KEY, OFF_VAL, OFF_ANCHOR = 57201, 57202, 57203
N_ANCHOR = 32


def unit(g, D):
    return T51.gen_vec(g, D)


def key_batch(g, m, max_cos=0.3):
    ks = []
    while len(ks) < m:
        k = R.KEY_SCALE * unit(g, M.D_K)
        if all(abs(float((k @ q) / (k.norm() * q.norm()))) <= max_cos for q in ks):
            ks.append(k)
    return ks


def val_batch(g, m, max_cos=0.3):
    vs = []
    while len(vs) < m:
        v = unit(g, M.D_V)
        if all(abs(float(v @ q)) <= max_cos for q in vs):
            vs.append(v)
    return vs


def null_q95(U, Vm, k, g):
    return T51.q95([T51.read_P(U, Vm, k, unit(g, M.D_V))[0] for _ in range(N_ANCHOR)])


def main():
    os.makedirs(OUT, exist_ok=True)
    same, diff = [], []
    for sd in SEEDS:
        P = R.build(sd, "V0")
        U0, V0 = P["U1"].clone(), P["V1"].clone()
        g = torch.Generator().manual_seed(sd + OFF_KEY)
        gv = torch.Generator().manual_seed(sd + OFF_VAL)
        ga = torch.Generator().manual_seed(sd + OFF_ANCHOR)
        kmax = max(M_LIST)
        ks = key_batch(g, kmax)
        vs_same = [unit(gv, M.D_V)] * kmax                       # 同一个值（单位向量）
        vs_dist = val_batch(gv, kmax)

        # ── 同值族：写一个、读全部已写
        U, Vm = U0.clone(), V0.clone()
        traj_same = []                                           # traj_same[j][i] = 写完 j+1 个后对 k_{i+1} 的 cos
        for j in range(kmax):
            U, Vm = T48.step_pair(U, Vm, ks[j], vs_same[j], 1.0)
            traj_same.append([T51.read_P(U, Vm, ks[i], vs_same[0])[0] for i in range(j + 1)])
        # ── 异值族：同样逐步记录
        Ud, Vd = U0.clone(), V0.clone()
        traj_diff = []
        for j in range(kmax):
            Ud, Vd = T48.step_pair(Ud, Vd, ks[j], vs_dist[j], 1.0)
            traj_diff.append([T51.read_P(Ud, Vd, ks[i], vs_dist[i])[0] for i in range(j + 1)])
        nulls = [null_q95(U0, V0, ks[i], ga) for i in range(kmax)]
        s = torch.linalg.svdvals(M._W(U, Vm).detach())
        same.append({"seed": sd, "traj": traj_same, "nulls": nulls,
                     "sigma": [float(x) for x in s[:3]]})
        diff.append({"seed": sd, "traj": traj_diff, "nulls": nulls,
                     "sigma": [float(x) for x in s[:3]]})

    def agg(rows, m):
        r"""`m` 个键全写完后：最早键 / 末键 / 全部键的 cos 中位；以及"全部超零假设"的比例。"""
        earliest = statistics.median([r["traj"][m - 1][0] for r in rows])
        last = statistics.median([r["traj"][m - 1][m - 1] for r in rows])
        allk = statistics.median([statistics.fmean(r["traj"][m - 1]) for r in rows])
        frac_all_ok = sum(1 for r in rows
                          if all(r["traj"][m - 1][i] > r["nulls"][i] for i in range(m))) / len(rows)
        return {"m": m, "earliest_median": earliest, "last_median": last,
                "mean_over_keys_median": allk, "frac_all_above_null": frac_all_ok,
                "null_median": statistics.median([r["nulls"][0] for r in rows])}

    res = {"probe": "57b", "device": "DIR-51 (write=dir48.step_pair)",
           "seeds": [SEEDS[0], SEEDS[-1]], "n_instances": len(SEEDS),
           "rank_cap": "rank(W) ≤ 2 (U∈R^{8×2}, V∈R^{4×2})",
           "same_value": [agg(same, m) for m in M_LIST],
           "distinct_values": [agg(diff, m) for m in M_LIST],
           "sigma_median_after_max_m": [statistics.median([r["sigma"][i] for r in same]) for i in range(3)],
           "same_traj_example": same[0]["traj"], "distinct_traj_example": diff[0]["traj"]}
    with open(os.path.join(OUT, "PROBE57B.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print("── 同值族（m 个键 → 同一个 v_P）：写完 m 个后的 cos")
    for r in res["same_value"]:
        print("   m=%d：最早键 %.4f · 末键 %.4f · 全键均值 %.4f · 全部超零假设(%.4f) %d/%d"
              % (r["m"], r["earliest_median"], r["last_median"], r["mean_over_keys_median"],
                 r["null_median"], round(r["frac_all_above_null"] * len(SEEDS)), len(SEEDS)))
    print("── 异值族（m 个键 → m 个不同值）")
    for r in res["distinct_values"]:
        print("   m=%d：最早键 %.4f · 末键 %.4f · 全键均值 %.4f · 全部超零假设 %d/%d"
              % (r["m"], r["earliest_median"], r["last_median"], r["mean_over_keys_median"],
                 round(r["frac_all_above_null"] * len(SEEDS)), len(SEEDS)))
    print("── rank 预算：写完 4 个后的奇异值中位 σ = [%.3f, %.3f, %.3f]"
          % tuple(res["sigma_median_after_max_m"]))


if __name__ == "__main__":
    main()