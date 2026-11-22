#!/opt/pytorch/bin/python
r"""DIR-57 · **前提探针 E（决定性复核）**：`cos` 在 rank-2 外积记忆里**不能**当"寻址"的度量。

★ 触发（为什么必须复核）：`PROBE57A/D` 的 `c_rand`（未写键对 `v_P` 的余弦）呈**±1 双峰**
   （q05 = **−0.983**、q95 = **+0.992**，`|c_rand| ≤ 0.3` 只有 **7/64**）——这正是
   **单笔写入后 `W` 近似 rank-1**（`W ≈ σ u vᵀ`）的表现：此时 `W k' ∥ v` 对**任意**键 `k'` 成立，
   余弦只剩 **±1**（符号取决于 `k'·u` 的正负）⇒ **"方向"不含寻址信息**。
★ 因此"跨键绑定"必须用**幅度**度量：`m_i = ‖W k_i‖`。本探针只做装置事实：
   ① 写 1 把键后：`m_a` 与 20 把**随机未写**键的 `m` 分布（比值中位 / 分位）；
   ② 写 2 把键后：`m_a, m_b` 与随机键的比值；
   ③ 同态的秩结构 `σ1, σ2`（`rank=2`）；
   ④ 未写键余弦的**逐键**分布（证实 ±1 双峰，不是"中位为负"这种统计假象）。
★ 探针纪律：不设阈值、不写判据。
用法：`/opt/pytorch/bin/python dir57/probe57e.py` → `dir57/out/PROBE57E.json`
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
SEEDS = [SEED0 + 58300 + i for i in range(32)]
OFF_K, OFF_V, OFF_R = 57401, 57402, 57403
N_RAND = 20


def write_pair(U, Vm, k, v):
    return T48.step_pair(U, Vm, k, v, 1.0)


def unit_key(g):
    return R.KEY_SCALE * T51.gen_vec(g, M.D_K)


def read_vec(U, Vm, k):
    return (M._W(U, Vm) @ k).detach()


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sd in SEEDS:
        P = R.build(sd, "V0")
        U0, V0 = P["U1"].clone(), P["V1"].clone()
        gk = torch.Generator().manual_seed(sd + OFF_K)
        gv = torch.Generator().manual_seed(sd + OFF_V)
        gr = torch.Generator().manual_seed(sd + OFF_R)
        kA, kB = unit_key(gk), unit_key(gk)
        vP = T51.gen_vec(gv, M.D_V)
        kRand = [unit_key(gr) for _ in range(N_RAND)]

        # ── 写 1 把
        U1, V1 = write_pair(U0, V0, kA, vP)
        mA = float(read_vec(U1, V1, kA).norm())
        mR1 = [float(read_vec(U1, V1, k).norm()) for k in kRand]
        cosR1 = [float((read_vec(U1, V1, k) @ vP) / (read_vec(U1, V1, k).norm() * vP.norm())) for k in kRand]
        cA1 = float((read_vec(U1, V1, kA) @ vP) / (read_vec(U1, V1, kA).norm() * vP.norm()))
        s1 = [float(x) for x in torch.linalg.svdvals(M._W(U1, V1).detach())[:3]]
        # ── 写 2 把（同值）
        U2, V2 = write_pair(U1, V1, kB, vP)
        mA2 = float(read_vec(U2, V2, kA).norm())
        mB2 = float(read_vec(U2, V2, kB).norm())
        mR2 = [float(read_vec(U2, V2, k).norm()) for k in kRand]
        s2 = [float(x) for x in torch.linalg.svdvals(M._W(U2, V2).detach())[:3]]

        rows.append({"seed": sd,
                     "m_a_one": mA, "m_rand_one_median": statistics.median(mR1),
                     "ratio_one": mA / statistics.median(mR1) if statistics.median(mR1) > 0 else float("inf"),
                     "cos_rand_one": cosR1, "cos_a_one": cA1, "sigma_one": s1,
                     "m_a_two": mA2, "m_b_two": mB2, "m_rand_two_median": statistics.median(mR2),
                     "ratio_a_two": mA2 / statistics.median(mR2) if statistics.median(mR2) > 0 else float("inf"),
                     "ratio_b_two": mB2 / statistics.median(mR2) if statistics.median(mR2) > 0 else float("inf"),
                     "sigma_two": s2})

    def q(vals, p):
        v = sorted(vals)
        return v[min(len(v) - 1, int(p * (len(v) - 1)))]

    def summ(key):
        vals = [x[key] for x in rows]
        return {"median": statistics.median(vals), "q05": q(vals, 0.05), "q95": q(vals, 0.95),
                "min": min(vals), "max": max(vals)}

    cos_all = [c for x in rows for c in x["cos_rand_one"]]
    abs_cos = [abs(c) for c in cos_all]
    res = {"probe": "57e", "n_instances": len(rows), "n_random_keys": N_RAND,
           "one_write": {"m_a": summ("m_a_one"), "m_rand": summ("m_rand_one_median"),
                         "ratio": summ("ratio_one"), "cos_a": summ("cos_a_one"),
                         "sigma": [statistics.median([x["sigma_one"][i] for x in rows]) for i in range(3)]},
           "two_writes": {"m_a": summ("m_a_two"), "m_b": summ("m_b_two"),
                          "m_rand": summ("m_rand_two_median"),
                          "ratio_a": summ("ratio_a_two"), "ratio_b": summ("ratio_b_two"),
                          "sigma": [statistics.median([x["sigma_two"][i] for x in rows]) for i in range(3)]},
           "unwritten_key_cos": {"n": len(cos_all), "median": statistics.median(cos_all),
                                 "frac_abs_gt_0_9": sum(1 for c in abs_cos if c > 0.9) / len(abs_cos),
                                 "q05": q(cos_all, 0.05), "q95": q(cos_all, 0.95)},
           "rows": rows}
    with open(os.path.join(OUT, "PROBE57E.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    o, t = res["one_write"], res["two_writes"]
    print("── 写 1 把：‖Wk_a‖ %.4f vs 随机键中位 %.4f ⇒ 比值 中位 %.3f（q05 %.3f q95 %.3f）"
          % (o["m_a"]["median"], o["m_rand"]["median"], o["ratio"]["median"], o["ratio"]["q05"], o["ratio"]["q95"]))
    print("    σ = [%.3f, %.3f, %.3f]" % tuple(o["sigma"]))
    print("── 写 2 把：‖Wk_a‖ %.4f ‖Wk_b‖ %.4f vs 随机 %.4f ⇒ 比值 a %.3f / b %.3f"
          % (t["m_a"]["median"], t["m_b"]["median"], t["m_rand"]["median"],
             t["ratio_a"]["median"], t["ratio_b"]["median"]))
    print("    σ = [%.3f, %.3f, %.3f]" % tuple(t["sigma"]))
    uc = res["unwritten_key_cos"]
    print("── 未写键余弦（%d 个读数）：中位 %.4f · |cos|>0.9 占 %.1f%% · q05 %.3f q95 %.3f"
          % (uc["n"], uc["median"], 100 * uc["frac_abs_gt_0_9"], uc["q05"], uc["q95"]))


if __name__ == "__main__":
    main()