#!/opt/pytorch/bin/python
r"""DIR-56 · **前提探针 A**：$P_v$ 是不是「**跨实例共享**的写入映射」？

★ 探针纪律：**不设阈值、不写判据、不产出裁决读数**（本文件只产"装置事实"）。
★ 出处：`framework/SGMN_v2.4_架构设计.md` §7.2「写入映射与门控 | $P_k,P_v$ 的固定、离线学习**或其他来源**」
   （第 523 行，**全文档仅此一处**提到 $P_k,P_v$ ⇒ 语义只能由**装置**给出）。

问的三件事：
  A1 `Pv1` 是否**逐实例独立**（实例 = 一次 `_mk(sd, …)` 抽出的装置状态）？
     ⇒ 若不独立/共享：`dir40.shared_pk1()` 那种"跨实例共享"的构造对 `Pv1` 是否也用得上？
  A2 `Pv1` 在装置里承担**几个角色**？（写入值 $v$ 的映射 vs 任务目标 $g_t$ 的映射）
     ⇒ 若**同一个** `Pv1` 同时定义"写什么"和"要预测什么"，则"学 $P_v$"不是选一个设计参数，
        而是在**改任务本身**（这是"能不能问"的关键）。
  A3 `Pv1` 是否在**可训参数集**里（`T40.init_params()`）？⇒ 定"固定"的现状。

用法：`/opt/pytorch/bin/python dir56/probe56a.py`
输出：`dir56/out/PROBE56A.json`
"""
import hashlib
import json
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
for _d in ("dir31", "dir33", "dir37", "dir38", "dir39", "dir40"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_d}")
import R1_task33 as R           # 任务层（逐字未改）
import R1_task37 as T37         # 测量层（_mk 逐字未改）
import R1_task40 as T40         # 协议层（init_params 逐字未改）

OUT = "/home/ec2-user/sgmn-loop/dir56/out"
SEED0 = 20261111
SEEDS = [SEED0 + 21000 + i for i in range(5)]      # 与 DIR-53 训练段同源（只取前 5 个实例）


def cos(a, b):
    return float((a.reshape(-1) @ b.reshape(-1)) / (a.norm() * b.norm()))


def main():
    os.makedirs(OUT, exist_ok=True)
    phi0 = T40.init_params()
    Ps = {sd: T37._mk(sd, phi0, train=True) for sd in SEEDS}

    # ── A3 先做：Pv1 是否可训
    keys = sorted(phi0.keys())
    pv_in_phi = any("Pv" in k for k in keys)

    rows = []
    for sd in SEEDS:
        P = Ps[sd]
        assert P["Pv1"].shape == (R.D_H, R.D_V), P["Pv1"].shape
        rows.append({"seed": sd,
                     "pv_fro": float(P["Pv1"].norm()), "pv_shape": list(P["Pv1"].shape),
                     "pk_fro": float(P["Pk1"].norm()), "pk_shape": list(P["Pk1"].shape),
                     "pv_sha12": hashlib.sha1(P["Pv1"].numpy().tobytes()).hexdigest()[:12]})

    # ── A1 跨实例相似度（Pv1 与 Pk1 各一张 5×5 余弦表）
    pv = [Ps[sd]["Pv1"] for sd in SEEDS]
    pk = [Ps[sd]["Pk1"] for sd in SEEDS]
    cospv = [[round(cos(pv[i], pv[j]), 6) for j in range(len(SEEDS))] for i in range(len(SEEDS))]
    cospk = [[round(cos(pk[i], pk[j]), 6) for j in range(len(SEEDS))] for i in range(len(SEEDS))]
    off_pv = [abs(cospv[i][j]) for i in range(len(SEEDS)) for j in range(len(SEEDS)) if i != j]
    off_pk = [abs(cospk[i][j]) for i in range(len(SEEDS)) for j in range(len(SEEDS)) if i != j]
    a1 = {"n_instances": len(SEEDS), "cospv": cospv, "cospk": cospk,
          "max_abs_offdiag_pv": max(off_pv), "max_abs_offdiag_pk": max(off_pk),
          "identical_instances": len(set(r["pv_sha12"] for r in rows)) < len(SEEDS)}

    # ── A2 角色：写入值 v 与任务目标 g_t 是否由**同一个** Pv1 定义
    P = Ps[SEEDS[0]]
    H = R.hs_of(P)
    V_write = torch.stack([torch.tanh(H[s] @ P["Pv1"]) for s in range(R.T)])       # 装置写步同式
    g_target = torch.stack(R.targets(P))                                          # 装置目标同式
    role_dev = float((V_write[:-1] - g_target).abs().max())                       # g_t = v(h_{t-1})

    # 同一个 map 同时驱动两者：对 Pv1 加扰动，看 v 与 g 的**相对**变化是否同步（成比例）
    g = torch.Generator().manual_seed(56_001)
    dPv = torch.randn(R.D_H, R.D_V, generator=g) * (P["Pv1"].norm() / (R.D_H * R.D_V) ** 0.5) * 1e-2
    P2 = dict(P)
    P2["Pv1"] = P["Pv1"] + dPv
    H = R.hs_of(P2)                      # h 只依赖 a,A1,b1 ⇒ 不受 Pv1 影响（严格：逐位相同）
    V2 = torch.stack([torch.tanh(H[s] @ P2["Pv1"]) for s in range(R.T)])
    g2 = torch.stack(R.targets(P2))
    pert = {"v_rel": float((V2 - V_write).norm() / V_write.norm()),
            "g_rel": float((g2 - g_target).norm() / g_target.norm()),
            "v_vs_g_bitwise_equal": bool(float((V2[:-1] - g2).abs().max()) == 0.0)}

    res = {"probe": "56a", "card": "probe-only", "seed0": SEED0, "seeds": SEEDS,
           "device": {"D_H": R.D_H, "D_K": R.D_K, "D_V": R.D_V, "T": R.T,
                      "pv1_shape": [R.D_H, R.D_V]},
           "A1_sharing": a1,
           "A2_role": {"max_abs_diff_v_write_vs_g_target": role_dev,
                       "n_pairs_compared": int(g_target.shape[0]),
                       "note": "g_t = tanh(h_{t-1}·Pv1) 与 v = tanh(h·Pv1) 同式 ⇒ 差应为 0",
                       "perturbation": pert},
           "A3_trainable": {"phi0_keys": keys, "pv_in_phi0": pv_in_phi,
                            "n_phi0": len(keys)},
           "rows": rows}
    with open(os.path.join(OUT, "PROBE56A.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print("── A3 Pv1 是否可训：phi0 共 %d 个键，含 Pv？%s" % (len(keys), pv_in_phi))
    print("── A1 跨实例余弦：|offdiag| 最大 Pv=%.3e、Pk=%.3e（相同实例？%s）"
          % (a1["max_abs_offdiag_pv"], a1["max_abs_offdiag_pk"], a1["identical_instances"]))
    print("── A2 角色：max|v - g_target| = %.3e（%d 对）；扰动 ⇒ v_rel=%.3e、g_rel=%.3e、逐位同？%s"
          % (role_dev, int(g_target.shape[0]), pert["v_rel"], pert["g_rel"],
             pert["v_vs_g_bitwise_equal"]))
    print("Pv1 例子：" + json.dumps(rows[0], ensure_ascii=False))


if __name__ == "__main__":
    main()