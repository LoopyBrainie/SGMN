#!/opt/pytorch/bin/python
r"""DIR-53 · 开跑前小档探针（**不进判据、不计 `SUCCEEDED`**）。

只回答三件事（`DEF-53-1` 的教训：**先证「梯度非零」与「控制真的生效」**）：
  P-1 `A53-6`：行范数投影**逐行** `< 1e-12`（否则"键变大/变小"会冒充收益）
  P-2 `A53-12`：离线目标**可训**（`∂L_off/∂P_k ≠ 0`）且 250 步后 `‖ΔP_k‖/‖P_k₀‖ > 0`
  P-3 `A53-11`：`∂L_task/∂P_k ≡ 0`（Freeze C 的机器化守卫）
★ 另测：离线学习**确定性**（两次跑逐位相同）与 **250 步 × 96 实例的真实成本**（供正式轮排程）。

用法：`/opt/pytorch/bin/python dir53/probe53b.py`
"""
import json
import sys
import time

import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir53")
import R1_task53 as T53

OUT = "/home/ec2-user/sgmn-loop/dir53/out/DIR53_probe_b.json"


def main():
    t0 = time.time()
    res = {"note": "小档探针：只验可行性与控制，不给判据"}
    pk0 = T53.pk1_for("P1")

    # ── P-1 / P-2：小档（8 实例）先验一次
    pk_a, om_a = T53.offline_learn_pk1(pk0, T53.TRAIN[:8], steps=30, lr=T53.LR)
    # ── 确定性：同输入再跑一次
    pk_b, om_b = T53.offline_learn_pk1(pk0, T53.TRAIN[:8], steps=30, lr=T53.LR)
    res["P-1_row_norm_dev_max"] = om_a["row_norm_dev_max"]
    res["P-1_ok"] = bool(om_a["row_norm_dev_max"] < 1e-12)
    res["P-2_rel_dPk"] = om_a["rel_dPk"]
    res["P-2_grad_pk_max"] = om_a["grad_pk_max"]
    res["P-2_ok"] = bool(om_a["rel_dPk"] > 0.0 and om_a["grad_pk_max"] > 0.0
                         and om_a["L_off_last20"] < om_a["L_off_first20"])
    res["determinism_sha1_equal"] = om_a["pk_sha1"] == om_b["pk_sha1"]
    res["determinism_ok"] = bool(res["determinism_sha1_equal"])

    # ── P-3：Freeze C 守卫（任务损失对 `P_k` 的梯度恒零）
    pkl = T53.pk1_for("P5").clone().requires_grad_(True)
    P = T53.T37._mk(T53.SMOKE_EVAL[0], T53.T40.init_params())
    L, _ = T53.T40.trace40(P, T53.N_REP, "learned", T53.SMOKE_EVAL[0],
                           T53.T40.init_params(), shared_pk1=pkl)
    L.backward()
    g_task = 0.0 if pkl.grad is None else float(pkl.grad.abs().max())
    res["P-3_task_grad_pk_max"] = g_task
    res["P-3_ok"] = (g_task == 0.0)

    # ── 真实成本：250 步 × 96 实例（正式轮的离线段）
    t1 = time.time()
    _, om_full = T53.offline_learn_pk1(pk0, T53.TRAIN, steps=T53.STEPS, lr=T53.LR)
    res["cost_full_offline_secs"] = time.time() - t1
    res["full_L_off_first20"] = om_full["L_off_first20"]
    res["full_L_off_last20"] = om_full["L_off_last20"]
    res["full_L_off_se_window"] = om_full["L_off_se_window"]
    res["full_rel_dPk"] = om_full["rel_dPk"]
    res["full_row_norm_dev_max"] = om_full["row_norm_dev_max"]
    res["n_pairs_full"] = om_full["n_pairs"]
    res["secs_total"] = time.time() - t0
    res["ok_all"] = bool(res["P-1_ok"] and res["P-2_ok"] and res["P-3_ok"]
                         and res["determinism_ok"] and res["full_row_norm_dev_max"] < 1e-12)

    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("★ 探针结论：" + ("全部成立" if res["ok_all"] else "存在问题（见上）"))
    return 0 if res["ok_all"] else 1


if __name__ == "__main__":
    sys.exit(main())