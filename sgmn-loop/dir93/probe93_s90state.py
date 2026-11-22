#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-93 · 状态溯源诊断（**不是判据**）—— `storage_diag` 是同一份代码，
   却给出 `0.1999`（`dir90/diag90.py`，写后立刻测）与 `1.6459`（`R1_task90.py:run_arm`，写→**读侧训练**→测）。
   本探针只回答一个问题：**是 `train_read_task` 把被测状态改了吗？**

   四个量，全部同一 `storage_diag` / 同一 `facts` / 同一 `chunks` / 同一 seed：
     s90_a  写后立刻测           —— 预期 0.1999（复现 diag90）
     s90_a2 再测一次（不训练）    —— 检验 storage_diag 自身是否幂等/自污染
     s90_b  读侧训练后测          —— 若 ≈1.6459 ⇒ 元凶是读侧训练
     dW    每层 max|W_after - W_before|  —— 直接量 W 是否被动过
   另记 no_write 标志在训练前后的取值（`train_read_task` 置 True 且**不恢复**）。
"""
import json
import pathlib
import sys

sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "dir90"))
import torch                                     # noqa: E402
import R1_task90 as R                            # noqa: E402  与 diag90.py 逐字同一口径

tok, model = R.model_and_tok()
facts = R.build_facts(tok)
chunks, owner = R.build_chunks(tok, facts)
train_keys = list(range(R.N_TRAIN_KEYS))

out = {"n_facts": len(facts), "n_train": len(train_keys), "eta_write": R.ETA_WRITE,
       "gamma0": R.GAMMA0}

st = R.make_stack(gamma=R.GAMMA0)
R.write_phase(model, st, chunks)
out["nowrite_after_write"] = {li: bool(m.no_write) for li, m in st.mems.items()}

W_before = {li: R.W_of(m).detach().clone() for li, m in st.mems.items()}

s90_a, per_a = R.storage_diag(model, st, facts, owner, chunks)
s90_a2, _ = R.storage_diag(model, st, facts, owner, chunks)
out["s90_a"] = s90_a
out["s90_a2"] = s90_a2
out["s90_idempotent_gap"] = abs(s90_a - s90_a2)
out["nowrite_after_diag"] = {li: bool(m.no_write) for li, m in st.mems.items()}
out["dW_after_diag"] = {li: float((R.W_of(m).detach() - W_before[li]).abs().max())
                        for li, m in st.mems.items()}

# ★ 嫌疑步骤：读侧训练（写冻结）
losses, viol, diverged = R.train_read_task(model, st, tok, facts, train_keys,
                                           __import__("random").Random(1000 + 20261028 % 100))
out["read_train"] = {"viol": viol, "diverged": diverged,
                     "loss_first": losses[0] if losses else None,
                     "loss_last": losses[-1] if losses else None}
out["nowrite_after_readtrain"] = {li: bool(m.no_write) for li, m in st.mems.items()}
out["dW_after_readtrain"] = {li: float((R.W_of(m).detach() - W_before[li]).abs().max())
                             for li, m in st.mems.items()}

s90_b, per_b = R.storage_diag(model, st, facts, owner, chunks)
out["s90_b"] = s90_b

# 逐层对照（per[fi][li]），并给出每层中位数，便于与 diag90 的 device_rel_err_median 比值
out["per_layer"] = {}
for li in W_before:
    out["per_layer"][li] = {
        "med_a": float(sorted(e[li] for e in per_a.values())[len(per_a) // 2]),
        "med_b": float(sorted(e[li] for e in per_b.values())[len(per_b) // 2])}

# W 范数：训练是否只是"衰减"
out["w_norm"] = {li: {"before": float(W_before[li].norm()),
                      "after_readtrain": float(R.W_of(m).detach().norm())}
                 for li, m in st.mems.items()}
st.remove()

p = pathlib.Path(__file__).resolve().parent / "out"
p.mkdir(exist_ok=True)
(p / "probe_s90state.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))