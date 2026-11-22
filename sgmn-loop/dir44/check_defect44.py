#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""check_defect44.py — 核实**探针 d 的缺陷**：`codebook` 臂的训练是否真的用了码本值？

★ 怀疑：`T42.train_reader42` 内部调用 `trace42`（`dir42`），**不读** `probe44.V_MODE`
⇒ ⇒ 两臂的**训练轨迹逐位相同**，只有**测量**用了码本 ⇒ 探针 d 的 `codebook` 臂是
「**device 损失训出来的控制器、在码本损失下测 ρ**」⇒ 混了两个协议。

★ 判据（本脚本）：两臂训后 `Wq1/bq1/g1/Wo1` 的**逐位差** = 0.0 ⇒ 缺陷成立。
"""
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for d in ("dir31", "dir33", "dir37", "dir40", "dir42", "dir43", "dir44"):
    sys.path.insert(0, os.path.join(ROOT, d))

import R1_task42 as T42          # noqa: E402
import probe44 as PR             # noqa: E402
import probe44d as PD            # noqa: E402

SD = PD.PROBE_SEEDS[0]
cb = PD.build_codebook()

PR.V_MODE = "device";  PR.V_CODEBOOK = None
a, _ = T42.train_reader42("learned", SD, steps=60, train_win=PD.TRAIN_WIN, lr=PD.LR)
PR.V_MODE = "codebook"; PR.V_CODEBOOK = cb
b, _ = T42.train_reader42("learned", SD, steps=60, train_win=PD.TRAIN_WIN, lr=PD.LR)

print("== 两臂训后参数的逐位差（应全 0 ⇒ 缺陷成立）==")
mx = 0.0
for k in a:
    d = float((a[k] - b[k]).abs().max())
    mx = max(mx, d)
    print(f"  {k:5s} max|Δ| = {d:.3e}")
print(f"\n结论：max|Δ| = {mx:.3e}  ⇒ " + ("**缺陷成立**：训练未用码本值" if mx == 0.0 else "训练确实不同"))
