#!/opt/pytorch/bin/python
r"""DIR-34 · **复现 rev 1.0 的首跑失败**（留档用）。

rev 1.0 的 Stage A 首跑未通过（A34-6 的「实例 max ≤ 0.90」在 400 实例上为 0.9951）。
首跑的输出文件被 rev 1.1 的重跑覆盖 ⇒ 本脚本**按 rev 1.0 的原始条款**重算并写回
`out/DIR34r10_stageA_FAILED_rev10.json`，使失败轮**可复现**（而不是抄写控制台数字）。
断言的其余部分与 rev 1.1 相同（同源 `R.stage_a()`），只有第二条子句回到「实例 max」。
"""
import json
import os
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir34")
import R1_task33 as R
import R1_task34 as D

OUT = "/home/ec2-user/sgmn-loop/dir34/out"
ok33, a33 = R.stage_a()
a = {}
a["A34-1_write_block_bitwise"] = a33["A33-1"]
a["A34-2_align_min"] = a33["A33-2_align_min"]
a["A34-2_sep_absmax"] = a33["A33-2_sep_absmax"]
a["A34-3_magnitude_match"] = a33["A33-3"]
a["A34-4_determinism"] = a33["A33-4"]
a["A34-5_G0_no_injection_rel"] = a33["A33-5"]
a["A34-8_worked_example"] = a33["A33-8"]
per_inst = []
for sd in D.EVAL:
    P = R.build(sd, "V0")
    R.period2(P, sd)
    g = R.targets(P)
    cs = [abs(float((g[i] * g[j]).sum() / (g[i].norm() * g[j].norm())))
          for i in range(len(g)) for j in range(len(g)) if (i - j) % 2 == 1]
    per_inst.append(max(cs))
a["A34-6_inst_mean"] = statistics.fmean(per_inst)
a["A34-6_inst_max"] = max(per_inst)          # ★ rev 1.0 的原始条款（极值）
ok = (a["A34-1_write_block_bitwise"] == 0.0 and a["A34-2_align_min"] >= 1 - 1e-14
      and a["A34-3_magnitude_match"] <= 1e-15 and a["A34-4_determinism"] == 0.0
      and a["A34-5_G0_no_injection_rel"] <= 1e-12 and a["A34-8_worked_example"] <= 1e-14
      and a["A34-6_inst_mean"] <= 0.60 and a["A34-6_inst_max"] <= 0.90)   # ← rev 1.0 原条款
print(f"rev 1.0 条款复现：{'★ 全部通过' if ok else '★ 有断言失败 —— 与首跑一致'}")
for k, v in a.items():
    print(f"  {k:26s} {v:.3e}")
os.makedirs(OUT, exist_ok=True)
json.dump({"ok": ok, "a": a, "note": "rev 1.0 原始条款（实例 max ≤ 0.90）的复现；失败点 A34-6_inst_max"},
          open(f"{OUT}/DIR34r10_stageA_FAILED_rev10.json", "w"), ensure_ascii=False, indent=1)
