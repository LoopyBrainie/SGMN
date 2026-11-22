#!/usr/bin/env python3
"""把 CPU 降级轮的判据 + 臂指标写入 DIR-18 裁决书的 §2.3（占位符 → 实测值）。

用法：python3 fill_verdict.py <results.json>
"""
import json, sys, pathlib, re

RES = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   "/workspace/sgmn-loop/revolve/DIR-18/.ops/cpu/R3_cpu_results.json")
V = pathlib.Path("/workspace/sgmn-loop/revolve/DIR-18/VERDICT.md")

d = json.load(open(RES))
arms, checks = d.get("arms", {}), d.get("checks", [])
SA = [c for c in checks if c["name"].startswith(("A1'", "A2'", "A4'"))]
SB = [c for c in checks if not c["name"].startswith(("A1'", "A2'", "A4'"))]

def g(a, k):
    v = a.get(k)
    if v is None:
        return "None"
    if isinstance(v, float) and v != v:
        return "nan"
    return f"{v:.4f}"

lines = []
lines.append(f"**Stage A**：**{sum(1 for c in SA if c['passed'])}/{len(SA)} PASS**，"
             f"FAIL {sum(1 for c in SA if not c['passed'])} 项。")
lines.append("")
lines.append(f"**Stage B（判据 PASS {sum(1 for c in SB if c['passed'])} / FAIL {sum(1 for c in SB if not c['passed'])}）**：")
lines.append("")
lines.append("| 判定 | 判据 | 读数 |")
lines.append("|---|---|---|")
for c in SB:
    mark = "✅ PASS" if c["passed"] else "❌ FAIL"
    lines.append(f"| {mark} | {c['name']} | {c['detail']} |")
lines.append("")
lines.append("**臂指标（降级轮，n_train=40 / n_eval=12 / 2 种子）**：")
lines.append("")
lines.append("| 臂 | `nll` | `hit` | `addr`（写入序） | `addr_slot`（槽空间） | `fid` | `fid_true` | `cos_y_vtrue` | `nll_ctrl` |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for name, a in arms.items():
    lines.append(f"| `{name}` | {g(a,'nll')} | {g(a,'hit')} | {g(a,'addr')} | {g(a,'addr_slot')} | "
                 f"{g(a,'fid')} | {g(a,'fid_true')} | {g(a,'cos_y_vtrue')} | {g(a,'nll_ctrl')} |")
lines.append("")
lines.append("> ★ 读法提醒：本轮为**降级轮**（算力 CPU 2 核/3 GB、规模 40/12/2）；"
             "**判据与阈值一字未改** ⇒ 可判定「效应方向与冻结判据是否一致」；"
             "**不能**替代 GPU 正式轮的阈值与统计功效判定 ⇒ **`SUCCEEDED` 计数不因本轮增加**。")

block = "\n".join(lines)
t = V.read_text(encoding="utf-8")
marker = "### 2.3 降级轮的任务层读数\n\n**（待填：15 臂 × 2 种子）**"
if marker in t:
    t = t.replace(marker, "### 2.3 降级轮的任务层读数\n\n" + block)
    V.write_text(t, encoding="utf-8")
    print(f"✓ VERDICT.md §2.3 已填入（{len(block)} 字符）")
else:
    print("★ 未找到占位符 ⇒ 跳过（VERDICT.md 可能已被更新）")
