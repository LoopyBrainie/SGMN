#!/usr/bin/env python3
"""DIR-18 · CPU 降级轮结果 → 把 D-25 的「② CPU 降级轮」段落写入 evidence/table.md。

用法：python3 finalize_d25.py <results.json>
★ 证据表**只增行、不改行** ⇒ 本脚本**不修改**已有的 D-25 草案行，
  而是在其**之后追加**一个「D-25-b（DIR-18 · CPU 降级轮的任务层读数）」新行。
"""
import json, sys, pathlib, re

RES = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   "/workspace/sgmn-loop/revolve/DIR-18/.ops/cpu/R3_cpu_results.json")
TABLE = pathlib.Path("/workspace/sgmn-loop/evidence/table.md")

d = json.load(open(RES))
arms, checks = d.get("arms", {}), d.get("checks", [])
SA = [c for c in checks if c["name"].startswith(("A1'", "A2'", "A4'"))]
SB = [c for c in checks if not c["name"].startswith(("A1'", "A2'", "A4'"))]

# —— 臂指标表 ——
def g(a, k):
    v = a.get(k)
    if v is None:
        return "None"
    if isinstance(v, float) and v != v:
        return "nan"
    return f"{v:.4f}"

arm_tbl = []
arm_tbl.append("| 臂 | `nll` | `hit` | `addr`（写入序） | `addr_slot`（槽空间） | `fid` | `fid_true` | `cos_y_vtrue` | `nll_ctrl` |")
arm_tbl.append("|---|---|---|---|---|---|---|---|---|")
for name, a in arms.items():
    arm_tbl.append(f"| `{name}` | {g(a,'nll')} | {g(a,'hit')} | {g(a,'addr')} | {g(a,'addr_slot')} | "
                   f"{g(a,'fid')} | {g(a,'fid_true')} | {g(a,'cos_y_vtrue')} | {g(a,'nll_ctrl')} |")
arm_tbl = "\n".join(arm_tbl)

# —— Stage B 判据 ——
sb = []
for c in SB:
    mark = "✅PASS" if c["passed"] else "❌FAIL"
    sb.append(f"| {mark} | {c['name']} | {c['detail']} |")
sb_tbl = "\n".join(sb)

sa_pass = sum(1 for c in SA if c["passed"])
sa_fail = len(SA) - sa_pass
sb_pass = sum(1 for c in SB if c["passed"])
sb_fail = len(SB) - sb_pass

supported = (
    f"在 **CPU 降级轮**（判据与阈值**一字未改**；算力降级为 CPU 2 核/3 GB，规模降级为 n_train 40 / n_eval 12 / 2 种子；"
    f"装置 `sgmn_module_hebb.py` md5 未变、模型与 dtype（float32）一致、语料 md5 逐位一致）下，"
    f"**Stage A {sa_pass}/{len(SA)} PASS（FAIL {sa_fail}）**、**Stage B 判据 PASS {sb_pass} / FAIL {sb_fail}**；"
    f"核心读数：**①** 跨环境逐位复现（`W0_verbatim` nll/hit/addr/addr_slot/cos_y_vtrue 与 `W3_hebb_full_eta7` 的 "
    f"nll/hit/addr/addr_slot/fid/fid_true/cos_y_vtrue **与 GPU 服务器 smoke#2 全部逐位一致**）⇒ "
    f"**降级轮的 13 处受限改动确实不改变任何数值**（md5 校验和、malloc_trim、safetensors mmap 加载、两/三阶段拆分、"
    f"内存卫生与增量落盘**只影响资源，不影响计算**）⇒ ⇒ **CPU 降级轮可作 GPU 正式轮的独立交叉验证**；"
    f"**②** 写规则可用性的**任务层**证据：原尺度（linear）主臂 `fid_true` **−0.0047**、`addr_slot` 0.6667 "
    f"⇒ **写规则几乎没做功**；键尺度归一后 `renorm`（η=1e-1）`fid_true` **+0.6496** / `addr_slot` **1.0000**、"
    f"`ortho`（η=1e-1）`fid_true` **+0.9833** / `addr_slot` **1.0000** / `hit` **1.0000** / `cos_y_vtrue` **0.9833** "
    f"⇒ ⇒ **写 → 寻址 → 读出整链贯通**，且**写规则的可用性不以牺牲寻址为代价**（两臂 `addr_slot` 均为 1.0000）"
    f"⇒ 与 probe18f/g 的装置层结论（ortho η=0.2 ⇒ fid 0.9995；置换对照塌到 +0.0056 ⇒ 非平凡）**方向与量级一致**；"
    f"**③** **H2″（可用配置下写步的任务贡献）**：ortho η=1e-1 下 `W5b`（写步开）nll **11.3328** vs "
    f"`Wnowrite_ortho_eta1`（写步从不执行、地址表照写）nll **14.1736** ⇒ **Δ(Δnll) = +2.8408 nats** "
    f"（冻结阈值 > +0.5 nats；两臂 `addr_slot` 均 1.0000 ⇒ **收益不是靠寻址换来的**）⇒ ⇒ "
    f"**在原尺度配置下做 `nowrite` 对照会完全看不到这个贡献**（linear 对照 Δ 仅 −0.022 nats）⇒ "
    f"**纪律 ㉘（归因对照须与主臂共用「让机制能工作」的配置）** 的直接收益"
)

not_extrapolate = (
    f"**不能外推为「DIR-18 主假说已证实」**：本行的 ② ③ 证明的是**机制链通**与**写步的任务贡献**，"
    f"**任务收益中「记忆内容」与「稀有词先验」各占多少，仍须由交换性对照（判据 H13 / H13b / H13c）判定** "
    f"（probe18e 在原尺度下已给出警示：那里的 +3.17 nats 超额是 **−0.329** ⇒ 几乎全是先验）"
    f"⇒ ⇒ **本行不得被读成「v2.4 写规则已在任务层产生内容驱动的收益」**；"
    f"**不能外推为「正式轮已完成」**：本轮是**降级轮**（算力与规模均降级）⇒ **裁决仍以 GPU 正式轮为准** "
    f"⇒ ⇒ **`SUCCEEDED` 计数不因本行增加**；"
    f"**不能外推为「降级轮的 p 值可与正式轮等价」**：n 减小 ⇒ 功效降低；CPU 与 GPU 的浮点累加顺序不同；"
    f"**不能外推为「寻址指标可与 DIR-17 的 0.8722 直接比较」**：规模不同 ⇒ 槽位回绕模式不同；"
    f"**不能外推为「renorm/ortho 是 v2.4 的现行条款」**：两者都是**对装置的偏离**（v2.4 原文只说 $k=P_kh$，**未规定键的范数标定**）"
    f"⇒ ⇒ 这是**待补的 Freeze B 条款**（「写入键必须逐写归一化到固定范数」），不是既有规定；"
    f"**不能外推为「键必须正交」**：`ortho` 只用于**隔离** DIR-13 已知障碍（键共线 cos 0.987–0.995），"
    f"`renorm`（只修尺度、保留 v2.4 的键方向）已给出可用性 ⇒ ⇒ **必要条款是「尺度归一」，「去相关」是更强的可选加强**"
)

row = (f"\n### D-25-b（DIR-18 · CPU 降级轮的任务层读数）\n\n"
       f"| 编号 | 实际支持的范围 | 不能外推为 |\n|---|---|---|\n"
       f"| **D-25-b（DIR-18 · CPU 降级轮）** | {supported} | {not_extrapolate} |\n\n"
       f"**判据明细（降级轮）**：\n\n"
       f"**Stage A**：{sa_pass}/{len(SA)} PASS，FAIL {sa_fail} 项。\n\n"
       f"**Stage B**：\n\n| 判定 | 判据 | 读数 |\n|---|---|---|\n{sb_tbl}\n\n"
       f"**臂指标（降级轮）**：\n\n{arm_tbl}\n")

s = TABLE.read_text(encoding="utf-8")
if "D-25-b（DIR-18 · CPU 降级轮的任务层读数）" in s:
    print("★ D-25-b 已存在 ⇒ 不重复追加（证据表只增行）")
else:
    s = s.rstrip("\n") + "\n" + row
    TABLE.write_text(s, encoding="utf-8")
    print(f"✓ D-25-b 已追加到 {TABLE}（+{len(row)} 字符）")
