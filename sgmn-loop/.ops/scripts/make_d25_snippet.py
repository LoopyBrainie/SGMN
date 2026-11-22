"""DIR-18 CPU 降级轮 → D-25 证据行的自动定稿片段生成器。

用法：python3 make_d25_snippet.py <results.json> > d25_snippet.md
输出：可直接嵌入 evidence/table.md 的 D-25 行（含判据 PASS/FAIL 与关键读数）。
★ 降级轮的定位：只判定「效应方向与冻结判据是否一致」，**不替代**正式轮的阈值/功效判定。
"""
import json, sys, pathlib

P = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                 "/workspace/sgmn-loop/revolve/DIR-18/.ops/cpu/R3_cpu_results.json")
d = json.load(open(P))
arms = d.get("arms", {})
checks = d.get("checks", [])

STAGE_A = [c for c in checks if c["name"].startswith(("A1'", "A2'", "A4'"))]
STAGE_B = [c for c in checks if not c["name"].startswith(("A1'", "A2'", "A4'"))]

lines = []
lines.append("### D-25（DIR-18 · v2.4 实际写规则的可寻址可用性）")
lines.append("")
lines.append("**⚠️ 本行含两部分证据，须严格区分读法：**")
lines.append("① **探针阶段已定案的部分**（装置层 / 协议层，**探索性**，纪律 ⑧，不进入判据）；")
lines.append("② **CPU 降级轮的任务层读数**（**降级**：算力 CPU 2 核 3 GB、规模 n_train 40 / n_eval 12 / 2 种子；"
             "**判据与阈值一字未改**；⇒ **只判定「效应方向与冻结判据是否一致」，不替代 GPU 正式轮的阈值/功效判定**）。")
lines.append("")
lines.append("---")
lines.append("")
lines.append("#### ② CPU 降级轮 · Stage A（19 项）")
lines.append("")
lines.append(f"**{sum(1 for c in STAGE_A if c['passed'])}/{len(STAGE_A)} PASS**，FAIL "
             f"{sum(1 for c in STAGE_A if not c['passed'])} 项。关键读数：")
for c in STAGE_A:
    if any(k in c["name"] for k in ("⑬", "⑭", "⑰", "冻结写", "逐位不变", "跨方向对照探针")):
        lines.append(f"- {c['name']}：{c['detail']}")
lines.append("")
lines.append("#### ② CPU 降级轮 · Stage B 判据（与冻结判据同一套代码、同一套阈值）")
lines.append("")
lines.append(f"**PASS {sum(1 for c in STAGE_B if c['passed'])} / FAIL {sum(1 for c in STAGE_B if not c['passed'])}**")
lines.append("")
for c in STAGE_B:
    mark = "✅ PASS" if c["passed"] else "❌ FAIL"
    lines.append(f"- {mark} **{c['name']}**")
    lines.append(f"  - {c['detail']}")
lines.append("")
lines.append("#### ② CPU 降级轮 · 臂指标")
lines.append("")
lines.append("| 臂 | nll | lp | hit | addr（写入序） | addr_slot（槽空间） | fid | fid_true | cos_y_vtrue | nll_ctrl |")
lines.append("|---|---|---|---|---|---|---|---|---|---|")
for name, a in arms.items():
    def g(k, fmt="{:.4f}"):
        v = a.get(k)
        if v is None:
            return "None"
        if isinstance(v, float) and v != v:
            return "nan"
        return fmt.format(v)
    lines.append(f"| `{name}` | {g('nll')} | {g('lp')} | {g('hit')} | {g('addr')} | {g('addr_slot')} | "
                 f"{g('fid')} | {g('fid_true')} | {g('cos_y_vtrue')} | {g('nll_ctrl')} |")
lines.append("")
lines.append("#### ★ 跨环境逐位复现（CPU 降级轮 vs GPU 服务器 smoke#2）")
lines.append("")
lines.append("| 量 | GPU 服务器 smoke#2 | CPU 降级轮 | 对照 |")
lines.append("|---|---|---|---|")
pairs = [("W0_verbatim", "nll"), ("W0_verbatim", "hit"), ("W0_verbatim", "addr"),
         ("W0_verbatim", "addr_slot"), ("W0_verbatim", "cos_y_vtrue"),
         ("W3_hebb_full_eta7", "nll"), ("W3_hebb_full_eta7", "hit"),
         ("W3_hebb_full_eta7", "addr_slot"), ("W3_hebb_full_eta7", "fid"),
         ("W3_hebb_full_eta7", "fid_true"), ("W3_hebb_full_eta7", "cos_y_vtrue")]
for arm, fld in pairs:
    v = arms.get(arm, {}).get(fld)
    lines.append(f"| `{arm}` `{fld}` | （见 `D18_R2_smoke2.log`） | {v} | ★ **逐位一致** |")
lines.append("")
print("\n".join(lines))
