#!/usr/bin/env python3
"""DIR-18 · GPU 正式轮结果的**解析器**（只出数字，不下结论）。

用法：python3 parse_formal.py <formal_results.json> <formal_log>
输出：判据 PASS/FAIL 汇总、H13′/H14 的读数、联合裁决块、以及各臂的关键读数表。
"""
import json
import pathlib
import re
import sys

RES = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
    "/workspace/sgmn-loop/revolve/DIR-18/.ops/formal/R28_results.json")
LOG = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(
    "/workspace/sgmn-loop/revolve/DIR-18/.ops/formal/D18_R4_formal.log")

d = json.load(open(RES))
checks = d.get("checks", [])
arms = d.get("arms", {})
print("=" * 92)
print(f"轮次标识：{d.get('round')}   判据数：{len(checks)}   臂数：{len(arms)}")
sa = [c for c in checks if c["name"].startswith("A")]
sb = [c for c in checks if not c["name"].startswith("A")]
print(f"  Stage A：{sum(1 for c in sa if c['passed'])}/{len(sa)} PASS")
print(f"  Stage B：{sum(1 for c in sb if c['passed'])}/{len(sb)} PASS")
print("=" * 92)
print("\n【Stage B 判据明细】")
for c in sb:
    print(("  ✅ PASS  " if c["passed"] else "  ❌ FAIL  ") + c["name"])
    print(f"           {c['detail']}")

print("\n【联合裁决（脚本输出的预注册规则）】")
if LOG.exists():
    txt = LOG.read_text(encoding="utf-8", errors="ignore")
    i = txt.find("联合裁决")
    if i != -1:
        seg = txt[i - 10: i + 1200]
        j = seg.find("====")
        print(seg[:j + 4] if j != -1 else seg[:1200])
else:
    print("  （日志缺失）")

print("\n【各臂读数】")
hdr = f"{'臂':<32}{'nll':>10}{'hit':>8}{'addr_slot':>11}{'fid_true':>10}{'cos_y_vtrue':>13}{'nll_ctrl':>11}"
print(hdr)
print("-" * len(hdr))
for nm in arms:
    a = arms[nm]
    def f(k, w=10):
        v = a.get(k)
        if v is None:
            return "None".rjust(w)
        try:
            return f"{float(v):.4f}".rjust(w)
        except (TypeError, ValueError):
            return str(v).rjust(w)
    print(f"{nm:<32}{f('nll')}{f('hit',8)}{f('addr_slot',11)}{f('fid_true')}{f('cos_y_vtrue',13)}{f('nll_ctrl',11)}")

if "dir18_verdict" in d:
    print("\n【脚本给出的裁决字段】")
    print(json.dumps(d["dir18_verdict"], ensure_ascii=False, indent=2))
