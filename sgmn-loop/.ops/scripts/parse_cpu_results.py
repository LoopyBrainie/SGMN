"""DIR-18 CPU 降级轮结果解析器：从 R28_smoke_results.json 提取判据表 + 臂指标。

用途：把降级轮的数字整理成可与正式轮对照的表格（供 evidence/table.md 的 D-25 定稿引用）。
★ 降级轮的定位：**只判定「效应方向与冻结判据是否一致」**，不替代正式轮的阈值/功效判定。
"""
import json, sys, pathlib

P = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                 "/workspace/sgmn-loop/revolve/DIR-18/.ops/cpu/R3_cpu_results.json")

d = json.load(open(P))
print("=" * 100)
print(f"round = {d.get('round')}  |  Stage A 检查数 = {len(d.get('checks', []))}")
print("=" * 100)

print("\n【Stage A】")
for c in d.get("checks", []):
    print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}")

print("\n【臂指标汇总】")
print(f"{'臂':28s} {'nll':>9s} {'lp':>9s} {'hit':>7s} {'addr':>7s} {'addr_s':>8s} "
      f"{'fid':>8s} {'fid_t':>8s} {'cos_v':>8s} {'nll_ctrl':>9s}")
for name, a in d.get("arms", {}).items():
    def g(k):
        v = a.get(k)
        return f"{v:9.4f}" if isinstance(v, (int, float)) else f"{'None':>9s}"
    print(f"{name:28s} {g('nll')} {g('lp')} "
          f"{(a.get('hit') if a.get('hit') is not None else float('nan')):7.4f} "
          f"{(a.get('addr') if a.get('addr') is not None else float('nan')):7.4f} "
          f"{(a.get('addr_slot') if a.get('addr_slot') is not None else float('nan')):8.4f} "
          f"{(a.get('fid') if a.get('fid') is not None else float('nan')):8.4f} "
          f"{(a.get('fid_true') if a.get('fid_true') is not None else float('nan')):8.4f} "
          f"{(a.get('cos_y_vtrue') if a.get('cos_y_vtrue') is not None else float('nan')):8.4f} "
          f"{g('nll_ctrl')}")

print("\n【Stage B 判据】")
npass = sum(1 for c in d.get("checks", []) if c["passed"] and not c["name"].startswith(("A1'", "A2'", "A4'")))
nfail = sum(1 for c in d.get("checks", []) if (not c["passed"]) and not c["name"].startswith(("A1'", "A2'", "A4'")))
print(f"  判据项：PASS {npass} / FAIL {nfail}")
for c in d.get("checks", []):
    if c["name"].startswith(("A1'", "A2'", "A4'")):
        continue
    print(f"\n  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}")
    print(f"      {c['detail']}")

if d.get("aborted"):
    print("\n★ aborted:", d["aborted"])
