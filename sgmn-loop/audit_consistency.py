#!/usr/bin/env python3
"""SGMN Loop · 一致性审计（纪律 ㉜ / #32 的落地脚本）

检查项：
 ① 纪律编号连续无缺号（圆圈编号 ①–㉝ 与平铺编号两种写法）
 ② 同一条纪律在「活文档」与「冻结阶段报告」里都能搜到
 ③ 证据表编号无重复（只增行、不改行）
 ④ 各文件的 SUCCEEDED 计数与方向数一致
 ⑤ 关键 md5 记录一致（装置 / 脚本）

用法：python3 audit_consistency.py
"""
import re, pathlib, hashlib, sys, os

# ★ 路径可移植（2026-11-07）：原实现硬编码 `/workspace/sgmn-loop`（旧沙箱布局）⇒
#   实验卷上运行时报错 ⇒ ⇒ 改为**环境变量优先 + 默认当前工作区**（不改变任何检查逻辑）。
ROOT = pathlib.Path(os.environ.get("SGMN_ROOT", "/home/ec2-user/sgmn-loop"))
LIVE = ROOT / "LOOP-MECHANISM.md"
FROZEN = ROOT / "LOOP_REPORT_DIR04-DIR17.md"
CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝"
fails = []

def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        fails.append(msg)

print("=" * 88)
print("SGMN Loop · 一致性审计")
print("=" * 88)

# ① 纪律编号连续
print("\n① 纪律编号连续性")
for name, f in (("活文档 LOOP-MECHANISM.md", LIVE), ("冻结报告 LOOP_REPORT_DIR04-DIR17.md", FROZEN)):
    if not f.exists():
        check(False, f"{name} 缺失")
        continue
    t = f.read_text(encoding="utf-8")
    found = {i for i, ch in enumerate(CIRC, 1) if ch in t}
    plain = {int(x) for x in re.findall(r"^(\d{1,2})\. \*\*", t, re.M)}
    alln = found | plain
    top = max(alln) if alln else 0
    gaps = [n for n in range(1, top + 1) if n not in alln]
    check(not gaps, f"{name}：编号 {min(alln)}–{top} 连续" + (f"，缺号 {gaps}" if gaps else ""))

# ② 双文件对照
print("\n② 活文档 ↔ 冻结报告 的纪律对照")
if LIVE.exists() and FROZEN.exists():
    lt = LIVE.read_text(encoding="utf-8")
    ft = FROZEN.read_text(encoding="utf-8")
    lf = {i for i, ch in enumerate(CIRC, 1) if ch in lt} | {int(x) for x in re.findall(r"^(\d{1,2})\. \*\*", lt, re.M)}
    ff = {i for i, ch in enumerate(CIRC, 1) if ch in ft} | {int(x) for x in re.findall(r"^(\d{1,2})\. \*\*", ft, re.M)}
    # ★ 冻结报告是 **DIR-04…DIR-17 阶段**的报告 ⇒ 只应承载当时已存在的纪律（≤26）；
    #   DIR-18 新增的 27–32 不要求回填（回填会篡改一份已冻结的报告）。
    only_live = sorted(n for n in (lf - ff) if n <= 26)
    check(not only_live, f"活文档有而冻结报告无的纪律（≤26 范围）：{only_live if only_live else '无'}")
    newer = sorted((lf - ff))
    print(f"      （说明：冻结报告之后的纪律 {newer} 按设计**不回填**到已冻结报告）")

# ③ 证据表编号
print("\n③ 证据表编号唯一性")
tb = ROOT / "evidence" / "table.md"
if tb.exists():
    t = tb.read_text(encoding="utf-8")
    ids = re.findall(r"^\| \*\*(D-\d+(?:-[a-z])?)[（(]", t, re.M)   # 只数**表行**，不计正文提及
    dup = sorted({i for i in ids if ids.count(i) > 1})
    check(not dup, f"证据表条目 {len(ids)} 条 / 唯一 {len(set(ids))} 条；重复：{dup if dup else '无'}")

# ④ SUCCEEDED 计数
print("\n④ SUCCEEDED 计数")
sm = ROOT / "SUMMARY.md"
if sm.exists():
    t = sm.read_text(encoding="utf-8")
    m = re.search(r"SUCCEEDED\s*=\s*(\d+)\s*/\s*(\d+)", t)
    check(bool(m) and m.group(1) == m.group(2), f"SUMMARY.md 头部 SUCCEEDED = {m.group(0) if m else '未找到'}")
    ndir = len(set(re.findall(r"DIR-\d{2}", t)))
    check(ndir >= 18, f"SUMMARY.md 覆盖方向数 = {ndir}")

# ⑤ 关键 md5
print("\n⑤ 关键 md5 记录")
MD5 = {
    # ★ 2026-11-07：路径从旧沙箱的 `revolve/DIR-18/...` 改为**恢复后工作区的真实布局** `dir18/...`
    #   （工作区是**部分**恢复：只带 DIR-15…DIR-20 + 台账；完整历史在交付包内）。
    #   md5 取**证据台账记录的**值 ⇒ 本项校验的是「工作区里的文件 == 台账记录的那一份」。
    "dir18/sgmn_module_hebb.py": "d62f833b91a1e317609e051d69e3caf8",
    "dir18/rounds/R2_write.py": "12dc22f6a5e7819e077fe779b0bddcd1",
    "dir18/rounds/R3_cpu.py": "dd631cf8801887c3bd8c7e6da92243b2",
    "revolve/DIR-20/R1_xsession.py": "d816507a7446d28c4204e65c1bd39e0a",
    "revolve/DIR-20/sgmn_module_hebb_v6.py": "341ab75583773dd85b65d1aaf04377a5",
    "revolve/DIR-20/VERDICT.md": "a0a71f57db12b17f41958f2d985f9e8c",
}
for rel, want in MD5.items():
    p = ROOT / rel
    if p.exists():
        got = hashlib.md5(p.read_bytes()).hexdigest()
        check(got == want, f"{rel} md5={got}" + ("" if got == want else f"（期望 {want}）"))
    else:
        check(False, f"{rel} 缺失")

# ⑥ 纪律计数跨文件一致（新增于纪律 ㉜/#32 之后）
print("\n⑥ 纪律计数跨文件一致")
import re as _re
_cnt = {}
for _f, _pat in (("LOOP-MECHANISM.md", r"合计\s*\*\*(\d+)\s*条"),
                 ("SUMMARY.md", r"纪律\s*(\d+)\s*条")):
    _t = ROOT / _f
    if _t.exists():
        _m = _re.search(_pat, _t.read_text(encoding="utf-8"))
        if _m:
            _cnt[_f] = int(_m.group(1))
_ok = len(set(_cnt.values())) == 1 and bool(_cnt)
check(_ok, f"纪律计数跨文件一致：{_cnt if _cnt else '（未读到）'}")

print("\n" + "=" * 88)
print(("★ 全部通过" if not fails else f"★ {len(fails)} 项未通过"))
for m in fails:
    print("   - " + m)
sys.exit(0 if not fails else 1)
