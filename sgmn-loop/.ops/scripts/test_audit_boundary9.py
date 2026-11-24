#!/usr/bin/env python3
"""`P134-14b` · 审计项 ⑨（证据行「边界」栏）的**可证伪性**自测。

discipline：本库已多次踩过的坑是「守卫看起来在跑，其实**永远不会红**」
（`⑧e` 之前恒真、`③` 重复编号恒空、`⑥b` 两份过期文件互证恒真）。⇒ 每一个新审计项
都必须在**注入**下实测能红，否则它只是装饰。

做法（**不碰仓库里的任何文件**）：
1. 在 `tempfile.mkdtemp()` 里建一个 ROOT 影子：只把审计**读到的东西**用软链挂进去，
   `evidence/table.md` 复制一份（★ 必须是副本，注入要写它）。
2. 干净副本跑一次 ⇒ 断言 **rc=0**（⑨a–⑨e 全绿）。
3. 逐个注入（每次从干净副本重建）⇒ 断言**出现**对应的 `✗ ⑨x` 行，且退出码非 0：
   - `⑨a` 清空边界栏
   - `⑨b` 边界栏写「无」（新增行规则 3 明令禁止）
   - `⑨c` 写成「见卡片。」（无任何限定词）
   - `⑨d` 标签漂移成「边界（草案）」（含「边界」但不在白名单）
   - `⑨e` 追加一个编号 ≥ `BOUNDARY_FROM` 的主块且**不带**边界栏
4. 打印 `N/N 通过`；任一不符 ⇒ rc=1。

★ 三条不许：不许改真表；不许把影子目录建在仓库内；不许把「注入没红」记成「守卫没问题」
（那正是本脚本存在的理由 —— 首版 ⑨b 就**假通过**过：单元格没剥行末 `|` ⇒ 写「无」也判过）。

用法：python3 test_audit_boundary9.py        # 全绿输出「自测：5 通过 / 0 不符」
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent                     # `.ops/scripts/` → 仓库根
AUDIT = HERE / "audit_consistency.py"
TABLE = ROOT / "evidence" / "table.md"
# ★ 影子 ROOT = **真树除 `evidence/table.md` 外全部软链**。首版只挂了 7 个具名条目 ⇒
#   别的审计项（⑤ md5 要读 `dir*/` 顶层、⑦ 要读 `revolve/`）在影子里先崩 ⇒ rc=1 却
#   **没有 ⑨ 红行** ⇒ 「干净副本」断言无从判断（实测踩到）。⇒ 改为整树软链，只把
#   `evidence/table.md` 换成副本（注入要写它）。★ 软链是只读用，注入只写副本。


def build_shadow(tmp):
    """建影子 ROOT（整树软链 + `table.md` 副本）；返回 (table 路径, 干净表文本)。"""
    sh = pathlib.Path(tmp)
    for src in ROOT.iterdir():
        if src.name == "evidence":
            continue
        (sh / src.name).symlink_to(src)
    (sh / "evidence").mkdir(parents=True, exist_ok=True)
    for sub in (ROOT / "evidence").iterdir():
        if sub.name != "table.md":
            (sh / "evidence" / sub.name).symlink_to(sub)
    clean = TABLE.read_text(encoding="utf-8")
    (sh / "evidence" / "table.md").write_text(clean, encoding="utf-8")
    return sh / "evidence" / "table.md", clean


def run_audit(sh):
    env = {**os.environ, "SGMN_ROOT": str(sh), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, str(AUDIT)], capture_output=True, text=True, env=env)
    red = [l.strip() for l in r.stdout.splitlines() if re.match(r"\s*- ⑨", l)]
    return r.returncode, red, r.stderr


def main():
    if not TABLE.exists():
        print(f"★ 自测中止：找不到 {TABLE}")
        return 2
    clean = TABLE.read_text(encoding="utf-8")
    m = re.search(r"^(\|\s*\*\*边界\*\*\s*\|)(.*)$", clean, re.M)
    if not m:
        print("★ 自测中止：证据表里没有 `| **边界** |` 行可注入")
        return 2

    cases = [
        ("⑨a 清空边界栏", "⑨a", lambda s: s.replace(m.group(0), m.group(1) + "  |", 1)),
        ("⑨b 边界栏写「无」", "⑨b", lambda s: s.replace(m.group(0), m.group(1) + " 无 |", 1)),
        ("⑨c 边界栏写「见卡片。」", "⑨c", lambda s: s.replace(m.group(0), m.group(1) + " 见卡片。 |", 1)),
        ("⑨d 标签漂移成「边界（草案）」", "⑨d",
         lambda s: s.replace(m.group(0), "| **边界（草案）** |" + m.group(2), 1)),
        ("⑨e 新主块缺边界栏", "⑨e",
         # ★ 编号必须 ≥ 审计里的 `BOUNDARY_FROM`；从真表里探出来，不硬编码。
         None),
    ]
    ok = bad = 0
    tmp = tempfile.mkdtemp(prefix="audit9_")
    try:
        tb, _ = build_shadow(tmp)
        rc, red, err = run_audit(pathlib.Path(tmp))
        if rc == 0 and not red:
            ok += 1
            print("  ✓ 干净副本：rc=0，⑨ 全绿")
        else:
            bad += 1
            print(f"  ✗ 干净副本就红了（rc={rc}）：{red or (err or '').strip()[-160:]}")
        for tag, want, mut in cases:
            if mut is None:                     # ⑨e：读出生效起点再注入
                src = pathlib.Path(p := str(AUDIT)).read_text(encoding="utf-8")
                mm = re.search(r"^BOUNDARY_FROM\s*=\s*(\d+)", src, re.M)
                if not mm:
                    bad += 1
                    print("  ✗ ⑨e 自测找不到 BOUNDARY_FROM 定义")
                    continue
                n = int(mm.group(1))
                mut = (lambda k: (lambda s: s + (
                    f"\n### **D-{k}** · DIR-999：注入（自测）\n\n"
                    "| **项** | **内容** |\n|---|---|\n| **问题** | x |\n")))(n)
            tb.write_text(mut(clean), encoding="utf-8")
            rc, red, err = run_audit(pathlib.Path(tmp))
            hit = [x for x in red if x.startswith(f"- {want}")]
            if hit and rc != 0:
                ok += 1
                print(f"  ✓ {tag}：实测能红（{hit[0][:78]}…）")
            else:
                bad += 1
                print(f"  ✗ {tag}：注入后**没红**（rc={rc}，⑨红={red}）"
                      " ⇒ 该子项是装饰，不是守卫")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n自测：{ok} 通过 / {bad} 不符")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
