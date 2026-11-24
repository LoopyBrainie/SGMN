#!/usr/bin/env python3
"""`P134-14d` · 审计项 ⑩（`K27` 判定式 ↔ 源码字符串同一性）的**可证伪性**自测。

discipline：本库反复踩的坑是「守卫看起来在跑，其实**永远不会红**」。⑩ 尤其危险 ——
它首版就跑绿，且两边都由我手写，很容易写成「自己跟自己比」。⇒ 必须证明**两侧任一处漂移都能红**：
  ① 归档侧改一个字符（`只(?:测` → `只(测`）⇒ 红
  ② 归档侧少一行（删 `R-B`）⇒ 红
  ③ 归档侧整块删除 ⇒ 红
  ④ **源码侧**改一个字符（`条件于` → `条件於`）⇒ 红  ← 这一条才是「单一来源」的真正证明
  ⑤ **源码侧**改定位器常量（默认 `kws` 加一项）⇒ 红

做法（**不碰仓库里任何文件**）：
1. 在 `tempfile.mkdtemp()` 里建 ROOT 影子：除 `evidence/`、`CARD-CHECKLIST.md`、`.ops/` 外全部软链；
   `evidence/*` 与 `.ops/*` 逐个子项软链（审计要读 `table.md`、`revolve/`、`dir*/`）；
   `CARD-CHECKLIST.md` 与 `.ops/scripts/check_cards.py` 放**副本**（注入要写它们）。
2. 干净副本跑一次 ⇒ 断言 `rc=0` 且**没有** ⑩ 红行。
3. 逐个注入（每次从干净副本重建）⇒ 断言出现对应 ⑩ 红行且退出码非 0。
4. 打印 `N/N 通过`；任一不符 ⇒ rc=1。

★ 三条不许：不许改真文件；不许把影子建在仓库内；不许把「注入没红」记成「守卫没问题」。

用法：python3 test_audit_k27judge.py        # 全绿输出「自测：6 通过 / 0 不符」
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # `.ops/scripts/` → 仓库根
AUDIT = HERE / "audit_consistency.py"
CHECKLIST = ROOT / "CARD-CHECKLIST.md"
CHECKCARDS = HERE / "check_cards.py"


def build_shadow(tmp):
    """建影子 ROOT；返回 (清单副本路径, check_cards 副本路径)。"""
    sh = pathlib.Path(tmp)
    keep_copy = {"evidence", "CARD-CHECKLIST.md", ".ops"}
    for src in ROOT.iterdir():
        if src.name not in keep_copy:
            (sh / src.name).symlink_to(src)
    (sh / "evidence").mkdir(parents=True, exist_ok=True)
    for sub in (ROOT / "evidence").iterdir():
        (sh / "evidence" / sub.name).symlink_to(sub)
    (sh / ".ops").mkdir(parents=True, exist_ok=True)
    for sub in (ROOT / ".ops").iterdir():
        if sub.name != "scripts":
            (sh / ".ops" / sub.name).symlink_to(sub)
    (sh / ".ops" / "scripts").mkdir(parents=True, exist_ok=True)
    for sub in (ROOT / ".ops" / "scripts").iterdir():
        dst = sh / ".ops" / "scripts" / sub.name
        if sub.name == "check_cards.py":
            dst.write_text(sub.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            dst.symlink_to(sub)
    cl = sh / "CARD-CHECKLIST.md"
    cl.write_text(CHECKLIST.read_text(encoding="utf-8"), encoding="utf-8")
    return cl, sh / ".ops" / "scripts" / "check_cards.py"


def run_audit(sh):
    env = {**os.environ, "SGMN_ROOT": str(sh), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, str(AUDIT)], capture_output=True, text=True, env=env)
    red = [l.strip() for l in r.stdout.splitlines() if re.match(r"\s*- ⑩", l)]
    return r.returncode, red, r.stderr


def main():
    for p in (CHECKLIST, CHECKCARDS, AUDIT):
        if not p.exists():
            print(f"★ 自测中止：找不到 {p}")
            return 2
    clean_cl = CHECKLIST.read_text(encoding="utf-8")
    clean_cc = CHECKCARDS.read_text(encoding="utf-8")
    # 注入锚点：从**真文件**里探出来，不硬编码（档块/正则会演进）
    m_pos = re.search(r"_U4_POS = re\.compile\((r'[^']*')", clean_cc)
    m_kws = re.search(r"def _u4_sections\(text, kws=\(\'边界\',\)\)", clean_cc)
    if not (m_pos and m_kws):
        print("★ 自测中止：在 check_cards.py 里找不到 `_U4_POS` / `_u4_sections` 注入锚点")
        return 2

    def arch_drop_line(s):
        return s.replace("  R-B: ", "  R-X: ", 1)     # 键名漂移 ⇒ 取不全四行

    def arch_del_block(s):
        m = re.search(r"```k27-judge\n.*?\n\s*```\n", s, re.S)
        return s.replace(m.group(0), "", 1) if m else s

    def src_pos(s):
        return s.replace("条件于", "条件於", 1)        # 源码侧一个字符

    def src_kws(s):
        return s.replace("kws=('边界',)", "kws=('边界', '红线')", 1)

    cases = [
        ("① 归档侧少一个字符（`只(?:` → `只(`）", "⑩ `K27` 判定式须与源码",
         "cl", lambda s: s.replace("只(?:测", "只(测", 1)),
        ("② 归档侧键名漂移（`R-B` → `R-X`）", "⑩ 须能在 `CARD-CHECKLIST.md` 找到",
         "cl", arch_drop_line),
        ("③ 归档侧整块删除", "⑩ 须能在 `CARD-CHECKLIST.md` 找到", "cl", arch_del_block),
        ("④ 源码侧一个字符（`条件于` → `条件於`）", "⑩ `K27` 判定式须与源码",
         "cc", src_pos),
        ("⑤ 源码侧定位器常量（默认 `kws` 加一项）", "⑩ `K27` 判定式须与源码",
         "cc", src_kws),
    ]
    ok = bad = 0
    tmp = tempfile.mkdtemp(prefix="audit10_")
    try:
        cl, cc = build_shadow(tmp)
        rc, red, err = run_audit(pathlib.Path(tmp))
        if rc == 0 and not red:
            ok += 1
            print("  ✓ 干净副本：rc=0，⑩ 全绿")
        else:
            bad += 1
            print(f"  ✗ 干净副本就红了（rc={rc}）：{red or (err or '').strip()[-200:]}")
        for tag, want, which, mut in cases:
            sh = pathlib.Path(tmp)
            f = ("CARD-CHECKLIST.md" if which == "cl" else ".ops/scripts/check_cards.py")
            base = clean_cl if which == "cl" else clean_cc
            (sh / f).write_text(mut(base), encoding="utf-8")
            rc, red, err = run_audit(sh)
            hit = [x for x in red if want in x]
            if hit and rc != 0:
                ok += 1
                print(f"  ✓ {tag}：实测能红（{hit[0][:76]}…）")
            else:
                bad += 1
                print(f"  ✗ {tag}：注入后**没红**（rc={rc}，⑩红={red}） ⇒ 该子项是装饰")
            (sh / f).write_text(base, encoding="utf-8")   # 复原，供下一条用例
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n自测：{ok} 通过 / {bad} 不符")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
