#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-42 · **冻结守卫自检**（`_card_guard`）

★ **这不是实验**：不导入装置（`dir31`/`dir33`）、不训练、不产生任何实验读数。
  它只把 `R1_task42.py` 里的 `_md5` / `_card_guard` 的**源码本体**抽出来，
  用**临时卡片文件**检验两条拒绝路径与一条放行路径，并做**结构性断言**
  （守卫是否真的挂在 `formal_arm` 与 `merge` 的入口上）。

⇒ ⇒ 为什么要有这个：**纪律「冻结后阈值不得改」不能只靠 Stage A 守。**
  `dir42/run_formal42.sh` 是**可独立调用**的脚本 ⇒ 若守卫只在 Stage A，
  就存在「绕过 Stage A 直接跑正式轮」的路�� ⇒ ⇒ 守卫必须**下沉**到
  `formal_arm` 与 `merge`（能产出判据读数的两个入口）。

用法：  python3 dir42/verify_freeze42.py
退出码：0 = 全部通过；1 = 有失败。
"""
import ast
import hashlib
import os
import pathlib
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "R1_task42.py")


def load_guard(card_path, expected):
    """抽出 `_md5` / `_card_guard` 的**源码本体**（不是复制品）"""
    src = open(SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    segs = [ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("_md5", "_card_guard")]
    if len(segs) != 2:
        raise SystemExit(f"★ 在 {SRC} 里只找到 {len(segs)}/2 个目标函数")
    ns = {"hashlib": hashlib, "pathlib": pathlib,
          "CARD_PATH": card_path, "CARD_MD5_EXPECTED": expected}
    exec("\n\n".join(segs), ns)
    return ns["_card_guard"]


def calls_in(tree, fname, callee):
    """某函数体内对 `callee` 的调用次数"""
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == fname:
            return sum(1 for c in ast.walk(n)
                       if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                       and c.func.id == callee)
    return -1  # 函数不存在


def main():
    fails = []

    def check(cond, msg):
        print(("  ✓ " if cond else "  ✗ ") + msg)
        if not cond:
            fails.append(msg)

    tmp = tempfile.mkdtemp()
    card = os.path.join(tmp, "CARD.md")
    with open(card, "w", encoding="utf-8") as f:
        f.write("# 合成卡片\n阈值 3.28\n")
    real = hashlib.md5(open(card, "rb").read()).hexdigest()

    print("① 拒绝路径 A：卡片**尚未冻结**（占位符）")
    g = load_guard(card, "__CARD_MD5__")
    try:
        g("formal test")
        check(False, "占位符时应当拒绝")
    except SystemExit as e:
        check("尚未冻结" in str(e), f"占位符时拒绝并说明原因：{str(e)[:60]}…")

    print("\n② 放行路径：md5 与冻结值一致")
    g = load_guard(card, real)
    try:
        got = g("formal test")
        check(got == real, f"md5 一致 ⇒ 放行（返回 {got[:8]}…）")
    except SystemExit as e:
        check(False, f"md5 一致时不应拒绝，却得到：{e}")

    print("\n③ 拒绝路径 B：卡片在冻结后被改过")
    with open(card, "a", encoding="utf-8") as f:
        f.write("阈值被改成 0.01\n")
    g = load_guard(card, real)
    try:
        g("merge test")
        check(False, "卡被改过时应当拒绝")
    except SystemExit as e:
        check("md5 不符" in str(e), f"卡被改过时拒绝并说明原因：{str(e)[:60]}…")

    print("\n④ 结构性断言：守卫挂在**能产出判据读数**的入口上")
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    n_formal = calls_in(tree, "formal_arm", "_card_guard")
    n_merge = calls_in(tree, "merge", "_card_guard")
    n_stage = calls_in(tree, "stage_a", "_card_guard")
    check(n_formal == 1, f"`formal_arm` 内调用 `_card_guard` 恰好 1 次（实测 {n_formal}）")
    check(n_merge == 1, f"`merge` 内调用 `_card_guard` 恰好 1 次（实测 {n_merge}）")
    check(n_stage == 0,
          f"`stage_a` **不**调用守卫（它要能在冻结前**报告** md5；实测 {n_stage}）")

    # `formal` 命令必须真的走到 `formal_arm`（否则守卫不可达）
    reach = False
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare):
            s = ast.dump(n.test)
            body = " ".join(ast.dump(x) for x in n.body)
            if "formal" in s and "formal_arm" in body:
                reach = True
    check(reach, "CLI 的 `formal` 分支确实调用 `formal_arm`（守卫可达）")

    # `run_formal42.sh` 是**可独立调用**的脚本 ⇒ 守卫必须下沉（已在 ④ 验证）
    sh = open(os.path.join(HERE, "run_formal42.sh"), encoding="utf-8").read()
    check("run_all42.sh" not in sh,
          "`run_formal42.sh` 确实可独立调用（⇒ 守卫必须在脚本层而非只在 Stage A）")

    print("\n★ 冻结守卫自检 " + ("全部通过" if not fails else "**有失败**"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
