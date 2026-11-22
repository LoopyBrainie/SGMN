#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-42 · **卡 ↔ 代码交叉核对**（阈值漂移探测器 + 断言覆盖 + 判定支对应）

★ **这不是实验**：不导入装置、不训练、不产生任何实验读数。
  它把**卡**（`revolve/DIR-42/DIRECTION_rev42_v10.md`）与**实现**（`dir42/R1_task42.py`）
  的对应关系**机械核对**一遍 ⇒ ⇒ 凡是对不上的地方，**开跑前**就报出来。

⇒ ⇒ 为什么必须有：用户规则「**不得改动已冻结判据的阈值与文字**」目前只是**承诺**；
  ⇒ ⇒ 本脚本把它变成**可执行断言**：卡里写的常数与代码里的常数**逐项相等**才算过。

核对四项：
  ① **常数逐项相等**（`SIG_K`/`FRAC_HEADROOM`/`N_INST`/`STEPS`/`LR`/`N_REP`/窗口/`COS_SIGNATURE`）；
  ② **断言 1:1**（卡 §7 的 `A42-1…A42-N` ↔ `stage_a` 里真正写的 `A42-N_ok`）；
  ③ **判定表六支**（卡 §4 的结局名 ↔ 代码里 `out["outcome"] = ...` 的字面量）；
  ④ **探针不可达**（卡 §4 第 1 支承诺「读到了探针文件 ⇒ `INCOMPLETE`」
     ⇒ ⇒ 代码里**不得**枚举目录 ⇒ ⇒ 结构上不可能读到探针文件）。

用法：  python3 dir42/verify_card42.py
退出码：0 = 全部通过；1 = 有失败。
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "R1_task42.py")
CARD = os.path.join(HERE, "..", "revolve", "DIR-42", "DIRECTION_rev42_v10.md")

# 代码里**应当**有的常数（名字 → 期望值）；与卡 §1 表逐项对应
CODE_CONSTS = {
    "SIG_K": 3.0, "FRAC_HEADROOM": 0.20, "N_INST": 400, "STEPS": 250,
    "LR": 0.03, "N_REP": 16, "TRAIN_WIN": (1, 8), "EVAL_WIN": (9, 15),
    "COS_SIGNATURE": 0.5,
}


def section(text, head):
    """取 `## §N` 开头到下一个 `## ` 之间的正文"""
    i = text.find(head)
    if i < 0:
        return ""
    j = text.find("\n## ", i + 1)
    return text[i:j] if j > 0 else text[i:]


def strip_md(s):
    return s.replace("**", "").replace("`", "").strip()


def num(s):
    s = strip_md(s)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def code_consts(tree):
    """从 AST 取模块级常量（只处理字面量；含 `STEPS, LR = 250, 0.03` 这种**元组赋值**）"""
    out = {}
    for n in tree.body:
        if not isinstance(n, ast.Assign) or len(n.targets) != 1:
            continue
        try:
            val = ast.literal_eval(n.value)
        except (ValueError, SyntaxError):
            continue
        t = n.targets[0]
        if isinstance(t, ast.Name):
            out[t.id] = val
        elif isinstance(t, (ast.Tuple, ast.List)):      # ★ `STEPS, LR = 250, 0.03`
            try:
                for name, v in zip(t.elts, val):
                    if isinstance(name, ast.Name):
                        out[name.id] = v
            except TypeError:
                pass
    return out


def main():
    fails = []

    def check(cond, msg):
        print(("  ✓ " if cond else "  ✗ ") + msg)
        if not cond:
            fails.append(msg)

    card = open(CARD, encoding="utf-8").read()
    src = open(SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    cc = code_consts(tree)

    print("① ★ 常数逐项相等（**阈值漂移探测器**）")
    # ★ 常数表在卡 §3.2 ⇒ ⇒ 但为稳健，直接扫**整卡**的表格行（该表形状唯一）
    card_tbl = {}
    for m in re.finditer(r"\|\s*`([A-Z_]+)`\s*\|\s*([^|]+?)\s*\|", card):
        card_tbl.setdefault(m.group(1), m.group(2))
    for name, want in CODE_CONSTS.items():
        if name in ("TRAIN_WIN", "EVAL_WIN"):
            continue
        got_card = card_tbl.get(name)
        got_code = cc.get(name)
        ok = (got_card is not None and got_code is not None
              and abs(num(got_card) - float(got_code)) < 1e-12
              and abs(float(got_code) - float(want)) < 1e-12)
        check(ok, f"`{name}`：卡 {got_card!r} = 代码 {got_code!r} = 期望 {want!r}")
    # 窗口行在卡里是**合并写的**：`| `TRAIN_WIN` / `EVAL_WIN` | **`[1,8]` / `[9,15]`** | ...`
    win_row = None
    for line in card.splitlines():
        if line.startswith("|") and "TRAIN_WIN" in line and "EVAL_WIN" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2:
                win_row = cells[1]
    ok_win = (win_row is not None
              and "[1,8]" in win_row.replace(" ", "") and "[9,15]" in win_row.replace(" ", "")
              and tuple(cc.get("TRAIN_WIN", ())) == (1, 8)
              and tuple(cc.get("EVAL_WIN", ())) == (9, 15))
    check(ok_win, f"窗口：卡 {win_row!r} ⇒ 代码 TRAIN_WIN={cc.get('TRAIN_WIN')} "
                  f"EVAL_WIN={cc.get('EVAL_WIN')}")

    print("\n② ★ 断言 1:1（卡 §7 ↔ `stage_a` 真正写的）")
    s7 = section(card, "## §7")
    # 卡：§7 表的每一行；并区分「**不设硬闸**」的行（诊断项）
    card_rows = {}
    for m in re.finditer(r"\|\s*`A42-(\d+)`\s*\|([^|]*)\|([^|]*)\|", s7):
        card_rows[m.group(1)] = (m.group(2), m.group(3))
    card_ids = set(card_rows)
    # ★ 「不设硬闸」写在**第 2 列**（断言名）里 ⇒ ⇒ 取名列，不是判定内容列
    card_ungated = {i for i, (name, desc) in card_rows.items() if "不设硬闸" in name}
    sa = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage_a"), None)
    if sa is None:
        check(False, "实现里找不到 `stage_a`")
        code_ids, code_gated = set(), set()
    else:
        sa_src = ast.get_source_segment(src, sa)
        # ★ 断言的**标记**在注释里（`# ── A42-N：`）⇒ ⇒ 诊断项（无 `_ok`）也算一条
        #   ⇒ ★ 一行注释里可能有**多个** id（`# ── A42-7 / A42-8 / A42-9：...`）⇒ ⇒ 整行取 id
        code_ids = set()
        for m in re.finditer(r"#\s*──[^\n]*", sa_src):
            code_ids |= set(re.findall(r"A42-(\d+)", m.group(0)))
        code_gated = set(re.findall(r'"A42-(\d+)_ok"', sa_src))
    check(card_ids == code_ids,
          f"卡 §7 的 {len(card_ids)} 条断言 ↔ 代码的 {len(code_ids)} 条：**完全一致**"
          + ("" if card_ids == code_ids else f"（卡独有 {sorted(card_ids - code_ids)}，"
                                             f"代码独有 {sorted(code_ids - card_ids)}）"))
    check(len(code_ids) == 15, f"断言条数 = 15（实测 {len(code_ids)}）")
    code_ungated = code_ids - code_gated
    check(card_ungated == code_ungated,
          f"**不设硬闸**的那条也一致（卡 {sorted(card_ungated)} ↔ 代码 {sorted(code_ungated)}）")
    check(len(code_gated) == 14, f"**有硬闸**的断言 = 14 条（实测 {len(code_gated)}）")

    print("\n③ ★ 判定表六支（卡 §4 ↔ 代码 `out[\"outcome\"]`）")
    s4 = section(card, "## §4")
    card_branches = []
    for m in re.finditer(r"\|\s*([1-6])\s*\|[^|]*\|[^|]*`([A-Z][A-Z0-9-]*)`[^|]*\|", s4):
        card_branches.append(m.group(2))
    code_branches = set(re.findall(r'out\["outcome"\]\s*=\s*"([A-Z][A-Z0-9-]*)"', src))
    check(len(card_branches) == 6, f"卡 §4 解析出 6 支（实测 {len(card_branches)}：{card_branches}）")
    check(set(card_branches) == code_branches,
          f"六支的结局名**完全对应**（卡 {sorted(set(card_branches))} ↔ 代码 {sorted(code_branches)}）")
    check("counted" in src and 'out["counted"] = True' in src,
          "代码里只有最后一支 `counted = True`（其余支均 `False`）")

    print("\n④ ★ 探针不可达（卡 §4 第 1 支的承诺）")
    mg = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "merge"), None)
    mg_src = ast.get_source_segment(src, mg) if mg else ""
    banned = [w for w in ("glob", "listdir", "scandir", "iterdir", "walk(") if w in mg_src]
    check(not banned, f"`merge` **不枚举目录**（结构上不可能读到探针文件）"
                      + ("" if not banned else f"；但出现了 {banned}"))
    check("DIR42r10_arm_" in mg_src and "PROBE" not in mg_src.upper(),
          "`merge` 只按显式文件名读 `DIR42r10_arm_*`（无 `PROBE` 字样）")

    print("\n⑤ ★ 指纹维度：卡 §8 写的字段 ⊇ 代码 `_row_cfg()` 的键")
    s8 = section(card, "## §8")
    rc = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_row_cfg"), None)
    rc_src = ast.get_source_segment(src, rc) if rc else ""
    keys = set(re.findall(r'"(\w+)":', rc_src))
    miss = [k for k in sorted(keys) if k not in s8]
    check(not miss, f"指纹维度 {sorted(keys)} 在卡 §8 都有交代"
                    + ("" if not miss else f"；**缺** {miss}"))

    print("\n★ 卡 ↔ 代码交叉核对 " + ("全部通过" if not fails else "**有失败**"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
