#!/usr/bin/env python3
"""`SUCCEEDED` 计数的**纯函数层**（`P118-3` / `DEF-118-11`）：让 ④b 能表达「撤回」。

★★ 为什么单独成模块：`audit_consistency.py` 是**直落式**脚本（第一行正文就是 `print`，末尾 `sys.exit`），
   导入它会**立刻跑完全部检查并退出** ⇒ 它自己**无法被单测**。而 `P118-3` 明确要求配一条
   **能断言红**的自测（`DEF-118-17`：自己的检验必须先证明能红），因此把「文本 → 计数」
   这两个纯函数剥到独立模块 —— 这是**为了可测性**而设的真实接缝，不是为抽象而抽象。

★★ 语义（`DEF-112-07` 的语义 + `DEF-118-11` 的修补）：
   `SUCCEEDED` = 「主假设被证实的方向数」。它**单调不减**地随证据行累积
   ⇒ 取所有 `SUCCEEDED = N / M` 提及中的**最大 N** 即为当前值（这条**一个字都不改**）。
   **唯一合法的递减是「撤回」**（例：`DIR-118` 判决后被 `CORR-31` 自我撤回 ⇒ 不再计入）。
   `max()` 在**结构上**看不见递减 ⇒ 故引入 `WITHDRAWN:` 标记，把撤回表达成
   **一列必须逐条对上账的 `−1` 步**。

★★ 为什么标记里要写 `55 -> 54` 而不是只写 `54`：
   若只写 `54`，脚本就是在**复述标记**、没有任何独立推导，写错也不会被发现。
   现在的极性是 —— **标记是待校验的断言**，脚本**自己**算出 `max(提及)` 与逐步的 `−1`，
   三者（推导值 / `from` / `to`）必须对上，否则抛 `CountError` 并让 ④b 报红。
"""
import re

__all__ = ["CountError", "MENTION_RE", "WITHDRAWN_RE", "DENOM_RE",
           "DENOM_SOURCES", "parse_withdrawals", "derive_count",
           "candidate_denominators", "derive_denominator", "parse_denom_declaration",
           "evidence_rows", "counted_directions", "codomain_violations",
           "PARALLEL_REPORT_KEYS"]

#: 计数提及。用 `=` 锚定，避免把附近的无关分数（如 "2/8 seed"）当成计数。
MENTION_RE = re.compile(r"SUCCEEDED\s*=\s*(\d+)\s*/\s*(\d+)")

#: 撤回标记（行首、独占一行）：
#:     WITHDRAWN: D-149 55 -> 54
#: `D-149` 允许带后缀（如 `D-32-ERRATA`）。
WITHDRAWN_RE = re.compile(r"^\s*WITHDRAWN:\s*(D-\d+(?:-\w+)?)\s+(\d+)\s*->\s*(\d+)\s*$",
                          re.M)


#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# `P118-6`：分母（方向总数）也必须能从归档**机械数出**，不能是四处硬写的常量。
#
# ★★ 先做的是**量**，不是选：`candidate_denominators()` 把「所有说得通的口径」
#   **一次全算出来**。分歧必须显示在审计输出里，**不允许**默默挑一个能对上现值的口径。
# ★★ 极性沿用 `P118-3`：`DENOM: source=<口径名> <数值>` 是**待校验的断言**，
#   脚本按**具名口径**自己重算一遍再比 ⇒ 写错、或口径名打错，都会当场报红。
#   只写数值（不写 source）会被拒绝 —— 那就是**复述**，没有独立推导。
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

#: 允许的「方向总数」口径名。★ 加名字 = 加一种定义，必须同时说明语义。
#:   revolve_dirs      —— `revolve/DIR-*/` 目录数（曾开过的方向，含无卡片者）
#:   revolve_direction —— 其中有冻结 `DIRECTION_*.md` 的（有卡片才算方向）
#:   revolve_verdict   —— 其中有 `VERDICT.md` 的（跑到裁决的）
#:   register_table    —— `DIRECTION-REGISTER.md` 状态表行数（★ 只到 DIR-16 后未再维护）
#:   register_basenums —— 登记表**全文**提及的 `DIR-<基号>` 去重（含字母后缀族）
DENOM_SOURCES = ("revolve_dirs", "revolve_direction", "revolve_verdict",
                 "register_table", "register_basenums")

#: 分母来源声明（行首、独占一行）：
#:     DENOM: source=revolve_direction 112
DENOM_RE = re.compile(r"^\s*DENOM:\s*source=(\w+)\s+(\d+)\s*$", re.M)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# `P118-7`（`Q4` 用户裁定第三条第 3 款）：**分子必须与分母共域**。
#
# ★★ 用户原话：「在写成 `54 / 112` 前，必须证明那 54 个 `SUCCEEDED` 都是这 112 个
#   冻结方向的子集。……若它贡献了成功数，却被排除在分母外，那么这个分数仍然不成立。」
#
# ★ 极性：本项**只做能机械判定的那一半** —— 把「哪些证据行声明了 +1」抽出来，
#   要求它们的方向**都有冻结卡片**（可判红）。剩下那一半（历史自由文本行）
#   **如实报告无法机械判定**，并拒绝在没有并列表述时写出比例。
#   ★★ 不假装能算：`counted_directions()` 显式返回 `undecidable` 计数，
#   让审计输出里永久可见「有多少行判不出来」，而不是静默当成 0。
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#

#: 证据行的表格化起点：`| **D-144（`DIR-113`）** |`
_EVIDENCE_ROW = re.compile(r"^\| \*\*(D-\d+(?:-\w+)?)（[`]*DIR-(\d+[A-Za-z-]*)", re.M)
#: 行内标准 `SUCCEEDED` 字段（★ 早期行没有这个字段，属自由文本）
_ROW_SUCCEEDED = re.compile(r"\*\*`SUCCEEDED`\*\*\s*\|([^|]{0,400})")

#: 共域未定时**必须**并列表述、不得组成比例的两个键（用户裁定原文给定的形式）
PARALLEL_REPORT_KEYS = ("SUCCEEDED_RECORDED", "AUDITABLE_FROZEN_DIRECTIONS")


def evidence_rows(tb_text):
    """把证据表切成 `[(D 名, 方向基号, 行正文), …]`（按文档顺序）。"""
    starts = [(m.start(), m.group(1), m.group(2)) for m in _EVIDENCE_ROW.finditer(tb_text)]
    out = []
    for i, (pos, dname, dirno) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(tb_text)
        out.append((dname, dirno, tb_text[pos:end]))
    return out


def counted_directions(tb_text):
    """返回 `(计+1 的 [(D, DIR)], 判不出的 [(D, DIR)])`。

    ★ 判据只有一条：该行有标准 `SUCCEEDED` 字段且含 `（+1）`。
      没有该字段（早期自由文本行）⇒ 归入 undecidable，**不猜**。
    """
    inc, und = [], []
    for dname, dirno, body in evidence_rows(tb_text):
        m = _ROW_SUCCEEDED.search(body)
        if m and ("（+1）" in m.group(1) or "(+1)" in m.group(1)):
            inc.append((dname, dirno))
        else:
            und.append((dname, dirno))
    return inc, und


def codomain_violations(tb_text, has_card):
    """`has_card(dirno) -> bool`；返回「声明了 +1 但该方向无冻结卡片」的 `[(D, DIR)]`。

    ★★ 这是共域检查里**唯一能判红**的那一半。返回空**不等于**共域成立 ——
      只等于「有标准字段的那些行没违规」；调用方必须把 `undecidable` 一并打印。
    """
    inc, _ = counted_directions(tb_text)
    return [(d, n) for d, n in inc if not has_card(n)]


#: 状态表数据行：`| DIR-01 | …`
_REG_TABLE_ROW = re.compile(r"^\|\s*(DIR-[\w-]+)\s*\|", re.M)
#: 登记表全文里任何 `DIR-<n>` 基号
_REG_BASENUM = re.compile(r"DIR-(\d+)")


class CountError(Exception):
    """撤回标记与推导值对不上账（`from` 不接续、或步长不是 −1）；或分母口径不合法。"""


def parse_withdrawals(tb_text):
    """返回 `[(方向名, from, to), …]`，按**文档顺序**（撤回必须是可追溯的时间序）。"""
    return [(m.group(1), int(m.group(2)), int(m.group(3)))
            for m in WITHDRAWN_RE.finditer(tb_text)]


def derive_count(tb_text):
    """从证据表推导当前 `SUCCEEDED` 分子；无提及时返回 `None`。

    ① `cur = max(所有提及的 N)` —— **原语义原样保留**（无撤回标记时行为与旧实现逐位相同）；
    ② 依次施加每条撤回：`from` 必须等于**当前**推导值（否则标记与推导不同源 ⇒ 抛错），
       `to` 必须等于 `from − 1`（一次撤回只撤掉**一个**方向 ⇒ 否则抛错）；
    ③ 返回最终 `cur`。
    """
    hits = [int(m.group(1)) for m in MENTION_RE.finditer(tb_text)]
    if not hits:
        return None
    cur = max(hits)
    for name, frm, to in parse_withdrawals(tb_text):
        if frm != cur:
            raise CountError(
                f"撤回合不上账：标记 {name} 声明从 {frm} 撤回，但推导出的当前值是 {cur}"
                f"（标记与推导必须同源；请核对标记的 from 是否写错，或中间是否漏了一条撤回）")
        if to != frm - 1:
            raise CountError(
                f"一次撤回必须恰好 −1：标记 {name} 写的是 {frm} -> {to}，步长 {to - frm}")
        cur = to
    return cur


def candidate_denominators(register_text, dir_entries):
    """把所有**语义上说得通**的分母口径一次算出来 ⇒ `{口径名: 计数}`。

    纯函数：只吃文本与列表，不碰文件系统 ⇒ 可单测。

    `dir_entries` 形如 `[{"name": "DIR-98", "has_direction": True,
    "has_verdict": True}, …]`，由调用方 glob 出来。
    """
    names = [e["name"] for e in dir_entries]
    return {
        "revolve_dirs": len(names),
        "revolve_direction": sum(1 for e in dir_entries if e["has_direction"]),
        "revolve_verdict": sum(1 for e in dir_entries if e["has_verdict"]),
        "register_table": len(_REG_TABLE_ROW.findall(register_text)),
        "register_basenums": len({m for m in _REG_BASENUM.findall(register_text)}),
    }


def derive_denominator(register_text, dir_entries, source):
    """按**具名口径**推导分母；口径名不合法则抛 `CountError`。

    ★ 不提供「默认口径」—— 默认值会让分歧重新隐身。
    """
    if source not in DENOM_SOURCES:
        raise CountError(
            f"未知的分母口径 {source!r}；允许的口径 = {list(DENOM_SOURCES)}。"
            f"★ 加口径必须同时写清它的语义，否则分歧会以「名字不同、数不同」的形式隐身。")
    return candidate_denominators(register_text, dir_entries)[source]


def parse_denom_declaration(text):
    """解析 `DENOM: source=<口径> <数值>`；无声明返回 `None`。

    多于一条声明 ⇒ 抛 `CountError`（分母只能有一个权威来源）。
    """
    hits = DENOM_RE.findall(text)
    if not hits:
        return None
    if len(hits) > 1:
        raise CountError(f"分母来源声明出现 {len(hits)} 条，只允许一条：{hits}")
    src, val = hits[0]
    return src, int(val)