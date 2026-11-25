"""P135-7 前瞻性验证探针 —— 纯本地、只读仓库、不新增任何卡片。

用 **真实的** check_cards.py 判据函数，对**候选条款文本**求四条读数：
  R-A  该行落在职能容器段内（`_u4_sec_line_ranges(text, _u4_sec_new())`）
  R-D  句型（`_u4_form`）= POS / NEG / BOTH / NONE
  R-C  含冻结轴值（`_U4_AX`）
  R-B  引号对上可检索归档对象（`any(_U4_ARCH.search(q) for q in _U4_Q.findall(t))`）

★★ 关键判据：一条条款只有**被 `_misread_hits` 检出**才算进 U4 族总体。
   若纯 POS 条款检不出 ⇒ `R-D` 的 `POS` 读数是**结构性 0**，不是「写不出来」。
"""
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".ops", "scripts"))
import check_cards as cc

print("=" * 96)
print("P135-7 前瞻性探针 · 用真实判据函数测「四条读数」在新条款上能否写出")
print("=" * 96)

# ---------- 0 · 两张词表的包含关系（结构性 0 的根因） ----------
print("\n[0] 两张「禁止」词表的包含关系")
print("    MISREAD_PROHIB（用于**检出**条款）= %r" % (cc.MISREAD_PROHIB,))
print("    _U4_NEG.pattern（用于**归类**句型）= %s" % cc._U4_NEG.pattern)
missing = [w for w in cc.MISREAD_PROHIB if not cc._U4_NEG.search(w)]
print("    ⇒ 检出词里不被 _U4_NEG 命中的 = %r  （空集 ⇒ 检出的条款必为 NEG 或 BOTH）" % (missing,))
print("    ⇒ 结构推论：**POS-only 在总体里不可能出现**")

# ---------- 1 · 真实的误读动词表 ----------
print("\n[1] MISREAD_VERBS = %r" % (cc.MISREAD_VERBS,))

# ---------- 2 · 候选条款 ----------
# 每项 = (标签, 段标题, 条款正文)
CANDS = [
    ("C1 自然禁令（引误读断言）",
     "## §6 边界",
     "- **不得**把 `ρ` 的池化读数读成「记忆无用」；该读数只覆盖单尺度。"),
    ("C2 自然禁令 + 引归档对象",
     "## §6 边界",
     "- **不得**把 `ρ` 的池化读数读成「记忆无用」（依「D-161」）；该读数只覆盖单尺度。"),
    ("C3 C2 + 冻结轴值",
     "## §6 边界",
     "- **不得**把 `dose=0.29` 下的池化读数读成「记忆无用」（依「D-161」）；该读数只覆盖单尺度。"),
    ("C4 纯 POS（无禁止语）",
     "## §6 边界",
     "- 本卡只测 `§7.2` 行 3 在 `dose=0.29` 的单一尺度；以此为该卡的界。"),
    ("C5 POS + 禁止语",
     "## §6 边界",
     "- **不得**外推；本卡只测 `dose=0.29` 的单一尺度。"),
    ("C6 四条齐备（边界段 + POS/BOTH + 轴值 + 归档）",
     "## §6 边界",
     "- **不得**把 `dose=0.29` 下的池化读数读成「记忆无用」（依「D-161」）；本卡只测该单一尺度。"),
    ("C7 四条齐备 · 换段名（判定表）",
     "## §5 判定表",
     "- **不得**把 `dose=0.29` 下的池化读数读成「记忆无用」（依「D-161」）；本卡只测该单一尺度。"),
    ("C8 四条齐备 · 段名不含关键词",
     "## §5 读数",
     "- **不得**把 `dose=0.29` 下的池化读数读成「记忆无用」（依「D-161」）；本卡只测该单一尺度。"),
    ("C9 四条齐备 · 假说段（故意不收）",
     "## §1 假说",
     "- **不得**把 `dose=0.29` 下的池化读数读成「记忆无用」（依「D-161」）；本卡只测该单一尺度。"),
    ("C10 四条齐备 · 一级标题作段",
     "# §6 边界",
     "- **不得**把 `dose=0.29` 下的池化读数读成「记忆无用」（依「D-161」）；本卡只测该单一尺度。"),
]

def readings(heading, clause):
    """在 (heading, clause) 构造的文档上求四条读数。"""
    doc = "DIRECTION 测试卡\n\n" + heading + "\n\n" + clause + "\n"
    lines = doc.split("\n")
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("- "))
    hits = cc._misread_hits(clause)
    in_sec = (idx in cc._u4_sec_line_ranges(doc, cc._u4_sec_new()))
    in_alt = (idx in cc._u4_sec_line_ranges(doc, cc._U4_SEC_ALT))
    pos_both = cc._u4_form(clause) in ("POS", "BOTH")
    pos_only = cc._u4_form(clause) == "POS"
    ax = bool(cc._U4_AX.search(clause))
    arch = any(cc._U4_ARCH.search(m) for m in cc._U4_Q.findall(clause))
    return dict(detected=len(hits) > 0, form=cc._u4_form(clause),
                RA=in_sec, RA_alt=in_alt, RB=arch, RC=ax,
                RD_pos=pos_only, RD_posboth=pos_both)

# ---------- 3 · 逐条读数 ----------
print("\n[2] 逐条读数（RD 两口径：pos_only 严格 / pos_both 宽）")
hdr = "%-46s %-9s %-5s %-4s %-4s %-4s %-5s %-8s" % (
    "候选", "检出", "句型", "R-A", "R-A'", "R-C", "R-B", "R-D(pos)")
print(hdr)
print("-" * 96)
res = {}
for label, heading, clause in CANDS:
    r = readings(heading, clause)
    res[label] = r
    print("%-46s %-9s %-5s %-4s %-4s %-4s %-5s %-8s" % (
        label, "✓" if r["detected"] else "✗(不可见)", r["form"],
        "✓" if r["RA"] else "✗", "✓" if r["RA_alt"] else "✗",
        "✓" if r["RC"] else "✗", "✓" if r["RB"] else "✗",
        "✓" if r["RD_pos"] else "✗"))

# ---------- 4 · 汇总：四条各自的可达性 ----------
print("\n[3] 汇总")
det = [k for k, r in res.items() if r["detected"]]
undet = [k for k, r in res.items() if not r["detected"]]
print("    被检出的候选 = %d / %d ；检不出的 = %r" % (len(det), len(CANDS), undet))
print("    ★ 检不出的那些**不进 U4 族总体** ⇒ 对它们谈四条读数没有意义")
allfour = [k for k, r in res.items() if r["detected"] and r["RA"] and r["RB"] and r["RC"] and r["RD_posboth"]]
allfour_strict = [k for k, r in res.items() if r["detected"] and r["RA"] and r["RB"] and r["RC"] and r["RD_pos"]]
print("    四条齐备（RD 用**宽**口径 POS|BOTH）= %d 条：%r" % (len(allfour), allfour))
print("    四条齐备（RD 用**严**口径 POS-only）= %d 条：%r" % (len(allfour_strict), allfour_strict))

# ---------- 5 · 结构性的证明：任何被检出条款都不可能是 POS-only ----------
print("\n[4] 「R-D 严格口径 = 结构性 0」的证明")
viol = []
for w in cc.MISREAD_PROHIB:
    if not cc._U4_NEG.search(w):
        viol.append(w)
print("    检出必须含 %r 之一；这些词全部都命中 _U4_NEG ⇒ form ∈ {NEG, BOTH}" % (cc.MISREAD_PROHIB,))
print("    反例（含禁止语却是 POS-only）需要：有禁止语 ∧ 不命中 _U4_NEG ⇒ 需 %r 中存在不被 _U4_NEG 命中的词" % "MISREAD_PROHIB")
print("    实测该类词 = %r ⇒ %s" % (viol, "无，故 POS-only 结构性不可达" if not viol else "存在，需复核"))

# ---------- 6 · 语料旁证：现库 55 条里 POS-only 的条数 ----------
print("\n[5] 现库旁证（R-D 的三口径，供对照 P135-4 §4-3 的 0/55 与 12/55）")
try:
    rc = cc._u4_recount("legacy")
    print("    n_clause = %d" % rc["n_clause"])
    print("    RD_pos_only    = %d / %d" % rc["RD_pos_only"])
    print("    RD_pos_and_both= %d / %d" % rc["RD_pos_and_both"])
    print("    RB_new         = %d / %d" % rc["RB_new"])
    print("    RC             = %d / %d" % rc["RC"])
    print("    RA_new         = %d / %d" % rc["RA_new"])
except Exception as e:
    print("    （recount 不可用：%r）" % (e,))