"""P135-7 探针 2 —— ① POS-only 到底可不可达（换禁止语）② 0/55 的机制（逐词普查）。"""
import sys, os, re
sys.path.insert(0, os.path.join(os.getcwd(), ".ops", "scripts"))
import check_cards as cc

print("=" * 96)
print("P135-7 探针 2 · POS-only 可达性 + `R-D` 的 0/55 机制")
print("=" * 96)

# ---------- A · 六个禁止语与 _U4_NEG 的关系 ----------
print("\n[A] 六个检出用禁止语 是否命中 `_U4_NEG`（= 一旦用它就必为 NEG/BOTH）")
for w in cc.MISREAD_PROHIB:
    hit = bool(cc._U4_NEG.search(w))
    print("    %-4s  命中 _U4_NEG = %-5s  ⇒ 用它的条款句型 %s" % (
        w, "是" if hit else "否", "必为 NEG/BOTH" if hit else "可为 POS-only"))

# ---------- B · POS-only 是否真可达（用 严禁 + 避开 未/非/无） ----------
print("\n[B] 用「严禁」写一条 POS-only 的四条齐备条款")
CASES = [
    ("C11 严禁 + 归档 + 轴值 + 本卡只测（避开 未/非/无）",
     "## §6 边界",
     "- **严禁**把 `dose=0.29` 的池化读数外推为「恒等映射」（依「D-161」）；本卡只测该单一尺度。"),
    ("C12 同上，但引语写成「记忆无用」（引入 无）",
     "## §6 边界",
     "- **严禁**把 `dose=0.29` 的池化读数外推为「记忆无用」（依「D-161」）；本卡只测该单一尺度。"),
    ("C13 同上，去掉 POS 短语（应落回 NONE/NEG）",
     "## §6 边界",
     "- **严禁**把 `dose=0.29` 的池化读数外推为「恒等映射」（依「D-161」）。"),
]
for label, heading, clause in CASES:
    doc = "DIRECTION 测试卡\n\n" + heading + "\n\n" + clause + "\n"
    lines = doc.split("\n")
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("- "))
    hits = cc._misread_hits(clause)
    ra = idx in cc._u4_sec_line_ranges(doc, cc._u4_sec_new())
    rb = any(cc._U4_ARCH.search(m) for m in cc._U4_Q.findall(clause))
    rc = bool(cc._U4_AX.search(clause))
    form = cc._u4_form(clause)
    print("    %-56s 检出=%s 句型=%-4s R-A=%s R-B=%s R-C=%s" % (
        label, "✓" if hits else "✗", form,
        "✓" if ra else "✗", "✓" if rb else "✗", "✓" if rc else "✗"))
    print("         NEG 词命中 = %r" % ([w for w in cc._U4_NEG.findall(
        cc._de_emph(clause))][:8],))

# ---------- C · 0/55 的机制：55 条里各禁止语用了多少 ----------
print("\n[C] 现库 55 条条款：逐词普查（这是 `R-D`=0/55 的机制，不是比例）")
fam = cc._u4_family_rows("legacy")
cl = [r for r in fam if r["clause"]]
print("    n_clause = %d" % len(cl))
cnt = {w: 0 for w in cc.MISREAD_PROHIB}
for r in cl:
    t = cc._de_emph(r["text"])
    for w in cc.MISREAD_PROHIB:
        if w in t:
            cnt[w] += 1
for w, n in cnt.items():
    print("    含 %-4s 的条款 = %2d / %d %s" % (
        w, n, len(cl), "（∈_U4_NEG ⇒ 必非 POS-only）" if cc._U4_NEG.search(w) else "（∉_U4_NEG ⇒ 可 POS-only）"))

only_hard = sum(1 for r in cl
                if any(w in cc._de_emph(r["text"]) for w in ("不得", "不可", "不能")))
only_soft = sum(1 for r in cl
                if any(w in cc._de_emph(r["text"]) for w in ("禁止", "不许", "严禁")))
print("    ⇒ 用了「不得/不可/不能」的 = %d ；只用了「禁止/不许/严禁」的 = %d" % (only_hard, only_soft))

# ---------- D · BOTH 的 12 条里，NEG 侧到底来自哪里 ----------
print("\n[D] 句型为 BOTH 的条款：NEG 侧来自真禁止语，还是来自 未/非/无 这类普通字？")
both = [r for r in cl if cc._u4_form(r["text"]) == "BOTH"]
print("    BOTH 条款数 = %d" % len(both))
for r in both:
    t = cc._de_emph(r["text"])
    hard = [w for w in ("不得", "不可", "不能") if w in t]
    soft = [w for w in ("禁止", "不许", "严禁") if w in t]
    loose = [w for w in ("未", "非", "无") if w in t]
    pos = [w for w in ("只测", "只覆盖", "只对", "仅在", "只在", "本卡只", "仅测", "仅限") if w in t]
    print("    DIR-%-3d:%-4d 硬禁止=%-8s 软禁止=%-6s 松散(未/非/无)=%-8s POS=%-10s" % (
        r["card"], r["line"], ",".join(hard) or "-", ",".join(soft) or "-",
        ",".join(loose) or "-", ",".join(pos) or "-"))