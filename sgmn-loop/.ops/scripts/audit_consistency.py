#!/usr/bin/env python3
"""SGMN Loop · 一致性审计（纪律 ㉜ / #32 的落地脚本）

检查项：
 ① 纪律编号连续无缺号（圆圈编号 ①–㉝ 与平铺编号两种写法）
 ② 同一条纪律在「活文档」与「冻结阶段报告」里都能搜到
 ③ 证据表编号无重复（只增行、不改行）——★ 同时数**两种**行式（`### **D-n** ·` 与 `| **D-n（…）** |`）
 ④ 各文件的 SUCCEEDED 计数与方向数一致
 ④b 三个头部的 SUCCEEDED 计数**彼此一致**且**等于证据表推导值**（★ 防「头部过期」）
 ⑤ 关键 md5 记录一致（装置 / 脚本）
 ⑧ 开卡清单 CARD-CHECKLIST.md 的格式完整性（条目数/字段/执行方式，声明 vs 机械点数）
 ⑧e–⑧h `判读` 字段与 `check_cards.py` 的 `INTERP` **逐字同一** + 标签合法（`P133-1`/`P134-1`）

用法：python3 audit_consistency.py
"""
import re, pathlib, hashlib, sys, os, ast
import audit_counts as AC   # ★ P118-3：SUCCEEDED 计数的纯函数层（可单测）
from collections import Counter

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
    # ★ DEF-26S-18（DIR-41/DIR-42 已披露）：原正则**只认表行** `| **D-n（…）** |` ⇒ 新式
    #   标题行 `### **D-n** · …` **全部漏数** ⇒ 「证据表条目 26 条」是**欠计**（真值 ≈ 4x）⇒
    #   ⇒ 审计**看不出**编号重复、也看不出缺行 ⇒ ⇒ 现在**两种行式都数**（去重后判重）。
    # ★★ DEF-173-01（Cycle 174 修）：本项此前**恒真**，两处叠加：
    #   (i) 先把两类 id 并成 `set()`，再在**已去重**的列表上 `ids.count(i)` ⇒ 恒为 1 ⇒ `dup` 恒空；
    #   (ii) 标题行正则 `^### \*\*(D-\d+)\*\*` 要求编号后**紧跟 `**`**，而多数真实行写作
    #        `### **D-103 · DIR-82：…`（编号后是 ` · `）⇒ **漏数**，于是同一编号的"两行"只被数到一行。
    #   ⇒ 实际后果：`D-103/104/105/106/108/109` 各有**两条无后缀行**（前一条属 DIR-82/83/84/86/90/91早期，
    #     后一条属 DIR-91 rev 1.9/2.2/2.4/2.5/2.7/2.8）—— 引用这些编号**有歧义**，而审计报了"无重复"。
    #   修法：① 标题行正则容忍 `**` 有无、编号后**任意字符**，后缀允许字母+数字（`-CORR2` 不得退化成 `-CORR`）；
    #         ② 重复计数在**原始标题行列表**上做（去重前）；③ 只对**标题行**判重
    #            （文件顶部的汇总表是**指针索引**，不是证据行，与标题天然同名，不得计入）；
    #         ④ 「已声明的历史冲突」写在 evidence/table.md 的 `DUP-OK:` 标记里（append-only，不改旧行）
    #            ⇒ 守卫对**未来任何新冲突**仍可失败。
    SUF = r"(?:-[A-Za-z]+\d*)?"
    ids_hdr = re.findall(rf"^#{{2,4}}\s+\*{{0,2}}(D-\d+{SUF})", t, re.M)      # 标题行 = 真正的证据行
    ids_tbl = re.findall(rf"^\|\s*\*\*(D-\d+{SUF})", t, re.M)               # 汇总表行 = 指针索引
    _cnt = Counter(ids_hdr)
    dup_all = sorted([k for k, v in _cnt.items() if v > 1],
                     key=lambda x: (int(re.findall(r"\d+", x)[0]), x))
    m_ok = re.search(r"DUP-OK:\s*([-\w,\s]+)", t)
    declared = {x.strip() for x in m_ok.group(1).split(",") if x.strip()} if m_ok else set()
    dup_undeclared = [d for d in dup_all if d not in declared]
    print(f"      （标题行 {len(ids_hdr)} 条 / 唯一 {len(_cnt)} 个编号；汇总表指针 {len(ids_tbl)} 行）")
    print(f"      （编号重复 {len(dup_all)} 个：已声明 {sorted(set(dup_all) & declared) or '无'}）")
    check(not dup_undeclared,
          f"证据表**未声明**的重复编号：{dup_undeclared if dup_undeclared else '无'}"
          f"（每个编号只应有一行；冲突见 evidence/table.md 的 `DUP-OK` 声明与消歧表）")

# ④ SUCCEEDED 计数
print("\n④ SUCCEEDED 计数")
sm = ROOT / "SUMMARY.md"
if sm.exists():
    t = sm.read_text(encoding="utf-8")
    # ★ DEF-96-11（2026-11-20，Cycle 188 修）：原实现用**裸首匹配**读 `SUCCEEDED = N / M`：
    #     m = re.search(r"SUCCEEDED\s*=\s*(\d+)\s*/\s*(\d+)", t)          # ← 旧版，留档备用
    #   问题是 `SUMMARY.md` 是**追加式**日志，首个匹配命中的是**最早的历史快照** `39 / 39`
    #   （实测 @32；而规范标记 `当前计数：SUCCEEDED = 49 / 49` 在 @462）⇒ 打印值误导。
    #   且旧断言只有 `n == d` —— **每个历史快照都满足** ⇒ 缺陷被掩盖（弱检查看不出读到过期值）。
    #   修法：与 §④b 一致，**优先读规范标记**，仅在无标记时退回裸匹配（并如实说明走的是哪条路）。
    m = re.search(r"当前计数：\s*SUCCEEDED\s*=\s*(\d+)\s*/\s*(\d+)", t)
    how = "规范标记"
    if not m:
        m = re.search(r"SUCCEEDED\s*=\s*(\d+)\s*/\s*(\d+)", t)   # 兜底（无标记时）
        how = "裸匹配·兜底（该文件无规范标记）"
    # ★★ `DEF-112-07`（Cycle 209 修，同 §④b 的同一处假设）：原断言 `m.group(1) == m.group(2)`
    #   把「分子==分母」当合法性条件 —— **当分母是方向总数（如 50 / 55）时必然失败**。
    #   真实语义：分子 = 已证实方向数，分母 = 方向总数 ⇒ 只要求 `1 <= 分子 <= 分母`。
    #   ★ 交叉校验（头部是否过期）由 §④b 对**证据表推导值**完成，不在本项重复。
    _ok = bool(m) and 1 <= int(m.group(1)) <= int(m.group(2))
    check(_ok, f"SUMMARY.md 头部 SUCCEEDED = {m.group(0) if m else '未找到'}（读自：{how}；"
               f"须 1 <= 分子 <= 分母）")
    ndir = len(set(re.findall(r"DIR-\d{2}", t)))
    check(ndir >= 18, f"SUMMARY.md 覆盖方向数 = {ndir}")

# ④b ★ 头部计数的一致性（DEF-26S-19：三个头部分别是 23/23、10/10、15/15 ⇒ 与证据表的 32/32 矛盾，
#   而原 ④ 只查「分子 == 分母」⇒ **看不出过期** ⇒ 本项把三者互校、并与证据表推导值对表）
# ★★★ `P118-3`（Cycle 225，`DEF-118-11` 修）：④b 原先**结构上无法表达「撤回」**。
#   旧实现（上面 `DEF-112-07` 那段语义）取「所有提及的最大 N」，理由是 `SUCCEEDED` 单调不减；
#   但 `DIR-118` 判决后被 `CORR-31` **自我撤回**后真实计数是 54，而 `D-149` 行里的 `55` 仍在表中
#   ⇒ 推导值只能读到 55 ⇒ **四头部被迫写 55 才能过检** ⇒ 计数与事实不符。
#   ⇒ 修法：把「文本 → 计数」剥成独立模块 `audit_counts.py`（直落式脚本无法被单测，而本卡要求自测能红）；
#     **`max()` 语义一个字不改**，撤回表达成**必须逐条对上账的 −1 步**：
#         WITHDRAWN: D-149 55 -> 54
#     标记是**待校验的断言**，不是答案：脚本自己算 `max(提及)` 与逐条 −1，
#     `from`/`to` 对不上就抛 `CountError` ⇒ 本项报红（不会静默复述标记）。
#   ★ 自测：`test_audit_counts.py`（8 条 + `--inject` 注入，已实测 5 个位置都能**断言红**）。
if tb.exists():
    try:
        _derived = AC.derive_count(tb.read_text(encoding="utf-8"))
    except AC.CountError as _e:
        _derived = None
        check(False, f"撤回标记与推导值合不上账：{_e}")
    heads = {}
    for rel in ("SUMMARY.md", "LOOP-MECHANISM.md", "DIRECTION-REGISTER.md", "LOOP-STATE.md"):
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        # ★ 优先读**规范标记**（`当前计数：SUCCEEDED = N / M`）⇒ 这三个文件是**追加式**日志，
        #   裸 `SUCCEEDED = N / N` 是**历史快照**（10/10、11/11、…）⇒ 用首个匹配会读到过期值。
        m = re.search(r"当前计数：\s*SUCCEEDED\s*=\s*(\d+)\s*/\s*(\d+)", txt)
        if not m:
            m = re.search(r"SUCCEEDED\s*=\s*(\d+)\s*/\s*(\d+)", txt)   # 兜底（无标记时）
        if m:
            heads[rel] = int(m.group(1))
    print(f"      证据表推导的 SUCCEEDED 计数 = {_derived}；各头部 = {heads}")
    check(len(set(heads.values())) <= 1, f"四个头部彼此一致：{heads}")
    if _derived is not None and heads:
        check(all(v == _derived for v in heads.values()),
              f"各头部 == 证据表推导值 {_derived}")

    # ④c ★★ `P118-6`：分母（方向总数）也必须能**从归档机械数出**。
    #   ④b 只校对了**分子**；分母 `M` 在此之前**从未被任何检查读过** —— `1 <= N <= M`
    #   这条约束对任何足够大的 M 都成立 ⇒ `54 / 56` 里有一半是**手工断言**。
    #   ★★ 本项**不猜口径**：把五个说得通的口径**一次全算出来**并打印，分歧必须现形。
    #   极性：`DENOM: source=<口径> <数值>` 是**待校验的断言**，脚本按具名口径自己重算再比。
    #   未声明时只**报告**不判红 —— 因为「选哪个口径」是**定义性决策**，不是本脚本能定的
    #   （`P118-6` 卡明确：两口径给出不同的数 ⇒ 如实记录分歧，**不得擅自选定**）。
    try:
        _reg = (ROOT / "DIRECTION-REGISTER.md")
        _reg_txt = _reg.read_text(encoding="utf-8") if _reg.exists() else ""
        _entries = []
        _rev = ROOT / "revolve"
        if _rev.exists():
            for _d in sorted(_rev.iterdir()):
                if _d.is_dir() and re.fullmatch(r"DIR-[\w-]+", _d.name):
                    _entries.append({
                        "name": _d.name,
                        "has_direction": any(_d.glob("DIRECTION_*.md")),
                        "has_verdict": (_d / "VERDICT.md").exists(),
                    })
        _cands = AC.candidate_denominators(_reg_txt, _entries)
        print(f"      分母候选口径（全部机械数出）= {_cands}")
        _decl = AC.parse_denom_declaration(tb.read_text(encoding="utf-8"))
        if _decl is None:
            _uniq = set(_cands.values())
            _m_used = re.search(r"当前计数：\s*SUCCEEDED\s*=\s*\d+\s*/\s*(\d+)",
                                (ROOT / "SUMMARY.md").read_text(encoding="utf-8")) \
                     if (ROOT / "SUMMARY.md").exists() else None
            _m_val = int(_m_used.group(1)) if _m_used else None
            print(f"      ⚠ 分母**未声明来源**（无 `DENOM: source=… ` 标记）；"
                  f"四头部写的分母 = {_m_val}")
            print(f"      ⚠ 候选值无一等于 {_m_val} ⇒ 该分母**无归档依据**"
                  if _m_val is not None and _m_val not in _uniq else
                  f"      ✓ 四头部写的分母 {_m_val} 与某个候选口径相等")
            print(f"      ⚠ 候选口径自身也不一致（{len(_uniq)} 个不同取值）⇒ "
                  f"选口径是**定义性决策**，未擅自选定（见 `P118-6`）")
        else:
            _src, _val = _decl
            _got = AC.derive_denominator(_reg_txt, _entries, _src)
            check(_got == _val,
                  f"分母声明与口径重算一致：source={_src} 声明 {_val} / 重算 {_got}")
    except AC.CountError as _e:
        check(False, f"分母校验失败：{_e}")

    # ④d ★★ `P118-7`（用户裁定第 3 条）：**分子必须与分母共域**。
    #   ④c 只保证「分母能被机械重算」，但一个分数 N/M 还要求 N ⊆ M。
    #   用户原话：「若它贡献了成功数，却被排除在分母外，那么这个分数仍然不成立。」
    #   ★ 本项只判**能机械判的那一半**（带标准 `SUCCEEDED` 字段的 +1 行必须有卡片）；
    #     另一半（早期自由文本行）**显式打印判不出的行数**，绝不静默当 0。
    #   ★★ 并强制：共域未定时，档案里**不得**出现比例式，只能并列表述。
    try:
        _tb_txt = tb.read_text(encoding="utf-8")
        _inc, _und = AC.counted_directions(_tb_txt)
        _viol = AC.codomain_violations(
            _tb_txt,
            lambda n: any((ROOT / f"revolve/DIR-{n}").is_dir()
                          and list((ROOT / f"revolve/DIR-{n}").glob("DIRECTION_*.md"))
                          for _ in [0]))
        print(f"      共域可判部分：判 +1 的行 = {len(_inc)}，**判不出**的行 = {len(_und)}"
              f"（早期行无标准 `SUCCEEDED` 字段）")
        check(not _viol,
              f"共域（可判部分）：{len(_inc)} 个声明 +1 的方向**全部有冻结卡片**"
              if not _viol else
              f"共域（可判部分）违规：声明 +1 但**无冻结卡片** = {_viol}")
        # 比例式禁写：共域未定时不得出现 `SUCCEEDED = N / <denom>` 这种**新**比例
        _ratio = re.findall(r"SUCCEEDED\s*=\s*(\d+)\s*/\s*(\d+)\s*$", _tb_txt, re.M)
        _has_parallel = all(k in _tb_txt for k in AC.PARALLEL_REPORT_KEYS)
        print(f"      共域整体状态：**未通过**（`DIR-01…DIR-10` 贡献成功数但卡片文件已失）"
              f"⇒ 只允许并列表述；并列键齐全 = {_has_parallel}")
        check(_has_parallel,
              "共域未定时，档案里有**并列表述**（"
              + " / ".join(AC.PARALLEL_REPORT_KEYS) + "）"
              if _has_parallel else
              "共域未定时**缺并列表述**（" + " / ".join(AC.PARALLEL_REPORT_KEYS) + "）")
    except AC.CountError as _e:
        check(False, f"共域校验失败：{_e}")

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

# ⑥b ★ DEF-26S-26：`合计 N 条` 的口径是「**最大在用纪律编号**」（不是该列表条目数）⇒
#   本项做**可证伪**的检查：① LOOP-MECHANISM.md 的声明值 == 全文件最大在用编号；
#   ② SUMMARY.md 的声明值 == 同一个数。★ 原 ⑥ 只比两个**都过期**的文件互相一致 ⇒ 恒真。
import re as _re2
_lm = ROOT / "LOOP-MECHANISM.md"
if _lm.exists():
    _t2 = _lm.read_text(encoding="utf-8")
    _m2 = _re2.search(r"## 机制级纪律（续编；全 loop 合计 \*\*(\d+)\s*条\*\*）", _t2)
    _decl = int(_m2.group(1)) if _m2 else None
    _maxn = max((int(x) for x in _re2.findall(r"### 纪律\s*\*{0,2}(\d{1,3})", _t2)), default=0)
    _sm = ROOT / "SUMMARY.md"
    _dsm = None
    if _sm.exists():
        _m3 = _re2.search(r"纪律到第\s*(\d+)\s*号", _sm.read_text(encoding="utf-8"))
        _dsm = int(_m3.group(1)) if _m3 else None
    check(_decl == _maxn,
          f"⑥b LOOP-MECHANISM 声明 {_decl} vs 全文件最大在用纪律编号 {_maxn}（须相等）")
    check(_dsm == _maxn,
          f"⑥b SUMMARY 声明 {_dsm} vs 全文件最大在用纪律编号 {_maxn}（须相等）")

# ⑦ ★ 归档完整性（2026-11-17，DEF-71-7）：审计原先只核**计数一致性**，
#    抓不到「某个 DIR 的 VERDICT 没进证据表 / 没进 register」这类**静默丢失**（本轮实际发生：D-92 未推上 trainer）。
print("\n⑦ 归档完整性（DIR ≥ 69：register 与证据表都必须覆盖）")
_revdirs = sorted([_p.name for _p in (ROOT / "revolve").glob("DIR-*") if _p.is_dir()])
_reg_t = (ROOT / "DIRECTION-REGISTER.md").read_text(encoding="utf-8") if (ROOT / "DIRECTION-REGISTER.md").exists() else ""
_tb_t = (ROOT / "evidence/table.md").read_text(encoding="utf-8") if (ROOT / "evidence/table.md").exists() else ""
_mreg, _mev = [], []
for _d in _revdirs:
    try:
        _n = int(_d.split("-")[1])
    except Exception:
        continue
    if _n < 69:
        continue
    if _d not in _reg_t:
        _mreg.append(_d)
    if (ROOT / "revolve" / _d / "VERDICT.md").exists() and _d not in _tb_t:
        _mev.append(_d)
check(not _mreg, f"⑦ DIR（≥69）都在 DIRECTION-REGISTER：缺 {_mreg or '无'}")
check(not _mev, f"⑦ 有 VERDICT 的 DIR（≥69）都在证据表出现：缺 {_mev or '无'}")

# ⑧ ★ P131-1（Cycle 244）：开卡清单（CARD-CHECKLIST.md）的**格式完整性**。
#   动机：`P131-1` 把 23 条纪律从 `handoff.md` 迁进该文件，但**没有任何守卫** ⇒
#   它会以与 `handoff.md` 完全相同的方式腐烂（本轮实测两次：漏写 `**依据**` 字段、
#   以及 `### K1 的回扫结果` 被当成条目 ⇒ 朴素切分器数出 25 条）。
#   本项不检查纪律**内容**（那属 `P132-1` 的核对器），只检查**形式**是否可被机械解析。
#   ★ 口径同 ⑥b：**不硬编码条目数**，而是断言「声明值 == 机械点数」（DEF-128-03）。
print("\n⑧ 开卡清单格式（CARD-CHECKLIST.md；P131-1）")
_cc = ROOT / "CARD-CHECKLIST.md"
if not _cc.exists():
    check(False, "⑧ CARD-CHECKLIST.md 缺失（开卡纪律的唯一归档地）")
else:
    _t8 = _cc.read_text(encoding="utf-8")
    # 条目切分：K1 用 `## K1 · `（表格式），K2+ 用 `### K<n> · `（项目式）
    _hits = [(int(m.group(1)), m.start()) for m in re.finditer(r"^#{2,3} K(\d+) · ", _t8, re.M)]
    _nums = [n for n, _ in _hits]
    check(_nums == list(range(1, len(_nums) + 1)),
          f"⑧ 条目编号须连续 1..{len(_nums)}（实际 {_nums}）")
    # 声明值 vs 机械点数（DEF-128-03：结构计数必须由代码数，不得手写）
    _m8 = re.search(r"当前条目数：\*{0,2}(\d+)", _t8)
    _decl8 = int(_m8.group(1)) if _m8 else None
    check(_decl8 == len(_nums),
          f"⑧ 声明条目数 {_decl8} vs 机械点数 {len(_nums)}（须相等）")
    # 每条：K1 只要 触发/依据/检查；K2+ 还要 原文
    _bad8 = []
    for _i, (_n, _pos) in enumerate(_hits):
        _end = _hits[_i + 1][1] if _i + 1 < len(_hits) else len(_t8)
        _b = _t8[_pos:_end]
        _need = ("触发", "依据", "检查") if _n == 1 else ("原文", "触发", "依据", "检查")
        for _f in _need:
            if f"**{_f}**" not in _b:
                _bad8.append(f"K{_n}缺{_f}")
        _mchk = re.search(r"\*\*检查\*\*\s*[：|]\s*([^\n]*)", _b)
        if not (_mchk and re.search(r"[①②③]", _mchk.group(1))):
            _bad8.append(f"K{_n}无执行方式")
    check(not _bad8, f"⑧ 每条须有 原文/触发/依据/检查 + 执行方式：缺 {_bad8 or '无'}")
    # 「原文」不得为空（迁入条目必须逐字保留措辞）
    _empty8 = [int(m.group(1)) for m in
               re.finditer(r"^### K(\d+) · [^\n]*\n- \*\*原文\*\*：\s*$", _t8, re.M)]
    check(not _empty8, f"⑧ 「原文」不得为空：空 {_empty8 or '无'}")
    # 声明分布 vs 机械分布（本项目已实际踩过：声明 15/9 而实测 14/10）
    _mk = []
    for _k in range(len(_hits)):
        _q = _hits[_k][1]
        _e = _hits[_k + 1][1] if _k + 1 < len(_hits) else len(_t8)
        _mm = re.search(r"\*\*检查\*\*\s*[：|]\s*([^\n]*)", _t8[_q:_e])
        _hit = re.search(r"[①②③]", _mm.group(1)) if _mm else None
        _mk.append(_hit.group(0) if _hit else "?")
    _real = Counter(_mk)
    _dbad = []
    for _sym, _lbl in (("①", "自动化"), ("②", "留痕"), ("③", "人工")):
        _m8b = re.search(rf"`{_sym}{_lbl}`\s*(\d+)\s*条", _t8)
        if _m8b and int(_m8b.group(1)) != _real.get(_sym, 0):
            _dbad.append(f"{_sym}{_lbl} 声明 {_m8b.group(1)} vs 实测 {_real.get(_sym, 0)}")
    check(not _dbad, f"⑧ 执行方式分布：{_dbad or '一致'}")
    # ⑧e–⑧g ★★ P133-1（Cycle 246）：`判读` 字段与 `check_cards.py` 的 `INTERP` **逐字同一**。
    #   动机：`check_cards.py` 为让谓词可执行，对 5 条纪律各自做了**窄化**，但窄化原本只写在
    #   各函数体的**局部变量**里 ⇒ 归档（本文件）无从核对，脚本一改，归档就是**谎**且无人发现。
    #   ★ 这正是 `P131-1` 已吃过的亏（纪律只活在每轮被覆盖的 `handoff.md` 里）**换了个载体**。
    #   口径：**字符串同一性**，不做语义判断（`P131-1` 的手写计数漂移 19/23 就是这么来的）。
    # ★★ `P134-1`：**不再 `import check_cards`**，改为从**源码**用 `ast.literal_eval` 取 `INTERP`。
    #   动机（本轮实测踩到）：`import` 走 `__pycache__`，而 pyc 的失效判据是 `(mtime, 源码字节数)`。
    #   于是「同尺寸、同一秒内改回」的脚本**会读到陈旧字节码** —— 实测把 `[降级]` 改成等长的
    #   `[随便]` 再改回，`import` 仍报 `[随便]` ⇒ ⑧e/⑧f/⑧h 会拿**旧文本**做「逐字同一」比对，
    #   而这条断言的全部价值就是「代码与归档不许漂移」。读源码同时去掉了 import 副作用
    #   （不让一个只读审计去执行被审模块）。
    _ccf = pathlib.Path(__file__).resolve().parent / "check_cards.py"
    _INTERP, _imp_err = None, None
    try:
        _cc_tree = ast.parse(_ccf.read_text(encoding="utf-8"))
        for _nd in _cc_tree.body:
            if (isinstance(_nd, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "INTERP" for t in _nd.targets)):
                _INTERP = ast.literal_eval(_nd.value)
                break
        if _INTERP is None:
            _imp_err = "源码里找不到模块级 `INTERP = {...}` 赋值"
    except Exception as _e:
        _imp_err = f"{type(_e).__name__}: {_e}"
    check(_INTERP is not None,
          f"⑧e 须能从 check_cards.py **源码**读出 INTERP（{_imp_err or 'OK'}）")
    if _INTERP is not None:
        # 逐块取 `判读` 行（**块内**搜索，避免命中「变更记录」里的说明文字）
        _blocks8 = {}
        for _i, (_n, _pos) in enumerate(_hits):
            _end = _hits[_i + 1][1] if _i + 1 < len(_hits) else len(_t8)
            _blocks8[_n] = _t8[_pos:_end]
        _got8 = {}
        for _n, _blk in _blocks8.items():
            _m8 = re.search(r"^- \*\*判读\*\*：(.*)$", _blk, re.M)
            if _m8:
                _got8[_n] = (_m8.group(1), _m8)
        # ⑧e 正向：INTERP 里**每一条**都必须在清单里有**逐字相同**的 `判读` 行。
        #   ★ P134-1：原实现只查以 `[窄化]` 开头的条目 ⇒ `K16` 降级为 `②留痕` 后
        #   若仍带 `判读`，⑧e 会**看不见**它，而 ⑧f 又会把它当孤儿 ⇒ 两头痛。
        #   现改为「凡进 INTERP 就必须镜像」，标签合法性交给 ⑧h。
        _e_miss, _e_diff = [], []
        for _k, _v in sorted(_INTERP.items()):
            if _k not in _got8:
                _e_miss.append(f"K{_k}")
            elif _got8[_k][0] != _v:
                _e_diff.append(f"K{_k}(清单{len(_got8[_k][0])}字 vs INTERP{len(_v)}字)")
        check(not _e_miss,
              f"⑧e INTERP 条目须在清单有 `判读` 行：缺 {_e_miss or '无'}")
        check(not _e_diff,
              f"⑧e `判读` 须与 INTERP **逐字相同**：不一致 {_e_diff or '无'}")
        # ⑧f 反向：不得有孤儿 `判读`（清单写了 判读、而 INTERP 里已无该键）
        _e_orphan = sorted(set(_got8) - set(_INTERP))
        check(not _e_orphan,
              f"⑧f 不得有孤儿 `判读`（INTERP 无此键）：多余 {_e_orphan or '无'}")
        # ⑧g 结构：`判读` 必须**紧接** `检查` 行（下一条非空行即 `判读`），便于机械解析
        _e_adj = []
        for _k, (_v, _m) in _got8.items():
            _head8 = _blocks8[_k][:_m.start()]
            _ln8 = [l for l in _head8.splitlines() if l.strip()]
            if not _ln8 or not _ln8[-1].startswith("- **检查**："):
                _e_adj.append(f"K{_k}")
        check(not _e_adj,
              f"⑧g `判读` 须紧接 `检查` 行（中间不得插其他字段）：不合 {_e_adj or '无'}")
        # ⑧h 标签合法：`判读` 必须以 `[窄化]` 或 `[降级]` 开头。
        #   ★ 为什么要这条：⑧e 现在「凡 INTERP 必镜像」，若允许任意前缀，
        #   `INTERP` 就成了自由文本框，`[窄化]`/`[降级]` 的区别（谁还在判、谁已不判）
        #   会静默消失 —— 读者无从知道某条纪律是不是还在被核对器执行。
        _e_tag = [f"K{_k}" for _k, _v in sorted(_INTERP.items())
                  if not (_v.startswith("[窄化]") or _v.startswith("[降级]"))]
        check(not _e_tag,
              f"⑧h `判读` 标签须为 `[窄化]` 或 `[降级]`：不合 {_e_tag or '无'}")


print("\n" + "=" * 88)
print(("★ 全部通过" if not fails else f"★ {len(fails)} 项未通过"))
for m in fails:
    print("   - " + m)
sys.exit(0 if not fails else 1)
