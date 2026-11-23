#!/usr/bin/env python3
"""check_cards.py —— 按 `CARD-CHECKLIST.md` 的 `①自动化` 条目核对方向卡（P132-1）。

★ 本脚本**只读**：不写任何卡片、不改任何冻结行、不运行任何实验代码。
★ 它**不**替代卡片自己的 Stage A 守卫；它只回答「这张卡在形式上是否满足了
   开卡清单里那些 `①自动化` 纪律」。（`P134-1` 起为 **13** 条：`K16` 已降为 `②留痕`，
   原因见 `INTERP[16]` —— 它必须真跑 `precheck` 子进程，而禁用词表只能由卡片自冻结。）

用法：
    python3 check_cards.py --dirs 126 127 128 129 130     # 校准集
    python3 check_cards.py --all                          # 全部 DIR
    python3 check_cards.py --selftest                     # 可证伪自测（造合成样本）
退出码：0 = 无 FAIL；1 = 有 FAIL；2 = 覆盖率不符（清单声明的 ①自动化 条目 ≠ 已实现）

------------------------------------------------------------------------------
设计约束（每一条都对应一次真实事故，见 decisions/ 决定二百三十五）
------------------------------------------------------------------------------
1. **`N/A` 必须有明确条件，不得把不适用记成 PASS**（`P132-1` 原文）。
   本脚本的 `N/A` 一律携带 `reason`，说明"为什么这条纪律对这张卡不适用"。
2. **每条结果都携带 `interp`（本脚本实际采用的判读）**。
   ★ 若某条纪律的原文比可机械判定的范围**宽**，本脚本按**窄**的那一侧实现，
   并把差异写进 `interp` —— 这样"我悄悄放宽了判据"永远不可能发生。
   凡 `interp` 以 `[窄化]` 开头的，后续都须由人复核。
3. **不硬编码条目数**（`DEF-128-03`）：`①自动化` 的集合**从清单现读**，
   与本脚本的 `REGISTRY` 比对；不一致即 `exit 2`，不静默漏判。
4. **不猜语义**：需要读产物语义才能判的（如 `②留痕` 10 条）**不在**本脚本内，
   本脚本只做 `①自动化`。
"""
import argparse
import ast
import io
import pathlib
import re
import sys
import tokenize

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__import__("os").environ.get("SGMN_ROOT", HERE.parent.parent))
CHECKLIST = ROOT / "CARD-CHECKLIST.md"
REVOLVE = ROOT / "revolve"
CJK = re.compile(r"[\u4e00-\u9fff]")
FLOAT = re.compile(r"(?<![\w.])-?\d+\.\d+(?![\w])")

PASS, FAIL, NA = "PASS", "FAIL", "N/A"


class R:
    """一条 (卡片 × 纪律) 的结果。"""

    def __init__(self, status, detail="", interp="", reason=""):
        self.status = status
        self.detail = detail
        self.interp = interp
        self.reason = reason

    def __repr__(self):
        return f"R({self.status},{self.detail[:40]!r})"


class Ctx:
    """一张卡 + 它的目录。所有谓词只读这个对象。"""

    def __init__(self, num):
        self.num = num
        self.cdir = REVOLVE / f"DIR-{num}"
        self.ddir = ROOT / f"dir{num}"
        # ★ P134-1 修：原先 key 用 `_v(\d+)\.md$` 且假定必有匹配 ⇒ `--all` 在
        #   `DIR-42/DIRECTION_rev42_v10_rev1.0_archived.md` 与 `DIR-91/DIRECTION_rev91_v28_r23.md`
        #   这类**非 `_vN.md` 结尾**的卡上 `AttributeError: 'NoneType' object has no attribute 'group'`。
        #   口径：① 版本号 `_v(\d+)` 允许出现在名字**任意位置**，取不到记 -1；
        #        ② **排除** 名字含 `archived` 的归档件（否则 `_v10` 与 `_v10_..._archived`
        #           同为版本 10，按名排序会让**归档件胜出**）；
        #        ③ 仍取最高版本，并列时按名字定序（确定性）。
        cands = sorted(
            (p for p in self.cdir.glob(f"DIRECTION_rev{num}_v*.md")
             if "archived" not in p.name),
            key=lambda p: ((int(m.group(1)) if (m := re.search(r"_v(\d+)", p.name)) else -1),
                           p.name),
        )
        self.card = cands[-1] if cands else None
        self.text = self.card.read_text(encoding="utf-8") if self.card else ""
        self.verdict = self.cdir / "VERDICT.md"
        self.py = sorted(self.ddir.glob("*.py")) if self.ddir.is_dir() else []
        self._secs = None

    # --- 派生视图 ---------------------------------------------------------
    def sections(self):
        """[(heading, body)]，按 `##`/`###` 切。"""
        if self._secs is None:
            out, head, buf = [], "", []
            for ln in self.text.splitlines():
                m = re.match(r"^(#{2,3})\s+(.*)$", ln)
                if m:
                    if head or buf:
                        out.append((head, "\n".join(buf)))
                    head, buf = m.group(2), []
                else:
                    buf.append(ln)
            out.append((head, "\n".join(buf)))
            self._secs = out
        return self._secs

    def sec_text(self, pat):
        """所有标题命中 `pat` 的小节正文拼接。"""
        return "\n".join(b for h, b in self.sections() if re.search(pat, h))

    def all_py_text(self):
        return {p.name: p.read_text(encoding="utf-8", errors="replace") for p in self.py}


# ---------------------------------------------------------------------------
# 谓词。每个返回 R；`NA` 必须给出 reason。
# ---------------------------------------------------------------------------

# ★★ P133-1（Cycle 246）：被**窄化**的 5 条纪律的判读文本，**唯一来源**。
#   动机：`check_cards.py` 让谓词可执行，对 5 条各自做了窄化，但窄化原本只写在
#   各函数体内部的局部变量里 ⇒ `CARD-CHECKLIST.md` 无从核对，两边必然漂移。
#   现改为：文本在此定义，函数引用 `INTERP[k]`，审计 ⑧ 断言清单里的 `判读`
#   行与本字典**逐字相同**（字符串同一性，不做语义判断）。
#   ★ 只有以 `[窄化]` 开头的条目才需要在清单里写 `判读` 行。
# ★★ P133-1（Cycle 246）+ P134-1（Cycle 247）：需要**显式判读**的条目，**唯一来源**。
#   `[窄化]` = 该条 `原文` 比可机械判定的范围宽，本脚本按窄的一侧实现；
#   `[降级]` = 该条**不再由本脚本判定**（`P134-1` 把 `K16` 由 `①自动化` 降为 `②留痕`）。
#   ★ 两种标签都必须在本字典里，并**逐字**镜像到 `CARD-CHECKLIST.md` 的 `判读` 行；
#   审计 ⑧e/⑧f 做**字符串同一性**核对，⑧h 校验标签属于上面两种。
INTERP = {
    7: "[窄化] 只抓 **>=3 位小数**的浮点（实测型读数，如 `0.8589`）；`0.05`/`0.0`/`0.5`/`0.95` 这类**惯例阈值**不算 —— `DEF-127-02` 的真实缺陷是**手抄实测值**，不是冻结一个显著性水平（首版按字面抓全部浮点 ⇒ 4/5 误报）；『派生』的等价标注放宽到 `机器算/算出/推导/computed`（卡片的实际用词）",
    8: "[窄化] 仅当『逐位相等』与『浮点』同句、且未出现 整数/有理数/Fraction、也不是『更正/改为/不可满足』句式才 FAIL（修订记录小节整体不计）—— 首版把『rev 1.0 曾要求浮点逐位相等、已因不可满足而改掉』误判为违规",
    16: "[降级] 本条由 `①自动化` **降为 `②留痕`**：静态核对器**不判**，也不得记 PASS。判定须在 trainer 上以**子进程**跑 `precheck{n}.py` 并扫 stdout —— 但『哪些词算 formal 的判据读数』**只能由卡片在开卡时自己冻结**（如 `dir130` 的 `S9_FORBIDDEN` + `T13` 自测）。核对器若自行拟一份通用禁用词表，要么误报合规卡、要么就是把已知的 11 张卡调到全绿 —— 后者正是 `P133-1` 要防的**静默窄化**。⇒ 现由卡片自带的 `S*` 执行，核对器不判。★★ `P134-2` 已**逐卡实跑裁定**（trainer，stdout 原样留档于 `revolve/P134-2-census/`，结论见 `revolve/P134-2-PRECHECK-CENSUS.md`）：**11 张 0 违规**，**无需**任何 `RECHECK.md`。允许打印：结构事实 / 已发表常数复现 / **单个工作例**的判据量与标签 / 检测器自测（含 level·power 仿真与预期标签）；禁止：本研究自身的判据**聚合**（`C2`/`C3`/`C4`、`p₃₅₀`/`p₃₀₀`）与**整条序列/分类**（机械佐证：11 份输出中含 ≥20 个数字的行数 = 0）。★ 故『`frac` 在单个工作例里打印』**不违规** —— `原文` 括号禁的是「**整条**序列/分类」，量词是「整条」。★ 更正 `P134-1` 的**错误论断**：`dir130` 的 `frac = 0.0` 属【工作例 §6】且下一行即声明 `p₃₅₀`/`p₃₀₀` 只在 `--stage formal` 打印 ⇒ 不是「自我豁免」，是**已声明的窄边界**；原记「未裁定」现裁定为**合规**。★★ 但查出**两处与 `K16` 无关的真实缺陷**：① `precheck125`/`precheck126` 的 `S12` 盲态前置条件（新剂量须 0 cell）在 `formal` 之后**永不可能再通过** ⇒ 重跑必 `rc=1`，**不是卡的问题**；② **8/11 根本没有 `--stage` 开关**，且至少 5 张读 `formal` 产物（`readings.json`/`out/cells`）⇒ 它们在**开卡时刻根本跑不起来**，`K16` 对它们只是**空洞成立**（纪律成立是因为检查从未在该时刻发生）⇒ 已派生 `P134-3`。",
    18: "[窄化] 仅当卡片真的在说**继承一个带定义域限制的求解器**时才适用；`û_s=0` 的『夹紧层』是**测量端点的定义**，不是求解器夹紧（首版按字面匹配 clamp/夹紧 ⇒ 4/5 误报）",
    22: "[窄化] 仅在卡片**真的用 α/抽样推断**时要求（首版对『否证/排除』全文触发，5/5 误报）；穷举型卡片没有抽样误差 ⇒ N/A",
    24: "[窄化] 只算**非三引号**、**双引号定界**的字面量（含 `f`/`r`/`u` 等前缀），且须含 CJK；长度用**同一把尺子**、三式同源：裸串 `len(源码) >= 60`（含 2 个定界符 ⇒ 等价于文本 >= 58）、带前缀式 `引号内文本 >= 58`、`f` 式只算**散文**（`{}` 之外，**排除**替换域与格式说明）`>= 58`；**隐式拼接**按同一尺子累加（仅括号内 `NL` 与 `COMMENT` 可连接，`NEWLINE` 逻辑换行**断开**）；三引号文档字符串与短消息不计（首版把全部 docstring 计入 ⇒ 5/5 误报）。★ `P134-1`：原先『段一』『段二』相邻是**两个**都不到 60 的 token ⇒ 漏判；补后 **6 张历史卡 PASS→FAIL**（`DIR-31/33/38/91/103/115`，逐条人工确认是真拼接）。★★ `P134-2` 更正与补全：① 原记『py3.12+ 把 `f` 式 tokenize 成 `FSTRING_START` ⇒ 逃逸』**机制说错了** —— 判据本要求以双引号起，各版本都把带前缀者排除，**结论从未因版本而变**；真缺口是**规则没覆盖带前缀式**（与 `原文`『不得用双引号定界』不符）。② 现补前缀式与 `f` 式，并给 `tokenize` 失败的文件加**逐行兜底**（原整份跳过 ⇒ 未闭合括号即可免检）。★ 尺子为何必须分式：实测 `f` 式若按**整段源码**计，历史全量 **1781 处 / 95 张卡**（量的是代码体积而非中文）；若把格式说明算进散文，`dir51/calib51.py:161` 会因 `:.6f` 越线而误报（该行中文仅 6 字）。★ 实测影响：**25 张卡 PASS→FAIL、无一反向**；兜底路径历史命中 **0**（现状休眠，仅防未来）。★ K24 对历史卡 FAIL 率 60/103 → **85/103** ⇒ **只能作新卡前置门，不得回溯判定历史卡**（`P130-1`/`P131-1`）。",
}

def k04(ctx):
    """K4 含「留一/多重合取」的判据必须在开卡前算功效（条件 grep）。"""
    interp = "[窄化] 触发词命中即要求『功效』小节；排除否定式提及（如『不是重采样近似』）"
    trig = re.compile(r"留一|leave[- ]one[- ]out|bootstrap|重采样|合取")
    neg = re.compile(r"(不是|不做|非)\s*[^\n]{0,8}(重采样|bootstrap|留一)")
    hits = [ln for ln in ctx.text.splitlines()
            if trig.search(ln) and not neg.search(ln)]
    if not hits:
        return R(NA, "无『留一/bootstrap/重采样/合取』触发词（或仅出现在否定式里）",
                 interp, reason="判据不含重采样型估计 ⇒ 无需功效段")
    has = any("功效" in h for h, _ in ctx.sections())
    if has:
        h = [h for h, _ in ctx.sections() if "功效" in h][0]
        return R(PASS, f"功效小节：{h[:48]}", interp)
    return R(FAIL, f"触发 {len(hits)} 处但无『功效』小节；首处：{hits[0].strip()[:70]}", interp)


def k05(ctx):
    """K5 写「否证/排除」条款前必须枚举备择族整个参数范围。"""
    interp = ("[窄化] 仅在卡片**具名一个备择族**（`H*-alt`）时要求枚举；"
              "『否证/排除』本身在中文科技写作里太常见，不能作触发词（首版实测 5/5 误报）")
    alts = sorted(set(re.findall(r"[A-Za-z]\w*-alt\w*", ctx.text)))
    if not alts:
        return R(NA, "卡片未具名备择族（无 `H*-alt`）", interp,
                 reason="没有备择族就谈不上『枚举备择族的参数范围』")
    enum = ("备择" in ctx.text) or ("参数范围" in ctx.text)
    interval = re.search(r"[（(]\s*\d+\s*[,，]\s*\d+\s*[\]）)]|\[\s*\d+\s*[,，]", ctx.text)
    if enum or interval:
        return R(PASS, f"具名 {alts}；有备择/参数范围/区间", interp)
    return R(FAIL, f"具名 {alts} 但全文无『备择』『参数范围』『区间』之一", interp)


def k07(ctx):
    """K7 冻结表里只冻整数/有理数，浮点一律做派生。"""
    interp = INTERP[7]
    secs = [(h, b) for h, b in ctx.sections() if re.search(r"判据|阈值|冻结", h)]
    if not secs:
        return R(NA, "无『判据/阈值/冻结』小节", interp,
                 reason="本卡没有冻结表 ⇒ 没有『只冻整数』的对象，K7 不适用")
    rows = []
    for head, body in secs:
        for ln in body.splitlines():
            if not ln.strip().startswith("|"):
                continue
            fs = re.findall(r"(?<![\w.])-?\d+\.\d{3,}(?![\w])", ln)
            if fs and not re.search(r"派生|derived|机器算|算出|推导|computed|by code", ln):
                rows.append((head, fs, ln.strip()))
    if not rows:
        return R(PASS, "冻结/判据/阈值小节内无未标『派生』的浮点", interp)
    tot = sum(len(f) for _, f, _ in rows)
    return R(FAIL, f"{tot} 个未标『派生』的浮点，例：{rows[0][2][:80]}", interp)


def k08(ctx):
    """K8 冻结卡里不得要求「浮点逐位相等」。"""
    interp = INTERP[8]
    bad = []
    for head, body in ctx.sections():
        if re.search(r"修订记录|revision|变更|§10", head):
            continue
        for sent in re.split(r"[。；\n]", body):
            if "逐位相等" not in sent and "逐位相同" not in sent:
                continue
            if re.search(r"整数|有理数|Fraction", sent):
                continue
            if re.search(r"更正|改为|已撤|不再要求|不可满足|rev 1\.0", sent):
                continue
            if re.search(r"浮点|\d+\.\d+", sent):
                bad.append((head, sent.strip()[:90]))
    if not bad:
        return R(PASS, "无『逐位相等』与浮点比较的组合", interp)
    return R(FAIL, f"{len(bad)} 处；例：{bad[0][1]}", interp)


def k12(ctx):
    """K12 常量不得手写：模块级不得有承载 seed/结构的大写字面量列表/字典。"""
    interp = ("[窄化] 只算**含 ≥5 个数值叶子**的大写字面量（纯字符串列表是列名/标签，"
              "不是 seed/结构；首版把 `KEYS=[10 个列名]` 误判为违规）")
    bad = []
    for p in ctx.py:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Name) or not tgt.id.isupper() or len(tgt.id) < 3:
                continue
            if not isinstance(node.value, (ast.List, ast.Dict)):
                continue
            nums = sum(1 for n in ast.walk(node.value)
                       if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                       and not isinstance(n.value, bool))
            if nums >= 5:
                bad.append(f"{p.name}:{tgt.id}({nums} 数值)")
    if not bad:
        return R(PASS, f"扫 {len(ctx.py)} 个 .py，无大写字面量数值常量", interp)
    return R(FAIL, f"{len(bad)} 处：{', '.join(bad[:4])}", interp)


def k14(ctx):
    """K14 源码级守卫必须用 AST，不得用文本行扫描。"""
    interp = ("[窄化] 仅对**真的在扫 .py 源码行**的脚本生效（`splitlines()` + 行内子串判定）；"
              "读二进制算 md5 不算")
    bad, scanned = [], []
    for p in ctx.py:
        s = p.read_text(encoding="utf-8", errors="replace")
        does_scan = re.search(r"splitlines\(\)", s) and re.search(
            r"if\s*\(?[^\n]{0,40}\bin\s+(ln|l|line)\b", s) and ".py" in s
        if not does_scan:
            continue
        scanned.append(p.name)
        if not re.search(r"^\s*import\s+ast\b|^\s*from\s+ast\s+import", s, re.M):
            bad.append(p.name)
    if not scanned:
        return R(NA, "无脚本对 .py 源码做文本行扫描", interp,
                 reason="不扫源码的脚本不涉及『用 AST 还是文本』的选择")
    if bad:
        return R(FAIL, f"扫源码但未 import ast：{', '.join(bad)}", interp)
    return R(PASS, f"扫源码的脚本均 import ast：{', '.join(scanned)}", interp)


def k15(ctx):
    """K15 「阶段纪律」必须配机械守卫：每处『不得打印/输出』须有 S 编号执行。"""
    interp = "按原文执行：`不得打印|不得输出` 的每一处所在小节内必须出现 `S\\d+`"
    hits = []
    for head, body in ctx.sections():
        for ln in body.splitlines():
            if re.search(r"不得打印|不得输出|禁止打印", ln):
                has_s = re.search(r"\bS\d+\b", ln) or re.search(r"\bS\d+\b", body)
                hits.append((head, ln.strip(), bool(has_s)))
    if not hits:
        return R(NA, "全文无『不得打印/不得输出』", interp,
                 reason="本卡未声明任何阶段打印纪律 ⇒ 无守卫可配")
    miss = [h for h in hits if not h[2]]
    if miss:
        return R(FAIL, f"{len(miss)}/{len(hits)} 处无 S 编号；例：{miss[0][1][:70]}", interp)
    return R(PASS, f"{len(hits)} 处均点名 S 编号", interp)


def k17(ctx):
    """K17 盲态守卫不得复用会产出判据结局的函数。"""
    interp = "按原文执行：盲态/结构核脚本的 import 不得含结局判定器（R1_task / adjudicate / verdict）"
    if not re.search(r"盲态|结构核", ctx.text):
        return R(NA, "卡片未声明盲态/结构核", interp,
                 reason="没有盲态守卫 ⇒ 无『复用结局函数』的风险")
    bad = []
    for p in ctx.py:
        if not re.search(r"precheck|blind|struct", p.name):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        imps = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imps += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                imps.append(n.module)
        hits = [i for i in imps if re.search(r"R1_task|adjudicate|verdict", i)]
        if hits:
            bad.append(f"{p.name}←{hits}")
    if bad:
        return R(FAIL, f"盲态脚本 import 了结局函数：{', '.join(bad)}", interp)
    return R(PASS, "盲态脚本未 import 结局函数", interp)


def k18(ctx):
    """K18 继承带定义域限制的求解器，必须打印夹紧是否被触发。"""
    interp = INTERP[18]
    solver = re.search(r"(继承|沿用|`?\w+\d*`?)\s*的?\s*求解器|求解器[^\n]{0,20}(定义域|域内|越界)",
                       ctx.text)
    if not solver:
        return R(NA, "未出现『继承带定义域限制的求解器』的表述", interp,
                 reason="卡片里的 clamp/夹紧 指测量端点（`û_s=0` 夹紧层 / `n_clamped`），"
                        "不是 DEF-124-01 的求解器定义域夹紧")
    printed = []
    for name, s in ctx.all_py_text().items():
        for ln in s.splitlines():
            if re.search(r"clamp|夹紧", ln) and re.search(r"print|log\(", ln):
                printed.append(name)
                break
    if printed:
        return R(PASS, f"打印 clamp 计数：{', '.join(sorted(set(printed)))}", interp)
    return R(FAIL, "卡片提到 clamp/夹紧，但无脚本打印该计数", interp)


def k19(ctx):
    """K19 占位脚手架不得留在自测/守卫/驱动里。"""
    interp = "按原文执行：`.py` 不得含 TODO/FIXME/PLACEHOLDER/XXX"
    bad = []
    for name, s in ctx.all_py_text().items():
        for m in re.finditer(r"\b(TODO|FIXME|PLACEHOLDER|XXX)\b", s):
            ln = s[: m.start()].count("\n") + 1
            bad.append(f"{name}:{ln}:{m.group(1)}")
    if not bad:
        return R(PASS, f"扫 {len(ctx.py)} 个 .py，无占位标记", interp)
    return R(FAIL, f"{len(bad)} 处：{', '.join(bad[:4])}", interp)


def k21(ctx):
    """K21 探查脚本一律落 dir*/probes/。"""
    interp = "按原文执行：`probe*.py` 不得落在 `dir*/` 根"
    if not ctx.ddir.is_dir():
        return R(NA, f"无 {ctx.ddir.name}/ 目录", interp, reason="没有目录就无从放置探查脚本")
    bad = [p.name for p in ctx.py if p.name.startswith("probe")]
    has_sub = (ctx.ddir / "probes").is_dir()
    if not bad:
        return R(PASS, "dir 根无 probe*.py", interp)
    return R(FAIL, f"{len(bad)} 个 probe 在根：{', '.join(bad)}"
                   f"（probes/ 子目录{'已存在' if has_sub else '不存在'}）", interp)


def k22(ctx):
    """K22 判决表必须显式含「若 size(判据) > α + 容差 ⇒ 记不可采用」。"""
    interp = INTERP[22]
    if not (re.search(r"\bα\b|alpha", ctx.text) or re.search(r"p\s*<\s*0", ctx.text)):
        return R(NA, "卡片未使用 α / 显著性阈值", interp,
                 reason="穷举或纯描述型判据没有抽样误差 ⇒ 不存在 size-vs-α 的取舍")
    if not ctx.verdict.exists():
        return R(NA, "无 VERDICT.md（尚未结题）", interp, reason="判决书未生成")
    vt = ctx.verdict.read_text(encoding="utf-8")
    has = {"size": bool(re.search(r"\bsize\b", vt)),
           "α": bool(re.search(r"\bα\b|alpha", vt)),
           "不可采用": "不可采用" in vt}
    if all(has.values()):
        return R(PASS, "VERDICT 含 size / α / 不可采用", interp)
    miss = [k for k, v in has.items() if not v]
    return R(FAIL, f"VERDICT 缺：{', '.join(miss)}", interp)


_K24_TEXT_MIN = 58
#   ★ 为何是 58 而非 60（`P134-2`）：裸串沿用 `P132-1` 的历史判据 `len(源码) >= 60`，而裸串源码
#   含 **2 个定界符** ⇒ 等价于「引号内文本 >= 58」。带前缀/`f"…"` 只能按**文本**度量，取 58
#   才与裸串**同一把尺子**；取 60 会平白留下一段 58–59 字符的残余漏判。
#   ★★ 而「文本」对 `f"…"` **必须排除替换域与格式说明**：本判据量的是**中文散文**，不是格式机器。
#   实测反例（差点变成误报）：`dir51/calib51.py:161` 的
#       f"  功效: mean(D^ortho){m_o:+.6f} sd={s_o:.6f} rel_se={relse:.4f} n_need={n_need}  墙钟 {w:.1f}s"
#   若把 `:.6f` 这类格式说明算进文本，长度 59 就越线 —— 而该行中文只有「功效」「墙钟」6 个字，
#   是一行**可读的**诊断输出，正是规则**不该**管的东西。故只收 `{}` **之外**的 `FSTRING_MIDDLE`。
_PREFIX = re.compile(r'^[rRbBuUfF]+')
#   tokenize 失败文件的兜底行扫描：只认**单行** `"…"`。`(?<!["\'])` 避免把 `""` 或 `'"'` 的
#   第二个引号当成起点。★ 保守设计：宁可漏判，也不误报（见 `_k24_linescan`）。
_K24_LINE_RE = re.compile(r'(?<!["\'])"((?:[^"\\\n]|\\.)*)"')


def _k24_hit(tok_or_text):
    """**裸**字面量的 K24 判据（`P132-1` 原规则，逐字节未动）。"""
    st = tok_or_text if isinstance(tok_or_text, str) else tok_or_text.string
    return (st.startswith('"') and not st.startswith('"""')
            and CJK.search(st) is not None and len(st) >= 60)


def _k24_prefixed_text(src):
    """带前缀（`r"`/`b"`/`u"`/`f"`…）的双引号字面量 ⇒ 返回**引号内文本**；非此形返回 `None`。"""
    m = _PREFIX.match(src)
    if not m:
        return None
    body = src[m.end():]
    if not body.startswith('"') or body.startswith('"""') or len(body) < 2:
        return None
    return body[1:-1]


def _k24_linescan(name, s):
    """`tokenize` 失败时的**兜底**扫描（`P134-2` 新增）。

    动机：原先整份文件 `continue` 跳过 ⇒ 未闭合括号这类**词法级**错误（`def f(:` 即是）
    即可让文件里所有长 CJK 双引号串**免检**（★ 实测自 `P132-1` 起即如此，非本轮引入）。
    ★ 保守：含三引号（`\"\"\"` 或 `'''`）的行**整行跳过** —— 多行串无法按行判定，
    宁可漏判不可误报；长度口径与带前缀式一致（`_K24_TEXT_MIN`）。
    """
    out = []
    for ln, line in enumerate(s.splitlines(), 1):
        if '"""' in line or "'''" in line:
            continue
        for m in _K24_LINE_RE.finditer(line):
            text = m.group(1)
            if len(text) >= _K24_TEXT_MIN and CJK.search(text):
                out.append(f"{name}:{ln}(兜底)")
    return out


def _k24_judge_run(run, bad, name):
    """对一段（一或多个）字面量单元应用 K24 判据。"""
    if not run:
        return
    if len(run) == 1:
        u = run[0]
        if u["kind"] == "bare":
            return                       # ① 已判（裸单串，逐字节沿用 `P132-1`）
        if CJK.search(u["text"]) and len(u["text"]) >= _K24_TEXT_MIN:
            bad.append(f"{name}:{u['row']}(前缀/f)")
        return
    if all(u["kind"] == "bare" for u in run):
        # ② `P134-1` 原规则，**逐字节未动**：合并源码长度 + 首段须以 `"` 起
        merged = "".join(u["src"] for u in run)
        if _k24_hit(merged) and run[0]["src"].startswith('"'):
            bad.append(f"{name}:{run[0]['row']}(拼接{len(run)}段)")
        return
    # ★ `P134-2` 新增：含前缀/f-string 的拼接 ⇒ 按**文本**长度累加
    text = "".join(u["text"] for u in run)
    if CJK.search(text) and len(text) >= _K24_TEXT_MIN:
        bad.append(f"{name}:{run[0]['row']}(拼接{len(run)}段·含前缀)")


def k24(ctx):
    """K24 不要用 ASCII 双引号内嵌写中文长字符串。"""
    interp = INTERP[24]
    bad = []
    # 连接符：**只允许** `NL`（括号内的换行）与 `COMMENT`。★ `NEWLINE`（逻辑换行）
    # 必须**断开** —— 否则相邻两条独立的字符串语句 `"甲…"。换行。"乙…"` 会被并成一段。
    CONNECT = {tokenize.NL, tokenize.COMMENT}
    for p in ctx.py:
        s = p.read_text(encoding="utf-8", errors="replace")
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(s).readline))
        except (tokenize.TokenError, IndentationError):
            # ★ `P134-2`：不再整份跳过，改为**逐行兜底扫描**（原先那是一处真实逃逸）。
            bad.extend(_k24_linescan(p.name, s))
            continue
        # ① 单字面量（`P132-1` 原规则，逐字节未动）
        for tok in toks:
            if tok.type == tokenize.STRING and _k24_hit(tok):
                bad.append(f"{p.name}:{tok.start[0]}")
        # 拆「字面量单元」：裸/带前缀 STRING 各占 1 token；f-string 占 START..END 一整段。
        #   ★ 为何要拆：`f"…"` 在 py3.12+ 是 `FSTRING_START`+`FSTRING_MIDDLE`+`FSTRING_END`，
        #   **根本不是 `STRING`** ⇒ 规则 ①② 的结构性盲区（历史全量实测：**20 处 / 12 张卡**）。
        units, i = [], 0
        while i < len(toks):
            t = toks[i]
            if t.type == tokenize.FSTRING_START:
                depth, j, mids, field = 0, i, [], 0
                while j < len(toks):
                    tj = toks[j]
                    if tj.type == tokenize.FSTRING_START:
                        depth += 1
                    elif tj.type == tokenize.OP and tj.string in ("{", "}"):
                        # ★ 替换域计数：`{`/`}` 成对。py3.12+ 把**格式说明**（`:.6f`）也发成
                        #   `FSTRING_MIDDLE`（与散文**同深度**），只能靠这个计数器区分。
                        field += 1 if tj.string == "{" else -1
                    elif (tj.type == tokenize.FSTRING_MIDDLE
                          and depth == 1 and field == 0):
                        mids.append(tj.string)      # 只取最外层、替换域**之外**的散文
                    elif tj.type == tokenize.FSTRING_END:
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if j < len(toks):
                    st, en = toks[i], toks[j]
                    # 只算**单行、非三引号**；文本口径 = 各 FSTRING_MIDDLE 源码长度之和
                    if st.start[0] == en.end[0] and '"""' not in st.string:
                        units.append({"i0": i, "i1": j, "kind": "f", "src": None,
                                      "text": "".join(mids), "row": st.start[0]})
                    i = j + 1
                    continue
            if t.type == tokenize.STRING:
                pre = _k24_prefixed_text(t.string)
                bare = t.string.startswith('"')
                units.append({"i0": i, "i1": i, "kind": "bare" if bare else "p",
                              "src": t.string,
                              "text": t.string[1:-1] if bare else (pre or ""),
                              "row": t.start[0]})
            i += 1
        # ② 隐式拼接：相邻单元之间**只隔** `NL`/`COMMENT` ⇒ 视为同一段（`P134-1` 语义）。
        run = []
        for u in units + [None]:
            if u is not None and (not run or all(
                    toks[k].type in CONNECT for k in range(run[-1]["i1"] + 1, u["i0"]))):
                run.append(u)
                continue
            _k24_judge_run(run, bad, p.name)
            run = [u] if u is not None else []
    bad = list(dict.fromkeys(bad))          # 去重但保序
    if not bad:
        return R(PASS, f"扫 {len(ctx.py)} 个 .py，无长 CJK 双引号单行串/拼接串", interp)
    return R(FAIL, f"{len(bad)} 处：{', '.join(bad[:5])}", interp)


# ★ `P134-1`：`16` **已移除** —— `K16` 由 `①自动化` 降为 `②留痕`（见 `INTERP[16]`）。
#   它要求真的执行 `precheck` 子进程才能判，而禁用词表只能由卡片自己冻结 ⇒ 本脚本不冒充执行。
REGISTRY = {4: k04, 5: k05, 7: k07, 8: k08, 12: k12, 14: k14, 15: k15,
            17: k17, 18: k18, 19: k19, 21: k21, 22: k22, 24: k24}


# ---------------------------------------------------------------------------
# 从清单现读 ①自动化 集合（不硬编码，DEF-128-03）
# ---------------------------------------------------------------------------
def checklist_auto():
    """返回 {K号: 标题}，只取标了 `①自动化` 的条目。"""
    if not CHECKLIST.exists():
        return {}
    t = CHECKLIST.read_text(encoding="utf-8")
    hits = [(int(m.group(1)), m.start()) for m in re.finditer(r"^#{2,3} K(\d+) · ", t, re.M)]
    out = {}
    for i, (n, pos) in enumerate(hits):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(t)
        blk = t[pos:end]
        m = re.search(r"\*\*检查\*\*\s*[：|]\s*([^\n]*)", blk)
        chk = m.group(1) if m else ""
        if "①" in chk:
            title = re.search(r"^#{2,3} K\d+ · ([^\n]*)", blk, re.M).group(1)
            out[n] = title
    return out


# ---------------------------------------------------------------------------
# 自测：合成样本，每条例律造一个「该红」的夹具，证明谓词非空转
# ---------------------------------------------------------------------------
def selftest():
    import tempfile
    import os

    cases = []  # (K, label, expect, card_md, {py_name: py_src})

    cases.append((5, "具名 H1-alt 却无备择枚举", FAIL,
                  "## §1\n`H1` 被否证 ⟺ `H1-alt` 不成立。\n", {}))
    cases.append((5, "具名 H1-alt 且给了参数范围", PASS,
                  "## §1\n`H1-alt` 的备择族与参数范围见下。\n", {}))
    cases.append((5, "完全不提 -alt", NA, "## §1\n`H1` 被否证。\n", {}))

    cases.append((7, "冻结表里裸实测浮点", FAIL,
                  "## §3 判据与冻结阈值\n| 量 | 值 |\n|---|---|\n| `r` | `0.8589` |\n", {}))
    cases.append((7, "冻结表里实测浮点标了派生", PASS,
                  "## §3 判据与冻结阈值\n| 量 | 值 |\n|---|---|\n| `r` | `0.8589`（派生） |\n", {}))
    cases.append((7, "实测浮点标了机器算", PASS,
                  "## §3 判据与冻结阈值\n| 量 | 值 |\n|---|---|\n| `r` | `-0.600000`（机器算） |\n", {}))
    cases.append((7, "惯例阈值 0.05 不算实测浮点", PASS,
                  "## §3 判据与冻结阈值\n| 量 | 值 |\n|---|---|\n| `ALPHA` | `0.05` |\n", {}))
    cases.append((7, "无冻结小节", NA, "## §1\n正文。\n", {}))

    cases.append((8, "要求浮点逐位相等", FAIL,
                  "## §1\n浮点必须逐位相等。\n", {}))
    cases.append((8, "整数逐位相等", PASS,
                  "## §1\n整数必须逐位相等。\n", {}))

    cases.append((12, "模块级大写数值常量", FAIL, "## §1\nx\n",
                  {"a.py": "SEEDS = [1, 2, 3, 4, 5, 6]\n"}))
    cases.append((12, "纯字符串列表（列名）", PASS, "## §1\nx\n",
                  {"a.py": 'KEYS = ["a", "b", "c", "d", "e", "f"]\n'}))

    cases.append((14, "扫源码行却未 import ast", FAIL, "## §1\nx\n",
                  {"precheck1.py": ('src = open("m.py").read()\n'
                                    'w = [ln.strip() for ln in src.splitlines()'
                                    ' if "open(" in ln]\n')}))
    cases.append((14, "扫源码且 import ast", PASS, "## §1\nx\n",
                  {"precheck1.py": ('import ast\nsrc = open("m.py").read()\n'
                                    'w = [ln for ln in src.splitlines() if "open(" in ln]\n')}))
    cases.append((14, "只算 md5，不扫源码", NA, "## §1\nx\n",
                  {"precheck1.py": 'import hashlib\nh=hashlib.md5(open("m","rb").read())\n'}))

    cases.append((15, "写了不得打印但无 S 编号", FAIL,
                  "## §6 阶段纪律\nprecheck 不得打印判据读数。\n", {}))
    cases.append((15, "写了不得打印且点名 S9", PASS,
                  "## §6 阶段纪律\nprecheck 不得打印判据读数，由 `S9` 执行。\n", {}))

    cases.append((18, "继承受限求解器但不打印夹紧", FAIL,
                  "## §2\n本卡继承带定义域限制的求解器，夹紧须报。\n", {"a.py": "x = 1\n"}))
    cases.append((18, "继承受限求解器且打印夹紧", PASS,
                  "## §2\n本卡继承带定义域限制的求解器，夹紧须报。\n",
                  {"a.py": 'print("夹紧次数 =", n)\n'}))
    cases.append((18, "夹紧层是测量端点定义（非求解器）", NA,
                  "## §2\n`û_s=0` 的夹紧层算作 `û_s<1`。\n", {"a.py": "x = 1\n"}))

    cases.append((19, "含 TODO", FAIL, "## §1\nx\n", {"a.py": "x = 1  # TODO 填\n"}))
    cases.append((19, "干净", PASS, "## §1\nx\n", {"a.py": "x = 1\n"}))

    cases.append((21, "probe 落在 dir 根", FAIL, "## §1\nx\n", {"probe_x.py": "x=1\n"}))
    cases.append((21, "无 probe", PASS, "## §1\nx\n", {"mech1.py": "x=1\n"}))

    cases.append((22, "用 α 但 VERDICT 缺三词", FAIL,
                  "## §3\n判据取 α=0.05。\n", {}))
    cases.append((22, "不用 α", NA, "## §3\n穷举全部结构。\n", {}))

    cases.append((24, "长 CJK 双引号单行串", FAIL, "## §1\nx\n",
                  {"a.py": 'print("这是一段刻意写得很长的中文提示信息用来越过六十个字符的阈值因此它必须被判定为违规并且还需要继续补充中文内容才能稳稳超过六十个字符的界限才行")\n'}))
    cases.append((24, "三引号文档字符串", PASS, "## §1\nx\n",
                  {"a.py": 'def f():\n    """这是一段很长很长的中文文档字符串用来确认三引号不被计入违规六十字符以上"""\n'}))
    # ★★ P134-1：隐式拼接（`"段一" "段二"`）原先是**漏判**的逃逸路径 ——
    #   两个都不到 60 字符的 token 各自过不了阈值。以下每条期望值都**实测**过（非手算）。
    _A, _B = "段" * 34, "乙" * 34          # 各 34 字 → 单条不过阈值、合并 68 过
    _a, _b = "短" * 20, "少" * 20          # 各 20 字 → 合并 40 仍不过
    cases.append((24, "隐式拼接两段(34+34)合并越线", FAIL, "## §1\nx\n",
                  {"a.py": f'x = "{_A}" "{_B}"\n'}))
    cases.append((24, "隐式拼接两段(20+20)合并未越线", PASS, "## §1\nx\n",
                  {"a.py": f'x = "{_a}" "{_b}"\n'}))
    cases.append((24, "括号内跨行拼接(34+34)", FAIL, "## §1\nx\n",
                  {"a.py": f'x = ("{_A}"\n     "{_B}")\n'}))
    cases.append((24, "括号内跨行拼接(20+20)未越线", PASS, "## §1\nx\n",
                  {"a.py": f'x = ("{_a}"\n     "{_b}")\n'}))
    cases.append((24, "三条拼接(24*3)", FAIL, "## §1\nx\n",
                  {"a.py": f'z = ("{"三"*24}"\n     "{"四"*24}"\n     "{"五"*24}")\n'}))
    # ★ 反向守卫：**两条相邻的裸字符串语句**不得被当成拼接（`NEWLINE` 必须断开 run）。
    #   ★ 初版夹具写成 `x = ".."\ny = ".."`，中间隔着 `y` 与 `=` ⇒ 无论 NEWLINE 是否算连接符
    #   都不会连成一段，**测不到该性质**（变异 M2 因此假绿）。现改为两条**裸**字符串语句。
    cases.append((24, "两条相邻裸字符串语句不得并成拼接", PASS, "## §1\nx\n",
                  {"a.py": f'"{_A}"\n"{_B}"\n'}))
    # ★★ `P134-2` 规则 ③：`f"…"` / `r"…"` 等**带前缀**的双引号字面量原先是结构性盲区
    #   （py3.12+ 的 `f"…"` 根本不是 `STRING` token）。口径：**引号内文本** >= 58。
    _T = "报" * 60                         # 60 字文本（>= 58）
    cases.append((24, "f-string 文本长(60)", FAIL, "## §1\nx\n",
                  {"a.py": f'print(f"{_T}")\n'}))
    # ★★ **关键回归守卫**：源码长但**文本短**（插值密集）**不得**判违规。
    #   若把尺子误换成「整段源码长度」，本条必红；而历史全量扫会有 **1781 处 / 95 张卡**误报
    #   —— 那会把 K24 变成「几乎所有卡都 FAIL」，规则随即失去分辨力。
    _iv = "".join("{" + f"v{i}" + "}" for i in range(12))   # {v0}..{v11} 每个 4 字符
    _dense = 'f"' + "甲乙丙丁戊己庚辛壬癸子丑" + _iv + '"'      # 文本 12 字、源码 62 字符
    assert len(_dense) >= 60 and 12 < _K24_TEXT_MIN, (len(_dense), 12)
    cases.append((24, "f-string 源码长但文本短(插值密集)", PASS, "## §1\nx\n",
                  {"a.py": "".join(f"v{i} = {i}\n" for i in range(12)) + f"print({_dense})\n"}))
    cases.append((24, "三引号 f-string 长文本", PASS, "## §1\nx\n",
                  {"a.py": 'def f():\n    return f"""' + "长" * 60 + '"""\n'}))
    cases.append((24, "r-string 文本长(60)", FAIL, "## §1\nx\n",
                  {"a.py": f'p = r"{_T}"\n'}))
    cases.append((24, "r-string 文本短", PASS, "## §1\nx\n",
                  {"a.py": 'p = r"' + "短" * 20 + '"\n'}))
    # ★★ `P134-2` 兜底：`tokenize` 失败的文件原先**整份跳过** ⇒ 未闭合括号即可免检
    #   （★ 实测 `def f(:` 亦抛 `TokenError: unexpected EOF in multi-line statement`）。
    cases.append((24, "tokenize 失败文件：兜底抓到长串", FAIL, "## §1\nx\n",
                  {"a.py": f'x = ("{_T}"\n'}))
    #   ★★ 同理：初版只写未终止三引号（`y = """…`），**行内没有闭口引号对**
    #   ⇒ 无论有无护栏，正则都匹配不到，变异**测不出来**。现让同一行既有 `"""` 又有一对
    #   闭口长 CJK 串（括号不闭合使 tokenize 失败）⇒ 无护栏则误报、有护栏才 PASS。
    cases.append((24, "tokenize 失败文件：三引号行不误报", PASS, "## §1\nx\n",
                  {"a.py": 'x = ("""abc""", "' + "长" * 60 + '"\n'}))
    # ★★ `P134-2` **含前缀/f 式的隐式拼接**：这是本轮 25 张翻转里**占 19 张**的那条分支
    #   （初版没有夹具，变异 M-f 因此测不出来 —— 又一类「夹具没踩在性质上」）。
    #   口径：各段**文本**长度累加（f 式只算散文），>= 58 即违规。
    _A3, _B3 = "段" * 30, "乙" * 30
    cases.append((24, "含前缀拼接越线(30+f+30)", FAIL, "## §1\nx\n",
                  {"a.py": "v = 1\n" + f'x = "{_A3}" f"{{v}}" "{_B3}"\n'}))
    cases.append((24, "含前缀拼接未越线(20+f+20)", PASS, "## §1\nx\n",
                  {"a.py": "v = 1\n" + f'x = "{"段" * 20}" f"{{v}}" "{"乙" * 20}"\n'}))

    # ★★ `P134-2` **格式说明不算散文**（`{}` 计数器）：本条的散文 < 58，但若把 `:.6f` 之类
    #   算进文本就越线 —— 这正是 `dir51/calib51.py:161` 的真实形状（该行中文仅 6 字）。
    #   去掉 `field` 计数器 ⇒ 本条必红（变异 M-d）。
    #   ★★ 夹具必须**跨阈值**才检验得到变异：取散文 50 字、格式说明 3×`+.6f`=12 字
    #   ⇒ 散文 50 < 58 而「散文+说明」62 >= 58。若散文远低于阈值（初版只 13 字），
    #   去掉计数器后仍不越线，变异**测不出来**（与上一轮 M2 同一类错误）。
    _pn, _nspec = 50, 12
    assert _pn < _K24_TEXT_MIN <= _pn + _nspec, (_pn, _nspec)
    _spec = ('print(f"' + "测" * _pn + '{a:+.6f}{b:+.6f}{c:+.6f}")\n'
             "a = b = c = 0.0\n")
    cases.append((24, "f-string 格式说明不计入散文(50/62)", PASS, "## §1\nx\n", {"a.py": _spec}))


    ok = bad = 0
    with tempfile.TemporaryDirectory() as td:
        troot = pathlib.Path(td)
        for i, (k, label, expect, md, pys) in enumerate(cases):
            num = 900 + i
            cdir = troot / "revolve" / f"DIR-{num}"
            ddir = troot / f"dir{num}"
            cdir.mkdir(parents=True)
            ddir.mkdir(parents=True)
            (cdir / f"DIRECTION_rev{num}_v10.md").write_text(md, encoding="utf-8")
            if k == 22 and expect == PASS:
                (cdir / "VERDICT.md").write_text("size α 不可采用", encoding="utf-8")
            elif k == 22 and expect == FAIL:
                # ★ 有 VERDICT、且判据确实用了 α，但判决书三个词缺两个 ⇒ 真 FAIL
                (cdir / "VERDICT.md").write_text("α=0.05 的判据已算。", encoding="utf-8")
            for name, src in pys.items():
                (ddir / name).write_text(src, encoding="utf-8")
            global REVOLVE, ROOT
            old = (REVOLVE, ROOT)
            REVOLVE, ROOT = troot / "revolve", troot
            try:
                got = REGISTRY[k](Ctx(num)).status
            finally:
                REVOLVE, ROOT = old
            mark = "✓" if got == expect else "✗"
            if got == expect:
                ok += 1
            else:
                bad += 1
            print(f"  {mark} K{k:<2} {label:<28} 期望 {expect:<4} 实得 {got}")
    print(f"\n自测：{ok} 通过 / {bad} 不符（共 {len(cases)}）")
    return 0 if bad == 0 else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    # ★ 覆盖率守卫：清单声明的 ①自动化 集合必须与本脚本实现的一致（DEF-128-03）
    auto = checklist_auto()
    if auto:
        miss = sorted(set(auto) - set(REGISTRY))
        extra = sorted(set(REGISTRY) - set(auto))
        if miss or extra:
            print(f"★ 覆盖率不符：清单 ①自动化 = {sorted(auto)}")
            print(f"   未实现 = {miss}；多实现 = {extra}")
            return 2

    if a.all:
        # ★ P134-1 修：原先 `int(p.name.split("-")[1])` 在 `revolve/DIR-26M`（变体目录）上
        #   直接 `ValueError: invalid literal for int(): '26M'` ⇒ `--all` **从未跑通**。
        #   口径：本脚本按 `dir{n}/` 定位，只认**纯数字** `DIR-{n}`；变体目录（`DIR-26B`/
        #   `DIR-87c` 等）没有同号目录 ⇒ **跳过并明示**（不静默截断）。
        canon, skip = [], []
        for p in sorted(REVOLVE.glob("DIR-*")):
            if not p.is_dir():
                continue
            m = re.fullmatch(r"DIR-(\d+)", p.name)
            (canon if m else skip).append(int(m.group(1)) if m else p.name)
        nums = sorted(canon)
        if skip:
            print(f"★ 跳过 {len(skip)} 个非规范目录（无同号 `dir{{n}}/`，本脚本不覆盖）："
                  f"{', '.join(sorted(skip))}")
    elif a.dirs:
        nums = a.dirs
    else:
        ap.error("需要 --dirs / --all / --selftest")

    ks = sorted(REGISTRY)
    total = {PASS: 0, FAIL: 0, NA: 0}
    print("=" * 100)
    print(f"方向卡纪律核对（CARD-CHECKLIST 的 {len(ks)} 条 ①自动化）ROOT={ROOT}")
    print("=" * 100)
    for n in nums:
        ctx = Ctx(n)
        print(f"\n■ DIR-{n}  card={ctx.card.name if ctx.card else '(缺卡)'}"
              f"  py={len(ctx.py)}")
        if not ctx.card:
            continue
        for k in ks:
            r = REGISTRY[k](ctx)
            total[r.status] += 1
            extra = f"  ⟵ {r.reason}" if r.reason else ""
            print(f"   {r.status:<4} K{k:<2} {r.detail[:96]}{extra}")
    print("\n" + "=" * 100)
    print(f"合计：PASS {total[PASS]} · FAIL {total[FAIL]} · N/A {total[NA]}"
          f"（{len(nums)} 卡 × {len(ks)} 条）")
    return 0 if total[FAIL] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())