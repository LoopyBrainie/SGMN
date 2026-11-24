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
import collections
import json
import io
import pathlib
import re
import sys
import tokenize
import warnings

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
    15: '[窄化] 「须有 `S\\d+` **执行**它」按**同一小节内**判定 —— 停在**位置约束**上，**不判**该 `S` 是否真执行（后者见 `P130-4`）。★ 为何**不**改成**卡级**（`P133-7` 量过、裁过）：① **前提不成立** ——「`DIR-130` 的纪律已被 `S9` 守住」只在**弱**意义下为真：`§6` 禁的是「`p₃₅₀`/`p₃₀₀` **及其任一中间聚合**」，而 `S9` 的禁用词只有**5 个字面 token**（`p₃₅₀ =`/`p₃₀₀ =`/`★ 判决:`/`C130-FRAC`/`C130-PROTOCOL`）⇒ 分子类中间聚合（如「`frac=0` 的 `C` 计数」）**不在**其内 ⇒ 卡级口径会对一张**守卫不足**的卡记 `PASS`，那正是 `P133-1` 防的**假 `PASS`**（比假 `N/A` 更坏：它把缺陷读成合规）；② **实测影响面 = 1 条判决**（`DIR-130` `FAIL`→`PASS`）、**`N/A`→`PASS` = 0 条**（`N/A` ⇔ 零命中，是算术必然），且唯一被改的正是**提出该问题的那张卡** ⇒ 与「改尺子让卡变绿」同形；③ **设计上**一张卡可有多处阶段纪律，卡级会让**一个** `S` 为全卡所有纪律背书 ⇒ 分辨力塌缩（本库现只有 1 张卡有命中，故 ③ 是**设计论证、非实测**，不得记作实测）。★ `DIR-130` 的缺口是**文档**性质（`S9` 于 `v1.2` 后补、`§6` 正文未回链），已照 `DIR-126` 先例在 `revolve/DIR-130/RECHECK.md` **追加**说明，卡片与 `VERDICT` **一字未改**。',
    16: "[降级] 本条由 `①自动化` **降为 `②留痕`**：静态核对器**不判**，也不得记 PASS。判定须在 trainer 上以**子进程**跑 `precheck{n}.py` 并扫 stdout —— 但『哪些词算 formal 的判据读数』**只能由卡片在开卡时自己冻结**（如 `dir130` 的 `S9_FORBIDDEN` + `T13` 自测）。核对器若自行拟一份通用禁用词表，要么误报合规卡、要么就是把已知的 11 张卡调到全绿 —— 后者正是 `P133-1` 要防的**静默窄化**。⇒ 现由卡片自带的 `S*` 执行，核对器不判。★★ `P134-2` 已**逐卡实跑裁定**（trainer，stdout 原样留档于 `revolve/P134-2-census/`，结论见 `revolve/P134-2-PRECHECK-CENSUS.md`）：**11 张 0 违规**，**无需**任何 `RECHECK.md`。允许打印：结构事实 / 已发表常数复现 / **单个工作例**的判据量与标签 / 检测器自测（含 level·power 仿真与预期标签）；禁止：本研究自身的判据**聚合**（`C2`/`C3`/`C4`、`p₃₅₀`/`p₃₀₀`）与**整条序列/分类**（机械佐证：11 份输出中含 ≥20 个数字的行数 = 0）。★ 故『`frac` 在单个工作例里打印』**不违规** —— `原文` 括号禁的是「**整条**序列/分类」，量词是「整条」。★ 更正 `P134-1` 的**错误论断**：`dir130` 的 `frac = 0.0` 属【工作例 §6】且下一行即声明 `p₃₅₀`/`p₃₀₀` 只在 `--stage formal` 打印 ⇒ 不是「自我豁免」，是**已声明的窄边界**；原记「未裁定」现裁定为**合规**。★★ 但查出**两处与 `K16` 无关的真实缺陷**：① `precheck125`/`precheck126` 的 `S12` 盲态前置条件（新剂量须 0 cell）在 `formal` 之后**永不可能再通过** ⇒ 重跑必 `rc=1`，**不是卡的问题**；② **8/11 根本没有 `--stage` 开关**，且至少 5 张读 `formal` 产物（`readings.json`/`out/cells`）⇒ 它们在**开卡时刻根本跑不起来**，`K16` 对它们只是**空洞成立**（纪律成立是因为检查从未在该时刻发生）⇒ 已派生 `P134-3`。",
    18: "[窄化] 仅当卡片真的在说**继承一个带定义域限制的求解器**时才适用；`û_s=0` 的『夹紧层』是**测量端点的定义**，不是求解器夹紧（首版按字面匹配 clamp/夹紧 ⇒ 4/5 误报）",
    22: "[窄化] 适用门**不再按符号搜 α**（`P133-3` 实测：本框架的 α 首先是**写入率/门控参数** —— `SGMN_v2.4_架构设计.md` §7.2 第 1 行「写入映射与门控」即「$\\alpha$ 的产生规则」，装置里是 `G = α(Wk−v)kᵀ + λW`，代码写作 `ALPHA`/`ALPHA, LAM`），改为要求**显著性用法的证据**：① 先抹掉 LaTeX 的 `\\alpha`/`$\\alpha$`；② α 被赋 **(0, 0.5)** 内的值（**排除 0** —— `decay` 臂把写率置零是控制臂，且零假阳性率的检验不存在）；③ `p` 与 α 直接比较；④ `p` 与具体水平比较（`p<0.05`/`p ≤ 0.0012`）；⑤ 显式点名「显著性水平」。★ 旧口径命中 **48** 张，逐张核过其中 **34 张是假阳性**（**17** 张只是 LaTeX 架构符号、**16** 张是写规则里的 α、**1** 张是 `α=0` 控制臂）⇒ 误报率 **71%**，FAIL 已失去分辨力；新口径命中 **15** 张，逐张核过**都是真显著性水平**（`DIR-19/20/21/23/92/104/112/113/114/120/121/122/123/124/126`）。★★ 另修一处**假阴性**：旧门只认 ASCII 的 `p<0…`，**不认 `≤`** ⇒ `DIR-112`（两条判据是 `p ≤ 0.0012` / `p≤0.035`）被**静默判 N/A** —— 与 `K15` 的假 N/A 同一物种。★ **更正一处我自己的错误论断**：`DIR-104`（`ALPHA = 0.05`）**不是**假阴性的例子（它被旧门看见靠的是自己卡里的 LaTeX `\alpha`，即假阳性那条路）；实测本库「大写 `ALPHA` 作显著性却对旧门不可见」的卡**零张**（`DIR-46/47/49/86/106/118` 的 `ALPHA` 全是写率/装置参数）。穷举型卡片没有抽样误差 ⇒ N/A",
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


_EMPH_RE = re.compile(r"[*_`]")


def _de_emph(s):
    """去掉 markdown 的强调/代码标记（`*` `_` 反引号）；**只用于匹配**，不改报告文本。

    ★ 为什么必须（`P133-4`）：`K15` 的谓词是字面 `不得打印|不得输出|禁止打印`，而
      `DIR-130` 的卡片写的是 `**不得**打印属于判据的**聚合读数**` —— markdown 强调把「不得」
      与「打印」**劈开**，谓词命中 **0** 处 ⇒ 全库 **103/103 判 `N/A`**。而 `N/A` 在报表里
      **最好看**（不是 `FAIL`、不被计数、不引起注意）⇒ **守卫静默失效**。
      更刺眼的是：`K15` 本身就是 `P130-4`（「阶段纪律必须配机械守卫」）的工具化产物，
      却对自己教训的原始卡片是**盲的**。
    ★ 单调性（**实测**，不是论证）：全库 122 个有卡目录里，去强调使命中数**只增不减**，
      且**唯一**新增命中是 `DIR-130`（0 → 2）。故本改动**不可能**把 `P133-2` 实测的
      6 个「包含/含有」假阳性（`DIR-65/88/91/31/26S/46`）拉回来 —— 那 6 个是**放宽词序**
      （`(不得|禁止|…)(…)(打印|输出|含)`）造成的；本改动只抹掉**夹在词间**的标记，
      **不改词序、不改词表**。
    """
    return _EMPH_RE.sub("", s)


def k15(ctx):
    """K15 「阶段纪律」必须配机械守卫：每处『不得打印/输出』须有 S 编号执行。"""
    interp = "按原文执行：`不得打印|不得输出` 的每一处所在小节内必须出现 `S\\d+`"
    hits = []
    for head, body in ctx.sections():
        for ln in body.splitlines():
            de = _de_emph(ln)          # ★ P133-4：先抹掉 `**`/反引号等再匹配
            if re.search(r"不得打印|不得输出|禁止打印", de):
                has_s = (re.search(r"\bS\d+\b", de)
                         or re.search(r"\bS\d+\b", _de_emph(body)))
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


# ★★ `P133-3`（Cycle 252）：K22 的**适用门**原先直接搜符号 `α` / `alpha` —— 那是错的，
#   因为在**本框架**里 α 首先不是显著性水平，而是**写入率/门控参数**：
#   `SGMN_v2.4_架构设计.md` §7.2 第 1 行「写入映射与门控」的括号里写的就是「$\alpha$ 的产生规则」，
#   装置里它是 `G = α(Wk−v)kᵀ + λW` 的 α，代码里写作 `ALPHA`/`ALPHA, LAM`。
#   只按符号搜 ⇒ **两个方向同时出错**（实测，见 INTERP[22]）：
#     ① 假阳性 34/48 —— 17 张只是 LaTeX 的 `\alpha`、16 张是写规则里的 α、1 张是 `α=0` 控制臂；
#     ② 假阴性 —— 旧门只认 ASCII 的 `p<0…`，**不认 `≤`**，且无 ASCII 折叠。`DIR-112` 的两条判据
#        就是 `p ≤ 0.0012` / `p≤0.035` ⇒ 旧门**静默判 N/A**（与 `K15` 的假 N/A 同一物种）。
#        ★ 更正一个**我一度写进草稿的错误论断**：`DIR-104`（`ALPHA = 0.05`）**不是**这个假阴性的例子 ——
#        它能被旧门看见，但靠的是它自己卡片里的 LaTeX `\alpha`（即假阳性的那条路）。
#        实测本库中「大写 `ALPHA` 作显著性水平却对旧门不可见」的卡 **一张也不存在**
#        （`DIR-46/47/49/86/106/118` 的 `ALPHA` 全是写率/装置参数）。
#   ⇒ 现改为要求「**显著性用法**的证据」，而不是「符号出现过」。
_LATEX_ALPHA = re.compile(r"\\alpha\b|\$\s*\\?alpha\s*\$")


def _uses_sig_alpha(text):
    """K22 的适用门：α 是否被当作**显著性水平**使用（而不是架构的写入率/门控参数）。

    判据（任一成立即算「真的用了 α」）：
      (a) α 被赋一个 **(0, 0.5)** 的值 —— ★ **排除 0**：写率置零是**控制臂**（`decay` 臂
          `G = 0·(Wk−v)kᵀ + λW`），不是 α 水平；而 α 也**不可能**是 0（零假阳性率的检验不存在）。
      (b) `p` 与 α **直接比较**（`p < α` / `p ≤ α`，容忍中间的 markdown/LaTeX 记号）。
      (c) `p` 与**具体水平**比较（`p<0.05`、`p ≤ 0.0012`）。
      (d) 显式点名「**显著性水平/阈值/水准**」。

    ★ 设计取舍：宁可**窄**（漏判一些表格式写法，如 `| α | 0.05 |`）也不要**宽** ——
      本条是**文档要求**，不是安全门；它的历史误报率（34/48）已经高到让 FAIL 失去分辨力。
      这正是本卡 `判读` 里那条「穷举型卡片没有抽样误差 ⇒ N/A」的同一个教训：**适用门比判据本身更容易错**。
    """
    s = _LATEX_ALPHA.sub(" ", text)          # 先抹掉 LaTeX 的架构符号 `\alpha` / `$\alpha$`
    for m in re.finditer(r"(?:α|\balpha\b|ALPHA)\s*=\s*([0-9]*\.?[0-9]+)", s):
        try:
            if 0.0 < float(m.group(1)) < 0.5:
                return True
        except ValueError:
            pass
    if re.search(r"\bp\s*(?:<|≤|<=|\\le)\s*[*`$_{} ]*(?:α|\balpha\b|ALPHA)", s):
        return True
    if re.search(r"\bp\s*(?:<|≤|<=|\\le)\s*0?\.\d+", s):
        return True
    if re.search(r"显著性(?:水平|阈值|水准)", s):
        return True
    return False


def k22(ctx):
    """K22 判决表必须显式含「若 size(判据) > α + 容差 ⇒ 记不可采用」。"""
    interp = INTERP[22]
    if not _uses_sig_alpha(ctx.text):
        return R(NA, "卡片未把 α 用作显著性水平（写率/门控参数不算）", interp,
                 reason="架构的 α 是写入率；穷举或纯描述型判据没有抽样误差 ⇒ 不存在 size-vs-α 的取舍")
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
    # ★★ `P133-4`（Cycle 253）：`K15` 的**假 `N/A`** —— markdown 强调把「不得」与「打印」**劈开**。
    #   下面的夹具在**旧实现**（无去强调）上会得到 `N/A`，与新期望值不同 ⇒ 真踩在改动上。
    #   ① 核心缺陷：`**不得**打印`（`DIR-130` 的原样写法；旧实现命中 0 ⇒ 静默 N/A）
    cases.append((15, "**不得**打印（强调夹在词间）", FAIL,
                  "## §6\n**不得**打印属于判据的**聚合读数**。\n", {}))
    #   ② 反引号同理（`不得`输出）—— 证明抹掉的是**标记**而不是只认星号
    cases.append((15, "`不得`输出（反引号夹在词间）", FAIL,
                  "## §6\n`不得`输出判决行。\n", {}))
    #   ③ 单星号强调（`*不得*`）—— 三段强调写法都覆盖
    cases.append((15, "*不得*打印（单星号）", FAIL,
                  "## §6\n*不得*打印 C2/C3。\n", {}))
    #   ④ **防矫枉过正**：去强调后仍须能判 PASS（S 编号在**同一节**内）
    cases.append((15, "**不得**打印且同节点名 S9", PASS,
                  "## §6\n**不得**打印 `C2`/`C3`。\n本节由 `S9` 机械执行。\n", {}))
    #   ⑤ **防漏判**：强调与 S 编号**都在同一行**
    cases.append((15, "**不得**打印且同行点名 S9", PASS,
                  "## §6\n**不得**打印判决行（`S9`）。\n", {}))
    #   ⑥ ★★ `P133-7`（Cycle 254）**裁定的机械钉子**：纪律写在 `§6`，而 `S9` 只出现在**别的节**
    #      ⇒ 现行**小节级**判 `FAIL`。★ 若日后有人把 `k15` 改成**卡级**（整张卡任一 `S\d+` 即算守住），
    #      本夹具会由 `FAIL` 变 `PASS` ⇒ **测试立刻变红**，裁定不会被静默推翻。
    cases.append((15, "纪律在某节、S 只在他节（P133-7 裁定仍 FAIL）", FAIL,
                  "## §5 Stage A\n| `S9` | 阶段纪律：precheck 不得含判据读数 |\n"
                  "\n## §6 工作例\n**不得**打印聚合读数。\n", {}))

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
    # ★★ `P133-3`（Cycle 252）：K22 适用门的**假阳性/假阴性**夹具。
    #   第 6 元 = 显式 VERDICT 文本（本块新增的机制：否则 NA 一律无 VERDICT，
    #   两个 NA 原因就分不开了 —— 旧夹具只能表达「卡里没 α」这一种 N/A）。
    #   ① 架构符号：LaTeX `$\alpha$`（= 「写入映射与门控」的参数），不是显著性
    cases.append((22, "架构符号 $\\alpha$ 不是显著性", NA,
                  "## §1\n判据权威：`SGMN_v2.4_架构设计.md` §7.2 第 1 行\n"
                  "> 「写入映射与门控」（$\\alpha$ 的产生规则 / 内容选择收益）\n",
                  {}, "本卡穷举全部 112 个结构。"))
    #   ② 写规则里的 α：`G = α(Wk−v)kᵀ + λW`，α 是写入率
    cases.append((22, "写规则里的 α=1.0 不是显著性", NA,
                  "## §2\n装置：`G = α(Wk−v)kᵀ + λW`，参数 `η=0.2, λ=0.05, α=1.0`。\n",
                  {}, "逐位复现 `R1_grad.py`。"))
    #   ③ `α=0` 是 decay 控制臂（写率置零），不是 α 水平
    cases.append((22, "α=0 控制臂不是显著性", NA,
                  "## §2\n`decay` 臂：`G = 0·(Wk−v)kᵀ + λW`（`α=0`，同样步数）。\n",
                  {}, "隔离 §5.5 的衰减项。"))
    #   ④ 大写 `ALPHA, LAM = 1.0, 0.05` —— 装置参数（**回归守卫**：旧门也判 N/A）
    cases.append((22, "ALPHA, LAM 装置参数不是显著性", NA,
                  "## §2\n`SGMNStack(model, MOUNT, d, d, R, ALPHA, LAM, 0.01, ...)`\n",
                  {}, "写入规则逐字不动。"))
    #   ⑤ 真显著性：`α = 0.05` 且判据以 p 为门 ⇒ 本条必须仍判 FAIL（防**矫枉过正**）
    cases.append((22, "真显著性 α=0.05 仍须判 FAIL", FAIL,
                  "## §3\n检验：精确置换（枚举 `9! = 362880`），单侧，`α = 0.05`。\n",
                  {}, "J126-2：ρ=-0.6，p=0.053999 > 0.05 ⇒ 未拒绝。"))
    #   ⑥ **假阴性**回归：旧门只认 `p<0`、不认 `≤` ⇒ `p ≤ 0.0012` 被静默判 N/A
    #      （实测对应 `DIR-112`；这是与 `K15` 同物种的静默失效）
    cases.append((22, "p ≤ 具体水平（旧门漏判）", FAIL,
                  "## §3\n两个量同时成立 ⇒ 联合 `p ≤ 0.0012`。\n",
                  {}, "联合 p 值已算。"))

    # ★★ `P133-3` **可证伪性见证**（`DEF-129-01`）：用**旧实现**的适用门跑同一段文本，
    #   证明上面的夹具真的踩在改动上，而不是“新旧都这么判”。旧门原文 = `P133-3` 之前的两行。
    _old_gate = lambda t: (bool(re.search(r"\bα\b|alpha", t))
                           or bool(re.search(r"p\s*<\s*0", t)))
    #   ★★ `P133-4`：`K15` 的**旧实现**（无去强调）—— 用来证明 K15 的强调夹具真踩在差异上。
    _old_k15 = lambda t: bool(re.search(r"不得打印|不得输出|禁止打印", t))
    _new_k15 = lambda t: bool(re.search(r"不得打印|不得输出|禁止打印", _de_emph(t)))
    #   每条 = (标签, 文本, 旧实现, 新实现)；两者**必须不同**，否则夹具未踩在差异上。
    _wit = [
        ("架构符号 $\\alpha$", "（$\\alpha$ 的产生规则）", _old_gate, _uses_sig_alpha),
        ("写规则 α=1.0", "`G = α(Wk−v)kᵀ + λW`，`α=1.0`", _old_gate, _uses_sig_alpha),
        ("α=0 控制臂", "`α=0`，同样步数", _old_gate, _uses_sig_alpha),
        ("p ≤ 0.0012", "联合 `p ≤ 0.0012`", _old_gate, _uses_sig_alpha),
        ("K15 **不得**打印", "**不得**打印聚合读数。", _old_k15, _new_k15),
        ("K15 `不得`输出", "`不得`输出判决行。", _old_k15, _new_k15),
        ("K15 *不得*打印", "*不得*打印 C2/C3。", _old_k15, _new_k15),
    ]

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
        for i, case in enumerate(cases):
            k, label, expect, md, pys = case[:5]
            v_override = case[5] if len(case) > 5 else None
            num = 900 + i
            cdir = troot / "revolve" / f"DIR-{num}"
            ddir = troot / f"dir{num}"
            cdir.mkdir(parents=True)
            ddir.mkdir(parents=True)
            (cdir / f"DIRECTION_rev{num}_v10.md").write_text(md, encoding="utf-8")
            if v_override is not None:
                (cdir / "VERDICT.md").write_text(v_override, encoding="utf-8")
            elif k == 22 and expect == PASS:
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

        # ★★ `P133-3` 可证伪性见证：旧门与新门在上述文本上**必须不同**。
        #   若两者一致 ⇒ 夹具没踩在差异上（变异**测不出来**），此处红 = 夹具无效。
        for _lbl, _txt, _ofn, _nfn in _wit:
            _o, _n = _ofn(_txt), _nfn(_txt)
            _ok = (_o != _n)
            ok, bad = (ok + 1, bad) if _ok else (ok, bad + 1)
            print(f"  {'✓' if _ok else '✗'} 见证 {_lbl:<16} 旧门={_o} 新门={_n}"
                  + ("  ⇒ 夹具踩在差异上" if _ok else "  ⇒ 夹具**未**踩在差异上"))
    # ★★★ `P133-8`：守卫的**覆盖度**（`S*` 存在 ≠ `S*` 够用）。
    #   ★ 这 4 条用**合成样本**钉**工具**的正确性；**实况**由 `--probe-guard-coverage` 报告
    #     （实况会随 `DIR-130` 被修好而变 ⇒ 不能写成夹具，否则修好即报红）。
    ok8 = bad8 = 0
    with tempfile.TemporaryDirectory() as td8:
        t8 = pathlib.Path(td8)

        # ① 词表判据**必然**漏掉语义集合里不含词表 token 的形式 ⇒ 「有守卫」≠「够用」可被检出
        _src8 = t8 / "synthetic_guard.py"
        _src8.write_text('FORB = ("a =", "b =")\n', encoding="utf-8")
        _f8 = _forbidden_from_source(_src8, "FORB")
        _forms8 = ("a = 1", "a=1", "n_pos = 12345")
        _hit8 = [f for f in _forms8 if any(w in f for w in _f8)]
        _ok8 = (len(_f8) == 2 and len(_hit8) == 1)
        ok8, bad8 = (ok8 + 1, bad8) if _ok8 else (ok8, bad8 + 1)
        print(f"  {'✓' if _ok8 else '✗'} P133-8-1 词表判据漏掉不含词表 token 的语义形式"
              f"  命中 {len(_hit8)}/{len(_forms8)}")

        # ② 带身份载体：**形式无关**地拦下全部 6 条被禁形式（remedy 可被检出）
        _res8 = _tagged_forms_rendered()
        _blk8 = sum(1 for _, o, _ in _res8 if o)
        _ok8 = (_blk8 == len(GUARD_FORMS_DIR130) == 6)
        ok8, bad8 = (ok8 + 1, bad8) if _ok8 else (ok8, bad8 + 1)
        print(f"  {'✓' if _ok8 else '✗'} P133-8-2 带身份载体拦下被禁形式 {_blk8}/6")

        # ③ 循环见证检测：以禁用词表为迭代源造见证 ⇒ 必须被检出
        _src8b = t8 / "synthetic_circ.py"
        _src8b.write_text(
            'FORB = ("x", "y")\n'
            'def t1():\n'
            '    hits = {w: det("a" + w) for w in FORB}\n'
            '    return (True, str(hits))\n'
            'def t2():\n'
            '    return (True, "unrelated")\n', encoding="utf-8")
        _c8 = _circular_witness(_src8b, "FORB")
        _ok8 = (_c8 == [("t1", "comprehension")])
        ok8, bad8 = (ok8 + 1, bad8) if _ok8 else (ok8, bad8 + 1)
        print(f"  {'✓' if _ok8 else '✗'} P133-8-3 检出循环见证 {_c8}")

        # ④ 循环见证检测**不得**误伤非循环的自测（防矫枉过正）
        _src8c = t8 / "synthetic_clean.py"
        _src8c.write_text(
            'FORB = ("x", "y")\n'
            'def t1():\n'
            '    return (True, "clean" + str(FORB))\n', encoding="utf-8")
        _c8b = _circular_witness(_src8c, "FORB")
        _ok8 = (_c8b == [])
        ok8, bad8 = (ok8 + 1, bad8) if _ok8 else (ok8, bad8 + 1)
        print(f"  {'✓' if _ok8 else '✗'} P133-8-4 非循环自测不误报 {_c8b}")

    ok += ok8
    bad += bad8

    # ★★★ `P133-10`：普查探针的**检测器**正确性。
    #   ★★ 见证集**手工书写**、**不由量词表/禁止语表反推** —— 否则重演 `P133-8` §二
    #      的「自证循环」（见证 = 被测判据 ⇒ 只证明「检测器非空转」）。
    #   ★ 夹具钉的是**工具**的行为；**实况**（哪条纪律是语义型）由普查 + 人工标注承担。
    ok10 = bad10 = 0

    def _t10(label, ok_, info=""):
        nonlocal ok10, bad10
        ok10, bad10 = (ok10 + 1, bad10) if ok_ else (ok10, bad10 + 1)
        print(f"  {'✓' if ok_ else '✗'} P133-10 {label} {info}")

    # ① 真命中：来源界定 + 全称量词（`K16` 的原句形态）
    _h = _sem_hit("**`precheck` 阶段不得打印属于 `formal` 的任何判据读数**")
    _t10("-1 命中「任何判据读数」（来源+全称）",
         bool(_h) and _h[0] == "任何" and "SRC" in _h[2] and "CLS" in _h[2], f"{_h}")

    # ② **假阳性防护**：本库的「逐」多是「逐位/逐字」复合词，**不是**全称量词
    _h = _sem_hit("冻结卡里不得要求浮点逐位相等")
    _t10("-2 「逐位相等」**不**命中（逐位≠量词）", _h is None, f"{_h}")

    # ③ **假阳性防护**：「均值」里的「均」是**名词的一部分**，不是量词
    _h = _sem_hit("禁止只对 5 张图的均值做一次 3se")
    _t10("-3 「均值」里的「均」**不**命中", _h is None, f"{_h}")

    # ④ **假阳性防护**：有量词、无禁止语 ⇒ 不是「禁止语句」
    _h = _sem_hit("任何判据读数都要在 VERDICT 里报告")
    _t10("-4 无禁止语**不**命中", _h is None, f"{_h}")

    # ⑤ **假阳性防护**：有禁止语、无量词 ⇒ 不是「全称型禁止语句」
    _h = _sem_hit("不得打印 `C2` 读数")
    _t10("-5 无量词**不**命中", _h is None, f"{_h}")

    # ⑥ 聚合层级（`DIR-130` 的「跨读数的聚合」物种）
    _h = _sem_hit("禁止输出一切跨读数的聚合")
    _t10("-6 命中「一切跨读数的聚合」（聚合层级）",
         bool(_h) and _h[0] == "一切" and "AGG" in _h[2], f"{_h}")

    # ⑦ markdown 强调把词劈开 ⇒ 仍须命中（`P133-4` 同物种：`N/A` 静默失效）
    _h = _sem_hit("**不得**打印**任何** `C2` 读数")
    _t10("-7 强调劈开仍命中（`P133-4` 同物种）",
         bool(_h) and _h[0] == "任何", f"{_h}")

    # ⑧ ★★★ **符号二义抓不到**（`K22` 的 `α`）：无全称量词 ⇒ 量词普查漏掉它
    #     ⇒ 证明第五段（符号扫描）**不可删**。
    _alpha_line = "判决表必须显式含「若 size(判据) > α + 容差 ⇒ 记不可采用」"
    _h = _sem_hit(_alpha_line)
    _t10("-8 符号二义**抓不到** ⇒ 符号扫描不可删",
         _h is None and "α" in re.findall(r"[α-ωΑ-Ω]", _de_emph(_alpha_line)), f"{_h}")

    # ⑨ `LIT`（显式 token 列举）**只能在原行**判 —— 去强调后反引号已抹掉
    #     （实测踩过：在去强调后的行上判 `LIT` 恒为 0）
    _h = _sem_hit("不得含**任何** `C2`/`C3` 读数")
    _t10("-9 `LIT`（`C2`/`C3` 列举）被认出",
         bool(_h) and _h[0] == "任何" and "LIT" in _h[2], f"{_h}")

    # ⑩ 被量化对象的短语：两侧都要抽（对象可能在量词**前**）
    _lo, _ri = _object_phrase("Stage A 守卫断言冻结任一阻断项失败",
                               _de_emph("Stage A 守卫断言冻结任一阻断项失败").find("任一"))
    _t10("-10 对象短语两侧抽出",
         "断言" in _lo and "失败" in _ri, f"左={_lo!r} 右={_ri!r}")

    # ⑪ ★★★ **完备性夹具**：本普查的三个**指定候选**必须**全部**被机械普查抓到。
    #     （`K15` 的「每处纪律」= 出现位置量词；`K16` 的「任何判据读数」= 来源+全称；
    #       `K22` 的 `α` = 符号二义 ⇒ **抓不到** ⇒ 由第五段符号扫描负责。）
    #     ★ 这条夹具的作用：**量词表再漏一族，三个候选里就有一个抓不到** ⇒ 立刻报红。
    _k15 = "每处 `不得打印` / `不得输出` 都必须有一条 `S\\d+` 断言**执行**它"
    _h = _sem_hit(_k15)
    _t10("-11 `K15` 的「每处」被量词表抓到",
         bool(_h) and _h[0] == "每处", f"{_h}")

    # ⑫ ★ 三个指定候选里，`K22` 的 `α` **必须**由**第五段符号扫描**抓到
    #     （而**不是**由量词普查抓到 —— 否则两个机制会互相顶替、掩盖漏检）。
    _alpha_blocks = [b for b in _checklist_blocks() if "α" in _de_emph(b[2])]
    _t10("-12 `K22` 的 `α` 由**符号扫描**抓到（量词普查抓不到它）",
         bool(_alpha_blocks) and all(_sem_hit(b[2]) is None or "SYM" not in _sem_hit(b[2])[2]
                                     for b in _alpha_blocks if b[0] == 22),
         f"K 条目 {sorted({b[0] for b in _alpha_blocks})}")

    # ⑬ ★★★ **已知正例夹具**：`DIR-130` `v1.3` 的「包括但不限于 …」式**开放列举**
    #     —— 这条**已知欠定义**（`P133-8` 实测：`v1.2` 的「及其任一中间聚合」不可执行）。
    #     ★ 该卡已**冻结**（`v1.3` 是终版）⇒ 可以写成夹具（不像 `--probe-guard-coverage`
    #       的实况那样会随修复而变 ⇒ 那里**不能**写夹具）。
    _d130 = cc_path = ROOT / "revolve" / "DIR-130" / "DIRECTION_rev130_v13.md"
    if _d130.exists():
        _ln130 = [l for l in _d130.read_text(encoding="utf-8").splitlines()
                  if "包括但不限于" in l]
        _h = _sem_hit(_ln130[0]) if _ln130 else None
        _t10("-13 `DIR-130` 的「包括但不限于」被认出（已知正例）",
             bool(_h) and _h[0] == "一切" and "AGG" in _h[2], f"{_h}")
    else:
        _t10("-13 `DIR-130` 的「包括但不限于」被认出（已知正例）", False, "缺 v1.3 卡片")

    ok += ok10
    bad += bad10

    # ------------------------------------------------------------------
    # ★ `P133-15`（Cycle 259）`K26` 探针夹具：8 条，**全部纯字符串**（不落盘）。
    #   目的：让「实测 ≠ 声明」可**证伪** —— 若 `_k26_state` 读了 `*_decl`，
    #   夹具 `-K26-9` 必红。
    # ------------------------------------------------------------------
    ok15 = bad15 = 0

    def _t15(label, got, expect):
        nonlocal ok15, bad15
        _ok = (got == expect)
        ok15, bad15 = (ok15 + 1, bad15) if _ok else (ok15, bad15 + 1)
        print(f"  {'✓' if _ok else '✗'} {label:<44} 期望 {expect!r} 实得 {got!r}")

    _MOD3 = 'FORB = ["聚合", "中间", "合并"]\n'
    _t15("-K26-1 模块级 3 项字符串表 ⇒ 1 个词表",
         _k26_ast_lit_lists(_MOD3), 1)
    _t15("-K26-2 只有 2 项 ⇒ 0（阈值 min_len=3，参数表不算词表）",
         _k26_ast_lit_lists('PAIR = ("a", "b")\n'), 0)
    _t15("-K26-3 3 项但非字符串 ⇒ 0（数字/flag 表不是禁止集）",
         _k26_ast_lit_lists('XS = [1, 2, 3]\n'), 0)
    _t15("-K26-4 3 项字符串但在**函数体内** ⇒ 0（判据词表必须模块级冻结）",
         _k26_ast_lit_lists('def f():\n    L = ["a", "b", "c"]\n    return L\n'), 0)
    _t15("-K26-5 载体类 + 审计函数都被认出",
         _k26_ast_carriers('class Agg:\n    pass\ndef s10_number_audit(x):\n    return True\n'),
         (["Agg"], ["s10_number_audit"]))

    _py_ok = [("x.py", _MOD3 + 'class Agg:\n    pass\n')]
    _n = _k26_nets("`S9` 禁用词 表 + 形式无关 + 源头 让被禁量不可打印", _py_ok)
    _t15("-K26-6 声明齐 + 实测齐 ⇒ 四个布尔全 True",
         (_n["lit_meas"], _n["lit_decl"], _n["free_meas"], _n["free_decl"]),
         (True, True, True, True))
    _n2 = _k26_nets("白名单 定义如下（未提任何 S 号守卫）", [])
    _t15("-K26-7 只说「白名单」而无 `S\\d+` ⇒ 字面(声) False（防当标点用）",
         _n2["lit_decl"], False)
    _n3 = _k26_nets("本纪律形式无关（但没说从哪一步掐断）", [])
    _t15("-K26-8 只说「形式无关」而无源头动作 ⇒ 形式无关(声) False",
         _n3["free_decl"], False)

    # ★★★ 本探针的**核心**可证伪点：只会声明、没有实测 ⇒ 必须判「零道」
    _only_decl = {"lit_meas": False, "lit_decl": True,
                  "free_meas": False, "free_decl": True}
    _t15("-K26-9 ★只会声明、零实测 ⇒ ZERO（**不得**判 TWO）",
         _k26_state(_only_decl, False, False), ("ZERO", "★★零道（实测）⇒ **未配网**"))
    _both = {"lit_meas": True, "lit_decl": False, "free_meas": True, "free_decl": False}
    _t15("-K26-10 两道实测齐（即使零声明）⇒ TWO",
         _k26_state(_both, False, False)[0], "TWO")
    _t15("-K26-11 只有字面实测 ⇒ ONE（只有一道字面网 = 未配网）",
         _k26_state({"lit_meas": True, "lit_decl": True,
                     "free_meas": False, "free_decl": True}, False, False)[0], "ONE")
    _t15("-K26-12 明示不可枚举 + 理由 ⇒ NONENUM（压过一切网读数）",
         _k26_state(_both, True, True)[0], "NONENUM")
    _t15("-K26-13 明示不可枚举但**无理由** ⇒ 不算合法出口（步骤 3 要求写明为什么）",
         _k26_state(_both, True, False)[0], "TWO")
    _ne_ev = _k26_nonenum("本纪律**不可枚举**\n因为禁止对象是推论方式")
    _t15("-K26-14 ③ 触发行被返回（让触发词可被人工复核）",
         (_ne_ev[0], _ne_ev[1], _ne_ev[2][0] if _ne_ev[2] else None),
         (True, True, "不可枚举"))
    # ★★★ 实测踩到的**误报**（`DIR-38/39/40/104`）：说的是**测量**不可判定，不是禁止集不可枚举。
    _w = _k26_nonenum("- ★ 不可判定：若 C 与 D 同号则本判据不可判定")
    _t15("-K26-15 ★「该判据不可判定」⇒ 强信号 False（不翻转状态）",
         (_w[0], _w[1], _w[2][0] if _w[2] else None), (False, False, "弱:不可判定"))
    _t15("-K26-16 弱信号不得让卡片走 ③ 出口",
         _k26_state(_only_decl, _w[0], _w[1])[0], "ZERO")

    ok += ok15
    bad += bad15
    print(f"   （其中 `P133-15` 的 `K26` 探针夹具 {ok15 + bad15} 条）")
    print(f"\n自测：{ok} 通过 / {bad} 不符（共 {len(cases)} 夹具 + {ok8 + bad8} 条 P133-8 探针夹具"
          f" + {ok10 + bad10} 条 P133-10 探针夹具 + {ok15 + bad15} 条 P133-15 探针夹具）")
    return 0 if bad == 0 else 1


# ---------------------------------------------------------------------------
def canonical_nums():
    """规范 `DIR-{n}/` 编号（升序）与被跳过的变体目录名（升序）。

    ★ `P134-1` 修：原先 `int(p.name.split("-")[1])` 在 `revolve/DIR-26M`（变体目录）上
      直接 `ValueError: invalid literal for int(): '26M'` ⇒ `--all` **从未跑通**。
      口径：本脚本按 `dir{n}/` 定位，只认**纯数字** `DIR-{n}`；变体目录（`DIR-26B`/
      `DIR-87c` 等）没有同号目录 ⇒ **跳过并明示**（不静默截断）。
    ★ `P133-7`：抽成函数，供 `--all` 与 `--probe-k15-scope` **共用**，避免两处枚举漂移。
    """
    canon, skip = [], []
    for p in sorted(REVOLVE.glob("DIR-*")):
        if not p.is_dir():
            continue
        m = re.fullmatch(r"DIR-(\d+)", p.name)
        (canon if m else skip).append(int(m.group(1)) if m else p.name)
    return sorted(canon), sorted(skip)


def probe_k15_scope():
    """`P133-7` 探针：对照 `K15` 的**小节级**（现行）与**卡级**（候选）两种口径。

    只**报告**，不改变 `k15()` 的判定 —— 用途是让「改口径会影响哪些判决」可复算，
    而不是拿影响面去反推该不该改（那只会在两个方向上都自证）。
    """
    nums, skip = canonical_nums()
    print("=" * 100)
    print("K15 口径对照探针（小节级 = 现行；卡级 = 候选：整张卡任一 `S\\d+` 即为守住）")
    print("=" * 100)
    if skip:
        print(f"★ 跳过 {len(skip)} 个非规范目录：{', '.join(skip)}")
    n_cards = n_hit_cards = 0
    deltas = []
    for n in nums:
        ctx = Ctx(n)
        if not ctx.card:
            continue
        n_cards += 1
        hits = []
        for head, body in ctx.sections():
            for ln in body.splitlines():
                de = _de_emph(ln)
                if re.search(r"不得打印|不得输出|禁止打印", de):
                    has_s = bool(re.search(r"\bS\d+\b", de)
                                 or re.search(r"\bS\d+\b", _de_emph(body)))
                    hits.append((head, ln.strip(), has_s))
        if not hits:
            continue
        n_hit_cards += 1
        card_s = sorted(set(re.findall(r"\bS\d+\b", _de_emph(ctx.text))),
                        key=lambda x: int(x[1:]))
        sec_ok = all(h[2] for h in hits)
        card_ok = bool(card_s)
        print(f"\n■ DIR-{n}")
        print(f"   命中 {len(hits)} 处 / 分布在 {len({h[0] for h in hits})} 个小节")
        print(f"   卡内 `S` 编号 {len(card_s)} 个：{', '.join(card_s) or '无'}")
        print(f"   小节级 = {'PASS' if sec_ok else 'FAIL'}    卡级 = {'PASS' if card_ok else 'FAIL'}")
        for head, ln, hs in hits:
            print(f"     [{'本节有S' if hs else '本节无S'}] {head[:30]} | {ln[:58]}")
        if sec_ok != card_ok:
            deltas.append((n, sec_ok, card_ok, len(hits), len(card_s)))
    print("\n" + "=" * 100)
    print(f"有命中的卡：**{n_hit_cards}** 张（其余 {n_cards - n_hit_cards} 张零命中 ⇒ 两口径同为 `N/A`）")
    if not deltas:
        print("两种口径全库**无差异**")
    else:
        print(f"口径差异：**{len(deltas)}** 张")
        for n, s_, c_, nh, ns in deltas:
            print(f"   DIR-{n}：小节级 {'PASS' if s_ else 'FAIL'} → 卡级 {'PASS' if c_ else 'FAIL'}"
                  f"   （命中 {nh} 处；卡内 {ns} 个 S）")
    print("★ 只报告，不改变 `k15()` 判定。")
    return 0


# ---------------------------------------------------------------------------
# ★ P133-8：守卫的**覆盖度**（`S*` 存在 ≠ `S*` 够用）
# ---------------------------------------------------------------------------

#: `DIR-130` §6 的被禁形式集合（6 条 = 2 条现行命中 + 4 条现行漏掉）。
#: ★ 这 6 条**按语义**挑（§6 禁的是「跨结构聚合」），**不是**按现行词表挑
#:   —— 若由词表自己出题，连「漏」都测不出来（自证循环，见 `_circular_witness`）。
GUARD_FORMS_DIR130 = (
    ("p₃₅₀ = 0.0",                    "终值（有空格）"),
    ("p₃₀₀ = 0.0",                    "终值（有空格）"),
    ("p₃₅₀=0.0",                      "终值（无空格）"),
    ("frac>0 的 (sub,C) 对数 = 12345", "分子（跨结构计数）"),
    ("可算 (sub,C) 对数 = 18000",      "分母（跨结构计数）"),
    ("k=3: frac==0 的 C 有 84 个",     "逐 k 计数"),
)


def _forbidden_from_source(path, name="S9_FORBIDDEN"):
    """从**源码**用 `ast` 取禁用词表（不 `import`；冻结文件只读）。"""
    try:
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets):
            try:
                return tuple(ast.literal_eval(n.value))
            except (ValueError, TypeError):
                return None
    return None


def _ast_has_class(path, name):
    """从**源码**用 `ast` 判定某个**类**是否存在（不 `import`；冻结文件只读）。

    ★ `DEF-130-02` 的延续：文本行扫描会读到文档字符串里的类名而假阳性 ⇒ 只认 `ast.ClassDef`。
    """
    try:
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    return any(isinstance(n, ast.ClassDef) and n.name == name for n in tree.body)


def _circular_witness(path, list_name="S9_FORBIDDEN"):
    """检测「见证集由被测判据反推」⇒ 见证只证明**非空转**，不证明**覆盖**。

    口径：在**自测函数**（名 `t<数字>`）里找**以禁用词表为迭代源**的循环 ——
    `for w in <词表>`（`ast.For`）**或** `{w: … for w in <词表>}`（`ast.comprehension`）——
    见证字符串直接由禁用词拼出 ⇒ 与被测判据**同源** ⇒ 自证循环。
    """
    try:
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    hits = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if not re.fullmatch(r"t\d+", fn.name):
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.For) and isinstance(n.iter, ast.Name) \
                    and n.iter.id == list_name:
                hits.append((fn.name, "for"))
                break
            if isinstance(n, ast.comprehension) and isinstance(n.iter, ast.Name) \
                    and n.iter.id == list_name:
                hits.append((fn.name, "comprehension"))
                break
    return sorted(set(hits))


class _Tagged:
    """**带身份**的被禁量载体：precheck 模式下不可格式化 ⇒ 「打印判据量」成为**不可能**。

    ★ 与文本黑名单的本质差别：它判的是**量本身**，不是**量的写法**
      ⇒ **形式无关**（有空格/无空格/换措辞/只留一个数字，全都拦得住）。
    ★ 这不是「更好的黑名单」，而是**换判据**：从「文本」换成「来源」。
    """
    __slots__ = ("v", "tag")

    def __init__(self, v, tag):
        self.v, self.tag = v, tag

    def _boom(self):
        raise PermissionError(f"判据量 {self.tag} 不得在 precheck 阶段被打印")

    def __format__(self, spec):
        self._boom()

    def __str__(self):
        self._boom()

    def __repr__(self):
        self._boom()


def _tagged_forms_rendered():
    """把 6 条被禁形式**经由带身份载体**渲染 ⇒ 每条都必须失败。"""
    cases = [
        ("p₃₅₀ = {v}", lambda: f"p₃₅₀ = {_Tagged(0.0, 'p350')}"),
        ("p₃₀₀ = {v}", lambda: f"p₃₀₀ = {_Tagged(0.0, 'p300')}"),
        ("p₃₅₀={v}（无空格）", lambda: f"p₃₅₀={_Tagged(0.0, 'p350')}"),
        ("分子计数 = {v}", lambda: f"frac>0 的 (sub,C) 对数 = {_Tagged(12345, 'n_pos')}"),
        ("分母计数 = {v}", lambda: f"可算 (sub,C) 对数 = {_Tagged(18000, 'n_comp')}"),
        ("逐 k 计数 = {v}", lambda: f"k=3: frac==0 的 C 有 {_Tagged(84, 'n_zero_k3')} 个"),
    ]
    out = []
    for label, fn in cases:
        try:
            s = fn()
            out.append((label, False, s))
        except PermissionError:
            out.append((label, True, "已拒绝"))
    return out


def _archive_leak():
    """★ `P133-8` 的**核心实测**：已归档的 precheck 输出是否已泄露判决分支。

    数据来源（**只读**）：`dir130/out/precheck130.log`（precheck 阶段实印）
    + `dir130/out/stage_formal.json`（formal 阶段的真值，用于对账）。
    """
    log_p = ROOT / "dir130" / "out" / "precheck130.log"
    fm_p = ROOT / "dir130" / "out" / "stage_formal.json"
    if not log_p.exists() or not fm_p.exists():
        return None
    import json
    try:
        log = log_p.read_text(encoding="utf-8")
        fm = json.loads(fm_p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    m_none = re.search(r"None=(\d+)", log)
    m_zero = re.search(r"零=(\d+)", log)
    m_per = re.search(r"每剂量=(\d+)/(\d+)", log)
    if not (m_none and m_zero and m_per):
        return None
    n_none_pool, n_zero_pool = int(m_none.group(1)), int(m_zero.group(1))
    per = (int(m_per.group(1)), int(m_per.group(2)))
    c2, c3 = fm.get("c2_p350") or {}, fm.get("c3_p300") or {}
    if not (c2 and c3):
        return None

    def nz(d):
        return d["n_comp"] - d["n_pos"]          # frac==0 的行数

    def un(d):
        return d["n_uncomp"]                       # frac is None 的行数

    truth = {"nz350": nz(c2), "nz300": nz(c3), "un350": un(c2), "un300": un(c3),
             "p350": c2.get("p"), "p300": c3.get("p"),
             "thr_max": c2.get("thr_max"), "thr_min": c3.get("thr_min")}
    # 对账：precheck 印的池化数 == formal 两个剂量的分项之和
    acct_zero = (truth["nz350"] + truth["nz300"] == n_zero_pool)
    acct_none = (truth["un350"] + truth["un300"] == n_none_pool)
    # 由 precheck 的三个数单独推出的**紧**界。
    #   联合约束：`n_zero(d) + n_none(d) <= N`（行划分为 None/零/正三类）
    #   ⇒ 给定 `a = n_none(d)`，另一剂量的零计数上限 = `N - (n_none_tot - a)`
    #   ⇒ `n_zero(d) >= n_zero_tot - (N - (n_none_tot - a))`；`p(d)` 对 `a` 单调 ⇒ 逐 `a` 取最大。
    N = max(per)
    nn_tot, nz_tot = n_none_pool, n_zero_pool
    bound_hi = 0.0
    for a in range(nn_tot + 1):
        comp = N - a
        if comp <= 0:
            continue
        nz_min = max(0, nz_tot - (N - (nn_tot - a)))
        if nz_min > comp:
            continue
        bound_hi = max(bound_hi, 1.0 - nz_min / comp)
    bound_lo = 0.0
    thr = truth["thr_min"]
    c3_determined = (thr is not None and bound_hi < thr)
    truth_inside = (bound_lo - 1e-12 <= truth["p350"] <= bound_hi + 1e-12
                    and bound_lo - 1e-12 <= truth["p300"] <= bound_hi + 1e-12)
    return {"n_none_pool": n_none_pool, "n_zero_pool": n_zero_pool, "per": per,
            "nz350": truth["nz350"], "nz300": truth["nz300"],
            "un350": truth["un350"], "un300": truth["un300"],
            "acct_zero": acct_zero, "acct_none": acct_none,
            "bound_lo": bound_lo, "bound_hi": bound_hi,
            "truth_inside": truth_inside,
            "thr_min": thr, "c3_determined": c3_determined,
            "p350": truth["p350"], "p300": truth["p300"],
            "verdict": fm.get("verdict") or fm.get("verdict_str")}


def probe_guard_coverage():
    """`P133-8` 探针：守卫的**覆盖度** —— `S*` 存在 ≠ `S*` 够用。

    只**报告**，不改任何判定（`K25` 是 `②留痕`，不接硬门 —— 「够不够」判不了）。
    """
    print("=" * 100)
    print("P133-8 探针 · 守卫的覆盖度（`S*` 存在 ≠ `S*` 够用）")
    print("=" * 100)

    # ---------- ① 全库普查：谁声明了「阶段打印纪律」，谁配了守卫 ----------
    nums, skip = canonical_nums()
    print(f"\n【① 全库普查】规范目录 {len(nums)} 个"
          + (f"；变体目录跳过 {len(skip)} 个" if skip else ""))
    pat = re.compile(r"不得打印|不得输出|禁止打印|不得产出|不得报告|不得泄露|只许打印")
    n_card = 0
    hits_cards = []
    for n in nums:
        ctx = Ctx(n)
        if not ctx.card:
            continue
        n_card += 1
        hits = [ln.strip() for ln in ctx.text.splitlines()
                if pat.search(_de_emph(ln))]
        if hits:
            hits_cards.append((n, ctx.card.name, hits))
    print(f"   有卡 {n_card} 张；含「阶段打印纪律」的卡 **{len(hits_cards)}** 张")
    for n, cn, hs in hits_cards:
        print(f"     DIR-{n}（{cn}）命中 {len(hs)} 行")
    print(f"   ⇒ 其余 {n_card - len(hits_cards)} 张卡零命中 ⇒ 无阶段纪律 ⇒ 无「阶段静默守卫」需求")

    # ---------- ② DIR-130 深挖：现行词表 vs 语义被禁集 ----------
    pc = ROOT / "dir130" / "precheck130.py"
    forb = _forbidden_from_source(pc)
    print(f"\n【② `DIR-130/S9` 的覆盖度】现行禁用词表（从源码 ast 取）= {forb}")
    if forb is None:
        print("   ★ 取不到禁用词表 ⇒ 跳过（源文件缺失或改版）")
    else:
        hit = miss = 0
        for form, kind in GUARD_FORMS_DIR130:
            h = [w for w in forb if w in form]
            hit += bool(h)
            miss += (not h)
            print(f"     {'命中' if h else '★漏  '} [{kind:<14}] {form}")
        print(f"   ⇒ 现行检测器：命中 {hit}/6，**漏 {miss}/6**")

    # ---------- ③ 数值碰撞：按「数值」匹配会误伤合法输出 ----------
    print("\n【③ 数值碰撞】同一语义量的数值，可能合法地出现在 precheck 输出里")
    src = pc.read_text(encoding="utf-8") if pc.exists() else ""
    m_bin = re.search(r"C\(9,k\) = ([0-9,]+)", src)
    if m_bin:
        nums_bin = [int(x) for x in m_bin.group(1).split(",")]
        print(f"   合法见证 `C(9,k) = {m_bin.group(1)}` 里有 {nums_bin.count(84)} 个 84")
        print(f"   被禁形式 #6 `k=3: frac==0 的 C 有 84 个` 也含 84")
        print("   ⇒ 按**数值**匹配的检测器 ⇒ 对合法输出**误伤** ⇒ 不 sound")
    else:
        print("   ★ 源码里找不到 `C(9,k)` 见证串 ⇒ 跳过")

    # ---------- ④ 自证循环：见证集由被测判据反推 ----------
    circ = _circular_witness(pc)
    print(f"\n【④ 自证循环】在自测函数里用 `<禁用词表>` 作迭代源造见证的："
          f"{circ if circ else '无'}")
    print("   ⇒ 见证与被测判据**同源** ⇒ 只证明「检测器非空转」，**不**证明「纪律被守住」")

    # ---------- ⑤ 方案演示：把判据从「文本」换成「来源」 ----------
    print("\n【⑤ 方案演示】带身份载体（判「量」而非判「写法」）")
    res = _tagged_forms_rendered()
    n_block = sum(1 for _, ok, _ in res if ok)
    for label, ok, info in res:
        print(f"     {'已拒绝' if ok else '★未拦住'}  {label:<20} {info if not ok else ''}")
    print(f"   ⇒ 被禁形式被拦 {n_block}/6（**形式无关**：不认空格、不认措辞）")
    legit = [f"C(9,k) = {','.join(str(x) for x in (9,36,84,126,126,84,36,9,1))}",
             "frac = 0.0", "m_tot = 17  n_low = 0"]
    print(f"   ⇒ 合法结构输出照常渲染：{legit[0]} / {legit[1]}")

    # ---------- ⑥ 已归档输出是否已泄露判决分支 ----------
    print("\n【⑥ 已归档 precheck 输出的判决分支泄露（实测）】")
    lk = _archive_leak()
    if lk is None:
        print("   ★ 缺 `dir130/out/precheck130.log` 或 `stage_formal.json` ⇒ 跳过")
    else:
        print(f"   precheck 印：每剂量 {lk['per'][0]}/{lk['per'][1]} 行、"
              f"零={lk['n_zero_pool']}、None={lk['n_none_pool']}")
        print(f"   formal 真值：n_zero(350)={lk['nz350']}、n_zero(300)={lk['nz300']}"
              f"；n_uncomp(350)={lk['un350']}、n_uncomp(300)={lk['un300']}"
              f"；p₃₅₀={lk['p350']:.6f}、p₃₀₀={lk['p300']:.6f}")
        print(f"   对账：n_zero 两剂量之和 == precheck 印的「零」？ "
              f"{'✓' if lk['acct_zero'] else '✗'}；n_uncomp 之和 == 「None」？ "
              f"{'✓' if lk['acct_none'] else '✗'}")
        print(f"   由 precheck 单独可推：p ∈ [{lk['bound_lo']:.6f}, {lk['bound_hi']:.6f}]"
              f" ⇒ c3 ⇔ p₃₀₀ ≥ {lk['thr_min']} ⇒ "
              f"{'**已可判定（必不成立）**' if lk['c3_determined'] else '不可判定'}")
        print(f"   自检：formal 真值落在界内？ {'✓' if lk['truth_inside'] else '✗（界算错）'}")
        print(f"   真判决 = {lk['verdict']}")

    # ---------- ⑦ `v1.3` 起：本卡的纪律有几道网（载体 + 数字来源审计）----------
    print("\n【⑦ `v1.3` 起的第二道网（结构性守卫 · 形式无关）】")
    card = ROOT / "revolve" / "DIR-130" / "DIRECTION_rev130_v13.md"
    if not card.exists():
        print("   ★ 缺 `DIRECTION_rev130_v13.md` ⇒ 尚未升级 ⇒ 只剩 `S9` 这一道（字面）网")
    else:
        ctext = card.read_text(encoding="utf-8")
        mech = ROOT / "dir130" / "mech130.py"
        pre = ROOT / "dir130" / "precheck130.py"
        has_agg = _ast_has_class(mech, "Agg")
        has_s10 = "s10_number_audit" in pre.read_text(encoding="utf-8")
        has_s11 = "`S11`" in ctext
        has_wl = "白名单" in ctext and "F1" in ctext and "F2" in ctext and "F3" in ctext
        print(f"   卡片 `v1.3` 在位 ✓ · 白名单 + `F1`/`F2`/`F3` 定义：{'✓' if has_wl else '✗'}"
              f" · `S11`（纪律不依赖词表）：{'✓' if has_s11 else '✗'}")
        print(f"   `mech130.Agg` 载体：{'✓' if has_agg else '✗'} · "
              f"`precheck130.s10_number_audit`（数字来源审计）：{'✓' if has_s10 else '✗'}")
        gj = ROOT / "dir130" / "out" / "probe130_v13_guard.json"
        if not gj.exists():
            print("   ★ 缺 `dir130/out/probe130_v13_guard.json` ⇒ 守卫验证尚未实跑")
        else:
            g = json.loads(gj.read_text(encoding="utf-8"))
            print(f"   守卫验证实测：V1（6 条被禁形式全被拦、形式无关）"
                  f"{'✓' if g.get('v1', {}).get('ok') else '✗'} · "
                  f"V2（合法输出零误伤）{'✓' if g.get('v2', {}).get('ok') else '✗'} · "
                  f"V5（新输出推不出任何分支）{'✓' if g.get('v5', {}).get('ok') else '✗'}")
            old_iv = (g.get("v5", {}).get("old", {}) or {}).get("interval_p300")
            new_iv = (g.get("v5", {}).get("old", {}) or {}).get("decides")
            if old_iv:
                print(f"   ★ 对照（旧归档输出）：p₃₀₀ 可行区间上界 = {old_iv[1]:.6f}"
                      f" ⇒ 旧输出**能**判掉分支 = {new_iv}")
            print("   ⇒ 纪律现有**两道**网：`S9`（字面，快但窄）+ `S10`+`Agg`（形式无关，慢但全覆盖）")
    print("\n★ 只报告，不改变任何判定。")
    return 0


# ---------------------------------------------------------------------------
# ★ `P133-10`（Cycle 257）普查探针：全库「**按语义定义的禁止集**」的纪律。
#
#   只**报告**，不判定、不进 `REGISTRY`（与 `K25` 同款 `②留痕`）——
#   「可枚举 / 欠定义」是**人工**判定，机械部分只做**候选收窄**与**线索标注**。
#
#   ★ 本探针的存在理由（`P133-8` §二 + `P133-9` 实测）：按**字面**定义的禁止集用
#   文本判据恰当；按**语义**定义的禁止集用文本判据**必漏**（实测漏 4/6）。
#   ⇒ 故须先量清「哪些纪律是语义型」，否则 `K26`（`P133-11`）会写成一个空条款。
# ---------------------------------------------------------------------------

# 全称量词表（★ 按**本库语料**校准过，见 `revolve/P133-10-SEMANTIC-DISCIPLINE-CENSUS.md`）：
#   ★★ **不含**裸「逐」—— 本库的「逐」几乎全是「逐位/逐字/逐字段/逐 seed/逐元素」
#      这类**复合词**（方式副词），**不是**全称量词；把「逐位相等」当量词是纯假阳性。
#   ★★ 也不含「逐条/逐字段」这类**逐项式**量词 —— 它们作用在**卡片自己的有限清单**上
#      ⇒ 按定义即「可枚举」，不属本普查对象。
QUANT_TOKENS = ("及其任一", "任一", "任何", "所有", "全部", "每一", "逐一", "逐个",
                "每个", "一切", "凡是", "无论",
                # ★★ 「每处」「每一处」「各处」「凡有」= 对**出现位置**的全称量化
                #   （`K15` 的「每处 `不得打印`」就是这一族）。★★ 这四个是**实测补的**：
                #   本普查的三个**指定候选**之一（`K15` 的「每处纪律」）在补它们之前
                #   **机械普查抓不到** ⇒ 量词表漏了整个「出现位置」量词族。
                "每处", "每一处", "各处", "凡有")

# ★★ 「均」「皆」**不**作无条件量词：本库「均值」「平均」极常见（`均` 在「均值」里是
#   **名词的一部分**，不是全称量词）。故二者只认**上下文**：后接情态/动词才算量词用法。
#   （实测：不加这个约束时，「禁止只对5张图的均值做一次3se」这类行会被误收。）
QUANT_CONTEXT = re.compile(r"均(?=须|应|需|不|为|已|未|是|要|当|取|可|能|得|照|依)"
                           r"|皆(?=须|应|不|是|为|已|未|需|要)")

# 禁止语表（★ **不含**「不可」—— 本库「不可」多为**属性标签**（「判据不可分辨」、
#   「不可判」「不可读」）而非**对读者的禁令**；混进来会把判据标签当纪律）。
PROHIB_TOKENS = ("不得", "禁止", "不许", "严禁", "不能", "只许", "仅限", "一律", "杜绝")

# 机制标记：一条纪律的禁止集是「语义型」的**四种可辨形态**。
_MECH = {
    "AGG":  re.compile(r"聚合|汇总|池化|中间量|中间聚合|派生的?|衍生|子集|推论|不限于"),
    "SRC":  re.compile(r"属于|来自|源自|formal|precheck|smoke|--stage|阶段"),
    "CLS":  re.compile(r"读数|量值|中间量|聚合量|结果|输出|结论|数字|数值|条目|项|内容|机制|架构|判决|读出|配置|实现|映射|种子"),
    "SYM":  re.compile(r"[α-ωΑ-Ω]"),
    "LIT":  re.compile(r"`[^`]+`(?:\s*[/、,，]\s*`[^`]+`)+"),
    "COMP": re.compile(r"之外|以外|其它|其他"),
}

# 描述性上下文：修订记录 / 缺陷表 / 历史 —— 这些行**记述**发生过什么，不是**对读者的禁令**。
_DESCRIPTIVE_HEAD = re.compile(r"修订|缺陷|历史|变更|变更记录|revision|勘误|沿革|回顾", re.I)
# 表格行（以 `|` 开头）= 判据标签 / 读数表 ⇒ 是**数据**，不是纪律语句。
_TABLE_ROW = re.compile(r"^\s*\|")


def _sem_hit(line):
    """一行是否是「含全称量词的禁止语句」。

    返回 `(量词, 禁止语, 机制标记元组)` 或 `None`。
    ★ 匹配前先 `_de_emph`（去 markdown 强调/反引号）—— 否则 `**不得**打印` 这类写法
      会因标记夹在词间而漏（`P133-4` 已实测过同一物种的假 `N/A`）。
    """
    d = _de_emph(line)
    qm = pm = None
    for tok in QUANT_TOKENS:
        i = d.find(tok)
        if i >= 0 and (qm is None or i < qm[0]):
            qm = (i, tok)
    if qm is None:
        m = QUANT_CONTEXT.search(d)
        if m:
            qm = (m.start(), m.group(0))
    for tok in PROHIB_TOKENS:
        i = d.find(tok)
        if i >= 0 and (pm is None or i < pm[0]):
            pm = (i, tok)
    if qm is None or pm is None:
        return None
    flags = tuple(k for k, r in _MECH.items() if r.search(line))
    #   ★ 标记在**原行**上判（不是去强调后的 `d`）：`LIT`（显式 token 列举）靠**反引号**识别，
    #   而 `_de_emph` 会把反引号抹掉 ⇒ 在 `d` 上判 `LIT` 恒为 0（实测踩过）。
    return (qm[1], pm[1], flags)


def _object_phrase(line, qstart):
    """量词**前后**的短语（各 ≤12 字）—— 人工标注时必须看到「量词管的是谁」。

    ★ 为什么两侧都要：被量化对象既可能**跟在**量词后（「不得含**任何** `C2`/`C3` 读数」），
      也可能**在**量词前（「Stage A 守卫断言冻结**任一**阻断项失败」⇒ 对象是「阻断项」，
      而「失败」在量词之后）。只取一侧会把对象抽错 ⇒ 标注就会跟着错。
    """
    d = _de_emph(line)
    left = d[max(0, qstart - 12):qstart]
    right = d[qstart:qstart + 16]
    cut = r"[（(，,、。；;：:「」『』\s|｜/\\]"
    left = re.split(cut, left)[-1][:12]
    right = re.split(cut, right)[0][:12]
    return left.strip(), right.strip()


def _norm_stmt(line):
    """禁止语句的归一化键（去强调 → 去空白 → 去标点 → 截断）。"""
    s = _de_emph(line)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，、,；;：:（）()【】\[\]「」『』|｜/\\]", "", s)
    return s[:72]


def _checklist_blocks():
    """`CARD-CHECKLIST.md` 的**全部**条目（条数由代码数，`DEF-128-03`）⇒ [(K号, 字段名, 行文本)]。"""
    if not CHECKLIST.exists():
        return []
    t = CHECKLIST.read_text(encoding="utf-8")
    hits = [(int(m.group(1)), m.start()) for m in re.finditer(r"^#{2,3} K(\d+) · ", t, re.M)]
    out = []
    for i, (n, pos) in enumerate(hits):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(t)
        blk = t[pos:end]
        for fld in ("原文", "检查"):
            m = re.search(rf"\*\*{fld}\*\*\s*[：|]\s*([^\n]*)", blk)
            if m:
                out.append((n, fld, m.group(1).strip()))
    return out


def _symbol_roles(sym, limit=6):
    """符号 `sym` 在全库里的**左上下文**（最近 6 个 CJK 字）分布 ⇒ 多义候选线索。

    ★ 只报告**线索**，多义与否**由人工**判（机械聚类不承载语义判断）。
    """
    cnt = collections.Counter()
    for n in canonical_nums()[0]:
        ctx = Ctx(n)
        if not ctx.card:
            continue
        d = _de_emph(ctx.text)
        for m in re.finditer(re.escape(sym), d):
            left = d[max(0, m.start() - 8):m.start()]
            left = re.sub(r"[^\u4e00-\u9fff]", "", left)[-6:]
            if left:
                cnt[left] += 1
    return cnt.most_common(limit)


def probe_semantic_discipline():
    """`P133-10` 探针：全库普查「**按语义定义的禁止集**」的纪律。

    只**报告**：不判 `PASS/FAIL`、不进 `REGISTRY`、不改任何判决。
    """
    print("=" * 100)
    print("P133-10 探针 · 全库普查「按语义定义的禁止集」的纪律（只报告，不判定）")
    print("=" * 100)
    # ★ `P133-11`（Cycle 258）：条目数**由代码数**，不写死（`DEF-128-03`）。
    #   `K26` 加入后清单从 25 条变 26 条 ⇒ 写死的 "25" 会立刻变成**谎**。
    _nA = len({n for n, _f, _t in _checklist_blocks()})
    print("口径：")
    print(f"  总体 A = `CARD-CHECKLIST.md` 的 {_nA} 条条目（`原文` + `检查` 行)")
    print("  总体 B = 105 个规范目录里每张卡的**规范当前 revision**（最高 `_vN`，排除 `archived`）")
    print("  普查单元 = 一条**禁止语句**（禁止语 + 全称量词同句），跨卡去重")
    print("  排除：修订记录/缺陷表等**描述性**小节、表格行（数据而非禁令）")
    print(f"  量词表（已按本库语料校准，**不含**裸「逐」——本库「逐」多是「逐位/逐字」复合词）= "
          f"{'、'.join(QUANT_TOKENS)}")
    print(f"  禁止语表（**不含**「不可」——本库「不可」多为属性标签而非禁令）= "
          f"{'、'.join(PROHIB_TOKENS)}")

    # ---------- ① 总体 A：清单条目 ----------
    print(f"\n【① 总体 A · 清单 {_nA} 条里的候选】")
    blocks = _checklist_blocks()
    a_hits = []
    for n, fld, txt in blocks:
        h = _sem_hit(txt)
        if h:
            a_hits.append((n, fld, txt, h))
            print(f"   K{n:<3} 【{fld}】 量词={h[0]!r} 禁止={h[1]!r} 机制={'|'.join(h[2]) or '—'}")
            print(f"        {txt[:120]}")
    print(f"   ⇒ {_nA} 条条目（×2 字段）里命中 **{len(a_hits)}** 条")

    # ---------- ② 总体 B：卡片 ----------
    nums, skip = canonical_nums()
    if skip:
        print(f"   ★ 跳过非规范目录 {len(skip)} 个：{', '.join(map(str, skip))}")
    print("\n【② 总体 B · 105 张卡里的候选】")
    stmts = {}
    n_card = 0
    for n in nums:
        ctx = Ctx(n)
        if not ctx.card:
            continue
        n_card += 1
        sec_ctx = []
        for head, body in ctx.sections():
            sec_ctx.append((head, body))
        for ln in ctx.text.splitlines():
            if _TABLE_ROW.match(ln):
                continue
            h = _sem_hit(ln)
            if not h:
                continue
            # 落在描述性小节里的行 ⇒ 排除（记述 ≠ 禁令）
            skip_line = False
            for head, body in sec_ctx:
                if ln in body.splitlines() and _DESCRIPTIVE_HEAD.search(head):
                    skip_line = True
                    break
            if skip_line:
                continue
            key = _norm_stmt(ln)
            stmts.setdefault(key, []).append((n, ln.strip(), h))
    n_occ = sum(len(v) for v in stmts.values())
    cards_hit = sorted({o[0] for v in stmts.values() for o in v})
    print(f"   有卡 {n_card} 张；命中行 **{n_occ}** 行；去重后不同禁止语句 **{len(stmts)}** 条；"
          f"涉及 **{len(cards_hit)}** 张卡")

    # ---------- ③ 机制标记分布 ----------
    print("\n【③ 机制标记分布（一条语句可带多个）】")
    mc = collections.Counter()
    for v in stmts.values():
        for f in v[0][2][2]:          # ★ v[0] = (卡号, 行文本, (量词, 禁止语, 标记)) ⇒ 标记在 [2][2]
            mc[f] += 1
    for f in sorted(_MECH):
        print(f"   {f:<5} {mc.get(f, 0):>4} 条")

    # ---------- ④ 候选逐条列出（人工标注用） ----------
    def _emit(title, group):
        print(f"\n【{title}（{len(group)} 条）—— 人工逐条标「可枚举 / 欠定义」】")
        for k, v in sorted(group, key=lambda kv: (-len(kv[1]), kv[0])):
            n, ln, h = v[0]
            locs = ",".join(f"D{x[0]}" for x in v)
            print(f"   [{'|'.join(h[2]) or '—':<17}] ×{len(v):<2} {locs:<24} {k[:46]}")
            lo, ri = _object_phrase(ln, _de_emph(ln).find(h[0]))
            print(f"        量词={h[0]!r} 禁={h[1]!r} 对象左={lo!r} 对象右={ri!r}")

    flagged = [(k, v) for k, v in stmts.items() if v[0][2][2]]
    plain = [(k, v) for k, v in stmts.items() if not v[0][2][2]]
    _emit("④ 带机制标记的候选", flagged)
    _emit("④b 无机制标记的候选（仍须逐条看对象 —— 「开放类别」不一定带机制词）", plain)
    print(f"\n   ⇒ 合计候选 **{len(stmts)}** 条（带标记 {len(flagged)} + 无标记 {len(plain)}）")

    # ---------- ⑤ 符号多义候选 ----------
    print("\n【⑤ 符号多义候选（第三机制：符号二义）】")
    print("   ★ 与①–④ **独立**：符号二义的纪律**未必**含全称量词 ⇒ 普查不得只靠量词抓它。")
    syms = set()
    for n, fld, txt in blocks:
        ss = set(re.findall(r"[α-ωΑ-Ω]", _de_emph(txt)))
        if ss:
            syms |= ss
            print(f"   K{n:<3} 【{fld}】含符号 {''.join(sorted(ss))}：{txt[:96]}")
    for s in sorted(syms):
        roles = _symbol_roles(s)
        print(f"   ⇒ `{s}` 的左上下文分布（前 {len(roles)}）：")
        for lc, c in roles:
            print(f"        {c:>4}×  …{lc}{s}")
    print("   ⇒ 多义与否**由人工判**；机械聚类不承载语义判断。")

    # ---------- ⑥ 不改判定 ----------
    print("\n【⑥ 本探针未触碰判定】")
    print(f"   `REGISTRY` 仍是 {len(REGISTRY)} 条：{sorted(REGISTRY)}")
    print(f"   清单 `①自动化` 集合仍是 {sorted(checklist_auto())}（两者相等 ⇒ 无新增硬门）")
    print("   ⇒ `PASS/FAIL/N/A` 计数与本探针**无关**；`--all` 的判决一字未动。")
    return 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ★ `P133-15`（Cycle 259）探针：`K26`（**禁止集的可执行性**）的**操作化**。
#
#   `K26` 是 `②留痕` ⇒ **不判 `PASS/FAIL`、不进 `REGISTRY`**，只把三步落成可抄的读数：
#     ① **能不能枚举**（候选形态，机械只报**标记**，判定仍归人）
#     ② **已配几道网** + **形态**（字面 / 形式无关；★★ 「**实测**」与「**声明**」分开报）
#     ③ **是否已明示「不可枚举」**
#
#   ★★★ 本探针的**核心区分**（`K26` 正文要求的两道网必须**形态不同**）：
#     · **字面网（实测）**  = 卡片的 `precheck*.py`/`mech*.py` 里存在**模块级**的
#                            字符串序列常量（≥3 项）⇒ 那是一个**真的 token 表**；
#     · **字面网（声明）**  = 卡片文本里 `S\d+` 与「禁用词/白名单」同现 ⇒ 只是**说法**；
#     · **形式无关网（实测）**= 卡片 py 里存在 **载体类**（`Agg`/`Carrier`/`Tagged`…）
#                            或 **数字来源审计函数**（`*audit*`）⇒ 能让被禁量**不可打印**；
#     · **形式无关网（声明）**= 卡片文本里「形式无关」+「源头/不打印/不算出」同现。
#   ★★ **实测 ≠ 声明**：`P133-8` 的教训是「有守卫 ≠ 守卫够」，此处同理
#      「**说**有两道网 ≠ 两道网在」，故四个布尔值**分列**，汇总只认**实测**。
# ---------------------------------------------------------------------------

#: 字面网**声明**：`S\d+` 守卫名 + token 表措辞（两者**同现**才算，防「白名单」一词被
#:   当标点用 —— 本库里「白名单」出现在**关于**白名单的讨论里很常见）。
_K26_LIT_DECL = re.compile(r"`?S\d+`?", re.I)
_K26_LIT_WORDS = re.compile(r"禁用词|禁止 token|禁止token|黑名单|白名单|词表|token 表|token表")
#: 形式无关网**声明**：必须同时有「形态无关的描述」与「源头动作」。
_K26_FREE_WORDS = re.compile(r"形式无关|不认措辞|不认空格|与措辞无关|形态无关")
_K26_SRC_ACTS = re.compile(r"源头|不可打印|不算出|不算出来|只报布尔|脱敏|身份载体|载体")
#: ③ 明示「不可枚举」——**强信号**：`K26` 步骤 3 的唯一合法出口。
#: ★★★ **实测收紧**（`Cycle 259`）：初版把「判不了 / 不可判定 / 欠定义」也算强信号，
#:   结果 `DIR-38`/`39`/`40`/`104` 四张卡被判成「已明示不可枚举」——全是**误报**：
#:   它们的原句是「★ 不可判定：若 …」，讲的是**测量**不可判定，**不是**禁止集不可枚举。
#:   ⇒ 强信号**只认**带「枚举/穷举/逐一列举」的措辞；三者降为**弱信号**（下），
#:     只报**不**翻转状态（否则 ③ 会变成一条只要写「判不了」就能走的**后门**）。
_K26_NONENUM = re.compile(r"不可枚举|无法枚举|不能枚举|枚举不了|无法穷举|不能穷举"
                          r"|无法逐一列举|不能逐一列举|无法逐项列举")
#: 弱信号：与可枚举性**相关但不特定**的措辞 —— 只报给人看，**不**参与状态机。
_K26_WEAK = re.compile(r"判不了|不可判定|欠定义|无法界定|说不清")
_K26_WHY = re.compile(r"因为|理由|原因|为什么|由于|开放集|推论|二义")

#: 载体类的名字（形式无关网的**实测**信号）—— 只认这些**结构性**名字，避免抓到普通变量。
_K26_CARRIER = re.compile(r"^(Agg|Aggregate|Carrier|Tagged|Wrapped|Sealed|Redacted|"
                          r"Bool|BoolOnly|Opaque)", re.I)
#: 数字来源审计函数名（`DIR-130` 的 `s10_number_audit` 就是这一族）。
_K26_AUDIT = re.compile(r"audit|provenance|trace_?num|num_?src", re.I)


def _k26_ast_lit_lists(src, min_len=3):
    """`src` 里**模块级**字符串序列常量（list/tuple/set，≥`min_len` 项）⇒ 计一个**字面网**。

    ★ 为什么是「模块级」：局部 list 多为中间量，不等于**判据词表**；判据词表必须是卡
      在**开卡时冻结**的东西（`K16` 判读理由）⇒ 只能在模块级。
    ★ 为什么 `min_len=3`：1–2 项的表更像**分支参数**（`("a","b")`）而非**禁止集**
      （实测本库 6 条禁止形式的词表是 `["聚合","中间","合并","累加","sum","mean"]` 这种量级）。
    """
    try:
        with warnings.catch_warnings():      # ★ 卡片 py 里常有 `\l` 这类 escape ⇒ 别吵
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(src)
    except SyntaxError:
        return 0
    n = 0
    for node in tree.body:                     # 只扫**顶层**（模块级）
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        val = node.value
        if not isinstance(val, (ast.List, ast.Tuple, ast.Set)):
            continue
        elts = val.elts
        if len(elts) < min_len:
            continue
        if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elts):
            n += 1
    return n


def _k26_ast_carriers(src):
    """`src` 里**形式无关网**的实测信号 ⇒ `(载体类名表, 审计函数名表)`。

    载体类：`class Agg(...)` 这类 —— 让被禁量**根本不以原样存在**；
    审计函数：`def s10_number_audit(...)` 这类 —— 让每个数字**必须带来源**。
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(src)
    except SyntaxError:
        return [], []
    classes, funcs = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _K26_CARRIER.match(node.name):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _K26_AUDIT.search(node.name):
            funcs.append(node.name)
    return classes, funcs


def _k26_nets(card_text, py_srcs):
    """`K26` 步骤 2 的四读数：`{lit_meas, lit_decl, free_meas, free_decl}`（**布尔**）。

    `py_srcs` = [(文件名, 源码)]，来自该卡的 `precheck*.py` / `mech*.py`。
    ★ 四个值**分开**报：实测是**事实**，声明是**说法**；汇总只认实测（`P133-8` 的教训）。
    """
    lit_meas = sum(_k26_ast_lit_lists(s) for _n, s in py_srcs) > 0
    free_meas = False
    for _n, s in py_srcs:
        cls, fns = _k26_ast_carriers(s)
        if cls or fns:
            free_meas = True
            break
    t = card_text
    lit_decl = bool(_K26_LIT_DECL.search(t)) and bool(_K26_LIT_WORDS.search(t))
    free_decl = bool(_K26_FREE_WORDS.search(t)) and bool(_K26_SRC_ACTS.search(t))
    return {"lit_meas": lit_meas, "lit_decl": lit_decl,
            "free_meas": free_meas, "free_decl": free_decl}


def _k26_evidence(py_srcs):
    """`(实)` 两类网的**证据串**（哪个文件、什么名字）⇒ 让 `✓` 可审，而不是一个裸勾。"""
    lit, free = [], []
    for name, src in py_srcs:
        k = _k26_ast_lit_lists(src)
        if k:
            lit.append(f"{name}: {k} 个模块级词表")
        cls, fns = _k26_ast_carriers(src)
        if cls:
            free.append(f"{name}: class {'/'.join(cls)}")
        if fns:
            free.append(f"{name}: def {'/'.join(fns)}")
    return lit, free


def _k26_nonenum(text):
    """`K26` 步骤 3：是否**明示**「不可枚举」、是否写了**为什么判不了**、**触发行**。

    ★★ 返回触发行是**必须**的：③ 的三个词（「判不了」「不可判定」「欠定义」）在本库
      也出现在**与可枚举性无关**的语境里（「该判据不可判定」说的是**测量**，不是
      **禁止集**）⇒ 只报布尔会让读者无法判断这是否是**真的**声明。故把**证据**摊开。
    """
    m = _K26_NONENUM.search(text)
    if not m:
        # ★ 强信号缺 ⇒ `(False, False, 弱信号或 None)`；**不**翻转状态（见 `_k26_state`）。
        w = _K26_WEAK.search(text)
        return False, False, (("弱:" + w.group(0), "（弱信号，不参与状态机）") if w else None)
    line = ""
    for ln in text.splitlines():
        if _K26_NONENUM.search(ln):
            line = _de_emph(ln).strip()
            break
    return True, bool(_K26_WHY.search(text)), (m.group(0), line[:96])


def _k26_forms(flags):
    """机制标记 ⇒ 可抄进开卡声明的**形态候选**（★ 候选，不是判定）。"""
    m = []
    if "LIT" in flags:
        m.append("E-LIT（显式 token 列举）")
    if "COMP" in flags:
        m.append("E-COMP（补集式，前提=列举完整）")
    if "AGG" in flags:
        m.append("U1-AGG（聚合/中间量）")
    if "SRC" in flags:
        m.append("U1-SRC（按来源/阶段）")
    if "CLS" in flags:
        m.append("U1-CLS（开放类别）")
    if "SYM" in flags:
        m.append("U3（符号二义）")
    return m or ["—（无机制标记 ⇒ 人工定）"]


def _k26_state(nets, ne, why):
    """`K26` 四状态（**只认实测**）⇒ `(code, 说明)`。

    ★★★ 抽出成函数是为了让它**可证伪**：`P133-15` 的核心主张是「**实测 ≠ 声明**」，
      若状态机读了 `*_decl`，一个只会声明的卡就会被判成「两道网齐备」——
      这正是 `P133-8`「有守卫 ≠ 守卫够」的同物种错误。夹具 `-K26-9` 专测这一点。
    """
    if ne and why:
        return "NONENUM", "③已明示不可枚举（合法出口）"
    if nets["lit_meas"] and nets["free_meas"]:
        return "TWO", "②两道网（实测齐备）⇒ 满足步骤 2"
    if nets["lit_meas"] or nets["free_meas"]:
        return "ONE", "★只有一道（实测）⇒ **未配网**"
    return "ZERO", "★★零道（实测）⇒ **未配网**"


def probe_k26():
    """`P133-15` 探针：把 `K26` 的三步**逐卡落成可抄读数**（只报告，不判定）。

    ★ 不进 `REGISTRY`、不改 `PASS/FAIL/N/A`、**不替卡片发明白名单**
      （那只能由卡片**开卡时**冻结，沿 `K16` 判读的理由）。
    """
    print("=" * 100)
    print("P133-15 探针 · `K26`（禁止集的**可执行性**）的操作化读数（只报告，不判定）")
    print("=" * 100)
    print("口径（★ 抄进开卡声明时**照抄这四行**）：")
    print("  ① 能枚举 ⇒ 至少**两道**网，且**形态互不相同**：(i) **字面网** ＋ (ii) **形式无关网**")
    print("     ★ 只有一道**字面**网的判「**未配网**」（`P133-8` 实测字面网命中 2/6、漏 4/6）")
    print("  ② 不能枚举 ⇒ **明示「不可枚举」＋ 写明为什么判不了** ＋ 该条降 `②留痕`；")
    print("     且**不得**再声称「已被 `S\\d+` 守住」")
    print("  ③ 本探针把**实测**与**声明**分列：`(实)` = 在卡片 py 里找到了结构证据，")
    print("     `(声)` = 只在卡片文本里**说**有。★★ 汇总只认 `(实)`。")
    print("  ④ `K26` 是 `②留痕` ⇒ 下列读数**不是** `PASS/FAIL`，而是**开卡时要声明的四个布尔值**")

    nums, skip = canonical_nums()
    if skip:
        print(f"   ★ 跳过非规范目录 {len(skip)} 个：{', '.join(map(str, skip))}")
    print(f"\n【① 逐卡扫描】规范目录 {len(nums)} 个")

    rows = []          # (n, cn, stmt, forms, nets, nonenum, why, has_free_decl_only)
    n_card = n_stmt = 0
    for n in nums:
        ctx = Ctx(n)
        if not ctx.card:
            continue
        n_card += 1
        sec_ctx = list(ctx.sections())
        # ★★ 只扫**冻结交付物**（`precheck*.py` / `mech*.py`）—— 与 §2 的机械定义**一致**。
        #   初版扫了 `dir{n}/*.py` 全部 ⇒ 读数取决于**工作区里恰好有哪些 probe 脚本**
        #   （本地镜像与 trainer 文件集不同 ⇒ 两侧输出无法逐字节比对，`Cycle 259` 实测）。
        #   「网」如果只存在于一张一次性 probe 里，它就不是**守卫**。
        py_srcs = []
        for f in sorted(ctx.py, key=lambda q: q.name):
            if not re.match(r"^(precheck|mech)[A-Za-z0-9_]*\.py$", f.name):
                continue
            try:
                py_srcs.append((f.name, f.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
        ctext = ctx.card.read_text(encoding="utf-8")
        ne, why, ne_ev = _k26_nonenum(ctext)
        nets = _k26_nets(ctext, py_srcs)          # ★ 卡片级，hoist 出语句循环
        ev_lit, ev_free = _k26_evidence(py_srcs)
        seen = set()
        for ln in ctx.text.splitlines():
            if _TABLE_ROW.match(ln):
                continue
            h = _sem_hit(ln)
            if not h:
                continue
            if any(ln in body.splitlines() and _DESCRIPTIVE_HEAD.search(head)
                   for head, body in sec_ctx):
                continue
            key = _norm_stmt(ln)
            if key in seen:
                continue
            seen.add(key)
            rows.append((n, ctx.card.name, _de_emph(ln).strip(), _k26_forms(h[2]),
                         nets, ne, why, ne_ev, ev_lit, ev_free))
            n_stmt += 1

    print(f"   有卡 {n_card} 张；含禁止语句 **{len({r[0] for r in rows})}** 张；"
          f"禁止语句 **{n_stmt}** 条")

    # ---------- ② 逐卡「开卡声明」草稿 ----------
    print("\n【② 逐卡「开卡声明」草稿（可直接抄）】")
    by_card = {}
    for r in rows:
        by_card.setdefault(r[0], []).append(r)
    #   ★ 汇总口径：四状态 —— 只认**实测**
    st_no_net = st_one_lit = st_two = st_non = 0
    for n in sorted(by_card):
        rs = by_card[n]
        cn = rs[0][1]
        nets = rs[0][4]
        ne, why, ne_ev = rs[0][5], rs[0][6], rs[0][7]
        ev_lit, ev_free = rs[0][8], rs[0][9]
        n_lit = int(nets["lit_meas"]) + int(nets["lit_decl"])
        n_free = int(nets["free_meas"]) + int(nets["free_decl"])
        print(f"\n   DIR-{n}（{cn}）· 禁止语句 {len(rs)} 条 · 卡片级 ③明示不可枚举："
              f"{'✓' if ne else '✗'}{'（附理由）' if ne and why else '（★无理由）' if ne else ''}"
              + (f" ⟵ 触发 `{ne_ev[0]}`：「{ne_ev[1]}」" if (ne and ne_ev) else
                 (f" ⟵ ★弱信号 `{ne_ev[0]}`（**不**算 ③ 的合法出口，仅提示人工复核）"
                  if ne_ev else "")))
        print(f"     ② 网 · 字面(实){'✓' if nets['lit_meas'] else '✗'}"
              f"/字面(声){'✓' if nets['lit_decl'] else '✗'}"
              f" · 形式无关(实){'✓' if nets['free_meas'] else '✗'}"
              f"/形式无关(声){'✓' if nets['free_decl'] else '✗'}")
        for e in (ev_lit + ev_free):
            print(f"       (实)证据：{e}")
        # 状态机（**只认实测**）
        _code, state = _k26_state(nets, ne, why)
        st_non += (_code == "NONENUM")
        st_two += (_code == "TWO")
        st_one_lit += (_code == "ONE")
        st_no_net += (_code == "ZERO")
        print(f"     ⇒ 判定**留给开卡人**；机械状态 = {state}")
        if not ne and not nets["free_meas"]:
            print("       ★ 若①判为「不可枚举」⇒ 必须**补 ③ 明示 + 理由**（`K26` 步骤 3）")
        for stmt, forms in [(r[2], r[3]) for r in rs]:
            print(f"       · ① 形态候选：{'、'.join(forms)}")
            print(f"         「{stmt[:96]}」")

    # ---------- ③ 汇总 ----------
    print("\n【③ 汇总（★ 只认实测；不构成 `PASS/FAIL`）】")
    print(f"  ③已明示不可枚举（合法出口）：**{st_non}** 张")
    print(f"  ②两道网实测齐备　　　　　　：**{st_two}** 张")
    print(f"  ★只有一道实测网（未配网）　：**{st_one_lit}** 张")
    print(f"  ★★零道实测网（未配网）　　：**{st_no_net}** 张")
    print("  ★ ★★ **未配网合计 = "
          f"{st_one_lit + st_no_net} 张** —— 但**不得**读成「这些卡违规」：")
    print("    `K26` 是 `②留痕`，判定「能不能枚举」的是**人**；本探针只把**事实**摆出来。")
    print("  ★ 同样**不得**读成「配了网的卡就合规」：实测只证明**网在**，不证明**网够**")
    print("    （那是 `K25`/`--probe-guard-coverage` 的问题，且它也判不了）。")

    # ---------- ④ 不改判定 ----------
    print("\n【④ 本探针未触碰判定】")
    print(f"   `REGISTRY` 仍是 {len(REGISTRY)} 条：{sorted(REGISTRY)}")
    print(f"   清单 `①自动化` 集合仍是 {sorted(checklist_auto())}（两者相等 ⇒ 无新增硬门）")
    print("   ⇒ `PASS/FAIL/N/A` 计数与本探针**无关**；`--all` 的判决一字未动。")
    print("\n★ 只报告，不改变任何判定。")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe-k15-scope", action="store_true")
    ap.add_argument("--probe-guard-coverage", action="store_true")
    ap.add_argument("--probe-semantic-discipline", action="store_true")
    ap.add_argument("--probe-k26", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if a.probe_k15_scope:
        return probe_k15_scope()

    if a.probe_guard_coverage:
        return probe_guard_coverage()

    if a.probe_semantic_discipline:
        return probe_semantic_discipline()

    if a.probe_k26:
        return probe_k26()

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
        nums, skip = canonical_nums()
        if skip:
            print(f"★ 跳过 {len(skip)} 个非规范目录（无同号 `dir{{n}}/`，本脚本不覆盖）："
                  f"{', '.join(skip)}")
    elif a.dirs:
        nums = a.dirs
    else:
        ap.error("需要 --dirs / --all / --selftest / --probe-k15-scope / --probe-guard-coverage"
                 " / --probe-semantic-discipline / --probe-k26")

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