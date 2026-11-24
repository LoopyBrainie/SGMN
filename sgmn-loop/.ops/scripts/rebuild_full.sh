#!/bin/bash
# DIR-18 · 重建完整交付包（SGMN_LOOP_full.zip / LOOP_REPORT_full.zip / MANIFEST）
#
# ★ 缺陷 D61（2026-11-06）：原脚本的**打包暂存区在 `/tmp/fzp/SGMN_LOOP_PACKAGE`**。
#   `/tmp` 是**容器临时盘** ⇒ 沙箱重建后被清空 ⇒ 脚本只在「暂存区还在」时才有意义：
#   一次沙箱回收后，暂存区只剩本次拷贝的 39 个文件 ⇒ **完整包从 368 个文件缩到 39 个**
#   （且 `01_FRAMEWORK` 等目录**整段丢失**，无任何报错）。
#   ⇒ 修法：① 暂存区改到**持久卷** `/workspace/.pkg/SGMN_LOOP_PACKAGE`；
#           ② 每次**从零重建**暂存区（`rm -rf` 后按固定目录清单重新拷贝）⇒ 脚本**自足、可重复**；
#           ③ 打包后**逐文件 md5 复核**（已有）**并断言文件数下限**（防静默缩水）。
set -u
# ★ 路径可移植（2026-11-07）：原实现硬编码 `/workspace/sgmn-loop` 与 `/workspace/attachments`
#   （旧沙箱布局）⇒ 实验卷上跑不通 ⇒ ⇒ 改为**环境变量优先 + 默认当前工作区/框架目录**。
ROOT=${SGMN_ROOT:-/home/ec2-user/sgmn-loop}
# ★ 2026-11-09（DEF-26P-10）：原默认暂存区硬编码 `/dev/sdf/...`（旧挂载点）。
#   持久卷已改为 `/mnt/sgmn` ⇒ `mkdir -p` 失败（permission denied）⇒ **zip 静默为空**
#   （`SGMN_LOOP_full.zip = B md5=`、复核 `FAILED=1`），而 log 仍打印「结束」。
#   ⇒ 默认改为**由 ROOT 派生**（`<ROOT 的父目录>/.pkg/...`），并对 PKGDIR 显式 `mkdir -p` + 失败即中止。
PKG=${SGMN_PKG:-$(dirname "$ROOT")/.pkg/SGMN_LOOP_PACKAGE}
LOG=$ROOT/.ops_pack.log
ATT=${SGMN_ATT:-$ROOT/framework}
PKGDIR=$(dirname "$PKG")
MIN_FILES_ABS=${MIN_FILES_ABS:-0}   # ★ 相对判据为主（入包 ≥ 源文件数）；绝对下限可选
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "===== 完整包重建（自足版 · 暂存区 $PKG）====="

# —— ⓿ 从零重建暂存区（目录名沿用**原交付包**的布局）——
rm -rf "$PKG"
mkdir -p "$PKGDIR" || { say "★ 致命：暂存区父目录不可建（$PKGDIR）⇒ 中止，不产出空包"; exit 1; }
mkdir -p "$PKG"/{01_FRAMEWORK,02_ROUNDS,03_LEDGERS,04_EVIDENCE,05_LOGS,06_CONTINUITY,07_BUNDLES} || { say "★ 致命：暂存区不可建（$PKG）⇒ 中止，不产出空包"; exit 1; }

# —— ① 01_FRAMEWORK：框架原文与上游附件（含 md5 与来源）——
#   ★ 旧实现指向旧沙箱的 `/workspace/attachments/<hash>/` 子目录 ⇒ 现指向 `$ATT`（= `$ROOT/framework`）的**平铺**文件。
cp -f "$ATT/SGMN_v2.4_架构设计.md" "$PKG/01_FRAMEWORK/" 2>/dev/null
cp -f "$ATT/记忆型LLM双系统.docx"          "$PKG/01_FRAMEWORK/" 2>/dev/null
cp -f "$ATT/神经网络多尺度方程的完整推导：从微观到宏观.txt" "$PKG/01_FRAMEWORK/" 2>/dev/null
{
  echo "# 01_FRAMEWORK · 本 loop 的**唯一框架来源**"
  echo
  echo "> 生成：$(date -u "+%Y-%m-%d %H:%M:%S") UTC"
  echo
  echo "| 文件 | md5 | 说明 |"
  echo "|---|---|---|"
  for f in "$PKG/01_FRAMEWORK"/*; do
    [ -f "$f" ] || continue
    _bn=$(basename "$f")
    [ "$_bn" = "00_PROVENANCE.md" ] && continue
    _md=$(md5sum "$f" | cut -d' ' -f1)
    if [ "${_bn#SGMN}" != "$_bn" ]; then
      _desc='★ **v2.4 架构设计原文** —— 本 loop 的唯一框架来源（§7.2 派生每个方向的 source）'
    else
      _desc='上游附件（背景材料，非判据来源）'
    fi
    printf "| \`%s\` | \`%s\` | %s |\n" "$_bn" "$_md" "$_desc"
  done
  echo
  echo "## 语料（不入包，按 md5 复原）"
  echo
  echo "| 文件 | md5 | 来源 |"
  echo "|---|---|---|"
  echo "| \`corpus/train.txt\` | \`$(md5sum "$ROOT/corpus/train.txt" 2>/dev/null | cut -d' ' -f1)\` | \`https://cdn.jsdelivr.net/gh/pytorch/examples@main/word_language_model/data/wikitext-2/\` |"
  echo "| \`corpus/valid.txt\` | \`$(md5sum "$ROOT/corpus/valid.txt" 2>/dev/null | cut -d' ' -f1)\` | 同上 |"
  echo
  echo "> 模型权重同样不入包：\`Qwen/Qwen2.5-0.5B-Instruct\`（float32，md5 见卡片）。"
} > "$PKG/01_FRAMEWORK/00_PROVENANCE.md"

# —— ② 单遍归档：**工作区的每个文件都进包一次**（按类别分流）——
#    ★ 2026-11-07：原实现只扫 `revolve/`（旧沙箱布局）⇒ 恢复后的工作区里
#      DIR-15…DIR-18 在**顶层**（`dir15/`…`dir18/`）⇒ 会被**漏掉** ⇒
#      ⇒ 改为扫**全部内容根**（`revolve` + `dir15…dir20` + `framework`）。
#    ★ 保证「不漏文件」：最后会断言「入包文件数 == 源文件数」。
SRC_LIST=$(mktemp)
: > "$SRC_LIST"
# ★ 2026-11-09（DIR-26-N 收口）：内容根补上 dir21…dir26n —— 原列表停在 dir20 ⇒
#   DIR-21…DIR-26-N 的**脚本与结果 JSON 全部漏包**（卡片与裁决在 revolve/ 下，故被扫到）。
# ★ 2026-11-09（DIR-26-P 收口，DEF-26P-9）：原实现是**手写目录白名单**，停在 dir26n
#   ⇒ DIR-26-O（dir26o/）与 DIR-26-P（dir26p/）的**脚本与结果 JSON 全部漏包**
#   （卡片与裁决在 revolve/ 下，故被扫到 ⇒ 「入包数 == 源数」的断言**照不出来**）。
#   ⇒ 改为**通配内容根**：`dir*` 自动覆盖现在与将来的所有方向目录，杜绝手写白名单漂移。
for _d in revolve framework $(cd "$ROOT" && ls -d dir* 2>/dev/null); do
  [ -d "$ROOT/$_d" ] || continue
  find "$ROOT/$_d" -type f ! -path "*__pycache__*" ! -name "*.pyc" >> "$SRC_LIST"
done
sort -o "$SRC_LIST" "$SRC_LIST"
N_SRC=$(wc -l < "$SRC_LIST")
say "源文件数（全部内容根）= $N_SRC"
while IFS= read -r f; do
  rel=${f#"$ROOT/"}
  nm=$(basename "$f")
  case "$nm" in
    *.log|*.out|*.txt)                     cat=05_LOGS ;;
    *.py)                                  cat=02_ROUNDS ;;
    *.md)                                  cat=04_EVIDENCE ;;
    *.json)                                cat=04_EVIDENCE ;;
    *)                                     cat=04_EVIDENCE ;;
  esac
  # 深度 > 2 的文件统一归到 02_ROUNDS（脚本与结果的子目录）
  d=$(dirname "$rel")
  if [ "$(echo "$d" | tr -cd '/' | wc -c)" -gt 2 ]; then cat=02_ROUNDS; fi
  mkdir -p "$PKG/$cat/$(dirname "$rel")"
  cp -f "$f" "$PKG/$cat/$rel" || { say "★★ 入包复制失败：$rel ⇒ 中止"; exit 7; }
done < "$SRC_LIST"

# —— ③ 03_LEDGERS：台账与报告 ——
cp -f "$ROOT/evidence/table.md"  "$PKG/03_LEDGERS/evidence_table.md"
cp -f "$ROOT/evidence/ledger.md" "$PKG/03_LEDGERS/evidence_ledger.md"
cp -f "$ROOT/DIRECTION-REGISTER.md" "$PKG/03_LEDGERS/DIRECTION-REGISTER.md"
cp -f "$ROOT/LOOP-MECHANISM.md"     "$PKG/03_LEDGERS/LOOP-MECHANISM.md"
cp -f "$ROOT/revisions.md"          "$PKG/03_LEDGERS/revisions.md"
# ★ DEF-91-34（Cycle 156）：COVERAGE-v24.md 是**终止条件的操作化载体**，
#   自 Cycle 140 引入起 rebuild_full.sh 从未收录 ⇒ 覆盖矩阵一直**不在包内**。补齐。
cp -f "$ROOT/COVERAGE-v24.md"        "$PKG/03_LEDGERS/COVERAGE-v24.md" \
  || { say "★★ COVERAGE-v24.md 入包失败 ⇒ 中止（终止判据依赖它）"; exit 7; }
# ★ DEF-26P-10（补）：台账是交付物，复制失败不得静默吞掉
cp -f "$ROOT/MANIFEST.md"           "$PKG/03_LEDGERS/MANIFEST.md" || { say "★★ MANIFEST.md 入包失败 ⇒ 中止"; exit 7; }
cp -f "$ROOT/SUMMARY.md"            "$PKG/03_LEDGERS/SUMMARY.md"  || { say "★★ SUMMARY.md 入包失败 ⇒ 中止"; exit 7; }
# ★ P130-1（Cycle 243）：`CARD-CHECKLIST.md`（开卡清单）是**治理文件** —— 它是纪律的唯一归档地
#   （`handoff.md` 每轮被覆盖，规则只写在那里会随之丢失）。它不在 ② 的内容根
#   （只有 revolve/framework/dir*）⇒ 与 COVERAGE-v24.md 同类，必须显式拷入，否则静默缺席归档。
cp -f "$ROOT/CARD-CHECKLIST.md"      "$PKG/03_LEDGERS/CARD-CHECKLIST.md" \
  || { say "★★ CARD-CHECKLIST.md 入包失败 ⇒ 中止（开卡纪律依赖它）"; exit 7; }
# ★ DEF-26P-12：`decisions/` 从未进 ② 的内容根（`revolve framework dir*`），也不在 ③ 的手写清单里
#   ⇒ **决策台账在终包里整段缺失**。此处按**保留路径**拷入（卡片里写的是 `decisions/DECISIONS.md`，路径必须可解析）。
mkdir -p "$PKG/03_LEDGERS/decisions"
cp -rf "$ROOT"/decisions/* "$PKG/03_LEDGERS/decisions/" || { say "★★ decisions/ 入包失败 ⇒ 中止"; exit 7; }
cp -f "$ROOT/REPORT.md" "$ROOT/SUMMARY.md" "$ROOT/LOOP_REPORT_DIR04-DIR17.md" "$PKG/03_LEDGERS/" || { say "★★ 报告类台账入包失败 ⇒ 中止"; exit 7; }
# ★ 续跑交接包：DIR-17 → DIR-20 的 compaction（当前状态 + 执行清单 + 缺陷留档）
cp -f "$ROOT/COMPACTION_DIR17-DIR20.md" "$PKG/06_CONTINUITY/" 2>/dev/null
cp -f "$ROOT/COMPACTION_DIR17-DIR20.md" "$PKG/03_LEDGERS/" 2>/dev/null

# —— ④ 05_LOGS 补：CPU 降级轮的完整日志与增量落盘 ——
cp -f "$ROOT"/revolve/DIR-18/.ops/cpu/*.log "$ROOT"/revolve/DIR-18/.ops/cpu/*.json \
      "$ROOT"/revolve/DIR-18/.ops/cpu/*.txt "$PKG/05_LOGS/" 2>/dev/null
cp -f "$ROOT/revolve/DIR-18/R28_smoke_results.json" "$PKG/05_LOGS/" 2>/dev/null

# —— ⑤ 06_CONTINUITY：运维 / 审计 / 打包脚本（可复现性资产）——
mkdir -p "$PKG/06_CONTINUITY"
# ★ DEF-26P-12：`cp -f "$ROOT"/.ops/*` 只碰 `.ops/` 顶层文件，对 `scripts/` 目录**无 `-r` ⇒ 静默失败**
#   ⇒ 归档里**没有任何打包/审计脚本**，包不可自重建、不可自审计。此处按路径显式拷入。
mkdir -p "$PKG/06_CONTINUITY/.ops/scripts"
# ★★ P118-3（Cycle 225）修：原白名单只有 3 个脚本，而 `audit_consistency.py` 现在
#   `import audit_counts` ⇒ 只带 audit_consistency.py 入包会让"可复现性资产"**导入即崩**。
#   ⇒ 把被依赖的纯函数层与其自测一并入包（★ 新增审计脚本时必须同步加到这里与下方 `_must`）。
# ★ P134-4（Cycle 250）：新增 blind_copy.sh / restore_out.sh —— 它们是「删档事故」的
#   机械守卫（决定二百三十九/二百四十）。★ 归档里若缺它们，事故的修法就不可复现。
# ★ P134-14b（Cycle 262）：新增 `test_audit_boundary9.py` —— 审计项 ⑨ 的**注入自测**
#   （证明 ⑨a–⑨e 每一项都真能红）。★ 与 `test_audit_counts.py` 同理：守卫的守卫也要入包，
#   否则"⑨ 不会红"这件事在归档里不可复核。
# ★ P134-14d（Cycle 264）：新增 `test_audit_k27judge.py` —— 审计项 ⑩（`K27` 判定式 ↔ 源码
#   字符串同一性）的**双向注入自测**。★ 这一条尤其需要入包：⑩ 的干净副本**天生就是绿的**，
#   若不带自测，「它会不会红」在归档里就只能靠相信。
for _s in rebuild_full.sh repack.sh audit_consistency.py audit_counts.py test_audit_counts.py \
          test_audit_boundary9.py test_audit_k27judge.py check_cards.py blind_copy.sh restore_out.sh; do
  cp -f "$ROOT/.ops/scripts/$_s" "$PKG/06_CONTINUITY/.ops/scripts/$_s" \
    || { say "★★ 打包脚本入包失败：$_s ⇒ 中止"; exit 7; }
done
cp -f "$ROOT"/.ops/* "$PKG/06_CONTINUITY/" 2>/dev/null
cp -f "$ROOT"/revolve/DIR-18/.ops/*.sh "$ROOT"/revolve/DIR-18/.ops/*.py "$PKG/06_CONTINUITY/" 2>/dev/null

# —— ⑥ 07_BUNDLES：各方向的独立包 ——
for z in "$ROOT"/DIR-*_artifacts.zip; do
  [ -f "$z" ] && cp -f "$z" "$PKG/07_BUNDLES/"
done

# —— ⓗ 00_README：包导航 ——
{
  echo "# SGMN v2.4 架构验证 Loop · 完整交付包"
  echo
  echo "> 生成：$(date -u "+%Y-%m-%d %H:%M:%S") UTC"
  echo "> 范围：**DIR-01…DIR-19 已闭合；DIR-20（跨会话状态语义）进行中**；\`SUCCEEDED = 14/14\`（**DIR-18 已 COMPLETE / SUCCESS** ⇒ 主假说【证实】）"
  echo
  echo "| 目录 | 内容 |"
  echo "|---|---|"
  echo "| \`01_FRAMEWORK/\` | **v2.4 架构设计原文**（本 loop 的唯一框架来源）+ 上游附件 + \`00_PROVENANCE.md\`（md5 与来源） |"
  echo "| \`02_ROUNDS/\` | 各方向的**装置 / 脚本 / 探针**（自包含单文件，附 md5 记录） |"
  echo "| \`03_LEDGERS/\` | 证据表 / 证据台账 / 方向登记表 / loop 机制 / 修订记录 / MANIFEST / 报告 |"
  echo "| \`04_EVIDENCE/\` | 各方向的**卡片（全部 revision）与裁决书**、结果 JSON |"
  echo "| \`05_LOGS/\` | 全部实验日志（含 **CPU 降级轮**的完整日志与增量落盘） |"
  echo "| \`06_CONTINUITY/\` | 运维 / 审计 / 打包脚本（**可复现性资产**） |"
  echo "| \`07_BUNDLES/\` | 各方向的独立产物包 |"
  echo
  echo "## 校验"
  echo
  echo "\`\`\`bash"
  echo "cd SGMN_LOOP_PACKAGE && md5sum -c 00_MANIFEST.md"
  echo "\`\`\`"
  echo
  echo "## ★ 阅读顺序建议"
  echo
  echo "1. \`03_LEDGERS/evidence_table.md\` —— **证据表**（只增行、不改行；含每条的「支持范围」与「不能外推为」）"
  echo "2. \`03_LEDGERS/SUMMARY.md\` —— 全 loop 摘要"
  echo "3. \`03_LEDGERS/LOOP-MECHANISM.md\` —— **loop 机制 + 46 条机制级纪律**"
  echo "4. \`03_LEDGERS/REPORT.md\` —— 完整报告"
  echo "5. \`04_EVIDENCE/DIR-20/VERDICT.md\` —— 最近一方向的裁决书（**跨会话状态语义**：折损 = 0，但收益的 69% 是通用注入）"
  echo
  echo "## ★ 一句话结论"
  echo
  echo "**v2.4 的实际写规则（Hebbian，键逐写归一化）在冻结基座上可用**，且**状态可跨会话搬运**（折损 = 0）；"
  echo "但**「记忆内容」只解释任务收益的一部分**（跨会话：内容特异 +1.72 / 通用注入 +3.79）"
  echo "⇒ ⇒ **「注入记忆结构 ≠ 召回记忆内容」**，任何只报告总收益的记忆结论都是不充分的。"
} > "$PKG/00_README.md"

# —— ⑧ 重算全包 MANIFEST ——
cd "$PKG"
find . -type f ! -name "00_MANIFEST.md" ! -path "*__pycache__*" ! -name "*.pyc" | sort | while read -r f; do
  md5sum "$f"
done > /tmp/_full_mf.txt
{
  echo "# 完整包 MANIFEST · SGMN v2.4 架构验证 Loop（DIR-01…DIR-20）"
  echo
  echo "> 生成：$(date -u "+%Y-%m-%d %H:%M:%S") UTC"
  echo "> 校验：\`cd SGMN_LOOP_PACKAGE && md5sum -c 00_MANIFEST.md\`"
  echo
  cat /tmp/_full_mf.txt
} > "$PKG/00_MANIFEST.md"
# ★ DEF-26P-12 硬断言：以下**归档必需品**必须在包内（缺一即中止，不许靠「源==包」断言糊过去，
#   因为 decisions/ 与 .ops/scripts/ 从来不进 ② 的内容根，两边同时漏 ⇒ 计数断言照不出来）。
for _must in "03_LEDGERS/COVERAGE-v24.md" \
              "03_LEDGERS/CARD-CHECKLIST.md" \
              "06_CONTINUITY/COMPLIANCE-BASELINE.md" \
              "03_LEDGERS/decisions/DECISIONS.md" \
             "06_CONTINUITY/.ops/scripts/rebuild_full.sh" \
             "06_CONTINUITY/.ops/scripts/repack.sh" \
             "06_CONTINUITY/.ops/scripts/audit_consistency.py" \
             "06_CONTINUITY/.ops/scripts/audit_counts.py" \
             "06_CONTINUITY/.ops/scripts/test_audit_counts.py" \
             "06_CONTINUITY/.ops/scripts/test_audit_boundary9.py" \
             "06_CONTINUITY/.ops/scripts/test_audit_k27judge.py" \
             "06_CONTINUITY/.ops/scripts/check_cards.py" \
             "06_CONTINUITY/.ops/scripts/blind_copy.sh" \
             "06_CONTINUITY/.ops/scripts/restore_out.sh"; do
  [ -f "$PKG/$_must" ] || { say "★★ 归档必需品缺失：$_must ⇒ 中止"; exit 9; }
done
say "归档必需品齐备：DECISIONS.md + CARD-CHECKLIST.md + COMPLIANCE-BASELINE.md + 六脚本（rebuild_full / repack / audit_consistency / check_cards / blind_copy / restore_out）"
N=$(wc -l < /tmp/_full_mf.txt)
say "全包文件数 = $N（源文件数 = $N_SRC）"
# ★ 2026-11-07：原实现用**硬编码下限 250**（旧沙箱布局的规模）⇒ 恢复后的工作区只有 186 个源文件
#   ⇒ 会被**误判为打包不完整** ⇒ ⇒ 改为**相对判据**：入包文件数必须 **≥ 源文件数**
#   （这才是 D61 「防静默缩水」的本意）。
if [ "$N" -lt "$N_SRC" ]; then
  say "★★ 完整包文件数 $N < 源文件数 $N_SRC ⇒ **有文件没进包**（D61 同型：静默缩水）⇒ 中止"
  exit 4
fi
if [ "$N" -lt "${MIN_FILES_ABS:-0}" ]; then
  say "★★ 完整包文件数 $N < 绝对下限 ${MIN_FILES_ABS} ⇒ 中止"
  exit 4
fi

# —— ⑨ 打包 ——
cd "$PKGDIR"
rm -f "$ROOT/SGMN_LOOP_full.zip" "$ROOT/LOOP_REPORT_full.zip"
# ★ DEF-26P-10（补）：`zip` 的 stderr 被丢弃 + 未校验产物 ⇒ 曾出现 **zip 为空而日志仍打印「结束」**。
#   ⇒ 不吞 stderr；产物必须存在且非空，否则**中止并且不打印成功横幅**。
if ! zip -qr "$ROOT/SGMN_LOOP_full.zip" SGMN_LOOP_PACKAGE -x "*__pycache__*" "*.pyc"; then
  say "★★ zip 返回非零 ⇒ 中止"; exit 6
fi
_SZ=$(stat -c%s "$ROOT/SGMN_LOOP_full.zip" 2>/dev/null || echo 0)
if [ "${_SZ:-0}" -le 0 ]; then
  say "★★ 完整包为空（${_SZ:-0} B）⇒ 中止，不产出/不报告成功"; exit 6
fi
if ! cp -f "$ROOT/SGMN_LOOP_full.zip" "$ROOT/LOOP_REPORT_full.zip"; then
  say "★★ LOOP_REPORT_full.zip 复制失败 ⇒ 中止"; exit 6
fi
say "SGMN_LOOP_full.zip = $(stat -c%s "$ROOT/SGMN_LOOP_full.zip") B  md5=$(md5sum "$ROOT/SGMN_LOOP_full.zip" | cut -d' ' -f1)"

# —— ⑩ 解压自检 + 逐文件 md5 复核 ——
rm -rf "$PKGDIR/_vfy" && mkdir -p "$PKGDIR/_vfy" && cd "$PKGDIR/_vfy"
unzip -q "$ROOT/SGMN_LOOP_full.zip"
cd SGMN_LOOP_PACKAGE
md5sum -c /tmp/_full_mf.txt > /tmp/_full_check.txt 2>&1
OK=$(grep -c ": OK$" /tmp/_full_check.txt)
BAD=$(grep -c ": FAILED$" /tmp/_full_check.txt)   # ★ 只匹配行尾的 FAILED（文件名里含 "FAILED" 不算失败）
say "全包 md5 复核：OK=$OK  FAILED=$BAD"
if [ "$BAD" != "0" ] || [ "$OK" != "$N" ]; then
  grep FAILED /tmp/_full_check.txt | head -10
  say "★★ 复核未通过（OK=$OK / 期望 $N，FAILED=$BAD）⇒ 中止，**不打印成功横幅**"
  exit 5
fi
# ★ DEF-26P-11：`md5sum -c` 只证明**清单内** N 个文件匹配；**清单外的条目不会被发现**。
#   实测 zip 非目录条目 = N + 1（唯一例外：包内自描述清单 `./00_MANIFEST.md`，其自身无法自列）。
#   ⇒ 显式数**非目录条目**并与 N+1 对齐；差集必须恰为 `./00_MANIFEST.md`，否则**中止**（不许靠改计数文字圆场）。
_NDIR=$(unzip -Z1 "$ROOT/SGMN_LOOP_full.zip" | grep -vc "/$")
if [ "$_NDIR" != "$((N+1))" ]; then
  say "★★ zip 非目录条目 $_NDIR != N+1（$((N+1))）⇒ 有清单外文件或多出/缺失 ⇒ 中止"
  unzip -Z1 "$ROOT/SGMN_LOOP_full.zip" | grep -v "/$" | sed 's|^SGMN_LOOP_PACKAGE/||;s|^|./|' | sort > /tmp/_z2.txt
  awk '{print $2}' /tmp/_full_mf.txt | sort > /tmp/_m2.txt
  say "  包内但清单缺：$(comm -23 /tmp/_z2.txt /tmp/_m2.txt | tr '\n' ' ')"
  say "  清单有但包内缺：$(comm -13 /tmp/_z2.txt /tmp/_m2.txt | tr '\n' ' ')"
  exit 8
fi
_EXTRA=$(unzip -Z1 "$ROOT/SGMN_LOOP_full.zip" | grep -v "/$" | sed 's|^SGMN_LOOP_PACKAGE/||;s|^|./|' | sort | comm -23 - <(awk '{print $2}' /tmp/_full_mf.txt | sort))
if [ "$_EXTRA" != "./00_MANIFEST.md" ]; then
  say "★★ 清单外条目不是唯一的 ./00_MANIFEST.md（实测：$_EXTRA）⇒ 中止"; exit 8
fi
say "zip 非目录条目 = $_NDIR = N(=$N) + 1（唯一例外 ./00_MANIFEST.md，自描述清单）"
say "===== 完整包重建结束（校验通过）====="
exit 0
