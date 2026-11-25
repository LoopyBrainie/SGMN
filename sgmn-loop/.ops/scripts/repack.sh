#!/bin/bash
# DIR-18 · 打包：更新 DIR-18_artifacts.zip / 两个 full zip / MANIFEST，并逐文件 md5 复核
set -u
# ★ 路径可移植（2026-11-07）：原实现硬编码 `/workspace/sgmn-loop`（旧沙箱布局）⇒
#   实验卷上运行时报错 ⇒ ⇒ 改为**环境变量优先 + 默认当前工作区**（不改变任何逻辑）。
ROOT=${SGMN_ROOT:-/home/ec2-user/sgmn-loop}
LOG=$ROOT/.ops_pack.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "===== DIR-18 打包开始 ====="

# ---------- ⓪ 快照保留 N 代（Cycle 250 · `P134-4` · 决定二百四十）----------
# ★★★ 事故根因（`决定二百三十九` §六b）：本脚本末尾的 `rebuild_full.sh` 会**覆盖**
#   `SGMN_LOOP_full.zip`，而工作区只留这**一份**。2026-11-23 03:45 那次覆盖把 02:46
#   唯一完好的全量快照抹掉 ⇒ 290 个深层 `out/` 文件**永久丢失**。
#   ⇒ 每次 repack **先**把**现有**全量包存成一代快照（带 UTC 时间戳），只留最近 KEEP 代。
#   这样「上一次覆盖前的状态」总是还在，事故当晚不会被第二次 repack 抹掉。
# ★ 只修剪**本机制自己的家族**（前缀 `SGMN_LOOP_full_g`）：`_final_c163…c168.zip` 等
#   历史快照是**别的**机制留下的、也是某些深层文件的唯一来源 ⇒ **绝不能**被这里删掉。
SNAP_DIR=$ROOT/.ops/archive
KEEP=${SGMN_SNAP_KEEP:-5}
case "$KEEP" in ''|*[!0-9]*) say "★★ SGMN_SNAP_KEEP='$KEEP' 非正整数 ⇒ 退回 5"; KEEP=5 ;; esac
[ "$KEEP" -ge 1 ] || { say "★★ SGMN_SNAP_KEEP=$KEEP <1 ⇒ 退回 5"; KEEP=5; }
if [ -f "$ROOT/SGMN_LOOP_full.zip" ]; then
  mkdir -p "$SNAP_DIR" || say "★★ 无法创建 $SNAP_DIR ⇒ 无快照保护"
  _tag=$(date -u +%Y%m%dT%H%M%SZ)
  _snap=$SNAP_DIR/SGMN_LOOP_full_g$_tag.zip
  _i=1; while [ -e "$_snap" ]; do _snap=$SNAP_DIR/SGMN_LOOP_full_g${_tag}_$_i.zip; _i=$((_i+1)); done
  if cp -f "$ROOT/SGMN_LOOP_full.zip" "$_snap"; then
    say "快照保留：$(basename "$_snap")（$(stat -c%s "$_snap") B）"
  else
    say "★★ 快照复制失败 ⇒ 本次覆盖**无退路**，请先手动另存"
  fi
  # 修剪：按 mtime 新→旧，删第 KEEP+1 代及以后（**只碰 g 家族**）
  _n_before=$(ls -1 "$SNAP_DIR"/SGMN_LOOP_full_g*.zip 2>/dev/null | wc -l)
  ls -1t "$SNAP_DIR"/SGMN_LOOP_full_g*.zip 2>/dev/null \
    | tail -n +$((KEEP + 1)) \
    | while IFS= read -r _old; do rm -f "$_old" && say "修剪旧快照：$(basename "$_old")"; done
  _n_after=$(ls -1 "$SNAP_DIR"/SGMN_LOOP_full_g*.zip 2>/dev/null | wc -l)
  say "快照代数：$_n_before → $_n_after（上限 $KEEP；★ 不改动 _final_* 历史快照与 /tmp 手动另存）"
else
  say "无现有全量包（首次打包）⇒ 跳过快照"
fi

# ---------- ① DIR-18 产物包 ----------
# ★ 2026-11-07：恢复后的工作区里 DIR-18 在**顶层** `dir18/`（旧沙箱布局是 `revolve/DIR-18/`）。
rm -f "$ROOT/DIR-18_artifacts.zip"
cd "$ROOT/dir18" && zip -qr "$ROOT/DIR-18_artifacts.zip" \
  DIRECTION_rev15.md VERDICT.md \
  sgmn_module_hebb.py sgmn_module_hebb_v2.py \
  R1_write.py R2_write.py R4_formal.py \
  rounds probes pipeline18.sh probe18*.py probe18*_out.json \
  -x "*__pycache__*" -x "*.server.py" 2>/dev/null
say "DIR-18_artifacts.zip = $(stat -c%s "$ROOT/DIR-18_artifacts.zip") B"

# ---------- ①' DIR-19 产物包（新增）----------
rm -f "$ROOT/DIR-19_artifacts.zip"
cd "$ROOT/revolve/DIR-19" && zip -qr "$ROOT/DIR-19_artifacts.zip" \
  DIRECTION.md DIRECTION_rev11.md DIRECTION_rev12.md DIRECTION_rev13.md DIRECTION_rev14.md DIRECTION_rev15.md \
  sgmn_module_hebb_v3.py R1_protect.py VERDICT.md IMPLEMENTATION-PLAN.md \
  -x "*__pycache__*" -x "*.server.py" 2>/dev/null
say "DIR-19_artifacts.zip = $(stat -c%s "$ROOT/DIR-19_artifacts.zip") B"

# ---------- ①'' DIR-20 产物包（跨会话状态语义）----------
rm -f "$ROOT/DIR-20_artifacts.zip"
cd "$ROOT/revolve/DIR-20" && zip -qr "$ROOT/DIR-20_artifacts.zip" \
  DIRECTION.md DIRECTION_rev11.md DIRECTION_rev12.md DIRECTION_rev13.md \
  DIRECTION_rev14.md DIRECTION_rev15.md DIRECTION_rev16.md \
  sgmn_module_hebb_v4.py sgmn_module_hebb_v5.py sgmn_module_hebb_v6.py \
  R1_xsession.py VERDICT.md R20_results.json \
  _blk_arms.py _blk_readonly.py _blk_main.py _blk_crit.py _blk_stageA.py \
  assemble_dir20.py D20_R1_smoke.log D20_R1_smoke2.log D20_R1_smoke3.log \
  D20_R1_smoke4.log D20_R1_smoke5.log D20_R1_smoke6.log D20_R1_smoke7.log \
  D20_R1_smoke8.log D20_R2_formal.log D20_R2_formal_attempt2_diskfull.log \
  -x "*__pycache__*" -x "*.server.py" 2>/dev/null
say "DIR-20_artifacts.zip = $(stat -c%s "$ROOT/DIR-20_artifacts.zip") B"
# ---------- ①''' DIR-21 产物包（时间与情节状态（一）· COMPLETE / FALSIFIED）----------
rm -f "$ROOT/DIR-21_artifacts.zip"
cd "$ROOT/revolve/DIR-21" && zip -qr "$ROOT/DIR-21_artifacts.zip" \
  DIRECTION_rev21.md DIRECTION_rev21_rev11.md \
  DIRECTION_rev21_v12.md DIRECTION_rev21_v13.md DIRECTION_rev21_v14.md \
  DIRECTION_rev21_v15.md DIRECTION_rev21_v16.md DIRECTION_rev21_v17.md \
  DIRECTION_rev21_v18.md sgmn_module_hebb_v7.py R1_granularity.py VERDICT.md \
  probe21_capacity.py probe21b_decay.py probe21c_task.py \
  -x "*__pycache__*" -x "*.server.py" 2>/dev/null
say "DIR-21_artifacts.zip = $(stat -c%s "$ROOT/DIR-21_artifacts.zip") B"


# ---------- ② full 包（交给自足版 rebuild_full.sh：持久卷暂存 + 从零重建 + 完整性断言）----------
# ★ 缺陷 D61：原实现在此处建 `/tmp/SGMN_LOOP_PACKAGE` 并只做增量拷贝 ⇒ 沙箱回收后
#   **完整包静默缩水 368 → 39 文件**且零报错。现统一由 rebuild_full.sh 自足重建。

# ---------- ③ MANIFEST ----------
cd "$ROOT"
{
  # ★ 2026-11-09（DIR-26-N 收口）：原实现把标题与目录**写死**在 DIR-21（"SUCCEEDED = 15/15"）
  #   ⇒ 每多一轮就与台账**不一致**（实测 MANIFEST.md 自 2026-11-07 起停在 DIR-21/15-15）。
  #   改为**从活文档现读**：SUCCEEDED 取 `SUMMARY.md` 的头部，方向目录**按 `revolve/DIR-*` + `dir*` 自动枚举**。
  # ★ DEF-188-01（2026-11-20，Cycle 188 修）：上一版修法用了 `grep -o ... | head -1` ——
  #   那是 `re.search` 首匹配的 shell 写法，而 `SUMMARY.md` 是**追加式**日志 ⇒ `head -1`
  #   **必然是历史上最早的那个快照**（实测读到 `SUCCEEDED = 39 / 39`，真值 `49 / 49`）⇒
  #   过期数字**被烧进交付包里的 MANIFEST.md**（比只在终端打印的 §④ 更严重：没有 ④b 那样的交叉核）。
  #   修法：与 `audit_consistency.py` 的 §④/§④b 一致，**优先读规范标记**，无标记时才退回裸匹配。
  _SUCC=$(grep -o "当前计数：SUCCEEDED = [0-9]* / [0-9]*" "$ROOT/SUMMARY.md" | head -1)
  if [ -z "$_SUCC" ]; then
    _SUCC=$(grep -o "SUCCEEDED = [0-9]* / [0-9]*" "$ROOT/SUMMARY.md" | head -1)   # 兜底（无标记时）
  fi
  _SUCC=${_SUCC#当前计数：}      # 打印时去掉标记前缀，保持 MANIFEST 原格式
  _NDIR=$(ls -d "$ROOT"/revolve/DIR-* 2>/dev/null | wc -l)
  echo "# MANIFEST · SGMN Loop（\`revolve/DIR-*\` 目录 ${_NDIR} 个〔非方向总数〕；${_SUCC:-SUCCEEDED 未读到}）"
  echo
  echo "> 生成：$(date -u "+%Y-%m-%d %H:%M:%S") UTC"
  echo "> 校验：逐文件 md5（\`md5sum -c\`）"
  echo
  echo "## 核心台账"
  find . -maxdepth 2 -name "*.md" -not -path "./.ops*" -not -name "MANIFEST.md" | sort | while read -r f; do
    md5sum "$f"
  done
  echo
  echo "## 各方向产物（自动枚举）"
  for _d in "$ROOT"/revolve/DIR-* "$ROOT"/dir*; do
    [ -d "$_d" ] || continue
    case "$(basename "$_d")" in __pycache__|.ops) continue ;; esac
    echo
    echo "### $(basename "$_d")"
    ( cd "$ROOT" && find "./${_d#"$ROOT"/}" -type f -not -path "*__pycache__*" -not -path "*.ops/*" | sort ) | while read -r f; do
      md5sum "$f"
    done
  done
  echo
  echo "## 证据表与台账"
  md5sum ./evidence/table.md ./evidence/ledger.md
} > "$ROOT/MANIFEST.md"
say "MANIFEST.md = $(wc -l < "$ROOT/MANIFEST.md") 行"

# ---------- ④ 校验 ----------
cd "$ROOT"
grep -E "^[0-9a-f]{32}  " MANIFEST.md > /tmp/_mf.txt
md5sum -c /tmp/_mf.txt > /tmp/_mf_check.txt 2>&1
OK=$(grep -c ": OK$" /tmp/_mf_check.txt)
BAD=$(grep -c ": FAILED$" /tmp/_mf_check.txt)   # ★ 只匹配行尾的 FAILED（文件名里含 "FAILED" 不算失败）
say "MANIFEST 校验：OK=$OK  FAILED=$BAD"
# ★★★ P135-17（Cycle 280）：原实现**只报告不阻断** —— `FAILED>0` 时仅打印，随后照旧
#   走到末尾打印「打包结束（全部校验通过）」并 exit 0 ⇒ **包已损坏却宣告全部通过**。
#   这正是本脚本 line 146 自己记载的 `DEF-26P-10`「假成功」同型缺陷，当时只修了
#   `rebuild_full.sh` 的退出码，**没修自己的 md5 校验**。注入实测（篡改一个受管文件）：
#   `OK=1 FAILED=1` → 「全部校验通过」→ rc=0。⇒ 现改为硬闸门。
if [ "$BAD" != "0" ]; then
  grep "FAILED" /tmp/_mf_check.txt | head -20 | tee -a "$LOG"
  say "★★ MANIFEST 校验 FAILED=$BAD ⇒ 归档已损坏或被改写，**不打印「打包结束」**"
  exit 10
fi
# ---------- ⑤ full 包重建（自足版）----------
# ★ DEF-26P-10（补）：原实现**不检查退出码** ⇒ 完整包构建失败时仍打印「打包结束」（假成功）
if ! bash "$ROOT/.ops/scripts/rebuild_full.sh"; then
  say "★★ 完整包构建失败（非零退出）⇒ 打包失败，**不打印「打包结束」**"; exit 1
fi
say "===== 打包结束（全部校验通过）====="
