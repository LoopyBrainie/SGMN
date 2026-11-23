#!/bin/bash
# ============================================================================
# restore_out.sh —— 从**任一**全量 zip 还原 `dir*/out/`（只增不改，绝不删除）
#   Cycle 250 · `P134-4` · 决定二百四十
#
# ★★★ 为什么需要它（实测事故，`决定二百三十九` §六）：
#   还原被删的 `dir*/out` 时我用了 `unzip 'SGMN_LOOP_PACKAGE/04_EVIDENCE/dir*/out/*'`
#   —— 但 `rebuild_full.sh` 把**深度 > 2** 的文件分流到 `02_ROUNDS/`、`<= 2` 到
#   `04_EVIDENCE/` ⇒ 那条命令**只还原了浅层文件**，`out/cells/*`、`out/FDSNAP/*`
#   等深层文件**一个都没还原**。而 `cp -n`（只增不改）**不会报错**，
#   「恢复了 1336 个」看起来完全像一个成功 ⇒ **半还原静默通过**。
#   ⇒ 本脚本**从两个分流一起取**，并在结束时打印**逐分流**候选数与
#     **结构性总数**（`find dir*/out -type f | wc -l`），让"只还原了一半"无所遁形。
#
# 用法:
#   restore_out.sh <zip> [--root DIR] [--dry-run] [--expect N]
#     <zip>        全量包（SGMN_LOOP_full*.zip 等，须含 SGMN_LOOP_PACKAGE/ 前缀）
#     --root DIR   还原到 DIR（默认 $SGMN_ROOT）
#     --dry-run    只报告将要还原什么，不落盘
#     --expect N   结束时断言 `find dir*/out -type f | wc -l` == N（不符即 exit 非零）
#
# 退出码: 0 成功 / 2 用法错 / 3 zip 无 out 内容 / 4 结构性断言不符 / 5 还原失败
# ============================================================================
set -u

SELF=$(basename "$0")
die()  { echo "[$SELF] ★★ $*" >&2; exit "${2:-4}"; }
info() { echo "[$SELF] $*"; }

usage() { echo "用法：$SELF <zip> [--root DIR] [--dry-run] [--expect N]" >&2; exit 2; }

ZIP=""; ROOT=${SGMN_ROOT:-/home/ec2-user/sgmn-loop}; DRY=0; EXPECT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --root)     ROOT=${2:-}; shift 2 ;;
    --dry-run)  DRY=1; shift ;;
    --expect)   EXPECT=${2:-}; shift 2 ;;
    -h|--help)  usage ;;
    -*)         die "未知选项：$1" 2 ;;
    *)          ZIP=$1; shift ;;
  esac
done
[ -n "$ZIP" ] || usage
[ -f "$ZIP" ] || die "zip 不存在：$ZIP" 2
[ -d "$ROOT" ] || die "root 不是目录：$ROOT" 2
command -v unzip >/dev/null || die "缺少 unzip" 2

ROOT_REAL=$(readlink -f "$ROOT") || die "无法解析 root" 2
# 粗检：像不像工作区（防把归档解到 / 或 home）
ls -d "$ROOT_REAL"/dir* >/dev/null 2>&1 || ls -d "$ROOT_REAL"/revolve >/dev/null 2>&1 \
  || die "'$ROOT_REAL' 里既无 dir*/ 也无 revolve/ ⇒ 不像工作区，拒绝写入" 2

n_out() { find "$ROOT_REAL"/dir*/out -type f 2>/dev/null | wc -l; }

# ---------------------------------------------------------------- ① 枚举候选
# ★ 同时扫**两个分流**，并把每个候选的**来源分流**记下来（这就是"覆盖两处"的证据）。
#   `rebuild_full.sh` 的分流规则：深度 > 2 → 02_ROUNDS/，深度 <= 2 → 04_EVIDENCE/。
#   ★ 也兼容**扁平**布局（老包里没有分流前缀）：此时来源记为 (flat)。
TMPL=$(mktemp /tmp/_restore_list.XXXXXX) || die "mktemp 失败" 5
ENTRIES=$(mktemp /tmp/_restore_all.XXXXXX) || die "mktemp 失败" 5
unzip -Z1 "$ZIP" > "$ENTRIES" 2>/dev/null || die "无法读 zip 目录：$ZIP" 3

: > "$TMPL"
# ★★ 关键：必须同时保留**两个**路径，不能混用（本脚本第一版就混了，导致
#   `unzip` 静默匹配 0 个成员却报「还原成功」—— 正是本脚本要防的那种静默半还原）：
#     $2 = zip **内**的完整成员名  → 只能用来寻址归档
#     $3 = **去掉分流前缀**后的目标相对路径 → 只能用来决定落盘位置
awk '
  {
    full = $0
    e = full
    sub(/^SGMN_LOOP_PACKAGE\//, "", e)          # 去包前缀（可有可无）
    if (e ~ /\/$/) next                          # 跳过目录项
    # ★★ 分流**不止两个**（Cycle 250 实测真包才发现）：`rebuild_full.sh` 先按**扩展名**分流
    #   （`*.log/*.out/*.txt → 05_LOGS`），再按**深度**分（`>2 → 02_ROUNDS`，`<=2 → 04_EVIDENCE`）。
    #   ⇒ `dirX/out/*.log` 落在 `05_LOGS/`，若只识别前两个分流就会**静默漏掉全部日志**
    #     （实测真包：只认两分流得 1388 个候选，磁盘上 `dir*/out` 有 1610 个 ⇒ 差 222 个正是日志）。
    #   故改为**泛化**：剥掉任意 `0X_LABEL/` 前缀，再把剩下的按 `dir*/out/...` 匹配。
    #   落盘目标恒为「剥掉分流前缀后的路径」，与分流无关 ⇒ 将来新增分流也不必改这里。
    stream = "(flat)"
    if (match(e, /^0[1-7]_[A-Z_]+\//)) {
      stream = substr(e, 1, RLENGTH - 1)
      e = substr(e, RLENGTH + 1)
    }
    # 只取 dir*/out/... 形态（其它分流与其它内容一概不碰）
    if (e ~ /^dir[^\/]+\/out\/.+/) print stream "\t" full "\t" e
  }
' "$ENTRIES" | sort -u > "$TMPL"

n_cand=$(wc -l < "$TMPL")
[ "$n_cand" -gt 0 ] || { rm -f "$TMPL" "$ENTRIES"; die "zip 里**没有任何** dir*/out/... 条目 ⇒ 这个包不含 out 内容（换个包）" 3; }

info "zip = $ZIP（$(stat -c%s "$ZIP") B）"
info "root = $ROOT_REAL"
info "候选条目 = $n_cand（**逐分流**统计，两处都必须看）"
# ★ 分流清单**从数据现读**、不硬编码（`DEF-128-03`：不把条目数/分布写死进脚本）
_streams=$(awk -F'\t' '{print $1}' "$TMPL" | sort -u)
for s in $_streams; do
  c=$(awk -F'\t' -v s="$s" '$1==s' "$TMPL" | wc -l)
  printf "[%s]   分流 %-14s 候选 %5d 个\n" "$SELF" "$s" "$c"
done
info "分流数 = $(echo "$_streams" | wc -w)（★ 真包通常 ≥3：02_ROUNDS/04_EVIDENCE/**05_LOGS**）"
info "候选涉及目录 = $(awk -F'\t' '{split($3,a,"/"); print a[1]}' "$TMPL" | sort -u | wc -l) 个"
_n_disk=$(n_out)
info "源树当前 out 文件数 = $_n_disk（**还原前**）"
# ★ 灾备问答（Cycle 250 · 真包实测 1610 候选 == 1610 磁盘文件，完全吻合）：
#   「这个包还能救回多少」是事故后第一个问题。候选 < 磁盘 ⇒ 包**解释不了**磁盘上现存的全部
#   out 文件（包更旧，或该文件从未入包）⇒ 只提示、**不判失败**（磁盘比包新是合法状态）。
if [ "$n_cand" -lt "$_n_disk" ]; then
  info "ℹ 包可解释磁盘上 $n_cand/$_n_disk 个 out 文件（差 $((_n_disk - n_cand)) 个：包更旧，或从未入包）"
else
  info "★ 包可解释磁盘上 $n_cand/$_n_disk 个 out 文件 ⇒ 若磁盘被毁，本包可**完整**还原 out/ 部分"
fi

if [ "$DRY" -eq 1 ]; then
  info "--dry-run：只报告，不落盘。前 5 条候选："
  head -5 "$TMPL" | awk -F'\t' '{printf "  [%s] %s\n", $1, $3}'
  rm -f "$TMPL" "$ENTRIES"; exit 0
fi

# ---------------------------------------------------------------- ② 解出到临时目录
TD=$(mktemp -d /tmp/_restore_tmp.XXXXXX) || die "mktemp -d 失败" 5
trap 'rm -rf "$TD"' EXIT

mapfile -t _want < <(awk -F'\t' '{print $2}' "$TMPL")
# ★ 扁平布局没有 SGMN_LOOP_PACKAGE/ 前缀 ⇒ 逐个试，存在即用
_n_want=${#_want[@]}
info "解压 $_n_want 个条目到临时目录…"
# 先按带前缀解；失败条目由第二遍（无前缀）补
unzip -q -o "$ZIP" "${_want[@]}" -d "$TD" 2>/dev/null
# 校验：所有候选都解出来了？（`unzip` 对不存在的成员只警告）
_missing=0
while IFS=$'\t' read -r _s _ziprel _rel; do
  [ -f "$TD/$_ziprel" ] || _missing=$((_missing+1))
done < "$TMPL"
if [ "$_missing" -gt 0 ]; then
  info "⚠ 有 $_missing 个条目没解出（可能前缀不同）⇒ 用无前缀模式补解"
  unzip -q -o "$ZIP" "$(awk -F'\t' '{print $2}' "$TMPL" | head -100000)" -d "$TD" 2>/dev/null || true
  # ★ 补解后**必须**复查：仍缺就如实失败，绝不带着缺口继续（那正是事故的形态）
  _still=0
  while IFS=$'\t' read -r _s _ziprel _rel; do
    [ -f "$TD/$_ziprel" ] || _still=$((_still+1))
  done < "$TMPL"
  [ "$_still" -eq 0 ] || die "解压后仍缺 $_still 个条目 ⇒ 归档与候选清单不一致，中止（不落盘）" 5
  info "补解后复查通过（0 缺失）"
fi

# ---------------------------------------------------------------- ③ 只增不改地落盘
added_total=0; kept_total=0; fail_total=0
declare -a _perdir=()
# ★★ 落盘必须**逐 (分流, 目录)** 进行：同一个 `dirX/out` 完全可能**同时**有
#   深层（在 `02_ROUNDS/`）与浅层（在 `04_EVIDENCE/`）两个来源 ⇒ 只按目录名合并会漏掉一半。
#   （本脚本第二版就漏了：候选找到了、解压也成功了，但复制时丢掉了分流前缀
#     ⇒ 全部目录报「缺 …/out」而计数器仍是 0 新增 —— 靠**结构性总数**才看出来。）
while IFS=$'\t' read -r _s d; do
  case "$_s" in
    "(flat)") src="$TD/SGMN_LOOP_PACKAGE/$d/out"; [ -d "$src" ] || src="$TD/$d/out" ;;
    *)        src="$TD/SGMN_LOOP_PACKAGE/$_s/$d/out" ;;
  esac
  [ -d "$src" ] || { info "⚠ 缺 [$_s] $d/out（跳过）"; continue; }
  dst="$ROOT_REAL/$d/out"
  b=$(find "$dst" -type f 2>/dev/null | wc -l)
  mkdir -p "$dst" || { info "★★ mkdir 失败：$dst"; fail_total=$((fail_total+1)); continue; }
  cp -rn "$src/." "$dst/" 2>/dev/null || info "⚠ cp 有报错：[$_s] $d"
  a=$(find "$dst" -type f 2>/dev/null | wc -l)
  if [ "$a" -gt "$b" ]; then
    added_total=$((added_total + a - b)); _perdir+=("$d +$((a - b))")
  else
    kept_total=$((kept_total + a))
  fi
done < <(awk -F'\t' '{split($3,a,"/"); print $1 "\t" a[1]}' "$TMPL" | sort -u)

# ★ `set -u` 下空数组的 `${arr[*]}` 在 bash < 4.4 会报 unbound ⇒ 先判长度再用
if [ "${#_perdir[@]}" -eq 0 ]; then
  info "逐目录新增：（无新增）"
else
  _pd=$(printf '%s | ' "${_perdir[@]:0:15}"); info "逐目录新增（最多 15 个）：${_pd% | }"
fi
# ---------------------------------------------------------------- ④ 结构性验收
final=$(n_out)
info "──────────────────────────────────────────────"
info "新增还原 = $added_total 个"
info "已存在（未触碰）涉及 = $kept_total 个"
info "★ 结构性总数：find dir*/out -type f | wc -l = **$final**"
info "──────────────────────────────────────────────"
if [ -n "$EXPECT" ]; then
  [ "$final" = "$EXPECT" ] || die "结构性断言不符：期望 $EXPECT，实测 $final（差 $((EXPECT - final))）" 4
  info "结构性断言通过：$final == $EXPECT"
fi
[ "$fail_total" -eq 0 ] || die "有 $fail_total 个目录还原失败" 5
info "✅ 完成（只增不改；**未删除任何文件**）"
exit 0