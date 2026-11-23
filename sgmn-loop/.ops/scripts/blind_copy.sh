#!/bin/bash
# ============================================================================
# blind_copy.sh —— 把工作区复制到一个**与源完全无别名**的目录
#   Cycle 250 · `P134-4` · 决定二百四十
#
# ★★★ 为什么需要它（实测事故，`决定二百三十九` §五/§六）：
#   2026-11-23「盲态实验」用 `cp -a ~/sgmn-loop /tmp/blind2` 复制工作区 ——
#   而 `~/sgmn-loop` 是**符号链接**（`-> /mnt/sgmn/home/ec2-user/sgmn-loop`）。
#   `cp -a` **保留符号链接本身** ⇒ `/tmp/blind2` 也**指向真实树** ⇒
#   随后 `cd /tmp/blind2 && rm -rf dir*/out` **删掉的是真实归档**
#   （125 个目录；其中 290 个深层文件永久丢失）。
#   ⇒ **任何**「复制工作区再破坏」的实验**必须**走本脚本：它用 `cp -rL`
#     （**解引用所有**符号链接）并在复制**之后**用四条断言证明副本与源无别名，
#     任一断言不过即 exit 非零。断言是**机械的**，不依赖操作者的记忆。
#
# 用法:
#   blind_copy.sh <dest> [--force]         复制工作区到 <dest>（须不存在或为空）
#   blind_copy.sh --assert-isolated <dir>  仅做隔离断言（**破坏性操作前**调用）
#
# 环境: SGMN_ROOT（默认 /home/ec2-user/sgmn-loop）
# 退出码: 0 成功 / 2 用法错 / 3 dest 非空 / 4 断言失败 / 5 复制失败
# ============================================================================
set -u

ROOT=${SGMN_ROOT:-/home/ec2-user/sgmn-loop}
MARK=.blind_copy_ok          # 隔离证明标记（由本脚本写入 dest 根）
SELF=$(basename "$0")

die()  { echo "[$SELF] ★★ $*" >&2; exit "${2:-4}"; }
info() { echo "[$SELF] $*"; }

# 文件数（**全部**常规文件，不做任何排除）
# ★ 注意：这与 `rebuild_full.sh` 的 `N_SRC` **口径不同**（后者只数内容根且排除
#   `__pycache__`）⇒ 两者**不可直接比较**（`决定二百三十九` §六 的实测教训：
#   `find` 测 3337、脚本测 3324，差 13 全是口径差）。本脚本只用它作**同一口径前后对比**。
nfiles() { find "$1" -type f 2>/dev/null | wc -l; }

# ---------------------------------------------------------------- 断言核心
# 四条断言：① dest 是**真目录**不是链接；② 真路径与源不同；
#           ③ inode 不同（防 bind-mount / 硬别名）；④ 内部无**逃逸**符号链接。
assert_isolated() {
  local d=$1 src_real=$2 why=""
  [ -d "$d" ] || die "断言①：'$d' 不是目录" 4
  [ ! -L "$d" ] || die "断言①：'$d' **是符号链接**（指向 $(readlink -f "$d")）⇒ 副本会与源别名" 4
  local d_real; d_real=$(readlink -f "$d") || die "断言②：无法解析 '$d'" 4
  [ "$d_real" != "$src_real" ] || die "断言②：副本真路径 == 源真路径（$src_real）" 4
  local si di; si=$(stat -c '%d:%i' "$src_real" 2>/dev/null); di=$(stat -c '%d:%i' "$d_real" 2>/dev/null)
  [ -n "$si" ] && [ -n "$di" ] || die "断言③：stat 失败" 4
  [ "$si" != "$di" ] || die "断言③：源与副本 inode 相同（$si）⇒ 它们是**同一份数据**" 4
  # ④ 逃逸符号链接：任何链接解析后落在 dest 之外 ⇒ 破坏 dest 会波及外部
  local esc=0 t
  while IFS= read -r lnk; do
    t=$(readlink -f "$lnk" 2>/dev/null) || t=""
    case "$t" in
      "$d_real"/*|"") : ;;
      *) esc=$((esc+1)); [ "$esc" -le 5 ] && echo "[$SELF]   ⚠ 逃逸链接：${lnk#$d_real/} -> $t" >&2 ;;
    esac
  done < <(find "$d_real" -type l 2>/dev/null)
  [ "$esc" -eq 0 ] || die "断言④：副本内有 $esc 个符号链接指向副本之外 ⇒ 破坏副本会波及外部" 4
  info "断言 ①②③④ 全部通过（副本 '$d_real' 与源 '$src_real' 无别名，内部无逃逸链接）"
  return 0
}

# ---------------------------------------------------------------- 子命令
if [ "${1:-}" = "--assert-isolated" ]; then
  d=${2:-}
  [ -n "$d" ] || die "用法：$SELF --assert-isolated <dir>" 2
  [ -d "$d" ] || die "'$d' 不存在" 4
  # ★ 只认**由本脚本生成**的目录：没有标记就说明它可能是个手抄的路径（正是事故的成因）
  [ -f "$d/$MARK" ] || die "'$d' 没有 $MARK 标记 ⇒ **不是** blind_copy.sh 生成的副本（拒绝在未证明隔离的目录上做破坏性操作）" 4
  rec_src=$(sed -n 's/^src=//p' "$d/$MARK" | head -1)
  src_real=$(readlink -f "$ROOT" 2>/dev/null) || die "无法解析 SGMN_ROOT='$ROOT'" 4
  [ "$rec_src" = "$src_real" ] || die "标记记录的源（$rec_src）与当前 SGMN_ROOT（$src_real）不符 ⇒ src/环境已变，拒绝" 4
  info "标记核验通过：src=$rec_src  复制时间=$(sed -n 's/^utc=//p' "$d/$MARK" | head -1)  文件数=$(sed -n 's/^files=//p' "$d/$MARK" | head -1)"
  assert_isolated "$d" "$src_real"
  info "★ 可以安全地在该目录内做破坏性操作"
  exit 0
fi

DEST=${1:-}; FORCE=0
[ -n "$DEST" ] || { echo "用法：$SELF <dest> [--force]  |  $SELF --assert-isolated <dir>" >&2; exit 2; }
[ "${2:-}" = "--force" ] && FORCE=1

[ -d "$ROOT" ] || die "SGMN_ROOT='$ROOT' 不是目录" 2
src_real=$(readlink -f "$ROOT") || die "无法解析 SGMN_ROOT='$ROOT'" 2

# dest 必须不存在或为空（除非 --force）；且不得与源同一路径
mkdir -p "$DEST" || die "无法创建 '$DEST'" 5
dest_real=$(readlink -f "$DEST")
[ "$dest_real" != "$src_real" ] || die "目标就是源（$src_real）⇒ 拒绝" 2
case "$dest_real" in
  "$src_real"/*) die "目标 '$dest_real' 在**源目录内部** ⇒ 会自我递归复制，拒绝" 2 ;;
esac
case "$src_real" in
  "$dest_real"/*) die "源 '$src_real' 在**目标内部** ⇒ 拒绝" 2 ;;
esac
if [ "$FORCE" -ne 1 ] && [ -n "$(ls -A "$dest_real" 2>/dev/null)" ]; then
  die "目标 '$dest_real' 非空（用 --force 覆盖；★ 注意 --force 不会删除目标内多余文件）" 3
fi

# ---------------------------------------------------------------- 复制
n_src_before=$(nfiles "$src_real")
info "源 '$src_real'（$( [ -L "$ROOT" ] && echo "★ 该路径本身是符号链接 -> $src_real" || echo "真目录" )）文件数=$n_src_before"
info "复制中：cp -rL（**解引用**所有符号链接）→ '$dest_real'"

# ★★ 关键：`-L`。用 `cp -a`（= -d，不 deref）会把符号链接原样搬过去 ⇒ 副本仍旧别名。
if ! cp -rLp "$src_real/." "$dest_real/" 2>/tmp/_blind_copy_err.$$; then
  echo "[$SELF] 复制报错（前 10 行）：" >&2; head -10 /tmp/_blind_copy_err.$$ >&2
  rm -f /tmp/_blind_copy_err.$$; die "复制失败" 5
fi
rm -f /tmp/_blind_copy_err.$$

n_dest=$(nfiles "$dest_real")
n_src_after=$(nfiles "$src_real")

# ---------------------------------------------------------------- 证据
echo "src=$src_real"  > "$dest_real/$MARK"
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$dest_real/$MARK"
echo "files=$n_dest" >> "$dest_real/$MARK"
info "副本文件数=$n_dest（源 $n_src_before → $n_src_after）"
# ★ 「复制没动到源」是事故的直接检验：源的文件数前后必须一致
[ "$n_src_before" = "$n_src_after" ] || die "复制**改变了源**的文件数（$n_src_before → $n_src_after）⇒ 严重异常" 5
[ "$n_dest" -gt 0 ] || die "副本为空" 5
# ★ 副本不得少于源（`cp -rL` 遇断链会跳过 ⇒ 必须报出来而不是静默变少）
if [ "$n_dest" -lt "$n_src_after" ]; then
  info "⚠ 副本比源少 $((n_src_after - n_dest)) 个文件（多为断链或权限）—— 若不该少，先查再破坏"
fi

assert_isolated "$dest_real" "$src_real"
info "✅ 完成：'$dest_real' 已证明与源无别名。破坏性操作前请先跑："
info "     $SELF --assert-isolated '$dest_real'"
exit 0