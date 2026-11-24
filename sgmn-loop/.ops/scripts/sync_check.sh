#!/usr/bin/env bash
# sync_check.sh —— 两侧 `.ops/scripts/` 的 **md5 清单双向差集**（只读）
#   Cycle 266 · `P134-14f` · 决定二百五十五
#
# ★★★ 为什么需要它（Cycle 265 **实测**踩到，不是假想）：
#   量「入包差集」时才发现本地 `.ops/scripts/_push114.sh` **在 trainer 上根本不存在** ——
#   历史「本地写了、从未 push」的残留。反向同理：「trainer 上手工改过、本地不知道」，
#   以及 `决定二百一十九` 那次**两侧同时被抹掉**的覆盖式同步事故。
#   两侧这份目录**从来没有过清单级断言**：`push52.sh` 只报「✔ 已推 N 个文件」，
#   从不报「**还有哪些没推**」；而 `audit_consistency.py` 只审包内一致性，看不见 trainer。
#
# ★ 纪律（写进脚本，不靠记忆）：
#   1. **只读**：不 push / 不 pull / 不删除 / 不覆盖任何文件。自动同步会掩盖
#      「该不该进包、该不该留」的人判断，且 `push52.sh` 的覆盖式写入曾是事故源。
#   2. **两侧同一套枚举口径**（同一段 `find … -printf`），否则「口径不同」会伪装成「文件不同」。
#   3. **ssh 失败必须单独报错**（`exit 5`），绝不能退化成「远端为空 ⇒ 全部显示为本地独有」——
#      假警报和静默失败一样会让人不再相信这个守卫（见 `.ops/ENV-NOTES.md` §9.4）。
#   4. 三份差集里**任何一项都要当场解释并留档**（是本轮新增未推？还是一侧误删？）。
#
# 用法：
#   .ops/scripts/sync_check.sh            # 只打印，退出码恒 0（供人看）
#   .ops/scripts/sync_check.sh --strict   # 任一差集非空 ⇒ exit 4（供闸门用）
set -uo pipefail

DEST_HOST="${DEST_HOST:-trainer}"
DEST_ROOT="${DEST_ROOT:-/home/ec2-user/sgmn-loop}"
DIR_REL="${DIR_REL:-.ops/scripts}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELF=sync_check.sh
STRICT=0

die() { echo "[$SELF] ★★ $*" >&2; exit "${2:-6}"; }

for a in "$@"; do
  case "$a" in
    --strict) STRICT=1 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "未知参数：$a（只接受 --strict / -h）" 2 ;;
  esac
done

[ -d "$REPO_ROOT/$DIR_REL" ] || die "本地目录不存在：$REPO_ROOT/$DIR_REL"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# —— 两侧**同一段**枚举/取 md5 的逻辑：一层普通文件、跳过 *.pyc、按名字排序 ——
SNAP='
cd "$1" || { echo "__DIR_MISSING__" >&2; exit 3; }
find . -maxdepth 1 -type f ! -name "*.pyc" -printf "%f\n" | sort | while IFS= read -r f; do
  printf "%s  %s\n" "$(md5sum "$f" | cut -d" " -f1)" "$f"
done
'

bash -c "$SNAP" _ "$REPO_ROOT/$DIR_REL" > "$TMP/local.txt" 2>"$TMP/local.err" \
  || die "本地枚举失败：$(cat "$TMP/local.err")" 3

# ★ ssh 失败 ⇒ exit 5。绝不能让它变成「远端为空」的空清单（否则全部误报为本地独有）。
if ! ssh -o ConnectTimeout=20 -o BatchMode=yes "$DEST_HOST" "bash -s -- '$DEST_ROOT/$DIR_REL'" \
      < <(printf '%s\n' "$SNAP") > "$TMP/remote.txt" 2>"$TMP/remote.err"; then
  die "ssh/枚举失败（远端 $DEST_HOST:$DEST_ROOT/$DIR_REL）：$(tr '\n' ' ' < "$TMP/remote.err")" 5
fi

awk '{print $2}' "$TMP/local.txt"  | sort > "$TMP/n_local.txt"
awk '{print $2}' "$TMP/remote.txt" | sort > "$TMP/n_remote.txt"
comm -23 "$TMP/n_local.txt"  "$TMP/n_remote.txt" > "$TMP/local_only.txt"
comm -13 "$TMP/n_local.txt"  "$TMP/n_remote.txt" > "$TMP/remote_only.txt"
sort -k2 "$TMP/local.txt"  > "$TMP/l.s"
sort -k2 "$TMP/remote.txt" > "$TMP/r.s"
join -j 2 "$TMP/l.s" "$TMP/r.s" | awk '$2!=$3{print $1}' | sort > "$TMP/differ.txt"

n_l=$(wc -l < "$TMP/n_local.txt"); n_r=$(wc -l < "$TMP/n_remote.txt")
n_lo=$(wc -l < "$TMP/local_only.txt"); n_ro=$(wc -l < "$TMP/remote_only.txt"); n_di=$(wc -l < "$TMP/differ.txt")

echo "== sync_check · $DIR_REL =="
echo "   本地 $REPO_ROOT/$DIR_REL : $n_l 个文件"
echo "   远端 $DEST_HOST:$DEST_ROOT/$DIR_REL : $n_r 个文件"
echo

show() {  # $1=标题 $2=文件 $3=提示
  local n; n=$(wc -l < "$2")
  if [ "$n" -eq 0 ]; then echo "✔ $1：0"; else
    echo "★★ $1：$n ——$3"
    sed 's/^/     · /' "$2"
  fi
}

echo "✔ 两侧 md5 相同：$(( n_l < n_r ? n_l : n_r )) 个共有文件（逐字节比对）"
show "仅本地有（本地改了/新建了，从未 push）" "$TMP/local_only.txt" " push 前先判断它该不该进归档"
show "仅远端有（远端新建/改名，本地没有）" "$TMP/remote_only.txt" " 下次 push 不会删它，但本地重建时会忘掉它"
show "两侧同名但内容不同" "$TMP/differ.txt" " 一次 push 就会覆盖其中一侧 ⇒ 先确认哪一侧是权威"

total=$(( n_lo + n_ro + n_di ))
echo
if [ "$total" -eq 0 ]; then
  echo "★ 结论：两侧 $DIR_REL 完全同步（$n_l 项，md5 全同）。"
  exit 0
fi
echo "★ 结论：$total 项分歧（本地独有 $n_lo + 远端独有 $n_ro + 内容不同 $n_di）。"
echo "  ★ 纪律（决定二百五十五）：每一处都要**当场解释并写进 decisions/** ——"
echo "    是本轮新增未 push？还是一侧误删？含糊放过 = 让下一次实验跑在陈旧代码上。"
if [ "$STRICT" = 1 ]; then
  echo "  ⇒ --strict：分歧非空，exit 4。"
  exit 4
fi
echo "  （未加 --strict，仅打印；退出码 0）"
exit 0
