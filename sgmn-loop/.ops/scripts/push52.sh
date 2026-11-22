#!/usr/bin/env bash
# push52.sh —— **唯一允许**把本地文件推到 trainer 的入口（Cycle 85 起）。
#
# 事故背景（DEF-52-5）：Cycle 84 的一条
#     rsync -a dir52/ revolve/DIR-52/ trainer:/home/ec2-user/sgmn-loop/ --relative
# 把**本地镜像里陈旧的** `dir52/out/DIR52r12_shard*.jsonl`（各 21 行）
# 覆盖到 trainer 的**权威输出**上 ⇒ 远程 shard 路径被换成 21 行新 inode，
# 而 4 个正式轮写手仍持有旧 inode 的 fd（`(deleted)`），其后续追加全部写进未链接 inode。
# 数据未丢（fd 可读回），但**路径被毒化**。
#
# 规则：`out/` 是**只出不进** —— 它可以被 rsync **下来**，永远不许被推上去。
#
# 用法：  .ops/scripts/push52.sh dir52/R1_task52.py revolve/DIR-52/VERDICT.md
#         （只接受**具体文件**；拒绝目录、拒绝任何位于 `out/` 下的路径）
set -euo pipefail

DEST_HOST="${DEST_HOST:-trainer}"
DEST_ROOT="${DEST_ROOT:-/home/ec2-user/sgmn-loop}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ "$#" -eq 0 ]; then
  echo "用法: push52.sh <repo-relative-file> [more files...]" >&2
  echo "  ★ 只推具体文件；out/ 只出不进。" >&2
  exit 2
fi

bad=0
for f in "$@"; do
  case "$f" in
    */out/*|*/out)   echo "拒绝: '$f' 位于 out/ 之下（out/ 只出不进）" >&2; bad=1 ;;
  esac
  if [ -d "$REPO_ROOT/$f" ]; then
    echo "拒绝: '$f' 是目录（请列具体文件，避免整目录覆盖）" >&2; bad=1
  elif [ ! -f "$REPO_ROOT/$f" ]; then
    echo "拒绝: '$f' 不是本地存在的常规文件" >&2; bad=1
  fi
done
[ "$bad" -eq 0 ] || exit 3

cd "$REPO_ROOT"
rsync -a --relative --exclude 'out/' "$@" "${DEST_HOST}:${DEST_ROOT}/"
echo "✔ 已推 ${#} 个文件 → ${DEST_HOST}:${DEST_ROOT}/"