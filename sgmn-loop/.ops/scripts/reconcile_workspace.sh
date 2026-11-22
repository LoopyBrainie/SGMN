#!/bin/bash
# SGMN Loop · 工作区 ↔ 交付包 **逐文件 md5 对账**（纪律 #43 的落地脚本）
#
# 为什么需要它（2026-11-07 实测）：把交付包解压回训练机后，
# `dir18/rounds/R2_write.py` 的 md5 是 `0545ecc7…`，而**交付包内**同一文件是
# `12dc22f6…`（= 证据台账里记录的值）⇒ ⇒ **工作区里的副本是 D48 时代的旧版**
# （`clear_slots(st)`），而**交付包与台账**是 D55/rev-1.2 修正版
# （沿用 DIR-17 原协议 + 精确槽号映射）⇒ ⇒ 工作区 ≠ 已审计的交付记录。
#
# 对账方式（**内容寻址**，不依赖路径猜测）：
#   ① 把工作区每个文件算一次 md5，建索引；
#   ② 对 MANIFEST 的每一条，找「同 md5 的文件」在工作区的**任何位置**：
#        - 命中 ⇒ OK（并报告是否**移动过位置**）；
#        - 同名但 md5 不同 ⇒ **STALE**（工作区是旧版 —— 危险！）；
#        - 没有同名也没有同 md5 ⇒ **MISSING**（工作区丢了）。
#
# 用法：bash reconcile_workspace.sh <包解压目录> [工作区根]
set -u
PKG=${1:?用法: reconcile_workspace.sh <包解压目录> [工作区根]}
ROOT=$(readlink -f "${2:-/home/ec2-user/sgmn-loop}")   # ★ 工作区是符号链接 ⇒ 必须解析成物理路径
echo "工作区（物理路径）= $ROOT"
MF="$PKG/SGMN_LOOP_PACKAGE/00_MANIFEST.md"
[ -f "$MF" ] || { echo "✗ 找不到 MANIFEST: $MF"; exit 2; }

WS_IDX=/tmp/_ws_md5_index.txt
: > "$WS_IDX"
find "$ROOT" -type f -not -path "*/.pkg/*" -not -path "*/__pycache__/*" -not -name "*.pyc" \
  | while read -r f; do md5sum "$f"; done > "$WS_IDX"
N_WS=$(wc -l < "$WS_IDX")
echo "工作区文件数 = $N_WS"

ok=0; stale=0; missing=0; moved=0
: > /tmp/_rec_missing.txt; : > /tmp/_rec_stale.txt; : > /tmp/_rec_moved.txt

grep -E "^[0-9a-f]{32}  " "$MF" | sed "s|^\([0-9a-f]\{32\}\)  |\1 |" | while IFS=' ' read -r want rel; do
  rel=${rel#./}
  hit=$(grep -m1 "^$want " "$WS_IDX" | cut -d' ' -f2-)
  base=$(basename "$rel")
  same=$(grep -m1 " $base\$" "$WS_IDX" | cut -d' ' -f1)
  if [ -n "$hit" ]; then
    echo "OK $rel ⇒ $hit" >> /tmp/_rec_ok.txt
  elif [ -n "$same" ]; then
    echo "STALE $rel  工作区同名文件 md5=$same（包内 $want）" >> /tmp/_rec_stale.txt
  else
    echo "MISSING $rel" >> /tmp/_rec_missing.txt
  fi
done

echo
echo "================ 对账结果 ================"
printf "OK      %s\n" "$( [ -f /tmp/_rec_ok.txt ] && wc -l < /tmp/_rec_ok.txt || echo 0)"
printf "STALE   %s   ★ 工作区存在同名文件但 md5 不同（= 旧版，**不能**当作已审计版本）\n" "$( [ -f /tmp/_rec_stale.txt ] && wc -l < /tmp/_rec_stale.txt || echo 0)"
printf "MISSING %s\n" "$( [ -f /tmp/_rec_missing.txt ] && wc -l < /tmp/_rec_missing.txt || echo 0)"
echo
echo "—— STALE 明细（最多 25 条）——"; [ -f /tmp/_rec_stale.txt ] && head -25 /tmp/_rec_stale.txt
echo "—— MISSING 明细（最多 15 条）——"; [ -f /tmp/_rec_missing.txt ] && head -15 /tmp/_rec_missing.txt
exit 0
