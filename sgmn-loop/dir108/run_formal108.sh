#!/bin/bash
# DIR-108 驱动：一格一进程（DEF-105-02：同进程混跑读数不可比）
# 用法：bash run_formal108.sh [cells|verdict]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/opt/pytorch/bin/python
cd "$HERE" || exit 1
mkdir -p out/cells
MODE=${1:-cells}

if [ "$MODE" = "cells" ]; then
  for spec in "gate 3000 7777" "gate 600 7777"; do
    set -- $spec
    tag="$1_$2_$3"
    if [ -f "out/cells/$tag.json" ]; then
      echo "[skip] $tag 已存在"; continue
    fi
    echo "===== [$tag] 开始 $(date -u +%H:%M:%S) ====="
    $PY R1_task108.py --stage cell --families "$1" --steps "$2" --seeds "$3" \
        2>&1 | grep -viE "futurewarning|pynvml" | tail -4
    echo "===== [$tag] 结束 $(date -u +%H:%M:%S) ====="
  done
else
  $PY R1_task108.py --stage verdict
fi
