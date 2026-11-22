#!/bin/bash
# DIR-119 驱动：Stage A（阻断）→ 自确定性双进程 → 18 格（**一格一进程**，DEF-105-02）→ merge 裁决。
# 每格落盘 dir119/out/cells/<ARM>_<dose>_<seed>.json，存在即跳过 ⇒ 可断点续跑。
cd "$(dirname "$0")/.." || exit 1
LOG=dir119/out/driver.log
PY=/opt/pytorch/bin/python
SEEDS="20271126 20271203 20271210 20260501 20260502 20260503 20260504 20260505 20260506"
mkdir -p dir119/out/cells dir119/out/det
echo "=== $(date -u +%FT%TZ) DIR-119 启动 PID $$ ===" >> "$LOG"

$PY dir119/R1_task119.py --stage A >> "$LOG" 2>&1
rc=$?; echo "--- stage A rc=$rc ---" >> "$LOG"
[ $rc -ne 0 ] && { echo "=== Stage A 未过，中止 ===" >> "$LOG"; exit 1; }

# 自确定性：同一 cell 两个独立进程（A119-12）
for tag in a b; do
  $PY dir119/R1_task119.py --stage det --tag "$tag" >> "$LOG" 2>&1
  echo "--- det $tag rc=$? ---" >> "$LOG"
done

# 18 格：一格一进程
for dose in 600 0; do
  for s in $SEEDS; do
    f=dir119/out/cells/COLOCATED_${dose}_${s}.json
    if [ -s "$f" ]; then echo "--- 跳过已有 $dose|$s ---" >> "$LOG"; continue; fi
    $PY dir119/R1_task119.py --stage cell --seed "$s" --dose "$dose" >> "$LOG" 2>&1
    echo "--- cell $dose|$s rc=$? ---" >> "$LOG"
  done
done

$PY dir119/R1_task119.py --stage merge >> "$LOG" 2>&1
echo "--- merge rc=$? ---" >> "$LOG"
echo "=== $(date -u +%FT%TZ) 完成 ===" >> "$LOG"
