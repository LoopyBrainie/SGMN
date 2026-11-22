#!/bin/bash
# DIR-125 驱动：Stage A（阻断）→ Smoke（阻断）→ 36 格（一格一进程，DEF-105-02）→ formal 裁决。
# 每格落盘 dir125/out/cells/COLOCATED_<dose>_<seed>.json，存在即跳过 ⇒ 可断点续跑。
cd "$(dirname "$0")/.." || exit 1
LOG=dir125/out/driver.log
PY=/opt/pytorch/bin/python
SEEDS="20260501 20260502 20260503 20260504 20260505 20260506 20271126 20271203 20271210"
DOSES="25 150 300 450"
mkdir -p dir125/out/cells dir125/out/det
echo "=== $(date -u +%FT%TZ) DIR-125 启动 PID $$ ===" >> "$LOG"

$PY dir125/precheck125.py >> "$LOG" 2>&1
rc=$?; echo "--- precheck rc=$rc ---" >> "$LOG"
[ $rc -ne 0 ] && { echo "=== 结构核未过，中止 ===" >> "$LOG"; exit 1; }

$PY dir125/test_mech125.py >> "$LOG" 2>&1
rc=$?; echo "--- selftest rc=$rc ---" >> "$LOG"
[ $rc -ne 0 ] && { echo "=== 测量层自测未过，中止 ===" >> "$LOG"; exit 1; }

$PY dir125/R1_task125.py --stage a >> "$LOG" 2>&1
rc=$?; echo "--- stage A rc=$rc ---" >> "$LOG"
[ $rc -ne 0 ] && { echo "=== Stage A 未过，中止 ===" >> "$LOG"; exit 1; }

$PY dir125/R1_task125.py --stage smoke >> "$LOG" 2>&1
rc=$?; echo "--- smoke rc=$rc ---" >> "$LOG"
[ $rc -ne 0 ] && { echo "=== Smoke 未过，中止 ===" >> "$LOG"; exit 1; }

# 36 格：一格一进程（先小剂量，早出读数；同剂量内按 seed）
for dose in $DOSES; do
  for s in $SEEDS; do
    f=dir125/out/cells/COLOCATED_${dose}_${s}.json
    if [ -s "$f" ]; then echo "--- 跳过已有 $dose|$s ---" >> "$LOG"; continue; fi
    $PY dir125/R1_task125.py --stage cell --dose "$dose" --seed "$s" >> "$LOG" 2>&1
    echo "--- cell $dose|$s rc=$? ---" >> "$LOG"
  done
done

$PY dir125/R1_task125.py --stage formal >> "$LOG" 2>&1
echo "--- formal rc=$? ---" >> "$LOG"
echo "=== $(date -u +%FT%TZ) 完成 ===" >> "$LOG"
