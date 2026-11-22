#!/bin/bash
# DIR-126 驱动：Stage A（阻断）→ Smoke（阻断）→ 27 格（一格一进程，DEF-105-02）→ formal 裁决。
# 每格落盘 dir126/out/cells/COLOCATED_<dose>_<seed>.json，存在即跳过 ⇒ 可断点续跑。
# ★ 剂量轴只是 DIR-125 的三点加密（{350,375,400}），装置与题集全复用。
cd "$(dirname "$0")/.." || exit 1
LOG=dir126/out/driver.log
PY=/opt/pytorch/bin/python
SEEDS="20260501 20260502 20260503 20260504 20260505 20260506 20271126 20271203 20271210"
DOSES="350 375 400"
mkdir -p dir126/out/cells dir126/out/det
echo "=== $(date -u +%FT%TZ) DIR-126 启动 PID $$ ===" >> "$LOG"

$PY dir126/precheck126.py >> "$LOG" 2>&1
rc=$?; echo "--- precheck rc=$rc ---" >> "$LOG"
[ $rc -ne 0 ] && { echo "=== 结构核未过，中止 ===" >> "$LOG"; exit 1; }

$PY dir126/test_mech126.py >> "$LOG" 2>&1
rc=$?; echo "--- selftest rc=$rc ---" >> "$LOG"
[ $rc -ne 0 ] && { echo "=== 测量层自测未过，中止 ===" >> "$LOG"; exit 1; }

$PY dir126/R1_task126.py --stage a >> "$LOG" 2>&1
rc=$?; echo "--- stage A rc=$rc ---" >> "$LOG"
[ $rc -ne 0 ] && { echo "=== Stage A 未过，中止 ===" >> "$LOG"; exit 1; }

$PY dir126/R1_task126.py --stage smoke >> "$LOG" 2>&1
rc=$?; echo "--- smoke rc=$rc ---" >> "$LOG"
[ $rc -ne 0 ] && { echo "=== Smoke 未过，中止 ===" >> "$LOG"; exit 1; }

# 27 格：一格一进程（先 350，早出读数）
for dose in $DOSES; do
  for s in $SEEDS; do
    f=dir126/out/cells/COLOCATED_${dose}_${s}.json
    if [ -s "$f" ]; then echo "--- 跳过已有 $dose|$s ---" >> "$LOG"; continue; fi
    $PY dir126/R1_task126.py --stage cell --dose "$dose" --seed "$s" >> "$LOG" 2>&1
    echo "--- cell $dose|$s rc=$? ---" >> "$LOG"
  done
done

$PY dir126/R1_task126.py --stage formal >> "$LOG" 2>&1
echo "--- formal rc=$? ---" >> "$LOG"
echo "=== $(date +%FT%TZ) 完成 ===" >> "$LOG"
