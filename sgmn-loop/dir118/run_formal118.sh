#!/bin/bash
# DIR-118 驱动：Stage A（含 36 格）→ 正式轮（继承）→ 判据。逐格落盘、存在即跳过 ⇒ 可断点续跑。
cd "$(dirname "$0")/.." || exit 1
LOG=dir118/out/driver.log
echo "=== $(date -u +%FT%TZ) DIR-118 启动 PID $$ ===" >> "$LOG"
/opt/pytorch/bin/python dir118/R1_task118.py --stage A >> "$LOG" 2>&1
echo "--- stage A rc=$? ---" >> "$LOG"
/opt/pytorch/bin/python dir118/R1_task118.py --stage formal >> "$LOG" 2>&1
echo "--- stage formal rc=$? ---" >> "$LOG"
echo "=== $(date -u +%FT%TZ) 完成 ===" >> "$LOG"
