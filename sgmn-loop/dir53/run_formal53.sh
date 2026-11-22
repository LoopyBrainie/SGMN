#!/bin/bash
# DIR-53 正式轮：15 臂（5 格 × FIX/OFF/ORACLE），4 路并发。
# ★ 顺序：先 5 个 OFF（最慢：离线学 + 训头），再 FIX，再 ORACLE ⇒ 尾部不拖长。
# ★ 用法：setsid nohup bash dir53/run_formal53.sh > dir53/logs/formal53_driver.log 2>&1 &
set -u
cd /home/ec2-user/sgmn-loop || exit 1
mkdir -p dir53/out dir53/logs
PY=/opt/pytorch/bin/python
{
  for c in P1 P2 P3 P4 P5; do echo "$c OFF"; done
  for c in P1 P2 P3 P4 P5; do echo "$c FIX"; done
  for c in P1 P2 P3 P4 P5; do echo "$c ORACLE"; done
} | xargs -P 4 -n 2 sh -c "$PY dir53/R1_task53.py formal \"\$0\" \"\$1\" > dir53/logs/D53_formal_\$0_\$1.log 2>&1; echo \"[\$(date +%H:%M:%S)] \$0/\$1 done rc=\$?\""
echo "[$(date +%H:%M:%S)] ★ 全部 15 臂结束"
$PY dir53/R1_task53.py merge
echo "[$(date +%H:%M:%S)] ★ merge 完成"