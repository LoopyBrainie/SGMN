#!/bin/bash
# DIR-59 · 59D rev3 启动器（6 次：3 格 × {AB, BA}，4 路并发）
set -u
cd /home/ec2-user/sgmn-loop || exit 1
mkdir -p dir59/out/logs
for job in "C1 AB" "C1 BA" "C2 AB" "C2 BA" "C3 AB" "C3 BA"; do echo "$job"; done | xargs -P 4 -n 2 bash -c '
  c=$0; a=$1
  log="dir59/out/logs/v3_${c}_${a}.log"
  [ -s "dir59/out/PROBE59Dv3_${c}_${a}.json" ] && { echo "skip $c $a"; exit 0; }
  /opt/pytorch/bin/python dir59/probe59d_v3.py --cell "$c" --arm "$a" > "$log" 2>&1
  echo "done $c $a rc=$? $(date +%H:%M:%S)"
'
echo "===== 59Dv3 结束 ====="
