#!/bin/bash
# DIR-59 · 正式轮启动器（5 格 × 4 臂 = 20 次训练；4 路并发；每格先跑满四臂）
set -u
cd /home/ec2-user/sgmn-loop || exit 1
mkdir -p dir59/out/logs
for j in 1 2 3 4 5; do for a in AA AB BA BB; do echo "$j $a"; done; done | xargs -P 4 -n 2 bash -c '
  j=$0; a=$1
  log="dir59/out/logs/f59_c${j}_${a}.log"
  [ -s "dir59/out/DIR59r10_c${j}_${a}.json" ] && { echo "skip c$j $a"; exit 0; }
  /opt/pytorch/bin/python dir59/R1_formal59.py --cell "$j" --arm "$a" > "$log" 2>&1
  echo "done c$j $a rc=$? $(date +%H:%M:%S)"
'
echo "===== DIR-59 正式轮结束 $(date +%F" "%T) ====="
