#!/bin/bash
# DIR-59 · 59D rev2 启动器（7 次训练：3×OWN + 3×FOREIGN + 1×NONE，4 路并发）
set -u
cd /home/ec2-user/sgmn-loop || exit 1
mkdir -p dir59/out/logs
for job in "C1 OWN" "C1 FOREIGN" "C2 OWN" "C2 FOREIGN" "C3 OWN" "C3 FOREIGN" "C1 NONE"; do
  echo "$job"
done | xargs -P 4 -n 2 bash -c '
  c=$0; a=$1
  log="dir59/out/logs/v2_${c}_${a}.log"
  [ -s "dir59/out/PROBE59Dv2_${c}_${a}.json" ] && { echo "skip $c $a"; exit 0; }
  /opt/pytorch/bin/python dir59/probe59d_v2.py --cell "$c" --arm "$a" > "$log" 2>&1
  echo "done $c $a rc=$? $(date +%H:%M:%S)"
'
echo "===== 59Dv2 结束 $(date +%F" "%T) ====="
