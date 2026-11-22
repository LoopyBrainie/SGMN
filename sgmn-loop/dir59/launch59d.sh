#!/bin/bash
# DIR-59 · 59D 探针启动器（9 格臂，4 路并发；跨 cycle 后台跑）
set -u
cd /home/ec2-user/sgmn-loop || exit 1
mkdir -p dir59/out/logs
for c in C1 C2 C3; do
  for a in OWN FOREIGN NONE; do
    echo "$c $a"
  done
done | xargs -P 4 -n 2 bash -c '
  c=$0; a=$1
  log="dir59/out/logs/${c}_${a}.log"
  [ -s "dir59/out/PROBE59D_${c}_${a}.json" ] && { echo "skip $c $a (已存在)"; exit 0; }
  /opt/pytorch/bin/python dir59/probe59d.py --cell "$c" --arm "$a" > "$log" 2>&1
  echo "done $c $a rc=$? $(date +%H:%M:%S)"
'
echo "===== 59D 全部格臂结束 $(date +%F' '%T) ====="
