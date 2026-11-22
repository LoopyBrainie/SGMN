#!/bin/bash
# DIR-47 · 驱动：smoke（不出判据）/ formal（2 臂 × 500，分片并发）
set -u
ROOT=/home/ec2-user/sgmn-loop
PY=/opt/pytorch/bin/python
MODE=${1:-smoke}
cd "$ROOT"
if [ "$MODE" = "smoke" ]; then
  for A in nor rep_on; do
    $PY dir47/R1_task47.py --arm $A --n 20 --seed0 25000 --out dir47/out/DIR47_smoke_arm_$A.jsonl --tag smoke
  done
  echo "SMOKE DONE"
else
  for A in nor rep_on; do
    for H in a b; do
      S0=$([ $H = a ] && echo 25200 || echo 25450)
      nohup $PY dir47/R1_task47.py --arm $A --n 250 --seed0 $S0 \
        --out dir47/out/DIR47r10_arm_${A}_${H}.jsonl --tag r10 > dir47/D47_formal_${A}_${H}.log 2>&1 &
    done
  done
  wait
  echo "ALL SHARDS DONE (formal)"
fi
