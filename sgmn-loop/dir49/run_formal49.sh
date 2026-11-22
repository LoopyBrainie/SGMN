#!/bin/bash
# DIR-49 · 驱动：smoke（不出判据）/ formal（4 臂 × 1000，各 2 分片并发）
set -u
ROOT=/home/ec2-user/sgmn-loop; PY=/opt/pytorch/bin/python
cd "$ROOT"
if [ "${1:-smoke}" = "smoke" ]; then
  for A in ortho d0.0 d0.5 d1.0; do
    $PY dir49/R1_task49.py --arm $A --n 20 --seed0 30400 --out dir49/out/DIR49_smoke_arm_${A}.jsonl --tag smoke
  done
  echo "SMOKE DONE"
else
  for A in ortho d0.0 d0.5 d1.0; do
    for H in a b; do
      case $H in a) S0=29400;; b) S0=29900;; esac
      nohup $PY dir49/R1_task49.py --arm $A --n 500 --seed0 $S0 \
        --out dir49/out/DIR49r10_arm_${A}_${H}.jsonl --tag r10 > dir49/D49_r10_formal_${A}_${H}.log 2>&1 &
    done
  done
  wait
  echo "ALL SHARDS DONE (formal)"
fi
