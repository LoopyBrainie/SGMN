#!/bin/bash
# DIR-50 · 驱动：smoke（不出判据）/ formal（5 条件 × 4000，2 分片并发）
set -u
ROOT=/home/ec2-user/sgmn-loop; PY=/opt/pytorch/bin/python
cd "$ROOT"
if [ "${1:-smoke}" = "smoke" ]; then
  for A in none; do $PY dir50/R1_task50.py --arm $A --n 20 --seed0 32400 --out dir50/out/DIR50_smoke_arm_none.jsonl --tag smoke; done
  for A in hit1 hit2; do for D in 0.0 1.0; do
    $PY dir50/R1_task50.py --arm $A --dose $D --n 20 --seed0 32400 --out dir50/out/DIR50_smoke_arm_${A}_${D}.jsonl --tag smoke
  done; done
  echo "SMOKE DONE"
else
  for A in none; do
    for H in a b; do
      case $H in a) S0=33000;; b) S0=35000;; esac
      nohup $PY dir50/R1_task50.py --arm $A --n 2000 --seed0 $S0 --out dir50/out/DIR50r10_arm_none_${H}.jsonl --tag r10 > dir50/D50_r10_formal_none_${H}.log 2>&1 &
    done
  done
  for A in hit1 hit2; do for D in 0.0 1.0; do
    for H in a b; do
      case $H in a) S0=33000;; b) S0=35000;; esac
      nohup $PY dir50/R1_task50.py --arm $A --dose $D --n 2000 --seed0 $S0 \
        --out dir50/out/DIR50r10_arm_${A}_${D}_${H}.jsonl --tag r10 > dir50/D50_r10_formal_${A}_${D}_${H}.log 2>&1 &
    done
  done; done
  wait
  echo "ALL SHARDS DONE (formal)"
fi
