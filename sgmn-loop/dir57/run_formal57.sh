#!/bin/bash
# DIR-57 rev 1.0 正式轮（4 臂 × 96 种子，种子段 59000..59095）
set -euo pipefail
cd /home/ec2-user/sgmn-loop
PY=/opt/pytorch/bin/python
mkdir -p dir57/out dir57/logs
for arm in m1 m2 m4 m4_int8; do
  echo "[$(date +%H:%M:%S)] $arm"
  $PY dir57/R1_task57.py --arm "$arm" --n 96 --seed0 59000 \
      --out "dir57/out/DIR57r10_${arm}.json" 2>&1 | grep -v "Warning\|pynvml\|Consider using" | tee "dir57/logs/DIR57r10_${arm}.log"
done
echo "[$(date +%H:%M:%S)] adjudicate"
$PY dir57/adjudicate57.py 2>&1 | grep -v "Warning\|pynvml" | tee dir57/logs/DIR57r10_adjudicate.log
