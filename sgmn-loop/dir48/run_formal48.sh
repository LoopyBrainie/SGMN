#!/bin/bash
# DIR-48 · 驱动：smoke（不出判据）/ formal（3 臂 × 1000，各 3 分片并发）
set -u
ROOT=/home/ec2-user/sgmn-loop; PY=/opt/pytorch/bin/python
cd "$ROOT"
if [ "${1:-smoke}" = "smoke" ]; then
  for A in quiet decay busy; do
    $PY dir48/R1_task48.py --arm $A --n 20 --seed0 28800 --out dir48/out/DIR48_smoke_arm_${A}.jsonl --tag smoke
  done
  echo "SMOKE DONE"
else
  for A in quiet decay busy; do
    for H in a b c; do
      # ★ `DEF-48-7`：分片起点必须与 `N_INST/段` 对齐（27400+334=27734、+334=28068），否则段内有缺口
      case $H in a) S0=27400;; b) S0=27734;; c) S0=28068;; esac
      N=334; [ $H = c ] && N=332
      nohup $PY dir48/R1_task48.py --arm $A --n $N --seed0 $S0 \
        --out dir48/out/DIR48r11_arm_${A}_${H}.jsonl --tag r10 > dir48/D48_r11_formal_${A}_${H}.log 2>&1 &
    done
  done
  wait
  echo "ALL SHARDS DONE (formal)"
fi
