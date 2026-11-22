#!/bin/bash
# DIR-46 · 正式轮驱动（4 分片并发：2 模式 × 2 半段；种子段 `M.SEED0+24000..24399`）
# 用法：run_formal46.sh smoke|formal
set -u
PY=/opt/pytorch/bin/python
cd /home/ec2-user/sgmn-loop
MODE_SET="period2 perstep"
case "${1:-formal}" in
  smoke)
    N=8; S0=23990; TAG=smoke            # ★ smoke 种子段 23990..23997（与正式 24000+、既往各段均不相交）
    for m in $MODE_SET; do
      $PY dir46/R1_task46.py --mode $m --n $N --seed0 $S0 --tag $TAG \
          --out dir46/out/DIR46_smoke_arm_$m.jsonl > dir46/D46_smoke_$m.log 2>&1 &
    done ;;
  formal)
    N=200; TAG=r10                      # 400 = 2 分片 × 200
    for m in $MODE_SET; do
      $PY dir46/R1_task46.py --mode $m --n $N --seed0 $((20261111+24000))      --tag $TAG \
          --out dir46/out/DIR46r10_arm_${m}_a.jsonl  > dir46/D46_formal_${m}_a.log 2>&1 &
      $PY dir46/R1_task46.py --mode $m --n $N --seed0 $((20261111+24200))      --tag $TAG \
          --out dir46/out/DIR46r10_arm_${m}_b.jsonl  > dir46/D46_formal_${m}_b.log 2>&1 &
    done ;;
esac
wait
echo "ALL SHARDS DONE ($1)"; ls -l dir46/out/ | tail -8
