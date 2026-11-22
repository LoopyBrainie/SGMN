#!/bin/bash
cd /home/ec2-user/sgmn-loop
for m in keys target; do
  for i in 0.01 0.03 0.1 0.3; do
    OMP_NUM_THREADS=1 /opt/pytorch/bin/python dir81/cell81.py $m $i > dir81/logs/cell_${m}_${i}.log 2>&1 &
  done
done
wait
echo "ALL CELLS DONE"
OMP_NUM_THREADS=1 /opt/pytorch/bin/python dir81/collect81.py > dir81/logs/collect81.log 2>&1
echo "COLLECT DONE"
