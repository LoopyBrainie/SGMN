#!/bin/bash
cd /home/ec2-user/sgmn-loop
mkdir -p dir67/out/cells dir67/out/logs
pkill -f "[R]1_task67.py"; sleep 1
W1="1e-01:20261118,1e-01:20261119,1e-01:20261120"
W2="1e-01:20261121,1e-01:20261122,3e-01:20261118"
W3="3e-01:20261119,3e-01:20261120"
W4="3e-01:20261121,3e-01:20261122"
i=1
for W in "$W1" "$W2" "$W3" "$W4"; do
  setsid nohup /opt/pytorch/bin/python dir67/R1_cells67.py --cells "$W" > dir67/out/logs/cells_w$i.log 2>&1 &
  echo "w$i pid=$! cells=$W"; i=$((i+1))
done
