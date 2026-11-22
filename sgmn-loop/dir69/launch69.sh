#!/bin/bash
cd /home/ec2-user/sgmn-loop
mkdir -p dir69/out/cells dir69/out/logs
i=1
for W in "1:20261118,1:20261119,1:20261120" "3:20261118,3:20261119,3:20261120"          "10:20261118,10:20261119,10:20261120" "20:20261118,20:20261119,20:20261120"; do
  setsid nohup /opt/pytorch/bin/python dir69/R1_cells69.py --cells "$W" > dir69/out/logs/w$i.log 2>&1 &
  echo "w$i pid=$! cells=$W"; i=$((i+1))
done
