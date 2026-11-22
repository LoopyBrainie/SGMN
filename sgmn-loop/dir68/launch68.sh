#!/bin/bash
cd /home/ec2-user/sgmn-loop
mkdir -p dir68/out/cells dir68/out/logs
W1="PAR:20261118,PAR:20261119"; W2="PAR:20261120,PAR:20261121"; W3="PAR:20261122"; W4="ORTH:20261118"
i=1
for W in "$W1" "$W2" "$W3" "$W4"; do
  setsid nohup /opt/pytorch/bin/python dir68/R1_cells68.py --cells "$W" > dir68/out/logs/w$i.log 2>&1 &
  echo "w$i pid=$! cells=$W"; i=$((i+1))
done
