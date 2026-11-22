#!/bin/bash
cd /home/ec2-user/sgmn-loop
mkdir -p dir67/out
setsid nohup /opt/pytorch/bin/python dir67/R1_task67.py --out dir67/out/DIR67r10_formal.json \
  > dir67/out/formal67.log 2>&1 &
echo "launched pid=$!"
