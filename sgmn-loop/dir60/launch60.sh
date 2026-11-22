#!/bin/bash
# PROBE60 正式探针：5 臂 × 2 格 = 10 次头训练，4 路并发（rev 1.3 冻结）
cd /home/ec2-user/sgmn-loop || exit 1
mkdir -p dir60/out/logs
JOBS="1 NONE
1 RAND
1 ORTH
1 RANK2
1 SCALE2
2 NONE
2 RAND
2 ORTH
2 RANK2
2 SCALE2"
echo "$JOBS" | xargs -P 4 -n 2 bash -c '/opt/pytorch/bin/python dir60/probe60a.py --cell $0 --arm $1 >> dir60/out/logs/c$0_$1.log 2>&1'
echo "=== PROBE60 全部结束 $(date "+%F %T") ==="
ls dir60/out/PROBE60_c*_*.json 2>/dev/null | wc -l
