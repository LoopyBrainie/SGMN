#!/bin/bash
# PROBE60B 正式探针：4 剂量 × 2 格 = 8 次头训练，4 路并发（`PROBE60B.md` rev 1.0 冻结，md5 539a84d9…）
cd /home/ec2-user/sgmn-loop || exit 1
mkdir -p dir60/out/logs
JOBS="1 0.5
2 0.5
1 1.5
2 1.5
1 3
2 3
1 4
2 4"
echo "$JOBS" | xargs -P 4 -n 2 bash -c '/opt/pytorch/bin/python dir60/probe60b.py --cell $0 --s $1 >> dir60/out/logs/b_c$0_s$1.log 2>&1'
echo "=== PROBE60B 全部结束 $(date "+%F %T") ==="
ls dir60/out/PROBE60B_c*_s*.json 2>/dev/null | wc -l
