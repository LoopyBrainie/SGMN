#!/bin/bash
# DIR-120 正式跑（纯再分析；不改 dir119/out）
set -u
ROOT=${SGMN_ROOT:-/home/ec2-user/sgmn-loop}
cd "$ROOT/dir120" || exit 1
PY=/opt/pytorch/bin/python
echo "===== DIR-120 Stage A ====="
$PY R1_task120.py --stage a 2>&1 | grep -vE "pynvml|^Loading weights"
rc=$?
echo "===== DIR-120 Formal ====="
$PY R1_task120.py --stage formal 2>&1 | grep -vE "pynvml|^Loading weights"
echo "===== 结束 ====="
