#!/bin/bash
# DIR-43 · **全流程**（preflight ⇒ Stage A ⇒ smoke ⇒ 正式轮 ⇒ merge）
# ★ Stage A 不过 ⇒ **拦下正式轮**（纪律：硬闸）
# ★ 判据只在 `R1_task43.py merge` 里（判定逻辑只有一份：`decide()`）
set -u
cd /home/ec2-user/sgmn-loop
mkdir -p dir43/out

echo "=== preflight（离线自检）==="
for t in dir43/verify_merge43.py dir43/verify_freeze43.py dir43/verify_card43.py; do
  OMP_NUM_THREADS=1 /opt/pytorch/bin/python "$t" > "dir43/D43_preflight_$(basename "$t" .py).log" 2>&1
  rc=$?
  tail -3 "dir43/D43_preflight_$(basename "$t" .py).log"
  if [ "$rc" != "0" ]; then
    echo "★ preflight 未通过（$t）⇒ **不启动正式轮**"; exit 1
  fi
done

echo "=== Stage A ==="
OMP_NUM_THREADS=1 /opt/pytorch/bin/python dir43/R1_task43.py stageA > dir43/D43_stageA_run1.log 2>&1
rcA=$?
tail -40 dir43/D43_stageA_run1.log
if [ "$rcA" != "0" ]; then echo "★ Stage A 未通过 ⇒ 正式轮被拦下"; exit 1; fi

echo "=== smoke ==="
OMP_NUM_THREADS=1 /opt/pytorch/bin/python dir43/R1_task43.py smoke > dir43/D43_smoke_run1.log 2>&1
rcS=$?
tail -20 dir43/D43_smoke_run1.log
if [ "$rcS" != "0" ]; then echo "★ smoke 失败"; exit 1; fi

echo "=== formal（3 臂 × 4 分片，4 路并发）==="
bash dir43/run_formal43.sh

echo "=== merge（独立裁决）==="
OMP_NUM_THREADS=1 /opt/pytorch/bin/python dir43/R1_task43.py merge > dir43/D43_merge_run1.log 2>&1
rcM=$?
cat dir43/D43_merge_run1.log
echo "merge rc=$rcM"
