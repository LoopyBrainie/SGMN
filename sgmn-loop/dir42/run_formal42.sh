#!/bin/bash
# DIR-42 · 正式轮驱动：3 臂（oracle / rand / learned）× 4 分片 = 12 个作业
# ★ trainer 4 核 ⇒ **4 路并发**（每波 4 个分片，共 3 波）
# ★ 每片 100 个实例 × 250 步 ⇒ 单片 ≈ 25 000 条轨迹 ⇒ 预计每波 ≈ 25–35 min
# ★ 判据与阈值**不在**本脚本里（只跑 ⇒ 裁决由 `merge` 独立完成）
set -u
cd /home/ec2-user/sgmn-loop
mkdir -p dir42/out

run() {  # arm shard
  local arm=$1 shard=$2
  OMP_NUM_THREADS=1 /opt/pytorch/bin/python dir42/R1_task42.py formal \
      --arm "$arm" --shard "$shard" --nshards 4 \
      > "dir42/D42_formal_${arm}_shard${shard}.log" 2>&1
  echo "wave done: ${arm} shard${shard} rc=$?"
}

echo "=== wave 1/3: rand ==="
for s in 0 1 2 3; do run rand "$s" & done; wait
echo "=== wave 2/3: learned ==="
for s in 0 1 2 3; do run learned "$s" & done; wait
echo "=== wave 3/3: oracle ==="
for s in 0 1 2 3; do run oracle "$s" & done; wait
echo "=== formal 全部完成 ==="
ls -la dir42/out/DIR42r10_arm_*.json
