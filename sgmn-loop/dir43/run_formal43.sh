#!/bin/bash
# DIR-43 · 正式轮驱动：3 臂（learned / rand / direct）× 4 分片 = 12 个作业
# ★ trainer 4 核 ⇒ **4 路并发**（每波 4 个分片，共 3 波）
# ★ learned 臂：400 实例 × 250 步（与 DIR-42 的 learned 臂同规模）⇒ 单片 100 实例 ≈ 35–45 min
# ★ rand 臂：100 实例（诊断对照）⇒ 单片 25 实例 ≈ 10 min
# ★ direct 臂：400 实例（纯对齐优化、无轨迹）⇒ 单片 100 实例 ≈ 3–6 min
# ★ 判据与阈值**不在**本脚本里（只跑 ⇒ 裁决由 `merge` 独立完成）
set -u
cd /home/ec2-user/sgmn-loop
mkdir -p dir43/out

JOBS=${JOBS:-4}          # ★ 并发度（默认 4 = trainer 满载；若同时有别的正式轮在跑 ⇒ JOBS=2）
NSHARDS=4               # ★ 分片数**恒为 4**（与 `merge` 的读法一致；并发度只改调度）

run() {  # arm shard
  local arm=$1 shard=$2
  OMP_NUM_THREADS=1 /opt/pytorch/bin/python dir43/R1_task43.py formal \
      --arm "$arm" --shard "$shard" --nshards "$NSHARDS" \
      > "dir43/D43_formal_${arm}_shard${shard}.log" 2>&1
  echo "wave done: ${arm} shard${shard} rc=$?"
}

wave() {  # arm
  local arm=$1
  local s=0
  while [ "$s" -lt "$NSHARDS" ]; do
    local k=0
    while [ "$k" -lt "$JOBS" ] && [ "$s" -lt "$NSHARDS" ]; do
      run "$arm" "$s" &
      s=$((s+1)); k=$((k+1))
    done
    wait
  done
}

echo "=== wave 1/3: learned（判据臂，n=400）==="
wave learned
echo "=== wave 2/3: rand（诊断对照臂，n=100）==="
wave rand
echo "=== wave 3/3: direct（判据-非空臂，n=400）==="
wave direct
echo "=== formal 全部完成 ==="
ls -la dir43/out/DIR43r10_arm_*.json
