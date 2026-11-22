#!/bin/bash
# DIR-41 · 正式轮驱动：18 臂（6 格 × 3 臂），**4 路并发**（trainer 4 核 / 15 GB，OMP_NUM_THREADS=1）
# ★ 不用探针的 9 路并发（Advisor 建议：4 路是已验证的跑法）
# ★ 每臂只写读数（DIR41r10_cell_{cell}_{arm}.json）；**裁决只在 `merge` 里**（冻结判定表）
set -u
cd /home/ec2-user/sgmn-loop
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/opt/pytorch/bin/python
mkdir -p dir41/out
JOBS=()
for cell in SH1 SH2 SH3 SH4 SH5 PI; do
  for arm in oracle rand learned; do
    JOBS+=("$cell $arm")
  done
done
echo "START $(date -Is)  jobs=${#JOBS[@]}  concurrency=4"
i=0
for j in "${JOBS[@]}"; do
  set -- $j
  cell=$1; arm=$2
  $PY dir41/R1_task41.py formal --cell "$cell" --arm "$arm" \
      > "dir41/D41_formal_${cell}_${arm}.log" 2>&1 &
  i=$((i+1))
  if [ $((i % 4)) -eq 0 ]; then
    echo "wave $((i/4)) done $(date -Is)"; wait
  fi
done
wait
echo "ALL DONE $(date -Is)"
