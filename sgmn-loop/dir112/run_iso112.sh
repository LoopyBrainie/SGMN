#!/bin/bash
# DIR-112 前置检查驱动 · **一格一进程**（`DEF-105-02` 的修法）
#
# 为什么必须这样：同一进程内连跑多格时，cuBLAS workspace 的分配受前序格影响 ⇒
# 归约顺序变化 ⇒ Adam 轨迹混沌放大 ⇒ 同一配置在不同「前序历史」下读数不同
# （实测：seed 7777 lin@600 隔离 0.8750，在 6 格之后 0.7812；基座 sha1 全程不变）。
#
# 本驱动对每个 (γ, arm, seed) 单独起一个 python 进程，各自落盘为
#   out/iso/g{G}_a{ARM}_s{SEED}.json      存在即跳过 ⇒ 可断点续跑
#
# 用法：
#   bash run_iso112.sh              # 跑全部格（8 格 ≈ 8 min）
#   GAMMAS="0.001" bash run_iso112.sh
set -u
cd "$(dirname "$0")" || exit 1
PY=/opt/pytorch/bin/python
OUTD=out/iso
mkdir -p "$OUTD"

GAMMAS="${GAMMAS:-0.001 0.01}"
SEEDS="${SEEDS:-20261118}"
ARMS="${ARMS:-T-NONE T-Q T-O T-BOTH}"

total=0
for g in $GAMMAS; do for a in $ARMS; do for s in $SEEDS; do total=$((total+1)); done; done; done

n=0; fail=0
for g in $GAMMAS; do
  for a in $ARMS; do
    for s in $SEEDS; do
      n=$((n+1))
      tag="g${g}_a${a}_s${s}"
      if [ -f "$OUTD/$tag.json" ]; then
        echo "[$n/$total] $tag 已有，跳过"
        continue
      fi
      echo "=============================================================="
      echo "[$n/$total] $tag  $(date -u +%H:%M:%S)"
      "$PY" probe_isolation.py --gamma "$g" --arm "$a" --seed "$s" \
            --out "$OUTD/$tag.json" > "$OUTD/$tag.log" 2>&1
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "  ✗ rc=$rc（见 $OUTD/$tag.log）"
        tail -5 "$OUTD/$tag.log"
        fail=$((fail+1))
        continue
      fi
      grep -E '^  \[' "$OUTD/$tag.log" | tail -1
    done
  done
done

echo "=============================================================="
echo "结束：计划 $total 格，失败 $fail"
exit $([ $fail -eq 0 ] && echo 0 || echo 1)