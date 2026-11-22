#!/bin/bash
# DIR-113 正式轮驱动 · **一格一进程**（卡片 §4；`DEF-105-02`/`DEF-112-02` 的协议纪律）
#
# 每格单独起一个 python 进程，各自落盘 out/iso/a{ARM}_s{SEED}.json
#   存在即跳过 ⇒ 可断点续跑
# 全部跑完后再 `--stage merge` 合成 stage_formal.json。
#
# 用法：  bash run_formal113.sh            # 48 格
set -u
cd "$(dirname "$0")" || exit 1
PY=/opt/pytorch/bin/python
OUTD=out/iso
mkdir -p "$OUTD"

G="${G:-0.001}"
SEEDS="${SEEDS:-20270401 20270415 20270429 20270513 20270527 20270610 20270624 20270708 20270722 20270805 20270819 20270902}"
ARMS="${ARMS:-T-NONE T-Q T-O T-BOTH}"

total=0
for a in $ARMS; do for s in $SEEDS; do total=$((total+1)); done; done

n=0; fail=0
for a in $ARMS; do
  for s in $SEEDS; do
    n=$((n+1))
    tag="a${a}_s${s}"
    if [ -f "$OUTD/$tag.json" ]; then
      echo "[$n/$total] $tag 已有，跳过"
      continue
    fi
    echo "=============================================================="
    echo "[$n/$total] $tag  $(date -u +%H:%M:%S)"
    "$PY" R1_task113.py --stage cell --arm "$a" --seed "$s" \
          --out "$OUTD/$tag.json" > "$OUTD/$tag.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "  ✗ rc=$rc（见 $OUTD/$tag.log）"; tail -6 "$OUTD/$tag.log"
      fail=$((fail+1)); continue
    fi
    grep -E '^  \[' "$OUTD/$tag.log" | tail -1
  done
done

echo "=============================================================="
echo "单格结束：计划 $total 格，失败 $fail"
if [ $fail -eq 0 ]; then
  echo "=== 合并 ==="
  "$PY" R1_task113.py --stage merge
fi
exit $([ $fail -eq 0 ] && echo 0 || echo 1)
