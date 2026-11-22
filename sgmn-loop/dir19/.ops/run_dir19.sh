#!/bin/bash
# DIR-19 · 启动脚本（服务器端）：先 smoke 验 Stage A，再开正式轮
#
# ★ 纪律 ①：**Stage A 任一 FAIL ⇒ 不得开主循环** ⇒ 本脚本在 smoke 的 Stage A 有 FAIL 时**中止**。
# ★ 纪律 ②：判据已冻结（卡片 rev 1.3）⇒ 脚本**不改**任何判据。
set -u
PY=/opt/pytorch/bin/python
D=~/sgmn-loop/dir19
cd "$D" || exit 9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "[$(date -u +%H:%M:%S)] ===== DIR-19 smoke（40/12/2 种子）====="
$PY -u R1_protect.py --smoke > D19_R1_smoke.log 2>&1
RC=$?
echo "[$(date -u +%H:%M:%S)] smoke 退出码=$RC"
if [ "$RC" != "0" ]; then
  echo "★ smoke 失败 ⇒ 中止（不��正式轮）"; tail -20 D19_R1_smoke.log; exit 3
fi
SA=$(grep -c "^\[PASS\]" D19_R1_smoke.log)
FA=$(grep -c "^\[FAIL\]" D19_R1_smoke.log || true)
echo "[$(date -u +%H:%M:%S)] Stage A/判据：PASS=$SA FAIL=$FA"
grep -E "^\[FAIL\]" D19_R1_smoke.log | head -10
if [ "$FA" != "0" ]; then
  echo "★ Stage A 有 FAIL ⇒ **中止，不开正式轮**（纪律 ①）"; exit 3
fi

echo "[$(date -u +%H:%M:%S)] ===== DIR-19 正式轮（200/60/3 种子 · 11 臂）====="
setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u R1_protect.py \
    > D19_R2_formal.log 2>&1 < /dev/null &
sleep 5
echo "[$(date -u +%H:%M:%S)] 正式轮已在后台启动（pid 见下）"
ps -ef | grep "[R]1_protect.py" | head -2
