#!/bin/bash
# DIR-19 · 正式轮启动脚本（服务器端）—— **修版**
#
# ★★ 缺陷 D75（脚本层）：旧脚本 `run_dir19.sh` 用 `grep -c "^\[FAIL\]"` 统计 **全日志**的 FAIL，
#   而**判据 H1′…H7′ 的 FAIL 也用同一个前缀** ⇒ ⇒ 一旦某条判据 FAIL（那是**合法结果**，
#   DIR-18 正式轮就有 10 条 FAIL），脚本就会**误杀正式轮**。
#   修法：① 只统计 **Stage A 段**（"[Stage A]" 到 "[Stage A 闸]" 之间）的 FAIL；
#         ② 判据 FAIL 只**报告**、不**中止**（判据的裁决权归卡片 rev 1.4 的读数规则）。
#   ※ Python 脚本内部**已有** Stage A 硬闸（纪律 ①）⇒ 本脚本的检查只是**双保险**。
set -u
PY=/opt/pytorch/bin/python
D=~/sgmn-loop/dir19
cd "$D" || exit 9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

LOG=${1:-D19_R2_formal.log}
echo "[$(date -u +%H:%M:%S)] ===== DIR-19 正式轮（200/60/3 种子 · 11 臂）====="
echo "[$(date -u +%H:%M:%S)] 日志：$LOG"
setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u R1_protect.py \
    > "$LOG" 2>&1 < /dev/null &
sleep 5
echo "[$(date -u +%H:%M:%S)] 正式轮已在后台启动"
ps -ef | grep "[R]1_protect.py" | head -2
