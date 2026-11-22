#!/usr/bin/bash
# DIR-18 · CPU 降级轮（**降级正式轮，不得当作正式轮结果**）
#
# ★ 触发条件：GPU 服务器 SSH 长期不可达时的**降级替代**。
# ★ 降级内容（全部写入卡片 rev 1.4 的「降级声明」段，不追认）：
#   - 规模：n_train 200→24、n_eval 60→10、种子 3→1（判据的统计功效随之下降）
#   - 精度：fp32 → bf16（CPU 内存 3 GB 限制；fp32 需 ~2.3 GB 常驻 + 激活 ⇒ OOM）
#   - 线程：2 CPU（服务器为 T4 GPU）
# ★ ⇒ 本轮**只能**判定「效应方向是否与冻结判据一致」，**不能**替代正式轮的阈值判定。
set -u
BASE=/workspace/sgmn-loop/revolve/DIR-18
LOG=$BASE/.ops/cpu_round.log
cd "$BASE/rounds"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "[$(date -u +%H:%M:%S)] CPU 降级轮启动" >> "$LOG"
/opt/pytorch/bin/python --version 2>/dev/null || true
python3 -u "$BASE/.ops/cpu_round.py" >> "$LOG" 2>&1
echo "[$(date -u +%H:%M:%S)] CPU 降级轮结束 rc=$?" >> "$LOG"
