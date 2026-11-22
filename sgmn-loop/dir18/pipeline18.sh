#!/bin/bash
# DIR-18 · 服务器侧流水线（一次连接内完成：probe18h → smoke#2 → 正式轮）
# 由本地 tar 管道传入后立即执行。
set -u
cd ~/sgmn-loop/dir18 || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/pytorch/bin/python

echo "===== [DIR-18 流水线开始] $(date -u +%H:%M:%S) ====="
md5sum R2_write.py sgmn_module_hebb.py probe18h.py 2>/dev/null || true
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# ---------- ① probe18h：协议等价性（clean vs dirty17 × 训练篇数 × 评测篇数）----------
echo "----- [1/3] probe18h（协议等价性）-----"
timeout 3000 $PY -u probe18h.py > D18_probe18h.log 2>&1
echo "probe18h rc=$?"
grep -E "^\[probe18h\]" D18_probe18h.log | tail -12

# ---------- ② smoke#2：新臂 + 新判据的 Stage A 全绿校验 ----------
echo "----- [2/3] smoke#2（R2_write.py --smoke）-----"
timeout 3000 $PY -u R2_write.py --smoke > D18_R2_smoke2.log 2>&1
echo "smoke2 rc=$?"
grep -E "^\[(PASS|FAIL)\]" D18_R2_smoke2.log | sed -n '1,30p'
echo "--- 判据块 ---"
sed -n '/\[汇总与判据\]/,$p' D18_R2_smoke2.log | grep -E "^\s*\[(PASS|FAIL)\]" | head -30

# Stage A 全绿才继续
if grep -q "^\[FAIL\]" D18_R2_smoke2.log; then
  echo "!!! Stage A 有 FAIL ⇒ 正式轮不启动"
  exit 2
fi

# ---------- ③ 正式轮 ----------
echo "----- [3/3] 正式轮（R2_write.py）-----"
setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u R2_write.py > D18_R1_formal.log 2>&1 < /dev/null &
sleep 5
pgrep -af "R2_write.py" | head -3
echo "===== [DIR-18 流水线结束] $(date -u +%H:%M:%S) ====="
