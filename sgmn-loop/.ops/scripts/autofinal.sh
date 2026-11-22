#!/bin/bash
# DIR-18 · 自动收尾（重连后自动执行）：拉回产物 → 启动正式轮 → 打包 → 校验
set -u
BASE=/workspace/sgmn-loop/revolve/DIR-18
ROOT=/workspace/sgmn-loop
S=$BASE/.ops/cm
KEY=/workspace/.ssh/SGMN.pem
HOST=ec2-user@ec2-3-106-240-137.ap-southeast-2.compute.amazonaws.com
OPTS="-S $S -o ControlPath=$S -i $KEY"
LOG=$BASE/.ops/package.log

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "===== DIR-18 自动收尾开始 ====="
mkdir -p "$BASE/rounds" "$BASE/probes"

# ---------- ① 拉回服务器产物 ----------
for f in D18_probe18b.log D18_probe18c.log D18_probe18d.log D18_probe18e.log \
         D18_probe18f.log D18_probe18g.log D18_probe18h.log \
         D18_R2_smoke2.log D18_R2_smoke2_v1_adrsNameError_FAILED.log \
         D18_R2_smoke2_v2_gatekey_FAILED.log D18_R1_formal.log \
         D18_R1_formal_v1_D48protocol_FAILED.log; do
  timeout 120 scp $OPTS "$HOST:~/sgmn-loop/dir18/$f" "$BASE/" 2>/dev/null && say "got $f"
done
for f in R28_results.json R28_smoke_results.json probe18h_out.json probe18c_out.json \
         probe18e_out.json probe18f_out.json probe18g_out.json; do
  timeout 120 scp $OPTS "$HOST:~/sgmn-loop/dir18/$f" "$BASE/" 2>/dev/null && say "got $f"
done
for f in probe18b.py probe18c.py probe18d.py probe18e.py probe18f.py probe18g.py probe18h.py; do
  timeout 120 scp $OPTS "$HOST:~/sgmn-loop/dir18/$f" "$BASE/probes/" 2>/dev/null && say "got probes/$f"
done
timeout 120 scp $OPTS "$HOST:~/sgmn-loop/dir18/R2_write.py" "$BASE/rounds/R2_write.server.py" 2>/dev/null
timeout 120 scp $OPTS "$HOST:~/sgmn-loop/dir18/sgmn_module_hebb.py" "$BASE/sgmn_module_hebb.server.py" 2>/dev/null

# ---------- ② 服务器/本地 md5 逐位对照 ----------
say "--- 服务器 md5 ---"
timeout 60 ssh $OPTS "$HOST" 'cd ~/sgmn-loop/dir18 && md5sum R2_write.py sgmn_module_hebb.py' 2>/dev/null | tee -a "$LOG"
say "--- 本地 md5 ---"
md5sum "$BASE/rounds/R2_write.py" "$BASE/sgmn_module_hebb.py" | tee -a "$LOG"

# ---------- ③ smoke#2 是否已通过（Stage A 闸） ----------
say "--- smoke#2 Stage A 闸 ---"
timeout 60 ssh $OPTS "$HOST" 'grep -A2 "Stage A 闸" ~/sgmn-loop/dir18/D18_R2_smoke2.log | head -4' 2>/dev/null | tee -a "$LOG"

# ---------- ④ 若正式轮未在跑，则启动 ----------
if ! timeout 40 ssh $OPTS "$HOST" 'pgrep -f "R2_write.py$" >/dev/null && echo RUNNING' 2>/dev/null | grep -q RUNNING; then
  say "--- 启动正式轮 ---"
  timeout 60 ssh $OPTS "$HOST" 'cd ~/sgmn-loop/dir18 && setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /opt/pytorch/bin/python -u R2_write.py > D18_R1_formal.log 2>&1 < /dev/null & sleep 4; pgrep -af R2_write | head -2' 2>/dev/null
else
  say "--- 正式轮已在运行 ---"
fi
say "===== 自动收尾结束 ====="
