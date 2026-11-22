#!/bin/bash
# DIR-18 · 端到端流水线（重连后自动跑完）：拉产物 → 启动正式轮 → 轮询完成 → 拉结果 → 打包
set -u
BASE=/workspace/sgmn-loop/revolve/DIR-18
ROOT=/workspace/sgmn-loop
S=$BASE/.ops/cm
KEY=/workspace/.ssh/SGMN.pem
HOST=ec2-user@ec2-3-106-240-137.ap-southeast-2.compute.amazonaws.com
OPTS="-S $S -o ControlPath=$S -i $KEY"
LOG=$BASE/.ops/pipeline.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "===== DIR-18 端到端流水线启动 ====="

# ---------- ① 拉回服务器已有产物 ----------
say "--- 拉回产物 ---"
mkdir -p "$BASE/rounds" "$BASE/probes" "$BASE/.ops/live"
for f in D18_probe18b.log D18_probe18c.log D18_probe18d.log D18_probe18e.log \
         D18_probe18f.log D18_probe18g.log D18_probe18h.log \
         D18_R2_smoke2.log D18_R2_smoke2_v1_adrsNameError_FAILED.log \
         D18_R2_smoke2_v2_gatekey_FAILED.log D18_R1_formal.log \
         D18_R1_formal_v1_D48protocol_FAILED.log; do
  timeout 90 scp $OPTS "$HOST:~/sgmn-loop/dir18/$f" "$BASE/" 2>/dev/null && say "got $f"
done
for f in R28_results.json R28_smoke_results.json probe18h_out.json probe18c_out.json \
         probe18e_out.json probe18f_out.json probe18g_out.json; do
  timeout 90 scp $OPTS "$HOST:~/sgmn-loop/dir18/$f" "$BASE/" 2>/dev/null && say "got $f"
done
for f in probe18b.py probe18c.py probe18d.py probe18e.py probe18f.py probe18g.py probe18h.py; do
  timeout 90 scp $OPTS "$HOST:~/sgmn-loop/dir18/$f" "$BASE/probes/" 2>/dev/null && say "got probes/$f"
done
timeout 90 scp $OPTS "$HOST:~/sgmn-loop/dir18/R2_write.py" "$BASE/rounds/R2_write.server.py" 2>/dev/null
timeout 90 scp $OPTS "$HOST:~/sgmn-loop/dir18/sgmn_module_hebb.py" "$BASE/sgmn_module_hebb.server.py" 2>/dev/null

# ---------- ② 逐位 md5 对照 ----------
say "--- md5 对照（服务器 vs 本地）---"
timeout 60 ssh $OPTS "$HOST" 'cd ~/sgmn-loop/dir18 && md5sum R2_write.py sgmn_module_hebb.py' 2>/dev/null | tee -a "$LOG"
md5sum "$BASE/rounds/R2_write.py" "$BASE/sgmn_module_hebb.py" | tee -a "$LOG"

# ---------- ③ Stage A 闸检查（smoke#2） ----------
say "--- smoke#2 Stage A 闸 ---"
timeout 60 ssh $OPTS "$HOST" 'grep -B1 -A3 "Stage A" ~/sgmn-loop/dir18/D18_R2_smoke2.log | head -12' 2>/dev/null | tee -a "$LOG"

# ---------- ④ 启动正式轮（若未在跑） ----------
RUNNING=$(timeout 40 ssh $OPTS "$HOST" 'pgrep -f "R2_write.py" >/dev/null && echo YES' 2>/dev/null)
if echo "$RUNNING" | grep -q YES; then
  say "--- 正式轮已在运行，跳过启动 ---"
else
  say "--- 启动正式轮（15 臂 × 3 种子）---"
  timeout 60 ssh $OPTS "$HOST" 'cd ~/sgmn-loop/dir18 && setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /opt/pytorch/bin/python -u R2_write.py > D18_R1_formal.log 2>&1 < /dev/null & sleep 5; pgrep -af "R2_write" | head -2' 2>/dev/null | tee -a "$LOG"
fi

# ---------- ⑤ 轮询直到正式轮完成 ----------
say "--- 轮询正式轮 ---"
for i in $(seq 1 240); do
  sleep 60
  DONE=$(timeout 40 ssh $OPTS "$HOST" 'grep -c "汇总与判据" ~/sgmn-loop/dir18/D18_R1_formal.log 2>/dev/null || echo 0' 2>/dev/null | tr -d "\r")
  ALIVE=$(timeout 40 ssh $OPTS "$HOST" 'pgrep -f "R2_write.py" >/dev/null && echo YES || echo NO' 2>/dev/null | tr -d "\r")
  say "poll $i: 判据块=$DONE  进程=$ALIVE"
  if echo "$DONE" | grep -qE "^[1-9]"; then say "--- 正式轮完成 ---"; break; fi
  if echo "$ALIVE" | grep -q NO; then
    say "--- 进程已退出但无判据块 ⇒ 需排查 ---"
    break
  fi
done

# ---------- ⑥ 拉回正式轮结果 ----------
say "--- 拉回正式轮结果 ---"
for f in D18_R1_formal.log R28_results.json; do
  timeout 90 scp $OPTS "$HOST:~/sgmn-loop/dir18/$f" "$BASE/" 2>/dev/null && say "got $f"
done

# ---------- ⑦ 打包 ----------
say "--- 重新打包 ---"
bash "$BASE/.ops/repack.sh" 2>&1 | tail -8 | tee -a "$LOG"

say "===== 流水线结束 ====="
