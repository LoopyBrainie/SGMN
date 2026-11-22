#!/bin/bash
# DIR-18 · 稳健推送 + 流水线（一次连接内完成传输与启动；带 md5 校验与重试）
# 用法: bash push18.sh <阶段>
#   阶段 = ship   : 传脚本 + 装置 + 探针，校验 md5
#   阶段 = smoke  : 传脚本 + 装置，校验 md5，启动 smoke#2
#   阶段 = formal : 传脚本 + 装置，校验 md5，启动正式轮
#   阶段 = pull   : 拉回全部日志与结果
set -u
BASE=/workspace/sgmn-loop/revolve/DIR-18
HOST=ec2-user@ec2-3-106-240-137.ap-southeast-2.compute.amazonaws.com
HOST=ec2-user@ec2-3-106-240-137.ap-southeast-2.compute.amazonaws.com
KEY=/workspace/.ssh/SGMN.pem
OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -o ConnectTimeout=20 -o ServerAliveInterval=30"
STAGE=${1:-ship}
LOG=$BASE/.ops/push18_$STAGE.log
mkdir -p "$BASE/.ops"

W_SCRIPT=$(md5sum "$BASE/rounds/R2_write.py" | cut -d' ' -f1)
W_MOD=$(md5sum "$BASE/sgmn_module_hebb.py" | cut -d' ' -f1)
W_PROBE=$(md5sum "$BASE/probes/probe18h.py" | cut -d' ' -f1)
echo "[$(date -u +%H:%M:%S)] stage=$STAGE want script=$W_SCRIPT mod=$W_MOD probe=$W_PROBE" >> "$LOG"

B_SCRIPT=$(base64 -w0 "$BASE/rounds/R2_write.py")
B_MOD=$(base64 -w0 "$BASE/sgmn_module_hebb.py")
B_PROBE=$(base64 -w0 "$BASE/probes/probe18h.py")

for i in $(seq 1 300); do
  case "$STAGE" in
    pull)
      OUT=$(timeout 60 ssh $OPTS $HOST 'cd ~/sgmn-loop/dir18 && for f in D18_pipeline.log D18_probe18h.log D18_R2_smoke2.log D18_R1_formal.log D18_probe18h.log probe18h_out.json R28_results.json R28_smoke_results.json; do [ -f "$f" ] && echo "===== $f =====" && cat "$f"; done; echo "===== PS ====="; pgrep -af "probe18h|R2_write" | head -4' 2>/dev/null)
      if [ -n "$OUT" ]; then
        echo "$OUT" > "$BASE/.ops/pull_$STAGE.txt"
        for f in probe18h_out.json R28_results.json R28_smoke_results.json; do
          timeout 90 scp $OPTS "$HOST:~/sgmn-loop/dir18/$f" "$BASE/.ops/" 2>/dev/null
        done
        echo "PULL_OK $(date -u +%H:%M:%S)" >> "$LOG"; exit 0
      fi
      ;;
    *)
      OUT=$(printf '%s\n%s\n%s\n' "$B_SCRIPT" "$B_MOD" "$B_PROBE" | timeout 45 ssh $OPTS $HOST '
          cd ~/sgmn-loop/dir18 || exit 1
          i=0
          while IFS= read -r line; do
            i=$((i+1))
            case $i in
              1) echo "$line" | base64 -d > R2_write.py.new ;;
              2) echo "$line" | base64 -d > sgmn_module_hebb.py.new ;;
              3) echo "$line" | base64 -d > probe18h.py.new ;;
            esac
          done
          mv R2_write.py.new R2_write.py
          mv sgmn_module_hebb.py.new sgmn_module_hebb.py
          mv probe18h.py.new probe18h.py
          md5sum R2_write.py sgmn_module_hebb.py probe18h.py' 2>/dev/null)
      G_S=$(echo "$OUT" | grep "R2_write.py" | cut -d' ' -f1)
      G_M=$(echo "$OUT" | grep "sgmn_module_hebb.py" | cut -d' ' -f1)
      G_P=$(echo "$OUT" | grep "probe18h.py" | cut -d' ' -f1)
      echo "[$(date -u +%H:%M:%S)] attempt $i script=$G_S mod=$G_M probe=$G_P" >> "$LOG"
      if [ "$G_S" = "$W_SCRIPT" ] && [ "$G_M" = "$W_MOD" ] && [ "$G_P" = "$W_PROBE" ]; then
        echo "MD5_OK $(date -u +%H:%M:%S)" >> "$LOG"
        case "$STAGE" in
          smoke)
            timeout 45 ssh $OPTS $HOST 'cd ~/sgmn-loop/dir18 && pkill -f "R2_write.py --smoke" 2>/dev/null; pkill -f "R2_write.py$" 2>/dev/null; sleep 2; setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /opt/pytorch/bin/python -u R2_write.py --smoke > D18_R2_smoke2.log 2>&1 < /dev/null & sleep 4; pgrep -af R2_write | head -3' 2>/dev/null
            ;;
          formal)
            timeout 45 ssh $OPTS $HOST 'cd ~/sgmn-loop/dir18 && setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /opt/pytorch/bin/python -u R2_write.py > D18_R1_formal.log 2>&1 < /dev/null & sleep 4; pgrep -af R2_write | head -3' 2>/dev/null
            ;;
        esac
        echo "LAUNCH_OK $(date -u +%H:%M:%S)" >> "$LOG"; exit 0
      fi
      ;;
  esac
  sleep 45
done
echo "PUSH_FAIL $(date -u +%H:%M:%S)" >> "$LOG"
exit 1
