#!/bin/bash
# 定期拉取 DIR-18 服务器端日志到本地（只读，不干扰服务器）
BASE=/workspace/sgmn-loop/revolve/DIR-18
S=$BASE/.ops/cm
KEY=/workspace/.ssh/SGMN.pem
HOST=ec2-user@ec2-3-106-240-137.ap-southeast-2.compute.amazonaws.com
OPTS="-S $S -o ControlPath=$S -i $KEY"
DEST=$BASE/.ops/live
mkdir -p "$DEST"
for i in $(seq 1 400); do
  OUT=$(timeout 60 ssh $OPTS "$HOST" 'cd ~/sgmn-loop/dir18 && for f in D18_R2_smoke2.log D18_R1_formal.log; do [ -f "$f" ] && echo "===== $f =====" && grep -vE "Loading weights|pynvml|RuntimeWarning|Consider using|  f\"" "$f"; done; echo "===== PS ====="; pgrep -af "R2_write|probe18h" | head -3; echo "===== GPU ====="; nvidia-smi --query-gpu=memory.used --format=csv,noheader' 2>/dev/null)
  if [ -n "$OUT" ]; then
    printf '%s\n' "$OUT" > "$DEST/snapshot.txt"
    echo "[$(date -u +%H:%M:%S)] pull $i ok ($(wc -c < "$DEST/snapshot.txt") bytes)" >> "$DEST/pull.log"
  else
    echo "[$(date -u +%H:%M:%S)] pull $i failed" >> "$DEST/pull.log"
  fi
  sleep 45
done
