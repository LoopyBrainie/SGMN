#!/bin/bash
# 耐心重连：首次静默 25 分钟，之后每 13 分钟一次（避免每次尝试都刷新封禁计时）
BASE=/workspace/sgmn-loop/revolve/DIR-18
S=$BASE/.ops/cm
KEY=/workspace/.ssh/SGMN.pem
HOST=ec2-user@ec2-3-106-240-137.ap-southeast-2.compute.amazonaws.com
LOG=$BASE/.ops/keepalive.log
chmod 400 "$KEY" 2>/dev/null

echo "[$(date -u +%H:%M:%S)] 进入静默期 1500s 后开始首次尝试" >> "$LOG"
sleep 1500

for i in $(seq 1 100); do
  rm -f "$S" 2>/dev/null
  if timeout 30 ssh -M -S "$S" -o ControlPath="$S" -o ControlPersist=14400 \
      -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o IdentitiesOnly=yes -o ServerAliveInterval=20 -o ConnectTimeout=18 \
      "$HOST" 'echo MASTER_UP' 2>/dev/null | grep -q MASTER_UP; then
    echo "[$(date -u +%H:%M:%S)] MASTER_UP (attempt $i)" >> "$LOG"
    bash "$BASE/.ops/pipeline2.sh"
    exit 0
  fi
  echo "[$(date -u +%H:%M:%S)] attempt $i failed (13min backoff)" >> "$LOG"
  sleep 780
done
