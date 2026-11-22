#!/bin/bash
# DIR-18 · 总控：等 SSH 恢复 → 建立 master → 自动收尾（拉回 + 启动正式轮）
BASE=/workspace/sgmn-loop/revolve/DIR-18
S=$BASE/.ops/cm
KEY=/workspace/.ssh/SGMN.pem
HOST=ec2-user@ec2-3-106-240-137.ap-southeast-2.compute.amazonaws.com
LOG=$BASE/.ops/keepalive.log
chmod 400 "$KEY" 2>/dev/null

for i in $(seq 1 400); do
  rm -f "$S" 2>/dev/null
  if timeout 30 ssh -M -S "$S" -o ControlPath="$S" -o ControlPersist=14400 \
      -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o IdentitiesOnly=yes -o ServerAliveInterval=20 -o ConnectTimeout=18 \
      "$HOST" 'echo MASTER_UP' 2>/dev/null | grep -q MASTER_UP; then
    echo "[$(date -u +%H:%M:%S)] MASTER_UP (attempt $i)" >> "$LOG"
    bash "$BASE/.ops/autofinal.sh"
    exit 0
  fi
  echo "[$(date -u +%H:%M:%S)] attempt $i failed" >> "$LOG"
  sleep 240
done
