#!/bin/bash
# DIR-18 · 维持 SSH master；master 一断就重连（短间隔重试，连接成功即复用）
BASE=/workspace/sgmn-loop/revolve/DIR-18
S=$BASE/.ops/cm
KEY=/workspace/.ssh/SGMN.pem
HOST=ec2-user@ec2-3-106-240-137.ap-southeast-2.compute.amazonaws.com
LOG=$BASE/.ops/keepalive.log
chmod 400 "$KEY" 2>/dev/null
mkdir -p "$BASE/.ops"

while true; do
  # master 活着吗？
  if timeout 25 ssh -S "$S" -o ControlPath="$S" -O check -i "$KEY" "$HOST" >/dev/null 2>&1; then
    sleep 40; continue
  fi
  rm -f "$S" 2>/dev/null
  echo "[$(date -u +%H:%M:%S)] reconnecting..." >> "$LOG"
  timeout 30 ssh -M -S "$S" -o ControlPath="$S" -o ControlPersist=14400 \
      -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o IdentitiesOnly=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=6 \
      -o ConnectTimeout=18 "$HOST" 'echo MASTER_UP' 2>/dev/null | grep -q MASTER_UP \
      && echo "[$(date -u +%H:%M:%S)] MASTER_UP" >> "$LOG"
  sleep 20
done
