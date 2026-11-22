#!/bin/bash
# 长退避重连：每 5 分钟试一次，成功即建立 master
S=/workspace/sgmn-loop/revolve/DIR-18/.ops/cm
KEY=/workspace/.ssh/SGMN.pem
HOST=ec2-user@ec2-3-106-240-137.ap-southeast-2.compute.amazonaws.com
LOG=/workspace/sgmn-loop/revolve/DIR-18/.ops/keepalive.log
for i in $(seq 1 300); do
  rm -f "$S" 2>/dev/null
  if timeout 30 ssh -M -S "$S" -o ControlPath="$S" -o ControlPersist=14400 \
      -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o IdentitiesOnly=yes -o ServerAliveInterval=20 -o ConnectTimeout=18 \
      "$HOST" 'echo MASTER_UP' 2>/dev/null | grep -q MASTER_UP; then
    echo "[$(date -u +%H:%M:%S)] MASTER_UP (attempt $i)" >> "$LOG"
    exit 0
  fi
  echo "[$(date -u +%H:%M:%S)] attempt $i failed" >> "$LOG"
  sleep 300
done
