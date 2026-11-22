#!/bin/bash
# 温和轮询：每 90 秒试一次连接；成功即建立长驻 master 并保持
HOST=ec2-user@ec2-3-106-240-137.ap-southeast-2.compute.amazonaws.com
KEY=/workspace/.ssh/SGMN.pem
OPS=/workspace/sgmn-loop/revolve/DIR-18/.ops
S=$OPS/cm
LOG=$OPS/wait_ssh.log
mkdir -p "$OPS"
chmod 400 "$KEY" 2>/dev/null
for i in $(seq 1 500); do
  rm -f "$S" 2>/dev/null
  if timeout 30 ssh -M -S "$S" -o ControlPath="$S" -o ControlPersist=14400 \
      -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o IdentitiesOnly=yes -o ServerAliveInterval=60 -o ConnectTimeout=20 \
      "$HOST" 'echo MASTER_ESTABLISHED' >/tmp/_mout 2>/dev/null; then
    if grep -q MASTER_ESTABLISHED /tmp/_mout 2>/dev/null; then
      echo "[$(date -u +%H:%M:%S)] MASTER_UP attempt $i" >> "$LOG"
      while true; do
        timeout 90 ssh -S "$S" -o ControlPath="$S" -i "$KEY" "$HOST" 'date -u +%H:%M:%S' >/dev/null 2>&1 || {
          echo "[$(date -u +%H:%M:%S)] master died" >> "$LOG"; break; }
        sleep 60
      done
      continue
    fi
  fi
  echo "[$(date -u +%H:%M:%S)] attempt $i failed" >> "$LOG"
  sleep 90
done
