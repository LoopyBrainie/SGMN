#!/bin/bash
# DIR-18 · 拉回服务器全部产物 + 重新打包 + 校验
set -u
BASE=/workspace/sgmn-loop/revolve/DIR-18
S=$BASE/.ops/cm
KEY=/workspace/.ssh/SGMN.pem
HOST=ec2-user@ec2-3-106-240-137.ap-southeast-2.compute.amazonaws.com
OPTS="-S $S -o ControlPath=$S -i $KEY"
LOG=$BASE/.ops/package.log

echo "[$(date -u +%H:%M:%S)] ===== DIR-18 打包开始 =====" >> "$LOG"

# ---------- ① 拉回服务器产物 ----------
mkdir -p "$BASE/rounds" "$BASE/probes"
# 日志
for f in D18_pipeline.log D18_probe18b.log D18_probe18c.log D18_probe18d.log D18_probe18e.log \
         D18_probe18f.log D18_probe18g.log D18_probe18h.log \
         D18_R2_smoke2.log D18_R2_smoke2_v1_adrsNameError_FAILED.log \
         D18_R2_smoke2_v2_gatekey_FAILED.log \
         D18_R1_formal.log D18_R1_formal_v1_D48protocol_FAILED.log; do
  timeout 90 scp $OPTS "$HOST:~/sgmn-loop/dir18/$f" "$BASE/" 2>/dev/null && echo "[$(date -u +%H:%M:%S)] got $f" >> "$LOG"
done
# JSON 结果
for f in R28_results.json R28_smoke_results.json probe18h_out.json probe18c_out.json probe18e_out.json \
         probe18f_out.json probe18g_out.json; do
  timeout 90 scp $OPTS "$HOST:~/sgmn-loop/dir18/$f" "$BASE/" 2>/dev/null && echo "[$(date -u +%H:%M:%S)] got $f" >> "$LOG"
done
# 探针脚本
for f in probe18b.py probe18c.py probe18d.py probe18e.py probe18f.py probe18g.py probe18h.py; do
  timeout 90 scp $OPTS "$HOST:~/sgmn-loop/dir18/$f" "$BASE/probes/" 2>/dev/null && echo "[$(date -u +%H:%M:%S)] got probes/$f" >> "$LOG"
done
# 服务器上的正式轮脚本（留档，与本地逐位对照）
timeout 90 scp $OPTS "$HOST:~/sgmn-loop/dir18/R2_write.py" "$BASE/rounds/R2_write.server.py" 2>/dev/null
timeout 90 scp $OPTS "$HOST:~/sgmn-loop/dir18/sgmn_module_hebb.py" "$BASE/sgmn_module_hebb.server.py" 2>/dev/null

echo "[$(date -u +%H:%M:%S)] ===== 拉取结束 =====" >> "$LOG"
ls -la "$BASE"/*.log "$BASE"/*.json 2>/dev/null | wc -l >> "$LOG"
