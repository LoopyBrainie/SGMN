#!/bin/bash
# DIR-18 · GPU 正式轮 → 沙箱 的**拉取 + 定稿**管线
#
# 用法：bash .ops/pull_formal.sh
# 前提：正式轮在服务器上已完成（48/48 臂，判据输出已打印）
set -u
K="-i /workspace/.ssh/SGMN.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25"
H=ec2-user@ec2-3-27-212-181.ap-southeast-2.compute.amazonaws.com
R=/workspace/sgmn-loop/revolve/DIR-18
DST=$R/.ops/formal
mkdir -p "$DST"

echo "[$(date -u +%H:%M:%S)] ① 检查服务器上正式轮状态"
timeout 120 ssh $K $H 'cd ~/sgmn-loop/dir18; echo "臂: $(grep -c "seed=" D18_R4_formal.log)/48"; grep -E "判据通过|联合裁决" D18_R4_formal.log | tail -6' || { echo "★ SSH 失败"; exit 9; }

echo "[$(date -u +%H:%M:%S)] ② 拉取产物"
for f in D18_R4_formal.log R28_results.json .ops/formal_incremental_full.json; do
  timeout 300 scp $H:~/sgmn-loop/dir18/$f "$DST/" 2>/dev/null && echo "  ✓ $f" || echo "  ✗ $f（缺失）"
done

echo "[$(date -u +%H:%M:%S)] ③ md5 核验"
timeout 120 ssh $K $H 'cd ~/sgmn-loop/dir18; md5sum D18_R4_formal.log R28_results.json 2>/dev/null' > "$DST/server_md5.txt" 2>/dev/null
( cd "$DST" && md5sum D18_R4_formal.log R28_results.json 2>/dev/null ) > "$DST/local_md5.txt" 2>/dev/null
echo "--- 服务器 ---"; cat "$DST/server_md5.txt"
echo "--- 本地   ---"; cat "$DST/local_md5.txt"
if [ -s "$DST/server_md5.txt" ] && [ -s "$DST/local_md5.txt" ]; then
  if diff <(sort "$DST/server_md5.txt") <(sort "$DST/local_md5.txt") > /dev/null; then
    echo "  ✓ md5 逐位一致"
  else
    echo "  ✗ md5 不一致 ⇒ 传输有问题 ⇒ 人工核对"
  fi
fi

echo "[$(date -u +%H:%M:%S)] ④ 抽取判据与联合裁决"
grep -E "^\[(PASS|FAIL)\]" "$DST/D18_R4_formal.log" > "$DST/criteria.txt" 2>/dev/null
sed -n '/联合裁决/,/^====/p' "$DST/D18_R4_formal.log" > "$DST/verdict_block.txt" 2>/dev/null
echo "  判据行数: $(wc -l < "$DST/criteria.txt")"
cat "$DST/verdict_block.txt"
echo "[$(date -u +%H:%M:%S)] ⑤ 完成"
