#!/bin/bash
# DIR-42 · **全流程**（preflight ⇒ Stage A ⇒ smoke ⇒ 正式轮 ⇒ merge）
# ★ Stage A 不过 ⇒ **拦下正式轮**（纪律：硬闸）
# ★ 判据只在 `R1_task42.py merge` 里（判定逻辑只有一份）
set -u
cd /home/ec2-user/sgmn-loop
mkdir -p dir42/out

# ── ★ preflight（**离线、零算力**）：四份合成自检 ⇒ 不过则**不花 trainer 时间**
#   ① verify_merge42.py：判定表**六支**与卡 §5 逐例一致（含 `INCOMPLETE`/`DIVERGED`）
#   ② verify_freeze42.py：冻结守卫（未冻结 / 卡被改过 ⇒ 拒绝）+ 守卫挂在 `formal_arm`/`merge` 上
#   ③ verify_resume42.py：续跑机制（追加原子性 / 指纹作废 / 截断尾行容忍 / 只重算未落盘的）
#   ④ verify_card42.py：卡 ↔ 代码（**阈值漂移探测器** / 断言 1:1 / 六支对应 / 探针不可达）
#   ★ 这四份自检**只读源码、不导入装置、不训练** ⇒ 可在本地或 trainer 上跑。
echo "=== preflight（离线自检）==="
for t in dir42/verify_merge42.py dir42/verify_freeze42.py dir42/verify_resume42.py dir42/verify_card42.py; do
  python3 "$t" > "dir42/D42_preflight_$(basename "$t" .py).log" 2>&1
  rc=$?
  tail -3 "dir42/D42_preflight_$(basename "$t" .py).log"
  if [ "$rc" != "0" ]; then
    echo "★ preflight 未通过（$t）⇒ **不启动正式轮**"; exit 1
  fi
done

echo "=== Stage A ==="
OMP_NUM_THREADS=1 /opt/pytorch/bin/python dir42/R1_task42.py stageA > dir42/D42_stageA_run1.log 2>&1
rcA=$?
tail -40 dir42/D42_stageA_run1.log
if [ "$rcA" != "0" ]; then echo "★ Stage A 未通过 ⇒ 正式轮被拦下"; exit 1; fi

echo "=== smoke ==="
OMP_NUM_THREADS=1 /opt/pytorch/bin/python dir42/R1_task42.py smoke > dir42/D42_smoke_run1.log 2>&1
rcS=$?
tail -20 dir42/D42_smoke_run1.log
if [ "$rcS" != "0" ]; then echo "★ smoke 失败"; exit 1; fi

echo "=== formal（3 臂 × 4 分片，4 路并发）==="
bash dir42/run_formal42.sh

echo "=== merge（独立裁决）==="
OMP_NUM_THREADS=1 /opt/pytorch/bin/python dir42/R1_task42.py merge > dir42/D42_merge_run1.log 2>&1
rcM=$?
cat dir42/D42_merge_run1.log
echo "merge rc=$rcM"
