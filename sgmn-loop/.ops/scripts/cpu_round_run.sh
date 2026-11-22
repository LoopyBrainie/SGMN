#!/bin/bash
# DIR-18 · CPU 降级轮（三阶段：Stage A 前半 → Stage A 后半 → 主循环）
# 改动 8/11：沙箱 3 GB 内存下，Stage A 与主循环必须**分进程**跑（峰值叠加 ⇒ OOM）
set -u
PY=/usr/bin/python3
R=/workspace/sgmn-loop/revolve/DIR-18
LOGD=$R/.ops/cpu
mkdir -p "$LOGD"

# —— 依赖自检 + 自愈重装（纪律 #31）——
if ! $PY -c "import torch" 2>/dev/null; then
  echo "[$(date -u +%H:%M:%S)] torch 缺失 ⇒ 重装" >&2
  $PY -m pip install --break-system-packages --no-deps --force-reinstall torch \
      --index-url https://download.pytorch.org/whl/cpu >&2 || exit 9
fi
if ! $PY -c "import transformers" 2>/dev/null; then
  echo "[$(date -u +%H:%M:%S)] transformers 缺失 ⇒ 重装" >&2
  $PY -m pip install --break-system-packages --no-deps --force-reinstall \
      transformers tokenizers regex safetensors huggingface_hub truststore hf-xet \
      httpx2 httpcore2 anyio annotated-doc click h11 idna sniffio certifi urllib3 \
      charset-normalizer requests pyarrow tqdm filelock packaging pyyaml sympy \
      setuptools mpmath numpy -i https://pypi.tuna.tsinghua.edu.cn/simple >&2 || exit 9
fi

export SGMN_MODEL=/workspace/models/qwen05b
export SGMN_ROUND_TAG=R3cpu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=131072
export SGMN_N_TRAIN=${SGMN_N_TRAIN:-40} SGMN_N_EVAL=${SGMN_N_EVAL:-12} SGMN_N_SEED=${SGMN_N_SEED:-2}
cd "$R/rounds"

run_phase () {   # $1=phase  $2=logfile  $3=标题
  # ★ 缺陷 D56b（2026-11-06）：原先用 `tee "$2"`（**无 -a**）⇒ 每次重启**截断日志**
  #   ⇒ 前一次尝试的输出（= 失败轮/中断轮留档）**被抹掉** ⇒ 违反「失败轮必须留档」。
  #   修法：`tee -a`（追加）+ 每次尝试写入**带时间戳的尝试分隔行**。
  echo "" | tee -a "$2"
  echo "========== [$3] 尝试开始 @ $(date -u +%Y-%m-%dT%H:%M:%SZ) (pid=$$) ==========" | tee -a "$2"
  SGMN_PHASE="$1" $PY -u R3_cpu.py --smoke 2>&1 \
    | grep --line-buffered -vE "it/s\]|UserWarning|Consider using|traj.append" | tee -a "$2"
  return ${PIPESTATUS[0]}
}

if [ "${SGMN_MAIN_ONLY:-0}" = "1" ] && [ -f "$LOGD/stageA_checks.json" ]; then
  echo "★ Stage A 缓存已存在 ⇒ 只跑主循环（阶段 3）"
  run_phase main "$LOGD/R3_cpu_main.log" "阶段 3/3：主循环（15 臂）" | tail -3
  RC3=${PIPESTATUS[0]}
  cp "$R/R28_smoke_results.json" "$LOGD/R3_cpu_results.json" 2>/dev/null
  echo "[$(date -u +%H:%M:%S)] 主循环退出码=$RC3"
  exit $RC3
fi

# ---- 阶段 1/3：Stage A 前半（①–⑭）----
run_phase stagea1 "$LOGD/R3_cpu_stageA1.log" "阶段 1/3：Stage A 前半（①–⑭）" | tail -3
RC=${PIPESTATUS[0]}
echo "[$(date -u +%H:%M:%S)] 阶段 1 退出码=$RC" | tee -a "$LOGD/R3_cpu_stageA1.log"
if [ "$RC" != "0" ]; then
  echo "★ 前半段未全通过 ⇒ **中止**（纪律 ①）"; exit 3
fi

# ---- 阶段 2/3：Stage A 后半（⑮–⑰ + A4'），复用前半段缓存 ----
run_phase stagea "$LOGD/R3_cpu_stageA2.log" "阶段 2/3：Stage A 后半（⑮–⑰ + A4'）" | tail -3
RC2=${PIPESTATUS[0]}
echo "[$(date -u +%H:%M:%S)] 阶段 2 退出码=$RC2" | tee -a "$LOGD/R3_cpu_stageA2.log"
if [ "$RC2" != "0" ]; then
  echo "★ Stage A 未全通过 ⇒ **中止，不开主循环**（纪律 ①）"; exit 3
fi

# ---- 阶段 3/3：主循环（复用完整 Stage A 缓存）----
run_phase main "$LOGD/R3_cpu_main.log" "阶段 3/3：主循环（15 臂）" | tail -3
RC3=${PIPESTATUS[0]}
echo "[$(date -u +%H:%M:%S)] 阶段 3 退出码=$RC3" | tee -a "$LOGD/R3_cpu_main.log"
cp "$R/R28_smoke_results.json" "$LOGD/R3_cpu_results.json" 2>/dev/null
echo "★ 三阶段完成。聚合结果：$LOGD/R3_cpu_results.json"
exit $RC3
