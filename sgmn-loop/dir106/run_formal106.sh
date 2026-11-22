#!/bin/bash
# DIR-106 · 一格一进程的形式轮驱动（`DEF-105-02` 强制）
#
# 为什么必须一格一进程：同进程连跑多格时，cuBLAS workspace 的分配受前序格影响
# ⇒ 归约顺序变化 ⇒ Adam 轨迹混沌放大 ⇒ 同一配置在不同「前序历史」下读数不同
# （`DIR-105` 实测：seed 7777 lin@600 隔离 0.8750 vs 6 格之后 0.7812）。见 `.ops/ENV-NOTES.md`。
#
# 格计划（卡片 §4，rev 1.2）：
#   主扫描  4 n_train × 4 split × 3 seed = 48   arm=main
#   零假设  4 n_train × 1 split × 3 seed = 12   arm=shuf   （A106-5）
#   W←W⁰    4 n_train × 1 split × 3 seed = 12   arm=w0     （内容对照）
#   合计 72 格（n=32 的 full 臂已在 Stage A 跑过 2 格，不重复）
#
# 用法：bash run_formal106.sh        # 约 72 × 100 s ≈ 2.0 h
set -u
cd "$(dirname "$0")" || exit 1
PY=/opt/pytorch/bin/python
OUTD=out/cells
mkdir -p "$OUTD"

NS=(16 20 24 28)
SEEDS=(20261028 7777 60013)

run_one () {           # $1=arm $2=n $3=split $4=seed
  local arm=$1 n=$2 sp=$3 sd=$4
  local tag="${arm}_${n}_${sp}_${sd}"
  [ -f "$OUTD/$tag.json" ] && { echo "  · $tag 已有，跳过"; return 0; }
  echo "=============================================================="
  echo "[$(date -u +%H:%M:%S)] $tag"
  "$PY" R1_task106.py --stage formal --arm "$arm" --n-train "$n" --split "$sp" --seed "$sd" \
        > "$OUTD/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "  ✗ rc=$rc（见 $OUTD/$tag.log）"; return 1; fi
  # 该格的 stage 产物 → out/cells/<tag>.json
  "$PY" - "$tag" <<'PYEOF'
import json, pathlib, sys
tag = sys.argv[1]
cand = [p for p in pathlib.Path("out").glob("cell_*.json")]
if not cand:
    print("  ✗ 未找到 cell_*.json"); sys.exit(1)
src = max(cand, key=lambda p: p.stat().st_mtime)
d = json.loads(src.read_text(encoding="utf-8"))
cs = d.get("cells", [])
assert len(cs) == 1, f"{tag}: 期望 1 格，实得 {len(cs)}"
pathlib.Path(f"out/cells/{tag}.json").write_text(
    json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
c = cs[0]
print(f"  acc_train={c['acc_train']:.4f} acc_held={c['acc_held']} "
      f"W_drift={c.get('W_drift')} {c['sec']}s")
src.unlink()
PYEOF
}

n=0; fail=0
# ---- 主扫描：4 n × 4 split × 3 seed
for nn in "${NS[@]}"; do
  for sp in 0 1 2 3; do
    for sd in "${SEEDS[@]}"; do
      n=$((n+1)); echo "--- [$n/72] main"
      run_one main "$nn" "$sp" "$sd" || fail=$((fail+1))
    done
  done
done
# ---- 零假设 + W←W⁰：每 n × 3 seed，split 固定 0
for arm in shuf w0; do
  for nn in "${NS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      n=$((n+1)); echo "--- [$n/72] $arm"
      run_one "$arm" "$nn" 0 "$sd" || fail=$((fail+1))
    done
  done
done

echo "=============================================================="
echo "完成：$((n-fail))/$n 格；失败 $fail"
ls -1 "$OUTD"/*.json 2>/dev/null | wc -l