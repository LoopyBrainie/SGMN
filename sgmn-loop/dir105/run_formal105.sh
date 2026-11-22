#!/bin/bash
# DIR-105 · 一格一进程的形式轮驱动（`DEF-105-02` 的修法）
#
# 为什么这样做：同一进程内连跑多格时，cuBLAS workspace 的分配受前序格影响 ⇒
# 归约顺序变化 ⇒ 600 步 Adam 轨迹混沌放大 ⇒ 同一配置在不同「前序历史」下读数不同
# （实测：seed 7777 lin@600 隔离跑 0.8750，在 6 格之后跑 0.7812；基座 sha1 全程不变）。
# 「一格一进程」使每格的分配器初态完全相同 ⇒ 可复现，且比全局确定性内核快 3.2×。
#
# 用法：bash run_formal105.sh            # 12 格全跑（约 36 min）
set -u
cd "$(dirname "$0")" || exit 1
PY=/opt/pytorch/bin/python
OUTD=out/cells
mkdir -p "$OUTD"

FAMS=(lin gate mlp1)
STEPS=(600 3000)
SEEDS=(20261028 7777)

n=0; fail=0
for fam in "${FAMS[@]}"; do
  for stp in "${STEPS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      n=$((n+1))
      tag="${fam}_${stp}_${sd}"
      if [ -f "$OUTD/$tag.json" ]; then
        echo "[$n/12] $tag 已有，跳过"
        continue
      fi
      echo "=============================================================="
      echo "[$n/12] $tag  $(date -u +%H:%M:%S)"
      "$PY" R1_task105.py --stage formal --families "$fam" --steps "$stp" --seeds "$sd" \
            > "$OUTD/$tag.log" 2>&1
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "  ✗ rc=$rc（见 $OUTD/$tag.log）"
        fail=$((fail+1))
        continue
      fi
      # 把这一格的 cells[0] 抽出来单独存档
      "$PY" - "$tag" <<'PYEOF'
import json, pathlib, sys
tag = sys.argv[1]
src = pathlib.Path("out/stage_formal.json")
dst = pathlib.Path(f"out/cells/{tag}.json")
d = json.loads(src.read_text(encoding="utf-8"))
cs = d["cells"]
assert len(cs) == 1, f"{tag}: 期望 1 格，实得 {len(cs)}"
dst.write_text(json.dumps(cs[0], ensure_ascii=False, indent=2), encoding="utf-8")
c = cs[0]
print(f"  acc_all32={c['acc_all32']:.4f} acc_W0={c['acc_W0']:.4f} "
      f"W_drift={c['W_drift']:.2e} steps={c['train_steps']}")
PYEOF
      rm -f out/stage_formal.json
    done
  done
done

echo "=============================================================="
echo "完成：$((12-fail))/12 格；失败 $fail"
ls -1 "$OUTD"/*.json 2>/dev/null | wc -l