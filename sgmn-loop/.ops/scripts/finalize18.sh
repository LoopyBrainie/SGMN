#!/bin/bash
# DIR-18 · CPU 降级轮完成后的**一键收尾**：解析 → D-25 定稿 → 台账/汇总/报告同步 → 打包 → 校验
set -u
ROOT=/workspace/sgmn-loop
R=$ROOT/revolve/DIR-18
LOG=$R/.ops/finalize.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

RES="$R/.ops/cpu/R3_cpu_results.json"
if [ ! -f "$RES" ]; then
  if [ -f "$R/rounds/R28_smoke_results.json" ]; then
    cp -f "$R/rounds/R28_smoke_results.json" "$RES"; say "✓ 结果 JSON 已从 rounds/ 取用"
  else
    say "★ 无结果 JSON ⇒ 降级轮尚未跑完 ⇒ 退出"; exit 1
  fi
fi

say "===== DIR-18 降级轮收尾开始 ====="

# —— ① 解析结果（人读格式）——
python3 "$R/.ops/parse_cpu_results.py" "$RES" > "$R/.ops/cpu/R3_cpu_readout.txt" 2>&1
say "读数已出：$R/.ops/cpu/R3_cpu_readout.txt"

# —— ② D-25 定稿（追加 D-25-b 行，不改既有行）——
python3 "$R/.ops/finalize_d25.py" "$RES" | tee -a "$LOG"

# —— ②b 裁决书 §2.3 填入实测判据 + 臂指标 ——
python3 "$R/.ops/fill_verdict.py" "$RES" | tee -a "$LOG"

# —— ③ 证据表行数校验（只增行）——
python3 - <<'PYEOF' | tee -a "$LOG"
import re, pathlib
t = pathlib.Path("/workspace/sgmn-loop/evidence/table.md").read_text(encoding="utf-8")
# ★ 行首锚定：证据行一律以 `| **D-xx（...）**` 开头；
#   原正则 `\*\*(D-\d+(?:-b)?)[（(]` 会把**正文里**的「**D-10（DIR-04）** 行」也计进来 ⇒ 假重复。
ids = re.findall(r"(?m)^\|\s*\*\*(D-\d+(?:-[a-z])?)", t)
print("证据表条目数 =", len(ids), "| 唯一编号 =", len(set(ids)))
dup = [i for i in set(ids) if ids.count(i) > 1]
print("重复编号:", dup if dup else "无")
if dup:
    raise SystemExit("★★ 证据表存在重复编号 ⇒ 违反「只增行、不改行」⇒ 中止")
PYEOF

# —— ④ 重新打包 + 全包重建 ——
bash "$R/.ops/repack.sh" 2>&1 | tail -4 | tee -a "$LOG"
bash "$R/.ops/rebuild_full.sh" 2>&1 | tail -4 | tee -a "$LOG"

# —— ⑤ 交付物核验 ——
for f in "$ROOT/SGMN_LOOP_full.zip" "$ROOT/LOOP_REPORT_full.zip" \
         "$ROOT/DIR-18_artifacts.zip" "$ROOT/MANIFEST.md" \
         "$ROOT/REPORT.md" "$ROOT/SUMMARY.md" "$ROOT/DIRECTION-REGISTER.md" \
         "$ROOT/LOOP-MECHANISM.md" "$ROOT/revisions.md" \
         "$ROOT/evidence/table.md" "$ROOT/evidence/ledger.md" \
         "$R/VERDICT.md" "$R/DIRECTION_rev14.md"; do
  if [ -f "$f" ]; then
    printf "  ✓ %-52s %10d B  md5=%s\n" "$(basename "$f")" "$(stat -c%s "$f")" "$(md5sum "$f" | cut -d' ' -f1)"
  else
    printf "  ✗ 缺失: %s\n" "$f"
  fi
done | tee -a "$LOG"
say "===== 收尾结束 ====="
