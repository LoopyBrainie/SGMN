#!/bin/bash
# 断言本轮产物**真的**进了包（DEF-72-7：push 被整条拒绝时，⑦ 的存在性检查会空过）
# 用法: assert_round.sh <证据行标签如 D-93> <源目录如 dir72> <revolve 目录如 DIR-72>
set -u
TAG=${1:?}
DIR=${2:?}
REV=${3:?}
Z=SGMN_LOOP_full.zip
ok=1
chk() { if [ "$1" -ge 1 ]; then echo "  ✓ $2"; else echo "  ✗ $2"; ok=0; fi; }
chk "$(unzip -p $Z SGMN_LOOP_PACKAGE/03_LEDGERS/evidence_table.md 2>/dev/null | grep -c "$TAG")" "证据表含 $TAG"
chk "$(unzip -l $Z | grep -c "$DIR/")" "包内含 $DIR/"
chk "$(unzip -l $Z | grep -c "revolve/$REV/")" "包内含 revolve/$REV/"
chk "$(grep -c "$REV" DIRECTION-REGISTER.md)" "register 含 $REV"
[ "$ok" = 1 ] && echo "★ assert_round PASS" || { echo "★ assert_round FAIL"; exit 1; }
