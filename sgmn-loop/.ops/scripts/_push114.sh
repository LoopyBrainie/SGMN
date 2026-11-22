set -e
cd ~/sgmn-loop
mkdir -p dir114/out
cat > dir114/R1_task114.py
cat > dir114/test_verdict114.py
cat > revolve/DIR-114/DIRECTION_rev114_v11.md
echo "--- md5 ---"
md5sum dir114/R1_task114.py dir114/test_verdict114.py revolve/DIR-114/DIRECTION_rev114_v11.md
