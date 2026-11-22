import sys, json, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir71"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir65")
import R2_bisect71 as B, ref70 as S, ref65 as R
ok = True
def chk(n, c, m=""):
    global ok; ok = ok and bool(c); print("%-52s %s | %s" % (n, "PASS" if c else "FAIL", m), flush=True)
# A71-1 二分自检（γ=0 锚括号）
r = B.bisect("target", 0.0, lo=0.884, hi=0.891)
good = r["ok"] and (0.884 < r["eta"] < 0.891) and r["hist"][0][1] is False and r["hist"][1][1] is True
chk("A71-1 二分自检 γ=0 锚括号", good, "η*=%.6f ∈(0.884,0.891) n_eval=%d" % (r["eta"], r["n_eval"]))
# A71-5 区间比 + n_eval
ratio = r["hi"] / r["lo"]; chk("A71-5 返回区间比 ≤1.02 且 n_eval 有记录", ratio <= 1.0201 and r["n_eval"] >= 2, "比=%.5f n_eval=%d" % (ratio, r["n_eval"]))
# A71-2 γ=0 两模式逐位复现 ref65
r0 = R.run(6.0, 0.5, 20000, seed=20261118, init=0.1)
for mode in ("target", "keys"):
    s0 = S.run_stack(6.0, 0.0, 0.5, 20000, seed=20261118, init=0.1, mode=mode)
    dw = float(np.max(np.abs(s0["W1"] - r0["W"])))
    chk("A71-2 γ=0 逐位复现 ref65 [%s]" % mode, dw == 0.0, "max|ΔW1|=%.1e" % dw)
# A71-3 坏括号被拒
bad = B.bisect("target", 3.0, lo=1.0, hi=1.0, verbose=False)
chk("A71-3 坏括号(两端同号)被拒", (bad["ok"] is False) and (not np.isfinite(bad["eta"])), "ok=%s eta=%s note=%s" % (bad["ok"], bad["eta"], bad["note"]))
# A71-4 发散格的 W1 必须 not finite（167）
d = S.run_stack(6.0, 10.0, 3.0, 200000, mode="keys")
chk("A71-4 发散格 W1 非有限（NaN 不静默通过）", d["div"] and (not np.isfinite(d["W1"]).all()), "div=%s finite=%s" % (d["div"], np.isfinite(d["W1"]).all()))
print("\n★ StageA-71: %d/5" % (5 if ok else -1), flush=True)
