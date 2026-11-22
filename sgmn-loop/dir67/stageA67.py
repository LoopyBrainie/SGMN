import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir66")
import ref66 as R
C, N = 6.0, 200000
ok = True
def chk(n, c, m=""):
    global ok; ok = ok and c; print("%-44s %s | %s" % (n, "PASS" if c else "FAIL", m))
# A67-1 装置锚
b = R.bisect_a(C, 0.3, N, seed=20261118)
chk("A67-1 装置锚 a=0.3 seed=181118", abs(b["eta"]-0.79704) <= 0.005, "η*=%.5f" % b["eta"])
# A67-2 种子换方向
w1 = R.run_a(C, 0.5, 0.1, 10, seed=20261118)["W"]; w2 = R.run_a(C, 0.5, 0.1, 10, seed=20261119)["W"]
rel = abs(np.linalg.norm(w1)-np.linalg.norm(w2))/max(np.linalg.norm(w1),1e-30)
chk("A67-2 种子确实换方向", rel >= 1e-6, "‖W0‖ 相对差=%.3e" % rel)
# A67-3 可算（抽样两格）
b2 = R.bisect_a(C, 0.1, N, seed=20261119); b3 = R.bisect_a(C, 0.3, N, seed=20261119)
chk("A67-3 可算（两格有限且两端异号）", all(np.isfinite(x["eta"]) and x["ok"] for x in (b2,b3)),
    "a=0.1:%.5f a=0.3:%.5f" % (b2["eta"], b3["eta"]))
# A67-4 分辨力
b4 = R.bisect_a(C, 0.1, N, seed=20261119)
chk("A67-4 分辨力（重跑 ≤0.5%）", abs(b2["eta"]-b4["eta"])/b2["eta"] <= 5e-3, "%.5f vs %.5f" % (b2["eta"], b4["eta"]))
print("\n★ Stage A: %d/4 PASS" % (4 if ok else -1))
