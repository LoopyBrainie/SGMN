import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir66")
import ref66 as R
C, N = 6.0, 200000
ok = True
def chk(n, c, m=""):
    global ok; ok = ok and c
    print("%-46s %s | %s" % (n, "PASS" if c else "FAIL", m))
# A66-1 二分可执行
chk("A66-1 二分可执行（0.5 有界 / 2.0 发散）", (not R.run_a(C,0.5,0.3,N)["div"]) and R.run_a(C,2.0,0.3,N)["div"])
# A66-2 逃逸位置可测
o = R.run_a(C,1.02*0.79705,0.3,N)   # DEF-66-1: η=1.5 时 T_div=25 < 记录间隔 ⇒ 无有限记录，守卫不可达；改用 T_div≈75 的 η
chk("A66-2 逃逸位置可测（末有限记录距失败≤25步）", np.isfinite(o["t_esc"]) and o["td"] is not None and (o["td"]-o["t_esc_step"]) <= 25,
    "t_esc=%.3f @%s, T_div=%s" % (o["t_esc"], o["t_esc_step"], o["td"]))
# A66-3 确定性
a1 = R.run_a(C,0.5,0.3,N)["W"]; a2 = R.run_a(C,0.5,0.3,N)["W"]
chk("A66-3 确定性（逐位相同）", np.array_equal(a1,a2), "max|Δ|=%.1e" % np.max(np.abs(a1-a2)))
# A66-4 分辨力
b1 = R.bisect_a(C,0.3,N); b2 = R.bisect_a(C,0.3,N)
chk("A66-4 分辨力（两次一致 ≤0.5%）", b1["ok"] and b2["ok"] and abs(b1["eta"]-b2["eta"])/b1["eta"] <= 5e-3,
    "η*=%.5f / %.5f" % (b1["eta"], b2["eta"]))
# A66-5 端点复现（仪器一致性）
chk("A66-5 端点复现 a=0.3（|η*−0.79705|≤0.005）", abs(b1["eta"]-0.79705) <= 0.005, "η*=%.5f (D-86: 0.79705)" % b1["eta"])
# A66-6 a 可独立设定
w1 = R.run_a(C,0.5,1e-4,10)["W"]; w2 = R.run_a(C,0.5,0.3,10)["W"]
chk("A66-6 a 可独立设定（‖W0‖ 差 ≥1e3×）", np.linalg.norm(w1)+1e-30 > 0 and (np.linalg.norm(w2)/(np.linalg.norm(w1)+1e-30)) >= 1e3,
    "‖W0‖: %.3e vs %.3e" % (np.linalg.norm(w1), np.linalg.norm(w2)))
print("\n★ Stage A: %d/6 PASS" % (6 if ok else -1))
