import sys, json, argparse, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir64")
import ref64 as R
FMT = ("mat_batch", "mat_cycle", "fac_batch", "fac_cycle")
CS = (2.0, 3.0, 6.0, 12.0, 24.0)
PRED = {  # 卡片 rev 1.2 §2 冻结值
    "mat_batch": {2.0: 5.2174, 3.0: 3.6364, 6.0: 1.9048, 12.0: 0.9756, 24.0: 0.4938},
    "fac_batch": {2.0: 3.0000, 3.0: 2.0000, 6.0: 1.0000, 12.0: 0.5000, 24.0: 0.2500},
    "mat_cycle": {2.0: 1.1424, 3.0: 0.7219, 6.0: 0.3456, 12.0: 0.1696, 24.0: 0.0840},
}
LIN_CYC = {2.0: 0.6540, 3.0: 0.3964, 6.0: 0.1814, 12.0: 0.0869, 24.0: 0.0425}
p = argparse.ArgumentParser(); p.add_argument("--out", default="dir64/out/DIR64r12_formal.json"); a = p.parse_args()
res = {"bound": {}, "recheck": {}, "saw": [], "init": {}, "horizon": {}}
for f in FMT:
    res["bound"][f] = {}
    for c in CS:
        b = R.bisect(f, c, n=100000, rel=0.005, lo=1e-4, hi=8.0)
        res["bound"][f][c] = b["eta"]
        print("%-10s c=%-5.1f η*=%.5f" % (f, c, b["eta"]))
        # N×5 复测（0.9 倍边界）
        if np.isfinite(b["eta"]):
            o = R.run(f, c, 0.9 * b["eta"], 500000)
            res["recheck"]["%s|%s" % (f, c)] = dict(eta=0.9 * b["eta"], div=o["div"], steps=o["steps"])
# 锯齿律：fac_cycle c=6，三点
b6 = res["bound"]["fac_cycle"][6.0]
Js = R.Jstar(6.0)
for frac in (0.3, 0.5, 0.7):
    eta = frac * b6
    o = R.run("fac_cycle", 6.0, eta, 200000, tail=12)
    rl = np.abs(np.array(o["tail_J"]) - Js) / Js
    # 早期窗口（t=1e5 附近）另取一次
    o2 = R.run("fac_cycle", 6.0, eta, 100000, tail=12)
    rl2 = np.abs(np.array(o2["tail_J"]) - Js) / Js
    res["saw"].append(dict(eta=eta, frac=frac, amp=float(rl.max() - rl.min()), amp_1e5=float(rl2.max() - rl2.min()), rel12=[float(x) for x in rl], rel12_1e5=[float(x) for x in rl2],
                           argmax=int(np.argmax(rl)), rel_mean=float(rl.mean()), div=o["div"]))
    print("saw η=%.5f amp=%.3e amp@1e5=%.3e argmax=%d" % (eta, rl.max() - rl.min(), rl2.max() - rl2.min(), int(np.argmax(rl))))
# init 依赖（必报）
for ini in (1e-3, 1e-2, 0.1, 0.3):
    res["init"]["fac_batch|%.0e" % ini] = R.bisect_init("fac_batch", 6.0, ini, n=100000, rel=0.005)
    res["init"]["fac_cycle|%.0e" % ini] = R.bisect_init("fac_cycle", 6.0, ini, n=100000, rel=0.005)
# 时域依赖（必报）
for n in (10000, 100000):
    res["horizon"]["fac_cycle|%d" % n] = R.bisect("fac_cycle", 6.0, n=n, rel=0.01)["eta"]
    res["horizon"]["fac_batch|%d" % n] = R.bisect("fac_batch", 6.0, n=n, rel=0.01)["eta"]
json.dump(res, open(a.out, "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
print("→", a.out)
