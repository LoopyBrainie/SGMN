import sys, json, argparse, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir66")
import ref66 as R
p = argparse.ArgumentParser(); p.add_argument("--out", default="dir67/out/DIR67r10_formal.json"); a = p.parse_args()
C, N = 6.0, 200000
AS = [5e-2, 0.1, 0.3]; SEEDS = [20261118, 20261119, 20261120, 20261121, 20261122]
res = {}
for aa in AS:
    vals = []
    for s in SEEDS:
        b = R.bisect_a(C, aa, N, seed=s); vals.append(b["eta"])
        print("a=%-8.0e seed=%d η*=%.5f" % (aa, s, b["eta"]))
    res["%.0e" % aa] = dict(vals=vals, mean=float(np.mean(vals)), spread=float(max(vals)/min(vals)-1))
    json.dump(res, open(a.out, "w"), indent=1, default=float)   # 增量落盘（每组 a 一写）
    print("   ⇒ a=%.0e mean=%.5f 跨种子极差=%.3f%%" % (aa, res["%.0e" % aa]["mean"], 100*res["%.0e" % aa]["spread"]))
json.dump(res, open(a.out, "w"), indent=1, default=float); print("→", a.out)
