import sys, json, argparse, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir65")
import ref65 as R
p = argparse.ArgumentParser(); p.add_argument("--out", default="dir65/out/DIR65r11_formal.json"); a = p.parse_args()
C = 6.0
res = {"bound": {}, "td": [], "drift": {}, "init": {}, "mech": {}}
for n in (10000, 100000, 1000000):
    b = R.bisect(C, n, rel=0.002); res["bound"]["N=%d" % n] = b["eta"]
    print("② 边界 N=%-8d η*=%.5f" % (n, b["eta"]))
b105 = res["bound"]["N=100000"]
for f in (1.02, 1.05, 1.10, 1.25):
    eta = f * b105
    o = R.run(C, eta, 2000000, trace_every=25)
    tr = [x[1] for x in o["trace"]]
    res["td"].append(dict(frac=f, eta=eta, td=o["td"], div=o["div"],
                          t_first=float(tr[0]) if tr else None, t_max=float(max(tr)) if tr else None,
                          t_min=float(min(tr)) if tr else None))
    print("④ η=%.4f(%.2f×) T_div=%-8s t: %.2f→max %.2f" % (eta, f, o["td"], tr[0] if tr else -1, max(tr) if tr else -1))
# ③ 漂移：失败运行 与 有界运行
for tag, eta in (("fail", 1.05 * b105), ("bound", 0.8 * b105)):
    o = R.run(C, eta, 2000000, trace_every=25)
    tr = np.array([x[1] for x in o["trace"]])
    res["drift"][tag] = dict(eta=eta, div=o["div"], td=o["td"], n=len(tr),
                             t0=float(tr[0]), tmax=float(np.max(np.maximum(tr, 1/tr))),
                             mono_frac=float(np.mean(np.diff(np.log(tr)) > 0)))
    print("③ %-6s η=%.4f div=%-5s max(t,1/t)=%.3f mono=%.2f" % (tag, eta, o["div"], res["drift"][tag]["tmax"], res["drift"][tag]["mono_frac"]))
# 探索性：init 依赖（必报）
for ini in (0.001, 0.03, 0.3):
    o = R.run(C, 0.85, 1, seed=1)  # 仅占位，避免空
    bb = R.bisect_init if hasattr(R, "bisect_init") else None
    # 直接二分（init 传参）
    lo, hi = 0.3, 1.2
    import math
    for _ in range(40):
        m = 0.5 * (math.log(lo) + math.log(hi))
        if R.run(C, math.exp(m), 100000, init=ini)["div"]: hi = math.exp(m)
        else: lo = math.exp(m)
    res["init"]["%.0e" % ini] = 0.5 * (lo + hi)
    print("   init=%.0e  η*=%.5f" % (ini, res["init"]["%.0e" % ini]))
# 探索性：瞬态峰值（区分"峰值更高"与"峰后增长"）
for eta in (0.79, 0.80):
    o = R.run(C, eta, 1000, trace_every=5)
    res["mech"]["%.2f" % eta] = dict(peak=float(max(np.sqrt(x[2]) for x in o["trace"])), td=o["td"])
    print("   机制 η=%.2f 峰值‖W‖=%.3f T_div=%s" % (eta, res["mech"]["%.2f" % eta]["peak"], o["td"]))
json.dump(res, open(a.out, "w"), indent=1, default=lambda o: float(o))
print("→", a.out)
