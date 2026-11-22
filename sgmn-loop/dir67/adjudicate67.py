import json, glob, sys, numpy as np
base = "dir67/out"
d = json.load(open(base + "/DIR67r10_formal.json"))
for f in glob.glob(base + "/cells/*.json"):
    c = json.load(open(f)); k = "%.0e" % c["a"]
    g = d.setdefault(k, {"vals": []})
    if c["seed"] not in [s for s in g.get("seeds", [])]:
        g["vals"].append(c["eta"]); g.setdefault("seeds", []).append(c["seed"])
for k, g in d.items():
    g["vals"] = [float(x) for x in g["vals"]]
    g["mean"] = float(np.mean(g["vals"])); g["spread"] = float(max(g["vals"]) / min(g["vals"]) - 1)
json.dump(d, open(base + "/DIR67r10_formal.json", "w"), indent=1)
for k in sorted(d): print("a=%-8s n=%d vals=%s mean=%.5f spread=%.3f%%" % (k, len(d[k]["vals"]),
      [round(x, 5) for x in d[k]["vals"]], d[k]["mean"], 100 * d[k]["spread"]))
sp = {k: d[k]["spread"] for k in d}
c1 = sum(1 for k in sp if sp[k] >= 0.01)
med = float(np.median([sp[k] for k in sp]))
c1_ok = c1 >= 2; c2_ok = med >= 0.015
print("\n① 极差≥1%% 的 a 个数 = %d/3 (需≥2) ⇒ %s" % (c1, "PASS" if c1_ok else "FAIL"))
print("② 三个极差中位数 = %.3f%% (需≥1.5%%) ⇒ %s" % (100 * med, "PASS" if c2_ok else "FAIL"))
means = {k: d[k]["mean"] for k in d}
print("③ 均值: %s" % {k: round(v, 5) for k, v in means.items()})
print("   a=0.3 相对 a=0.1 均值下降 = %.4f (%.1f%%)" % (means["1e-01"] - means["3e-01"], 100 * (1 - means["3e-01"] / means["1e-01"])))
print("   a=0.3 相对 a=5e-02 均值下降 = %.4f (%.1f%%)" % (means["5e-02"] - means["3e-01"], 100 * (1 - means["3e-01"] / means["5e-02"])))
print("   方向极差/幅度均值差(a=0.3) = %.1f×" % (sp["3e-01"] / (1 - means["3e-01"] / means["1e-01"])))
verdict = "SUCCEEDED" if (c1_ok and c2_ok) else ("PARTIAL" if (c1_ok or c2_ok) else "FALSIFIED")
print("\n★ 裁决 = %s" % verdict)
json.dump(dict(verdict=verdict, c1=dict(count=c1, ok=c1_ok), c2=dict(median=med, ok=c2_ok),
               spreads=sp, means=means), open(base + "/DIR67_adjudication.json", "w"), indent=1)
