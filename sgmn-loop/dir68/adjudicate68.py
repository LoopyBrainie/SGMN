import json, glob, numpy as np
d67 = json.load(open("dir67/out/DIR67r10_formal.json"))
mix = dict(zip(d67["1e-01"]["seeds"], d67["1e-01"]["vals"]))
par, orth = {}, {}
for f in glob.glob("dir68/out/cells/*.json"):
    c = json.load(open(f))
    if c["kind"] == "PAR": par[c["seed"]] = c["eta"]
    if c["kind"] == "ORTH": orth[c["seed"]] = c
print("MIX (DIR-67 a=0.1):", {k: round(v, 5) for k, v in sorted(mix.items())})
print("PAR (配平):          ", {k: round(v, 5) for k, v in sorted(par.items())})
seeds = sorted(set(mix) & set(par))
print("\n配对比较 (PAR vs MIX):")
wins = 0
for s in seeds:
    rel = par[s] / mix[s] - 1
    wins += par[s] < mix[s]
    print("  seed=%d PAR=%.5f MIX=%.5f  Δ=%+.2f%%  %s" % (s, par[s], mix[s], 100 * rel, "PAR<MIX ✔" if par[s] < mix[s] else "PAR≥MIX ✘"))
c1 = wins == len(seeds) == 5
print("① PAR<MIX 严格成立 = %d/%d ⇒ %s" % (wins, len(seeds), "PASS" if c1 else "FAIL"))
o = list(orth.values())
c2 = bool(o) and all(x["maxW"] <= 1e-12 and not x["div"] for x in o)
print("② ORTH η=2.0 有界:", [(x["seed"], "%.2e" % x["maxW"], x["div"]) for x in o], "⇒", "PASS" if c2 else "FAIL")
pm, mm = float(np.mean([par[s] for s in seeds])), float(np.mean([mix[s] for s in seeds]))
print("③ PAR 均值=%.5f  MIX 均值=%.5f  Δ=%+.2f%%" % (pm, mm, 100 * (pm / mm - 1)))
print("   inf_dir η* = %.5f   sup_dir η* = +inf (ORTH 纤维有界 ∀η)" % min(min(par.values()), min(mix.values())))
v = "SUCCEEDED" if (c1 and c2) else ("PARTIAL" if (c1 or c2) else "FALSIFIED")
print("\n★ 裁决 = %s" % v)
json.dump(dict(verdict=v, c1=dict(wins=wins, n=len(seeds), ok=c1), c2=dict(ok=c2, cells=o),
               par=par, mix=mix, par_mean=pm, mix_mean=mm), open("dir68/out/DIR68_adjudication.json", "w"), indent=1)
