import sys, json, numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else "dir64/out"; F = sys.argv[2] if len(sys.argv) > 2 else "DIR64r13_formal.json"
d = json.load(open("%s/%s" % (OUT, F)))
CS = [2.0, 3.0, 6.0, 12.0, 24.0]
PRED = {"mat_batch": {2.0: 5.2174, 3.0: 3.6364, 6.0: 1.9048, 12.0: 0.9756, 24.0: 0.4938},
        "fac_batch": {2.0: 3.0000, 3.0: 2.0000, 6.0: 1.0000, 12.0: 0.5000, 24.0: 0.2500},
        "mat_cycle": {2.0: 1.1424, 3.0: 0.7219, 6.0: 0.3456, 12.0: 0.1696, 24.0: 0.0840}}
LIN_CYC = {2.0: 0.6540, 3.0: 0.3964, 6.0: 0.1814, 12.0: 0.0869, 24.0: 0.0425}
deg = lambda x: float(next(v for k, v in d["bound"][x].items() if abs(float(k) - CS[0]) < 1e-9)) if False else None
def bd(f, c):
    return float(d["bound"][f][str(c)] if str(c) in d["bound"][f] else d["bound"][f][c])
V = {}
rel = {(f, c): abs(bd(f, c) / PRED[f][c] - 1) for f in PRED for c in CS}
V["c1"] = {"rel": {"%s|%g" % (f, c): rel[(f, c)] for f, c in rel},
           "max_by_fmt": {f: max(rel[(f, c)] for c in CS) for f in PRED}}
V["c1"]["ok"] = bool(all(rel[(f, c)] <= 0.03 for f in PRED for c in CS))
V["c2"] = {"chain": {("%g" % c): [bd("fac_cycle", c), bd("mat_cycle", c), bd("fac_batch", c), bd("mat_batch", c)] for c in CS}}
V["c2"]["ok"] = bool(all(bd("fac_cycle", c) < bd("mat_cycle", c) < bd("fac_batch", c) < bd("mat_batch", c) for c in CS))
G = {c: LIN_CYC[c] / bd("fac_cycle", c) for c in CS}
V["c3"] = {"G": {"%g" % c: G[c] for c in CS}, "min": min(G.values()), "ok": bool(min(G.values()) >= 2)}
eta = np.array([s["eta"] for s in d["saw"]]); amp = np.array([s["amp"] for s in d["saw"]]); amp1 = np.array([s["amp_1e5"] for s in d["saw"]])
slope = float(np.polyfit(np.log(eta), np.log(amp), 1)[0])
decay = np.abs(amp - amp1) / amp1
V["c4"] = {"slope": slope, "amp": [float(x) for x in amp], "amp_1e5": [float(x) for x in amp1],
           "decay_rel": [float(x) for x in decay], "argmax": [s["argmax"] for s in d["saw"]],
           "ok": bool(0.8 <= slope <= 1.2 and max(decay) <= 0.10)}
V["ratios_fac_batch_over_Doverc"] = {("%g" % c): bd("fac_batch", c) / PRED["fac_batch"][c] for c in CS}
V["recheck"] = d.get("recheck", {}); V["init"] = d.get("init", {}); V["horizon"] = d.get("horizon", {})
V["verdict"] = "SUCCEEDED" if all(V[k]["ok"] for k in ("c1", "c2", "c3", "c4")) else "FALSIFIED"
for k in ("c1", "c2", "c3", "c4"):
    print("%s: %s" % (k, "PASS" if V[k]["ok"] else "FAIL"))
print("① 逐格式最大相对差:", {f: "%.4f" % v for f, v in V["c1"]["max_by_fmt"].items()})
print("② 包含链:", all(V["c2"]["ok"] for _ in [0]))
print("③ G:", {k: "%.2f" % v for k, v in G.items()}, "min=%.2f" % min(G.values()))
print("④ 斜率=%.2f 衰减=%s" % (slope, ["%.0f%%" % (100 * x) for x in decay]))
print("fac_batch/D·c:", {k: "%.3f" % v for k, v in V["ratios_fac_batch_over_Doverc"].items()})
print("★ %s" % V["verdict"])
json.dump(V, open("%s/DIR64_adjudication.json" % OUT, "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
