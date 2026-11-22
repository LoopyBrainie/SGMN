r"""DIR-63 裁决器（rev 1.0，机械套用 §4 四子句；不做人工调节）"""
import json
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "dir63/out"
F = sys.argv[2] if len(sys.argv) > 2 else "DIR63_formal.json"
d = json.load(open("%s/%s" % (OUT, F)))
V = {}

# ① 层内闭式命中
r0 = max(s["rel0"] for row in d["stack"] for s in row["seeds"])
r1 = max(s["rel1"] for row in d["stack"] for s in row["seeds"])
V["c1"] = {"rel0_max": r0, "rel1_max": r1, "ok": r0 <= 1e-6 and r1 <= 1e-6}

# ② 剂量律：逐点 + 严格单调
pt = max(abs(np.mean([s["J1"] for s in row["seeds"]]) - row["Jc_analytic"]) / row["Jc_analytic"]
         for row in d["stack"])
mono = all(all(d["stack"][i]["seeds"][j]["delta"] < d["stack"][i + 1]["seeds"][j]["delta"]
               for i in range(len(d["stack"]) - 1)) for j in range(len(d["stack"][0]["seeds"])))
V["c2"] = {"pointwise_rel_max": pt, "monotone": bool(mono), "ok": bool(pt <= 1e-6 and mono)}

# ③ 两端代价
d0 = np.mean([s["delta"] for s in d["stack"][0]["seeds"]])
d90 = np.mean([s["delta"] for s in d["stack"][-1]["seeds"]])
ratio = d90 / d["Jd_analytic"]
V["c3"] = {"dJ0": d0, "dJ90": d90, "ratio": ratio,
           "ok": abs(d0 - 0.004626) <= 1e-6 and abs(d90 - 0.478503) <= 1e-6 and 10.04 <= ratio <= 10.06}

# ④ 数据加工下界 + 直连基线
lb = all(all(s["J1"] >= d["Jd_analytic"] - 1e-12 for s in row["seeds"]) for row in d["stack"])
relD = max(x["rel"] for x in d["direct"])
V["c4"] = {"lower_bound_all": bool(lb), "direct_rel_max": relD, "ok": bool(lb and relD <= 1e-6)}

# 必报量
V["report"] = {"ranks": {row["theta_deg"]: {"W0": row["rank_W0"], "W1": row["rank_W1"]} for row in d["stack"]},
               "first_step": {row["theta_deg"]: [s["first_step"] for s in row["seeds"]] for row in d["stack"]},
               "J1_by_theta": {row["theta_deg"]: float(np.mean([s["J1"] for s in row["seeds"]])) for row in d["stack"]},
               "table": [(row["theta_deg"], row["Jc_analytic"], row["delta_analytic"]) for row in d["stack"]]}
V["status"] = "SUCCEEDED" if all(V[k]["ok"] for k in ("c1", "c2", "c3", "c4")) else "FALSIFIED"
json.dump(V, open("%s/DIR63_adjudication.json" % OUT, "w"), indent=1,
              default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))

for k in ("c1", "c2", "c3", "c4"):
    print("%s: %s" % (k, "PASS" if V[k]["ok"] else "**FAIL**"))
print("① rel0=%.2e rel1=%.2e" % (V["c1"]["rel0_max"], V["c1"]["rel1_max"]))
print("② pointwise_rel=%.2e monotone=%s" % (V["c2"]["pointwise_rel_max"], V["c2"]["monotone"]))
print("③ ΔJ(0)=%.6f ΔJ(90°)=%.6f ratio=%.3f" % (V["c3"]["dJ0"], V["c3"]["dJ90"], V["c3"]["ratio"]))
print("④ 下界全成立=%s 直连 rel=%.2e" % (V["c4"]["lower_bound_all"], V["c4"]["direct_rel_max"]))
print("必报 ranks=%s" % V["report"]["ranks"])
print("必报 first_step(rel<1e-6)=%s" % V["report"]["first_step"])
print("\n★ ① %s · ② %s · ③ %s · ④ %s ⇒ **%s**"
      % tuple(["PASS" if V[k]["ok"] else "FAIL" for k in ("c1", "c2", "c3", "c4")] + [V["status"]]))
