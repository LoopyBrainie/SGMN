r"""DIR-62 裁决器（rev 1.1，机械套用卡 §4 的四条判据；不做任何人工调节）"""
import json
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "dir62/out"
d = json.load(open("%s/DIR62r11_formal.json" % OUT))
A = {(a["tag"], a["r"], a["device"]): a for a in d["arms"]}
S = d["sens"]
LD = 2 / 2.05
LP = 0.05 / 2.05
ver = {}

# ---- ① 存活律：WEAK r=4 ----
a = A[("WEAK", 4, "W")]
ok = all(s["dgmax"] <= 1e-9 and s["dgdata"] <= 1e-9 for s in a["seeds"])
ver["c1"] = {"arm": "WEAK/r4/DeviceW", "dgmax": max(s["dgmax"] for s in a["seeds"]),
             "dgdata": max(s["dgdata"] for s in a["seeds"]),
             "gains_mean": [sum(s["gains"][j] for s in a["seeds"]) / len(a["seeds"]) for j in range(3)],
             "gdata_mean": sum(s["gdata"] for s in a["seeds"]) / len(a["seeds"]), "ok": ok}

# ---- ② 稀释律 ----
a = A[("DILUTE", 2, "W")]
ok = all(s["d_data"] <= 1e-9 and s["d_P"] <= 1e-9 for s in a["seeds"])
ver["c2"] = {"c_data_mean": sum(s["c_data"] for s in a["seeds"]) / len(a["seeds"]),
             "c_P_mean": sum(s["c_P"] for s in a["seeds"]) / len(a["seeds"]),
             "pred": [LD, LP], "d_max": max(max(s["d_data"], s["d_P"]) for s in a["seeds"]), "ok": ok}

# ---- ③ 争秩律：两档 × r=1..4 ----
c3 = {}
for tag in ("WEAK", "STRONG"):
    for r in (1, 2, 3, 4):
        a = A[(tag, r, "W")]
        ok = all(s["rel"] <= 1e-6 and s["dgmax"] <= 1e-6 and s["dgdata"] <= 1e-6 for s in a["seeds"])
        c3["%s/r%d" % (tag, r)] = {
            "Jref": a["Jref"], "J_mean": sum(s["J"] for s in a["seeds"]) / len(a["seeds"]),
            "rel_max": max(s["rel"] for s in a["seeds"]),
            "gains_ref": a["gref"],
            "gains_mean": [sum(s["gains"][j] for s in a["seeds"]) / len(a["seeds"]) for j in range(3)],
            "gdata_mean": sum(s["gdata"] for s in a["seeds"]) / len(a["seeds"]), "ok": ok}
evict = abs(c3["STRONG/r1"]["gdata_mean"]) <= 1e-6
ver["c3"] = {"arms": c3, "eviction_reproduced": evict, "ok": all(x["ok"] for x in c3.values()) and evict}

# ---- ④ 参数化对照：DeviceD r=1 ----
c4 = {}
for tag in ("WEAK", "STRONG"):
    a = A[(tag, 1, "D")]
    ok = all(s["rel"] <= 1e-6 and s["dgmax"] <= 1e-6 and s["dgdata"] <= 1e-6 for s in a["seeds"])
    c4[tag] = {"Jref": a["Jref"], "rel_max": max(s["rel"] for s in a["seeds"]),
               "gains_mean": [sum(s["gains"][j] for s in a["seeds"]) / len(a["seeds"]) for j in range(3)],
               "gdata_mean": sum(s["gdata"] for s in a["seeds"]) / len(a["seeds"]), "ok": ok}
ver["c4"] = {"arms": c4, "ok": all(x["ok"] for x in c4.values())}

# ---- 判据外必报量 ----
ver["report"] = {
    "dJ_param_WEAK": A[("WEAK", 1, "W")]["Jref"] - A[("WEAK", 1, "D")]["Jref"],
    "dJ_param_STRONG": A[("STRONG", 1, "W")]["Jref"] - A[("STRONG", 1, "D")]["Jref"],
    "full_arm_rel": {t: max(s["rel"] for s in A[(t, 6, "W(full)")]["seeds"]) for t in ("WEAK", "STRONG")},
    "sens": S}
ver["status"] = "SUCCEEDED" if all(ver[k]["ok"] for k in ("c1", "c2", "c3", "c4")) else "FALSIFIED"
open("%s/DIR62r11_adjudication.json" % OUT, "w").write(json.dumps(ver, indent=1))
for k in ("c1", "c2", "c3", "c4"):
    print("%s: %s" % (k, "PASS" if ver[k]["ok"] else "**FAIL**"))
print("① gains=%s gdata=%.9f (Δ=%.1e)" % (["%.9f" % g for g in ver["c1"]["gains_mean"]], ver["c1"]["gdata_mean"], ver["c1"]["dgmax"]))
print("② (c_data,c_P)=(%.12f, %.12f) 预测=(%.12f, %.12f) Δ=%.1e"
      % (ver["c2"]["c_data_mean"], ver["c2"]["c_P_mean"], LD, LP, ver["c2"]["d_max"]))
for k, x in ver["c3"]["arms"].items():
    print("③ %-11s Jref=%.10f J=%.10f rel=%.1e gains=%s gdata=%.6f"
          % (k, x["Jref"], x["J_mean"], x["rel_max"], ["%.4f" % g for g in x["gains_mean"]], x["gdata_mean"]))
print("③ 驱逐复现: %s | ④ %s" % (ver["c3"]["eviction_reproduced"],
                                 {t: "rel=%.1e gdata=%.6f" % (v["rel_max"], v["gdata_mean"]) for t, v in c4.items()}))
print("必报: ΔJ_param WEAK=%.4f STRONG=%.4f | full_arm_rel=%s"
      % (ver["report"]["dJ_param_WEAK"], ver["report"]["dJ_param_STRONG"], ver["report"]["full_arm_rel"]))
print("\n★ ① %s · ② %s · ③ %s · ④ %s ⇒ **%s**"
      % tuple(["PASS" if ver[k]["ok"] else "FAIL" for k in ("c1", "c2", "c3", "c4")] + [ver["status"]]))
