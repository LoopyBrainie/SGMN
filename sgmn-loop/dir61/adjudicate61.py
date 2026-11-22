r"""DIR-61 裁决器（rev 1.1，机械套用卡 §4 的三段判据；不做任何人工调节）"""
import json
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "dir61/out"
init = json.load(open("%s/DIR61r11_formal_init.json" % OUT))
A = init["arms"]
ver = {}

# ---- ① 陷阱集精确性 ----
c1 = []
for nm in ("ZERO", "ORTHO"):
    a = A[nm]
    ok = a["dW_max"] <= 1e-12 and a["J_max_dev"] <= 1e-12
    c1.append({"arm": nm, "dW_max": a["dW_max"], "J_max_dev": a["J_max_dev"], "ok": ok, "want": "frozen"})
for nm in ("VNULL", "UNULL", "RAND", "DEADBLK"):
    a = A[nm]
    ok = a["dW_N"] >= 1e-2
    c1.append({"arm": nm, "dW_N": a["dW_N"], "ok": ok, "want": "move"})
vals = [x.get("dW_max", x.get("dW_N")) for x in c1]
inter = [v for v in vals if 1e-12 < v < 1e-2]
ver["c1"] = {"arms": c1, "no_intermediate": len(inter) == 0,
             "ok": all(x["ok"] for x in c1) and len(inter) == 0}

# ---- ② 逃逸律（三变体皆须通过） ----
c2 = {}
for v in ("d2r1", "d4r1", "d4r2"):
    d = json.load(open("%s/DIR61r11_formal_esc_%s.json" % (OUT, v)))
    f = d["fits"]
    ok = (f["log"]["R2"] >= 0.99 and f["log"]["b"] > 0 and f["R2_gap"] >= 0.02 and f["n"] >= 8)
    c2[v] = {"R2_log": f["log"]["R2"], "b": f["log"]["b"], "R2_pow": f["pow"]["R2"],
             "gap": f["R2_gap"], "n": f["n"], "ok": ok}
ver["c2"] = {"variants": c2, "ok": all(x["ok"] for x in c2.values())}

# ---- ③ 机制一致性（|k^T V| 单调 17/17） ----
c3 = {}
for v in ("d2r1", "d4r1", "d4r2"):
    d = json.load(open("%s/DIR61r11_formal_esc_%s.json" % (OUT, v)))
    mono = sum(1 for r in d["rows"] if r["align_monotone"])
    viol = sum(r["align_n_monotone_viol"] for r in d["rows"])
    steps = sum(r["n_steps"] - 1 for r in d["rows"])
    md = max(r["align_max_rel_drop"] for r in d["rows"])
    # 机制替换诊断（**非判据**）：值侧投影 |v^T U|
    mono_v = sum(1 for r in d["rows"] if r["vtu_monotone"])
    viol_v = sum(r["vtu_n_viol"] for r in d["rows"])
    c3[v] = {"rows_monotone": mono, "viol": viol, "steps": steps, "max_rel_drop": md,
             "ok": mono == 17,
             "diag_vtu_rows_monotone": mono_v, "diag_vtu_viol": viol_v,
             "diag_vtu_steps": steps}
ver["c3"] = {"variants": c3, "ok": all(x["ok"] for x in c3.values())}

ver["status"] = ("SUCCEEDED" if (ver["c1"]["ok"] and ver["c2"]["ok"] and ver["c3"]["ok"])
                 else "FALSIFIED")
print(json.dumps(ver, indent=1))
open("%s/DIR61r11_adjudication.json" % OUT, "w").write(json.dumps(ver, indent=1))
print("\n★ ① %s · ② %s · ③ %s ⇒ **%s**"
      % ("PASS" if ver["c1"]["ok"] else "FAIL", "PASS" if ver["c2"]["ok"] else "FAIL",
         "PASS" if ver["c3"]["ok"] else "FAIL", ver["status"]))
