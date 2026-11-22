"""DIR-73 裁决（rev 1.1 §2 字面；判据与阈值冻结）
① 两格（target 30、keys 10）ratio = η*_near/η*_lin ≥ 0.80 两格都
② 对照格 target 3 ratio ∈[0.90,1.10]
③ 报告条款：字段齐全（不单独定裁决）
裁决：①∧② ⇒ SUCCEEDED；①∧¬② ⇒ PARTIAL；¬① ⇒ FALSIFIED；端点同号 ⇒ CENSORED
"""
import json, os, sys
OUT = "/home/ec2-user/sgmn-loop/dir73/out/grid"
def load(mode, g, d="1e-3"):
    p = os.path.join(OUT, "near_%s_g%g_d%s.json" % (mode, g, ("%g" % float(d))))
    return json.load(open(p))
cells = [load("target", 3.0), load("target", 30.0), load("keys", 10.0)]
print("%-7s %-4s %10s %11s %8s %8s %s" % ("mode", "γ", "δ", "η*_near", "η*_lin", "ratio", "ok/bracket"))
for c in cells:
    print("%-7s %-4g %10.0e %11.6g %11.6g %8.4f %s %s"
          % (c["mode"], c["gamma"], c["delta"], c["eta_near"], c["eta_lin"], (c["ratio"] or float("nan")),
             c["ok"], ["%.6g" % x for x in c["bracket"]]))
def ratio_of(mode, g):
    for c in cells:
        if c["mode"] == mode and c["gamma"] == float(g): return c["ratio"], c["ok"]
r30, ok30 = ratio_of("target", 30.0); rk10, okk10 = ratio_of("keys", 10.0); r3, ok3 = ratio_of("target", 3.0)
cens = not all(c["ok"] for c in cells)
c1 = (not cens) and r30 >= 0.80 and rk10 >= 0.80
c2 = (not cens) and 0.90 <= r3 <= 1.10
verdict = "CENSORED" if cens else ("SUCCEEDED" if (c1 and c2) else ("PARTIAL" if c1 else "FALSIFIED"))
far = {("target", 3.0): 0.9977, ("target", 30.0): 0.5273, ("keys", 10.0): 0.3298}
print("\n① 两格 ratio ≥0.80 : %s  (target30=%.4f, keys10=%.4f)" % (c1, r30, rk10))
print("② 对照 target3 ∈[0.90,1.10] : %s  (%.4f)" % (c2, r3))
print("③ 字段齐全 : %s" % all(all(k in c for k in ("eta_near", "eta_lin", "ratio", "delta", "pert_meas", "n_eval", "bracket")) for c in cells))
print("\n远起点(init=0.1, D-93) → 近不动点(δ=1e-3, 本卡) 比值对照:")
for c in cells:
    k = (c["mode"], c["gamma"])
    print("  %-6s γ=%-3g  远 %.4f → 近 %.4f  (Δ=%+.4f)" % (c["mode"], c["gamma"], far[k], c["ratio"], c["ratio"] - far[k]))
json.dump(dict(cells=cells, c1=c1, c2=c2, c3=True, verdict=verdict, far=dict((("%s_%g" % k), v) for k, v in far.items())),
          open("/home/ec2-user/sgmn-loop/dir73/out/verdict73.json", "w"), indent=1)
print("\nVERDICT", verdict)
