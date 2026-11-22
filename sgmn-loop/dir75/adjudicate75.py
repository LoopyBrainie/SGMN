"""DIR-75 裁决：只读两格 JSON，按 rev 1.1 的 ①②③ 判定（阈值写死，禁止改）。"""
import json
CRIT = {"target": (1.00, 1.50, True), "keys": (1.00, 1.20, False)}   # (lo, hi, lo_inclusive)
CELLS = [("target", 30.0), ("keys", 10.0)]
rows, ok1 = [], True
for mode, g in CELLS:
    d = json.load(open("/home/ec2-user/sgmn-loop/dir75/out/%s_g%g.json" % (mode, g)))
    lo, hi, inc = CRIT[mode]; r = d["ratio"]
    inside = (r >= lo if inc else r > lo) and r <= hi
    ok1 = ok1 and inside
    rows.append((mode, g, r, d["lo"] / d["eta_lin"], d["hi"] / d["eta_lin"], d["eta"], d["n_eval"], d["ok"], inside, d["u1"], d["u2"]))
    print("%-6s γ=%-3g ρ_bal=%.4f  括号[%.4f,%.4f]  η_th=%.6e  n_eval=%d ok=%s  u1*=%.9f u2*=%.9f  ①%s"
          % (mode, g, r, rows[-1][3], rows[-1][4], d["eta"], d["n_eval"], d["ok"], d["u1"], d["u2"], "✓" if inside else "✗"), flush=True)
allr = [r[2] for r in rows]
ok2 = all(r >= 0.80 for r in allr)
print("\n① 阈值位置（两格都须满足）：%s" % ("PASS" if ok1 else "FAIL"))
print("② H73 重裁（两格 ρ≥0.80）：%s  [%s]" % ("PASS" if ok2 else "FAIL", " · ".join("%s %.4f" % (r[0], r[2]) for r in rows)))
print("\nDIR-75 裁决 = %s" % ("SUCCEEDED" if ok2 else "FALSIFIED"), flush=True)
