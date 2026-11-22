"""DIR-76 裁决：读 5 个刻度（3 新 + DIR-75 两格冻结 JSON），按卡 ①②③ 判定。"""
import json
BAND = (0.95, 1.25)
CELLS = [("target", 3.0, "dir76"), ("target", 10.0, "dir76"), ("keys", 3.0, "dir76"),
         ("target", 30.0, "dir75"), ("keys", 10.0, "dir75")]
rows, ok_new = [], True
for mode, g, d in CELLS:
    j = json.load(open("/home/ec2-user/sgmn-loop/%s/out/%s_g%g.json" % (d, mode, g)))
    r = j["ratio"]; inside = BAND[0] <= r <= BAND[1]
    if d == "dir76": ok_new = ok_new and inside
    rows.append((mode, g, r, j["lo"] / j["eta_lin"], j["hi"] / j["eta_lin"], j["eta"], j["n_eval"], j["ok"], inside, d))
    print("%-6s γ=%-3g ρ=%.4f  括号[%.4f,%.4f]  η_th=%.6e  n_eval=%-2d ok=%s  ①%s  (%s)"
          % (mode, g, r, rows[-1][3], rows[-1][4], j["eta"], j["n_eval"], j["ok"], "✓" if inside else "✗", d), flush=True)
ref = {("target", 30.0): 1.0486, ("keys", 10.0): 1.1522}
cite = all(abs(r[2] - ref[(r[0], r[1])]) < 5e-4 for r in rows if r[9] == "dir75")
print("\n★ DIR-75 冻结读数复现（|Δ|<5e-4）：%s" % ("✓" if cite else "✗"))
print("① 三个新格都在 [0.95,1.25]：%s" % ("PASS" if ok_new else "FAIL"))
allin = all(r[8] for r in rows)
print("② 5 个刻度全在带内：%s  [%s]" % ("PASS" if allin else "FAIL", " · ".join("%s%g %.4f" % (r[0][0], r[1], r[2]) for r in rows)))
print("\nDIR-76 裁决 = %s" % ("SUCCEEDED" if (ok_new and allin) else "FALSIFIED"), flush=True)
