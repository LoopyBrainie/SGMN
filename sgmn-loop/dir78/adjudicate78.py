"""DIR-78 裁决器：warm 行（本轮）+ bal 行（D-98 冻结引用）→ 判据 ①②③。"""
import json, glob, os
BAL = {("target", 3.0): (1.0532, 1.0551), ("target", 10.0): (1.0537, 1.0551), ("target", 30.0): (1.0539, 1.0486),
       ("keys", 3.0): (1.0473, 1.0472), ("keys", 10.0): (1.1481, 1.1522)}   # (ρ_λ, ρ_div) 来自 D-98
D94 = {("target", 30.0): 0.9965, ("keys", 10.0): 0.5679, ("target", 3.0): 0.9940}
rows, ok1, ok2 = [], 0, 0
for f in sorted(glob.glob("/home/ec2-user/sgmn-loop/dir78/out/warm_*.json")):
    d = json.load(open(f)); k = (d["mode"], float(d["gamma"]))
    bl, bd = BAL[k]
    rel = abs(d["rho_div"] - d["rho_lam"]) / d["rho_lam"]
    c1 = rel <= 0.02; c2 = d["rho_lam"] < bl
    ok1 += c1; ok2 += c2
    rows.append(dict(cell="%s %g" % k, bal_lam=bl, bal_div=bd, warm_lam=d["rho_lam"], warm_div=d["rho_div"],
                     rel=rel, c1=c1, c2=c2, bracket=[d["lo"] / d["eta_lin"], d["hi"] / d["eta_lin"]],
                     n_eval=d["n_eval"], dW=d["dW"], d94=D94.get(k)))
print("格         ρ_λ^bal  ρ_div^bal | ρ_λ^warm ρ_div^warm 相对差  ①  ②  warm括号        n_ev  ΔW        D-94对照")
for r in sorted(rows, key=lambda x: (x["cell"])):
    print("%-9s %8.4f %9.4f | %8.4f %9.4f %6.2f%% %s %s  [%.4f,%.4f] %3d  %.1e  %s"
          % (r["cell"], r["bal_lam"], r["bal_div"], r["warm_lam"], r["warm_div"], 100 * r["rel"],
             "✓" if r["c1"] else "✗", "✓" if r["c2"] else "✗", r["bracket"][0], r["bracket"][1], r["n_eval"],
             r["dW"], ("%.4f" % r["d94"]) if r["d94"] else "-"))
n = len(rows)
c1f, c2f = ok1 >= 4, ok2 == 5
print("\n① 相对差 ≤2%%：%d/%d ⇒ %s（要求 ≥4/5）" % (ok1, n, "PASS" if c1f else "FAIL"))
print("② ρ_λ^warm < ρ_λ^bal 严格：%d/%d ⇒ %s（要求 5/5）" % (ok2, n, "PASS" if c2f else "FAIL"))
print("③ 报告：%s" % ("warm 行逐格读数见上；target 30 的 ρ_λ^warm 为 %s" % [("%.4f" % r["warm_lam"]) for r in rows if r["cell"] == "target 30"]))
verd = "SUCCEEDED" if (c1f and c2f and n == 5) else ("PARTIAL" if c2f else ("CENSORED" if n < 3 else "FALSIFIED"))
print("\n★ 裁决：%s（可算格 %d/5）" % (verd, n))
json.dump(dict(rows=rows, ok1=ok1, ok2=ok2, c1=c1f, c2=c2f, n=n, verdict=verd),
          open("/home/ec2-user/sgmn-loop/dir78/out/adjudication78.json", "w"), indent=1)
