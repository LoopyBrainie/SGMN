"""DIR-80 协调器：读 4 格 JSON，算判据 ①②③，写出 DIR80_formal.json。"""
import json, glob, math, os
import numpy as np

OUT = "/home/ec2-user/sgmn-loop/dir80/out"
GAM = [10.0, 20.0, 30.0]
GAP_T = {10.0: 0.18486729169987637, 20.0: 0.5075515924376042, 30.0: 0.8836529416648964}  # D-100 冻结
EXP_T, SUP_EXP_T = 0.422, 0.511
cells = {}
for g in (3.0, 10.0, 20.0, 30.0):
    p = os.path.join(OUT, "cell_%g.json" % g)
    if not os.path.exists(p):
        cells[str(g)] = dict(missing=True); continue
    d = json.load(open(p))
    if d.get("stage") != "done":
        cells[str(g)] = dict(incomplete=d.get("stage")); continue
    cells[str(g)] = dict(mode=d["mode"], eta_th=d["eta_th"], rho=d["rho"], eta_warm=d["eta_warm"],
                         gap=d["rho"] / d["eta_th"] - 1, fp_res=d["fp_res"], fp_steps=d["fp_steps"],
                         sup=d["sup"]["sup"], peak_t=d["sup"]["peak_t"],
                         interior=d["sup"]["interior_peak"], at_t=d["sup"]["at_t"],
                         nev=d["nev"], cap_hit=d["cap_hit"], width=d["width"],
                         rho_diag={k: d["rho_diag"][k] for k in ("nev", "cap_hit", "width", "lmax_at", "lmax_half")})

V = {}
ok_cells = all(str(g) in cells and "eta_th" in cells[str(g)] for g in GAM)
if ok_cells:
    r = [cells[str(g)]["gap"] / GAP_T[g] for g in GAM]
    c1 = all(0.5 <= x <= 2.0 for x in r)
    lg = np.log(GAM)
    a = float(np.polyfit(lg, np.log([cells[str(g)]["gap"] for g in GAM]), 1)[0])
    c2 = abs(a - EXP_T) <= 0.25
    b = float(np.polyfit(lg, np.log([cells[str(g)]["sup"] for g in GAM]), 1)[0])
    c3 = abs(b - SUP_EXP_T) <= 0.25
    V = dict(C1=dict(ok=bool(c1), ratio=[float(x) for x in r], band=[0.5, 2.0], n_pass=int(sum(0.5 <= x <= 2.0 for x in r))),
             C2=dict(ok=bool(c2), a=a, ref=EXP_T, band=[EXP_T - 0.25, EXP_T + 0.25]),
             C3=dict(ok=bool(c3), b=b, ref=SUP_EXP_T, band=[SUP_EXP_T - 0.25, SUP_EXP_T + 0.25], gating=False))
else:
    V = dict(error="cells incomplete", cells=cells)
out = dict(cells=cells, verdict=V)
json.dump(out, open(os.path.join(OUT, "DIR80_formal.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(V, ensure_ascii=False, indent=1))
if V and "C1" in V:
    print("判据: ① %s ② %s ③(非门禁) %s" % tuple("PASS" if V[k]["ok"] else "FAIL" for k in ("C1", "C2", "C3")))
