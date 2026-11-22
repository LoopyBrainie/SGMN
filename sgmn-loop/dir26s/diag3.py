#!/opt/pytorch/bin/python
import json, sys
for _p in ("dir26s","dir26r","dir26o","dir26k","dir26j","dir26i","dir26h","dir26g","dir26f","dir26e","dir26c","dir26b","dir26","dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")
import numpy as np, torch
import R1_dose_ar1_v3 as V3
from R1_gating12 import cal_at, SEEDS
disk = json.load(open("/home/ec2-user/sgmn-loop/dir26s/out/DIR26S_rows.json"))["rows"]
dmap = {(r["phi"], r["seed"]): r for r in disk}
tau = cal_at(V3.N_KV)
sd = SEEDS[0]
r0 = V3.run_unit_dose(sd, 0.0, tau, cnt0=None); cnt0 = r0["counts"]
r9 = V3.run_unit_dose(sd, 0.9, tau, cnt0=cnt0)
row9 = dmap[(0.9, sd)]
print("V3 φ=0.9 keysig K =", r9["keysig"]["K"][:16], " disk φ=0.9 keysig K =", row9["keysig"]["K"][:16], " match=", r9["keysig"]["K"]==row9["keysig"]["K"])
print("V3 φ=0.9 cnt[:8] =", r9["counts"][:8])
print("disk φ=0.9 cnt[:8] =", row9["counts"][:8])
print("V3 φ=0.9 cnt_source =", r9["cnt_source"], " disk =", row9["cnt_source"])
print("V3 φ=0.9 diag =", r9.get("diag"), " disk diag =", row9.get("diag"))
