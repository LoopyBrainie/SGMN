#!/opt/pytorch/bin/python
import json, sys
for _p in ("dir26s","dir26r","dir26o","dir26k","dir26j","dir26i","dir26h","dir26g","dir26f","dir26e","dir26c","dir26b","dir26","dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")
import numpy as np, torch
import R1_dose_ar1_v3 as V3
import probe_perround as PP
from R1_gating12 import cal_at, SEEDS
disk = json.load(open("/home/ec2-user/sgmn-loop/dir26s/out/DIR26S_rows.json"))["rows"]
dmap = {(r["phi"], r["seed"]): r for r in disk}
tau = cal_at(V3.N_KV); sd = SEEDS[0]
# φ=0 拿 cnt0
r0 = V3.run_unit_dose(sd, 0.0, tau, cnt0=None); cnt0 = r0["counts"]
print("cnt0[:6]=", cnt0[:6])
# run_one φ=0.9 with frozen cnt0
ro = PP.run_one(sd, 0.9, tau, cnt0)
print("run_one Fu mu =", ro["F"]["mu"], " Ku mu =", ro["K"]["mu"], " cnt[:6]=", ro["cnt"][:6])
print("disk φ=0.9 Fu mu =", dmap[(0.9,sd)]["arms"]["Fu"]["mu"], " Ku =", dmap[(0.9,sd)]["arms"]["Ku"]["mu"])
# V3 直接 φ=0.9 frozen
r9 = V3.run_unit_dose(sd, 0.9, tau, cnt0=cnt0)
print("V3 φ=0.9 Fu mu =", r9["arms"]["Fu"]["mu"], " Ku =", r9["arms"]["Ku"]["mu"])
