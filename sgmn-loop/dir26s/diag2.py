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
# φ=0 拿 cnt0
r0 = V3.run_unit_dose(sd, 0.0, tau, cnt0=None)
cnt0 = r0["counts"]
print("φ=0 cnt0[:8]=", cnt0[:8], "keysig K=", r0["keysig"]["K"][:12])
# φ=0.9 用 frozen cnt0
r9 = V3.run_unit_dose(sd, 0.9, tau, cnt0=cnt0)
row9 = dmap[(0.9, sd)]; row0 = dmap[(0.0, sd)]
for pk in ("Fu","Ku"):
    print(f"{pk}: rerun_mu={r9['arms'][pk]['mu']:.7f}  disk_mu={row9['arms'][pk]['mu']:.7f}  diff={abs(r9['arms'][pk]['mu']-row9['arms'][pk]['mu']):.7f}")
    print(f"     rerun_wsig={r9['arms'][pk]['wsig'][:16]}  disk_wsig={row9['arms'][pk]['wsig'][:16]}  cnt_src={row9['cnt_source']}")
print("φ=0 复现:")
for pk in ("Fu","Ku"):
    print(f"{pk}: rerun_mu={r0['arms'][pk]['mu']:.7f}  disk_mu={row0['arms'][pk]['mu']:.7f}  diff={abs(r0['arms'][pk]['mu']-row0['arms'][pk]['mu']):.7f}")
print("disk φ=0 keysig K=", row0["keysig"]["K"][:12], " φ=0.9 keysig K=", row9["keysig"]["K"][:12])
