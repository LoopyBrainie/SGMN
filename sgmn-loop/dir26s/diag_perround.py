#!/opt/pytorch/bin/python
import json, sys
for _p in ("dir26s","dir26r","dir26o","dir26k","dir26j","dir26i","dir26h","dir26g",
           "dir26f","dir26e","dir26c","dir26b","dir26","dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")
import numpy as np, torch
from R1_gating import D_MODEL, M, drive, make_core, unit
from R1_gating2 import item_counts
from R1_gating5 import theta_of
from R1_gating7 import build_keep, realized_profile
from R1_gating12 import N_READ, R0, R_HO8, R_PRES, cal_at, read_stream, wsig, ret_jitter_margin
import R1_ar1 as RA
N_KV = 60
def rerun(seed, phi, pk):
    D = RA.build_stream_ar1(N_KV, seed, D_MODEL, int(N_KV), R_PRES, theta_of(0.9), phi)
    RS8 = read_stream(seed, R_HO8); vb = D["vb"][:N_READ]
    _t = cal_at(N_KV); tau = float(_t["c_tau"]) if isinstance(_t, dict) else float(_t)
    cB = make_core(D_MODEL, seed, R0); kB, logB = drive(cB, D, "B", tau=tau)
    cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, N_KV)
    keep, _ = build_keep(N_KV, R_PRES, cnt, __import__("R1_gating12", fromlist=["PROFILE"]).PROFILE[pk])
    c = make_core(D_MODEL, seed, R0); drive(c, D, "R", keep=keep)
    W = c._W().detach(); P = W - c.W0
    m = ret_jitter_margin(P, RS8["K"], vb, N_READ, R_HO8)
    return m["mu"], wsig(W), m["mu_sd"]
disk = json.load(open("/home/ec2-user/sgmn-loop/dir26s/out/DIR26S_rows.json"))["rows"]
dmap = {(r["phi"], r["seed"]): r for r in disk}
for seed in (20261028, 20261104):
    for phi in (0.0, 0.9):
        for pk in ("F", "K"):
            mu_r, ws, sd = rerun(seed, phi, pk)
            mu_d = dmap[(phi, seed)]["arms"][pk + "u"]["mu"]
            print(f"seed={seed} phi={phi} {pk}u: rerun_mu={mu_r:.7f} disk_mu={mu_d:.7f} diff={abs(mu_r-mu_d):.7f} wsig={ws} sd={sd:.5f}")
