"""DIR-72 正式轮：五格求谱 ⇒ η*_pred（因子空间 / 矩阵空间），与 D-92 的二分值比对。
用法: R5_spec72.py <mode> <gamma> <eta_meas>
"""
import sys, os, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, ref70, step72, spec72
mode, gamma, eta_meas = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
eta_f = 1e-3
OUT = "/home/ec2-user/sgmn-loop/dir72/out/grid"
os.makedirs(OUT, exist_ok=True)
fp = step72.fp_factors(mode, gamma, eta_warm=0.3 * eta_meas)
xs = spec72.pack(fp["U1"], fp["V1"], fp["U2"], fp["V2"])
Jfac = spec72.jac_factor(xs, mode, gamma, eta_f)
eta_pred, dg = spec72.bound(Jfac, eta_f)
vb = spec72.verify_bound(Jfac, eta_f, eta_pred)
p = ref70.pred_target(6.0, gamma) if mode == "target" else ref70.pred_keys(6.0, gamma)
W1s, W2s = p["u1"] * ref70.P, p["u2"] * ref70.P
Jfld = spec72.jac_field(W1s, W2s, mode, gamma, eta_f)
eta_pred_mat, dgm = spec72.bound(Jfld, eta_f)
rec = dict(mode=mode, gamma=gamma, eta_meas=eta_meas, eta_pred_factor=eta_pred, eta_pred_matrix=eta_pred_mat,
           ratio_factor=eta_meas / eta_pred if np.isfinite(eta_pred) else None,
           ratio_matrix=eta_meas / eta_pred_mat if np.isfinite(eta_pred_mat) else None,
           fp_res=fp["res"], fp_steps=fp["steps"], zeros=dg["zeros"], n_neg=dg["n_neg"],
           lmax_decay_factor=dg["lmax_decay"], top5=[(float(b), str(l)) for b, l in dg["top5"]],
           verify=vb, field_zeros=dgm["zeros"], field_lmax_decay=dgm["lmax_decay"])
json.dump(rec, open(os.path.join(OUT, "spec_%s_g%g.json" % (mode, gamma)), "w"), indent=1)
print("DONE spec %s γ=%g η_meas=%.6g | factor pred=%.6g ratio=%.4f (zeros=%d, verify=%s) | matrix pred=%.6g ratio=%.4f"
      % (mode, gamma, eta_meas, eta_pred, eta_meas / eta_pred, dg["zeros"], vb["ok"], eta_pred_mat, eta_meas / eta_pred_mat), flush=True)
