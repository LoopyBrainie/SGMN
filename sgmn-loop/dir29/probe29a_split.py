"""DIR-29 · Stage A 诊断（**不产出任何判据读数**）：A29-4 的两块对照量随 T 的行为。

目的：区分「实现错」与「有限 T 的 QR 收敛残差」。
判据量（η_c / R / P1 / P2）本脚本**一律不计算**。
"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir29")
import numpy as np
from R1_tangent import run


def full_qr_top(jac_seq, mask_seq, dim):
    S_ = jac_seq[0].shape[0]
    q = np.zeros((S_, dim, dim)); q[..., np.arange(dim), np.arange(dim)] = 1.0
    acc = np.zeros((S_, dim))
    marr = np.stack(mask_seq)
    dead = ~marr
    has_dead = dead.any(axis=0)
    t_dead = np.where(has_dead, dead.argmax(axis=0) + 1, marr.shape[0])
    for t, Jt in enumerate(jac_seq):
        M = Jt @ q
        Q, R = np.linalg.qr(M)
        d = np.diagonal(R, axis1=-2, axis2=-1)
        m = mask_seq[t]
        acc += np.where(m[:, None], np.log(np.maximum(np.abs(d), 1e-300)), 0.0)
        q = np.where(m[:, None, None], Q, q)
    return np.max(acc / np.maximum(t_dead, 1.0)[:, None], axis=1), t_dead


eta4, gamma4 = 0.3, 0.5
for T4 in (3000, 10000, 30000):
    rf = run("full", [eta4], gamma4, t_steps=T4, record=True)
    df, td = full_qr_top(rf["jac_seq"], rf["mask_seq"], 2)
    d_full = float(np.max(np.abs(df - np.maximum(rf["lam"][:, 0], rf["lam2"][:, 0]))))
    rfc = run("fac", [eta4], gamma4, t_steps=T4, record=True)
    dfc, _ = full_qr_top(rfc["jac_seq"], rfc["mask_seq"], 4)
    blk = np.maximum(np.maximum(rfc["lam"][:, 0], rfc["lam2"][:, 0]),
                     np.maximum(rfc["lam3"][:, 0], rfc["lam4"][:, 0]))
    d_fac = float(np.max(np.abs(dfc - blk)))
    print(f"T={T4:>6}: d_full={d_full:.6g}  d_fac={d_fac:.6g}  "
          f"lam1={float(np.median(rf['lam'])):+.6f} lam2={float(np.median(rf['lam2'])):+.6f}  "
          f"fac_top={float(np.median(rfc['lam_top'])):+.4f}  "
          f"t_dead_med(full)={float(np.median(td)):.0f}  n_trunc(full)={rf['n_trunc']}  "
          f"n_trunc(fac)={rfc['n_trunc']}", flush=True)
