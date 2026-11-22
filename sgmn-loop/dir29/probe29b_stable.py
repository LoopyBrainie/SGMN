"""DIR-29 · Stage A 诊断 2（**不产出任何判据读数**）：A29-4 对照量在**稳定区**的行为。

rev 1.0 的 A29-4 参数 (γ=0.5, η=0.3, T=3000) 落在**两臂都发散**的区间
（实测：full 臂 64/64 列在 t≈368 死亡、fac 臂更早）⇒ 雅可比溢出 ⇒ 对照量被污染。
本探针在**稳定区**（无死亡）找 A29-4 可执行的参数点。判据量一律不算。
"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir29")
import numpy as np
from R1_tangent import run


def full_qr_top(jac_seq, mask_seq, dim, t_dead):
    S_ = jac_seq[0].shape[0]
    q = np.zeros((S_, dim, dim)); q[..., np.arange(dim), np.arange(dim)] = 1.0
    acc = np.zeros((S_, dim))
    for t, Jt in enumerate(jac_seq):
        M = Jt @ q
        Q, R = np.linalg.qr(M)
        d = np.diagonal(R, axis1=-2, axis2=-1)
        m = mask_seq[t]
        acc += np.where(m[:, None], np.log(np.maximum(np.abs(d), 1e-300)), 0.0)
        q = np.where(m[:, None, None], Q, q)
    return np.max(acc / np.maximum(np.asarray(t_dead).reshape(-1), 1.0)[:, None], axis=1)


for eta4 in (0.02, 0.05, 0.1):
    for T4 in (3000, 10000):
        rf = run("full", [eta4], 0.5, t_steps=T4, record=True)
        df = full_qr_top(rf["jac_seq"], rf["mask_seq"], 2, rf["t_dead"])
        d_full = float(np.max(np.abs(df - np.maximum(rf["lam"][:, 0], rf["lam2"][:, 0]))))
        rfc = run("fac", [eta4], 0.5, t_steps=T4, record=True)
        dfc = full_qr_top(rfc["jac_seq"], rfc["mask_seq"], 4, rfc["t_dead"])
        blk = np.maximum(np.maximum(rfc["lam"][:, 0], rfc["lam2"][:, 0]),
                         np.maximum(rfc["lam3"][:, 0], rfc["lam4"][:, 0]))
        d_fac = float(np.max(np.abs(dfc - blk)))
        print(f"η={eta4:<5} T={T4:>6}: d_full={d_full:.6g}  d_fac={d_fac:.6g}  "
              f"lam1={float(np.median(rf['lam'])):+.6f} lam2={float(np.median(rf['lam2'])):+.6f}  "
              f"fac_top={float(np.median(rfc['lam_top'])):+.6f}  "
              f"trunc(full)={rf['n_trunc']} trunc(fac)={rfc['n_trunc']}", flush=True)
