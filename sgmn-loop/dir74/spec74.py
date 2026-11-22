"""DIR-74 量测层：线性化放大曲线。
派生：`dir73/near73.py`（`perturb`/`freeze_fp` 原样调用）+ `dir72/spec72.py`（`pack`/`unpack`/`jac_factor` 原样调用）+ `dir70/ref70.py`（`P`/`pred_target`）。
**唯一新增**：解析读出 R 与增益曲线 g(t)/σ_max(t)。
"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, step72, spec72, ref70
P = ref70.P; D, Rr = 6, 2

def _slots(n, shape):
    """第 n 个因子在 48 维向量里的切片。"""
    step_ = D * Rr
    return slice(n * step_, (n + 1) * step_)

def readout(xs):
    """R: 72×48，vec(W1,W2) 对 x 的一阶解析读出。"""
    U1, V1, U2, V2 = xs
    Rm = np.zeros((72, 48))
    for j in range(48):
        e = np.zeros(48); e[j] = 1.0
        dU1 = e[_slots(0, None)].reshape(D, Rr); dV1 = e[_slots(1, None)].reshape(D, Rr)
        dU2 = e[_slots(2, None)].reshape(D, Rr); dV2 = e[_slots(3, None)].reshape(D, Rr)
        Rm[:36, j] = (dU1 @ V1.T + U1 @ dV1.T).ravel()
        Rm[36:, j] = (dU2 @ V2.T + U2 @ dV2.T).ravel()
    return Rm

def whiten(Rm, rtol=1e-10):
    """右逆白化 W(48×r)，R W = I_r（取数值秩）。"""
    U, S, Vt = np.linalg.svd(Rm, full_matrices=False)
    r = int(np.sum(S > rtol * S[0]))
    return Vt[:r].T / S[:r], r

def delta0(xstar, delta=1e-3, seed=near73.SD):
    x0, meas = near73.perturb(xstar, delta, seed)
    return np.concatenate([X.ravel() for X in x0]) - np.concatenate([X.ravel() for X in xstar]), meas

def curves(xs, mode, gamma, eta, d0, nmax=200000, log_every=1.15, want_G=True, Rw=None):
    """返回 g(t)/包络曲线与峰值。g(t)=max_层‖R_l M_t δ0‖ / max_层‖R_l δ0‖（D-94 的 per-layer max 语义）。"""
    Rm = readout(xs)
    Den = max(np.linalg.norm(Rm[:36] @ d0), np.linalg.norm(Rm[36:] @ d0))
    M = spec72.jac_factor(spec72.pack(*xs), mode, gamma, eta)
    M_t = np.eye(48); best = dict(g=0.0, t=None, absmax=0.0, G=0.0, tG=None); out = []
    nxt_log = 1.0
    for t in range(1, nmax + 1):
        M_t = M @ M_t
        Rm_t = Rm @ M_t
        a1 = np.linalg.norm(Rm_t[:36] @ d0); a2 = np.linalg.norm(Rm_t[36:] @ d0)
        g = max(a1, a2) / Den
        if g > best["g"]:
            best.update(g=float(g), t=int(t), absmax=float(max(a1, a2)))
        if want_G:
            G = float(np.linalg.svd(Rm_t @ Rw, compute_uv=False)[0])
            if G > best["G"]:
                best.update(G=G, tG=int(t))
        if t >= nxt_log:
            out.append((int(t), float(g)))
            nxt_log *= log_every
    best.update(Den=float(Den), g_req=float(1e3 / Den), n_eval=int(nmax))
    return best, out
