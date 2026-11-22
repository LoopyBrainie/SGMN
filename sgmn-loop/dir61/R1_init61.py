r"""DIR-61 正式轮 · 初始化陷阱集（d=4, r=2, N_trap=2000，float64）

逐字派生 dir61/workex61b.py（装置不变，只把"臂"做成 §2.2 的 2×2 因子设计）：
  k = e1+e2, v = e3+e4, lam=0.05, eta=0.05
  G = (Wk - v) k^T + lam (W - W0);  U <- U - eta G V;  V <- V - eta G^T U;  W = U V^T
判据量：max_t ||W_t - W0||_F 与 max_t |J_t - J0|。
"""
import argparse
import json
import os

import numpy as np

D, R, LAM, ETA, N = 4, 2, 0.05, 0.05, 2000
K = np.zeros((D, 1)); K[0, 0] = K[1, 0] = 1.0
V = np.zeros((D, 1)); V[2, 0] = V[3, 0] = 1.0
KH = K / np.linalg.norm(K); VH = V / np.linalg.norm(V)
PK = np.zeros((D, 2)); PK[2, 0] = PK[3, 1] = 1.0      # 列 ⊥ k（span{e3,e4}）
PV = np.zeros((D, 2)); PV[0, 0] = PV[1, 1] = 1.0      # 列 ⊥ v（span{e1,e2}）
SEED = 20261117


def J(W, W0):
    res = W @ K - V
    return 0.5 * float(np.sum(res * res)) + 0.5 * LAM * float(np.sum((W - W0) ** 2))


def arms(seed=SEED):
    rng = np.random.default_rng(seed)
    A = lambda: rng.normal(size=(2, 2)) + 2.0 * np.eye(2)      # 可逆、非退化
    out = {}
    out["ZERO"] = (np.zeros((D, 2)), np.zeros((D, 2)), "frozen")
    out["ORTHO"] = (PV @ A(), PK @ A(), "frozen")
    out["VNULL"] = (PV @ A() + 0.6 * VH @ np.ones((1, 2)), PK @ A(), "move")
    out["UNULL"] = (PV @ A(), PK @ A() + 0.6 * KH @ np.ones((1, 2)), "move")
    c1 = rng.normal(size=(D, 1)); c1 /= np.linalg.norm(c1)
    c2 = rng.normal(size=(D, 1)); c2 /= np.linalg.norm(c2)
    out["DEADBLK"] = (np.hstack([PV @ A()[:, :1], c1]), np.hstack([PK @ A()[:, :1], c2]), "move")
    out["RAND"] = (rng.normal(size=(D, 2)), rng.normal(size=(D, 2)), "move")
    return out


def step(U, Vm, W0, sync=True, eta=ETA):
    W = U @ Vm.T
    G = (W @ K - V) @ K.T + LAM * (W - W0)
    if sync:
        return U - eta * (G @ Vm), Vm - eta * (G.T @ U)
    U2 = U - eta * (G @ Vm)
    W2 = U2 @ Vm.T
    G2 = (W2 @ K - V) @ K.T + LAM * (W2 - W0)
    return U2, Vm - eta * (G2.T @ U2)


def run(U0, V0, n=N, sync=True, eta=ETA):
    W0 = U0 @ V0.T
    U, Vm = U0.copy(), V0.copy()
    j0 = J(W0, W0)
    dmax = 0.0; jmaxdev = 0.0; trace = []
    for t in range(1, n + 1):
        U, Vm = step(U, Vm, W0, sync=sync, eta=eta)
        Wn = U @ Vm.T
        dmax = max(dmax, float(np.linalg.norm(Wn - W0)))
        jmaxdev = max(jmaxdev, abs(J(Wn, W0) - j0))
        if t in (1, 10, 100, 500, n):
            trace.append([t, float(np.linalg.norm(Wn - W0)), J(Wn, W0)])
    Wn = U @ Vm.T
    return {"dW_max": dmax, "dW_N": float(np.linalg.norm(Wn - W0)), "J0": j0,
            "J_N": J(Wn, W0), "J_max_dev": jmaxdev, "trace": trace, "W0": W0, "W_N": Wn}


def run_full(W0, n=N, eta=ETA):
    W = W0.copy(); dmax = 0.0; j0 = J(W0, W0)
    for _ in range(n):
        G = (W @ K - V) @ K.T + LAM * (W - W0)
        W = W - eta * G
        dmax = max(dmax, float(np.linalg.norm(W - W0)))
    return {"dW_max": dmax, "J0": j0, "J_N": J(W, W0)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=""); ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    res = {"seed": a.seed, "d": D, "r": R, "lam": LAM, "eta": ETA, "N": N, "arms": {}}
    A = arms(a.seed)
    for name, (U0, V0, pred) in A.items():
        r = run(U0, V0)
        rr = {k: v for k, v in r.items() if k not in ("W0", "W_N")}
        rr.update({"pred": pred,
                   "kTV": float(np.abs(K.T @ V0).max()),
                   "UTv": float(np.abs(U0.T @ V).max()),
                   "full": run_full(U0 @ V0.T)})
        # 复活诊断（DEADBLK）：死列方向 D0 = (PV a1)(PK b1)^T 上的分量变化
        if name == "DEADBLK":
            u1 = U0[:, 0:1]; v1 = V0[:, 0:1]
            D0 = u1 @ v1.T; D0 /= np.linalg.norm(D0)
            rr["revival_deadcomp_0"] = float(abs(np.sum((U0 @ V0.T) * D0)))
            rr["revival_deadcomp_N"] = float(abs(np.sum(r["W_N"] * D0)))
        res["arms"][name] = rr
        print("%-8s pred=%-6s |kT V|=%.2e |UT v|=%.2e  dW_max=%.3e dW_N=%.3e  J %.4f->%.4f  full_dW=%.3e%s"
              % (name, pred, rr["kTV"], rr["UTv"], rr["dW_max"], rr["dW_N"], rr["J0"], rr["J_N"],
                 rr["full"]["dW_max"],
                 ("  revival %.2e->%.2e" % (rr["revival_deadcomp_0"], rr["revival_deadcomp_N"]))
                 if name == "DEADBLK" else ""))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("→", a.out)
