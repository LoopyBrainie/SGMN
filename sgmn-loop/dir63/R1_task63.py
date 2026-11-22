r"""DIR-63 正式轮装置：两层线性堆叠（rev 1.0 §3 逐字实现）

层0：W0=U0V0^T，键 k，目标 v0=P0 k
层1：W1=U1V1^T，键 h=sg(W0 k)，目标 v1=P k
步：G_l=(W_l k_l - v_l) k_l^T + λ W_l ; U_l <- U_l - η G_l V_l ; V_l <- V_l - η G_l^T U_l
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref63 import (D, LAM, ETA, N, SEEDS, THETAS, KEYS, P, P0, v0, v1,
                   J_layer, Jd_analytic, Jc_analytic, generic)


def run_stack(th, seed, n=N, eta=ETA, r=2, track=False):
    """rev 1.1 批梯度：每步 6 键平均；h_i = sg(W0 k_i) 用当前 W0"""
    rng = np.random.default_rng(seed)
    U0 = rng.normal(size=(D, r)) * 0.3; V0 = rng.normal(size=(D, r)) * 0.3
    U1 = rng.normal(size=(D, r)) * 0.3; V1 = rng.normal(size=(D, r)) * 0.3
    tgt0 = [v0(k, th) for k in KEYS]; tgt1 = [v1(k) for k in KEYS]
    C0 = sum(k @ k.T for k in KEYS) / len(KEYS)
    B0 = sum(tgt0[i] @ KEYS[i].T for i in range(D)) / len(KEYS)
    first = None
    for t in range(n):
        W0 = U0 @ V0.T
        G0 = (W0 @ C0 - B0) + LAM * W0
        U0 = U0 - eta * (G0 @ V0); V0 = V0 - eta * (G0.T @ U0)
        W0 = U0 @ V0.T
        hs = [W0 @ k for k in KEYS]
        C1 = sum(h @ h.T for h in hs) / len(hs)
        B1 = sum(tgt1[i] @ hs[i].T for i in range(D)) / len(hs)
        W1 = U1 @ V1.T
        G1 = (W1 @ C1 - B1) + LAM * W1
        U1 = U1 - eta * (G1 @ V1); V1 = V1 - eta * (G1.T @ U1)
        if track and t % 2000 == 0 and t > 0:
            j1 = J_layer(U1 @ V1.T, [W0 @ kk for kk in KEYS], tgt1)
            if first is None and abs(j1 - Jc_analytic(th)) / Jc_analytic(th) <= 1e-6:
                first = t
    W0 = U0 @ V0.T; W1 = U1 @ V1.T
    return dict(W0=W0, W1=W1, J0=J_layer(W0, KEYS, tgt0),
                J1=J_layer(W1, [W0 @ kk for kk in KEYS], tgt1), first=first)


def run_direct(seed, n=N, eta=ETA, r=2):
    """rev 1.1 批梯度：单层（直连臂），C_x=I、B=P"""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(D, r)) * 0.3; V = rng.normal(size=(D, r)) * 0.3
    tgt = [v1(k) for k in KEYS]
    C = sum(k @ k.T for k in KEYS) / len(KEYS)
    B = sum(tgt[i] @ KEYS[i].T for i in range(D)) / len(KEYS)
    for t in range(n):
        W = U @ V.T
        G = (W @ C - B) + LAM * W
        U = U - eta * (G @ V); V = V - eta * (G.T @ U)
    return dict(J=J_layer(U @ V.T, KEYS, tgt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dir63/out/DIR63_formal.json")
    ap.add_argument("--n", type=int, default=N)
    a = ap.parse_args()
    res = {"lam": LAM, "eta": ETA, "N": a.n, "seeds": SEEDS, "thetas_deg": THETAS,
           "Jd_analytic": Jd_analytic(), "stack": [], "direct": []}
    for thd in THETAS:
        th = np.deg2rad(thd)
        g = generic(th)
        row = {"theta_deg": thd, "Jc_analytic": Jc_analytic(th), "Jd_analytic": Jd_analytic(),
               "J0_analytic": LAM / (1 + LAM), "rank_W0": g["r0"], "rank_W1": g["r1"],
               "delta_analytic": Jc_analytic(th) - Jd_analytic(), "seeds": []}
        for s in SEEDS:
            o = run_stack(th, s, n=a.n, track=True)
            row["seeds"].append({"seed": s, "J0": o["J0"], "J1": o["J1"], "first_step": o["first"],
                                 "rel0": abs(o["J0"] - LAM / (1 + LAM)) / (LAM / (1 + LAM)),
                                 "rel1": abs(o["J1"] - Jc_analytic(th)) / Jc_analytic(th),
                                 "delta": o["J1"] - Jd_analytic(),
                                 "rank_W0": int(np.linalg.matrix_rank(o["W0"], tol=1e-9)),
                                 "rank_W1": int(np.linalg.matrix_rank(o["W1"], tol=1e-9))})
        res["stack"].append(row)
        print("θ=%2.0f°  J1c=%.9f (analytic %.9f)  Δ=%.6f  ranks=%d,%d"
              % (thd, np.mean([x["J1"] for x in row["seeds"]]), row["Jc_analytic"],
                 row["delta_analytic"], row["rank_W0"], row["rank_W1"]))
    for s in SEEDS:
        o = run_direct(s, n=a.n)
        res["direct"].append({"seed": s, "J": o["J"], "rel": abs(o["J"] - Jd_analytic()) / Jd_analytic()})
    print("直连 J_d = %.9f (analytic %.9f)" % (np.mean([x["J"] for x in res["direct"]]), Jd_analytic()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("→", a.out)
