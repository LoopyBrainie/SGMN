r"""DIR-62 正式轮装置（卡 rev 1.0 逐字实现）

DeviceW: W = U V^T           （实现形，§3.3；秩预算压在 W 上）
DeviceD: W = W^0 + U V^T     （ΔW 形，§5.1；秩预算压在 ΔW 上）
步：G = (Wk - v)k^T + λ(W - W^0);  U <- U - η G V;  V <- V - η G^T U
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref62 import D, LAM, k, v, preload, J, closed, svd_r

ETA, N = 0.05, 200000
SEEDS = [20261118 + i for i in range(5)]
E = lambda i: np.eye(D)[:, i:i + 1]
C = k @ k.T
RSYM = np.linalg.cholesky(C + LAM * np.eye(D))
REG = {"WEAK": [(1.40, E(2), E(0)), (0.50, E(3), E(1)), (0.15, E(4), E(4))],
       "STRONG": [(8.00, E(2), E(0)), (6.00, E(3), E(1)), (1.00, E(4), E(4))]}


def run(U, V, W0, r, device, seed, n=N, eta=ETA):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(D, r)) * 0.3
    V = rng.normal(size=(D, r)) * 0.3
    for _ in range(n):
        W = U @ V.T + (W0 if device == "D" else 0.0)
        G = (W @ k - v) @ k.T + LAM * (W - W0)
        U = U - eta * (G @ V)
        V = V - eta * (G.T @ U)
    return U @ V.T + (W0 if device == "D" else 0.0)


def gains(W, pairs):
    g = [float((W @ kj).T @ vj) / float(vj.T @ vj) for (_, kj, vj) in pairs]
    gd = float((W @ k).T @ v) / float(v.T @ v)
    return g, gd


def ref_reads(W0, r):
    Ws = closed(W0)
    Wr = svd_r(Ws @ RSYM, r) @ np.linalg.inv(RSYM)
    return Ws, Wr


def arm_rank(tag, r, device):
    pairs = REG[tag]
    W0 = preload(pairs)
    Ws, Wr = ref_reads(W0, r)
    # device D：参照是 §5.1 的 ΔW 最优（=W*，因为 rank M̄=1≤r）
    Wref = Wr if device == "W" else Ws
    Jref = J(Wref, W0)
    gref, gdref = gains(Wref, pairs)
    out = {"tag": tag, "r": r, "device": device, "Jref": Jref, "gref": gref, "gdref": gdref, "seeds": []}
    for s in SEEDS:
        W = run(None, None, W0, r, device, s)
        g, gd = gains(W, pairs)
        out["seeds"].append({"seed": s, "J": J(W, W0), "rel": abs(J(W, W0) - Jref) / Jref,
                             "gains": g, "gdata": gd,
                             "dgmax": float(max(abs(a - b) for a, b in zip(g, gref))),
                             "dgdata": abs(gd - gdref)})
    return out


def arm_dilution():
    vP = E(0)
    W0 = preload([(1.0, k, vP)])
    Jref = J(closed(W0), W0)
    out = {"tag": "DILUTE", "r": 2, "device": "W", "Jref": Jref, "seeds": []}
    for s in SEEDS:
        W = run(None, None, W0, 2, "W", s)
        read = W @ k
        cd = float(read.T @ v) / float(v.T @ v)
        cp = float(read.T @ vP) / float(vP.T @ vP)
        out["seeds"].append({"seed": s, "J": J(W, W0), "rel": abs(J(W, W0) - Jref) / Jref,
                             "c_data": cd, "c_P": cp,
                             "d_data": abs(cd - 2 / 2.05), "d_P": abs(cp - 0.05 / 2.05)})
    return out


def arm_sensitivity():
    """A62-6：把 γ1 的键改成 k_P=e3+1e-2 e1（不再正交）⇒ 存活律必须失效"""
    kP = E(2) + 1e-2 * E(0)
    pairs = [(1.40, kP, E(0)), (0.50, E(3), E(1)), (0.15, E(4), E(4))]
    W0 = preload(pairs)
    Ws = closed(W0)
    W4 = svd_r(Ws @ RSYM, 4) @ np.linalg.inv(RSYM)
    g, gd = gains(W4, pairs)
    return {"g_ref": g[0], "gamma1": 1.40, "dev": abs(g[0] - 1.40), "gains": g, "gdata": gd}


def arm_full(tag):
    """r=d 全矩阵（无秩约束）参照臂，用于核对 ① 的闭式"""
    pairs = REG[tag]
    W0 = preload(pairs)
    out = {"tag": tag, "r": D, "device": "W(full)", "Jref": J(closed(W0), W0), "seeds": []}
    for s in SEEDS:
        W = run(None, None, W0, D, "W", s)
        g, gd = gains(W, pairs)
        out["seeds"].append({"seed": s, "J": J(W, W0), "rel": abs(J(W, W0) - out["Jref"]) / out["Jref"],
                             "gains": g, "gdata": gd})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {"eta": ETA, "N": N, "lam": LAM, "seeds": SEEDS, "arms": []}
    if a.what in ("all", "ranks"):
        for tag in ("WEAK", "STRONG"):
            for r in (1, 2, 3, 4):
                res["arms"].append(arm_rank(tag, r, "W"))
            res["arms"].append(arm_rank(tag, 1, "D"))
        for tag in ("WEAK", "STRONG"):
            res["arms"].append(arm_full(tag))
    if a.what in ("all", "dilute"):
        res["arms"].append(arm_dilution())
    if a.what in ("all", "sens"):
        res["sens"] = arm_sensitivity()
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    for x in res["arms"]:
        if "seeds" not in x:
            print("SENS g1=%.6f dev=%.3e" % (x["g_ref"], x["dev"]))
            continue
        rels = [s["rel"] for s in x["seeds"]]
        print("%-7s r=%d %s | Jref=%.10f rel_max=%.1e | Δg_max=%.1e | gdata=%s"
              % (x["tag"], x["r"], x["device"], x["Jref"], max(rels),
                 max([s.get("dgmax", s.get("d_data", 0)) for s in x["seeds"]]),
                 "%.6f" % np.mean([s.get("gdata", s.get("c_data", 0)) for s in x["seeds"]])))
    if a.out:
        print("→", a.out)
