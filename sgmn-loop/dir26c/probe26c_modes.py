"""DIR-26-C · probe26c_modes.py —— 侦察探针：三种读键模式下的量级（**不是正式轮，不出判据**）

卡片尚未冻结；本探针只回答一件事：**读键换成"跨情节"的抖动实例键（$k_{i,\\rho^\\star}$，$\rho^\\star$ 未参与写）后，
三臂（A/B/P）之间还剩多少量程？** 用来决定 §4(G) 量程闸门与判据格点是否能算。

三种读键（写侧一字不改）：
  `clean`  $q=k^{\\rm proto}_i=\\sqrt5\\,e_i$ —— D-33/D-34 口径（**读键=写键坐标**的对齐假设）
  `Jin`    $q=k_{i,\\rho}$，$\\rho\\in[0,R)$ —— **参与过写**的实例键（同情节查询）
  `Jout` ★ $q=k_{i,\\rho^\\star}$ —— **未参与写**的全新抖动（跨情节查询；框架 §7.2「查询坐标对齐方式」）
聚合：`mean`（逐读命中率）与 `maj`（$R$ 次读里多数票命中，$R{=}4$ 取 $\\ge3$；$R{=}2$ 取 $2$）。

用法：`--d 16 --n 8 --r0 2`（工作算例）｜`--d 896 --n 35 --r0 8`（量级侦察）
"""
import argparse, json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26b")

import R1_gating as G
from R1_gating import build_stream, drive, make_core, metrics_plas, unit
from R1_gating2 import keep_positions

HO_OFFSET = 999                      # 跨情节查询流的种子偏移（同一批原型、全新抖动）
R_PRES = 4
R_HO = 4
THETA = math.radians(25.8419)


def jitter_metrics(W, W0, Kq, vb, n, r_q, thresh=None):
    """读键 $q=k_{i,\\rho^\\star}$：逐 (i,$\\rho$) 命中 + 两种聚合。$Kq$ 形状 $(r_q,n,d)$。"""
    P = W - W0
    hit = np.zeros((r_q, n), dtype=bool)
    for rho in range(r_q):
        Y = P @ Kq[rho].T                                     # (d, n)
        S = unit(Y.T) @ unit(vb).T                            # S[i,j] = cos(P q_i, v_j)
        for i in range(n):
            hit[rho, i] = int(torch.argmax(S[i]).item()) == i
    maj_n = r_q // 2 + 1 if r_q % 2 == 0 else (r_q + 1) // 2
    return {"ret1_mean": float(hit.mean()),
            "ret1_maj": float(np.mean([hit[:, i].sum() >= maj_n for i in range(n)])),
            "maj_need": int(maj_n),
            "per_item_mean": [float(x) for x in hit.mean(0)]}


def run(d, n, r0, seed, rho=0.4, tag=""):
    D = build_stream(seed, d, n, R_PRES, THETA)
    DH = build_stream(seed + HO_OFFSET, d, n, R_HO, THETA)     # ★ 跨情节查询键（同原型 E、全新 u）
    assert float(torch.allclose(D["proto"], DH["proto"]))
    tau = float(np.quantile([e["cos"] for e in drive(make_core(d, seed, r0), D, "A")[1]], 1 - rho))
    out = {"tag": tag, "d": d, "n": n, "r0": r0, "seed": seed, "rho": rho, "c_tau": tau, "arms": {}}
    tgt = {"A": ("A", None), "B": ("B", tau)}
    for arm, (mode, kw) in tgt.items():
        c = make_core(d, seed, r0)
        kwd = {"tau": kw} if kw is not None else {}
        k, _ = drive(c, D, mode, **kwd)
        W = c._W().detach()
        m = metrics_plas(W, c.W0, D["proto"], D["vb"], n)
        out["arms"][arm] = {"issued": k, "ret1_clean": m["ret1"], "rowm": m["rowm"],
                            "Jin": jitter_metrics(W, c.W0, D["K"], D["vb"], n, R_PRES),
                            "Jout": jitter_metrics(W, c.W0, DH["K"], D["vb"], n, R_HO)}
    # P 臂（等发放数，content-blind）
    kB = out["arms"]["B"]["issued"]
    keepP = keep_positions(n, R_PRES, kB)
    cP = make_core(d, seed, r0)
    kP, _ = drive(cP, D, "R", keep=keepP)
    W = cP._W().detach()
    m = metrics_plas(W, cP.W0, D["proto"], D["vb"], n)
    out["arms"]["P"] = {"issued": kP, "ret1_clean": m["ret1"], "rowm": m["rowm"],
                        "Jin": jitter_metrics(W, cP.W0, D["K"], D["vb"], n, R_PRES),
                        "Jout": jitter_metrics(W, cP.W0, DH["K"], D["vb"], n, R_HO)}
    T = R_PRES * n * 2 * 2
    print(f"\n[{tag}] d={d} N={n} r={r0} seed={seed} ρ={rho} c_τ={tau:+.4f} T={T}")
    for arm in ("A", "B", "P"):
        a = out["arms"][arm]
        print(f"  {arm} k={a['issued']:3d} | clean {a['ret1_clean']:.4f} | "
              f"Jin mean {a['Jin']['ret1_mean']:.4f} maj {a['Jin']['ret1_maj']:.4f} | "
              f"Jout mean {a['Jout']['ret1_mean']:.4f} maj {a['Jout']['ret1_maj']:.4f}")
    for md, key in (("clean", "ret1_clean"), ("Jin_maj", None), ("Jout_maj", None)):
        if key:
            v = {a: out["arms"][a][key] for a in ("A", "B", "P")}
        else:
            k2 = "ret1_maj"
            v = {a: out["arms"][a][md.split("_")[0]][k2] for a in ("A", "B", "P")}
        rng = max(v.values()) - min(v.values())
        print(f"  Δ_BP({md}) = {v['B'] - v['P']:+.4f} | 量程 {rng:.4f} {'★NO-RANGE' if rng < 0.20 else ''}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--r0", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"meta": {"d": a.d, "n": a.n, "seed": a.seed, "R_pres": R_PRES, "R_ho": R_HO,
                    "theta_deg": round(math.degrees(THETA), 4), "ho_offset": HO_OFFSET,
                    "note": "侦察探针（非正式轮）"}, "cells": []}
    r0s = [a.r0] if a.r0 else ([2, 4] if a.d == 16 else [6, 8, 12])
    for r0 in r0s:
        res["cells"].append(run(a.d, a.n, r0, a.seed, tag=f"d{a.d}"))
    outp = a.out or f"/home/ec2-user/sgmn-loop/dir26c/out/probe26c_d{a.d}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}")


if __name__ == "__main__":
    main()
