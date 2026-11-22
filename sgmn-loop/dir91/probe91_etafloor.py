#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""DIR-91 · rev 2.0 步②续：**残余地板 v.s. 步长的定律检验**（诊断）

`probe91_coexist.py` 的 G3 给出一个**未解释的数**：$n\ge4$ 时 rel 从 $E{=}10$ 到 $E{=}200$
**完全平坦**在 $0.0992\sim0.1078$，且几乎**与 $n$ 无关**；而单对 $n{=}1$ 收敛到 $\mathbf{0.0000}$。
同时 $\eta\alpha\lVert k\rVert^2=0.1099\approx0.104$ —— 二者近得可疑。

**预注册假设（运行前冻结，不得按结果改）**：
  $$ \mathrm{floor}_{\rm rel}\ \approx\ c\,\eta\,\alpha\,\lVert k\rVert^2 ,\qquad c=O(1) $$
  即残余地板**正比于每步收缩因子**，而不是容量或键病态所致。
  推论：$\eta$ 缩小 10 倍 ⇒ 地板缩小 10 倍（但需 ~10 倍 epoch 才到达稳态）。

**判读（预注册）**：
  - 若 $c$ 在跨 $\eta$（≥1 个数量级）与跨 $n$ **稳定在常数** ⇒ 假设成立，
    地板是**步长噪声底**，不是存储缺陷 ⇒ 存储容量与调度可解。
  - 若 $c$ 随 $\eta$ 单调漂移或量级跳变 ⇒ 假设被推翻，地板另有来源（须新假设）。

★ 只做测量；不产生裁决、不写证据行、不改任何已冻结阈值。
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, MOUNT, N_FACTS, build_chunks, build_facts, dump,
                       key_matrices, make_stack, model_and_tok)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GAMMA = 0.05
NS = [1, 8, 32]
ETAS = [3e-5, 1e-5, 3e-6, 1e-6]
EPOCH = 400


def rel(W, k, v):
    return float(((W @ k) - v).norm() / (v.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok); chunks, owner = build_chunks(tok, facts)
    st = make_stack(gamma=GAMMA, no_write=False, seed=a.seed)
    li = MOUNT[0]; m = st.mems[li]
    for mm in st.mems.values():
        mm.no_write = True
    K, V = key_matrices(model, st, facts, chunks, owner)
    Ks = [K[li][i].detach().float() for i in range(N_FACTS)]
    Vs = [V[li][i].detach().float() for i in range(N_FACTS)]
    W0 = m.W0.detach().clone()
    alpha, lam = float(m.alpha), float(m.lam)
    k2 = float((torch.stack(Ks[:32]).norm(dim=1) ** 2).mean())
    res = {"seed": a.seed, "alpha": alpha, "lam": lam, "ktwo": k2,
           "epoch": EPOCH, "ns": NS, "etas": ETAS}
    print(f"[设定] alpha={alpha} lam={lam} ||k||^2_mean={k2:.2f} epoch={EPOCH}", flush=True)
    res["rows"] = []
    for n in NS:
        for eta in ETAS:
            W = W0.clone()
            for _ in range(EPOCH):
                for i in range(n):
                    k, v = Ks[i], Vs[i]
                    G = alpha * torch.outer(W @ k - v, k) + lam * (W - W0)
                    W = W - eta * G
            rs = [rel(W, Ks[i], Vs[i]) for i in range(n)]
            fl = float(np.mean(rs))
            fac = eta * alpha * k2
            c = fl / fac if fac > 0 else float("nan")
            res["rows"].append({"n": n, "eta": eta, "floor": fl, "factor": fac, "c": c,
                                "rel_max": float(np.max(rs))})
            print(f"  n={n:<3} eta={eta:.0e}  floor={fl:.5f}  "
                  f"eta*alpha*||k||^2={fac:.5e}  c={c:.4f}", flush=True)
    # 线性检验：log(floor) v.s. log(eta)
    print("\n[定律检验] 对每个 n 做 log(floor) ~ log(eta) 斜率（预注册：≈1）", flush=True)
    res["slopes"] = {}
    for n in NS:
        rr = [(r["eta"], r["floor"]) for r in res["rows"] if r["n"] == n and r["floor"] > 1e-12]
        if len(rr) >= 2:
            x = np.log([p[0] for p in rr]); y = np.log([p[1] for p in rr])
            sl = float(np.polyfit(x, y, 1)[0])
            res["slopes"][n] = sl
            print(f"  n={n:<3} slope={sl:+.3f}", flush=True)
    st.remove()
    dump(res, "probe91_etafloor.json")
    print("\n★ 诊断完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
