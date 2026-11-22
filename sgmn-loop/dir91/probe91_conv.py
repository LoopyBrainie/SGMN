#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""DIR-91 · rev 2.0 步②终：**0.10 是「真地板」还是「病态慢收敛」？**（诊断）

已测事实（`probe91_lsfloor.py` / `probe91_etafloor.py`）：
  * 键两两余弦 $0.9973$、有效秩 $\mathbf{1.03/32}$；值同样 $\mathbf{1.02/32}$；
  * $v_i$ 不能由 $\mathrm{span}\{k_j\}$ 解释的部分 $0.9821$，但 $\lVert v_i-\bar v\rVert/\lVert v_i\rVert=\mathbf{0.0481}$；
  * 正则最小二乘**闭式解**地板 $0.00080$（$n{=}32$）—— 极小；
  * SGD 地板 $\approx0.10$，且**与 $\eta$ 无关**（斜率 $-0.035/-0.047$）。

**假设（运行前冻结）**：$0.10$ 不是真地板，而是**病态慢收敛**——
判别性信息住在 Gram 的小特征值子空间，收缩率 $\eta\alpha\lambda_i$ 极小，
故在固定 epoch 数下看起来像地板；$\eta$ 越小同样越慢 ⇒ 解释了 $\eta$ 无关性。

**判读（预注册）**：
  - 若 $E$ 增 25×（400 → 10000）时 rel **持续下降**（$\to<0.02$）⇒ 假设成立，
    $0.10$ 是**伪地板**；修复方向是**预条件/键白化/归一化**，不是容量。
  - 若 rel 在 $E$ 上**真正停住**（$|\Delta|<0.01$ 跨 10×）⇒ 真地板，与步长无关，
    须另找结构原因（$\lambda$ 平衡或 $v$ 侧的均值塌缩）。

★ 只做测量；不产生裁决、不写证据行、不改任何已冻结阈值。
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, MOUNT, N_FACTS, build_chunks, build_facts, dump,
                       key_matrices, make_stack, model_and_tok)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GAMMA = 0.05
PLAN = [(8, [400, 2000, 5000, 10000]), (32, [400, 2000, 5000])]
ETA = 3e-5


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
    Ks = torch.stack([K[li][i].detach().float() for i in range(N_FACTS)])
    Vs = torch.stack([V[li][i].detach().float() for i in range(N_FACTS)])
    W0 = m.W0.detach().float().clone()
    alpha, lam = float(m.alpha), float(m.lam)
    res = {"seed": a.seed, "eta": ETA, "alpha": alpha, "lam": lam, "plan": PLAN}

    # Gram 谱（解释收缩率）
    G32 = alpha * (Ks.T @ Ks)
    ev = torch.linalg.eigvalsh(G32).clamp(min=0).flip(0)      # 降序
    res["spectrum_top"] = [float(x) for x in ev[:8]]
    res["spectrum_tail"] = [float(x) for x in ev[-6:]]
    print(f"[Gram] alpha*K^T K 前 8 特征值: {[f'{float(x):.3e}' for x in ev[:8]]}", flush=True)
    print(f"[Gram] 尾部 6: {[f'{float(x):.3e}' for x in ev[-6:]]}", flush=True)
    print(f"[Gram] 收缩率 eta*lambda_i 前8: "
          f"{[f'{ETA*float(x):.2e}' for x in ev[:8]]}", flush=True)
    print(f"[Gram] 收缩率 eta*lambda_i 尾部: "
          f"{[f'{ETA*float(x):.2e}' for x in ev[-6:]]}", flush=True)

    res["rows"] = []
    for n, Es in PLAN:
        for E in Es:
            W = W0.clone()
            for _ in range(E):
                for i in range(n):
                    k, v = Ks[i], Vs[i]
                    G = alpha * torch.outer(W @ k - v, k) + lam * (W - W0)
                    W = W - ETA * G
            rs = [rel(W, Ks[i], Vs[i]) for i in range(n)]
            res["rows"].append({"n": n, "E": E, "rel_mean": float(np.mean(rs)),
                                "rel_max": float(np.max(rs))})
            print(f"  n={n:<3} E={E:<6} rel mean={np.mean(rs):.5f} max={np.max(rs):.5f}",
                  flush=True)
    st.remove()
    dump(res, "probe91_conv.json")
    print("\n★ 诊断完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
