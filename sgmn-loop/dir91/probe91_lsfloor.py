#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""DIR-91 · rev 2.0 步②终：**0.10 的地板是不是「正则最小二乘最优」？**（诊断）

`probe91_etafloor.py` **推翻了** `floor ≈ c·ηα‖k‖²`（$n\ge8$ 时斜率 $-0.035/-0.047\approx0$，
跨 30× $\eta$ 地板恒在 $0.098\sim0.116$）⇒ **地板与步长无关**，故不是优化器噪声底，
而应是**目标函数本身的最优点**。本探针直接解析算出该最优点并与之比对。

**目标函数**（逐字取自 `write` 的 $G$，SGD 正是在它上面下降）：
  $$ L(W)=\sum_i \alpha\|Wk_i-v_i\|^2+\lambda\|W-W^0\|_F^2 $$
**闭式解**：$W^\star=(\alpha V^{\top}K+\lambda W^0)(\alpha K^{\top}K+\lambda I)^{-1}$

同时报三项**结构性诊断**（用来区分「键冲突」与「其他」）：
  S1 键两两余弦分布（$\max$/均值），以及 Gram 的**有效秩**（谱熵）；
  S2 $v_i$ 中**不可由 $\{k_j\}$ 线性解释**的分量占比 —— 若该量与地板同量级，
     则地板是**键-值内在冲突**（任何 $W$ 都做不到），与优化器无关；
  S3 $\lambda\to0$ 时的纯最小二乘地板，确认地板是否由 $\lambda$ 主导（预期：不是）。

**预注册判读（运行前冻结）**：
  - 若 $W^\star$ 的地板 $\approx$ 实测地板（$0.0992\sim0.1078$，差 $<20\%$）⇒
    **地板 = 正则最小二乘最优**，优化器已尽力，天花板由**键-值结构**决定。
  - 若 $W^\star$ 地板 $\ll$ 实测 ⇒ **优化器未达最优**（步长/epoch 不足或动力学另有陷阱），
    须改判为「调度仍有余量」。
  - 若 S2 的最小二乘残差与地板同量级 ⇒ **32 个事实在键空间不可分**，是**内在冲突**。

★ 只做测量；不产生裁决、不写证据行、不改任何已冻结阈值。
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, MOUNT, N_FACTS, build_chunks, build_facts, dump,
                       key_matrices, make_stack, model_and_tok)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GAMMA = 0.05
NS = [1, 4, 8, 16, 32]


def rel(W, k, v):
    return float(((W @ k) - v).norm() / (v.norm() + 1e-30))


def ls_solution(K, V, W0, alpha, lam):
    """W* = (alpha V^T K + lam W0) (alpha K^T K + lam I)^{-1}；K:(n,d_k) V:(n,d_v)"""
    n = K.shape[0]
    A = alpha * (V.T @ K) + lam * W0                      # (d_v,d_k)
    B = alpha * (K.T @ K) + lam * torch.eye(K.shape[1], device=K.device, dtype=K.dtype)
    return torch.linalg.solve(B.T, A.T).T                 # A @ B^{-1}


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
    Ks = torch.stack([K[li][i].detach().float() for i in range(N_FACTS)])   # (32,d_k)
    Vs = torch.stack([V[li][i].detach().float() for i in range(N_FACTS)])   # (32,d_v)
    W0 = m.W0.detach().float().clone()
    alpha, lam = float(m.alpha), float(m.lam)
    res = {"seed": a.seed, "alpha": alpha, "lam": lam, "ns": NS}

    # ---------- S1 键结构
    Kn = Ks / Ks.norm(dim=1, keepdim=True)
    Gc = (Kn @ Kn.T)
    off = Gc[~torch.eye(N_FACTS, dtype=bool, device=Gc.device)]
    ev = torch.linalg.eigvalsh(Ks @ Ks.T).clamp(min=0)
    p = ev / ev.sum()
    effrank = float(torch.exp(-(p * torch.log(p + 1e-30)).sum()))
    res["S1"] = {"cos_key_mean": float(off.mean()), "cos_key_max": float(off.max()),
                 "cos_key_absmean": float(off.abs().mean()), "eff_rank": effrank}
    print(f"[S1] 键两两余弦 mean={float(off.mean()):+.4f} |mean|={float(off.abs().mean()):.4f} "
          f"max={float(off.max()):+.4f}  有效秩={effrank:.2f}/{N_FACTS}", flush=True)

    # ---------- S2/S3 各 n 的最小二乘地板
    res["rows"] = []
    for n in NS:
        Kn_, Vn_ = Ks[:n], Vs[:n]
        r = {"n": n}
        for tag, La in (("lam", lam), ("zero", 0.0)):
            A = alpha * (Vn_.T @ Kn_) + (La * W0 if La > 0 else torch.zeros_like(W0))
            B = alpha * (Kn_.T @ Kn_) + (La * torch.eye(Kn_.shape[1], device=Kn_.device)
                                         if La > 0 else torch.zeros(Kn_.shape[1], Kn_.shape[1], device=Kn_.device))
            if La > 0:
                W = torch.linalg.solve(B.T, A.T).T
            else:  # 纯最小二乘：W = V^T K (K^T K)^+
                W = (Vn_.T @ Kn_) @ torch.linalg.pinv(Kn_.T @ Kn_)
            rs = [rel(W, Kn_[i], Vn_[i]) for i in range(n)]
            r[f"floor_{tag}"] = float(np.mean(rs))
            r[f"relmax_{tag}"] = float(np.max(rs))
        res["rows"].append(r)
        print(f"  n={n:<3} 正则最优地板={r['floor_lam']:.5f}  "
              f"纯LS(lambda=0)地板={r['floor_zero']:.5f}", flush=True)

    # ---------- S2b 键/值两侧的退化度（内在冲突的直接度量）
    Vn_ = Vs / Vs.norm(dim=1, keepdim=True)
    Gv = (Vn_ @ Vn_.T)
    offv = Gv[~torch.eye(N_FACTS, dtype=bool, device=Gv.device)]
    evv = torch.linalg.eigvalsh(Vs @ Vs.T).clamp(min=0)
    pv = evv / evv.sum()
    effrank_v = float(torch.exp(-(pv * torch.log(pv + 1e-30)).sum()))
    res["S2v"] = {"cos_val_mean": float(offv.mean()), "cos_val_max": float(offv.max()),
                  "eff_rank": effrank_v}
    print(f"[S2-v] 值两两余弦 mean={float(offv.mean()):+.4f} max={float(offv.max()):+.4f} "
          f"有效秩={effrank_v:.2f}/{N_FACTS}", flush=True)

    # v_i 有多少**无法**由 span{k_j} 线性解释（任何 W 都拿不到的部分）
    Kpinv = torch.linalg.pinv(Ks.T)          # (n,d_k) 的伪逆 -> (32, d_k)
    res["S2"] = []
    for i in range(N_FACTS):
        c = Kpinv @ Vs[i]                    # (32,)
        fit = Ks.T @ c                       # (d_k,) 以 k 为基去拟合 v_i
        resid = float((fit - Vs[i]).norm() / (Vs[i].norm() + 1e-30))
        res["S2"].append(resid)
    s2 = float(np.mean(res["S2"]))
    print(f"[S2-k] v_i 无法由 span{{k_j}} 解释的分量 mean={s2:.4f} "
          f"(min={min(res['S2']):.4f} max={max(res['S2']):.4f})", flush=True)

    # 对照：v_i 距「公共均值方向」多远（键退化时，W 只能产出这个共同方向）
    vbar = Vs.mean(dim=0, keepdim=True)
    dv = [(float((Vs[i] - vbar.squeeze(0)).norm() / (Vs[i].norm() + 1e-30)))
          for i in range(N_FACTS)]
    res["S2_vbar"] = {"mean": float(np.mean(dv)), "min": min(dv), "max": max(dv)}
    print(f"[S2-vbar] ||v_i - mean(v)|| / ||v_i|| mean={np.mean(dv):.4f} "
          f"(min={min(dv):.4f} max={max(dv):.4f})", flush=True)
    print("     ↑ 键退化到 1 维时，任何 W 对每个 k 只能给出**同一个**输出方向，"
          "该项即其不可消除残差", flush=True)
    st.remove()
    dump(res, "probe91_lsfloor.json")
    print("\n★ 诊断完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
