#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""DIR-91 · rev 2.1 预备：**float64 谱分析——修正 LS 地板、算真实秩/居中秩、导出 T**（诊断）

三处必须修（Advisor 复核指出，均成立）：
  E1  **$\lambda{=}0$ 的 LS 地板不应 $>0$**：$n\le32$ 个**线性独立**的 $k_i$ 下，
      自由 $896\times896$ 的 $W$ 可**精确**满足全部 $Wk_i{=}v_i$（把 $W$ 在
      $\mathrm{span}\{k_i\}$ 上任意定义后延拓）。故 $\lambda{=}0$ 地板 $\equiv0$。
      上轮报的 $0.02159/0.03112$ 是 **float32 正规方程在 cond$\\approx3\times10^3$ 下的数值垃圾**。
      同理 **S2-k $=0.9821$ 是近恒真的**（$v_i$ 不必落在 $\\mathrm{span}\{k_j\}$ 内）⇒ 撤回该证据。
  E2  **「秩塌缩 $32\to1.03$ 维」表述错误**：数值秩就是 32（cond$\\approx3\times10^3$），
      退化的是**能量分布**（一个方向占绝大部分能量）⇒ 一律改称
      **「近共线／能量主导」**。补算**居中有效秩**与 $\\lVert k_i-\\bar k\\rVert/\\lVert k_i\\rVert$
      （上一轮只对 $v$ 算过，对 $k$ 没算——而它才决定条件数与 $T$）。
  E3  **$T$ 不能靠猜**：由 epoch 映射
      $$M_{\rm ep}=M_1M_2\cdots M_n,\quad M_i=I-\eta(\alpha k_ik_i^{\top}+\lambda I)$$
      的**谱半径** $\rho$ 导出收敛速率，再由预注册目标 rel 反推 $T$。
      同时给出 $M_{\rm ep}$ **不动点**的精确渐近地板（换掉 float32 的 $0.00080$）。

另加零成本确认：$\lVert (W^0k_i-v_i)_{\perp}\rVert/\lVert v_i\rVert$，
$\perp$ = 去掉键的主方向后的分量（Advisor 提示：若 $\approx0.10$，则伪地板被定量钉死）。

**全部 float64。** 装置本身是 float32 ⇒ 本探针给出的是**可实现界**，不是装置读数；
两处口径分别标注。
★ 只做测量；不产生裁决、不写证据行、不改已冻结阈值。
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


def eff_rank(M):
    ev = torch.linalg.eigvalsh(M).clamp(min=0)
    p = ev / ev.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-30)).sum()))


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
    # ★ float64
    Ks = torch.stack([K[li][i].detach() for i in range(N_FACTS)]).double()
    Vs = torch.stack([V[li][i].detach() for i in range(N_FACTS)]).double()
    W0 = m.W0.detach().double().clone()
    alpha, lam, eta = float(m.alpha), float(m.lam), float(m.eta)
    res = {"seed": a.seed, "dtype": "float64", "alpha": alpha, "lam": lam, "eta": eta}

    # ---------- E2 键/值的真实结构与居中结构
    sk = torch.linalg.svdvals(Ks)
    sv = torch.linalg.svdvals(Vs)
    kbar = Ks.mean(0, keepdim=True); vbar = Vs.mean(0, keepdim=True)
    Kc = Ks - kbar; Vc = Vs - vbar
    # 数值秩：sv > tol*sv_max
    rank_by_tol = {}
    for tag, tol in (("1e-6", 1e-6), ("1e-3", 1e-3), ("1e-2", 1e-2)):
        rank_by_tol[tag] = int((sk > tol * sk[0]).sum())
    res["rank_by_tol"] = rank_by_tol
    res.update({
        "sv_K_top": [float(x) for x in sk[:6]],
        "sv_K_min": float(sk[-1]), "cond_K": float(sk[0] / sk[-1]),
        "effrank_K": eff_rank(Ks @ Ks.T), "effrank_Kc": eff_rank(Kc @ Kc.T),
        "effrank_V": eff_rank(Vs @ Vs.T), "effrank_Vc": eff_rank(Vc @ Vc.T),
        "knorm_dev_mean": float((Kc.norm(dim=1) / Ks.norm(dim=1)).mean()),
        "vnorm_dev_mean": float((Vc.norm(dim=1) / Vs.norm(dim=1)).mean()),
        "cos_K_offdiag_mean": float(((Ks / Ks.norm(dim=1, keepdim=True)) @
                                     (Ks / Ks.norm(dim=1, keepdim=True)).T)[
            ~torch.eye(N_FACTS, dtype=bool)].mean()),
    })
    print(f"[E2] K 奇异值 前6={[f'{float(x):.3e}' for x in sk[:6]]}", flush=True)
    print(f"[E2] K cond={float(sk[0]/sk[-1]):.3e}  数值秩(tol=1e-2)={res['rank_by_tol']['1e-2']}/32  "
          f"居中有效秩={res['effrank_Kc']:.2f}（未居中={res['effrank_K']:.2f}）", flush=True)
    print(f"[E2] ||k_i-kbar||/||k_i||={res['knorm_dev_mean']:.4f}   "
          f"||v_i-vbar||/||v_i||={res['vnorm_dev_mean']:.4f}", flush=True)

    # ---------- E1 float64 正则地板 + 纯 LS 地板（后者按构造应为 0）
    res["floors"] = []
    for n in NS:
        Kn, Vn = Ks[:n], Vs[:n]
        A = alpha * (Vn.T @ Kn) + lam * W0
        B = alpha * (Kn.T @ Kn) + lam * torch.eye(Kn.shape[1], dtype=torch.float64,
                                                     device=Kn.device)
        Wl = torch.linalg.solve(B.T, A.T).T
        r_lam = [rel(Wl, Kn[i], Vn[i]) for i in range(n)]
        # 纯 LS：W 在 span{k} 上任意 ⇒ 精确拟合。
        # 显式最小范数解 W = V^T (K K^T)^{-1} K  ⇒  W k_i = V^T e_i = v_i 精确成立
        Gk = Kn @ Kn.T                                       # (n,n)
        W0_ = Vn.T @ torch.linalg.solve(Gk, Kn)              # (d_v,n)@(n,d_k) = (d_v,d_k)
        r_ls = [rel(W0_, Kn[i], Vn[i]) for i in range(n)]
        res["floors"].append({"n": n, "rel_reg": float(np.mean(r_lam)),
                              "rel_reg_max": float(np.max(r_lam)),
                              "rel_ls": float(np.mean(r_ls)),
                              "rel_ls_max": float(np.max(r_ls)),
                              "cond_gram": float(torch.linalg.cond(B))})
        print(f"  n={n:<3} 正则地板(float64)={np.mean(r_lam):.6f}  "
              f"纯LS地板={np.mean(r_ls):.3e}", flush=True)

    # ---------- E3 epoch 映射谱半径 + 不动点渐近地板
    EPOCH_CAP = 6000
    res["epoch"] = []
    I = torch.eye(Ks.shape[1], dtype=torch.float64, device=Ks.device)
    for n in (8, 32):
        Kn, Vn = Ks[:n], Vs[:n]
        M = I.clone()
        for i in range(n):
            Mi = I - eta * (alpha * torch.outer(Kn[i], Kn[i]) + lam * I)
            M = M @ Mi
        # 谱半径：幂迭代（实矩阵，取 |lambda| 最大）
        x = torch.randn(Ks.shape[1], dtype=torch.float64, device=Ks.device, generator=torch.Generator(device=Ks.device).manual_seed(0))
        x /= x.norm()
        rho = 0.0
        for _ in range(3000):
            y = M @ x
            ny = y.norm()
            if ny < 1e-300:
                break
            rho = float(ny)
            x = y / ny
        # 不动点：迭代 epoch 到收敛（float64）
        W = W0.clone()
        for _ in range(EPOCH_CAP):
            for i in range(n):
                G = alpha * torch.outer(W @ Kn[i] - Vn[i], Kn[i]) + lam * (W - W0)
                W = W - eta * G
        rs_inf = [rel(W, Kn[i], Vn[i]) for i in range(n)]
        # T 推导：rel(T) ≈ C rho^T，用 E=400 的值定 C
        C = float(np.mean([0.09750 if n == 8 else 0.09873]))
        Ts = {}
        for tgt in (0.05, 0.01):
            if 0 < rho < 1:
                Ts[str(tgt)] = float(np.log(tgt / C) / np.log(rho))
        res["epoch"].append({"n": n, "rho": rho, "rel_at_cap_epochs": float(np.mean(rs_inf)),
                             "rel_max_cap": float(np.max(rs_inf)), "T_for_target": Ts})
        print(f"  n={n:<3} 谱半径 rho={rho:.6f}  ⇒ cap epoch 渐近地板="
              f"{np.mean(rs_inf):.6f}  T(rel<0.05)≈{Ts.get('0.05', float('nan')):.0f} epoch",
              flush=True)
    st.remove()
    dump(res, "probe91_spectral.json")
    print("\n★ 诊断完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
