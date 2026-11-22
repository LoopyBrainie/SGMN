#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""DIR-91 · rev 2.0 步①②：**非恒真守卫 + 多对共存门**（诊断）

背景（Cycle 151 已定，`DEF-91-12`）：
  写入规则在**单对**上完全收敛（rel→0.0000, cos→1.0000, t=100）；
  完整跑 rel≈2.95 的真因是**调度×步长**——$\eta\alpha\|k\|^2=1.099e-1$，
  单对需 50–100 步，而每个不同对只得到约 1 步。
  但**单目标可达不蕴含多目标共存**（干扰/容量另计）。本探针补这一环。

三个检查（每个都**可能失败**，故非恒真；见 `DEF-91-13` 教训）：

  G1  **写回放守卫**：取受控状态调模块自己的 `m.write(hs)`，
      核验 `Wfull` 的增量是否**逐元素等于**我按同一公式手算的 $-\eta G$。
      ⇒ 若不等，本探针后续所有回放都无意义。
  G2  **读重构守卫**（替代恒真的 C1）：用固定随机 $c$ 调模块自己的 `m.read(hs)`，
      核验其输出是否等于我独立组合的 $W_o\,W\,W_q\,c$。
      ⇒ 这是**不同代码路径**，可失败。
  G3  **多对共存门**：$n\in\{1,4,8,16,32\}$ 个**真实** $(k_i,v_i)$（取自 `key_matrices`），
      $W$ 重置为 $W^0$，每 epoch 顺序把 $n$ 对各写一次，共 $E\in\{1,10,50,200\}$ epoch；
      报每对的 $\mathrm{rel}_i$、$\cos_i$、超阈计数、Gram 条件数。

★ 预注册读法（**在运行前写定，不得事后按结果调整**）：
  - 若 $n=32$ 在 $E=200$ 下仍 mean rel $>0.5$ ⇒ **容量/干扰**是真限制，调度不是唯一原因。
  - 若 $n=32$ 在 $E=200$ 下 mean rel $<0.05$ ⇒ **容量充足（≥32 对）**，
    原失败**纯属每对欠训练**，多对共存不构成瓶颈。
  - 中间态（$0.05\sim0.5$）⇒ 按 $n$ 的依赖曲线判：若 rel 随 $n$ 单调恶化 ⇒ 干扰起作用。

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
EPOCHS = [1, 10, 50, 200]
REL_SAT = 0.05          # 预注册：rel<0.05 记「已满足」（仅报告用，非裁决）


def cos(a, b):
    a = a.detach().float().reshape(-1); b = b.detach().float().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def rel(W, k, v):
    return float(((W @ k) - v).norm() / (v.norm() + 1e-30))


def step(W, k, v, alpha, lam, eta, W0):
    """逐字复刻 sgmn_module.LayerMem.write（use_factorized=False 支路）。"""
    G = alpha * torch.outer(W @ k - v, k) + lam * (W - W0)
    return W - eta * G


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok); chunks, owner = build_chunks(tok, facts)
    st = make_stack(gamma=GAMMA, no_write=False, seed=a.seed)
    res = {"seed": a.seed, "ns": NS, "epochs": EPOCHS, "rel_sat": REL_SAT}
    li = MOUNT[0]; m = st.mems[li]

    # ---------- G1 写回放守卫（可失败）
    model(input_ids=chunks[0].to(DEV))
    model(input_ids=chunks[0].to(DEV))      # 第二次：prev_hs 就绪，write 才不会早退
    assert m.prev_hs is not None, "prev_hs 未就绪，G1 无法进行"
    hs_in = torch.randn(1, 32, m.d_model, device=DEV)     # 受控 hs_in ⇒ v 的来源明确
    k_hand = ((m.prev_hs.reshape(-1, m.prev_hs.shape[-1]) @ m.Pk).mean(0)) * m.pk_scale
    v_hand = (hs_in.reshape(-1, m.d_model) @ m.Pv).mean(0)
    W_before = m.Wfull.detach().clone()
    G_hand = m.alpha * torch.outer(W_before @ k_hand - v_hand, k_hand) \
             + m.lam * (W_before - m.W0)
    expect = W_before - m.eta * G_hand
    m.write(hs_in, False)                   # 模块自己走一遍
    got = m.Wfull.detach().clone()
    g1 = float((got - expect).norm() / (expect.norm() + 1e-30))
    res["G1_write_replay_relerr"] = g1
    res["G1_pass"] = bool(g1 < 1e-6)
    print(f"[G1] 写回放 rel.err = {g1:.3e}  => {'PASS' if g1 < 1e-6 else 'FAIL'}", flush=True)

    # ---------- G2 读重构守卫（可失败；走不同代码路径）
    c3 = torch.randn(1, 8, m.d_model, device=DEV)
    got_r = m.read(c3).detach().float()
    Wc = m.Wfull.detach().float()
    cf = c3.float().reshape(-1, m.d_model)
    q_ = (m.Wq.float() @ cf.T).T
    y_ = (Wc @ q_.T).T
    hand = (m.Wo.float() @ y_.T).T.reshape(c3.shape)
    g2 = float((got_r - hand).norm() / (got_r.norm() + 1e-30))
    res["G2_read_recon_relerr"] = g2
    res["G2_pass"] = bool(g2 < 1e-5)
    print(f"[G2] 读重构 rel.err = {g2:.3e}  => {'PASS' if g2 < 1e-5 else 'FAIL'}", flush=True)
    res["dims"] = {"d_model": int(m.d_model), "d_k": int(m.d_k), "d_v": int(m.d_v),
                   "R": int(m.r), "use_factorized": bool(m.use_factorized)}

    # ---------- G3 多对共存门
    for mm in st.mems.values():
        mm.no_write = True
    K, V = key_matrices(model, st, facts, chunks, owner)
    Ks = [K[li][i].detach().float() for i in range(N_FACTS)]
    Vs = [V[li][i].detach().float() for i in range(N_FACTS)]
    W0 = m.W0.detach().clone()
    alpha, lam, eta = float(m.alpha), float(m.lam), float(m.eta)
    print(f"[超参] alpha={alpha} lam={lam} eta={eta}  W={tuple(W0.shape)}  "
          f"||k||~{float(Ks[0].norm()):.1f} ||v||~{float(Vs[0].norm()):.1f}", flush=True)

    res["G3"] = {}
    for n in NS:
        Ks_n, Vs_n = Ks[:n], Vs[:n]
        Kmat = torch.stack(Ks_n)                      # (n,d_k)
        gram = (Kmat @ Kmat.T).float()
        ev = torch.linalg.eigvalsh(gram)
        cond = float(ev.max() / max(float(ev.min()), 1e-30))
        rec = {"cond_gram": cond, "eta_alpha_k2_mean":
               float(eta * alpha * float((Kmat.norm(dim=1) ** 2).mean()))}
        for E in EPOCHS:
            W = W0.clone()                      # 每个 E 都从 W^0 重来
            for _ in range(E):
                for i in range(n):
                    W = step(W, Ks_n[i], Vs_n[i], alpha, lam, eta, W0)
            rs = [rel(W, Ks_n[i], Vs_n[i]) for i in range(n)]
            cs = [cos(W @ Ks_n[i], Vs_n[i]) for i in range(n)]
            rec[f"E{E}"] = {"rel_mean": float(np.mean(rs)), "rel_max": float(np.max(rs)),
                            "rel_min": float(np.min(rs)), "cos_mean": float(np.mean(cs)),
                            "n_sat": int(sum(1 for r in rs if r < REL_SAT)),
                            "rel_per_pair": [round(r, 4) for r in rs]}
            print(f"  n={n:<3} E={E:<4} rel mean={np.mean(rs):.4f} max={np.max(rs):.4f} "
                  f"cos={np.mean(cs):+.4f} 满足(<{REL_SAT})={sum(1 for r in rs if r < REL_SAT)}/{n}",
                  flush=True)
        print(f"   ↳ n={n}: Gram cond={cond:.3e}  eta*alpha*||k||^2={rec['eta_alpha_k2_mean']:.3e}",
              flush=True)
        res["G3"][n] = rec
    st.remove()
    dump(res, "probe91_coexist.json")
    print("\n★ 诊断完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
