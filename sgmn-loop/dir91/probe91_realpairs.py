#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""DIR-91 · rev2.2 **缺口(a)(b) 实测**

(a) **块错位**：`write()` 的 $k=\mathrm{mean}(P_k h_{t-1})$ 取自**前一块**、
    $v=\mathrm{mean}(P_v h_t)$ 取自**当前块**；而 `key_matrices` **两块都取自同一块**
    （`owner.index(fi)+1`）。故二者口径不同。本探针 **patch `write` 抓真实 $(k,v)$ 序列**，
    量化：真实写序列长度、每事实写次数、**边界写**（$k$ 来自别的块）个数，
    以及「用参考对评估」v.s.「用真实写入对评估」的 rel 差。
(b) **$k,v$ 漂移**：$h^+{=}h+\gamma m$，$\gamma{=}0$ 时前向无注入 ⇒ 隐状态**与 $W$ 无关**
    ⇒ 参考对在全过程**恒定**。本探针**写前/写后各算一次** $K,V$ 并报 $\max|\Delta|$ 直接验证。

**归因**：同总写入数下，用参考对模拟
  ① **blocked**（按真实顺序，每事实连续写 $c$ 次）② **round-robin**（每轮每事实写 1 次）
再与**真实** rel 比较 ⇒ 判定 20× 差异来自**顺序**还是**对本身**。

★ 只测量；不产生裁决、不写证据行、不改冻结阈值。
"""
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, MOUNT, N_FACTS, build_chunks, build_facts, dump,
                       key_matrices, make_stack, model_and_tok, W_of, write_phase)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def relvec(W, K, V):
    return ((K @ W.T) - V).norm(dim=1) / V.norm(dim=1).clamp_min(1e-30)


def eff_rank(M):
    ev = torch.linalg.eigvalsh(M).clamp(min=0)
    p = ev / ev.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-30)).sum()))


def runs_of(owner):
    out, i = [], 0
    while i < len(owner):
        j = i
        while j + 1 < len(owner) and owner[j + 1] == owner[i]:
            j += 1
        out.append((int(owner[i]), i, j - i + 1))
        i = j + 1
    return out


def simulate(K, V, eta, alpha, lam, W0, seq):
    W = W0.clone()
    for f in seq:
        k, v = K[f], V[f]
        G = alpha * torch.outer(W @ k - v, k) + lam * (W - W0)
        W = W - eta * G
    return W


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok); chunks, owner = build_chunks(tok, facts)
    st = make_stack(gamma=0.0, seed=a.seed)

    K0, V0 = key_matrices(model, st, facts, chunks, owner)      # 写前参考对

    rec = {li: [] for li in st.mems}
    for li, m in st.mems.items():
        orig = m.write
        def make_w(li=li, m=m, orig=orig):
            def w(hs_in, drop_sg=False):
                if m.prev_hs is None or m.no_write:
                    return orig(hs_in, drop_sg)
                pf = m.prev_hs.reshape(-1, m.prev_hs.shape[-1])
                hf = hs_in.reshape(-1, hs_in.shape[-1])
                k = ((pf @ m.Pk).mean(dim=0)) * m.pk_scale
                v = ((hf @ m.Pv).mean(dim=0))
                rec[li].append((k.detach().clone(), v.detach().clone()))
                return orig(hs_in, drop_sg)
            return w
        m.write = make_w()

    write_phase(model, st, chunks)
    K1, V1 = key_matrices(model, st, facts, chunks, owner)       # 写后（γ=0）

    res = {"seed": a.seed, "gamma": 0.0, "dtype_dev": "float32",
           "n_chunks": len(chunks), "mount": list(MOUNT), "per_layer": {}}
    runs = runs_of(owner)
    res["runs"] = [{"fact": f, "start": s, "count": c} for f, s, c in runs]
    res["n_runs"] = len(runs)

    for li in MOUNT:
        m = st.mems[li]
        W = W_of(m).float(); W0m = m.W0.detach().float()
        recs = rec[li]
        eta, alpha, lam = float(m.eta), float(m.alpha), float(m.lam)

        Kr, Vr = [], []
        for f, s, c in runs:
            c_last = s + c - 1
            j = c_last - 1
            if 0 <= j < len(recs):
                Kr.append(recs[j][0]); Vr.append(recs[j][1])
            else:
                Kr.append(K0[li][f]); Vr.append(V0[li][f])
        Kr = torch.stack(Kr).float(); Vr = torch.stack(Vr).float()

        bnd = sum(1 for f, s, c in runs if s - 1 >= 0 and owner[s - 1] != f)
        G0 = K0[li] @ K0[li].T
        G1 = Kr @ Kr.T

        seq_block = []
        for i, (f, s, c) in enumerate(runs):
            seq_block += [f] * (c - 1 if i == 0 else c)
        Wb = simulate(K0[li], V0[li], eta, alpha, lam, W0m, seq_block)
        seq_rr = [f for f, s, c in runs] * 18
        Wr = simulate(K0[li], V0[li], eta, alpha, lam, W0m, seq_rr)

        kn = torch.nn.functional.normalize(Kr, dim=1)
        C = kn @ kn.T
        off = C[~torch.eye(len(Kr), dtype=torch.bool, device=C.device)]
        Kc = Kr - Kr.mean(0, keepdim=True)

        res["per_layer"][li] = {
            "n_write_calls": len(recs), "boundary_writes": bnd,
            "drift_K_max": float((K1[li] - K0[li]).abs().max()),
            "drift_V_max": float((V1[li] - V0[li]).abs().max()),
            "rel_baseline_W0": float(relvec(W0m, K0[li], V0[li]).median()),
            "rel_vs_reference": float(relvec(W, K0[li], V0[li]).median()),
            "rel_vs_real": float(relvec(W, Kr, Vr).median()),
            "rel_sim_blocked_ref": float(relvec(Wb, K0[li], V0[li]).median()),
            "rel_sim_roundrobin_ref": float(relvec(Wr, K0[li], V0[li]).median()),
            "n_sim_blocked": len(seq_block), "n_sim_rr": len(seq_rr),
            "cos_real_key_mean": float(off.mean()), "cos_real_key_max": float(off.max()),
            "effrank_centered_real": eff_rank(Kc @ Kc.T),
            "effrank_centered_ref": eff_rank((K0[li] - K0[li].mean(0, keepdim=True)) @
                                            (K0[li] - K0[li].mean(0, keepdim=True)).T),
        }
        r = res["per_layer"][li]
        print(f"[L{li}] writes={r['n_write_calls']} boundary={bnd} "
              f"driftK={r['drift_K_max']:.2e} driftV={r['drift_V_max']:.2e}", flush=True)
        print(f"     rel: base={r['rel_baseline_W0']:.4f} vs_ref={r['rel_vs_reference']:.4f} "
              f"vs_real={r['rel_vs_real']:.4f} | sim blocked={r['rel_sim_blocked_ref']:.4f} "
              f"rr={r['rel_sim_roundrobin_ref']:.4f}", flush=True)
        print(f"     real keys: cos={r['cos_real_key_mean']:.4f} "
              f"centered_effrank={r['effrank_centered_real']:.2f} "
              f"(ref {r['effrank_centered_ref']:.2f})", flush=True)

    st.remove()
    dump(res, "probe91_realpairs.json")
    print("\n★ 完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
