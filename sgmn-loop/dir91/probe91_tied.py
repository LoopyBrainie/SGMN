#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 · rev 1.7 第二探针：**把读侧查询绑定到写入侧键投影**（诊断，不产生裁决）

前一个探针（`probe91_identity.py`）的决定性读数：
  M1 cos(Wq, pk_scale·Pk^T) ≈ 0.000（训练前后不变）；
  训练键 常规 q acc(题)=1.000，**同一批训练键换 oracle-q = 0.062**。
⇒ 全部能力在 Wq/Wo/gamma 里，W 几乎不参与 —— 该通路是「计算」而非「检索」。

本探针检验**唯一自然的修法**：读侧查询不自由训练，而**与写入键同源**：
      q_t = pk_scale · (Pk^T c_t)        （= 写入时 k 的定义，sgmn_module.py:86）
则 W q ≈ W k_i ≈ v_i 才是真的检索，而不是用 W 当随机矩阵。

三臂（同一次写入、同一批键、同一评测）：
  T0  绑定查询、**不训练**            —— 冻结装置的裸读数
  T1  绑定查询、只训练 Wo 与 gamma    —— 检验"绑定后读侧还能不能学到有用的输出图"
  T2  绑定查询、训练 Wq 也放开        —— 对照：若放开又退回计算通路，说明绑定是必要的
另记 T1 的训练键读数（能背 v.s. 能推）与每层 cos(W q, v_i)（写入是否真的把 v_i 放到 k_i 上）。

★ 不改任何已冻结判据；只产出诊断数字。
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from R2_task91 import (  # noqa: E402
    DEV, N_FACTS, N_TRAIN_KEYS, READ_LR, READ_STEPS, acc_by_question,
    build_chunks, build_facts, dump, key_matrices, make_eval_items, make_stack,
    model_and_tok, nll_of_candidate, train_read_task, write_phase,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GAMMA = 0.05


def flat_cos(a, b):
    a = a.detach().reshape(-1).double()
    b = b.detach().reshape(-1).double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def patch_tied_read(st, train_wo=True, train_wq=False):
    """把每层 read 换成 q = pk_scale·(Pk^T c)。返回原 read 以便还原。"""
    orig = {li: m.read for li, m in st.mems.items()}
    for li, m in st.mems.items():
        def make(mem):
            def tied(hs_in, drop_sg=False):
                c = hs_in.detach()
                cf = c.reshape(-1, c.shape[-1])
                q = (cf @ mem.Pk).detach() * float(mem.pk_scale)   # (N,d_k) = Pk^T c · p
                W = (mem.W0 + mem.U @ mem.V.T) if mem.use_factorized else mem.Wfull
                y = (W @ q.T).T
                mm = (mem.Wo @ y.T).T
                mem.injection = mm.reshape(hs_in.shape)
                return mem.injection
            return tied
        m.read = make(m)
    return orig


def train_selected(model, st, tok, facts, keys, rng, train_wq, train_wo, steps=None):
    """只对选定参数做 Adam 训练；损失与 train_read_task 完全一致（4 选 1 候选 CE）。"""
    steps = steps or READ_STEPS
    ps = []
    for li, m in st.mems.items():
        if train_wq:
            ps.append(m.Wq)
        if train_wo:
            ps.append(m.Wo)
        ps.append(m.gamma)
    opt = torch.optim.Adam(ps, lr=READ_LR)
    import torch.nn.functional as F
    batches = []
    for ki in keys:
        f = facts[ki]
        others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != f["val"]]
        distr = rng.sample(sorted(set(others)), 3)
        cands = [f["val"]] + distr
        rng.shuffle(cands)
        ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in cands]
        batches.append((f["prompt_ids"], ids, cands.index(f["val"])))
    losses = []
    for step in range(steps):
        prompt_ids, ids, tgt = batches[step % len(batches)]
        P = len(prompt_ids)
        seq = torch.tensor([prompt_ids + [c] for c in ids], dtype=torch.long, device=DEV)
        opt.zero_grad(set_to_none=True)
        lg = model(input_ids=seq).logits.float()
        idx = torch.arange(len(ids), device=DEV)
        pick = lg[idx, P - 1, torch.tensor(ids, device=DEV)].unsqueeze(0)
        loss = F.cross_entropy(pick, torch.tensor([tgt], device=DEV))
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses


def eval_plain(model, st, tok, facts, keys, rng_seed=555):
    for m in st.mems.values():
        m.no_write = True
    items = make_eval_items(tok, facts, keys, random.Random(rng_seed))
    det = []
    for it in items:
        nlls = [nll_of_candidate(model, st, it["prompt_ids"], c[0]) for c in it["cand"]]
        pick = int(np.argmin(nlls))
        det.append({"ok": bool(it["cand"][pick][1])})
    acc = float(np.mean([d["ok"] for d in det]))
    accq, _ = acc_by_question(items, det)
    return acc, accq


def wv_readings(st, K, V):
    """每层 cos(W k_i, v_i) 与相对误差 —— 写入是否真把 v_i 放到 k_i 上（与读侧无关）。"""
    out = {}
    for li, m in st.mems.items():
        W = (m.W0 + m.U @ m.V.T) if m.use_factorized else m.Wfull
        cs, rs = [], []
        for i in range(len(K[li])):
            Wk = W.detach() @ K[li][i]
            cs.append(flat_cos(Wk, V[li][i]))
            rs.append(float((Wk - V[li][i]).norm() / (V[li][i].norm() + 1e-30)))
        out[li] = {"cos_Wk_v_mean": float(np.mean(cs)),
                   "rel_err_mean": float(np.mean(rs)),
                   "cos_per_fact": [round(c, 4) for c in cs]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args()
    seed = a.seed
    os.makedirs(OUT, exist_ok=True)

    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    order = list(range(N_FACTS))
    random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]
    print(f"训练键 {len(train_keys)} 留出键 {len(eval_keys)} LR={READ_LR} steps={READ_STEPS}",
          flush=True)

    res = {"seed": seed, "gamma": GAMMA, "READ_LR": READ_LR,
           "train_keys": train_keys, "eval_keys": eval_keys, "arms": {}}

    # ---------- 写入一次，三条臂共用同一 W（各自 make_stack 但 seed 相同 ⇒ 同初值同序列）
    print("\n--- 基线 W 的写入质量（与读侧无关）---", flush=True)
    st0 = make_stack(gamma=GAMMA, no_write=False, seed=seed)
    write_phase(model, st0, chunks)
    K, V = key_matrices(model, st0, facts, chunks, owner)
    wv0 = wv_readings(st0, K, V)
    res["W_vs_v_untrained_read"] = wv0
    for li, d in wv0.items():
        print(f"  L{li}: cos(W k_i, v_i)={d['cos_Wk_v_mean']:+.4f}  "
              f"rel_err={d['rel_err_mean']:.4f}", flush=True)
    st0.remove()

    def one_arm(tag, train_wo, train_wq):
        st = make_stack(gamma=GAMMA, no_write=False, seed=seed)
        write_phase(model, st, chunks)
        _K, _V = key_matrices(model, st, facts, chunks, owner)
        patch_tied_read(st, train_wo, train_wq)
        if train_wo or train_wq:
            losses = train_selected(model, st, tok, facts, train_keys,
                                    random.Random(1000), train_wq, train_wo)
            fin = float(np.mean(losses[-50:]))
        else:
            fin = None
        acc, accq = eval_plain(model, st, tok, facts, eval_keys)
        acc_t, accq_t = eval_plain(model, st, tok, facts, train_keys, rng_seed=556)
        d = {"train_wo": train_wo, "train_wq": train_wq, "final_loss": fin,
             "acc_eval": acc, "accq_eval": accq,
             "acc_train": acc_t, "accq_train": accq_t}
        res["arms"][tag] = d
        print(f"  [{tag}] loss={fin}  留出键 acc(项)={acc:.3f} acc(题)={accq:.3f} | "
              f"训练键 acc(项)={acc_t:.3f} acc(题)={accq_t:.3f}", flush=True)
        dump(res, "probe91_tied.json")
        st.remove()
        return d

    print("\n--- 三臂（查询一律绑定到写入侧键投影）---", flush=True)
    one_arm("T0_tied_untrained", False, False)
    one_arm("T1_tied_train_Wo_gamma", True, False)
    one_arm("T2_tied_train_all", True, True)

    print("\n★ 诊断完成（不产生裁决）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())