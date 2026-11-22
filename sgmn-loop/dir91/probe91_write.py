#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 · rev 1.9 附：**装置真正写入的 (k,v) 对** 与 **真检索** （诊断）

背景矛盾：rev1.7 报 `cos(W k_i, v_i)=0.98`（"存储不是瓶颈"），而 `probe91_gate.py` 用
`key_matrices` 的 k（取自 `st._hs_in[li]`）只得到 **0.069/0.007/0.001**。
根因候选：`sgmn_module.write` 的 k 取自 **`self.prev_hs`**（上一次模块调用的 hs_in，
注释写明 `h_{l,t-1}`），而 `key_matrices` 取的是 `st._hs_in[li]`（**当前**层 hs_in）⇒ **两者不是同一张量**。

本探针 hook `write`，抓**真正参与更新**的 (k, v)，再报：
  cos(W k_true, v_true)   —— 被最小化的量，应当很高（若 0.98 复现 ⇒ rev1.7 对，诊断 k 错）
  cos(W q_prompt, v_true) —— **真检索**（查询位 vs 被存目标）
  cos(W q_prompt, e_ans)  —— 落到答案几何上的分量
并报 cos(k_true, q_prompt) 作为编码/查询键匹配度。

★ 只做测量。
"""
import argparse, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, N_FACTS, N_TRAIN_KEYS, MOUNT, build_chunks, build_facts, dump,
                       make_stack, model_and_tok, write_phase)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GAMMA = 0.05


def cos(a, b):
    a = a.detach().float().reshape(-1); b = b.detach().float().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok); chunks, owner = build_chunks(tok, facts)
    res = {"seed": a.seed}

    st = make_stack(gamma=GAMMA, no_write=False, seed=a.seed)
    CAP = {}
    for li, m in st.mems.items():
        orig = m.write
        def wrap(hs_in, drop_sg=False, li=li, m=m, orig=orig):
            before = None if m.prev_hs is None else m.prev_hs.clone()
            r = orig(hs_in, drop_sg)
            if before is not None:
                kt = ((before.reshape(-1, before.shape[-1]) @ m.Pk).mean(0)) * m.pk_scale
                vt = (hs_in.reshape(-1, hs_in.shape[-1]) @ m.Pv).mean(0)
                CAP[li] = (kt.detach().float().clone(), vt.detach().float().clone())
            return r
        m.write = wrap
    write_phase(model, st, chunks)          # 训练装置
    for m in st.mems.values():
        m.no_write = True

    # 查询位：prompt 末位
    QP = {li: [] for li in MOUNT}
    for fi in range(N_FACTS):
        f = facts[fi]
        with torch.no_grad():
            model(input_ids=torch.tensor([f["prompt_ids"]], dtype=torch.long, device=DEV))
        for li in MOUNT:
            hs = st._hs_in[li][0]
            QP[li].append(((hs[-1] @ st.mems[li].Pk) * st.mems[li].pk_scale).float())

    e_ans = model.get_input_embeddings().weight[
        torch.tensor([facts[fi]["val_ids"][0] for fi in range(N_FACTS)], device=DEV)].detach().float()

    res["true_pair"] = {}
    for li in MOUNT:
        m = st.mems[li]
        W = (m.W0 + m.U @ m.V.T).detach().float()
        k_t, v_t = CAP[li]                      # ★ 真正写入的那一对（最后一次）
        row = {"cos_Wk_v": cos(W @ k_t, v_t)}
        qc, qe = [], []
        for fi in range(N_FACTS):
            q = QP[li][fi]
            qc.append(cos(W @ q, v_t)); qe.append(cos(W @ q, e_ans[fi]))
        row["cos_Wq_v_true"] = float(np.mean(qc))
        row["cos_Wq_eans"] = float(np.mean(qe))
        qs = [cos(QP[li][fi], k_t) for fi in range(N_FACTS)]
        row["cos_k_true_q"] = float(np.mean(qs))
        res["true_pair"][li] = row
        print(f" L{li}: ||k_true||={float(k_t.norm()):.3f} ||v_true||={float(v_t.norm()):.3f}", flush=True)
        print(f"   ★ cos(W k_true, v_true) = {row['cos_Wk_v']:+.4f}  ← 被最小化（rev1.7 报 0.98）", flush=True)
        print(f"   ★ cos(W q, v_true)      = {row['cos_Wq_v_true']:+.4f}  ← 真检索", flush=True)
        print(f"   ★ cos(W q, e_ans)       = {row['cos_Wq_eans']:+.4f}", flush=True)
        print(f"     cos(k_true, q_prompt) = {row['cos_k_true_q']:+.4f}", flush=True)
        dump(res, "probe91_write.json")
    st.remove()
    print("\n★ 诊断完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
