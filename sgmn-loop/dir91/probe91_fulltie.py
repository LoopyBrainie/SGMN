#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 · rev 1.7 第四探针：**整条读链无学习参数**（诊断，不产生裁决）

前三探针：
  ① 写入好：cos(W k_i, v_i)=0.987/0.981/0.963
  ② 不绑定查询 ⇒ 读侧绕开 W（训练键 1.000 v.s. 换真键 0.062）
  ③ 绑定查询 ⇒ 训练键 1.000、留出键 0.000；且泛化随训练事实数 k **下降**（0.214→0.097→0.062 按项）
⇒ 失败在读出解码器 Wo 的泛化。

本探针把**读出也绑定**：m_t = Pv (W q_t)，q_t = pk_scale·(Pk^T c_t)
⇒ 读链 = Pv ∘ W ∘ Pk^T，**无任何可学习参数**（gamma 仅作标量放大，取 1.0 与 0.05 两档对照）。
若此臂在**留出键**上显著优于机会 ⇒ 架构可行，坏的是"可学习读侧"；
若仍贴地板 ⇒ 该基座/该注入位置上，"线性检索出的 v 无法被 LLM 用作答案"。

★ 不改任何已冻结判据。
"""
import argparse, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (N_FACTS, N_TRAIN_KEYS, acc_by_question, build_chunks, build_facts,
                       dump, make_eval_items, make_stack, model_and_tok,
                       nll_of_candidate, write_phase)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

def patch_fulltie(st):
    orig = {li: m.read for li, m in st.mems.items()}
    for li, m in st.mems.items():
        def make(mem):
            def ft(hs_in, drop_sg=False):
                cf = hs_in.detach().reshape(-1, hs_in.shape[-1])
                q = (cf @ mem.Pk).detach() * float(mem.pk_scale)      # (N,d_k)
                W = (mem.W0 + mem.U @ mem.V.T) if mem.use_factorized else mem.Wfull
                y = (W @ q.T).T                                       # (N,d_v) ≈ v_i
                mm = (y @ mem.Pv.T) * float(mem.gamma)                # 读回残差空间
                mem.injection = mm.reshape(hs_in.shape)
                return mem.injection
            return ft
        m.read = make(m)
    return orig

def ev(model, st, tok, facts, keys, seed=555):
    for m in st.mems.values():
        m.no_write = True
    items = make_eval_items(tok, facts, keys, random.Random(seed))
    det = []
    for it in items:
        nlls = [nll_of_candidate(model, st, it["prompt_ids"], c[0]) for c in it["cand"]]
        det.append({"ok": bool(it["cand"][int(np.argmin(nlls))][1])})
    return float(np.mean([d["ok"] for d in det])), acc_by_question(items, det)[0]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok); chunks, owner = build_chunks(tok, facts)
    order = list(range(N_FACTS)); random.Random(777).shuffle(order)
    tr, evk = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]
    res = {"seed": a.seed, "arms": {}}
    for g in (1.0, 0.05, 0.2):
        st = make_stack(gamma=g, no_write=False, seed=a.seed)
        write_phase(model, st, chunks)
        patch_fulltie(st)
        a_e, q_e = ev(model, st, tok, facts, evk)
        a_t, q_t = ev(model, st, tok, facts, tr, seed=556)
        res["arms"][f"fulltie_g{g}"] = {"gamma": g, "acc_eval": a_e, "accq_eval": q_e,
                                        "acc_train": a_t, "accq_train": q_t}
        print(f"  [fulltie γ={g}] 训练键 {a_t:.3f}/{q_t:.3f} | 留出键 {a_e:.3f}/{q_e:.3f}",
              flush=True)
        dump(res, "probe91_fulltie.json")
        st.remove()
    print("\n★ 诊断完成（不产生裁决）", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
