#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 · rev 1.7 第三探针：**读出解码器的泛化随训练事实数的标度**（诊断，不产生裁决）

前两探针结论：
  ① 写入是好的：cos(W k_i, v_i) = 0.987/0.981/0.963（rel_err 0.16/0.19/0.27）
  ② 不绑定查询 ⇒ 读侧绕开 W（训练键常规 q 1.000，换 oracle-q 0.062）
  ③ 绑定查询（q = pk_scale·Pk^T c）⇒ 训练键 1.000、**留出键 0.000**
⇒ 失败点不在 q、不在 W，而在 Wo（读出解码器）只会背见过的 v。

本探针把训练事实数 k 扫 {4, 8, 16, 24}，评测**仅在留出键**上：
  若留出 acc 恒 ~0 ⇒ 解码器根本无法从这个装置的 16-24 条样本里学到"检索"（架构级结论）。
  若留出 acc 随 k 上升 ⇒ DIR-91 的失败仅源于事实数太少（设定级结论）。
★ 绑定查询（T1 配置：只训 Wo 与 gamma）为主口径；另记不绑定（T2'）作对照。
★ 不改任何已冻结判据。
"""
import argparse, os, random, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, N_FACTS, READ_LR, READ_STEPS, acc_by_question, build_chunks,
                       build_facts, dump, key_matrices, make_eval_items, make_stack,
                       model_and_tok, nll_of_candidate, write_phase)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GAMMA = 0.05

def patch_tied(st):
    orig = {li: m.read for li, m in st.mems.items()}
    for li, m in st.mems.items():
        def make(mem):
            def tied(hs_in, drop_sg=False):
                cf = hs_in.detach().reshape(-1, hs_in.shape[-1])
                q = (cf @ mem.Pk).detach() * float(mem.pk_scale)
                W = (mem.W0 + mem.U @ mem.V.T) if mem.use_factorized else mem.Wfull
                mm = (mem.Wo @ ((W @ q.T).T).T).T
                mem.injection = mm.reshape(hs_in.shape)
                return mem.injection
            return tied
        m.read = make(m)
    return orig

def train_wo_gamma(model, st, tok, facts, keys, rng, steps=READ_STEPS):
    ps = []
    for m in st.mems.values():
        ps += [m.Wo, m.gamma]
    opt = torch.optim.Adam(ps, lr=READ_LR)
    batches = []
    for ki in keys:
        f = facts[ki]
        others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != f["val"]]
        cands = [f["val"]] + rng.sample(sorted(set(others)), 3)
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
        loss.backward(); opt.step(); losses.append(float(loss))
    return losses

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok); chunks, owner = build_chunks(tok, facts)
    order = list(range(N_FACTS)); random.Random(777).shuffle(order)
    res = {"seed": a.seed, "ks": [4, 8, 16, 24], "rows": {}}
    print(f"N_FACTS={N_FACTS}", flush=True)
    for k in res["ks"]:
        tr, ev_ = order[:k], order[k:]
        st = make_stack(gamma=GAMMA, no_write=False, seed=a.seed)
        write_phase(model, st, chunks)
        patch_tied(st)
        losses = train_wo_gamma(model, st, tok, facts, tr, random.Random(1000))
        a_ev, q_ev = ev(model, st, tok, facts, ev_)
        a_tr, q_tr = ev(model, st, tok, facts, tr, seed=556)
        row = {"k_train": k, "n_eval": len(ev_), "final_loss": float(np.mean(losses[-50:])),
               "acc_eval": a_ev, "accq_eval": q_ev, "acc_train": a_tr, "accq_train": q_tr}
        res["rows"][str(k)] = row
        print(f"  k={k:<3} n_eval={len(ev_):<3} loss={row['final_loss']:.2e} | "
              f"训练键 {a_tr:.3f}/{q_tr:.3f} | 留出键 {a_ev:.3f}/{q_ev:.3f}", flush=True)
        dump(res, "probe91_scale.json")
        st.remove()
    print("\n★ 诊断完成（不产生裁决）", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
