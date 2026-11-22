#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 · rev 1.8 追加对照：**幅值与方向，哪一个才是缺陷？**（诊断，不产生裁决）

`probe91_stub.py` 已给：
  S-A e_ans 在 rho>=1.0 得 1.000；rho=0.5 得 0.875(题)
  S-B e_rand 全 0.000（tied lm_head 下它抬高的是干扰项的 logit）
  M5 实际全绑定臂 ||Δ||/||h|| = 0.127~0.151，且 |cos(Δ,e_ans)| = 0.010~0.028
     ≈ 随机单位向量期望值 sqrt(2/(pi*896)) = 0.0267

剩下唯一含糊处：Δ 的**幅值只有 0.13**，而桩在 rho=0.5 就已 0.875。
故须问：**同样 0.13 的幅值，若方向对齐，能不能答对？**
  A  rho = 0.13 且 e_ans  （幅值 = 全绑定臂，方向理想）
  B  rho = 0.13 且 e_rand （幅值同上，方向随机）→ 与 A 对照即「方向的贡献」
  C  rho = 0.25 且 e_ans  （幅值再加倍，看曲线）
若 A 已明显高于 BASE ⇒ 缺陷主因是**方向**（幅值够）；
若 A 仍 ≈ BASE     ⇒ 缺陷主因是**幅值**（方向再好也不够）。

★ 只补 3 个配置 × 2 个层组；不改任何已冻结判据。
"""
import argparse, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, N_FACTS, N_TRAIN_KEYS, acc_by_question, build_chunks,
                       build_facts, dump, make_eval_items, make_stack, model_and_tok,
                       nll_of_candidate)
from probe91_stub import STUB, embed_of, install_stub, silence_read, correct_tid

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok)
    order = list(range(N_FACTS)); random.Random(777).shuffle(order)
    eval_keys = order[N_TRAIN_KEYS:]
    items = make_eval_items(tok, facts, eval_keys, random.Random(555))
    tid_ans = [correct_tid(it) for it in items]
    tid_rand = []
    for it in items:
        others = [c[0][0] for c in it["cand"] if not c[1]]
        tid_rand.append(others[len(tid_rand) % max(1, len(others))])

    res = {"seed": a.seed, "arms": {}}
    CFG = [("A_ans", tid_ans, 0.13), ("B_rand", tid_rand, 0.13),
           ("C_ans", tid_ans, 0.25), ("D_ans", tid_ans, 0.35)]
    for tag, tids, rho in CFG:
        for lset in ((16,), (8, 12, 16)):
            st = make_stack(gamma=1.0, no_write=True, seed=a.seed)
            install_stub(st, lset); silence_read(st, layers=set(lset))
            det = []
            for it, t in zip(items, tids):
                STUB["e"] = embed_of(model, t)
                STUB["idx"] = len(it["prompt_ids"]) - 1
                STUB["rho"] = rho
                nlls = [nll_of_candidate(model, st, it["prompt_ids"], c[0]) for c in it["cand"]]
                det.append({"ok": bool(it["cand"][int(np.argmin(nlls))][1])})
            ai = float(np.mean([d["ok"] for d in det])); aq = acc_by_question(items, det)[0]
            k = f"{tag}|rho={rho}|L={lset}"
            res["arms"][k] = {"acc_item": ai, "acc_q": aq}
            print(f"  {tag:<7} rho={rho:<5} L={str(lset):<11} acc(项)={ai:.3f} acc(题)={aq:.3f}",
                  flush=True)
            dump(res, "probe91_stub2.json")
            STUB["e"] = None; st.remove()
    print("\n★ 诊断完成（不产生裁决）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
