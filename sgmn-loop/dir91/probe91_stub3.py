#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 · rev 1.8 幅值重定标：在**真实**注入幅值上重测对齐桩（诊断，不产生裁决）

`probe91_align2.py` 用 hook 抓真实注入后修正：装置真实 ||Δ||/||h|| = **0.049/0.052/0.054**，
而非解析式给出的 0.13（解析式用了被后续写入改过的最终 W，rel.err 1.08）。
故 `probe91_stub2.py` 在 rho=0.13 上得到的 0.438**不是**装置实际幅值下的读数，须重测。

本探针在 rho ∈ {0.03, 0.049, 0.08, 0.13} 上跑：
  A  e_ans（对齐）      B  e_wrong（同题干扰候选 token，非随机方向——改名，见更正节）
  G  e_gauss（真正各向同性随机方向，新范数匹配）—— 用于区分「语义错」与「方向随机」
若 A 在 rho≈0.049 仍明显高于 BASE ⇒ 幅值不是限制、方向是；
若 A 在 rho≈0.049 已回落 BASE 附近 ⇒ 幅值也构成限制，rev1.8 的「幅值足够」**不成立**。

★ 只做测量，不改判据。
"""
import argparse, math, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, N_FACTS, N_TRAIN_KEYS, acc_by_question, build_facts, dump,
                       make_eval_items, make_stack, model_and_tok, nll_of_candidate)
from probe91_stub import STUB, embed_of, install_stub, silence_read, correct_tid

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def run(model, items, tids, rho, lset, seed, gauss=None):
    st = make_stack(gamma=1.0, no_write=True, seed=seed)
    install_stub(st, lset); silence_read(st, layers=set(lset))
    det = []
    for n, (it, t) in enumerate(zip(items, tids)):
        if gauss is not None:
            gen = torch.Generator(device="cpu"); gen.manual_seed(1000 + n)
            STUB["e"] = torch.randn(896, generator=gen).to(DEV)
        else:
            STUB["e"] = embed_of(model, t)
        STUB["idx"] = len(it["prompt_ids"]) - 1
        STUB["rho"] = rho
        nlls = [nll_of_candidate(model, st, it["prompt_ids"], c[0]) for c in it["cand"]]
        det.append({"ok": bool(it["cand"][int(np.argmin(nlls))][1])})
    ai = float(np.mean([d["ok"] for d in det])); aq = acc_by_question(items, det)[0]
    STUB["e"] = None; st.remove()
    return ai, aq


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok)
    order = list(range(N_FACTS)); random.Random(777).shuffle(order)
    eval_keys = order[N_TRAIN_KEYS:]
    items = make_eval_items(tok, facts, eval_keys, random.Random(555))
    tid_ans = [correct_tid(it) for it in items]
    tid_wrong = []
    for it in items:
        others = [c[0][0] for c in it["cand"] if not c[1]]
        tid_wrong.append(others[len(tid_wrong) % max(1, len(others))])

    res = {"seed": a.seed, "base": {"acc_item": 0.250, "acc_q": 0.125}, "arms": {}}
    print("BASE = 0.250 项 / 0.125 题（前轮实测）；装置真实幅值 0.049~0.054", flush=True)
    for rho in (0.03, 0.049, 0.08, 0.13):
        for tag, tids, gauss in (("A_ans", tid_ans, None),
                                 ("B_wrong", tid_wrong, None),
                                 ("G_gauss", tid_ans, True)):
            ai, aq = run(model, items, tids, rho, (8, 12, 16), a.seed, gauss)
            res["arms"][f"{tag}|rho={rho}"] = {"acc_item": ai, "acc_q": aq}
            print(f"  {tag:<8} rho={rho:<6} acc(项)={ai:.3f} acc(题)={aq:.3f}", flush=True)
            dump(res, "probe91_stub3.json")
    print("\n★ 诊断完成（不产生裁决）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
