#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""DIR-91 · rev2.4 **在 $\gamma{=}0$ 下重测解码门**（补 rev1.9 理由 (a) 的 $\gamma$ 混淆）

**背景**（`DEF-91-22`(E)）：rev1.9 的门 FAIL 理由 (a)「答案在查询位不可解（$z{=}+0.01$）」
出自 `probe91_gate.py`，其 `GAMMA=0.05`（第 30 行）⇒ **$\gamma\neq0$ 混淆**。
**另**：该脚本第 101 行的 $W$ 用了 `W0 + U@V.T`（**冻结臂**，见 `DEF-91-23`），
**但**解码门（第 123–127 行）只用 `st._hs_in` 与 `e_ans`，**完全不碰 $W$** ⇒ **理由 (a) 不受冻结 $W$ 影响**，
**只**受 $\gamma$ 影响。本探针只补 $\gamma$。

**设计**：**同一脚本**内跑三臂——
  ① $\gamma{=}0$，真写；② $\gamma{=}0$，`no_write=True`；③ $\gamma{=}0.05$，真写。
臂 ① 与 ② **必须逐位相同**（漂移恒为 0 的端到端检验，`DEF-91-20`(2)）；
若 ①$\neq$② 则说明有别的 $W\to$ 隐态耦合通路，须停。

★ 只测量；无裁决、无证据行、不改冻结阈值。
"""
import argparse, json, os, random, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, MOUNT, N_FACTS, N_TRAIN_KEYS, build_chunks, build_facts,
                       dump, make_stack, model_and_tok, write_phase)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def ridge_cv(X, Y, folds=4, lam=1e-2):
    n = len(X); idx = np.arange(n); pred = torch.zeros_like(Y)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        Xtr, Ytr = X[tr], Y[tr]
        G = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=Xtr.device)
        A = torch.linalg.solve(G, Xtr.T @ Ytr)
        pred[te] = X[te] @ A
    rel = float(((pred - Y).norm(dim=1) / (Y.norm(dim=1) + 1e-30)).mean())
    c = float(torch.nn.functional.cosine_similarity(pred, Y, dim=1).mean())
    return rel, c


def arm(tok, model, facts, chunks, gamma, no_write, seed):
    """返回 (HP, HA, HB) 与写入次数；HP/HA/HB 只来自 st._hs_in，不碰 W。"""
    st = make_stack(gamma=gamma, no_write=no_write, seed=seed)
    try:
        for m in st.mems.values():
            m.prev_hs = None
        write_phase(model, st, chunks)
        nw = int(sum(m.write_calls for m in st.mems.values()) // max(len(st.mems), 1))
        HP = {li: [] for li in MOUNT}; HA = {li: [] for li in MOUNT}
        for fi in range(N_FACTS):
            f = facts[fi]
            with torch.no_grad():
                model(input_ids=torch.tensor([f["prompt_ids"]], dtype=torch.long, device=DEV))
            for li in MOUNT:
                hs = st._hs_in[li][0]
                HP[li].append(hs[-1].float()); HA[li].append(hs.mean(0).float())
        HB = {li: [] for li in MOUNT}
        for fi in range(N_FACTS):
            f = facts[fi]
            ids = tok(f["prompt"] + " " + f["val"], add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], dtype=torch.long, device=DEV))
            for li in MOUNT:
                HB[li].append(st._hs_in[li][0, -1].float())
        return HP, HA, HB, nw
    finally:
        st.remove()


def decode(HP, HA, HB, e_ans, tag_out):
    out = {}
    for li in MOUNT:
        for tag, H in (("prompt_last", HP), ("prompt_mean", HA), ("answer_last", HB)):
            X = torch.stack(H[li]); Y = e_ans
            rel, c = ridge_cv(X, Y)
            rng = random.Random(4242); nulls = []
            for _ in range(40):
                perm = list(range(N_FACTS)); rng.shuffle(perm)
                _, cl = ridge_cv(X, Y[perm]); nulls.append(cl)
            out[f"L{li}|{tag}"] = {
                "rel_err": rel, "cos": c,
                "null_cos_mean": float(np.mean(nulls)), "null_cos_sd": float(np.std(nulls)),
                "z": float((c - np.mean(nulls)) / (np.std(nulls) + 1e-30))}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok); chunks, owner = build_chunks(tok, facts)
    e_ans = model.get_input_embeddings().weight[
        torch.tensor([facts[fi]["val_ids"][0] for fi in range(N_FACTS)], device=DEV)
    ].detach().float()

    res = {"seed": a.seed, "note": "decodability only; no W used; three arms in ONE script",
           "dtype": "float32", "guard": {}, "arms": {}}
    arms = [("g0_write", 0.0, False), ("g0_nowrite", 0.0, True), ("g05_write", 0.05, False)]
    got = {}
    for name, g, nw_flag in arms:
        HP, HA, HB, nw = arm(tok, model, facts, chunks, g, nw_flag, a.seed)
        got[name] = (HP, HA, HB)
        res["arms"][name] = {"gamma": g, "no_write": nw_flag, "n_writes": nw,
                             "decodability": decode(HP, HA, HB, e_ans, name)}
        d = res["arms"][name]["decodability"]
        print(f"\n### {name}  (gamma={g}, no_write={nw_flag}, writes={nw})", flush=True)
        for li in MOUNT:
            k = f"L{li}|prompt_last"
            print(f"  L{li} prompt_last z={d[k]['z']:+.2f} cos={d[k]['cos']:+.4f} "
                  f"rel={d[k]['rel_err']:.3f} | answer_last z={d[f'L{li}|answer_last']['z']:+.2f}",
                  flush=True)

    # ★ 守卫：γ=0 下 write 臂与 no_write 臂的隐态必须逐位相同（漂移恒 0 的端到端检验）
    dmax = 0.0
    for li in MOUNT:
        for i in range(N_FACTS):
            for A, B in ((got["g0_write"][0][li][i], got["g0_nowrite"][0][li][i]),
                         (got["g0_write"][2][li][i], got["g0_nowrite"][2][li][i])):
                dmax = max(dmax, float((A - B).abs().max()))
    res["guard"]["max_abs_hs_diff_g0_write_vs_nowrite"] = dmax
    res["guard"]["pass"] = (dmax == 0.0)
    print(f"\n★ 守卫 γ=0 write v.s. no_write 隐态 max|Δ| = {dmax:.3e} "
          f"=> {'PASS（漂移≡0 端到端成立）' if dmax == 0.0 else 'FAIL（存在别的 W→隐态通路，须停）'}",
          flush=True)

    dump(res, "probe91_decod2.json")
    print("\n★ 完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
