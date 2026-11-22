#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 · rev 1.9 **解码可行性门**（诊断，不产生裁决）

rev1.9 要求先过此门再谈 E1/E2/E3。本探针同时回答三件事：

(1) **编码/查询是否同一个量**（本轮新发现的候选真因）
    `sgmn_module.write`：`k = mean_over_all_tokens(Pk·prev_hs)*pk_scale`，
    `v = mean_over_all_tokens(Pv·hs_in)` —— **每个 chunk 只存 1 对「序列均值」**。
    而 `read`：`q = Wq·c`，`m = Wo·W·q` —— **逐 token 注入**，用的是单个 token 的 c。
    ⇒ 训练对是 (mean_k → mean_v)，查询却是 W·q_token ⇒ **查询落在 W 从未训练的区域**。
    ★ 若成立，则 rev1.7 的 `cos(W k_i, v_i)=0.98` 只是把**被最小化的损失**当证据报了出来
      （`G = alpha(Wk - v)k^T + lam(W-W0)` 就是在拟合这一对），**不是检索能力**。

(2) **形式化的解码门**：值 token 身份能否从键的隐态在**留出键**上线性解出（ridge + CV）。
    对照：标签置换（打乱 fact→value 对应）⇒ 同口径零分布。

(3) **值 token 是否在提示位可用**：提示末位**尚未生成答案**，故 e_ans 不可能在该位可解；
    作为阳性对照，在**答案位**上重解一次（应当显著可解）。

★ 只做测量，不改判据、不产生裁决。
"""
import argparse, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, N_FACTS, N_TRAIN_KEYS, MOUNT, build_chunks, build_facts, dump,
                       key_matrices, make_stack, model_and_tok, write_phase)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GAMMA = 0.05


def cos(a, b):
    a = a.detach().float().reshape(-1); b = b.detach().float().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def ridge_cv(X, Y, folds=4, lam=1e-2):
    """k-fold：在训练折解 A（最小二乘），在留出折报 相对误差 与 余弦。"""
    n = len(X)
    idx = np.arange(n)
    pred = torch.zeros_like(Y)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        Xtr, Ytr = X[tr], Y[tr]
        # A^T = (Xtr^T Xtr + lam I)^-1 Xtr^T Ytr
        G = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=Xtr.device)
        A = torch.linalg.solve(G, Xtr.T @ Ytr)          # (d_in, d_out)
        pred[te] = X[te] @ A
    rel = float(((pred - Y).norm(dim=1) / (Y.norm(dim=1) + 1e-30)).mean())
    c = float(torch.nn.functional.cosine_similarity(pred, Y, dim=1).mean())
    return rel, c, pred


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    order = list(range(N_FACTS)); random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]
    res = {"seed": a.seed, "layers": MOUNT}

    # ---------- 训练装置（真写），取得训练后的 W 与 chunk 均值 (K,V)
    st = make_stack(gamma=GAMMA, no_write=False, seed=a.seed)
    write_phase(model, st, chunks)
    K, V = key_matrices(model, st, facts, chunks, owner)

    # ---------- 提示位 / 答案位的隐态（no_write，不污染 W）
    for m in st.mems.values():
        m.no_write = True
    HP, HA, QP = {li: [] for li in MOUNT}, {li: [] for li in MOUNT}, {li: [] for li in MOUNT}
    for fi in range(N_FACTS):
        f = facts[fi]
        with torch.no_grad():
            model(input_ids=torch.tensor([f["prompt_ids"]], dtype=torch.long, device=DEV))
        for li in MOUNT:
            hs = st._hs_in[li][0]                       # (L_prompt, d)
            HP[li].append(hs[-1].float())               # 提示末位
            QP[li].append(((hs[-1] @ st.mems[li].Pk) * st.mems[li].pk_scale).float())
            HA[li].append(hs.mean(0).float())           # 阳性对照用：提示均值（答案位另跑）
    # 答案位：跑「提示+答案」
    HB = {li: [] for li in MOUNT}
    for fi in range(N_FACTS):
        f = facts[fi]
        ids = tok(f["prompt"] + " " + f["val"], add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], dtype=torch.long, device=DEV))
        for li in MOUNT:
            HB[li].append(st._hs_in[li][0, -1].float())  # 答案末位
    for m in st.mems.values():
        m.no_write = False

    e_ans = model.get_input_embeddings().weight[
        torch.tensor([facts[fi]["val_ids"][0] for fi in range(N_FACTS)], device=DEV)].detach().float()

    res["encode_query"] = {}
    for li in MOUNT:
        m = st.mems[li]
        W = (m.W0 + m.U @ m.V.T).detach().float()
        rows = []
        for fi in range(N_FACTS):
            k_i, v_i = K[li][fi], V[li][fi]
            q_i = QP[li][fi]
            Wk, Wq = W @ k_i, W @ q_i
            rows.append({"fi": fi,
                         "cos_k_q": cos(k_i, q_i),                 # 编码键 v.s. 查询键
                         "cos_Wk_v": cos(Wk, v_i),                 # ★ 被最小化的量（tautology）
                         "cos_Wq_v": cos(Wq, v_i),                 # ★ 真检索
                         "cos_Wq_eans": cos(Wq, e_ans[fi]),        # 真检索 → 答案几何
                         "rel_parallel": float((q_i @ k_i) / (k_i @ k_i + 1e-30))})
        agg = {key: float(np.mean([r[key] for r in rows]))
               for key in ("cos_k_q", "cos_Wk_v", "cos_Wq_v", "cos_Wq_eans")}
        agg["n"] = len(rows)
        res["encode_query"][li] = agg
        print(f"\n L{li}: cos(k_chunk_mean, q_prompt_token) = {agg['cos_k_q']:+.4f}", flush=True)
        print(f"   cos(W k_i, v_i)   = {agg['cos_Wk_v']:+.4f}   ← 被最小化的量（tautology）", flush=True)
        print(f"   ★ cos(W q_i, v_i) = {agg['cos_Wq_v']:+.4f}   ← 真检索", flush=True)
        print(f"   ★ cos(W q_i, e_ans)= {agg['cos_Wq_eans']:+.4f}", flush=True)
        dump(res, "probe91_gate.json")

    # ---------- (2) 形式化解码门：e_ans 能否从提示位隐态在留出键上线性解出
    res["decodability"] = {}
    for li in MOUNT:
        for tag, H in (("prompt_last", HP[li]), ("prompt_mean", HA[li]), ("answer_last", HB[li])):
            X = torch.stack(H); Y = e_ans
            rel, c, _ = ridge_cv(X, Y)
            # 标签置换零分布
            rng = random.Random(4242); nulls = []
            for _ in range(40):
                perm = list(range(N_FACTS)); rng.shuffle(perm)
                rl, cl, _ = ridge_cv(X, Y[perm])
                nulls.append(cl)
            res["decodability"][f"L{li}|{tag}"] = {
                "rel_err": rel, "cos": c,
                "null_cos_mean": float(np.mean(nulls)), "null_cos_sd": float(np.std(nulls)),
                "z": float((c - np.mean(nulls)) / (np.std(nulls) + 1e-30))}
            print(f" [解码] L{li} {tag:<12} rel={rel:.3f} cos={c:+.4f} "
                  f"零={np.mean(nulls):+.4f}±{np.std(nulls):.4f} z="
                  f"{(c-np.mean(nulls))/(np.std(nulls)+1e-30):+.2f}", flush=True)
            dump(res, "probe91_gate.json")
    st.remove()
    print("\n★ 诊断完成（不产生裁决）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
