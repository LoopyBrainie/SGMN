#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 rev 2.3 · 输出映射的可辨识性（task: DIR-91-rev2.5）

问题（由 D-104 判决 A 派生）：同一 W_o、只换 q 时准确率仅 0.125/0.188 ⇒
  「v_i -> 答案 token」的映射能否从 16 个训练键学到并**泛化**到 16 个留出键？

三臂（每 seed；写阶段后 W 冻结，三臂同初值/同步数/同数据顺序）：
  real        q = Wq c（原生），训 600 步           基线
  oracleP1    q = k_i，**仅读出位 (P-1)**，训 600 步   主臂（修 DEF-91-33 的常量偏置）
  oracleALL   q = k_i，**全位置**，训 600 步          对照（复现 DEF-91-33）

主判决（§1，冻结）：按题准确率 acc_q（n_q=16，chance=0.25），每臂在
  **同一份权重**上评两次：train_keys（fit）与 heldout_keys（gen）。
  分支：GENERALIZES(held>=.50) / FIT_ONLY(train>=.50 & held<=.25) /
        BOTH_LOW(train<.50 & held<=.25) / PARTIAL
  H1 = FIT_ONLY（"查表而非规则"）。两 seed 不一致 => INCONCLUSIVE。

解析量（无需训练）：M2 = V->答案类的岭回归可辨识性（必要条件/上限）；
  M3 = Wo 的参与比（有效秩）；M4 = 恒等读出下 argmax_c (u_c · v_i) 的命中率。

守卫 A91-40..47（任一 FAIL 阻断正式轮）。★ 本文件**不修改** sgmn_module.py /
R2_task91.py：读通路的 oracle 替换是本文件内的 monkey-patch。
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

import sgmn_module
from sgmn_module import LayerMem, sg
from R2_task91 import (ALPHA, DEV, ETA_WRITE, LAM, N_CAND, N_DRAWS, N_FACTS,
                       N_TRAIN_KEYS, OUT, READ_LR, READ_STEPS, W_of, acc_by_question,
                       build_chunks, build_facts, key_matrices, make_eval_items,
                       make_stack, model_and_tok, nll_of_candidate, read_params,
                       score_and_accuracy, train_read_task, write_params,
                       write_phase)

RES = {"card": "DIR-91 rev 2.3", "task": "DIR-91-rev2.5",
       "alpha": ALPHA, "lam": LAM, "eta": ETA_WRITE, "read_steps": READ_STEPS,
       "read_lr": READ_LR, "n_train_keys": N_TRAIN_KEYS, "n_cand": N_CAND,
       "n_draws": N_DRAWS, "guards": {}, "arms": [], "analytic": {}}
FAILS = []

# §1 判决门（冻结）
GATE_FIT = 0.50
GATE_GEN = 0.50
GATE_LOW = 0.25
CHANCE = 1.0 / N_CAND


def rec(name, ok, detail, value=None):
    RES["guards"][name] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


# --------------------------------------------- 读通路：q 的 oracle 替换（可定位）
_ORIG_READ = LayerMem.read


def _read_patched(self, hs_in, drop_sg=False):
    """与 sgmn_module.py:65 逐字相同；**唯一差别** = 指定位置(或全部)的 q 可被替换。

    _q_oracle: (d_k,) 替换值；_q_pos: None=全位置替换，int=只替换该行（读出位）。
    ★ DEF-91-33 的修复：pos 为 int 时，**其余行与原生逐位相同**（A91-41 断言）。
    """
    ov = getattr(self, "_q_oracle", None)
    if ov is None:
        return _ORIG_READ(self, hs_in, drop_sg)          # 无替换 => 原生，保证逐位相同
    self.read_calls += 1
    c = sg(hs_in, drop_sg)
    cf = c.reshape(-1, c.shape[-1])
    W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
    q = (self.Wq @ cf.T).T
    ovv = ov.to(device=cf.device, dtype=cf.dtype)
    pos = getattr(self, "_q_pos", None)
    if pos is None:
        q = ovv.unsqueeze(0).expand(cf.shape[0], -1)
    else:
        # ★ 哨兵 "readout" => 倒数第二行（prompt 末位；其后恰有 1 个候选 token）。
        #   不得用 bool/True 当索引：torch 会把标量布尔当**布尔掩码**选中**所有**行。
        idx = (cf.shape[0] - 2) if pos == "readout" else int(pos)
        assert isinstance(idx, int) and not isinstance(pos, bool), pos
        q = q.clone()
        q[idx] = ovv
    y = (sg(W, drop_sg) @ q.T).T
    m = (self.Wo @ y.T).T
    self.injection = m.reshape(hs_in.shape)
    return self.injection


def set_oracle(st, K, fi, pos=None):
    for li, m in st.mems.items():
        m._q_oracle = None if fi is None else K[li][fi]
        m._q_pos = pos


def clear_oracle(st):
    for m in st.mems.values():
        m._q_oracle = None
        m._q_pos = None


def snapshot(st):
    """(W, Wq, Wo, gamma) 的逐位快照。"""
    d = {}
    for li, m in st.mems.items():
        d[li] = {"W": W_of(m).detach().clone(), "Wq": m.Wq.detach().clone(),
                 "Wo": m.Wo.detach().clone(), "gamma": m.gamma.detach().clone()}
    return d


def maxdiff_snap(a, b):
    return max(float((a[li][k] - b[li][k]).abs().max())
               for li in a for k in a[li])


# --------------------------------------------- 训练 / 评测（带可定位 oracle）
def train_read_oracle(model, st, tok, facts, train_keys, rng, K, pos):
    """= train_read_task 逐字；**唯一差别** = 前向前把 q 置为 k_i（可只置读出位）。"""
    for m in st.mems.values():
        m.no_write = True
    opt = torch.optim.Adam(read_params(st), lr=READ_LR)
    losses, violations, diverged = [], 0, False
    batches = []
    for ki in train_keys:
        f = facts[ki]
        others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != f["val"]]
        distr = rng.sample(sorted(set(others)), N_CAND - 1)
        cands = [f["val"]] + distr
        rng.shuffle(cands)
        ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in cands]
        batches.append((ki, f["prompt_ids"], ids, cands.index(f["val"])))
    for step in range(READ_STEPS):
        ki, prompt_ids, ids, tgt = batches[step % len(batches)]
        P = len(prompt_ids)
        seq = torch.tensor([prompt_ids + [c] for c in ids], dtype=torch.long, device=DEV)
        opt.zero_grad(set_to_none=True)
        set_oracle(st, K, ki, pos)
        try:
            lg = model(input_ids=seq).logits.float()
        finally:
            clear_oracle(st)
        assert seq.shape[1] == P + 1 and seq.shape[1] - 2 == P - 1, "读出位约定失效"
        idx = torch.arange(len(ids), device=DEV)
        pick_logits = lg[idx, P - 1, torch.tensor(ids, device=DEV)].unsqueeze(0)
        assert pick_logits.shape == (1, len(ids)), pick_logits.shape
        loss = torch.nn.functional.cross_entropy(pick_logits, torch.tensor([tgt], device=DEV))
        loss.backward()
        if step % 50 == 0:
            for p in write_params(st):
                if p.grad is not None and float(p.grad.abs().max()) != 0.0:
                    violations += 1
        torch.nn.utils.clip_grad_norm_([p for p in read_params(st) if p.grad is not None], 1.0)
        opt.step()
        losses.append(float(loss))
        if not all(bool(torch.isfinite(W_of(m)).all()) for m in st.mems.values()):
            diverged = True
            break
    for m in st.mems.values():
        m.no_write = False
    return losses, violations, diverged


def eval_keys(model, st, tok, facts, keys, K=None, use_oracle=False, pos=None):
    """= run_arm 的评测段；use_oracle 时按 prompt 反查事实下标并置 q=k_i（可只置读出位）。"""
    for m in st.mems.values():
        m.no_write = True
    items = make_eval_items(tok, facts, keys, __import__("random").Random(555))
    p2fi = {tuple(f["prompt_ids"]): i for i, f in enumerate(facts)}
    if use_oracle:
        def scorer(p, cs):
            fi = p2fi[tuple(p)]
            set_oracle(st, K, fi, pos)
            try:
                return [nll_of_candidate(model, st, p, c) for c in cs]
            finally:
                clear_oracle(st)
    else:
        scorer = lambda p, cs: [nll_of_candidate(model, st, p, c) for c in cs]   # noqa: E731
    acc, _, det = score_and_accuracy(items, scorer)
    acc_q, nq = acc_by_question(items, det)
    margins = []
    for it, dd in zip(items, det):
        nlls = np.array(dd["nll"], dtype=float)
        ok_idx = [j for j, c in enumerate(it["cand"]) if c[1]]
        bad = [j for j in range(len(nlls)) if j not in ok_idx]
        margins.append(float(min(nlls[b] for b in bad) - nlls[ok_idx[0]]))
    return {"acc_item": acc, "acc_q": acc_q, "n_q": nq, "n_item": len(items),
            "margin_mean": float(np.mean(margins)),
            "margin_pos_frac": float(np.mean([m > 0 for m in margins]))}


# --------------------------------------------- 解析量 M2/M3/M4
def ridge_scores(Xtr, ytr, Xte, nclass, lam=1.0):
    """对偶形式岭回归到 one-hot：W = Xc^T (Xc Xc^T + lam I)^{-1} Y。返回 (nte,nclass)。"""
    Xtr = Xtr.detach().to("cpu", torch.float64)
    Xte = Xte.detach().to("cpu", torch.float64)
    mu = Xtr.mean(dim=0, keepdim=True)
    Xc, Xt = Xtr - mu, Xte - mu
    Y = torch.zeros(len(ytr), nclass, dtype=Xtr.dtype)
    Y[torch.arange(len(ytr)), torch.as_tensor(ytr)] = 1.0
    Kmat = Xc @ Xc.T
    A = torch.linalg.solve(Kmat + lam * torch.eye(len(ytr), dtype=Xtr.dtype), Y)
    return Xt @ Xc.T @ A


def top1(scores, ytrue):
    pred = scores.argmax(dim=1).tolist()
    return float(np.mean([int(p) == int(t) for p, t in zip(pred, ytrue)])), pred


def participation_ratio(M):
    s = torch.linalg.svdvals(M.float())
    return float((s.sum() ** 2) / (s ** 2).sum())


def class_ids(facts):
    """答案类 = 值 token 的去重排序下标（VALUE_POOL 有 27 值 / 32 事实 ⇒ 不是 1:1）。"""
    vals = sorted({f["val"] for f in facts})
    v2c = {v: i for i, v in enumerate(vals)}
    return [v2c[f["val"]] for f in facts], len(vals)


# ===================================================== Stage A
def stage_a(tok, model, facts, chunks, owner):
    print("\n===== Stage A 守卫 =====\n", flush=True)
    seed = 20261028
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    try:
        write_phase(model, st, chunks)
        K, V = key_matrices(model, st, facts, chunks, owner)
        order = list(range(N_FACTS))
        __import__("random").Random(777).shuffle(order)
        tr, he = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]

        # ---- A91-40：无替换时与原生逐位相同
        for m in st.mems.values():
            m.no_write = True
        LayerMem.read = _ORIG_READ
        model(input_ids=chunks[0].to(DEV))
        m_orig = {li: m.injection.detach().clone() for li, m in st.mems.items()}
        LayerMem.read = _read_patched
        clear_oracle(st)
        model(input_ids=chunks[0].to(DEV))
        d = max(float((m.injection - m_orig[li]).abs().max()) for li, m in st.mems.items())
        rec("A91-40", d == 0.0,
            f"打补丁的读通路在**无替换**时与原生**逐位相同**：max|Δm|={d:.1e}（须 ==0）",
            value={"d": d})

        # ---- A91-41 ★ DEF-91-33 的修复：只读出位不同，其余逐位相同
        ids0 = chunks[0].to(DEV)
        N = int(ids0.shape[0]) if ids0.dim() == 1 else int(ids0.shape[-1])
        pos = N - 1                                   # 末位当作"读出位"做结构断言
        LayerMem.read = _ORIG_READ
        model(input_ids=ids0)
        m_orig = {li: m.injection.detach().clone() for li, m in st.mems.items()}
        LayerMem.read = _read_patched
        set_oracle(st, K, 0, "readout")
        try:
            model(input_ids=ids0)
            rpos = N - 2                      # 哨兵 "readout" 应命中的行
            d_pos = max(float((m.injection[:, rpos] - m_orig[li][:, rpos]).abs().max())
                        for li, m in st.mems.items())
            mask = torch.ones(N, dtype=torch.bool)
            mask[rpos] = False
            d_oth = max(float((m.injection[:, mask] - m_orig[li][:, mask]).abs().max())
                        for li, m in st.mems.items())
        finally:
            clear_oracle(st)
        # 对照：全位置替换时"其余位置"必然也不同（证明上一条不是无信息量的 0）
        set_oracle(st, K, 0, None)
        try:
            model(input_ids=ids0)
            mask2 = torch.ones(N, dtype=torch.bool)
            mask2[rpos] = False
            d_all_oth = max(float((m.injection[:, mask2] - m_orig[li][:, mask2]).abs().max())
                            for li, m in st.mems.items())
        finally:
            clear_oracle(st)
        rec("A91-41", (d_pos > 0) and (d_oth == 0.0) and (d_all_oth > 0),
            f"★ `DEF-91-33` 修复被实测断言（哨兵 \"readout\" ⇒ 行 {N - 2}）："
            f"**只读出位**替换时该位 max|Δm|={d_pos:.2e}（须 >0）、"
            f"**其余位置 max|Δm|={d_oth:.1e}（须 ==0）**；对照「全位置替换」下其余位置 "
            f"max|Δm|={d_all_oth:.2e}（须 >0，证明上一条的 0 非无信息量）",
            value={"d_pos": d_pos, "d_other": d_oth, "d_all_other": d_all_oth})

        # ---- A91-42：M2 通路天花板（合成数据，**每类都在训练集出现**）
        ncls = 27
        g = torch.Generator().manual_seed(5)
        oh = torch.eye(ncls, dtype=torch.float64)
        ytr_c = [c for c in range(ncls)]
        Xoh_tr, Xoh_te, yoh_te = oh.clone(), oh.clone(), list(range(ncls))
        s_oh = ridge_scores(Xoh_tr, ytr_c, Xoh_te, ncls)
        acc_oh, _ = top1(s_oh, yoh_te)
        rng_p = np.random.default_rng(11)
        y_perm = rng_p.permutation(ytr_c).tolist()
        Xr = torch.randn(ncls, 32, dtype=torch.float64, generator=g)
        Xtr_r, Xte_r = Xr[: ncls // 2 + 1], Xr[ncls // 2 + 1:]
        s_pm = ridge_scores(Xtr_r, y_perm[: len(Xtr_r)], Xte_r, ncls)
        acc_pm, _ = top1(s_pm, y_perm[len(Xtr_r):])
        rec("A91-42", (abs(acc_oh - 1.0) < 1e-12) and (acc_pm <= GATE_LOW),
            f"M2 通路天花板：**one-hot（每类都在训练集）⇒ 留出 top-1={acc_oh:.12f}（须 ==1.0）**；"
            f"随机置换标签 ⇒ {acc_pm:.3f}（须 ≤{GATE_LOW}）",
            value={"acc_onehot": acc_oh, "acc_permuted": acc_pm, "n_class": ncls})

        # ---- A91-43：acc_by_question 已算例（k 题全对 => 恰好 k/n）
        fake, kk, nq = [], 3, 8
        for qi in range(nq):
            for dr in range(N_DRAWS):
                fake.append({"key_idx": qi})
        det_fake = [{"ok": (i // N_DRAWS) < kk} for i in range(len(fake))]
        aq, nq_got = acc_by_question(fake, det_fake)
        rec("A91-43", (abs(aq - kk / nq) < 1e-12) and (nq_got == nq),
            f"`acc_by_question` 已算例：构造 {kk}/{nq} 题全对 ⇒ 返回 {aq:.6f}（须 == {kk / nq:.6f}）、"
            f"n_q={nq_got}（须 =={nq}）", value={"acc": aq, "n_q": nq_got})

        # ---- A91-44：键划分
        inter = set(tr) & set(he)
        rec("A91-44", (not inter) and len(tr) == 16 and len(he) == 16 and len(set(tr) | set(he)) == 32,
            f"键划分：train∩held={len(inter)}（须 0）、|train|={len(tr)}、|held|={len(he)}、"
            f"并集={len(set(tr) | set(he))}（须 32）",
            value={"train": sorted(tr), "held": sorted(he)})

        # ---- A91-45 ★ 4 选 1 打分通路天花板：答案在上下文中
        LayerMem.read = _ORIG_READ
        for m in st.mems.values():
            m.no_write = True
        rng_e = __import__("random").Random(2027)
        hit, n_ctx = 0, 8
        for fi in range(n_ctx):
            f = facts[fi]
            prompt_ctx = f["prompt"] + " " + f["val"] + ", and this was double-checked"
            pids = tok(prompt_ctx, add_special_tokens=False)["input_ids"]
            others = [g2["val"] for j, g2 in enumerate(facts) if g2["val"] != f["val"]]
            cands = [f["val"]] + rng_e.sample(sorted(set(others)), N_CAND - 1)
            rng_e.shuffle(cands)
            nlls = [nll_of_candidate(model, st, pids,
                                     tok(" " + c, add_special_tokens=False)["input_ids"])
                    for c in cands]
            if cands[int(np.argmin(nlls))] == f["val"]:
                hit += 1
        acc_ctx = hit / n_ctx
        rec("A91-45", acc_ctx >= 0.75,
            f"★ 4 选 1 打分通路天花板：**答案明写在上下文里** ⇒ 命中 {hit}/{n_ctx}={acc_ctx:.3f}"
            f"（须 ≥0.75）⇒ 证明该通路**能**达到 0.50 以上的门（`CORR-10`）",
            value={"acc_in_context": acc_ctx, "hit": hit, "n": n_ctx})

        # ---- A91-46：fit/gen 两次评测之间权重逐位不变
        for m in st.mems.values():
            m.no_write = True
        s0 = snapshot(st)
        _ = eval_keys(model, st, tok, facts, tr)
        _ = eval_keys(model, st, tok, facts, he)
        s1 = snapshot(st)
        d_w = maxdiff_snap(s0, s1)
        rec("A91-46", d_w == 0.0,
            f"同一份权重：`fit` 与 `gen` 两次评测之间 W/Wq/Wo/γ **逐位不变** "
            f"max|Δ|={d_w:.1e}（须 ==0）", value={"d": d_w})

        # ---- A91-47：M3 参与比已算例（秩1=>1；随机 896² => MP 理论 E[s]^2/E[s^2]=0.7203·n）
        # ★ rev 2.4（DEF-91-35）：原写 0.360·n 是**我的算术错**（混淆半圆律与四分之一圆）；
        #   Marchenko–Pastur: E[s]=8/(3π)=0.8488, E[s^2]=1 ⇒ PR/n = 0.7203。实测 0.720·n。
        n = 896
        r1 = torch.zeros(n, n)
        r1[0, 0] = 3.0
        pr1 = participation_ratio(r1)
        gr = torch.Generator().manual_seed(3)
        rnd = torch.randn(n, n, generator=gr)
        prr = participation_ratio(rnd)
        lo, hi = 0.60 * n, 0.85 * n
        rec("A91-47", (abs(pr1 - 1.0) < 1e-6) and (lo < prr < hi),
            f"M3 参与比已算例：秩 1 ⇒ PR={pr1:.9f}（须 ==1）；各向同性随机 {n}×{n} ⇒ "
            f"PR={prr:.1f}（须落在 ({lo:.0f},{hi:.0f}) = Marchenko–Pastur 理论 0.7203·n≈{0.7203 * n:.0f}）",
            value={"pr_rank1": pr1, "pr_random": prr, "band": [lo, hi]})
    finally:
        LayerMem.read = _read_patched
        st.remove()


# ===================================================== 正式轮
def run_seed(tok, model, facts, chunks, owner, tr, he, seed):
    print(f"\n===== seed {seed} =====", flush=True)
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    out = {"seed": seed}
    try:
        write_phase(model, st, chunks)
        K, V = key_matrices(model, st, facts, chunks, owner)
        base = snapshot(st)          # ★ 写后基准态：三臂每次从它恢复（同初值）
        cls, ncls = class_ids(facts)
        out["n_class"] = ncls
        out["class_seen_frac_held"] = float(
            np.mean([cls[i] in {cls[j] for j in tr} for i in he]))
        RES["analytic"]["class_ids"] = cls

        arms = [("real", None, False, None),
                ("oracleP1", K, True, "readout"),
                ("oracleALL", K, True, None)]
        for name, KK, use_or, pos in arms:
            for li, m in st.mems.items():                     # 恢复同初值
                with torch.no_grad():
                    m.Wfull.copy_(base[li]["W"])
                    m.Wq.copy_(base[li]["Wq"])
                    m.Wo.copy_(base[li]["Wo"])
                    m.gamma.copy_(base[li]["gamma"])
            t0 = time.time()
            if use_or:
                losses, viol, div = train_read_oracle(
                    model, st, tok, facts, tr, __import__("random").Random(1000 + seed % 100),
                    KK, pos)
            else:
                losses, viol, div = train_read_task(
                    model, st, tok, facts, tr, __import__("random").Random(1000 + seed % 100))
            fit = eval_keys(model, st, tok, facts, tr, KK, use_or, pos)
            gen = eval_keys(model, st, tok, facts, he, KK, use_or, pos)
            model(input_ids=chunks[0].to(DEV))
            g = float(st.mems[list(st.mems)[0]].gamma)
            a = {"arm": name, "seed": seed, "gamma_final": g,
                 "acc_train": fit["acc_q"], "acc_held": gen["acc_q"],
                 "acc_item_train": fit["acc_item"], "acc_item_held": gen["acc_item"],
                 "margin_train": fit["margin_mean"], "margin_held": gen["margin_mean"],
                 "margin_pos_train": fit["margin_pos_frac"],
                 "margin_pos_held": gen["margin_pos_frac"],
                 "violations": viol, "diverged": div,
                 "final_loss": float(np.mean(losses[-50:])) if losses else None,
                 "sec": round(time.time() - t0, 1)}
            RES["arms"].append(a)
            print(f"  [{name}] acc_train={a['acc_train']:.3f} acc_held={a['acc_held']:.3f} "
                  f"margin(tr/ho)={a['margin_train']:+.2f}/{a['margin_held']:+.2f} "
                  f"γ={g:+.4f} viol={viol} div={div} {a['sec']}s", flush=True)
            out[name] = a

        # ---- 解析量 M3（Wo 有效秩，取 real 臂终值；三臂初值相同，秩结构可比）
        pr = {}
        for li, m in st.mems.items():
            pr[li] = participation_ratio(m.Wo.detach())
        out["wo_pr"] = pr
        print(f"  [M3] Wo 参与比(real 臂终值) = " +
              " ".join(f"L{li}:{v:.1f}" for li, v in pr.items()) +
              f"（随机基线≈{0.7203 * 896:.0f}，MP 理论）", flush=True)

        # ---- 解析量 M2：V → 答案类（岭回归；16/16 与 LOO）
        m2 = {}
        for li in st.mems:
            Vl = V[li].float()
            ytr_c = [cls[i] for i in tr]
            yhe_c = [cls[i] for i in he]
            s = ridge_scores(Vl[tr], ytr_c, Vl[he], ncls)
            a_ho, _ = top1(s, yhe_c)
            s_lo = ridge_scores(Vl, cls, Vl, ncls)        # LOO 近似（留一在下方精确做）
            loo = 0
            for i in range(N_FACTS):
                keep = [j for j in range(N_FACTS) if j != i]
                si = ridge_scores(Vl[keep], [cls[j] for j in keep], Vl[[i]], ncls)
                loo += int(int(si.argmax(dim=1)[0]) == cls[i])
            m2[li] = {"held_top1": a_ho, "loo_top1": loo / N_FACTS,
                      "insample_top1": top1(s_lo, cls)[0]}
        out["m2"] = {str(k): v for k, v in m2.items()}
        print("  [M2] V→答案类 " + " | ".join(
            f"L{li}: held={v['held_top1']:.3f} loo={v['loo_top1']:.3f}" for li, v in m2.items()),
            flush=True)

        # ---- 解析量 M4：恒等读出 argmax_c (u_c · v_i)
        head = model.lm_head.weight.detach().float()
        cids = [tok(" " + f["val"], add_special_tokens=False)["input_ids"][0] for f in facts]
        U = head[torch.tensor(cids)]                       # (32, d_model)
        m4 = {}
        for li in st.mems:
            Vl = V[li].float()
            S = Vl @ U.T                                   # (32 facts, 32 fact-value rows)
            hit = sum(int(int(S[i].argmax()) == i) for i in range(N_FACTS))
            m4[li] = hit / N_FACTS
        out["m4_identity_top1"] = m4
        print("  [M4] 恒等读出命中 " + " ".join(f"L{li}:{v:.3f}" for li, v in m4.items()),
              flush=True)
        RES["analytic"][str(seed)] = out
        return out
    finally:
        st.remove()


def branch_of(acc_tr, acc_he):
    if acc_he >= GATE_GEN:
        return "GENERALIZES"
    if acc_tr >= GATE_FIT and acc_he <= GATE_LOW:
        return "FIT_ONLY"
    if acc_tr < GATE_FIT and acc_he <= GATE_LOW:
        return "BOTH_LOW"
    return "PARTIAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["guards", "all"])
    ap.add_argument("--seeds", default="20261028,7777")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    RES["seeds"] = seeds

    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    print(f"事实 {len(facts)}；块 {len(chunks)}；训练键 {N_TRAIN_KEYS}；"
          f"留出键 {N_FACTS - N_TRAIN_KEYS}；n_q={N_FACTS - N_TRAIN_KEYS}（chance={CHANCE:.2f}）",
          flush=True)

    LayerMem.read = _read_patched
    try:
        stage_a(tok, model, facts, chunks, owner)
        if FAILS:
            print(f"\n★ Stage A FAIL: {FAILS} ⇒ 阻断正式轮", flush=True)
            RES["blocked"] = True
            dump_out(RES, "probe91_womap_guards")
            raise SystemExit(1)
        print("\n★ Stage A 全部 PASS", flush=True)
        dump_out(RES, "probe91_womap_guards")
        if args.stage == "guards":
            return

        order = list(range(N_FACTS))
        __import__("random").Random(777).shuffle(order)
        tr, he = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]

        for sd in seeds:
            run_seed(tok, model, facts, chunks, owner, tr, he, sd)

        # ---- 判决（主臂 = oracleP1）
        verd = {}
        for sd in seeds:
            a = RES["analytic"][str(sd)]["oracleP1"]
            verd[sd] = {"branch": branch_of(a["acc_train"], a["acc_held"]),
                        "acc_train": a["acc_train"], "acc_held": a["acc_held"]}
        RES["verdicts"] = {str(k): v for k, v in verd.items()}
        bs = {v["branch"] for v in verd.values()}
        final = bs.pop() if len(bs) == 1 else "INCONCLUSIVE"
        RES["branch_final"] = final
        RES["H1_confirmed"] = (final == "FIT_ONLY")
        print("\n" + "=" * 88, flush=True)
        for sd in seeds:
            v = verd[sd]
            print(f"★ seed {sd}: acc_train={v['acc_train']:.3f} acc_held={v['acc_held']:.3f} "
                  f"⇒ **{v['branch']}**", flush=True)
        print(f"\n★ 判决 = **{final}**；H1(fit-but-not-generalizable) "
              f"{'确认' if RES['H1_confirmed'] else '否证'}", flush=True)
        # 对照臂读数（DEF-91-33 的定量代价）
        for sd in seeds:
            al = RES["analytic"][str(sd)]["oracleALL"]
            print(f"  [对照] seed {sd} oracleALL: acc_train={al['acc_train']:.3f} "
                  f"acc_held={al['acc_held']:.3f} loss={al['final_loss']}", flush=True)
        dump_out(RES, "probe91_womap")
        print("\n★ 完成 ⇒ out/probe91_womap.json", flush=True)
    finally:
        LayerMem.read = _ORIG_READ


def dump_out(obj, name):
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()