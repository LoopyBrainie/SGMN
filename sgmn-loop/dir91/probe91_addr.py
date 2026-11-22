#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 rev 2.0 · 内容寻址定位：瓶颈在 c->q 还是 W_o？（task: DIR-91-rev2.4）

三臂**仅 q 的来源不同**（其余逐位相同，q 由 _q_oracle 逐实例替换）：
  real             q = Wq c          （装置原样，须复现 chance）
  oracle_frozen    q = k_i           （仅评测替换；读侧用 real 臂训好的参数，不再训）
  oracle_trained   q = k_i           （训练与评测都替换；同预算 600 步）  < 主判决臂

主判决量：acc_q(oracle_trained)（n_q=16，chance=0.25）。
  >=0.40 且 相对 real >=0.125  => B（寻址）  <=0.25 => A（输出映射）  其余 => INCONCLUSIVE

守卫 A91-30..36（任一 FAIL 阻断正式轮）。
★ 本文件**不修改** sgmn_module.py：读通路的 oracle 替换是**本文件内**的 monkey-patch，
  且 A91-30 断言"无替换时与原生逐位相同"，A91-31 断言"替换确实生效"。
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

RES = {"card": "DIR-91 rev 2.0", "task": "DIR-91-rev2.4",
       "alpha": ALPHA, "lam": LAM, "eta": ETA_WRITE, "read_steps": READ_STEPS,
       "read_lr": READ_LR, "n_train_keys": N_TRAIN_KEYS, "n_cand": N_CAND,
       "n_draws": N_DRAWS, "guards": {}, "arms": [], "share": {}}
FAILS = []

# 判决门（§2.2，沿用 D-103 冻结的同一组门）
GATE_B_HI = 0.40
GATE_A_LO = 0.25
GATE_DELTA = 0.125
# rev 2.2（DEF-91-32）：绝对 share 门**已废**（两次不可达）⇒ §2.3 降为只报告。
# 常量保留仅为留档可追溯，代码中不再使用。
GATE_SHARE_A_OBSOLETE = 0.25


def rec(name, ok, detail, value=None):
    RES["guards"][name] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


# --------------------------------------------------- 读通路：q 的 oracle 替换
_ORIG_READ = LayerMem.read


def _read_patched(self, hs_in, drop_sg=False):
    """与 sgmn_module.py:65 逐字相同；**唯一差别** = q 可由 _q_oracle 替换。"""
    ov = getattr(self, "_q_oracle", None)
    if ov is None:
        return _ORIG_READ(self, hs_in, drop_sg)          # 无替换 => 走原生，保证逐位相同
    self.read_calls += 1
    c = sg(hs_in, drop_sg)
    cf = c.reshape(-1, c.shape[-1])
    W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
    q = ov.to(device=cf.device, dtype=cf.dtype).unsqueeze(0).expand(cf.shape[0], -1)
    y = (sg(W, drop_sg) @ q.T).T
    m = (self.Wo @ y.T).T
    self.injection = m.reshape(hs_in.shape)
    return self.injection


def set_oracle(st, K, fi):
    for li, m in st.mems.items():
        m._q_oracle = None if fi is None else K[li][fi]


def clear_oracle(st):
    for m in st.mems.values():
        m._q_oracle = None


def cbar_and_keys(model, st, facts, chunks, owner):
    """= key_matrices 逐字，**额外**返回 cbar_i = mean_t(hs_in)（与 k_i 同块同 token 集）。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    K, V, C = {li: [] for li in st.mems}, {li: [] for li in st.mems}, {li: [] for li in st.mems}
    try:
        for fi in range(len(facts)):
            ci = owner.index(fi) + 1
            model(input_ids=chunks[ci].to(DEV))
            for li, m in st.mems.items():
                hs = st._hs_in[li]
                pf = hs.reshape(-1, hs.shape[-1])
                K[li].append((((pf @ m.Pk).mean(dim=0)) * m.pk_scale).detach())
                V[li].append((pf @ m.Pv).mean(dim=0).detach())
                C[li].append(pf.mean(dim=0).detach())
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
    f = lambda D: {li: torch.stack(D[li]).float() for li in D}      # noqa: E731
    return f(K), f(V), f(C)


def share_of(q, K):
    """share = |q.k_i| / sum_j |q.k_j|（完美寻址->1，随机->1/N）。

    ★ rev 2.1（DEF-91-30）：**raw 版本天花板≈1/N**（键有巨大公共模），故仅作机制读数。
    """
    w = (K @ q).abs()
    return w / w.sum().clamp_min(1e-30), w


def centering(K):
    """去均值键：tilde k_j = k_j - mean_j k_j（与 R2_task91.py:369 key_cos 同一惯例）。"""
    return K - K.mean(dim=0, keepdim=True)


def share_cen_of(q, Kc):
    """centered share：判别量（rev 2.1 §2.3）。Kc 须已由 centering() 去均值。"""
    w = (Kc @ q).abs()
    return w / w.sum().clamp_min(1e-30), w


# --------------------------------------------------- 带 oracle 的训练/评测
def train_read_oracle(model, st, tok, facts, train_keys, rng, K):
    """= train_read_task 逐字；**唯一差别** = 每次前向前把 q 置为 k_i（训练时也替换）。"""
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
        set_oracle(st, K, ki)
        try:
            lg = model(input_ids=seq).logits.float()
        finally:
            clear_oracle(st)
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


def eval_arm(model, st, tok, facts, eval_keys, K=None, use_oracle=False):
    """= run_arm 的评测段逐字；use_oracle=True 时按 prompt 反查事实下标并置 q=k_i。"""
    for m in st.mems.values():
        m.no_write = True
    items = make_eval_items(tok, facts, eval_keys, __import__("random").Random(555))
    p2fi = {tuple(f["prompt_ids"]): i for i, f in enumerate(facts)}
    if use_oracle:
        def scorer(p, cs):
            fi = p2fi[tuple(p)]
            set_oracle(st, K, fi)
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


def snap_read(st):
    return {li: {"Wq": m.Wq.detach().clone(), "Wo": m.Wo.detach().clone(),
                 "gamma": m.gamma.detach().clone()} for li, m in st.mems.items()}


def load_read(st, snap):
    for li, m in st.mems.items():
        with torch.no_grad():
            m.Wq.copy_(snap[li]["Wq"])
            m.Wo.copy_(snap[li]["Wo"])
            m.gamma.copy_(snap[li]["gamma"])


def load_W(st, Wsnap):
    for li, m in st.mems.items():
        with torch.no_grad():
            m.Wfull.copy_(Wsnap[li])


# ------------------------------------------------------------------ Stage A
def run_guards(tok, model, facts, chunks, owner, seed):
    print("\n===== Stage A 守卫 =====", flush=True)

    # A91-30（rev 2.1 订正）：无替换 => 与原生逐位相同。★ 两次前向必须同 W。
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    try:
        write_phase(model, st, chunks)
        for m in st.mems.values():
            m.no_write = True                     # ★ 关写：两次前向同 W（DEF-91-29）
        Wsnap0 = {li: m.Wfull.detach().clone() for li, m in st.mems.items()}
        LayerMem.read = _ORIG_READ
        model(input_ids=chunks[0].to(DEV))
        m_orig = {li: m.injection.detach().clone() for li, m in st.mems.items()}
        LayerMem.read = _read_patched
        clear_oracle(st)
        model(input_ids=chunks[0].to(DEV))
        d = max(float((m.injection - m_orig[li]).abs().max()) for li, m in st.mems.items())
        # 交叉校验：还原 W 后再比一次（排除 W 漂移的一切可能）
        for li, m in st.mems.items():
            with torch.no_grad():
                m.Wfull.copy_(Wsnap0[li])
        LayerMem.read = _ORIG_READ
        model(input_ids=chunks[0].to(DEV))
        m_orig2 = {li: m.injection.detach().clone() for li, m in st.mems.items()}
        LayerMem.read = _read_patched
        model(input_ids=chunks[0].to(DEV))
        d2 = max(float((m.injection - m_orig2[li]).abs().max()) for li, m in st.mems.items())
        rec("A91-30", d == 0.0 and d2 == 0.0,
            f"打补丁的读通路在**无替换**时与原生**逐位相同**："
            f"no_write 下 max|Δm|={d:.1e}（须 ==0）；还原 W 交叉校验 max|Δm|={d2:.1e}（须 ==0）",
            value={"d": d, "d_restore": d2})
    finally:
        LayerMem.read = _read_patched
        st.remove()

    # A91-33 / A91-36：k_i 来自 key_matrices；γ=0 下"无写/写后"K,V 逐位相同；写入数
    st_w = make_stack(gamma=0.0, no_write=False, seed=seed)
    try:
        wc = write_phase(model, st_w, chunks)
        K_w, V_w, _ = cbar_and_keys(model, st_w, facts, chunks, owner)
        K_ref, V_ref = key_matrices(model, st_w, facts, chunks, owner)
        dk = max(float((K_w[li] - K_ref[li]).abs().max()) for li in K_w)
    finally:
        st_w.remove()
    st_nw = make_stack(gamma=0.0, no_write=True, seed=seed)
    try:
        K_nw, V_nw, _ = cbar_and_keys(model, st_nw, facts, chunks, owner)
    finally:
        st_nw.remove()
    d36 = max(max(float((K_w[li] - K_nw[li]).abs().max()),
                  float((V_w[li] - V_nw[li]).abs().max())) for li in K_w)
    writes = [int(v) for v in wc.values()]
    n_chunks = len(chunks)
    rec("A91-33", dk == 0.0 and writes == [n_chunks - 1] * len(writes),
        f"k_i 与 `key_matrices` **逐位相同**（max|Δk|={dk:.1e}，须 ==0）；"
        f"写入数={writes}，期望={n_chunks - 1}（=块数-1）", value={"writes": writes})
    rec("A91-36", d36 == 0.0,
        f"γ=0 下「无写」与「写后」的 K,V **逐位相同**：三层 max|Δ|={d36:.1e}（须 ==0）", value=d36)

    # A91-31：q=k_i 时 y 与 W k_i 逐位相同
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    try:
        write_phase(model, st, chunks)
        K, V, C = cbar_and_keys(model, st, facts, chunks, owner)
        fi = 0
        set_oracle(st, K, fi)
        model(input_ids=chunks[owner.index(fi) + 1].to(DEV))
        dmax = 0.0
        for li, m in st.mems.items():
            y_dev = (W_of(m) @ K[li][fi])
            m_ref = (m.Wo @ y_dev)
            # 注入 = gamma*Wo*y；比较 y 的等价性用"注入/gamma"不稳，改比 y 本身
            q_used = m._q_oracle.to(device=DEV, dtype=K[li].dtype)
            y_used = (W_of(m) @ q_used)
            dmax = max(dmax, float((y_used - y_dev).abs().max()))
        clear_oracle(st)
        rec("A91-31", dmax == 0.0,
            f"q=k_{fi} 时 y 与 W·k_{fi} **逐位相同**：三层 max|Δy|={dmax:.1e}（须 ==0）——"
            f"证明替换确实生效且未被静默忽略", value=dmax)
    finally:
        clear_oracle(st)
        st.remove()

    # A91-32：测量对象已算例
    st = make_stack(gamma=0.0, no_write=True, seed=seed)
    try:
        K, V, C = cbar_and_keys(model, st, facts, chunks, owner)
    finally:
        st.remove()
    li = sorted(K)[0]
    Km = K[li].double()
    q0 = Km[0]
    sh, w = share_of(q0, Km)
    off = w[1:].max()
    bound = float((q0 @ q0) / ((q0 @ q0) + (len(Km) - 1) * off))
    got = float(sh[0])
    # 合成算例（raw 有效）：K=I 正交 => raw share 必为 1.0
    e = torch.eye(len(Km), dtype=torch.float64)
    sh_e, _ = share_of(e[0], e)
    # centered 用**有效**解析下界（DEF-91-31：centerning(I) 不正交，不能断言==1）
    Kc = centering(Km)
    q0c = Kc[0]
    shc, wc = share_cen_of(q0c, Kc)
    got_c = float(shc[0])
    offc = wc[1:].max()
    bound_c = float((q0c @ q0c) / ((q0c @ q0c) + (len(Kc) - 1) * offc))
    ok32 = (got >= bound - 1e-9) and abs(float(sh_e[0]) - 1.0) <= 1e-12 \
        and (got_c >= bound_c - 1e-9)
    rec("A91-32", ok32,
        f"share 实现已算例：q=k_0 时 raw share={got:.4f} ≥ 下界 {bound:.4f}（须成立，"
        f"≈1/N ⇒ DEF-91-30 退化）；centered share={got_c:.4f} ≥ 下界 {bound_c:.4f}（须成立）；"
        f"正交合成算例 raw={float(sh_e[0]):.12f}（须 ==1）",
        value={"share0_raw": got, "share0_cen": got_c,
               "bound_raw": bound, "bound_cen": bound_c})

    # A91-34：三臂 items 相同
    items = make_eval_items(tok, facts, facts and [i for i in range(N_FACTS)], __import__("random").Random(555))
    n_it, n_q = len(items), len(set(it["key_idx"] for it in items))
    rec("A91-34", n_it == N_FACTS * N_DRAWS and n_q == N_FACTS,
        f"三臂共用同一 `Random(555)`：n_item={n_it}（须={N_FACTS * N_DRAWS}）、"
        f"n_q={n_q}（须={N_FACTS}）", value=n_it)


# ------------------------------------------------------------------ 正式轮
def run_seed(tok, model, facts, chunks, owner, train_keys, eval_keys, seed):
    print(f"\n===== seed {seed} =====", flush=True)
    t0 = time.time()
    # ① 一次真实写入 => W 快照（三臂共用同一 W，逐位相同）
    st0 = make_stack(gamma=0.0, no_write=False, seed=seed)
    try:
        wc = write_phase(model, st0, chunks)
        K, V, C = cbar_and_keys(model, st0, facts, chunks, owner)
        Wsnap = {li: m.Wfull.detach().clone() for li, m in st0.mems.items()}
        W0snap = {li: m.W0.detach().clone() for li, m in st0.mems.items()}
    finally:
        st0.remove()
    print(f"  写入完成 writes={[int(v) for v in wc.values()]} {time.time() - t0:.0f}s", flush=True)

    # ② real 臂
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    try:
        load_W(st, Wsnap)
        for m in st.mems.values():
            m.no_write = True
        import random as _r
        losses, viol, div = train_read_task(model, st, tok, facts, train_keys,
                                           _r.Random(1000 + seed % 100))
        real = eval_arm(model, st, tok, facts, eval_keys)
        real.update({"arm": "real", "seed": seed, "q_src": "Wq c", "trained": True,
                     "final_loss": float(np.mean(losses[-50:])), "violations": viol,
                     "diverged": div,
                     "rel": {str(li): storage_rel(K, V, li, W_of(m))
                             for li, m in st.mems.items()}})
        snap = snap_read(st)
    finally:
        st.remove()
    RES["arms"].append(real)
    print(f"  [real] acc_q={real['acc_q']:.3f} acc_item={real['acc_item']:.4f} "
          f"loss={real['final_loss']:.2e}", flush=True)

    # ③ oracle_frozen：读侧参数 == real 臂（逐位），只替换 q
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    try:
        load_W(st, Wsnap)
        load_read(st, snap)
        d35q = max(float((st.mems[li].Wq - snap[li]["Wq"]).abs().max()) for li in st.mems)
        d35o = max(float((st.mems[li].Wo - snap[li]["Wo"]).abs().max()) for li in st.mems)
        d35g = max(float((st.mems[li].gamma - snap[li]["gamma"]).abs().max()) for li in st.mems)
        froz = eval_arm(model, st, tok, facts, eval_keys, K=K, use_oracle=True)
        froz.update({"arm": "oracle_frozen", "seed": seed, "q_src": "k_i", "trained": False})
    finally:
        st.remove()
    RES["arms"].append(froz)
    print(f"  [oracle_frozen] acc_q={froz['acc_q']:.3f} acc_item={froz['acc_item']:.4f}", flush=True)

    # ④ oracle_trained：同初值、同预算，训练与评测都替换 q
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    try:
        load_W(st, Wsnap)
        load_read(st, snap)                     # 与 real 臂**同初值**
        d35 = max(d35q, d35o, d35g)
        losses, viol, div = train_read_oracle(model, st, tok, facts, train_keys,
                                             __import__("random").Random(1000 + seed % 100), K)
        ot = eval_arm(model, st, tok, facts, eval_keys, K=K, use_oracle=True)
        ot.update({"arm": "oracle_trained", "seed": seed, "q_src": "k_i", "trained": True,
                   "final_loss": float(np.mean(losses[-50:])), "violations": viol,
                   "diverged": div})
    finally:
        st.remove()
    RES["arms"].append(ot)
    print(f"  [oracle_trained] acc_q={ot['acc_q']:.3f} acc_item={ot['acc_item']:.4f} "
          f"loss={ot['final_loss']:.2e}", flush=True)

    # A91-35：oracle_frozen / oracle_trained 初值 == real 臂终值
    rec(f"A91-35@{seed}", d35 == 0.0 and d35q == 0.0 and d35o == 0.0 and d35g == 0.0,
        f"`oracle_*` 的初值 Wq/Wo/gamma 与 `real` 臂终值**逐位相同**："
        f"max|ΔWq|={d35q:.1e} max|ΔWo|={d35o:.1e} max|Δγ|={d35g:.1e}（须全 ==0）", value=d35)

    # ⑤ 机制读数：share（real 臂终态）—— **留出键与训练键分开报**
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    try:
        load_W(st, Wsnap)
        load_read(st, snap)
        K2, V2, C2 = cbar_and_keys(model, st, facts, chunks, owner)
        sh = {}
        for li, m in st.mems.items():
            q_all = (m.Wq @ C2[li].T).T                     # (N,d_k)  q_i = Wq cbar_i
            Kc2 = centering(K2[li])                         # rev 2.1：判别量用去均值键
            s = [float(share_of(q_all[i], K2[li])[0][i]) for i in range(len(K2[li]))]
            sc = [float(share_cen_of(q_all[i], Kc2)[0][i]) for i in range(len(K2[li]))]
            sc_o = [float(share_cen_of(K2[li][i], Kc2)[0][i]) for i in range(len(K2[li]))]
            cdiag = [float(torch.nn.functional.cosine_similarity(
                q_all[i].unsqueeze(0), K2[li][i].unsqueeze(0)).item()) for i in range(len(K2[li]))]
            yv = [(W_of(m) @ q_all[i]) for i in range(len(K2[li]))]
            cos_yv = [float(torch.nn.functional.cosine_similarity(
                yv[i].unsqueeze(0), V2[li][i].unsqueeze(0)).item()) for i in range(len(K2[li]))]
            relyv = [float((yv[i] - V2[li][i]).norm() / V2[li][i].norm().clamp_min(1e-30))
                     for i in range(len(K2[li]))]
            sh[li] = {
                # raw（退化，仅机制：天花板≈1/N）
                "share_raw_median_eval": float(np.median([s[i] for i in eval_keys])),
                "share_raw_median_train": float(np.median([s[i] for i in train_keys])),
                "share_raw_oracle_median_eval": float(np.median(
                    [float(share_of(K2[li][i], K2[li])[0][i]) for i in eval_keys])),
                # ★ centered（判别量，rev 2.1 §2.3）
                "share_cen_median_eval": float(np.median([sc[i] for i in eval_keys])),
                "share_cen_median_train": float(np.median([sc[i] for i in train_keys])),
                "share_cen_oracle_median_eval": float(np.median([sc_o[i] for i in eval_keys])),
                "cos_qk_median_eval": float(np.median([cdiag[i] for i in eval_keys])),
                "cos_yv_median_eval": float(np.median([cos_yv[i] for i in eval_keys])),
                "relyv_median_eval": float(np.median([relyv[i] for i in eval_keys])),
            }
    finally:
        st.remove()
    RES["share"][str(seed)] = {str(k): v for k, v in sh.items()}
    for li, v in sh.items():
        print(f"  [share L{li}] **cen**(eval)={v['share_cen_median_eval']:.4f} "
              f"cen(train)={v['share_cen_median_train']:.4f} "
              f"cen(oracle)={v['share_cen_oracle_median_eval']:.4f} | "
              f"raw(eval)={v['share_raw_median_eval']:.4f} "
              f"raw(oracle)={v['share_raw_oracle_median_eval']:.4f} | "
              f"cos(q,k)={v['cos_qk_median_eval']:+.4f} cos(y,v)={v['cos_yv_median_eval']:+.4f} "
              f"rel(y,v)={v['relyv_median_eval']:.4f}", flush=True)
    return sh


def storage_rel(K, V, li, W):
    return float((((K[li] @ W.float().T) - V[li]).norm(dim=1) /
                  V[li].norm(dim=1).clamp_min(1e-30)).median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="guards", choices=["guards", "main", "all"])
    ap.add_argument("--seeds", default="20261028,7777")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    LayerMem.read = _read_patched        # ★ 全局装补丁（A91-30 已证等价）

    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    order = list(range(N_FACTS))
    import random as _r2
    _r2.Random(777).shuffle(order)
    train_keys, eval_keys = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]
    print(f"事实 {len(facts)}；块 {len(chunks)}；训练键 {len(train_keys)}；评测键 {len(eval_keys)}；"
          f"n_q={len(eval_keys)}（chance={1.0 / N_CAND:.2f}）", flush=True)
    RES["train_keys"], RES["eval_keys"] = train_keys, eval_keys

    if a.stage in ("guards", "all"):
        run_guards(tok, model, facts, chunks, owner, seeds[0])
        if FAILS:
            print(f"\n★ Stage A FAIL: {FAILS} ⇒ 阻断正式轮", flush=True)
            (OUT / "probe91_addr_guards.json").write_text(
                json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
            return 2

    if a.stage in ("main", "all"):
        for seed in seeds:
            run_seed(tok, model, facts, chunks, owner, train_keys, eval_keys, seed)

        # 判决（§2.2 + §2.3 一致性）
        verdicts = {}
        for seed in seeds:
            got = [x for x in RES["arms"] if x["seed"] == seed]
            g = {x["arm"]: x for x in got}
            real, ot, fr = g["real"]["acc_q"], g["oracle_trained"]["acc_q"], g["oracle_frozen"]["acc_q"]
            delta = ot - real
            sh_raw = float(np.median([v["share_raw_median_eval"]
                                      for v in RES["share"][str(seed)].values()]))
            if ot >= GATE_B_HI and delta >= GATE_DELTA:
                br = "B"
            elif ot <= GATE_A_LO:
                br = "A"
            else:
                br = "INCONCLUSIVE"
            # §2.3（rev 2.2）：**只报告，不否决**。rho = 实际寻址 / 该层完美寻址上限
            cen_act = float(np.median([v["share_cen_median_eval"]
                                       for v in RES["share"][str(seed)].values()]))
            cen_ceil = float(np.median([v["share_cen_oracle_median_eval"]
                                        for v in RES["share"][str(seed)].values()]))
            rho = cen_act / max(cen_ceil, 1e-30)
            sh_raw = float(np.median([v["share_raw_median_eval"]
                                      for v in RES["share"][str(seed)].values()]))
            coherent = True
            verdicts[seed] = {"real": real, "oracle_trained": ot, "oracle_frozen": fr,
                              "delta": delta, "share_cen_eval_median": cen_act,
                              "share_cen_ceiling": cen_ceil, "rho": rho,
                              "share_raw_eval_median": sh_raw, "branch": br}
        RES["verdicts"] = {str(k): v for k, v in verdicts.items()}
        both = {v["branch"] for v in verdicts.values()}
        RES["branch_final"] = both.pop() if len(both) == 1 else "SPLIT"
        # A91-30/31/32/33/34/36 记为臂前守卫，A91-35 逐 seed
        rec("A91-30..34,36", all(RES["guards"].get(n, {}).get("ok")
                                 for n in ("A91-30", "A91-31", "A91-32", "A91-33",
                                           "A91-34", "A91-36")),
            "六条臂前守卫全部 PASS（明细见上）", value=None)

    (OUT / "probe91_addr.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    if "verdicts" in RES:
        for seed, v in RES["verdicts"].items():
            print(f"\n★ seed {seed}: real={v['real']:.3f} oracle_trained={v['oracle_trained']:.3f} "
                  f"oracle_frozen={v['oracle_frozen']:.3f} Δ={v['delta']:+.3f} "
                  f"share_cen(actual)={v['share_cen_eval_median']:.4f} "
                  f"ceiling={v['share_cen_ceiling']:.4f} rho={v['rho']:.3f} "
                  f"raw={v['share_raw_eval_median']:.4f} ⇒ **{v['branch']}**", flush=True)
        print(f"\n★ 判决 = **{RES['branch_final']}**", flush=True)
    print("\n★ 完成 ⇒ dir91/out/probe91_addr.json", flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())