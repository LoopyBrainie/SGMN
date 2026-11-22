#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 rev 2.8 · **W_o 的失败是优化/目标问题，还是注入通路表达力问题？**

卡片：revolve/DIR-91/DIRECTION_rev91_v28.md（判据冻结；本文件不得改阈值）
派生自 D-108 §八「可分辨的下一问」。

六臂（**唯一独立变量 = 读出位的注入向量 m 从哪来**；写侧一律 A_mean，写后 W 冻结）：
  zero        m ≡ 0                                              对照：无注入
  adam16      m = Wo y，Wo 由 Adam 600 步学（复现 D-108 的 A_mean，γ init 0）
  adam16_g1   同上但 γ init 1.0                                   定向诊断
  free        每事实一个自由向量 m_i（逐层 896 维，只优化 m_i）    ✘ 表达力天花板
  ridge16     m = M y，M 闭式岭回归拟合 16 条训练                  可实现 · 主判据
  ridge32     同上，拟合集 = 全部 32                              ✘ 声明式泄漏上界

★ 不修改 sgmn_module.py / R2_task91.py：读通路注入替换 + y 抓取全部在本文件内 monkey-patch。
★ 状态共享：一个 seed 只写一次；{zero, free, ridge16, ridge32} 只替换 m，不动任何参数 ⇒ 共用一个 stack。
"""
import argparse, json, os, sys, time
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)

import numpy as np
import torch

import sgmn_module
from sgmn_module import LayerMem, sg
import probe91_womap as W91
import probe91_valread as V27
import R2_task91 as R2
from R2_task91 import (ALPHA, B as N_CHUNKS, CHUNK, DEV, ETA_WRITE, LAM, MOUNT, N_CAND,
                       N_DRAWS, N_FACTS, N_TRAIN_KEYS, OUT, READ_LR, READ_STEPS,
                       acc_by_question, build_chunks, build_facts, make_eval_items,
                       make_stack, model_and_tok, score_and_accuracy)

# ---------------------------------------------------------------- §2 冻结
G_GEN = 0.50
G_LOW = 0.25
G_DELTA = 0.125
CHANCE = 1.0 / N_CAND
SEEDS = [20261028, 7777]
ARMS = ["zero", "adam16", "adam16_g1", "free", "ridge16", "ridge32"]
STEPS_FREE = 120
LR_FREE = 0.05
RIDGE_REL = 1e-3
ALL32 = list(range(N_FACTS))

RES = {"card": "DIR-91 rev 2.8", "file": "probe91_woform.py",
       "gates": {"G_gen": G_GEN, "G_low": G_LOW, "G_delta": G_DELTA, "chance": CHANCE},
       "consts": {"alpha": ALPHA, "lam": LAM, "eta": ETA_WRITE, "read_steps": READ_STEPS,
                  "read_lr": READ_LR, "n_chunk": N_CHUNKS, "n_facts": N_FACTS,
                  "n_train_keys": N_TRAIN_KEYS, "n_cand": N_CAND, "n_draws": N_DRAWS,
                  "steps_free": STEPS_FREE, "lr_free": LR_FREE, "ridge_rel": RIDGE_REL},
       "guards": {}, "arms": [], "branch": {}, "worked": {}}
FAILS = []


def rec(name, ok, detail, value=None):
    RES["guards"][name] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"{'[PASS]' if ok else '[FAIL]'} {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def dump(obj, name):
    p = OUT / f"{name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print(f"  -> {p}", flush=True)


# ------------------------------------------------- 读通路：注入替换 / y 抓取
_ORIG_READ = LayerMem.read


def _read_r28(self, hs_in, drop_sg=False):
    """与 sgmn_module.py:65 逐字相同；**唯一差别** = 读出位那一行的 m 可被替换/记录。

    `_m_override`: (d_model,) 覆盖读出位；None ⇒ 走原生（逐位相同）。
    `_y_capture` : True 时把读出位 y 存进 `self._y_out`（不改变注入）。
    """
    mo = getattr(self, "_m_override", None)
    cap = getattr(self, "_y_capture", False)
    if mo is None and not cap:
        return _ORIG_READ(self, hs_in, drop_sg)
    self.read_calls += 1
    c = sg(hs_in, drop_sg)
    cf = c.reshape(-1, c.shape[-1])
    W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
    q = (self.Wq @ cf.T).T
    y = (sg(W, drop_sg) @ q.T).T
    # ★ DEF-94-01：读出位 = **每一行**的 prompt 末位（b*L + L-2），不是展平批的倒数第二个。
    #   因果注意力 ⇒ 各行的该位注入相同；单行时退化为 L-2。
    Bn, Ln = hs_in.shape[0], hs_in.shape[1]
    idx = torch.arange(Bn, device=cf.device) * Ln + (Ln - 2)
    if cap:
        self._y_out = y[idx[0]].detach().clone()
        self._y_spread = float((y[idx] - y[idx[0]]).abs().max()) if Bn > 1 else 0.0
    m = (self.Wo @ y.T).T
    if mo is not None:
        m = m.clone()
        m[idx] = mo.to(device=m.device, dtype=m.dtype)
    self.injection = m.reshape(hs_in.shape)
    return self.injection


# ★ 全文只装一次：无 _m_override / _y_capture 时 _read_r28 逐位委托给 _ORIG_READ（A94-01 断言）
LayerMem.read = _read_r28


def count_mounted(model, layers):
    """数各层 forward hook 里我们的注入钩子个数（★ DEF-93-01：数，不假设 n_stacks=1）。"""
    n = 0
    ls = model.model.layers if hasattr(model, "model") else model.layers
    for li in layers:
        for h in ls[li]._forward_hooks.values():
            if getattr(h, "__name__", "") == "fwd_hook":
                n += 1
    return n


def set_m(st, vec):
    for li, m in st.mems.items():
        m._m_override = None if vec is None else vec[li]


def set_capture(st, on):
    for m in st.mems.values():
        m._y_capture = bool(on)
        if not on:
            m._y_out = None


# ------------------------------------------------------------------ 写一次
def set_gamma(st, val):
    with torch.no_grad():
        for m in st.mems.values():
            m.gamma.fill_(float(val))


def build_state(tok, model, facts, chunks, owner, seed, gamma=0.0):
    """★ DEF-94-03：**写阶段一律 γ=0** —— 读注入否则会扰动写轨迹，
    使 W 依赖 γ 与"当时有没有挂 override"。γ 在本函数内不生效，写完后由 set_gamma 设定。"""
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    V27.install_write()
    try:
        V27.ARM["name"] = "A_mean"
        wc = V27.write_phase_arm(model, st, chunks, owner, {})
    finally:
        V27.restore_write()
    Wsnap = {li: W91.W_of(m).detach().clone() for li, m in st.mems.items()}
    return st, wc, Wsnap


def capture_y(model, st, facts, dummy):
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    Y = {li: [] for li in st.mems}
    try:
        set_m(st, None)
        set_capture(st, True)
        for f in facts:
            seq = torch.tensor([f["prompt_ids"] + [dummy]], dtype=torch.long, device=DEV)
            model(input_ids=seq)
            for li, m in st.mems.items():
                Y[li].append(m._y_out.clone())
    finally:
        set_capture(st, False)
        for li, m in st.mems.items():
            m.no_write = saved[li]
    return {li: torch.stack(Y[li]).float() for li in Y}


# ------------------------------------------------------------------ 闭式岭
def ridge_closed(Y, T, rel=RIDGE_REL):
    """M = T Yᵀ (Y Yᵀ + λI)⁻¹；Y,T 为 (d, n)。λ = rel·tr(YYᵀ)/n。"""
    Yd, Td = Y.double(), T.double()
    A = Yd @ Yd.T
    lam = rel * float(torch.diagonal(A).sum()) / max(Yd.shape[1], 1)
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    M = Td @ Yd.T @ torch.linalg.inv(A + lam * I)
    res = float((M @ Yd - Td).norm() / Td.norm().clamp_min(1e-30))
    return M, lam, res


# ------------------------------------------------------------------ 注入式评测
def eval_inject(model, st, tok, facts, keys, Mvec, rng):
    """判据口径 = R2_task91.make_eval_items + score_and_accuracy + acc_by_question（A93-08）。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    items = make_eval_items(tok, facts, keys, rng)
    p2fi = {tuple(f["prompt_ids"]): i for i, f in enumerate(facts)}
    try:
        def scorer(p, cs):
            fi = p2fi[tuple(p)]
            if Mvec == "native":
                set_m(st, None)
            elif Mvec is None:
                set_m(st, {li: torch.zeros(st.mems[li].d_model, device=DEV)
                           for li in st.mems})
            else:
                set_m(st, {li: Mvec[li][fi] for li in st.mems})
            P = len(p)
            # score_and_accuracy 传进来的是 ids_list = [c[0] for c in cand] ⇒ 每项已是 id 列表
            for cids in cs:
                assert len(cids) == 1, f"候选不是单 token：{cids}"   # 与 R2.train_read_task 同假设
            seq = torch.tensor([list(p) + [cids[0]] for cids in cs],
                               dtype=torch.long, device=DEV)
            lg = model(input_ids=seq).logits.float()
            out = []
            for r, cids in enumerate(cs):
                lp = torch.log_softmax(lg[r, P - 1], dim=-1)
                out.append(float(-lp[cids[0]]))
            return out
        acc_item, preds, detail = score_and_accuracy(items, scorer)
        acc_q, n_q = acc_by_question(items, detail)
        mg = []
        for it, d in zip(items, detail):
            k = [i for i, c in enumerate(it["cand"]) if c[1]][0]
            others = [v for i, v in enumerate(d["nll"]) if i != k]
            mg.append(min(others) - d["nll"][k])
        return {"acc_q": acc_q, "n_q": n_q, "acc_item": acc_item,
                "margin_mean": float(np.mean(mg)),
                "margin_pos_frac": float(np.mean([m > 0 for m in mg])),
                "n_item": len(items)}
    finally:
        set_m(st, None)
        for li, m in st.mems.items():
            m.no_write = saved[li]


# ------------------------------------------------------------------ free 优化
def optimise_free(model, st, tok, facts, keys, rng, steps=STEPS_FREE):
    """★ rng 一律用 Random(555)（= 判据题集）：天花板必须在**被判的那批题**上取，
    否则"天花板"实际是在没见过的干扰集上判，会低估表达力（DEF-94-04）。"""
    """每事实每层一个自由向量 m_i；只优化 m_i（其余冻结、no_write、不写 W）。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    items = make_eval_items(tok, facts, keys, rng)
    by_fi = {}
    for it in items:
        by_fi.setdefault(it["key_idx"], []).append(it)
    Mv = {li: {} for li in st.mems}
    ls_by_fi = {}
    try:
        set_m(st, None)
        for fi in keys:
            its = by_fi[fi]
            P = len(its[0]["prompt_ids"])
            nc = len(its[0]["cand"])
            for it in its:
                for cids, _ in it["cand"]:
                    assert len(cids) == 1, f"候选不是单 token：{cids}"
            seqs = [it["prompt_ids"] + [c[0][0]] for it in its for c in it["cand"]]
            seq = torch.tensor(seqs, dtype=torch.long, device=DEV)
            tok_ids = torch.tensor([s[-1] for s in seqs], device=DEV)
            tgt = torch.tensor([next(j for j, c in enumerate(it["cand"]) if c[1])
                                for it in its], dtype=torch.long, device=DEV)
            pars = {li: torch.zeros(st.mems[li].d_model, device=DEV, requires_grad=True)
                    for li in st.mems}
            opt = torch.optim.Adam(pars.values(), lr=LR_FREE)
            ls = []
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                set_m(st, {li: pars[li] for li in st.mems})
                lg = model(input_ids=seq).logits.float()
                ar = torch.arange(len(seqs), device=DEV)
                pick = lg[ar, P - 1, tok_ids].reshape(len(its), nc)
                loss = torch.nn.functional.cross_entropy(pick, tgt)
                loss.backward()
                opt.step()
                ls.append(float(loss))
            for li in st.mems:
                Mv[li][fi] = pars[li].detach().clone()
            ls_by_fi[fi] = {"head": float(np.mean(ls[:20])), "tail": float(np.mean(ls[-20:]))}
    finally:
        set_m(st, None)
        for li, m in st.mems.items():
            m.no_write = saved[li]
    M = {li: torch.stack([Mv[li][fi] for fi in keys]).float() for li in st.mems}
    return M, ls_by_fi


# ------------------------------------------------------------------ Adam 臂
def run_adam(arm, tok, model, facts, chunks, owner, tr, he, seed):
    ginit = 1.0 if arm == "adam16_g1" else 0.0
    st, wc, _ = build_state(tok, model, facts, chunks, owner, seed)
    t0 = time.time()
    try:
        set_gamma(st, ginit)                # 写完后才设 γ（写阶段恒 γ=0）
        for m in st.mems.values():
            m.no_write = True
        rng = __import__("random").Random(1000 + seed % 100)
        losses, viol, div = R2.train_read_task(model, st, tok, facts, tr, rng)
        fit = eval_inject(model, st, tok, facts, tr, "native",
                          __import__("random").Random(555))
        gen = eval_inject(model, st, tok, facts, he, "native",
                          __import__("random").Random(555))
        g = float(st.mems[list(st.mems)[0]].gamma)
        a = {"arm": arm, "seed": seed, "gamma_init": ginit, "gamma_final": g,
             "write_calls": wc,
             "acc_train": fit["acc_q"], "acc_held": gen["acc_q"],
             "acc_item_train": fit["acc_item"], "acc_item_held": gen["acc_item"],
             "margin_train": fit["margin_mean"], "margin_held": gen["margin_mean"],
             "margin_pos_held": gen["margin_pos_frac"],
             "n_q": gen["n_q"], "n_item": gen["n_item"],
             "info": {"violations": viol, "diverged": div,
                      "final_loss": float(np.mean(losses[-50:])) if losses else None},
             "sec": round(time.time() - t0, 1)}
        print(f"  [{arm}] acc_train={a['acc_train']:.3f} acc_held={a['acc_held']:.3f} "
              f"margin(tr/ho)={a['margin_train']:+.2f}/{a['margin_held']:+.2f} "
              f"γ={g:+.4f} viol={viol} {a['sec']}s", flush=True)
        return a
    finally:
        st.remove()


# ------------------------------------------------------------------ Stage A
def stage_a(tok, model, facts, chunks, owner, tr, he):
    seed = SEEDS[0]
    # ---- A94-00 配置 + worked example（量测对象先证明非空转）
    rec("A94-00", MOUNT == [8, 12, 16] and CHUNK == 64 and N_CHUNKS == 600,
        f"MOUNT={MOUNT} CHUNK={CHUNK} B={N_CHUNKS} n_q=16 n_draws={N_DRAWS}")
    st, wc, Wsnap = build_state(tok, model, facts, chunks, owner, seed)
    try:
        set_gamma(st, 1.0)
        f0 = facts[0]
        # ★ 必须 no_write：capture_y 内部置 no_write=True；此处不得裸跑（否则 W 被写、y 变动）
        y_a = capture_y(model, st, [f0], f0["prompt_ids"][-1])
        y_b = capture_y(model, st, [f0], facts[1]["prompt_ids"][-1])
        y0 = {li: y_a[li][0] for li in st.mems}
        d_dummy = max(float((y_a[li][0] - y_b[li][0]).abs().max()) for li in st.mems)
        # 手算复核：y = sg(W) Wq c（同一 no_write 状态下重算）
        saved = {li: m.no_write for li, m in st.mems.items()}
        for m in st.mems.values():
            m.no_write = True
        try:
            set_m(st, None)
            set_capture(st, True)
            model(input_ids=torch.tensor([f0["prompt_ids"] + [f0["prompt_ids"][-1]]],
                                         dtype=torch.long, device=DEV))
            man = 0.0
            for li, m in st.mems.items():
                hs = m.prev_hs if False else st._hs_in[li]
                cf = hs.reshape(-1, hs.shape[-1])
                W = (m.W0 + m.U @ m.V.T) if m.use_factorized else m.Wfull
                yy = (sg(W) @ (m.Wq @ cf.T)).T
                man = max(man, float((yy[cf.shape[0] - 2] - y0[li]).abs().max()))
        finally:
            set_capture(st, False)
            for li, m in st.mems.items():
                m.no_write = saved[li]
        rec("A94-05", d_dummy == 0.0 and man < 1e-5,
            f"换 dummy token 的 y 逐位差={d_dummy:.1e}(须=0，因果性)；"
            f"与手算 W·Wq·c 差={man:.2e}(须<1e-5)", {"d_dummy": d_dummy, "manual": man})

        # A94-00 worked example：‖y‖/‖m_free‖/‖M y‖ + 两种 m 下的 4 个 NLL
        tg, _ = optimise_free(model, st, tok, facts, [0], __import__("random").Random(555))
        Yall = capture_y(model, st, facts, facts[0]["prompt_ids"][-1])
        Ms, lam0, res0 = {}, {}, {}
        Y2 = {li: Yall[li][[0, 1]] for li in st.mems}
        T2 = {li: tg[li] for li in st.mems}
        M16s, M32s = {}, {}
        for li in st.mems:
            M16s[li], lam0[str(li)], res0[str(li)] = ridge_closed(tg[li].T, tg[li].T)
        li0 = list(st.mems)[0]
        mfree = tg[li0][0]
        rec("A94-00b", True,
            f"worked: 层{li0} 事实0 ‖y0‖={float(Yall[li0][0].norm()):.3f} "
            f"‖m_free‖={float(mfree.norm()):.4f}", 
            {"y_norm": float(Yall[li0][0].norm()), "m_free_norm": float(mfree.norm())})
        RES["worked"] = {"layer": li0, "fact": 0,
                         "y_norm": float(Yall[li0][0].norm()),
                         "m_free_norm": float(mfree.norm())}

        # A94-01 无替换 ⇒ 逐位相同
        set_m(st, None)
        set_capture(st, False)
        inj = {}
        saved = {li: m.no_write for li, m in st.mems.items()}
        for m in st.mems.values():
            m.no_write = True
        model(input_ids=torch.tensor([f0["prompt_ids"] + [f0["prompt_ids"][-1]]],
                                     dtype=torch.long, device=DEV))
        for li, m in st.mems.items():
            inj[li] = m.injection.detach().clone()
        LayerMem.read = _ORIG_READ
        try:
            model(input_ids=torch.tensor([f0["prompt_ids"] + [f0["prompt_ids"][-1]]],
                                         dtype=torch.long, device=DEV))
            patched = {li: m.injection.detach().clone() for li, m in st.mems.items()}
        finally:
            LayerMem.read = _read_r28
            for li, m in st.mems.items():
                m.no_write = saved[li]
        d01 = max(float((inj[li] - patched[li]).abs().max()) for li in inj)
        rec("A94-01", d01 == 0.0, f"无替换时 patched 与原生 injection 逐位差={d01:.1e}(须=0)")

        # A94-02 只改读出位
        v = torch.ones(st.mems[li0].d_model, device=DEV) * 0.25
        if True:
            set_m(st, {li: (v if li == li0 else torch.zeros(st.mems[li].d_model, device=DEV))
                       for li in st.mems})
            model(input_ids=torch.tensor([f0["prompt_ids"] + [f0["prompt_ids"][-1]]],
                                         dtype=torch.long, device=DEV))
            got = st.mems[li0].injection.detach().clone().reshape(-1, v.shape[0])
            base = inj[li0].reshape(-1, v.shape[0])
            set_m(st, None)
        ridx = got.shape[0] - 2
        d_idx = float((got[ridx] - v).abs().max())
        others = [j for j in range(got.shape[0]) if j != ridx]
        d_oth = max(float((got[j] - base[j]).abs().max()) for j in others)
        rec("A94-02", d_idx == 0.0 and d_oth == 0.0,
            f"读出位行={d_idx:.1e}(须=0)；其余 {len(others)} 行逐位差={d_oth:.1e}(须=0)",
            {"d_idx": d_idx, "d_oth": d_oth})

        # A94-03 注入真被用上 + m=0 与 zero 臂等价
        set_capture(st, False)
        seq = torch.tensor([f0["prompt_ids"] + [f0["prompt_ids"][-1]]],
                           dtype=torch.long, device=DEV)
        P = len(f0["prompt_ids"])

        def lg_with(vec):
            if True:
                set_m(st, None if vec == "native" else
                      {li: (vec if vec is not None else
                            torch.zeros(st.mems[li].d_model, device=DEV)) for li in st.mems})
                out = model(input_ids=seq).logits.float().detach()
                set_m(st, None)
                return out

        l_nat = lg_with("native")
        l_zero = lg_with(None)
        l_big = lg_with(torch.ones(st.mems[li0].d_model, device=DEV) * 0.5)
        d03a = float((l_big[:, P - 1] - l_zero[:, P - 1]).abs().max())
        d03b = float((l_zero[:, P - 1] - l_nat[:, P - 1]).abs().max())
        rec("A94-03", d03a > 1e-3,
            f"m≠0 与 m=0 在读出位 logits 差={d03a:.3f}(须>1e-3) ⇒ 注入真被用上；"
            f"（m=0 与 native 差={d03b:.3f}，仅供参考）", {"d_big": d03a, "d_nat": d03b})

        # A94-04 γ 可见性
        g_zero_arms = float(st.mems[li0].gamma)
        rec("A94-04", g_zero_arms == 1.0,
            f"闭式/free/zero 臂的实际 γ={g_zero_arms:.3f}(须=1.0，否则注入被 γ 缩放或归零)")

        # A94-06 闭式解正确性（合成数据 vs lstsq）
        g = torch.Generator(device="cpu").manual_seed(7)
        # ★ ① n=64 > d_v=16 ⇒ 超定、解唯一；② T = M_true·Y ⇒ **精确相容**，
        #   否则「不相容系统的最小二乘解」与 lstsq 合法地不同（DEF-94-02）。
        Ys = torch.randn(16, 64, generator=g).double()
        M_true = torch.randn(64, 16, generator=g).double()
        Ts = M_true @ Ys
        ref = torch.linalg.lstsq(Ys.T, Ts.T).solution.T      # (d_model, d_v)
        M0, _, res0_ = ridge_closed(Ys, Ts, rel=0.0)         # ★ λ=0 ⇒ 纯代数，须与 lstsq 一致
        d_syn = float((M0 - ref).norm() / ref.norm().clamp_min(1e-30))
        Mf_, lam_, res_f = ridge_closed(Ys, Ts)              # 冻结 λ
        d_lam = float((Mf_ - ref).norm() / ref.norm().clamp_min(1e-30))
        rec("A94-06", d_syn < 1e-4 and res_f < 1e-2 and d_lam < 1e-2,
            f"λ=0 时 rel(M, lstsq)={d_syn:.2e}(须<1e-4，纯代数)；相容解残差={res_f:.1e}"
            f"(须<1e-2 —— λ>0 时残差本就是 O(λ/σ²)，要求更小等于要求正则化不生效)；"
            f"冻结 λ 的偏移={d_lam:.2e}(须<1e-2，即 λ 只是小扰动)",
            {"d_syn_l0": d_syn, "res": res_f, "d_lam": d_lam})

        # A94-08（CORR-14 强制）臂×独立变量表：走正式轮同一条构造路径
        tgt1, _ = optimise_free(model, st, tok, facts, [0, 1, 2, 3, 4, 5, 6, 7],
                               __import__("random").Random(555))
        Ys2 = {li: Yall[li][[0, 1, 2, 3, 4, 5, 6, 7]] for li in st.mems}
        mrow = {}
        mrow["zero"] = {li: torch.zeros(st.mems[li].d_model, device=DEV) for li in st.mems}
        mrow["free"] = {li: tgt1[li][0] for li in st.mems}
        Mfit1 = {}
        for li in st.mems:
            M, _, _ = ridge_closed(Ys2[li].T[:, :4], tgt1[li].T[:, :4])
            Mfit1[li] = (M.float() @ Ys2[li][0])
        mrow["ridge16"] = Mfit1
        Mfit2 = {}
        for li in st.mems:
            M, _, _ = ridge_closed(Ys2[li].T[:, :8], tgt1[li].T[:, :8])
            Mfit2[li] = (M.float() @ Ys2[li][0])
        mrow["ridge32"] = Mfit2
        z = float(max(float(mrow["zero"][li].abs().max()) for li in st.mems))
        keys3 = ["free", "ridge16", "ridge32"]
        dmin = min(float((mrow[a][li0] - mrow[b][li0]).abs().max())
                   for i, a in enumerate(keys3) for b in keys3[i + 1:])
        rec("A94-08", z == 0.0 and dmin > 1e-3,
            f"臂×变量表(结构检查，拟合集 4 vs 8 条)：zero 恒={z:.1e}(须=0)；"
            f"{{free,ridge16,ridge32}} 两两最小 max|Δm|={dmin:.3e}(须>1e-3)",
            {"zero": z, "dmin": dmin})

        # A94-09 free 收敛（走正式轮同一步数，只 1 条事实）
        tg2, ls2 = optimise_free(model, st, tok, facts, [0], __import__("random").Random(555))
        hd, tl = ls2[0]["head"], ls2[0]["tail"]
        nn = float(tg2[li0][0].norm())
        rec("A94-09", tl < 0.2 * hd and nn > 0,
            f"free 事实0 loss 首/尾={hd:.3f}/{tl:.3f}(须 tail<0.2·head)；‖m‖={nn:.4f}(须>0)",
            {"head": hd, "tail": tl, "m_norm": nn})

        # A94-11 评测口径
        items = make_eval_items(tok, facts, he, __import__("random").Random(555))
        ceil_ok = all(any(c[1] for c in it["cand"]) for it in items)
        n_q = len({it["key_idx"] for it in items})
        rec("A94-11", ceil_ok and n_q == 16 and len(items) == 48,
            f"n_q={n_q}(须16) n_item={len(items)}(须48) chance={CHANCE:.3f} "
            f"正确项必在候选={ceil_ok}")

        # A94-07 闭式机器阳性对照（in-sample，须 ≥0.75）
        Mv = {li: torch.stack([(ridge_closed(Ys2[li].T[:, :8], tgt1[li].T[:, :8])[0].float()
                                @ Yall[li][i]) for i in range(N_FACTS)]).float()
              for li in st.mems}
        zf = eval_inject(model, st, tok, facts, [0, 1, 2, 3, 4, 5, 6, 7], Mv,
                         __import__("random").Random(555))
        rec("A94-07", zf["acc_q"] >= 0.75,
            f"闭式机器在 8 条拟合事实上的 acc_q={zf['acc_q']:.3f}(须≥0.75) ⇒ 闭式解能拟合",
            {"acc_q": zf["acc_q"]})
    finally:
        st.remove()
    # ---- A94-10 写侧未被扰动（DEF-93-04 的正面回归）——**必须在本 state 已卸载之后**：
    #      两个 stack 同时挂载时，γ≠0 的那个会注入并扰动另一个的写轨迹（DEF-94-07）
    n_mounted = count_mounted(model, MOUNT)
    stx = make_stack(gamma=0.0, no_write=False, seed=seed)
    try:
        if True:
            V27.install_write()
            V27.ARM["name"] = "A_mean"
            W_init = {li: m.Wfull.detach().clone() for li, m in stx.mems.items()}

            def write_once():
                for li, m in stx.mems.items():
                    with torch.no_grad():
                        m.Wfull.copy_(W_init[li])
                    m.prev_hs = None
                    m.steps = 0
                    m.write_calls = 0
                V27.write_phase_arm(model, stx, chunks, owner, {})
                return {li: m.Wfull.detach().clone() for li, m in stx.mems.items()}

            Wa = write_once()
            Wb = write_once()
            d_rep = max(float((Wa[li] - Wb[li]).abs().max()) for li in Wa)
            # γ=0 时，即使把读出位注入置成非零，写轨迹也必须逐位不变
            set_m(stx, {li: torch.ones(stx.mems[li].d_model, device=DEV) * 0.5
                        for li in stx.mems})
            Wc = write_once()
            set_m(stx, None)
            d_ovr = max(float((Wa[li] - Wc[li]).abs().max()) for li in Wa)
            wcx = {li: m.write_calls for li, m in stx.mems.items()}
    finally:
        V27.restore_write()
        stx.remove()
    n_after = count_mounted(model, MOUNT)
    if True:
        rec("A94-10", d_rep == 0.0 and d_ovr == 0.0 and n_mounted == 0 and n_after == 0,
            f"挂载计数 {n_mounted}→{n_after}(须都是 0：不得有两个 stack 同时挂载)；"
            f"同初始 W 连写两遍逐位差={d_rep:.1e}(须=0，写路径可复现)；"
            f"γ=0 下挂非零 override 后与基线差={d_ovr:.1e}(须=0，读侧不得污染写侧)；"
            f"write_calls={wcx}",
            {"d_rep": d_rep, "d_override": d_ovr, "write_calls": wcx,
             "n_mounted_before": n_mounted, "n_mounted_after": n_after})

    return FAILS == []


# ------------------------------------------------------------------ 一个 seed
def run_seed(seed, tok, model, facts, chunks, owner, tr, he):
    t0 = time.time()
    st, wc, Wsnap = build_state(tok, model, facts, chunks, owner, seed)
    out = []
    try:
        set_gamma(st, 1.0)                  # 闭式/free/zero 臂：γ≡1（A94-04）
        Y = capture_y(model, st, facts, facts[0]["prompt_ids"][-1])
        rngf = __import__("random").Random(1000 + seed % 100)
        tgt32, floss = optimise_free(model, st, tok, facts, ALL32,
                                     __import__("random").Random(555))
        RES["arms"].append({"arm": "free_losses", "seed": seed,
                            "info": {str(k): v for k, v in floss.items()}})
        # zero
        for nm, Mv in (("zero", None), ("free", tgt32)):
            fit = eval_inject(model, st, tok, facts, tr, Mv, __import__("random").Random(555))
            gen = eval_inject(model, st, tok, facts, he, Mv, __import__("random").Random(555))
            out.append({"arm": nm, "seed": seed, "gamma_final": float(st.mems[list(st.mems)[0]].gamma),
                        "write_calls": wc, "acc_train": fit["acc_q"], "acc_held": gen["acc_q"],
                        "acc_item_train": fit["acc_item"], "acc_item_held": gen["acc_item"],
                        "margin_train": fit["margin_mean"], "margin_held": gen["margin_mean"],
                        "margin_pos_held": gen["margin_pos_frac"], "n_q": gen["n_q"],
                        "n_item": gen["n_item"], "info": {}, "sec": 0})
            print(f"  [{nm}] acc_train={out[-1]['acc_train']:.3f} "
                  f"acc_held={out[-1]['acc_held']:.3f} "
                  f"margin(tr/ho)={out[-1]['margin_train']:+.2f}/{out[-1]['margin_held']:+.2f}",
                  flush=True)
        # ridge16 / ridge32
        for nm, fitkeys in (("ridge16", tr), ("ridge32", ALL32)):
            Ms, lams, ress = {}, {}, {}
            for li in st.mems:
                Yl = torch.stack([Y[li][i] for i in fitkeys])
                Tl = torch.stack([tgt32[li][j] for j in range(len(fitkeys))])
                M, lam, res = ridge_closed(Yl.T, Tl.T)
                Ms[li], lams[str(li)], ress[str(li)] = M.float(), lam, res
            Mv = {li: (Ms[li] @ Y[li].T).T for li in st.mems}
            fit = eval_inject(model, st, tok, facts, tr, Mv, __import__("random").Random(555))
            gen = eval_inject(model, st, tok, facts, he, Mv, __import__("random").Random(555))
            out.append({"arm": nm, "seed": seed,
                        "gamma_final": float(st.mems[list(st.mems)[0]].gamma),
                        "write_calls": wc, "acc_train": fit["acc_q"], "acc_held": gen["acc_q"],
                        "acc_item_train": fit["acc_item"], "acc_item_held": gen["acc_item"],
                        "margin_train": fit["margin_mean"], "margin_held": gen["margin_mean"],
                        "margin_pos_held": gen["margin_pos_frac"], "n_q": gen["n_q"],
                        "n_item": gen["n_item"],
                        "info": {"lambda": lams, "rel_res": ress,
                                 "n_fit": len(fitkeys)}, "sec": 0})
            print(f"  [{nm}] acc_train={out[-1]['acc_train']:.3f} "
                  f"acc_held={out[-1]['acc_held']:.3f} "
                  f"margin(tr/ho)={out[-1]['margin_train']:+.2f}/{out[-1]['margin_held']:+.2f} "
                  f"n_fit={len(fitkeys)}", flush=True)
    finally:
        st.remove()
    for arm in ("adam16", "adam16_g1"):
        out.append(run_adam(arm, tok, model, facts, chunks, owner, tr, he, seed))
    for a in out:
        if a["sec"] == 0:
            a["sec"] = round(time.time() - t0, 1)
    return out


def decide():
    per = {}
    for s in SEEDS:
        d = {a["arm"]: a for a in RES["arms"] if a["seed"] == s and "acc_held" in a}
        if not all(k in d for k in ("free", "ridge16", "ridge32")):
            continue
        f, r16, r32 = d["free"]["acc_held"], d["ridge16"]["acc_held"], d["ridge32"]["acc_held"]
        if f <= G_LOW:
            b = "EXPR_FAIL"
        elif r16 >= G_GEN:
            b = "OPT_LIMIT"
        elif r16 <= G_LOW and r32 >= G_GEN:
            b = "SAMPLE_LIMIT"
        elif r16 <= G_LOW and r32 <= G_LOW:
            b = "LINEAR_LIMIT"
        else:
            b = "PARTIAL"
        per[str(s)] = {"free": f, "ridge16": r16, "ridge32": r32, "branch": b}
    bs = {v["branch"] for v in per.values()}
    return {"per_seed": per, "branch": (bs.pop() if len(bs) == 1 else "INCONCLUSIVE")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    # ★ A94-10：写轨迹须可复现 —— 关 dropout + 关 cudnn 自动算法选择（否则同一进程内
    #   不同时机的两遍写会因 kernel 选择不同而差 ~5e-4，与"读侧污染写侧"无法区分）
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tok, model = model_and_tok()
    model.eval()                       # ★ A94-10：写轨迹须逐位可复现（关掉 dropout）
    torch.set_grad_enabled(True)
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    tr = list(range(N_TRAIN_KEYS))
    he = list(range(N_TRAIN_KEYS, N_FACTS))
    if a.stage_a:
        print("===================== Stage A =====================", flush=True)
        ok = stage_a(tok, model, facts, chunks, owner, tr, he)
        RES["stage_a_ok"] = ok
        dump(RES, "probe91_woform_guards")
        print("\n* Stage A " + ("全部通过" if ok else "存在 FAIL —— 正式轮被阻断"), flush=True)
        sys.exit(0 if ok else 2)
    seeds = SEEDS[:1] if a.smoke else SEEDS
    for seed in seeds:
        print(f"===== seed {seed} =====", flush=True)
        RES["arms"].extend(run_seed(seed, tok, model, facts, chunks, owner, tr, he))
    RES["branch"] = decide()
    dump(RES, f"probe91_woform{'_smoke' if a.smoke else ''}")
    ex = {a["arm"]: a for a in RES["arms"] if a.get("seed") == seeds[0] and "acc_held" in a}
    for s, v in RES["branch"]["per_seed"].items():
        print(f"  seed {s}: free={v['free']:.3f} ridge16={v['ridge16']:.3f} "
              f"ridge32={v['ridge32']:.3f} -> {v['branch']}", flush=True)
    print(f"\n* 分支 = {RES['branch']['branch']}", flush=True)


if __name__ == "__main__":
    main()
