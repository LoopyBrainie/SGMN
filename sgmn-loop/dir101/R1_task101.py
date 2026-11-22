#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-101 —— 注入接口 `y → m` 是否为**共享函数**？测它的**秩**。

卡片：revolve/DIR-101/DIRECTION_rev101_v10.md（阈值冻结，不得改）

装置：`dir100/R1_task100.py` 的 **white 键臂**逐字（键正交、`maxcos=0`、内容 32/32 已由 D-128 判定）。
唯一自变量 = **`Wo` 的秩 `r`**（`Wo = U Vᵀ`，`U,V ∈ R^{896×r}`）。
训练集 = **全部 32 个事实**（本轮问**容量**，不问留出 —— 与 `D-111` 的 16 样本问题互补）。
主读数 = `acc_all32(r)`；`r_min = min{r : acc ≥ 0.99}`。

★ 不 import `probe91_woform`（它在 import 期就把 `LayerMem.read` 换掉，会与本文件的 patch 冲突）；
  `optimise_free` / `capture_y` 按 `dir91/probe91_woform.py:156-282` **逐字派生**到本文件。
"""
import argparse
import json
import pathlib
import random
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "dir91"))
sys.path.insert(0, str(_HERE.parent / "dir99"))
sys.path.insert(0, str(_HERE.parent / "dir100"))

import probe91_kpool as KP                       # noqa: E402
import R2_task91 as R91                          # noqa: E402
import R1_task100 as D100                        # noqa: E402
import torch                                     # noqa: E402
from sgmn_module import LayerMem, sg             # noqa: E402

DEV = R91.DEV
SEEDS = [20261028, 7777]
# ★ 冻结秩网格（卡片 §3）：含 27 = 不同答案值数（预测点）、32 = 事实数（判据点）、896 = 满秩（守卫点）
RANKS = [1, 2, 4, 8, 16, 27, 32, 896]
KEY_ARM = "white"          # ★ 键固定 white（D-128：32/32 内容）；自变量只有 rank(Wo)
THRESH_PASS = 0.99         # C101-CAP
THRESH_FAIL = 0.90         # C101-FAIL
STEPS_FREE = 120           # = probe91_woform.STEPS_FREE
LR_FREE = 0.05             # = probe91_woform.LR_FREE
OUT = _HERE / "out"
RES = {"guards": {}, "cells": []}

_ORIG_READ = LayerMem.read
_ORIG_WRITE = LayerMem.write


def _grad(n):
    """在 `set_grad_enabled(False)` 的全局上下文里为优化循环临时开梯度（`DEF-101-04`）。"""
    for i in range(n):
        with torch.enable_grad():
            yield i


def rec(cid, ok, detail, value=None):
    RES["guards"][cid] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"  {'✓' if ok else '✗ FAIL'} {cid:<11} {detail}", flush=True)
    return ok


# ==================================================================== 读侧 patch
def _read_r101(self, hs_in, drop_sg=False):
    """= `dir100/R1_task100._read_tf` 逐字；**新增**三件（互不干扰）：
       ① `Wo` 秩因子化：`self._WU @ self._WV.T`（None ⇒ 原生 `self.Wo`）
       ② `_m_override` 覆盖读出位（自由向量臂）
       ③ `_y_capture` 记录读出位 `y`（A101-2 用）
    """
    u = getattr(self, "_WU", None)
    mo = getattr(self, "_m_override", None)
    cap = getattr(self, "_y_capture", False)
    if u is None and mo is None and not cap:
        return D100._read_tf(self, hs_in, drop_sg)
    self.read_calls += 1
    c = sg(hs_in, drop_sg)
    cf = c.reshape(-1, c.shape[-1])
    q = (self.Wq @ cf.T).T
    q = self.q_tf(q)
    W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
    y = (sg(W, drop_sg) @ q.T).T
    if u is None:
        m = (self.Wo @ y.T).T
    else:
        m = (y @ (self._WV @ u.T))                 # y (N,d_v) · (d_v,r)(r,d_model) = (N,d_model) = Wo_eff @ y
    if cap:
        Bn, Ln = hs_in.shape[0], hs_in.shape[1]
        idx = torch.arange(Bn, device=cf.device) * Ln + (Ln - 2)
        self._y_out = y[idx[0]].detach().clone()
    if mo is not None:
        Bn, Ln = hs_in.shape[0], hs_in.shape[1]
        idx = torch.arange(Bn, device=cf.device) * Ln + (Ln - 2)
        m = m.clone()
        m[idx] = mo.to(device=m.device, dtype=m.dtype)
    self.injection = m.reshape(hs_in.shape)
    return self.injection


def install():
    LayerMem.write = D100._write_tf
    LayerMem.read = _read_r101


def restore():
    LayerMem.write = _ORIG_WRITE
    LayerMem.read = _ORIG_READ


def set_m(st, vec):
    """★ `vec` 是 **{li: (d_model,)}**；必须按层取用（`DEF-101-02`：曾把整个 dict 塞进一层）。"""
    for li, m in st.mems.items():
        m._m_override = None if vec is None else vec[li]


def set_capture(st, on):
    for m in st.mems.values():
        m._y_capture = bool(on)
        if not on:
            m._y_out = None


def set_gamma(st, val):
    """`gamma` 是注入的总幅度；`make_stack(gamma=0.0)` ⇒ **注入被整体乘零**（`DEF-101-05`）。
    `optimise_free` 只优化 `m`、**不训 `gamma`** ⇒ 自由向量臂必须显式置 1（同 `probe91_woform` 的做法）。"""
    for m in st.mems.values():
        m.gamma.data.fill_(float(val))


def set_rank(st, U, V):
    for li, m in st.mems.items():
        m._WU = None if U is None else U[li]
        m._WV = None if V is None else V[li]


# ==================================================================== 自由向量（A101-1）
def optimise_free(model, st, tok, facts, keys, rng, steps=STEPS_FREE):
    """逐字派生 `dir91/probe91_woform.py:236`：每事实每层一个自由向量 m_i，只优化 m_i。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    items = R91.make_eval_items(tok, facts, keys, rng)
    by_fi = {}
    for it in items:
        by_fi.setdefault(it["key_idx"], []).append(it)
    Mv = {li: {} for li in st.mems}
    try:
        set_m(st, None)
        for fi in keys:
            its = by_fi[fi]
            P = len(its[0]["prompt_ids"])
            nc = len(its[0]["cand"])
            seqs = [it["prompt_ids"] + [c[0][0]] for it in its for c in it["cand"]]
            seq = torch.tensor(seqs, dtype=torch.long, device=DEV)
            tok_ids = torch.tensor([s[-1] for s in seqs], device=DEV)
            tgt = torch.tensor([next(j for j, c in enumerate(it["cand"]) if c[1])
                                for it in its], dtype=torch.long, device=DEV)
            pars = {li: torch.zeros(st.mems[li].d_model, device=DEV, requires_grad=True)
                    for li in st.mems}
            opt = torch.optim.Adam(pars.values(), lr=LR_FREE)
            for _ in _grad(steps):
                opt.zero_grad(set_to_none=True)
                set_m(st, {li: pars[li] for li in st.mems})
                lg = model(input_ids=seq).logits.float()
                ar = torch.arange(len(seqs), device=DEV)
                pick = lg[ar, P - 1, tok_ids].reshape(len(its), nc)
                loss = torch.nn.functional.cross_entropy(pick, tgt)
                loss.backward()
                opt.step()
            for li in st.mems:
                Mv[li][fi] = pars[li].detach().clone()
    finally:
        set_m(st, None)
        for li, m in st.mems.items():
            m.no_write = saved[li]
    return Mv


def capture_y(model, st, facts, dummy):
    """逐字派生 `probe91_woform.py:156`；用本文件的 `_y_out`。"""
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


# ==================================================================== 评测
@torch.no_grad()
def eval_acc(model, st, tok, facts, keys, seed=555):
    """4 选 1、按题计（同题 N_DRAWS 次全对才算命中）；`no_write`；返回 (acc, n_q)。"""
    rng = random.Random(seed)
    items = R91.make_eval_items(tok, facts, keys, rng)
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    per = {}
    try:
        set_m(st, None)
        for it in items:
            P = len(it["prompt_ids"])
            seqs = [it["prompt_ids"] + [c[0][0]] for c in it["cand"]]
            seq = torch.tensor(seqs, dtype=torch.long, device=DEV)
            tok_ids = torch.tensor([s[-1] for s in seqs], device=DEV)
            lg = model(input_ids=seq).logits.float()
            ar = torch.arange(len(seqs), device=DEV)
            pick = lg[ar, P - 1, tok_ids]
            best = int(torch.argmax(pick))
            per.setdefault(it["key_idx"], []).append(it["cand"][best][1])
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
    nq = len(per)
    hit = sum(1 for v in per.values() if all(v))
    return hit / nq, nq


# ==================================================================== 秩因子化训练
def init_rank_from_Wo(st, rank):
    """把每层原生 `Wo`（= 装置里的 `mk(d_model,d_v)` 初始化）做**秩 `rank` 截断**
    ⇒ `U@Vᵀ` = `Wo_init` 的最佳秩 `rank` 近似。★ 使各 rank 的**初始分布可比**。"""
    U, V = {}, {}
    for li, m in st.mems.items():
        Wo = m.Wo.detach().float()                      # (d_model, d_v)
        if rank >= Wo.shape[0]:
            raise ValueError("rank>=d 应走满秩分支")
        Uu, Ss, Vv = torch.linalg.svd(Wo, full_matrices=False)
        s = Ss[:rank].clamp_min(0.0).sqrt()
        U[li] = (Uu[:, :rank] * s).clone().requires_grad_(True)          # (d_model, r)
        V[li] = (Vv[:rank, :].T * s).clone().requires_grad_(True)        # (d_v, r)
    return U, V


def train_rank_wq_gamma(model, st, tok, facts, train_keys, U, V, rng,
                        steps=R91.READ_STEPS, lr=R91.READ_LR):
    """逐字派生 `dir100.R1_task100.train_read_frozen_wq`（`freeze_wq=False` 分支）；
    **唯一差别**：`Wo` 不直接训练，改为训练 `U,V`（`Wo_eff = U Vᵀ`）。
    `W` 冻结（no_write）、`W0/U/V/Wfull` 不进阶（守卫 A101-4）。"""
    for m in st.mems.values():
        m.no_write = True
    ps = []
    for li, m in st.mems.items():
        ps += [U[li], V[li], m.Wq, m.gamma]
    opt = torch.optim.Adam(ps, lr=lr)
    losses, viol, div = [], 0, False
    batches = []
    for ki in train_keys:
        f = facts[ki]
        others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != f["val"]]
        distr = rng.sample(sorted(set(others)), R91.N_CAND - 1)
        cands = [f["val"]] + distr
        rng.shuffle(cands)
        ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in cands]
        batches.append((f["prompt_ids"], ids, cands.index(f["val"])))
    with torch.enable_grad():
      for step in range(steps):
        prompt_ids, ids, tgt = batches[step % len(batches)]
        P = len(prompt_ids)
        seq = torch.tensor([prompt_ids + [c] for c in ids], dtype=torch.long, device=DEV)
        set_rank(st, U, V)
        opt.zero_grad(set_to_none=True)
        lg = model(input_ids=seq).logits.float()
        idx = torch.arange(len(ids), device=DEV)
        pick = lg[idx, P - 1, torch.tensor(ids, device=DEV)].unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(pick, torch.tensor([tgt], device=DEV))
        loss.backward()
        if step % 50 == 0:
            for p in R91.write_params(st):
                if p.grad is not None and float(p.grad.abs().max()) != 0.0:
                    viol += 1
        torch.nn.utils.clip_grad_norm_([p for p in ps if p.grad is not None], R91.CLIP)
        opt.step()
        losses.append(float(loss))
        if not all(bool(torch.isfinite(R91.W_of(m)).all()) for m in st.mems.values()):
            div = True
            break
    set_rank(st, None, None)
    for m in st.mems.values():
        m.no_write = False
    return losses, viol, div


def train_full_wo(model, st, tok, facts, train_keys, rng, steps=R91.READ_STEPS):
    """±896 = **满秩对照**：直接用原生 `Wo`（与 U,V 满秩等价），派生自
    `R91.train_read_task` 的 `read_params` 分支（`Wq`/`Wo`/`gamma` 全训）。"""
    for m in st.mems.values():
        m.no_write = True
    ps = [p for m in st.mems.values() for p in (m.Wq, m.Wo, m.gamma)]
    opt = torch.optim.Adam(ps, lr=R91.READ_LR)
    losses, viol, div = [], 0, False
    batches = []
    for ki in train_keys:
        f = facts[ki]
        others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != f["val"]]
        distr = rng.sample(sorted(set(others)), R91.N_CAND - 1)
        cands = [f["val"]] + distr
        rng.shuffle(cands)
        ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in cands]
        batches.append((f["prompt_ids"], ids, cands.index(f["val"])))
    with torch.enable_grad():
      for step in range(steps):
        prompt_ids, ids, tgt = batches[step % len(batches)]
        P = len(prompt_ids)
        seq = torch.tensor([prompt_ids + [c] for c in ids], dtype=torch.long, device=DEV)
        opt.zero_grad(set_to_none=True)
        lg = model(input_ids=seq).logits.float()
        idx = torch.arange(len(ids), device=DEV)
        pick = lg[idx, P - 1, torch.tensor(ids, device=DEV)].unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(pick, torch.tensor([tgt], device=DEV))
        loss.backward()
        if step % 50 == 0:
            for p in R91.write_params(st):
                if p.grad is not None and float(p.grad.abs().max()) != 0.0:
                    viol += 1
        torch.nn.utils.clip_grad_norm_([p for p in ps if p.grad is not None], R91.CLIP)
        opt.step()
        losses.append(float(loss))
        if not all(bool(torch.isfinite(R91.W_of(m)).all()) for m in st.mems.values()):
            div = True
            break
    for m in st.mems.values():
        m.no_write = False
    return losses, viol, div


# ==================================================================== 手算算例（A101-9）
def toy_example():
    U = torch.eye(2)
    V = torch.eye(2)
    y = torch.tensor([[3.0, 4.0]])
    Wo = U @ V.T
    m = (y @ (t := Wo).T)                          # = Wo y（行向量约定）
    d = float((m.squeeze(0) - torch.tensor([3.0, 4.0])).abs().max())
    nrm = float(m.norm())
    rk = int(torch.linalg.matrix_rank(Wo))
    return rec("A101-9", d < 1e-12 and abs(nrm - 5.0) < 1e-12 and rk == 2,
               f"手算：‖Wo@yᵀ−[3,4]‖={d:.1e}(须=0)、‖m‖={nrm:.1f}(须=5)、rank={rk}(须=2)",
               {"diff": d, "norm": nrm, "rank": rk})


# ==================================================================== 单格
def run_cell(model, tok, facts, seed, rank, stage, chunks, owner):
    t0 = time.time()
    st = R91.make_stack(gamma=0.0, no_write=False, seed=seed)
    res = {"seed": seed, "rank": rank, "stage": stage}
    try:
        # --- 写 600 块（W 真写），白键臂
        for m in st.mems.values():
            m.key_tf = D100.IdentityTF()
            m.q_tf = D100.IdentityTF()
            m.v_neg_shift = False
            m.v_hold = None
        install()
        Kraw, V = D100.capture_keys(model, st, facts, chunks, owner)
        TF = {}
        for li in Kraw:
            TF[li] = D100.build_tf(KEY_ARM, Kraw[li], V[li], li)
            st.mems[li].key_tf = TF[li]
            st.mems[li].q_tf = TF[li].qmap
        KP.write_phase_arm(model, st, chunks, owner,
                           {i: None for i in range(len(facts))})
        W_write = {li: R91.W_of(m).detach().clone() for li, m in st.mems.items()}
        allk = list(range(len(facts)))
        rng = random.Random(555)

        if rank == "free":                                    # A101-1 表达力上限
            set_gamma(st, 1.0)                                # ★ DEF-101-05
            Mv = optimise_free(model, st, tok, facts, allk, random.Random(555))
            # 评测：逐事实代入 m_i（同题只用该事实的 m_i）
            acc = eval_free(model, st, tok, facts, allk, Mv)
            res.update({"acc_all32": acc, "nq": len(allk)})
        elif rank == 896 or rank == "wo_init":                 # 满秩 / 未训练
            if rank == "wo_init":
                set_gamma(st, 1.0)                            # ★ DEF-101-05：否则测的是"无注入"
                acc, nq = eval_acc(model, st, tok, facts, allk)
                res.update({"acc_all32": acc, "nq": nq})
            else:
                U, V = None, None
                ls, viol, div = train_full_wo(model, st, tok, facts, allk, rng)
                acc, nq = eval_acc(model, st, tok, facts, allk)
                res.update({"acc_all32": acc, "nq": nq, "loss_tail": ls[-20:],
                            "viol": viol, "div": div})
        else:                                                  # 秩因子化
            U, V = init_rank_from_Wo(st, rank)
            ls, viol, div = train_rank_wq_gamma(model, st, tok, facts, allk, U, V, rng)
            set_rank(st, U, V)
            acc, nq = eval_acc(model, st, tok, facts, allk)
            set_rank(st, None, None)
            eff = {li: int(torch.linalg.matrix_rank((U[li] @ V[li].T).detach().float()))
                   for li in st.mems}
            res.update({"acc_all32": acc, "nq": nq, "loss_tail": ls[-20:],
                        "viol": viol, "div": div, "rank_eff": eff})

        # A101-4：读训练不得改 W
        dW = max(float((R91.W_of(m).detach() - W_write[li]).norm()
                       / W_write[li].norm().clamp_min(1e-30))
                 for li, m in st.mems.items())
        res["W_drift"] = dW
    finally:
        restore()
        st.remove()
        torch.cuda.empty_cache()
    res["secs"] = round(time.time() - t0, 1)
    return res


@torch.no_grad()
def eval_free(model, st, tok, facts, keys, Mv, seed=555):
    """自由向量臂的评测：每题的 m 用**该题自己**的 `m_i`（逐题逐层）。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    rng = random.Random(seed)
    items = R91.make_eval_items(tok, facts, keys, rng)
    per = {}
    try:
        for it in items:
            P = len(it["prompt_ids"])
            seqs = [it["prompt_ids"] + [c[0][0]] for c in it["cand"]]
            seq = torch.tensor(seqs, dtype=torch.long, device=DEV)
            tok_ids = torch.tensor([s[-1] for s in seqs], device=DEV)
            set_m(st, {li: Mv[li][it["key_idx"]] for li in st.mems})
            lg = model(input_ids=seq).logits.float()
            ar = torch.arange(len(seqs), device=DEV)
            pick = lg[ar, P - 1, tok_ids]
            best = int(torch.argmax(pick))
            per.setdefault(it["key_idx"], []).append(it["cand"][best][1])
    finally:
        set_m(st, None)
        for li, m in st.mems.items():
            m.no_write = saved[li]
    nq = len(per)
    return sum(1 for v in per.values() if all(v)) / nq


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stageA", choices=["stageA", "formal", "smoke"])
    ap.add_argument("--ranks", default="", help="逗号分隔，覆盖 RANKS")
    ap.add_argument("--seeds", default="", help="逗号分隔，覆盖 SEEDS")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    tok, model = KP.model_and_tok()
    model.eval()
    facts = R91.build_facts(tok)
    chunks, owner = R91.build_chunks(tok, facts)
    ranks = [x for x in a.ranks.split(",") if x] or RANKS
    # ★ DEF-101-08（Cycle 189 修）：原写 `[int(x) if x.isdigit() else x for x in ranks]` ——
    #   当 `--ranks` 为空时 `ranks` 就是 **`RANKS` 本身（已经是 int）** ⇒ `x.isdigit()` 抛
    #   `AttributeError: 'int' object has no attribute 'isdigit'`。
    #   该缺陷只在**默认路径**触发（显式传 `--ranks` 时元素是 str，反而正常）⇒ Cycle 187 的 smoke 未暴露。
    #   修法：用 `isinstance(x, str)` 判型，两种写法都能跑。
    ranks = [int(x) if isinstance(x, str) and x.isdigit() else x for x in ranks]
    seeds = [int(x) for x in a.seeds.split(",") if x] or (SEEDS if a.stage == "formal" else SEEDS[:1])
    C = []
    chunk_cache = {"_chunks": chunks, "_owner": owner}
    try:
        toy_example()
        print(f"[setup] facts={len(facts)} chunks={len(chunks)} "
              f"值池={len(KP.VALUE_POOL) if hasattr(KP,'VALUE_POOL') else len(R91.VALUE_POOL)} "
              f"不同答案值={len({f['val'] for f in facts})} ranks={ranks} seeds={seeds}", flush=True)

        # ---- A101-0 装置复现 + A101-1/A101-2 自由向量闸（只跑一次 seed0）
        s0 = seeds[0]
        st = R91.make_stack(gamma=0.0, no_write=False, seed=s0)
        try:
            for m in st.mems.values():
                m.key_tf = D100.IdentityTF()
                m.q_tf = D100.IdentityTF()
                m.v_neg_shift = False
                m.v_hold = None
            install()
            Kraw, Vv = D100.capture_keys(model, st, facts, chunks, owner)
            TF = {}
            for li in Kraw:
                TF[li] = D100.build_tf(KEY_ARM, Kraw[li], Vv[li], li)
                st.mems[li].key_tf = TF[li]
                st.mems[li].q_tf = TF[li].qmap
            # ★ white 键 = 过 tf 之后的键（W 里实际写的就是它）；raw 键近共线是 D-114 的已知事实
            cosmax = {li: float(_maxoff(D100.norm_rows((tf.A @ Kraw[li].float().T).T)))
                      for li, tf in TF.items()}
            rec("A101-0", max(cosmax.values()) < 1e-4,
                "white（过 tf 后）键两两 |cos| 最大 = " +
                " / ".join(f"{v:.2e}" for v in cosmax.values()) + "（须 <1e-4，复现 D-128/D-130）",
                {"cosmax": cosmax, "note": "raw 键近共线属 D-114 已知，不作守卫"})
            KP.write_phase_arm(model, st, chunks, owner, {i: None for i in range(len(facts))})
            allk = list(range(len(facts)))
            set_gamma(st, 1.0)          # ★ DEF-101-05：optimise_free 只优化 m、不训 gamma
            Mv = optimise_free(model, st, tok, facts, allk, random.Random(555))
            accF = eval_free(model, st, tok, facts, allk, Mv)
            rec("A101-1", accF >= THRESH_PASS,
                f"free 臂（逐事实自由 m_i）acc_all32={accF:.4f}（须 ≥{THRESH_PASS}）", {"acc": accF})
            Y = capture_y(model, st, facts, facts[0]["prompt_ids"][-1])
            ok2, det2 = _guard_closed_form(st, Kraw, Y, Mv, model, tok, facts, allk)
            # ★ DEF-101-07（rev 1.1）：A101-2 **降级为 report-only** —— 它对"任意选中的解集"做回归，
            #   与卡片 §1 否决任务行路线的理由**同源**，不能当阻断式守卫。读数照记、不隐瞒。
            RES["report"] = RES.get("report", {})
            RES["report"]["A101-2"] = {"ok": ok2, "detail": det2[0], "value": det2[1],
                                       "status": "report-only（DEF-101-07）"}
            print(f"  · A101-2(report-only) {det2[0]}", flush=True)
            # ---- A101-6：`y` 是否携带信息（装置性质，与解集无关）
            yr = _guard_y_info(Y, facts, allk)
            rec("A101-6", yr["rank_y"] >= 8,
                f"rank({{y_i}})={yr['rank_y']}（须 ≥8）；"
                f"|cos| 同值对均值={yr['same']:.3f} vs 不同值对均值={yr['diff']:.3f}",
                yr)
            RES["y_info"] = yr
            set_gamma(st, 1.0)          # ★ DEF-101-05：A101-3 须测"有注入但 Wo 未训练"
            acc0, nq0 = eval_acc(model, st, tok, facts, allk)
            rec("A101-3", acc0 <= 0.35, f"未训练 Wo（gamma=1）的 acc_all32={acc0:.4f}（须 ≤0.35）",
                {"acc": acc0})
        finally:
            restore()
            st.remove()
            torch.cuda.empty_cache()

        # ---- 秩网格
        for seed in seeds:
            for r in ranks:
                res = run_cell(model, tok, facts, seed, r, a.stage, chunks, owner)
                C.append({k: v for k, v in res.items() if not k.startswith("_")})
                print(f"[{seed} r={r}] acc_all32={res['acc_all32']:.4f} "
                      f"W_drift={res.get('W_drift', float('nan')):.2e} {res['secs']}s", flush=True)
        RES["cells"] = C
    finally:
        torch.set_grad_enabled(True)

    # ---- A101-4 / A101-5
    if C:
        dmax = max(c.get("W_drift", 0.0) for c in C)
        rec("A101-4", dmax < 1e-12, f"读训练期 W 漂移最大 = {dmax:.3e}（须 =0，A97-4）", {"dmax": dmax})
        rec("A101-5", all(c.get("nq", 0) > 0 for c in C),
            f"所有 {len(C)} 格均有读数（nq>0）", {"n": len(C)})

    # ---- 判据
    if a.stage == "formal" and all(g["ok"] for g in RES["guards"].values()):
        by = {}
        for c in C:
            by.setdefault(c["rank"], {})[c["seed"]] = c["acc_all32"]
        means = {r: sum(d.values()) / len(d) for r, d in by.items()}
        r_min = None
        for r in sorted(x for x in by if isinstance(x, int)):
            if means[r] >= THRESH_PASS:
                r_min = r
                break
        cap = means.get(896, means.get(32, 0.0))
        if cap >= THRESH_PASS and r_min is not None:
            verdict = "A-SHARED-LOWRANK"
        elif cap < THRESH_FAIL and all(v < THRESH_FAIL for v in means.values()):
            verdict = "A-NOT-SHARED"
        else:
            verdict = "A-PARTIAL"
        RES.update({"means": {str(k): v for k, v in means.items()},
                    "r_min": r_min, "verdict": verdict, "n_distinct_vals": len({f['val'] for f in facts})})
        print("[DIR-101] 各秩 acc_all32 = " +
              " ".join(f"r{k}={v:.4f}" for k, v in sorted(means.items(), key=lambda t: (isinstance(t[0], str), t[0]))),
              flush=True)
        print(f"[DIR-101] r_min（≥{THRESH_PASS}）= {r_min}；不同答案值数 = {RES['n_distinct_vals']}", flush=True)
        print(f"[DIR-101] 判决 = {verdict}", flush=True)
    else:
        RES["verdict"] = "BLOCKED" if a.stage == "formal" else "STAGE_A_ONLY"
        if a.stage == "formal":
            bad = [k for k, g in RES["guards"].items() if not g["ok"]]
            print(f"[DIR-101] 守卫失败 ⇒ BLOCKED：{bad}", flush=True)

    (OUT / f"stage_{a.stage}.json").write_text(
        json.dumps(RES, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[完成] → {OUT / f'stage_{a.stage}.json'}", flush=True)
    return 0


def _maxoff(N):
    """N 已行归一 ⇒ N Nᵀ 的非对角最大绝对值。"""
    G = N @ N.T
    d = torch.diagonal(G)
    G = G - torch.diag(d)
    return float(G.abs().max())


@torch.no_grad()
def _guard_closed_form(st, Kraw, Y, Mv, model, tok, facts, allk):
    """A101-2：闭式 `G = M Yᵀ(YYᵀ+λI)⁻¹`（**全部 32**）⇒ 装成 `Wo` 后 acc_all32 须 ≥0.99。
    ★ 这是**接口可行性**守卫：证明共享线性 `y→m` **能承载**这些解（与优化器无关）。"""
    info = {}
    saved = {li: m.Wo.detach().clone() for li, m in st.mems.items()}
    try:
        for li, m in st.mems.items():
            M = torch.stack([Mv[li][i] for i in allk], dim=1).double()      # (d_model, 32)
            Yl = Y[li].T.double()                                           # (d_v, 32)
            G = _ridge_closed(M, Yl)                                        # (d_model, d_v)
            info[str(li)] = float((G @ Yl - M).norm() / M.norm().clamp_min(1e-30))
            m.Wo.data.copy_(G.to(dtype=m.Wo.dtype, device=m.Wo.device))
        acc, nq = eval_acc(model, st, tok, facts, allk)
        return acc >= THRESH_PASS, (
            f"闭式 G=M Yᵀ(YYᵀ+λI)⁻¹（全 32）⇒ acc_all32={acc:.4f}（须 ≥{THRESH_PASS}）；"
            f"拟合相对残差 {max(info.values()):.2e}", {"acc": acc, "res": info})
    finally:
        for li, m in st.mems.items():
            m.Wo.data.copy_(saved[li])


@torch.no_grad()
def _guard_y_info(Y, facts, allk):
    """A101-6：`y` 携带多少信息。与 `optimise_free` 的解集**无关**（装置性质）。
    `rank_y` = `{y_i}` 的秩；`same/diff` = 同答案值对 vs 不同答案值对上的 |cos| 均值。"""
    out = {}
    for li, Ym in Y.items():                       # Ym (32, d_v)
        Yt = Ym.float().T                          # (d_v, 32)
        sv = torch.linalg.svdvals(Yt)
        rk = int((sv > sv[0] * 1e-6).sum())
        Yn = D100.norm_rows(Ym.float())
        Cm = (Yn @ Yn.T).abs()
        n = Cm.shape[0]
        off = ~torch.eye(n, dtype=torch.bool, device=Cm.device)
        same, diff = [], []
        for i in range(n):
            for j in range(i + 1, n):
                (same if facts[i]["val"] == facts[j]["val"] else diff).append(float(Cm[i, j]))
        out[str(li)] = {"rank_y": rk, "sv1": float(sv[0]),
                        "same": sum(same) / max(len(same), 1), "n_same": len(same),
                        "diff": sum(diff) / max(len(diff), 1), "n_diff": len(diff),
                        "off_max": float(Cm[off].max())}
    f0 = out[str(min(out, key=lambda x: int(x)))]
    return {"rank_y": f0["rank_y"], "same": f0["same"], "diff": f0["diff"],
            "n_same": f0["n_same"], "n_diff": f0["n_diff"], "per_layer": out}


def _ridge_closed(T, Yd, rel=1e-6):
    """逐字派生 `probe91_woform.ridge_closed`：M = T Yᵀ (Y Yᵀ + λI)⁻¹；Y,T 为 (d, n)。"""
    A = Yd @ Yd.T
    lam = rel * float(torch.diagonal(A).sum()) / max(Yd.shape[1], 1)
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    return T @ Yd.T @ torch.linalg.inv(A + lam * I)


if __name__ == "__main__":
    sys.exit(main())