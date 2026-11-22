#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-105 —— `\\mathcal O_l` 的**非线性族**与**更长训练**：`acc_all32` 的 `0.9688` 是预算假象还是结构上限？

卡片：revolve/DIR-105/DIRECTION_rev105_v10.md（rev 1.0 冻结；阈值 0.99 / 0.50 不得改）

装置：`dir103/R1_task103.py` **逐字派生**（white 键臂 = 上界臂、内容 `D-128` 判 31–32/32）。
唯一自变量（两项，正是 `D-133` §4 的**剩余缺口 ①**）：
    --family {lin, gate, mlp1}      ← `\\mathcal O_l` 的内部结构（框架 §7.2 明确开放项）
    --steps  {600, 3000}            ← 更长训练（`O103-1` 的推测）
    lin : m = Wo @ y + Wc @ c              （= `D-133` 的 `cy` 臂，逐字，作复现臂）
    gate: g = sigmoid(Wg @ c); m = Wo @ (g * y) + Wc @ c   （乘性门控：用 c 的身份选 y 的值）
    mlp1: h = GELU(W1 @ [y;c] + b1); m = W2 @ h            （通用单隐层非线性）
训练集 = 全部 32 个事实（问**容量**，不问留出）。
主判据 = `acc_all32 >= 0.99`  **且**  `acc_W0 <= 0.50`（后者排除读出侧记忆化，`D-133` 的教训）。

★ `A105-1` 要求 `lin@600` **逐位复现** `D-133` 的 `cy` = 31/32, 28/32
  ⇒ 所有新参数必须用**本地 Generator** 初始化，**严禁**扰动全局 RNG（`DEF-103-*` 的教训）。
★ 不 import `probe91_woform`（import 期会换掉 `LayerMem.read`）。
"""
import argparse
import hashlib
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
sys.path.insert(0, str(_HERE.parent / "dir101"))

import probe91_kpool as KP                       # noqa: E402
import R2_task91 as R91                          # noqa: E402
import R1_task100 as D100                        # noqa: E402
import torch                                     # noqa: E402
from sgmn_module import LayerMem, sg             # noqa: E402

DEV = R91.DEV
SEEDS = [20261028, 7777]
KEY_ARM = "white"                 # ★ 键固定 white（D-128：内容 31–32/32）
FAMILIES = ["lin", "gate", "mlp1"]
STEPS_GRID = [600, 3000]
# ---- 冻结阈值（卡片 §2.1；**不得改**）----
THR_ROUTE = 0.99                  # `M105-1`：`acc_all32 >= 0.99`（沿用 D-133，一字未改）
THR_NOTMEM = 0.50                 # `M105-2`：`acc_W0 <= 0.50`（非记忆化）
# ---- `A105-1` 复现 D-133 的 `cy`（逐位，按题计）----
REPRO_LIN600 = {20261028: 31, 7777: 28}   # 31/32 = 0.96875、28/32 = 0.8750（D-133 §1.1）
NQ_EXPECT = 32                    # `A105-5`
OUT = _HERE / "out"
RES = {"guards": {}, "cells": [], "toy": {}}

_ORIG_READ = LayerMem.read
_ORIG_WRITE = LayerMem.write

# ★ 全局状态（patch 内读取；逐格显式设置/还原）
_FAM = "lin"               # lin | gate | mlp1
_PG = {}                   # {li: Wg}     gate
_PM = {}                   # {li: (W1,W2,b1)}  mlp1
_WC = {}                   # {li: Tensor(d_model,d_model)}
_C_CAP = False
_C_OUT = None


def rec(cid, ok, detail, value=None):
    RES["guards"][cid] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"  {'✓' if ok else '✗ FAIL'} {cid:<11} {detail}", flush=True)
    return ok


def rec_only(cid, detail, value=None):
    """只记录、不参与 `guards_ok` 判定（`DEF-101-07` 的先例）。"""
    RES["guards"][cid] = {"ok": True, "detail": detail, "value": value, "report_only": True}
    print(f"  ○ {cid:<11} {detail}", flush=True)


# ==================================================================== 读侧 patch
def _read_r105(self, hs_in, drop_sg=False):
    """= `dir103/R1_task103._read_r103` 逐字；**唯一改动**：`m` 的族可换（`_FAM`）。

    `lin`  : m = Wo @ y (+ Wc @ c)                      ← `D-133` 的 `cy` 臂，逐字
    `gate` : g = sigmoid(Wg @ c); m = Wo @ (g * y) (+ Wc @ c)   ← 乘性门控
    `mlp1` : h = GELU(W1 @ [y;c] + b1); m = W2 @ h                ← 单隐层非线性
    ★ `y` 的公式与 `c` 的 sg 位置**一字未改**（`D-133` 的 `A103-4` 已核过 c 的捕获）。
    """
    wc = _WC.get(id(self)) if _WC else None
    cap = _C_CAP
    self.read_calls += 1
    c = sg(hs_in, drop_sg)
    cf = c.reshape(-1, c.shape[-1])
    q = (self.Wq @ cf.T).T
    q = self.q_tf(q)
    W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
    y = (sg(W, drop_sg) @ q.T).T                       # (N, d_v)
    if _FAM == "lin":
        m = (self.Wo @ y.T).T
        if wc is not None:
            m = m + (wc @ cf.T).T
    elif _FAM == "gate":
        wg = _PG.get(id(self))
        if wg is None:                                 # `DEF-105-01`：给出可读错误
            raise RuntimeError(f"gate 族参数缺失（layer id={id(self)}）；"
                               f"`set_family` 必须在任何前向之前调用")
        g = torch.sigmoid((wg @ cf.T).T)               # (N, d_v)
        m = (self.Wo @ (g * y).T).T
        if wc is not None:
            m = m + (wc @ cf.T).T
    elif _FAM == "mlp1":
        pm = _PM.get(id(self))
        if pm is None:
            raise RuntimeError(f"mlp1 族参数缺失（layer id={id(self)}）")
        w1, w2, b1 = pm
        z = torch.cat([y, cf], dim=-1)                 # (N, d_v + d_model)
        h = torch.nn.functional.gelu((w1 @ z.T).T + b1)   # (N, h_dim)
        m = (w2 @ h.T).T
        if wc is not None:
            m = m + (wc @ cf.T).T
    else:
        raise AssertionError(_FAM)
    if cap:
        Bn, Ln = hs_in.shape[0], hs_in.shape[1]
        idx = torch.arange(Bn, device=cf.device) * Ln + (Ln - 2)
        # ★ 保留整段（不只第一个 batch），供 A103-4 逐位比对
        global _C_OUT
        _C_OUT = cf[idx].detach().clone()
    self.injection = m.reshape(hs_in.shape)
    return self.injection


def install():
    LayerMem.write = D100._write_tf
    LayerMem.read = _read_r105


def restore():
    LayerMem.write = _ORIG_WRITE
    LayerMem.read = _ORIG_READ


def set_family(fam, st, make_wc=True):
    """设置族并（按需）用**本地 Generator** 建新参数；返回逐层参数快照。

    ★ 初始化规则（卡片 §3.3 冻结）：`torch.Generator(device="cpu").manual_seed(7919+li+K)`，
      `K=100 -> Wg`、`K=200 -> W1`、`K=300 -> W2`；`b1` 由 `K=300` 的同一 generator 续取。
      **严禁触碰全局 RNG**，否则 `A105-1` 无法逐位复现 `D-133`。
    """
    global _FAM, _PG, _PM
    assert fam in ("lin", "gate", "mlp1"), fam
    _FAM, _PG, _PM = fam, {}, {}
    out = {}
    for li, m in st.mems.items():
        dv, dm = m.d_v, m.d_model
        if fam == "gate":
            g100 = torch.Generator(device="cpu").manual_seed(7919 + li + 100)
            m._Wgp = (torch.randn(dv, dm, generator=g100)
                      / (dm ** 0.5)).to(m.Wo.device).requires_grad_(True)
            _PG[id(m)] = m._Wgp
            # ★ DIR-108 rev1.0 §7 唯一新增（测量层）：初始化快照，供 §3.1 的 ρ 作分母。
            #   只 clone、不改数值；`detach()` ⇒ 不进计算图、不影响任何梯度。
            m._Wg0p = m._Wgp.detach().clone()
            out[li] = m._Wgp
        elif fam == "mlp1":
            g200 = torch.Generator(device="cpu").manual_seed(7919 + li + 200)
            m._W1p = (torch.randn(dm, dv + dm, generator=g200)
                      / ((dv + dm) ** 0.5)).to(m.Wo.device).requires_grad_(True)
            m._B1p = torch.zeros(dm, device=m.Wo.device).requires_grad_(True)
            g300 = torch.Generator(device="cpu").manual_seed(7919 + li + 300)
            m._W2p = (torch.randn(dm, dm, generator=g300)
                      / (dm ** 0.5)).to(m.Wo.device).requires_grad_(True)
            _PM[id(m)] = (m._W1p, m._W2p, m._B1p)
            out[li] = [m._W1p, m._W2p, m._B1p]
        if make_wc:
            g = torch.Generator(device="cpu").manual_seed(7919 + li)
            m._WCp = (torch.randn(dm, dm, generator=g)
                      / (dm ** 0.5)).to(m.Wo.device).requires_grad_(True)
            _WC[id(m)] = m._WCp
    return out


def _assert_family_ready(st, fam):
    """`A105-13`：族参数必须对**每一层**都已就绪，否则读侧会静默走错分支。

    这条守卫是 `DEF-105-01` 的直接产物：原实现让 `_FAM` 跨格泄漏、
    且 `set_family` 晚于 `capture_keys`，于是 `_PG[id(self)]` 抛 `KeyError`。
    """
    miss = []
    for li, m in st.mems.items():
        if fam == "gate" and id(m) not in _PG:
            miss.append(li)
        if fam == "mlp1" and id(m) not in _PM:
            miss.append(li)
    assert not miss, f"{fam} 族参数未就绪的层 = {miss}（`DEF-105-01`）"


def fam_params(st):
    """本族**待训**的读出侧新参数（不含 `Wq/Wo/gamma`，那些由 train 里加）。"""
    out = []
    for m in st.mems.values():
        if _FAM == "gate":
            out.append(m._Wgp)
        elif _FAM == "mlp1":
            out += [m._W1p, m._W2p, m._B1p]
    return out


def clear_wc(st):
    global _WC, _PG, _PM, _FAM
    _WC, _PG, _PM = {}, {}, {}
    _FAM = "lin"                       # ★ 复位：否则下一格在 `capture_keys` 阶段就会走错族
    for m in st.mems.values():
        m._WCp = None
        m._Wg0p = None


def set_capture(on):
    global _C_CAP, _C_OUT
    _C_CAP = bool(on)
    if not on:
        _C_OUT = None


def set_gamma(st, val):
    for m in st.mems.values():
        m.gamma.data.fill_(float(val))


# ==================================================================== 评测
@torch.no_grad()
def eval_acc(model, st, tok, facts, keys, seed=555):
    """4 选 1、按题计（同题 `N_DRAWS` 次全对才算命中）；`no_write`；返回 (acc, n_q)。
    = `dir101/R1_task101.eval_acc` 逐字。"""
    rng = random.Random(seed)
    items = R91.make_eval_items(tok, facts, keys, rng)
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    per = {}
    try:
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


# ==================================================================== 训练（唯一改动：可含 Wc）
def train_read_v105(model, st, tok, facts, train_keys, rng, steps, lr=R91.READ_LR):
    """= `dir103.train_read_v2` 逐字；**唯一改动**：待训参数按 `_FAM` 组装、`steps` 由外部给。
    `W` 冻结（`no_write=True`）；`W0/U/V/Wfull` 不进阶（`A105-2`）。
    ★ `Wo`/`Wc` 在**三族**里都参与训练（`D-133` 的 `cy` 臂就训 `Wo`+`Wc`）⇒ `lin` 与旧臂逐位等价。"""
    for m in st.mems.values():
        m.no_write = True
    ps = []
    for li, m in st.mems.items():
        ps += [m.Wq, m.gamma, m.Wo]
        if getattr(m, "_WCp", None) is not None:
            ps.append(m._WCp)
    ps += fam_params(st)
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


# ==================================================================== 手算算例（A103-9）
def toy_example():
    """卡片 §2.4 (a)-(e)：**写码前逐条手算**（`CORR-24`）。四条数值 + 一卷算术。

    (a) lin  : W=diag(1,2)、Wq=Wo=Wc=I、c=(3,4) ⇒ y=(3,8) ⇒ m=(6,12)
    (b) gate : Wg=I、Wo=I、Wc=0 ⇒ m=(sigma(3)*3, sigma(4)*8)=(2.8577223805, 7.8561103203)
    (c) gate 退化恒等式：Wg=0 ⇒ g=sigma(0)=0.5 ⇒ m = 0.5*(Wo y) = (1.5, 4.0)
    (d) mlp1 : c=(0.3,0.4) ⇒ y=(0.3,0.8)；W1=[I|0]、b1=0、W2=I ⇒ m=GELU(y)=(0.1853734267, 0.6305156811)
    (e) acc  : per={0:[T,T,T],1:[T,F,T],2:[T,T,T]} ⇒ hit=2, nq=3, acc=2/3
    """
    # ★ 用 float64 做这组手算：float32 的 sigmoid/GELU 有 ~1e-8 舍入，
    #   若用 float32 就得把容差放宽到 1e-7 —— 那正是 `DEF-84-1` 的错法（拿容差盖住精度）。
    D_ = torch.float64
    W = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=D_)
    eye = torch.eye(2, dtype=D_)
    c3 = torch.tensor([[3.0, 4.0]], dtype=D_)

    # (a) lin
    y3 = (W @ ((eye @ c3.T).T).T).T                       # q = I c = (3,4); y = W q
    m_lin = (eye @ y3.T).T
    d_a = float((m_lin.squeeze(0) - torch.tensor([3.0, 8.0], dtype=D_)).abs().max())
    m_lin_cy = m_lin + (eye @ c3.T).T
    d_a2 = float((m_lin_cy.squeeze(0) - torch.tensor([6.0, 12.0], dtype=D_)).abs().max())

    # (b) gate with Wg=I
    g = torch.sigmoid((eye @ c3.T).T)
    m_g = (eye @ (g * y3).T).T
    want_g = torch.tensor([0.9525741268 * 3.0, 0.9820137900 * 8.0], dtype=D_)
    d_b = float((m_g.squeeze(0) - want_g).abs().max())

    # (c) gate degeneracy: Wg=0 ⇒ g=0.5 ⇒ m = 0.5*(Wo y)
    g0 = torch.sigmoid((torch.zeros(2, 2, dtype=D_) @ c3.T).T)
    m_g0 = (eye @ (g0 * y3).T).T
    d_c = float((m_g0 - 0.5 * m_lin).abs().max())

    # (d) mlp1: c=(0.3,0.4) ⇒ y = W c = (0.3, 0.8); W1=[I|0] ⇒ z 取 y；m = GELU(y)
    c2 = torch.tensor([[0.3, 0.4]], dtype=D_)
    y2 = (W @ c2.T).T
    W1 = torch.cat([torch.eye(2, dtype=D_), torch.zeros(2, 2, dtype=D_)], dim=1)
    h = torch.nn.functional.gelu((W1 @ torch.cat([y2, c2], dim=-1).T).T)
    d_d = float((h.squeeze(0) - torch.tensor([0.1853734267, 0.6305156811], dtype=D_)).abs().max())

    # (e) acc 算术（N_DRAWS=3、按题计）
    def acc_of(per):
        nq_ = len(per)
        return sum(1 for v in per.values() if all(v)) / nq_, nq_
    a1, n1 = acc_of({0: [True, True, True], 1: [True, False, True], 2: [True, True, True]})
    a2, _ = acc_of({0: [True, True, True]})
    d_e = max(abs(a1 - 2 / 3), abs(a2 - 1.0))
    thr_ok = (31 / 32) < THR_ROUTE                       # ★ 阈值不可被 31/32 满足

    RES["toy"] = {"lin_y": d_a, "lin_cy": d_a2, "gate": d_b,
                  "gate_deg": d_c, "mlp1": d_d, "acc_delta": d_e}
    # ★ 两档容差，各有依据（`DEF-84-1` 的教训：容差必须匹配「被比较对象的精度」）：
    #   tol_ex   = 1e-12：闭式算术（(a)(c)(e)），无超验常数 ⇒ 只受 float64 舍入限制
    #   tol_const= 1e-9 ：含**写到 10 位小数**的超验常数（(b) sigmoid、(d) GELU）
    #              ⇒ 单项舍入 ≤5e-11，乘 3/8 后 ≤~3e-10 ⇒ 1e-9 是**匹配常量精度**，不是放宽阈值
    tol_ex, tol_const = 1e-12, 1e-9
    ok = (d_a < tol_ex and d_a2 < tol_ex and d_b < tol_const and d_c < tol_ex
          and d_d < tol_const and d_e < tol_ex and thr_ok)
    return rec("A105-9", ok,
               f"手算 (a)lin_y={d_a:.0e} (a)lin_cy={d_a2:.0e} (b)gate={d_b:.1e} "
               f"(c)gate退化={d_c:.0e} (d)mlp1={d_d:.1e} (e)acc差={d_e:.0e}；"
               f"31/32={31/32:.5f}<{THR_ROUTE} 成立={thr_ok}（七条须全成立）",
               RES["toy"])


# ==================================================================== 单格
def run_cell(model, tok, facts, seed, fam, steps, stage, chunks, owner):
    """`fam ∈ {lin,gate,mlp1}`、`steps ∈ {600,3000}`。★ 一格 = **一次全新栈**（`DEF-95-02`）。"""
    t0 = time.time()
    _base_before = model_hash(model)     # ★ DIR-108 `A108-4`：基座 sha1 前
    st = R91.make_stack(gamma=0.0, no_write=False, seed=seed)
    res = {"seed": seed, "family": fam, "steps": steps, "stage": stage}
    try:
        # ★★ 必须**先**建族参数再跑任何前向：`capture_keys`/`write_phase_arm` 都会调用
        #    `LayerMem.read`，若此时 `_FAM` 还停在上一格的族且 `_PG/_PM` 为空 ⇒ `KeyError`
        #    （`DEF-105-01`，Cycle 196 实测踩到）。
        set_family(fam, st, make_wc=True)
        _assert_family_ready(st, fam)
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

        # `A105-8`：white 键两两 |cos| 最大
        mc = 0.0
        for li in Kraw:
            Kt = TF[li](Kraw[li])
            Kn = Kt / Kt.norm(dim=1, keepdim=True).clamp_min(1e-12)
            G = Kn @ Kn.T
            G.fill_diagonal_(0.0)
            mc = max(mc, float(G.abs().max()))
        res["maxcos"] = mc

        ls, viol, div = train_read_v105(model, st, tok, facts, allk, rng, steps)
        acc, nq = eval_acc(model, st, tok, facts, allk)

        # `A105-6`：新参数**确实参与**（梯度非零且有限）
        grad = {}
        for m in st.mems.values():
            if fam == "gate":
                gp = m._Wgp.grad
                grad[f"Wg{id(m)}"] = None if gp is None else float(gp.abs().max())
            elif fam == "mlp1":
                for nm, pp in (("W1", m._W1p), ("W2", m._W2p), ("b1", m._B1p)):
                    gg = pp.grad
                    grad[f"{nm}{id(m)}"] = None if gg is None else float(gg.abs().max())
        gmax = max([v for v in grad.values() if v is not None], default=None)
        gany_none = any(v is None for v in grad.values())
        # ★ DIR-108 rev1.1 `M108-2`（report-only，非判据）：W_g 梯度的**分布**
        #   因为 Adam 的 eps=1e-8 是**逐坐标**地板（卡片 §2.2'），只看 max 会高估位移能力。
        gdist = None
        if fam == "gate":
            ga = torch.cat([m._Wgp.grad.detach().abs().reshape(-1).float()
                            for m in st.mems.values()
                            if m._Wgp.grad is not None])
            if ga.numel():
                q = torch.tensor([0.50, 0.90, 0.99], device=ga.device)
                qs = torch.quantile(ga, q).tolist()
                gdist = {"p50": qs[0], "p90": qs[1], "p99": qs[2],
                         "max": float(ga.max()),
                         "frac_gt_eps": float((ga > 1e-8).float().mean()),
                         "n": int(ga.numel())}
        res.update({"acc_all32": acc, "nq": nq, "loss_tail": ls[-20:],
                    "viol": viol, "div": div, "train_steps": len(ls),
                    "grad_max": gmax, "grad_any_none": gany_none,
                    "grad_dist": gdist,
                    "gamma_final": {li: [float(x) for x in m.gamma.detach().reshape(-1)]
                                    for li, m in st.mems.items()}})

        # ★ DIR-108 rev1.0 §3.1：`W_g` 的**相对位移** ρ = ||ΔW_g||_F / ||W_g^init||_F
        #   三层拼接后取 F 范数。纯读取、纯算术 ⇒ 不扰动训练路径（`A108-2` 会逐位验）。
        rho = None
        if fam == "gate":
            g0 = [m._Wg0p for m in st.mems.values()]
            g1 = [m._Wgp.detach() for m in st.mems.values()]
            assert all(t is not None for t in g0), "`_Wg0p` 缺失（DIR-108 测量层未生效）"
            flat0 = torch.cat([t.reshape(-1).float() for t in g0])
            flat1 = torch.cat([t.reshape(-1).float() for t in g1])
            n0 = float(flat0.norm()); nd = float((flat1 - flat0).norm())
            rho = nd / max(n0, 1e-30)
            res.update({"rho": rho, "Wg_norm_init": n0, "Wg_norm_delta": nd,
                        "Wg_init_sha1": _sha1_of(flat0),
                        "Wg_final_sha1": _sha1_of(flat1),
                        "Wg_finite": bool(torch.isfinite(flat1).all())})
        else:
            res["rho"] = None

        # `A105-2`：读训练不得改 W
        dW = max(float((R91.W_of(m).detach() - W_write[li]).norm()
                       / W_write[li].norm().clamp_min(1e-30))
                 for li, m in st.mems.items())
        res["W_drift"] = dW

        # `M105-2`：`W <- W0`（抹掉 600 块写入）、其余参数不动，重测 ⇒ 非记忆化对照
        for li, m in st.mems.items():
            if m.use_factorized:
                m.U.data.zero_()
                m.V.data.zero_()
            else:
                m.Wfull.data.copy_(m.W0.data)
        dW0 = max(float((R91.W_of(m).detach() - m.W0.detach()).norm()
                        / m.W0.detach().norm().clamp_min(1e-30))
                  for m in st.mems.values())
        acc0, _ = eval_acc(model, st, tok, facts, allk)
        res.update({"acc_W0": acc0, "W0_rel": dW0})
    finally:
        clear_wc(st)
        set_capture(False)
        restore()
        st.remove()
        torch.cuda.empty_cache()
    res["base_sha1_before"] = _base_before
    res["base_sha1_after"] = model_hash(model)   # `A108-4`：须与 before 相同
    res["secs"] = round(time.time() - t0, 1)
    return res


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stageA",
                    choices=["stageA", "smoke", "formal", "merge", "adjudicate",
                             "premise", "synth", "cell", "verdict"])
    ap.add_argument("--families", default="", help="逗号分隔，覆盖 FAMILIES")
    ap.add_argument("--steps", default="", help="逗号分隔，覆盖 STEPS_GRID")
    ap.add_argument("--seeds", default="", help="逗号分隔，覆盖 SEEDS")
    ap.add_argument("--from", dest="from_json", default="",
                    help="adjudicate：从既有 stage_*.json 重算判决（不重跑格子）")
    ap.add_argument("--force", action="store_true",
                    help="cell：覆盖已存在的格（默认跳过 = 断点续跑）")
    ap.add_argument("--cells-dir", default="out/cells",
                    help="merge：一格一进程的逐格 JSON 目录（`DEF-105-02`）")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    # ★★ DIR-108 新增阶段（测量/审计层；不进入任何训练路径）
    if a.stage == "premise":
        premise_main(a); return
    if a.stage == "synth":
        synth_main(a); return
    if a.stage == "cell":
        cell_main(a); return
    if a.stage == "verdict":
        verdict108_main(a); return

    # ★ merge：一格一进程（`run_formal105.sh`）跑完后合成再裁决（`DEF-105-02`）
    if a.stage == "merge":
        cd = pathlib.Path(a.cells_dir)
        files = sorted(cd.glob("*.json"))
        cells = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        # 按 (family, steps, seed) 固定排序，使判决与文件系统顺序无关
        cells.sort(key=lambda c: (c["family"], c["steps"], c["seed"]))
        RES["cells"] = cells
        print(f"[merge] 载入 {len(cells)} 格（{cd}）", flush=True)
        for c in cells:
            print(f"  {c['family']:<5}{c['steps']:>5} {c['seed']:>8} "
                  f"acc={c['acc_all32']:.4f} acc_W0={c['acc_W0']:.4f}", flush=True)
        RES["stage"] = "merge"
        (OUT / "stage_formal.json").write_text(
            json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
        verdict_main(None)
        return

    # ★ adjudicate：只重算判决（判决必须能离线重算）
    if a.stage == "adjudicate":
        assert a.from_json, "adjudicate 需要 --from <stage_*.json>"
        src = json.loads(pathlib.Path(a.from_json).read_text(encoding="utf-8"))
        RES["cells"] = src["cells"]
        RES["guards"] = src.get("guards", {})
        RES["toy"] = src.get("toy", {})
        RES["stage"] = "adjudicate"
        print(f"[adjudicate] 载入 {a.from_json}：{len(RES['cells'])} 格", flush=True)
        verdict_main(None)
        return

    tok, model = KP.model_and_tok()
    model.eval()
    facts = R91.build_facts(tok)
    chunks, owner = R91.build_chunks(tok, facts)
    fams = [x for x in a.families.split(",") if x] or FAMILIES
    grids = [int(x) for x in a.steps.split(",") if x] or STEPS_GRID
    seeds = [int(x) for x in a.seeds.split(",") if x] or (
        SEEDS if a.stage == "formal" else SEEDS[:1])
    print(f"族 {fams} × 步数 {grids} × 种子 {seeds}", flush=True)

    # ---------------- Stage A（手算守卫；不需训练）
    if a.stage == "stageA":
        toy_example()
        # `A105-6` 的前置：三族都能建出参数且形状正确（不训练）
        seed0 = seeds[0]
        st = R91.make_stack(gamma=0.0, no_write=False, seed=seed0)
        try:
            install()
            shapes, dgen = {}, 0.0
            for fam in FAMILIES:
                set_family(fam, st, make_wc=True)
                if fam == "gate":
                    bad = [li for li, m in st.mems.items() if tuple(m._Wgp.shape) != (m.d_v, m.d_model)]
                    shapes[fam] = bad
                elif fam == "mlp1":
                    bad = [li for li, m in st.mems.items()
                           if tuple(m._W1p.shape) != (m.d_model, m.d_v + m.d_model)
                           or tuple(m._W2p.shape) != (m.d_model, m.d_model)
                           or tuple(m._B1p.shape) != (m.d_model,)]
                    shapes[fam] = bad
                else:
                    shapes[fam] = []
            ok_shape = all(len(v) == 0 for v in shapes.values())
            rec("A105-11", ok_shape,
                f"三族参数形状：gate/mlp1 逐层符合（坏层：{shapes}）", shapes)
            # `A105-12`：本地 Generator ⇒ 新建参数**不扰动**全局 RNG
            t0_ = torch.randn(3, generator=torch.Generator().manual_seed(12345))
            torch.manual_seed(999)
            _ = torch.randn(4)
            ref = torch.randn(3)
            torch.manual_seed(999)
            _ = torch.randn(4)
            set_family("mlp1", st, make_wc=True)     # 内部只用本地 Generator
            got = torch.randn(3)
            dgen = float((got - ref).abs().max())
            rec("A105-12", dgen == 0.0,
                f"建参数后全局 RNG 流未变：‖got−ref‖∞ = {dgen:.1e}（须 =0）⇒ `A105-1` 可逐位复现", dgen)
        finally:
            clear_wc(st)
            restore()
            st.remove()
            torch.cuda.empty_cache()
        RES["stage"] = "stageA"
        (OUT / "stage_stageA.json").write_text(
            json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n[Stage A] 手算/形状/RNG 守卫完成；其余守卫需读数 ⇒ 见 smoke/formal", flush=True)
        return

    # ---------------- smoke / formal：跑格
    cells = [c for c in RES.get("cells", [])]
    for seed in seeds:
        for fam in fams:
            for stp in grids:
                print(f"\n{'='*88}\n[{seed} fam={fam} steps={stp}]\n{'='*88}", flush=True)
                c = run_cell(model, tok, facts, seed, fam, stp, a.stage, chunks, owner)
                print(f"[{seed} fam={fam} steps={stp}] acc={c['acc_all32']:.4f} "
                      f"acc_W0={c['acc_W0']:.4f} W_drift={c['W_drift']:.2e} "
                      f"grad_max={c['grad_max']} {c['secs']}s", flush=True)
                cells.append(c)
                RES["cells"] = cells
                (OUT / f"stage_{a.stage}.json").write_text(
                    json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
    RES["cells"] = cells
    (OUT / f"stage_{a.stage}.json").write_text(
        json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")

    if a.stage == "formal":
        verdict_main(model)


def verdict_main(model):
    """判决（卡片 §2.1 + §6）。**阈值一字不改**。"""
    C = RES["cells"]
    key = lambda c: (c["family"], c["steps"], c["seed"])
    by = {}
    for c in C:
        by.setdefault((c["family"], c["steps"]), {})[c["seed"]] = c
    seeds = sorted({c["seed"] for c in C})
    combos = sorted(by)

    # `A105-10`：格计划 12 格、六个组合各 2 seed、无删失
    n_expect = len(FAMILIES) * len(STEPS_GRID) * len(SEEDS)
    complete = (len(C) == n_expect
                and all(len(by[k]) == len(SEEDS) for k in combos)
                and not any(c.get("div") for c in C))
    rec("A105-10", complete,
        f"格数 {len(C)}/{n_expect}；组合 {[(f, s, sorted(by[(f, s)])) for f, s in combos]}；"
        f"diverged={any(c.get('div') for c in C)}")

    # `A105-1`：`lin@600` 逐位复现 D-133 的 `cy`（按题计 = 31/32, 28/32）
    rep = {}
    ok1 = True
    for sd, want_hit in REPRO_LIN600.items():
        c = by.get(("lin", 600), {}).get(sd)
        got_hit = None if c is None else round(c["acc_all32"] * c["nq"])
        rep[sd] = (got_hit, want_hit)
        ok1 = ok1 and (got_hit == want_hit)
    rec("A105-1", ok1,
        f"`lin@600` 复现 `D-133` 的 `cy`：实测/期望命中 {rep}（须逐位相等）", rep)

    # `A105-8`
    mc = max((c.get("maxcos", 9.9) for c in C), default=9.9)
    rec("A105-8", mc < 1e-4, f"white 键两两 |cos| 最大 = {mc:.2e}（须 <1e-4）")

    # `A105-2` / `A105-3`
    dw = max((c.get("W_drift", 9.9) for c in C), default=9.9)
    vi = sum(c.get("viol", 0) for c in C)
    rec("A105-2", dw == 0.0, f"读训练期 W 漂移最大 = {dw:.3e}（须逐位 =0）")
    rec("A105-3", vi == 0, f"写侧梯度违规计数合计 = {vi}（须 0）")

    # `A105-5`
    nqs = sorted({c.get("nq") for c in C})
    rec("A105-5", nqs == [NQ_EXPECT], f"评测问题数 nq = {nqs}（须全 ={NQ_EXPECT}）")

    # `A105-4`
    div = any(c.get("div") for c in C)
    rec("A105-4", not div, f"diverged = {div}（须 False）")

    # `A105-7`：`--steps` 已接线（600 与 3000 的实际步数不同且等于请求值）
    stp_ok = all(c.get("train_steps") == c["steps"] for c in C)
    rec("A105-7", stp_ok,
        f"每格 train_steps == 请求 steps：{sorted({(c['steps'], c.get('train_steps')) for c in C})}")

    # `A105-6`：非线性族的新参数梯度非零且有限（`lin` 族无新参数，跳过）
    nl_cells = [c for c in C if c["family"] != "lin"]
    g_ok = all((c.get("grad_max") is not None and c["grad_max"] > 0.0
                and not c.get("grad_any_none")) for c in nl_cells) if nl_cells else True
    rec("A105-6", g_ok,
        f"非线性族新参数 |grad|max：{[(c['family'], c['steps'], c['seed'], c.get('grad_max')) for c in nl_cells]}"
        f"（须全 >0 且无一为 None）")

    # ---------------- 判据
    guards_ok = all(v["ok"] for v in RES["guards"].values())
    rows = {}
    for (fam, stp) in combos:
        d = by[(fam, stp)]
        a32 = [d[s]["acc_all32"] for s in seeds if s in d]
        a0 = [d[s]["acc_W0"] for s in seeds if s in d]
        rows[f"{fam}@{stp}"] = {
            "acc_all32": a32, "acc_W0": a0,
            "m1_pass": all(x >= THR_ROUTE for x in a32),
            "m2_pass": all(x <= THR_NOTMEM for x in a0),
            "hit": all(x >= THR_ROUTE and y <= THR_NOTMEM
                       for x, y in zip(a32, a0)),
        }
    RES["rows"] = rows

    if not guards_ok:
        RES["verdict"] = "C105-BLOCKED"
        print("[判决] C105-BLOCKED（Stage A 守卫未全过）", flush=True)
        _dump_verdict(); return
    if len(combos) < len(FAMILIES) * len(STEPS_GRID):
        RES["verdict"] = "C105-INCOMPLETE"
        print(f"[判决] C105-INCOMPLETE（只有组合 {combos}）", flush=True)
        _dump_verdict(); return

    nl_hit = [k for k in rows if not k.startswith("lin@") and rows[k]["hit"]]
    nl_reach_only = [k for k in rows if not k.startswith("lin@")
                     and rows[k]["m1_pass"] and not rows[k]["m2_pass"]]
    lin_hit = [k for k in rows if k.startswith("lin@") and rows[k]["hit"]]

    if nl_hit:
        RES["verdict"] = "C105-NL-REACHED"
    elif nl_reach_only:
        RES["verdict"] = "C105-MEMORIZED"
    elif lin_hit:
        RES["verdict"] = "C105-BUDGET-ONLY"
    else:
        RES["verdict"] = "C105-NL-NOT-ENOUGH"

    RES["readings"] = {
        "rows": rows, "nl_hit": nl_hit, "nl_reach_only": nl_reach_only,
        "lin_hit": lin_hit,
        "d_nl600": (rows.get("gate@600", {}).get("acc_all32", [None])[0] or 0)
                   - (rows.get("lin@600", {}).get("acc_all32", [None])[0] or 0),
        "d_steps": (rows.get("lin@3000", {}).get("acc_all32", [None])[0] or 0)
                   - (rows.get("lin@600", {}).get("acc_all32", [None])[0] or 0),
    }
    print(f"[判决] {RES['verdict']} · rows={json.dumps(rows, ensure_ascii=False)}", flush=True)
    _dump_verdict()


def _dump_verdict():
    p = OUT / ("stage_adjudicate.json" if RES.get("stage") == "adjudicate"
               else "stage_verdict.json")
    p.write_text(json.dumps(RES, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] → {p}", flush=True)



# ==================================================================== DIR-108 新增：测量/审计层
# ★ 本段**只读**既有读数、或只做与训练无关的合成检验；不触碰任何训练路径。
RHO_THR = 0.01                     # ★ 卡片 §3.2 冻结阈值：ρ ≥ 0.01 ⇒ 「移动了」
BASE_SHA1_EXPECT = "26f966c3b98ccc30"   # `A108-4`
GRAD_BASE_3000_7777 = 8.699483089458226e-08   # dir105/out/cells/gate_3000_7777.json
ACC_BASE_3000_7777 = 0.90625
RHO_HAND = [(("diag(1,1)", [1.0, 1.0]), [0.5, 0.0], 0.353553),
            (("diag(.03,.03)", [0.03, 0.03]), [0.001, 0.0], 0.023570)]


def _sha1_of(t):
    """张量的 sha1（float32、CPU、连续）—— 用于 `A108-1` 的初始化可重建性。"""
    a = t.detach().float().cpu().contiguous().numpy()
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


def model_hash(model):
    """基座参数 + buffer 的 sha1（逐张量，按名字排序）。
    ★ 逐字复制自 `dir105/diag105_state.py` —— 定义必须同源，否则对不上 `26f966c3b98ccc30`。"""
    h = hashlib.sha1()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    for name, b in sorted(model.named_buffers()):
        h.update(name.encode())
        h.update(b.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def check_hand_rho():
    """`A108-7`：卡片 §4.1 的手算 ρ **逐位复算**。
    ★ 手算值取自卡片 §4.1（冻结前已写死），不是这里临时算的。"""
    out = {}
    for (name, w0), dw, want in RHO_HAND:
        n0 = sum(x * x for x in w0) ** 0.5
        nd = sum(x * x for x in dw) ** 0.5
        got = nd / n0
        ok = abs(got - want) < 1e-6
        out[name] = {"n0": n0, "nd": nd, "rho": got, "hand": want, "ok": ok}
        print(f"  [A108-7] {name:16s} rho={got:.6f} 手算={want:.6f} {'✓' if ok else '✗'}", flush=True)
    return out, all(v["ok"] for v in out.values())


def synth_adam():
    """`A108-3`（rev 1.1 修正版，`DEF-108-01`）：Adam 步长的**两个 regime**。

    ★ rev 1.0 断言「都 = -lr」是**错的**（漏了 eps）：正确式 step = lr*g/(|g|+eps)。
      平台区 |g| >> eps ⇒ ≈ lr；线性区 |g| ~ eps ⇒ ≈ lr*g/eps。
    本函数断言**两个 regime 各自**符合正确公式，而**不是**「尺度不变」。"""
    lr, eps = 1e-3, 1e-8

    def one(g):
        # ★ rev 1.2：显式 float64。本检验只验**公式**，与模型 dtype 无关；
        #   float32 下相对误差 ~1e-7（= float32 eps）会把公式的验证淹没在舍入里（DEF-108-02）。
        th = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([th], lr=lr, eps=eps)
        opt.zero_grad(set_to_none=True)
        th.grad = torch.tensor([float(g)], dtype=torch.float64)
        opt.step()
        return abs(float(th.item()))

    GS = (1e-9, 8.699e-08, 1e-2, 1e2, 1e4)     # ★ 同一份 g 列表派生 res/pred，避免键名不一致
    res = {f"g={g:g}": one(g) for g in GS}
    pred = {f"g={g:g}": lr * g / (g + eps) for g in GS}
    relerr = {k: abs(res[k] - pred[k]) / pred[k] for k in res}

    # ① 平台区：g=1e-2 与 g=1e+2 尺度不变（相对差 < 1e-6），且都 ≈ lr
    plateau = [res[f"g={1e2:g}"], res[f"g={1e4:g}"]]
    # ★★ rev 1.3：不再断言"平台区恰好 = lr"（那要 |g| >> eps，而 1e-2 只有 1e6*eps，
    #    其真实偏差恰为 eps/g = 1e-6 ⇒ rev1.2 的 1e-12 阈值**在数学上不可满足**，DEF-108-03）。
    #    改为直接断言**ε 律本身**：|step/lr - 1| <= (eps/|g|)*1.01 + 1e-12，对每个 g 都成立。
    ratio = {k: res[k] / lr for k in res}
    bound = {f"g={g:g}": (eps / g) * 1.01 + 1e-12 for g in GS}
    ok_law = all(abs(ratio[f"g={g:g}"] - 1.0) <= bound[f"g={g:g}"] for g in GS)
    # 平台区（真正的平台：g=1e4 ⇒ eps/g=1e-12）步长与 lr 相对差 < 1e-10
    ok_plateau = abs(ratio[f"g={1e4:g}"] - 1.0) < 1e-10
    # 线性区：g=1e-9 严格按 lr*g/(g+eps) 缩水
    ok_linear = relerr[f"g={1e-9:g}"] < 1e-12
    # 公式本身：所有 g 点与 lr*g/(g+eps) 的相对误差 < 1e-12
    ok_formula = all(v < 1e-12 for v in relerr.values())

    for k in res:
        print(f"  [A108-3] {k:14s} 实测 step={res[k]:.6e}  公式={pred[k]:.6e}  "
              f"相对误差={relerr[k]:.2e}", flush=True)
    print(f"  [A108-3] eps律={ok_law}  平台区={ok_plateau}  线性区缩水={ok_linear}  "
          f"公式全中={ok_formula}", flush=True)
    for g in GS:
        k = f"g={g:g}"
        print(f"  [A108-3]   {k:14s} step/lr={ratio[k]:.12f}  "
              f"上界(eps/g*1.01)={bound[k]:.3e}", flush=True)
    return {"step": res, "pred": pred, "relerr": relerr, "ratio": ratio,
            "ok_law": ok_law, "ok_plateau": ok_plateau, "ok_linear": ok_linear,
            "ok_formula": ok_formula,
            "ok": bool(ok_law and ok_plateau and ok_linear and ok_formula)}


def premise_main(a):
    """§2 前提审计：**纯算术，读既有 `dir105/out/cells/*.json`**。
    ★ 卡片 §2 已声明：这些数字在冻结前就见过 ⇒ **不作判据**，只留档。"""
    import math
    src = pathlib.Path(a.cells_dir)
    files = sorted(src.glob("*.json"))
    assert files, f"没有读数：{src}"
    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("grad_max") is None:
            continue
        lt = d.get("loss_tail") or [None]
        rows.append({"cell": f.stem, "family": d["family"], "steps": d["steps"],
                     "seed": d["seed"], "acc_all32": d["acc_all32"],
                     "grad_max": d["grad_max"], "loss": lt[-1],
                     "grad_over_loss": d["grad_max"] / lt[-1]})
    print(f"[premise] 非线性格 {len(rows)} 个（来自 {src}）", flush=True)
    print(f"  {'cell':22s}{'acc':>8s}{'grad_max':>12s}{'loss':>12s}{'g/loss':>9s}", flush=True)
    for r in rows:
        print(f"  {r['cell']:22s}{r['acc_all32']:>8.4f}{r['grad_max']:>12.3e}"
              f"{r['loss']:>12.3e}{r['grad_over_loss']:>9.3f}", flush=True)

    conv = [r for r in rows if r["loss"] < 2e-05]
    band = [r["grad_over_loss"] for r in conv]
    xs = [math.log10(r["loss"]) for r in rows]; ys = [math.log10(r["grad_max"]) for r in rows]
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx; r2 = sxy * sxy / (sxx * syy)

    pairs = []
    for stp in (600, 3000):
        for sd in (20261028, 7777):
            g = [r for r in rows if r["family"] == "gate" and r["steps"] == stp and r["seed"] == sd]
            m = [r for r in rows if r["family"] == "mlp1" and r["steps"] == stp and r["seed"] == sd]
            if g and m:
                pairs.append({"steps": stp, "seed": sd,
                              "gate": g[0]["grad_max"], "mlp1": m[0]["grad_max"],
                              "mlp1_over_gate": m[0]["grad_max"] / g[0]["grad_max"],
                              "mlp1_acc": m[0]["acc_all32"], "mlp1_loss": m[0]["loss"]})
    gate_ge = sum(1 for p in pairs if p["mlp1_over_gate"] <= 1.0)
    out = {"rows": rows, "converged_band": {"min": min(band), "max": max(band), "n": len(band)},
           "regression": {"slope": slope, "r2": r2, "n": n},
           "pairs": pairs, "gate_ge_mlp1_pairs": gate_ge,
           "note": "★ 已见数据 ⇒ 不作判据（卡片 §2 方法论声明 / DEF-107-07）",
           "readlr_fact": float(R91.READ_LR), "argparse_lr_claimed_by_P105_1": 1e-2}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "premise.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[premise] 已收敛 {len(band)} 格的 grad/loss 窄带 = [{min(band):.3f}, {max(band):.3f}]", flush=True)
    print(f"[premise] log-log 斜率={slope:.3f} r2={r2:.4f}；4 对里 gate≥mlp1 的有 {gate_ge} 对", flush=True)
    print(f"[premise] 实测 READ_LR={R91.READ_LR}（P105-1 写 1e-2 ⇒ 笔误）", flush=True)


def synth_main(a):
    """`A108-1`/`A108-3`/`A108-7`/`A108-10` —— 全部不训练。"""
    OUT.mkdir(parents=True, exist_ok=True)
    G = {}
    print("[synth] A108-7 手算 ρ 复算", flush=True)
    hand, ok_hand = check_hand_rho(); G["A108-7"] = ok_hand

    print("[synth] A108-3 合成 Adam 尺度不变性", flush=True)
    ad = synth_adam(); G["A108-3"] = ad["ok"]

    print("[synth] A108-10 Wg 确在优化器参数表里", flush=True)
    torch.set_grad_enabled(False); torch.manual_seed(0)
    tok, model = KP.model_and_tok(); model.eval()
    import inspect
    src_txt = inspect.getsource(train_read_v105)
    lr_ok = abs(float(R91.READ_LR) - 1e-3) < 1e-12
    src_ok = ("ps += fam_params(st)" in src_txt) or ("fam_params(st)" in src_txt)
    st = R91.make_stack(gamma=0.0, no_write=False, seed=7777)
    base_h = model_hash(model)
    try:
        set_family("gate", st, make_wc=True)
        fp = fam_params(st)
        shapes = [list(t.shape) for t in fp]
        mem_ok = (len(fp) == 3 and all(t.requires_grad for t in fp)
                  and all(id(m) in _PG for m in st.mems.values()))
        print(f"  [A108-10] READ_LR={R91.READ_LR} (ok={lr_ok})  fam_params={len(fp)} 层 "
              f"shapes={shapes}  全在 _PG={mem_ok}  源码含 fam_params={src_ok}", flush=True)
        G["A108-10"] = bool(lr_ok and src_ok and mem_ok)

        print("[synth] A108-1 Wg 初始化可逐位重建", flush=True)
        h1 = _sha1_of(torch.cat([m._Wg0p.reshape(-1) for m in st.mems.values()]))
        n1 = float(torch.cat([m._Wg0p.reshape(-1) for m in st.mems.values()]).norm())
        clear_wc(st); set_family("gate", st, make_wc=True)
        h2 = _sha1_of(torch.cat([m._Wg0p.reshape(-1) for m in st.mems.values()]))
        n2 = float(torch.cat([m._Wg0p.reshape(-1) for m in st.mems.values()]).norm())
        print(f"  [A108-1] 第一次 sha1={h1} ||Wg||={n1:.6f}", flush=True)
        print(f"  [A108-1] 第二次 sha1={h2} ||Wg||={n2:.6f}  同={h1 == h2}", flush=True)
        G["A108-1"] = bool(h1 == h2 and abs(n1 - n2) < 1e-9)
    finally:
        clear_wc(st); st.remove(); torch.cuda.empty_cache()

    print(f"[synth] A108-4 基座 sha1 = {base_h}（期望 {BASE_SHA1_EXPECT}）", flush=True)
    G["A108-4"] = (base_h == BASE_SHA1_EXPECT)

    out = {"guards": G, "hand": hand, "adam": ad,
           "base_sha1": base_h, "readlr": float(R91.READ_LR)}
    (OUT / "synth.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[synth] 守卫：" + "  ".join(f"{k}={'✓' if v else '✗'}" for k, v in G.items()), flush=True)


def cell_main(a):
    """一格一进程（`DEF-105-02`）；落盘 `out/cells/<fam>_<steps>_<seed>.json`；存在即跳过。"""
    assert a.families and a.steps and a.seeds, "cell 需要 --families/--steps/--seeds 单值"
    fam = a.families.split(",")[0]; stp = int(a.steps.split(",")[0]); sd = int(a.seeds.split(",")[0])
    cd = pathlib.Path(a.cells_dir); cd.mkdir(parents=True, exist_ok=True)
    tag = f"{fam}_{stp}_{sd}"
    dst = cd / f"{tag}.json"
    if dst.exists() and not a.force:
        print(f"[cell] {tag} 已存在 ⇒ 跳过（断点续跑）", flush=True); return
    torch.set_grad_enabled(False); torch.manual_seed(0)
    tok, model = KP.model_and_tok(); model.eval()
    facts = R91.build_facts(tok); chunks, owner = R91.build_chunks(tok, facts)
    c = run_cell(model, tok, facts, sd, fam, stp, "cell108", chunks, owner)
    dst.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cell] {tag} acc={c['acc_all32']:.4f} rho={c['rho']} "
          f"grad_max={c['grad_max']} base={c['base_sha1_after']} {c['secs']}s → {dst}", flush=True)


def verdict108_main(a):
    """§5 分支 + §6 守卫。阈值一字不改（`RHO_THR = 0.01`）。"""
    cd = pathlib.Path(a.cells_dir)
    cells = {}
    for f in sorted(cd.glob("gate_*.json")):
        d = json.loads(f.read_text(encoding="utf-8")); cells[f.stem] = d
    synth = {}
    sp = OUT / "synth.json"
    if sp.exists():
        synth = json.loads(sp.read_text(encoding="utf-8"))
    G = dict(synth.get("guards", {}))

    c3000 = cells.get("gate_3000_7777"); c600 = cells.get("gate_600_7777")
    print(f"[verdict108] 载入格：{sorted(cells)}", flush=True)
    for k in ("gate_3000_7777", "gate_600_7777"):
        if k not in cells:
            G[f"missing_{k}"] = False

    rho = None
    if c3000:
        rho = c3000.get("rho")
        G["A108-2"] = (c3000.get("acc_all32") == ACC_BASE_3000_7777
                       and abs(c3000.get("grad_max", 0) - GRAD_BASE_3000_7777) / GRAD_BASE_3000_7777 < 1e-3)
        G["A108-4"] = (c3000.get("base_sha1_before") == c3000.get("base_sha1_after")
                       == BASE_SHA1_EXPECT)
        G["A108-5"] = (c3000.get("W_drift") == 0.0)
        G["A108-6"] = bool(c3000.get("Wg_finite")) and rho is not None and rho == rho
        G["A108-8"] = (c3000.get("grad_max") is not None
                       and 1e-09 < c3000["grad_max"] < 1e-05)
        G["A108-9"] = True
        if c600 and synth:
            G["A108-1"] = G.get("A108-1", False) and (c600.get("Wg_init_sha1") == c3000.get("Wg_init_sha1"))

    guards_ok = all(bool(v) for v in G.values()) if G else False
    print("\n[verdict108] §6 守卫：", flush=True)
    for k in sorted(G):
        print(f"  {k:10s} {'✓' if G[k] else '✗'}", flush=True)

    if not guards_ok or rho is None:
        branch = "C108-INCOMPLETE"
        why = "守卫失败或 ρ 未计算 ⇒ 判据从未被计算（不得记 PASS/FAIL）"
    elif rho >= RHO_THR:
        branch = "C108-PREMISE-INVALID"
        why = (f"ρ={rho:.6g} ≥ {RHO_THR} ⇒ W_g 位移达自身范数的 {rho*100:.2f}% ⇒ "
               "「几乎未被推动」为假 ⇒ P105-1 前提塌陷")
    else:
        branch = "C108-GATE-UNLEARNED"
        why = f"ρ={rho:.6g} < {RHO_THR} ⇒ W_g 基本停在初始化 ⇒ 前提成立、门控确实没学起来"

    out = {"branch": branch, "why": why, "rho": rho, "rho_thr": RHO_THR,
           "guards": G, "guards_all_ok": guards_ok,
           "rho_600": (c600 or {}).get("rho"),
           "cells": {k: {kk: v.get(kk) for kk in
                         ("acc_all32", "rho", "grad_max", "grad_dist", "W_drift",
                          "Wg_norm_init", "Wg_norm_delta", "Wg_init_sha1",
                          "base_sha1_after", "secs")}
                     for k, v in cells.items()},
           "synth": synth.get("adam")}
    (OUT / "verdict108.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*80}\n★ 判决：{branch}\n  {why}", flush=True)
    # ★ DEF-108-04：`rho` 可为 None（缺格/守卫失败）⇒ 打印必须 None-safe，
    #   否则「判据从未被计算」这条路自己会抛 TypeError，把 INCOMPLETE 变成崩溃。
    def _fmt(x):
        return "None" if x is None else f"{x:.6g}"
    if out["rho_600"] is not None or rho is not None:
        ratio = ("n/a" if (rho is None or out["rho_600"] in (None, 0.0))
                 else f"{rho / out['rho_600']:.2f}×")
        print(f"  ρ(600)={_fmt(out['rho_600'])}   ρ(3000)={_fmt(rho)}   比值={ratio}", flush=True)
    print(f"{'='*80}", flush=True)
    return out


if __name__ == "__main__":
    main()
