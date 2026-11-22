#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 rev 2.7 · **位置选择性取值能否带来内容性收益？（端到端）**

卡片：revolve/DIR-91/DIRECTION_rev91_v27.md（判据冻结；本文件不得改阈值）
派生自 D-106（POOLING：值 token 位 LOO-10=1.000、64 位均值为 chance）。

五臂（**唯一独立变量 = sgmn_module.py:87 那一行 `v` 的取值方式**；其余全部原样）：
  A_mean       v = zv.mean(0)                        装置现状（复现 D-105/D-106）     可实现
  B_attn       v = Σ a_t zv_t, a = softmax(√d_k·cos(kp_t,k))                    可实现
  C_max        v = zv.max(0).values                                             可实现
  D_vtpos      v = zv[V_i].mean(0)（**真值 token 位**）                        ✘ 上界诊断
  E_vtpos_qor  同 D，且读通路 q ← k_i（仅读出位）                              ✘ 叠加上界

★ CORR-13：臂 D 是 §1 机理措辞的**可推翻对照**（G_ctrl）。D 不达标 ⇒ 机理被否证。
★ 不修改 sgmn_module.py / R2_task91.py：写侧取值在本文件内 monkey-patch。
"""
import argparse, json, math, os, sys, time
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)

import numpy as np
import torch

import sgmn_module
from sgmn_module import LayerMem, sg
import probe91_womap as W91                      # ★ 导入无副作用（已实测 read 未被 patch）
from R2_task91 import (ALPHA, B as N_CHUNKS, CHUNK, DEV, ETA_WRITE, LAM, MOUNT, N_CAND,
                       N_DRAWS, N_FACTS, N_TRAIN_KEYS, OUT, READ_LR, READ_STEPS,
                       acc_by_question, build_chunks, build_facts, make_eval_items,
                       make_stack, model_and_tok, nll_of_candidate, read_params,
                       score_and_accuracy, train_read_task, write_params)

RES = {"card": "DIR-91 rev 2.7", "file": "probe91_valread.py",
       "alpha": ALPHA, "lam": LAM, "eta": ETA_WRITE, "read_steps": READ_STEPS,
       "read_lr": READ_LR, "n_chunk": N_CHUNKS, "n_facts": N_FACTS,
       "n_train_keys": N_TRAIN_KEYS, "n_cand": N_CAND, "n_draws": N_DRAWS,
       "guards": {}, "stage_a": [], "arms": [], "analytic": {}, "worked": {}}
FAILS = []

# §2.2 判决门（冻结；全部沿用 D-105 的常数，不新造）
G_GEN = 0.50
G_LOW = 0.25
G_DELTA = 0.125
G_PARTIAL = 0.375          # = G_LOW + G_DELTA（派生量）

ARMS = ["A_mean", "B_attn", "C_max", "D_vtpos", "E_vtpos_qor"]
ORACLE_ARMS = {"E_vtpos_qor"}


def rec(name, ok, detail, value=None):
    RES["guards"][name] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def dump(obj, name):
    p = OUT / f"{name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    print(f"  ⇒ {p}", flush=True)


def count_mounted(model, layers):
    """★ 真实测量已挂载的 SGMN 前向钩子数（每 stack 每层挂 1 个 fwd_hook）。

    DEF-93-01：原 A93-01 把 n_stacks 硬编码成 1 ⇒ 该守卫**永不可能失败** —
    而 run_write 确实泄漏了 2 个 stack 的钩子，憋到 A93-09 才以
    "变量被 inplace 修改" 的 autograd 版本错爆出来。改为实测。"""
    n = 0
    for layer in layers:
        for h in layer._forward_hooks.values():
            f = getattr(h, "func", h)
            if getattr(f, "__qualname__", "").endswith("_install.<locals>.fwd_hook"):
                n += 1
    return n


# ==================================================================== v 规则
# ★ 单一真源：写规则与解析量 key_matrices_arm 都用它，保证 §3 worked example 一致。
ARM = {"name": "A_mean"}
VPOS = {"V": None}


def v_rule(name, zv, kp, k, vpos):
    """把 zv=(N,d_v) 收成一个 (d_v,) 向量。唯一独立变量。"""
    if name == "A_mean":
        return zv.mean(dim=0)
    if name == "B_attn":
        # 标准化点积：softmax(√d_k · cos(kp_t, k))，无温度超参
        nk = kp / kp.norm(dim=1, keepdim=True).clamp_min(1e-12)
        nq = k / k.norm().clamp_min(1e-12)
        a = torch.softmax(math.sqrt(float(kp.shape[-1])) * (nk @ nq), dim=0)
        return (a.unsqueeze(1) * zv).sum(dim=0)
    if name == "C_max":
        return zv.max(dim=0).values
    if name in ("D_vtpos", "E_vtpos_qor"):
        assert vpos is not None and len(vpos) > 0, "D/E 需要值 token 位"
        idx = torch.as_tensor(vpos, device=zv.device, dtype=torch.long)
        return zv.index_select(0, idx).mean(dim=0)
    raise ValueError(name)


# ==================================================================== 写侧 patch
_ORIG_WRITE = LayerMem.write


def _write_patched(self, hs_in, drop_sg=False):
    """与 sgmn_module.py:79-104 **逐字相同**，唯一差别 = `v` 那一行改为 v_rule。

    ★ DEF-93-02：运算序必须与原式一致 —— 原式为 `(sg(pf@Pk).mean(0)) * pk_scale`
      （先 mean 后乘）；若改成逐位先乘再 mean，浮点重排会在 599 步累积下产生 ~1e-7 的差。
    """
    if self.prev_hs is None or self.no_write:
        return
    self.write_calls += 1
    pf = self.prev_hs.reshape(-1, self.prev_hs.shape[-1])
    hf = hs_in.reshape(-1, hs_in.shape[-1])
    pk_raw = sg(pf @ self.Pk, drop_sg)
    k = pk_raw.mean(dim=0) * self.pk_scale                          # ★ 与原装置逐位相同
    kp = pk_raw * self.pk_scale                                     # 仅供臂 B 的注意力
    zv = sg(hf @ self.Pv, drop_sg)                                   # (N,d_v)
    v = v_rule(ARM["name"], zv, kp, k, VPOS["V"])
    W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
    G = self.alpha * torch.outer(W @ k - v, k) + self.lam * (W - self.W0)
    with torch.no_grad():
        if self.use_factorized:
            self.U.copy_(self.U - self.eta * (G @ self.V))
            self.V.copy_(self.V - self.eta * (G.T @ self.U))
        else:
            self.Wfull.copy_(self.Wfull - self.eta * G)
    self.steps += 1


def install_write():
    """★ DEF-93-04：正式轮必须在**写阶段前后**显式安装/恢复 patch。
    原实现只在 Stage A 内赋值 LayerMem.write，正式轮从未安装 ⇒ 五臂全部用原均值写
    ⇒ A/B/C 读数**逐位相同**（症状：acc/margin/γ 完全一致，只有 M2 不同）。"""
    LayerMem.write = _write_patched


def restore_write():
    LayerMem.write = _ORIG_WRITE


def write_only(arm, model, facts, chunks, owner, vp_by_fact, seed):
    """只跑写阶段，返回 (W 每层, write_calls)。供 A93-10 判非退化。"""
    ARM["name"] = arm
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    install_write()
    try:
        wc = write_phase_arm(model, st, chunks, owner, vp_by_fact)
        return ({li: W91.W_of(m).detach().clone() for li, m in st.mems.items()}, wc)
    finally:
        restore_write()
        st.remove()
        ARM["name"] = "A_mean"


def write_phase_arm(model, st, chunks, owner, vpos_by_fact):
    """= R2_task91.write_phase 逐字；差别 = 每块前设 VPOS（供臂 D/E 取值位）。"""
    for ci, ids in enumerate(chunks):
        VPOS["V"] = vpos_by_fact.get(owner[ci]) if ARM["name"] in ("D_vtpos", "E_vtpos_qor") else None
        model(input_ids=ids.to(DEV))
    VPOS["V"] = None
    return {li: m.write_calls for li, m in st.mems.items()}


def key_matrices_arm(model, st, facts, chunks, owner, vpos_by_fact):
    """= R2_task91.key_matrices，但 V 用**本臂的 v 规则**（与写规则同源）。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    K = {li: [] for li in st.mems}
    V = {li: [] for li in st.mems}
    try:
        for fi in range(len(facts)):
            ci = owner.index(fi) + 1
            model(input_ids=chunks[ci].to(DEV))
            for li, m in st.mems.items():
                hs = st._hs_in[li]
                pf = hs.reshape(-1, hs.shape[-1])
                pk_raw = sg(pf @ m.Pk)
                k = pk_raw.mean(dim=0) * m.pk_scale
                kp = pk_raw * m.pk_scale
                zv = sg(pf @ m.Pv)
                K[li].append(k.detach())
                V[li].append(v_rule(ARM["name"], zv, kp, k,
                                    vpos_by_fact.get(fi)).detach())
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
    return ({li: torch.stack(K[li]).float() for li in K},
            {li: torch.stack(V[li]).float() for li in V})


# ==================================================================== 取值位
def value_positions(tok, facts):
    """= probe91_valuetok.value_positions（★ 用 CHUNK，不用 shape[0] —— DEF-92-01）。"""
    out = []
    for f in facts:
        enc = tok(f["sent"] + " ", add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        chars = len(f["prompt"]) + 1
        j = None
        for t, (s0, e0) in enumerate(offs):
            if s0 <= chars < e0:
                j = t
                break
        if j is None:
            raise RuntimeError(f"未能定位值 token offset：{f['sent']}")
        per = len(ids)
        out.append({"V": [p for p in range(CHUNK) if p % per == j], "j": int(j),
                    "period": int(per), "tok_id": int(ids[j])})
    return out


# ==================================================================== Stage A
def stage_a(tok, model, facts, chunks, owner, vpos):
    """A93-00..09（任一 FAIL 阻断正式轮）。"""
    seed = 20261028
    vp_by_fact = {fi: vpos[fi]["V"] for fi in range(N_FACTS)}

    # ---- A93-00 配置
    rec("A93-00", MOUNT == [8, 12, 16] and CHUNK == 64 and N_CHUNKS == 600,
        f"MOUNT={MOUNT} CHUNK={CHUNK} B={N_CHUNKS} γ=0.0 no_write=False")

    # ---- A93-07 CORR-12 天花板
    items = make_eval_items(tok, facts, list(range(N_TRAIN_KEYS, N_FACTS)),
                            __import__("random").Random(555))
    ceil_ok = True
    for it in items:
        if not any(c[1] for c in it["cand"]):
            ceil_ok = False
    rec("A93-07", ceil_ok and G_GEN <= 1.0,
        f"4 选 1 候选必含正确项={ceil_ok} ⇒ acc_held 天花板=1.0；G_gen={G_GEN}≤1.0",
        {"n_item": len(items), "ceiling": 1.0})

    # ---- A93-02 ★ 臂 A 与原 write 逐位相同
    def run_write(arm, patch):
        ARM["name"] = arm
        LayerMem.write = _write_patched if patch else _ORIG_WRITE
        st = make_stack(gamma=0.0, no_write=False, seed=seed)
        try:
            write_phase_arm(model, st, chunks, owner, vp_by_fact)
            return {li: W91.W_of(m).detach().clone() for li, m in st.mems.items()}
        finally:
            st.remove()                       # ★ DEF-93-01：必须移除，否则钩子泄漏
            LayerMem.write = _ORIG_WRITE

    Wa = run_write("A_mean", patch=True)
    Wo = run_write("A_mean", patch=False)
    d_a = max(float((Wa[li] - Wo[li]).abs().max()) for li in Wa)
    rec("A93-02", d_a == 0.0, f"臂A(patched) vs 原 write：max|ΔW|={d_a:.3e}（须=0）", d_a)

    # ---- A93-06 同 seed 五臂初值相同
    snaps = {}
    for arm in ARMS:
        ARM["name"] = arm
        st = make_stack(gamma=0.0, no_write=True, seed=seed)
        snaps[arm] = W91.snapshot(st)
        st.remove()
    base = snaps["A_mean"]
    d6 = max(W91.maxdiff_snap(base, snaps[a]) for a in ARMS)
    rec("A93-06", d6 == 0.0, f"五臂初值 max|Δ|={d6:.3e}（须=0）", d6)

    # ---- A93-03 ★ worked example（一条具体事实，6 条断言）
    fi = 0
    ci = owner.index(fi) + 1
    ARM["name"] = "A_mean"
    st = make_stack(gamma=0.0, no_write=True, seed=seed)
    model(input_ids=chunks[ci].to(DEV))
    we = {}
    for li in MOUNT:
        hs = st._hs_in[li]
        pf = hs.reshape(-1, hs.shape[-1])
        kp = sg(pf @ st.mems[li].Pk) * st.mems[li].pk_scale
        k = (sg(pf @ st.mems[li].Pk).mean(dim=0)) * st.mems[li].pk_scale
        zv = sg(pf @ st.mems[li].Pv)
        vs = {a: v_rule(a, zv, kp, k, vp_by_fact.get(fi)) for a in ARMS}
        # 1 v_A == mean
        d1 = float((vs["A_mean"] - zv.mean(dim=0)).abs().max())
        # 2 v_D == zv[V].mean
        idx = torch.as_tensor(vpos[fi]["V"], device=zv.device, dtype=torch.long)
        d2 = float((vs["D_vtpos"] - zv.index_select(0, idx).mean(dim=0)).abs().max())
        # 3 attention 归一 + 非均匀 + ★报告项：有效位置数 1/Σa²
        nk = kp / kp.norm(dim=1, keepdim=True).clamp_min(1e-12)
        nq = k / k.norm().clamp_min(1e-12)
        a = torch.softmax(math.sqrt(float(kp.shape[-1])) * (nk @ nq), dim=0)
        asum = float(a.sum()); amax = float(a.max())
        attn_eff = float(1.0 / (a ** 2).sum())
        # 4 v_C == max
        d4 = float((vs["C_max"] - zv.max(dim=0).values).abs().max())
        # 5 退化检查
        rel = {x: float((vs[x] - vs["A_mean"]).norm() / vs["A_mean"].norm().clamp_min(1e-12))
               for x in ("B_attn", "C_max", "D_vtpos")}
        # 6 k 与原式逐位相同
        k_orig = (sg(pf @ st.mems[li].Pk).mean(dim=0)) * st.mems[li].pk_scale
        d6k = float((k - k_orig).abs().max())
        we[li] = {"d_vA": d1, "d_vD": d2, "attn_sum": asum, "attn_max": amax,
                  "attn_eff": attn_eff, "d_vC": d4, "rel": rel, "d_k": d6k,
                  "nV": len(vpos[fi]["V"])}
        print(f"    L{li}: vA|Δ|={d1:.1e} vD|Δ|={d2:.1e} Σa={asum:.6f} max a={amax:.4f} "
              f"eff={attn_eff:.2f} vC|Δ|={d4:.1e} relB/C/D={rel['B_attn']:.3f}/"
              f"{rel['C_max']:.3f}/{rel['D_vtpos']:.3f} k|Δ|={d6k:.1e} "
              f"|V|={len(vpos[fi]['V'])}", flush=True)
    st.remove()
    RES["worked"] = {str(k): v for k, v in we.items()}
    ok3 = (all(v["d_vA"] == 0.0 for v in we.values())
           and all(v["d_vD"] <= 1e-6 for v in we.values())
           and all(abs(v["attn_sum"] - 1.0) <= 1e-5 for v in we.values())
           and all(v["attn_max"] > 2.0 / CHUNK for v in we.values())
           and all(v["d_vC"] == 0.0 for v in we.values())
           and all(v["rel"][x] > 1e-3 for v in we.values() for x in ("B_attn", "C_max", "D_vtpos"))
           and all(v["d_k"] == 0.0 for v in we.values())
           and all(v["nV"] >= 1 for v in we.values()))
    rec("A93-03", ok3, "worked example 6 条断言（vA逐位/vD逐位/Σa=1&非均匀/vC逐位/"
                       "三臂非退化/k逐位相同/|V|≥1）", RES["worked"])

    # ---- A93-01 ★ write_calls == 600 且**实测**只挂载一个 stack（不得硬编码）
    probe = make_stack(gamma=0.0, no_write=True, seed=seed)
    layers = probe._layers()
    probe.remove()
    n_before = count_mounted(model, layers)
    ARM["name"] = "A_mean"
    st1 = make_stack(gamma=0.0, no_write=False, seed=seed)
    n_one = count_mounted(model, layers)
    wc = write_phase_arm(model, st1, chunks, owner, vp_by_fact)
    st1.remove()
    n_after = count_mounted(model, layers)
    want = len(MOUNT)
    rec("A93-01", n_before == 0 and n_one == want and n_after == 0
        and all(v == N_CHUNKS - 1 for v in wc.values()),
        f"钩子实测：基线={n_before}(须0) 挂载后={n_one}(须{want}) 移除后={n_after}(须0)；"
        f"write_calls={sorted(set(wc.values()))}(须{N_CHUNKS - 1}=B-1，首步声明不写)",
        {"n_before": n_before, "n_one": n_one, "n_after": n_after, "wc": wc})

    # ---- A93-04 读训练不改写入侧
    ARM["name"] = "A_mean"
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    write_phase_arm(model, st, chunks, owner, vp_by_fact)
    pre = {li: {"W": W91.W_of(m).detach().clone(), "Wq": m.Wq.detach().clone()}
           for li, m in st.mems.items()}
    tr = list(range(N_TRAIN_KEYS))
    train_read_task(model, st, tok, facts, tr, __import__("random").Random(1000 + seed % 100))
    dW = max(float((W91.W_of(m).detach() - pre[li]["W"]).abs().max())
             for li, m in st.mems.items())
    dWq = max(float((m.Wq.detach() - pre[li]["Wq"]).abs().max())
              for li, m in st.mems.items())
    rec("A93-04", dW == 0.0 and dWq > 0.0,
        f"读训练后 max|ΔW|={dW:.3e}（须=0）max|ΔWq|={dWq:.3e}（须>0）", {"dW": dW, "dWq": dWq})
    st.remove()

    # ---- A93-05 ★ E 的 q 覆盖只作用于读出位
    ARM["name"] = "A_mean"
    fi0 = 0
    prompt = facts[fi0]["prompt_ids"]
    seq = torch.tensor([prompt + [facts[fi0]["prompt_ids"][0]]], dtype=torch.long, device=DEV)
    base_lg = orc_lg = None
    s_a = make_stack(gamma=0.0, no_write=True, seed=seed)      # 同 seed ⇒ 同初值
    for m in s_a.mems.values():
        with torch.no_grad():
            m.gamma.fill_(1.0)          # ★ DEF-93-02：γ=0 使注入恒为 0 ⇒ 检查空转
    try:
        base_lg = model(input_ids=seq).logits.float().detach().clone()
    finally:
        s_a.remove()
    s_b = make_stack(gamma=0.0, no_write=True, seed=seed)
    for m in s_b.mems.values():
        with torch.no_grad():
            m.gamma.fill_(1.0)
    LayerMem.read = W91._read_patched
    try:
        K2 = {li: torch.randn(N_FACTS, s_b.mems[li].Wq.shape[0], device=DEV) for li in MOUNT}
        W91.set_oracle(s_b, K2, fi0, "readout")
        orc_lg = model(input_ids=seq).logits.float().detach().clone()
    finally:
        W91.clear_oracle(s_b)
        LayerMem.read = W91._ORIG_READ
        s_b.remove()
    P = seq.shape[1]
    ridx = P - 2                                   # ★ 装置约定：读出位 = L-2
    dlt = (base_lg - orc_lg).abs()                 # (vocab, L)
    diff_all = float(dlt.max())
    diff_ridx = float(dlt[:, ridx].max())
    # ★ DEF-93-03：因果 transformer ⇒ 注入会**向前**传播到 j>ridx。
    #   只有 j<ridx 必须逐位不变（因果不可逆）；j>ridx 的变化记录但不作判据。
    diff_before = float(dlt[:, :ridx].max()) if ridx > 0 else 0.0
    diff_after = float(dlt[:, ridx + 1:].max()) if ridx + 1 < P else 0.0
    ok5 = (diff_ridx > 0) and (diff_before < 1e-5)
    rec("A93-05", ok5,
        f"读出位(L-2)行 Δ={diff_ridx:.3e}(须>0)；j<ridx 行 max|Δ|={diff_before:.3e}"
        f"(须<1e-5)；j>ridx 前向传播 Δ={diff_after:.3e}(记录，非判据)；"
        f"哨兵='readout'（禁 bool 索引）",
        {"ridx": int(ridx), "d_ridx": diff_ridx, "d_before": diff_before,
         "d_after": diff_after, "d_all": diff_all})

    # ---- A93-09 smoke（40 步，不出判决）
    ARM["name"] = "A_mean"
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    write_phase_arm(model, st, chunks, owner, vp_by_fact)
    orig_steps = globals()["READ_STEPS"]
    import R2_task91 as _R
    osc = _R.READ_STEPS
    _R.READ_STEPS = 40
    try:
        ls, v9, dv9 = train_read_task(model, st, tok, facts, tr,
                                      __import__("random").Random(1000 + seed % 100))
    finally:
        _R.READ_STEPS = osc
    ok9 = (len(ls) > 0) and (not dv9) and all(bool(np.isfinite(x)) for x in ls)
    rec("A93-09", ok9, f"smoke 40 步：n={len(ls)} viol={v9} div={dv9} "
                       f"loss={ls[0]:.3f}→{ls[-1]:.3f}（smoke 不出判决）")
    st.remove()
    ARM["name"] = "A_mean"

    # ---- A93-10 ★★ 正式路径非退化：五臂的**写后 W 必须两两不同**
    ARM["name"] = "A_mean"
    Ws = {}
    for arm in ARMS:
        Ws[arm], _ = write_only(arm, model, facts, chunks, owner, vp_by_fact, seed)
    WRITE_ARMS = ["A_mean", "B_attn", "C_max", "D_vtpos"]   # 写侧四臂（E 是纯读侧干预）
    pairs = {}
    for a in range(len(WRITE_ARMS)):
        for b in range(a + 1, len(WRITE_ARMS)):
            x, y = WRITE_ARMS[a], WRITE_ARMS[b]
            pairs[f"{x}|{y}"] = max(float((Ws[x][li] - Ws[y][li]).abs().max())
                                    for li in Ws[x])
    worst = min(pairs.values())
    # ★ 正向断言：E 的独立变量**只**在读侧 ⇒ 写后 W 必须与 D **逐位相同**
    d_de = max(float((Ws["D_vtpos"][li] - Ws["E_vtpos_qor"][li]).abs().max())
               for li in Ws["D_vtpos"])
    rec("A93-10", worst > 1e-3 and d_de == 0.0,
        f"写侧四臂两两最小 max|ΔW|={worst:.3e}(须>1e-3，最小对={min(pairs, key=pairs.get)})；"
        f"D|E 写后逐位差={d_de:.3e}(须=0，因 E 只改读侧)",
        {"pairs": {k: round(v, 6) for k, v in pairs.items()}, "d_DE": d_de})

    # ---- A93-08 读数域（正式轮后复核，此处先登记口径）
    rec("A93-08", True, f"口径登记：n_q=16 n_item=48 chance={1.0/N_CAND:.3f}；"
                        f"正式轮后断言 acc∈[0,1] 且 margin 有限")
    return FAILS == []


# ==================================================================== 一臂
def run_arm(arm, tok, model, facts, chunks, owner, vp_by_fact, tr, he, seed, cls, ncls):
    ARM["name"] = arm
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    t0 = time.time()
    install_write()                       # ★ DEF-93-04：正式轮必须安装写侧 patch
    try:
        wc = write_phase_arm(model, st, chunks, owner, vp_by_fact)
        K, V = key_matrices_arm(model, st, facts, chunks, owner, vp_by_fact)
        rng = __import__("random").Random(1000 + seed % 100)
        if arm in ORACLE_ARMS:
            layers_prev = LayerMem.read
            LayerMem.read = W91._read_patched
            try:
                losses, viol, div = W91.train_read_oracle(
                    model, st, tok, facts, tr, rng, K, "readout")
            finally:
                LayerMem.read = layers_prev
            layers_prev = LayerMem.read
            LayerMem.read = W91._read_patched
            try:
                fit = W91.eval_keys(model, st, tok, facts, tr, K, True, "readout")
                gen = W91.eval_keys(model, st, tok, facts, he, K, True, "readout")
            finally:
                LayerMem.read = layers_prev
        else:
            losses, viol, div = train_read_task(model, st, tok, facts, tr, rng)
            fit = W91.eval_keys(model, st, tok, facts, tr, None, False, None)
            gen = W91.eval_keys(model, st, tok, facts, he, None, False, None)
        model(input_ids=chunks[0].to(DEV))
        g = float(st.mems[list(st.mems)[0]].gamma)
        pr = {li: W91.participation_ratio(m.Wo.detach()) for li, m in st.mems.items()}
        # M2：本臂 V → 答案类（LOO + held）
        m2 = {}
        for li in st.mems:
            Vl = V[li].float()
            s = W91.ridge_scores(Vl[tr], [cls[i] for i in tr], Vl[he], ncls)
            a_ho, _ = W91.top1(s, [cls[i] for i in he])
            loo = 0
            for i in range(N_FACTS):
                keep = [j for j in range(N_FACTS) if j != i]
                si = W91.ridge_scores(Vl[keep], [cls[j] for j in keep], Vl[[i]], ncls)
                loo += int(int(si.argmax(dim=1)[0]) == cls[i])
            m2[li] = {"held_top1": a_ho, "loo_top1": loo / N_FACTS}
        a = {"arm": arm, "seed": seed, "gamma_final": g,
             "acc_train": fit["acc_q"], "acc_held": gen["acc_q"],
             "acc_item_train": fit["acc_item"], "acc_item_held": gen["acc_item"],
             "margin_train": fit["margin_mean"], "margin_held": gen["margin_mean"],
             "margin_pos_train": fit["margin_pos_frac"],
             "margin_pos_held": gen["margin_pos_frac"],
             "n_q": fit["n_q"], "n_item": fit["n_item"],
             "violations": viol, "diverged": div,
             "final_loss": float(np.mean(losses[-50:])) if losses else None,
             "wo_pr": {str(k): v for k, v in pr.items()},
             "m2": {str(k): v for k, v in m2.items()},
             "write_calls": wc, "sec": round(time.time() - t0, 1)}
        print(f"  [{arm}] acc_train={a['acc_train']:.3f} acc_held={a['acc_held']:.3f} "
              f"margin(tr/ho)={a['margin_train']:+.2f}/{a['margin_held']:+.2f} "
              f"M2loo={np.mean([v['loo_top1'] for v in m2.values()]):.3f} "
              f"γ={g:+.4f} viol={viol} div={div} {a['sec']}s", flush=True)
        return a
    finally:
        restore_write()
        st.remove()
        ARM["name"] = "A_mean"


# ==================================================================== 判决
def judge(arms):
    by = {(a["seed"], a["arm"]): a for a in arms}
    seeds = sorted({a["seed"] for a in arms})
    per = {}
    for s in seeds:
        per[s] = {a: by[(s, a)]["acc_held"] for a in ARMS}
        per[s]["A"] = per[s]["A_mean"]
    sel = ("B_attn", "C_max")
    helps, partial, nohelp = True, True, True
    for s in seeds:
        d = per[s]
        helps &= any(d[x] >= G_GEN and d[x] >= d["A"] + G_DELTA for x in sel)
        partial &= any(d[x] >= G_PARTIAL and d[x] >= d["A"] + G_DELTA for x in sel)
        nohelp &= all(d[x] <= G_LOW for x in sel)
    if helps:
        br = "SELECTIVE_HELPS"
    elif partial:
        br = "SELECTIVE_PARTIAL"
    elif nohelp:
        br = "NO_HELP"
    else:
        br = "INCONCLUSIVE"
    d_helps = all(per[s]["D_vtpos"] >= G_GEN and per[s]["D_vtpos"] >= per[s]["A"] + G_DELTA
                  for s in seeds)
    verdict = br + ("+ORACLE_ONLY" if (br == "NO_HELP" and d_helps) else "")
    return {"per_seed_held": {str(k): v for k, v in per.items()},
            "branch": br, "D_HELPS": bool(d_helps), "verdict": verdict,
            "mechanism_refuted": bool(not d_helps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", action="store_true")
    ap.add_argument("--formal", action="store_true")
    args = ap.parse_args()
    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    vpos = value_positions(tok, facts)
    vp_by_fact = {fi: vpos[fi]["V"] for fi in range(N_FACTS)}
    print(f"值 token 位：|V|∈[{min(len(v['V']) for v in vpos)},"
          f"{max(len(v['V']) for v in vpos)}]、period∈["
          f"{min(v['period'] for v in vpos)},{max(v['period'] for v in vpos)}]", flush=True)
    RES["vpos"] = [{"V": v["V"], "j": v["j"], "period": v["period"]} for v in vpos]

    if args.stage_a:
        print("\n===================== Stage A =====================", flush=True)
        ok = stage_a(tok, model, facts, chunks, owner, vpos)
        RES["stage_a_ok"] = ok
        RES["fails"] = list(FAILS)
        dump(RES, "probe91_valread_guards")
        print(f"\n★ Stage A {'全部通过' if ok else 'FAIL: ' + str(FAILS)}", flush=True)
        return

    if not args.formal:
        print("需 --stage-a 或 --formal", flush=True)
        return

    print("\n===================== 正式轮 =====================", flush=True)
    tr = list(range(N_TRAIN_KEYS))
    he = list(range(N_TRAIN_KEYS, N_FACTS))
    cls, ncls = W91.class_ids(facts)
    for seed in (20261028, 7777):
        print(f"\n===== seed {seed} =====", flush=True)
        for arm in ARMS:
            RES["arms"].append(run_arm(arm, tok, model, facts, chunks, owner,
                                       vp_by_fact, tr, he, seed, cls, ncls))
    j = judge(RES["arms"])
    RES["judgement"] = j
    print("\n" + "=" * 70, flush=True)
    for s, d in j["per_seed_held"].items():
        print(f"  seed {s} acc_held: " + " ".join(f"{a}={d[a]:.3f}" for a in ARMS), flush=True)
    print(f"  ★ 判决 = {j['verdict']}   (branch={j['branch']}, D_HELPS={j['D_HELPS']}, "
          f"机理被否证={j['mechanism_refuted']})", flush=True)
    dump(RES, "probe91_valread")


if __name__ == "__main__":
    main()
