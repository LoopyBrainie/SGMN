#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-21 · **写侧容量/衰减探针（第二版：走真实扫描路径）** —— ★ **诊断用；不是科学结论** ★

------------------------------------------------------------------
**为什么要 probe21b（探针 1 的三个不足）**：
  ① 探针 1 **没训练** ⇒ `ortho_idx` 的相位是 0；正式路径里训练已烧掉
     `200 篇 × 68 token` ⇒ 扫描的键列**起点不是 0** ⇒ **列复用情节随相位改变**
     ⇒ 探针 1 算出的复用情节（14 / 225 / 133 / 225）**不能直接用**。
  ② 探针 1 **没清 `keys_written`** ⇒ 正式路径里训练残留仍在 `keys_written` 里
     ⇒ 「文档 1 的原始 (k,v)」**取不到**（缺陷 **D111**）。
  ③ 探针 1 测的是**余弦**；Advisor 指出：**PASS/FAIL 的 R(K) 必须保持冻结的归一化余弦**
     （改成信号投影会让 H21-2/3/4 变成代数恒等式 ⇒ 制造伪通过）⇒
     ⇒ 本探针把**余弦（PASS/FAIL 口径）**、**槽位余弦（阴性对照）**、
     **信号投影（仅作 Rp 诊断）** 三者同时测出来。

------------------------------------------------------------------
**本探针只回答四件事（开跑前把协议钉死）**：
  Q1 正式路径下**锚条目**（文档 1 的码 token 条目）的**原始 (k, v)** 能否取到？
  Q2 归一化余弦 R(K) 的**首个 ≤0.5 的 K**（= 冻结的半衰期定义）在各臂落在哪里？
     ⇒ 决定扫描网格必须多长（否则 H21-3 是**删失**，不是 PASS）。
  Q3 锚条目的**键列**在正式路径下于哪个情节被**再次写入**（列复用）
     ⇒ 决定每条保持曲线的**有效窗口**。
  Q4 `r_cnt`（驻留且自寻址的条目数）与 `occupied`（槽占用）是否给出**同一个容量膝点**
     ⇒ 决定 H21-1b 的测量对象（冻结的 `R_cnt` 保留哪一个口径）。

**边界**：不训练出科学结论；不产生判据；不进证据表；只用于**冻结 rev 1.7 的协议口径**。
"""

import json
import math
import pathlib
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R1_granularity as RG            # noqa: E402

SEED = 20261028
KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 256
N_TRAIN = 200
OUT_JSON = HERE.parent / "dir21_probe21b.json"


def set_arm(st, eta, stride, pick):
    """与 runner 的 `_set_arm` 逐字同构。"""
    for m in st.mems.values():
        m.eta = float(eta)
        m.core.eta = float(eta)
        m.write_stride = int(stride)
        m.stride_pick = str(pick)


def cos_ab(a, b):
    na = float(a.norm())
    nb = float(b.norm())
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float((a @ b) / (na * nb))


def run_arm(model, d, name, eta, stride, pick, ev, base0):
    st = RG.make_stack(model, d, "cue", seed=SEED, mem_kind="hebbian_full",
                       eta=RG.ETA_BASE, key_mode="ortho", variant="normal")
    RG.clear_slots(st)
    # ★★ 缺陷 D111：`clear_slots` **不清** `keys_written` / `vals_written`
    #   ⇒ 训练残留仍在 ⇒ 「文档 1 的原始 (k,v)」会取错 ⇒ 必须在扫描前显式清空。
    for m in st.mems.values():
        m.keys_written.clear()
        m.vals_written.clear()
    set_arm(st, eta, stride, pick)
    m0 = st.mems[list(st.mems)[0]]
    N = int(m0.core.N)
    trig = list(m0.trigger_positions(len(ev[0]["ids"])))
    m_doc = 2 * len(trig)
    code_i = len(trig) - 1
    t_code = int(trig[code_i])

    # —— 写文档 1，捕获它的原始 (k, v)（**按写入顺序的前 len(trig) 条**）——
    st.unfreeze_write()
    RG.write_both(st, ev[0])
    anchor = []
    for m in st.mems.values():
        ks = [t.detach().clone() for t in m.keys_written[:len(trig)]]
        vs = [t.detach().clone() for t in m.vals_written[:len(trig)]]
        anchor.append((ks[code_i], vs[code_i]))
    slot_anchor = code_i % N

    rows = []
    for K in range(1, KMAX + 1):
        if K > 1:
            st.unfreeze_write()
            RG.write_both(st, ev[K - 1])
        writes = int(m0.core.count)
        cs_o, cs_s, proj = [], [], []
        for m, (k, v) in zip(st.mems.values(), anchor):
            with torch.no_grad():
                W = m.core._W()
                cs_o.append(cos_ab(W @ k, v))                    # ★ 冻结口径：cos(W·k, v)
                cs_s.append(float(m.core.fidelity(slot_anchor)))  # 阴性对照：当前槽住户
                proj.append(float(((W - m.core.W0) @ k).norm()))   # Rp 诊断：信号投影范数
        rows.append({"K": K, "writes": writes,
                     "cos_orig": float(np.mean(cs_o)),
                     "cos_slot": float(np.mean(cs_s)),
                     "proj": float(np.mean(proj)),
                     "r_cnt": RG.resident_r_cnt(st),
                     "occupied": min(writes, N)})
    st.remove()
    torch.cuda.empty_cache()
    return {"name": name, "m_doc": int(m_doc), "n_entries_doc": int(len(trig)),
            "base0": int(base0), "rows": rows}


def column_schedule(trig, base0, kmax):
    """★ 锚条目键列的**列复用情节**（按真实相位 base0 精确计算）。

    键列：目标前视起点 = `base0 + 68·(e−1)`，干扰前视起点 = 再 +34；
    写入列 = 前视起点 + (t−1)（t ∈ 触发位置）。
    锚条目 = 文档 1 目标前视的码 token ⇒ 列 `base0 + 33`（mod 896）。
    """
    D = 896
    j_anchor = (base0 + 33) % D
    per_ep = 68
    for e in range(2, kmax + 2):
        bt = base0 + per_ep * (e - 1)
        bd = bt + 34
        cols = [((bt + t - 1) % D) for t in trig] + [((bd + t - 1) % D) for t in trig]
        if j_anchor in cols:
            return e, j_anchor
    return None, j_anchor


def summarize(arm, reuse_ep):
    rows = arm["rows"]
    c0 = rows[0]["cos_orig"]
    out = {"name": arm["name"], "m_doc": arm["m_doc"], "base0": arm["base0"],
           "cos_orig_K1": c0, "cos_slot_K1": rows[0]["cos_slot"],
           "proj_K1": rows[0]["proj"], "reuse_episode": reuse_ep}
    # —— 冻结的半衰期定义：首个「逐种子归一余弦」≤0.5 的 K ——
    hl = next((r["K"] for r in rows if c0 > 0 and (r["cos_orig"] / c0) <= 0.5), None)
    out["half_life_K_cos"] = hl
    out["half_life_writes_cos"] = (hl * arm["m_doc"]) if hl else None
    out["half_life_censored_cos"] = hl is None
    # —— 槽位口径（阴性对照）会给出什么 ——
    s0 = rows[0]["cos_slot"]
    hls = next((r["K"] for r in rows if s0 > 0 and (r["cos_slot"] / s0) <= 0.5), None)
    out["half_life_K_slotpolluted"] = hls
    # —— 容量口径：r_cnt 与 occupied 的**首个最大值** ——
    for key in ("r_cnt", "occupied"):
        vals = [(r["K"], r[key]) for r in rows]
        mx = max(v for _, v in vals)
        out["first_max_" + key] = next((k for k, v in vals if v == mx), None)
        out["peak_" + key] = mx
    return out


def main():
    print("=== DIR-21 probe21b（真实扫描路径 · 诊断用 · 非科学结论）===", flush=True)
    tok = AutoTokenizer.from_pretrained(RG.MODEL)
    model = AutoModelForCausalLM.from_pretrained(RG.MODEL, dtype=torch.float32).to(RG.DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size
    tr, ev = RG.build_docs(tok, SEED, N_TRAIN, KMAX + 2)
    print(f"[数据] 训练 {len(tr)} 篇 / 评测 {len(ev)} 篇；文档长度 {len(ev[0]['ids'])}", flush=True)

    # —— 真实路径：训练一次 → 快照 → 恢复 → 清槽 → 扫描 ——
    st = RG.make_stack(model, d, "cue", seed=SEED, mem_kind="hebbian_full",
                       eta=RG.ETA_BASE, key_mode="ortho", variant="normal")
    print("[训练] 200 篇（token 级 · ortho · η=1e−1）…", flush=True)
    RG.train_arm(st, tr, "cue", variant="normal")
    snap = st.snapshot_all()
    base0 = int(st.mems[list(st.mems)[0]].ortho_idx)
    print(f"[训练] 完成；扫描起点 ortho_idx = {base0}（★ 相位 ≠ 0 ⇒ 列复用情节随臂改变）",
          flush=True)
    st.remove()
    torch.cuda.empty_cache()

    arms, summaries = [], []
    for (name, eta, stride, pick) in RG.ARMS21:
        st2 = RG.make_stack(model, d, "cue", seed=SEED, mem_kind="hebbian_full",
                            eta=RG.ETA_BASE, key_mode="ortho", variant="normal")
        st2.restore_all(snap)
        RG.clear_slots(st2)
        for m in st2.mems.values():
            m.keys_written.clear()
            m.vals_written.clear()
        st2.remove()
        torch.cuda.empty_cache()

        arm = run_arm(model, d, name, eta, stride, pick, ev, base0)
        trig = list(range(1, 35)) if stride == 1 else (
            [17, 34] if stride == 17 else ([18, 34] if stride == 18 else [34]))
        reuse_ep, j_anchor = column_schedule(trig, base0, KMAX)
        arm["anchor_col"] = j_anchor
        arm["reuse_episode"] = reuse_ep
        arms.append(arm)
        s = summarize(arm, reuse_ep)
        summaries.append(s)
        print(f"\n--- {name}  m_doc={arm['m_doc']}  每篇条目={arm['n_entries_doc']}  "
              f"锚列={j_anchor}  列复用情节={reuse_ep} ---", flush=True)
        print(f"    余弦 K=1：原对象 {s['cos_orig_K1']:.4f}（PASS/FAIL 口径）｜"
              f"槽住户 {s['cos_slot_K1']:.4f}（阴性对照）｜信号投影范数 {s['proj_K1']:.4g}（Rp）",
              flush=True)
        print(f"    冻结半衰期（首个归一余弦 ≤0.5）：K={s['half_life_K_cos']} ⇒ "
              f"写步={s['half_life_writes_cos']}（删失={s['half_life_censored_cos']}）；"
              f"槽口径半衰期 K={s['half_life_K_slotpolluted']}", flush=True)
        print(f"    容量：r_cnt 首个最大值 K={s['first_max_r_cnt']}（峰 {s['peak_r_cnt']}）｜"
              f"occupied 首个最大值 K={s['first_max_occupied']}（峰 {s['peak_occupied']}）", flush=True)
        show = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256]
        print("      K  writes  cos_orig  cos_slot  proj      r_cnt  occupied", flush=True)
        for r in arm["rows"]:
            if r["K"] in show and r["K"] <= KMAX:
                c0 = arm["rows"][0]["cos_orig"]
                print(f"     {r['K']:4d} {r['writes']:7d}  {r['cos_orig']:.4f}   {r['cos_slot']:.4f}"
                      f"   {r['proj']:.5f}   {r['r_cnt']:5d}   {r['occupied']:5d}"
                      f"   R={r['cos_orig']/c0:.4f}", flush=True)

    res = {"probe": "DIR-21 probe21b (real scan path)", "note": "DIAGNOSTIC ONLY",
           "seed": SEED, "kmax": KMAX, "base0": base0, "n_train": N_TRAIN,
           "arms": arms, "summary": summaries}
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写出 {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
