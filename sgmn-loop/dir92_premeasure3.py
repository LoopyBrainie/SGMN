#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
r"""DIR-92 · **开卡前测量 ③**（**不是判据**）：`D-108` 的 0.148 与今天 V0 的 0.5594 差 3.8×，
   同一装置/同 $\eta$/同语料/同 600 块 —— 必有一个统计量口径问题。本探针在**同一个 stack 上**
   同时跑 `dir90.storage_diag` 的原函数与我的口径，二选一定出到底是哪个在变。
   若两者同 stack 同值 ⇒ 差异来自**上游状态**（写入块数/写入是否发生），须另查。
"""
import json, pathlib, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir90")
import numpy as np, torch
import R2_task91 as R91
import R1_task90 as R90

tok, model = R91.model_and_tok()
facts = R90.build_facts(tok)                       # V0 语料
chunks, owner = R90.build_chunks(tok, facts)

st = R91.make_stack(gamma=0.0, no_write=False)
try:
    # ★ 关键：先记录**写入前**的 W 与写入调用次数
    w_before = {li: R91.W_of(st.mems[li]).clone() for li in st.mems}
    n_blocks = len(chunks)
    R91.write_phase(model, st, chunks)
    print(f"chunks={n_blocks}  write_calls={ {li: m.write_calls for li, m in st.mems.items()} }", flush=True)
    dW = {li: float((R91.W_of(st.mems[li]) - w_before[li]).norm()) for li in st.mems}
    print("||W_after - W_before|| =", {k: round(v, 4) for k, v in dW.items()}, flush=True)

    # --- 口径 A：dir90.storage_diag 的原函数（median over facts of mean over layers） ---
    medA, perA = R90.storage_diag(model, st, facts, owner, chunks)
    # --- 口径 B：我的（每层 rel.median()，再跨层平均） ---
    K, V = R91.key_matrices(model, st, facts, chunks, owner)
    perB = {}
    for li in sorted(st.mems):
        W = R91.W_of(st.mems[li]).float()
        rel = ((K[li] @ W.T) - V[li]).norm(dim=1) / V[li].norm(dim=1).clamp_min(1e-30)
        perB[li] = float(rel.median())
    medB = float(np.mean(list(perB.values())))
    # --- 口径 C：median over facts of mean over layers，但用 key_matrices 的 K/V ---
    perC = []
    for fi in range(len(facts)):
        ls = [float(((K[li][fi] @ R91.W_of(st.mems[li]).float().T) - V[li][fi]).norm()
                    / V[li][fi].norm().clamp_min(1e-30)) for li in sorted(st.mems)]
        perC.append(float(np.mean(ls)))
    medC = float(np.median(perC))
    # --- 基线：不写入（W0）---
    st2 = R91.make_stack(gamma=0.0, no_write=True)
    try:
        K0, V0_ = R91.key_matrices(model, st2, facts, chunks, owner)
        base = {}
        for li in sorted(st2.mems):
            W = R91.W_of(st2.mems[li]).float()
            rel = ((K0[li] @ W.T) - V0_[li]).norm(dim=1) / V0_[li].norm(dim=1).clamp_min(1e-30)
            base[li] = float(rel.median())
        baseMed = float(np.mean(list(base.values())))
    finally:
        st2.remove()
    out = {"chunks": n_blocks,
           "write_calls": {li: m.write_calls for li, m in st.mems.items()},
           "dW_norm": dW,
           "A_dir90_storage_diag": medA, "A_per_fact": perA,
           "B_perlayer_median_then_mean": medB, "B_per_layer": perB,
           "C_median_over_facts_of_layer_mean": medC,
           "baseline_W0": baseMed, "baseline_per_layer": base}
    print(f"\nA dir90.storage_diag 原函数 = {medA:.4f}", flush=True)
    print(f"B 每层中位数再跨层均值       = {medB:.4f}", flush=True)
    print(f"C fact中位数(层均值)         = {medC:.4f}", flush=True)
    print(f"基线 W0（不写入）            = {baseMed:.4f}", flush=True)
    print(f"改善倍数 = {baseMed/medB:.2f}×", flush=True)
    (pathlib.Path(__file__).resolve().parent / "dir92" / "out" / "premeasure3.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
finally:
    st.remove()
