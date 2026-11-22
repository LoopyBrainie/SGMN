#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-91 rev 2.7 · **判决后的补充诊断**（非判据、不改任何冻结阈值）

正式轮判 `NO_HELP` 且 `D_HELPS=False`（机理被否证）。本文件回答"那约束在哪"：

(A) **关联式是否成立**：W 是写规则学到的关联记忆，理想上 `W k_i ≈ v_i`。
    算 rel(W k_i, v_i)、cos(W k_i, v_i)，**分训练键 / 留出键**。
    若训练键 rel 小而留出键 rel ≈1 ⇒ **关联不泛化**，且这不是"值不携带答案"
    （臂 D 的 M2loo=0.312 已证明值携带答案），而是**键不可分**导致 W 无法同时满足 32 条。
(B) **键几何**（复现 D-104 机制 1）：raw share 与 centered share。
(C) 对照：W^0（未写）的同一 rel —— 排除"rel 大是因为 v 本身尺度"。
"""
import json, os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here)); sys.path.insert(0, _here)
import numpy as np, torch
import probe91_womap as W91
import probe91_valread as V
from R2_task91 import DEV, MOUNT, N_FACTS, N_TRAIN_KEYS, build_chunks, build_facts, \
    make_stack, model_and_tok

R = {"kind": "post-verdict supplementary (non-gating)", "arms": {}}
TR = list(range(N_TRAIN_KEYS)); HE = list(range(N_TRAIN_KEYS, N_FACTS))


def shares(Km):
    """K: (32,d_k) -> (raw share 中位, centered share 中位)"""
    raw, cen = [], []
    Kc = Km - Km.mean(dim=0, keepdim=True)
    for i in range(Km.shape[0]):
        raw.append(float(Km[i] @ Km[i]) / float((Km * Km).sum()))
        cen.append(float(Kc[i] @ Kc[i]) / float((Kc * Kc).sum()))
    return float(np.median(raw)), float(np.median(cen))


def main():
    tok, model = model_and_tok()
    facts = build_facts(tok); chunks, owner = build_chunks(tok, facts)
    vpos = V.value_positions(tok, facts)
    vp = {fi: vpos[fi]["V"] for fi in range(N_FACTS)}
    seed = 20261028
    for arm in ("A_mean", "D_vtpos"):
        W, wc = V.write_only(arm, model, facts, chunks, owner, vp, seed)
        V.ARM["name"] = arm
        st = make_stack(gamma=0.0, no_write=True, seed=seed)
        try:
            K, Vv = V.key_matrices_arm(model, st, facts, chunks, owner, vp)
        finally:
            st.remove(); V.ARM["name"] = "A_mean"
        row = {}
        for li in MOUNT:
            Wl = W[li].float().double(); Kl = K[li].double(); Vl = Vv[li].double()
            W0l = (st0 := None)  # 占位
            y = (Wl @ Kl.T).T                      # (32, d_v) = W k_i
            rel = ((y - Vl).norm(dim=1) / Vl.norm(dim=1).clamp_min(1e-30))
            cos = torch.nn.functional.cosine_similarity(y, Vl, dim=1)
            rs, cs = shares(Kl)
            row[str(li)] = {
                "rel_train": float(rel[TR].median()), "rel_held": float(rel[HE].median()),
                "cos_train": float(cos[TR].median()), "cos_held": float(cos[HE].median()),
                "share_raw": rs, "share_centered": cs,
                "k_norm_med": float(Kl.norm(dim=1).median()),
                "v_norm_med": float(Vl.norm(dim=1).median())}
            print(f"  [{arm}] L{li}: rel(tr/ho)={row[str(li)]['rel_train']:.3f}/"
                  f"{row[str(li)]['rel_held']:.3f}  cos(tr/ho)="
                  f"{row[str(li)]['cos_train']:+.4f}/{row[str(li)]['cos_held']:+.4f}  "
                  f"share raw/centered={rs:.4f}/{cs:.4f}  |k|={row[str(li)]['k_norm_med']:.1f}",
                  flush=True)
        R["arms"][arm] = row
    # (C) 对照：W^0（无写）
    V.ARM["name"] = "A_mean"
    st = make_stack(gamma=0.0, no_write=True, seed=seed)
    try:
        K, Vv = V.key_matrices_arm(model, st, facts, chunks, owner, vp)
        W0, wc0 = V.write_only("A_mean", model, facts, chunks, owner, vp, seed)
    finally:
        st.remove()
    # 用 make_stack 未写时的 W0：直接取 st.mems 的 W0
    st2 = make_stack(gamma=0.0, no_write=True, seed=seed)
    try:
        ctrl = {}
        for li in MOUNT:
            W0l = st2.mems[li].W0.detach().double()
            Kl = K[li].double(); Vl = Vv[li].double()
            y = (W0l @ Kl.T).T
            rel = ((y - Vl).norm(dim=1) / Vl.norm(dim=1).clamp_min(1e-30))
            ctrl[str(li)] = {"rel_train": float(rel[TR].median()),
                             "rel_held": float(rel[HE].median())}
            print(f"  [W0 对照] L{li}: rel(tr/ho)={ctrl[str(li)]['rel_train']:.3f}/"
                  f"{ctrl[str(li)]['rel_held']:.3f}", flush=True)
    finally:
        st2.remove()
    R["control_W0"] = ctrl
    with open(os.path.join(V.OUT, "probe91_assoc.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    print("\n★ 完成 ⇒ out/probe91_assoc.json", flush=True)


if __name__ == "__main__":
    main()
