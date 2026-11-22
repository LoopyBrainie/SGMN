#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe18h · DIR-18 · **协议等价性探针**（纪律 ⑯ / ⑲ / ⑳）

**要回答的唯一问题**：DIR-18 的 `W0_verbatim`（= DIR-17 的 `E1_cue_train` 同构臂）
在本轮的**评测协议**下 `addr`/`hit` = 0.6389（seed 20261028, 40 训练篇, 12 评测篇），
而 DIR-17 的正式轮 `E1_cue_train` `hit` = **0.8722**（200 训练篇, 60 评测篇, 3 种子）。
⇒ 差异来自**哪一个变量**？

**四个候选变量**（本探针逐一分离，**每个格点只改一个变量**）：

| 变量 | 取值 |
|---|---|
| (V1) 评测协议 | `clean`（本轮 D48：每篇评测前**清空槽表**，保留 W）<br>`dirty17`（DIR-17 原协议：评测前**恢复训练后快照**⇒ 残留 68 条训练条目 + 128 槽回绕） |
| (V2) 训练篇数 | 40 / 200 |
| (V3) 评测篇数 | 12 / 60 |
| (V4) 种子 | 固定 20261028 |

**设计**：训练**只做一次**（按 n_train），然后在**同一批训练好的核状态**上跑
`{clean, dirty17} × {12 篇, 60 篇}` ⇒ V1 与 V3 的效应**在同一核状态**上比较
⇒ 不受训练随机性干扰（这正是纪律 ⑯ 的要求：跨方向复用协议时先做装置逐位对照探针）。

**探针性质**：**探索性**（纪律 ⑧）—— 只用于定位协议差异，**不进入判据**。

输出：`probe18h_out.json`
"""
import os, sys, json, math
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "rounds"))

from sgmn_module_hebb import SGMNStack as SGS        # noqa: E402
from R2_write import (build_docs, build_ctrl_ids, MOUNT, MODEL, R, ALPHA, LAM, GAMMA0,  # noqa: E402
                       DEV, N_ENTRIES, make_stack, clear_slots, write_both, train_arm)

SEED = 20261028


# ---------------------------------------------------------------- 两种评测协议
def eval_clean(st, eval_docs, ctrl_ids=None):
    """**DIR-18 rev 1.2（D48）**：每篇评测前清空槽表（保留 W）⇒ 无陈旧、无回绕。"""
    rows = []
    snap0 = st.snapshot_all()
    for d in eval_docs:
        st.unfreeze_write()
        st.restore_all(snap0)
        clear_slots(st)
        write_both(st, d)
        vals = {}
        for li, m in st.mems.items():
            vals[li] = torch.stack(m.vals_written) if m.vals_written else None
            m.vals_written.clear(); m.keys_written.clear()
        st.freeze_write()
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            lg = st.model(input_ids=pre).logits.float()
        lp_all = F.log_softmax(lg[0, -1], dim=-1)
        hits, addr, n_stale = [], [], []
        for li, m in st.mems.items():
            yr = m.last_yr[0] if m.last_yr is not None else None
            V = vals[li]
            j_sel = m.last_cue_idx
            addr.append(1.0 if (j_sel is not None and int(j_sel) == int(d["true_idx"])) else 0.0)
            n = min(int(m.core.count), int(m.core.N))
            n_stale.append(int(n) - (0 if V is None else int(V.shape[0])))
            if yr is None or V is None or V.shape[0] == 0:
                hits.append(0.0); continue
            cs = F.cosine_similarity(V, yr.unsqueeze(0), dim=1)
            hits.append(1.0 if int(torch.argmax(cs)) == int(d["true_idx"]) else 0.0)
        r = {"lp": float(lp_all[d["code_ids"][0]]), "nll": -float(lp_all[d["code_ids"][0]]),
             "hit": float(np.mean(hits)), "addr": float(np.mean(addr)),
             "n_stale": int(max(n_stale)), "n_entries": int(vals[list(vals)[0]].shape[0])
             if vals and vals[list(vals)[0]] is not None else 0}
        if ctrl_ids is not None:
            r["nll_ctrl"] = float(-lp_all[ctrl_ids].mean())
        rows.append(r)
    st.unfreeze_write()
    return rows


def eval_dirty17(st, eval_docs, ctrl_ids=None):
    """**DIR-17 原协议**：评测前**恢复训练后快照**（⇒ 核内残留训练条目）⇒ 128 槽回绕。"""
    rows = []
    snap0 = st.snapshot_all()
    for d in eval_docs:
        st.unfreeze_write()
        st.restore_all(snap0)
        write_both(st, d)
        vals = {}
        for li, m in st.mems.items():
            vals[li] = torch.stack(m.vals_written) if m.vals_written else None
            m.vals_written.clear(); m.keys_written.clear()
        st.freeze_write()
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=DEV).unsqueeze(0)
        with torch.no_grad():
            lg = st.model(input_ids=pre).logits.float()
        lp_all = F.log_softmax(lg[0, -1], dim=-1)
        hits, addr, n_stale = [], [], []
        for li, m in st.mems.items():
            yr = m.last_yr[0] if m.last_yr is not None else None
            V = vals[li]
            j_sel = m.last_cue_idx
            # 写入序口径：DIR-17 的 `hit` 是**读出值口径**，此处同时给两个口径
            addr.append(1.0 if (j_sel is not None and int(j_sel) == int(d["true_idx"])) else 0.0)
            n = min(int(m.core.count), int(m.core.N))
            n_stale.append(int(n) - (0 if V is None else int(V.shape[0])))
            if yr is None or V is None or V.shape[0] == 0:
                hits.append(0.0); continue
            cs = F.cosine_similarity(V, yr.unsqueeze(0), dim=1)
            hits.append(1.0 if int(torch.argmax(cs)) == int(d["true_idx"]) else 0.0)
        r = {"lp": float(lp_all[d["code_ids"][0]]), "nll": -float(lp_all[d["code_ids"][0]]),
             "hit": float(np.mean(hits)), "addr": float(np.mean(addr)),
             "n_stale": int(max(n_stale)), "n_entries": int(vals[list(vals)[0]].shape[0])
             if vals and vals[list(vals)[0]] is not None else 0}
        if ctrl_ids is not None:
            r["nll_ctrl"] = float(-lp_all[ctrl_ids].mean())
        rows.append(r)
    st.unfreeze_write()
    return rows


def main():
    out = {"probe": "probe18h", "kind": "exploratory", "seed": SEED, "cells": []}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    d = model.config.hidden_size

    # ★ 评测集**固定**：一次构造 n_train=200 / n_eval=60 的文档池
    #   ⇒ 两种协议、两种评测篇数、两种训练篇数**共用同一批评测文档**
    #   ⇒ V1（协议）与 V2（训练篇数）的比较**不含评测集差异**这个混淆。
    tr_all, ev_all = build_docs(tok, SEED, 200, 60)
    for n_train in (40, 200):
        tr = tr_all[:n_train]
        st = make_stack(model, d, "cue", mem_kind="explicit_kv")
        train_arm(st, tr, "cue")
        ctrl = build_ctrl_ids(tok, SEED, tr_all, ev_all)
        # 同一核状态上跑 2 协议 × 2 评测篇数
        for proto, fn in (("clean", eval_clean), ("dirty17", eval_dirty17)):
            for n_eval in (12, 60):
                rows = fn(st, ev_all[:n_eval], ctrl_ids=ctrl)
                cell = {"n_train": n_train, "protocol": proto, "n_eval": n_eval,
                        "hit": float(np.mean([r["hit"] for r in rows])),
                        "addr": float(np.mean([r["addr"] for r in rows])),
                        "nll": float(np.mean([r["nll"] for r in rows])),
                        "nll_ctrl": float(np.mean([r["nll_ctrl"] for r in rows])),
                        "n_stale": int(max(r["n_stale"] for r in rows)),
                        "n_entries": int(rows[0]["n_entries"]),
                        "n_docs": len(rows)}
                out["cells"].append(cell)
                print(f"[probe18h] n_train={n_train:3d} {proto:8s} n_eval={n_eval:2d} "
                      f"hit={cell['hit']:.4f} addr={cell['addr']:.4f} nll={cell['nll']:.4f} "
                      f"nll_ctrl={cell['nll_ctrl']:.4f} n_stale={cell['n_stale']} "
                      f"n_entries={cell['n_entries']}", flush=True)
        st.remove()
        torch.cuda.empty_cache()

    with open(os.path.join(HERE, "probe18h_out.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[probe18h] done ->", os.path.join(HERE, "probe18h_out.json"))


if __name__ == "__main__":
    main()
