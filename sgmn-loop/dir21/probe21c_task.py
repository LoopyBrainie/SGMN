#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-21 · **任务层归因探针**（诊断用；不是科学结论）—— 缺陷 D119 的验证

------------------------------------------------------------------
**假设**：DIR-21 的**训练栈**用 η=ETA(=1e-5)，而五个臂用 η=ETA_BASE(=1e-1)
（且 `_set_arm` 在**训练之后**才改 η）⇒ ⇒ 读出映射 (W_o, γ) 是在
「内容**几乎没写进去**的读出」上训练的 ⇒ ⇒ 评测时（内容以 η=1e-1 好好写入）
读出映射**学不到「用记忆里的内容」** ⇒ ⇒ 任务层 NLL 停在基座水平
⇒ ⇒ **H21-1 的臂间差被这一缺陷抹平** ⇒ ⇒ H21-1 的读数**不是**科学结果。

**DIR-20 的对照**：X1 臂的 `make_stack(..., eta=1e-1, ...)` ⇒ 训练与评测**同一 η**
⇒ ⇒ 跨会话增益 +5.51 nats ⇒ ⇒ 两轮的唯一差别就在这里。

------------------------------------------------------------------
**本探针只做一件事**：对 `η_train ∈ {1e-5, 1e-1}` 各跑一次
「训练 200 篇 → 快照 → 清槽 → 写文档 1 → 只读读出」，打印
训练损失首末、基座 NLL、带记忆的 NLL、hit、保真 ⇒ ⇒ **一次跑完即判定 D119**。
"""

import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import R1_granularity as RG            # noqa: E402

SEED = 20261028
N_TRAIN = 200
N_EVAL = 260
ETAS = [1e-5, 1e-1]


def run(eta_train, tok, model, d, tr, ev, ctrl_ids):
    st = RG.make_stack(model, d, "cue", seed=SEED, mem_kind="hebbian_full",
                       eta=eta_train, key_mode="ortho", variant="normal")
    traj, rep = RG.train_arm(st, tr, "cue", variant="normal")
    loss_first = float(traj[0][0]) if traj else float("nan")
    loss_last = float(traj[-1][0]) if traj else float("nan")
    snap = st.snapshot_all()
    # —— 评测：DIR-21 rev 1.7 的扫描路径 ——
    st.restore_all(snap)
    RG.clear_slots(st)
    RG.clear_write_records(st)
    for _m in st.mems.values():          # ★ 与 runner 的 `_set_arm` 逐字同构
        _m.eta = float(RG.ETA_BASE)
        _m.core.eta = float(RG.ETA_BASE)
        _m.write_stride = 1
        _m.stride_pick = "last"
    st.unfreeze_write()
    RG.write_both(st, ev[0])
    L = len(ev[0]["ids"])
    trig = list(st.mems[list(st.mems)[0]].trigger_positions(L))
    kv = RG.capture_doc1_entries(st, len(trig))
    code_i = len(trig) - 1
    spec = {"prefix": list(ev[0]["ids"][:L - 1]),
            "code_ids": [int(ev[0]["ids"][L - 1])],
            "true_idx": int(code_i), "true_slot": int(code_i),
            "dist_code": ev[0].get("dist_code", "0")}
    r = RG.eval_arm_readonly(st, spec, ctrl_ids)
    cos = RG.cos_orig(st, [kv[_l][code_i] for _l in range(len(kv))])
    # —— 基座（无记忆）读同一篇 ——
    st.freeze_write()
    pre = torch.tensor(spec["prefix"], dtype=torch.long, device=RG.DEV).unsqueeze(0)
    with torch.no_grad():
        lg = model(input_ids=pre).logits.float()
    lp_all = torch.log_softmax(lg[0, -1], dim=-1)
    nll_base = float(-lp_all[spec["code_ids"][0]])
    out = {"eta_train": eta_train, "loss_first": loss_first, "loss_last": loss_last,
           "nll_base": nll_base, "nll_mem": float(r["nll"]), "hit": float(r["hit"]),
           "addr_slot": float(r["addr_slot"]), "cos_orig": cos,
           "gain": nll_base - float(r["nll"]),
           "gamma": [float(m.gamma) for m in st.mems.values()]}
    st.remove()
    torch.cuda.empty_cache()
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("=== DIR-21 任务层归因探针（D119 · 诊断用 · 非科学结论）===", flush=True)
    tok = AutoTokenizer.from_pretrained(RG.MODEL)
    model = AutoModelForCausalLM.from_pretrained(RG.MODEL, dtype=torch.float32).to(RG.DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size
    tr, ev = RG.build_docs(tok, SEED, N_TRAIN, N_EVAL)
    ctrl_ids = RG.build_ctrl_ids(tok, SEED, tr, ev)
    rows = []
    for eta in ETAS:
        r = run(eta, tok, model, d, tr, ev, ctrl_ids)
        rows.append(r)
        print(f"\n--- η_train = {eta:.6g} ---", flush=True)
        print(f"    训练损失 首/末 = {r['loss_first']:.4f} / {r['loss_last']:.4f}"
              f"（⇒ 训练时内容写进去的强度）", flush=True)
        print(f"    基座 NLL（无记忆）= {r['nll_base']:.4f}", flush=True)
        print(f"    带记忆 NLL = {r['nll_mem']:.4f} ⇒ 增益 = {r['gain']:+.4f} nats", flush=True)
        print(f"    hit={r['hit']:.4f}  addr_slot={r['addr_slot']:.4f}  cos_orig={r['cos_orig']:.4f}"
              f"  gamma={[round(g, 5) for g in r['gamma']]}", flush=True)
    d119 = rows[1]["gain"] - rows[0]["gain"]
    print(f"\n=== D119 判定：η_train 1e-5 → 1e-1 的任务层增益差 = {d119:+.4f} nats"
          f" ⇒ {'D119 成立（训练侧 η 必须 = 臂侧 η）' if d119 > 0.5 else 'D119 不成立'} ===",
          flush=True)


if __name__ == "__main__":
    main()
