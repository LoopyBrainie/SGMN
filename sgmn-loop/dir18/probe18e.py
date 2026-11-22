#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-18 · probe18e —— 判定「训练收益 = 内容驱动」还是「稀有词先验驱动」。

【probe18d 留下的疑点】probe18d 用**两个极稀有 token**（池中频次最低）作对照，
  得 hebb_train: Δcode=−3.167 而 Δrare=−0.481 ⇒ 看似「内容驱动」。
  BUT 这两个对照 token 在 base 上 NLL 只有 12.97（比真码 14.17 **更低**），
  ⇒ 它们**与码不可交换**（base 的先验不同）⇒ Δ 的差可能只是「base 先验差异」，
  不是「内容被读出」。

【本探针的修正对照】从**与码同一个池、同一抽样分布**里取 K=32 个对照 token
  （码本身是 `rng.choice(code_pool)` 的**均匀抽样** ⇒ 对照也必须均匀抽样）：
    · 排除所有训练码/评测码/干扰码
    · 同一批对照 token 用于**所有臂**（配对）
  ⇒ Δcode 与 Δctrl 都是「同一批 token 的配对差」，先验差异被配对消掉。

【判读表】令 ΔX = NLL_base(X) − NLL_arm(X)（正 = 该目标被臂预测得更好）：
    · Δcode ≈ Δctrl ⇒ 收益 = **稀有词先验**（内容没被读出）⇒ DIR-18 主假说不成立
    · Δcode ≫ Δctrl ⇒ 收益**超出**先验 ⇒ 内容被读出（须再归因到 Hebbian 写步）

【附加】概率质量账：P(code) vs 「稀有池 4000 个 token 的均摊份额」P_pool/4000
    ⇒ 若 P(code) ≈ 均摊份额 ⇒ 该码只拿到了「池内平均待遇」⇒ 先验解释成立。

【额外】cos(y_read, v_true) 与 cos(W k_true, v_true) 在**真值槽**上重测。

用法：/opt/pytorch/bin/python -u probe18e.py
"""
import importlib.util
import json
import math
import pathlib
import random
import re as _re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

D = "/home/ec2-user/sgmn-loop/dir18"
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("r1w", D + "/R1_write.py")
r1w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1w)

OUT = pathlib.Path(D) / "probe18e_out.json"
rep = {"arms": {}, "meta": {}}


def log(*a):
    print(*a, flush=True)


def make_nowrite_variant(st):
    for m in st.mems.values():
        core = m.core

        def patched(k, v, sig=None, core=core):
            with torch.no_grad():
                i = core.count % core.N
                core.count += 1
                core.keys[i].copy_(k)
                core.vals[i].copy_(v)
                if sig is not None:
                    core.sigs[i].copy_(sig)
        core.write = patched


def eval_measure(st, ev, ctrl_ids, pool_set):
    """一次读前向上同时测：真码 / 干扰码 / K 个交换性对照 / 中频词 + 概率质量账。"""
    rows = []
    snap0 = st.snapshot_all()
    for d in ev:
        st.unfreeze_write()
        st.restore_all(snap0)
        r1w.write_both(st, d)
        vals = {}
        for m in st.mems.values():
            vals[id(m)] = torch.stack(m.vals_written) if m.vals_written else None
            m.vals_written.clear()
            m.keys_written.clear()
        st.freeze_write()
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=r1w.DEV).unsqueeze(0)
        with torch.no_grad():
            lg = st.model(input_ids=pre).logits.float()
        lp = F.log_softmax(lg[0, -1], dim=-1)
        p = lp.exp()
        code = int(d["code_ids"][0])
        dcode = int(d["dist_code"])
        row = {"nll_code": -float(lp[code]),
               "nll_dcode": -float(lp[dcode]),
               "nll_ctrl_mean": float(-lp[ctrl_ids].mean()),
               "nll_ctrl_med": float(-lp[ctrl_ids].median()),
               "p_code": float(p[code]),
               "p_pool_mass": float(p[list(pool_set)].sum())}
        # 真值槽上的写保真度与读出余弦
        fids, cys = [], []
        for li, m in st.mems.items():
            V = vals.get(id(m))
            if V is None or V.shape[0] == 0:
                continue
            ti = int(d["true_idx"])
            if hasattr(m.core, "vals"):
                cs = F.cosine_similarity(m.core.vals.detach().float(),
                                         V[ti].unsqueeze(0), dim=1)
                true_slot = int(torch.argmax(cs).item())
                if hasattr(m.core, "fidelity"):
                    fids.append(float(m.core.fidelity(true_slot)))
                yr = m.last_yr[0] if m.last_yr is not None else None
                if yr is not None:
                    cys.append(float(F.cosine_similarity(
                        V[ti].unsqueeze(0), yr.unsqueeze(0), dim=1).item()))
        row["fid_true_slot"] = float(np.mean(fids)) if fids else float("nan")
        row["cos_y_vtrue"] = float(np.mean(cys)) if cys else float("nan")
        row["p_pool_share"] = row["p_pool_mass"] / max(len(pool_set), 1)
        rows.append(row)
    st.unfreeze_write()
    return rows


def summarize(name, rows):
    s = {k: float(np.mean([r[k] for r in rows])) for k in rows[0].keys()}
    rep["arms"][name] = s
    log(f"[{name}] NLL: code={s['nll_code']:.4f} | dcode={s['nll_dcode']:.4f} | "
        f"ctrl(交换性对照)={s['nll_ctrl_mean']:.4f} | fid(真值槽)={s['fid_true_slot']:.4f} "
        f"| cos(y,v_true)={s['cos_y_vtrue']:.4f}")
    log(f"          P(code)={s['p_code']:.3e} | 池均摊份额={s['p_pool_share']:.3e} "
        f"| 比值={s['p_code']/max(s['p_pool_share'],1e-30):.2f}")
    return s


def main():
    tok = AutoTokenizer.from_pretrained(r1w.MODEL)
    model = AutoModelForCausalLM.from_pretrained(r1w.MODEL, dtype=torch.float32).to(r1w.DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size
    tr, ev = r1w.build_docs(tok, 20261028, 40, 12)

    # ---- 与 build_docs 同源地重建 pool，取 code_pool
    text = r1w.load_filler()
    paras = [q.strip() for q in _re.split(r"\n\s*\n", text) if len(q.strip()) > 120]
    pool = []
    for q in paras:
        pool.extend(tok(q).input_ids)
        if len(pool) > 160000:
            break
    code_pool = r1w.rare_word_tokens(tok, pool)

    used = set()
    for x in list(tr) + list(ev):
        used.add(int(x["code_ids"][0]))
        used.add(int(x["dist_code"]))
    # **交换性对照**：从 code_pool 均匀抽 K 个（码本身也是均匀抽 ⇒ 分布一致）
    rng = random.Random(777)
    cand = [int(t) for t in code_pool if int(t) not in used]
    rng.shuffle(cand)
    ctrl_ids = cand[:32]
    assert len(ctrl_ids) == 32, "对照不足"
    pool_set = set(int(t) for t in code_pool)
    log(f"code_pool={len(code_pool)} | 交换性对照 K={len(ctrl_ids)}（与码同池同分布）")

    def run(name, qm, mkind, eta, train, variant=None):
        st = r1w.make_stack(model, d, qm, seed=20261028, mem_kind=mkind, eta=eta)
        if variant == "nowrite":
            make_nowrite_variant(st)
        if train:
            r1w.train_arm(st, tr, qm)
        rows = eval_measure(st, ev, ctrl_ids, pool_set)
        st.remove()
        torch.cuda.empty_cache()
        return summarize(name, rows)

    log("=== 归因：内容 vs 稀有词先验（交换性对照） ===")
    run("hebb_eta7_cue_train", "cue", "hebbian_full", 1e-7, True)
    run("hebb_eta7_nowrite_train", "cue", "hebbian_full", 1e-7, True, variant="nowrite")
    run("verbatim_cue_train", "cue", "explicit_kv", None, True)
    run("verbatim_cue_notrain", "cue", "explicit_kv", None, False)

    # 基座
    st_rows = []
    for x in ev:
        pre = torch.tensor(x["prefix"], dtype=torch.long, device=r1w.DEV).unsqueeze(0)
        with torch.no_grad():
            lg = model(input_ids=pre).logits.float()
        lp = F.log_softmax(lg[0, -1], dim=-1)
        p = lp.exp()
        code = int(x["code_ids"][0])
        dcode = int(x["dist_code"])
        st_rows.append({"nll_code": -float(lp[code]), "nll_dcode": -float(lp[dcode]),
                        "nll_ctrl_mean": float(-lp[ctrl_ids].mean()),
                        "nll_ctrl_med": float(-lp[ctrl_ids].median()),
                        "p_code": float(p[code]),
                        "p_pool_mass": float(p[list(pool_set)].sum()),
                        "fid_true_slot": float("nan"), "cos_y_vtrue": float("nan"),
                        "p_pool_share": float(p[list(pool_set)].sum()) / max(len(pool_set), 1)})
    s = {k: float(np.mean([r[k] for r in st_rows])) for k in st_rows[0].keys()}
    rep["arms"]["base"] = s
    log(f"[base] NLL: code={s['nll_code']:.4f} | dcode={s['nll_dcode']:.4f} | "
        f"ctrl={s['nll_ctrl_mean']:.4f} | P(code)={s['p_code']:.3e}")

    # ---- 归因表
    log("\n=== 归因表（配对：同一批 token，Δ = base − arm） ===")
    b = rep["arms"]["base"]
    rep["deltas"] = {}
    for name, a in rep["arms"].items():
        if name == "base":
            continue
        dc = b["nll_code"] - a["nll_code"]
        dd = b["nll_dcode"] - a["nll_dcode"]
        dk = b["nll_ctrl_mean"] - a["nll_ctrl_mean"]
        rep["deltas"][name] = {"d_code": dc, "d_dcode": dd, "d_ctrl": dk,
                               "excess_over_prior": dc - dk,
                               "ratio_code_to_ctrl": a["nll_code"] / a["nll_ctrl_mean"]}
        log(f"  {name:24s} Δcode={dc:+7.3f}  Δdcode={dd:+7.3f}  Δctrl={dk:+7.3f}  "
            f"⇒ 超额(Δcode−Δctrl)={dc - dk:+7.3f}")

    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n写出 {OUT}")


if __name__ == "__main__":
    main()
