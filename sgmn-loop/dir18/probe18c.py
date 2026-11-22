#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-18 · probe18c —— 一次性判定三件事（全部取自装置的实际输出路径）：

  Q1【D46 口径】`addr` 指标恒 0 是**指标缺陷**还是**机制失败**？
      探针：同一批评测文档上，同时打印
            · `last_cue_idx`（**核槽位空间**的索引）
            · `true_idx`（**写入顺序空间**的索引）
            · 真值条目实际所在的**槽位**（由值内容反查）
            · 核心 `count`（写前/写后）与**陈旧条目数**
      ⇒ 若「匹配槽 == 真值槽」成立而「匹配槽 != true_idx」⇒ 纯口径错位。

  Q2【陈旧槽位污染】DIR-17 协议把核状态从**训练后**快照恢复 ⇒ 评测时核里残留训练条目，
      且 68+68 > N_ENTRIES(128) ⇒ **槽位回绕**。探针：打印残留条目数、地址表中
      真值槽的排名、以及「地址选中陈旧槽」的比例。

  Q3【D47 P_v 梯度】`W4_hebb_full_trainPv` 与主臂**逐位相同** ⇒ P_v 是否真的拿到梯度？

  Q4【收益归因】主臂相对 base 的 +2.97 nats 是**内容驱动**还是**读接口泛化**？
      探针（全部训练后再评测，NLL 口径）：
        A  主臂（cue + Hebbian 写）              —— 复现 smoke 的 +2.97
        B  内容对照：misaddress（读邻位键）
        C  只读不写：写地址表但不更新 W（⇒ 地址正确、**无内容关联**）
        D  逐字上界（verbatim）
        E  基座
      ⇒ 若 A≈B≈C ⇒ 收益**不是内容驱动**；若 A≫B 且 A≫C ⇒ 内容确实被用上。
      并测量 cos(y_read, v_true) 与 cos(W k_true, v_true)【真值槽】。

用法：/opt/pytorch/bin/python -u probe18c.py
"""
import importlib.util
import json
import math
import pathlib
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

OUT = pathlib.Path(D) / "probe18c_out.json"
rep = {"arms": {}, "notes": []}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 装置变体
def make_nowrite_variant(st):
    """只读不写：地址表照常写（keys/sigs），但**不更新 W** ⇒ 地址正确、内容无关联。"""
    for m in st.mems.values():
        core = m.core

        def patched(k, v, sig=None, core=core, orig=core.write):
            with torch.no_grad():
                i = core.count % core.N
                core.count += 1
                core.keys[i].copy_(k)
                core.vals[i].copy_(v)
                if sig is not None:
                    core.sigs[i].copy_(sig)
            return None
        core.write = patched


def make_shuffle_variant(st, shift=11):
    """写错配：键不变，值取自别的位置 ⇒ 键-值关联被打乱（内容对照的极端形式）。"""
    for m in st.mems.values():
        core = m.core
        orig = core.write
        state = {"buf": []}

        def patched(k, v, sig=None, core=core, orig=orig, state=state):
            state["buf"].append((k, v, sig))
            if len(state["buf"]) <= shift:
                # 未满 shift 时先按原样写
                return orig(k, v, sig)
            k2, v2, sig2 = state["buf"][-shift]
            return orig(k2, v2, sig2)
        core.write = patched


# ---------------------------------------------------------------- 评测
def eval_probe(st, ev, tag):
    """复刻 DIR-18 eval_arm，但**额外**记录槽位空间信息。"""
    rows = []
    snap0 = st.snapshot_all()
    base_count = {li: int(m.core.count) for li, m in st.mems.items()}
    for d in ev:
        st.unfreeze_write()
        st.restore_all(snap0)
        cnt_before = {li: int(m.core.count) for li, m in st.mems.items()}
        r1w.write_both(st, d)
        cnt_after = {li: int(m.core.count) for li, m in st.mems.items()}
        vals = {}
        for li, m in st.mems.items():
            vals[li] = torch.stack(m.vals_written) if m.vals_written else None
            m.vals_written.clear()
            m.keys_written.clear()
        st.freeze_write()
        pre = torch.tensor(d["prefix"], dtype=torch.long, device=r1w.DEV).unsqueeze(0)
        with torch.no_grad():
            lg = st.model(input_ids=pre).logits.float()
        lp = float(F.log_softmax(lg[0, -1], dim=-1)[d["code_ids"][0]])
        info = {"lp": lp, "nll": -lp, "true_idx": int(d["true_idx"]), "layers": {}}
        for li, m in st.mems.items():
            V = vals[li]
            j_sel = m.last_cue_idx
            # 真值槽（由**值内容**反查：core.vals 里与 V[true_idx] 相同的那一槽）
            true_slot = None
            if V is not None and V.shape[0] > 0 and hasattr(m.core, "vals"):
                cs = F.cosine_similarity(m.core.vals.detach().float(),
                                         V[int(d["true_idx"])].unsqueeze(0), dim=1)
                true_slot = int(torch.argmax(cs).item())
            # 真值槽在地址表（sigs）中的排名
            rank = None
            n_searched = None
            if hasattr(m.core, "sigs") and j_sel is not None:
                n = min(int(m.core.count), int(m.core.N))
                n_searched = n
                q = m.last_cue_scores
                if q is not None and true_slot is not None and true_slot < len(q):
                    rank = int((q > q[true_slot]).sum().item())
            yr = m.last_yr[0] if m.last_yr is not None else None
            cos_y_vtrue = float("nan")
            if yr is not None and V is not None and V.shape[0] > 0:
                cos_y_vtrue = float(F.cosine_similarity(
                    V[int(d["true_idx"])].unsqueeze(0), yr.unsqueeze(0), dim=1).item())
            fid_true = float("nan")
            if hasattr(m.core, "fidelity") and true_slot is not None:
                fid_true = float(m.core.fidelity(true_slot))
            fid_sel = float("nan")
            if hasattr(m.core, "fidelity") and j_sel is not None:
                fid_sel = float(m.core.fidelity(int(j_sel)))
            info["layers"][li] = {
                "j_sel": (None if j_sel is None else int(j_sel)),
                "true_idx": int(d["true_idx"]),
                "true_slot": true_slot,
                "true_slot_eq_writeorder": (true_slot == int(d["true_idx"])),
                "j_sel_eq_true_slot": (j_sel is not None and true_slot is not None
                                       and int(j_sel) == true_slot),
                "j_sel_eq_true_idx": (j_sel is not None and int(j_sel) == int(d["true_idx"])),
                "rank_true_slot": rank,
                "n_searched": n_searched,
                "n_stale": (n_searched - V.shape[0]) if (n_searched is not None and V is not None) else None,
                "cos_y_vtrue": cos_y_vtrue,
                "fid_true_slot": fid_true,
                "fid_selected": fid_sel,
                "cnt_before": cnt_before[li], "cnt_after": cnt_after[li],
            }
        rows.append(info)
    st.unfreeze_write()
    return rows, base_count


def agg(rows, key):
    out = []
    for r in rows:
        for li, L in r["layers"].items():
            out.append(L[key])
    return out


def summarize(name, rows):
    def m(key):
        v = [x for x in agg(rows, key) if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(v)) if v else float("nan")
    s = {
        "nll": float(np.mean([r["nll"] for r in rows])),
        "addr_writeorder": m("j_sel_eq_true_idx"),
        "addr_slotspace": m("j_sel_eq_true_slot"),
        "true_slot_eq_writeorder": m("true_slot_eq_writeorder"),
        "rank_true_slot_mean": m("rank_true_slot"),
        "n_searched_mean": m("n_searched"),
        "n_stale_mean": m("n_stale"),
        "cos_y_vtrue": m("cos_y_vtrue"),
        "fid_true_slot": m("fid_true_slot"),
        "fid_selected": m("fid_selected"),
    }
    rep["arms"][name] = s
    log(f"[{name}] nll={s['nll']:.4f} | addr(写入序)={s['addr_writeorder']:.4f} "
        f"addr(槽空间)={s['addr_slotspace']:.4f} | 真值槽==true_idx: {s['true_slot_eq_writeorder']:.3f} "
        f"| 真值槽排名={s['rank_true_slot_mean']:.2f} | 搜索槽数={s['n_searched_mean']:.1f} "
        f"陈旧={s['n_stale_mean']:.1f} | cos(y,v_true)={s['cos_y_vtrue']:.4f} "
        f"| fid(真值槽)={s['fid_true_slot']:.4f} fid(选中槽)={s['fid_selected']:.4f}")
    return s


def main():
    tok = AutoTokenizer.from_pretrained(r1w.MODEL)
    model = AutoModelForCausalLM.from_pretrained(r1w.MODEL, dtype=torch.float32).to(r1w.DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size
    tr, ev = r1w.build_docs(tok, 20261028, 40, 12)

    def run(name, qm, mkind, eta, train, variant=None):
        st = r1w.make_stack(model, d, qm, seed=20261028, mem_kind=mkind, eta=eta)
        if variant == "nowrite":
            make_nowrite_variant(st)
        elif variant == "shuffle":
            make_shuffle_variant(st)
        if train:
            r1w.train_arm(st, tr, qm)
        rows, _ = eval_probe(st, ev, name)
        st.remove()
        torch.cuda.empty_cache()
        return summarize(name, rows)

    log("=== Q1/Q2：口径与槽位空间 ===")
    run("hebb_eta7_cue_train", "cue", "hebbian_full", 1e-7, True)
    run("hebb_eta7_cue_notrain", "cue", "hebbian_full", 1e-7, False)
    run("verbatim_cue_train", "cue", "explicit_kv", None, True)
    run("verbatim_cue_notrain", "cue", "explicit_kv", None, False)

    log("\n=== Q4：收益归因（内容对照） ===")
    run("hebb_eta7_misaddr_train", "misaddress", "hebbian_full", 1e-7, True)
    run("hebb_eta7_nowrite_train", "cue", "hebbian_full", 1e-7, True, variant="nowrite")
    run("hebb_eta7_shuffle_train", "cue", "hebbian_full", 1e-7, True, variant="shuffle")

    # 基座
    rows_b = r1w.eval_base(model, ev)
    rep["arms"]["base"] = {"nll": float(np.mean([r["nll"] for r in rows_b]))}
    log(f"[base] nll={rep['arms']['base']['nll']:.4f}")

    log("\n=== Q3：P_v 是否拿到梯度（D47） ===")
    st = r1w.make_stack(model, d, "cue", seed=20261028, mem_kind="hebbian_full", eta=1e-7)
    traj, repp = r1w.train_arm(st, tr, "cue", train_pv=True)
    pv_info = []
    for li, m in st.mems.items():
        g = None if m.Pv.grad is None else float(m.Pv.grad.norm())
        pv_info.append({"layer": li, "pv_grad": g, "pv_requires_grad": bool(m.Pv.requires_grad)})
        log(f"  层 {li}: ‖∂L/∂P_v‖ = {g}（None ⇒ 从未参与反向）")
    rep["pv_grad"] = pv_info
    st.remove()
    torch.cuda.empty_cache()

    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n写出 {OUT}")


if __name__ == "__main__":
    main()
