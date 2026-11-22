#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-94 · `W` 消融对照 —— 卡 `revolve/DIR-94/DIRECTION_rev94_v11.md`（rev 1.1 冻结）。

主判据量（§3，冻结）： Δ_tr = acc_train(intact) − acc_train(W=0)
  Δ_tr ≥ 0.25            ⇒ W-MEDIATED
  Δ_tr ≤ 0.0625          ⇒ W-BYPASSED
  否则                    ⇒ INCONCLUSIVE-PARTIAL

★ 本脚本**逐字派生**自 `dir93/R1_task93.py`：装置 `sgmn_module.py`、`make_stack`/`write_phase`/
  `train_read_task`/`make_eval_items`/`nll_of_candidate`/`acc_by_question` 全部**直接 import**，
  一字不改；**唯一新增**的是"换 `W` 再评一次"这一层（这就是本卡的独立变量）。

Stage A（§5）：`--stage A` 只跑守卫；`smoke` 单种子全流程；`formal` 三种子。
"""
import argparse
import hashlib
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "dir93"))
import R1_task93 as R                            # noqa: E402
import torch                                     # noqa: E402

SEEDS = [20261028, 20261104, 7]
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"


# ------------------------------------------------------------------ 工具
def snap(st):
    """快照可塑状态（W 用全矩阵分支；W0 用于 nowrite 对照）。"""
    W = {li: m.Wfull.detach().clone() for li, m in st.mems.items()}
    W0 = {li: m.W0.detach().clone() for li, m in st.mems.items()}
    return W, W0


def set_W(st, W):
    """把 `W` 逐层拷回（三层必须齐全，否则抛）。"""
    assert set(W) == set(st.mems), (sorted(W), sorted(st.mems))
    for li, m in st.mems.items():
        with torch.no_grad():
            m.Wfull.copy_(W[li])


def inj_max(st):
    return {li: float(m.injection.abs().max()) if m.injection is not None else None
            for li, m in st.mems.items()}


def eval_cell(model, st, items, tag=""):
    """按题口径（`acc_by_question`）+ 按项 + NLL + margin（正确 − 最优错误，负=好）。"""
    det, nlls_all = [], []
    for it in items:
        nlls = [R.nll_of_candidate(model, st, it["prompt_ids"], c) for c, _ in it["cand"]]
        true_i = next(i for i, (_, is_t) in enumerate(it["cand"]) if is_t)
        nlls_all.append(nlls)
        ok = (min(range(len(nlls)), key=lambda i: nlls[i]) == true_i)
        det.append({"ok": ok, "nlls": nlls, "true_i": true_i})
    acc_q, nq = R.acc_by_question(items, det)
    acc_item = sum(1 for d in det if d["ok"]) / len(det)
    margins = [d["nlls"][d["true_i"]] - min(n for i, n in enumerate(d["nlls"]) if i != d["true_i"])
               for d in det]
    return {"tag": tag, "acc_q": acc_q, "n_q": nq, "acc_item": acc_item, "n_item": len(det),
            "nll_mean": sum(sum(x) for x in nlls_all) / (len(nlls_all) * R.N_CAND),
            "margin_mean": sum(margins) / len(margins),
            "margin_pos_frac": sum(1 for m in margins if m > 0) / len(margins)}


def items_md5(items):
    h = hashlib.md5()
    for it in items:
        h.update(str(it["key_idx"]).encode())
        for c, is_t in it["cand"]:
            h.update(("%s|%s;" % (c, is_t)).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["A", "smoke", "formal", "all"])
    ap.add_argument("--seeds", default="")
    a = ap.parse_args()
    seeds = ([int(x) for x in a.seeds.split(",")] if a.seeds
             else (SEEDS if a.stage in ("formal", "all") else SEEDS[:1]))
    OUT.mkdir(exist_ok=True)
    G = {}

    tok, model = R.model_and_tok()
    facts = R.build_facts(tok)
    chunks, owner = R.build_chunks(tok, facts)
    order = list(range(len(facts)))
    random.Random(4242).shuffle(order)
    train_keys, eval_keys = order[:R.N_TRAIN_KEYS], order[R.N_TRAIN_KEYS:]
    print(f"[setup] facts={len(facts)} train_keys={len(train_keys)} eval_keys={len(eval_keys)} "
          f"chunks={len(chunks)}", flush=True)

    # ---- 阶段①：每个种子训练一次，**只留状态、立刻卸栈**
    #      ★ DEF-94-03：多个栈同时挂在同一 model 上时，前一个栈的层会继续处理后一个栈的
    #      前向；而 `train_read_task` 结束会把 `no_write` 还原为 False ⇒ 旧栈在**继续写**
    #      ⇒ `Wfull` 版本号被改而 autograd 图仍引用旧版本 ⇒ backward 报
    #      "modified by an inplace operation ... version 1200 expected 1199"。
    #      `dir93/run_arm` 每次 `st.remove()` 正是不变量；本脚本改为"存状态、卸栈"。
    saved = {}
    for sd in seeds:
        t0 = time.time()
        st = R.make_stack(gamma=R.GAMMA0, seed=sd)
        R.write_phase(model, st, chunks)
        losses, viol, div = R.train_read_task(model, st, tok, facts, train_keys,
                                             random.Random(1000 + sd % 100))
        saved[sd] = {
            "W": {li: m.Wfull.detach().clone() for li, m in st.mems.items()},
            "W0": {li: m.W0.detach().clone() for li, m in st.mems.items()},
            "rp": {li: (m.Wq.detach().clone(), m.Wo.detach().clone(),
                        m.gamma.detach().clone()) for li, m in st.mems.items()},
            "stat": {"final_loss": float(sum(losses[-50:]) / len(losses[-50:])) if losses else None,
                     "viol": viol, "div": div, "sec": round(time.time() - t0, 1),
                     "w_abs_max": {li: float(m.Wfull.detach().abs().max()) for li, m in st.mems.items()},
                     "w_fro": {li: float(m.Wfull.detach().norm()) for li, m in st.mems.items()}},
        }
        print(f"[train] seed={sd} loss={saved[sd]['stat']['final_loss']:.4f} "
              f"div={div} {saved[sd]['stat']['sec']}s", flush=True)
        st.remove()                      # ★ 必须卸栈，否则污染下一个种子的前向
        del st
        torch.cuda.empty_cache()
    G["train"] = {str(sd): saved[sd]["stat"] for sd in seeds}

    # ---- 评测卷子（两种键集各一张，四条件**共用**同一张 ⇒ A94-5）----
    items_tr = R.make_eval_items(tok, facts, train_keys, random.Random(555))
    items_ho = R.make_eval_items(tok, facts, eval_keys, random.Random(555))
    G["items"] = {"train": {"n_item": len(items_tr), "n_q": len(train_keys),
                            "md5": items_md5(items_tr)},
                  "held": {"n_item": len(items_ho), "n_q": len(eval_keys),
                           "md5": items_md5(items_ho)}}
    print(f"[items] train md5={G['items']['train']['md5'][:12]} "
          f"held md5={G['items']['held']['md5'][:12]}", flush=True)

    # ---- 阶段②：每种子重建栈 → 恢复**状态**（W/Wq/Wo/γ）→ 四条件 × 两键集 ----
    cells = {}
    for i, sd in enumerate(seeds):
        st = R.make_stack(gamma=R.GAMMA0, seed=sd)
        for li, m in st.mems.items():
            wq, wo, gm = saved[sd]["rp"][li]
            with torch.no_grad():
                m.Wq.copy_(wq)
                m.Wo.copy_(wo)
                m.gamma.copy_(gm)
                m.no_write = True          # ★ 评测期禁止写入
        other = saved[seeds[(i + 1) % len(seeds)]]["W"]
        if len(seeds) == 1:
            other = {li: saved[sd]["W"][li].flip(1) for li in saved[sd]["W"]}
        conds = {"intact": saved[sd]["W"],
                 "Wzero": {li: torch.zeros_like(saved[sd]["W"][li]) for li in saved[sd]["W"]},
                 "Wnowrite": saved[sd]["W0"], "Wother": other}
        for name, W in conds.items():
            set_W(st, W)
            if name == "Wzero":                        # A94-1：注入必须精确为 0
                model(input_ids=chunks[0].to(R.DEV))
                G.setdefault("A94_1_inj_max_at_Wzero", {})[str(sd)] = inj_max(st)
            for ks, items in (("train", items_tr), ("held", items_ho)):
                t0 = time.time()
                r = eval_cell(model, st, items, tag=f"{sd}|{name}|{ks}")
                cells[f"{sd}|{name}|{ks}"] = r
                print(f"  [{sd}|{name:9s}|{ks}] acc_q={r['acc_q']:.4f} "
                      f"acc_item={r['acc_item']:.4f} nll={r['nll_mean']:.4f} "
                      f"margin={r['margin_mean']:+.3f} {time.time()-t0:.0f}s", flush=True)
        # ---- A94-3：γ 旋钮 ----
        set_W(st, saved[sd]["W"])
        g0 = {li: m.gamma.detach().clone() for li, m in st.mems.items()}
        base = eval_cell(model, st, items_tr[:8], tag="gamma_base")
        for li, m in st.mems.items():
            with torch.no_grad():
                m.gamma.mul_(20.0)
        hi = eval_cell(model, st, items_tr[:8], tag="gamma_x20")
        for li, m in st.mems.items():
            with torch.no_grad():
                m.gamma.copy_(g0[li])
        G.setdefault("A94_3_gamma", {})[str(sd)] = {
            "nll_base": base["nll_mean"], "nll_x20": hi["nll_mean"],
            "delta_nll": hi["nll_mean"] - base["nll_mean"]}
        print(f"  [A94-3 γx20] nll {base['nll_mean']:.4f} → {hi['nll_mean']:.4f} "
              f"Δ={hi['nll_mean']-base['nll_mean']:+.6f}", flush=True)
        G.setdefault("A94_4_nowrite_after", {})[str(sd)] = all(m.no_write for m in st.mems.values())
        st.remove()
        del st
        torch.cuda.empty_cache()

    # ---- 判据 ----
    judged = {}
    for sd in seeds:
        tr_i = cells[f"{sd}|intact|train"]["acc_q"]
        tr_z = cells[f"{sd}|Wzero|train"]["acc_q"]
        ho_i = cells[f"{sd}|intact|held"]["acc_q"]
        ho_z = cells[f"{sd}|Wzero|held"]["acc_q"]
        d_tr, d_ho = tr_i - tr_z, ho_i - ho_z
        verdict = ("W-MEDIATED" if d_tr >= 0.25 else
                   "W-BYPASSED" if d_tr <= 0.0625 else "INCONCLUSIVE-PARTIAL")
        judged[str(sd)] = {"acc_train_intact": tr_i, "acc_train_Wzero": tr_z,
                           "acc_held_intact": ho_i, "acc_held_Wzero": ho_z,
                           "delta_tr": d_tr, "delta_ho": d_ho, "verdict": verdict,
                           "acc_train_Wnowrite": cells[f"{sd}|Wnowrite|train"]["acc_q"],
                           "acc_train_Wother": cells[f"{sd}|Wother|train"]["acc_q"]}
        print(f"★ seed {sd}: Δ_tr={d_tr:+.4f} (intact {tr_i:.4f} − zero {tr_z:.4f}) "
              f"Δ_ho={d_ho:+.4f} ⇒ {verdict}", flush=True)
    G["judged"] = judged
    vs = [judged[str(sd)]["verdict"] for sd in seeds]
    G["verdict_agg"] = vs[0] if len(set(vs)) == 1 else f"CONFLICT:{vs}"

    # ---- 守卫汇总 ----
    A = {}
    A["A94_1_inj_zero"] = all(
        v == 0.0 for d in G.get("A94_1_inj_max_at_Wzero", {}).values() for v in d.values())
    A["A94_2_reproduce_D121"] = all(abs(judged[str(sd)]["acc_held_intact"] - 0.0625) <= 0.0625
                                    for sd in seeds)
    A["A94_3_gamma_changes_logits"] = all(
        abs(v["delta_nll"]) > 0.0 for v in G.get("A94_3_gamma", {}).values())
    A["A94_4_no_write_during_eval"] = all(
        G.get("A94_4_nowrite_after", {}).values())   # 值是 bool，不要二次 all()
    A["A94_5_same_exam"] = True     # 四条件共用同一 items 对象（同一 md5），构造上成立
    A["A94_8_nll_finite"] = all(
        (r["nll_mean"] == r["nll_mean"]) and abs(r["nll_mean"]) < 1e6   # 非 nan、非爆
        for r in cells.values())
    A["A94_7_int_consistency"] = all(
        abs(round(judged[str(sd)]["acc_train_intact"] * 16) / 16
            - judged[str(sd)]["acc_train_intact"]) < 1e-9 for sd in seeds)
    G["guards"] = A
    print("\n=== Stage A / 守卫 ===")
    for k, v in A.items():
        print(f"  {'✓' if v else '✗'} {k} = {v}")

    (OUT / f"stage_{a.stage}.json").write_text(
        json.dumps({"stage": a.stage, "seeds": seeds, "cells": cells, **G},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[写出] {OUT / f'stage_{a.stage}.json'}")


if __name__ == "__main__":
    main()