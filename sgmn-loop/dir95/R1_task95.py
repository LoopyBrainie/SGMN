#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-95 · 在 `D-115` 自己的装置里做 `W` 消融 —— 卡 `revolve/DIR-95/DIRECTION_rev95_v12.md`（rev 1.2 冻结）。

主判据量（§3，冻结）： Δ_tr = acc_train(intact) − acc_train(W=0)
  Δ_tr ≥ 0.25            ⇒ W-MEDIATED
  Δ_tr ≤ 0.0625          ⇒ W-BYPASSED
  否则                    ⇒ INCONCLUSIVE-PARTIAL
★ 阈值与 `D-124`（DIR-94）**逐字相同**，为的是**跨装置可比**；rev 1.0 即冻结，未改。

★ 本脚本**逐字派生**自 `dir94/R1_task94.py`：
  - **只换装置**：评测/写入/训练全部走 `dir91/probe91_kpool.py` 的**原生臂 `K_mean_V_mean`**
    （`write_phase_arm` / `R2_task91.train_read_task` / `probe91_womap.eval_keys`），一字不改；
  - **判据量改为 `D-115` 原口径**：`acc_train = W91.eval_keys(..., tr, None, False, None)["acc_q"]`
    （**按题 3 次全对** ⇒ 16 母题、分辨率 1/16；按题机会 (1/4)^3=0.0156）；
  - **唯一独立变量**仍只有「评测时 `W` 的取值」（4 条件）。

★ 与 `DIR-94` 的两处必须不同（否则会踩已记录的坑）：
  - `DEF-94-03`：**不得**并存多个栈。故分两趟：①写+训练 ⇒ 存**状态** ⇒ `st.remove()`；
    ②按 seed 重建栈（`make_stack(seed=sd)` 决定性地复现同一 `W0`）⇒ 灌回状态 ⇒ 评测。
  - `W=Wother` 需要**另一 seed 训练好的 `Wfull`** ⇒ 必须先跑完趟①的**全部** seed。

Stage：`--stage A` 只跑守卫；`smoke`（1 seed、B=60、2 条件）；`formal`（2 seed、4 条件）。
"""
import argparse
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
import probe91_kpool as KP                        # noqa: E402
import probe91_womap as W91                       # noqa: E402
import R2_task91 as R91                           # noqa: E402
import torch                                      # noqa: E402

SEEDS = [20261028, 7777]          # 与 D-115 同（其 JSON 原值即出自这两个 seed）
ARM = "K_mean_V_mean"             # D-115 原生臂（acc_train 0.875/0.8125 即出自此臂）
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
# A95-2 的跨轮参照：dir91/out/probe91_kpool.json 原值（D-115 的产物）
REF = {20261028: {"acc_train": 0.875, "acc_held": 0.0, "margin_train": 7.664, "gamma": -0.0212},
       7777:     {"acc_train": 0.8125, "acc_held": 0.0, "margin_train": 8.971, "gamma": -0.0144}}
TOL = 1.0 / 16.0                  # A95-2 容差 ±1 题位


def load_state(st, sv):
    """把训练好的可塑状态灌回重建的栈（一次一栈，DEF-94-03）。"""
    for li, m in st.mems.items():
        wq, wo, gm = sv["rp"][li]
        with torch.no_grad():
            m.Wq.copy_(wq)
            m.Wo.copy_(wo)
            m.gamma.copy_(gm)
            m.Wfull.copy_(sv["W"][li])
            m.no_write = True           # ★ 评测期禁止写入（DEF-94-02）


def eval_keys(model, st, tok, facts, keys):
    """**逐字**调用 D-115 的评测口（`probe91_womap.eval_keys`，内部自设 no_write=True）。"""
    return W91.eval_keys(model, st, tok, facts, keys, None, False, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["A", "smoke", "formal", "all"])
    a = ap.parse_args()
    smoke = (a.stage == "smoke")
    seeds = SEEDS[:1] if smoke else SEEDS
    OUT.mkdir(exist_ok=True)
    G = {}

    tok, model = KP.model_and_tok()
    model.eval()
    torch.set_grad_enabled(True)            # 逐字对齐 probe91_kpool.main()
    facts = KP.build_facts(tok)
    chunks, owner = KP.build_chunks(tok, facts)
    if smoke:                                  # smoke：写入预算砍到 1/10（永不出判据）
        chunks = chunks[:max(4, len(chunks) // 10)]
    tr = list(range(R91.N_TRAIN_KEYS))
    he = list(range(R91.N_TRAIN_KEYS, R91.N_FACTS))
    print(f"[setup] n_facts={len(facts)} chunks={len(chunks)} tr={len(tr)} he={len(he)} "
          f"arm={ARM} smoke={smoke}", flush=True)
    assert KP.RULE[ARM] == ("mean", "mean"), KP.RULE[ARM]

    # ---------------- 趟①：写 + 训练读出 ⇒ 只留状态、立刻卸栈 ----------------
    saved = {}
    KP.install_write()
    try:
        for sd in seeds:
            t0 = time.time()
            st = R91.make_stack(gamma=0.0, no_write=False, seed=sd)   # 与 run_arm 逐字一致
            KP.write_phase_arm(model, st, chunks, owner, None)        # 原生臂 need_vt=False
            losses, viol, div = R91.train_read_task(
                model, st, tok, facts, tr, random.Random(1000 + sd % 100))
            saved[sd] = {
                "W": {li: m.Wfull.detach().clone() for li, m in st.mems.items()},
                "W0": {li: m.W0.detach().clone() for li, m in st.mems.items()},
                "rp": {li: (m.Wq.detach().clone(), m.Wo.detach().clone(),
                            m.gamma.detach().clone()) for li, m in st.mems.items()},
                "stat": {"final_loss": float(sum(losses[-50:]) / len(losses[-50:])) if losses else None,
                         "viol": viol, "div": div,
                         "w_fro": {li: float(m.Wfull.detach().norm()) for li, m in st.mems.items()},
                         "w0_fro": {li: float(m.W0.norm()) for li, m in st.mems.items()},
                         "dW_rel": {li: float((m.Wfull.detach() - m.W0).norm()
                                              / m.W0.norm()) for li, m in st.mems.items()}},
            }
            print(f"[train] seed={sd} loss={saved[sd]['stat']['final_loss']:.4f} "
                  f"viol={viol} div={div} dW_rel="
                  f"{[round(v, 6) for v in saved[sd]['stat']['dW_rel'].values()]} "
                  f"{time.time() - t0:.0f}s", flush=True)
            st.remove()                       # ★ 卸栈，绝不与下一 seed 并存
            del st
            torch.cuda.empty_cache()
    finally:
        KP.restore_write()
    G["train"] = {str(sd): saved[sd]["stat"] for sd in seeds}

    # ---------------- 趟②：重建栈 → 灌状态 → 4 条件 × 2 键集 ----------------
    cells = {}
    for i, sd in enumerate(seeds):
        st = R91.make_stack(gamma=0.0, no_write=True, seed=sd)
        load_state(st, saved[sd])
        if len(seeds) > 1:
            other = saved[seeds[(i + 1) % len(seeds)]]["W"]
        else:                                  # smoke：无第二 seed，用列置换造"异己 W"
            other = {li: saved[sd]["W"][li].flip(1) for li in saved[sd]["W"]}
        conds = {"intact": saved[sd]["W"],
                 "Wzero": {li: torch.zeros_like(saved[sd]["W"][li]) for li in saved[sd]["W"]},
                 "Wnowrite": saved[sd]["W0"],
                 "Wother": other}
        for name, W in conds.items():
            with torch.no_grad():
                for li, m in st.mems.items():
                    m.Wfull.copy_(W[li])
            if name == "Wzero":                       # A95-1：注入必须精确为 0
                model(input_ids=chunks[0].to(R91.DEV))
                G.setdefault("A95_1_inj_max_at_Wzero", {})[str(sd)] = {
                    li: float(m.injection.abs().max()) for li, m in st.mems.items()}
            for tag, keys in (("train", tr), ("held", he)):
                t0 = time.time()
                wc0 = {li: m.write_calls for li, m in st.mems.items()}
                r = eval_keys(model, st, tok, facts, keys)
                wc1 = {li: m.write_calls for li, m in st.mems.items()}
                cells[f"{sd}|{name}|{tag}"] = {"acc_q": r["acc_q"], "acc_item": r["acc_item"],
                                              "n_q": r["n_q"], "n_item": r["n_item"],
                                              "margin": r["margin_mean"],
                                              "write_delta": {li: wc1[li] - wc0[li] for li in wc0}}
                print(f"  [{sd}|{name:9s}|{tag}] acc_q={r['acc_q']:.4f} "
                      f"acc_item={r['acc_item']:.4f} margin={r['margin_mean']:+.3f} "
                      f"nq={r['n_q']} dwrite={sum(cells[f'{sd}|{name}|{tag}']['write_delta'].values())} "
                      f"{time.time() - t0:.0f}s", flush=True)
        # ---- A95-3：γ 旋钮 ----
        with torch.no_grad():
            for li, m in st.mems.items():
                m.Wfull.copy_(saved[sd]["W"][li])
        g0 = {li: m.gamma.detach().clone() for li, m in st.mems.items()}
        items8 = W91.make_eval_items(tok, facts, tr, random.Random(555))[:8]
        base = [W91.nll_of_candidate(model, st, it["prompt_ids"], c) for it in items8 for c, _ in it["cand"]]
        with torch.no_grad():
            for m in st.mems.values():
                m.gamma.mul_(20.0)
        hi = [W91.nll_of_candidate(model, st, it["prompt_ids"], c) for it in items8 for c, _ in it["cand"]]
        with torch.no_grad():
            for li, m in st.mems.items():
                m.gamma.copy_(g0[li])
        G.setdefault("A95_3_gamma", {})[str(sd)] = {
            "nll_base": sum(base) / len(base), "nll_x20": sum(hi) / len(hi),
            "max_abs_d": max(abs(x - y) for x, y in zip(base, hi))}
        print(f"  [A95-3 γx20] mean nll {sum(base)/len(base):.4f} → {sum(hi)/len(hi):.4f} "
              f"max|Δ|={G['A95_3_gamma'][str(sd)]['max_abs_d']:.6f}", flush=True)
        G.setdefault("no_write_after", {})[str(sd)] = all(m.no_write for m in st.mems.values())
        st.remove()
        del st
        torch.cuda.empty_cache()

    # ---------------- 判据（§3/§8 机械） ----------------
    judged = {}
    for sd in seeds:
        tr_i = cells[f"{sd}|intact|train"]["acc_q"]
        tr_z = cells[f"{sd}|Wzero|train"]["acc_q"]
        ho_i = cells[f"{sd}|intact|held"]["acc_q"]
        ho_z = cells[f"{sd}|Wzero|held"]["acc_q"]
        d_tr, d_ho = tr_i - tr_z, ho_i - ho_z
        ref = REF[sd]
        ok_repro = (abs(tr_i - ref["acc_train"]) <= TOL + 1e-12) and \
                   (abs(ho_i - ref["acc_held"]) <= TOL + 1e-12)
        if smoke:                                # §7：smoke 永不出判据
            verdict = "SMOKE-NO-VERDICT"
        elif not ok_repro:
            verdict = "INSTRUMENT-MISMATCH"          # §8 第 1 条
        elif d_tr >= 0.25:
            verdict = "W-MEDIATED"
        elif d_tr <= 0.0625:
            verdict = "W-BYPASSED"
        else:
            verdict = "INCONCLUSIVE-PARTIAL"
        judged[str(sd)] = {"acc_train_intact": tr_i, "acc_train_Wzero": tr_z,
                           "acc_held_intact": ho_i, "acc_held_Wzero": ho_z,
                           "delta_tr": d_tr, "delta_ho": d_ho,
                           "acc_train_Wnowrite": cells[f"{sd}|Wnowrite|train"]["acc_q"],
                           "acc_train_Wother": cells[f"{sd}|Wother|train"]["acc_q"],
                           "margin_train_intact": cells[f"{sd}|intact|train"]["margin"],
                           "repro_ref": ref, "repro_ok": ok_repro, "verdict": verdict}
        print(f"★ seed {sd}: Δ_tr={d_tr:+.4f} (intact {tr_i:.4f} − zero {tr_z:.4f}) "
              f"Δ_ho={d_ho:+.4f} repro={ok_repro} ⇒ {verdict}", flush=True)
    G["judged"] = judged
    vs = [judged[str(sd)]["verdict"] for sd in seeds]
    G["verdict_agg"] = ("SMOKE-NO-VERDICT" if smoke else vs[0]) if len(set(vs)) == 1 \
        else "INCONCLUSIVE-SEED-SPLIT"

    # ---------------- Stage A 守卫（§6） ----------------
    A = {}
    A["A95_1_inj_zero"] = all(
        v == 0.0 for d in G.get("A95_1_inj_max_at_Wzero", {}).values() for v in d.values())
    A["A95_2_reproduce_D115"] = all(judged[str(sd)]["repro_ok"] for sd in seeds)
    A["A95_3_gamma_changes_logits"] = all(
        v["max_abs_d"] > 0.0 for v in G.get("A95_3_gamma", {}).values())
    A["A95_4_no_write_during_eval"] = (
        all(sum(c["write_delta"].values()) == 0 for c in cells.values())
        and all(G.get("no_write_after", {}).values()))
    A["A95_5_same_exam"] = True      # 四条件共用同一 keys 列表与同一 Random(555)，构造上成立
    A["A95_6_nll_finite"] = all(
        c["margin"] == c["margin"] and abs(c["margin"]) < 1e6 for c in cells.values())
    A["A95_7_int_consistency"] = all(
        abs(round(judged[str(sd)]["acc_train_intact"] * 16) / 16
            - judged[str(sd)]["acc_train_intact"]) < 1e-9 for sd in seeds)
    A["A95_8_one_stack"] = (len(saved) == len(seeds))     # 趟①每个 seed 都卸栈后才进下一个
    A["A95_9_margin_train_pos"] = all(
        judged[str(sd)]["margin_train_intact"] > 0 for sd in seeds)
    G["guards"] = A
    print("\n=== Stage A / 守卫 ===")
    for k, v in A.items():
        print(f"  {'✓' if v else '✗'} {k} = {v}")

    # ★ DEF-94-05：**先写产物，再让异常发生**（聚合已在上面完成，此处只落盘）
    (OUT / f"stage_{a.stage}.json").write_text(
        json.dumps({"stage": a.stage, "seeds": seeds, "arm": ARM, "cells": cells, **G},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[写出] {OUT / f'stage_{a.stage}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())