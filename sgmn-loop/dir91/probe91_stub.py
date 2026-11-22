#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 · rev 1.8 分离探针：注入通路可用吗？`v` 里到底有没有「是哪个值 token」？

**性质：诊断**。不产生裁决、不写证据行、不改任何已冻结判据。

task.md（rev1.8）要求先回答：全绑定臂里 $Wk_i\\approx v_i$、$q=k_i$、注入 $=P_vv_i$，
却仍只有 BASE 水平 —— 是 (a) 注入位置/尺度不对，还是 (b) $v$ 编码不含可用语义？
并明确：**必须先做一个分离桩**（把「正确答案 token 的 embedding」注入同一位置），
若该桩也答不对 ⇒ 注入通路本身在本基座/本位置不可用，与 $v$ 无关。

★ 本基座 `tie_word_embeddings=True`（已实测：`lm_head.weight` 与 `embed_tokens.weight`
  是同一块内存，hidden=896，24 层）。因此 `logit_t ≈ <norm(h_last), e_t>`，
  把 $e_{\\rm ans}$ 加到最后位置**近乎代数地**抬高该 token 的 logit。
  ⇒ 单看「桩能答对」是**同义反复**。故本探针的桩一律带 **同范数随机 token 对照**，
  并另立一条不依赖桩的定量判据 M5（对齐余弦）。

四部分：

  P0  BASE —— 无栈裸基座（本次自测的参照，避免与旧轮读数混用）。
  S   **分离桩**：在挂载层注入 $\\rho\\lVert h\\rVert\\hat e$。
        S-A `e_ans`（该题正确答案 token 的 embedding）
        S-B `e_rand`（同范数、随机**另一个** token 的 embedding）—— 特异性对照
        S-C `e_ans` 但只放单层 8 / 12 / 16 —— 位置分离
      若 S-A 有效而 S-B 无效 ⇒ 通路可用且可特异驱动输出；
      若 S-A 与 S-B 同样有效（或同样无效）⇒ 桩不可解释，须以 M5 为准。
  M   **尺度与对齐**：
        M1 $\\lVert\\Delta\\rVert/\\lVert h\\rVert$（$\\Delta=P_vv_i$ 与学到的 $\\gamma W_oW W_qc$）
        M2 **M5 对齐**：$\\cos(\\Delta,\\ e_{\\rm ans})$ 与 $\\cos(\\Delta,\\ e_{\\rm distr})$ 的**边际**
           （tied lm_head ⇒ 这正是注入对答案 logit 的贡献方向）
        M3 参照量 $\\cos(e_{\\rm ans},e_{\\rm distr})$（答案 embedding 之间本身有多像）
  L   **全绑定单层注入**（8 / 12 / 16 各一次 + 三层齐上），无学习参数。

★ 全部为诊断读数；不得当作 $\\Delta_{\\rm content}$ 的计算。
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from R2_task91 import (  # noqa: E402
    DEV, N_FACTS, N_TRAIN_KEYS, acc_by_question, build_chunks, build_facts, dump,
    key_matrices, make_eval_items, make_stack, model_and_tok, nll_of_candidate,
    write_phase,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GAMMA = 0.05
MOUNT = [8, 12, 16]

STUB = {"e": None, "idx": None, "rho": 0.0}


# ------------------------------------------------------------------ 工具
def embed_of(model, tok_id):
    """(d_model,) 的输入 embedding。"""
    with torch.no_grad():
        w = model.get_input_embeddings().weight
        return w[torch.tensor([tok_id], device=DEV)][0].detach().float().clone()


def cos(a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def install_stub(st, layers, rho_key="rho"):
    """把指定层的 read 换成桩：最后位置注入 rho*||h||*e_hat。gamma 置 1。"""
    for li in layers:
        m = st.mems[li]
        m.gamma.data.fill_(1.0)

        def make(mem):
            def stub_read(hs_in, drop_sg=False):
                delta = torch.zeros_like(hs_in)
                if STUB["e"] is not None and STUB["idx"] is not None:
                    i = STUB["idx"]
                    ref = float(hs_in[0, i, :].norm())
                    e = STUB["e"]
                    scale = STUB[rho_key] * ref / (float(e.norm()) + 1e-30)
                    delta[:, i, :] = scale * e
                mem.injection = delta
                return delta
            return stub_read
        m.read = make(m)


def install_fulltie(st, layers):
    """读链 = Pv o W o Pk^T，无学习参数（探针四的构造）。"""
    for li in layers:
        m = st.mems[li]

        def make(mem):
            def ft(hs_in, drop_sg=False):
                cf = hs_in.detach().reshape(-1, hs_in.shape[-1])
                q = (cf @ mem.Pk).detach() * float(mem.pk_scale)
                W = mem.W0 + mem.U @ mem.V.T
                y = (W @ q.T).T
                delta = (y @ mem.Pv.T) * float(mem.gamma)
                mem.injection = delta.reshape(hs_in.shape)
                return mem.injection
            return ft
        m.read = make(m)


def silence_read(st, layers=None):
    """其余挂载层不注入（保证单层臂只在该层生效）。"""
    for li, m in st.mems.items():
        if layers is not None and li in layers:
            continue
        m.read = lambda hs_in, drop_sg=False: torch.zeros_like(hs_in)


def correct_tid(it):
    for ids, ok in it["cand"]:
        if ok:
            return ids[0]
    raise AssertionError("no correct candidate")


def score_items(model, st, items):
    det = []
    for it in items:
        nlls = [nll_of_candidate(model, st, it["prompt_ids"], c[0]) for c in it["cand"]]
        det.append({"ok": bool(it["cand"][int(np.argmin(nlls))][1])})
    return det


def acc_pair(model, st, items):
    det = score_items(model, st, items)
    return float(np.mean([d["ok"] for d in det])), acc_by_question(items, det)[0]


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    order = list(range(N_FACTS))
    random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]
    res = {"seed": a.seed, "gamma": GAMMA, "mount": MOUNT,
           "tie_word_embeddings": bool(model.config.tie_word_embeddings)}
    print(f"tie_word_embeddings={res['tie_word_embeddings']} d={model.config.hidden_size} "
          f"L={model.config.num_hidden_layers}", flush=True)

    # ---------------- P0 BASE：无栈
    items_ev = make_eval_items(tok, facts, eval_keys, random.Random(555))
    det0 = score_items(model, None, items_ev)
    base_i = float(np.mean([d["ok"] for d in det0]))
    base_q = acc_by_question(items_ev, det0)[0]
    res["base"] = {"acc_item": base_i, "acc_q": base_q}
    print(f"\n[P0 BASE] 无栈  留出键 acc(项)={base_i:.3f} acc(题)={base_q:.3f}", flush=True)
    dump(res, "probe91_stub.json")

    # ---------------- S 分离桩
    print("\n--- S：分离桩（最后位置注入 rho*||h||*e_hat）---", flush=True)
    res["stub"] = {}
    tid_ans = [correct_tid(it) for it in items_ev]
    # 同范数随机对照所用的「另一个」token：取该题某个干扰候选的 token
    tid_rand = []
    for it in items_ev:  # rng 固定，可复现
        others = [c[0][0] for c in it["cand"] if not c[1]]
        tid_rand.append(others[len(tid_rand) % max(1, len(others))])

    for tag, tid in (("S-A_e_ans", tid_ans), ("S-B_e_rand", tid_rand)):
        for rho in (0.5, 1.0, 2.0, 4.0):
            for lset in ((16,), (8, 12, 16)):
                st = make_stack(gamma=1.0, no_write=True, seed=a.seed)
                install_stub(st, lset)
                silence_read(st, layers=set(lset))
                det = []
                for it, t in zip(items_ev, tid):
                    STUB["e"] = embed_of(model, t)
                    STUB["idx"] = len(it["prompt_ids"]) - 1
                    STUB["rho"] = rho
                    nlls = [nll_of_candidate(model, st, it["prompt_ids"], c[0])
                            for c in it["cand"]]
                    det.append({"ok": bool(it["cand"][int(np.argmin(nlls))][1])})
                ai = float(np.mean([d["ok"] for d in det]))
                aq = acc_by_question(items_ev, det)[0]
                k = f"{tag}|rho={rho}|L={lset}"
                res["stub"][k] = {"acc_item": ai, "acc_q": aq}
                print(f"  {tag:<11} rho={rho:<4} L={str(lset):<11} "
                      f"acc(项)={ai:.3f} acc(题)={aq:.3f}", flush=True)
                dump(res, "probe91_stub.json")
                STUB["e"] = None
                st.remove()

    # ---------------- M 尺度与对齐（M5）
    print("\n--- M：尺度与对齐余弦 ---", flush=True)
    st = make_stack(gamma=GAMMA, no_write=False, seed=a.seed)
    write_phase(model, st, chunks)
    K, V = key_matrices(model, st, facts, chunks, owner)
    e_ans = {ki: embed_of(model, facts[ki]["val_ids"][0]) for ki in eval_keys}
    # 参照：答案 embedding 之间的余弦（值是 VALUE_POOL 里的词，彼此可能相似）
    ref_cos = []
    for i in eval_keys:
        for j in eval_keys:
            if i != j:
                ref_cos.append(cos(e_ans[i], e_ans[j]))
    res["M3_cos_e_ans_pair"] = {"mean": float(np.mean(ref_cos)),
                               "max": float(np.max(np.abs(ref_cos)))}
    print(f"  M3 cos(e_ans_i, e_ans_j) 均值={np.mean(ref_cos):+.4f} "
          f"最大绝对值={np.max(np.abs(ref_cos)):.4f}", flush=True)

    # ★ 采集 hs_in 前必须静音全部读，否则 8/12 层的注入会污染 12/16 层的 hs_in。
    silence_read(st)
    HS = {li: {} for li in MOUNT}
    for ki in eval_keys:
        f = facts[ki]
        ids = torch.tensor([f["prompt_ids"]], dtype=torch.long, device=DEV)
        with torch.no_grad():
            model(input_ids=ids)
        for li in MOUNT:
            HS[li][ki] = st._hs_in[li][0, len(f["prompt_ids"]) - 1, :].detach().float()

    res["M"] = {}
    for li in MOUNT:
        m = st.mems[li]
        rows = []
        for ki in eval_keys:
            hs = HS[li][ki]
            W = (m.W0 + m.U @ m.V.T).detach()
            # key_matrices 按 fi 顺序堆叠 ⇒ 下标就是事实号 ki（不是 eval_keys 内的位置）
            v, k = V[li][ki], K[li][ki]
            d_tie = (m.Pv.detach() @ (W @ k))
            q = (hs @ m.Pk.detach()) * float(m.pk_scale)
            d_full = (m.Pv.detach() @ (W @ q)) * float(m.gamma)
            hn = float(hs.norm()) + 1e-30
            e_a = e_ans[ki]
            # 干扰答案 embedding 的均值方向
            e_d = torch.stack([e_ans[j] for j in eval_keys if j != ki]).mean(0)
            rows.append({
                "ki": ki,
                "rel_tie": float(d_tie.norm()) / hn,
                "rel_full": float(d_full.norm()) / hn,
                "cos_tie_ans": cos(d_tie, e_a), "cos_tie_distr": cos(d_tie, e_d),
                "cos_full_ans": cos(d_full, e_a), "cos_full_distr": cos(d_full, e_d),
                "h_norm": hn,
            })
        agg = {k: float(np.mean([r[k] for r in rows])) for k in
               ("rel_tie", "rel_full", "cos_tie_ans", "cos_tie_distr",
                "cos_full_ans", "cos_full_distr", "h_norm")}
        agg["margin_tie"] = agg["cos_tie_ans"] - agg["cos_tie_distr"]
        agg["margin_full"] = agg["cos_full_ans"] - agg["cos_full_distr"]
        res["M"][li] = agg
        print(f"  L{li}: ||h||={agg['h_norm']:.1f} | "
              f"||Δtie||/||h||={agg['rel_tie']:.4f} ||Δfull||/||h||={agg['rel_full']:.4f}",
              flush=True)
        print(f"       cos(Δtie,e_ans)={agg['cos_tie_ans']:+.4f} "
              f"vs e_distr={agg['cos_tie_distr']:+.4f}  边际={agg['margin_tie']:+.4f}",
              flush=True)
        print(f"       cos(Δfull,e_ans)={agg['cos_full_ans']:+.4f} "
              f"vs e_distr={agg['cos_full_distr']:+.4f}  边际={agg['margin_full']:+.4f}",
              flush=True)
        dump(res, "probe91_stub.json")
    st.remove()

    # ---------------- L 全绑定单层
    print("\n--- L：全绑定单层注入 ---", flush=True)
    res["fulltie_layers"] = {}
    for lset in ((8,), (12,), (16,), (8, 12, 16)):
        st = make_stack(gamma=GAMMA, no_write=False, seed=a.seed)
        write_phase(model, st, chunks)
        install_fulltie(st, lset)
        silence_read(st, layers=set(lset))
        ai, aq = acc_pair(model, st, items_ev)
        res["fulltie_layers"][str(lset)] = {"acc_item": ai, "acc_q": aq}
        print(f"  fulltie L={str(lset):<11} 留出键 acc(项)={ai:.3f} acc(题)={aq:.3f}",
              flush=True)
        dump(res, "probe91_stub.json")
        st.remove()

    print("\n★ 诊断完成（不产生裁决）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())