#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 · rev 1.7 前置机制探针（**诊断，不产生裁决**）

任务（task.md 顶部 ★「必须先做的复核 ③」）：
  「`W_q` 是否学到了近似恒等映射（需 Wq k_i ≈ k_i 才能让 W 命中第 i 个槽）」

★ 订正任务里的一句话：这个条件**不是**"恒等"，而是更精确的一条 ——
    y = W q,  q = Wq c,  m = Wo y      （sgmn_module.py:66-74）
    k = pk_scale · (Pk^T h)            （sgmn_module.py:86）
  要让 q ≈ k_i，需要  **Wq ≈ pk_scale · Pk^T**（两者同为 (d_k, d_model)）。
  故本探针测 M1 = cos(Wq, pk_scale·Pk^T) 与相对 Frobenius 误差，而非"是否恒等"。

六项读数（全部逐层、逐 MOUNT）：
  M1 读映射与写入侧键投影的接近度：cos(Wq, p·Pk^T)、rel_F、范数比
  M2 训练键上的 q 对齐：cos(Wq c_i, k_i)
  M3 **留出键**上的 q 对齐：cos(Wq c_j, k_j)   ← M2-M3 的落差 = 泛化落差
  M4 **oracle-q 上界**：强制 q = k_i，绕开 Wq ⇒ 直接问"W 里的内容可读吗"
  M5 训练后 Wq/Wo/gamma 的位移量（训练是否真的动了它们）
  M6 未训练对照（同装置、跳过训练）的同名读数

★ 本探针**不改任何已冻结判据阈值**，只产出诊断数字。裁决另开卡片。
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from R2_task91 import (  # noqa: E402
    DEV, N_FACTS, N_TRAIN_KEYS, READ_LR, READ_STEPS, acc_by_question, build_chunks,
    build_facts, dump, key_matrices, make_eval_items, make_stack, model_and_tok,
    nll_of_candidate, train_read_task, write_phase,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


# ------------------------------------------------------------------ 工具
def flat_cos(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def rel_f(a, b):
    a = a.double()
    b = b.double()
    return float((a - b).norm() / (b.norm() + 1e-30))


def read_map_readings(st, K, V):
    """M1/M2/M3：读映射接近度与 q 对齐。"""
    out = {}
    for li, m in st.mems.items():
        p = float(m.pk_scale)
        target = p * m.Pk.T.detach()                     # (d_k, d_model)
        out[li] = {
            "pk_scale": p,
            "M1_cos_Wq_PkT": flat_cos(m.Wq.detach(), target),
            "M1_relF": rel_f(m.Wq.detach(), target),
            "M1_norm_ratio": float(m.Wq.detach().double().norm()
                                   / (target.double().norm() + 1e-30)),
        }
    return out


def q_alignment(st, model, facts, chunks, owner, K, keys):
    """对给定键集合，测 cos(Wq c_i, k_i)。c_i 取写入时同位置（第 owner[i]+1 块）。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    acc = {li: [] for li in st.mems}
    try:
        for ki in keys:
            ci = owner[ki] + 1
            model(input_ids=chunks[ci].to(DEV))
            for li, m in st.mems.items():
                hs = st._hs_in[li]
                cf = hs.reshape(-1, hs.shape[-1]).detach()
                cf = cf.to(m.Wq.dtype)
                q = (m.Wq @ cf.T).T                      # (N,d_k)
                qi = q.mean(dim=0)                        # 该块的平均 q
                acc[li].append(flat_cos(qi, K[li][ki]))
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
    return {li: float(np.mean(v)) for li, v in acc.items()} if acc else {}


def run_arm_with_oracle(name, tok, model, facts, chunks, owner, train_keys,
                        eval_keys, seed, train=True):
    """复刻 run_arm 的 MEM 臂，并额外测 M4（oracle-q）与 M6（未训练对照）。"""
    rec = {"arm": name, "seed": seed, "trained": train}

    # ---- ① 装置 + 写入（与 run_arm 完全一致：gamma=GAMMA_RUN, do_write=True）
    st = make_stack(gamma=0.05, no_write=False, seed=seed)
    wc = write_phase(model, st, chunks)
    rec["write_calls"] = wc

    # ---- ② 写入后、训练前的键/值真值
    K, V = key_matrices(model, st, facts, chunks, owner)

    # ---- ③ M6：未训练对照读数
    if not train:
        rec["M1"] = read_map_readings(st, K, V)
        rec["M2_train_keys"] = q_alignment(st, model, facts, chunks, owner, K, train_keys)
        rec["M3_eval_keys"] = q_alignment(st, model, facts, chunks, owner, K, eval_keys)
        acc, _ = eval_arm(model, st, tok, facts, eval_keys, oracle=None)
        rec["acc_normal"] = acc
        st.remove()
        return rec

    # ---- ④ 训练读侧
    losses, viol, div = train_read_task(model, st, tok, facts, train_keys,
                                        random.Random(1000 + seed % 100))
    rec.update({"violations": viol, "diverged": div,
                "final_loss": float(np.mean(losses[-50:])) if losses else None})

    # ---- ⑤ M1 / M2 / M3
    rec["M1"] = read_map_readings(st, K, V)
    rec["M2_train_keys"] = q_alignment(st, model, facts, chunks, owner, K, train_keys)
    rec["M3_eval_keys"] = q_alignment(st, model, facts, chunks, owner, K, eval_keys)

    # ---- ⑥ M4：oracle-q 上界 + 常规读数
    acc_n, acc_q_n = eval_arm(model, st, tok, facts, eval_keys, oracle=None)
    acc_o, acc_q_o = eval_arm(model, st, tok, facts, eval_keys, oracle=K)
    rec["acc_normal"] = acc_n
    rec["acc_oracle"] = acc_o
    rec["acc_q_normal"] = acc_q_n
    rec["acc_q_oracle"] = acc_q_o

    # 额外：oracle-q 也在训练键上看一眼（区分"能背"与"能推"）
    acc_ot, acc_q_ot = eval_arm(model, st, tok, facts, train_keys, oracle=K,
                                rng_seed=556)
    rec["acc_q_oracle_trainkeys"] = acc_q_ot
    rec["acc_q_normal_trainkeys"] = eval_arm(model, st, tok, facts, train_keys,
                                             oracle=None, rng_seed=556)[1]

    st.remove()
    return rec


def eval_arm(model, st, tok, facts, keys, oracle=None, rng_seed=555):
    """评测。oracle=K 时把 read 换成 q = K[li][key_idx]（绕开 Wq），其余一律不变。

    oracle 机制：三层各自 mem.read 被替换为返回 Wo(W q_i)，q_i 用**该题的写入键真值**。
    ⇒ 记忆内容与写入侧一切不变，唯一变化是"q 是否取对了"。
    """
    for m in st.mems.values():
        m.no_write = True
    items = make_eval_items(tok, facts, keys, random.Random(rng_seed))
    orig = {li: m.read for li, m in st.mems.items()}
    cur = {"ki": 0}          # 可变容器：patched 闭包读到当前题号
    if oracle is not None:
        for li, m in st.mems.items():
            def make_patched(mem, li_):
                def patched(hs_in, drop_sg=False):
                    n = hs_in.reshape(-1, hs_in.shape[-1]).shape[0]
                    q = oracle[li_][cur["ki"]].detach().to(hs_in.device)
                    q = q.unsqueeze(0).expand(n, -1)
                    W = (mem.W0 + mem.U @ mem.V.T) if mem.use_factorized else mem.Wfull
                    y = (W @ q.T).T
                    mm = (mem.Wo @ y.T).T
                    mem.injection = mm.reshape(hs_in.shape)
                    return mem.injection
                return patched
            m.read = make_patched(m, li)
    try:
        preds, det = [], []
        for it in items:
            cur["ki"] = it["key_idx"]
            nlls = [nll_of_candidate(model, st, it["prompt_ids"], c[0]) for c in it["cand"]]
            pick = int(np.argmin(nlls))
            ok = bool(it["cand"][pick][1])
            preds.append(pick)
            det.append({"ok": ok})
        acc = float(np.mean([d["ok"] for d in det]))
        accq, _ = acc_by_question(items, det)
        return acc, accq
    finally:
        for li, m in st.mems.items():
            m.read = orig[li]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="20261028")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    order = list(range(N_FACTS))
    random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]
    print(f"事实 {len(facts)}；训练键 {len(train_keys)}；留出键 {len(eval_keys)}；"
          f"READ_LR={READ_LR} READ_STEPS={READ_STEPS}", flush=True)

    res = {"READ_LR": READ_LR, "READ_STEPS": READ_STEPS,
           "train_keys": train_keys, "eval_keys": eval_keys, "arms": {}}

    for seed in seeds:
        # 未训练对照（M6）
        r0 = run_arm_with_oracle("MEM-untrained", tok, model, facts, chunks, owner,
                                 train_keys, eval_keys, seed, train=False)
        res["arms"][f"untrained|{seed}"] = r0
        print(f"\n[{seed}] 未训练对照：acc={r0['acc_normal']:.3f}", flush=True)
        for li in r0["M1"]:
            d = r0["M1"][li]
            print(f"   L{li} M1 cos(Wq,p·Pk^T)={d['M1_cos_Wq_PkT']:+.4f} "
                  f"relF={d['M1_relF']:.4f} |norm|={d['M1_norm_ratio']:.3f}", flush=True)
        print(f"   M2 训练键 cos(q,k)={r0['M2_train_keys']}", flush=True)
        print(f"   M3 留出键 cos(q,k)={r0['M3_eval_keys']}", flush=True)

        # 训练臂
        r1 = run_arm_with_oracle("MEM-trained", tok, model, facts, chunks, owner,
                                 train_keys, eval_keys, seed, train=True)
        res["arms"][f"trained|{seed}"] = r1
        print(f"\n[{seed}] 训练后：acc(项)={r1['acc_normal']:.3f} "
              f"acc(题)={r1['acc_q_normal']:.3f} | loss={r1['final_loss']:.4f} "
              f"div={r1['diverged']}", flush=True)
        for li in r1["M1"]:
            d = r1["M1"][li]
            print(f"   L{li} M1 cos(Wq,p·Pk^T)={d['M1_cos_Wq_PkT']:+.4f} "
                  f"relF={d['M1_relF']:.4f} |norm|={d['M1_norm_ratio']:.3f}", flush=True)
        print(f"   M2 训练键 cos(q,k)={r1['M2_train_keys']}", flush=True)
        print(f"   M3 留出键 cos(q,k)={r1['M3_eval_keys']}", flush=True)
        print(f"   ★ M4 oracle-q：留出键 acc(项)={r1['acc_oracle']:.3f} "
              f"acc(题)={r1['acc_q_oracle']:.3f}", flush=True)
        print(f"     常规 q：    留出键 acc(项)={r1['acc_normal']:.3f} "
              f"acc(题)={r1['acc_q_normal']:.3f}", flush=True)
        print(f"     训练键：正常 q acc(题)={r1['acc_q_normal_trainkeys']:.3f} "
              f"vs oracle-q acc(题)={r1['acc_q_oracle_trainkeys']:.3f}", flush=True)
        dump(res, "probe91_identity.json")

    # ---- 汇总
    print("\n=== 汇总 ===", flush=True)
    for k, r in res["arms"].items():
        if "M1" not in r:
            continue
        m1 = np.mean([v["M1_cos_Wq_PkT"] for v in r["M1"].values()])
        print(f"  {k:<16} M1cos={m1:+.4f} M2={np.mean(list(r['M2_train_keys'].values())):+.4f} "
              f"M3={np.mean(list(r['M3_eval_keys'].values())):+.4f}", flush=True)
    dump(res, "probe91_identity.json")
    print("\n★ 诊断完成（不产生裁决）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())