#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 · rev 1.8 读数强度更正：**对齐量的测量对象修正**（诊断，不产生裁决）

前一轮（`probe91_stub.py`）的对齐读数有两处**测量对象错误**，本探针修正：
  (1) **RMSNorm 的逐通道增益 g**：`logit_t = <norm(x), e_t>` 且 `norm(x)=g*x/||x||`
      ⇒ `logit_t = <x, g ⊙ e_t>/||x||`。**因果量是 cos(Δ, g ⊙ e_t)，不是 cos(Δ, e_t)。**
      `g ⊙ e_t` 与 `e_t` 方向可差很多，故原读数须重算。
  (2) **归一化会去掉 Δ 沿 h 的分量**：一阶只剩 Δ_⊥h。故另报 cos(Δ_⊥h, g ⊙ e_ans)。
  (3) **零模型**：原用各向同性期望 sqrt(2/(pi*d))，但答案 embedding 之间 cos 均值 +0.18
      （池内词高度同簇）⇒ 各向同性零模型**不成立**。改报
      **配对边际** cos(Δ,g⊙e_ans) − cos(Δ,g⊙e_distr) 及其**标签置换零分布**。
  (4) **值池重复污染**：`VALUE_POOL` 仅 **27** 词而 `N_FACTS=32` ⇒ `i%27` 使 5 个事实
      复用前 5 个值；`A91-7` 只查池唯一性、**未查分配唯一性**。故按「该值是否在训练键中出现」
      分层报告（seen / unseen）。
  (5) **Δ 须实测**：用 hook 抓真正注入的张量，与解析式比对 rel.err（否则 0.13 无据）。

★ 只做测量，不改任何已冻结判据，不产生裁决。
"""
import argparse, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, N_FACTS, N_TRAIN_KEYS, build_chunks, build_facts, dump,
                       key_matrices, make_stack, model_and_tok, write_phase, VALUE_POOL)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GAMMA = 0.05
MOUNT = [8, 12, 16]


def cos(a, b):
    a = a.detach().float().reshape(-1); b = b.detach().float().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    order = list(range(N_FACTS)); random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]
    res = {"seed": a.seed, "value_pool_len": len(VALUE_POOL), "N_FACTS": N_FACTS}

    # ---- (4) 值分配唯一性
    assigned = [facts[i]["val"] for i in range(N_FACTS)]
    dup = {v: [i for i in range(N_FACTS) if assigned[i] == v]
           for v in set(assigned) if assigned.count(v) > 1}
    vals_train = {facts[i]["val"] for i in train_keys}
    seen = [i for i in eval_keys if facts[i]["val"] in vals_train]
    unseen = [i for i in eval_keys if facts[i]["val"] not in vals_train]
    res["value_contamination"] = {
        "n_distinct_assigned": len(set(assigned)),
        "duplicated_values": {k: v for k, v in dup.items()},
        "eval_keys_value_seen_in_train": seen,
        "n_eval_seen": len(seen), "n_eval_unseen": len(unseen),
        "A91-7_checks_pool_only": len(set(VALUE_POOL)) == len(VALUE_POOL),
    }
    print(f"[值污染] 池 {len(VALUE_POOL)} 词 / {N_FACTS} 事实 ⇒ 去重后仅 "
          f"{len(set(assigned))} 个不同值", flush=True)
    print(f"  重复值 = { {k: v for k, v in dup.items()} }", flush=True)
    print(f"  留出键中「值已在训练键出现」= {seen}（{len(seen)}/{len(eval_keys)}）", flush=True)
    print(f"  A91-7 只查池唯一性 = {res['value_contamination']['A91-7_checks_pool_only']}", flush=True)
    dump(res, "probe91_align2.json")

    # ---- 终层 norm 的逐通道增益 g
    g = None
    for nm, p in model.named_parameters():
        if nm.endswith("model.norm.weight") or nm == "model.norm.weight":
            g = p.detach().float().clone()
    assert g is not None, "找不到终层 RMSNorm 的 weight"
    res["norm_g"] = {"mean": float(g.mean()), "std": float(g.std()),
                     "min": float(g.min()), "max": float(g.max())}
    print(f"\n[终层 norm g] 均值={float(g.mean()):.4f} 标准差={float(g.std()):.4f} "
          f"范围=[{float(g.min()):.4f}, {float(g.max()):.4f}]", flush=True)

    # ---- 采集：hs_in 与「真正注入的张量」
    st = make_stack(gamma=GAMMA, no_write=False, seed=a.seed)
    write_phase(model, st, chunks)
    K, V = key_matrices(model, st, facts, chunks, owner)
    CAP = {}
    for li, m in st.mems.items():
        orig = m.read
        def wrap(hs_in, drop_sg=False, li=li, m=m, orig=orig):
            d = orig(hs_in, drop_sg)
            CAP[li] = (d.detach().float().clone() * float(m.gamma)).clone()
            return d
        m.read = wrap

    HS, ACTUAL, ANAL = {li: {} for li in MOUNT}, {li: {} for li in MOUNT}, {li: {} for li in MOUNT}
    for ki in eval_keys:
        f = facts[ki]
        ids = torch.tensor([f["prompt_ids"]], dtype=torch.long, device=DEV)
        with torch.no_grad():
            model(input_ids=ids)
        for li in MOUNT:
            HS[li][ki] = st._hs_in[li][0, len(f["prompt_ids"]) - 1, :].detach().float()
            ACTUAL[li][ki] = CAP[li][0, len(f["prompt_ids"]) - 1, :]
            m = st.mems[li]
            W = (m.W0 + m.U @ m.V.T).detach()
            q = (HS[li][ki] @ m.Pk.detach()) * float(m.pk_scale)
            ANAL[li][ki] = ((m.Pv.detach() @ (W @ q)) * float(m.gamma))

    def embed_of(tid):
        return model.get_input_embeddings().weight[torch.tensor([tid], device=DEV)][0].detach().float().clone()

    e_ans = {ki: embed_of(facts[ki]["val_ids"][0]) for ki in range(N_FACTS)}

    # ---- 重算对齐
    res["align"] = {}
    for li in MOUNT:
        verr = [float((ACTUAL[li][ki] - ANAL[li][ki]).norm() /
                      (ANAL[li][ki].norm() + 1e-30)) for ki in eval_keys]
        rows = []
        for ki in eval_keys:
            h, d = HS[li][ki], ACTUAL[li][ki]
            # Δ_⊥h：去掉沿 h 的分量（RMSNorm 后只保这一部分）
            hn = h / (h.norm() + 1e-30)
            dp = d - (d @ hn) * hn
            ge = g * e_ans[ki]                     # ★ 修正：g ⊙ e_t
            gd = torch.stack([g * e_ans[j] for j in eval_keys if j != ki]).mean(0)
            k = K[li][ki]
            W = (st.mems[li].W0 + st.mems[li].U @ st.mems[li].V.T).detach()
            d_tie = (st.mems[li].Pv.detach() @ (W @ k)) * float(st.mems[li].gamma)
            rows.append({
                "ki": ki, "is_unseen_value": ki in unseen,
                "rel": float(d.norm()) / (float(h.norm()) + 1e-30),
                "cos_e": cos(d, e_ans[ki]), "cos_ge": cos(d, ge),
                "cos_ge_dp": cos(dp, ge),
                "cos_ge_d": cos(d, gd),
                "m_ge": cos(d, ge) - cos(d, gd),
                "m_gedp": cos(dp, ge) - cos(dp, gd),
                "rel_tie_sameq": float(d_tie.norm()) / (float(h.norm()) + 1e-30),
            })
        agg = {k: float(np.mean([r[k] for r in rows])) for k in
               ("rel", "cos_e", "cos_ge", "cos_ge_dp", "cos_ge_d", "m_ge", "m_gedp",
                "rel_tie_sameq")}
        # 标签置换零分布（配对边际）
        rng = random.Random(2026)
        null = []
        for _ in range(400):
            perm = list(eval_keys); rng.shuffle(perm)
            ms = []
            for idx, ki in enumerate(eval_keys):
                h, d = HS[li][ki], ACTUAL[li][ki]
                hn = h / (h.norm() + 1e-30); dp = d - (d @ hn) * hn
                ge = g * e_ans[perm[idx]]
                gd = torch.stack([g * e_ans[perm[j]] for j in range(len(perm)) if j != idx]).mean(0)
                ms.append(cos(dp, ge) - cos(dp, gd))
            null.append(float(np.mean(ms)))
        null = np.array(null)
        agg["null_m_gedp_mean"] = float(null.mean()); agg["null_m_gedp_sd"] = float(null.std())
        agg["z_m_gedp"] = float((agg["m_gedp"] - null.mean()) / (null.std() + 1e-30))
        sub = [r for r in rows if r["is_unseen_value"]]
        agg["m_gedp_unseen_only"] = float(np.mean([r["m_gedp"] for r in sub])) if sub else None
        agg["n_unseen"] = len(sub)
        res["align"][li] = agg
        res["align"][li]["actual_vs_analytic_rel_err_max"] = float(np.max(verr))
        # 原口径（错）与修正后口径并列
        print(f"\n L{li}: ||h||={float(np.mean([float(HS[li][k].norm()) for k in eval_keys])):.1f} "
              f"||Δ||/||h||={agg['rel']:.4f}  (同 q 的 tie 臂 {agg['rel_tie_sameq']:.4f})", flush=True)
        print(f"   实测 vs 解析 rel.err 最大 = {float(np.max(verr)):.2e}", flush=True)
        print(f"   旧口径 cos(Δ, e_ans)          = {agg['cos_e']:+.4f}", flush=True)
        print(f"   ★修正 cos(Δ, g⊙e_ans)        = {agg['cos_ge']:+.4f}", flush=True)
        print(f"   ★修正 cos(Δ_⊥h, g⊙e_ans)     = {agg['cos_ge_dp']:+.4f}", flush=True)
        print(f"   ★配对边际 m(Δ_⊥h)            = {agg['m_gedp']:+.4f}  "
              f"零分布 {null.mean():+.4f}±{null.std():.4f}  z={agg['z_m_gedp']:+.2f}", flush=True)
        print(f"   仅未见值 (n={len(sub)}) 边际     = {agg['m_gedp_unseen_only']}", flush=True)
        dump(res, "probe91_align2.json")
    st.remove()
    print("\n★ 诊断完成（不产生裁决）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
