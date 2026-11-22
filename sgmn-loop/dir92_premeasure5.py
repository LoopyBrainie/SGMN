#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
r"""DIR-92 · **开卡前测量 ⑤**（**不是判据**）：给出一对**同口径**（装置口径，含 `pk_scale`）的
   存储读数与随机基线，用以更正 `D-109`/`A91-17`。两种语料都测，确认语料无影响。
   统计量同时报两种，避免再犯"口径不同却相减"的错：
     (i) 每层 fact 中位数，再跨层平均；(ii) `dir90.storage_diag` 原式（fact 中位数的层均值）。
"""
import json, pathlib, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir90")
import numpy as np, torch
import R2_task91 as R91
import R1_task90 as R90

def dev_kv(model, st, facts, chunks, owner):
    """**装置口径**：k 含 pk_scale（与 `sgmn_module.py:86` 一致）；v 同。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    K = {li: [] for li in st.mems}; V = {li: [] for li in st.mems}
    try:
        for fi in range(len(facts)):
            ci = owner.index(fi) + 1
            model(input_ids=chunks[ci].to(R91.DEV))
            for li, m in st.mems.items():
                hs = st._hs_in[li]
                pf = hs.reshape(-1, hs.shape[-1])
                K[li].append((((pf @ m.Pk).mean(dim=0)) * m.pk_scale).detach())
                V[li].append(((pf @ m.Pv).mean(dim=0)).detach())
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
    return ({li: torch.stack(K[li]).float() for li in K},
            {li: torch.stack(V[li]).float() for li in V})

def both_stats(K, V, W):
    per_layer, per_fact = {}, []
    for li in sorted(K):
        rel = ((K[li] @ W[li].T) - V[li]).norm(dim=1) / V[li].norm(dim=1).clamp_min(1e-30)
        per_layer[li] = float(rel.median())
    for fi in range(K[sorted(K)[0]].shape[0]):
        ls = [float(((K[li][fi] @ W[li].T) - V[li][fi]).norm() / V[li][fi].norm().clamp_min(1e-30))
              for li in sorted(K)]
        per_fact.append(float(np.mean(ls)))
    return {"i_perlayer_median_then_mean": float(np.mean(list(per_layer.values()))),
            "ii_factmedian_of_layermean": float(np.median(per_fact)),
            "per_layer": per_layer}

out = {"note": "pre-card measurement; device-faithful (k includes pk_scale); NOT a criterion"}
for tag in ("V0", "V3"):
    tok, model = R91.model_and_tok()
    if tag == "V0":
        facts = R90.build_facts(tok); chunks, owner = R90.build_chunks(tok, facts)
    else:
        facts = R91.build_facts(tok); chunks, owner = R91.build_chunks(tok, facts)
    stW = R91.make_stack(gamma=0.0, no_write=False)
    stN = R91.make_stack(gamma=0.0, no_write=True)
    try:
        R91.write_phase(model, stW, chunks)
        KW, VW = dev_kv(model, stW, facts, chunks, owner)
        KN, VN = dev_kv(model, stN, facts, chunks, owner)
        Ww = {li: R91.W_of(stW.mems[li]).float() for li in stW.mems}
        Wn = {li: R91.W_of(stN.mems[li]).float() for li in stN.mems}
        sw, sb = both_stats(KW, VW, Ww), both_stats(KN, VN, Wn)
        kn = {li: float(KW[li].norm(dim=1).mean()) for li in KW}
        out[tag] = {"stored": sw, "baseline_W0": sb, "dev_k_norm_mean": kn,
                    "dev_eta_k2": {li: float(R91.ETA_WRITE * kn[li] ** 2) for li in kn},
                    "improvement_i": sb["i_perlayer_median_then_mean"] / sw["i_perlayer_median_then_mean"],
                    "improvement_ii": sb["ii_factmedian_of_layermean"] / sw["ii_factmedian_of_layermean"]}
        print(f"\n=== {tag}（装置口径）===", flush=True)
        print(f"  ||k|| = " + "/".join(f"{kn[li]:.2f}" for li in sorted(kn)), flush=True)
        print(f"  已写存储 (i)={sw['i_perlayer_median_then_mean']:.4f}  (ii)={sw['ii_factmedian_of_layermean']:.4f}", flush=True)
        print(f"  基线 W0  (i)={sb['i_perlayer_median_then_mean']:.4f}  (ii)={sb['ii_factmedian_of_layermean']:.4f}", flush=True)
        print(f"  改善倍数 (i)={out[tag]['improvement_i']:.2f}×  (ii)={out[tag]['improvement_ii']:.2f}×", flush=True)
    finally:
        stW.remove(); stN.remove()

print("\n=== 汇总 ===", flush=True)
print(json.dumps({t: {k: out[t][k] for k in ("improvement_i", "improvement_ii")} for t in ("V0", "V3")},
                 ensure_ascii=False, indent=1), flush=True)
(pathlib.Path(__file__).resolve().parent / "dir92" / "out" / "premeasure5.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
