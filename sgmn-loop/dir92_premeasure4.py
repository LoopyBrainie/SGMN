#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
r"""DIR-92 · **开卡前测量 ④**（**不是判据**）：A=0.1995（dir90.storage_diag）与 C=0.5598（我的口径）
   在同一 stack 上分歧 2.8×。逐元素比 K/V，定位到底是 k 不同、v 不同，还是 W 不同。
"""
import json, pathlib, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir90")
import numpy as np, torch
import R2_task91 as R91
import R1_task90 as R90

tok, model = R91.model_and_tok()
facts = R90.build_facts(tok)
chunks, owner = R90.build_chunks(tok, facts)
st = R91.make_stack(gamma=0.0, no_write=False)
try:
    R91.write_phase(model, st, chunks)

    # ---- 路径 A：逐字复刻 storage_diag 的内层，并在同一 forward 里同时收 k、v、pred ----
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    A = {li: {"k": [], "v": [], "pred": []} for li in st.mems}
    for fi in range(len(facts)):
        ci = owner.index(fi) + 1
        model(input_ids=chunks[ci].to(R91.DEV))
        for li, m in st.mems.items():
            hs = st._hs_in[li]
            pf = hs.reshape(-1, hs.shape[-1])
            hf = hs.reshape(-1, hs.shape[-1])       # 与 storage_diag 一致
            k = ((pf @ m.Pk).mean(dim=0)) * m.pk_scale
            v = (hf @ m.Pv).mean(dim=0)
            A[li]["k"].append(k.detach().float())
            A[li]["v"].append(v.detach().float())
            A[li]["pred"].append((R90.W_of(m) @ k).detach().float())
    for li, m in st.mems.items():
        m.no_write = saved[li]

    # ---- 路径 B：key_matrices ----
    K, V = R91.key_matrices(model, st, facts, chunks, owner)

    out = {}
    for li in sorted(st.mems):
        kA = torch.stack(A[li]["k"]); vA = torch.stack(A[li]["v"]); pA = torch.stack(A[li]["pred"])
        kB, vB = K[li].float(), V[li].float()
        W = R91.W_of(st.mems[li]).float()
        pB = kB @ W.T
        errA = ((pA - vA).norm(dim=1) / vA.norm(dim=1).clamp_min(1e-30))
        errB = ((pB - vB).norm(dim=1) / vB.norm(dim=1).clamp_min(1e-30))
        out[f"L{li}"] = {
            "dK_max_abs": float((kA - kB).abs().max()),
            "dV_max_abs": float((vA - vB).abs().max()),
            "dPred_max_abs": float((pA - pB).abs().max()),
            "errA_med": float(errA.median()), "errB_med": float(errB.median()),
            "kA_norm_mean": float(kA.norm(dim=1).mean()),
            "kB_norm_mean": float(kB.norm(dim=1).mean()),
            # k 与 v 的相关性：若 k,v 由**同一块** hs 得出，二者高度相关 -> 存储"容易"
            "cos_kA_vA_mean": float(torch.nn.functional.cosine_similarity(kA, vA, dim=1).mean()),
        }
        q = out[f"L{li}"]
        print(f"L{li}: dK={q['dK_max_abs']:.3e} dV={q['dV_max_abs']:.3e} "
              f"dPred={q['dPred_max_abs']:.3e} | errA={q['errA_med']:.4f} errB={q['errB_med']:.4f} "
              f"| |kA|={q['kA_norm_mean']:.2f} |kB|={q['kB_norm_mean']:.2f} "
              f"| cos(k,v)={q['cos_kA_vA_mean']:.5f}", flush=True)
    (pathlib.Path(__file__).resolve().parent / "dir92" / "out" / "premeasure4.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
finally:
    st.remove()
