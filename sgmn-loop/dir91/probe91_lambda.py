#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-91 · 决定性诊断（**不是判据**）：装置达到的 rel.err=0.556 由**什么**决定？
   两个竞争机制：
     (M1) 键近共线 ⇒ 精确解要求巨大 ||W|| ⇒ 装置的 λ=0.05 把解**衰减掉**（平衡点问题）
     (M2) 写入预算不足 ⇒ 更多写入会更好（**已被 probe91_storage 否证**：64→600 全平）
   做法：同一个 (K,V) 上做 ridge，扫描 λ，看 rel.err 怎么随 λ 变。
        若 λ=0.05 处 rel.err ≈ 0.55 ⇒ M1 成立（λ 通过病态性起作用，**不是**通过每步衰减）。
        同时给 K 的奇异值谱（条件数），量化病态程度。
"""
import json, pathlib, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
import numpy as np, torch
import R2_task91 as R

tok, model = R.model_and_tok()
facts = R.build_facts(tok)
chunks, owner = R.build_chunks(tok, facts)

st = R.make_stack(gamma=0.0, no_write=True)
try:
    K, V = R.key_matrices(model, st, facts, chunks, owner)
finally:
    st.remove()

out = {}
for li in sorted(K):
    Km, Vm = K[li], V[li]
    sc = float((Km.T @ Km).diagonal().mean())          # 尺度 = tr(KᵀK)/d
    sv = torch.linalg.svdvals(Km)
    per = {}
    for lam in (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 0.5, 5.0):
        G = Km.T @ Km + (lam * sc) * torch.eye(Km.shape[1], device=Km.device)
        Wls = torch.linalg.solve(G, Km.T @ Vm).T
        rel = ((Km @ Wls.T) - Vm).norm(dim=1) / Vm.norm(dim=1).clamp_min(1e-30)
        per[f"lambda_{lam}"] = {"rel_med": float(rel.median()),
                                "W_norm": float(Wls.norm())}
    out[li] = {"lambda_sweep": per,
               "sv_max": float(sv.max()), "sv_min": float(sv.min()),
               "cond": float(sv.max() / sv.min()),
               "k_norm_mean": float(Km.norm(dim=1).mean()),
               "v_norm_mean": float(Vm.norm(dim=1).mean()),
               "scale_trKK_over_d": sc}
    print(f"L{li} cond={out[li]['cond']:.3e} |k|={out[li]['k_norm_mean']:.1f} "
          f"|v|={out[li]['v_norm_mean']:.1f}", flush=True)
    for lam in (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 0.5, 5.0):
        q = per[f"lambda_{lam}"]
        print(f"   λ={lam:<8g} rel.err={q['rel_med']:.4e}  ‖W‖={q['W_norm']:.3e}", flush=True)

(pathlib.Path(__file__).resolve().parent / "out" / "probe91_lambda.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
