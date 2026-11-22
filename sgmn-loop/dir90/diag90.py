#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-90 · 事后诊断（**不是判据**）：内容为什么没存进去？
   ① k_i 之间的余弦（键是否近共线）
   ② **最佳线性可实现性**：ridge 解 min_W Σ||W k_i - v_i||² 的残差 —— 若残差本来就大，
      则任何写规则都不可能存住，零结果是**任务设计**问题而非 SGMN 问题。
"""
import json, pathlib, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
import R1_task90 as R   # 复用同一装置/事实集/块构造，逐字同一口径

tok, model = R.model_and_tok()
facts = R.build_facts(tok)
chunks, owner = R.build_chunks(tok, facts)
per = [19]*24 + [18]*8
order_pos, pos = {}, 0
for idx in range(len(facts)):
    order_pos[idx] = pos; pos += per[idx]

st = R.make_stack(gamma=R.GAMMA0, no_write=True)
K, V = {li: [] for li in st.mems}, {li: [] for li in st.mems}
for fi in range(len(facts)):
    ci = order_pos[fi] + 1
    model(input_ids=chunks[ci].to(R.DEV))
    for li, m in st.mems.items():
        hs = st._hs_in[li]
        pf = hs.reshape(-1, hs.shape[-1])
        K[li].append(((pf @ m.Pk).mean(dim=0)) * m.pk_scale)
        V[li].append((pf @ m.Pv).mean(dim=0))
st.remove()

out = {}
for li in K:
    Km = torch.stack(K[li]).float()            # (32, dk)
    Vm = torch.stack(V[li]).float()            # (32, dv)
    Kn = torch.nn.functional.normalize(Km, dim=1)
    C = (Kn @ Kn.T).abs()
    off = C[~torch.eye(len(Km), dtype=torch.bool, device=C.device)]
    # 最佳线性 W（ridge，最小二乘）：min ||W K^T - V^T||^2
    lam_r = 1e-6 * float((Km.T @ Km).diagonal().mean())
    G = Km.T @ Km + lam_r * torch.eye(Km.shape[1], device=Km.device)
    Wls = torch.linalg.solve(G, Km.T @ Vm).T    # (dv, dk)
    pred = Km @ Wls.T
    rel = ((pred - Vm).norm(dim=1) / Vm.norm(dim=1).clamp_min(1e-30))
    out[li] = {"cos_mean": float(off.mean()), "cos_min": float(off.min()),
               "cos_max": float(off.max()), "rank_K": int(torch.linalg.matrix_rank(Km)),
               "ls_rel_err_median": float(rel.median()), "ls_rel_err_max": float(rel.max()),
               "k_norm_mean": float(Km.norm(dim=1).mean()),
               "v_norm_mean": float(Vm.norm(dim=1).mean())}

# 装置实测 W 的 rel.err（与正式轮同口径，作为对照）
st2 = R.make_stack(gamma=R.GAMMA0)
R.write_phase(model, st2, chunks)
s90, _ = R.storage_diag(model, st2, facts, owner, chunks)
for li in out:
    Km, Vm = torch.stack(K[li]).float(), torch.stack(V[li]).float()
    pred = (R.W_of(st2.mems[li]).float() @ Km.T).T
    rel = ((pred - Vm).norm(dim=1) / Vm.norm(dim=1).clamp_min(1e-30))
    out[li]["device_rel_err_median"] = float(rel.median())
st2.remove()
out["_s90_median"] = s90
print(json.dumps(out, ensure_ascii=False, indent=1))
(pathlib.Path(__file__).resolve().parent / "out" / "diag.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
