#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-90 · 事后诊断 B（**不是判据**）：把 S90 的**污染**去掉。
   写入规则与 γ 无关 ⇒ 取 γ=0（前向无扰动、读路径未训练）时量 ||W k_i - v_i|| / ||v_i||，
   这才是"存没存进去"的干净读数；并给出 no_write（W=W0）基线以核卡片"2× 改善"条款。
   同时报告**线性可实现性下限**（ridge 最优 W 的残差）作为精度参照。
"""
import json, pathlib, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop")
import torch
import R1_task90 as R

tok, model = R.model_and_tok()
facts = R.build_facts(tok)
chunks, owner = R.build_chunks(tok, facts)

def k_v(st):
    K, V = {li: [] for li in st.mems}, {li: [] for li in st.mems}
    for fi in range(len(facts)):
        ci = owner.index(fi) + 1          # ★ 直接用 owner 定位该事实 run 的第 2 块
        model(input_ids=chunks[ci].to(R.DEV))
        for li, m in st.mems.items():
            hs = st._hs_in[li]; pf = hs.reshape(-1, hs.shape[-1])
            K[li].append(((pf @ m.Pk).mean(dim=0)) * m.pk_scale)
            V[li].append((pf @ m.Pv).mean(dim=0))
    return K, V

out = {}
for tag, gamma, no_write in (("written", 0.0, False), ("baseline", 0.0, True)):
    st = R.make_stack(gamma=gamma, no_write=no_write)
    if not no_write:
        R.write_phase(model, st, chunks)
    K, V = k_v(st)
    per = {}
    for li, m in st.mems.items():
        Km, Vm = torch.stack(K[li]).float(), torch.stack(V[li]).float()
        W = R.W_of(m).float()
        rel = ((Km @ W.T) - Vm).norm(dim=1) / Vm.norm(dim=1).clamp_min(1e-30)
        Kn = torch.nn.functional.normalize(Km, dim=1); C = (Kn @ Kn.T).abs()
        off = C[~torch.eye(len(Km), dtype=torch.bool, device=C.device)]
        lam_r = 1e-6 * float((Km.T @ Km).diagonal().mean())
        G = Km.T @ Km + lam_r * torch.eye(Km.shape[1], device=Km.device)
        Wls = torch.linalg.solve(G, Km.T @ Vm).T
        rel_ls = ((Km @ Wls.T) - Vm).norm(dim=1) / Vm.norm(dim=1).clamp_min(1e-30)
        per[li] = {"rel_err_median": float(rel.median()),
                   "ls_floor_median": float(rel_ls.median()),
                   "cos_mean_abs": float(off.mean()),
                   "k_norm": float(Km.norm(dim=1).mean()),
                   "v_norm": float(Vm.norm(dim=1).mean())}
    out[tag] = per
    st.remove()

med = {t: float((sum(v["rel_err_median"] for v in out[t].values())) / len(out[t])) for t in out}
out["_median_written"] = med["written"]
out["_median_baseline"] = med["baseline"]
out["_improve_ratio"] = med["baseline"] / max(med["written"], 1e-30)
print(json.dumps(out, ensure_ascii=False, indent=1))
(pathlib.Path(__file__).resolve().parent / "out" / "diagB.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
