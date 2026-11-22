#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-18 · probe18g —— **反平凡性检验**：probe18f 的高保真度是「真存储」还是「平凡高分」？

【probe18f 的结果（装置层，单文档）】
    renorm(线性键归一化到 ‖k‖²=5) η=0.1 : fid_mean=0.76  cos_read=0.78
    ortho(正交键，‖k‖²=5)        η=0.2 : fid_mean=0.9994 cos_read=0.9999
  ⇒ 写规则 §3.1/§3.2 **本身是好的**；此前失败是**键尺度**问题（装置 `pk_scale` 用随机
    N(0,1) 标定，而真实 hidden state 范数大得多 ⇒ 实际 ‖k‖≈136–350 而非 √5）。

【但必须排除一个致命混淆】若同一文档内各槽的**值** $v_j=P_v h_t$ **本来就彼此高度相似**
  （相邻位置的 hidden state 本就相似）⇒ 那么「$W$ 学会输出那个公共方向」也能拿到
  cos(W k_j, v_j) ≈ 1 ⇒ **高 fid 是平凡的**，不证明「按槽分别存储」。
  ⇒ ⇒ **高 fid 本身不足以支撑「写规则可用」的结论。**

【本探针的三项判别】
  ① **值的平凡性**：同一文档内 $v_j$ 与 $v_{j'}$（$j\ne j$）的余弦分布
     —— 若 off-diag 余弦 ≈ 0.9+ ⇒ 本任务下「按槽存储」几乎是不可观测的。
  ② **交叉保真矩阵**：$C_{jj'}=\cos(Wk_j,\;v_{j'})$
     —— 真正的按槽存储 ⇒ **对角显著高于非对角**（且非对角 ≈ 值间相似度）。
  ③ **置换对照（null）**：把值 $v$ 随机打乱后写入（键不变）⇒ 若打乱后 fid 仍高
     ⇒ ⇒ **「高 fid」与「存对内容」无关** ⇒ 平凡。

【附加：协议尺度】真实协议 = 每篇 68 条写入、连续多篇写进**同一个 W**
  ⇒ 测「写完 doc B 后，doc A 的保真度掉多少」⇒ 多篇干扰代价。

用法：/opt/pytorch/bin/python -u probe18g.py
"""
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

D = "/home/ec2-user/sgmn-loop/dir18"
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("r1w", D + "/R1_write.py")
r1w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1w)
spec2 = importlib.util.spec_from_file_location("hebb", D + "/sgmn_module_hebb.py")
hebb = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(hebb)

OUT = pathlib.Path(D) / "probe18g_out.json"
rep = {}


def log(*a):
    print(*a, flush=True)


def make_keys(hs, Pk, mode, scale_to=5.0, Q=None):
    L = hs.shape[0]
    keys = []
    for t in range(1, L):
        if mode == "ortho":
            k = Q[:, (t - 1) % Q.shape[1]].clone()
        else:
            k = (hs[t - 1].reshape(1, -1) @ Pk).mean(dim=0).detach()
        if mode in ("renorm", "ortho"):
            k = k * (math.sqrt(scale_to) / float(k.norm().clamp_min(1e-30)))
        keys.append(k)
    return keys


def vals_of(hs, Pv):
    return [(hs[t].reshape(1, -1) @ Pv).mean(dim=0).detach() for t in range(1, hs.shape[0])]


def cross_matrix(W, keys, vals):
    """C[j, j'] = cos(W k_j, v_j')"""
    A = torch.stack([W @ k for k in keys])          # (n, d_v)
    B = torch.stack(vals)                            # (n, d_v)
    A = A / A.norm(dim=1).clamp_min(1e-30).unsqueeze(1)
    B = B / B.norm(dim=1).clamp_min(1e-30).unsqueeze(1)
    return (A @ B.T).cpu().numpy()


def value_sim(vals):
    B = torch.stack(vals)
    B = B / B.norm(dim=1).clamp_min(1e-30).unsqueeze(1)
    C = (B @ B.T).cpu().numpy()
    n = C.shape[0]
    off = C[~np.eye(n, dtype=bool)]
    return {"v_offdiag_mean": float(off.mean()), "v_offdiag_max": float(off.max()),
            "v_offdiag_min": float(off.min()),
            "v_offdiag_p90": float(np.percentile(off, 90))}


def analyze(tag, W, keys, vals):
    C = cross_matrix(W, keys, vals)
    n = C.shape[0]
    diag = np.diag(C)
    off = C[~np.eye(n, dtype=bool)]
    out = {"fid_diag_mean": float(diag.mean()), "fid_diag_min": float(diag.min()),
           "cross_offdiag_mean": float(off.mean()), "cross_offdiag_max": float(off.max()),
           "diag_minus_offdiag": float(diag.mean() - off.mean())}
    out.update(value_sim(vals))
    out["triviality_ratio"] = out["cross_offdiag_mean"] / max(out["v_offdiag_mean"], 1e-12)
    log(f"  {tag}: diag={out['fid_diag_mean']:+.4f}(min {out['fid_diag_min']:+.4f}) "
        f"cross_offdiag={out['cross_offdiag_mean']:+.4f} "
        f"(对角−非对角={out['diag_minus_offdiag']:+.4f}) | "
        f"值间相似 offdiag_mean={out['v_offdiag_mean']:+.4f} max={out['v_offdiag_max']:+.4f} "
        f"⇒ 平凡性比={out['triviality_ratio']:.3f}")
    return out


def main():
    tok = AutoTokenizer.from_pretrained(r1w.MODEL)
    model = AutoModelForCausalLM.from_pretrained(r1w.MODEL, dtype=torch.float32).to(r1w.DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = r1w.DEV
    d_model = model.config.hidden_size
    d_k = d_v = d_model
    alpha, lam = 1.0, 0.05
    tr, ev = r1w.build_docs(tok, 20261028, 8, 2)

    seed = 20261028
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    Pk = (torch.randn(d_model, d_k, generator=gen) / math.sqrt(d_model)).to(dev)
    Pv = (torch.randn(d_model, d_v, generator=gen) / math.sqrt(d_model)).to(dev)

    # 抓每层的真实 hidden states（用装置挂载跑一遍，取 _hs_in）
    st = r1w.make_stack(model, d_model, "cue", seed=seed, mem_kind="hebbian_full", eta=1e-7)
    with torch.no_grad():
        ids = torch.tensor(tr[0]["ids"], dtype=torch.long, device=dev).unsqueeze(0)
        st.model(input_ids=ids)
        hs_layers = {li: st._hs_in[li][0].detach().clone() for li in st.mems}
    st.remove(); torch.cuda.empty_cache()

    g = torch.Generator(device="cpu"); g.manual_seed(1234)
    A = torch.randn(d_model, d_model, generator=g)
    Q, _ = torch.linalg.qr(A)
    Q = Q.to(dev)

    for li, hs in hs_layers.items():
        log(f"\n=== 层 {li} ===")
        vals = vals_of(hs, Pv)
        log(f"  [① 值的平凡性] {value_sim(vals)}")
        for mode, eta in (("renorm", 0.1), ("ortho", 0.2), ("renorm", 0.01), ("ortho", 0.1)):
            keys = make_keys(hs, Pk, mode, Q=Q)
            # —— 正常写入
            gen2 = torch.Generator(device="cpu"); gen2.manual_seed(seed * 1000 + li)
            core = hebb.HebbianCore(d_k, d_v, alpha, lam, eta, gen2, dev,
                                    factorized=False, n_entries=128, d_model_sig=d_model)
            with torch.no_grad():
                for j in range(len(keys)):
                    core.write(keys[j], vals[j], sig=hs[j].detach())
            rep[f"{li}_{mode}_eta{eta:g}"] = analyze(f"正常写 {mode} η={eta:g}", core.W, keys, vals)
            # —— ③ 置换对照：值随机打乱
            gperm = torch.Generator(device="cpu"); gperm.manual_seed(999 + li)
            perm = torch.randperm(len(vals), generator=gperm).tolist()
            vals_p = [vals[i] for i in perm]
            gen3 = torch.Generator(device="cpu"); gen3.manual_seed(seed * 1000 + li)
            core2 = hebb.HebbianCore(d_k, d_v, alpha, lam, eta, gen3, dev,
                                     factorized=False, n_entries=128, d_model_sig=d_model)
            with torch.no_grad():
                for j in range(len(keys)):
                    core2.write(keys[j], vals_p[j], sig=hs[j].detach())
            rep[f"{li}_{mode}_eta{eta:g}_PERM"] = analyze(
                f"★置换 {mode} η={eta:g}", core2.W, keys, vals)   # ★ 用**真值**评 ⇒ 打乱后应崩

    # —— ④ 协议尺度：多篇连续写入同一 W，看干扰
    log("\n=== 协议尺度：连续多篇写入同一 W（每篇 34 条） ===")
    for mode, eta in (("renorm", 0.1), ("ortho", 0.2)):
        keys_all, vals_all, hs_all = [], [], []
        for d in tr[:6]:
            with torch.no_grad():
                ids = torch.tensor(d["ids"], dtype=torch.long, device=dev).unsqueeze(0)
                st2 = r1w.make_stack(model, d_model, "cue", seed=seed, mem_kind="hebbian_full", eta=eta)
                st2.model(input_ids=ids)
                hs_all.append(st2._hs_in[8][0].detach().clone())
                st2.remove(); torch.cuda.empty_cache()
        gen4 = torch.Generator(device="cpu"); gen4.manual_seed(seed)
        core = hebb.HebbianCore(d_k, d_v, alpha, lam, eta, gen4, dev,
                                factorized=False, n_entries=128, d_model_sig=d_model)
        ks0 = vs0 = None
        for di, h in enumerate(hs_all):
            ks = make_keys(h, Pk, mode, Q=Q)
            vs = vals_of(h, Pv)
            with torch.no_grad():
                for j in range(len(ks)):
                    core.write(ks[j], vs[j], sig=h[j].detach())
            if di == 0:
                ks0, vs0 = ks, vs
                rep[f"multi_{mode}_eta{eta:g}_doc0"] = analyze(
                    f"{mode} η={eta:g} 写完 doc0", core.W, ks, vs)
            else:
                # 干扰：doc0 的保真度还剩多少
                rep[f"multi_{mode}_eta{eta:g}_after{di+1}docs"] = analyze(
                    f"{mode} η={eta:g} 写完 {di+1} 篇后回看 doc0", core.W, ks0, vs0)

    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n写出 {OUT}")


if __name__ == "__main__":
    main()
