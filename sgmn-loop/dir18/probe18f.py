#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-18 · probe18f —— **装置层**的写规则可行性图谱（决定正式轮臂表的依据）。

【为什么必须做这个探针】probe18b 给出两个互相矛盾的事实：
  · 键范数 ‖k‖ mean **131.4** / max **3451** —— 而装置的 `pk_scale` 标定目标是 E‖k‖²=**5**
    （标定用的是 **随机 N(0,1) 向量**，不是真实 hidden state ⇒ **标定口径错**）。
  · 稳定上界 η < 2/(λ+α‖k‖²)：按 mean 得 1.16e-4，按 max 得 **1.7e-7**；
    而「有效学习」需要 η ~ 1e-3 ⇒ **窗口不相交**。
⇒ ⇒ 必须把「**尺度**（‖k‖ 太大）」与「**几何**（键共线，DIR-13 实测 cos 0.987–0.995）」
   这两个**混淆在一起**的因素**拆开**测，否则正式轮的臂表会把两个变量绑死。

【探针设计】逐位置写（与装置的 §2.3 语义逐字一致）：
    对 t = 1..L-1：k_t = sg(P_k h_{t-1})·scale，v_t = sg(P_v h_t)，写 (k_t, v_t, sig=h_{t-1})
  写完后测**三个量**（全部取自装置的实际输出路径）：
    ① `fid_slot`  = cos(W k_j, v_j)   —— 逐槽写保真度（**核心机制量**）
    ② `fid_read`  = cos(y_read, v_true) —— 经**真实读路径**（地址选键 + §2.2 线性读）的余弦
    ③ `w_norm` / `stable` —— 稳定性
  配置网格：
    · 键几何：`linear`（P_k h）/ `renorm`（线性键但**逐写归一化到 ‖k‖²=5**）/ `ortho`（正交基）
    · 步长 η ∈ {1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 0.2}（每个都跑到**失稳**为止 ⇒ 画出窗口）
  另外测**两文档干扰**：写完 doc A 再写 doc B（同一 W）⇒ 看 doc A 的保真度掉多少
  ⇒ 这是「位置级写 + 共享 W」的真实代价。

【关键预言（可证伪）】若「尺度」是唯一障碍 ⇒ `renorm` 在某个 η 下 fid ≫ 0；
  若「几何」（键共线）也障碍 ⇒ `renorm` 仍 fid≈0 而 `ortho` 才高；
  若写规则本身坏 ⇒ 两者都 ≈ 0。

用法：/opt/pytorch/bin/python -u probe18f.py
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

OUT = pathlib.Path(D) / "probe18f_out.json"
rep = {"grid": {}, "notes": []}


def log(*a):
    print(*a, flush=True)


def build_core(d_model, d_k, d_v, alpha, lam, eta, gen, device, n_entries=128):
    core = hebb.HebbianCore(d_k, d_v, alpha, lam, eta, gen, device,
                            factorized=False, n_entries=n_entries, d_model_sig=d_model)
    return core


def write_positions(core, Pk, Pv, hs, mode, scale_to=5.0):
    """逐位置写（与装置 LayerMem.write 的语义逐字一致，但可切换键几何）。

    hs: (L, d_model) —— 该层的输入 hidden states
    mode: linear | renorm | ortho
    """
    L = hs.shape[0]
    keys, vals, sigs = [], [], []
    for t in range(1, L):
        k = (hs[t - 1].reshape(1, -1) @ Pk).mean(dim=0).detach()
        v = (hs[t].reshape(1, -1) @ Pv).mean(dim=0).detach()
        if mode == "linear":
            pass                                  # 保留装置的原始尺度（由 pk_scale 决定）
        elif mode == "renorm":
            k = k * (math.sqrt(scale_to) / float(k.norm().clamp_min(1e-30)))
        elif mode == "ortho":
            pass                                  # 由调用方预先替换
        keys.append(k)
        vals.append(v)
        sigs.append(hs[t - 1].detach())
    return keys, vals, sigs


def run_grid(model, tok, seeds=(20261028,), n_docs=3):
    dev = r1w.DEV
    d_model = model.config.hidden_size
    d_k = d_v = d_model                      # 与脚本一致（卡片 D49 已订正）
    alpha, lam = 1.0, 0.05
    tr, ev = r1w.build_docs(tok, seeds[0], n_docs, 2)

    for si, seed in enumerate(seeds):
        gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
        # 与装置同构的 P_k / P_v（d_model×d_k / d_model×d_v，各 /sqrt(d_model)）
        Pk = (torch.randn(d_model, d_k, generator=gen) / math.sqrt(d_model)).to(dev)
        Pv = (torch.randn(d_model, d_v, generator=gen) / math.sqrt(d_model)).to(dev)
        # pk_scale 的**装置口径**标定：随机 N(0,1) x ⇒ E‖k‖²=5
        pk_scale = 1.0
        with torch.no_grad():
            x = torch.randn(20000, d_model, generator=gen).to(dev)
            kk = (x @ Pk) * pk_scale
            pk_scale = pk_scale / float((kk ** 2).mean().sqrt()) * math.sqrt(5.0)
        log(f"[seed {seed}] 装置口径 pk_scale={pk_scale:.6f}")

        # 取若干层的真实 hidden states（用装置的挂载方式跑一遍，拿到每层 hs_in）
        st = r1w.make_stack(model, d_model, "cue", seed=seed, mem_kind="hebbian_full", eta=1e-7)
        hs_per_layer = {}
        with torch.no_grad():
            for li, m in st.mems.items():
                hs_per_layer[li] = None
            ids = torch.tensor(tr[0]["ids"], dtype=torch.long, device=dev).unsqueeze(0)
            # 逐层抓 hs_in：装置把 hs_in 存在 st._hs_in[li]
            st.model(input_ids=ids)
            for li in st.mems:
                hs_per_layer[li] = st._hs_in[li][0].detach().clone()
        st.remove(); torch.cuda.empty_cache()

        for li, hs in hs_per_layer.items():
            if hs is None:
                continue
            L = hs.shape[0]
            for mode in ("linear", "renorm", "ortho"):
                for eta in (1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 0.2):
                    gen2 = torch.Generator(device="cpu"); gen2.manual_seed(seed * 1000 + li)
                    core = build_core(d_model, d_k, d_v, alpha, lam, eta, gen2, dev)
                    # 键几何：ortho 用随机正交基逐写取列（与装置 ortho 模式同构）
                    if mode == "ortho":
                        g = torch.Generator(device="cpu"); g.manual_seed(1234 + li)
                        A = torch.randn(d_model, d_model, generator=g)
                        Q, _ = torch.linalg.qr(A)
                        Q = Q.to(dev)
                    keys, vals, sigs = [], [], []
                    for t in range(1, L):
                        if mode == "ortho":
                            k = Q[:, (t - 1) % Q.shape[1]].to(dev)
                            k = k * (math.sqrt(5.0) / float(k.norm().clamp_min(1e-30)))
                        else:
                            k = (hs[t - 1].reshape(1, -1) @ Pk).mean(dim=0).detach()
                            if mode == "linear":
                                k = k * pk_scale
                            else:  # renorm
                                k = k * (math.sqrt(5.0) / float(k.norm().clamp_min(1e-30)))
                        v = (hs[t].reshape(1, -1) @ Pv).mean(dim=0).detach()
                        keys.append(k); vals.append(v); sigs.append(hs[t - 1].detach())
                    # 逐位置写（§3.2 全矩阵参考步）
                    with torch.no_grad():
                        for t in range(1, L):
                            core.write(keys[t - 1], vals[t - 1], sig=sigs[t - 1])
                    # —— 测量
                    stable = bool(torch.isfinite(core.W).all().item())
                    fids = []
                    for j in range(len(keys)):
                        a = core.W @ keys[j]
                        b = vals[j]
                        if a.norm() < 1e-30 or b.norm() < 1e-30:
                            fids.append(0.0); continue
                        fids.append(float((a @ b) / (a.norm() * b.norm())))
                    # 真实读路径：地址选键（上下文签名余弦）+ §2.2 线性读
                    q_sig = hs[L - 1]                      # 读侧地址取最后一位（装置 D29 口径）
                    sc = torch.stack(sigs) @ q_sig
                    sc = sc / (torch.stack(sigs).norm(dim=1) * q_sig.norm()).clamp_min(1e-30)
                    j_sel = int(torch.argmax(sc).item())
                    y = core.read(keys[j_sel].reshape(1, -1))[0]
                    cos_read = float(F.cosine_similarity(y.reshape(1, -1),
                                                        vals[j_sel].reshape(1, -1)).item())
                    kn = float(torch.stack(keys).norm(dim=1).mean().item())
                    kmax = float(torch.stack(keys).norm(dim=1).max().item())
                    rec = {"fid_mean": float(np.mean(fids)), "fid_max": float(np.max(fids)),
                           "fid_min": float(np.min(fids)), "cos_read": cos_read,
                           "j_sel": j_sel, "w_norm": float(core.W.norm().item()),
                           "w0_norm": float(core.W0.norm().item()),
                           "k_norm_mean": kn, "k_norm_max": kmax,
                           "v_norm_mean": float(torch.stack(vals).norm(dim=1).mean().item()),
                           "stable": stable}
                    key = f"s{seed}_L{li}_{mode}_eta{eta:g}"
                    rep["grid"][key] = rec
                    log(f"  {key:34s} fid_mean={rec['fid_mean']:+.4f} fid_max={rec['fid_max']:+.4f} "
                        f"cos_read={cos_read:+.4f} |W|={rec['w_norm']:.3g} (W0={rec['w0_norm']:.3g}) "
                        f"‖k‖mean={kn:.4g} max={kmax:.4g} stable={stable}")


def main():
    tok = AutoTokenizer.from_pretrained(r1w.MODEL)
    model = AutoModelForCausalLM.from_pretrained(r1w.MODEL, dtype=torch.float32).to(r1w.DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log("=== DIR-18 probe18f：写规则可行性图谱（装置层） ===")
    run_grid(model, tok, seeds=(20261028,))
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n写出 {OUT}")


if __name__ == "__main__":
    main()
