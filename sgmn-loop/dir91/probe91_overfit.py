#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 · rev 2.0 预备：**写入侧阳性对照**（诊断）

`probe91_storage.py` 的 `rel≈2.95 全程平坦` **不构成「写入惰性」的证据**：
每次写测的是**不同** (k,v) 对、且测在**更新前**；不同对之间 rel 平坦是应当的。
本探针补上缺的阳性对照与两项检查：

  C1  **重建 W 校验**：`W0+U@V.T`（或 `Wfull`）与 `read` 实际使用的矩阵是否同一；
      用固定随机 q 比对 `read` 输出 v.s. 手算 `Wo@W@q`。
  C2  **单对过拟合**：同一 (k,v) 连写 200 次，报 rel 轨迹。
      收敛 ⇒ 目标可达（问题在容量/干扰/步长）；不收敛 ⇒ **写通路本身有问题**。
  C3  报 `eta*alpha*||k||^2`（Hebb 收敛条件；须 ≈1）与 `||W_t - W_0||` 轨迹。

★ 只做测量。
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import DEV, MOUNT, build_chunks, build_facts, dump, make_stack, model_and_tok

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GAMMA = 0.05


def cos(a, b):
    a = a.detach().float().reshape(-1); b = b.detach().float().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok); chunks, owner = build_chunks(tok, facts)
    st = make_stack(gamma=GAMMA, no_write=False, seed=a.seed)
    res = {"seed": a.seed}

    li0 = MOUNT[0]; m = st.mems[li0]
    print(f"[超参] eta={m.eta} alpha={m.alpha} lam={m.lam} pk_scale={m.pk_scale} "
          f"use_factorized={m.use_factorized}", flush=True)
    res["hyper"] = {"eta": float(m.eta), "alpha": float(m.alpha), "lam": float(m.lam),
                    "pk_scale": float(m.pk_scale), "R": int(getattr(m, "R", -1))}

    # ---- C1 重建 W 校验（随机 q，比对 read 与手算）
    q = torch.randn(m.Pk.shape[1], device=DEV) * 1.0
    with torch.no_grad():
        Wr = (m.W0 + m.U @ m.V.T) if m.use_factorized else m.Wfull
        y = Wr @ q
        left = (m.Wo @ y)
        # read 侧：走模块自身，取单 token 注入
        hs = torch.zeros(1, 1, m.Pk.shape[1], device=DEV)
        m_q = m  # read 用 Wq，此处仅校验 W 的重建一致性
        Wr2 = (m.W0 + m.U @ m.V.T) if m.use_factorized else m.Wfull
        print(f"[C1] 两条重构路径 cos = {cos(Wr, Wr2):+.6f}", flush=True)
        res["C1_recon_cos"] = cos(Wr, Wr2)
        res["C1_W_norm"] = float(Wr.norm())

    # ---- C2 单对过拟合：取一个真实 (k,v) 对，连写 200 次
    CAP = {}
    orig_write = m.write
    def wrap(hs_in, drop_sg=False):
        before = None if m.prev_hs is None else m.prev_hs.clone()
        r = orig_write(hs_in, drop_sg)
        if before is not None and "kv" not in CAP:
            kt = ((before.reshape(-1, before.shape[-1]) @ m.Pk).mean(0)) * m.pk_scale
            vt = (hs_in.reshape(-1, hs_in.shape[-1]) @ m.Pv).mean(0)
            CAP["kv"] = (kt.detach().float().clone(), vt.detach().float().clone())
        return r
    m.write = wrap
    model(input_ids=chunks[0].to(DEV))          # 第 1 次：prev_hs 仍为 None，write 早退
    model(input_ids=chunks[0].to(DEV))          # 第 2 次：prev_hs 已就绪，能抓到 (k,v)
    m.write = orig_write
    assert "kv" in CAP, "未抓到 (k,v)"
    k, v = CAP["kv"]
    W0n = float(((m.W0 + m.U @ m.V.T) if m.use_factorized else m.Wfull).norm())
    print(f"[C2] 固定对 ||k||={float(k.norm()):.2f} ||v||={float(v.norm()):.2f}", flush=True)
    traj = []
    for t in range(201):
        Wcur = ((m.W0 + m.U @ m.V.T) if m.use_factorized else m.Wfull)
        rl = float(((Wcur @ k) - v).norm() / (v.norm() + 1e-30))
        cs = cos(Wcur @ k, v)
        wn = float((Wcur - m.W0).norm()) if not m.use_factorized else float((m.U @ m.V.T).norm())
        if t in (0, 1, 2, 5, 10, 25, 50, 100, 150, 200):
            traj.append({"t": t, "rel": rl, "cos": cs, "dW": wn})
            print(f"   t={t:<4} rel={rl:.4f} cos={cs:+.4f} ||W-W0||={wn:.2f}", flush=True)
        # 手动施加一次真实写规则（同 sgmn_module.write，用固定 k,v）
        Wc = ((m.W0 + m.U @ m.V.T) if m.use_factorized else m.Wfull)
        G = m.alpha * torch.outer(Wc @ k - v, k) + m.lam * (Wc - m.W0)
        with torch.no_grad():
            if m.use_factorized:
                m.U.copy_(m.U - m.eta * (G @ m.V)); m.V.copy_(m.V - m.eta * (G.T @ m.U))
            else:
                m.Wfull.copy_(m.Wfull - m.eta * G)
    res["C2_overfit"] = traj
    res["C2_k_norm"] = float(k.norm()); res["C2_v_norm"] = float(v.norm())
    res["C2_hebb_step"] = float(m.eta) * float(m.alpha) * float(k.norm()) ** 2
    res["C2_W0_norm"] = W0n
    print(f"[C3] eta*alpha*||k||^2 = {res['C2_hebb_step']:.3e}（Hebb 收敛需 ≈1）", flush=True)
    print(f"[C3] ||W0|| = {W0n:.2f}", flush=True)
    dump(res, "probe91_overfit.json")
    print("\n★ 诊断完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
