#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-96 事后核 · `D-126` 的三处修补（全部由 advisor 复核意见触发，逐条留档）。

修 1 · **秩统计量**：删掉我原用的 `1/((σ₁/‖ΔW‖)²+(σ₂/σ₁)²)`（混合了两种归一化，**不是**定义良好的量）。
       改用 **stable rank = ‖ΔW‖_F²/σ₁²**（= 1/Σ(σ_i/σ₁)²，精确）与 **尾能量 = 1 − σ₁²/‖ΔW‖_F²**。
修 2 · **溯源**：全部读数**由实测张量直接算出**，不再由手打的 5 位小数 ν/σ₁ 反推（原先第 5 位是转录噪声）。
       本脚本落盘 `stage_attrib.json`，行/裁决引用其 md5。
修 3 · **归因（决定 `O96-1` 值不值得开）**：rank-1 可能来自**键共线**（右因子）或**误差/值向量共线**（左因子）。
       测 `cos(v₁, k̂)`、`cos(u₁, ē)`、`cos(u₁, v̄)`、`cos(u₁, k̂)`、`cos(v₁, ē)`，以及**去均值后**键/值协方差谱。

★ 另测 **`D-126` 的"饱和"解释是否成立**：我原先说"32 事实写完后 `Wk≈v` ⇒ 数据项消失"。
  更简约的替代 = **衰减平衡**：`0 = α E[(Wk−v)kᵀ] + λ(W−W⁰)` ⇒ `W = (α r + λW⁰)(αS+λI)^{-1}`，`S=E[kkᵀ]`、`r=E[vkᵀ]`。
  该式 **与 η 无关**、且当 `S` 近 rank-1 时 `ΔW` **必然 rank-1（右因子 = k̂）** —— 一条式子同时解释"块数无关 + rank-1"。
  本脚本**直接算该式**并与实测 `ΔW` 对比（余弦 + 范数比），**不做近似**。
"""
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
import probe91_kpool as KP                       # noqa: E402
import R2_task91 as R91                          # noqa: E402
import sgmn_module                               # noqa: E402
import torch                                     # noqa: E402

SEEDS = [20261028, 7777]
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
DEV = R91.DEV


def cap_write(st):
    """返回 (patched_ctx, BUF)。在 write() 里按**与源码逐字相同**的公式复算 k,v 并留存。"""
    BUF = {li: [] for li in st.mems}
    orig = sgmn_module.LayerMem.write

    def patched(self, hs_in, drop_sg=False):
        if self.prev_hs is not None and not self.no_write:
            for li, m in st.mems.items():
                if m is self:
                    pf = self.prev_hs.reshape(-1, self.prev_hs.shape[-1])
                    hf = hs_in.reshape(-1, hs_in.shape[-1])
                    agg = (lambda z: z.amax(dim=0)) if self.op == "maxpool" else (lambda z: z.mean(dim=0))
                    BUF[li].append(((agg(pf @ self.Pk) * self.pk_scale).detach().clone(),
                                    agg(hf @ self.Pv).detach().clone()))
                    break
        return orig(self, hs_in, drop_sg)

    class C:
        def __enter__(s):
            sgmn_module.LayerMem.write = patched
            return BUF

        def __exit__(s, *a):
            sgmn_module.LayerMem.write = orig
            return False

    return C()


def stats(X):
    """stable rank / 尾能量 / 精确奇异值谱。"""
    sv = torch.linalg.svdvals(X)
    fro2 = float((X ** 2).sum())
    s1 = float(sv[0])
    p = (sv ** 2) / fro2
    ent = float(-(p * torch.log(p.clamp_min(1e-30))).sum().exp())     # 熵有效秩（精确，全谱）
    return {"fro": math.sqrt(fro2), "sigma1": s1,
            "stable_rank": fro2 / (s1 ** 2), "tail_energy": 1.0 - (s1 ** 2) / fro2,
            "entropy_rank": ent, "sv_top8": [float(x) for x in sv[:8]]}


def main():
    OUT.mkdir(exist_ok=True)
    tok, model = KP.model_and_tok()
    model.eval()
    torch.set_grad_enabled(True)
    facts = KP.build_facts(tok)
    chunks, owner = KP.build_chunks(tok, facts)
    tr = list(range(R91.N_TRAIN_KEYS))
    G = {"seeds": {}}
    alpha, lam = R91.ALPHA, R91.LAM

    KP.install_write()
    try:
        for sd in SEEDS:
            st = R91.make_stack(gamma=0.0, no_write=False, seed=sd)
            with cap_write(st) as BUF:
                KP.write_phase_arm(model, st, chunks, owner, None)
            R91.train_read_task(model, st, tok, facts, tr, random.Random(1000 + sd % 100))
            R = {}
            for li, m in st.mems.items():
                W0, W = m.W0, m.Wfull.detach()
                dW = W - W0
                nu = float(dW.norm() / W0.norm())
                K = torch.stack([b[0] for b in BUF[li]])          # (B, d_k)
                V = torch.stack([b[1] for b in BUF[li]])          # (B, d_v)
                B = K.shape[0]
                S = (K.T @ K) / B
                r = (V.T @ K) / B
                I = torch.eye(S.shape[0], device=DEV, dtype=S.dtype)
                Wpred = (alpha * r + lam * W0) @ torch.linalg.inv(alpha * S + lam * I)
                dWp = Wpred - W0
                # ---- 归因 ----
                U, sv, Vh = torch.linalg.svd(dW, full_matrices=False)
                u1, v1 = U[:, 0], Vh[0]
                kbar = K.mean(0)
                vbar = V.mean(0)
                ebar = (W @ K.T).T.sub(V).mean(0)     # 每块误差 (W k_b − v_b) 的均值 ⇒ (d_v,)
                c = lambda a, b: float((a @ b) / (a.norm() * b.norm()))
                # ---- 去均值后的键/值协方差谱 ----
                Kc = K - kbar
                Vc = V - vbar
                Sc = (Kc.T @ Kc) / B
                svSc = torch.linalg.svdvals(Sc)
                pr = float((svSc.sum() ** 2) / (svSc ** 2).sum())      # 参与比（秩代理）
                R[str(li)] = {
                    "nu": nu, "dW": stats(dW), "dW_pred": stats(dWp),
                    "pred_cos": c(dWp.flatten(), dW.flatten()),
                    "pred_norm_ratio": float(dWp.norm() / dW.norm()),
                    "n_blocks": int(B),
                    "cos_v1_kbar": c(v1, kbar), "cos_u1_ebar": c(u1, ebar),
                    "cos_u1_vbar": c(u1, vbar),
                    "kbar_norm": float(kbar.norm()), "vbar_norm": float(vbar.norm()),
                    "ebar_norm": float(ebar.norm()),
                    "kcenter_participation": pr,
                    "kcenter_sv_top8": [float(x) for x in svSc[:8]],
                    "alpha_over_lambda": alpha / lam,
                    "eta_alpha_k2": float(R91.ETA_WRITE * alpha * (kbar.norm() ** 2)),
                }
                print(f"[{sd} L{li}] ν={nu:.5f} stable_rank={R[str(li)]['dW']['stable_rank']:.4f} "
                      f"tail={R[str(li)]['dW']['tail_energy']:.5f} "
                      f"cos(v1,k̄)={R[str(li)]['cos_v1_kbar']:+.4f} "
                      f"cos(u1,ē)={R[str(li)]['cos_u1_ebar']:+.4f} "
                      f"pred_cos={R[str(li)]['pred_cos']:.4f} "
                      f"pred‖·‖/‖·‖={R[str(li)]['pred_norm_ratio']:.3f}", flush=True)
            G["seeds"][str(sd)] = R
            st.remove()
            del st
            torch.cuda.empty_cache()
    finally:
        KP.restore_write()

    (OUT / "stage_attrib.json").write_text(json.dumps(G, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[写出] {OUT/'stage_attrib.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())