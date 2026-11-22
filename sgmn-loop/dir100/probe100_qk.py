#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-100 · **前提核**（非实验轮）—— 事实键能否从「问题末位」线性解码、且跨事实泛化？

逐字**复用** `dir99/R1_task99.py`（`capture_keys` / `IdentityTF` / `install` / `restore`），
不重写任何键/写/读逻辑 ⇒ `k_i` 与正式轮真写用的是**同一个函数**。

判据**跑前冻结**在 `revolve/DIR-100/PREMISE-100.md` §3：
  `cos_in` = 全 32 拟合、全 32 算（天花板）
  `cos_tr` = 仅 16 训练事实拟合、这 16 上算（拟合质量，守卫 ≥0.90）
  `cos_he` = 同一 `M` 在另 16 上算（★ 主读数）
  P100-A `cos_he ≥ 0.50`（线性解存在 ⇒ 优化问题）
  P100-B `cos_he ≤ 0.20`（无线性泛化 ⇒ 表示性不可达）
  P100-C 0.20 < cos_he < 0.50（灰区）
  P100-D `cos_tr < 0.90` 而 `cos_in ≥ 0.90`（16 点定不出 ⇒ 改用留一法）
  INVALID `cos_in < 0.90`（问题末位没装事实键）

★ 本探针**不产生** `SUCCEEDED`、**不写证据行**。
"""
import json
import pathlib
import sys

sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir99")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
import probe91_kpool as KP                       # noqa: E402
import R1_task99 as R99                          # noqa: E402   ← 逐字复用其 capture_keys
import R2_task91 as R91                          # noqa: E402
import torch                                     # noqa: E402

SEED = 20261028
LAM = 1e-6
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
DEV = R91.DEV


def _cos_rows(P, Q):
    """逐行余弦，返回 (n,) 1-D。"""
    P = P.float()
    Q = Q.float()
    return (P * Q).sum(dim=1) / (P.norm(dim=1) * Q.norm(dim=1)).clamp_min(1e-12)


def _ridge_A(C, K, lam=LAM):
    """最小范数最小二乘：A = Cᵀ(CCᵀ+λI)⁻¹K，(n,d)→(d,d_k)。★ n=16 ≪ d=896。"""
    C = C.float()
    K = K.float()
    n = C.shape[0]
    G = C @ C.T + lam * torch.eye(n, device=C.device, dtype=C.dtype)
    return C.T @ torch.linalg.solve(G, K)                       # (d, d_k)


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_grad_enabled(False)
    tok, model = KP.model_and_tok()
    model.eval()
    facts = KP.build_facts(tok)
    chunks, owner = KP.build_chunks(tok, facts)
    tr = list(range(16))
    he = list(range(16, 32))
    print(f"[setup] facts={len(facts)} chunks={len(chunks)} tr={len(tr)} he={len(he)}", flush=True)

    st = R91.make_stack(gamma=0.0, no_write=False, seed=SEED)
    for m in st.mems.values():
        m.key_tf = R99.IdentityTF()
        m.q_tf = R99.IdentityTF()
        m.v_neg_shift = False
        m.v_hold = None
    R99.install()
    G = {"seed": SEED, "layers": {}, "rules": {
        "P100-A": "cos_tr>=0.90 and cos_he>=0.50",
        "P100-B": "cos_tr>=0.90 and cos_he<=0.20",
        "P100-C": "cos_tr>=0.90 and 0.20<cos_he<0.50",
        "P100-D": "cos_tr<0.90 and cos_in>=0.90",
        "INVALID": "cos_in<0.90"}}
    try:
        # ---- 键：逐字复用正式轮的 capture_keys（恒等 T）
        K, V = R99.capture_keys(model, st, facts, chunks, owner)

        # ---- 问题末位读出向量 c_i（判据位置 = train_read_task 的 P-1）
        C = {li: [] for li in st.mems}
        Pk_scale = {}
        for li, m in st.mems.items():
            Pk_scale[li] = float(m.pk_scale)
        lens = []
        for fi in range(len(facts)):
            pid = facts[fi]["prompt_ids"]
            lens.append(len(pid))
            model(input_ids=torch.tensor([pid], dtype=torch.long, device=DEV))
            for li in st.mems:
                hs = st._hs_in[li]
                h = hs[0] if hs.dim() == 3 else hs
                C[li].append(h[len(pid) - 1].detach().clone())      # ★ P-1
        print(f"[prompt] len min/median/max = {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}",
              flush=True)

        # ---- 逐层拟合 + 判定
        verdicts = []
        for li in st.mems:
            Kl = K[li].float()                                     # (32, d_k)
            Cl = torch.stack(C[li]).float()                        # (32, d)
            Pk = st.mems[li].Pk.detach().float()                   # (d, d_k)
            s = Pk_scale[li]
            nrm = R99.norm_rows
            # 1) 天花板：全 32 拟合、全 32 算
            A_all = _ridge_A(Cl, Kl)
            cos_in = float(_cos_rows(Cl @ A_all, Kl).mean())
            # 2/3) 只 16 训练拟合 → tr / he
            A_tr = _ridge_A(Cl[tr], Kl[tr])
            c_tr = _cos_rows(Cl[tr] @ A_tr, Kl[tr])
            c_he = _cos_rows(Cl[he] @ A_tr, Kl[he])
            cos_tr, cos_he = float(c_tr.mean()), float(c_he.mean())
            # 对照：候选机制 A_pk = pk_scale · Pkᵀ
            pk_tr = float(_cos_rows((Cl[tr] @ Pk) * s, Kl[tr]).mean())
            pk_he = float(_cos_rows((Cl[he] @ Pk) * s, Kl[he]).mean())
            # 参考（不进判定）：事实 chunk 自己的末位 c 与 k_i
            own = []
            for fi in range(len(facts)):
                ci = owner.index(fi) + 1
                model(input_ids=chunks[ci].to(DEV))
                hs = st._hs_in[li]
                h = hs[0] if hs.dim() == 3 else hs
                own.append(h[h.shape[0] - 1].detach().clone())
            own = torch.stack(own).float()
            cos_own = float(_cos_rows((own @ Pk) * s, Kl).mean())

            row = {"cos_in": cos_in, "cos_tr": cos_tr, "cos_he": cos_he,
                   "he_min": float(c_he.min()), "he_max": float(c_he.max()),
                   "pk_tr": pk_tr, "pk_he": pk_he, "cos_own_chunk_last": cos_own,
                   "n_he_pos": int((c_he > 0.5).sum())}
            if cos_in < 0.90:
                row["verdict"] = "INVALID"
            elif cos_tr < 0.90:
                row["verdict"] = "P100-D"
            elif cos_he >= 0.50:
                row["verdict"] = "P100-A"
            elif cos_he <= 0.20:
                row["verdict"] = "P100-B"
            else:
                row["verdict"] = "P100-C"
            verdicts.append(row["verdict"])
            G["layers"][str(li)] = row
            print(f"[L{li}] cos_in={cos_in:.4f} cos_tr={cos_tr:.4f} "
                  f"cos_he={cos_he:.4f} (min {row['he_min']:+.3f} max {row['he_max']:+.3f}, "
                  f">0.5 的 he 数={row['n_he_pos']}/16) | A_pk: tr={pk_tr:.4f} he={pk_he:.4f} "
                  f"| own_chunk_last={cos_own:.4f} ⇒ {row['verdict']}", flush=True)

        maj = max(set(verdicts), key=verdicts.count)
        G["verdicts_per_layer"] = verdicts
        G["verdict"] = maj
        print(f"[PREMISE-100] 逐层 = {verdicts} ⇒ 多数判决 = **{maj}**", flush=True)
        (OUT / "premise100.json").write_text(
            json.dumps(G, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  ⇒ {OUT/'premise100.json'}", flush=True)
    finally:
        R99.restore()
        st.remove()
        del st
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())