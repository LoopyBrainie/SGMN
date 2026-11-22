#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-97 · rev 1.2 —— 写入规则存下的内容能不能分辨 32 个事实？

逐字派生自 `dir96/R1_task96.py` + `dir91/probe91_kpool.py`。
**只改测量层**：不跑 600 步读侧训练（`W` 由写规则驱动，`read()` 里 `y = sg(W) q`），
但 `A97-4` **实测**这一前提（跑一次读侧训练并断言 `W` 未变），不以代码阅读代替。

判据（rev 1.2，冻结）：
  主 `N_hit^v(W) = Σ_i [ argmax_j cos(W k_i, v_j) ∈ S_i ]`，`S_i = {j : v_j ≡ v_i}`
  副 `N_hit^s(W)`（严格下标）、`N_hit^v_L2(W)`
  `G97-1` 阳性 `N_hit^v(W_or) ≥ 30/32`；`G97-2` 阴性 `N_hit^v(W_or^π) ≤ 3/32`
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time

sys.path.insert(0, "/home/ec2-user/sgmn-loop")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir91")
import probe91_kpool as KP                       # noqa: E402
import R2_task91 as R91                          # noqa: E402
import torch                                     # noqa: E402

SEEDS = [20261028, 7777]
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
DEV = R91.DEV
CHECKS = []


def rec(cid, ok, detail):
    CHECKS.append({"id": cid, "ok": bool(ok), "detail": detail})
    print(f"  {'✓' if ok else '✗'} {cid}  {detail}", flush=True)


def norm_rows(X):
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-30)


def n_hit(W, K, V, Ssets):
    """N_hit^v（值一致即命中）、N_hit^s（严格下标）、N_hit^v_L2。**同一个函数**，只换形参。"""
    R = (W @ K.T).T                                     # (n, d_v)
    COS = norm_rows(R) @ norm_rows(V).T                 # (n, n)
    am = COS.argmax(dim=1)
    strict = int((am == torch.arange(K.shape[0], device=DEV)).sum())
    valmatch = int(sum(int(am[i]) in Ssets[i] for i in range(K.shape[0])))
    D = (R.unsqueeze(1) - V.unsqueeze(0)).pow(2).sum(-1)   # (n, n) 距离
    am2 = D.argmin(dim=1)
    valmatch_l2 = int(sum(int(am2[i]) in Ssets[i] for i in range(K.shape[0])))
    return valmatch, strict, valmatch_l2


def stable_rank(X):
    sv = torch.linalg.svdvals(X)
    fro2 = float((X ** 2).sum())
    return fro2 / float(sv[0]) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="formal", choices=["stageA", "formal"])
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    stageA = (a.stage == "stageA")

    tok, model = KP.model_and_tok()
    model.eval()
    torch.set_grad_enabled(True)
    facts = KP.build_facts(tok)
    chunks, owner = KP.build_chunks(tok, facts)
    n = len(facts)
    tr = list(range(R91.N_TRAIN_KEYS))
    print(f"[setup] facts={n} chunks={len(chunks)} tr={len(tr)} seeds={SEEDS} stage={a.stage} dev={DEV}", flush=True)

    # ---------------- 值重复结构（rev 1.2 的核心） ----------------
    vals = [f["val"] for f in facts]
    uniq = sorted(set(vals))
    Ssets = [[j for j in range(n) if vals[j] == vals[i]] for i in range(n)]
    sum_S = sum(len(s) for s in Ssets)
    E_random = sum_S / n
    print(f"[values] n={n} distinct={len(uniq)} Σ|S_i|={sum_S} E_random={E_random:.4f}")

    # ---------------- 逐 seed：写相（无读侧训练）+ 判据 ----------------
    perm = list(range(n))
    random.Random(20261120).shuffle(perm)
    blocks_per_fact = {fi: sum(1 for o in owner if o == fi) for fi in range(n)}
    G = {"stage": a.stage, "n_facts": n, "n_distinct_vals": len(uniq),
         "sum_S": sum_S, "E_random": E_random, "seeds": {}, "checks": CHECKS}

    for sd in SEEDS:
        t0 = time.time()
        KP.install_write()
        try:
            st = R91.make_stack(gamma=0.0, no_write=False, seed=sd)
            KP.write_phase_arm(model, st, chunks, owner, None)
            Ws = {li: m.Wfull.detach().clone() for li, m in st.mems.items()}
            W0s = {li: m.W0.detach().clone() for li, m in st.mems.items()}
            W_before = {li: m.Wfull.detach().clone() for li, m in st.mems.items()}

            # ---- A97-4：读侧训练不得改动 W（实测 sg(W)），每 seed 各测一次 ----
            R91.train_read_task(model, st, tok, facts, tr, random.Random(1000 + sd % 100))
            for m in st.mems.values():                 # DEF-95-01：显式重封
                m.no_write = True
            rel = max(float((m.Wfull.detach() - W_before[li]).norm() / W_before[li].norm())
                      for li, m in st.mems.items())
            rec(f"A97-4/seed{sd}", rel < 1e-12,
                f"‖W_after_read − W_write‖/‖W_write‖ = {rel:.3e} (< 1e-12)")

            # ---- 键/值矩阵（与写规则同源） ----
            K, V = KP.key_matrices_arm(model, st, facts, chunks, owner, None)
            for m in st.mems.values():
                m.no_write = True
            st.remove()
            del st
            torch.cuda.empty_cache()
        finally:
            KP.restore_write()

        RS = {}
        for li in sorted(K.keys()):
            Kl, Vl = K[li], V[li]
            W, W0 = Ws[li], W0s[li]
            Kp = torch.linalg.pinv(Kl.T)                     # (n, d_k)
            W_or = Vl.T @ Kp                                 # 最小范数解：W_or K^T = V^T
            Vp = Vl[torch.as_tensor(perm, device=DEV)]
            W_or_pi = Vp.T @ Kp
            resid = float((W_or @ Kl.T - Vl.T).norm() / Vl.T.norm())
            nv, ns, nl2 = n_hit(W, Kl, Vl, Ssets)
            w0v, w0s, _ = n_hit(W0, Kl, Vl, Ssets)
            orv, ors, _ = n_hit(W_or, Kl, Vl, Ssets)
            piv, pis, _ = n_hit(W_or_pi, Kl, Vl, Ssets)
            dW = W - W0
            Vu = torch.stack([Vl[vals.index(u)] for u in uniq])
            Cn = norm_rows(Vu) @ norm_rows(Vu).T
            max_pair = float(Cn.fill_diagonal_(-2.0).max())
            kn = Kl.norm(dim=1)
            RS[str(li)] = {
                "n_hit_v": nv, "n_hit_s": ns, "n_hit_v_L2": nl2,
                "n_hit_v_W0": w0v, "n_hit_s_W0": w0s,
                "n_hit_v_oracle": orv, "n_hit_s_oracle": ors,
                "n_hit_v_oracle_perm": piv, "n_hit_s_oracle_perm": pis,
                "oracle_resid": resid, "stable_rank": stable_rank(dW),
                "nu": float(dW.norm() / W0.norm()),
                "max_pair_cos_v_unique": max_pair,
                "knorm_ratio": float(kn.max() / kn.min()),
                "blocks_min": min(blocks_per_fact.values()),
            }
            print(f"[seed {sd} L{li}] N_hit^v: intact={nv}/32 W0={w0v}/32 oracle={orv}/32 "
                  f"oracle^π={piv}/32 | strict={ns}/32 | L2={nl2}/32 "
                  f"| stable_rank={RS[str(li)]['stable_rank']:.4f} "
                  f"| max_pair_cos(v_uniq)={max_pair:.4f} | {time.time()-t0:.0f}s", flush=True)
        G["seeds"][str(sd)] = RS

    # ---------------- Stage A 判定（跨全部 seed × 层取最坏） ----------------
    flat = {str(li): [G["seeds"][str(sd)][str(li)] for sd in SEEDS] for li in sorted(K.keys())}
    cells = [c for li in flat for c in flat[li]]
    worst = lambda f: [c[f] for c in cells]                            # noqa: E731
    rec("A97-1", min(o - p for o, p in zip(worst("n_hit_v_oracle"), worst("n_hit_v_oracle_perm"))) >= 30,
        f"解算力 min(oracle − oracle^π) = "
        f"{min(o - p for o, p in zip(worst('n_hit_v_oracle'), worst('n_hit_v_oracle_perm')))}/32 (≥30)")
    rec("A97-2", max(worst("knorm_ratio")) < 10.0,
        f"max ‖k_i‖ 比 = {max(worst('knorm_ratio')):.3f} (< 10)")
    rec("A97-3", max(worst("oracle_resid")) < 1e-6,
        f"max oracle 残差 = {max(worst('oracle_resid')):.3e} (< 1e-6)")
    rec("A97-6", min(worst("blocks_min")) >= 18, f"min 每事实块数 = {min(worst('blocks_min'))} (≥18)")
    rec("A97-7", max(worst("stable_rank")) <= 1.02,
        f"max stable rank(ΔW) = {max(worst('stable_rank')):.4f} (≤1.02，复现 D-126)")
    rec("A97-9", sum_S == sum(m * m for m in __import__("collections").Counter(vals).values())
        and len(uniq) == 27 and abs(E_random - sum_S / n) < 1e-12,
        f"Σ|S_i|={sum_S}==Σm_v²；不同值={len(uniq)}==27；E_random={E_random:.4f}")
    rec("G97-1", min(worst("n_hit_v_oracle")) >= 30,
        f"min N_hit^v(oracle) = {min(worst('n_hit_v_oracle'))}/32 (≥30)")
    rec("G97-2", max(worst("n_hit_v_oracle_perm")) <= 3,
        f"max N_hit^v(oracle^π) = {max(worst('n_hit_v_oracle_perm'))}/32 (≤3)")
    rec("A97-5", all(c["n_hit_v_oracle"] != c["n_hit_v_oracle_perm"] for c in cells),
        "N_hit^v 为同一函数；oracle 与 oracle^π 得分不同")
    rec("M97-2", max(worst("n_hit_v_W0")) <= E_random + 3,
        f"max N_hit^v(W0) = {max(worst('n_hit_v_W0'))}/32 (≤ E_random+3 = {E_random + 3:.3f})")
    G["device_max_pair_cos_v_unique"] = [min(worst("max_pair_cos_v_unique")),
                                         max(worst("max_pair_cos_v_unique"))]
    # ---- 主判据 M97-1（只在 Stage A 全过时解释；此处仅记录，不在此处下判决） ----
    G["M97-1_reading"] = {"per_cell_n_hit_v": [c["n_hit_v"] for c in cells],
                          "note": "≥24 确认 / ≤8 否证 / (8,24) AMBIGUOUS —— 判决在 VERDICT.md"}

    G["all_stageA_pass"] = all(c["ok"] for c in CHECKS)
    (OUT / f"stage_{'A' if stageA else 'formal'}.json").write_text(
        json.dumps(G, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[Stage A] {'全部通过' if G['all_stageA_pass'] else '有 FAIL'}  "
          f"→ {OUT}/stage_{'A' if stageA else 'formal'}.json")
    return 0 if G["all_stageA_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())