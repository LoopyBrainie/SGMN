#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-103 · 纯诊断（**不进判据**）—— `c` 到底能把 32 条身份分开到什么程度？

**为什么需要它**：正式轮 `c`-only 在**两个 seed 上都是 0.8750 = 28/32**。
若是优化噪声，不会同一值；若是结构上界，须给出**可计算的量**。
→ 本脚本量 `c` 的**线性身份可解码上界**（闭式，与优化器无关，方法同 `A101-2`/`A101-6`）。

读数：
 ① `rank({c_i})`、奇异值谱
 ② 闭式 ridge 解码「身份」(`c → onehot(i)`) 的**留一**准确率上界 —— ★ 单标签 32 类
 ③ 闭式 ridge 解码「值」(`c → onehot(val)`) —— 若高 ⇒ 值其实在 `c` 里（前提破裂）
 ④ 同值对 / 异值对 `|cos(c)|`（镜像 `A101-6` 的口径）
★ 一律 **report-only**：不改任何判据、不改任何阈值。
"""
import json
import pathlib
import random
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "dir91"))
sys.path.insert(0, str(_HERE.parent / "dir99"))
sys.path.insert(0, str(_HERE.parent / "dir100"))

import probe91_kpool as KP                        # noqa: E402
import R2_task91 as R91                           # noqa: E402
import R1_task100 as D100                         # noqa: E402
import R1_task103 as D103                         # noqa: E402
import torch                                      # noqa: E402

DEV = R91.DEV
OUT = _HERE / "out"


@torch.no_grad()
def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    tok, model = KP.model_and_tok()
    model.eval()
    facts = R91.build_facts(tok)
    allk = list(range(len(facts)))
    res = {}

    # ---- 在一栈上取每层的 `c`（读出位，即 prompt 最后一个位置）
    st = R91.make_stack(gamma=0.0, no_write=False, seed=20261028)
    try:
        for m in st.mems.values():
            m.key_tf = D100.IdentityTF()
            m.q_tf = D100.IdentityTF()
            m.v_neg_shift = False
            m.v_hold = None
        D103.install()
        # ★ 写完之后再取 `c`：`c` 与 `W` 无关（`c = sg(hs_in)`），故写不写不影响 `c`；
        #   但保持与正式轮同一状态顺序，避免任何口径歧义。
        chunks, owner = R91.build_chunks(tok, facts)
        Kraw, V = D100.capture_keys(model, st, facts, chunks, owner)
        TF = {}
        for li in Kraw:
            TF[li] = D100.build_tf("white", Kraw[li], V[li], li)
            st.mems[li].key_tf = TF[li]
            st.mems[li].q_tf = TF[li].qmap
        KP.write_phase_arm(model, st, chunks, owner, {i: None for i in allk})

        D103.set_wc(st, make=True)
        D103.set_inp("cy")
        D103.set_capture(True)
        C = {}
        for ki in allk:
            f = facts[ki]
            ids = torch.tensor([f["prompt_ids"]], dtype=torch.long, device=DEV)
            # 逐层 forward 会被 LayerMem.read 捕获；用**单题**序列 ⇒ `_C_OUT` 就是该题读出位
            D103.set_capture(True)
            _ = model(input_ids=ids)
            C[ki] = D103._C_OUT.clone()          # (B*1, d_model) with B=1
        D103.set_capture(False)

        for li in sorted(st.mems):
            # ★ 捕获按层覆盖，故只保留**最后被读的那一层**的 c；
            #   为拿逐层的 c，改为逐层单独 forward 太重 ⇒ 只报 `MOUNT` 中**最后一层**，
            #   并在结果里显式标出是哪一层（不得冒充"全部层"）。
            pass
        c_last = torch.stack([C[k][0] for k in allk])      # (32, d_model)
        li_last = max(st.mems)

        X = c_last.float()
        res["layer"] = int(li_last)
        res["d_model"] = int(X.shape[1])
        res["shape"] = list(X.shape)

        # ① 秩与奇异值
        sv = torch.linalg.svdvals(X - X.mean(0, keepdim=True))
        res["rank_raw"] = int(torch.linalg.matrix_rank(X))
        res["rank_centered"] = int(torch.linalg.matrix_rank(X - X.mean(0, keepdim=True)))
        res["sv_top8"] = [round(float(x), 4) for x in sv[:8]]
        res["sv_ratio_1_2"] = round(float(sv[0] / sv[1].clamp_min(1e-30)), 4)

        # ④ 同值对 / 异值对 |cos|
        Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
        G = (Xn @ Xn.T).abs()
        vals = [facts[i]["val"] for i in allk]
        same, diff = [], []
        for i in range(len(allk)):
            for j in range(i + 1, len(allk)):
                (same if vals[i] == vals[j] else diff).append(float(G[i, j]))
        res["cos_same_mean"] = round(sum(same) / max(len(same), 1), 4)
        res["cos_diff_mean"] = round(sum(diff) / max(len(diff), 1), 4)
        res["cos_gap"] = round(res["cos_diff_mean"] - res["cos_same_mean"], 4)
        res["n_same_pairs"] = len(same)

        # ② ③ 闭式 ridge（留一）—— 与优化器无关
        def ridge_loo(T, Yr, lam_frac=1e-3):
            """留一：每次用 31 个样本求闭式解、在第 32 个上测（`T` 目标哈希表）。"""
            n = T.shape[0]
            hit_ident, hit_val = 0, 0
            idx = torch.arange(n)
            for k in range(n):
                tr = idx[idx != k]
                A, b = T[tr], Yr[tr]
                AtA = A.T @ A
                lam = lam_frac * float(AtA.diagonal().mean().clamp_min(1e-12))
                Gm = torch.linalg.solve(AtA + lam * torch.eye(A.shape[1], device=DEV), A.T @ b)
                pred = (T[k:k + 1] @ Gm)[0]
                hit_ident += int(int(torch.argmax(pred)) == int(Yr[k].argmax()))
                # 值的预测：值只 27 类，但同一 `onehot` 空间的 argmax 即"最像哪个训练样本"
                hit_val += int(int(torch.argmax(pred)) == int(Yr[k].argmax()))
            return hit_ident / n, hit_val / n

        n = len(allk)
        Iden = torch.eye(n, device=DEV)
        # 身份：目标 = onehot(i)
        acc_id, _ = ridge_loo(X, Iden)
        res["ridge_identity_loo"] = round(acc_id, 4)
        res["ridge_identity_loo_n"] = f"{round(acc_id*n)}/{n}"

        # 值：目标 = onehot(该事实的值在值池中的下标)（同值事实共享一行 ⇒ 用样本 onehot 不可行，
        #     改为"值类别"目标：把值映射到 0..26 的类别下标再 onehot）
        pool = sorted({f["val"] for f in facts})
        vcls = {v: i for i, v in enumerate(pool)}
        Yv = torch.zeros(n, len(pool), device=DEV)
        for i, k in enumerate(allk):
            Yv[i, vcls[facts[k]["val"]]] = 1.0
        # 值类别的留一：预测 argmax 落在该样本自己的值类 ⇒ 命中
        hitv = 0
        idx = torch.arange(n)
        for k in range(n):
            tr = idx[idx != k]
            A, b = X[tr], Yv[tr]
            AtA = A.T @ A
            lam = 1e-3 * float(AtA.diagonal().mean().clamp_min(1e-12))
            Gm = torch.linalg.solve(AtA + lam * torch.eye(A.shape[1], device=DEV), A.T @ b)
            pred = (X[k:k + 1] @ Gm)[0]
            hitv += int(int(torch.argmax(pred)) == int(Yv[k].argmax()))
        res["ridge_value_loo"] = round(hitv / n, 4)
        res["ridge_value_loo_n"] = f"{hitv}/{n}"

        # ⑤ 中心化后同一套（更严格口径）
        Xc = X - X.mean(0, keepdim=True)
        acc_idc, _ = ridge_loo(Xc, Iden)
        res["ridge_identity_loo_centered"] = round(acc_idc, 4)
        res["ridge_identity_loo_centered_n"] = f"{round(acc_idc*n)}/{n}"
    finally:
        D103.set_capture(False)
        D103.set_inp("cy")
        D103.clear_wc(st)
        D103.restore()
        st.remove()
        torch.cuda.empty_cache()

    (OUT / "diag_c.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print("\n" + "=" * 84)
    for k, v in res.items():
        print(f"{k:>34} = {v}")
    print("=" * 84, flush=True)


if __name__ == "__main__":
    main()