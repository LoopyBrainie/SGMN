#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 rev 2.5 · **判决之后的补充诊断**（非判据、不参与任何门）

正式轮（probe91_valuetok.py）判 `POOLING`：值 token 位 LOO-10=1.000、块均值=机会水平。
本文件回答两个**架构可操作**的后续问题（**都是补充，不改卡片 §2 的任何阈值**）：

  (a) **影响锥**：答案身份在"离值 token 多少个 token 的距离"上仍可线性辨识？
      对每个事实按块内相位 o=(j+d) mod per 取位置集，逐 d 求 LOO-10。
      ⇒ 若 d=0 之外迅速掉到 chance，则"只有值 token 自身携带答案"，
         ⇒ 取值必须**位置选择性**，任何对邻位加权/池化的方案都不会有帮助。
  (b) **剂量-反应对照**：块均值能否辨识**同样稀疏**的另一个句内 token（动词，1–2/64 质量）？
      6 类（VERBS[i%6]，每类重数>=5，天花板 1.0）。若也不能 ⇒ 稀释解释在数量上自洽。

只读：不训练、不写、不设门。
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from R2_task91 import MOUNT, N_FACTS, build_chunks, build_facts, make_stack, model_and_tok
from probe91_valuetok import loo_predictions, value_positions, acc_on   # noqa: E402

OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
R = {"card": "DIR-91 rev 2.5", "kind": "post-verdict supplementary (non-gating)",
     "seeds": [], "profile": {}, "verb_control": {}, "dilution": {}}


def main():
    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    vals = sorted({f["val"] for f in facts})
    v2c = {v: i for i, v in enumerate(vals)}
    cls = [v2c[f["val"]] for f in facts]
    ncls = len(vals)
    vcls = [i % 6 for i in range(N_FACTS)]          # 动词类（VERBS[i%6]）
    cnt = {}
    for c in cls:
        cnt[c] = cnt.get(c, 0) + 1
    idx10 = [i for i in range(N_FACTS) if cnt[cls[i]] >= 2]
    per_info = value_positions(tok, facts)
    DS = list(range(-12, 13))

    for seed in (20261028, 7777):
        st = make_stack(gamma=0.0, no_write=True, seed=seed)
        Z, per = {}, per_info
        for li in MOUNT:
            Z[li] = []
        for fi in range(N_FACTS):
            model(input_ids=chunks[owner.index(fi) + 1].to(DEV_OK()))
            for li in MOUNT:
                pf = st._hs_in[li].reshape(-1, st._hs_in[li].shape[-1])
                Z[li].append((pf @ st.mems[li].Pv).detach().float().cpu())
        R["seeds"].append(seed)

        # ---------------- (a) 影响锥
        for li in MOUNT:
            prof = {}
            for d in DS:
                xs = []
                for fi in range(N_FACTS):
                    j, p = per[fi]["j"], per[fi]["period"]
                    o = (j + d) % p
                    P = [q for q in range(Z[li][fi].shape[0]) if q % p == o]
                    if not P:
                        xs.append(None); continue
                    xs.append(Z[li][fi][P].mean(dim=0))
                keep = [i for i, x in enumerate(xs) if x is not None]
                if len(keep) < N_FACTS:
                    continue
                X = torch.stack(xs)
                acc = acc_on(loo_predictions(X, cls, ncls), cls, idx10)
                prof[str(d)] = acc
            R["profile"][f"{seed}_L{li}"] = prof
            print(f"  seed {seed} L{li} 影响锥: " + " ".join(
                f"{d:+d}:{prof[str(d)]:.2f}" for d in DS if str(d) in prof), flush=True)

        # ---------------- (b) 剂量-反应：动词 6 类
        for li in MOUNT:
            row = {}
            for arm in ("mean_all", "vt_pos", "nonvt"):
                xs = []
                for fi in range(N_FACTS):
                    z = Z[li][fi]
                    V = per[fi]["V"]
                    if arm == "mean_all":
                        xs.append(z.mean(dim=0))
                    elif arm == "vt_pos":
                        xs.append(z[V].mean(dim=0))
                    else:
                        m = torch.ones(z.shape[0], dtype=torch.bool); m[V] = False
                        xs.append(z[m].mean(dim=0))
                X = torch.stack(xs)
                row[arm] = float(np.mean([int(a) == int(b) for a, b in
                                          zip(loo_predictions(X, vcls, 6), vcls)]))
            R["verb_control"][f"{seed}_L{li}"] = row
            print(f"  seed {seed} L{li} 动词6类(天花板1.0) LOO: " + " ".join(
                f"{a}={row[a]:.3f}" for a in row), flush=True)

        # ---------------- (c) 稀释量化
        for li in MOUNT:
            cs = []
            for fi in range(N_FACTS):
                z = Z[li][fi]; V = per[fi]["V"]
                a = z.mean(dim=0); b = z[V].mean(dim=0)
                cs.append(float(torch.nn.functional.cosine_similarity(a, b, dim=0)))
            R["dilution"][f"{seed}_L{li}"] = {
                "cos_mean_vs_vt": float(np.mean(cs)),
                "mass_frac": float(np.mean([len(per[fi]["V"]) / 64 for fi in range(N_FACTS)]))}
            print(f"  seed {seed} L{li} 稀释: cos(mean_all, vt_pos)={np.mean(cs):+.4f} "
                  f"质量占比={R['dilution'][f'{seed}_L{li}']['mass_frac']:.4f}", flush=True)

    with open(os.path.join(OUTD, "probe91_valprofile.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    print("\n★ 补充诊断完成 ⇒ out/probe91_valprofile.json", flush=True)


def DEV_OK():
    from R2_task91 import DEV
    return DEV


if __name__ == "__main__":
    main()
