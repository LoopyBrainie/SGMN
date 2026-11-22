#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""补充对照：**排除"6 类比 27 类容易"的混淆**。

`mean_all -> 动词(6 类)` = 1.000，而 `mean_all -> 值(27 类)` = 机会水平。
差异可能只是**类数**（6 vs 27），不是"值身份被位置局限"。
本文件用**同为 6 类**的值分组目标重测：
  verb6 = i % 6            （动词 VERBS[i%6]，5-6 事实/类）
  vgrp6 = cls % 6          （27 个值类按类号 mod 6 分 6 组，5-6 事实/组）
若 `vgrp6` 也落在机会水平而 `verb6` 在 1.0，则**类数不是解释**。
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from R2_task91 import MOUNT, N_FACTS, build_chunks, build_facts, make_stack, model_and_tok, DEV
from probe91_valuetok import loo_predictions, value_positions

OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
R = {"kind": "supplementary control (class-count confound)", "rows": {}}

tok, model = model_and_tok()
facts = build_facts(tok)
chunks, owner = build_chunks(tok, facts)
vals = sorted({f["val"] for f in facts})
v2c = {v: i for i, v in enumerate(vals)}
cls = [v2c[f["val"]] for f in facts]
tgt = {"verb6": [i % 6 for i in range(N_FACTS)], "vgrp6": [c % 6 for c in cls]}
cnt = {k: {c: v.count(c) for c in set(v)} for k, v in tgt.items()}
per = value_positions(tok, facts)
print("类分布:", {k: sorted(v.values()) for k, v in cnt.items()}, flush=True)

for seed in (20261028, 7777):
    st = make_stack(gamma=0.0, no_write=True, seed=seed)
    Z = {li: [] for li in MOUNT}
    for fi in range(N_FACTS):
        model(input_ids=chunks[owner.index(fi) + 1].to(DEV))
        for li in MOUNT:
            pf = st._hs_in[li].reshape(-1, st._hs_in[li].shape[-1])
            Z[li].append((pf @ st.mems[li].Pv).detach().float().cpu())
    for li in MOUNT:
        arms = {}
        for arm in ("mean_all", "vt_pos", "nonvt"):
            xs = []
            for fi in range(N_FACTS):
                z, V = Z[li][fi], per[fi]["V"]
                if arm == "mean_all":
                    xs.append(z.mean(dim=0))
                elif arm == "vt_pos":
                    xs.append(z[V].mean(dim=0))
                else:
                    m = torch.ones(z.shape[0], dtype=torch.bool); m[V] = False
                    xs.append(z[m].mean(dim=0))
            X = torch.stack(xs)
            for name, y in tgt.items():
                acc = float(np.mean([int(a) == int(b) for a, b in
                                     zip(loo_predictions(X, y, 6), y)]))
                arms[f"{arm}__{name}"] = acc
        R["rows"][f"{seed}_L{li}"] = arms
        print(f"  seed {seed} L{li}: " + " | ".join(
            f"{k}={v:.3f}" for k, v in arms.items()), flush=True)

with open(os.path.join(OUTD, "probe91_valgroup.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("\n★ 完成 ⇒ out/probe91_valgroup.json", flush=True)
