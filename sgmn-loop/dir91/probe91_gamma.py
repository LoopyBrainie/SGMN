#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""DIR-91 · rev2.2 **更正轮：在同一脚本内隔离 $\gamma$**（`DEF-91-22`）

上一轮把「$\gamma{=}0.05\Rightarrow\mathrm{rel}{=}2.95$」与「$\gamma{=}0\Rightarrow0.1914$」
当作「只改 $\gamma$」，**不成立**：两者出自**两个脚本**，且**聚合口径不同**
（前者 `np.mean` over 32、后者 `.median()`）、**测量时刻不同**（写入瞬间 v.s. 终态）、
**$prev\_hs$ 污染不同**（600 v.s. 599）、且**前者从未报 $W^0$ 基线**。

本探针在**同一脚本、同一口径、同一时刻、同一对约定**下扫 $\gamma\in\{0,0.05\}$ × 2 seeds，
**同时报 mean 与 median、含 $W^0$ 基线**，并把 $prev\_hs$ 显式置 `None` 对齐正式协议（599 次写）。

另按 `DEF-91-22` 修正聚合：**报逐层比值，不报跨层中位数之比**。

★ 只测量；不产生裁决、不写证据行、不改冻结阈值。
"""
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from R2_task91 import (DEV, MOUNT, N_FACTS, build_chunks, build_facts, dump,
                       key_matrices, make_stack, model_and_tok, W_of, write_phase)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def relvec(W, K, V):
    return ((K @ W.T) - V).norm(dim=1) / V.norm(dim=1).clamp_min(1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[20261028, 7777])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.05])
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    tok, model = model_and_tok()
    facts = build_facts(tok); chunks, owner = build_chunks(tok, facts)
    res = {"note": "same script/convention/timing; mean AND median; W0 baseline included",
           "dtype_dev": "float32", "mount": list(MOUNT), "rows": []}

    for seed in a.seeds:
        for g in a.gammas:
            st = make_stack(gamma=g, seed=seed)
            try:
                for m in st.mems.values():
                    m.prev_hs = None            # 对齐正式协议：首块不写 ⇒ 599 次
                write_phase(model, st, chunks)
                K, V = key_matrices(model, st, facts, chunks, owner)   # 终态、同块对
                row = {"seed": seed, "gamma": g,
                       "n_writes": int(sum(m.write_calls for m in st.mems.values()) // len(st.mems)),
                       "per_layer": {}}
                for li in MOUNT:
                    m = st.mems[li]
                    r_post = relvec(W_of(m).float(), K[li], V[li])
                    r_base = relvec(m.W0.detach().float(), K[li], V[li])
                    post_m, post_md = float(r_post.mean()), float(r_post.median())
                    base_m, base_md = float(r_base.mean()), float(r_base.median())
                    row["per_layer"][li] = {
                        "post_mean": post_m, "post_median": post_md,
                        "base_mean": base_m, "base_median": base_md,
                        "ratio_mean": base_m / max(post_m, 1e-30),
                        "ratio_median": base_md / max(post_md, 1e-30)}
                res["rows"].append(row)
                r0 = row["per_layer"]
                print(f"[seed={seed} gamma={g}] writes={row['n_writes']} | "
                      f"mean {r0[MOUNT[0]]['post_mean']:.4f}/{r0[MOUNT[1]]['post_mean']:.4f}/"
                      f"{r0[MOUNT[2]]['post_mean']:.4f} | "
                      f"median {r0[MOUNT[0]]['post_median']:.4f}/{r0[MOUNT[1]]['post_median']:.4f}/"
                      f"{r0[MOUNT[2]]['post_median']:.4f}", flush=True)
                print(f"          base mean {r0[MOUNT[0]]['base_mean']:.4f}/"
                      f"{r0[MOUNT[1]]['base_mean']:.4f}/{r0[MOUNT[2]]['base_mean']:.4f} | "
                      f"ratio(mean) {r0[MOUNT[0]]['ratio_mean']:.2f}/"
                      f"{r0[MOUNT[1]]['ratio_mean']:.2f}/{r0[MOUNT[2]]['ratio_mean']:.2f}", flush=True)
            finally:
                st.remove()
    dump(res, "probe91_gamma.json")
    print("\n★ 完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
