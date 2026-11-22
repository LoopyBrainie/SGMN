#!/opt/pytorch/bin/python
r"""DIR-47 · 开卡前诊断 ②（**不进判据**）：三臂版 `replay` 干预的**变量分解**是否可行？

**为什么要有第二份探针**：`probe47a` 的两臂设计（`nor` vs `rep`）**把两种后果捆在一个差里** ——
`rep` 相对 `nor` 同时做了两件事：① **不出新干扰**（最后 `R` 次写没写新对）② **重申被测对**（纠错被反复施加）。
「replay 的作用」若只报 `Δ_on`，就无法说清它是"**免除了新干扰**"还是"**真的把旧对修好了**"。
⇒ 三臂设计用**前缀上下文对**（写完就再也不过问、**永不测**）做隔离：

| 臂 | 前 `ctx+S` 次写 | 最后 `R` 次写 | 被测槽位 |
|---|---|---|---|
| `nor` | 上下文 `ctx` + 被测 `S` | `R` 个**新对** | 被测块的 `1..S−1` |
| `rep_on` | 同上（**逐位相同**） | `R` 次**重放被测块** | 同上 |
| `rep_off` | 同上（**逐位相同**） | `R` 次**重放上下文块** | 同上 |

⇒ `Δ_on = C(rep_on) − C(nor)`（**replay 的作用 = 主判据**）· `Δ_off = C(rep_off) − C(nor)`（**只免干扰**）
⇒ `Δ_pure = Δ_on − Δ_off`（**纯重申**部分；卡上作**次判据/报告**，不单独判 `PASS`）。
三臂的**总写入数相同**（`ctx+S+R`）、**被测槽位相同**、**共享前缀逐位相同**（本文件当场断言）。

用法：`probe47b.py --n 80 --seed0 24780 --out dir47/out/DIR47_probe_b.json`
"""
import argparse
import json
import math
import os
import sys
import time

import torch

torch.set_default_dtype(torch.float64)

ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task46 as T46      # noqa: E402
from probe47a import write_pair, D_H   # noqa: E402  ★ 写入内层循环**逐字**复用（不重写）

N_REP = T46.N_REP
REPLAY_TAG = 9899            # 与 probe47a / 后续实现**同一**重放抽取器
CELLS = ((4, 8, 0), (4, 8, 1), (4, 8, 2))     # (ctx, S, R)（`R=0` = **锚点**：无额外写）
ARMS = ("nor", "rep_on", "rep_off")


def run_arm(sd, ctx, S, Rr, arm):
    r"""上下文块 `0..ctx−1` 永不测；被测块 `ctx..ctx+S−1`，测其内部槽 `1..S−1`。"""
    P = R.build(sd, "V0")
    tot = ctx + S + Rr
    a = T46.content(sd, tot, "perstep")
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(tot)]
    zero = torch.zeros(D_H)
    K = [torch.tanh((zero if t == 0 else H[t - 1]) @ P["Pk1"]) for t in range(tot)]
    Vv = [torch.tanh(H[t] @ P["Pv1"]) for t in range(tot)]

    U, V = P["U1"].clone(), P["V1"].clone()
    n_writes = 0
    for t in range(ctx + S):                    # ★ 三臂**逐位相同**的前缀
        U, V = write_pair(U, V, K[t], Vv[t]); n_writes += 1
    g = torch.Generator().manual_seed(int(sd) + REPLAY_TAG + 17)
    if arm == "nor" or Rr == 0:
        extra = list(range(ctx + S, tot))
    elif arm == "rep_on":
        # 被测块的**内部槽** `1..S−1`（跳过块首槽 0，其 `k` 含上下文或为零）
        extra = [ctx + int(torch.randint(1, S, (1,), generator=g).item()) for _ in range(Rr)]
    else:                                        # rep_off：重放**上下文块**（永不测）
        extra = [int(torch.randint(0, ctx, (1,), generator=g).item()) for _ in range(Rr)]
    for t in extra:
        U, V = write_pair(U, V, K[t], Vv[t]); n_writes += 1
    W = M._W(U, V)
    assert n_writes == tot, (sd, arm, n_writes, tot)

    finite = bool(torch.isfinite(W).all())
    cos, n_skip = [], 0
    if finite:
        for s in range(1, S):
            t = ctx + s
            num = float(W @ K[t] @ Vv[t])
            den = float((W @ K[t]).norm() * Vv[t].norm())
            c = num / den if den > 0 else float("nan")
            if math.isfinite(c):
                cos.append(c)
            else:
                n_skip += 1
    else:
        n_skip = S - 1
    assert len(cos) + n_skip == S - 1, (sd, arm)
    return {"C": (sum(cos) / len(cos)) if cos else None, "n_read": len(cos),
            "n_skip": n_skip, "finite": finite, "n_writes": n_writes, "extra": extra,
            "wnorm": (float(W.norm()) if finite else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed0", type=int, default=24780)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for (ctx, S, Rr) in CELLS:
        got = {a: [] for a in ARMS}
        div = {a: 0 for a in ARMS}
        pair = {"on": [], "off": [], "pure": []}
        for i in range(args.n):
            sd = args.seed0 + i
            o = {a: run_arm(sd, ctx, S, Rr, a) for a in ARMS}
            assert len({o[a]["n_writes"] for a in ARMS}) == 1
            for a in ARMS:
                if o[a]["C"] is None:
                    div[a] += 1
                else:
                    got[a].append(o[a]["C"])
            if all(o[a]["C"] is not None for a in ARMS):
                pair["on"].append(o["rep_on"]["C"] - o["nor"]["C"])
                pair["off"].append(o["rep_off"]["C"] - o["nor"]["C"])
                pair["pure"].append(o["rep_on"]["C"] - o["rep_off"]["C"])
            rows.append({"seed": sd, "ctx": ctx, "S": S, "R": Rr, "arms": o})

        def stat(v):
            if not v:
                return {"mean": None, "sd": None, "n": 0, "3se": None}
            m = sum(v) / len(v)
            s = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
            return {"mean": m, "sd": s, "n": len(v), "3se": 3 * s / math.sqrt(len(v)) if len(v) else None}
        summ = {"ctx": ctx, "S": S, "R": Rr, "M": ctx + S + Rr, "n": args.n,
                "cens": {a: div[a] / args.n for a in ARMS},
                "levels": {a: stat(got[a]) for a in ARMS},
                "d_on": stat(pair["on"]), "d_off": stat(pair["off"]), "d_pure": stat(pair["pure"])}
        print(json.dumps(summ, ensure_ascii=False), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cells": [list(c) for c in CELLS], "arms": list(ARMS), "n": args.n,
                   "seed0": args.seed0, "replay_tag": REPLAY_TAG, "rows": rows}, f,
                  ensure_ascii=False)
    print(f"\nDONE n={args.n} → {args.out}  墙钟 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
