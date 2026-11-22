#!/opt/pytorch/bin/python
r"""DIR-47 · 开卡前诊断 ③（**不进判据**）：把 `C` **按槽位切开**，看 `replay` 的作用是「局部修复」还是「全局稳定」。

**为什么必须切**（`probe47c` 之前的两份探针暴露的混杂）：
`probe47c` 的 `Δ_on` 是在 **7 个被测槽位的均值**上算的，而 `rep_on` 臂**只重放了其中 `R` 个**。
若「重放哪个槽位就修好哪个槽位」，则均值会**自动**抬升约 `(R/7)×(1−C_nor)` —— 这**不是** replay 保护记忆，
而是「**被点到的槽位被修好**」这个算术后果。⇒ 必须把被测槽分成两组，分别报：

| 组 | 含义 | 判据角色 |
|---|---|---|
| `hit` | **被重放抽到**的槽位（`picks` 去重） | **P1**：`rep_on` 的 `C` 是否真的高（`replay` 修好了它碰到的） |
| `rest` | 未被抽到的槽位 | **P2**：`rep_on` 的 `C` 相对 `nor` 是否**不更差**（有无**全局惩罚**） |

★ 关键：`picks` 只由 `(seed, REPLAY_TAG)` 决定、**与臂无关** ⇒ `nor` 臂可按**同一组 `picks`** 切分 ⇒ 两组**配对可比**。

**要回答的**：`C_on(hit)`（应近 1）· `C_nor(hit)`（应近 0，因为该槽被后续新写覆盖）· `C_on(rest)` vs `C_nor(rest)`（差是否 ≈ 0）。

用法：`probe47c.py --n 200 --seed0 24960 --ctx 4 --S 8 --R 2 --out dir47/out/DIR47_probe_d.json`
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
from probe47a import write_pair, D_H   # noqa: E402

N_REP = T46.N_REP
REPLAY_TAG = 9899
REPLAY_OFF = 17
ARMS = ("nor", "rep_on")


def picks_of(sd, ctx, S, Rr):
    r"""`picks` **只**依赖 `(sd, ctx, S, Rr)` —— 与臂无关（两臂共用同一组槽位）。"""
    g = torch.Generator().manual_seed(int(sd) + REPLAY_TAG + REPLAY_OFF)
    return [ctx + int(torch.randint(1, S, (1,), generator=g).item()) for _ in range(Rr)]


def run_arm(sd, ctx, S, Rr, arm):
    P = R.build(sd, "V0")
    tot = ctx + S + Rr
    a = T46.content(sd, tot, "perstep")
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(tot)]
    zero = torch.zeros(D_H)
    K = [torch.tanh((zero if t == 0 else H[t - 1]) @ P["Pk1"]) for t in range(tot)]
    Vv = [torch.tanh(H[t] @ P["Pv1"]) for t in range(tot)]
    U, V = P["U1"].clone(), P["V1"].clone()
    for t in range(ctx + S):
        U, V = write_pair(U, V, K[t], Vv[t])
    extra = list(range(ctx + S, tot)) if arm == "nor" else picks_of(sd, ctx, S, Rr)
    for t in extra:
        U, V = write_pair(U, V, K[t], Vv[t])
    W = M._W(U, V)
    if not bool(torch.isfinite(W).all()):
        return {"finite": False, "slots": None, "picks": extra}
    per = {}
    for s in range(1, S):
        t = ctx + s
        num = float(W @ K[t] @ Vv[t])
        den = float((W @ K[t]).norm() * Vv[t].norm())
        per[s] = (num / den) if den > 0 else float("nan")
    return {"finite": all(math.isfinite(v) for v in per.values()), "slots": per, "picks": extra}


def agg(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return {"mean": None, "sd": None, "n": 0, "3se": None}
    m = sum(vals) / len(vals)
    s = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return {"mean": m, "sd": s, "n": len(vals), "3se": 3 * s / math.sqrt(len(vals)) if len(vals) else None}


def main():
    ap = argparse.ArgumentParser()
    for k, d in (("n", 200), ("seed0", 24960), ("ctx", 4), ("S", 8), ("R", 2)):
        ap.add_argument("--" + k, type=int, default=d)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ctx, S, Rr = args.ctx, args.S, args.R
    t0 = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    G = {"on_hit": [], "on_rest": [], "nor_hit": [], "nor_rest": [], "on_all": [], "nor_all": []}
    d_hit, d_rest = [], []
    n_div = 0
    dup = []
    rows = []
    for i in range(args.n):
        sd = args.seed0 + i
        pk = picks_of(sd, ctx, S, Rr)
        hit = sorted({p - ctx for p in pk})                 # 转到被测块内部编号 1..S−1
        o = {a: run_arm(sd, ctx, S, Rr, a) for a in ARMS}
        rows.append({"seed": sd, "picks": [p for p in pk], "hit": hit,
                     "nor": o["nor"], "rep_on": o["rep_on"]})
        if not (o["nor"]["finite"] and o["rep_on"]["finite"]):
            n_div += 1
            continue
        dup.append(Rr - len(hit))
        for a, key in (("rep_on", "on"), ("nor", "nor")):
            per = o[a]["slots"]
            hv = [per[s] for s in hit]
            rv = [per[s] for s in range(1, S) if s not in hit]
            G[key + "_hit"].append(sum(hv) / len(hv) if hv else None)
            G[key + "_rest"].append(sum(rv) / len(rv) if rv else None)
            G[key + "_all"].append(sum(per.values()) / len(per))
        d_hit.append(o["rep_on"]["slots"][hit[0]] if len(hit) == 1 else None)
        rh = [o["rep_on"]["slots"][s] for s in hit]
        rn = [o["nor"]["slots"][s] for s in hit]
        r2 = [o["rep_on"]["slots"][s] for s in range(1, S) if s not in hit]
        r3 = [o["nor"]["slots"][s] for s in range(1, S) if s not in hit]
        d_hit[-1] = (sum(rh) / len(rh)) - (sum(rn) / len(rn)) if rh else None
        d_rest.append((sum(r2) / len(r2)) - (sum(r3) / len(r3)) if r2 else None)

    out = {"ctx": ctx, "S": S, "R": Rr, "n": args.n, "seed0": args.seed0,
           "n_div": n_div, "cens": n_div / args.n,
           "dup_picks_mean": (sum(dup) / len(dup)) if dup else None,
           "levels": {k: agg(v) for k, v in G.items()},
           "d_hit": agg([x for x in d_hit if x is not None]),
           "d_rest": agg([x for x in d_rest if x is not None]),
           "wall_s": time.time() - t0}
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({**out, "rows": rows}, f, ensure_ascii=False)
    print(f"DONE → {args.out}  墙钟 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
