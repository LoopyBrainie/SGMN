#!/opt/pytorch/bin/python
r"""DIR-47 · 开卡前诊断（**不进判据**，卡 §0 明文）：`replay` 能否把容量失效点推后？

**出发点**：DIR-46（D-69）的机制读数是「`period2` 不失效，因为第 `s` 与第 `s−2` 次写**同一对** ⇒ 纠错被反复重申」。
⇒ 本探针把这句话**变成可判定的干预**：在**总写入数相同**、**被测槽位相同**的前提下，
把最后 `R` 次写从「**新对**」换成「**旧对的重放**」⇒ 唯一的自变量就是这 `R` 次写**写谁**。

**匹配设计（关键，否则 `PASS` 会是假象）**
- 臂 `nor`：`S` 个新对 + `R` 个**新对**（共 `M=S+R` 次写）⇒ 测**最早**的 `S−1` 个槽位。
- 臂 `rep`：`S` 个新对 + `R` 次**重放旧对**（从槽 `1..S−1` 均匀抽，共 `M=S+R` 次写）⇒ 测**同一批** `S−1` 个槽位。
- ⇒ 两臂**总写入数相同**（`M`）、**被测槽位相同**（同一批 `k_s,v_s`）、**种子相同**；
  `content(sd,·)` 逐行归一化 ⇒ 前 `S` 行**逐位相同** ⇒ `K[0..S−1]`/`V[0..S−1]` **两臂逐位相等**（本文件当场断言）。

**探针要回答的四件事**（DIR-46 的四条 + 一条新条）
1. `C(2)` 型锚点是否非空（`S ≥ 2` 时可读槽位非空）；
2. 判据点上的 `sd(C)` ⇒ 定 `n`；
3. `3se ≤ 0.05` 能否达到（`n=400`）；
4. **`N_div` 前置**：两个臂在 `M` 上的发散率是否都 ≤ `CENS_MAX=0.20`（DIR-46 已测 `perstep` 在 `N_w=16` 就 21.7% ⇒ 若这里也超，判据点必须下移）；
5. **等写入数校验**：`M_nor == M_rep` 逐实例断言（不是"设计上相等"，而是**数出来**相等）。

用法：`probe47a.py --n 80 --seed0 24700 --out dir47/out/DIR47_probe_a.json`
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
for _d in ("dir31", "dir33", "dir46"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task46 as T46      # noqa: E402  ★ 内容生成器 `content()` 与写入常数**逐字**沿用

N_REP = T46.N_REP
D_H = T46.D_H
REPLAY_TAG = 9899            # 重放抽取的生成器偏移（**开卡后冻结**）
CELLS = ((4, 4), (8, 2), (8, 4), (8, 8), (12, 4))   # (S, R)


def write_pair(U, V, k, v):
    r"""一次「对」的写入 = `N_REP` 步，**逐字**抄 `R1_task46.trace_one` 的内层循环。"""
    for _ in range(N_REP):
        W = M._W(U, V)
        G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
        U = U - M.ETA * (G @ V)
        V = V - M.ETA * (G.t() @ U)
    return U, V


def measure(W, K, Vv, S):
    r"""测槽 `1..S−1`（槽 0 恒跳过：`k_0 = 0` ⇒ `W·k_0 = 0`，见 `DEF-46-3`）。"""
    cos, n_skip = [], 0
    if not bool(torch.isfinite(W).all()):
        return None, 0, (S - 1), False
    for s in range(1, S):
        num = float(W @ K[s] @ Vv[s])
        den = float((W @ K[s]).norm() * Vv[s].norm())
        c = num / den if den > 0 else float("nan")
        if math.isfinite(c):
            cos.append(c)
        else:
            n_skip += 1
    return (sum(cos) / len(cos) if cos else None), len(cos), n_skip, True


def run_arm(sd, S, Rr, arm):
    r"""臂 `nor`（后 `Rr` 次写新对）vs `rep`（后 `Rr` 次重放旧对）。`M = S + Rr` 两臂相同。"""
    P = R.build(sd, "V0")
    M_tot = S + Rr
    a = T46.content(sd, M_tot, "perstep")          # ★ 前 S 行与 content(sd,S) 逐位相同（逐行归一化）
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(M_tot)]
    zero = torch.zeros(D_H)
    K = [torch.tanh((zero if s == 0 else H[s - 1]) @ P["Pk1"]) for s in range(M_tot)]
    Vv = [torch.tanh(H[s] @ P["Pv1"]) for s in range(M_tot)]

    U, V = P["U1"].clone(), P["V1"].clone()
    n_writes = 0
    for s in range(S):                              # 前 S 次写：两臂**逐位相同**
        U, V = write_pair(U, V, K[s], Vv[s]); n_writes += 1
    if arm == "nor":
        for s in range(S, M_tot):                   # 后 R 次：写**新**对
            U, V = write_pair(U, V, K[s], Vv[s]); n_writes += 1
        picks = list(range(S, M_tot))
    else:
        g = torch.Generator().manual_seed(int(sd) + REPLAY_TAG)
        # ★ DEF-47-1：`torch.randint(1, S)` 已是**半开区间** `[1, S)` = `1..S−1` ⇒ **不得**再 `+1`
        #   （run-1 就是因此把 p 抽成 `2..S`，撞上下面的 assert；该失败轮留档）
        picks = [int(torch.randint(1, S, (1,), generator=g).item()) for _ in range(Rr)]
        for p in picks:                             # 后 R 次：**重放**旧对（槽 1..S−1）
            assert 1 <= p <= S - 1
            U, V = write_pair(U, V, K[p], Vv[p]); n_writes += 1
    W = M._W(U, V)
    C, n_read, n_skip, finite = measure(W, K, Vv, S)
    assert n_writes == M_tot, (sd, arm, n_writes, M_tot)      # ★ 等写入数（数出来）
    assert n_read + n_skip == S - 1, (sd, arm, n_read, n_skip, S)
    return {"C": C, "n_read": n_read, "n_skip": n_skip, "finite": finite,
            "n_writes": n_writes, "picks": picks,
            "wnorm": (float(W.norm()) if finite else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed0", type=int, default=24700)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for (S, Rr) in CELLS:
        cs = {"nor": [], "rep": []}
        div = {"nor": 0, "rep": 0}
        diffs = []
        for i in range(args.n):
            sd = args.seed0 + i
            o = {arm: run_arm(sd, S, Rr, arm) for arm in ("nor", "rep")}
            # ★ 等写入数 + 前缀逐位相同（两臂共享前 S 次写的全部输入）
            assert o["nor"]["n_writes"] == o["rep"]["n_writes"] == S + Rr
            for arm in ("nor", "rep"):
                if o[arm]["C"] is None:
                    div[arm] += 1
                else:
                    cs[arm].append(o[arm]["C"])
            if o["nor"]["C"] is not None and o["rep"]["C"] is not None:
                diffs.append(o["rep"]["C"] - o["nor"]["C"])
            rows.append({"seed": sd, "S": S, "R": Rr, "nor": o["nor"], "rep": o["rep"]})
        def stat(v):
            if not v:
                return {"mean": None, "sd": None, "n": 0}
            m = sum(v) / len(v)
            sdv = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
            return {"mean": m, "sd": sdv, "n": len(v)}
        sn, sr, sdd = stat(cs["nor"]), stat(cs["rep"]), stat(diffs)
        summ = {"S": S, "R": Rr, "M": S + Rr, "n": args.n,
                "n_meas_slots": S - 1,
                "nor": sn, "rep": sr, "diff_rep_minus_nor": sdd,
                "n_div": div, "cens": {k: div[k] / args.n for k in div},
                "3se_nor": (3 * sn["sd"] / math.sqrt(sn["n"]) if sn["sd"] and sn["n"] else None),
                "3se_rep": (3 * sr["sd"] / math.sqrt(sr["n"]) if sr["sd"] and sr["n"] else None),
                "3se_diff": (3 * sdd["sd"] / math.sqrt(sdd["n"]) if sdd["sd"] and sdd["n"] else None)}
        print(json.dumps(summ, ensure_ascii=False), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cells": [c for c in CELLS], "n": args.n, "seed0": args.seed0,
                   "replay_tag": REPLAY_TAG, "wall_s": None, "rows": rows}, f,
                  ensure_ascii=False)
    print(f"\nDONE n={args.n} seed0={args.seed0} → {args.out}  墙钟 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
