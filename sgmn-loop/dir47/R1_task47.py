#!/opt/pytorch/bin/python
r"""DIR-47 · 实现：`replay` 的作用 —— **按槽位分组**判「局部修复 vs 全局改变」。

卡：`revolve/DIR-47/DIRECTION_rev47_v10.md`（rev 1.0，md5 `58cb94d44327ae00ca2236d6e56d67d2`，**阈值已冻结**）。
★ **逐字派生**：装置（`dir31/R1_grad.py` + `dir33/R1_task33.py`）、内容生成器与写入内层循环（`dir46/R1_task46.py`）
   **一字不改**；本文件只加**一个自变量**：末尾 `R=2` 次写**写谁**（`nor` = 新对 / `rep_on` = 重放被测块内部槽位）。
★ 判据量：`C(hit)`（P1）、`Δ_rest = C_on(rest) − C_nor(rest)`（P2，逐实例配对）。

用法：`R1_task47.py --arm {nor,rep_on} --n 500 --seed0 25200 --out PATH [--tag r10]`
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
import R1_task46 as T46      # noqa: E402

# ══════ 冻结常数（卡 §2 / §6；**开卡后一字不改**）
CTX = 3                 # 上下文块大小（写完再也不过问、永不测）
S_MEAS = 8              # 被测块大小（内部槽 1..7 被测）
R_EXTRA = 2             # 末尾额外写次数（两臂相同）
M_TOT = CTX + S_MEAS + R_EXTRA      # = 13
N_REP = T46.N_REP       # 16
D_H = T46.D_H
N_SLOTS = S_MEAS - 1    # = 7 个被测槽位
CONTENT_TAG = 7777      # 逐字沿用 dir46
REPLAY_TAG = 9899       # 逐字沿用 probe47a/b/c
REPLAY_OFF = 17
ARMS = ("nor", "rep_on")


def write_pair(U, V, k, v):
    r"""一次「对」的写入 = `N_REP` 步（**逐字**抄 `R1_task46.trace_one` 内层循环）。"""
    for _ in range(N_REP):
        W = M._W(U, V)
        G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
        U = U - M.ETA * (G @ V)
        V = V - M.ETA * (G.t() @ U)
    return U, V


def picks_of(sd):
    r"""末尾 `R_EXTRA` 次重放的目标槽位（**只依赖 seed**，与臂无关 ⇒ 两臂可配对切分）。"""
    g = torch.Generator().manual_seed(int(sd) + REPLAY_TAG + REPLAY_OFF)
    return [int(torch.randint(1, S_MEAS, (1,), generator=g).item()) for _ in range(R_EXTRA)]


def build_KV(sd, P):
    a = T46.content(sd, M_TOT, "perstep")
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(M_TOT)]
    zero = torch.zeros(D_H)
    K = [torch.tanh((zero if t == 0 else H[t - 1]) @ P["Pk1"]) for t in range(M_TOT)]
    Vv = [torch.tanh(H[t] @ P["Pv1"]) for t in range(M_TOT)]
    return K, Vv


def trace_one(sd, arm):
    r"""单实例单臂。返回逐槽位 `c_s`（`s=1..7`）、分组均值、`picks`、计数。"""
    assert arm in ARMS
    P = R.build(sd, "V0")
    K, Vv = build_KV(sd, P)
    U, V = P["U1"].clone(), P["V1"].clone()
    n_writes = 0
    for t in range(CTX + S_MEAS):                    # ★ 前缀：两臂**逐位相同**
        U, V = write_pair(U, V, K[t], Vv[t]); n_writes += 1
    pk = picks_of(sd)
    if arm == "nor":
        targets = list(range(CTX + S_MEAS, M_TOT))   # 新对 t=11,12
    else:
        targets = [CTX + s for s in pk]              # 重放被测块内部槽 `s=1..7`
    for t in targets:
        U, V = write_pair(U, V, K[t], Vv[t]); n_writes += 1

    W = M._W(U, V)
    finite = bool(torch.isfinite(W).all())
    slots, n_skip = {}, 0
    if finite:
        for s in range(1, S_MEAS):
            t = CTX + s
            num = float(W @ K[t] @ Vv[t])
            den = float((W @ K[t]).norm() * Vv[t].norm())
            c = num / den if den > 0 else float("nan")
            if math.isfinite(c):
                slots[str(s)] = c
            else:
                n_skip += 1
    else:
        n_skip = N_SLOTS

    hit = sorted({s for s in pk})                    # 1..7
    rest = [s for s in range(1, S_MEAS) if s not in hit]
    def mean_over(idx):
        vs = [slots[str(s)] for s in idx if str(s) in slots]
        return (sum(vs) / len(vs)) if vs else None
    row = {"seed": sd, "arm": arm, "n_writes": n_writes, "picks": pk,
           "hit": hit, "rest": rest, "finite": finite,
           "wnorm": (float(W.norm()) if finite else None),
           "slots": slots, "n_read": len(slots), "n_skip": n_skip,
           "C_hit": mean_over(hit), "C_rest": mean_over(rest),
           "C_all": mean_over(range(1, S_MEAS))}
    # ★ A47-1 / A47-5 逐实例当场断言
    assert n_writes == M_TOT, (sd, arm, n_writes)
    assert len(hit) + len(rest) == N_SLOTS, (sd, arm, hit, rest)
    assert len(slots) + n_skip == N_SLOTS, (sd, arm, len(slots), n_skip)
    assert finite or not slots, (sd, arm, "删失格却有可读槽")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=25200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="r10")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            sd = args.seed0 + i
            row = trace_one(sd, args.arm)
            row["tag"] = args.tag
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"[{args.arm}] {i+1}/{args.n}  {time.time()-t0:.1f}s", flush=True)
    print(f"DONE {args.arm} n={args.n} seed0={args.seed0} → {args.out}  墙钟 {time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
