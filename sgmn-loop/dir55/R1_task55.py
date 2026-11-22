#!/opt/pytorch/bin/python
r"""DIR-55 装置：多 trait 身份装置（卡 **rev 1.2** 逐条实现；`revolve/DIR-55/DIRECTION_rev55_v12.md`）。

★ 逐字复用（未改一字）：`dir31/R1_grad.py`（`_W` 与常数）· `dir33/R1_task33.py`（`build`/`KEY_SCALE`）·
  `dir48/R1_task48.py`（`step_pair` 写动力学 · `content` 自然内容生成式）。
★ 唯一自变量 = **写入协议**（六臂）：`none` / `natural_m1` / `reinforce_m1` / `natural_r8` / `order012` / `order210`。
★ 测量对象 = 余弦串扰矩阵 $C_{ij}=\cos(W k_i, v_j)$（卡 §2.3）；幸存集 $S=\{i: C_{ii}\ge0.50\}$。
★ 本文件**只产出读数**，不判结局（结局由 `adjudicate55.py` 独立重算）。
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
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M
import R1_task33 as R
import R1_task48 as T48

D_H, D_K, D_V = M.D_H, M.D_K, M.D_V
KEY_SCALE = R.KEY_SCALE

# ── 冻结常数（卡 §2.2 / §7 `A55-12`；**逐字**，不得改）
STORE = 64
SURV = 0.50
R_NAT = 8
K_MIN = 0.25    # rev 1.4：守卫界（实测 min‖k‖=0.4963，n=864）
SEED0 = 56000
N_SEEDS = 96
SEEDS = list(range(SEED0, SEED0 + N_SEEDS))
PROBE_SEEDS = set(range(55000, 55024))          # 探针段（`A55-11`：不得与之相交）

ARMS_N2 = ("none", "natural_r8")
ARMS_N3 = ("none", "natural_m1", "reinforce_m1", "order012", "order210")


def orthonormal_vals(g, n):
    r"""$n$ 个单位正交值（与探针 `probe55a.orthonormal_vals` 逐字同式）。"""
    assert n <= D_V, (n, D_V)
    while True:
        Q, _ = torch.linalg.qr(torch.randn(D_V, n, generator=g))
        if float(Q.abs().max()) < 0.999:
            return [Q[:, i].contiguous() for i in range(n)]


def build_traits(seed, n, order=None):
    r"""键 = 标准基×`KEY_SCALE`；值 = QR 正交归一（同一 `Generator(seed)`）。"""
    g = torch.Generator().manual_seed(seed)
    P = R.build(seed, "V0")
    keys_all = [KEY_SCALE * torch.eye(D_K)[i] for i in range(n)]
    vals_all = orthonormal_vals(g, n)
    idx = list(range(n)) if order is None else [int(c) for c in order]
    return P, [keys_all[i] for i in idx], [vals_all[i] for i in idx], idx


def write_seq(U, Vm, pairs, reps=1):
    r"""**轮转**写（`DEF-55-1`：逐对写满会引入近因偏置）。"""
    for _ in range(reps):
        for k, v in pairs:
            U, Vm = T48.step_pair(U, Vm, k, v, 1.0)
    return U, Vm


def cmat(U, Vm, keys, vals):
    W = M._W(U, Vm).detach()
    out = []
    for k in keys:
        r = W @ k
        nr = float(r.norm())
        out.append([float((r @ v) / nr) if nr > 0 else 0.0 for v in vals])
    return out


def natural_pair(P, content_seed):
    r"""**非退化**自然对（`DEF-55-5`）：取 `T48.content` 的 $t{=}1$ 项 ⇒ $k\neq0$。"""
    a = T48.content(content_seed + 7919, 2)
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(2)]
    k = torch.tanh(H[0] @ P["Pk1"])
    v = torch.tanh(H[1] @ P["Pv1"])
    return k, v


def diag(C):
    return [C[i][i] for i in range(len(C))]


def survivors(C, thr=SURV):
    return [i for i, x in enumerate(diag(C)) if x >= thr]


def stored_state(seed, n, order=None):
    P, keys, vals, idx = build_traits(seed, n, order)
    U, Vm = P["U1"].clone(), P["V1"].clone()
    U, Vm = write_seq(U, Vm, list(zip(keys, vals)), reps=STORE)
    return P, keys, vals, idx, U, Vm


def cell(seed, n, arm):
    r"""一个格：返回可序列化读数（**不含判据**）。"""
    assert seed in SEEDS and seed not in PROBE_SEEDS
    if arm == "order210":
        assert n == 3, ("order210 只在 n=3", n)
        order = (2, 1, 0)
    elif arm in ("none", "order012"):
        order = (0, 1, 2) if n == 3 else None
    else:
        order = (0, 1, 2) if n == 3 else None
    P, keys, vals, idx, U, Vm = stored_state(seed, n, order)
    C_store = cmat(U, Vm, keys, vals)
    out = {"seed": seed, "n": n, "arm": arm, "idx": idx,
           "C_store": C_store, "diag_store": diag(C_store), "S": survivors(C_store)}
    if arm in ("none", "order012", "order210"):
        out["C"] = C_store
        out["diag"] = diag(C_store)
        out["S_after"] = survivors(C_store)
    elif arm == "natural_m1":
        k, v = natural_pair(P, seed)
        assert float(k.norm()) >= K_MIN, ("DEF-55-5 守卫", float(k.norm()))
        U2, V2 = write_seq(U, Vm, [(k, v)], reps=1)
        C = cmat(U2, V2, keys, vals)
        out.update({"C": C, "diag": diag(C), "S_after": survivors(C),
                    "k_norm": float(k.norm())})
    elif arm == "reinforce_m1":
        U2, V2 = write_seq(U, Vm, [(keys[0], vals[0])], reps=1)
        C = cmat(U2, V2, keys, vals)
        out.update({"C": C, "diag": diag(C), "S_after": survivors(C)})
    elif arm == "natural_r8":
        reps = []
        for r in range(R_NAT):
            k, v = natural_pair(P, seed * 100 + r)     # 8 个**不同**内容种子（A55-14）
            assert float(k.norm()) >= K_MIN, ("DEF-55-5 守卫", float(k.norm()))
            U2, V2 = write_seq(U, Vm, [(k, v)], reps=1)   # ★ 每条都从**同一** (U,Vm) 出发
            C = cmat(U2, V2, keys, vals)
            reps.append({"r": r, "C": C, "diag": diag(C), "k_norm": float(k.norm())})
        out["reps"] = reps
        out["diag"] = diag(C_store)          # 基线（供裁决器算 Δ）
        out["S_after"] = out["S"]
    else:
        raise ValueError(arm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("cells", "single", "smoke"))
    ap.add_argument("--seed", type=int, default=SEED0)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--arm", default="none")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "dir55", "out"))
    ap.add_argument("--n-smoke", type=int, default=8)
    # rev 2.0：种子段与产物前缀可参数化（默认值 = rev 1.4 的冻结值，行为不变）
    ap.add_argument("--seed0", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--prefix", default="DIR55r13")
    ap.add_argument("--card-rev", default="rev1.4")
    ap.add_argument("--card-md5", default="ae4300936747b92a9b510b9011dbd411")
    a = ap.parse_args()
    t0 = time.time()
    global SEEDS
    if a.seed0 is not None or a.n_seeds is not None:
        s0 = SEED0 if a.seed0 is None else a.seed0
        nS = N_SEEDS if a.n_seeds is None else a.n_seeds
        SEEDS = list(range(s0, s0 + nS))
        assert not (set(SEEDS) & PROBE_SEEDS), ("种子段与探针段相交", len(set(SEEDS) & PROBE_SEEDS))
    os.makedirs(a.out_dir, exist_ok=True)
    if a.mode == "single":
        print(json.dumps(cell(a.seed, a.n, a.arm), ensure_ascii=False))
        return
    if a.mode == "smoke":
        seeds = SEEDS[: a.n_smoke]
    else:                                   # cells：**全量**（96 种子）
        seeds = SEEDS
    assert len(seeds) == (a.n_smoke if a.mode == "smoke" else len(SEEDS))
    jobs = [(2, arm) for arm in ARMS_N2] + [(3, arm) for arm in ARMS_N3]
    for n, arm in jobs:
        rows = [cell(s, n, arm) for s in seeds]
        res = {"dir": "DIR-55", "card": a.card_rev, "card_md5": a.card_md5,
               "n": n, "arm": arm, "seeds": seeds, "rows": rows,
               "consts": {"STORE": STORE, "SURV": SURV, "R_NAT": R_NAT, "K_MIN": K_MIN},
               "secs": time.time() - t0}
        name = f"{a.prefix}_{arm}_n{n}.json" if a.mode == "cells" else f"DIR55smoke_{arm}_n{n}.json"
        path = os.path.join(a.out_dir, name)
        json.dump(res, open(path, "w"), ensure_ascii=False)
        print(f"[{arm} n={n}] {len(rows)} 行 → {path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()