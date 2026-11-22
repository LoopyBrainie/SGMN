#!/opt/pytorch/bin/python
r"""DIR-55 · §0 定向探针：**多 trait 身份装置**能否给出「串扰矩阵」的可分辨读数？

要问的问题（出处 = `v2.4 §7.2 记忆语义 · 「人格」命题`；任务行 = DIR-51 遗留 ①②）：
  在 rank-$r$ 低秩记忆里同时维持 $n\ge 2$ 个身份 trait（键值对）时，
  **自读** $\mathrm{diag}\,C_{ii}=\cos(W k_i, v_i)$ 与**交叉读** $C_{ij}=\cos(W k_i, v_j)$ 的结构，
  以及**定向写入**（重复强化 trait 0）与**非定向写入**（自然内容）对它做了什么。

★ 本探针**不出判据**：只回答三件装置事实
  (D1) 串扰矩阵是否**可分辨**（diag 与 off-diag 的分离 ≫ 跨种子噪声底）；
  (D2) 是否存在**容量压力**（$n>r$ 时 diag 明显下降 ⇒ 「多 trait」不是免费）；
  (D3) 两个自变量（强化 / 自然写入）是否产生**非饱和、可测**的差异（含"定向性"读数）。
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


def unit(g, D):
    v = torch.randn(D, generator=g)
    return v / v.norm()


def build_traits(seed, n, key_mode):
    """返回 (P, keys, vals)。keys 已乘 `KEY_SCALE`（与 DIR-51 同式）。"""
    g = torch.Generator().manual_seed(seed)
    P = R.build(seed, "V0")
    if key_mode == "ortho":
        # 正交键：D_K 维标准基的前 n 个（n ≤ D_K 时逐位正交）
        assert n <= D_K, (n, D_K)
        keys = [KEY_SCALE * torch.eye(D_K)[i] for i in range(n)]
    elif key_mode == "rand":
        keys = [KEY_SCALE * unit(g, D_K) for _ in range(n)]
    else:
        raise ValueError(key_mode)
    vals = orthonormal_vals(g, n)
    return P, keys, vals


def orthonormal_vals(g, n):
    r"""$n$ 个单位正交值（$D_V\ge n$ 时逐位正交）—— 否则随机单位向量的交叉余弦地板 ≈ $1/\sqrt{D_V}$（$D_V{=}4$ 时 0.5）。"""
    assert n <= D_V, (n, D_V)
    while True:
        Q, _ = torch.linalg.qr(torch.randn(D_V, n, generator=g))
        if float(Q.abs().max()) < 0.999:      # 非退化
            return [Q[:, i].contiguous() for i in range(n)]


def write_seq(U, Vm, pairs, reps=1, alpha=1.0, interleave=True):
    r"""把 `pairs` 写 `reps` 轮。

    ★ `interleave=True`（**轮转**）：每轮把**所有**对写一次 —— 否则"逐对写满再写下一对"会给出**近因偏置**
      （探针 v3 实测：$n{=}2$ 顺序写时 $\cos$ 只有 0.61，因最后一对把前一对挤掉）。
    """
    if interleave:
        for _ in range(reps):
            for k, v in pairs:
                U, Vm = T48.step_pair(U, Vm, k, v, alpha)
    else:
        for k, v in pairs:
            for _ in range(reps):
                U, Vm = T48.step_pair(U, Vm, k, v, alpha)
    return U, Vm


def cmat(U, Vm, keys, vals):
    r"""**余弦**串扰矩阵：$C_{ij}=\cos(W k_i, v_j)$。

    ★ 为什么用余弦而不是投影 $((Wk_i)\cdot v_j)$：写规则的平衡点是 $c=\alpha/(\alpha+\lambda)$（幅度被正则压小），
      投影会被幅度**整体缩放**、把"关联是否正确"与"存得多不多"混在一起；余弦只问**方向**。
      （`proj` 同时返回，仅作强度诊断，**不作判据**。）
    """
    W = M._W(U, Vm).detach()
    out = []
    for k in keys:
        r = W @ k
        nr = float(r.norm())
        out.append([float((r @ v) / nr) if nr > 0 else 0.0 for v in vals])
    return out


def pmat(U, Vm, keys, vals):
    W = M._W(U, Vm).detach()
    return [[float((W @ k) @ v) for v in vals] for k in keys]


def natural_pairs(P, seed, m):
    """自然内容写入（逐字沿用 DIR-51 的生成式：`T48.content` + 装置 `Pk1/Pv1`）。"""
    a = T48.content(seed + 7919, m)
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(m)]
    z = torch.zeros(D_H)
    K = [torch.tanh((z if t == 0 else H[t - 1]) @ P["Pk1"]) for t in range(m)]
    V = [torch.tanh(H[t] @ P["Pv1"]) for t in range(m)]
    return list(zip(K, V))


def converge_trace(seed, n, key_mode):
    P, keys, vals = build_traits(seed, n, key_mode)
    out = {}
    U, Vm = P["U1"].clone(), P["V1"].clone()
    done = 0
    for target in (1, 4, 16, 64):
        U, Vm = write_seq(U, Vm, list(zip(keys, vals)), reps=target - done)
        done = target
        out[str(target)] = diag(cmat(U, Vm, keys, vals))
    return out


def diag(C):
    return [C[i][i] for i in range(len(C))]


def offdiag(C):
    n = len(C)
    return [C[i][j] for i in range(n) for j in range(n) if i != j]


def run_cell(seed, n, key_mode, arm, m=8, reps_store=64):
    P, keys, vals = build_traits(seed, n, key_mode)
    U, Vm = P["U1"].clone(), P["V1"].clone()
    U, Vm = write_seq(U, Vm, list(zip(keys, vals)), reps=reps_store)   # 轮转（见 write_seq 注释）
    C0 = cmat(U, Vm, keys, vals)
    if arm == "none":
        C1 = C0
    elif arm == "reinforce":                       # 定向：重复强化 trait 0
        U, Vm = write_seq(U, Vm, [(keys[0], vals[0])], reps=m)   # 单对，轮转无关
        C1 = cmat(U, Vm, keys, vals)
    elif arm == "natural":                         # 非定向：自然内容写入
        U, Vm = write_seq(U, Vm, natural_pairs(P, seed, m), reps=1)
        C1 = cmat(U, Vm, keys, vals)
    elif arm == "ortho_other":                     # 非定向：正交键 + 随机值（非身份内容）
        g = torch.Generator().manual_seed(seed + 555)
        pairs = []
        base = torch.eye(D_K)
        for j in range(m):
            kk = KEY_SCALE * base[(n + j) % D_K]
            pairs.append((kk, unit(g, D_V)))
        U, Vm = write_seq(U, Vm, pairs, reps=1)
        C1 = cmat(U, Vm, keys, vals)
    else:
        raise ValueError(arm)
    d0, d1 = diag(C0), diag(C1)
    return {"seed": seed, "n": n, "key_mode": key_mode, "arm": arm,
            "C0": C0, "C1": C1, "P1": pmat(U, Vm, keys, vals), "diag0": d0, "diag1": d1,
            "delta_diag": [b - a for a, b in zip(d0, d1)],
            "off0": offdiag(C0), "off1": offdiag(C1)}


def summarize(rows, n, key_mode, arm):
    sel = [r for r in rows if r["n"] == n and r["key_mode"] == key_mode and r["arm"] == arm]
    assert sel, (n, key_mode, arm)
    dd = [[r["delta_diag"][i] for r in sel] for i in range(n)]
    d0 = [[r["diag0"][i] for r in sel] for i in range(n)]
    d1 = [[r["diag1"][i] for r in sel] for i in range(n)]
    off = [x for r in sel for x in r["off1"]]
    return {"n": n, "key_mode": key_mode, "arm": arm, "n_seeds": len(sel),
            "diag0_mean": [statistics_fmean(x) for x in d0],
            "diag1_mean": [statistics_fmean(x) for x in d1],
            "delta_diag_mean": [statistics_fmean(x) for x in dd],
            "delta_diag_sd": [statistics_stdev(x) for x in dd],
            "offdiag_absmax_mean": statistics_fmean([abs(o) for o in off]),
            "C1_seed0": sel[0]["C1"]}


def statistics_fmean(xs):
    return math.fsum(xs) / len(xs)


def statistics_stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = statistics_fmean(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--reps-store", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    rows = []
    for s in range(a.seeds):
        seed = 55000 + s
        for n in (2, 3, 4):
            for km in ("ortho", "rand"):
                for arm in ("none", "reinforce", "natural", "ortho_other"):
                    rows.append(run_cell(seed, n, km, arm, m=a.m, reps_store=a.reps_store))
    summ = [summarize(rows, n, km, arm) for n in (2, 3, 4) for km in ("ortho", "rand")
            for arm in ("none", "reinforce", "natural", "ortho_other")]
    # ── D1 可分辨性：n=2 ortho，diag 与 |offdiag| 的分离
    s = {(x["n"], x["key_mode"], x["arm"]): x for x in summ}
    d1 = s[(2, "ortho", "none")]
    d1_sep = statistics_fmean(d1["diag1_mean"]) - d1["offdiag_absmax_mean"]
    # ── D2 容量压力：n=2 → n=3 的 diag 下降
    press = statistics_fmean(d1["diag1_mean"]) - statistics_fmean(s[(3, "ortho", "none")]["diag1_mean"])
    # ── D3 定向性：强化臂 (Δdiag0 − mean Δdiag_{others}) vs 自然臂同式
    def directed(cell):
        dd = cell["delta_diag_mean"]
        return dd[0] - statistics_fmean(dd[1:])
    dir_re = directed(s[(3, "ortho", "reinforce")])
    dir_nat = directed(s[(3, "ortho", "natural")])
    chk = {
        "D1_diag_mean": statistics_fmean(d1["diag1_mean"]),
        "D1_offdiag_absmax_mean": d1["offdiag_absmax_mean"],
        "D1_separation": d1_sep,
        "D1_ok": bool(d1_sep > 0.90 and statistics_fmean(d1["diag1_mean"]) > 0.99
                      and d1["offdiag_absmax_mean"] < 0.02),
        "D2_ortho_n3_diag_mean": statistics_fmean(s[(3, "ortho", "none")]["diag1_mean"]),
        "D2_pressure": press,
        "D2_ok": bool(press > 0.05),
        "D3_reinforce_directed": dir_re,
        "D3_natural_directed": dir_nat,
        "D3_ok": bool(abs(dir_re) > 3 * s[(3, "ortho", "reinforce")]["delta_diag_sd"][0] / math.sqrt(a.seeds)),
        "D4_not_saturated": bool(0.02 < statistics_fmean(d1["diag1_mean"]) < 0.98),
    }
    conv = {f"n{n}_{km}": converge_trace(55000, n, km) for n in (2, 3, 4) for km in ("ortho", "rand")}
    res = {"phase": "probe55a_v2", "seeds": a.seeds, "m": a.m, "reps_store": a.reps_store,
           "secs": time.time() - t0, "converge_trace_seed55000": conv,
           "checks": chk, "cells": summ}
    print(json.dumps({"checks": chk}, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()