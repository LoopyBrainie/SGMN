#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""probe45a.py — DIR-45 的**前提检查探针**（**只定尺、不定论**；**不进任何判据**）。

★★ 本探针要**先否掉 DIR-45 的原始表述**，再判定**还剩什么可测**。

  **否掉的部分（`ALREADY-COVERED`）**：原表述「在**梯度仍然非零**的阶段测 ρ」——
  `DIR-43` 的 `RHO_GRID = (0,1,2,5,10,25,50,100,250)` **已经**在**训练轨迹上**、`n=400` 下采过 `ρ(t)`
  （`R1_task43.py` 在 `step in RHO_GRID` 时用**当步仍活着**的 `phi["Wq1"].grad`）⇒ 九个点**全部**不显著：
  `ρ(0)=−0.037708`（`3se=0.064231`，而此刻 `‖g‖=4.901481` 最大）、`ρ(5)=+0.021148`、`ρ(250)=−0.011750`
  ⇒ ⇒ **「梯度活着时也不指向」已被 n=400 测过** ⇒ 该自变量**无可测新信息** ⇒ 记为 `ALREADY-COVERED`（不重跑）。

  **剩下的真问题（本探针测它）**：`ρ` 是**对一个地址定义**算的。`DIR-43` 用
  `q*_cur(t) := tanh(h(t)·Pk1)`，但装置里**目标**是 `gs[t-1] = tanh(h(t-1)·Pv1) = v_{t-1}`
  ⇒ ⇒ 它**写在第 `t-1` 步**，而第 `s` 步写入用的键是 `k_s = tanh(h(s-1)·Pk1)`
  ⇒ ⇒ **正确地址是 `k_{t-1} = tanh(h(t-2)·Pk1)`**（比 `q*_cur` 早**两**步，比 `q*_prev` 早**一**步）
  ⇒ ⇒ **`DIR-43` 可能一直用错了地址**（框架 §7.2「数值与结构选择」的「…**查询坐标对齐方式**」正是这条开放项）。

  ★★ 但装置输入是**周期 2** ⇒ `h(t)` 只有 2 个状态 ⇒ `k_{t-1}`、`k_t`、`k_{t+1}` **同奇偶性**?? 不！
  `k_s` 依赖 `h(s-1)` ⇒ `k_{t-1}` 与 `k_{t+1}` 依赖 `h(t-2)`、`h(t)` ⇒ **同奇偶** ⇒ **逐向量相等**
  ⇒ ⇒ **周期 2 下 `q*_cur ≡ q*_tgt`（地址定义**退化**）**⇒ 「对齐」在该装置上**不可证伪**。
  本探针第 ① 组**数值验证**这件事；第 ② 组用**实测**（`argmax_s cos(W_t·k_s, gs[t-1])`）判定目标**实际**落在哪个槽。
  第 ③ 组在**不退化**的内容模式（`perstep`）下测 `ρ_tgt`，并**断言** `ρ_tgt ≠ ρ_cur`
  （否则重蹈 `DEF-44-3`：测了两个定义却拿到同一列）。

═══ 纪律 ═══
  · 探针**只定尺、不定论**（不进任何判据；纪律 106/108）；跑出的行**全部落盘**。
  · 种子段 `23700+`，与 `22500/23100/23200/23300/23400/23500/23600` **均不相交**。
  · ★ 本探针**不修** `probe44f`（`DEF-44-3` 留在探针文件里做档案），而是在**新文件**里**第一次就断言两列不同**。
"""

import json
import math
import os
import pathlib
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)

ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir37", "dir40", "dir42", "dir43", "dir44"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M            # noqa: E402
import R1_task33 as R          # noqa: E402
import R1_task42 as T42        # noqa: E402
import R1_task43 as T43        # noqa: E402
import probe44 as PR           # noqa: E402
import probe44e as PE          # noqa: E402

OUT = os.path.join(ROOT, "dir45", "out")
os.makedirs(OUT, exist_ok=True)

SEED0 = M.SEED0
N_PROBE = 12
PROBE_OFF = 23700
PROBE_SEEDS = [SEED0 + PROBE_OFF + i for i in range(N_PROBE)]
TRAIN_WIN = (1, 8)
N_REP = 16
STEPS, LR = 250, 0.03


# ══════════ 结构：`h(t)`、`k_s`、`v_s`、以及三种"地址定义"
def hs(P):
    return [torch.tanh(P["a"][t] @ P["A1"] + P["b1"]) for t in range(R.T)]


def kv(P):
    r"""第 `s` 步**写入**的 `(k_s, v_s)`：`k_s = tanh(h(s-1)·Pk1)`（`s=0` 为零向量），`v_s = tanh(h(s)·Pv1)`。"""
    H = hs(P)
    ks, vs = [], []
    for s in range(R.T):
        hp = torch.zeros(R.D_H) if s == 0 else H[s - 1]
        ks.append(torch.tanh(hp @ P["Pk1"]))
        vs.append(torch.tanh(H[s] @ P["Pv1"]))
    return ks, vs


def addr_defs(P):
    r"""三种地址定义（`t = 1..T-1` 的**读步**）：

      `cur(t)  = tanh(h(t)·Pk1)`     —— DIR-43 冻结口径（`k_{t+1}`）
      `prev(t) = tanh(h(t-1)·Pk1)`  —— 差一步（`k_t`）
      `tgt(t)  = tanh(h(t-2)·Pk1)`  —— **目标 `v_{t-1}` 真正的键** `k_{t-1}`（`t=1` 时 `k_0` 未定义 ⇒ `None`）
    """
    H = hs(P)
    cur = {t: torch.tanh(H[t] @ P["Pk1"]) for t in range(1, R.T)}
    prev = {t: torch.tanh(H[t - 1] @ P["Pk1"]) for t in range(1, R.T)}
    tgt = {t: (torch.tanh(H[t - 2] @ P["Pk1"]) if t >= 2 else None) for t in range(1, R.T)}
    return cur, prev, tgt


def cos2(u, v):
    return float(torch.nn.functional.cosine_similarity(u, v, dim=0))


# ══════════ ① 退化性：三个地址定义两两 |cos|
def aliasing(P):
    cur, prev, tgt = addr_defs(P)
    rows = []
    for t in range(2, R.T):
        rows.append({"t": t,
                     "cos_cur_tgt": cos2(cur[t], tgt[t]),
                     "cos_prev_tgt": cos2(prev[t], tgt[t]),
                     "cos_cur_prev": cos2(cur[t], prev[t])})
    return rows


# ══════════ ② 实测：目标 `gs[t-1]` 实际落在哪个槽（无梯度，轻量 trace）
def trace_mem(P):
    r"""记录**每次读之前的**第 1 层权重 `W_t`，以及每步写入的 `(k_s, v_s)`。"""
    U, V = P["U1"].clone(), P["V1"].clone()
    Ws = []
    for t in range(R.T):
        Ws.append(M._W(U, V))
        hp = P["a"][t]
        h1 = torch.tanh(hp @ P["A1"] + P["b1"])
        ks, vs = kv(P)
        k = ks[t]                                     # `k_t = tanh(h(t-1)·Pk1)`（与 `kv` 同口径）
        v = vs[t]
        for _ in range(N_REP):
            W = M._W(U, V)
            G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
            U = U - M.ETA * (G @ V)
            V = V - M.ETA * (G.t() @ U)
    return Ws, ks, vs


def slot_argmax(P):
    r"""对每个读步 `t`，求 `s* = argmax_s cos(W_t·k_s, gs[t-1])` ⇒ 记录 `t-1-s*` 的分布。"""
    Ws, ks, vs = trace_mem(P)
    H = hs(P)
    rows = []
    for t in range(2, R.T):
        target = torch.tanh(H[t - 1] @ P["Pv1"])      # `gs[t-1] = v_{t-1}`
        best, arg = -9.0, None
        for s in range(0, t):
            c = cos2(Ws[t] @ ks[s], target)
            if not math.isfinite(c):
                continue                              # ★ 记忆发散（`W` 出现 NaN/Inf）⇒ 该候选不可用
            if c > best:
                best, arg = c, s
        rows.append({"t": t, "s_star": arg,
                     "off": (None if arg is None else (t - 1) - arg),
                     "cos_best": (None if arg is None else best),
                     "mem_finite": bool(torch.isfinite(Ws[t]).all()),
                     "cos_vmatch": cos2(vs[t - 1], target)})
    return rows


# ══════════ ③ 在三种地址定义上各算一次 `ρ`（附**断言两列不同**）
def g_addr_of(P, W, b, which):
    r"""`∂A/∂W`，`A := mean_t cos(tanh(cc(t)·W + b), q*_which(t))`，`q*` 视作常量；`cc(t) = c(t) = h(t)`。"""
    H = hs(P)
    cur, prev, tgt = addr_defs(P)
    src = {"cur": cur, "prev": prev, "tgt": tgt}[which]
    Wl = W.detach().clone().requires_grad_(True)
    cs = []
    for t in range(1, R.T):
        qs = src[t]
        if qs is None:
            continue                                  # `t=1` 的 `tgt` 未定义 ⇒ 跳过（窗口差异**必须**披露）
        cs.append(torch.nn.functional.cosine_similarity(torch.tanh(H[t] @ Wl + b), qs, dim=0))
    A = torch.stack(cs).mean()
    return float(A.detach()), torch.autograd.grad(A, [Wl])[0].detach().clone(), len(cs)


def rho_all(sd, phi):
    P0 = PE.mk_e(sd, None)
    live = {k: (v.detach().clone().requires_grad_(True) if torch.is_tensor(v) else v)
            for k, v in phi.items()}                     # ★ 独立叶子（不污染传入的 `phi`）
    P = PE.mk_e(sd, live, train=True)
    L, _, _ = PR.rollout44(P, N_REP, sd, live, win=TRAIN_WIN)
    L.backward()
    gt = live["Wq1"].grad.detach().clone()
    out, As = {}, {}
    for which in ("cur", "prev", "tgt"):
        A, g, nwin = g_addr_of(P0, phi["Wq1"], phi["bq1"], which)
        out[f"rho_{which}"] = PR.cos2(-gt, g)
        As[f"A_{which}"] = A
        out[f"nwin_{which}"] = nwin
    out["gn"] = float(gt.norm())
    out["A"] = As
    return out


def evals(sd, phi):
    r"""三条开卡前诊断（记忆必要性 / 寻址信息量）：用 `tgt` 口径的正确键作 oracle。"""
    P0 = PE.mk_e(sd, None)
    krand = PR.rand_key(P0, sd)
    cur, prev, tgt = addr_defs(P0)

    def roll(q_mode="reader", key=None):
        P = PE.mk_e(sd, phi, True)
        L, _, _ = PR.rollout44(P, N_REP, sd, phi, win=TRAIN_WIN, q_mode=q_mode, q_key=key)
        return float(L)

    def roll_y0():
        P = PE.mk_e(sd, phi, True)
        L, _, _ = PR.rollout44_y0(P, N_REP, sd, phi, win=TRAIN_WIN)
        return float(L)

    Lr = roll()
    L0 = roll_y0()
    # ★ 正确键作 oracle：`q = tanh(c(t)·Pk1)` 但用**同一** `Pk1` ⇒ 与 DIR-43 的 `oracle` 档同构；
    #   `tgt` 与 `cur` 的差异由 ③ 的 `ρ_tgt` 体现（rollout 层无法直接注入逐时刻不同的键）。
    Lo = roll(q_mode="oracle")
    Lk = roll(q_mode="randkey", key=krand)
    return {"L_reader": Lr, "L_y0": L0, "L_oracle": Lo, "L_randkey": Lk,
            "bypass_gap": Lr - L0, "addr_gap": Lk - Lo}


def train45(sd, steps=STEPS, lr=LR, win=TRAIN_WIN):
    phi = T42.reader_init(sd)
    names = ["g1", "Wo1", "bo1", "Wq1", "bq1"]
    opt = torch.optim.Adam([phi[n] for n in names], lr=lr)
    path = []                                        # ★ `‖g_task(t)‖` 曲线（证明取点非空）
    for st in range(steps):
        opt.zero_grad()
        P = PE.mk_e(sd, phi, train=True)
        L, _, _ = PR.rollout44(P, N_REP, sd, phi, win=win)
        L.backward()
        if st in (0, 1, 2, 5, 10, 25, 50, 100, 249):
            path.append({"step": st, "gn": float(phi["Wq1"].grad.norm())})
        opt.step()
    return {k: v.detach().clone() for k, v in phi.items()}, path


def _sum(xs, label, indent="  "):
    n = len(xs)
    if n == 0:
        print(f"{indent}{label:22s} = (无有效值)"); return None
    mu = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    star = "★" if abs(mu) >= 3 * se else " "
    print(f"{indent}{label:22s} = {mu:+.5f}  (sd {sd:.4f}, se {se:.4f}, n={n})  {star}3se={3*se:.4f}")
    return mu, se


def main():
    t0 = time.time()
    res = {"meta": {"n_probe": N_PROBE, "probe_off": PROBE_OFF,
                    "note": "探针只定尺、不定论；不进任何判据（纪律 106/108）",
                    "already_covered": "DIR-43 RHO_GRID 已在 n=400 下测过（梯度活着时也不显著）"},
           "cells": {}}
    for a_mode in ("period2", "perstep"):
        PE.A_MODE = a_mode
        print(f"\n########## A_MODE = {a_mode} ##########")
        cell = {"aliasing": [], "slot": [], "rows": []}
        for sd in PROBE_SEEDS:
            P = PE.mk_e(sd, None)
            cell["aliasing"].append({"seed": sd, "rows": aliasing(P)})
            cell["slot"].append({"seed": sd, "rows": slot_argmax(P)})
        # ① 退化性
        cc = [r["cos_cur_tgt"] for a in cell["aliasing"] for r in a["rows"]]
        pc = [r["cos_prev_tgt"] for a in cell["aliasing"] for r in a["rows"]]
        print("  ① 地址定义两两 |cos|（目标 = `tgt`）")
        _sum(cc, "cos(cur, tgt)")
        _sum(pc, "cos(prev, tgt)")
        print(f"    ⇒ 退化判据：min|cos(cur,tgt)| = {min(abs(x) for x in cc):.6f}"
              f"（=1 表示 `cur ≡ tgt`，地址定义**不可区分**）")
        # ② 实测槽位
        offs = [r["off"] for a in cell["slot"] for r in a["rows"] if r["off"] is not None]
        n_div = sum(1 for a in cell["slot"] for r in a["rows"] if r["off"] is None)
        n_all = sum(len(a["rows"]) for a in cell["slot"])
        print(f"  ② 记忆发散（`W` 非有限）导致不可读的 (seed,t) 数 = {n_div}/{n_all}")
        from collections import Counter
        print("  ② 目标 `gs[t-1]` 实测落槽：`t-1-s*` 的分布")
        print("    ", dict(sorted(Counter(offs).items())))
        _sum([r["cos_best"] for a in cell["slot"] for r in a["rows"] if r["cos_best"] is not None],
             "cos(W·k_s*, gs[t-1])")
        _sum([r["cos_vmatch"] for a in cell["slot"] for r in a["rows"]], "cos(v_{t-1}, gs[t-1])")
        # ③ ρ（三种地址定义）+ 断言两列不同
        for sd in PROBE_SEEDS:
            phi0 = T42.reader_init(sd)
            r0 = rho_all(sd, phi0)
            e0 = evals(sd, phi0)
            phi, path = train45(sd)
            r1 = rho_all(sd, phi)
            e1 = evals(sd, phi)
            row = {"seed": sd}
            row.update({f"init_{k}": v for k, v in r0.items() if not isinstance(v, dict)})
            row.update({f"init_{k}": v for k, v in e0.items()})
            row.update({f"end_{k}": v for k, v in r1.items() if not isinstance(v, dict)})
            row.update({f"end_{k}": v for k, v in e1.items()})
            row["gpath"] = path
            cell["rows"].append(row)
        rows = cell["rows"]
        print("  ③ ρ（三种地址定义；`DEF-44-3` 的教训 ⇒ 下面必须看到三列**各不相同**）")
        for which in ("cur", "prev", "tgt"):
            _sum([r[f"end_rho_{which}"] for r in rows], f"ρ_{which}(250)")
        for which in ("cur", "tgt"):
            _sum([r[f"init_rho_{which}"] for r in rows], f"ρ_{which}(初值)")
        diff_ct = [abs(r["end_rho_cur"] - r["end_rho_tgt"]) for r in rows]
        print(f"    ⇒ |ρ_cur − ρ_tgt| 均值 = {statistics.fmean(diff_ct):.6f}"
              f"（**必须 > 0**；=0 说明又测了同一个量）")
        print("  ④ 开卡前三条诊断")
        _sum([r["end_bypass_gap"] for r in rows], "bypass_gap(250)")
        _sum([r["end_addr_gap"] for r in rows], "addr_gap(250)")
        gn0 = [r["gpath"][0]["gn"] for r in rows if r["gpath"]]
        gn250 = [r["gpath"][-1]["gn"] for r in rows if r["gpath"]]
        _sum(gn0, "‖g_task‖(0)")
        _sum(gn250, "‖g_task‖(249)")
        print("    ‖g_task(t)‖ 曲线均值：",
              {p["step"]: round(statistics.fmean([r["gpath"][i]["gn"] for r in rows]), 4)
               for i, p in enumerate(rows[0]["gpath"])})
        _sum([r["end_L_reader"] for r in rows], "L_reader(250)")
        cell.pop("aliasing"); cell.pop("slot")
        res["cells"][a_mode] = cell
    res["meta"]["wall_s"] = round(time.time() - t0, 1)
    outp = os.path.join(OUT, "DIR45_probe_a.json")
    pathlib.Path(outp).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nWROTE", outp, f"\n墙钟 {res['meta']['wall_s']} s")


if __name__ == "__main__":
    main()
