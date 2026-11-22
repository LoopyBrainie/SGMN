#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""probe44e.py — DIR-44 的**第五探针**（**只定尺、不定论**；**不进任何判据**）。

★★ 先说清「为什么还要第五个探针」（必须留档，否则下一 cycle 会重复踩坑）：

  **probe44d 的设计有两处**构造性缺陷**（本轮**实测**确认，见 `dir44/check_defect44.py`）：

  **缺陷 ①（已实测）**：`T42.train_reader42` 内部走 `trace42`（`dir42`），**不读** `probe44.V_MODE`
    ⇒ ⇒ 两臂**训后参数逐位相同**（`max|Δ| = 0.000e+00`）⇒ 探针 d 的 `codebook` 臂其实是
    「**device 损失训出来的控制器、在码本损失下测 ρ**」⇒ **混了两个协议** ⇒ `+0.293` **不可作为开卡依据**。

  **缺陷 ②（读代码即可确认）**：装置里的目标是 `gs[t-1] = tanh(hs[t-1]·Pv1)` ⇒ 它**恒等于**该步写入的值
    ⇒ **把『值』换成码本** ⇒ 记忆里存的**不再是目标** ⇒ 记忆**变得无用**（不是变得必要）
    ⇒ 探针 d 的 `L(250)=13.76`（vs device `0.058`）正是这个 ⇒ ⇒ 「降损只能靠寻址」**不成立**。

★★ **更深的根源（本探针要验证的那一条）**：
  装置输入 `a[t]` 是**周期 2**（`R.period2`）⇒ ⇒ `h(t)` 只有 **2 个状态**、`gs[t-1]` 只有 **2 个取值**
  ⇒ ⇒ 读头**只需记住这 2 个常数**就能降损（`c(t)` 也只有 2 个状态，足以索引它们）
  ⇒ ⇒ **记忆在任何时刻都不是必需的** ⇒ ⇒ `L(mem) − L(y=0) ≈ 0`（probe44 run-1 已实测：`+0.34 ± 1.82`）
  ⇒ ⇒ **`ρ≈0` 是这一结构的必然结果**，与「基座可分性」「`c` 通道」**都无关**。

★★ **本探针的唯一自变量**：把输入从**周期 2** 换成**逐步独立**（`perstep`）——其余**逐字不变**
  （同一 `A1,b1`、同一 `Pk1,Pv1`、同一尺度 `√D_H`、同一写入规则、同一读器初值、同一 Adam/250 步）。
  预期机制链：16 个**互不相同**的内容 ⇒ 目标 `gs[t-1] = tanh(h(t-1)·Pv1)` **不再能从 `c(t) = h(t)` 读出**
  （要从 `h(t)` 反演 `h(t-1)`，而非线性 tanh 网络做不到）⇒ ⇒ **记忆变成必需**
  ⇒ ⇒ `L(mem) − L(y=0)` 应当**显著 > 0**；且键逐步不同 ⇒ ⇒ 寻址**有信息**
  ⇒ ⇒ `L(randkey) − L(oracle)` 应当**显著 > 0**；⇒ ⇒ `ρ(250)` 应当**转正**。

★★ 判定「这条修改**可不可用**」的三条**开卡前必须先算**（纪律 55/56）：
  ① **记忆必要性**：`bypass_gap(250) = L_reader − L_y0` 显著 > 0（否则记忆仍可旁路 ⇒ 开了卡也是空判据）；
  ② **寻址信息量**：`addr_gap(250) = L_randkey − L_oracle` 显著 > 0（否则寻址与任务无关 ⇒ ρ 无意义）；
  ③ **判据非空**：`direct` 臂能达 `cos ≥ 0.9`；且 `L` 能**训下来**（`L(250) ≪ L(0)`）。
  ★ 三条**任一不成立** ⇒ DIR-44 **不得**以这一装置 revision 开卡 ⇒ 必须记 `INFEASIBLE-DEVICE`（**不是** PASS/FAIL）。

═══ 纪律 ═══
  · 探针**只定尺、不定论**；读数**不得**进任何判据（纪律 106）。
  · 种子段 `23500+`，与 `23100/23200/23300/23400` 探针段、DIR-43 正式轮 `22500+` **均不相交**。
  · 跑出的行**必须落盘**（纪律 108）⇒ `dir44/out/DIR44_probe_e.json`。
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
import R1_task37 as T37        # noqa: E402
import R1_task42 as T42        # noqa: E402
import R1_task43 as T43        # noqa: E402
import probe44 as PR           # noqa: E402  ★ 它的 `rollout44` 在 device 模式下**逐位**等于 `trace42`

OUT = os.path.join(ROOT, "dir44", "out")
os.makedirs(OUT, exist_ok=True)

SEED0 = M.SEED0
N_PROBE = 12
PROBE_OFF = 23500
PROBE_SEEDS = [SEED0 + PROBE_OFF + i for i in range(N_PROBE)]
TRAIN_WIN = (1, 8)
N_REP = 16
STEPS, LR = 250, 0.03

A_MODE = "period2"          # ★ 模块级开关："period2"（装置原样）| "perstep"（逐步独立）


# ══════════ 装置入口（唯一改动点 = `a[t]` 的取法）
def mk_e(sd, phi=None, train=False):
    r"""`T37._mk` 的**逐字派生** + `A_MODE` 开关（只改 `P["a"]` 的取法，其余逐字）。"""
    P = R.build(sd, "V0")
    if A_MODE == "perstep":
        # ★ 逐步独立：16 个互不相同的内容，**同尺度**（`√D_H`），同一生成器流
        g = torch.Generator().manual_seed(int(sd) + 7777)
        x = torch.randn(R.T, R.D_H, generator=g)
        x = x / x.norm(dim=1, keepdim=True) * (R.D_H ** 0.5)
        P["a"] = x
    else:
        R.period2(P, sd)
    if phi:
        P.update({k: (v if train else v.clone()) for k, v in phi.items()})
    return P


def train_reader_e(sd, steps=STEPS, lr=LR, win=TRAIN_WIN, qsrc="learned"):
    r"""`T42.train_reader42` 的**逐字派生**，只把装置入口换成 `mk_e`。"""
    phi = T42.reader_init(sd)
    names = ["g1", "Wo1", "bo1"] + (["Wq1", "bq1"] if qsrc == "learned" else [])
    opt = torch.optim.Adam([phi[n] for n in names], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        P = mk_e(sd, phi, train=True)
        L, _, _ = PR.rollout44(P, N_REP, sd, phi, win=win, q_mode=qsrc)
        L.backward()
        opt.step()
    return {k: v.detach().clone() for k, v in phi.items()}


def eval_arms(sd, phi):
    r"""四条读源在同一控制器上的损失（`Pr`/`Pk`/`P0` 与 `trace42` 同口径）。"""
    krand = PR.rand_key(mk_e(sd, None), sd)
    Lr, _, _ = PR.rollout44(mk_e(sd, phi, True), N_REP, sd, phi, win=TRAIN_WIN)
    Lo, _, _ = PR.rollout44(mk_e(sd, phi, True), N_REP, sd, phi, win=TRAIN_WIN, q_mode="oracle")
    Lk, _, _ = PR.rollout44(mk_e(sd, phi, True), N_REP, sd, phi, win=TRAIN_WIN,
                            q_mode="randkey", q_key=krand)
    L0, _, _ = PR.rollout44_y0(mk_e(sd, phi, True), N_REP, sd, phi, win=TRAIN_WIN)
    return float(Lr), float(Lo), float(Lk), float(L0)


def rho_of(sd, phi):
    r"""`ρ = cos(−g_task, g_addr)`（**DIR-43 §0.1 的下降方向约定**，逐字同口径）。"""
    live = {k: (v.detach().clone().requires_grad_(True) if torch.is_tensor(v) else v)
            for k, v in phi.items()}
    P = mk_e(sd, live, train=True)
    L, _, _ = PR.rollout44(P, N_REP, sd, live, win=TRAIN_WIN)
    L.backward()
    g_task = live["Wq1"].grad.detach().clone()
    P0 = mk_e(sd, None)
    _, g_addr = T43.align_grad(P0, live, W=live["Wq1"], b=live["bq1"])
    return PR.cos2(-g_task, g_addr), float(g_task.norm())


def direct_e(sd, steps=STEPS, lr=LR):
    r"""`T43.direct43` 的**逐字派生**（只换装置入口）：纯对齐优化 ⇒ `cos_end`。"""
    phi = T42.reader_init(sd)
    P = mk_e(sd, None)
    c, qs = T43.ctx_of(P), T43.qstar_of(P)
    W = phi["Wq1"].detach().clone().requires_grad_(True)
    b = phi["bq1"].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        cs = [torch.nn.functional.cosine_similarity(torch.tanh(c[t] @ W + b), qs[t], dim=0)
              for t in range(1, R.T)]
        (-torch.stack(cs).mean()).backward()
        opt.step()
    with torch.no_grad():
        cs = [float(torch.nn.functional.cosine_similarity(torch.tanh(c[t] @ W + b), qs[t], dim=0))
              for t in range(1, R.T)]
    return {"cos_end": min(cs), "cos_mean": statistics.fmean(cs)}


def run_one(sd):
    phi0 = T42.reader_init(sd)
    Lr0, Lo0, Lk0, L00 = eval_arms(sd, phi0)
    r0, g0 = rho_of(sd, phi0)
    phi = train_reader_e(sd, steps=STEPS)
    Lr1, Lo1, Lk1, L01 = eval_arms(sd, phi)
    r1, g1 = rho_of(sd, phi)
    d = direct_e(sd)
    # ★ 结构诊断：内容两两余弦（`perstep` 下应当**不再**是 2 个状态）
    P = mk_e(sd, None)
    c = PR.contexts(P)
    cc = max(abs(float(torch.nn.functional.cosine_similarity(c[i], c[j], dim=0)))
             for i in range(len(c)) for j in range(i + 1, len(c)))
    kk = PR.keys_of(P)
    kk_max = max(abs(float(torch.nn.functional.cosine_similarity(kk[i], kk[j], dim=0)))
                 for i in range(len(kk)) for j in range(i + 1, len(kk)))
    return {"seed": sd,
            "L_reader_init": Lr0, "L_oracle_init": Lo0, "L_randkey_init": Lk0, "L_y0_init": L00,
            "bypass_gap_init": Lr0 - L00, "addr_gap_init": Lk0 - Lo0,
            "rho_init": r0, "gn_init": g0,
            "L_reader_250": Lr1, "L_oracle_250": Lo1, "L_randkey_250": Lk1, "L_y0_250": L01,
            "bypass_gap_250": Lr1 - L01, "addr_gap_250": Lk1 - Lo1,
            "rho_250": r1, "gn_250": g1,
            "cos_end_direct": d["cos_end"], "cos_mean_direct": d["cos_mean"],
            "ctx_pair_cos_max": cc, "key_pair_cos_max": kk_max}


def _sig(xs, label):
    n = len(xs)
    mu = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    star = "★" if abs(mu) >= 3 * se else " "
    print(f"  {label:18s} = {mu:+.5f}  (sd {sd:.4f}, se {se:.4f}, n={n})"
          f"  {star}3se={3*se:.4f}  正号 {sum(1 for x in xs if x > 0)}/{n}")
    return mu, se


def main():
    t0 = time.time()
    res = {"meta": {"n_probe": N_PROBE, "probe_off": PROBE_OFF,
                    "note": "探针只定尺、不定论；不进任何判据（纪律 106/108）"},
           "cells": {}}
    for mode in ("period2", "perstep"):
        globals()["A_MODE"] = mode
        rows = [run_one(sd) for sd in PROBE_SEEDS]
        res["cells"][mode] = rows
        print(f"\n===== A_MODE = {mode} =====")
        for k, lab in (("bypass_gap_init", "bypass(初值)"), ("bypass_gap_250", "bypass(250)"),
                       ("addr_gap_init", "addr_gap(初值)"), ("addr_gap_250", "addr_gap(250)"),
                       ("rho_init", "ρ(初值)"), ("rho_250", "ρ(250)"),
                       ("cos_end_direct", "direct cos_end"),
                       ("ctx_pair_cos_max", "ctx 两两|cos|max"),
                       ("key_pair_cos_max", "key 两两|cos|max")):
            _sig([r[k] for r in rows], lab)
        print(f"  L: 初值 {statistics.fmean([r['L_reader_init'] for r in rows]):.4f}"
              f" → 250 步 {statistics.fmean([r['L_reader_250'] for r in rows]):.4f}"
              f"  ‖g_task‖ 初值 {statistics.fmean([r['gn_init'] for r in rows]):.4f}"
              f" → 250 步 {statistics.fmean([r['gn_250'] for r in rows]):.4f}")
    res["meta"]["wall_s"] = round(time.time() - t0, 1)
    outp = os.path.join(OUT, "DIR44_probe_e.json")
    pathlib.Path(outp).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nWROTE", outp, f"\n墙钟 {res['meta']['wall_s']} s")


if __name__ == "__main__":
    main()
