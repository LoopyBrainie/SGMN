#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""probe44f.py — DIR-44 的**第六探针**（**只定尺、不定论**；**不进任何判据**）。

★★ 本探针要判定的是**一条定义层面的错位**，它来自 probe44e 的**负** `addr_gap`：

  probe44e 实测（n=12，`perstep`）：`addr_gap(250) = L(randkey) − L(oracle) = −1.86`
  ⇒ ⇒ **用「随机键」比用 `q*`（当前键）损失更小** ⇒ 说明 **DIR-43 的 `q*` 定义取错了键**。

  **为什么取错（可推导，一步代数）**：`trace42` 在第 `t` 步写入的**值**是 `v_t = tanh(h(t)·Pv1)`，
  而**键**是 `k_{t-1} = tanh(h(t-1)·Pk1)`；第 `t` 步的**目标**是 `gs[t-1] = tanh(h(t-1)·Pv1) = v_{t-1}`
  ⇒ ⇒ 要取回目标，查询必须对上 **`v_{t-1}` 的键**，即 `k_{t-2} = tanh(h(t-2)·Pk1)`
  ⇒ ⇒ 而 DIR-43 的 `q*(t) := tanh(c(t)·Pk1) = k_t`（**当前**键，连差两步）
  ⇒ ⇒ **「寻址方向」被测成了「把查询对准下一步将要写的键」** ⇒ `ρ≈0` 有一个**定义性**来源。

  ★ 且 `A43-6` 的守恒（`k1s[t-1] ≡ q*(t)`）当时是**在 `trace42` 的配对口径下**成立的
  —— 那是「读在第 `t` 步读到的槽位」，**不是**「目标所在槽位」⇒ ⇒ 该守恒**不构成**对齐正确的证据。

★★ **两条自变量（各自单变量，且都出自框架 §7.2 的开放项）**：

  | 自变量 | 取值 | 框架出处 |
  |---|---|---|
  | `Q_OFF`（**查询坐标对齐**） | `0` = 用 `c(t)`（装置原样）· `1` = 用 `c(t-1)` | 「数值与结构选择」行：「$\mathcal Q_l,\mathcal O_l$ 的…**查询坐标对齐方式**」 |
  | `A_MODE`（内容持续度） | `period2` · `perstep` | 「记忆语义」行：「信息寿命分解、长期干扰、replay 作用、**模态持续度**」 |

  ★ 本探针**不改**判据的尺度（`ρ` 仍按 **DIR-43 §0.1 的下降方向约定**），只改**被测的地址**：
  `q*_prev(t) := tanh(c(t-1)·Pk1)`（= **目标 `v_{t-1}` 的键**）⇒ ⇒ `ρ_prev := cos(−g_task, g_addr_prev)`。

═══ 本探针要给出的四个数（**开卡前必须先算**，纪律 55/56）═══
  ① `ρ_prev`（`Q_OFF=1` 格、训练后）的点估计与 sd ⇒ 功效；
  ② `ρ_cur`（同格）作对照 ⇒ 分辨「对齐」与「随便哪个方向」；
  ③ `addr_gap_prev := L(randkey_prev) − L(oracle_prev)` 是否**转正**（寻址**有信息**）；
  ④ `bypass_gap` 是否仍显著 < 0（**记忆仍是必需的**，否则空判据）。

═══ 纪律 ═══
  · 探针**只定尺、不定论**；读数**不得**进任何判据（纪律 106）。
  · 种子段 `23600+`，与 `23100…23500` 探针段、DIR-43 正式轮 `22500+` **均不相交**。
  · 落盘 `dir44/out/DIR44_probe_f.json`（纪律 108）。
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
import probe44 as PR           # noqa: E402
import probe44e as PE          # noqa: E402  ★ `mk_e` / `train_reader_e` 的 `A_MODE` 开关

OUT = os.path.join(ROOT, "dir44", "out")
os.makedirs(OUT, exist_ok=True)

SEED0 = M.SEED0
N_PROBE = 12
PROBE_OFF = 23600
PROBE_SEEDS = [SEED0 + PROBE_OFF + i for i in range(N_PROBE)]
TRAIN_WIN = (1, 8)
N_REP = 16
STEPS, LR = 250, 0.03

Q_OFF = 0                  # ★ 模块级开关：0 = `c(t)`（装置原样）| 1 = `c(t-1)`（对齐）


# ══════════ 带 `Q_OFF` 的 rollout（`trace42` 的逐字派生 + 只改 `q` 的取上下文）
def rollout_f(P, n_rep, sd, phi, win=TRAIN_WIN, q_src="reader", q_key=None, y_zero=False):
    r"""`trace42` 的**逐字派生**：唯一改动 = 第 1 层查询的**上下文**由 `Q_OFF` 选（`c(t)` / `c(t-1)`）。"""
    L = P["a"].new_zeros(())
    gs = R.targets(P)
    h_pre_prev = [None, None]
    c_prev = None
    for t in range(T := R.T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if li == 0:
                c = h
                cc = c if (Q_OFF == 0 or c_prev is None) else c_prev   # ★ 唯一改动
                if q_src == "oracle":
                    q = torch.tanh(cc @ P["Pk1"])
                elif q_src == "randkey":
                    q = torch.tanh(cc @ q_key)
                else:
                    q = torch.tanh(cc @ P["Wq1"] + P["bq1"])
            else:
                c = hp_prev.detach()
                q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            Wd = M._W(P[f"U{l}"], P[f"V{l}"]).detach()
            y = Wd @ q
            if y_zero and li == 0:
                y = torch.zeros_like(y)
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            hp[li] = h + P[f"g{l}"] * m
            hp_prev = hp[li]
        if t >= 1 and (win is None or (win[0] <= t <= win[1])):
            L = L + ((R.pout() @ hp[0] - gs[t - 1]) ** 2).sum()
        upd = {}
        for li, l in enumerate((1, 2)):
            hprev = h_pre_prev[li]
            if hprev is None:
                hprev = torch.zeros(R.D_H)
            k = torch.tanh(hprev.detach() @ P[f"Pk{l}"])
            v = torch.tanh(h_pre[li].detach() @ P[f"Pv{l}"])
            Uo, Vo = P[f"U{l}"], P[f"V{l}"]
            for _ in range(n_rep):
                W = M._W(Uo, Vo)
                G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
                upd[f"U{l}"] = Uo - M.ETA * (G @ Vo)
                upd[f"V{l}"] = Vo - M.ETA * (G.t() @ Uo)
            Uo, Vo = upd[f"U{l}"], upd[f"V{l}"]
            P[f"U{l}"], P[f"V{l}"] = Uo.detach(), Vo.detach()
        h_pre_prev = [hh.detach() for hh in h_pre]
        c_prev = h_pre[0].detach() if h_pre[0] is not None else None
        c_prev = torch.tanh(P["a"][t] @ P["A1"] + P["b1"]).detach()
    return L


def _qc(c, prev_c1, t):
    r"""查询上下文：`Q_OFF=0` ⇒ `c(t)`（装置原样）；`Q_OFF=1` ⇒ `c(t-1)`（对齐到目标的键）。"""
    if Q_OFF == 0:
        return c
    if t == 0 or prev_c1 is None:
        return c
    return prev_c1


def train_f(sd, steps=STEPS, lr=LR, win=TRAIN_WIN, q_src="learned"):
    phi = T42.reader_init(sd)
    names = ["g1", "Wo1", "bo1", "Wq1", "bq1"]
    opt = torch.optim.Adam([phi[n] for n in names], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        P = PE.mk_e(sd, phi, train=True)
        L = rollout_f(P, N_REP, sd, phi, win=win, q_src=q_src)
        L.backward()
        opt.step()
    return {k: v.detach().clone() for k, v in phi.items()}


def qstars(P):
    r"""`q*_cur(t) := tanh(c(t)·Pk1)`（DIR-43 原定义）与 `q*_prev(t) := tanh(c(t-1)·Pk1)`（目标 `v_{t-1}` 的键）。"""
    c = PR.contexts(P)
    return ({t: torch.tanh(c[t] @ P["Pk1"]) for t in range(1, R.T)},
            {t: torch.tanh(c[t - 1] @ P["Pk1"]) for t in range(1, R.T)})


def g_addr_of(P, phi, key=None, which="cur"):
    c = PR.contexts(P)
    cur, prev = qstars(P)
    qs = cur if which == "cur" else prev
    Wl = phi["Wq1"].detach().clone().requires_grad_(True)
    bl = phi["bq1"].detach().clone()
    kk = P["Pk1"] if key is None else key
    cs = []
    for t in range(1, R.T):
        cc = c[t] if Q_OFF == 0 else c[t - 1]
        tgt = torch.tanh(cc @ kk)
        cs.append(torch.nn.functional.cosine_similarity(torch.tanh(cc @ Wl + bl), tgt, dim=0))
    A = torch.stack(cs).mean()
    return torch.autograd.grad(A, [Wl])[0].detach().clone()


def rho_pair(sd, phi):
    r"""`ρ` 对**两种地址定义**各算一次（`g_task` 只算一次，`Q_OFF` 一致）。"""
    live = {k: (v.detach().clone().requires_grad_(True) if torch.is_tensor(v) else v)
            for k, v in phi.items()}
    P0 = PE.mk_e(sd, None)
    krand = PR.rand_key(P0, sd)
    P = PE.mk_e(sd, live, train=True)
    L = rollout_f(P, N_REP, sd, live, win=TRAIN_WIN, q_src="reader")
    L.backward()
    gt = live["Wq1"].grad.detach().clone()
    out = {}
    for which in ("cur", "prev"):
        out[f"rho_{which}"] = PR.cos2(-gt, g_addr_of(P0, live, which=which))
        out[f"rho_{which}_rand"] = PR.cos2(-gt, g_addr_of(P0, live, key=krand, which=which))
    out["gn"] = float(gt.norm())
    return out


def evals(sd, phi):
    P0 = PE.mk_e(sd, None)
    krand = PR.rand_key(P0, sd)
    Lr = float(rollout_f(PE.mk_e(sd, phi, True), N_REP, sd, phi))
    Lo_p = float(rollout_f(PE.mk_e(sd, phi, True), N_REP, sd, phi, q_src="oracle"))
    Lk_p = float(rollout_f(PE.mk_e(sd, phi, True), N_REP, sd, phi, q_src="randkey", q_key=krand))
    L0 = float(rollout_f(PE.mk_e(sd, phi, True), N_REP, sd, phi, y_zero=True))
    return {"L_reader": Lr, "L_oracle": Lo_p, "L_randkey": Lk_p, "L_y0": L0,
            "addr_gap_prev": Lk_p - Lo_p, "bypass_gap": Lr - L0}


def run_one(sd):
    phi0 = T42.reader_init(sd)
    r0 = rho_pair(sd, phi0)
    e0 = evals(sd, phi0)
    phi = train_f(sd, steps=STEPS)
    r1 = rho_pair(sd, phi)
    e1 = evals(sd, phi)
    row = {"seed": sd, "Q_OFF": Q_OFF}
    for k, v in r0.items():
        row[f"init_{k}"] = v
    for k, v in e0.items():
        row[f"init_{k}"] = v
    for k, v in r1.items():
        row[f"end_{k}"] = v
    for k, v in e1.items():
        row[f"end_{k}"] = v
    return row


def _sig(xs, label):
    n = len(xs)
    mu = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    star = "★" if abs(mu) >= 3 * se else " "
    print(f"  {label:20s} = {mu:+.5f}  (sd {sd:.4f}, se {se:.4f}, n={n})"
          f"  {star}3se={3*se:.4f}  正号 {sum(1 for x in xs if x > 0)}/{n}")
    return mu, se


def main():
    t0 = time.time()
    res = {"meta": {"n_probe": N_PROBE, "probe_off": PROBE_OFF,
                    "note": "探针只定尺、不定论；不进任何判据（纪律 106/108）"}, "cells": {}}
    for a_mode in ("period2", "perstep"):
        for q_off in (0, 1):
            PE.A_MODE = a_mode
            globals()["Q_OFF"] = q_off
            rows = [run_one(sd) for sd in PROBE_SEEDS]
            res["cells"][f"{a_mode}_q{q_off}"] = rows
            print(f"\n===== A_MODE={a_mode}  Q_OFF={q_off} =====")
            _sig([r["end_rho_prev"] for r in rows], "ρ_prev(250)")
            _sig([r["end_rho_cur"] for r in rows], "ρ_cur(250) [对照]")
            _sig([r["init_rho_prev"] for r in rows], "ρ_prev(初值)")
            _sig([r["end_addr_gap_prev"] for r in rows], "addr_gap_prev(250)")
            _sig([r["end_bypass_gap"] for r in rows], "bypass_gap(250)")
            print(f"  L: 初值 {statistics.fmean([r['init_L_reader'] for r in rows]):.4f}"
                  f" → 250 步 {statistics.fmean([r['end_L_reader'] for r in rows]):.4f}"
                  f" ‖g_task‖ 250 步 {statistics.fmean([r['end_gn'] for r in rows]):.4f}")
    res["meta"]["wall_s"] = round(time.time() - t0, 1)
    outp = os.path.join(OUT, "DIR44_probe_f.json")
    pathlib.Path(outp).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nWROTE", outp, f"\n墙钟 {res['meta']['wall_s']} s")


if __name__ == "__main__":
    main()
