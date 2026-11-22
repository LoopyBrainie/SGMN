#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""probe44d.py — DIR-44 的**第四探针**（**只定尺、不定论**；**不进任何判据**）。

★★ **机制更正**（来自 probe44.py 落盘的逐实例数据 —— 必须**先看清再开卡**）：
  run-1 的 `addr_gap = +0.556` 是**全局均值**；其**逐实例 sd = 3.21** ⇒ `se = 0.655`
  ⇒ **统计上不显著** ⇒ **初值处地址信息在损失里并不存在**（真实的 `addr_gap ≈ 0`）。
  ★ 这与 DIR-43 的 `ρ ≈ 0` **一致**（并非矛盾）⇒ ⇒ **更深层的问题：地址信息**根本没进入损失**。
  （probe44b 的「逐时刻相消」是**损失对寻址的局部敏感度在特定 q(0) 附近的瞬时形状**，不等于「损失携带寻址信息」。）

★★ **根本原因（机制解释）**：
  装置里**任务的『目标值』是** `gs[t-1] = tanh(c(t-1)·Pv1)` ⇒ **目标值是** `c` 的确定性函数**。
  ⇒ ⇒ 读头 `O_1` **只需**从 `c` 通道**直接逼近**目标值（记忆 `y=Wq` 经 `tanh(cat(c,y))` 进入 `m`，
  而 `m` 又直接决定 `hp` → 直接进损失），**不必「寻对那把键」** ⇒ ⇒ **降损**不必**指向寻址 ⇒ ⇒ `ρ ≈ 0`。
  ★ 这正是**「检索有用」与「目标可从输入直接推出」**的**根本冲突** ⇒ 在这一装置下**结构性地**无解
  （无论基座、无论 `c` 通道怎么改）。

★★ **真正的修复（本探针要验证的假设）**：
  **把『目标值』与** `c` **解耦** —— 让**值**来自**`c` 无法推出的外部码本**（离线、按**奇偶**索引）⇒ ⇒
  读头**必须**用 `y = W·q` 取回**正确的值** ⇒ ⇒ 降损**只能**靠「寻对那把键」⇒ ⇒ `ρ` 应当显著 > 0。
  ★ 出处（**不是自造议题**）：框架 §7.2「**记忆语义**」的「**信息寿命分解**／**长期干扰**／**replay 作用**／**模态持续度**」
  —— 装置里的『目标』现在是**瞬时**的（同一轮、可由 `c` 推），而记忆的核心问题是「**记忆比瞬时目标更持久**」。
  用**码本**=「值来自**过去**、且 `c` 不知道」⇒ 正是把记忆问题**变成必须依赖记忆**的那一步。

═══ 本探针要给的三个数（**开卡前必须先算**，纪律 55/56）═══
  ① **可达性**：码本臂下，损失对寻址敏感（`addr_gap` 显著 > 0）、训练**能**降损、
     `direct` 臂**仍**达 `cos ≥ 0.9`（判据**非空**，P3）。
  ② **功效**：`ρ`（训练后）的点估计与 sd ⇒ `n=400` 时 `3se` 能否检出。
  ③ **副作用**：码本臂下损失**会不会变平凡**（若 `L` 直接塌 0 ⇒ 判据**空**）。

═══ 纪律 ═══
  · 探针**只定尺、不定论**；读数**不得**进任何判据（纪律 106）。
  · 种子段 `23400+`，与 `23100/23200/23300` 探针段、DIR-43 正式轮 `22500+` **均不相交**。
  · 跑出的行**必须落盘**（纪律 108）⇒ `dir44/out/DIR44_probe_d.json`。
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
import probe44 as PR           # noqa: E402  ★ run-1 的 `rollout44`（A44-1 已证逐位等价）

OUT = os.path.join(ROOT, "dir44", "out")
os.makedirs(OUT, exist_ok=True)

SEED0 = M.SEED0
N_PROBE = 12
PROBE_OFF = 23400
PROBE_SEEDS = [SEED0 + PROBE_OFF + i for i in range(N_PROBE)]
TRAIN_WIN = (1, 8)
N_REP = 16
STEPS, LR = 250, 0.03
K_SHORT = 60


def build_codebook(layers=(1, 2)):
    r"""**离线码本**：按**奇偶**索引的 `(u_p0, u_p1)`，各层各自一份 ⇒ 值**无法由 `c` 推出**。

    ★ 与装置**同型**（`tanh(u·Pv1)`、`u` 周期 2、幅度 ≈ `√D_H`）⇒ 记忆动力学**不变**。
    ★ **独立种子流**（`seed + 424242`）⇒ 与实例种子、`a`、`Pk1`、`Pv1` **全无相关** ⇒
      ⇒ 目标值**不是** `c` 的函数 ⇒ ⇒ 堵死「`O_1` 从 `c` 通道直接逼近」这条捷径。
    """
    cb = {}
    for sd in PROBE_SEEDS[:1]:            # ★ 只需**一组**（所有实例共用同一码本）
        g = torch.Generator().manual_seed(int(sd) + 424242)
        for l in layers:
            x0 = torch.randn(R.D_H, generator=g) * (R.D_H ** 0.5)
            x1 = torch.randn(R.D_H, generator=g) * (R.D_H ** 0.5)
            cb[l] = (x0, x1)
    return cb


def rho_of(sd, phi):
    r"""`ρ = cos(−g_task, g_addr)`（与 DIR-43 同口径）。★ 必须重建 **live** 叶子。"""
    live = {k: (v.detach().clone().requires_grad_(True) if torch.is_tensor(v) else v)
            for k, v in phi.items()}
    P0 = T37._mk(sd, None)
    P = T37._mk(sd, live, train=True)
    L, _, _ = PR.rollout44(P, N_REP, sd, live, win=TRAIN_WIN)
    L.backward()
    g_task = live["Wq1"].grad.detach().clone()
    live["Wq1"].grad = None
    _, g_addr = T43.align_grad(P0, live, W=live["Wq1"], b=live["bq1"])
    return PR.cos2(-g_task, g_addr), float(g_task.norm())


def run_one(sd, v_mode, steps=K_SHORT, cb=None):
    r"""同一装置、同一初值；唯一自变量 = 值的来源。返回 `base`（不训）+ 训练后两档。"""
    if v_mode == "codebook":
        PR.V_MODE = "codebook"
        PR.V_CODEBOOK = cb
    else:
        PR.V_MODE = "device"
        PR.V_CODEBOOK = None

    phi0 = T42.reader_init(sd)
    P0 = T37._mk(sd, None)
    krand = PR.rand_key(P0, sd)

    def run_eval(phi):
        P = T37._mk(sd, phi, train=True)
        Lr, _, _ = PR.rollout44(P, N_REP, sd, phi, win=TRAIN_WIN)
        Lo, _, _ = PR.rollout44(T37._mk(sd, phi, train=True), N_REP, sd, phi,
                                win=TRAIN_WIN, q_mode="oracle")
        Lk, _, _ = PR.rollout44(T37._mk(sd, phi, train=True), N_REP, sd, phi,
                                win=TRAIN_WIN, q_mode="randkey", q_key=krand)
        L0, _, _ = PR.rollout44_y0(T37._mk(sd, phi, train=True), N_REP, sd, phi, win=TRAIN_WIN)
        return float(Lr), float(Lo), float(Lk), float(L0)

    Lr, Lo, Lk, L0 = run_eval(phi0)
    r0, gn0 = rho_of(sd, phi0)

    phi, _ = T42.train_reader42("learned", sd, steps=steps, train_win=TRAIN_WIN, lr=LR)
    Lr1, Lo1, Lk1, L01 = run_eval(phi)
    r1, gn1 = rho_of(sd, phi)

    phi2, _ = T42.train_reader42("learned", sd, steps=STEPS, train_win=TRAIN_WIN, lr=LR)
    Lr2, _, _, _ = run_eval(phi2)
    r2, gn2 = rho_of(sd, phi2)

    return {
        "seed": sd,
        "L_reader_init": Lr, "L_oracle_init": Lo, "L_randkey_init": Lk, "L_y0_init": L0,
        "addr_gap_init": Lk - Lo, "bypass_gap_init": Lr - L0,
        "rho_init": r0, "gn_task_init": gn0,
        "L_reader_K": Lr1, "rho_after_K": r1, "gn_task_after_K": gn1,
        "L_reader_250": Lr2, "rho_after_250": r2, "gn_task_after_250": gn2,
    }


def main():
    t0 = time.time()
    res = {"meta": {"n_probe": N_PROBE, "probe_off": PROBE_OFF,
                    "note": "探针只定尺、不定论；不进任何判据（纪律 106/108）"}}
    cb = build_codebook()
    rows = []
    for sd in PROBE_SEEDS:
        base = run_one(sd, "device")
        code = run_one(sd, "codebook", cb=cb)
        rows.append({"seed": sd, "device": base, "codebook": code})
    res["rows"] = rows

    # ── `direct` 臂的可达性（`cos_end` 与值来源无关 ⇒ 应**几乎不变**）
    drows = []
    for sd in PROBE_SEEDS[:3]:
        d = T43.direct43(sd, steps=STEPS)
        drows.append({"seed": sd, **d})
    res["direct_arm"] = drows
    res["meta"]["wall_s"] = round(time.time() - t0, 1)

    outp = os.path.join(OUT, "DIR44_probe_d.json")
    pathlib.Path(outp).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("WROTE", outp)

    for tag in ("device", "codebook"):
        print(f"\n== {tag}（值来源 = {'装置 h_pre（与 c 耦合）' if tag=='device' else '离线码本（与 c 解耦）'}）==")
        for key, label in (("addr_gap_init", "addr_gap(初值)"), ("bypass_gap_init", "bypass(初值)"),
                           ("rho_init", "ρ(初值)"), ("rho_after_K", f"ρ(K={K_SHORT})"),
                           ("rho_after_250", "ρ(250)")):
            xs = [r[tag][key] for r in rows]
            print(f"  {label:14s} = {statistics.fmean(xs):+.5f}"
                  f"  (sd {statistics.stdev(xs):.5f}, n={len(xs)})"
                  f"  正号 {sum(1 for x in xs if x > 0)}/{len(xs)}")
        print(f"  L 初值 {statistics.fmean([r[tag]['L_reader_init'] for r in rows]):.4f}"
              f" → K 步 {statistics.fmean([r[tag]['L_reader_K'] for r in rows]):.4f}"
              f" → 250 步 {statistics.fmean([r[tag]['L_reader_250'] for r in rows]):.4f}")
    print("\n== direct 臂 ==")
    for d in drows:
        print(" ", {k: round(v, 6) if isinstance(v, float) else v for k, v in d.items()})
    print(f"\n墙钟 {res['meta']['wall_s']} s")


if __name__ == "__main__":
    main()
