#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""probe44c.py — DIR-44 的**第三探针**（**只定尺、不定论**；**不进任何判据**）。

★ 机制（来自 `probe44b.py` run-1，分解**精确**：自检 `max_err = 8.9e-16`）：
  · 地址成分**逐时刻都存在**：`mean_t |cos(g_t, ᾡ_t)| = 0.457`（初值）/ `0.351`（训练后）；
  · 但**符号随时刻翻转**（正 ≈ 3.6–4.1 / 负 ≈ 3.9–4.4，共 8 个时刻）⇒ **逐时刻求和相消**：
    相消比 `‖Σg_t‖ / Σ‖g_t‖` = **0.698**（初值）/ **0.1025**（训练后）⇒ ⇒ `ρ ≈ 0`。
  · **窗口错配**（任务梯度 `t∈[1,8]` vs DIR-43 的 `A` 窗口 `t∈[1,15]`）**只贡献 ~0.004** ⇒ **不是**主因。

★★ 由此得到的**机制解释（假设 H44）**：
  装置里，任务的**目标** `gs[t-1] = tanh(c(t-1)·Pv1)` 是**读上下文 `c` 的确定性函数**
  ⇒ 而 `c` 周期为 2 ⇒ 只有 **2 个** `(c → 目标)` 对 ⇒ ⇒ **读头 `O_1` 只需记住 2 个向量就能
  完全绕过寻址**（`m` 由 `c` 通道直接算出，与 `y` 无关）⇒ ⇒ **降损不必指向寻址** ⇒ ⇒ `ρ ≈ 0`。
  ⇒ ⇒ 干预：**让读头 `O_1` 用不到 `c` 通道**（`Wo1` 的 c-block 冻结为 0）⇒ ⇒ 降损的**唯一**通道
  变成「取对钥匙」⇒ ⇒ 任务梯度**应当**指向寻址。
  ★ 出处：框架 §7.2「数值与结构选择」的「**$\mathcal Q_l,\mathcal O_l$ 的内部结构**」
  （框架 §3 的**签名** `m_{l,t}=\mathcal O_l(c_{l,t},y_{l,t};\phi_l^o)` **不变**；
  零权重 = 「内部结构选择不使用 `c`」，**不是**改签名）。

═══ 本探针要给的三个数（**开卡前必须先算**，纪律 55/56）═══
  ① **可达性**：新结构下损失**仍**对寻址敏感（`addr_gap > 0`）、训练**仍**能降损、
     `direct` 臂**仍**达 `cos ≥ 0.9`（判据**非空**，P3）。
  ② **功效**：`ρ` 的点估计与 sd（⇒ `n=400` 时 `3se` 能否检出）。
  ③ **副作用**：`c` 通道置零后损失**会不会变平凡**（`L` 直接塌到 0 ⇒ 判据**空**）。

═══ 纪律 ═══
  · 探针**只定尺、不定论**；读数**不得**进任何判据（纪律 106）。
  · 种子段 `23300+`，与 `23100+`（run-1）、`23200+`（run-2）、DIR-43 正式轮 `22500+` **均不相交**。
  · 跑出的行**必须落盘**（纪律 108）⇒ `dir44/out/DIR44_probe_c.json`。
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

OUT = os.path.join(ROOT, "dir44", "out")
os.makedirs(OUT, exist_ok=True)

SEED0 = M.SEED0
N_PROBE = 12
PROBE_OFF = 23300
PROBE_SEEDS = [SEED0 + PROBE_OFF + i for i in range(N_PROBE)]
TRAIN_WIN = (1, 8)
N_REP = 16
STEPS, LR = 250, 0.03
K_SHORT = 60                  # ★ 探针用**短**训练（只为定尺；正式轮仍 250 步）


def zero_c_block(phi):
    r"""把 `Wo1` 的 **`c` 通道**（前 `D_H` 行）置零 ⇒ `O_1` 只看 `y`。"""
    with torch.no_grad():
        phi["Wo1"][:R.D_H, :].zero_()
    return phi


def train_variant(sd, steps=K_SHORT, c_frozen=True):
    r"""`train_reader42` 的**逐字派生** + 一个自变量：**`Wo1` 的 c-block 冻结为 0**。

    ★ 冻结方式：c-block **不进优化器**（`names` 只含 `Wo1` 的 y-block 视图）⇒
      ⇒ 训练**不能**把 c-block 填回去 ⇒ ⇒ 干预在训练全程**有效**（这正是
      「内部结构选择不使用 `c`」的实现）。
    ★ 其它一切逐字不变：`reader_init`、Adam、`lr`、损失窗口、`trace42`。
    """
    phi = T42.reader_init(sd)
    if c_frozen:
        zero_c_block(phi)
    # ★ `Wo1` 必须是**叶子**才能进优化器 ⇒ 用**数据**拼（不带图）
    _w = torch.cat([phi["Wo1"][:R.D_H, :].detach(),
                    phi["Wo1"][R.D_H:, :].detach().clone()], dim=0)
    phi["Wo1"] = _w.detach().clone().requires_grad_(True)
    names = ["g1", "Wo1", "bo1", "Wq1", "bq1"]
    opt = torch.optim.Adam([phi[n] for n in names], lr=LR)
    for _ in range(steps):
        opt.zero_grad()
        P = T37._mk(sd, phi, train=True)
        L, _ = T42.trace42(P, N_REP, "learned", sd, phi, win=TRAIN_WIN)
        L.backward()
        if c_frozen:                      # ★ 把 c-block 的梯度**清零** ⇒ 它永远学不动
            with torch.no_grad():
                phi["Wo1"].grad[:R.D_H, :].zero_()
        opt.step()
        if c_frozen:                      # ★ 兜底：每步后 c-block 仍为 0
            with torch.no_grad():
                phi["Wo1"][:R.D_H, :].zero_()
    if c_frozen:
        with torch.no_grad():
            phi["Wo1"][:R.D_H, :].zero_()
    return phi


def eval_L(sd, phi, q_mode="reader", q_key=None, win=TRAIN_WIN):
    P = T37._mk(sd, phi, train=True)
    L, _, ex = PR.rollout44(P, N_REP, sd, phi, win=win, q_mode=q_mode, q_key=q_key)
    return float(L), ex


def rho_of(sd, phi):
    P0 = T37._mk(sd, None)
    P = T37._mk(sd, phi, train=True)
    L, _, _ = PR.rollout44(P, N_REP, sd, phi, win=TRAIN_WIN)
    L.backward()
    g_task = phi["Wq1"].grad.detach().clone()
    phi["Wq1"].grad = None
    _, g_addr = T43.align_grad(P0, phi, W=phi["Wq1"], b=phi["bq1"])
    return PR.cos2(-g_task, g_addr), float(g_task.norm())


def main():
    t0 = time.time()
    res = {"meta": {"n_probe": N_PROBE, "probe_off": PROBE_OFF, "k_short": K_SHORT,
                    "note": "探针只定尺、不定论；不进任何判据（纪律 106/108）"}}
    rows = []
    for sd in PROBE_SEEDS:
        P0 = T37._mk(sd, None)
        krand = PR.rand_key(P0, sd)

        # ── 基线（c-block 不冻结）vs 变体（c-block 冻结为 0），**同一**种子
        out = {"seed": sd}
        for tag, cf in (("base", False), ("cfrozen", True)):
            phi0 = T42.reader_init(sd)
            L_reader, _ = eval_L(sd, phi0)
            L_oracle, _ = eval_L(sd, phi0, q_mode="oracle")
            L_rand, _ = eval_L(sd, phi0, q_mode="randkey", q_key=krand)
            r0, gn0 = rho_of(sd, phi0)

            phi = train_variant(sd, steps=K_SHORT, c_frozen=cf)
            L1, _ = eval_L(sd, phi)
            r1, gn1 = rho_of(sd, phi)
            out[tag] = {
                "L_reader_init": L_reader, "L_oracle_init": L_oracle,
                "L_randkey_init": L_rand,
                "addr_gap_init": L_rand - L_oracle,
                "rho_init": r0, "gn_task_init": gn0,
                "L_after_K": L1, "rho_after_K": r1, "gn_task_after_K": gn1,
                "L_drop_ratio": (L1 - L_reader) / max(abs(L_reader), 1e-300),
            }
        rows.append(out)

    res["rows"] = rows
    # ── `direct` 臂的可达性（判据**非空**，P3）：`cos_end` 与 c-block 无关 ⇒ 应几乎不变
    drows = []
    for sd in PROBE_SEEDS[:3]:
        phi = T42.reader_init(sd)
        d = T43.direct43(sd, steps=STEPS)
        drows.append({"seed": sd, **{k: v for k, v in d.items()}})
    res["direct_arm"] = drows
    res["meta"]["wall_s"] = round(time.time() - t0, 1)

    outp = os.path.join(OUT, "DIR44_probe_c.json")
    pathlib.Path(outp).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("WROTE", outp)

    for tag in ("base", "cfrozen"):
        print(f"\n== {tag} ==")
        print(f"  addr_gap(初值) = {statistics.fmean([r[tag]['addr_gap_init'] for r in rows]):+.6f}")
        print(f"  ρ(初值)   = {statistics.fmean([r[tag]['rho_init'] for r in rows]):+.4f}"
              f"  (sd {statistics.stdev([r[tag]['rho_init'] for r in rows]):.4f})")
        print(f"  ρ(K={K_SHORT} 步后) = {statistics.fmean([r[tag]['rho_after_K'] for r in rows]):+.4f}"
              f"  (sd {statistics.stdev([r[tag]['rho_after_K'] for r in rows]):.4f})")
        print(f"  L 初值 {statistics.fmean([r[tag]['L_reader_init'] for r in rows]):.4f}"
              f" → K 步后 {statistics.fmean([r[tag]['L_after_K'] for r in rows]):.4f}"
              f"  (降幅 {statistics.fmean([r[tag]['L_drop_ratio'] for r in rows]):+.3f})")
        print(f"  ‖g_task‖ 初值 {statistics.fmean([r[tag]['gn_task_init'] for r in rows]):.4f}"
              f" · K 步后 {statistics.fmean([r[tag]['gn_task_after_K'] for r in rows]):.4f}")
    print("\n== direct 臂 ==")
    for d in drows:
        print(" ", {k: v for k, v in d.items() if k != "meta_keys"})
    print(f"\n墙钟 {res['meta']['wall_s']} s")


if __name__ == "__main__":
    main()
