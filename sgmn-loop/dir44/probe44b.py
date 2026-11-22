#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""probe44b.py — DIR-44 的**第二探针**（**只定尺、不定论**；**不进任何判据**）。

★ 为什么有这一探针（接 `probe44.py` run-1 的读数）：
  run-1 测到三件事：
   ① **键基座其实可分**：装置上 `sep_ctx = 0.340`、`sep_key = 0.301`
      ⇒ DIR-33 标定的 `0.9416` 是**标定构造**（未正交化的输入 ⇒ `sep = 1.0000`）的**产物**，
      **不是**装置的值 ⇒ ⇒ **「基座不可分 ⇒ 地址信息进不了梯度」这个前提被否证**。
   ② **损失确实携带地址信息**：初值处 `L(q_rand) − L(q*) = +0.555709`（正确地址明显更优）。
   ③ **记忆不可旁路**：`L(mem) − L(y=0) = +0.340245`。
  ⇒ ⇒ 既然「信息在」而「方向不在」，**稀释必然发生在梯度的组装过程里**。

★ 本探针要定位**稀释发生在哪里**（**逐时刻**分解）：
  装置的关键结构事实（逐字读 `dir42.trace42`）：
    · 层-2 的读上下文 `c = hp_prev.detach()` ⇒ **层-1 → 层-2 的反馈路径已被 stop-gradient 切断**；
    · 损失只加 `t ∈ [1,15]` 的项 ⇒ **`q(0)` 不进损失**；
    · ⇒ **`∂L/∂q(t)` 只走「检索」这一条路** ⇒ **`g_task` 可**逐时刻**精确分解**：
        `g_task = Σ_{t=1..15} g_t`，`g_t := (∂L/∂q(t) ⊙ (1−q(t)²)) ⊗ c(t)`
      同理 `g_addr = (1/15) Σ_{t=1..15} ᾡ_t`，`ᾡ_t := (∂A/∂q(t) ⊙ (1−q(t)²)) ⊗ c(t)`。
  ★ 分解结果给出三种**互斥**读法（下一张卡据此选**唯一**干预）：
    (A) **符号冲突**：逐时刻 `cos(g_t, ᾡ_t)` 里**正负混杂** ⇒ ⇒ 逐时刻求和**相消**
        ⇒ 干预方向 = **消掉冲突时刻**（例如把损失窗口挪到记忆成形之后）。
    (B) **逐时刻都对、只是量级小** ⇒ ⇒ 稀释来自**目标几何**（对齐目标要求 16 维精确对齐，
        而损失只关心 `row(W)` 内的 2 维方向）⇒ 干预方向 = **改寻址目标的定义/查询坐标对齐**。
    (C) **逐时刻几乎无关**（`|cos(g_t, ᾡ_t)| ≈ 0`）⇒ ⇒ 损失在每个时刻都**不**偏好寻址
        ⇒ 干预方向 = **读侧结构**（`\mathcal O_l` 的内部结构 / `\mathcal Q_l` 的坐标对齐）。
  ★ 另报**相消比** `‖Σg_t‖ / Σ‖g_t‖`（=1 ⇒ 无相消；≪1 ⇒ 相消主导）。

═══ 纪律 ═══
  · 探针**只定尺、不定论**；读数**不得**进任何判据（纪律 106）。
  · 种子段 `23200+`，与 run-1（`23100+`）、DIR-43 正式轮（`22500+`）**均不相交**。
  · 跑出的行**必须落盘**（纪律 108）⇒ `dir44/out/DIR44_probe_b.json`。
"""

import json
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
N_PROBE = 24
PROBE_OFF = 23200
PROBE_SEEDS = [SEED0 + PROBE_OFF + i for i in range(N_PROBE)]
TRAIN_WIN = (1, 8)
N_REP = 16
STEPS, LR = 250, 0.03


def decompose(sd, phi):
    r"""逐时刻分解：`g_task = Σ g_t`、`g_addr = (1/15) Σ ᾡ_t`。"""
    # ★ `train_reader42` 返回的是 **detach 过的**副本 ⇒ 必须重建带图的叶子
    live = {k: (v.detach().clone().requires_grad_(True) if torch.is_tensor(v) else v)
            for k, v in phi.items()}
    P0 = T37._mk(sd, None)
    P = T37._mk(sd, live, train=True)
    L, _, ex = PR.rollout44(P, N_REP, sd, phi, win=TRAIN_WIN)
    qg = ex["qg"]
    c = PR.contexts(P0)
    qstar = T43.qstar_of(P0)

    # ── 逐时刻 `∂L/∂q(t)`
    dLdq = []
    for t in range(len(qg)):
        gq = torch.autograd.grad(L, qg[t], retain_graph=True, allow_unused=True)[0]
        dLdq.append(None if gq is None else gq.detach().clone())

    # ── `g_t`（**只在损失窗口内**：窗口外的 `q(t)` 对 `L` 无梯度 ⇒ `dLdq[t]` 为 None）
    tl = list(range(TRAIN_WIN[0], TRAIN_WIN[1] + 1))
    gt = []
    for t in tl:
        if dLdq[t] is None:
            continue
        q = qg[t].detach()
        fac = (1.0 - q ** 2)
        gt.append(torch.outer(c[t], dLdq[t] * fac))

    # ── 寻址方向的逐时刻贡献；**两套窗口**（原版 `t∈[1,15]` vs 与任务对齐的 `t∈[1,8]`）
    def read_dirs(tlist):
        out = []
        for t in tlist:
            q = qg[t].detach()
            fac = (1.0 - q ** 2)
            cs = float(torch.nn.functional.cosine_similarity(q, qstar[t], dim=0))
            dA_dq = (qstar[t] - cs * q) / float(qstar[t].norm())
            out.append(torch.outer(c[t], dA_dq * fac) / len(tlist))
        return out

    all_t = list(range(1, R.T))                    # DIR-43 原版：`A = mean_{t=1..T-1}`
    gt_read = read_dirs(all_t)
    gt_read_win = read_dirs(tl)                    # ★ 与任务梯度**窗口对齐**的版本
    G = torch.stack(gt).sum(0)
    A = torch.stack(gt_read).sum(0)
    A_win = torch.stack(gt_read_win).sum(0)
    # ── 自检：`g_task`（autograd）== `Σ g_t` ⇒ 分解必须**精确**
    L2, _, _ = PR.rollout44(T37._mk(sd, live, train=True), N_REP, sd, live, win=TRAIN_WIN)
    L2.backward()
    g_task = live["Wq1"].grad.detach().clone()
    live["Wq1"].grad = None
    decomp_err = float((G - g_task).abs().max())

    # 逐时刻余弦：**同窗口**配对（`g_t` 只在损失窗口内，`ᾡ_t` 取同窗口版本）
    gt_win = read_dirs(tl)
    cos_per_t = [PR.cos2(gt[i], gt_win[i]) for i in range(len(gt))]
    signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in cos_per_t]
    norms = [float(x.norm()) for x in gt]
    cancel = float(G.norm()) / max(sum(norms), 1e-300)
    return {
        "seed": sd,
        "rho_total": PR.cos2(-g_task, A),             # ★ DIR-43 原版（`A` 窗口 = t∈[1,15]）
        "rho_win_aligned": PR.cos2(-g_task, A_win),   # ★ 与任务梯度**窗口对齐**的 `A`（t∈[1,8]）
        "cos_per_t": cos_per_t,
        "signs": signs,
        "n_pos": sum(1 for s in signs if s > 0),
        "n_neg": sum(1 for s in signs if s < 0),
        "n_zero": sum(1 for s in signs if s == 0),
        "mean_abs_cos_per_t": statistics.fmean(abs(x) for x in cos_per_t),
        "cancel_ratio": cancel,
        "norm_gt_sum": sum(norms),
        "norm_G": float(G.norm()),
        "decomp_max_err": decomp_err,                 # ★ 必须 ≈ 0
        "norm_g_task": float(g_task.norm()),
    }


def main():
    t0 = time.time()
    res = {"meta": {"n_probe": N_PROBE, "probe_off": PROBE_OFF,
                    "note": "探针只定尺、不定论；不进任何判据（纪律 106/108）"}}
    rows = [decompose(sd, T42.reader_init(sd)) for sd in PROBE_SEEDS]
    res["decomp_init"] = rows
    errs = [r["decomp_max_err"] for r in rows]
    res["A44b-1_decomposition_exact"] = {"ok": max(errs) < 1e-12, "max_err": max(errs)}
    assert max(errs) < 1e-12, "★ A44b-1 FAIL：逐时刻分解与 autograd 不一致 ⇒ 分解无效"

    # ── 训练后（`learned` 臂，250 步）再做一次分解 ⇒ 对齐 `ρ(STEPS)`
    rows_t = []
    for sd in PROBE_SEEDS[:12]:
        phi, _ = T42.train_reader42("learned", sd, steps=STEPS, train_win=TRAIN_WIN, lr=LR)
        rows_t.append(decompose(sd, phi))
    res["decomp_trained"] = rows_t

    res["meta"]["wall_s"] = round(time.time() - t0, 1)
    outp = os.path.join(OUT, "DIR44_probe_b.json")
    pathlib.Path(outp).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("WROTE", outp)

    def summarize(name, rs):
        print(f"\n== {name} ==")
        print(f"  ρ(原版 A 窗口 t∈[1,15]) 均值 = {statistics.fmean([r['rho_total'] for r in rs]):+.4f}"
              f"  sd = {statistics.stdev([r['rho_total'] for r in rs]):.4f}")
        print(f"  ρ(窗口对齐 A t∈[1,8])   均值 = {statistics.fmean([r['rho_win_aligned'] for r in rs]):+.4f}"
              f"  sd = {statistics.stdev([r['rho_win_aligned'] for r in rs]):.4f}")
        print(f"  逐时刻 cos(g_t, ᾡ_t): 均值 {statistics.fmean([r['mean_abs_cos_per_t'] for r in rs]):.4f}"
              f" (绝对值均值)")
        print(f"  符号：正 {statistics.fmean([r['n_pos'] for r in rs]):.1f} / "
              f"负 {statistics.fmean([r['n_neg'] for r in rs]):.1f} / "
              f"零 {statistics.fmean([r['n_zero'] for r in rs]):.1f} （共 15 个时刻）")
        print(f"  相消比 ‖Σg_t‖/Σ‖g_t‖ = {statistics.fmean([r['cancel_ratio'] for r in rs]):.4f}")
        print(f"  分解自检 max_err = {max(r['decomp_max_err'] for r in rs):.3e}")

    summarize("初值（ρ(0) 对应）", rows)
    summarize("训练后（ρ(STEPS) 对应）", rows_t)


if __name__ == "__main__":
    main()
