#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""probe46a.py — DIR-46 的**开卡前诊断**（**只定尺、不定论**；**不进任何判据**）。

★ 框架出处：§7.2「**记忆语义**」行 ——「**信息寿命分解、长期干扰、replay 作用、模态持续度**」。
★ 本探针**不判**容量是否有限（那是 DIR-46 卡的主判据）；它只回答卡上**必须先算**的四条：
  ① **可测性**（`C(N_w)` 的 sd ⇒ 定 `n`）；② **判据非空**（`C(1) ≥ 0.9`？）；
  ③ **功效**（由 sd 反推使 `3se ≤ 0.05` 所需的 `n`）；④ **稳定性前置**（`N_div` = 首个 `W` 非有限的 `N_w`）。
  ⇒ ⇒ 若 ② 不成立或 ④ 在 `N_w=1` 就命中，DIR-46 的卡**按其触发条款**会被判 `EMPTY-CRITERION`/`DIVERGED-EARLY`
  ⇒ 那种情况下**开卡前就应改装置 revision**，而不是开一张注定被遮蔽的卡。

═══ 测量对象（**逐字沿用装置**）═══
  · 写入：`W ← W − ETA·(G Vᵀ)`、`G = ALPHA·(W k − v) kᵀ + LAM·W`（`M._W`/`M.ALPHA`/`M.LAM`/`M.ETA` 同 `trace42`），
    每对 `(k,v)` 重复 `n_rep=16` 次（装置协议）。
  · `k_s = tanh(h(s-1)·Pk1)`（`s=0` 为零向量）、`v_s = tanh(h(s)·Pv1)`；`h(s) = tanh(a[s]·A1 + b1)`。
  · 内容：`period2`（装置原样，交替）与 `perstep`（逐步独立，同尺度 `√D_H`、同生成器流 `seed+7777`），长度 = `N_w`。
  · **取回**：`C(N_w) := mean_{s < N_w} cos(W^{(N_w)}·k_s, v_s)`（`s=0` 的 `k_0=0` ⇒ 不可读 ⇒ **跳过并计数**）。
  · **发散**：`W` 出现非有限 ⇒ 该实例记 `diverged=True`（不计入 `C` 的均值，但**必须报占比**）。

═══ 纪律 ═══
  · 探针只定尺、不定论（纪律 106/108）；行**全部落盘** `dir46/out/DIR46_probe_a.json`。
  · 种子段 `23800+`（n=12），与 `22500/23100…23700` **均不相交**；正式轮将换**新种子段**重测同曲线。
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
for _d in ("dir31", "dir33", "dir37", "dir40", "dir42", "dir43", "dir44", "dir45"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M            # noqa: E402
import R1_task33 as R          # noqa: E402
import probe44e as PE          # noqa: E402  ★ 复用其 `mk_e`（`A_MODE` 开关）以保装置逐字一致

OUT = os.path.join(ROOT, "dir46", "out")
os.makedirs(OUT, exist_ok=True)

SEED0 = M.SEED0
N_PROBE = 12
PROBE_OFF = 23800
PROBE_SEEDS = [SEED0 + PROBE_OFF + i for i in range(N_PROBE)]
N_REP = 16
NW_GRID = (1, 2, 4, 8, 16, 32, 64, 128)


def content(sd, n_w, mode):
    r"""长度 `n_w` 的内容序列：`period2`（交替，装置原样）或 `perstep`（逐步独立，同尺度/同流）。"""
    g = torch.Generator().manual_seed(int(sd) + 7777)
    x = torch.randn(n_w, R.D_H, generator=g)
    x = x / x.norm(dim=1, keepdim=True) * (R.D_H ** 0.5)
    if mode == "period2":
        x = torch.stack([x[t % 2] for t in range(n_w)])
    return x


def curve_one(sd, mode, n_w):
    r"""单个 `(seed, mode, N_w)` 单元：写入 `n_w` 对后，测全部槽位的取回相似度与 `‖W‖`。"""
    P = R.build(sd, "V0")
    P["a"] = content(sd, n_w, mode)
    H = [torch.tanh(P["a"][t] @ P["A1"] + P["b1"]) for t in range(n_w)]
    ks = [torch.tanh((torch.zeros(R.D_H) if s == 0 else H[s - 1]) @ P["Pk1"]) for s in range(n_w)]
    vs = [torch.tanh(H[s] @ P["Pv1"]) for s in range(n_w)]
    U, V = P["U1"].clone(), P["V1"].clone()
    for s in range(n_w):
        k, v = ks[s], vs[s]
        for _ in range(N_REP):
            W = M._W(U, V)
            G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
            U = U - M.ETA * (G @ V)
            V = V - M.ETA * (G.t() @ U)
    W = M._W(U, V)
    finite = bool(torch.isfinite(W).all())
    cs, skipped = [], 0
    if finite:
        for s in range(1, n_w):                       # `k_0 = 0` ⇒ 不可读 ⇒ 跳过（计数）
            c = float(torch.nn.functional.cosine_similarity(W @ ks[s], vs[s], dim=0))
            if math.isfinite(c):
                cs.append(c)
            else:
                skipped += 1
    else:
        skipped = n_w - 1
    return {"seed": sd, "mode": mode, "n_w": n_w, "finite": finite,
            "wnorm": (float(W.norm()) if finite else None),
            "C": (statistics.fmean(cs) if cs else None),
            "n_read": len(cs), "n_skip": skipped}


def _sum(xs, label, indent="    "):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        print(f"{indent}{label:26s} = (无有效值)"); return None, None
    mu = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    print(f"{indent}{label:26s} = {mu:+.5f}  (sd {sd:.4f}, se {se:.4f}, n={n})  "
          f"{'★' if abs(mu) >= 3*se else ' '}3se={3*se:.4f}")
    return mu, sd


def main():
    t0 = time.time()
    res = {"meta": {"n_probe": N_PROBE, "probe_off": PROBE_OFF, "n_rep": N_REP,
                    "nw_grid": list(NW_GRID),
                    "note": "开卡前诊断（只定尺）；不进任何判据（纪律 106/108）"}, "cells": []}
    for mode in ("period2", "perstep"):
        print(f"\n########## mode = {mode} ##########")
        rows = []
        for n_w in NW_GRID:
            cell = [curve_one(sd, mode, n_w) for sd in PROBE_SEEDS]
            rows.extend(cell)
            fin = [c for c in cell if c["finite"]]
            div = len(cell) - len(fin)
            mu, sd = _sum([c["C"] for c in fin], f"C({n_w})")
            _sum([c["wnorm"] for c in fin], f"  ‖W‖({n_w})")
            nr = statistics.fmean([c["n_read"] for c in fin]) if fin else float("nan")
            print(f"      发散 {div}/{len(cell)}   有效读槽均值 {nr:.2f}/{max(n_w-1,0)}"
                  + ("   ★ 全格发散 ⇒ 该 N_w **无可用数据**（不得记成 0 或删失）" if not fin else ""))
            res["cells"].append({"mode": mode, "n_w": n_w, "C_mean": mu, "C_sd": sd,
                                 "n_finite": len(fin), "n_div": div,
                                 "rows": cell})
        # ① 判据非空：★ `N_w=1` **不可读**（`k_0 = 0` ⇒ `W·0 = 0`）⇒ 锚点必须取 `N_w=2`
        print("  ① 判据非空：`C(1)` **不存在**（`k_0 = 0` ⇒ 首次写入的键是零向量 ⇒ `W·k_0 = 0` ⇒ 该槽**永不可读**；"
              "且 `G = ALPHA·(W·0 − v)·0ᵀ + LAM·W = LAM·W` ⇒ 第 0 步只**衰减**、不写值）⇒ 锚点改用 `C(2)`：")
        c2 = [c["C_mean"] for c in res["cells"] if c["mode"] == mode and c["n_w"] == 2]
        print("      C(2) =", c2)
        # ② 可测性 / ③ 功效
        for n_w in NW_GRID:
            sds = [c["C_sd"] for c in res["cells"] if c["mode"] == mode and c["n_w"] == n_w]
            s = sds[0]
            if s and math.isfinite(s):
                n_need = math.ceil((3 * s / 0.05) ** 2)
                print(f"  ③ N_w={n_w:3d}: sd={s:.4f} ⇒ 使 3se≤0.05 需 n≈{n_need}")
        # ④ 稳定性前置
        nd = [c["n_w"] for c in res["cells"] if c["mode"] == mode and c["n_div"] > 0]
        print("  ④ `N_div`（首个出现发散的 N_w）=", min(nd) if nd else None,
              "  发散占比：", {c["n_w"]: f"{c['n_div']}/{N_PROBE}" for c in res["cells"]
                              if c["mode"] == mode and c["n_div"] > 0})
        print("  ⑤ `C(N_w)` 曲线（均值±sd，n=有限实例数）：")
        print("      ", "  ".join(f"{c['n_w']}:{('-', '-') if c['C_mean'] is None else (round(c['C_mean'],4), round(c['C_sd'],4))}"
                                 f"({c['n_finite']})" for c in res["cells"] if c["mode"] == mode))
    res["meta"]["wall_s"] = round(time.time() - t0, 1)
    outp = os.path.join(OUT, "DIR46_probe_a.json")
    pathlib.Path(outp).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nWROTE", outp, f"\n墙钟 {res['meta']['wall_s']} s")


if __name__ == "__main__":
    main()
