#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""probe44.py — DIR-43 收束后的**第一探针**（**只定尺、不定论**；**不进任何判据**）。

★ 动机（DIR-43 `VERDICT.md` §4 的授权候选「**可分性更强的键基座**」）：
  DIR-43 判 `ρ ≈ 0`（任务梯度的下降方向**不**指向寻址）。
  DIR-43 `VERDICT.md` §2.3 与 §6.7 把「**基座不可分 ⇒ 地址信息无法进入梯度**」列为**相容解释**。
  ★★ 本探针**先检验这个前提是否真的成立**，再决定下一张卡改什么 ⇒
  —— **前提若被否证，就必须换方向，而不是硬开一张卡**。

═══ 测量（全部**只读**既有装置，不改任何冻结文件）═══

  **P1 前提检验**：装置上**真实**的
      · `sep_ctx  = |cos(c_0, c_1)|`（两个交替上下文的夹角）
      · `sep_key  = |cos(k_0, k_1)|`（两把**已写入**的键的夹角）
    ★★ **并排**给 DIR-33 标定构造的值（**同样**的度量、**不做**输入正交化）
    ⇒ 直接回答「DIR-33 的 `0.9416` 是**装置的值**还是**标定构造的值**」。

  **P2 损失是否携带地址信息**（**代换检验**，**零训练**）：
      同一装置、同一（随机）头，只把**读查询**换成
      `q*(t) = tanh(c(t)·Pk1)`（正确地址）  vs  `q_rand(t) = tanh(c(t)·Pk_rand)`（错误地址）
      ⇒ `L(q*) − L(q_rand)` ⇒ **损失本身**是否偏好正确地址。

  **P3 任务是否可被旁路**（**零训练**）：
      把层-1 的 `y` 置 0（头**只**能看到 `c`）⇒ `L(y=0) − L(mem)`
      ⇒ 若≈0 ⇒ 头**根本不需要**寻址 ⇒ 任务被**旁路**。

  **P4 重叠分解**（**零训练**）：`∂L/∂q(t)` 是否被限制在 `row(W)` 内；
      `q*(t)`、`q(t)` 在 `span{keys}` 内的份额。

  **P5 候选干预的 `ρ(0)` 筛**（**零训练**，`n` 小 ⇒ 只定尺）：
      逐个**单变量**干预下，测量 `ρ(0)`、`sep_key`、`L` 的地址落差 ⇒
      ⇒ 挑出**唯一**值得开卡的干预；**若全为 0 ⇒ 前提被否证 ⇒ 换方向**。

═══ 纪律 ═══
  · 探针**只定尺、不定论**；读数**不得**进任何判据（纪律 106）。
  · 探针段与**正式轮段不相交**（本探针用 `23100+`，DIR-43 正式轮用 `22500+`、探针用 `22000+`）。
  · 本探针跑的真实单元**必须落盘**（纪律 108）⇒ 输出 `dir44/out/DIR44_probe.json`。
  · 断言必须**真的读被断言的对象**（纪律 97/108）⇒ `A44-1` 断言本地的 `rollout44`
    在默认模式下与 `dir42.trace42` **逐位相等**。
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
for _d in ("dir31", "dir33", "dir37", "dir40", "dir42", "dir43"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M            # noqa: E402  装置（逐字零改）
import R1_task33 as R          # noqa: E402  任务层（逐字零改）
import R1_task37 as T37        # noqa: E402
import R1_task42 as T42        # noqa: E402  测量层/训练层（逐字零改）
import R1_task43 as T43        # noqa: E402  DIR-43 的测量层（逐字零改）

OUT = os.path.join(ROOT, "dir44", "out")
os.makedirs(OUT, exist_ok=True)

SEED0 = M.SEED0
N_PROBE = 24
PROBE_OFF = 23100                      # ★ 与所有既有段不相交
PROBE_SEEDS = [SEED0 + PROBE_OFF + i for i in range(N_PROBE)]
TRAIN_WIN = (1, 8)                    # 同 DIR-42/43
N_REP = 16                             # 同 DIR-42/43
STEPS, LR = 250, 0.03

# 键映射的随机正交对照（同 DIR-43 的口径：种子 = 实例种子 + offset）
RAND_KEY_SEED_OFFSET = 7_777_777


# ════════════════════════════════════════════════════════════════════
#  本地 rollout：**trace42 的逐字派生**，只加两个**只读**开关
#    q_mode ∈ {"reader", "oracle", "randkey"}   ← 强制读查询（**不**动写键）
#    y_mode ∈ {"mem", "zero"}                    ← 层-1 的 `y` 置 0（旁路检验）
#  ★ 默认模式必须与 `trace42` **逐位相等**（`A44-1`）。
# ════════════════════════════════════════════════════════════════════
def rollout44(P, n_rep, sd, phi, win=None, q_mode="reader", q_key=None):
    r"""`dir42.trace42` 的**逐字派生** + 两个开关。返回 `(L, diag, extras)`。"""
    L = P["a"].new_zeros(())
    gs = R.targets(P)
    h_pre_prev = [None, None]
    q1s, k1s = [], []
    qg = []                                  # 带图（供 `∂L/∂q` 用）
    acc = {"c_eq_h": 0.0}
    q_fp = 0.0
    for t in range(T := R.T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if li == 0:
                c = h
                acc["c_eq_h"] = max(acc["c_eq_h"], float((c - h).abs().max()))
                if q_mode == "oracle":
                    q = torch.tanh(h @ P["Pk1"])
                elif q_mode == "randkey":
                    q = torch.tanh(h @ q_key)
                else:
                    q = torch.tanh(c @ P["Wq1"] + P["bq1"])
                q1s.append(q.detach().clone())
                qg.append(q)
                q_fp = q_fp + float(q.sum())
            else:
                c = hp_prev.detach()
                q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            Wd = W.detach()
            y = Wd @ q
            if li == 0 and y_mode_zero:
                y = torch.zeros_like(y)
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            gam = P[f"g{l}"]
            hp[li] = h + gam * m
            hp_prev = hp[li]
        if t >= 1 and (win is None or (win[0] <= t <= win[1])):
            L = L + ((R.pout() @ hp[0] - gs[t - 1]) ** 2).sum()
        upd = {}
        for li, l in enumerate((1, 2)):
            hprev = h_pre_prev[li]
            if hprev is None:
                hprev = torch.zeros(R.D_H)
            k = torch.tanh(hprev.detach() @ P[f"Pk{l}"])
            if V_MODE == "codebook":
                # ★ 值来自**离线码本**（按**奇偶**索引 ⇒ 记忆收敛；且**不可由 `c` 解码** ⇒ 堵死捷径）
                v = torch.tanh(V_CODEBOOK[l][t % 2] @ P[f"Pv{l}"])
            else:
                v = torch.tanh(h_pre[li].detach() @ P[f"Pv{l}"])
            if li == 0:
                k1s.append(k.detach().clone())
            Uo, Vo = P[f"U{l}"], P[f"V{l}"]
            for _ in range(n_rep):
                W = M._W(Uo, Vo)
                G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
                upd[f"U{l}"] = Uo - M.ETA * (G @ Vo)
                upd[f"V{l}"] = Vo - M.ETA * (G.t() @ Uo)
                Uo, Vo = upd[f"U{l}"], upd[f"V{l}"]
        upd = {kk: vv.detach() for kk, vv in upd.items()}
        P.update(upd)
        h_pre_prev = h_pre
    diag = {"c_eq_h": acc["c_eq_h"], "q_fp": q_fp}
    return L, diag, {"q1s": q1s, "k1s": k1s, "qg": qg, "W1": M._W(P["U1"], P["V1"]).detach().clone()}


y_mode_zero = False          # ★ 模块级开关（`rollout44` 内部读取）
V_MODE = "device"            # ★ 值来源开关："device"（默认：`h_pre`）| "codebook"（离线码本）
V_CODEBOOK = None            # ★ {layer: (u_parity0, u_parity1)}；仅 `V_MODE="codebook"` 时用


def rollout44_y0(P, n_rep, sd, phi, win=None):
    r"""`rollout44` 的 `y_mode="zero"` 版本（层-1 的 `y` 置 0 ⇒ 头只看 `c`）。"""
    global y_mode_zero
    old = y_mode_zero
    y_mode_zero = True
    try:
        return rollout44(P, n_rep, sd, phi, win=win, q_mode="reader")
    finally:
        y_mode_zero = old


# ════════════════════════════════════════════════════════════════════
#  工具
# ════════════════════════════════════════════════════════════════════
def cos2(a, b):
    na, nb = float(a.norm()), float(b.norm())
    if na <= 1e-300 or nb <= 1e-300:
        return float("nan")
    return float((a * b).sum()) / (na * nb)


def proj_frac(v, basis):
    r"""`v` 在 `basis`（行向量组构成的子空间）内的能量份额。"""
    B = torch.stack([b for b in basis])              # (m, d)
    # 正交化（Gram-Schmidt）
    ortho = []
    for row in B:
        u = row.clone()
        for o in ortho:
            u = u - (u @ o) / (o @ o) * o
        if float(u @ u) > 1e-300:
            ortho.append(u)
    if not ortho:
        return 0.0
    O = torch.stack(ortho)
    coef = O @ v
    return float((coef @ coef) / (v @ v))


def rand_key(P, sd):
    g = torch.Generator().manual_seed(int(sd) + RAND_KEY_SEED_OFFSET)
    A = torch.randn(R.D_K, R.D_K, generator=g, dtype=torch.float64)
    Q, Rr = torch.linalg.qr(A)
    return P["Pk1"] @ (Q * torch.sign(torch.diagonal(Rr)).unsqueeze(0))


def contexts(P):
    r"""层-1 读上下文 `c(t) = tanh(a[t]·A1 + b1)`。"""
    return [torch.tanh(P["a"][t] @ P["A1"] + P["b1"]) for t in range(R.T)]


def keys_of(P):
    r"""**已写入**的键（第 1 层）：`k_s = tanh(c(s-1)·Pk1)`，`s = 1..T-1`（与 `trace42` 同口径）。"""
    c = contexts(P)
    return [torch.tanh(c[t - 1] @ P["Pk1"]) for t in range(1, R.T)]


def calib_construction(sd, P):
    r"""DIR-33 `calib33.key_sep` 的**标定构造**（**不做**输入正交化）⇒ 并排对照。"""
    g = torch.Generator().manual_seed(int(sd) + 7777)
    x = torch.randn(2, M.D_H, generator=g, dtype=torch.float64)
    a = torch.stack([x[t % 2] for t in range(M.T_STEPS)])
    h_prev = a[0]
    ks = []
    for t in range(M.T_STEPS):
        h = torch.tanh(h_prev @ P["A1"] + P["b1"])
        ks.append(torch.tanh(h @ P["Pk1"]))
        h_prev = a[t]
    return ks


# ════════════════════════════════════════════════════════════════════
#  P1：前提检验（**零训练**）
# ════════════════════════════════════════════════════════════════════
def p1_premise(seeds):
    r"""装置上真实的 `sep_ctx` / `sep_key`；并排 DIR-33 的标定构造。"""
    rows = []
    for sd in seeds:
        P = T37._mk(sd, None)
        c = contexts(P)
        ks = keys_of(P)
        # 装置：只有两个**不同**的上下文（周期 2）⇒ 取 c[0], c[1] 与 k[0], k[1]
        dev_ctx = abs(cos2(c[0], c[1]))
        dev_key = abs(cos2(ks[0], ks[1]))
        # 相邻键（写入序列里相邻的两把）
        adj = [abs(cos2(ks[i], ks[i - 1])) for i in range(1, len(ks))]
        # 标定构造（未正交化）
        cks = calib_construction(sd, P)
        cal_adj = [abs(cos2(cks[i], cks[i - 1])) for i in range(1, len(cks))]
        rows.append({
            "seed": sd,
            "dev_sep_ctx": dev_ctx, "dev_sep_key": dev_key,
            "dev_sep_key_adj_max": max(adj), "dev_sep_key_adj_mean": statistics.fmean(adj),
            "calib_sep_key_adj_max": max(cal_adj), "calib_sep_key_adj_mean": statistics.fmean(cal_adj),
            "n_distinct_ctx": len({tuple(round(v, 12) for v in c[t].detach().tolist()) for t in range(R.T)}),
        })
    return rows


# ════════════════════════════════════════════════════════════════════
#  P2 / P3 / P4：零训练的诊断（**同一**装置、**同一**头 ⇒ 配对）
# ════════════════════════════════════════════════════════════════════
def p2_p3_p4(seeds, phi_of):
    r"""地址落差 / 旁路检验 / 重叠分解。`phi_of(sd)` 给读器参数（初值或训练后）。"""
    rows = []
    for sd in seeds:
        phi = phi_of(sd)
        P0 = T37._mk(sd, None)
        krand = rand_key(P0, sd)
        ks = keys_of(P0)

        def run(q_mode, y_zero=False):
            P = T37._mk(sd, phi, train=True)
            L, diag, ex = (rollout44_y0(P, N_REP, sd, phi, win=TRAIN_WIN) if y_zero
                           else rollout44(P, N_REP, sd, phi, win=TRAIN_WIN,
                                          q_mode=q_mode, q_key=krand))
            return float(L), diag, ex

        L_reader, _, ex_reader = run("reader")
        L_oracle, _, _ = run("oracle")
        L_rand, _, _ = run("randkey")
        L_y0, _, _ = run("reader", y_zero=True)

        # ── 重叠分解（用**读者的** q 与训练窗口的 L）
        P = T37._mk(sd, phi, train=True)
        Lg, _, ex = rollout44(P, N_REP, sd, phi, win=TRAIN_WIN)
        qg = ex["qg"]
        dLdq = []
        for t in range(len(qg)):
            gq = torch.autograd.grad(Lg, qg[t], retain_graph=True, allow_unused=True)[0]
            # ★ t=0 的 q **不进**损失（损失从 t=1 起）⇒ 梯度为 None ⇒ 跳过（不许假装是 0）
            if gq is None:
                continue
            dLdq.append(gq.detach().clone())
        W1 = ex["W1"]
        # `row(W1)` 的正交基（W1 是 D_V×D_K ⇒ 行空间在 R^D_K）
        U_, S_, Vt_ = torch.linalg.svd(W1)
        rowbasis = [Vt_[i] for i in range(int((S_ > 1e-10 * float(S_[0])).sum()))]
        # 已写入键的 span
        keybasis = [ks[0], ks[1]]
        qstar = T43.qstar_of(P0)
        frac_rowW = [proj_frac(dLdq[t], rowbasis) for t in range(len(dLdq))]
        frac_qstar_rowW = [proj_frac(qstar[t], rowbasis) for t in range(R.T)]
        frac_qstar_keyspan = [proj_frac(qstar[t], keybasis) for t in range(R.T)]
        frac_q_keyspan = [proj_frac(ex["q1s"][t].clone(), keybasis) for t in range(len(ex["q1s"]))]

        rows.append({
            "seed": sd,
            "L_reader": L_reader, "L_oracle": L_oracle, "L_randkey": L_rand, "L_y0": L_y0,
            "addr_gap": L_rand - L_oracle,          # 正 = 正确地址更优
            "bypass_gap": L_reader - L_y0,          # 正 = 记忆有用（不可旁路）
            "rank_W1": len(rowbasis),
            "frac_dLdq_in_rowW_mean": statistics.fmean(frac_rowW),
            "frac_qstar_in_rowW_mean": statistics.fmean(frac_qstar_rowW),
            "frac_qstar_in_keyspan_mean": statistics.fmean(frac_qstar_keyspan),
            "frac_q_in_keyspan_mean": statistics.fmean(frac_q_keyspan),
        })
    return rows


# ════════════════════════════════════════════════════════════════════
#  P5：候选**单变量**干预的 `ρ(0)` 筛（**零训练**）
# ════════════════════════════════════════════════════════════════════
def build_variant(sd, cfg):
    r"""按 `cfg` 构造装置（**只**改一个变量）。`cfg` 键：
      `key_scale`（Pk 缩放）· `d_k`（键维）· `rank`（记忆秩）· `ortho_ctx`（基座输出正交化）
    """
    P = T37._mk(sd, None)
    if "key_scale" in cfg:
        for l in (1, 2):
            P[f"Pk{l}"] = P[f"Pk{l}"] * (cfg["key_scale"] / R.KEY_SCALE)
    if "d_k" in cfg:
        # 键维变了 ⇒ `Pk` 的形状变了 ⇒ 需要重新抽样（**不是**同一装置了 ⇒ 只作筛）
        raise SystemExit("d_k 变体需另建装置（暂不支持）")
    if "rank" in cfg:
        raise SystemExit("rank 变体需另建装置（暂不支持）")
    if cfg.get("ortho_ctx"):
        # ★ 把正交化从**输入层**移到**基座输出层**：`c(t)` 的两个不同值正交
        c = contexts(P)
        c0, c1 = c[0].clone(), c[1].clone()
        c1 = c1 - (c1 @ c0) / (c0 @ c0) * c0
        a = []
        for t in range(R.T):
            a.append(c0 if t % 2 == 0 else c1)
        # 反解：`c = tanh(a@A1+b1)` ⇒ 直接改 `a` 无法保证 `c` 仍是 `tanh(a@A1+b1)`
        #   ⇒ 改为**在 rollout 内**替换上下文（见 `ctx_override`）
    return P


def rho_at_init(seeds, cfg):
    r"""`ρ(0)`（**零训练**）：与 DIR-43 同口径（`g_desc = −g_task` vs `g_addr`）。"""
    out = []
    for sd in seeds:
        P0 = T37._mk(sd, None)
        phi = T42.reader_init(sd)
        P = T37._mk(sd, phi, train=True)
        L, _, _ = rollout44(P, N_REP, sd, phi, win=TRAIN_WIN)
        L.backward()
        g_task = phi["Wq1"].grad.detach().clone()
        _, g_addr = T43.align_grad(P0, phi, W=phi["Wq1"], b=phi["bq1"])
        out.append({"seed": sd, "rho0": cos2(-g_task, g_addr),
                    "gn_task": float(g_task.norm())})
    return out


# ════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    res = {"meta": {"n_probe": N_PROBE, "probe_off": PROBE_OFF,
                    "seeds": [PROBE_SEEDS[0], PROBE_SEEDS[-1]],
                    "note": "探针只定尺、不定论；读数不进任何判据（纪律 106/108）"}}

    # ── A44-1：本地 rollout 与 `trace42` **逐位相等**（断言必须真读被断言对象）
    sd = PROBE_SEEDS[0]
    phi = T42.reader_init(sd)
    P1_ = T37._mk(sd, phi, train=True)
    L_local, _, _ = rollout44(P1_, N_REP, sd, phi, win=TRAIN_WIN)
    P2_ = T37._mk(sd, phi, train=True)
    L_ref, _ = T42.trace42(P2_, N_REP, "learned", sd, phi, win=TRAIN_WIN)
    bit_eq = (float(L_local) == float(L_ref))
    res["A44-1_rollout44_bit_equal_trace42"] = {"ok": bool(bit_eq),
                                                "local": float(L_local), "ref": float(L_ref)}
    assert bit_eq, "★ A44-1 FAIL：本地 rollout44 与 trace42 不逐位相等 ⇒ 探针无效"

    # ── P1
    res["P1_premise"] = p1_premise(PROBE_SEEDS)
    # ── P2/P3/P4（初值读器 ⇒ 零训练）
    res["P2P3P4_init"] = p2_p3_p4(PROBE_SEEDS, lambda sd: T42.reader_init(sd))
    # ── P5：`ρ(0)` 筛（基线）
    res["P5_rho0_baseline"] = rho_at_init(PROBE_SEEDS, {})

    # ── P5b：单变量干预 —— `KEY_SCALE`（键尺度 ⇒ tanh 工作点 ⇒ 键的线性度）
    for ks in (0.25, 1.0):
        rows = []
        for sd in PROBE_SEEDS:
            P0 = T37._mk(sd, None)
            for l in (1, 2):
                P0[f"Pk{l}"] = P0[f"Pk{l}"] * (ks / R.KEY_SCALE)
            phi = T42.reader_init(sd)
            P = T37._mk(sd, phi, train=True)
            # ★ 覆写 `Pk` ⇒ 写入与 `q*` 用同一张（缩放过的）键图 ⇒ 口径一致
            P["Pk1"] = P0["Pk1"].clone()
            P["Pk2"] = P0["Pk2"].clone()
            L, _, _ = rollout44(P, N_REP, sd, phi, win=TRAIN_WIN)
            L.backward()
            g_task = phi["Wq1"].grad.detach().clone()
            _, g_addr = T43.align_grad(P0, phi, W=phi["Wq1"], b=phi["bq1"])
            ksx = keys_of(P0)
            rows.append({"seed": sd, "rho0": cos2(-g_task, g_addr),
                         "sep_key": abs(cos2(ksx[0], ksx[1])),
                         "addr_gap": None})
        res[f"P5b_rho0_keyscale_{ks}"] = rows

    res["meta"]["wall_s"] = round(time.time() - t0, 1)
    outp = os.path.join(OUT, "DIR44_probe.json")
    pathlib.Path(outp).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("WROTE", outp)

    # ── 摘要
    p1 = res["P1_premise"]
    print("\n== P1 前提检验 ==")
    print(f"  装置 sep_ctx  = {statistics.fmean([r['dev_sep_ctx'] for r in p1]):.4f}")
    print(f"  装置 sep_key  = {statistics.fmean([r['dev_sep_key'] for r in p1]):.4f}"
          f"  (adj_max {statistics.fmean([r['dev_sep_key_adj_max'] for r in p1]):.4f})")
    print(f"  标定 sep_key  = {statistics.fmean([r['calib_sep_key_adj_max'] for r in p1]):.4f}"
          f"  (adj_max) ← DIR-33 报的 0.9416 在这里")
    print(f"  不同上下文数  = {sorted({r['n_distinct_ctx'] for r in p1})}")
    p2 = res["P2P3P4_init"]
    print("\n== P2/P3/P4（初值，零训练）==")
    print(f"  地址落差 addr_gap   = {statistics.fmean([r['addr_gap'] for r in p2]):+.6f}"
          f"  (正 = 正确地址更优)")
    print(f"  旁路落差 bypass_gap = {statistics.fmean([r['bypass_gap'] for r in p2]):+.6f}"
          f"  (正 = 记忆不可旁路)")
    print(f"  ∂L/∂q 在 row(W) 内  = {statistics.fmean([r['frac_dLdq_in_rowW_mean'] for r in p2]):.4f}")
    print(f"  q* 在 row(W) 内      = {statistics.fmean([r['frac_qstar_in_rowW_mean'] for r in p2]):.4f}")
    print(f"  q* 在 span{keys} 内  = {statistics.fmean([r['frac_qstar_in_keyspan_mean'] for r in p2]):.4f}")
    print(f"  q  在 span{keys} 内  = {statistics.fmean([r['frac_q_in_keyspan_mean'] for r in p2]):.4f}")
    p5 = res["P5_rho0_baseline"]
    print("\n== P5 ρ(0)（基线，零训练）==")
    print(f"  mean ρ(0) = {statistics.fmean([r['rho0'] for r in p5]):+.4f}"
          f"  sd = {statistics.stdev([r['rho0'] for r in p5]):.4f}")
    for ks in (0.25, 1.0):
        rr = res[f"P5b_rho0_keyscale_{ks}"]
        print(f"  KEY_SCALE={ks}: mean ρ(0) = {statistics.fmean([r['rho0'] for r in rr]):+.4f}"
              f"  sep_key = {statistics.fmean([r['sep_key'] for r in rr]):.4f}")
    print(f"\n墙钟 {res['meta']['wall_s']} s")


if __name__ == "__main__":
    main()
