#!/opt/pytorch/bin/python
r"""DIR-40 · **开卡前前置探针**（★ 不进判据）—— 甲「可表达性」· 乙「真正的 §8」。

卡：**尚未开卡**（本探针是开卡**前**的设计输入；纪律 92：探针读数**不进判据**）。
派生自：`dir39/R1_task39.py`（DIR-39 的 trace/初值/训练协议，**逐字**）·
        装置 `dir31/R1_grad.py`（md5 `8fa40f708600bec7e6c484727d2180c7`，**逐字零改**）·
        任务层 `dir33/R1_task33.py`（md5 `96ef89e87ec61556b52e0e464850087e`，**逐字零改**）。

★ 本文件**唯一**新增的东西：
  ① `trace40` = `trace39` 的**逐字派生**，只加三个读数与两个钩子：
     - 读数 `cos_q_k`（**余弦**：⟨q_t, k_{t−1}⟩/(‖q_t‖‖k_{t−1}‖)）★ DIR-39 的 `align_q_k` 是**投影比**，
       不可当余弦（D-63-CORR）⇒ 本探针**两个都报**，只把余弦当寻址读数；
     - 钩子 `shared_pk1`（**探针乙的装置 revision**：把层 1 的写键映射 `Pk1` 改成**跨实例共享**）；
     - 钩子：控制器初值与 `Pk1` 的**距离**（`‖Wq1 − Pk1_eff‖`）⇒ 读控制器是否学到了**真值键映射**。
  ② 两个探针的训练/评估协议（见下）。

═══════════════════════════════════════════════════════════════════════════════
**探针甲 · 可表达性**（per-instance 读控制器）
───────────────────────────────────────────────────────────────────────────────
- 问：**逐实例**的读控制器 `(Wq1_i, bq1_i)`（★ **随机初值**，★ **禁止**从 `Pk1_i` 初始化）
  在**该实例自己的损失**上训练后，能否达到 `oracle` 读的质量？
- 头：**冻结**（在 `oracle` 臂上训的那一份）⇒ 变量只有「读控制器」。
- 控制器初值 = **所有实例共用同一份**随机初值（⇒ 与「不训」逐位配对）。
- 读数：① `L_oracle_i`（上限）② `L_rand_i`（同初值不训）③ `L_learned_i`（训过）
  ④ `cos_q_k` ⑤ `‖Wq1_i − Pk1_i‖`（**真值映射距离**：读参数化是否学到了 `Pk1_i`）。
- ★ 口径：**只答「可表达性」**（读参数化够不够），**不答**泛化、**不答**多对记忆。

═══════════════════════════════════════════════════════════════════════════════
**探针乙 · 真正的 §8**（`Pk1` 跨实例共享 + 控制器仍共享）
───────────────────────────────────────────────────────────────────────────────
- 装置 revision：层 1 的写键映射 `Pk1` 对**所有实例取同一张矩阵**（分布与装置逐字一致：
  `randn(D_H,D_K)/√D_H × KEY_SCALE`，固定种子）⇒ ⇒ 键**映射**可共享，而值 `Pv1`/`W` 仍逐实例随机。
- 三臂（**分臂联合训练头**：每臂各自把自己的头训出来 ⇒ 失败可归因）：
  ① `oracle`：`q = tanh(h·Pk1_shared)`（描述性上限），头在 `oracle` 臂上训；
  ② `rand`  ：控制器**冻结**（与 ③ **同一初值**），头在**冻结控制器**的读上训（⇒ 强基线）；
  ③ `learned`：控制器**被任务梯度训过**（同初值），头**与之联合训练**。
- 读数：各臂 `L`、`cos_q_k`、`‖Wq1_shared − Pk1_shared‖`（**共享真值映射距离**）、配对功效参考。
- ★ 口径：这是 §8「读取负责学习如何使用关联」在**共享参数**下的**最小**版本。

═══════════════════════════════════════════════════════════════════════════════
**种子段**（全部新段，与全部历史段不相交）
───────────────────────────────────────────────────────────────────────────────
  甲的实例：`17000..17047`（48）· 乙的训练：`17100..17163`（64）· 乙的评估：`17200..17263`（64）
  （历史已用至 `16411`：DIR-38 `12500–14383`、DIR-39 `15000–16411`）

用法：probe
"""
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir37")
import R1_task33 as R           # 任务层（逐字未改；import 时改写装置的有效常数）
import R1_grad as M             # 装置（逐字未改）
import R1_task37 as T37         # DIR-37 的测量层（_mk / 训练协议）

OUT = "/home/ec2-user/sgmn-loop/dir40/out"
D_H, D_V, D_K, T = R.D_H, R.D_V, R.D_K, R.T
SEED0 = M.SEED0

# ══════════ 冻结的协议常数（与 DIR-39 的卡一致；**探针阶段**的量级参数）
N_REP = 16                     # 写侧：DIR-38 的 `n_rep=16`（逐字冻结）
STEPS_JIA, STEPS_YI = 250, 120  # 甲：逐实例 250 步；乙：分臂联合训练 120 步
LR = 0.03                      # 与 DIR-37/39 的训练协议逐字相同（Adam）
N_JIA = 48                     # 甲的实例数（探针量级）
N_YI_TRAIN, N_YI_EVAL = 64, 64  # 乙的训练/评估实例数（探针量级）
SEEDS_JIA = [SEED0 + 17000 + i for i in range(N_JIA)]
SEEDS_YI_TRAIN = [SEED0 + 17100 + i for i in range(N_YI_TRAIN)]
SEEDS_YI_EVAL = [SEED0 + 17200 + i for i in range(N_YI_EVAL)]
SHARED_PK1_SEED = 20261112     # 乙的共享 `Pk1` 的固定种子（写进 JSON，留档）


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def init_params():
    r"""DIR-39 的初值生成器（**逐字**）：`g1,Wo1,bo1` 前三抽、`Wq1,bq1` 后两抽（同一生成器 4242）。

    ⇒ 探针甲的**逐实例**控制器初值 = 这份 `Wq1,bq1`（**随机**，★ 不是 `Pk1`）
    ⇒ 探针乙的三臂初值 = 同一份 ⇒ **逐位配对**。
    """
    gin = torch.Generator().manual_seed(4242)
    return {
        "g1": torch.full((1,), M.GAMMA, requires_grad=True),
        "Wo1": (torch.randn(D_H + D_V, D_H, generator=gin)
                / math.sqrt(D_H + D_V)).requires_grad_(True),
        "bo1": (torch.randn(D_H, generator=gin) * 0.1).requires_grad_(True),
        "Wq1": (torch.randn(D_H, D_K, generator=gin) * 0.354).requires_grad_(True),
        "bq1": torch.zeros(D_K, requires_grad=True),
    }


def shared_pk1():
    r"""探针乙的**装置 revision**：`Pk1` 跨实例共享。

    ★ 分布与装置的逐实例 `Pk1` **逐字一致**：`randn(D_H,D_K)/√D_H`（装置的 `s_h`）再乘
      `KEY_SCALE`（DIR-33 的键尺度 0.5）⇒ 只改「是否共享」，不改分布。
    """
    g = torch.Generator().manual_seed(SHARED_PK1_SEED)
    return (torch.randn(D_H, D_K, generator=g) / math.sqrt(D_H)) * R.KEY_SCALE


def trace40(P, n_rep, qsrc, sd, phi, want_diag=False, shared_pk1=None):
    r"""`T39.trace39` 的**逐字派生**：只加（i）`shared_pk1` 钩子（ii）余弦与真值映射距离读数。

    ★ 层 1：`c = h`（**所有臂逐字相同** ⇒ 只换 `q`）；层 2：`c = hp_prev.detach()`、`q = tanh(c·Wq2+bq2)`。
    ★ 读：`y = W.detach() @ q`（写通道隔离）；写：`n_rep` 次重复的同步因子更新。
    ★ `qsrc`：`"oracle"` ⇒ `tanh(h·Pk1)`；`"rand"`/`"learned"` ⇒ `tanh(h·Wq1+bq1)`
      （两者的**唯一**区别是 `(Wq1,bq1)` 训不训 ⇒ 由调用方保证）。
    """
    if shared_pk1 is not None:
        P["Pk1"] = shared_pk1
    L = P["a"].new_zeros(())
    gs = R.targets(P)
    h_pre_prev = [None, None]
    q1s, k1s = [], []
    acc = {"q_norm": 0.0, "y_norm": 0.0, "m_abs": 0.0, "n": 0}
    wmax = 0.0
    for t in range(T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if li == 0:
                c = h                                   # ★ **各臂逐字相同**（oracle 的 c）
                if qsrc == "oracle":
                    q = torch.tanh(h @ P["Pk1"])
                else:
                    q = torch.tanh(c @ P["Wq1"] + P["bq1"])
                q1s.append(q.detach().clone())
            else:
                c = hp_prev.detach()
                q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            Wd = W.detach()
            y = Wd @ q
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            if want_diag and li == 0:
                acc["q_norm"] += float(q.norm())
                acc["y_norm"] += float(y.norm())
                acc["m_abs"] += float(m.abs().mean())
                acc["n"] += 1
            gam = P[f"g{l}"]
            hp[li] = h + gam * m
            hp_prev = hp[li]
        if t >= 1:
            L = L + ((R.pout() @ hp[0] - gs[t - 1]) ** 2).sum()
        upd = {}
        for li, l in enumerate((1, 2)):
            hprev = h_pre_prev[li]
            if hprev is None:
                hprev = torch.zeros(D_H)
            k = torch.tanh(hprev.detach() @ P[f"Pk{l}"])
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
        if want_diag:
            wmax = max(wmax, float(M._W(P["U1"], P["V1"]).norm()))
    diag = {"Wmax": wmax}
    if want_diag:
        n = max(acc["n"], 1)
        diag["q_norm"] = acc["q_norm"] / n
        diag["y_norm"] = acc["y_norm"] / n
        diag["m_abs"] = acc["m_abs"] / n
        coss, alpr = [], []
        for t in range(1, len(q1s)):
            q, kp = q1s[t], k1s[t - 1]
            nq, nk = float(q.norm()), float(kp.norm())
            if nq > 1e-300 and nk > 1e-300:
                coss.append(float(q @ kp) / (nq * nk))      # ★ 余弦
                alpr.append(float(q @ kp) / nk)              # 投影比（DIR-39 的口径，留作连续性）
        diag["cos_q_k"] = statistics.fmean(coss) if coss else float("nan")
        diag["align_q_k"] = statistics.fmean(alpr) if alpr else float("nan")
        diag["cos_q_k_series"] = coss
    return L, diag


def train_arm40(qsrc, seeds, shared_pk=None, train_ctrl=False, steps=STEPS_YI, lr=LR):
    r"""**分臂联合训练**（探针乙的协议）：头 `(g1,Wo1,bo1)` 必训；控制器 `(Wq1,bq1)` 可选。

    ★ 初值与 DIR-39 的 `init_params()` **逐字同源**（同一生成器 4242、同一抽取顺序）
      ⇒ 三臂（`oracle`/`rand`/`learned`）的**头初值逐位相同**、`rand` 与 `learned` 的
      **控制器初值逐位相同** ⇒ 逐位配对。
    ★ `train_ctrl=False` ⇒ `(Wq1,bq1)` **不在优化器里** ⇒ **逐位冻结**（这是**可检查**的冻结：
      冻结臂与训练臂的差别**只**来自「训不训」）。
    """
    init = init_params()
    phi = {"g1": init["g1"], "Wo1": init["Wo1"], "bo1": init["bo1"],
           "Wq1": init["Wq1"], "bq1": init["bq1"]}
    names = ["g1", "Wo1", "bo1"]
    if train_ctrl:
        names += ["Wq1", "bq1"]
    # ★ 纪律 97：`rand` 臂的 `(Wq1,bq1)` **存在但不在优化器里** ⇒ 训练后**逐位不变**
    #   （这才是**可检查的冻结**；DIR-39 的 A39-8 用 `init−init` 是空断言 ⇒ DEF-39-5）。
    opt = torch.optim.Adam([phi[n] for n in names], lr=lr)
    gn = {}
    for _ in range(steps):
        opt.zero_grad()
        for sd in seeds:
            P = T37._mk(sd, phi, train=True)
            L, _ = trace40(P, N_REP, qsrc, sd, phi, shared_pk1=shared_pk)
            (L / len(seeds)).backward()
        for n in names:
            if phi[n].grad is not None:
                gn[n] = max(gn.get(n, 0.0), float(phi[n].grad.norm()))
        opt.step()
    return {k: v.detach().clone() for k, v in phi.items()}, gn


def train_per_instance(sd, phi_head, steps=STEPS_JIA, lr=LR):
    r"""探针甲：**逐实例**训 `(Wq1_i,bq1_i)`（★ 随机初值；★ **头冻结**）。

    ★ 控制器初值 = `init_params()` 的**同一份**随机初值（所有实例共用 ⇒ 与「不训」逐位配对）。
    ★ 头**冻结** ⇒ `phi_head` 的三个张量都 `detach()` ⇒ 梯度**只**进控制器。
    """
    init = init_params()
    # ★ DEF-40-1：`init` 的张量本身 requires_grad ⇒ `.clone()` 得到的是**非叶子** ⇒
    #   优化器拒绝（"can't optimize a non-leaf Tensor"）⇒ 必须先 `detach()` 再 `clone()`。
    phi = {"Wq1": init["Wq1"].detach().clone().requires_grad_(True),
           "bq1": init["bq1"].detach().clone().requires_grad_(True)}
    ph = {k: v.detach().clone() for k, v in phi_head.items()}
    opt = torch.optim.Adam([phi["Wq1"], phi["bq1"]], lr=lr)
    gn = {"Wq1": 0.0, "bq1": 0.0}
    for _ in range(steps):
        opt.zero_grad()
        P = T37._mk(sd, {**ph, **phi}, train=True)
        L, _ = trace40(P, N_REP, "learned", sd, phi)
        L.backward()
        for n in ("Wq1", "bq1"):
            if phi[n].grad is not None:
                gn[n] = max(gn[n], float(phi[n].grad.norm()))
        opt.step()
    out = {k: v.detach().clone() for k, v in phi.items()}
    return out, gn


def eval40(qsrc, phi, seeds, shared_pk=None, init_ctrl=None):
    Ls, coss, alprs, dn, qn, yn, ma, df = [], [], [], [], [], [], [], []
    for sd in seeds:
        P = T37._mk(sd, phi, train=False)
        L, dg = trace40(P, N_REP, qsrc, sd, phi, want_diag=True, shared_pk1=shared_pk)
        Ls.append(float(L))
        coss.append(dg["cos_q_k"])
        alprs.append(dg["align_q_k"])
        qn.append(dg["q_norm"]); yn.append(dg["y_norm"]); ma.append(dg["m_abs"])
        if "Wq1" in phi:
            dn.append(float((P["Wq1"] - P["Pk1"]).norm()))
            if init_ctrl is not None:
                df.append(float((P["Wq1"] - init_ctrl).norm()))
    # ★ DEF-40-2：`stdev` 要求 ≥2 个数据点 ⇒ 探针甲的**逐实例**评估只有 1 个实例 ⇒ 必须兜底。
    out = {"mean_L": statistics.fmean(Ls),
           "sd_L": statistics.stdev(Ls) if len(Ls) > 1 else 0.0,
           "L_series": Ls, "cos_q_k": statistics.fmean(coss),
           "align_q_k": statistics.fmean(alprs),
           "q_norm": statistics.fmean(qn), "y_norm": statistics.fmean(yn),
           "m_abs": statistics.fmean(ma)}
    if dn:
        out["dWq1_Pk1"] = statistics.fmean(dn)
        out["dWq1_Pk1_series"] = dn
    if df:
        out["dWq1_from_init"] = statistics.fmean(df)
        out["dWq1_from_init_max"] = max(df)
    return out


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    se = sd / len(d) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": 3.0 * se, "n": len(d)}


# ══════════ 探针甲
def probe_jia():
    t0 = time.time()
    print("══ 探针甲 · 可表达性（per-instance 读控制器；头冻结） ══", flush=True)
    print(f"[甲] 训头（oracle 臂，n_rep={N_REP}）n={N_YI_TRAIN} ...", flush=True)
    phi_head, gn_head = train_arm40("oracle", SEEDS_YI_TRAIN, shared_pk=None,
                                    train_ctrl=False, steps=STEPS_YI)
    print(f"[甲] 头完成 {time.time()-t0:.0f}s  梯度 {({k: round(v,4) for k,v in gn_head.items()})}",
          flush=True)
    init = init_params()
    rows = []
    for i, sd in enumerate(SEEDS_JIA):
        phi_ctl, gn = train_per_instance(sd, phi_head)
        phi_o = {**{k: v.detach().clone() for k, v in phi_head.items()}}
        phi_r = {**phi_o, "Wq1": init["Wq1"].clone(), "bq1": init["bq1"].clone()}
        phi_l = {**phi_o, "Wq1": phi_ctl["Wq1"], "bq1": phi_ctl["bq1"]}
        e_o = eval40("oracle", phi_o, [sd])
        e_r = eval40("rand", phi_r, [sd])
        e_l = eval40("learned", phi_l, [sd])
        rows.append({
            "seed": sd,
            "L_oracle": e_o["mean_L"], "L_rand": e_r["mean_L"], "L_learned": e_l["mean_L"],
            "cos_oracle": e_o["cos_q_k"], "cos_rand": e_r["cos_q_k"], "cos_learned": e_l["cos_q_k"],
            "dWq1_Pk1_learned": e_l.get("dWq1_Pk1"), "dWq1_Pk1_rand": e_r.get("dWq1_Pk1"),
            "gn_Wq1": gn["Wq1"], "gn_bq1": gn["bq1"],
        })
        if (i + 1) % 8 == 0:
            print(f"  甲 {i+1}/{N_JIA}: L_o={rows[-1]['L_oracle']:.4f} "
                  f"L_r={rows[-1]['L_rand']:.4f} L_l={rows[-1]['L_learned']:.4f} "
                  f"cos_l={rows[-1]['cos_learned']:.4f} "
                  f"|Wq-Pk|={rows[-1]['dWq1_Pk1_learned']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    L_o = [r["L_oracle"] for r in rows]
    L_r = [r["L_rand"] for r in rows]
    L_l = [r["L_learned"] for r in rows]
    res = {
        "n": N_JIA, "steps": STEPS_JIA, "head": "oracle 臂，冻结",
        "L_oracle_mean": statistics.fmean(L_o), "L_rand_mean": statistics.fmean(L_r),
        "L_learned_mean": statistics.fmean(L_l),
        "paired_rand_minus_learned": paired(L_r, L_l),
        "paired_oracle_minus_learned": paired(L_o, L_l),
        "cos_oracle": statistics.fmean([r["cos_oracle"] for r in rows]),
        "cos_rand": statistics.fmean([r["cos_rand"] for r in rows]),
        "cos_learned": statistics.fmean([r["cos_learned"] for r in rows]),
        "dWq1_Pk1_rand": statistics.fmean([r["dWq1_Pk1_rand"] for r in rows]),
        "dWq1_Pk1_learned": statistics.fmean([r["dWq1_Pk1_learned"] for r in rows]),
        "rows": rows,
    }
    print(f"[甲] 完成 {time.time()-t0:.0f}s  L_o={res['L_oracle_mean']:.6f} "
          f"L_r={res['L_rand_mean']:.6f} L_l={res['L_learned_mean']:.6f}  "
          f"cos: o={res['cos_oracle']:.4f} r={res['cos_rand']:.4f} l={res['cos_learned']:.4f}  "
          f"|Wq-Pk|: r={res['dWq1_Pk1_rand']:.4f} l={res['dWq1_Pk1_learned']:.4f}", flush=True)
    print(f"[甲] 配对 rand−learned: mean={res['paired_rand_minus_learned']['mean']:.6f} "
          f"3se={res['paired_rand_minus_learned']['3se']:.6f}", flush=True)
    print(f"[甲] 配对 oracle−learned: mean={res['paired_oracle_minus_learned']['mean']:.6f} "
          f"3se={res['paired_oracle_minus_learned']['3se']:.6f}", flush=True)
    return res


# ══════════ 探针乙
def probe_yi():
    t0 = time.time()
    print("\n══ 探针乙 · 真正的 §8（Pk1 跨实例共享；控制器仍共享；分臂联合训练头） ══", flush=True)
    spk = shared_pk1()
    arms, gns = {}, {}
    for name, (qsrc, tctrl) in (("oracle", ("oracle", False)),
                                ("rand", ("rand", False)),
                                ("learned", ("learned", True))):
        print(f"[乙] 训臂 {name}（qsrc={qsrc}, 控制器{'训' if tctrl else '冻'}）n={N_YI_TRAIN} ...",
              flush=True)
        phi, gn = train_arm40(qsrc, SEEDS_YI_TRAIN, shared_pk=spk, train_ctrl=tctrl)
        gns[name] = gn
        e = eval40(qsrc, phi, SEEDS_YI_EVAL, shared_pk=spk, init_ctrl=init_params()["Wq1"])
        arms[name] = e
        print(f"  乙 {name:8s} L={e['mean_L']:.6f} (sd {e['sd_L']:.4f}) cos={e['cos_q_k']:.4f} "
              f"|Wq-Pk|={e.get('dWq1_Pk1', float('nan')):.4f} "
              f"|Wq-init|={e.get('dWq1_from_init_max', float('nan')):.3e}  [{time.time()-t0:.0f}s]",
              flush=True)
    pr = paired(arms["rand"]["L_series"], arms["learned"]["L_series"])
    po = paired(arms["oracle"]["L_series"], arms["rand"]["L_series"])
    R_head = arms["rand"]["mean_L"] - arms["oracle"]["mean_L"]
    res = {
        "n_train": N_YI_TRAIN, "n_eval": N_YI_EVAL, "steps": STEPS_YI,
        "shared_pk1_seed": SHARED_PK1_SEED,
        "shared_pk1_md5": hashlib.md5(spk.numpy().tobytes()).hexdigest(),
        "arms": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_series")}
                 for k, v in arms.items()},
        "L_series": {k: v["L_series"] for k, v in arms.items()},
        "gradnorms": gns,
        "headroom_R_rand_minus_oracle": R_head,
        "paired_rand_minus_learned": pr,
        "paired_oracle_minus_rand": po,
        "power_ref_3se_at_400": 3.0 * pr["sd"] / math.sqrt(400),
    }
    print(f"[乙] 完成 {time.time()-t0:.0f}s  R={R_head:.6f}  "
          f"配对 rand−learned mean={pr['mean']:.6f} 3se={pr['3se']:.6f}  "
          f"功效参考 3se@400={res['power_ref_3se_at_400']:.6f}", flush=True)
    return res


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    out = {
        "note": "DIR-40 开卡前前置探针（**不进判据**；纪律 92）",
        "protocol": {"N_REP": N_REP, "STEPS_JIA": STEPS_JIA, "STEPS_YI": STEPS_YI,
                     "LR": LR, "N_JIA": N_JIA, "N_YI_TRAIN": N_YI_TRAIN,
                     "N_YI_EVAL": N_YI_EVAL, "SHARED_PK1_SEED": SHARED_PK1_SEED},
        "md5": {"device": _md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py"),
                "task": _md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py"),
                "dir37": _md5("/home/ec2-user/sgmn-loop/dir37/R1_task37.py"),
                "probe40": _md5("/home/ec2-user/sgmn-loop/dir40/probe40.py")},
        "effective_constants": {"D_H": D_H, "D_V": D_V, "D_K": D_K, "T": T,
                                "ETA": M.ETA, "LAM": M.LAM, "ALPHA": M.ALPHA,
                                "GAMMA": M.GAMMA, "KEY_SCALE": R.KEY_SCALE},
        "jia": probe_jia(),
        "yi": probe_yi(),
        "secs": time.time() - t0,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUT, "DIR40_probe.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n探针完成，用时 {out['secs']:.0f}s → dir40/out/DIR40_probe.json")
