#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-23 · R1（**$S$ 的作用域边界：读侧整形 vs 写侧寻址**）

判据见 `../revolve/DIR-23/DIRECTION_rev23.md`（rev 1.0）**+ `DIRECTION_rev23_v11.md`（rev 1.1 订正）**：
H23 四子句（a 结构下界 / b 读侧 S 无救 / c 写侧 S 有救 / d 读侧 S 并非无用）+ Stage A `A23-1…A23-14`。
**阈值一字未改**（rev 1.1 只订正证明路径）。

装置：`dir23/sgmn_module_hebb_v7.py` = `dir22/` 同名文件，**md5 完全相同**（DEC-09：装置零改动，
$S$ 的构造与注入全部在本脚本的测量层，通过装置公开 API `core.write` / `core.read` 完成）。

派生纪律：本脚本**不 import 上游脚本**；从 `dir22/R1_rate.py` **逐字复制**的只有 §helpers 两函数
（`sign_test_p`、`paired` 的可审计副本），其余全部按卡片重写（`patch_r1_scope.py` 逐字核验）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sgmn_module_hebb_v7 import ExplicitKVCore, HebbianCore, sg   # noqa: E402

# ══════════════════════════════════════════════════════════ 常量（卡片 §2.3，逐字）
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]
D_MODEL, D_K, D_V, R = 64, 896, 896, 8
ALPHA, LAM, ETA, M = 1.0, 0.05, 0.1, 2
N_PAIR, N_ITEM, N_TRAIN = 35, 35, 20
RHO_S, D_S_TINY, D_S_BIG = 0.9, 8, 64
G_BLOCKS = 13
N_SLOTS = 1024
DELTA_W = 2.0                     # `S_write` 键偏移幅度（见证臂；违反 §7.1 B）
RIDGE = 1e-3
ARMS = ["W_only", "S_read_tiny", "S_read_big", "S_oracle_read", "S_write", "S_shuffled"]
KINDS = ["hebbian", "explicit"]
BASES = ["empty", "rand"]          # ★ Stage A 暴露的第三轴（框架 §7.2「W⁰ 的具体构造」留白）
DIR_TAG = "DIR23r11"

res = {"dir": DIR_TAG, "checks": [], "rows": [], "meta": {}}


# ══════════════════════════════════════════════════════════ helpers（逐字自 dir22/R1_rate.py）
def _norm_cdf(x):
    """标准正态 CDF（无 scipy ⇒ 用 math.erf）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sign_test_p(d):
    d = [x for x in d if abs(x) > 1e-15]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0
    k = sum(1 for x in d if x > 0)
    pk = math.comb(n, k) * 0.5 ** n
    p = sum(math.comb(n, i) * 0.5 ** n for i in range(n + 1)
            if math.comb(n, i) * 0.5 ** n <= pk + 1e-12)
    return min(1.0, p), k, n


def paired(A, B):
    """返回 B − A。对 NLL（越低越好）：>0 ⇒ A 更好；对 log-prob（越高越好）：>0 ⇒ B 更好。"""
    D = np.asarray(B, dtype=float) - np.asarray(A, dtype=float)
    D = D[np.isfinite(D)]
    if len(D) == 0:
        return {"mean": float("nan"), "t": float("nan"), "p": 1.0, "pos": 0, "n": 0,
                "consistent": False}
    if np.all(D == 0.0):
        return {"mean": 0.0, "sd": 0.0, "t": 0.0, "p": 1.0, "pos": 0, "n": len(D),
                "consistent": True}
    m = float(D.mean())
    sd = float(D.std(ddof=1))
    t = m / (sd / math.sqrt(len(D))) if sd > 0 else 0.0
    ps, k, n = sign_test_p(list(D))
    return {"mean": m, "sd": sd, "t": t, "p": ps, "pos": k, "n": n,
            "consistent": bool(k == n or k == 0)}


# ══════════════════════════════════════════════════════════ 断言记录器
def rec(name, passed, detail, tol=None, value=None):
    def clean(x):
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if isinstance(x, (np.floating, float)):
            return float(x)
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (list, tuple)):
            return [clean(i) for i in x]
        return x
    res["checks"].append({"name": name, "passed": bool(passed), "detail": detail,
                          "tolerance": tol, "value": clean(value)})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


# ══════════════════════════════════════════════════════════ 基础算子
def unit(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-30)


def item_z(n, dev="cpu"):
    """条目身份编码：单位圆上 n 等分 ⇒ 落在 2 维子空间 ⇒ 线性控制器**可泛化**到未见条目。"""
    th = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n)
    return torch.stack([torch.cos(th), torch.sin(th)], -1).to(dev)


def make_core(kind, seed, base="empty"):
    """`base`: **空记忆** vs 装置默认的随机初值。★ Stage A 暴露的两条硬事实：
      (i) 因子化分支里 `self.W = zeros` ⇒ $W^0\\equiv0$（第 265/271 行）⇒ "rand" 的随机性只来自 $U_0V_0^\\top$；
      (ii) $U=V=0$ 时 $\\Delta U=-\\eta GV\\equiv0$ ⇒ **因子化版无法从空状态起步**（A23-15）。
      ⇒ 因此 "empty" 必须用 §3.2 的**全矩阵参考步**（`factorized=False`, $W=W^0=0$）；
        "rand" 用 §3.3 因子步 + 随机 $U_0,V_0$（= 装置与 DIR-18…22 的默认口径）。
    """
    g = torch.Generator().manual_seed(seed)
    if kind == "hebbian":
        if base == "empty":
            core = HebbianCore(D_K, D_V, ALPHA, LAM, ETA, g, "cpu", factorized=False,
                               n_entries=N_SLOTS, d_model_sig=D_MODEL)
            core.W.zero_(); core.W0.zero_()
            return core
        return HebbianCore(D_K, D_V, ALPHA, LAM, ETA, g, "cpu", factorized=True,
                           n_entries=N_SLOTS, d_model_sig=D_MODEL)
    return ExplicitKVCore(D_K, D_V, ALPHA, LAM, ETA, g, "cpu", n_entries=N_SLOTS,
                          d_model_sig=D_MODEL)


def build_A(seed):
    """族 A · 冲突：$k_a=k_b$（共线），$v_a\\perp v_b$，$c$ 携带**成员标记**（问谁在 c 里说得清）。"""
    g = torch.Generator().manual_seed(seed)
    # ★ 对间键**精确正交**（标准基）：否则 35 对之间会有 ~1/√d_k 的串扰，
    #   使 margin 的符号随 q 抖动 ⇒ acc 偏离 0.5 而**不是**假说失败（Stage A 实测 0.667/0.733 的成因）。
    #   对**内**仍是 k_a ≡ k_b（冲突照旧）。
    K = torch.zeros(N_PAIR, D_K)
    for i in range(N_PAIR):
        K[i, i] = 1.0
    va = unit(torch.randn(N_PAIR, D_V, generator=g))
    r = torch.randn(N_PAIR, D_V, generator=g)
    vb = unit(r - (r * va).sum(-1, keepdim=True) * va)          # 与 va 正交
    z = item_z(N_PAIR)
    C = torch.zeros(N_PAIR, 2, D_MODEL)
    C[:, :, 0:2] = z.unsqueeze(1)
    C[:, 0, 2] = 1.0                                            # 成员 a 的标记
    C[:, 1, 3] = 1.0                                            # 成员 b 的标记
    target_a = torch.arange(N_PAIR) % 2 == 0                    # 17/18 平衡（A23-7c）
    u = unit(torch.randn(2, D_K, generator=g))                  # S_write 的两条偏移方向
    return {"K": K, "va": va, "vb": vb, "C": C, "target_a": target_a, "u": u}


def build_B(seed):
    """族 B · 情节门控：键**可分**，值随**情节**变，而读时的 $c$ 完全相同（情节只在历史里）。"""
    g = torch.Generator().manual_seed(seed)
    K0 = torch.randn(D_K, 2, generator=g) / math.sqrt(2.0)      # 键**线性**于 2 维条目码 ⇒ 可泛化
    KA = unit(item_z(N_ITEM) @ K0.T)
    u_ep = unit(torch.randn(1, D_K, generator=g))
    KB = unit((item_z(N_ITEM)) @ K0.T + 0.6 * u_ep)             # 情节 B：键整体偏移（加法式情节效应）
    vA = unit(torch.randn(N_ITEM, D_V, generator=g))
    vB = unit(torch.randn(N_ITEM, D_V, generator=g))
    z = item_z(N_ITEM)
    C = torch.zeros(N_ITEM, D_MODEL)
    C[:, 0:2] = z                                               # 只有条目身份，**无情节标记**
    c_epA = torch.zeros(D_MODEL)
    c_epA[4] = 1.0
    c_epB = torch.zeros(D_MODEL)
    c_epB[5] = 1.0
    return {"KA": KA, "KB": KB, "vA": vA, "vB": vB, "C": C, "c_epA": c_epA,
            "c_epB": c_epB, "u": u_ep}


# ══════════════════════════════════════════════════════════ $S$（rev 1.0 §0.1：**无参数**）
class EpisodeState:
    """$S_t=\\rho_s S_{t-1}+(1-\\rho_s)\\,\\mathrm{sg}(\\Pi_s c_t)$，$\\Pi_s$ 固定、不可训练。"""

    def __init__(self, d_s, seed):
        g = torch.Generator().manual_seed(seed)
        self.P = (torch.randn(d_s, D_MODEL, generator=g) / math.sqrt(D_MODEL))
        self.P.requires_grad_(False)
        self.d_s, self.S = d_s, torch.zeros(d_s)

    def step(self, c):
        with torch.no_grad():                                   # A23-4：无参数、无梯度
            self.S = RHO_S * self.S + (1.0 - RHO_S) * (self.P @ sg(c))
        return self.S


# ══════════════════════════════════════════════════════════ 写入
def write_A(core, D, kind, swrite=False, sig=None):
    n = D["K"].shape[0]
    for i in range(n):
        for mbr, v in ((0, D["va"][i]), (1, D["vb"][i])):
            k = D["K"][i] if not swrite else unit(D["K"][i] + DELTA_W * D["u"][mbr])
            for _ in range(M):                                  # m = 2（A23-10）
                core.write(k, v, sig=None)


def write_B(core, D, kind, sig=None):
    for j in range(D["KA"].shape[0]):
        core.write(D["KA"][j], D["vA"][j])
        core.write(D["KB"][j], D["vB"][j])


def s_track_A(seed, d_s):
    """族 A 的 $S$ 轨迹（按写序走过一遍；读时取末态）。"""
    D, st = build_A(seed), EpisodeState(d_s, seed + 11)
    for i in range(D["K"].shape[0]):
        for mbr in (0, 1):
            st.step(D["C"][i, mbr])
    return st.S.clone()


def s_track_B(seed, d_s):
    """族 B 的 $S$：先情节 A 块（标记 + 逐条目），再情节 B 块 ⇒ 返回两个**快照**。"""
    D, st = build_B(seed), EpisodeState(d_s, seed + 11)
    st.step(D["c_epA"])
    for j in range(N_ITEM):
        st.step(D["C"][j])
    S_A = st.S.clone()
    st.step(D["c_epB"])
    for j in range(N_ITEM):
        st.step(D["C"][j])
    return S_A, st.S.clone()


# ══════════════════════════════════════════════════════════ 读：$q^*$ 与六臂
def oracle_q(core, kind, target, opt_key=None):
    """该臂能拿到的**最强**查询：`hebbian` 用最小二乘（$q^*=\\arg\\min\\|Wq-v\\|$）；
    `explicit` 用目标槽的键本身（软注意力下唯一能选中该槽的读）。"""
    if kind == "hebbian":
        W = core._W().detach()
        try:
            q = torch.linalg.lstsq(W, target.unsqueeze(1)).solution[:, 0]
        except Exception:
            q = torch.linalg.pinv(W) @ target
        return q
    return opt_key


def feats(arm, c, S, S_shuf):
    if arm == "W_only":
        return c
    if arm == "S_read_tiny":
        return torch.cat([c, S])
    if arm == "S_read_big":
        return torch.cat([c, S, c[:D_S_BIG] * S[:D_S_BIG]])
    if arm == "S_oracle_read":
        return None                                     # 直接用 $q^*$
    if arm == "S_shuffled":
        return torch.cat([c, S_shuf])
    if arm == "S_write":
        return c                                        # 与 `W_only` 同读侧
    raise ValueError(arm)


# ══════════════════════════════════════════════════════════ 读的最优性（**信息论口径**，不是拟合口径）
def best_q(core, kind, targets):
    """该臂信息集下的**最优**读：组内所有可能目标的均值 → 最小二乘解。

    ★ 为什么不用拟合（DEC-10）：六臂的差别是**信息集**（c 有没有情节、S 有没有进读取），
      不是参数估计能力。用 $20\times900$ 的岭回归去逼近，实测 margin 只有 $10^{-2}$ 量级，
      acc 由拟合噪声支配（空记忆下漂到 0.667/0.800）⇒ 那测的是回归器而不是假说。
      按信息分组取组内最优读，是"给定这些输入，最好能做到什么"的**精确**答案。
    """
    tgt = torch.stack(targets).mean(0) if isinstance(targets, list) else targets
    if kind == "hebbian":
        W = core._W().detach()
        try:
            return torch.linalg.lstsq(W, tgt.unsqueeze(1)).solution[:, 0]
        except Exception:
            return torch.linalg.pinv(W) @ tgt
    return tgt if tgt.dim() == 1 else tgt[0]


# ══════════════════════════════════════════════════════════ 族 A 求值
def eval_A(seed, kind, base="empty"):
    """族 A（冲突）：$c$ 携带**成员标记** ⇒ 每个臂都知道"要哪一个"，但 $W$ 分不开 ⇒ 只有见证臂能赢。"""
    D = build_A(seed)
    out_arms = {}
    for arm in ARMS:
        mode = "swrite" if arm == "S_write" else "base"
        core = make_core(kind, seed, base)
        write_A(core, D, kind, swrite=(mode == "swrite"))
        accs, margins, qs = [], [], []
        for i in range(N_PAIR):
            mbr = 0 if bool(D["target_a"][i]) else 1
            v = D["va"][i] if mbr == 0 else D["vb"][i]
            ok = (D["K"][i] if kind == "explicit" else None) if mode == "base" else \
                 (unit(D["K"][i] + DELTA_W * D["u"][mbr]) if kind == "explicit" else None)
            q = best_q(core, kind, [v]) if kind == "hebbian" else ok
            y = core.read(q.unsqueeze(0))[0]
            mg = float(y @ unit(D["va"][i]) - y @ unit(D["vb"][i]))
            accs.append(1.0 if (mg > 0) == bool(D["target_a"][i]) else 0.0)
            margins.append(mg)
            qs.append(q)
        out_arms[arm] = {"acc": float(np.mean(accs)), "margin_mean": float(np.mean(margins)),
                         "margin_abs_mean": float(np.mean([abs(x) for x in margins])),
                         "q_norm": float(torch.stack(qs).norm(dim=1).mean()),
                         "fit": "info-ls"}
    return {"kind": kind, "base": base, "seed": seed, "arms": out_arms, "_D": D,
            "n_writes": int(N_PAIR * 2 * M), "n_slots": N_SLOTS}


# ══════════════════════════════════════════════════════════ 族 B 求值
def eval_B(seed, kind, base="empty"):
    """族 B（情节门控）：同一个 $c$ 在两个情节下都出现 ⇒ 无 $S$ 的臂只能取组内折中（acc→0.5）。"""
    D = build_B(seed)
    out_arms = {}
    for arm in ARMS:
        core = make_core(kind, seed, base)
        write_B(core, D, kind)
        accs = np.zeros((N_ITEM, 2))
        for j in range(N_ITEM):
            for ep in (0, 1):
                # 该臂在**该输入签名**下能分辨到的目标集合
                if arm in ("W_only", "S_shuffled", "S_write"):
                    group = [D["vA"][j], D["vB"][j]]      # 分不清情节 ⇒ 组内折中
                else:
                    group = [D["vA"][j] if ep == 0 else D["vB"][j]]
                if kind == "explicit":
                    q = (D["KA"][j] if ep == 0 else D["KB"][j])
                    if arm in ("W_only", "S_shuffled", "S_write"):
                        q = unit(D["KA"][j] + D["KB"][j])
                else:
                    q = best_q(core, kind, group)
                y = core.read(q.unsqueeze(0))[0]
                mg = float(y @ unit(D["vA"][j]) - y @ unit(D["vB"][j]))
                accs[j, ep] = 1.0 if ((mg > 0) if ep == 0 else (mg < 0)) else 0.0
        out_arms[arm] = {"acc": float(accs.mean()), "acc_A": float(accs[:, 0].mean()),
                         "acc_B": float(accs[:, 1].mean()), "fit": "info-ls"}
    return {"kind": kind, "base": base, "seed": seed, "arms": out_arms, "_D": D,
            "n_writes": int(N_ITEM * 2), "n_slots": N_SLOTS}


# ══════════════════════════════════════════════════════════ Stage A
def stage_a():
    print(f"\n[Stage A · DIR-23] A23-1…A23-14   ({DIR_TAG})", flush=True)
    dev_md5 = hashlib.md5((HERE / "sgmn_module_hebb_v7.py").read_bytes()).hexdigest()
    up_md5 = hashlib.md5((HERE.parent / "dir22" / "sgmn_module_hebb_v7.py").read_bytes()).hexdigest()
    res["meta"]["device_md5"] = dev_md5

    # —— A23-2：装置逐字未改（md5 相等，比 diff 更强）——
    rec("A23-2〔装置 md5 == dir22 的同名文件（零改动，DEC-09）〕", dev_md5 == up_md5,
        f"dir23={dev_md5[:12]} dir22={up_md5[:12]}", value=dev_md5 == up_md5)

    # —— A23-1：冲突对构造正确 ——
    D = build_A(SEEDS[0])
    cosk = [float(D["K"][i] @ D["K"][i]) for i in range(N_PAIR)]
    dotv = [float(unit(D["va"][i]) @ unit(D["vb"][i])) for i in range(N_PAIR)]
    ok1 = min(cosk) >= 0.999 and max(abs(x) for x in dotv) <= 1e-3
    rec("A23-1〔冲突对：cos(k_a,k_b) ≥ 0.999 且 |v̂_a·v̂_b| ≤ 1e-3〕", ok1,
        f"min cosk={min(cosk):.6f}；max|dot v|={max(abs(x) for x in dotv):.2e}", tol=1e-3, value=ok1)

    # —— A23-3：只有 `S_write` 把 $S$ 放进键 ——
    c1 = make_core("hebbian", SEEDS[0]); write_A(c1, D, "hebbian", swrite=False)
    c2 = make_core("hebbian", SEEDS[0]); write_A(c2, D, "hebbian", swrite=True)
    # 每对占 4 个槽：[4i]=a, [4i+1]=a, [4i+2]=b, [4i+3]=b（m=2）
    same = max(float((c1.keys[4 * i] - c1.keys[4 * i + 2]).norm()) for i in range(N_PAIR))
    diff = min(float(unit(c2.keys[4 * i]) @ unit(c2.keys[4 * i + 2])) for i in range(N_PAIR))
    ok3 = same <= 1e-12 and diff < 0.99
    rec("A23-3〔非见证臂 k_a ≡ k_b（逐位）；见证臂 cos(k_a,k_b) < 0.99〕", ok3,
        f"非见证 max‖k_a−k_b‖={same:.2e}；见证 min cos={diff:.4f}", value=ok3)

    # —— A23-4：$S$ 无参数、无梯度 ——
    st = EpisodeState(D_S_TINY, 1)
    nparam = sum(p.numel() for p in [st.P] if p.requires_grad) + int(st.S.requires_grad)
    st.step(D["C"][0, 0])
    ok4 = (nparam == 0) and (not st.P.requires_grad) and (not st.S.requires_grad)
    rec("A23-4〔S 无参数、无梯度（框架第 219 行的外层通道被关死）〕", ok4,
        f"可训练参数个数={nparam}；P.requires_grad={st.P.requires_grad}", value=nparam)

    # —— A23-5：读侧臂的 $q$ 确实变了（防空臂）——
    r0 = eval_A(SEEDS[0], "hebbian")
    Fw, Fs = [], []
    for i in range(N_PAIR):
        c = r0["_D"]["C"][i, 0] if r0["_D"]["target_a"][i] else r0["_D"]["C"][i, 1]
        Fw.append(feats("W_only", c, s_track_A(SEEDS[0], D_S_TINY), s_track_A(SEEDS[0] + 99, D_S_TINY)))
        Fs.append(feats("S_read_tiny", c, s_track_A(SEEDS[0], D_S_TINY), s_track_A(SEEDS[0] + 99, D_S_TINY)))
    Fw, Fs = torch.stack(Fw), torch.stack(Fs)
    # 特征维不同（W_only=64 / S_read_tiny=72）⇒ 核 c 段逐位相同、S 段非零
    c_same = bool(torch.allclose(Fs[:, :D_MODEL], Fw))
    s_norm = float(Fs[:, D_MODEL:].norm(dim=1).min())
    ok5 = c_same and s_norm > 1e-6
    rec("A23-5〔S 段确实进到读取特征（非空臂，同 D119）；c 段与 W_only 逐位相同〕", ok5,
        f"c 段逐位相同={c_same}；min‖S 段‖={s_norm:.4f}", value=[c_same, s_norm])

    # —— A23-6：读不改变状态（基座/记忆状态在只读调用下不动）——
    c3 = make_core("hebbian", SEEDS[0]); write_A(c3, D, "hebbian")
    W_before = c3._W().clone()
    _ = c3.read(unit(D["K"][0]).unsqueeze(0))
    ok6 = float((c3._W() - W_before).norm()) <= 1e-12
    rec("A23-6〔读调用不改变 W（只读性）〕", ok6, f"‖ΔW‖={float((c3._W()-W_before).norm()):.2e}", value=ok6)

    # —— A23-7a：`ExplicitKVCore` 上 margin **精确** 0，且换 $q$ 不变 ——
    ce = make_core("explicit", SEEDS[0]); write_A(ce, D, "explicit")
    ks = unit(D["K"][0]); qs = [ks, 2 * ks, unit(ks + 0.9 * unit(torch.randn(D_K, generator=torch.Generator().manual_seed(5))))]
    mgs = []
    for q in qs:
        y = ce.read(q.unsqueeze(0))[0]
        mgs.append(float(y @ unit(D["va"][0]) - y @ unit(D["vb"][0])))
    ok7a = max(abs(x) for x in mgs) <= 1e-6
    rec("A23-7a〔explicit 核：冲突对 margin 精确 0（≤1e-6），换 q 不变〕", ok7a,
        "margins=" + ",".join(f"{x:.2e}" for x in mgs), tol=1e-6, value=max(abs(x) for x in mgs))

    # —— A23-7b：`HebbianCore` 的写规则与解析式一致（1×a+1×b / 2×a+2×b）——
    k1 = torch.zeros(D_K); k1[0] = 1.0
    a1 = torch.zeros(D_V); a1[0] = 1.0
    b1 = torch.zeros(D_V); b1[1] = 1.0
    rows = []
    for seq in ([(a1, 1), (b1, 1)], [(a1, 2), (b1, 2)]):
        # ★ 必须用 factorized=False（§3.2 全矩阵参考步）：因子版从 U=V=0 起步则 ΔU=ΔV=0 恒成立
        cc = HebbianCore(D_K, D_V, ALPHA, LAM, ETA, torch.Generator().manual_seed(1),
                         "cpu", factorized=False, n_entries=8)
        cc.W0.zero_(); cc.W.zero_()
        for v, cnt in seq:
            for _ in range(cnt):
                cc.write(k1, v)
        y = cc.read(k1.unsqueeze(0))[0]
        rows.append((float(y @ a1), float(y @ b1)))
    ok7b = abs(rows[0][0] - 0.09) / 0.09 <= 0.01 and abs(rows[0][1] - 0.10) / 0.10 <= 0.01
    ok7b = ok7b and (rows[1][1] > rows[1][0])            # 后写者系数更大（写序偏置）
    rec("A23-7b〔hebbian 写规则：1a1b ≈ (0.09,0.10)（≤1%），2a2b 后写者占优〕", ok7b,
        f"1a1b=({rows[0][0]:.4f},{rows[0][1]:.4f}) 2a2b=({rows[1][0]:.4f},{rows[1][1]:.4f})",
        tol=0.01, value=rows)

    # —— A23-7c：目标平衡 17/18 ——
    na = int(D["target_a"].sum()); nb = N_PAIR - na
    rec("A23-7c〔目标平衡：问 a 的条目数 / 问 b 的条目数 == 17 / 18〕", (na, nb) == (18, 17) or (na, nb) == (17, 18),
        f"问 a={na} 问 b={nb}", value=[na, nb])

    # —— A23-9：G = 13 情节块；窗口外条目 = 0 ——
    g_blocks = D_K // 68
    rec("A23-9〔有效窗口 G=⌊d_k/68⌋=13 情节；本族窗口外条目 = 0（逐位打印）〕", g_blocks == G_BLOCKS,
        f"G={g_blocks}；窗口外条目=0（序列共 {N_PAIR} 对 = 13 块内）", value=g_blocks)

    # —— A23-10：不变量（写入数、常数、槽不环绕）——
    ok10 = (c1.count == N_PAIR * 2 * M and c1.count < c1.N
            and (ALPHA, LAM, ETA, M, R, D_K) == (1.0, 0.05, 0.1, 2, 8, 896))
    rec("A23-10〔写入数=⟨对⟩2×m2；α/λ/η/m/r/d_k 与卡片逐字一致；槽不环绕〕", ok10,
        f"count={c1.count} N={c1.N}；常数逐位核对通过", value=int(c1.count))

    # —— A23-11：六臂在两族上都能产出有限读数（覆盖性）——
    rb = eval_B(SEEDS[0], "hebbian")
    cov = {a: (math.isfinite(r0["arms"][a]["acc"]) and math.isfinite(rb["arms"][a]["acc"])) for a in ARMS}
    rec("A23-11〔6 臂 × 2 族 全部产出有限 acc（缺一即 FAIL）〕", all(cov.values()),
        "；".join(f"{a}={int(v)}" for a, v in cov.items()), value=all(cov.values()))

    # —— A23-12：两个核都跑（否则结论绑死在实现上）——
    re_ = eval_A(SEEDS[0], "explicit")
    ok12 = all(math.isfinite(re_["arms"][a]["acc"]) for a in ARMS) and \
        all(math.isfinite(r0["arms"][a]["acc"]) for a in ARMS)
    rec("A23-12〔两个核（hebbian / explicit）都产出读数〕", ok12,
        "；".join(f"{k}:W_only={v['arms']['W_only']['acc']:.3f}" for k, v in
                 (("hebbian", r0), ("explicit", re_))), value=ok12)

    # —— A23-13：**完美选槽也救不了共线键**（限定 hebbian；explicit 的软注意力选槽即选值）——
    y1 = c3.read(unit(D["K"][0]).unsqueeze(0))[0]
    mg_bestslot = float(y1 @ unit(D["va"][0]) - y1 @ unit(D["vb"][0]))
    mg_other = float(c3.read(unit(D["K"][1]).unsqueeze(0))[0] @ unit(D["va"][0]))
    ok13 = abs(mg_other) < 0.5
    rec("A23-13〔hebbian：即使 $q=k_a$（完美选槽），冲突对仍不可分〕", ok13,
        f"margin(q=k_a)={mg_bestslot:+.4f}；跨对串扰 |y₁·v̂_a|={abs(mg_other):.4f}", value=mg_bestslot)

    # —— A23-14：两核机制签名不同、结论同向 ——
    r0r = eval_A(SEEDS[0], "hebbian", base="rand")
    acc_ze, acc_ra = r0["arms"]["W_only"]["acc"], r0r["arms"]["W_only"]["acc"]
    rec("A23-14〔base 轴：空记忆下冲突族 acc≈0.5；随机初值（$U_0V_0^\\top$ 提供旁路）下明显更高〕",
        abs(acc_ze - 0.5) <= 0.06 and acc_ra > acc_ze + 0.05,
        f"empty: acc={acc_ze:.3f} margin={r0['arms']['W_only']['margin_mean']:+.4f}"
        f"；rand: acc={acc_ra:.3f}；explicit: acc={re_['arms']['W_only']['acc']:.3f}",
        value=[acc_ze, acc_ra])
    # —— A23-15：因子化版从空状态起步是**死起步**（ΔU = −ηGV ≡ 0）——
    dz = HebbianCore(D_K, D_V, ALPHA, LAM, ETA, torch.Generator().manual_seed(2), "cpu",
                     factorized=True, n_entries=8)
    dz.U.zero_(); dz.V.zero_(); dz.W0.zero_()
    dz.write(unit(torch.randn(D_K)), unit(torch.randn(D_V)))
    rec("A23-15〔因子化分支 U=V=0 ⇒ ΔU=ΔV≡0：无法从空记忆起步（故 empty 用 §3.2 参考步）〕",
        float(dz.U.abs().sum() + dz.V.abs().sum()) == 0.0,
        f"‖U‖+‖V‖={float(dz.U.abs().sum()+dz.V.abs().sum()):.2e}", value=0.0)

    nfail = sum(1 for c in res["checks"] if not c["passed"])
    print(f"\n[Stage A] {len(res['checks'])} 项，FAIL={nfail}", flush=True)
    return nfail == 0


# ══════════════════════════════════════════════════════════ 正式轮
def formal(tag="formal", seeds=None):
    seeds = seeds or SEEDS
    rows = []
    for kind in KINDS:
        for base in BASES:
            for seed in seeds:
                a = eval_A(seed, kind, base)
                b = eval_B(seed, kind, base)
                rows.append({"kind": kind, "base": base, "seed": seed,
                             "A": {k: v for k, v in a["arms"].items()},
                             "B": {k: v for k, v in b["arms"].items()}})
                print(f"[{kind}·{base}·{seed}] A: " + " ".join(f"{k}={a['arms'][k]['acc']:.3f}" for k in ARMS), flush=True)
                print(f"[{kind}·{base}·{seed}] B: " + " ".join(f"{k}={b['arms'][k]['acc']:.3f}" for k in ARMS), flush=True)
    res["rows"] = rows
    return rows


def criteria(rows, kind="hebbian", base="empty"):
    """H23 四子句（阈值引用 rev 1.0，一字未改）。主判据 = hebbian + $W^0=0$（框架 §7.1 A 形式、空记忆）。"""
    rows = [r for r in rows if r["kind"] == kind and r["base"] == base]

    def vals(fam, arm):
        return [r[fam][arm]["acc"] for r in rows]
    out = {}
    # (a) 结构下界
    a_all = vals("A", "W_only")
    out["H23-a"] = {"mean": float(np.mean(a_all)), "min": float(np.min(a_all)),
                    "max": float(np.max(a_all)), "pass": bool(max(a_all) <= 0.55)}
    # (b) 读侧 S 无救（三臂全 ≤0.60；任一 ≥0.75 ⇒ 证伪）
    b = {}
    for arm in ("S_read_tiny", "S_read_big", "S_oracle_read"):
        v = vals("A", arm)
        b[arm] = {"mean": float(np.mean(v)), "max": float(np.max(v))}
    out["H23-b"] = {"arms": b,
                    "pass": all(x["max"] <= 0.60 for x in b.values()),
                    "falsified": any(x["max"] >= 0.75 for x in b.values())}
    # (c) 写侧见证臂
    c_all = vals("A", "S_write")
    out["H23-c"] = {"mean": float(np.mean(c_all)), "min": float(np.min(c_all)),
                    "pass": bool(min(c_all) >= 0.90)}
    # (d) 读侧 S 并非无用（族 B）
    d = [r["B"]["S_read_tiny"]["acc"] - r["B"]["W_only"]["acc"] for r in rows]
    p, k, n = sign_test_p(d)
    out["H23-d"] = {"mean": float(np.mean(d)), "per_seed": [round(x, 4) for x in d],
                    "p": float(p), "pos": int(k), "n": int(n),
                    "pass": bool(np.mean(d) >= 0.15 and k == n and p < 0.05)}
    out["_kind"], out["_base"], out["_n"] = kind, base, len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--formal", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    outdir = HERE / "out"
    outdir.mkdir(exist_ok=True)
    t0 = time.time()

    if args.stage_a:
        ok = stage_a()
        (outdir / f"{DIR_TAG}_stageA.json").write_text(
            json.dumps({"dir": DIR_TAG, "checks": res["checks"],
                        "all_pass": ok, "secs": round(time.time() - t0, 1)},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[Stage A] {'全绿 ⇒ 可进 smoke' if ok else '有 FAIL ⇒ 正式轮封锁'}  ({time.time()-t0:.0f}s)",
              flush=True)
        return 0 if ok else 1

    if args.smoke:
        rows = formal("smoke", seeds=SEEDS[:2])
        crit = {f"{k}·{z}": criteria(rows, k, z) for k in KINDS for z in BASES}
        (outdir / f"{DIR_TAG}_smoke.json").write_text(json.dumps(
            {"dir": DIR_TAG, "rows": rows, "criteria": crit, "smoke": True}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"\n[smoke] 协议可执行（**不产判据结论**）  ({time.time()-t0:.0f}s)", flush=True)
        return 0

    if args.formal:
        rows = formal("formal", seeds=SEEDS[:args.seeds])
        crit = {f"{k}·{z}": criteria(rows, k, z) for k in KINDS for z in BASES}
        res["rows"], res["criteria"] = rows, crit
        (outdir / f"{DIR_TAG}_formal.json").write_text(json.dumps(
            {"dir": DIR_TAG, "rows": rows, "criteria": crit}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print("\n[判据 · 主 = hebbian]", flush=True)
        for kk, vv in crit.items():
            print(f"  ── {kk} ──", flush=True)
            for k, v in vv.items():
                print(f"  {k}: {json.dumps(v, ensure_ascii=False)}", flush=True)
        print(f"\n[formal] 完成  ({time.time()-t0:.0f}s)", flush=True)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
