#!/opt/pytorch/bin/python
r"""DIR-51 · 实现：**人格漂移代理量** —— 非人格写入相对**无写入对照**的超额 trait 余弦损失。

卡：`revolve/DIR-51/DIRECTION_rev51_v10.md`（rev 1.0，判据结构与规则已冻结；阈值数值于 rev 1.1 回填）。
★ **逐字派生**：装置（`dir31/R1_grad.py` + `dir33/R1_task33.py` + `dir47/R1_task47.py`）、
   写入内层循环 `dir48/R1_task48.step_pair`、内容生成器 `dir48/R1_task48.content` **一字不改**；
   **唯一改动** = 把「写入臂」提为显式自变量 + 增加 per-step 读数轨迹落盘。

★ **主判据量**（用户 2026-11-16 锁定）：
   `D_P^{(a)}(N) = c_P^{α=0}(t0+N) − c_P^{(a)}(t0+N)`
   其中 `α=0` = 无数据写入对照（保留正则与其余状态动力学，**在线步数相同**）。

用法：`R1_task51.py --arm {alpha0,random,ortho,orthoP,replay} --n N --seed0 S --out PATH [--steps K]`
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

torch.set_default_dtype(torch.float64)

ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task46 as T46      # noqa: E402
import R1_task47 as T47      # noqa: E402
import R1_task48 as T48      # noqa: E402

D_H = M.D_H
D_K = M.D_K
D_V = M.D_V
KEY_SCALE = R.KEY_SCALE
N_REP = T47.N_REP

ARMS = ("alpha0", "random", "ortho", "orthoP", "replay")
# ★ 种子派生偏移（与 DIR-48 的内容流种子 7777、DIR-51 探针的 7919/12345 均不同）
OFF_KEY, OFF_VAL, OFF_PERP, OFF_ANCHOR = 12345, 999, 777, 555
STEPS = 20          # ★ rev 1.1 冻结的终点 N（沿用探针的 20 步，不自造）
N_ANCHOR = 32       # ★ DEF-51-3：实例内随机锚点数（零假设水平）
FLOOR_REL = 1e-8    # ★ DEF-51-2：数值噪声底（相对 float64 精度 × 状态范数）


def q95(v):
    """单侧 95 分位（线性插值）。"""
    s = sorted(v)
    if not s:
        return float("nan")
    if len(s) == 1:
        return float(s[0])
    idx = 0.95 * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(s) - 1)
    f = idx - lo
    return float(s[lo] * (1 - f) + s[hi] * f)


def gen_vec(g, D):
    v = torch.randn(D, generator=g)
    return v / v.norm()


def perp_to(kP, g):
    r"""在 `D_K` 维内对 `k_P` 严格正交的单位向量。

    ★ `DEF-51-1`：初版用「减投影后归一化」⇒ 当随机基向量 `b` 与 `k_P` 近平行时残差趋零，
      归一化后方向随机 ⇒ 实测 `|k_P·perp|` 可达 **0.67**（"严格正交"前提**不成立**）。
      ⇒ 修法：**拒绝采样**（要求 `|b·k_P| ≤ 0.5`，保证残差不近零）+ 精确正交化
      ⇒ `|k_P·perp| ≤ 1e-15`（float64 舍入量级）。"""
    for _ in range(64):
        b = gen_vec(g, D_K)
        c = float(b @ kP)
        if abs(c) <= 0.5 * float(kP.norm()):
            # ★ 投影必须除以 ‖kP‖²：`KEY_SCALE=0.5` ⇒ ‖kP‖²=0.25 ≠ 1
            #   （初版漏除 ⇒ `kP·p = (b·kP)(1−‖kP‖²) ≠ 0` ⇒ 正交性彻底失效，实测 |cos| 可达 0.67）
            p = b - (c / float(kP @ kP)) * kP
            return p / p.norm()
    raise RuntimeError("perp_to：拒绝采样 64 次未命中（装置异常）")


def read_P(U, Vm, kP, vP):
    r"""**纯读**：返回 `(cos(Wk_P, v_P^0), ||W k_P||)`；不改动状态。"""
    Wm = M._W(U, Vm)
    r = (Wm @ kP).detach()
    nr = float(r.norm())
    nv = float(vP.norm())
    c = float((r @ vP) / (nr * nv)) if (nr > 0 and nv > 0) else float("nan")
    return c, nr


def nonperson_vec(g, vP, max_cos=0.5):
    r"""采一个与 `v_P^0` **显著不平行**的单位向量（`D_V=4` 下随机向量 `|cos|` 可达 0.99）。

    ★ `DEF-51-8`：初版直接用随机向量 ⇒ 实测 `max|cos(v_rand, v_P^0)| = 0.986`
      ⇒ 「非人格值」在个别步几乎与人格值平行 ⇒ 条件定义被污染。
      ⇒ 修法：**拒绝采样**，要求 `|cos(v_rand, v_P^0)| ≤ 0.5`。"""
    for _ in range(256):
        v = gen_vec(g, D_V)
        c = abs(float(v @ vP) / (v.norm() * vP.norm()))
        if c <= max_cos:
            return v
    raise RuntimeError("nonperson_vec：拒绝采样 256 次未命中（装置异常）")


def run_one(seed, arm, steps=STEPS, alpha0_kv=None):
    r"""一个实例 = 一个人格 trait + `steps` 步干扰写入。

    返回：读数轨迹 `cs`（长度 steps）、范数轨迹 `ns`、`c0`/`n0`（t0 时刻）、
          `n0_rand`（t0 时刻对**随机锚点**的余弦 = 零假设读数）、
          `n0_base`（**基座** `W^0` 对随机键的读出范数 = 数值噪声底候选）、
          `W0norm`、`final_wnorm`。
    """
    assert arm in ARMS, arm
    P = R.build(seed, "V0")
    g_base = torch.Generator().manual_seed(seed + OFF_KEY)
    g_val = torch.Generator().manual_seed(seed + OFF_VAL)
    g_anchor = torch.Generator().manual_seed(seed + OFF_ANCHOR)   # ★ DEF-51-9：E1 锚点用**独立** generator
    g_perp = torch.Generator().manual_seed(seed + OFF_PERP)
    kP = KEY_SCALE * gen_vec(g_base, D_K)
    vP = gen_vec(g_base, D_V)                 # v_P^0（原始人格参考值，单位向量）
    v_anchor = gen_vec(g_anchor, D_V)         # 随机锚点（零假设读数用，实例内 1 个）
    anchors_rand = [gen_vec(g_anchor, D_V) for _ in range(N_ANCHOR)]  # ★ DEF-51-3：实例内多锚点
    k_rand = KEY_SCALE * gen_vec(g_base, D_K)  # 基座噪声读出用（随机键）
    U, Vm = P["U1"].clone(), P["V1"].clone()
    W0 = M._W(U, Vm).detach().clone()
    # ★ 初始状态指纹（Stage A：五臂起跑状态逐位相同的可核证据）
    init_fp = hashlib.sha256(
        torch.cat([U.flatten(), Vm.flatten(), kP, vP]).detach().numpy().tobytes()).hexdigest()[:16]

    # ── 人格写入（t0 之前）：trait (k_P, v_P^0) 写入一次
    U, Vm = T48.step_pair(U, Vm, kP, vP, 1.0)

    # ── t0 读数
    c0, n0 = read_P(U, Vm, kP, vP)
    c0_rand, _ = read_P(U, Vm, kP, v_anchor)      # 零假设：随机锚点余弦（单锚点，参考）
    # ★ DEF-51-3：实例内 **多锚点**零假设水平（同一条 r_P 对 32 个随机锚点的余弦 q95）
    #   ⇒ E1 在**实例内配对**，避免「跨种子 q95」把 cos 的维度涨落误当噪声
    cs_anchor = []
    for va in anchors_rand:
        ca, _ = read_P(U, Vm, kP, va)
        cs_anchor.append(ca)
    c0_rand_q95 = q95(cs_anchor)
    # ★ DEF-51-2：数值噪声底 = **相对精度**尺度（float64 eps × 状态范数），而非「基座读出范数」
    #   （后者是**正常读出尺度**，用它当门槛会误杀 91% 实例）
    w_t0 = float(M._W(U, Vm).detach().norm())
    n_floor = FLOOR_REL * w_t0
    n0_base = float((W0 @ k_rand).norm())         # 基座对随机键的读出范数（仅参考，不作门槛）

    cs, ns, ks, anchors, vrand_cos = [], [], [], [], []
    for j in range(steps):
        if arm == "alpha0":                        # 无数据写入对照（α=0：数据项恒为 0）
            # ★ `alpha0_kv` 仅供 Stage A 的 A51-3（换 (k,v) 不应改变 α=0 臂结果）
            kv, vv, al = (kP, vP, 0.0) if alpha0_kv is None else (alpha0_kv[0], alpha0_kv[1], 0.0)
        elif arm == "random":                      # 同键换**非人格**值（阳性对照）
            kv, vv, al = kP, nonperson_vec(g_val, vP), 1.0
        elif arm == "ortho":                       # 正交键 + 非人格值（**主干扰条件**）
            kv, vv, al = KEY_SCALE * perp_to(kP, g_perp), nonperson_vec(g_val, vP), 1.0
        elif arm == "orthoP":                      # 正交键 + 人格值（**阴性对照**：结构同 ortho）
            kv, vv, al = KEY_SCALE * perp_to(kP, g_perp), vP, 1.0
        elif arm == "replay":                      # 同键同人值（强化/可学习性对照）
            kv, vv, al = kP, vP, 1.0
        else:
            raise ValueError(arm)
        if arm in ("ortho", "orthoP"):
            ks.append(float((kv @ kP) / (kv.norm() * kP.norm())))   # 键正交性实测
        anchors.append(vv.clone())
        if arm in ("random", "ortho"):
            vrand_cos.append(float((vv @ vP) / (vv.norm() * vP.norm())))
        U, Vm = T48.step_pair(U, Vm, kv, vv, al)
        c, n = read_P(U, Vm, kP, vP)
        cs.append(c)
        ns.append(n)

    # ★ 机制诊断：末态读出是否被「最近写入的**值**」带走（`v_anchor_last` / `v_anchor_first`）
    c_anchor_last, _ = read_P(U, Vm, kP, anchors[-1])
    c_anchor_first, _ = read_P(U, Vm, kP, anchors[0])

    finite = all(math.isfinite(x) for x in cs) and math.isfinite(c0)
    return {"seed": seed, "arm": arm, "steps": steps,
            "init_fp": init_fp, "vrand_cos": vrand_cos,
            "c0": c0, "n0": n0, "c0_rand": c0_rand, "c0_rand_q95": c0_rand_q95,
            "n0_base": n0_base, "n_floor": n_floor, "w_t0": w_t0,
            "cs": cs, "ns": ns, "ks": ks,
            "c_anchor_last": c_anchor_last, "c_anchor_first": c_anchor_first,
            "W0norm": float(W0.norm()), "final_wnorm": float(M._W(U, Vm).norm()),
            "finite": finite,
            "undef": (not finite)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=51000)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="r11")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(a.n):
            row = run_one(a.seed0 + i, a.arm, a.steps)
            row["tag"] = a.tag
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"[{a.arm}] {i+1}/{a.n}  {time.time()-t0:.1f}s", flush=True)
    print(f"DONE {a.arm} n={a.n} seed0={a.seed0} steps={a.steps} → {a.out}  墙钟 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
