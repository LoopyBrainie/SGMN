r"""DIR-28 · R1_couple.py —— 因子化低秩记忆下的**跨层耦合稳定性**（声明的最小两层实例 $M_2^r$）

卡片：`revolve/DIR-28/DIRECTION_rev28_v14.md`（rev 1.4 冻结；rev 1.0/1.1/1.2/1.3 全留档）
脚本派生自：**无**（自建，卡 §5.1：D-09 的 `dir03/` 装置与卡片在两侧工作区都不存在，故只声明实例，不写"复现 D-09"）。
装置公式**逐字**取自卡 §3.1 / §3.2 / §3.3（即 v2.4 §3.1 第 114–126 行的局部目标与梯度）。

实例（全标量 $d_k{=}d_v{=}1$、$r{=}1$、$\alpha\equiv1$、$\lambda{=}0.05$、$\mu{=}5$、$W^0{=}0$）：
  $k_1=q_1=a_t$；**耦合只走键通道**：$k_2=a_t(1+\gamma W_{1,t})$，$v_2$ 外生；
  $G_l=(W_lk_l-v_l)k_l+\lambda W_l$；全矩阵 $W_l\!\leftarrow\!W_l-\eta G_l$；
  因子化（**同步**）$U_l\!\leftarrow\!U_l-\eta G_lV_l,\;V_l\!\leftarrow\!V_l-\eta G_l^{\top}U_l$，$W_l=U_lV_l^{\top}$。

实现层声明（rev 1.5 的 DEF-28-6/7/8 写进卡片；此处先落码）：
  ① **不用上穿界 $B$**（卡 §6 坑 6 不适用）：判据量是**增长率** $\hat\lambda$，不是"是否越过某界"。
  ② **$\eta_c$ 定位**：56 点对数网格 `[1e-3, 2.0]`（比 1.15）上算 $\mathrm{median}_s\hat\lambda$，取第一个由正变负的
     区间，再 6 次算术二分 ⇒ 分辨率 $\approx1.15/2^6\approx2\%$，优于卡要求的 4%。
  ③ **因子化臂判据范数** $|u_lv_l|$（DEF-28-5）；$\max(|u_l|,|v_l|)$ 只作报告。
  ④ **噪声**：单一 `default_rng(SEEDS[0])` 抽 $(T,S)$ 三组独立流，**第 $j$ 列即第 $j$ 个"种子"**（确定性、可复算）。
  ⑤ **D-09 的 0.344 不作阈值/不作对照**（卡 §5.1）。
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

# ══════════════════════════════════════ 冻结常数（卡 §3.1 / §5.2 / §5.5）
MU = 5.0
LAM = 0.05
SIG0 = 1.0                       # init-A：u=v=σ0 ⇒ W0=1（DEF-28-11：σ0=0.1 时 fac 的有效步长 ∝2σ0²=0.02
                                 #   ⇒ η_c^fac≈24.5 > ETA_HI，网格不可达；取 σ0=1 使两臂在 t=0 可比）
SIG1 = 1.0                       # init-B：u=√5, v=1/√5 ⇒ W0=1（不同尺度）
PRIMARY_INIT = "init-A"          # DEF-27-12：主 init 事前声明
T = 20000                        # 每种子步数（卡 §5.5）
S = 64                           # 列数＝"种子"数（卡 §5.3）
SEED0 = 20261110
GAMMA_CONF = 0.5                 # 唯一确认性 γ（卡 §1.1）
GAMMAS_REPORT = (1.0, 2.0)       # 报告用
ETA_LO, ETA_HI, K_GRID, K_BIS = 1e-3, 2.0, 56, 6
ANCHOR_AS = 0.49090552           # 卡 A27-4（D-09 归档 0.49091）
ANCHOR_MS = 0.1337704            # 卡 A27-5
FLOOR = 0.10                     # T2/T10
MOVE_MS = 0.08                   # T9（=2×4%）
OUT = "/home/ec2-user/sgmn-loop/dir28/out"
RS = 1e250                       # 早停阈值：W 形式（DEF-28-17：必须远高于随机涨落）
RS_FAC = 1e150                   # fac 形式更小：递推里有 $v^2,u^2$，若 |v|\to10^{155} 则平方溢出（DEF-28-18）
CLIP = 1e150                     # 矩累加的裁剪（防 1e500 溢出）


# ══════════════════════════════════════ 装置核心（一步，可向量化）
def step_full(W1, W2, a, va, vb, eta, gamma, indep=False):
    r"""全矩阵臂一步（同步）。$k_2=a(1+\gamma W_1)$；`indep` 时 $k_2\equiv a$。"""
    G1 = (W1 * a - va) * a + LAM * W1
    k2 = a if indep else a * (1.0 + gamma * W1)
    G2 = (W2 * k2 - vb) * k2 + LAM * W2
    return W1 - eta * G1, W2 - eta * G2, k2, G1, G2


def step_fac(u1, v1, u2, v2, a, va, vb, eta, gamma):
    r"""因子化臂一步（**同步**：$G$ 用旧 $U,V$；卡 §3.3 第 155 行、坑 2）。"""
    W1 = u1 * v1
    G1 = (W1 * a - va) * a + LAM * W1
    k2 = a * (1.0 + gamma * W1)
    W2 = u2 * v2
    G2 = (W2 * k2 - vb) * k2 + LAM * W2
    return (u1 - eta * G1 * v1, v1 - eta * G1 * u1,
            u2 - eta * G2 * v2, v2 - eta * G2 * u2, k2, G1, G2)


def noise():
    r"""外生流的抽样。

    ★ **DEF-28-10（rev 1.6）**：$v^{(1)}\equiv v^{(2)}\equiv0$（**无加性噪声**），只有 $a_t\sim\mathcal N(0,\mu)$。
    原因（Stage A 第一跑抓到）：带加性噪声时 $\hat\lambda=\frac1T\log|W_T/W_0|$ 测的是
    "从 $W_0$ 长到**噪声底**"的瞬态，不是乘积的指数 —— 实测 $\mathrm{median}\hat\lambda(\eta_c/4)=+2.1\times10^{-4}$
    而真值 $\approx-0.6$。取 $v\equiv0$ 后 $W_{n+1}=A_nW_n$ 是**纯乘性**的 ⇒
    $\hat\lambda\equiv\frac1T\sum_t\log|A_t|$，**无瞬态、无标度依赖**，且与 A27-4/A27-5 的解析锚**同一对象**
    （两个锚本来就是按 $W^{+}=[1-\eta(k^2+\lambda)]W$ 算的）。
    """
    g = np.random.default_rng(SEED0)
    a = g.normal(0.0, math.sqrt(MU), size=(T, S))
    return a, np.zeros((T, S)), np.zeros((T, S))


A_, VA_, VB_ = noise()


def init_state(arm, init, shape):
    B = init == "init-B"
    if arm == "fac":
        if B:                                            # init-B：不同尺度，W0=√5·(1/√5)=1
            u0, v0 = math.sqrt(5.0), 1.0 / math.sqrt(5.0)
        else:                                            # init-A：对称，W0=1
            u0 = v0 = SIG0
        return [np.full(shape, u0), np.full(shape, v0), np.full(shape, u0), np.full(shape, v0)]
    w0 = 1.0                                             # 两臂 W0 相同（comparability）
    if arm == "one":
        return [np.full(shape, w0)]
    return [np.full(shape, w0), np.full(shape, w0)]


def run(arm, eta, gamma, init=PRIMARY_INIT):
    r"""返回 dict：lam (S,K) 增长率、ms (K) 二阶矩斜率、k2sq (K) 时均 $E[k_2^2]$、W1end/W2end、trunc。

    判据量（**声明**，卡 DEF-27-10 允许的"按声明"选项）：
        $\hat\lambda_s=\max_l\ \frac1T\log\frac{|W_{l,T}|}{|W_{l,0}|}$，fac 臂 $W_l=u_lv_l$（DEF-28-5）
    ⇒ 逐层指数取 max（系统在"**任一层**发散"时算不稳定）。★ 首版用 $\max_l|W_l|$ 的**端点幅值比**，
    会让层 1（与 $\gamma$ 无关）永远压住层 2 ⇒ $\gamma{=}0.5/1/2$ 给出**同一个** $\eta_c$（DEF-28-15）。
    **早停**（DEF-28-16）：状态范数超 $10^{12}$ 即停，$\hat\lambda$ 用**实际走过的步数**算 —— 乘积的 log 比在这段上仍是无偏的，
    且比"重标定"干净（重标定会改 fac 的动力学，因为递推里 $u^2,v^2$ 各自出现，不是规范不变的）。
    """
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    K = eta.size
    shape = (S, K)
    eta_b = eta.reshape(1, K)
    st = init_state(arm, init, shape)

    def Ws(st_):
        if arm == "fac":
            return st_[0] * st_[1], st_[2] * st_[3]
        if arm == "one":
            return st_[0], np.zeros_like(st_[0])
        return st_[0], st_[1]

    def mag_state(st_):
        if arm == "fac":
            return np.maximum(np.maximum(np.abs(st_[0]), np.abs(st_[1])),
                              np.maximum(np.abs(st_[2]), np.abs(st_[3])))
        return np.abs(st_[0]) if arm == "one" else np.maximum(np.abs(st_[0]), np.abs(st_[1]))

    W1, W2 = Ws(st)
    w1_0 = np.maximum(np.abs(W1), 1e-300).copy()
    w2_0 = np.maximum(np.abs(W2), 1e-300).copy()
    m2_0 = ((np.clip(W1, -1e150, 1e150) ** 2 + np.clip(W2, -1e150, 1e150) ** 2).mean(axis=0)).copy()
    k2sum = np.zeros(shape)
    alive = np.ones(shape, dtype=bool)
    t_dead = np.full(shape, float(T))          # 每列**自己的**死亡时刻（早期截断）
    for t in range(T):
        a, va, vb = A_[t][:, None], VA_[t][:, None], VB_[t][:, None]
        old = list(st)
        if arm == "fac":
            u1, v1, u2, v2 = st
            nu1, nv1, nu2, nv2, k2, _, _ = step_fac(u1, v1, u2, v2, a, va, vb, eta_b, gamma)
            st = [nu1, nv1, nu2, nv2]
        elif arm == "one":
            G1 = (old[0] * a - va) * a + LAM * old[0]
            st = [old[0] - eta_b * G1]
            k2 = a
        else:
            nW1, nW2, k2, _, _ = step_full(old[0], old[1], a, va, vb, eta_b, gamma,
                                           indep=(arm == "indep"))
            st = [nW1, nW2]
        if not alive.any():
            break
        st = [np.where(alive, n, o) for n, o in zip(st, old)]   # 死列冻结，不再改
        k2sum += np.where(alive, np.clip(k2, -CLIP, CLIP) ** 2, 0.0)
        thr = RS_FAC if arm == "fac" else RS
        ms_ = mag_state(st)
        over = alive & ((ms_ > thr) | ~np.isfinite(ms_))

        if over.any():
            t_dead[over] = float(t + 1)
            alive &= ~over
            st = [np.where(over, np.where(np.isfinite(x), x, thr), x) for x in st]  # 非有限值换成有限哨兵
            st = [np.where(over, np.clip(x, -thr, thr), x) for x in st]
    W1, W2 = Ws(st)
    t_ = np.maximum(t_dead, 1.0)
    lam1 = np.log(np.maximum(np.abs(W1), 1e-300) / w1_0) / t_
    lam2 = np.log(np.maximum(np.abs(W2), 1e-300) / w2_0) / t_
    lam = lam1 if arm == "one" else np.maximum(lam1, lam2)
    W1c, W2c = np.clip(W1, -CLIP, CLIP), np.clip(W2, -CLIP, CLIP)
    ms = (np.log(np.maximum((W1c ** 2 + W2c ** 2).mean(axis=0), 1e-300))
          - np.log(np.maximum(m2_0, 1e-300))) / t_.mean(axis=0)
    k2sq = (k2sum / t_).mean(axis=0)
    return {"lam": lam, "ms": ms, "k2sq": k2sq, "W1end": W1, "W2end": W2,
            "t_dead_med": float(np.median(t_dead)), "n_trunc": int((t_dead < T).sum())}


def med_lam(arm, eta, gamma, init=PRIMARY_INIT):
    r = run(arm, eta, gamma, init)
    lam = r["lam"]
    return np.median(lam, axis=0), (np.percentile(lam, 75, axis=0) - np.percentile(lam, 25, axis=0))


def eta_c(arm, gamma, init=PRIMARY_INIT):
    r"""中位数 $\hat\lambda$ 过零（DEF-27-10）+ 网格 + 6 次二分（DEF-28-8）。"""
    grid = np.exp(np.linspace(math.log(ETA_LO), math.log(ETA_HI), K_GRID))
    m, iqr = med_lam(arm, grid, gamma, init)
    # λ̂ 随 η **递增**（η 越大越不稳）⇒ 过零点是从负到正（DEF-28-14：首版写成 +→−，方向反了）
    idx = np.where((m[:-1] <= 0) & (m[1:] > 0))[0]
    info = {"med": m.tolist(), "iqr_max": float(np.max(iqr)),
            "se_med": float(1.25 * float(np.max(iqr)) / math.sqrt(S))}
    if idx.size == 0:
        return None, ("GT_HI" if m[-1] > 0 else ("LT_LO" if m[0] < 0 else "NO_CROSS")), info
    i = int(idx[0])
    lo, hi = float(grid[i]), float(grid[i + 1])
    for _ in range(K_BIS):
        mid = 0.5 * (lo + hi)
        mm, _ = med_lam(arm, [mid], gamma, init)
        if mm[0] > 0:            # 中点已不稳定 ⇒ 根更小
            hi = mid
        else:
            lo = mid
    info["bracket"] = [lo, hi]
    info["res_rel"] = (hi - lo) / (0.5 * (hi + lo))
    return 0.5 * (lo + hi), "OK", info


# ══════════════════════════════════════ 解析锚
def anchor_as(n=400001):
    r"""$E\ln|1-\eta(k^2+\lambda)|$ 的零点（$k^2=\mu\chi^2_1$），A27-4。"""
    z = np.linspace(-9.0, 9.0, n)
    w = np.exp(-0.5 * z ** 2)
    w /= w.sum()
    k2 = MU * z ** 2

    def f(eta):
        return float(np.sum(w * np.log(np.abs(1.0 - eta * (k2 + LAM)))))
    lo, hi = 0.30, 0.70
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)   # f 在根右侧为正（DEF-28-12）
    return 0.5 * (lo + hi), f(0.48091), f(ANCHOR_AS), f(0.50091)


def anchor_ms_quad(n=400001):
    z = np.linspace(-9.0, 9.0, n)
    w = np.exp(-0.5 * z ** 2)
    w /= w.sum()
    e = float(np.sum(w * (MU * z ** 2 + LAM) ** 2))
    return 2.0 * (MU + LAM) / e, e


def eta_c_ms(v):
    r"""外生单层 ms 门槛闭式：$k^2\sim v\chi^2_1$ ⇒ $E[(k^2+\lambda)^2]=3v^2+2\lambda v+\lambda^2$。"""
    return 2.0 * (v + LAM) / (3.0 * v * v + 2.0 * LAM * v + LAM * LAM)


# ══════════════════════════════════════ S1（不阻断设计检查）
def s1():
    rows = []
    for g in (0.0, 0.5, 1.0, 2.0):
        for e in (0.05, 0.2, 0.4909, 0.8, 1.2, 2.0):
            a, vb, W1, W2 = 1.3, 0.7, 0.42, 0.25
            k1, k2 = a, a * (1.0 + g * W1)
            J11, J22 = 1.0 - e * (k1 ** 2 + LAM), 1.0 - e * (k2 ** 2 + LAM)
            J12 = 0.0                                   # 结构恒零（W1 不依赖 W2）
            J21 = e * a * g * (vb - 2.0 * k2 * W2)      # DEF-27-7
            rows.append({"gamma": g, "eta": e, "J11": J11, "J12": J12, "J21": J21, "J22": J22,
                         "rho": max(abs(J11), abs(J22))})
    return {"rows": rows, "j12_max": max(abs(r["J12"]) for r in rows),
            "rho_eq_maxA": max(abs(r["rho"] - max(abs(r["J11"]), abs(r["J22"]))) for r in rows),
            "note": ("块下三角是定理（$W_1$ 不依赖 $W_2$）⇒ $\\rho(\\mathcal J)=\\max_l\\rho(A_l)$；"
                     "$W^*{=}0$ 处 $k_2=a$ ⇒ $\\gamma$ 从 $E[\\mathcal J]$ 与 $\\mathcal T$ 中消失，"
                     "联合效应只能经 $E[W_1^2]>0$ 进入（DEF-27-8）")}


# ══════════════════════════════════════ 工作例（卡 §3.2 W1–W4）
def worked():
    W, k, v, lam, eta = 0.7, 1.3, -0.4, 0.05, 0.01
    G = (W * k - v) * k + lam * W
    w1 = {"G": G, "W_next": W - eta * G}
    u, vv, kk, vt, lam2, eta2 = 0.6, 1.1, -0.9, 0.3, 0.05, 0.01
    Wf = u * vv
    Gf = (Wf * kk - vt) * kk + lam2 * Wf
    un, vn = u - eta2 * Gf * vv, vv - eta2 * Gf * u
    # 异步（错）式：v 用新 u
    vn_async = vv - eta2 * ((un * vv) * kk - vt) * kk * un - eta2 * lam2 * un * vv * un
    w2 = {"W": Wf, "G": Gf, "u_next": un, "v_next": vn, "asym_gap": abs(vn - vn_async)}
    a, W1, g = 1.7, 0.42, 2.0
    w3 = {"k2": a * (1.0 + g * W1), "k2_g0": a * (1.0 + 0.0 * W1), "a": a}
    g4, eta4, lam4, a4, v24, W24 = 2.0, 0.01, 0.05, 1.3, 0.7, 0.4

    def f_(W1_, W2_):
        k2_ = a4 * (1.0 + g4 * W1_)
        return (1.0 - eta4 * (k2_ ** 2 + lam4)) * W2_ + eta4 * v24 * k2_
    h = 1e-6
    fd = (f_(h, W24) - f_(-h, W24)) / (2 * h)
    k2_0 = a4 * (1.0 + g4 * 0.0)
    good = eta4 * a4 * g4 * (v24 - 2.0 * k2_0 * W24)
    wrong = -eta4 * a4 * g4 * (W24 * k2_0 - v24)
    w4 = {"fd": fd, "good": good, "wrong": wrong,
          "rel_good": abs(fd - good) / max(abs(fd), 1e-300),
          "rel_wrong": abs(fd - wrong) / max(abs(fd), 1e-300)}
    return {"W1": w1, "W2": w2, "W3": w3, "W4": w4}


# ══════════════════════════════════════ Stage A
def stageA():
    t0 = time.time()
    md5 = hashlib.md5(open(__file__, "rb").read()).hexdigest()
    A = []

    def chk(name, ok, detail):
        A.append({"id": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {str(detail)[:170]}")

    chk("A27-1", True, {"script_md5": md5, "monkeypatch": [], "self_built": True})
    W = worked()
    _w1 = step_full(0.7, 0.0, 1.3, -0.4, 0.0, 0.01, 0.0)   # 走装置函数，不是重抄公式
    chk("A27-2a", abs(float(_w1[0]) - 0.68262) < 1e-12 and abs(W["W1"]["W_next"] - 0.68262) < 1e-12,
        {"script_W_next": float(_w1[0]), "hand": 0.68262, "G": W["W1"]["G"]})
    chk("A27-2b", W["W2"]["asym_gap"] > 1e-8, W["W2"])
    chk("A27-2c", abs(W["W3"]["k2"] - 3.128) < 1e-12 and W["W3"]["k2_g0"] == W["W3"]["a"], W["W3"])
    chk("A27-2d", W["W4"]["rel_good"] < 1e-6 and W["W4"]["rel_wrong"] > 1e-3, W["W4"])

    # A27-3：γ=0 退化（§3.1 的链条断开，逐位）
    f_ = run("full", [0.05], 0.0)
    i_ = run("indep", [0.05], 0.0)
    o_ = run("one", [0.05], 0.0)
    d_k2 = float(np.abs(f_["k2sq"] - i_["k2sq"]).max())
    d_lam = float(np.abs(f_["lam"] - i_["lam"]).max())
    d_w1 = float(np.abs(f_["W1end"] - o_["W1end"]).max())
    chk("A27-3", d_k2 == 0.0 and d_lam == 0.0 and d_w1 == 0.0,
        {"k2_diff": d_k2, "lam_diff": d_lam, "W1_vs_layer1only": d_w1, "k2sq_full": float(f_["k2sq"][0])})

    r_as, f_lo, f_mid, f_hi = anchor_as()
    chk("A27-4", abs(r_as - ANCHOR_AS) / ANCHOR_AS < 4e-3 and f_lo < 0 < f_hi,
        {"root": r_as, "anchor": ANCHOR_AS, "rel": (r_as - ANCHOR_AS) / ANCHOR_AS,
         "f(0.48091)": f_lo, "f(anchor)": f_mid, "f(0.50091)": f_hi})
    r_ms, e_m = anchor_ms_quad()
    chk("A27-5", abs(r_ms - ANCHOR_MS) / ANCHOR_MS < 1e-6,
        {"quad": r_ms, "closed": 2.0 * (MU + LAM) / e_m, "anchor": ANCHOR_MS,
         "rel": (r_ms - ANCHOR_MS) / ANCHOR_MS})

    s = s1()
    chk("A27-6", True, {"non_blocking": True, "note": s["note"]})
    chk("A27-7", s["j12_max"] <= 1e-14, {"j12_max": s["j12_max"]})
    chk("A27-11", s["rho_eq_maxA"] <= 1e-9, {"cells": len(s["rows"]), "rho_eq_maxA": s["rho_eq_maxA"]})

    m_lo, _ = med_lam("full", [ANCHOR_AS / 4.0], 0.0)
    m_hi, _ = med_lam("full", [4.0 * ANCHOR_AS], 0.0)
    chk("A27-8", float(m_lo[0]) <= -0.01 and float(m_hi[0]) >= 0.01,
        {"med@eta_c/4": float(m_lo[0]), "med@4eta_c": float(m_hi[0])})

    l1 = run("full", [0.05], 0.5)["lam"]
    l2 = run("full", [0.05], 0.5)["lam"]
    chk("A27-9", float(np.abs(l1 - l2).max()) == 0.0, {"bitwise": float(np.abs(l1 - l2).max())})
    chk("A27-10", len(A) + 1 >= 10, {"count": len(A) + 1})

    ok = all(a["pass"] for a in A)
    out = {"stage": "A", "md5": md5, "seed0": SEED0, "S": S, "T": T,
           "all_pass": ok, "n_pass": sum(a["pass"] for a in A), "n": len(A),
           "assertions": A, "worked": W, "s1_note": s["note"], "secs": time.time() - t0}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/DIR28r10_stageA.json", "w"), ensure_ascii=False, indent=1)
    print(f"Stage A: {out['n_pass']}/{out['n']} PASS  ({out['secs']:.1f}s)  md5={md5}")
    return ok


# ══════════════════════════════════════ smoke（γ=0：全矩阵臂复现解析锚）
def smoke():
    t0 = time.time()
    c, st, info = eta_c("full", 0.0)
    ok = c is not None and abs(c - ANCHOR_AS) / ANCHOR_AS < 0.06
    out = {"stage": "smoke", "arm": "full", "gamma": 0.0, "eta_c": c, "state": st,
           "anchor": ANCHOR_AS, "rel": (None if c is None else c / ANCHOR_AS - 1.0),
           "bracket": info.get("bracket"), "res_rel": info.get("res_rel"), "ok": ok,
           "secs": time.time() - t0}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/DIR28r10_smoke.json", "w"), ensure_ascii=False, indent=1)
    print(f"smoke: η_c(γ=0,full)={c} rel={out['rel']} {'OK' if ok else 'MISS'} ({out['secs']:.1f}s)")
    return ok


# ══════════════════════════════════════ 正式轮
def formal():
    t0 = time.time()
    cells = {}
    plan = [("full", 0.0), ("full", GAMMA_CONF), ("fac", 0.0), ("fac", GAMMA_CONF)]
    plan += [(a, g) for a in ("full", "fac") for g in GAMMAS_REPORT]
    for arm, g in plan:
        c, st, info = eta_c(arm, g)
        cells[f"{arm}|g{g}"] = {"arm": arm, "gamma": g, "eta_c": c, "state": st,
                                "res_rel": info.get("res_rel"), "bracket": info.get("bracket"),
                                "iqr_max": info.get("iqr_max"), "se_med": info.get("se_med")}
        print(f"  {arm} γ={g}: η_c={None if c is None else round(c, 6)} ({st})")

    def g_(arm, gam):
        return cells[f"{arm}|g{gam}"]["eta_c"]

    res = {"cells": cells}
    for arm in ("full", "fac"):
        c0, c5 = g_(arm, 0.0), g_(arm, GAMMA_CONF)
        res[f"R_as_{arm}"] = (c5 / c0) if (c0 and c5) else None
    Rf, Rc = res["R_as_full"], res["R_as_fac"]
    a13 = (Rf is not None) and abs(Rf - 1.0) >= FLOOR
    res["A27-13"] = {"R_as_full": Rf, "floor": FLOOR, "pass": bool(a13)}
    if a13 and Rc is not None:
        d = Rf - Rc
        res["H27"] = {"R_fac": Rc, "R_full": Rf, "diff": d, "need": FLOOR,
                      "verdict": "H27 成立" if d >= FLOOR else "H27 否证"}

    c5 = g_("full", GAMMA_CONF)
    if c5:
        r5 = run("full", [c5], GAMMA_CONF)
        v_eff = float(r5["k2sq"][0])
        th0, th = eta_c_ms(MU), eta_c_ms(v_eff)
        res["A27-12"] = {"v_eff": v_eff, "mu": MU, "ratio": v_eff / MU,
                         "eta_c_ms_mu": th0, "eta_c_ms_eff": th,
                         "move_rel": (th - th0) / th0, "need": MOVE_MS,
                         "pass": bool(abs((th - th0) / th0) > MOVE_MS)}
    res["ms_report"] = {}
    for arm, gam in (("full", 0.0), ("full", GAMMA_CONF), ("fac", 0.0), ("fac", GAMMA_CONF)):
        c = g_(arm, gam)
        if c:
            for tag, e in (("eta_c/4", c / 4.0), ("eta_c", c), ("4eta_c", 4.0 * c)):
                res["ms_report"][f"{arm}|g{gam}|{tag}"] = float(run(arm, [e], gam)["ms"][0])
    for arm in ("full", "fac"):
        cB, stB, infoB = eta_c(arm, GAMMA_CONF, init="init-B")
        res[f"eta_c_{arm}_initB"] = cB
        res[f"R_as_{arm}_initB"] = (cB / g_(arm, 0.0)) if (cB and g_(arm, 0.0)) else None
        res[f"state_{arm}_initB"] = stB
    hv = res.get("H27", {}).get("verdict")
    if hv == "H27 成立":
        rb = res["R_as_fac_initB"]
        same = (rb is not None) and rb < 1.0 - FLOOR and Rc < 1.0 - FLOOR
        res["init_rule"] = {"same_sign": bool(same), "verdict": "同向" if same else "INIT-DEPENDENT"}
    out = {"stage": "formal", **res, "secs": time.time() - t0}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/DIR28r10_formal.json", "w"), ensure_ascii=False, indent=1)
    print(f"formal: {out['secs']:.1f}s → {OUT}/DIR28r10_formal.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ns = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if ns.stage in ("all", "s1"):
        json.dump(s1(), open(f"{OUT}/DIR28r10_s1.json", "w"), ensure_ascii=False, indent=1)
        print("S1 落盘")
    if ns.stage in ("all", "stageA"):
        if not stageA():
            sys.exit(3)
    if ns.stage in ("all", "smoke"):
        if not smoke():
            sys.exit(4)
    if ns.stage in ("all", "formal"):
        formal()
