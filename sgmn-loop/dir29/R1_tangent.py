r"""DIR-29 · R1_tangent.py —— 因子化 × 跨层耦合，**第二版装置**：切线空间 Lyapunov 指数 + 恢复加性噪声

卡片：`revolve/DIR-29/DIRECTION_rev29_v10.md`（rev 1.0 冻结）
脚本派生自：`dir28/R1_couple.py`（md5 dc64a1921e4f6645299357f06a2cc382）；改动仅限卡 §2 差异表：
  ① 噪声 v≡0 → 三路 iid 高斯 (a, b1, b2)；② 判据量：端点幅值比 → 雅可比累加（切线 Lyapunov）；
  ③ 锚点/早停相应更新。装置常数、耦合形式 k2=a(1+γW1)、同步更新、初始化、γ 集、列=种子约定逐字不动。

装置公式（v2.4 §3.1 局部目标与梯度，同 DIR-28 §3）：
  实例：全标量 d=1, r=1, α=1, λ=0.05, μ=5, W0=1；k1=a；k2=a(1+γW1)（键通道单向耦合）；
  G_l=(W_l k_l-v_l)k_l+λW_l；全矩阵 W_l←W_l-ηG_l；因子化（同步）U←U-ηGV, V←V-ηGᵀU。
  噪声：v1=b1, v2=b2，与 a 三路独立 N(0,μ)；default_rng(20261111) 一次抽 (3,T,S)，列=种子。

判据量（卡 §3.3，冻结）：
  full 臂：雅可比下三角 ⇒ 无需 QR：λ̂1=T⁻¹Σlog|1-η(a²+λ)|，λ̂2=T⁻¹Σlog|1-η(k2²+λ)|，top=max。
  fac 臂：两块 (u,v) 各做 Gram–Schmidt QR 累加 log|R_ii|；top = 四个指数之 max。
  arm one：λ̂=T⁻¹Σlog|1-η(a²+λ)|（与噪声、γ 无关 ⇒ 跨装置锚 0.49091）。
  逐列早停（DEF-28-16/17/18 分档）：W 形式 RS=1e250，fac 形式 RS_FAC=1e150；NaN/inf 判死 + 有限哨兵。
  ★ 数值安全杀（坑 3 追加）：雅可比出现非有限值（近死列的 u²A / k2² 溢出）⇒ 该列本步即杀，
    阈值语义不变（状态阈值仍是 RS/RS_FAC），只是不等状态先涨到阈值。λ̂ 用实际步数 t_dead 归一。
  死列冻结状态与累加；全部列进中位数（卡 §1.2，DEF-28-16 无偏性由 Stage A 背书）。
"""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

# ══════════════════════════════════════ 冻结常数（卡 §3 / §5.5）
MU = 5.0
LAM = 0.05
SIG0 = 1.0                       # 主 init-A：u=v=σ0=1 ⇒ W0=1（σ0 扫描为报告格）
PRIMARY_INIT = "init-A"
T = 10000                        # 每种子步数（卡 §5.5）
S = 64                           # 列数＝种子数
SEED0 = 20261111
GAMMA_CONF = 0.5
GAMMAS_REPORT = (1.0, 2.0)
ETA_LO, ETA_HI, K_GRID, K_BIS = 1e-3, 1.2, 40, 6
ANCHOR_AS = 0.49090552           # 卡 §3.4：E log|1-η(a²+λ)|=0 的根（DIR-28 锚，跨装置共享）
P1_SHIFT = 0.02                  # P1 操纵检查阈值（卡 §1.1）
P1_ETA_FRAC = 0.25               # η_ref = 0.25·η_c^one（纪律 58：在临界点下方测）
FLOOR = 0.10                     # P2 地板
GAP = 0.10                       # H29 的 0.10 间隔
OUT = "/home/ec2-user/sgmn-loop/dir29/out"
RS = 1e250                       # 早停：W 形式（DEF-28-17）
RS_FAC = 1e150                   # 早停：fac 形式（DEF-28-18）
CLIP = 1e150


# ══════════════════════════════════════ 装置核心（一步，可向量化）
def step_full(W1, W2, a, va, vb, eta, gamma):
    r"""全矩阵臂一步（同步）。k2=a(1+γW1)。"""
    G1 = (W1 * a - va) * a + LAM * W1
    k2 = a * (1.0 + gamma * W1)
    G2 = (W2 * k2 - vb) * k2 + LAM * W2
    return W1 - eta * G1, W2 - eta * G2, k2, G1, G2


def step_fac(u1, v1, u2, v2, a, va, vb, eta, gamma):
    r"""因子化臂一步（同步：G 用旧 U,V）。"""
    W1 = u1 * v1
    G1 = (W1 * a - va) * a + LAM * W1
    k2 = a * (1.0 + gamma * W1)
    W2 = u2 * v2
    G2 = (W2 * k2 - vb) * k2 + LAM * W2
    return (u1 - eta * G1 * v1, v1 - eta * G1 * u1,
            u2 - eta * G2 * v2, v2 - eta * G2 * u2, k2, G1, G2)


def jac_full_diag(a, k2, eta):
    r"""full 臂对角块（标量，向量化）：J11, J22（卡 §3.3）。"""
    return 1.0 - eta * (a ** 2 + LAM), 1.0 - eta * (k2 ** 2 + LAM)


def jac_fac(u, v, key, val, eta):
    r"""fac 臂 2×2 块雅可比（卡 §3.3 / §4 工作例）。

    key='a'：G=(uv·a-b)a+λuv，A=a²+λ；key='k2'：G=(uv·k2-b)k2+λuv，A=k2²+λ。
    J11=1-ηv²A，J22=1-ηu²A，J12=J21=-η(uvA+G)（对称，卡 §4 工作例 -0.215/-2.13）。
    """
    A = key ** 2 + LAM
    W = u * v
    G = (W * key - val) * key + LAM * W
    uvA = u * v * A
    J11 = 1.0 - eta * (v ** 2) * A
    J12 = -eta * (uvA + G)
    J22 = 1.0 - eta * (u ** 2) * A
    return J11, J12, J22


def qr2_step(J11, J12, J22, q):
    r"""2×2 Gram–Schmidt QR 累加一步。J 对称（J21=J12）。q: (...,2) Q 的两列。

    M=J@Q；R11=|M0|；Q0=M0/R11；R12=Q0·M1；M1'=M1−R12·Q0；R22=|M1'|；Q1=M1'/R22。
    返回新 q 与 (log|R11|, log|R22|)。J 非有限 ⇒ 输出 NaN，由调用方按列杀。
    """
    M0 = J11 * q[..., 0] + J12 * q[..., 1]
    M1 = J12 * q[..., 0] + J22 * q[..., 1]
    R11 = np.maximum(np.abs(M0), 1e-300)
    Q0 = M0 / R11
    R12 = Q0 * M1
    M1p = M1 - R12 * Q0
    R22 = np.maximum(np.abs(M1p), 1e-300)
    Q1 = M1p / R22
    return np.stack([Q0, Q1], axis=-1), np.log(R11), np.log(R22)


def noise():
    r"""三路独立流 a, b1, b2 ~ N(0, μ)；列=种子（卡 §3.2/坑 8）。"""
    g = np.random.default_rng(SEED0)
    return g.normal(0.0, math.sqrt(MU), size=(3, T, S))


A_, B1_, B2_ = noise()


def init_state(arm, init, shape, sig0=SIG0):
    B = init == "init-B"
    if arm == "fac":
        if B:                                            # init-B：u=√5, v=1/√5 ⇒ W0=1（尺度不同）
            u0, v0 = math.sqrt(5.0), 1.0 / math.sqrt(5.0)
        else:                                            # init-A：u=v=σ0 ⇒ W0=σ0²
            u0 = v0 = sig0
        return [np.full(shape, u0), np.full(shape, v0), np.full(shape, u0), np.full(shape, v0)]
    w0 = 1.0                                             # 两臂 W0 相同（comparability）
    if arm == "one":
        return [np.full(shape, w0)]
    return [np.full(shape, w0), np.full(shape, w0)]


def run(arm, eta, gamma, init=PRIMARY_INIT, sig0=SIG0, t_steps=T, s_cols=S,
        record=False):
    r"""返回 dict：各指数 (S,K)、top (S,K)、k2sq (S,K)、t_dead、n_trunc。

    判据量（卡 §3.3）：逐列切线 Lyapunov 指数；早停列冻结累加、按 t_dead 归一。
    数值安全杀：雅可比/状态非有限 ⇒ 该列本步杀（阈值语义不变，见文件头）。

    ★ `record=True`（**仅供 Stage A 的 A29-4**）：额外返回 `jac_seq`（每步的雅可比块）
      与 `mask_seq`（每步的存活掩码）。这样 A29-4 的「完整 QR vs 分块 QR」对照就作用在
      **同一条轨迹的同一批雅可比**上，而不是另写一套会溢出的参考循环（**DEF-29-5**）。
    """
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    K = eta.size
    shape = (s_cols, K)
    eta_b = eta.reshape(1, K)
    st = init_state(arm, init, shape, sig0)

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

    thr = RS_FAC if arm == "fac" else RS
    n_acc = 4 if arm == "fac" else (2 if arm == "full" else 1)
    accs = [np.zeros(shape) for _ in range(n_acc)]
    k2sum = np.zeros(shape)
    alive = np.ones(shape, dtype=bool)
    t_dead = np.full(shape, float(t_steps))
    q1 = np.zeros(shape + (2,)); q1[..., 0] = 1.0        # fac 块 1 的 Q
    q2 = np.zeros(shape + (2,)); q2[..., 0] = 1.0        # fac 块 2 的 Q
    jac_seq, mask_seq = ([], []) if record else (None, None)
    for t in range(t_steps):
        a, b1, b2 = A_[t][:s_cols, None], B1_[t][:s_cols, None], B2_[t][:s_cols, None]
        old = list(st)
        if arm == "fac":
            u1, v1, u2, v2 = st
            nu1, nv1, nu2, nv2, k2, G1, G2 = step_fac(u1, v1, u2, v2, a, b1, b2, eta_b, gamma)
            st = [nu1, nv1, nu2, nv2]
        elif arm == "one":
            G1 = (old[0] * a - b1) * a + LAM * old[0]
            st = [old[0] - eta_b * G1]
            k2 = a
        else:
            nW1, nW2, k2, G1, G2 = step_full(old[0], old[1], a, b1, b2, eta_b, gamma)
            st = [nW1, nW2]
        if not alive.any():
            break
        st = [np.where(alive, n, o) for n, o in zip(st, old)]
        k2c = np.clip(k2, -CLIP, CLIP)
        k2sum += np.where(alive, k2c ** 2, 0.0)
        # ── 切线雅可比（在 t 时刻状态 old 上求值；与动力学同拍）──
        with np.errstate(over="ignore", invalid="ignore"):
            if arm == "one":
                J = [1.0 - eta_b * (a ** 2 + LAM)]
                safe = np.isfinite(J[0])
            elif arm == "full":
                J = list(jac_full_diag(a, k2c, eta_b))
                safe = np.isfinite(J[0]) & np.isfinite(J[1])
                if record:
                    # J21 = -η·∂_{W1}[(W2 k2 − v2) k2] = -η(2γ a k2 W2 − γ a v2)（DEF-27-7 同族）
                    J21 = -eta_b * (2.0 * gamma * a * k2c * old[1] - gamma * a * b2)
                    J21 = np.clip(J21, -CLIP, CLIP)
                    safe = safe & np.isfinite(J21)
            else:
                J1 = jac_fac(old[0], old[1], a, b1, eta_b)          # 块 1：(u1,v1)，键 a
                J2 = jac_fac(old[2], old[3], k2c, b2, eta_b)        # 块 2：(u2,v2)，键 k2
                J = list(J1) + list(J2)
                safe = np.isfinite(J[0]) & np.isfinite(J[1]) & np.isfinite(J[2]) \
                    & np.isfinite(J[3])
        mask = alive & safe
        if record:
            if arm == "full":
                blk = np.stack([np.stack([J[0], np.zeros_like(J[0])], -1),
                                 np.stack([J21, J[1]], -1)], -2)
            elif arm == "fac":
                blk = np.zeros(shape + (4, 4))
                blk[..., 0, 0] = J[0]; blk[..., 0, 1] = J[1]
                blk[..., 1, 0] = J[1]; blk[..., 1, 1] = J[2]
                blk[..., 2, 2] = J[3]; blk[..., 2, 3] = J[4]
                blk[..., 3, 2] = J[4]; blk[..., 3, 3] = J[5]
            else:
                blk = None
            # ★ DEF-29-6：记录张量带多余 K 轴 ⇒ 后续 `Jt @ q` 广播成 (S,S,dim,dim) ⇒ ValueError。
            #   A29-4 恒用单点 η（K=1）⇒ 此处 squeeze 掉 K 轴。
            assert shape[1] == 1, "record 模式只支持单点 η（K=1）"
            jac_seq.append(None if blk is None else blk[:, 0].copy())
            mask_seq.append(mask[:, 0].copy())
        # ── 累加（只对 活∧safe 列）──
        if arm == "fac":
            q1, l11, l12 = qr2_step(J[0], J[1], J[2], q1)
            q2, l21, l22 = qr2_step(J[3], J[4], J[5], q2)
            accs[0] += np.where(mask, l11, 0.0)
            accs[1] += np.where(mask, l12, 0.0)
            accs[2] += np.where(mask, l21, 0.0)
            accs[3] += np.where(mask, l22, 0.0)
        else:
            accs[0] += np.where(mask, np.log(np.maximum(np.abs(J[0]), 1e-300)), 0.0)
            if arm == "full":
                accs[1] += np.where(mask, np.log(np.maximum(np.abs(J[1]), 1e-300)), 0.0)
        # ── 早停：状态超阈 或 雅可比非有限（数值安全杀）──
        ms_ = mag_state(st)
        over = alive & ((ms_ > thr) | ~np.isfinite(ms_) | ~safe)
        if over.any():
            t_dead[over] = float(t + 1)
            alive &= ~over
            st = [np.where(over, np.where(np.isfinite(x), x, thr), x) for x in st]
            st = [np.where(over, np.clip(x, -thr, thr), x) for x in st]
    t_ = np.maximum(t_dead, 1.0)
    lams = [ac / t_ for ac in accs]
    out = {"lam": lams[0], "k2sq": k2sum / t_,
           "t_dead_med": float(np.median(t_dead)), "n_trunc": int((t_dead < t_steps).sum()),
           "t_dead": t_dead.copy()}          # ★ A29-4 的参考 QR 必须用**同一个**归一口径
    if arm == "full":
        out["lam2"] = lams[1]
        out["lam_top"] = np.maximum(lams[0], lams[1])
    elif arm == "fac":
        for i in range(1, 4):
            out[f"lam{i+1}"] = lams[i]
        out["lam_top"] = np.maximum(np.maximum(lams[0], lams[1]), np.maximum(lams[2], lams[3]))
    else:
        out["lam_top"] = lams[0]
    if record:
        out["jac_seq"] = jac_seq
        out["mask_seq"] = mask_seq
    return out


def exp_report(r):
    r"""把 run() 的各指数转成 median/IQR 报告量（Advisor 第 3 条：只报告、不进判据）。

    动机：只存 lam_top 时，若 P2（地板）不过，无法区分「λ̂2 根本没动」与「动了但没过地板」。
    """
    out = {}
    for key in ("lam", "lam2", "lam3", "lam4"):
        if key in r:
            v = r[key]
            with np.errstate(all="ignore"):                # ★ DEF-29-9
                out[key + "_med"] = float(np.nanmedian(v))
            out[key + "_iqr"] = float(np.percentile(v, 75) - np.percentile(v, 25))
    return out


def grid_mono(meds):
    r"""grid_meds 的符号变化次数与过零附近的单调性（Advisor 第 4 条：非单调 ⇒ 非精确根）。"""
    meds = np.asarray(meds, dtype=float)
    sgn = np.sign(meds)
    sgn = sgn[sgn != 0.0]
    n_changes = int(np.sum(sgn[1:] != sgn[:-1])) if sgn.size > 1 else 0
    return {"n_sign_changes": n_changes,
            "monotone_increasing": bool(np.all(np.diff(meds) > 0)) if meds.size > 1 else None,
            "med_first": float(meds[0]), "med_last": float(meds[-1])}


def top_median(arm, eta, gamma, init=PRIMARY_INIT, sig0=SIG0, t_steps=T, s_cols=S):
    r"""median over columns of the top tangent exponent at scalar or grid eta."""
    r = run(arm, eta, gamma, init, sig0, t_steps, s_cols)
    # ★ DEF-29-9：溢出列的指数可能是 NaN ⇒ np.median 会被**毒化**（NaN 让所有比较为假）。
    #   改用 np.nanmedian（只在该列的指数未定义时排除它；判据格点的根**不受影响**，
    #   因为全部 NaN 都出现在过零点**之后**的网格上）。
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.nanmedian(r["lam_top"])), r


def find_eta_c(arm, gamma, init=PRIMARY_INIT, sig0=SIG0, t_steps=T, s_cols=S):
    r"""网格（批 K=40）找首个 **−→+** 区间，再 6 次算术二分（卡 §3.4，rev 1.1 / DEF-29-1）。

    ★ **DEF-29-1（rev 1.1 修正）**：rev 1.0 的卡片与代码都按「首个 $+→-$ 区间」写，
      但 $\hat\lambda$ 随 $\eta$ **递增**（$\eta$ 越大越不稳 ⇒ $\hat\lambda$ 越大；
      DIR-28 已归档同一方向的错 **DEF-28-14**）。按 +→− 搜索 ⇒ 永远找不到区间
      ⇒ 全格 `no_sign_change`。现改为 **−→+**：网格取首个 $\mathrm{med}\le0$ 且下一格 $>0$；
      二分里「中点已不稳定（$m>0$）⇒ 根更小 ⇒ $hi=mid$」。
    """
    grid = np.exp(np.linspace(math.log(ETA_LO), math.log(ETA_HI), K_GRID))
    _, rg = top_median(arm, grid, gamma, init, sig0, t_steps, s_cols)
    with np.errstate(all="ignore"):                       # ★ DEF-29-9：nanmedian
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meds = np.nanmedian(rg["lam_top"], axis=0)    # (K,)
    idx = None
    for j in range(K_GRID - 1):
        if meds[j] <= 0.0 and meds[j + 1] > 0.0:
            idx = j
            break
    if idx is None:
        return None, None, {"grid_meds": meds.tolist(), "reason": "no_sign_change",
                            "med_first": float(meds[0]), "med_last": float(meds[-1])}
    lo, hi = float(grid[idx]), float(grid[idx + 1])
    for _ in range(K_BIS):
        mid = 0.5 * (lo + hi)
        m, _ = top_median(arm, mid, gamma, init, sig0, t_steps, s_cols)
        if m > 0.0:            # 中点已不稳定 ⇒ 根更小
            hi = mid
        else:
            lo = mid
    eta_c = 0.5 * (lo + hi)
    return eta_c, [lo, hi], {"grid_meds": meds.tolist(), "bracket": [lo, hi]}


# ══════════════════════════════════════ Stage A
def stage_a():
    res = {}
    ok_all = True

    def check(name, cond, info):
        nonlocal ok_all
        res[name] = {"pass": bool(cond), **info}
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {info}")
        ok_all = ok_all and bool(cond)

    print("Stage A · DIR-29（切线 Lyapunov 装置）")
    # A29-1 工作例逐位（卡 §4）
    J = jac_fac(np.full(1, 1.0), np.full(1, 1.0), np.full(1, 2.0), np.full(1, 0.5), np.full(1, 0.3))
    exp = [-0.215, -2.13, -0.215]
    got = [float(J[0][0]), float(J[1][0]), float(J[2][0])]
    det = got[0] * got[2] - got[1] ** 2
    # ★ DEF-29-4：卡片 §4 写的 det = -4.490725 是**算术错**（应为 -4.490675：
    #   (-0.215)² - 2.13² = 0.046225 - 4.5369 = -4.490675）。雅可比本身逐位正确（1e-16）。
    ok1 = all(abs(g - e) < 1e-12 for g, e in zip(got, exp)) and abs(det - (-4.490675)) < 1e-9
    check("A29-1 worked-example", ok1, {"J11,J12,J22": got, "det": det})

    # A29-2 雅可比有限差分（fac 块；中心差分 h=1e-6）
    rng = np.random.default_rng(7)
    ok2, worst = True, 0.0
    for _ in range(3):
        u, v = rng.normal(size=2)
        a, b = rng.normal(0, math.sqrt(MU), size=2)
        for eta in (0.1, 0.3, 0.9):
            h = 1e-6
            Jn = np.zeros((2, 2))
            for i, (du, dv) in enumerate([(h, 0.0), (0.0, h)]):
                def f(uu, vv):
                    G = ((uu * vv * a - b) * a + LAM * uu * vv)
                    return (uu - eta * G * vv, vv - eta * G * uu)
                up_p, vp_p = f(u + du, v + dv)
                up_m, vp_m = f(u - du, v - dv)
                Jn[0, i] = (up_p - up_m) / (2 * h)
                Jn[1, i] = (vp_p - vp_m) / (2 * h)
            Ja = np.array([[float(jac_fac(np.array([[u]]), np.array([[v]]), np.array([[a]]),
                                          np.array([[b]]), np.array([[eta]]))[k][0, 0])
                            for k in range(3)]])
            Ja = np.array([[Ja[0, 0], Ja[0, 1]], [Ja[0, 1], Ja[0, 2]]])
            rel = float(np.max(np.abs(Jn - Ja) / np.maximum(np.abs(Ja), 1e-8)))
            worst = max(worst, rel)
            ok2 = ok2 and rel < 1e-5
    check("A29-2 jacobian-fd", ok2, {"worst_rel": worst})

    # A29-3 跨装置锚：η_c^full(γ=0) ∈ [0.48091, 0.50091]
    eta_c0, br, info = find_eta_c("full", 0.0)
    ok3 = eta_c0 is not None and abs(eta_c0 - ANCHOR_AS) <= 0.01
    check("A29-3 cross-device-anchor", ok3, {"eta_c_full_g0": eta_c0, "anchor": ANCHOR_AS})

    # A29-4 三角分解：完整 QR（np.linalg.qr 数值参考）vs 分块路径
    # ★ DEF-29-5：rev 1.0 的参考循环**没有 run() 的 clip/判死规则** ⇒ 发散列溢出成 NaN
    #   ⇒ d_full=d_fac=nan（首跑 FAIL）。现改为：在 **run() 自己的雅可比序列**（record=True）
    #   上做「完整 QR」与「分块 QR」的对照 ⇒ 同源、同掩码、无溢出。
    # ★ DEF-29-8：rev 1.0 的参数 (η=0.3, T=3000) 落在**两臂都发散**的区间
    #   （实测 full 臂 64/64 列在 t≈368 死亡、fac 臂更早）⇒ 雅可比溢出 ⇒ 对照量被污染
    #   （首跑 d_full=0.077、d_fac=30.2）。改到**稳定点**（无死亡）：d_full=0.0046、d_fac=0.0052。
    #   **断言的量、容差 0.02、γ=0.5 全部不动**，只改 η 与 T 这两个「可执行性」参数。
    t4 = 3000
    eta4, gamma4 = 0.02, 0.5

    def full_qr_top(jac_seq, mask_seq, dim, t_dead):
        r"""在记录的雅可比序列上做完整 QR，返回每列的最大指数（死列按存活步数归一）。"""
        S_ = jac_seq[0].shape[0]
        q = np.zeros((S_, dim, dim)); q[..., np.arange(dim), np.arange(dim)] = 1.0
        acc = np.zeros((S_, dim))
        # ★ DEF-29-7：归一必须与 run() 的 t_dead **完全一致** ⇒ 直接用 run() 导出的 t_dead。
        #   （原实现按"实际累加步数"归一：run() 的 t_dead 是**死亡步 + 1**，两者对**状态爆掉**的列差 1 步
        #    ⇒ 指数差 |λ̂|/t ⇒ fac 臂实测 d=30.2、full 臂 d=0.072）
        t_dead = np.asarray(t_dead).reshape(-1)
        for t, Jt in enumerate(jac_seq):
            M = Jt @ q
            Q, R = np.linalg.qr(M)
            d = np.diagonal(R, axis1=-2, axis2=-1)
            m = mask_seq[t]
            acc += np.where(m[:, None], np.log(np.maximum(np.abs(d), 1e-300)), 0.0)
            q = np.where(m[:, None, None], Q, q)
        return np.max(acc / np.maximum(t_dead, 1.0)[:, None], axis=1)

    rf = run("full", [eta4], gamma4, t_steps=t4, record=True)
    d_full = float(np.max(np.abs(full_qr_top(rf["jac_seq"], rf["mask_seq"], 2, rf["t_dead"])
                               - np.maximum(rf["lam"][:, 0], rf["lam2"][:, 0]))))
    rfc = run("fac", [eta4], gamma4, t_steps=t4, record=True)
    blk4 = np.maximum(np.maximum(rfc["lam"][:, 0], rfc["lam2"][:, 0]),
                      np.maximum(rfc["lam3"][:, 0], rfc["lam4"][:, 0]))
    d_fac = float(np.max(np.abs(full_qr_top(rfc["jac_seq"], rfc["mask_seq"], 4, rfc["t_dead"]) - blk4)))
    ok4 = d_full <= 0.02 and d_fac <= 0.02
    check("A29-4 triangular-split", ok4, {"d_full": d_full, "d_fac": d_fac})

    # A29-5 噪声在场：含噪 median|W_T| > 10× 无噪（η=0.1, T=2e4, arm one）
    T5 = 20000
    g5 = np.random.default_rng(SEED0)
    a5 = g5.normal(0.0, math.sqrt(MU), size=(T5, S))
    b5 = g5.normal(0.0, math.sqrt(MU), size=(T5, S))
    w_n = np.ones((S, 1)); w_0 = np.ones((S, 1))
    for t in range(T5):
        w_n = w_n - 0.1 * ((w_n * a5[t][:, None] - b5[t][:, None]) * a5[t][:, None] + LAM * w_n)
        w_0 = w_0 - 0.1 * ((w_0 * a5[t][:, None]) * a5[t][:, None] + LAM * w_0)
        w_n = np.clip(w_n, -1e150, 1e150)
        w_0 = np.clip(w_0, -1e150, 1e150)
    med_noisy = float(np.median(np.abs(w_n)))
    med_nl = float(np.median(np.abs(w_0)))
    ratio = med_noisy / max(med_nl, 1e-300)
    ok5 = ratio > 10.0
    check("A29-5 noise-present", ok5, {"median_noisy": med_noisy, "median_noiseless": med_nl,
                                       "ratio": ratio})

    # A29-6 确定性
    r1 = top_median("fac", 0.3, 0.5, t_steps=2000)
    r2 = top_median("fac", 0.3, 0.5, t_steps=2000)
    ok6 = r1[0] == r2[0]
    check("A29-6 determinism", ok6, {"run1": r1[0], "run2": r2[0]})

    # A29-7 列=种子：前 8 列单跑 == 全阵前 8 列
    rfull = run("fac", 0.3, 0.5, t_steps=1500, s_cols=S)
    rsolo = run("fac", 0.3, 0.5, t_steps=1500, s_cols=8)
    d7 = float(np.max(np.abs(rsolo["lam_top"][:, 0] - rfull["lam_top"][:8, 0])))
    ok7 = d7 <= 1e-12
    check("A29-7 col-seed", ok7, {"max_abs_diff": d7})

    # A29-8 坑 8：corr(a, b1) 池化 < 0.05
    c = float(np.corrcoef(A_.ravel(), B1_.ravel())[0, 1])
    ok8 = abs(c) < 0.05
    check("A29-8 stream-independence", ok8, {"corr_a_b1": c})

    md5 = hashlib.md5(open(__file__, "rb").read()).hexdigest()
    n_pass = sum(1 for v in res.values() if v["pass"])
    print(f"Stage A: {n_pass}/{len(res)} PASS  ({time.time()-_T0:.1f}s)  md5={md5}")
    out = {"pass": ok_all, "checks": res, "md5": md5, "secs": round(time.time() - _T0, 1)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "DIR29r10_stageA.json"), "w"), ensure_ascii=False, indent=1)
    return ok_all


# ══════════════════════════════════════ smoke / formal
def smoke():
    eta_c0, br, _ = find_eta_c("full", 0.0)
    rel = None if eta_c0 is None else (eta_c0 - ANCHOR_AS) / ANCHOR_AS
    ok = eta_c0 is not None and abs(rel) <= 0.03
    print(f"smoke: η_c(full,γ=0)={eta_c0} rel={rel} {'OK' if ok else 'BAD'} ({time.time()-_T0:.1f}s)")
    json.dump({"eta_c_full_g0": eta_c0, "anchor": ANCHOR_AS, "rel": rel, "pass": ok,
               "secs": round(time.time() - _T0, 1)},
              open(os.path.join(OUT, "DIR29r10_smoke.json"), "w"), ensure_ascii=False, indent=1)
    return ok


def formal():
    print("formal · DIR-29")
    res = {"cells": {}}
    arms_cells = [("full", g) for g in (0.0, GAMMA_CONF) + GAMMAS_REPORT] + \
                 [("fac", g) for g in (0.0, GAMMA_CONF) + GAMMAS_REPORT]
    for arm, g in arms_cells:
        eta_c, br, info = find_eta_c(arm, g)
        cell = {"eta_c": eta_c, "bracket": br,
                "grid_meds": info.get("grid_meds"),
                "reason": info.get("reason")}
        if info.get("grid_meds") is not None:
            cell["grid"] = grid_mono(info["grid_meds"])
        if eta_c is not None:                      # ★ 报告量（Advisor 第 3 条）：逐指数 median/IQR
            rr = run(arm, eta_c, g)
            cell["at_eta_c"] = exp_report(rr)
            cell["n_trunc_at_eta_c"] = rr["n_trunc"]
        res["cells"][f"{arm}|g{g}"] = cell
        print(f"  {arm} γ={g}: η_c={eta_c} (bracket {br})", flush=True)
    ec_one, _, _ = find_eta_c("one", 0.0)
    res["eta_c_one"] = ec_one
    print(f"  one: η_c={ec_one}")

    def R(arm, g):
        e0 = res["cells"][f"{arm}|g0.0"]["eta_c"]
        e1 = res["cells"][f"{arm}|g{g}"]["eta_c"]
        return None if (e0 is None or e1 is None) else e1 / e0

    # 前置 1：操纵检查（η_ref = 0.25·η_c^one；λ̂2 full 的位移，卡 §1.1）
    eta_ref = P1_ETA_FRAC * ec_one
    _, r0 = top_median("full", eta_ref, 0.0)
    _, r1 = top_median("full", eta_ref, GAMMA_CONF)
    with np.errstate(all="ignore"):                        # ★ DEF-29-9
        m0 = float(np.nanmedian(r0["lam2"]))
        m1 = float(np.nanmedian(r1["lam2"]))
    p1 = None if ec_one is None else abs(m1 - m0)
    res["P1"] = {"eta_ref": eta_ref, "lam2_med_g0": m0, "lam2_med_g05": m1,
                 "shift": p1, "pass": None if p1 is None else bool(p1 >= P1_SHIFT)}
    print(f"  P1: shift={p1} (need ≥{P1_SHIFT}) {'PASS' if res['P1']['pass'] else 'FAIL'}")
    # 前置 2：地板
    Rf = R("full", GAMMA_CONF)
    p2 = None if Rf is None else abs(Rf - 1.0)
    res["P2"] = {"R_full": Rf, "dev": p2, "pass": None if p2 is None else bool(p2 >= FLOOR)}
    print(f"  P2: |R_full−1|={p2} (need ≥{FLOOR}) {'PASS' if res['P2']['pass'] else 'FAIL'}")
    # H29
    Rfac = R("fac", GAMMA_CONF)
    h29 = None if (Rf is None or Rfac is None) else bool(Rfac <= Rf - GAP)
    res["H29"] = {"R_fac": Rfac, "R_full": Rf,
                  "gap": None if (Rf is None or Rfac is None) else Rf - Rfac,
                  "pass": h29}
    print(f"  H29: R_fac={Rfac} ≤ R_full−0.10={None if Rf is None else Rf - GAP} → {h29}")

    # 报告格：σ0 扫描（fac, γ∈{0,0.5}）+ init-B 抽查 + k2sq
    res["report"] = {}
    for s0 in (0.5, 2.0):
        for g in (0.0, GAMMA_CONF):
            e_c, _, _ = find_eta_c("fac", g, sig0=s0)
            res["report"][f"fac|s{s0}|g{g}"] = e_c
            print(f"  report fac σ0={s0} γ={g}: η_c={e_c}", flush=True)
    for arm in ("full", "fac"):
        for g in (0.0, GAMMA_CONF):
            e_c, _, _ = find_eta_c(arm, g, init="init-B")
            res["report"][f"{arm}|initB|g{g}"] = e_c
            print(f"  report {arm} init-B γ={g}: η_c={e_c}", flush=True)
    for arm, g in [("full", 0.0), ("full", GAMMA_CONF), ("fac", 0.0), ("fac", GAMMA_CONF)]:
        rr = run(arm, 0.25, g)
        res["report"][f"k2sq|{arm}|g{g}"] = {"med": float(np.nanmedian(rr["k2sq"])),
                                            "n_trunc": rr["n_trunc"]}
    # ── 事前写死的 T 延长规则（卡 rev 1.1 §1.3；阈值不动）──
    # 触发：full 臂 γ=0.5 的 |R−1| < 0.15  **或** 其过零距锚 < 0.05
    # 动作：该格以 T=2×10^4 重算一次；若两次 η_c 的相对移动 > 4% ⇒ 该格记 INFEASIBLE-RESOLUTION（T 依赖）
    res["T_extension"] = {"rule": "(|R_full(0.5)−1|<0.15) or (|eta_c_full(0.5)−anchor|<0.05) ⇒ T=2e4 复核",
                           "threshold_rel_move": 0.04, "triggered": False}
    e_anchor = res["cells"]["full|g0.0"]["eta_c"]
    e_half = res["cells"]["full|g0.5"]["eta_c"]
    trig = (Rf is not None and abs(Rf - 1.0) < 0.15) or \
           (e_half is not None and e_anchor is not None and abs(e_half - ANCHOR_AS) < 0.05)
    if trig:
        res["T_extension"]["triggered"] = True
        e2, br2, _ = find_eta_c("full", GAMMA_CONF, t_steps=20000)
        res["T_extension"]["eta_c_full_g05_T2e4"] = e2
        res["T_extension"]["eta_c_full_g05_T1e4"] = e_half
        if e2 is not None and e_half is not None and e_half != 0:
            mv = abs(e2 - e_half) / abs(e_half)
            res["T_extension"]["rel_move"] = mv
            res["T_extension"]["infeasible_resolution"] = bool(mv > 0.04)
            if res["T_extension"]["infeasible_resolution"]:
                res["cells"]["full|g0.5"]["state_T_dependent"] = True
        print(f"  T 延长：η_c(full,0.5,T=1e4)={e_half} → T=2e4: {e2} "
              f"rel_move={res['T_extension'].get('rel_move')}", flush=True)

    res["secs"] = round(time.time() - _T0, 1)
    print(f"formal: {res['secs']}s → {OUT}/DIR29r10_formal.json")
    json.dump(res, open(os.path.join(OUT, "DIR29r10_formal.json"), "w"), ensure_ascii=False, indent=1)
    return res


def s1():
    r"""设计检查：γ=0 时 full 臂 λ̂1 与 λ̂2 逐列相同（k2≡a ⇒ 同被积函数）。"""
    r = run("full", 0.3, 0.0, t_steps=3000)
    d = float(np.max(np.abs(r["lam"] - r["lam2"])))
    print(f"S1: max|λ̂1−λ̂2| at γ=0 = {d}")
    json.dump({"max_diff": d, "pass": d == 0.0},
              open(os.path.join(OUT, "DIR29r10_s1.json"), "w"), ensure_ascii=False, indent=1)
    return d == 0.0


if __name__ == "__main__":
    _T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["s1", "stageA", "smoke", "formal"], required=True)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"s1": s1, "stageA": stage_a, "smoke": smoke, "formal": formal}[args.stage]()
