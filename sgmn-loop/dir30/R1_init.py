r"""DIR-30 · R1_init.py —— **初始化作为一等公民**：因子化臂的操作域与耦合敏感性如何依赖初始化的两个尺度分量

卡片：`revolve/DIR-30/DIRECTION_rev30_v10.md`（rev 1.0 冻结）
脚本派生自：`dir29/R1_tangent.py`（md5 cb4992d25b42a0af2b2230fcfc6f36c1）；**动力学逐字不动**，
改动仅限卡 §2 差异表：
  ① 判据量：中位数后翻根 → **逐列过零阈值**（修掉 DIR-29 的悬崖；卡 §3.3）；
  ② 独立变量：耦合强度 → **初始化的两个尺度分量**（Axis A 偏斜 σ_f、Axis B 对称尺度 σ0）；
  ③ run() 允许**逐列 η**（形状 (S,1)），供逐列二分使用（测量层需要）；
  ④ Stage A 断言集换成 A30-1…A30-9（含两条**归档锚** A30-4 / A30-9）。

装置公式（v2.4 §3.1 局部目标与梯度，同 DIR-28/29 §3，**逐字不动**）：
  实例：全标量 d=1, r=1, α=1, λ=0.05, μ=5；k1=a；k2=a(1+γW1)（键通道单向耦合）；
  G_l=(W_l k_l-v_l)k_l+λW_l；全矩阵 W_l←W_l-ηG_l；因子化（同步）U←U-ηGV, V←V-ηGᵀU。
  噪声：v1=b1, v2=b2，与 a 三路独立 N(0,μ)；default_rng(20261111) 一次抽 (3,T,S)，列=种子。

判据量（卡 §3.3，冻结）：
  full 臂：雅可比下三角 ⇒ 无需 QR：λ̂1=T⁻¹Σlog|1-η(a²+λ)|，λ̂2=T⁻¹Σlog|1-η(k2²+λ)|，top=max。
  fac 臂：两块 (u,v) 各做 Gram–Schmidt QR 累加 log|R_ii|；top = 四个指数之 max。
  arm one：λ̂=T⁻¹Σlog|1-η(a²+λ)|（与噪声、γ 无关 ⇒ 跨装置锚 0.49091）。
  **逐列早停**（DEF-28-16/17/18 分档）：W 形式 RS=1e250，fac 形式 RS_FAC=1e150；NaN/inf 判死 + 有限哨兵。
  死列冻结状态与累加；全部列进统计（DEF-29-3/纪律 61：排除死列＝可选停止）。
  ★ **逐列过零阈值**（本卡新判据量）：每列各自找首个 −→+ 区间并各自二分 6 次 ⇒ η_j；
    臂阈值 = nanmedian(η_j)（纪律 60）；**删失列不排除**，按网格上沿 +1 格插补（卡 §3.3）。

"""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

# ★ DIR-30：fac 臂在门槛**之上**必然溢出（这正是要测的东西）⇒ 抑制 overflow/invalid 告警。
np.seterr(over="ignore", invalid="ignore", divide="ignore")

# ══════════════════════════════════════ 冻结常数（卡 §3 / §5.5）
MU = 5.0
LAM = 0.05
SIG0 = 1.0                       # 主 init-A：u=v=σ0=1 ⇒ W0=1（σ0 扫描为报告格）
PRIMARY_INIT = "init-A"
T = 10000                        # 每种子步数（卡 §5.5）
T_NOISE = 20000                  # ★ DEF-30-1：噪声数组步数（卡 §5 A30-3 的快二分需 T=2e4）
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
OUT = "/home/ec2-user/sgmn-loop/dir30/out"
RS = 1e250                       # 早停：W 形式（DEF-28-17）
RS_FAC = 1e150                   # 早停：fac 形式（DEF-28-18）
CLIP = 1e150
# ── DIR-30 新增（卡 §1/§3.2/§3.3/§5 全部冻结值）──
GRID_RATIO = 1.1993                 # 相邻网格比 e^{ln1200/39}（DIR-29 §3.4，同值）
CENSOR_IMPUTE = ETA_HI * GRID_RATIO  # 删失列插补 = 网格上沿 +1 格 = 1.43916
AMP_MIN = 0.15                      # H30 主判据：R 的振幅下界（卡 §1.1）
TOL_SELF = 0.05                     # A30-4/A30-9：与归档读数的相对差容差
AXIS_A = (0.25, 0.5, 1.0, 2.0, 4.0, math.sqrt(5.0))   # 偏斜轴 σ_f（W0≡1 固定）
AXIS_B = (0.5, 1.0, 2.0)            # 对称尺度轴 σ0（W0=σ0²）
SQRT5 = math.sqrt(5.0)
# DIR-29 归档读数（d29/out/DIR29r10_formal.json）—— A30-4 / A30-9 的**跨装置锚**
D29_FULL_G0 = 0.4933106             # full, γ=0（DIR-29 归档）
D29_FULL_G05 = 0.1879949            # full, γ=0.5（DIR-29 归档）
D29_FAC_S1_G0 = 0.1768623           # fac, σ0=1(init-A), γ=0
D29_FAC_S1_G05 = 0.1440873          # fac, σ0=1(init-A), γ=0.5
D29_INITB_G0 = 0.07391435204943869  # fac, init-B(u=√5,v=1/√5), γ=0
D29_INITB_G05 = 0.059757595748992066  # fac, init-B, γ=0.5


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
    r"""三路独立流 a, b1, b2 ~ N(0, μ)；列=种子（卡 §3.2/坑 8）。

    ★ DEF-30-1：噪声数组抽 **T_NOISE=2×10⁴** 步（卡 §5 A30-3 的快二分要 T=2e4，
      而原数组只有 T=1e4 ⇒ 越界 IndexError）。`default_rng` 按行填充 ⇒ **前 1e4 步逐位不变**
      ⇒ 正式轮（T=1e4）的噪声与 DIR-29 完全一致，跨装置可比性不受影响。
    """
    g = np.random.default_rng(SEED0)
    # ★ DEF-30-4（rev 1.3 修正 DEF-30-1）：**不能**直接抽 (3, T_NOISE, S) ——
    #   `default_rng.normal` 按 C 序填充，元素 (i,t,j) 的偏移是 i*T*S + t*S + j ⇒
    #   把第一维从 T 改成 T_NOISE 会**平移第 2、3 条流的起点** ⇒ 前 1e4 步**不再**与 DIR-29 相同
    #   （实测：同口径同格子复现不出来，A30-4(a) 差 15.4%）。
    #   **正确做法**：先按 DIR-29 的**原样**抽 (3, T, S) ⇒ 前 1e4 步**逐位相同**；
    #   再**追加**抽 (3, T_NOISE-T, S) 供 A30-3 的 T=2e4 快二分用（这批不与任何归档读数比较）。
    base = g.normal(0.0, math.sqrt(MU), size=(3, T, S))
    if T_NOISE > T:
        extra = g.normal(0.0, math.sqrt(MU), size=(3, T_NOISE - T, S))
        base = np.concatenate([base, extra], axis=1)
    return base


A_, B1_, B2_ = noise()


def init_state(arm, init, shape, sig0=SIG0):
    B = init == "init-B"
    S_ = init == "init-S"                              # ★ DIR-30 Axis A：偏斜，W0≡1 固定
    if arm == "fac":
        if B:                                            # init-B：u=√5, v=1/√5 ⇒ W0=1（尺度不同）
            u0, v0 = math.sqrt(5.0), 1.0 / math.sqrt(5.0)
        elif S_:                                        # init-S：u=σ_f, v=1/σ_f ⇒ W0=1（卡 §3.2）
            u0, v0 = sig0, 1.0 / sig0
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
    if eta.ndim == 1:
        eta = eta.reshape(1, -1)                 # (1,K)：全体同一点
    K = eta.shape[-1]
    shape = (s_cols, K)
    # ★ DIR-30：允许 (S,K) 的**逐列 η**（逐列二分用；卡 §6 坑 6）
    eta_b = eta if eta.shape[0] == s_cols else eta.reshape(1, K)
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


# ══════════════════════════════════════ DIR-30 新判据量：逐列过零阈值（卡 §3.3）
def flip_indices(lam):
    r"""逐列找**首个 −→+** 区间（DEF-29-1 的方向：λ̂ 随 η **递增**）⇒ 返回每列的区间下标；
    −1 = **删失**（整条曲线始终 ≤0 ⇒ 网格内从未翻正）。"""
    lam = np.asarray(lam, dtype=float)
    S, K = lam.shape
    flip = np.full(S, -1, dtype=int)
    for k in range(K - 1):
        cond = (flip < 0) & (lam[:, k] <= 0.0) & (lam[:, k + 1] > 0.0)
        flip[cond] = k
    return flip


def find_eta_c_col(arm, gamma, init=PRIMARY_INIT, sig0=SIG0, t_steps=T, s_cols=S,
                   run_fn=None):
    r"""**逐列过零阈值**（卡 §3.3，本卡主判据量）。

    每列各自在 40 点网格上找首个 −→+ 区间，再各自二分 6 次（一次 run() 用 η∈R^{S×1} 推进所有列）
    ⇒ 列阈值 η_j；臂阈值 = nanmedian(η_j)（纪律 60）。
    删失列（−1）**不排除**（排除＝可选停止，DEF-28-17）⇒ 插补为 CENSOR_IMPUTE（网格上沿 +1 格）。
    返回 (threshold, eta_j (S,), info)。
    """
    rf = run_fn if run_fn is not None else run
    grid = np.exp(np.linspace(math.log(ETA_LO), math.log(ETA_HI), K_GRID))
    r = rf(arm, grid, gamma, init, sig0, t_steps, s_cols)
    lam = r["lam_top"]
    flip = flip_indices(lam)
    censored = flip < 0
    idx = np.clip(flip, 0, K_GRID - 2)
    lo = grid[idx].copy()
    hi = grid[idx + 1].copy()
    for _ in range(K_BIS):
        mid = 0.5 * (lo + hi)
        r2 = rf(arm, mid.reshape(s_cols, 1), gamma, init, sig0, t_steps, s_cols)
        m = np.asarray(r2["lam_top"], dtype=float)[:, 0]
        # ★ NaN（指数未定义＝溢出＝已发散）算**不稳定** ⇒ 根更小 ⇒ hi=mid
        unstable = (~np.isfinite(m)) | (m > 0.0)
        hi = np.where(unstable, mid, hi)
        lo = np.where(unstable, lo, mid)
    eta_j = 0.5 * (lo + hi)
    eta_j = np.where(censored, CENSOR_IMPUTE, eta_j)
    with np.errstate(all="ignore"):
        thr = float(np.nanmedian(eta_j))
    info = {"censored_frac": float(censored.mean()), "n_censored": int(censored.sum()),
            "n_cols": int(s_cols),
            "eta_j_q": [float(np.nanpercentile(eta_j, q)) for q in (5, 25, 50, 75, 95)]}
    return thr, eta_j, info


# ══════════════════════════════════════ Stage A
def stage_a():
    res = {}
    ok_all = True

    def check(name, cond, info):
        nonlocal ok_all
        res[name] = {"pass": bool(cond), **info}
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {info}", flush=True)
        ok_all = ok_all and bool(cond)

    print("Stage A · DIR-30（初始化作为一等公民）")

    # A30-1 工作例逐位（卡 §4，与 DIR-29 A29-1 同款）
    J = jac_fac(np.full(1, 1.0), np.full(1, 1.0), np.full(1, 2.0), np.full(1, 0.5), np.full(1, 0.3))
    exp = [-0.215, -2.13, -0.215]
    got = [float(J[0][0]), float(J[1][0]), float(J[2][0])]
    det = got[0] * got[2] - got[1] ** 2
    ok1 = all(abs(g - e) < 1e-12 for g, e in zip(got, exp)) and abs(det - (-4.490675)) < 1e-9
    check("A30-1 worked-example", ok1, {"J11,J12,J22": got, "det": det})

    # A30-2 雅可比有限差分（fac 块；中心差分 h=1e-6）
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
    check("A30-2 jacobian-fd", ok2, {"worst_rel": worst})

    # A30-3 跨装置锚：η_c^full(γ=0) ∈ [0.48091, 0.50091]（T=2e4 快二分）
    eta_c0, br, info = find_eta_c("full", 0.0, t_steps=20000)
    ok3 = eta_c0 is not None and abs(eta_c0 - ANCHOR_AS) <= 0.01
    check("A30-3 cross-device-anchor", ok3, {"eta_c_full_g0": eta_c0, "anchor": ANCHOR_AS})

    # A30-4 **与归档同口径的自洽检查**（DEF-30-2 后改写）：
    #   ★ 首跑证明「新统计量（逐列过零）」**≠** DIR-29 的统计量（中位数后翻根）：
    #     fac 主格相对差 16.4%（γ=0）/ 8.2%（γ=0.5）⇒ 两个量在 fac 臂上**测的不是同一件事**
    #     （机制：fac 臂有大量列的 λ̂ 被钉在 ≈0 的**边际平台**上 ⇒ 中位数后翻根测的是
    #      「中位列自己翻正」的那一点，被平台顶得更高 ⇒ 偏大；逐列统计量测「列阈值分布的中位数」）。
    #   ⇒ 自洽检查改成**两条**（各用自己的口径）：
    #     (a) **DIR-29 统计量**在主格（init-S σ_f=1）vs DIR-29 归档的 init-A σ0=1 读数（≤5%）；
    #     (b) **新统计量**在**全矩阵臂**（init-A）vs DIR-29 归档的 full 读数（≤5%）。
    #   ★ fac 臂上两种口径的差**只作报告量**（⇒ DEF-30-2 的证据），不设阈值。
    a4 = {}
    ok4 = True
    for g, ref in ((0.0, D29_FAC_S1_G0), (0.5, D29_FAC_S1_G05)):
        thr_med, _, _ = find_eta_c("fac", g, init="init-S", sig0=1.0)
        rel = None if (thr_med is None or ref is None) else abs(thr_med - ref) / ref
        a4[f"med_stat_g{g}"] = {"thr_D29stat": thr_med, "ref_D29": ref, "rel": rel}
        ok4 = ok4 and (rel is not None and rel <= TOL_SELF)
    for g, ref in ((0.0, D29_FULL_G0), (0.5, D29_FULL_G05)):
        thr_col, ej, inf = find_eta_c_col("full", g, init="init-A", sig0=1.0)
        rel = None if (thr_col is None or ref is None) else abs(thr_col - ref) / ref
        a4[f"new_stat_full_g{g}"] = {"thr_col": thr_col, "ref_D29": ref, "rel": rel}
        ok4 = ok4 and (rel is not None and rel <= TOL_SELF)
    check("A30-4 archive-selfconsistency", ok4, a4)

    # A30-5 **删失插补规则**：人造一条「永不过零」的列 ⇒ 该列必须被插补为 CENSOR_IMPUTE 且**不被排除**
    def fake_run(arm, eta, gamma, init=PRIMARY_INIT, sig0=SIG0, t_steps=T, s_cols=S, **kw):
        eta = np.atleast_1d(np.asarray(eta, dtype=float))
        K = eta.size
        # ★ DEF-30-3：rev 1.1 的假装置把 lam[1:] 整片置 +1 ⇒ 那些列**从来不是 ≤0**
        #   ⇒ 它们也成了「删失」⇒ n_censored=64 而非 1 ⇒ A30-5 首跑 FAIL（**测试装置的错**）。
        eta = np.atleast_1d(np.asarray(eta, dtype=float))
        K = eta.size
        lam = np.full((s_cols, K), -1.0)          # 全部列恒负 ⇒ 第 0 列删失
        lam[1:, -1] = 1.0                         # 其余列在最后一格翻正 ⇒ k=K-2 处过零
        return {"lam_top": lam}
    thr5, ej5, inf5 = find_eta_c_col("fac", 0.5, init="init-S", sig0=1.0, run_fn=fake_run)
    ok5 = (inf5["n_censored"] == 1
           and abs(ej5[0] - CENSOR_IMPUTE) < 1e-12
           and abs(CENSOR_IMPUTE - 1.2 * GRID_RATIO) < 1e-9
           and inf5["n_cols"] == S)
    check("A30-5 censoring-imputation", ok5,
          {"n_censored": inf5["n_censored"], "eta_j0": float(ej5[0]),
           "CENSOR_IMPUTE": CENSOR_IMPUTE, "n_cols": inf5["n_cols"]})

    # A30-6 确定性：同参数两次运行前 12 位相同
    _, r6a = top_median("fac", 0.15, 0.5)
    _, r6b = top_median("fac", 0.15, 0.5)
    d6 = float(np.max(np.abs(r6a["lam_top"] - r6b["lam_top"])))
    check("A30-6 determinism", d6 == 0.0, {"max_abs_diff": d6})

    # A30-7 列=种子可复算：第 0 列单跑 == 全阵第 0 列
    r7a = run("fac", 0.15, 0.5, s_cols=S)
    r7b = run("fac", 0.15, 0.5, s_cols=1)
    d7 = float(np.max(np.abs(r7a["lam_top"][0] - r7b["lam_top"][0])))
    check("A30-7 column-seed-repro", d7 == 0.0, {"max_abs_diff": d7})

    # A30-8 **两轴在 σ=1 重合**：init-S(σ_f=1) 的状态与 init-A(σ0=1) **逐位相同**
    #   ⇒ 且（T=2000 的短跑）阈值也逐位相同。
    shp = (4, 1)
    st_s = init_state("fac", "init-S", shp, 1.0)
    st_a = init_state("fac", "init-A", shp, 1.0)
    same_state = all(np.array_equal(x, y) for x, y in zip(st_s, st_a))
    t8a, _, _ = find_eta_c_col("fac", 0.5, init="init-S", sig0=1.0, t_steps=2000, s_cols=8)
    t8b, _, _ = find_eta_c_col("fac", 0.5, init="init-A", sig0=1.0, t_steps=2000, s_cols=8)
    ok8 = same_state and (t8a == t8b)
    check("A30-8 axes-coincide-at-1", ok8,
          {"same_state": bool(same_state), "thr_skew": t8a, "thr_sym": t8b})

    # A30-9 **偏斜轴 = init-B**：用 **DIR-29 统计量**（同口径）在 σ_f=√5 上 vs 归档的 init-B 读数
    #   （≤5%）⇒ 证明「偏斜轴 √5 ≡ 归档的 init-B」（动力学同源）。
    #   ★ DEF-30-2：新统计量在 fac 臂上与归档口径差 8–16% ⇒ 这里**必须**用归档口径比。
    a9 = {}
    ok9 = True
    for g, ref in ((0.0, D29_INITB_G0), (0.5, D29_INITB_G05)):
        thr_med, _, _ = find_eta_c("fac", g, init="init-S", sig0=SQRT5)
        rel = None if (thr_med is None or ref is None) else abs(thr_med - ref) / ref
        a9[f"g{g}"] = {"thr_D29stat": thr_med, "ref_D29": ref, "rel": rel}
        ok9 = ok9 and (rel is not None and rel <= TOL_SELF)
    check("A30-9 skew-axis-equals-initB", ok9, a9)

    res["all_pass"] = bool(ok_all)
    json.dump(res, open(os.path.join(OUT, "DIR30r10_stageA.json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    print(f"Stage A: {'ALL PASS' if ok_all else 'FAILED'}")
    return ok_all


# ══════════════════════════════════════ smoke / formal
def smoke():
    eta_c0, br, _ = find_eta_c("full", 0.0)
    rel = None if eta_c0 is None else (eta_c0 - ANCHOR_AS) / ANCHOR_AS
    ok = eta_c0 is not None and abs(rel) <= 0.03
    print(f"smoke: η_c(full,γ=0)={eta_c0} rel={rel} {'OK' if ok else 'BAD'} ({time.time()-_T0:.1f}s)")
    json.dump({"eta_c_full_g0": eta_c0, "anchor": ANCHOR_AS, "rel": rel, "pass": ok,
               "secs": round(time.time() - _T0, 1)},
              open(os.path.join(OUT, "DIR30r10_smoke.json"), "w"), ensure_ascii=False, indent=1)
    return ok


def formal():
    print("formal · DIR-30", flush=True)
    res = {"cells": {}, "axes": {}}

    def cell(arm, g, init, sig0, key):
        thr, ej, inf = find_eta_c_col(arm, g, init=init, sig0=sig0)
        c = {"eta_c_col": thr, "censored_frac": inf["censored_frac"],
             "n_censored": inf["n_censored"], "eta_j_q": inf["eta_j_q"],
             "axis": key}
        # ★ DIR-30：同一格同时算 **DIR-29 的统计量**（中位数后翻根）⇒ 两种口径的对照（报告量）
        thr_med, _, _ = find_eta_c(arm, g, init=init, sig0=sig0)
        c["eta_c_med_D29stat"] = thr_med
        if thr is not None:
            rr = run(arm, thr, g, init, sig0)
            c["at_eta_c"] = exp_report(rr)
            c["n_trunc_at_eta_c"] = rr["n_trunc"]
        res["cells"][f"{arm}|{key}|g{g}"] = c
        print(f"  {arm} {key} γ={g}: eta_c(col)={thr} eta_c(D29stat)={thr_med} "
              f"(censored {inf['censored_frac']:.3f})", flush=True)
        return c

    # ── Axis A：偏斜（W0≡1 固定）──
    for sf in AXIS_A:
        for g in (0.0, GAMMA_CONF):
            cell("fac", g, "init-S", sf, f"skew{sf:g}")
    # ── Axis B：对称尺度（W0=σ0²）──；σ0=1 与 Axis A σ_f=1 重合 ⇒ 复用（卡 §5.5）
    for s0 in AXIS_B:
        if s0 == 1.0:
            for g in (0.0, GAMMA_CONF):
                # ★ DEF-30-5：键名格式必须与 Axis A 写入时**完全一致**（`{1.0:g}` ⇒ "1"，
                #   不是 "1.0"）⇒ rev 1.3 的正式轮在 σ0=1 复用格上抛 KeyError。
                #   另：复用格必须**连同 eta_c_med_D29stat 一起复制**，否则 D29 口径的
                #   振幅会漏掉 sym1 这一格（两种口径的样本集就不一样了）。
                src = res["cells"][f"fac|skew{1.0:g}|g{g}"]
                res["cells"][f"fac|sym{s0:g}|g{g}"] = {
                    "eta_c_col": src["eta_c_col"], "censored_frac": src["censored_frac"],
                    "n_censored": src["n_censored"], "eta_j_q": src["eta_j_q"],
                    "eta_c_med_D29stat": src.get("eta_c_med_D29stat"),
                    "axis": f"sym{s0:g}", "reused_from": f"skew{1.0:g}"}
            continue
        for g in (0.0, GAMMA_CONF):
            cell("fac", g, "init-A", s0, f"sym{s0:g}")
    # ── 全矩阵臂（W0≡1 ⇒ 与初始化无关）与 one 锚 ──
    for g in (0.0, GAMMA_CONF):
        cell("full", g, "init-A", 1.0, "full")
    ec_one, _, _ = find_eta_c("one", 0.0)
    res["eta_c_one"] = ec_one
    print(f"  one: η_c={ec_one}", flush=True)

    def R_of(arm_key_prefix, g, axis_map):
        e0 = res["cells"].get(f"{arm_key_prefix}|g0.0", {}).get("eta_c_col")
        e1 = res["cells"].get(f"{arm_key_prefix}|g{g}", {}).get("eta_c_col")
        return None if (e0 is None or e1 is None) else e1 / e0

    # ── 逐初始化的 R（fac）──
    Rmap = {}
    for axis, sfs in (("skew", AXIS_A), ("sym", AXIS_B)):
        for x in sfs:
            k0 = f"fac|{axis}{x:g}|g0.0"
            k1 = f"fac|{axis}{x:g}|g{GAMMA_CONF}"
            e0 = res["cells"].get(k0, {}).get("eta_c_col")
            e1 = res["cells"].get(k1, {}).get("eta_c_col")
            cen0 = res["cells"].get(k0, {}).get("censored_frac", 0.0)
            cen1 = res["cells"].get(k1, {}).get("censored_frac", 0.0)
            R = None if (e0 is None or e1 is None) else e1 / e0
            # 同格的 DIR-29 统计量口径
            m0 = res["cells"].get(k0, {}).get("eta_c_med_D29stat")
            m1 = res["cells"].get(k1, {}).get("eta_c_med_D29stat")
            Rmed = None if (m0 is None or m1 is None) else m1 / m0
            Rmap[f"{axis}{x:g}"] = {"R": R, "R_D29stat": Rmed,
                                    "eta_c_g0": e0, "eta_c_g05": e1,
                                    "eta_c_g0_D29stat": m0, "eta_c_g05_D29stat": m1,
                                    "censored_max": max(cen0, cen1)}
            print(f"  R[{axis}{x:g}] = {R}   (D29stat R = {Rmed})", flush=True)
    res["R_map"] = Rmap
    R_full = None
    e_f0 = res["cells"].get("fac|full|g0.0", {}).get("eta_c_col")
    # full 臂自己的 R
    f0 = res["cells"].get("full|full|g0.0", {}).get("eta_c_col")
    f1 = res["cells"].get(f"full|full|g{GAMMA_CONF}", {}).get("eta_c_col")
    R_full = None if (f0 is None or f1 is None) else f1 / f0
    res["R_full"] = R_full
    res["eta_c_full"] = {"g0": f0, "g05": f1}

    # ── 前置 1：操纵检查（η_ref = 0.25·η_c^one；同 DIR-29 P1 口径与阈值）──
    eta_ref = P1_ETA_FRAC * ec_one
    _, r0 = top_median("full", eta_ref, 0.0)
    _, r1 = top_median("full", eta_ref, GAMMA_CONF)
    with np.errstate(all="ignore"):
        m0 = float(np.nanmedian(r0["lam2"]))
        m1 = float(np.nanmedian(r1["lam2"]))
    p1 = None if ec_one is None else abs(m1 - m0)
    res["P1"] = {"eta_ref": eta_ref, "lam2_med_g0": m0, "lam2_med_g05": m1,
                 "shift": p1, "pass": None if p1 is None else bool(p1 >= P1_SHIFT)}
    print(f"  P1: shift={p1} (need ≥{P1_SHIFT}) {'PASS' if res['P1']['pass'] else 'FAIL'}", flush=True)

    # ── 前置 2：地板 ──
    p2 = None if R_full is None else abs(R_full - 1.0)
    res["P2"] = {"R_full": R_full, "dev": p2, "pass": None if p2 is None else bool(p2 >= FLOOR)}
    print(f"  P2: |R_full−1|={p2} (need ≥{FLOOR}) {'PASS' if res['P2']['pass'] else 'FAIL'}", flush=True)

    # ── H30：振幅 ──
    Rs = {k: v["R"] for k, v in Rmap.items() if v["R"] is not None}
    Rs_med = {k: v["R_D29stat"] for k, v in Rmap.items() if v["R_D29stat"] is not None}
    censored_cells = [k for k, v in Rmap.items() if v["censored_max"] > 0.5]
    amp = None
    if Rs and not censored_cells:
        amp = float(max(Rs.values()) - min(Rs.values()))
    amp_med = None
    if Rs_med and not censored_cells:
        amp_med = float(max(Rs_med.values()) - min(Rs_med.values()))
    # ★ 卡 rev 1.2 的**统计量稳健性**条款：两种口径的振幅若分居 0.15 两侧 ⇒ 判定记
    #   `STATISTIC-SENSITIVE`（不写成立/否证）——不追认任何一个口径。
    state = ("INFEASIBLE-RESOLUTION" if censored_cells else
             ("NOT-COMPUTED" if amp is None else "COMPUTED"))
    stat_sensitive = (amp is not None and amp_med is not None
                      and ((amp >= AMP_MIN) != (amp_med >= AMP_MIN)))
    if stat_sensitive:
        state = "STATISTIC-SENSITIVE"
    res["H30"] = {"R_map": Rs, "R_map_D29stat": Rs_med,
                  "amplitude": amp, "amplitude_D29stat": amp_med,
                  "threshold": AMP_MIN, "censored_cells": censored_cells,
                  "statistic_sensitive": bool(stat_sensitive),
                  "pass": (None if (amp is None or stat_sensitive) else bool(amp >= AMP_MIN)),
                  "state": state}
    print(f"  H30: amplitude={amp} (need >= {AMP_MIN}) -> {res['H30']['pass']}"
          f"  [D29stat amplitude={amp_med}]  state={state}"
          f"  censored_cells={censored_cells}", flush=True)

    # ── 报告量：交叉尺度 σ*（R_fac 跨越 R_full 与 R_full−0.10）与 Axis A 单调性 ──
    cross = {}
    for target, name in ((R_full, "R_full"), (None if R_full is None else R_full - GAP, "R_full-0.10")):
        if target is None:
            continue
        hit = [k for k, v in sorted(Rmap.items()) if v["R"] is not None and v["R"] <= target]
        cross[name] = {"target": target, "cells_below": hit}
    res["crossing"] = cross
    sk = [(float(k.replace("skew", "")), v["R"]) for k, v in Rmap.items()
          if k.startswith("skew") and v["R"] is not None]
    sk.sort()
    res["axisA_monotone_nonincreasing"] = bool(all(sk[i][1] >= sk[i + 1][1] - 1e-12
                                                   for i in range(len(sk) - 1)))
    res["axisA_points"] = sk
    skm = [(float(k.replace("skew", "")), v["R_D29stat"]) for k, v in Rmap.items()
           if k.startswith("skew") and v["R_D29stat"] is not None]
    skm.sort()
    res["axisA_monotone_nonincreasing_D29stat"] = bool(all(
        skm[i][1] >= skm[i + 1][1] - 1e-12 for i in range(len(skm) - 1)))
    res["axisA_points_D29stat"] = skm
    print(f"  Axis A 单调（非增）: {res['axisA_monotone_nonincreasing']}  points={sk}", flush=True)

    # ── 报告量：DIR-29 统计量在主格上的同轮对照 ──
    med_g0, _, _ = find_eta_c("fac", 0.0, init="init-S", sig0=1.0)
    med_g05, _, _ = find_eta_c("fac", GAMMA_CONF, init="init-S", sig0=1.0)
    res["median_stat_same_run"] = {"g0": med_g0, "g05": med_g05,
                                   "note": "DIR-29 的统计量（中位数后翻根）在同一装置上的读数"}
    print(f"  同轮对照（DIR-29 统计量）：g0={med_g0} g05={med_g05}", flush=True)

    res["secs"] = round(time.time() - _T0, 1)
    print(f"formal: {res['secs']}s → {OUT}/DIR30r10_formal.json", flush=True)
    json.dump(res, open(os.path.join(OUT, "DIR30r10_formal.json"), "w"),
              ensure_ascii=False, indent=1, default=float)
    return res


def s1():
    r"""设计检查：γ=0 时 full 臂 λ̂1 与 λ̂2 逐列相同（k2≡a ⇒ 同被积函数）。"""
    r = run("full", 0.3, 0.0, t_steps=3000)
    d = float(np.max(np.abs(r["lam"] - r["lam2"])))
    print(f"S1: max|λ̂1−λ̂2| at γ=0 = {d}")
    json.dump({"max_diff": d, "pass": d == 0.0},
              open(os.path.join(OUT, "DIR30r10_s1.json"), "w"), ensure_ascii=False, indent=1)
    return d == 0.0


if __name__ == "__main__":
    _T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["s1", "stageA", "smoke", "formal"], required=True)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"s1": s1, "stageA": stage_a, "smoke": smoke, "formal": formal}[args.stage]()
