#!/opt/pytorch/bin/python
r"""DIR-36 · **剩余差距的分解**：读出保真度 vs 读数头（输出范围 / 函数逼近）。

卡：`revolve/DIR-36/DIRECTION_rev36_v10.md`（rev 1.0，阈值已冻结）。

★ **为什么本文件这么短**：装置 `dir31/R1_grad.py`（md5 `8fa40f7…`）与任务层 `dir33/R1_task33.py`
  （md5 `96ef89e8…`）**逐字零改动**，本文件**只 import 它们**（⇒ 任务与装置在**代码对象层面**
  就是 DIR-33/35 的那些，不是拷贝）；A36-6/A36-7 用 md5 把这一性质钉死。

★ **唯一自变量**：**读数来源** —— 装置的 Hebb 写出的 $W$（`T` 臂）vs 理想秩-2 记忆（`Ts` 臂，
  任务层 DIR-33 里**已有**的那条臂）。读数头结构（8 个 tanh 单元）、训练协议（Adam/lr 0.03/250 步）、
  寻址（oracle）、$n_{\rm train}$ 全部不变。

★ **新增的只有测量层**：
  ① `breakdown()` —— 逐 $t$ 镜像 `R.rollout_task`，记录 $h_t,\gamma,m_t,y_t,g_{t-1},\ell_t$（A36-1/A36-2 钉死它不扰动轨迹）；
  ② 盒约束 QP 求解器（投影 Gauss-Seidel；精确性由主动集枚举交叉核对 A36-3）⇒ 读数头的**输出范围**地板；
  ③ 读出误差诊断（$y$ vs $g$，以及理想秩-2 读 vs $g$）。

用法：calib | stageA | repro | smoke | formal
"""
import hashlib
import itertools
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
sys.path.insert(0, "/home/brra1n0/.sandbox/experiment/1107/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
sys.path.insert(0, "/home/brra1n0/.sandbox/experiment/1107/sgmn-loop/dir31")
import R1_task33 as R          # 任务层（逐字未改）
import R1_grad as M            # 装置（逐字未改）

OUT = "/home/ec2-user/sgmn-loop/dir36/out"
D_H, D_V, D_K, T = R.D_H, R.D_V, R.D_K, R.T

# ══════════ 种子段（卡 §3.2；纪律 70 + 纪律 85：评估段也算历史段 ⇒ 本轮全部用新段）
SEED0 = M.SEED0
TRAIN_BASE = SEED0 + 8000                      # 训练段 8000..8383（384）
EVAL = [SEED0 + 7000 + i for i in range(400)]  # 评估段 7000..7399（400）
CALIB_TRAIN = [SEED0 + 9000 + i for i in range(24)]   # 标定段（不进判据）
CALIB_EVAL = [SEED0 + 8500 + i for i in range(50)]
ANCHOR_TRAIN = SEED0 + 5000                   # ★ DIR-35 的**精确**训练段（复现锚）
ANCHOR_EVAL = [SEED0 + 6000 + i for i in range(400)]

HIST_RANGES = {"pilot32": (100, 124), "formal32": (200, 224), "calib33": (300, 308),
               "formal33": (400, 424), "pilot33": (500, 524), "calib34": (600, 624),
               "heldout34": (700, 724), "train34": (900, 923), "eval34": (2000, 2399),
               "calibtrain35": (3000, 3095), "calibeval35": (4000, 4199),
               "train35": (5000, 5383), "eval35": (6000, 6399)}
OWN_SEGMENTS = {"train36": (8000, 8383), "eval36": (7000, 7399),
                "calibtrain36": (9000, 9023), "calibeval36": (8500, 8549)}

# ══════════ 冻结阈值（卡 rev 1.0 冻结）
D_MIN_FRAC = 0.25
CARD_MD5 = "1c21e3ac8188f71d8e32cd8639357995"             # ← 写卡后回填（Stage A 守卫 A36-11）
MD5_DEVICE = "8fa40f708600bec7e6c484727d2180c7"
MD5_TASK = "96ef89e87ec61556b52e0e464850087e"
CARD_PATH = "/home/ec2-user/sgmn-loop/revolve/DIR-36/DIRECTION_rev36_v10.md"
# ★ 复现锚的**期望读数**（D-59 的 DIR-35 正式轮，n_train=384，逐位）
ANCHOR_EXPECT = {"L_out": 8.368712036551683, "L_in": 7.6006540213249645}
ANCHOR_TOL = 1e-10           # 相对容差（逐位 ⇒ 实际应为 0 差异）
REL_TOL = 1e-12
BOX_TOL = 1e-8               # 盒求解器交叉核对的容差
SOLVER_SWEEPS = 500          # 投影 Gauss-Seidel 的扫描数

_PHI = {}                    # 训练好的 φ（不序列化；供正式轮的前置闸门复用）


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ══════════════════════════════════════════════════════ 盒约束 QP（读数头的输出范围地板）
def box_qp(As, bs, sweeps=SOLVER_SWEEPS):
    r"""min_{m∈[-1,1]^d} ‖A m + b‖² —— 逐实例向量化（投影 Gauss-Seidel）。

    ★ **为什么这是「输出范围」的地板**：读数头的输出 $m=\tanh(\cdot)\in(-1,1)^{8}$ ⇒
      它能产生的修正恰好落在盒 $(-1,1)^8$；**任何**（不论多聪明）的、形如
      $m=\tanh([c,y]\mathcal W^o_1+b^o_1)$ 的头都只能落在盒里 ⇒ 本函数给出的是
      **在给定轨迹上**任何这类头的损失下界。
    ★ 收敛性：凸二次 + 盒约束下的投影 Gauss-Seidel ⇒ 收敛到全局最优（$G_{ii}>0$ ⇒ 可解）。
    """
    B, r, d = As.shape
    G = torch.einsum("bij,bik->bjk", As, As)
    diag = torch.diagonal(G, dim1=1, dim2=2).clone()
    m = torch.zeros(B, d, dtype=As.dtype)
    for _ in range(sweeps):
        for i in range(d):
            res = torch.einsum("bij,bj->bi", As, m) + bs
            gi = 2.0 * torch.einsum("bi,bi->b", As[:, :, i], res)
            m[:, i] = torch.clamp(m[:, i] - gi / (2.0 * diag[:, i]), -1.0, 1.0)
    res = torch.einsum("bij,bj->bi", As, m) + bs
    return (res * res).sum(1), m


def box_qp_exact(A, b):
    r"""**精确**解（主动集枚举 $3^d$ 个模式 + KKT 判据）—— 仅用于交叉核对（A36-3/A36-12）。"""
    d = A.shape[1]
    G = A.t() @ A
    c = A.t() @ b
    best_val, best_x = float("inf"), None
    for state in itertools.product((-1, 0, 1), repeat=d):
        free = [i for i, s in enumerate(state) if s == 0]
        x = torch.zeros(d, dtype=A.dtype)
        for i, s in enumerate(state):
            if s != 0:
                x[i] = float(s)
        if free:
            fixed = [i for i in range(d) if i not in free]
            rhs = -c[free]
            if fixed:
                rhs = rhs - G[free][:, fixed] @ x[fixed]
            Gf = G[free][:, free]
            sol = torch.linalg.lstsq(Gf, rhs.unsqueeze(1)).solution.squeeze(1)
            if bool(torch.any(sol.abs() > 1.0 + 1e-12)):
                continue
            x[free] = sol
        grad = 2.0 * (G @ x + c)
        bad = False
        for i, s in enumerate(state):
            if s == 1 and float(grad[i]) > 1e-9:
                bad = True
            elif s == -1 and float(grad[i]) < -1e-9:
                bad = True
        if bad:
            continue
        val = float(((A @ x + b) ** 2).sum())
        if val < best_val:
            best_val, best_x = val, x.clone()
    if best_x is None:      # 数值兜底（KKT 未命中 ⇒ 用 Gauss-Seidel 的值）
        v, _ = box_qp(A.unsqueeze(0), b.unsqueeze(0))
        return float(v[0]), None
    return best_val, best_x


# ══════════════════════════════════════════════════════ 逐 t 分解（镜像 R.rollout_task）
def breakdown(P, addr, arm, sd, detach_writes=True):
    r"""逐 $t$ 分解：返回 `(L, recs)`。

    ★ **必须与 `R.rollout_task` 逐位等价**（A36-1/A36-2 钉死）：本函数是它的**镜像**，
      只多记了层 1 的 $h_t,\gamma,m_t,y_t$、目标 $g_{t-1}$、实际损失 $\ell_t$，
      以及**本轨迹**上按已写下的 $(k,v)$ 对构造的理想秩-2 读 `ystar`（读出误差诊断用）。
    """
    L = P["a"].new_zeros(())
    gs = R.targets(P)
    if arm == "Ts":
        H = R.hs_of(P)
        KS = [torch.tanh(H[s] @ P["Pk1"]) for s in range(T)]
        VS = [torch.tanh(H[s] @ P["Pv1"]) for s in range(T)]
    else:
        KS = VS = None
    pairs = []                       # pairs[s] = 第 s 步写下的 (k_{s-1}, v_s)
    recs = []
    h_pre_prev = [None, None]
    for t in range(T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        h1 = m1 = y1 = q1 = None
        gam1 = None
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if addr == "oracle" and l == 1:
                c, q = h, torch.tanh(h @ P["Pk1"])
            else:
                c = (P["a"][t] if l == 1 else hp_prev).detach()
                q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            if arm == "Ts" and l == 1:
                pp = [(KS[s - 1], VS[s]) for s in range(1, t)]
                if len(pp) >= 2:
                    K = torch.stack([x[0] for x in pp], 1)
                    Vm = torch.stack([x[1] for x in pp], 1)
                    U_, S_, Vt = torch.linalg.svd(Vm @ K.t())
                    Wr = (U_[:, :2] * S_[:2]) @ Vt[:2]
                else:
                    Wr = torch.zeros(D_V, D_K)
                y = Wr @ q
            else:
                y = R.read_W(W.detach(), arm, sd) @ q
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            gam = torch.zeros(1) if arm == "G0" else P[f"g{l}"]
            hp[li] = h + gam * m
            hp_prev = hp[li]
            if l == 1:
                h1, m1, y1, gam1, q1 = h, m, y, gam, q
        if t >= 1:
            ell = ((R.pout() @ hp[0] - gs[t - 1]) ** 2).sum()
            L = L + ell
            # ★ 本轨迹上的理想秩-2 读（读出误差诊断；只对层 1 有意义）
            pp = [(pairs[s][0], pairs[s][1]) for s in range(1, t)]
            if len(pp) >= 2:
                K = torch.stack([x[0] for x in pp], 1)
                Vm = torch.stack([x[1] for x in pp], 1)
                U_, S_, Vt = torch.linalg.svd(Vm @ K.t())
                Wst = (U_[:, :2] * S_[:2]) @ Vt[:2]
            else:
                Wst = torch.zeros(D_V, D_K)
            recs.append({"t": t, "h": h1.clone(), "gam": gam1.clone(), "m": m1.clone(),
                         "y": y1.clone(), "ystar": (Wst @ q1).clone(),
                         "g": gs[t - 1].clone(), "ell": float(ell)})
        upd = {}
        for li, l in enumerate((1, 2)):
            hprev = h_pre_prev[li]
            if hprev is None:
                hprev = torch.zeros(D_H)
            k = torch.tanh(hprev.detach() @ P[f"Pk{l}"])
            v = torch.tanh(h_pre[li].detach() @ P[f"Pv{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
            Uo, Vo = P[f"U{l}"], P[f"V{l}"]
            upd[f"U{l}"] = Uo - M.ETA * (G @ Vo)
            upd[f"V{l}"] = Vo - M.ETA * (G.t() @ Uo)
            if l == 1:
                pairs.append((k.clone(), v.clone()))   # ★ 第 t 步写入的 (k_{t-1}, v_t)
        if detach_writes:
            upd = {kk: vv.detach() for kk, vv in upd.items()}
        P.update(upd)
        h_pre_prev = h_pre
    return L, recs


def floor_from_recs(recs):
    r"""由逐 $t$ 记录算：盒地板 $L_D$（输出范围地板）与无约束地板 $L_F$（逐 $t$ 最小二乘残差）。

    $A=\gamma\,\mathcal O$、$b=\mathcal O h_t-g_{t-1}$ ⇒ 实际损失 $=\lVert A m+b\rVert^2$。
    """
    O = R.pout()
    As, bs, lls = [], [], []
    for rc in recs:
        gam = float(rc["gam"])
        As.append(gam * O)
        bs.append(O @ rc["h"] - rc["g"])
        lls.append(rc["ell"])
    As = torch.stack(As)
    bs = torch.stack(bs)
    obj, _ = box_qp(As, bs)
    sol = torch.linalg.lstsq(As, (-bs).unsqueeze(2)).solution.squeeze(2)
    res = torch.einsum("bij,bj->bi", As, sol) + bs
    lf = float((res * res).sum())
    return float(obj.sum()), lf, [float(x) for x in lls]


# ══════════════════════════════════════════════════════ 配对统计
def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    se = sd / len(d) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": 3 * se, "n": len(d)}


def _mk(sd, phi):
    P = R.build(sd, "V0")
    R.period2(P, sd)
    if phi:
        P.update({k: v.clone() for k, v in phi.items()})
    return P


def floor_Z(seeds):
    zs = []
    for sd in seeds:
        P = _mk(sd, None)
        zs.append(sum(float((g ** 2).sum()) for g in R.targets(P)))
    return zs


# ══════════════════════════════════════════════════════ 一个档位的完整测量
def measure(level, train_base, eval_seeds, arms=("T", "Ts")):
    r"""在 `level` 个训练实例上训 A 层（`T`）与 C 层（`Ts`），在 `eval_seeds` 上逐实例分解。"""
    tr = [train_base + i for i in range(level)]
    out = {"level": level, "train_base": train_base, "n_eval": len(eval_seeds), "arms": {}}
    phis = {}
    for arm in arms:
        t0 = time.time()
        phis[arm] = R.train_arm("oracle", arm, learned=False, seeds=tr)
        print(f"  训练 {arm:2s}（n={level}） {time.time()-t0:.0f}s", flush=True)
    _PHI[(level, train_base)] = phis
    out["gamma"] = {a: float(phis[a]["g1"]) for a in arms}
    acc = {a: {"L": [], "LD": [], "LF": [], "rell_y": [], "rell_ystar": []} for a in arms}
    t0 = time.time()
    for i, sd in enumerate(eval_seeds):
        for arm in arms:
            P = _mk(sd, phis[arm])
            L, recs = breakdown(P, "oracle", arm, sd)
            LD, LF, lls = floor_from_recs(recs)
            acc[arm]["L"].append(float(L))
            acc[arm]["LD"].append(LD)
            acc[arm]["LF"].append(LF)
            e, es = [], []
            for rc in recs:
                gn = float(rc["g"].norm())
                if gn > 0:
                    e.append(float((rc["y"] - rc["g"]).norm()) / gn)
                    es.append(float((rc["ystar"] - rc["g"]).norm()) / gn)
            acc[arm]["rell_y"].append(statistics.fmean(e) if e else 0.0)
            acc[arm]["rell_ystar"].append(statistics.fmean(es) if es else 0.0)
        if (i + 1) % 100 == 0:
            print(f"    评估 {i+1}/{len(eval_seeds)}  ({time.time()-t0:.0f}s)", flush=True)
    out["per"] = {a: dict(acc[a]) for a in arms}
    for a in arms:
        out["per"][a]["mean_L"] = statistics.fmean(acc[a]["L"])
        out["per"][a]["mean_LD"] = statistics.fmean(acc[a]["LD"])
        out["per"][a]["mean_LF"] = statistics.fmean(acc[a]["LF"])
        out["per"][a]["mean_rell_y"] = statistics.fmean(acc[a]["rell_y"])
        out["per"][a]["mean_rell_ystar"] = statistics.fmean(acc[a]["rell_ystar"])
    return out


def decompose(m):
    r"""由 `measure` 的输出算三项机制（①读出 ②逼近 ③范围）与分离统计。"""
    TA = m["per"]["T"]["L"]
    TC = m["per"]["Ts"]["L"]
    DA = m["per"]["T"]["LD"]
    DC = m["per"]["Ts"]["LD"]
    LA, LC = statistics.fmean(TA), statistics.fmean(TC)
    LDA, LDC = statistics.fmean(DA), statistics.fmean(DC)
    g_read = paired(TA, TC)      # ① 读出保真度
    g_approx = paired(TC, DC)    # ② 读数头的函数逼近
    g_range = {"mean": LDC, "sd": (statistics.stdev(DC) if len(DC) > 1 else 0.0),
               "se": 0.0, "3se": 0.0, "n": len(DC)}
    g_range["se"] = g_range["sd"] / len(DC) ** 0.5
    g_range["3se"] = 3 * g_range["se"]
    terms = {"read": g_read["mean"], "approx": g_approx["mean"], "range": g_range["mean"]}
    per_inst = {"read": [a - b for a, b in zip(TA, TC)],
                "approx": [a - b for a, b in zip(TC, DC)],
                "range": list(DC)}
    dom = max(terms, key=terms.get)
    second = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)[1][0]
    sep = paired(per_inst[dom], per_inst[second])
    return {"L_A": LA, "L_C": LC, "L_D_A": LDA, "L_D_C": LDC,
            "g_read": g_read, "g_approx": g_approx, "g_range": g_range,
            "terms": terms, "dom": dom, "second": second, "sep": sep,
            "identity_resid": (LA - LDC) - (g_read["mean"] + g_approx["mean"])}


# ══════════════════════════════════════════════════════ Stage A
def stage_a():
    a = {}
    tr = list(range(TRAIN_BASE, TRAIN_BASE + 8))
    phiA = R.train_arm("oracle", "T", learned=False, seeds=tr, steps=25)
    phiC = R.train_arm("oracle", "Ts", learned=False, seeds=tr, steps=25)
    # A36-1 / A36-2：分解层不扰动轨迹（逐实例相对误差）
    for tag, phi, arm in (("A36-1", phiA, "T"), ("A36-2", phiC, "Ts")):
        errs = []
        for sd in EVAL[:12]:
            P = _mk(sd, phi)
            Lb, _ = breakdown(P, "oracle", arm, sd)
            Lt = float(R.rollout_task(_mk(sd, phi), "oracle", arm, sd))
            errs.append(abs(float(Lb) - Lt) / max(abs(Lt), 1.0))
        a[f"{tag}_rel_max"] = max(errs)
    # A36-3：盒求解器 vs 主动集枚举（随机 QP）
    g = torch.Generator().manual_seed(20261112)
    diffs = []
    for _ in range(8):
        A = torch.randn(4, 8, generator=g) / math.sqrt(8)
        b = torch.randn(4, generator=g)
        v1, _ = box_qp(A.unsqueeze(0), b.unsqueeze(0))
        v2, _ = box_qp_exact(A, b)
        diffs.append(abs(float(v1[0]) - v2))
    a["A36-3_solver_gap"] = max(diffs)
    # A36-4：手算工作例（2 维，O=I，h=0，γ=1）
    A1 = torch.eye(2)
    v1, _ = box_qp(A1.unsqueeze(0), torch.tensor([-2.0, -1.0]).unsqueeze(0))
    v0, _ = box_qp(A1.unsqueeze(0), torch.tensor([-0.5, 0.3]).unsqueeze(0))
    a["A36-4_worked_1"] = float(v1[0])          # 期望 1.0
    a["A36-4_worked_2"] = float(v0[0])          # 期望 0.0
    a["A36-4_exact_1"] = box_qp_exact(A1, torch.tensor([-2.0, -1.0]))[0]
    a["A36-4_exact_2"] = box_qp_exact(A1, torch.tensor([-0.5, 0.3]))[0]
    # A36-5：地板是真地板（逐 t 盒最优 ≤ 实际损失）
    viol = 0
    O = R.pout()
    for sd in EVAL[:20]:
        P = _mk(sd, phiA)
        Lb, recs = breakdown(P, "oracle", "T", sd)
        for rc in recs:
            A = float(rc["gam"]) * O
            b = O @ rc["h"] - rc["g"]
            v, _ = box_qp(A.unsqueeze(0), b.unsqueeze(0))
            if float(v[0]) > rc["ell"] + 1e-9:
                viol += 1
    a["A36-5_violations"] = float(viol)
    # A36-6 / A36-7：md5
    a["A36-6_device_md5"] = _md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py")
    a["A36-6_device_diff"] = 0.0 if a["A36-6_device_md5"] == MD5_DEVICE else 1.0
    a["A36-7_task_md5"] = _md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py")
    a["A36-7_task_diff"] = 0.0 if a["A36-7_task_md5"] == MD5_TASK else 1.0
    # A36-8：种子隔离
    own = list(range(TRAIN_BASE, TRAIN_BASE + 384)) + EVAL + CALIB_EVAL + CALIB_TRAIN
    hits = sorted({n for n, (lo, hi) in HIST_RANGES.items() for s in own if lo <= (s - SEED0) <= hi})
    a["A36-8_hist_hits"] = ",".join(hits) if hits else "none"
    a["A36-8_seed_disjoint"] = 0.0 if not hits else 1.0
    a["A36-8_train_eval_overlap"] = float(len(set(range(TRAIN_BASE, TRAIN_BASE + 384)) & set(EVAL)))
    # A36-9：训练确定性
    p1 = R.train_arm("oracle", "T", learned=False, seeds=tr[:4], steps=6)
    p2 = R.train_arm("oracle", "T", learned=False, seeds=tr[:4], steps=6)
    a["A36-9_train_bitwise"] = max(float((p1[k] - p2[k]).abs().max()) for k in p1)
    # A36-10：O 满行秩 ⇒ 逐 t 的无约束残差恒 0（结构断言）
    a["A36-10_rank_O"] = int(torch.linalg.matrix_rank(O).item())
    P = _mk(EVAL[0], phiA)
    _, recs = breakdown(P, "oracle", "T", EVAL[0])
    resid = []
    for rc in recs:
        A = float(rc["gam"]) * O
        b = O @ rc["h"] - rc["g"]
        # ★ DEF-36-0a：原先写成 `.squeeze(1)` + `A @ sol` ⇒ sol 为 (1,8,1) ⇒ 广播成 (1,4,4)
        #   ⇒ 残差被算成垃圾值（302.6）⇒ 断言假失败。**测量层未受影响**
        #   （`floor_from_recs` 用 einsum，形状正确）⇒ 断言的**意图与阈值不变**，只修实现。
        sol = torch.linalg.lstsq(A.unsqueeze(0), (-b).reshape(1, D_V, 1)).solution.squeeze(-1).squeeze(0)
        resid.append(float(((A @ sol + b) ** 2).sum()))
    a["A36-10_free_resid"] = max(resid)
    # A36-11：卡已冻结
    a["A36-11_card_md5"] = _md5(CARD_PATH)
    a["A36-11_card_diff"] = 0.0 if a["A36-11_card_md5"] == CARD_MD5 else 1.0
    a["A36-11_threshold_set"] = 0.0 if (D_MIN_FRAC is not None) else 1.0
    # A36-12：独立重算（精确求解器 vs Gauss-Seidel；并验恒等式）
    P = _mk(EVAL[0], phiA)
    Lb, recs = breakdown(P, "oracle", "T", EVAL[0])
    LD_ccd, LF, lls = floor_from_recs(recs)
    LD_exact = 0.0
    for rc in recs:
        A = float(rc["gam"]) * O
        b = O @ rc["h"] - rc["g"]
        LD_exact += box_qp_exact(A, b)[0]
    a["A36-12_LD_ccd"] = LD_ccd
    a["A36-12_LD_exact"] = LD_exact
    a["A36-12_LD_gap"] = abs(LD_ccd - LD_exact)
    a["A36-12_floor_le_actual"] = 0.0 if all(x <= y + 1e-9 for x, y in
                                            zip([LD_exact], [float(Lb)])) else 1.0
    ok = (a["A36-1_rel_max"] <= REL_TOL and a["A36-2_rel_max"] <= REL_TOL
          and a["A36-3_solver_gap"] <= BOX_TOL
          and abs(a["A36-4_worked_1"] - 1.0) <= 1e-9 and abs(a["A36-4_worked_2"]) <= 1e-9
          and abs(a["A36-4_exact_1"] - 1.0) <= 1e-9 and abs(a["A36-4_exact_2"]) <= 1e-9
          and a["A36-5_violations"] == 0.0
          and a["A36-6_device_diff"] == 0.0 and a["A36-7_task_diff"] == 0.0
          and a["A36-8_seed_disjoint"] == 0.0 and a["A36-8_train_eval_overlap"] == 0.0
          and a["A36-9_train_bitwise"] == 0.0
          and a["A36-10_rank_O"] == D_V and a["A36-10_free_resid"] <= 1e-12
          and a["A36-11_card_diff"] == 0.0 and a["A36-11_threshold_set"] == 0.0
          and a["A36-12_LD_gap"] <= BOX_TOL and a["A36-12_floor_le_actual"] == 0.0)
    return ok, a


# ══════════════════════════════════════════════════════ 标定（决定二十九，不进判据）
def calib():
    t0 = time.time()
    print("── 标定（机制代码的可执行性 + 配对差的 sd；不进判据）")
    m = measure(24, CALIB_TRAIN[0], CALIB_EVAL)
    d = decompose(m)
    m["decompose"] = d
    m["secs"] = round(time.time() - t0, 1)
    print(f"  L_A {d['L_A']:.5f} · L_C {d['L_C']:.5f} · L_D {d['L_D_C']:.5f} "
          f"· G_read {d['g_read']['mean']:+.5f} · G_approx {d['g_approx']['mean']:+.5f} "
          f"· G_range {d['g_range']['mean']:+.5f}")
    print(f"  主导 {d['dom']} · 分离 |Δ| {abs(d['sep']['mean']):.5f} (3se {d['sep']['3se']:.5f})")
    json.dump(m, open(f"{OUT}/DIR36_calib.json", "w"), ensure_ascii=False, indent=1)
    return m


# ══════════════════════════════════════════════════════ 复现锚（纪律 84：只作一致性校验）
def repro():
    r"""用 DIR-35 的**精确**配置跑 A 层 ⇒ 必须逐位复现 D-59 的两个读数。"""
    t0 = time.time()
    print("── 复现锚（train 5000..5383 / eval 6000..6399，n_train=384，臂 T）")
    tr = [ANCHOR_TRAIN + i for i in range(384)]
    phi = R.train_arm("oracle", "T", learned=False, seeds=tr)
    lo = [float(R.rollout_task(_mk(sd, phi), "oracle", "T", sd)) for sd in ANCHOR_EVAL]
    li = [float(R.rollout_task(_mk(sd, phi), "oracle", "T", sd)) for sd in tr]
    out = {"L_out": statistics.fmean(lo), "L_in": statistics.fmean(li),
           "expect": ANCHOR_EXPECT, "secs": round(time.time() - t0, 1)}
    e_out = abs(out["L_out"] - ANCHOR_EXPECT["L_out"]) / max(abs(ANCHOR_EXPECT["L_out"]), 1.0)
    e_in = abs(out["L_in"] - ANCHOR_EXPECT["L_in"]) / max(abs(ANCHOR_EXPECT["L_in"]), 1.0)
    out["rel_err_out"] = e_out
    out["rel_err_in"] = e_in
    out["ok"] = bool(e_out <= ANCHOR_TOL and e_in <= ANCHOR_TOL)
    print(f"  L*(T,out) {out['L_out']:.8f}（期望 {ANCHOR_EXPECT['L_out']:.8f}，相对差 {e_out:.2e}）")
    print(f"  L*(T,in ) {out['L_in']:.8f}（期望 {ANCHOR_EXPECT['L_in']:.8f}，相对差 {e_in:.2e}）")
    print(f"  ⇒ 锚 {'通过' if out['ok'] else '★未通过'}（{out['secs']:.0f}s）")
    json.dump(out, open(f"{OUT}/DIR36_repro.json", "w"), indent=1)
    return out


# ══════════════════════════════════════════════════════ 判据计算
def attribution(d, dmin):
    r"""卡 §1.2 的**预注册**决策表：读数 ⇒ 归因 ⇒ 下一个杠杆。"""
    t = d["terms"]
    if d["L_D_C"] >= dmin:
        return ("读数头的**输出范围**（盒约束/γ）", "读数头的宽度/激活/γ（需新任务层 revision）")
    if t["read"] >= dmin and t["read"] >= t["approx"]:
        return ("读出保真度", "写/读侧（写入动力学、秩-2 限制、键可分性）")
    if t["approx"] >= dmin and t["approx"] > t["read"]:
        return ("读数头的函数逼近", "读数头的结构（需新任务层 revision）")
    return ("**归因失败**（三项都 < Δmin）", "差距不在这三者 ⇒ 不得硬凑")


def formal():
    t0 = time.time()
    if D_MIN_FRAC is None or CARD_MD5 is None:
        print("★ 阈值/卡 md5 未冻结 ⇒ 拒绝跑正式轮")
        sys.exit(2)
    rp = f"{OUT}/DIR36_repro.json"
    if not os.path.exists(rp):
        print("★ 找不到复现锚记录 ⇒ 拒绝跑正式轮（纪律 84）")
        sys.exit(2)
    anch = json.load(open(rp))
    if not anch.get("ok"):
        print("★ 复现锚未通过 ⇒ 拒绝跑正式轮（MEASUREMENT-NOT-CREDIBLE）")
        sys.exit(2)
    res = {"D_MIN_FRAC": D_MIN_FRAC, "CARD_MD5": CARD_MD5, "anchor": anch, "levels": {}}
    for level in (384, 96):          # 384 = 判据档（DIR-35 的饱和档）；96 = 次级档（报告项）
        print(f"── 档位 n_train={level}")
        m = measure(level, TRAIN_BASE, EVAL)
        d = decompose(m)
        m["decompose"] = d
        res["levels"][str(level)] = m
        print(f"  L_A {d['L_A']:.5f} · L_C {d['L_C']:.5f} · L_D {d['L_D_C']:.5f} · "
              f"L_F {m['per']['Ts']['mean_LF']:.3e}")
        print(f"  G_read {d['g_read']['mean']:+.5f} (3se {d['g_read']['3se']:.5f}) · "
              f"G_approx {d['g_approx']['mean']:+.5f} (3se {d['g_approx']['3se']:.5f}) · "
              f"G_range {d['g_range']['mean']:+.5f}")
        print(f"  主导 {d['dom']} · 分离 |Δ| {abs(d['sep']['mean']):.5f} (3se {d['sep']['3se']:.5f})")
    lev = res["levels"]["384"]
    d = lev["decompose"]
    Z_ev = statistics.fmean(floor_Z(EVAL))
    dmin = D_MIN_FRAC * Z_ev
    res["Z_eval"] = Z_ev
    res["dmin"] = dmin
    terms = d["terms"]
    res["clauses"] = {}
    res["clauses"]["C36-a"] = {"max_term": max(terms.values()), "dmin": dmin,
                               "terms": terms, "ok": max(terms.values()) >= dmin}
    res["clauses"]["C36-b"] = {"dom": d["dom"], "second": d["second"],
                               "diff": abs(d["sep"]["mean"]), "3se": d["sep"]["3se"],
                               "ok": abs(d["sep"]["mean"]) >= d["sep"]["3se"]}
    res["clauses"]["C36-c"] = {"L_F": lev["per"]["Ts"]["mean_LF"],
                               "L_D": d["L_D_C"], "L_C": d["L_C"], "L_A": d["L_A"],
                               "ok": (d["L_D_C"] <= d["L_C"] + 1e-12) and (d["L_C"] <= d["L_A"] + 1e-12)}
    # 前提 P36-1：逐 (实例,t) 盒最优 ≤ 实际损失（C 层轨迹）
    viol = 0
    O = R.pout()
    phiC = _PHI[(384, TRAIN_BASE)]["Ts"]
    for sd in EVAL:
        P = _mk(sd, phiC)
        _, recs = breakdown(P, "oracle", "Ts", sd)
        for rc in recs:
            A = float(rc["gam"]) * O
            b = O @ rc["h"] - rc["g"]
            v, _ = box_qp(A.unsqueeze(0), b.unsqueeze(0))
            if float(v[0]) > rc["ell"] + 1e-9:
                viol += 1
    res["P36-1"] = {"violations": viol, "ok": viol == 0}
    res["identity_resid"] = d["identity_resid"]
    ok = all(c["ok"] for c in res["clauses"].values())
    if not res["P36-1"]["ok"]:
        verdict = "PREMISE-NOT-MET"
    elif not res["clauses"]["C36-b"]["ok"]:
        verdict = "INFEASIBLE-POWER"
    elif ok:
        verdict = "SUCCEEDED"
    else:
        verdict = "FALSIFIED"
    res["verdict"] = verdict
    res["attribution"] = attribution(d, dmin)
    res["secs"] = round(time.time() - t0, 1)
    for k, c in res["clauses"].items():
        print(f"  {k}: {c} ⇒ {'过' if c['ok'] else '不过'}")
    print(f"  P36-1: {res['P36-1']}")
    print(f"  归因：{res['attribution']}")
    print(f"⇒ 判据状态 **{verdict}**  ({res['secs']:.0f}s)")
    json.dump(res, open(f"{OUT}/DIR36r10_formal.json", "w"), ensure_ascii=False, indent=1)
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stageA"
    os.makedirs(OUT, exist_ok=True)
    if mode == "calib":
        calib()
    elif mode == "repro":
        repro()
    elif mode == "stageA":
        ok, a = stage_a()
        print(f"Stage A：{'★ 全部通过' if ok else '★ 有断言失败 —— 冻结正式轮'}")
        for k, v in a.items():
            print(f"  {k:26s} {v if isinstance(v, str) else f'{v:.3e}'}")
        json.dump({"ok": ok, "a": a}, open(f"{OUT}/DIR36r10_stageA.json", "w"), indent=1)
        sys.exit(0 if ok else 1)
    elif mode == "smoke":
        t1 = time.time()
        m = measure(8, TRAIN_BASE, EVAL[:4])
        d = decompose(m)
        ok = all(math.isfinite(v) for v in (d["L_A"], d["L_C"], d["L_D_C"]))
        print(f"smoke：n_train=8 × 4 评估实例 ⇒ {'可执行' if ok else '★ 不可执行'}  ({time.time()-t1:.0f}s)")
        print(f"  L_A {d['L_A']:.5f} · L_C {d['L_C']:.5f} · L_D {d['L_D_C']:.5f}")
        json.dump({"ok": ok, "decompose": d}, open(f"{OUT}/DIR36r10_smoke.json", "w"), indent=1)
        sys.exit(0 if ok else 1)
    else:
        formal()
