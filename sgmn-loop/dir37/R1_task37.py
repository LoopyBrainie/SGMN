#!/opt/pytorch/bin/python
r"""DIR-37 · **公平的读侧基准**（最小二乘最优检索）+ 剩余差距的**四项**重分解 + 「头在用读的什么性质」。

卡：`revolve/DIR-37/DIRECTION_rev37_v10.md`（rev 1.0，阈值已冻结；md5 `3ce25d5b…`，由 A37-11 守卫）。

★ **为什么本文件这么短**：装置 `dir31/R1_grad.py`（md5 `8fa40f7…`）与任务层 `dir33/R1_task33.py`
  （md5 `96ef89e8…`）**逐字零改动**，本文件**只 import 它们**（⇒ 任务与装置在**代码对象层面**
  就是 DIR-33/35 的那些，不是拷贝）；A37-8 用 md5 把这一性质钉死。

★ **唯一自变量**：**读数来源** —— 装置的 Hebb 写出的 $W$（`dev`）vs **最小二乘最优检索**（`ref`）。
  读数头结构（8 个 tanh 单元）、训练协议（Adam/lr 0.03/250 步）、寻址（oracle）、$\gamma$ 训练方式全不变。

★ **五层反事实**（每层只放松一个约束）：
  A（装置）→ B（装置的**冻结**头 + 参考读）→ C（在参考读上**重训**的头）→ D（盒地板）→ F（无约束地板）。

★ **新增的只有测量层**：
  ① `trace()` —— 逐 $t$ 镜像 `R.rollout_task`，记录 $h_t,\gamma,m_t,y_t,g_{t-1},\ell_t$ 并**在线重建参考读**；
  ② 盒约束 QP（投影 Gauss-Seidel，**按实例批处理** + 收敛判据；主动集枚举交叉核对）；
  ③ 参考读（Moore–Penrose 伪逆的最小二乘最优检索）；
  ④ 固定轨迹探针（5 个读变体）。

用法：calib | stageA | repro | smoke | formal
"""
import hashlib
import glob
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

OUT = "/home/ec2-user/sgmn-loop/dir37/out"
D_H, D_V, D_K, T = R.D_H, R.D_V, R.D_K, R.T

# ══════════ 种子段（卡 §6；纪律 70 + 85：全部新段）
SEED0 = M.SEED0
TRAIN_BASE = SEED0 + 11000                    # 训练段 11000..11383（384）
EVAL = [SEED0 + 10000 + i for i in range(400)]  # 评估段 10000..10399（400）
CALIB_TRAIN = [SEED0 + 12000 + i for i in range(24)]
CALIB_EVAL = [SEED0 + 11500 + i for i in range(50)]
ANCHOR_TRAIN = SEED0 + 8000                   # ★ DIR-36 的**精确**训练段（复现锚）
ANCHOR_EVAL = [SEED0 + 7000 + i for i in range(400)]
PROBE_LEVELS = (48, 96, 384)

HIST_RANGES = {"pilot32": (100, 124), "formal32": (200, 224), "calib33": (300, 308),
               "formal33": (400, 424), "pilot33": (500, 524), "calib34": (600, 624),
               "heldout34": (700, 724), "train34": (900, 923), "eval34": (2000, 2399),
               "calibtrain35": (3000, 3095), "calibeval35": (4000, 4199),
               "train35": (5000, 5383), "eval35": (6000, 6399),
               "train36": (8000, 8383), "eval36": (7000, 7399),
               "calibtrain36": (9000, 9023), "calibeval36": (8500, 8549)}
OWN_SEGMENTS = {"train37": (11000, 11383), "eval37": (10000, 10399),
                "calibtrain37": (12000, 12023), "calibeval37": (11500, 11549)}

# ══════════ 冻结阈值（卡 rev 1.0 冻结）
D_MIN_FRAC = 0.25
CARD_MD5 = "3ce25d5b590ee8745ac903ee4d34a474"
MD5_DEVICE = "8fa40f708600bec7e6c484727d2180c7"
MD5_TASK = "96ef89e87ec61556b52e0e464850087e"
CARD_PATH = "/home/ec2-user/sgmn-loop/revolve/DIR-37/DIRECTION_rev37_v10.md"
ANCHOR_EXPECT = {"L_A": 8.257212940724042,      # ← D-60 的 DIR-36 正式轮读数（逐位）
                 "L_A_96": 9.421045543145368}  # ← D-60 的次级档读数（同一配置的 n=96）
ANCHOR_TOL = 1e-10
REL_TOL = 1e-12
BOX_TOL = 1e-8
SOLVER_SWEEPS = 500
SOLVER_CONV = 1e-14                           # 投影 Gauss-Seidel 的收敛判据
MONO_TOL_FRAC = 1e-6                          # C37-d：四项 ≥ −1e−6·Z


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ══════════════════════════════════════════════════════ 参考读：最小二乘最优检索
def ref_W(pairs):
    r"""$W^{\rm LS}=V_mK^{+}$ —— 已写下的配对 $(k_s,v_s)$ 上的**最小二乘最优线性检索**。

    ★ 为什么不是 DIR-33/36 的 $\mathrm{svd}(V_mK^\top)$ 秩-2 截断：后者最小化的是**矩阵**范数
      $\lVert V_mK^\top-W\rVert_F$，不是**检索误差** $\sum_s\lVert Wk_s-v_s\rVert^2$
      ⇒ 它不是最优检索（D-60 实测：比装置自己的读差 7.04×）。
    ★ 伪逆（Moore–Penrose）**无自由参数** ⇒ 不存在「调参把基准调好看」的空间。
    ★ 秩：周期 2 ⇒ $K$ 的列空间 $\le2$ 维 ⇒ $\mathrm{rank}(W^{\rm LS})\le2$ ⇒ 与装置的秩-2 同阶。
    """
    if not pairs:
        return None
    K = torch.stack([p[0] for p in pairs], 1)      # (D_K, n)
    Vm = torch.stack([p[1] for p in pairs], 1)     # (D_V, n)
    return Vm @ torch.linalg.pinv(K)


def _read_variant(kind, y_dev, P, h_pre_prev, gs, t):
    r"""探针的**单变量**读变换（卡 §2.2）：

    - `scale`：$y'=\lVert g\rVert/\lVert y\rVert\cdot y$（**只**改尺度；方向不动）；
    - `phase`：$y'=W_t q_{t-1}$（同一装置状态 $W_t$、**另一相位**的查询 ⇒ 改可辨识性，
      尺度分布近似不变）；
    - `zero`：$y\equiv0$；`oracle`：$y'=g$。
    ★ $t=0$ 步无损失 ⇒ 该步保持装置读（不参与任何读数）。
    """
    if kind == "scale":
        if t == 0:
            return y_dev
        g = gs[t - 1]
        gn, yn = float(g.norm()), float(y_dev.norm())
        if yn <= 0.0:
            return y_dev
        return y_dev * (gn / yn)
    if kind == "phase":
        if h_pre_prev[0] is None:
            return y_dev
        q_prev = torch.tanh(h_pre_prev[0].detach() @ P["Pk1"])
        W1 = M._W(P["U1"], P["V1"])
        return W1.detach() @ q_prev
    if kind == "zero":
        return torch.zeros_like(y_dev)
    if kind == "oracle":
        if t == 0:
            return y_dev
        return gs[t - 1].detach().clone()
    return y_dev


# ══════════════════════════════════════════════════════ 盒约束 QP（读数头的输出范围地板）
def box_qp(As, bs, sweeps=SOLVER_SWEEPS, conv=SOLVER_CONV):
    r"""min_{m∈[-1,1]^d} ‖A m + b‖² —— **按实例批处理**（修 DIR-36 的 DEF-36-0b）+ 收敛判据。

    ★ 投影 Gauss-Seidel 对凸二次 + 盒约束**单调不增** ⇒ 收敛到全局最优（$G_{ii}>0$ ⇒ 可解）。
    ★ 与 DIR-36 的差别只有两点：**批处理**（不再逐（实例,$t$）调用）与**收敛提前退出**
      （阈值 1e-14 ⇒ 不改变解，只省时间；精确性由主动集枚举交叉核对 ⇒ A37-5）。
    """
    B, r, d = As.shape
    G = torch.einsum("bij,bik->bjk", As, As)
    diag = torch.diagonal(G, dim1=1, dim2=2).clone()
    m = torch.zeros(B, d, dtype=As.dtype)
    prev = None
    for _ in range(sweeps):
        for i in range(d):
            res = torch.einsum("bij,bj->bi", As, m) + bs
            gi = 2.0 * torch.einsum("bi,bi->b", As[:, :, i], res)
            m[:, i] = torch.clamp(m[:, i] - gi / (2.0 * diag[:, i]), -1.0, 1.0)
        if prev is not None:
            ch = float((m - prev).abs().max())
            if ch < conv:
                break
        prev = m.clone()
    res = torch.einsum("bij,bj->bi", As, m) + bs
    return (res * res).sum(1), m


def box_qp_exact(A, b):
    r"""**精确**解（主动集枚举 $3^d$ 个模式 + KKT 判据）—— 仅用于交叉核对（A37-5）。"""
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
    if best_x is None:
        v, _ = box_qp(A.unsqueeze(0), b.unsqueeze(0))
        return float(v[0]), None
    return best_val, best_x


# ══════════════════════════════════════════════════════ 逐 t 迹线（镜像 R.rollout_task）
def trace(P, read_src, sd, phi, want_recs=True, read_var=None):
    r"""逐 $t$ 迹线：返回 `(L, recs)`。

    ★ **必须与 `R.rollout_task` 逐位等价**（A37-1 钉死）：本函数是它的**镜像**，
      唯一区别是层 1 的读可切换为**参考读**（`read_src="ref"`）并额外记录逐 $t$ 量。
    ★ 参考读只依赖**已写下的配对**（层 1 的前激活与注入无关）⇒ 参考读**不随 $\gamma$ 变**
      ⇒ 换臂是**干净的单变量替换**（DIR-36 的 DEF-36-2 在本装置上不成立）。
    ★ `read_var`（探针）：对**装置读**做**单变量**变换（`scale`/`phase`/`zero`/`oracle`）。
      `read_var=None` 或 `"dev"` ⇒ 走的是**与装置逐字相同**的分支 ⇒
      探针的 `dev` 对照与装置损失**逐位相同**（A37-12）。
    """
    L = P["a"].new_zeros(())
    gs = R.targets(P)
    h_pre_prev = [None, None]
    pairs = []                      # 第 1 层已写下的 (k, v)
    recs = []
    for t in range(T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        h1 = m1 = y1 = gam1 = None
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if li == 0:
                c, q = h, torch.tanh(h @ P["Pk1"])       # oracle 寻址
            else:
                c = hp_prev.detach()
                q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            if li == 0 and read_src == "ref":
                Wr = ref_W(pairs)
                y = (Wr @ q) if Wr is not None else torch.zeros(D_V)
            else:
                y = W.detach() @ q                        # 装置读（与 R.read_W(W,"T",sd) 同）
                if li == 0 and read_var is not None and read_var != "dev":
                    y = _read_variant(read_var, y, P, h_pre_prev, gs, t)
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            gam = P[f"g{l}"]
            hp[li] = h + gam * m
            hp_prev = hp[li]
            if li == 0:
                h1, m1, y1, gam1 = h, m, y, gam
        if t >= 1:
            ell = ((R.pout() @ hp[0] - gs[t - 1]) ** 2).sum()
            L = L + ell
            if want_recs:
                recs.append({"t": t, "h": h1.detach().clone(), "gam": float(gam1),
                             "m": m1.detach().clone(), "y": y1.detach().clone(),
                             "g": gs[t - 1].detach().clone(), "ell": float(ell)})
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
            if li == 0:
                pairs.append((k.detach().clone(), v.detach().clone()))
        upd = {kk: vv.detach() for kk, vv in upd.items()}
        P.update(upd)
        h_pre_prev = h_pre
    return L, recs


def floors(recs):
    r"""由逐 $t$ 记录算盒地板 $L_D$（输出范围地板）与无约束地板 $L_F$。"""
    O = R.pout()
    As = torch.stack([rc["gam"] * O for rc in recs])
    bs = torch.stack([O @ rc["h"] - rc["g"] for rc in recs])
    obj, _ = box_qp(As, bs)
    sol = torch.linalg.lstsq(As, (-bs).unsqueeze(2)).solution.squeeze(2)
    res = torch.einsum("bij,bj->bi", As, sol) + bs
    return float(obj.sum()), float((res * res).sum())


# ══════════════════════════════════════════════════════ 构造与训练
def _mk(sd, phi, train=False):
    P = R.build(sd, "V0")
    R.period2(P, sd)
    if phi:
        P.update({k: (v if train else v.clone()) for k, v in phi.items()})
    return P


def train_head(read_src, seeds, steps=250, lr=0.03, read_var=None):
    r"""训练读数头（协议与 `R.train_arm` 逐字相同：同一初值生成器 4242、Adam、lr、250 步）。

    ★ `read_src="dev"` 时必须与 `R.train_arm("oracle","T",...)` **逐位相同**（A37-1b）。
    ★ `read_var` 非 `None` 时是**探针**：读被单变量变换，训练协议**完全不变**
      ⇒ `read_var="dev"` 与 `read_var=None` **逐位相同**（探针对照的构造性保证）。
    """
    gin = torch.Generator().manual_seed(4242)
    phi = {"g1": torch.full((1,), M.GAMMA, requires_grad=True),
           "Wo1": (torch.randn(D_H + D_V, D_H, generator=gin)
                   / math.sqrt(D_H + D_V)).requires_grad_(True),
           "bo1": (torch.randn(D_H, generator=gin) * 0.1).requires_grad_(True)}
    opt = torch.optim.Adam(list(phi.values()), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        for sd in seeds:
            P = _mk(sd, phi, train=True)
            L, _ = trace(P, read_src, sd, phi, want_recs=False, read_var=read_var)
            (L / len(seeds)).backward()
        opt.step()
    return {k: v.detach().clone() for k, v in phi.items()}


# ══════════════════════════════════════════════════════ 统计
def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    se = sd / len(d) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": 3 * se, "n": len(d)}


def floor_Z(seeds):
    zs = []
    for sd in seeds:
        P = _mk(sd, None)
        zs.append(sum(float((g ** 2).sum()) for g in R.targets(P)))
    return zs


# ══════════════════════════════════════════════════════ 一个档位的主测量
_HEADS = {}          # (train_base, level, kind) -> 训练好的头（避免重复训练）


def measure(level, train_base, eval_seeds):
    r"""三臂：A（装置）、B（装置的冻结头 + 参考读）、C（参考读 + 重训头）。"""
    tr = [train_base + i for i in range(level)]
    out = {"level": level, "train_base": train_base, "n_eval": len(eval_seeds)}
    t0 = time.time()
    phiA = train_head("dev", tr)
    _HEADS[(train_base, level, "dev")] = phiA
    print(f"  训练 A(dev) n={level} {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    phiC = train_head("ref", tr)
    print(f"  训练 C(ref) n={level} {time.time()-t0:.0f}s", flush=True)
    out["gamma"] = {"A": float(phiA["g1"]), "C": float(phiC["g1"])}
    acc = {k: {"L": [], "LD": [], "LF": [], "rell": []} for k in ("A", "B", "C")}
    t0 = time.time()
    for i, sd in enumerate(eval_seeds):
        for arm, (src, phi) in (("A", ("dev", phiA)), ("B", ("ref", phiA)), ("C", ("ref", phiC))):
            P = _mk(sd, phi)
            L, recs = trace(P, src, sd, phi)
            LD, LF = floors(recs)
            acc[arm]["L"].append(float(L))
            acc[arm]["LD"].append(LD)
            acc[arm]["LF"].append(LF)
            e = []
            for rc in recs:
                gn = float(rc["g"].norm())
                if gn > 0:
                    e.append(float((rc["y"] - rc["g"]).norm()) / gn)
            acc[arm]["rell"].append(statistics.fmean(e) if e else 0.0)
        if (i + 1) % 100 == 0:
            print(f"    评估 {i+1}/{len(eval_seeds)}  ({time.time()-t0:.0f}s)", flush=True)
    out["per"] = {}
    for k in ("A", "B", "C"):
        out["per"][k] = {kk: (statistics.fmean(vv) if kk != "L" else None)
                         for kk, vv in acc[k].items()}
        out["per"][k]["mean_L"] = statistics.fmean(acc[k]["L"])
        out["per"][k]["mean_LD"] = statistics.fmean(acc[k]["LD"])
        out["per"][k]["mean_LF"] = statistics.fmean(acc[k]["LF"])
        out["per"][k]["mean_rell"] = statistics.fmean(acc[k]["rell"])
        out["per"][k]["L_series"] = acc[k]["L"]
        out["per"][k]["LD_series"] = acc[k]["LD"]
    return out


def decompose(m):
    r"""四项机制：$G_{\rm inst}=L_A-L_B$、$G_{\rm adapt}=L_B-L_C$、$G_{\rm head}=L_C-L_D$、$G_{\rm range}=L_D-L_F$。"""
    LA = m["per"]["A"]["mean_L"]
    LB = m["per"]["B"]["mean_L"]
    LC = m["per"]["C"]["mean_L"]
    LDA = m["per"]["A"]["mean_LD"]
    LDC = m["per"]["C"]["mean_LD"]
    LFA = m["per"]["A"]["mean_LF"]
    LFC = m["per"]["C"]["mean_LF"]
    LA_s, LB_s, LC_s = m["per"]["A"]["L_series"], m["per"]["B"]["L_series"], m["per"]["C"]["L_series"]
    LD_s = m["per"]["C"]["LD_series"]
    g_inst = paired(LA_s, LB_s)
    g_adapt = paired(LB_s, LC_s)
    g_head = paired(LC_s, LD_s)
    g_range = {"mean": LDC, "sd": (statistics.stdev(LD_s) if len(LD_s) > 1 else 0.0),
               "se": 0.0, "3se": 0.0, "n": len(LD_s)}
    g_range["se"] = g_range["sd"] / len(LD_s) ** 0.5
    g_range["3se"] = 3 * g_range["se"]
    terms = {"inst": g_inst["mean"], "adapt": g_adapt["mean"],
             "head": g_head["mean"], "range": g_range["mean"]}
    # ★ 四项的**逐实例**序列（分离判据要用「配对逐实例差」）
    per_inst = {"inst": [a - b for a, b in zip(LA_s, LB_s)],
                "adapt": [b - c for b, c in zip(LB_s, LC_s)],
                "head": [c - d for c, d in zip(LC_s, LD_s)],
                "range": list(LD_s)}
    dom = max(terms, key=terms.get)
    second = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)[1][0]
    sep = paired(per_inst[dom], per_inst[second])
    resid = (LA - LFC) - (terms["inst"] + terms["adapt"] + terms["head"] + terms["range"])
    return {"L_A": LA, "L_B": LB, "L_C": LC, "L_D_A": LDA, "L_D_C": LDC,
            "L_F_A": LFA, "L_F_C": LFC,
            "g_inst": g_inst, "g_adapt": g_adapt, "g_head": g_head, "g_range": g_range,
            "terms": terms, "dom": dom, "second": second, "sep": sep,
            "identity_resid": resid,
            "read_side_total": terms["inst"] + terms["adapt"]}


# ══════════════════════════════════════════════════════ 固定轨迹探针（机制探针，不进判据）
def probe_eval(phi, eval_seeds, variants=("dev", "scale", "phase", "zero", "oracle")):
    r"""在**冻结轨迹**上评估给定的头（每个变体 = 一次**单变量**读替换）。

    ★ 轨迹不变（层 1 前激活只由输入决定）⇒ 变体之间**只有读不同**。
    ★ `dev` 变体走的分支与装置**逐字相同** ⇒ 其损失与装置损失**逐位相同**（A37-12）。
    """
    out = {}
    for name in variants:
        rv = None if name == "dev" else name
        Ls = []
        for sd in eval_seeds:
            P = _mk(sd, phi)
            L, _ = trace(P, "dev", sd, phi, want_recs=False, read_var=rv)
            Ls.append(float(L))
        out[name] = {"L": statistics.fmean(Ls), "L_series": Ls}
    return out


def run_probe(level, train_base, eval_seeds, variants=None, path=None):
    r"""固定轨迹探针：每个读变体**在自己的读上重训一个头**（卡 §2.2），
    然后在**冻结的装置轨迹**上评估 ⇒ 测的是「头在给定读下能做到多少」。

    ★ 变体 `dev` 复用主测量已训好的装置头（逐位同一）⇒ 对照成立。
    ★ `variants`/`path` 支持**分段并行**（同一档的不同变体分开算，最后合并）。
    """
    variants = list(variants) if variants else ["dev", "scale", "phase", "zero", "oracle"]
    tr = [train_base + i for i in range(level)]
    out = {"level": level, "n_eval": len(eval_seeds)}
    if path and os.path.exists(path):
        out.update(json.load(open(path)))
    for name in variants:
        if isinstance(out.get(name), dict) and "L" in out[name]:
            continue
        t0 = time.time()
        if name == "dev":
            phi = _HEADS.get((train_base, level, "dev"))
            if phi is None:
                phi = train_head("dev", tr)
                _HEADS[(train_base, level, "dev")] = phi
        else:
            phi = train_head("dev", tr, read_var=name)
        ev = probe_eval(phi, eval_seeds, variants=(name,))
        out[name] = ev[name]
        print(f"    探针 {name:6s} L {out[name]['L']:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        if path:
            json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    if all(k in out and isinstance(out[k], dict) and "L" in out[k]
           for k in ("dev", "scale", "phase", "zero", "oracle")):
        span = out["zero"]["L"] - out["oracle"]["L"]
        out["span"] = span
        out["scale_share"] = (out["dev"]["L"] - out["scale"]["L"]) / span if span > 0 else None
        out["ident_share"] = (out["scale"]["L"] - out["phase"]["L"]) / span if span > 0 else None
        out["scale_hyp"] = bool(span > 0 and (out["dev"]["L"] - out["scale"]["L"]) >= 0.5 * span)
        out["ident_hyp"] = bool(span > 0 and (out["scale"]["L"] - out["phase"]["L"]) >= 0.5 * span)
    if path:
        json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    return out


# ══════════════════════════════════════════════════════ Stage A
def stage_a():
    a = {}
    tr = list(range(TRAIN_BASE, TRAIN_BASE + 8))
    phiA = train_head("dev", tr)
    # A37-1：迹线层不扰动轨迹（与 R.rollout_task 逐位相同）
    errs = []
    for sd in EVAL[:12]:
        P = _mk(sd, phiA)
        Lb, _ = trace(P, "dev", sd, phiA)
        Lt = float(R.rollout_task(_mk(sd, phiA), "oracle", "T", sd))
        errs.append(abs(float(Lb) - Lt) / max(abs(Lt), 1.0))
    a["A37-1_rel_max"] = max(errs)
    # A37-1b：训练循环与 R.train_arm 逐位相同
    ph_ref = R.train_arm("oracle", "T", learned=False, seeds=tr, steps=6)
    ph_mine = train_head("dev", tr, steps=6)
    a["A37-1b_train_bitwise"] = max(float((ph_ref[k] - ph_mine[k]).abs().max()) for k in ph_ref)
    # A37-2：参考读在已写键上复现写入值（LS 的定义性质）
    g = torch.Generator().manual_seed(20261112)
    worst = 0.0
    for _ in range(5):
        K = torch.randn(D_K, 6, generator=g) / math.sqrt(D_K)
        Vm = torch.randn(D_V, 6, generator=g)
        W = Vm @ torch.linalg.pinv(K)
        worst = max(worst, float((W @ K - Vm).abs().max()))
    a["A37-2_ls_resid"] = worst
    # A37-3：pinv 解 vs **正规方程解**（良态随机数据 ⇒ $KK^\top$ 可逆）
    diffs = []
    for _ in range(5):
        K = torch.randn(D_K, D_K + 4, generator=g) / math.sqrt(D_K)
        Vm = torch.randn(D_V, D_K + 4, generator=g)
        W1 = Vm @ torch.linalg.pinv(K)                       # Moore–Penrose
        W2 = Vm @ K.t() @ torch.linalg.solve(K @ K.t(), torch.eye(D_K))   # 正规方程
        diffs.append(float((W1 - W2).abs().max()))
    a["A37-3_pinv_gap"] = max(diffs)
    # A37-4：参考读的秩 ≤ 2
    P = _mk(EVAL[0], phiA)
    _, recs = trace(P, "dev", EVAL[0], phiA)
    ranks = []
    for rc in recs:
        ks = []
        # 重建该步已写下的键（与 trace 内部一致）
        Pk = R.build(EVAL[0], "V0")
        R.period2(Pk, EVAL[0])
        H = R.hs_of(Pk)
        n = rc["t"] - 1
        for s in range(1, rc["t"]):
            ks.append(torch.tanh(H[s - 1] @ Pk["Pk1"]))
        if len(ks) >= 2:
            K = torch.stack(ks, 1)
            Vm = torch.stack([torch.tanh(H[s] @ Pk["Pv1"]) for s in range(1, rc["t"])], 1)
            W = Vm @ torch.linalg.pinv(K)
            ranks.append(int(torch.linalg.matrix_rank(W, tol=1e-8).item()))
    a["A37-4_max_rank"] = max(ranks) if ranks else 0
    # A37-5：盒求解器 vs 主动集枚举
    diffs = []
    for _ in range(8):
        A = torch.randn(4, 8, generator=g) / math.sqrt(8)
        b = torch.randn(4, generator=g)
        v1, _ = box_qp(A.unsqueeze(0), b.unsqueeze(0))
        v2, _ = box_qp_exact(A, b)
        diffs.append(abs(float(v1[0]) - v2))
    a["A37-5_solver_gap"] = max(diffs)
    # A37-6：手算工作例
    A1 = torch.eye(2)
    v1, _ = box_qp(A1.unsqueeze(0), torch.tensor([-2.0, -1.0]).unsqueeze(0))
    v0, _ = box_qp(A1.unsqueeze(0), torch.tensor([-0.5, 0.3]).unsqueeze(0))
    a["A37-6_worked_1"] = float(v1[0])
    a["A37-6_worked_2"] = float(v0[0])
    a["A37-6_exact_1"] = box_qp_exact(A1, torch.tensor([-2.0, -1.0]))[0]
    a["A37-6_exact_2"] = box_qp_exact(A1, torch.tensor([-0.5, 0.3]))[0]
    # A37-7：**公平性**（P37-1）—— 参考读在同一指标上不得劣于装置读（A 轨迹）
    re_dev, re_ref = [], []
    for sd in EVAL[:40]:
        P = _mk(sd, phiA)
        _, recs = trace(P, "dev", sd, phiA)
        e = []
        for rc in recs:
            gn = float(rc["g"].norm())
            if gn > 0:
                e.append(float((rc["y"] - rc["g"]).norm()) / gn)
        re_dev.append(statistics.fmean(e) if e else 0.0)
        # 参考读（同一轨迹、同一批已写配对）
        Pk = R.build(sd, "V0")
        R.period2(Pk, sd)
        H = R.hs_of(Pk)
        _, recs2 = trace(P, "ref", sd, phiA)
        e2 = []
        for rc in recs2:
            gn = float(rc["g"].norm())
            if gn > 0:
                e2.append(float((rc["y"] - rc["g"]).norm()) / gn)
        re_ref.append(statistics.fmean(e2) if e2 else 0.0)
    md_, mr_ = statistics.fmean(re_dev), statistics.fmean(re_ref)
    a["A37-7_rell_dev"] = md_
    a["A37-7_rell_ref"] = mr_
    a["A37-7_fair_ok"] = 1.0 if mr_ <= md_ + 1e-12 else 0.0
    # A37-8：md5 双守卫
    a["A37-8_device_diff"] = 0.0 if _md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py") == MD5_DEVICE else 1.0
    a["A37-8_task_diff"] = 0.0 if _md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py") == MD5_TASK else 1.0
    # A37-9：种子隔离
    own = list(range(TRAIN_BASE, TRAIN_BASE + 384)) + EVAL + CALIB_EVAL + CALIB_TRAIN
    hits = sorted({n for n, (lo, hi) in HIST_RANGES.items() for s in own if lo <= (s - SEED0) <= hi})
    a["A37-9_hist_hits"] = ",".join(hits) if hits else "none"
    a["A37-9_seed_disjoint"] = 0.0 if not hits else 1.0
    a["A37-9_train_eval_overlap"] = float(len(set(range(TRAIN_BASE, TRAIN_BASE + 384)) & set(EVAL)))
    # A37-10：训练确定性
    p1 = train_head("dev", tr[:4], steps=6)
    p2 = train_head("dev", tr[:4], steps=6)
    a["A37-10_train_bitwise"] = max(float((p1[k] - p2[k]).abs().max()) for k in p1)
    # A37-11：卡已冻结
    a["A37-11_card_md5"] = _md5(CARD_PATH)
    a["A37-11_card_diff"] = 0.0 if a["A37-11_card_md5"] == CARD_MD5 else 1.0
    a["A37-11_threshold_set"] = 0.0 if (D_MIN_FRAC is not None) else 1.0
    # A37-12：**探针对照** —— V_dev 的损失必须与装置损失逐位相同
    phi_dev = train_head("dev", tr)
    dev_losses = []
    for sd in EVAL[:8]:
        P = _mk(sd, phi_dev)
        L, _ = trace(P, "dev", sd, phi_dev, want_recs=False)
        dev_losses.append(float(L))
    phi_probe = train_head("dev", tr, read_var="dev")
    a["A37-12_probe_head_gap"] = max(float((phi_dev[k] - phi_probe[k]).abs().max()) for k in phi_dev)
    pr = probe_eval(phi_probe, EVAL[:8], variants=("dev",))
    a["A37-12_probe_dev_rel"] = abs(pr["dev"]["L"] - statistics.fmean(dev_losses)) / max(abs(statistics.fmean(dev_losses)), 1.0)
    # A37-13：恒等式自洽（用标定级的小样本检验）
    mm = measure(8, TRAIN_BASE, EVAL[:6])
    dd = decompose(mm)
    a["A37-13_identity_rel"] = abs(dd["identity_resid"]) / max(abs(dd["L_A"]), 1.0)
    # A37-14：盒地板是真地板
    viol = 0
    O = R.pout()
    for sd in EVAL[:20]:
        P = _mk(sd, phiA)
        _, recs = trace(P, "dev", sd, phiA)
        for rc in recs:
            A = float(rc["gam"]) * O
            b = O @ rc["h"] - rc["g"]
            v, _ = box_qp(A.unsqueeze(0), b.unsqueeze(0))
            if float(v[0]) > rc["ell"] + 1e-9:
                viol += 1
    a["A37-14_violations"] = float(viol)
    ok = (a["A37-1_rel_max"] <= REL_TOL and a["A37-1b_train_bitwise"] == 0.0
          and a["A37-2_ls_resid"] <= 1e-9 and a["A37-3_pinv_gap"] <= 1e-8
          and a["A37-4_max_rank"] <= 2 and a["A37-5_solver_gap"] <= BOX_TOL
          and abs(a["A37-6_worked_1"] - 1.0) <= 1e-9 and abs(a["A37-6_worked_2"]) <= 1e-9
          and abs(a["A37-6_exact_1"] - 1.0) <= 1e-9 and abs(a["A37-6_exact_2"]) <= 1e-9
          and a["A37-7_fair_ok"] == 1.0
          and a["A37-8_device_diff"] == 0.0 and a["A37-8_task_diff"] == 0.0
          and a["A37-9_seed_disjoint"] == 0.0 and a["A37-9_train_eval_overlap"] == 0.0
          and a["A37-10_train_bitwise"] == 0.0
          and a["A37-11_card_diff"] == 0.0 and a["A37-11_threshold_set"] == 0.0
          and a["A37-12_probe_dev_rel"] <= REL_TOL
          and a["A37-13_identity_rel"] <= 1e-9
          and a["A37-14_violations"] == 0.0)
    return ok, a


# ══════════════════════════════════════════════════════ 标定（决定二十九，不进判据）
def calib():
    t0 = time.time()
    print("── 标定（机制代码的可执行性；不进判据）")
    m = measure(24, CALIB_TRAIN[0], CALIB_EVAL)
    d = decompose(m)
    m["decompose"] = d
    pr = run_probe(24, CALIB_TRAIN[0], CALIB_EVAL)
    m["probe"] = pr
    m["secs"] = round(time.time() - t0, 1)
    print(f"  L_A {d['L_A']:.5f} · L_B {d['L_B']:.5f} · L_C {d['L_C']:.5f} · L_D {d['L_D_C']:.5f}")
    print(f"  G_inst {d['g_inst']['mean']:+.5f} · G_adapt {d['g_adapt']['mean']:+.5f} · "
          f"G_head {d['g_head']['mean']:+.5f} · G_range {d['g_range']['mean']:+.5f}")
    print(f"  主导 {d['dom']} · 读侧合计 {d['read_side_total']:+.5f} · 恒等式残差 {d['identity_resid']:.2e}")
    print(f"  探针 span {pr['span']:.5f} · scale_share {pr['scale_share']} · ident_share {pr['ident_share']}")
    json.dump(m, open(f"{OUT}/DIR37_calib.json", "w"), ensure_ascii=False, indent=1)
    return m


# ══════════════════════════════════════════════════════ 复现锚（纪律 84）
def repro():
    r"""用 DIR-36 的**精确**配置跑 A 臂 ⇒ 必须逐位复现 D-60 的 $L_A$。

    ★ 两个读数（纪律 84 + 卡 §2.2 的探针纪律「复现 D-60 的 ≥2 个读数」）：
      ① $n_{\rm train}=384$：$L_A=8.257212940724042$；
      ② 次级档 $n_{\rm train}=96$（同一训练段前 96 个、同一评估段）：$L_A=9.421045543145368$。
    """
    t0 = time.time()
    print("── 复现锚（train 8000..8383 / eval 7000..7399，n_train=384，臂 A）")
    tr = [ANCHOR_TRAIN + i for i in range(384)]
    phi = train_head("dev", tr)
    lo = []
    for sd in ANCHOR_EVAL:
        P = _mk(sd, phi)
        L, _ = trace(P, "dev", sd, phi, want_recs=False)
        lo.append(float(L))
    out = {"L_A": statistics.fmean(lo), "expect": ANCHOR_EXPECT,
           "secs": round(time.time() - t0, 1)}
    e = abs(out["L_A"] - ANCHOR_EXPECT["L_A"]) / max(abs(ANCHOR_EXPECT["L_A"]), 1.0)
    out["rel_err"] = e
    # 第二个读数：次级档 n=96（同一评估段）
    t1 = time.time()
    phi96 = train_head("dev", tr[:96])
    lo96 = []
    for sd in ANCHOR_EVAL:
        P = _mk(sd, phi96)
        L, _ = trace(P, "dev", sd, phi96, want_recs=False)
        lo96.append(float(L))
    out["L_A_96"] = statistics.fmean(lo96)
    e96 = abs(out["L_A_96"] - ANCHOR_EXPECT["L_A_96"]) / max(abs(ANCHOR_EXPECT["L_A_96"]), 1.0)
    out["rel_err_96"] = e96
    out["secs_96"] = round(time.time() - t1, 1)
    out["ok"] = bool(e <= ANCHOR_TOL and e96 <= ANCHOR_TOL)
    print(f"  L_A {out['L_A']:.12f}（期望 {ANCHOR_EXPECT['L_A']:.12f}，相对差 {e:.2e}）")
    print(f"  L_A(n=96) {out['L_A_96']:.12f}（期望 {ANCHOR_EXPECT['L_A_96']:.12f}，相对差 {e96:.2e}）")
    print(f"  ⇒ 锚 {'通过' if out['ok'] else '★未通过'}（{out['secs']:.0f}s + {out['secs_96']:.0f}s）")
    json.dump(out, open(f"{OUT}/DIR37_repro.json", "w"), indent=1)
    return out


def _pearson(x, y):
    n = len(x)
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


# ══════════════════════════════════════════════════════ 判据计算
def attribution(d, dmin):
    r"""卡 §2.1 的**预注册**决策表：四项 ⇒ 机制 ⇒ 下一个杠杆。"""
    t = d["terms"]
    if d["read_side_total"] >= dmin and d["read_side_total"] >= max(t["head"], t["range"]):
        return ("读侧（检索 + 头的适应）", "写/读侧：写规则、寻址、键可分性（**装置/协议** revision）")
    if t["head"] >= dmin and t["head"] >= d["read_side_total"]:
        return ("读数头的函数逼近", "读数头的结构（**任务层** revision）")
    if t["range"] >= dmin:
        return ("读数头的输出范围", "γ / tanh 饱和 / 归一化（**任务层** revision）")
    if t["inst"] >= dmin:
        return ("读的瞬时质量", "写/读侧（**装置/协议** revision）")
    if t["adapt"] >= dmin:
        return ("头对读分布的适应", "读数头的训练/结构（**任务层** revision）")
    return ("**归因失败**（四项都 < Δmin）", "差距不在这四者 ⇒ 不得硬凑")


def finalize(res):
    r"""由已测的档位算条款、归因与**判据状态**（正式轮与分段合并共用同一段逻辑）。"""
    lev = res["levels"]["384"]
    d = lev["decompose"]
    Z_ev = statistics.fmean(floor_Z(EVAL))
    dmin = D_MIN_FRAC * Z_ev
    res["Z_eval"] = Z_ev
    res["dmin"] = dmin
    terms = d["terms"]
    tol_mono = MONO_TOL_FRAC * Z_ev
    res["clauses"] = {}
    res["clauses"]["C37-a"] = {"max_term": max(terms.values()), "dmin": dmin, "terms": terms,
                               "ok": max(terms.values()) >= dmin}
    res["clauses"]["C37-b"] = {"dom": d["dom"], "second": d["second"],
                               "diff": abs(d["sep"]["mean"]), "3se": d["sep"]["3se"],
                               "ok": abs(d["sep"]["mean"]) >= d["sep"]["3se"]}
    res["clauses"]["C37-c"] = {"identity_resid": d["identity_resid"],
                               "L_F": d["L_F_C"], "L_D": d["L_D_C"], "L_C": d["L_C"], "L_A": d["L_A"],
                               "ok": (abs(d["identity_resid"]) <= 1e-9 * max(abs(d["L_A"]), 1.0)
                                      and d["L_F_C"] <= d["L_D_C"] + 1e-12)}
    res["clauses"]["C37-d"] = {"tol": tol_mono, "terms": terms,
                               "ok": all(v >= -tol_mono for v in terms.values())}
    # P37-1（公平性）在正式轮重查
    res["P37-1"] = {"rell_dev": lev["per"]["A"]["mean_rell"], "rell_ref": lev["per"]["C"]["mean_rell"],
                    "ok": lev["per"]["C"]["mean_rell"] <= lev["per"]["A"]["mean_rell"] + 1e-12}
    ok = all(c["ok"] for c in res["clauses"].values())
    if not res["P37-1"]["ok"]:
        verdict = "PREMISE-NOT-MET"
    elif not res["clauses"]["C37-b"]["ok"]:
        verdict = "INFEASIBLE-POWER"
    elif ok:
        verdict = "SUCCEEDED"
    else:
        verdict = "FALSIFIED"
    res["verdict"] = verdict
    res["attribution"] = attribution(d, dmin)
    for k, c in res["clauses"].items():
        print(f"  {k}: {c} ⇒ {'过' if c['ok'] else '不过'}")
    print(f"  P37-1: {res['P37-1']}")
    print(f"  归因：{res['attribution']}")
    print(f"⇒ 判据状态 **{verdict}**")
    return res


def formal_main():
    r"""主测量（两档 × 三臂）+ 条款（**不含探针**）⇒ 写 `DIR37r10_formal_main.json`。"""
    t0 = time.time()
    if D_MIN_FRAC is None or CARD_MD5 is None:
        print("★ 阈值/卡 md5 未冻结 ⇒ 拒绝跑正式轮")
        sys.exit(2)
    rp = f"{OUT}/DIR37_repro.json"
    if not os.path.exists(rp):
        print("★ 找不到复现锚记录 ⇒ 拒绝跑正式轮（纪律 84）")
        sys.exit(2)
    anch = json.load(open(rp))
    if not anch.get("ok"):
        print("★ 复现锚未通过 ⇒ 拒绝跑正式轮（MEASUREMENT-NOT-CREDIBLE）")
        sys.exit(2)
    res = {"D_MIN_FRAC": D_MIN_FRAC, "CARD_MD5": CARD_MD5, "anchor": anch, "levels": {}}
    for level in (384, 96):
        print(f"── 档位 n_train={level}")
        m = measure(level, TRAIN_BASE, EVAL)
        d = decompose(m)
        m["decompose"] = d
        res["levels"][str(level)] = m
        print(f"  L_A {d['L_A']:.5f} · L_B {d['L_B']:.5f} · L_C {d['L_C']:.5f} · L_D {d['L_D_C']:.5f}")
        print(f"  G_inst {d['g_inst']['mean']:+.5f} · G_adapt {d['g_adapt']['mean']:+.5f} · "
              f"G_head {d['g_head']['mean']:+.5f} · G_range {d['g_range']['mean']:+.5f}")
        print(f"  主导 {d['dom']} · 读侧合计 {d['read_side_total']:+.5f} · "
              f"恒等式残差 {d['identity_resid']:.2e}")
    res["secs"] = round(time.time() - t0, 1)
    return res


def formal_probe(lv, variants=None):
    r"""单个探针档的**一组**变体（**机制**，不进判据）⇒ 写 `DIR37r10_probe_{lv}.<变体集>.json`。

    ★ 分段并行：每个进程只算自己那几个变体、只写自己那个文件 ⇒ **无写冲突**；
      `formal_merge` 再把同档的多个文件合并（变体字典按名字合并）。
    """
    vs = list(variants) if variants else ["dev", "scale", "phase", "zero", "oracle"]
    t0 = time.time()
    print(f"── 探针档 n_train={lv} 变体 {','.join(vs)}")
    path = f"{OUT}/DIR37r10_probe_{lv}.{'_'.join(vs)}.json"
    pr = run_probe(lv, TRAIN_BASE, EVAL, variants=vs, path=path)
    pr["secs"] = round(time.time() - t0, 1)
    for name in vs:
        print(f"  探针 {name:6s} L {pr[name]['L']:.5f}")
    print(f"  ⇒ {path}  ({pr['secs']:.0f}s)")
    return pr


def formal_merge():
    r"""把主测量与三个探针档合并 ⇒ 写正式的 `DIR37r10_formal.json`（含判据与机制）。"""
    mp = f"{OUT}/DIR37r10_formal_main.json"
    if not os.path.exists(mp):
        print("★ 找不到主测量记录 ⇒ 拒绝合并")
        sys.exit(2)
    res = json.load(open(mp))
    res["probe"] = {}
    for lv in PROBE_LEVELS:
        merged = {"level": lv, "n_eval": len(EVAL)}
        parts = sorted(glob.glob(f"{OUT}/DIR37r10_probe_{lv}.*.json"))
        if not parts:
            print(f"★ 找不到探针档 {lv} ⇒ 拒绝合并")
            sys.exit(2)
        for p in parts:
            d = json.load(open(p))
            for name, val in d.items():
                if isinstance(val, dict) and "L" in val:
                    merged[name] = val
        if not all(k in merged for k in ("dev", "scale", "phase", "zero", "oracle")):
            print(f"★ 探针档 {lv} 的变体不全（{sorted(k for k in merged if k in ('dev','scale','phase','zero','oracle'))}）⇒ 拒绝合并")
            sys.exit(2)
        span = merged["zero"]["L"] - merged["oracle"]["L"]
        merged["span"] = span
        merged["scale_share"] = (merged["dev"]["L"] - merged["scale"]["L"]) / span if span > 0 else None
        merged["ident_share"] = (merged["scale"]["L"] - merged["phase"]["L"]) / span if span > 0 else None
        merged["scale_hyp"] = bool(span > 0 and (merged["dev"]["L"] - merged["scale"]["L"]) >= 0.5 * span)
        merged["ident_hyp"] = bool(span > 0 and (merged["scale"]["L"] - merged["phase"]["L"]) >= 0.5 * span)
        merged["parts"] = [os.path.basename(p) for p in parts]
        res["probe"][str(lv)] = merged
        print(f"  探针档 {lv}: " + " · ".join(f"{n} {merged[n]['L']:.5f}"
              for n in ("dev", "scale", "phase", "zero", "oracle"))
              + f" · span {span:.5f} · scale_hyp {merged['scale_hyp']} · ident_hyp {merged['ident_hyp']}")
    lv_keys = sorted(res["probe"].keys(), key=int)
    dev_ls = [res["probe"][k]["dev"]["L"] for k in lv_keys]
    sp_ls = [res["probe"][k]["span"] for k in lv_keys]
    n_scale = sum(1 for k in lv_keys if res["probe"][k]["scale_hyp"])
    n_ident = sum(1 for k in lv_keys if res["probe"][k]["ident_hyp"])
    try:
        r_ds = _pearson(dev_ls, sp_ls)
    except Exception:
        r_ds = None
    res["probe_consistency"] = {"levels": lv_keys, "dev_L": dev_ls, "span": sp_ls,
                                "scale_hyp_levels": n_scale, "ident_hyp_levels": n_ident,
                                "pearson_dev_span": r_ds,
                                "note": "描述项：跨三档的一致性，不进判据"}
    finalize(res)
    res["secs_total"] = res.get("secs", 0)
    json.dump(res, open(f"{OUT}/DIR37r10_formal.json", "w"), ensure_ascii=False, indent=1)
    print(f"⇒ 已写出 {OUT}/DIR37r10_formal.json")
    return res


def formal():
    r"""**串行**全量（主测量 + 三档探针 + 合并）—— 与分段运行等价。"""
    res = formal_main()
    json.dump(res, open(f"{OUT}/DIR37r10_formal_main.json", "w"), ensure_ascii=False, indent=1)
    res["probe"] = {}
    for lv in PROBE_LEVELS:
        res["probe"][str(lv)] = formal_probe(lv)
    finalize(res)
    res["secs"] = round(res.get("secs", 0), 1)
    json.dump(res, open(f"{OUT}/DIR37r10_formal.json", "w"), ensure_ascii=False, indent=1)
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
            print(f"  {k:28s} {v if isinstance(v, str) else f'{v:.3e}'}")
        json.dump({"ok": ok, "a": a}, open(f"{OUT}/DIR37r10_stageA.json", "w"), indent=1)
        sys.exit(0 if ok else 1)
    elif mode == "smoke":
        t1 = time.time()
        m = measure(8, TRAIN_BASE, EVAL[:4])
        d = decompose(m)
        ok = all(math.isfinite(v) for v in (d["L_A"], d["L_B"], d["L_C"], d["L_D_C"]))
        print(f"smoke：n_train=8 × 4 评估实例 ⇒ {'可执行' if ok else '★ 不可执行'}  ({time.time()-t1:.0f}s)")
        print(f"  L_A {d['L_A']:.5f} · L_B {d['L_B']:.5f} · L_C {d['L_C']:.5f} · L_D {d['L_D_C']:.5f}")
        json.dump({"ok": ok, "decompose": d}, open(f"{OUT}/DIR37r10_smoke.json", "w"), indent=1)
        sys.exit(0 if ok else 1)
    elif mode == "formal_main":
        res = formal_main()
        finalize(res)
        json.dump(res, open(f"{OUT}/DIR37r10_formal_main.json", "w"), ensure_ascii=False, indent=1)
        print(f"  （主测量完成 {res['secs']:.0f}s）")
    elif mode == "formal_probe":
        lv = int(sys.argv[2]) if len(sys.argv) > 2 else 384
        vs = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        formal_probe(lv, vs)
    elif mode == "formal_merge":
        formal_merge()
    else:
        formal()
