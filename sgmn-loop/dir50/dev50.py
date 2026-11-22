#!/opt/pytorch/bin/python
r"""DIR-50 · **两层装置**（层 1 / 层 2 = 两个「模态通道」）—— 逐字派生自 `dir49/R1_task49.py`，
**只新增**：① `dir31/R1_grad.rollout` 的**两层接线**（层 2 的输入 = 层 1 的**注入后**隐状态 ⇒ 含层 1 的记忆读出）；
② **键空间内构造的干扰键**（剂量 `cosθ` **精确可控**、范数与自然键**逐位匹配**）。

★ **血统（逐字不改）**：
- 写动力学：`dir48/R1_task48.step_pair`（`N_REP=16`、`α`、`λ=0.05`、`η=0.2`）—— **一字不改**；
- 读数：`dir48/R1_task48.read_target` —— **一字不改**；
- 内容流：`dir46/R1_task46.content(sd, n, "perstep")`（`NMARK=36` 固定长度 ⇒ 治 `DEF-48-4`）—— **一字不改**；
- 装置参数：`dir33/R1_task33.build`（⇒ `dir31/R1_grad.build` 的**两层**参数 + `KEY_SCALE=0.5`）—— **一字不改**；
- 两层前向接线：`dir31/R1_grad.rollout` 的 `for li, l in enumerate((1,2))` 段 —— **逐字照搬**。

★ **两层的结构性事实（本卡的全部机制基础）**：
- **层 1 的 `(k,v)` 流与层 1 的记忆**无关**：`k_1[t] = tanh(h_pre_1[t−1] @ Pk1)`、`h_pre_1[t] = tanh(a[t] @ A1 + b1)`
  ⇒ 层 1 的键/值**只依赖输入 `a`** ⇒ 层 1 的记忆状态**不反馈**回层 1 的写入流；
- **层 2 的 `(k,v)` 流依赖层 1 的记忆**：层 2 的输入 `hp_1[t] = h_pre_1[t] + γ_1·m_1[t]`，
  而 `m_1[t]` 含 `y_1[t] = sg(W_1) q_1[t]` ⇒ ★ **写层 1 ⇒ 改变层 1 的读出 ⇒ 改变层 2 的 `(k,v)`**（**跨模态级联通道**）；
- ★ **耦合是单向的**：层 2 的记忆**不**影响层 1 ⇒ 臂 `hit2` 的**层 1 轨迹与 `none` 逐位相同**（**Stage A 的结构不变量**）。

用法：`dev50.rollout(P, a, arm, dose)`；`arm ∈ {none, hit1, hit2}`；`dose ∈ {None, 0.0, 1.0}`（`cosθ`）。
"""
import math
import os
import sys

import torch

torch.set_default_dtype(torch.float64)

ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task46 as T46      # noqa: E402
import R1_task48 as T48      # noqa: E402

# ══════ 装置常数（沿用 DIR-49；**开卡后一字不改**）
D_H, D_K, D_V = M.D_H, M.D_K, M.D_V
N_REP = T48.N_REP            # 16
ALPHA, LAM, ETA = M.ALPHA, M.LAM, M.ETA     # 1.0 / 0.05 / 0.2
CTX = 3                      # 上下文步数
T_TGT = CTX                  # 3：目标步
T_HIT = CTX + 1              # 4：干扰写所在步
W_AFTER = 7                  # 干扰后的自然步数（step 5..11）
T_STEPS = CTX + 1 + 1 + W_AFTER              # 12
NMARK = T48.NMARK            # 36
ARMS = ("none", "hit1", "hit2")
DOSES = (0.0, 1.0)           # `cosθ`：0.0 = 正交、1.0 = 与目标键同向


def content(sd):
    """内容流（**逐字**沿用 `dir46`）。"""
    return T46.content(sd, NMARK, "perstep")


def step_pair(U, Vm, k, v, alpha):
    """一次「对」的写入（**逐字**沿用 `dir48.step_pair`）。"""
    return T48.step_pair(U, Vm, k, v, alpha)


def read_target(U, Vm, k, v):
    """纯读（**逐字**沿用 `dir48.read_target`）：返回 `(cos, ‖W‖)`。"""
    return T48.read_target(U, Vm, k, v)


def _ortho_unit(u, seed):
    r"""取一个与单位向量 `u` **正交**的单位向量（Gram–Schmidt + 种子化随机向量）。"""
    g = torch.Generator().manual_seed(int(seed))
    r = torch.randn(u.numel(), generator=g, dtype=u.dtype)
    r = r - (r @ u) / (u @ u) * u
    n = float(r.norm())
    if n < 1e-12:
        r = torch.zeros_like(r)
        r[0] = 1.0
        r = r - (r @ u) / (u @ u) * u
        n = float(r.norm())
    return r / n


def construct_key(k_tgt, k_nat, cos_theta, seed):
    r"""构造干扰键：`|cos(k_int, k_tgt)| = cosθ`（**精确**）、`‖k_int‖ = ‖k_nat‖`（**逐位匹配**）。

    ★ `k_int = (cosθ·û_tgt + sinθ·û_⊥) · ‖k_nat‖`，其中 `û_tgt = k_tgt/‖k_tgt‖`、
      `û_⊥ ⊥ û_tgt`（种子化随机）⇒ **剂量精确、尺度匹配**，且**与内容流无关** ⇒
      **四臂的内容流逐位相同**（只改**写入的键**）。
    ★ 边界（卡 §8）：构造键**不在** `tanh` 自然流形上 ⇒ 本卡是**写侧机制探针**，不是内容级扰动。
    """
    ut = k_tgt / float(k_tgt.norm())
    up = _ortho_unit(ut, seed)
    kt = cos_theta * ut + math.sqrt(max(0.0, 1.0 - cos_theta ** 2)) * up
    return kt * float(k_nat.norm())


def rollout(P, a, arm="none", dose=None, W_after=W_AFTER, record_writes=False):
    r"""两层 rollout。返回 dict：

    - `tgt[l]`：层 `l` 的**目标对** `(k, v)`（在目标步捕获，逐字等于写入值）；
    - `c0[l]`：目标步之后的目标余弦；
    - `curve[l][d−1]`：`d = 1..W_after`，干扰步之后第 `d` 步的目标余弦；
    - `k_int`、`v_int`、`ov_k`、`res`：干扰对（`arm="none"` 时为 `None`）；
    - `y1`：层 1 读出 `y_1` 的轨迹（`W_after+1` 个，step 4 起），供**中介量**计算。
    """
    assert arm in ARMS
    assert (arm == "none") == (dose is None)
    U = [P["U1"].clone(), P["U2"].clone()]
    Vm = [P["V1"].clone(), P["V2"].clone()]
    h_pre_prev = [None, None]
    tgt = [None, None]
    c0 = [None, None]
    curve = [[], []]
    y1 = []
    k_int = v_int = None
    ov_k = None
    res = None
    med = {"y1": [], "h1p": [], "h2pre": [], "k2": [], "v2": []}   # ★ 中介衰减链（step T_HIT 起）
    writes = [] if record_writes else None      # ★ 仅**记录**，不改动动力学（供 Stage A 的「写入流逐位相同」核）
    for t in range(T_STEPS):
        # ── 前向（`dir31.rollout` 的两层段，逐字照搬）
        hp_prev = a[t]
        h_pre, hp, yv = [None, None], [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            c = (a[t] if l == 1 else hp_prev).detach()      # $c=\operatorname{sg}(h_{l-1,t})$
            q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            Wm = M._W(U[li], Vm[li])
            y = Wm.detach() @ q                              # $y=\operatorname{sg}(W)q$
            yv[li] = y
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            hp[li] = h + P[f"g{l}"] * m                       # $h^+=h+\gamma m$
            hp_prev = hp[li]
        # ── 写入（`dir31.rollout` 的写段，逐字照搬；`α` 显式化沿用 `dir48`）
        for li, l in enumerate((1, 2)):
            hprev = h_pre_prev[li]
            if hprev is None:
                hprev = torch.zeros(D_H)
            k = torch.tanh(hprev.detach() @ P[f"Pk{l}"])     # $k=\operatorname{sg}(P_k(h_{t-1}))$
            v = torch.tanh(h_pre[li].detach() @ P[f"Pv{l}"])  # $v=\operatorname{sg}(P_v(h_t))$
            if t == T_HIT and arm != "none":
                li_hit = 0 if arm == "hit1" else 1
                if li == li_hit:
                    k_nat = k
                    k = construct_key(tgt[li][0], k_nat, dose, 1000 + 7919 * t + li)
                    v_int = v
                    k_int = k.clone()
                    ov_k = abs(float(k @ tgt[li][0]) / (float(k.norm() * tgt[li][0].norm()) + 1e-300))
                    res = float((M._W(U[li], Vm[li]) @ k - v).norm())
            if t >= T_HIT and li == 1:
                med["k2"].append(k.clone()); med["v2"].append(v.clone())
            if record_writes:
                writes.append((t, li, k.clone(), v.clone()))
            U[li], Vm[li] = step_pair(U[li], Vm[li], k, v, ALPHA)
            if t == T_TGT:
                tgt[li] = (k.clone(), v.clone())
        # ── 读数
        if t == T_TGT:
            for li in (0, 1):
                c0[li] = read_target(U[li], Vm[li], *tgt[li])[0]
        elif t > T_TGT:
            d = t - T_TGT
            if d <= W_after:
                for li in (0, 1):
                    curve[li].append(read_target(U[li], Vm[li], *tgt[li])[0])
        if t >= T_HIT:
            y1.append(yv[0].clone())
            med["y1"].append(yv[0].clone())
            med["h1p"].append(hp[0].clone())
            med["h2pre"].append(h_pre[1].clone())
        h_pre_prev = h_pre
    return {"tgt": tgt, "c0": c0, "curve": curve, "k_int": k_int, "v_int": v_int,
            "ov_k": ov_k, "res": res, "y1": y1, "med": med, "writes": writes}


def rel_curve(rr, li):
    r"""层 `li` 的相对存活曲线 `rel(d) = c(d)/c0`（`d = 1..W_after`）；`c0 ≤ 0` 或非有限 ⇒ 该点为 `None`。"""
    c0 = rr["c0"][li]
    out = []
    for c in rr["curve"][li]:
        if math.isfinite(c0) and c0 > 0 and math.isfinite(c):
            out.append(c / c0)
        else:
            out.append(None)
    return out


def build(sd):
    """装置参数（**逐字**沿用 `dir33.build`，两层）。"""
    return R.build(sd, "V0")
