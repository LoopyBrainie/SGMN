#!/usr/bin/env python
r"""DIR-31 · 集成接口验收（一）：**梯度隔离** —— 最小可微实例（torch）

方向出处：`framework/SGMN_v2.4_架构设计.md` §7.2「集成接口验收」行（四项：**前向记忆敏感性**、
**读取参数的非零有限梯度**、**任务梯度不进入写入通道**、**冻结基座不发生意外更新**）
+ §7.1 C「C 是**必须满足的接口定义**，不是『已通过 EXP-03』的事实陈述」
+ 第 218–219 行「$\gamma_l=0$ 时…$\nabla_{\phi_l}\mathcal L=0$；$\gamma_l$ 自身是否能脱离零点取
  决于 $\langle\delta^+_{l,t},m_{l,t}\rangle$」+「**非零梯度、有限梯度和记忆扰动确实改变输出，
  都不是 stop-gradient 定义本身能够保证的**，仍须由 EXP-03 类接口实验验收」
+ 第 219 行「**参数冻结与输入梯度是否保留是两件事**」。

装置血统：写入侧**逐字**取自 §2.3/§3.3（$k=\operatorname{sg}(P_k(h_{t-1}))$、
$v=\operatorname{sg}(P_v(h_t))$、$W=W^0+UV^\top$、$G=\alpha(Wk-v)k^\top+\lambda(W-W^0)$、
同步低秩步 $U\leftarrow U-\eta GV$、$V\leftarrow V-\eta G^\top U$），即 `dir30/R1_init.py` 标量
实例在 $d_k,d_v>1$ 下的直译；**本方向唯一新增的是测量层**（读取头 $\mathcal Q,\mathcal O,\gamma$
＋任务损失＋autograd 探针）。

用法：pilot | stageA | smoke | formal
"""
import json
import math
import os
import sys
import time

import torch

# ★ bit-exact 断言需要双精度；float32 下「恰好为 0」与「数值噪声」不可区分。
torch.set_default_dtype(torch.float64)

# ══════════════════════════════════════ 冻结常数（卡 §3 / §5）
D_H, D_K, D_V, R = 8, 4, 4, 2     # 隐藏/键/读取结果维与低秩
T_STEPS = 8                       # 每实例在线步数（BPTT 深度）
N_INST = 24                       # 实例数（纪律 67：多工作点 × 多任务实例）
SEED0 = 20261111
ALPHA = 1.0                       # 局部目标数据项门控（§3.1）
LAM = 0.05                        # 局部目标基准回归（§3.1）
ETA = 0.05                        # 局部步长（构型标定 sweep31.py：写通道活性）
GAMMA = 0.5                       # 注入强度（参考构型）
INIT_SD = 0.5                     # U,V 初值尺度（构型标定：ρ_mem≈0.16，记忆可辨但未主导）
LR_OPT = 0.05                     # C4a 行为检验的优化器步长
OUT = "/home/ec2-user/sgmn-loop/dir31/out"

# ── 冻结阈值（卡 §1.1 / §1.2 / §5；pilot + sweep31.py 标定后冻结，此后不可改）──
RHO_MIN = 0.05        # C1 前向记忆敏感性下界（实测 min 0.075，24 实例）
GREAD_MIN = 1e-3      # C2 读取参数梯度下界（实测 min 4.77e-2）
GIN_MIN = 1e-3        # C4b 输入梯度下界（实测 min 1.756）
ALIVE_MIN = 0.01      # P1 写通道活性下界（实测 min 0.0475）
FD_TOL = 1e-6         # A31-3：T=1 截断图上 autograd vs 中心差分
WORK_TOL = 1e-14      # A31-1/A31-2：工作例 vs 手算表
GRAD_TOL = 1e-10      # A31-1(ii)：G 是否为局部目标的梯度

# ── 冻结的变体判别表（卡 §1.4）：变体 × 子句 ⇒ 预测（1=PASS, 0=FAIL）──
PREDICT = {
    "V0": dict(C1=1, C2=1, C3=1, C4a=1, C4b=1),
    "V1": dict(C1=1, C2=1, C3=0, C4a=1, C4b=1),
    "V2": dict(C1=1, C2=0, C3=1, C4a=1, C4b=1),
    "V3": dict(C1=1, C2=0, C3=1, C4a=1, C4b=0),
    "V4": dict(C1=0, C2=0, C3=1, C4a=1, C4b=1),
    "V5": dict(C1=1, C2=1, C3=1, C4a=0, C4b=1),
}
CLAUSES = ("C1", "C2", "C3", "C4a", "C4b")

VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5")
VARIANT_DOC = {
    "V0": "参考实现（正确）",
    "V1": "读路去掉 sg（$y=Wq$）⇒ 任务梯度应进入写通道",
    "V2": "把 $\\mathcal Q$ 的**输出** detach（而非输入 $c$）⇒ $\\phi^q$ 应断流",
    "V3": "基座前向包在 `torch.no_grad()`（最常见的省显存写法）⇒ 输入梯度应消失",
    "V4": "$\\gamma=0$（§3.4 退化点）⇒ 前向应无记忆敏感性、读取参数梯度应为 0",
    "V5": "忘了冻结基座（$\\theta$ 仍 requires_grad）⇒ 基座参数应拿到梯度并被更新",
}

READ_KEYS = tuple(f"{p}{l}" for l in (1, 2) for p in ("Wq", "bq", "Wo", "bo", "g"))
BASE_KEYS = tuple(f"{p}{l}" for l in (1, 2) for p in ("A", "b"))
WRITE_KEYS = tuple(f"{p}{l}" for l in (1, 2) for p in ("U", "V"))


# ══════════════════════════════════════ 实例构造
def build(seed, variant):
    r"""构造一个实例的全部张量。基座按**预训练权重**的常态构造（V5 保持 requires_grad=True）。"""
    g = torch.Generator().manual_seed(seed)
    base_rg = (variant == "V5")

    def rn(*shape, scale=1.0, rg=False):
        return (torch.randn(*shape, generator=g) * scale).requires_grad_(rg)

    s_h = 1.0 / math.sqrt(D_H)
    P = {}
    for l in (1, 2):
        P[f"A{l}"] = rn(D_H, D_H, scale=s_h, rg=base_rg)      # 基座（冻结）
        P[f"b{l}"] = rn(D_H, scale=0.1, rg=base_rg)
        P[f"Pk{l}"] = rn(D_H, D_K, scale=s_h)                 # 写入映射（固定）
        P[f"Pv{l}"] = rn(D_H, D_V, scale=s_h)
        P[f"Wq{l}"] = rn(D_H, D_K, scale=s_h, rg=True)        # $\phi^q$
        P[f"bq{l}"] = rn(D_K, scale=0.1, rg=True)
        P[f"Wo{l}"] = rn(D_H + D_V, D_H, scale=1.0 / math.sqrt(D_H + D_V), rg=True)  # $\phi^o$
        P[f"bo{l}"] = rn(D_H, scale=0.1, rg=True)
        P[f"g{l}"] = (torch.zeros(1, requires_grad=True) if variant == "V4"
                      else torch.full((1,), GAMMA, requires_grad=True))
        P[f"U{l}"] = rn(D_V, R, scale=INIT_SD, rg=True)       # 低秩因子
        P[f"V{l}"] = rn(D_K, R, scale=INIT_SD, rg=True)
    P["a"] = rn(T_STEPS, D_H, rg=True)                        # 输入（须保留输入梯度 ⇒ 叶子）
    P["tgt"] = rn(T_STEPS, D_H)
    return P


def _W(U, V, no_memory=False):
    W = U @ V.t()
    return torch.zeros_like(W) if no_memory else W


def rollout(P, variant, no_memory=False):
    r"""T 步 rollout。读取用 $W_t$、写入产生 $W_{t+1}$（§2.2）。

    写入链**全程留在计算图中**（不包 no_grad）：若读路漏了 sg，泄漏会沿 T 步反传进
    $U_0,V_0$ ⇒ C3 的检验对泄漏**最敏感**。$\mathrm{sg}$ 本身用 `.detach()`。
    """
    L = P["a"].new_zeros(())
    outs, Wmax, Winit = [], [], {}
    for l in (1, 2):
        Winit[l] = _W(P[f"U{l}"], P[f"V{l}"]).detach().clone()
    h_pre_prev = [None, None]
    for t in range(T_STEPS):
        hp_prev = P["a"][t]                       # $h_{0,t}=h_{0,t}^+=$ 输入
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            if variant == "V3":                   # 基座 no_grad（失败模式）
                with torch.no_grad():
                    h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            else:
                h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            c = (P["a"][t] if l == 1 else hp_prev).detach()   # $c=\operatorname{sg}(h_{l-1,t})$
            q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])      # $\mathcal Q_l$：detach 在输入上
            if variant == "V2":
                q = q.detach()                                 # 失败模式：detach 错了位置
            W = _W(P[f"U{l}"], P[f"V{l}"], no_memory)
            Wmax.append(W.abs().max())
            y = (W if variant == "V1" else W.detach()) @ q     # $y=\operatorname{sg}(W)q$
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            hp[li] = h + P[f"g{l}"] * m                        # $h^+=h+\gamma m$
            hp_prev = hp[li]
        outs.append(hp[1])
        # ── 局部写入规则（§2.3/§3.3 逐字）：k,v 取**注入前**隐状态且被 sg ──
        upd = {}
        for li, l in enumerate((1, 2)):
            hprev = h_pre_prev[li]
            if hprev is None:
                hprev = torch.zeros(D_H)
            k = torch.tanh(hprev.detach() @ P[f"Pk{l}"])       # $k=\operatorname{sg}(P_k(h_{t-1}))$
            v = torch.tanh(h_pre[li].detach() @ P[f"Pv{l}"])   # $v=\operatorname{sg}(P_v(h_t))$
            W = _W(P[f"U{l}"], P[f"V{l}"], no_memory)
            G = ALPHA * torch.outer(W @ k - v, k) + LAM * W     # $G=\alpha(Wk-v)k^\top+\lambda(W-W^0)$
            Uo, Vo = P[f"U{l}"], P[f"V{l}"]
            upd[f"U{l}"] = Uo - ETA * (G @ Vo)                 # 同步：两式都用**旧** U,V
            upd[f"V{l}"] = Vo - ETA * (G.t() @ Uo)
        P.update(upd)
        h_pre_prev = h_pre
        L = L + ((hp[1] - P["tgt"][t]) ** 2).sum()
    dW, WT, W0 = 0.0, 0.0, 0.0
    for l in (1, 2):
        Wt = _W(P[f"U{l}"], P[f"V{l}"])
        dW = max(dW, float((Wt - Winit[l]).norm()))
        WT = max(WT, float(Wt.norm()))
        W0 = max(W0, float(Winit[l].norm()))
    return L, torch.stack(outs), {"Wmax": float(torch.stack(Wmax).max()),
                                  "dW": dW, "WT": WT, "W0": W0}


def _stat(t):
    if t is None:
        return {"none": True, "max": 0.0, "finite": True}
    return {"none": False, "max": float(t.abs().max()), "finite": bool(torch.isfinite(t).all())}


def probe(seed, variant):
    r"""单个实例的四项验收读数。`.grad is None` = **结构上不连通**（纪律 67）。"""
    P = build(seed, variant)
    leaves = {k: P[k] for k in READ_KEYS + WRITE_KEYS + BASE_KEYS + ("a",)}
    L, outs, diag = rollout(P, variant)
    _, outs0, _ = rollout(build(seed, variant), variant, no_memory=True)
    L.backward()
    r = {"seed": seed, "variant": variant, "L": float(L)}
    den = float(outs.norm())
    r["rho_mem"] = float((outs - outs0).norm()) / den if den > 0 else 0.0
    r["read"] = {k: _stat(leaves[k].grad) for k in READ_KEYS}
    r["write"] = {k: _stat(leaves[k].grad) for k in WRITE_KEYS}
    r["base"] = {k: _stat(leaves[k].grad) for k in BASE_KEYS}
    r["ain"] = _stat(leaves["a"].grad)
    r["gread_min"] = min(r["read"][k]["max"] for k in READ_KEYS)
    r["gread_finite_all"] = all(r["read"][k]["finite"] for k in READ_KEYS)
    r["gwrite_max"] = max(r["write"][k]["max"] for k in WRITE_KEYS)
    r["gwrite_none_all"] = all(r["write"][k]["none"] for k in WRITE_KEYS)
    r["gbase_max"] = max(r["base"][k]["max"] for k in BASE_KEYS)
    r["gbase_none_all"] = all(r["base"][k]["none"] for k in BASE_KEYS)
    r["alive_dW"] = diag["dW"]
    r["alive_act"] = diag["dW"] / (diag["WT"] + diag["W0"] + 1e-300)
    r["optmoved"] = opt_check(seed, variant)
    return r


def opt_check(seed, variant):
    r"""C4a 的**行为**检验：把所有参数（含基座）交给同一个优化器走一步，
    基座必须**逐位不变**、读取参数必须**确实变化**。「参数冻结」不是靠读 .grad 自证。"""
    P = build(seed, variant)
    leaves = {k: P[k] for k in READ_KEYS + WRITE_KEYS + BASE_KEYS}
    pre = {k: P[k].detach().clone() for k in leaves}
    L, _, _ = rollout(P, variant)
    L.backward()
    opt = torch.optim.SGD([leaves[k] for k in READ_KEYS + WRITE_KEYS + BASE_KEYS], lr=LR_OPT)
    opt.step()
    dbase = max(float((leaves[k] - pre[k]).abs().max()) for k in BASE_KEYS)
    dread = max(float((leaves[k] - pre[k]).abs().max()) for k in READ_KEYS)
    return {"base_moved": dbase, "read_moved": dread}


def pilot():
    r"""阈值标定：只测**量级**，不进判据。"""
    print(f"torch {torch.__version__}  d_h={D_H} d_k={D_K} d_v={D_V} r={R} T={T_STEPS} "
          f"α={ALPHA} λ={LAM} η={ETA} γ={GAMMA} sd={INIT_SD}")
    hdr = f"{'var':4s} {'L':>9s} {'ρ_mem':>8s} {'gread_min':>10s} {'gwrite':>10s} {'gbase':>10s} {'g_in':>10s} {'ΔW':>9s}"
    print(hdr)
    for v in VARIANTS:
        rs = [probe(SEED0 + i, v) for i in range(4)]
        print(f"{v:4s} {min(r['L'] for r in rs):9.4f} {min(r['rho_mem'] for r in rs):8.5f} "
              f"{min(r['gread_min'] for r in rs):10.3e} {max(r['gwrite_max'] for r in rs):10.3e} "
              f"{max(r['gbase_max'] for r in rs):10.3e} {min(r['ain']['max'] for r in rs):10.3e} "
              f"{min(r['alive_dW'] for r in rs):9.3e}")
    print("\n结构信号与行为信号（V0 应全 None + 基座不动）：")
    for v in VARIANTS:
        r = probe(SEED0, v)
        print(f"  {v}: write_none={r['gwrite_none_all']} base_none={r['gbase_none_all']} "
              f"read_finite={r['gread_finite_all']} ρ={r['rho_mem']:.5f} "
              f"base_moved={r['optmoved']['base_moved']:.3e} read_moved={r['optmoved']['read_moved']:.3e}")



# ══════════════════════════════════════ 五条子句的机械判定（卡 §1.1）
def clauses(r):
    r"""把一个实例的读数折成 C1..C4 的 PASS/FAIL。**机械**，不含判断。"""
    return {
        "C1": int(r["rho_mem"] >= RHO_MIN),
        "C2": int(all(not r["read"][k]["none"] for k in READ_KEYS)
                  and r["gread_finite_all"] and r["gread_min"] >= GREAD_MIN),
        "C3": int(r["gwrite_none_all"]),
        "C4a": int(r["gbase_none_all"]
                   and r["optmoved"]["base_moved"] == 0.0
                   and r["optmoved"]["read_moved"] > 0.0),
        "C4b": int((not r["ain"]["none"]) and r["ain"]["finite"] and r["ain"]["max"] >= GIN_MIN),
    }


def matrix(variants=VARIANTS, n=None):
    r"""变体 × 子句 的判定矩阵（全部实例都满足才算 PASS），附最坏值读数。"""
    n = N_INST if n is None else n
    M, worst = {}, {}
    for v in variants:
        rs = [probe(SEED0 + i, v) for i in range(n)]
        cell = {c: int(all(clauses(r)[c] for r in rs)) for c in CLAUSES}
        M[v] = cell
        worst[v] = {
            "rho_min": min(r["rho_mem"] for r in rs),
            "gread_min": min(r["gread_min"] for r in rs),
            "gwrite_max": max(r["gwrite_max"] for r in rs),
            "gwrite_none_all": all(r["gwrite_none_all"] for r in rs),
            "gbase_max": max(r["gbase_max"] for r in rs),
            "gbase_none_all": all(r["gbase_none_all"] for r in rs),
            "gin_min": min(r["ain"]["max"] for r in rs),
            "read_finite_all": all(r["gread_finite_all"] for r in rs),
            "alive_min": min(r["alive_act"] for r in rs),
            "base_moved_max": max(r["optmoved"]["base_moved"] for r in rs),
            "read_moved_min": min(r["optmoved"]["read_moved"] for r in rs),
            "L_min": min(r["L"] for r in rs),
        }
    return M, worst


# ══════════════════════════════════════ 工作例（卡 §4，纯 float，独立转写）
def werk_write(u, w, k, val, alpha=ALPHA, lam=LAM, eta=0.01):
    r"""§3 写入规则的一步，标量直译（独立于 rollout 的向量化实现）。"""
    W = u * w
    G = alpha * (W * k - val) * k + lam * W
    return W, G, u - eta * G * w, w - eta * G * u


def werk_read(c, W, Wq, bq, Woc, Woy, bo, gam, h):
    q = math.tanh(Wq * c + bq)
    y = W * q
    m = math.tanh(Woc * c + Woy * y + bo)
    return q, y, m, h + gam * m


def local_objective(W, k, v, alpha=ALPHA, lam=LAM, W0=0.0):
    r"""§3.1：$\ell=\frac\alpha2\|Wk-v\|^2+\frac\lambda2\|W-W^0\|^2$。"""
    return alpha / 2 * float(((W @ k - v) ** 2).sum()) + lam / 2 * float(((W - W0) ** 2).sum())


def fd_grad_autograd(variant="V0", t_steps=1, entry=(0, 0), h=1e-6, seed=SEED0, key="Wo1"):
    r"""中心差分 vs autograd。返回 (autograd, fd, rel)。

    ★ $T{=}1$ 时写通道不回流（读取用 $W_0$、写入在损失之后）⇒ 两者**必须**一致；
      $T{>}1$ 时 FD 含被 $\operatorname{sg}$ 禁止的写通道反馈 ⇒ **必然不等**（卡 §6 坑 1）。
    """
    old = T_STEPS
    globals()["T_STEPS"] = t_steps
    try:
        P = build(seed, variant)
        L, _, _ = rollout(P, variant)
        g = torch.autograd.grad(L, [P[key]], allow_unused=True)[0]
        if g is None:
            return None, None, None
        i, j = entry
        one_d = (g.dim() == 1)
        Qp = build(seed, variant); Qm = build(seed, variant)
        if one_d:
            Qp[key].data[i] += h; Qm[key].data[i] -= h
        else:
            Qp[key].data[i, j] += h; Qm[key].data[i, j] -= h
        Lp, _, _ = rollout(Qp, variant)
        Lm, _, _ = rollout(Qm, variant)
        fd = (float(Lp) - float(Lm)) / (2 * h)
        an = float(g[i] if one_d else g[i, j])
        return an, fd, abs(fd - an) / max(abs(an), 1e-300)
    finally:
        globals()["T_STEPS"] = old


# ══════════════════════════════════════ Stage A（卡 §5，七条断言：硬闸）
def stage_a():
    t0, res = time.time(), []

    def chk(tag, ok, detail=""):
        res.append({"tag": tag, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag}  {detail}")

    print("A31-1 写入规则：手算表 + 'G 是局部目标的梯度'")
    W, G, u2, w2 = werk_write(0.1, 0.2, 2.0, 3.0)
    W2 = u2 * w2
    tab = {"W": (W, 0.02), "G": (G, -5.919), "u'": (u2, 0.111838),
           "w'": (w2, 0.205919), "W'": (W2, 0.023029569122)}
    dmax = max(abs(a - b) / max(abs(b), 1e-300) for a, b in tab.values())
    chk("A31-1(i) 工作例 A 与手算表一致", dmax < WORK_TOL, f"max rel={dmax:.2e}")
    # (ii) G 是否是 §3.1 目标的梯度：方向导数 vs 内积
    torch.manual_seed(7)
    Wt = torch.randn(D_V, D_K); k = torch.randn(D_K); vt = torch.randn(D_V)
    Gt = ALPHA * torch.outer(Wt @ k - vt, k) + LAM * Wt
    Dd = torch.randn(D_V, D_K); eps = 1e-4   # 二次型的中心差分**精确**；残差=舍入，∝1/eps
    fd = (local_objective(Wt + eps * Dd, k, vt) - local_objective(Wt - eps * Dd, k, vt)) / (2 * eps)
    an = float((Gt * Dd).sum())
    rel = abs(fd - an) / max(abs(an), 1e-300)
    chk("A31-1(ii) G == ∇ℓ(W)（§3.1 目标）", rel < GRAD_TOL, f"fd={fd:+.10e} an={an:+.10e} rel={rel:.2e}")

    print("A31-2 读取路径：手算例 B")
    q, y, m, hp = werk_read(1.0, 2.0, 3.0, 0.5, 0.5, 0.25, 0.1, 0.5, 1.0)
    tab2 = {"q": (q, 0.998177897611199), "y": (y, 1.996355795222397),
            "m": (m, 0.800171532226446), "h+": (hp, 1.400085766113223)}
    d2 = max(abs(a - b) / max(abs(b), 1e-300) for a, b in tab2.values())
    chk("A31-2 工作例 B 与手算表一致", d2 < WORK_TOL, f"max rel={d2:.2e}")

    print("A31-3（P3-②）无截断路径的组合上 autograd ≡ 中心差分")
    # ★ DEF-31-1：T=1 并不等于「无截断」—— $c_2=\operatorname{sg}(h_1)$ 在**第一步**就截断了
    #   第 1 层读取参数对 $m_2$ 的影响。真正无截断的只有「只经 $\gamma_2$ 的路径」：
    #   $\gamma_2\to h_2^{+}\to\mathcal L$ 是唯一通路（$h_2$ 的注入前值取 $h_2$，且无下一步）。
    an, fd, rel = fd_grad_autograd(t_steps=1, entry=(0, 0), key="g2")
    chk("A31-3 (γ2, T=1) rel < FD_TOL", rel < FD_TOL, f"an={an:+.10e} fd={fd:+.10e} rel={rel:.2e}")
    for tag, ts, key, e in (("c2 截断@T=1", 1, "Wo1", (0, 0)),
                            ("c2 截断@T=1", 1, "g1", (0, 0)),
                            ("写通道+c 截断@T=8", T_STEPS, "Wo1", (0, 0)),
                            ("写通道+c 截断@T=8", T_STEPS, "g1", (0, 0))):
        a2, f2, r2 = fd_grad_autograd(t_steps=ts, entry=e, key=key)
        print(f"       [读数] {tag} {key}: an={a2:+.6e} fd={f2:+.6e} rel={r2:.4f}"
              f"  ← sg 截断的前向依赖，**不是 bug**（§6 坑 1）")

    print("A31-4（P3-①）确定性与可复现性")
    a, b = probe(SEED0, "V0"), probe(SEED0, "V0")
    L1, _, _ = rollout(build(SEED0, "V0"), "V0")
    L2, _, _ = rollout(build(SEED0, "V0"), "V0")
    chk("A31-4 同实例两次 / 两次 build 逐位相同",
        a["L"] == b["L"] and float(L1) == float(L2), f"L={a['L']:.12f}")

    print("A31-5（P1）写通道活性")
    n5 = min(N_INST, 8)
    act = min(probe(SEED0 + i, "V0")["alive_act"] for i in range(n5))
    chk("A31-5 act >= ALIVE_MIN", act >= ALIVE_MIN, f"act_min={act:.5f} (n={n5})")

    print("A31-6（P2）变体判别表与预测逐格一致")
    n6 = min(N_INST, 8)
    Mv, _ = matrix(n=n6)
    bad = [(v, c, Mv[v][c], PREDICT[v][c]) for v in VARIANTS for c in CLAUSES if Mv[v][c] != PREDICT[v][c]]
    chk("A31-6 矩阵与卡 §1.4 一致", not bad,
        "全部 30 格一致" if not bad else f"错位 {bad}")

    print("A31-7 结构性信号方向")
    r0, r1, r5 = probe(SEED0, "V0"), probe(SEED0, "V1"), probe(SEED0, "V5")
    chk("A31-7 V0 的 U,V,θ 全 None；V1 的 U 非 None；V5 的 θ 非 None",
        r0["gwrite_none_all"] and r0["gbase_none_all"]
        and not r1["gwrite_none_all"] and not r5["gbase_none_all"],
        f"V0 w/base={r0['gwrite_none_all']}/{r0['gbase_none_all']} "
        f"V1 w={r1['gwrite_none_all']} V5 b={r5['gbase_none_all']}")

    ok = all(r["pass"] for r in res)
    print(f"\nStage A {'★ 全过' if ok else '✗ 有失败'}（{time.time()-t0:.1f} s，{sum(r['pass'] for r in res)}/{len(res)}）")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/DIR31r10_stageA.json", "w"), ensure_ascii=False, indent=1)
    return 0 if ok else 1


# ══════════════════════════════════════ smoke（协议可执行性，不出判据）
def smoke():
    t0 = time.time()
    Mv, worst = matrix(n=2)
    print(f"smoke: {len(VARIANTS)} 变体 × 2 实例，L_min={worst['V0']['L_min']:.4f}，{time.time()-t0:.1f} s")
    for v in VARIANTS:
        print(f"  {v}: {Mv[v]}")
    json.dump({"matrix": Mv, "secs": round(time.time() - t0, 1)},
              open(f"{OUT}/DIR31r10_smoke.json", "w"), indent=1)
    print("smoke PASS（协议可执行；不出判据）")


# ══════════════════════════════════════ 正式轮
def formal():
    t0 = time.time()
    Mv, worst = matrix()
    # ── P1：写通道活性（V0 的 24 实例最小值）──
    p1 = worst["V0"]["alive_min"] >= ALIVE_MIN
    # ── P2：判别表逐格一致 ──
    bad = [(v, c, Mv[v][c], PREDICT[v][c]) for v in VARIANTS for c in CLAUSES if Mv[v][c] != PREDICT[v][c]]
    p2 = not bad
    # ── P3：确定性与截断图自洽（Stage A 已断；此处再取一次读数）──
    a = probe(SEED0, "V0")
    # ★ DEF-31-1：P3-② 的口径必须是「**无截断路径**的组合」，即 (γ2, T=1)；
    #   用 (Wo1, T=1) 会踩 $c_2=\operatorname{sg}(h_1)$ 的截断 ⇒ 恒 FAIL（那是读数，不是自洽性）。
    rel1 = fd_grad_autograd(t_steps=1, entry=(0, 0), key="g2")[2]
    p3 = (a["L"] == probe(SEED0, "V0")["L"]) and rel1 < FD_TOL
    # ── C1–C4（V0 的 24 实例全满足）──
    c0 = {c: Mv["V0"][c] for c in CLAUSES}
    pre_ok = p1 and p2 and p3
    allC = all(c0[c] == 1 for c in CLAUSES)
    if not pre_ok:
        verdict = "MEASUREMENT-NOT-CREDIBLE"
    elif allC:
        verdict = "SUCCEEDED"
    else:
        verdict = "INTERFACE-NOT-ACCEPTED"
    out = {
        "variant": "DIR-31 rev 1.0", "secs": round(time.time() - t0, 1), "N_inst": N_INST,
        "thresholds": {"RHO_MIN": RHO_MIN, "GREAD_MIN": GREAD_MIN, "GIN_MIN": GIN_MIN,
                       "ALIVE_MIN": ALIVE_MIN, "FD_TOL": FD_TOL},
        "P1_alive": p1, "P2_matrix": p2, "P3_credible": p3,
        "P2_mismatch": bad, "T1_fd_rel": rel1,
        "C": c0, "matrix": Mv, "predicted": PREDICT, "worst": worst,
        "verdict": verdict,
    }
    json.dump(out, open(f"{OUT}/DIR31r10_formal.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n═══ DIR-31 正式轮（{out['secs']} s，N={N_INST}）═══")
    print(f"P1 写通道活性 act_min={worst['V0']['alive_min']:.5f} ≥ {ALIVE_MIN} ⇒ {'PASS' if p1 else 'FAIL'}")
    print(f"P2 判别表：{len(VARIANTS)}×{len(CLAUSES)}=30 格，错位 {len(bad)} ⇒ {'PASS' if p2 else 'FAIL'}")
    print(f"P3 无截断组合 (γ2,T=1) rel={rel1:.2e} < {FD_TOL:.0e} ⇒ {'PASS' if p3 else 'FAIL'}")
    print(f"\nV0 的最坏读数（24 实例）：ρ_min={worst['V0']['rho_min']:.5f} "
          f"gread_min={worst['V0']['gread_min']:.4e} g_in_min={worst['V0']['gin_min']:.4f} "
          f"gwrite_none={worst['V0']['gwrite_none_all']} gbase_none={worst['V0']['gbase_none_all']} "
          f"base_moved={worst['V0']['base_moved_max']:.1e} read_moved={worst['V0']['read_moved_min']:.3e}")
    print("子句： " + "  ".join(f"{c}={'PASS' if c0[c] else 'FAIL'}" for c in CLAUSES))
    print(f"\n★ 判定：{verdict}")
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    os.makedirs(OUT, exist_ok=True)
    {"pilot": pilot, "stageA": stage_a, "smoke": smoke, "formal": formal,
     "matrix": lambda: print(json.dumps(matrix()[0], indent=1))}[mode]()
