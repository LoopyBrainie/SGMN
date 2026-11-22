#!/opt/pytorch/bin/python
r"""DIR-32 **开卡前可行性标定**（决定二十九：只决定装置放在哪，**不进判据**）。

要回答三件事，任何一件不过 ⇒ 本轮不可跑，须先改装置或先判前置失败：
  A0  写入规则是否真的能学到「键 $k(h_{s-1})\to$ 值 $v(h_s)$」的关联（否则没有内容可读）；
  A1  **共享读取头 + 逐实例随机写入映射**下，「学会寻址」是否可能 —— 若不可能，
      这就是一条**装置层不可能性**结论（机制产物，比数字值钱）；
  A2  **oracle 寻址**（$\mathcal W^q_l := P_{k,l}$ 冻结、$c:=h_{1,t}$）下，内容性收益是否可测。

关联方向的代数（务必与 `dir31/R1_grad.py` 的写入块逐字一致）：
  第 $s$ 步写：$k_s=\operatorname{sg}(P_k(h_{s-1}))$、$v_s=\operatorname{sg}(P_v(h_s))$ ⇒ 记忆里存 $(k_s\to v_s)$。
  第 $t$ 步读：$q_t=\mathcal Q(c_t)$、$y_t=\operatorname{sg}(W_t)q_t$。
  **周期 2 输入**（$a_t=a_{t\bmod 2}$）⇒ $h_{t-2}=h_t$ ⇒ $q_t=k(h_t)=k(h_{t-2})$ 恰好命中 $s=t-1$ 写入的键
  ⇒ $y_t\approx v_{t-1}=v(h_{t-1})$。故**目标取 $g_t:=v(h_{t-1})$**，它**不是** $c_t=h_t$ 的函数（另一奇偶的值），
  且两个奇偶的**配对关系逐实例随机** ⇒ 共享读取头无法靠 $c$ 直接算出它 ⇒ 判别力存在。
"""
import json
import math
import os
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
import R1_grad as M                       # ★ 复用**冻结的装置**：常量/build/_W/写入块

OUT = "/home/ec2-user/sgmn-loop/dir32/out"
PILOT_SEEDS = [M.SEED0 + 100 + i for i in range(24)]   # ★ 与正式轮种子**不相交**（纪律 70）
D_H, D_K, D_V = M.D_H, M.D_K, M.D_V
T = M.T_STEPS


# ══════════════════════════════════ 周期 2 输入（唯一的自变量：任务）
_POUT = {}


def pout():
    r"""任务层的**冻结共享解码矩阵** $P_{\rm out}\in\mathbb R^{d_v\times d_h}$（一次抽定、全局共享）。
    目标活在 $d_v$ 里、读数活在 $d_h$ 里，必须有一个确定的解码口；它若逐实例不同，
    共享读取头就有额外自由度，任务会变成「猜解码器」而不是「用记忆」。"""
    if "m" not in _POUT:
        g = torch.Generator().manual_seed(int(M.SEED0) + 31337)
        _POUT["m"] = torch.randn(D_V, D_H, generator=g) / math.sqrt(D_H)
    return _POUT["m"]


def period2(P):
    g = torch.Generator().manual_seed(int(P["_seed"]) + 7777)
    x = torch.randn(2, D_H, generator=g)
    P["a"] = torch.stack([x[t % 2] for t in range(T)])


def targets(P):
    r"""$g_t=v(h_{t-1})$，$t\ge 1$（基座前向、无梯度；不进计算图）。"""
    A, b, Pv = P["A1"], P["b1"], P["Pv1"]
    hs = [torch.tanh(P["a"][s] @ A + b) for s in range(T)]
    return hs, [torch.tanh(hs[t - 1] @ Pv) for t in range(1, T)]


# ══════════════════════════════════ 任务 rollout（写入块与装置**逐字**相同）
def rollout_task(P, addr, read_arm="true", detach_writes=True):
    r"""addr ∈ {"oracle","device"}；read_arm ∈ {"true","shuf","zero"}。
    detach_writes=True 时把 U,V 的在线更新切成常量（数值**不变**，只为不让任务梯度回溯到 $U_0,V_0$）。"""
    L = P["a"].new_zeros(())
    hs, gs = targets(P)
    h_pre_prev = [None, None]
    perm = torch.tensor([1, 3, 0, 2])
    for t in range(T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if addr == "oracle" and l == 1:
                c, q = h, torch.tanh(h @ P["Pk1"])                 # $q=k(h_t)$，$\mathcal W^q_1:=P_{k,1}$
            else:
                c = (P["a"][t] if l == 1 else hp_prev).detach()
                q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            Wd = W.detach()
            if read_arm == "shuf":
                Wd = Wd[:, perm]                                   # 列置换：‖W‖ 与条目分布不变，键-值对应破坏
            elif read_arm == "zero":
                Wd = torch.zeros_like(Wd)
            y = Wd @ q
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            hp[li] = h + P[f"g{l}"] * m
            hp_prev = hp[li]
        if t >= 1:
            L = L + ((pout() @ hp[1] - gs[t - 1]) ** 2).sum()   # 目标 $g_t=v(h_{t-1})$ 活在 $d_v$
        # ── 冻结装置的写入块（与 M.rollout 逐字一致）──
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
        if detach_writes:
            upd = {kk: vv.detach() for kk, vv in upd.items()}
        P.update(upd)
        h_pre_prev = h_pre
    return L, hs, gs


def write_quality(P):
    r"""A0：第 1 层沿轨迹的实际关联残差 $\lVert W_t k_s-v_s\rVert$（读得出来的前提）。"""
    out = []
    for l in (1, 2):
        W = M._W(P[f"U{l}"], P[f"V{l}"])
        res = []
        for s in range(1, T):
            hs = torch.tanh(P["a"][s - 1] @ P[f"A{l}"] + P[f"b{l}"]) if l == 1 else None
            if hs is None:
                continue
            k = torch.tanh(hs @ P[f"Pk{l}"])
            v = torch.tanh(torch.tanh(P["a"][s] @ P[f"A{l}"] + P[f"b{l}"]) @ P[f"Pv{l}"])
            res.append(float((W @ k - v).norm()))
        out.append(res)
    return out[0]


def train(addr, steps=300, lr=0.03, learned=False):
    r"""在 R_true 臂上训练读取参数（只碰 $\gamma,\mathcal W^o,b^o$；learned 时另碰 $\mathcal W^q,b^q$）。"""
    P0 = M.build(PILOT_SEEDS[0], "V0")
    names = ["g1", "Wo1", "bo1"] + (["Wq1", "bq1"] if learned else [])
    torch.manual_seed(4242)
    for k in names:                                        # 所有实例**共享**同一组读取参数
        if k in ("g1",):
            P0[k] = torch.full((1,), M.GAMMA, requires_grad=True)
        elif k == "Wq1":
            P0[k] = (torch.randn(D_H, D_K) * 0.354).requires_grad_(True)
        elif k == "bq1":
            P0[k] = torch.zeros(D_K, requires_grad=True)
    opt = torch.optim.Adam([P0[k] for k in names], lr=lr)
    hist = []
    for it in range(steps):
        tot = 0.0
        opt.zero_grad()
        for sd in PILOT_SEEDS:
            P = M.build(sd, "V0")
            P["_seed"] = sd
            period2(P)
            for k in names:
                P[k] = P0[k]
            L, _, _ = rollout_task(P, addr)
            (L / len(PILOT_SEEDS)).backward()
            tot += float(L) / len(PILOT_SEEDS)
        opt.step()
        opt.zero_grad()
        hist.append(tot)
    return {k: P0[k].detach().clone() for k in names}, hist


def evaluate(phi, addr):
    r"""$L$ 三臂（**冻结的 $\phi$** 上比，不比训练曲线）。"""
    acc = {a: 0.0 for a in ("true", "shuf", "zero")}
    wq = []
    for sd in PILOT_SEEDS:
        P = M.build(sd, "V0")
        P["_seed"] = sd
        period2(P)
        for k, v in phi.items():
            P[k] = v.clone()
        for a in acc:
            L, _, _ = rollout_task(P, addr, read_arm=a)
            acc[a] += float(L) / len(PILOT_SEEDS)
        Q = M.build(sd, "V0")
        Q["_seed"] = sd
        period2(Q)
        Q.update({k: v.clone() for k, v in phi.items()})
        rollout_task(Q, addr, detach_writes=True)
        wq.append(write_quality(Q))
    res = dict(acc)
    res["content"] = acc["shuf"] - acc["true"]
    res["channel"] = acc["zero"] - acc["shuf"]
    res["wq_mean"] = sum(sum(w) / len(w) for w in wq) / len(wq)
    return res


def crosscheck():
    r"""A-守卫：本文件的写入块必须与**冻结装置** `M.rollout` 逐位一致（纪律 68：同一断言不得有两份副本）。"""
    P1 = M.build(PILOT_SEEDS[0], "V0")
    P2 = M.build(PILOT_SEEDS[0], "V0")
    M.rollout(P1, "V0")                       # 装置的 T 步
    rollout_task(P2, "device", detach_writes=False)
    d = max(float((P1[k] - P2[k]).abs().max()) for k in ("U1", "V1", "U2", "V2"))
    return d


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print(f"torch {torch.__version__}  pilot seeds = SEED0+100..123（与正式轮不相交）")
    d = crosscheck()
    print(f"\n[A-守卫] 写入块 vs 冻结装置 M.rollout：逐位差 {d:.3e}  "
          f"{'OK' if d == 0.0 else '★ 不一致 —— 禁止开卡'}")
    P = M.build(PILOT_SEEDS[0], "V0")
    P["_seed"] = PILOT_SEEDS[0]
    period2(P)
    rollout_task(P, "device", detach_writes=True)
    wq = write_quality(P)
    print(f"[A0] 第 1 层关联残差 ‖W_t k_s − v_s‖（t=7 为止）："
          f" min {min(wq):.4f}  mean {sum(wq)/len(wq):.4f}  max {max(wq):.4f}"
          f"   （初值 ‖v‖ 量级 {float(torch.tanh(torch.tanh(P['a'][1] @ P['A1'] + P['b1']) @ P['Pv1']).norm()):.3f}）")
    out = {"crosscheck": d, "wq": wq}
    for addr, learned in (("device", True), ("oracle", False)):
        t1 = time.time()
        phi, hist = train(addr, learned=learned)
        r = evaluate(phi, addr)
        tag = "A1 共享头·学寻址" if addr == "device" else "A2 oracle 寻址"
        print(f"\n[{tag}] 训练 {len(hist)} 步：L {hist[0]:.4f} → {hist[-1]:.4f}  ({time.time()-t1:.1f}s)")
        print(f"   L_true {r['true']:9.5f} · L_shuf {r['shuf']:9.5f} · L_zero {r['zero']:9.5f}")
        print(f"   **内容性收益** {r['content']:+9.6f}   **通用注入收益** {r['channel']:+9.5f}"
              f"   写入残差 {r['wq_mean']:.4f}")
        r["hist0"], r["hist1"] = hist[0], hist[-1]
        out[addr] = r
    json.dump(out, open(f"{OUT}/pilot32.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n合计 {time.time()-t0:.1f}s ⇒ {OUT}/pilot32.json")
