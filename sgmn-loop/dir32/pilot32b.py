#!/opt/pytorch/bin/python
r"""DIR-32 可行性标定 **v2** —— 修正 v1 的两个方法学缺陷（v1 存档不删，见 pilot32.py）。

v1 缺陷：
  **DEF-32-0a**：「列置换 = 内容破坏」**错**。$W'=WP\Rightarrow W'q=W(Pq)$，可被可学查询头整体吸收
                ⇒ 那不是内容破坏，是**键空间重新贴标签**。改用**拍平条目置换**（$\lVert\cdot\rVert_F$ 与条目分布逐位保持，
                且不是 $(I\otimes A)$ 型映射 ⇒ 无法被读数头吸收）。
  **DEF-32-0b**：只在 R_true 上训练、再拿冻结 $\phi$ 去评 shuf/zero ⇒ 是**分布外**读数。$L_{shuf}\gg L_{zero}$
                正是这条失真的签名。改用**每条臂在自己的分布里训练**，比「可达损失」$L^\ast(\cdot)$。

四臂阶梯（$G0 \supset N \supset S \supset T$）—— 直接实现用户规则要求的拆解：
  **G0** $\gamma\equiv0$、$W$ 不进读数 ⇒ 纯基座，无注入
  **N**  $\gamma$ 可训、读数 $y\equiv0$ ⇒ **通用注入**（有额外容量，无记忆内容）
  **S**  $\gamma$ 可训、读数用**拍平置换**后的 $W$ ⇒ 幅度匹配、关联破坏
  **T**  $\gamma$ 可训、读数用真 $W$ ⇒ 内容可用
  · **通用注入收益 $=L^\ast(G0)-L^\ast(N)$**   · **内容性收益 $=L^\ast(S)-L^\ast(T)$**（幅度匹配，唯一可计数项）
  · $L^\ast(N)-L^\ast(S)$ 单列（「逐实例随机信号」的收益，不属两者，更不计数）
"""
import json
import math
import os
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
import R1_grad as M

OUT = "/home/ec2-user/sgmn-loop/dir32/out"
PILOT_SEEDS = [M.SEED0 + 100 + i for i in range(24)]
D_H, D_K, D_V, T = M.D_H, M.D_K, M.D_V, M.T_STEPS
ARMS = ("G0", "N", "S", "T")
_POUT = {}
_W0 = {}


def pout():
    if "m" not in _POUT:
        g = torch.Generator().manual_seed(int(M.SEED0) + 31337)
        _POUT["m"] = torch.randn(D_V, D_H, generator=g) / math.sqrt(D_H)
    return _POUT["m"]


def period2(P, seed):
    g = torch.Generator().manual_seed(int(seed) + 7777)
    x = torch.randn(2, D_H, generator=g)
    P["a"] = torch.stack([x[t % 2] for t in range(T)])


def targets(P):
    hs = [torch.tanh(P["a"][s] @ P["A1"] + P["b1"]) for s in range(T)]
    return [torch.tanh(hs[t - 1] @ P["Pv1"]) for t in range(1, T)]


def scratch(seed):
    r"""拍平条目置换：$\lVert W'\rVert_F=\lVert W\rVert_F$，条目分布不变，键-值对应破坏。"""
    g = torch.Generator().manual_seed(int(seed) + 9091)
    return torch.randperm(D_V * D_K, generator=g)


def read_W(W, arm, seed):
    if arm in ("G0", "N"):
        return torch.zeros_like(W)
    if arm == "S":
        return W.flatten()[scratch(seed)].reshape(D_V, D_K)
    return W


def rollout_task(P, addr, arm, seed, detach_writes=True):
    L = P["a"].new_zeros(())
    gs = targets(P)
    h_pre_prev = [None, None]
    for t in range(T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if addr == "oracle" and l == 1:
                c, q = h, torch.tanh(h @ P["Pk1"])
            else:
                c = (P["a"][t] if l == 1 else hp_prev).detach()
                q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            y = read_W(W.detach(), arm, seed) @ q
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            gam = torch.zeros(1) if arm == "G0" else P[f"g{l}"]
            hp[li] = h + gam * m
            hp_prev = hp[li]
        if t >= 1:
            L = L + ((pout() @ hp[1] - gs[t - 1]) ** 2).sum()
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
    return L


def train_arm(addr, arm, steps=250, lr=0.03, learned=False):
    r"""**在本臂自己的分布里训练**（比可达损失）。只碰读取参数；$\phi$ 在 24 个实例间共享。"""
    names = ([] if arm == "G0" else ["g1"]) + ["Wo1", "bo1"] + (["Wq1", "bq1"] if learned else [])
    torch.manual_seed(4242)
    phi0 = {}
    for k in names:
        if k == "g1":
            phi0[k] = torch.full((1,), M.GAMMA, requires_grad=True)
        elif k == "Wq1":
            phi0[k] = (torch.randn(D_H, D_K) * 0.354).requires_grad_(True)
        elif k == "bq1":
            phi0[k] = torch.zeros(D_K, requires_grad=True)
    if not names:
        return {}, [0.0, 0.0]
    opt = torch.optim.Adam(list(phi0.values()), lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        tot = 0.0
        for sd in PILOT_SEEDS:
            P = M.build(sd, "V0")
            period2(P, sd)
            P.update({k: v for k, v in phi0.items()})
            L = rollout_task(P, addr, arm, sd)
            (L / len(PILOT_SEEDS)).backward()
            tot += float(L) / len(PILOT_SEEDS)
        opt.step()
        hist.append(tot)
    return {k: v.detach().clone() for k, v in phi0.items()}, [hist[0], hist[-1]]


def eval_arm(addr, arm, phi):
    acc = 0.0
    for sd in PILOT_SEEDS:
        P = M.build(sd, "V0")
        period2(P, sd)
        P.update({k: v.clone() for k, v in phi.items()})
        acc += float(rollout_task(P, addr, arm, sd)) / len(PILOT_SEEDS)
    return acc


def crosscheck():
    P1 = M.build(PILOT_SEEDS[0], "V0")
    P2 = M.build(PILOT_SEEDS[0], "V0")
    M.rollout(P1, "V0")
    rollout_task(P2, "device", "T", PILOT_SEEDS[0], detach_writes=False)
    return max(float((P1[k] - P2[k]).abs().max()) for k in ("U1", "V1", "U2", "V2"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    d = crosscheck()
    print(f"[A-守卫] 写入块 vs 冻结装置：逐位差 {d:.3e}  {'OK' if d == 0.0 else '★ 禁止开卡'}")
    res = {"crosscheck": d}
    for addr, learned in (("device", True), ("oracle", False)):
        tag = "学寻址（共享头）" if addr == "device" else "oracle 寻址"
        print(f"\n══ {tag} ══")
        Ls, hs = {}, {}
        for arm in ARMS:
            t1 = time.time()
            phi, h = train_arm(addr, arm, learned=learned)
            Ls[arm] = eval_arm(addr, arm, phi)
            hs[arm] = h
            print(f"  {arm:2s}  L* {Ls[arm]:9.5f}   训练 {h[0]:8.3f}→{h[1]:8.3f}  ({time.time()-t1:.0f}s)")
        print(f"  ⇒ **通用注入收益** L*(G0)−L*(N) = {Ls['G0']-Ls['N']:+9.5f}")
        print(f"  ⇒ 逐实例随机信号 L*(N)−L*(S) = {Ls['N']-Ls['S']:+9.5f}   （不属两者，不计数）")
        print(f"  ⇒ **内容性收益**   L*(S)−L*(T) = {Ls['S']-Ls['T']:+9.5f}   ★唯一可计数项")
        res[addr] = {"L": Ls, "hist": hs}
    json.dump(res, open(f"{OUT}/pilot32b.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n合计 {time.time()-t0:.0f}s")
