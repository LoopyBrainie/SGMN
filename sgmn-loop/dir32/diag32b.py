#!/opt/pytorch/bin/python
r"""DIR-32 前提诊断 **v2**：写规则的**收敛交叉点**（修正 diag32.py 的一处标签错误）。

DEF-32-0c：`diag32.py` 把 `cos(q,k_{t-1})` 印成「oracle 查询对不对得上键」，但第 $t-1$ 步写入的键是
$k(h_{t-2})$，而打印的是 $k(h_{t-1})$ ⇒ 那个 0.1588 是**两个不同键的余弦（键可分性）**，
不是对齐残差。oracle 对齐按构造**恰好为 1**（$q_t=\tanh(h_tP_k)=k(h_{t-2})$，周期 2）。本文件分开印。

问题：8 步内 $\lVert\Delta W\rVert/\lVert W_0\rVert\approx0.21$、$\cos\approx0.09$ 而 rank-2 上界 $0.95$。
⇒ 是「**时域太短**」还是「**规则存不进**」？做法：固定一个 (k,v) 对，让**同一条冻结写入规则**跑
$N\in\{8,16,32,64,128,256,512\}$ 步，量 $\cos(W k,v)$ 与 $\lVert Wk-v\rVert/\lVert v\rVert$。
"""
import json
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
import R1_grad as M

OUT = "/home/ec2-user/sgmn-loop/dir32/out"
SEEDS = [M.SEED0 + 100 + i for i in range(24)]
D_H, D_V, T = M.D_H, M.D_V, M.T_STEPS
NS = (8, 16, 32, 64, 128, 256, 512)


def cos(u, v):
    nu, nv = float(u.norm()), float(v.norm())
    return float((u * v).sum()) / (nu * nv) if nu > 0 and nv > 0 else 0.0


_UV0 = {}


def _W0(sd):
    r"""该实例的初始 $W^0=U_0V_0^{\top}$（逐位可复现，用于相对位移）。"""
    if sd not in _UV0:
        Q = M.build(sd, "V0")
        _UV0[sd] = M._W(Q["U1"], Q["V1"]).detach().clone()
    return _UV0[sd]


def convergence(sd):
    r"""单个固定 (k,v) 对，重复写入 $N$ 步（§2.3 的 $G$ 与同步低秩步，逐字）。"""
    P = M.build(sd, "V0")
    g = torch.Generator().manual_seed(int(sd) + 5150)
    h = torch.randn(D_H, generator=g)
    k = torch.tanh(h @ P["Pk1"])
    v = torch.tanh(h @ P["Pv1"])
    Winit = _W0(sd)
    n0 = float(Winit.norm())
    out = {}
    for n in range(1, max(NS) + 1):
        W = M._W(P["U1"], P["V1"])
        G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
        Uo, Vo = P["U1"], P["V1"]
        P["U1"] = Uo - M.ETA * (G @ Vo)
        P["V1"] = Vo - M.ETA * (G.t() @ Uo)
        if n in NS:
            Wn = M._W(P["U1"], P["V1"])
            out[n] = {"cos": cos(Wn @ k, v),
                      "rel": float((Wn @ k - v).norm() / v.norm()),
                      "dW": float((Wn - Winit).norm()) / n0}
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = [convergence(sd) for sd in SEEDS]
    print("固定一对 (k,v)，同一条冻结写入规则重复 $N$ 步：")
    print(f"{'N':>5s} {'cos(Wk,v)':>11s} {'‖Wk−v‖/‖v‖':>12s} {'‖ΔW‖/‖W0‖':>11s}")
    agg = {}
    for n in NS:
        c = sum(r[n]["cos"] for r in rows) / len(rows)
        e = sum(r[n]["rel"] for r in rows) / len(rows)
        d = sum(r[n]["dW"] for r in rows) / len(rows)
        agg[n] = {"cos": c, "rel": e, "dW": d}
        print(f"{n:5d} {c:+11.4f} {e:12.4f} {d:11.4f}")
    # 交叉点：cos 首次 ≥ 0.9（可检索）
    hit = next((n for n in NS if agg[n]["cos"] >= 0.9), None)
    print(f"\n交叉点：cos ≥ 0.9 的最早 N = **{hit}**" if hit else "\n交叉点：在 N ≤ 512 内**未达到** cos ≥ 0.9")
    json.dump({"agg": {str(k): v for k, v in agg.items()}, "cross": hit},
              open(f"{OUT}/diag32b.json", "w"), ensure_ascii=False, indent=1)
