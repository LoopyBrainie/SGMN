#!/opt/pytorch/bin/python
r"""DIR-32 **前提诊断**：不训练任何东西，直接问「写规则在 $T{=}8$ 步里存进去了没有」。

若 oracle 查询（$q_t=k(h_{t-2})=k(h_t)$，周期 2 ⇒ 恰是第 $t-1$ 步写入的键）都取不回目标，
那「前向任务收益」在本装置族里**无从谈起**，必须先改装置再开卡 —— 这是 MVP 闸门前置。
"""
import json
import math
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
import R1_grad as M

OUT = "/home/ec2-user/sgmn-loop/dir32/out"
SEEDS = [M.SEED0 + 100 + i for i in range(24)]
D_H, D_K, D_V, T = M.D_H, M.D_K, M.D_V, M.T_STEPS


def period2(P, sd):
    g = torch.Generator().manual_seed(int(sd) + 7777)
    x = torch.randn(2, D_H, generator=g)
    P["a"] = torch.stack([x[t % 2] for t in range(T)])


def cos(u, v):
    nu, nv = float(u.norm()), float(v.norm())
    return float((u * v).sum()) / (nu * nv) if nu > 0 and nv > 0 else 0.0


def run(sd):
    r"""在 V0 上走 T 步，逐步记下 $W_t$ 与 $k,v$；只做前向测量（不训练）。"""
    P = M.build(sd, "V0")
    period2(P, sd)
    hs = [torch.tanh(P["a"][s] @ P["A1"] + P["b1"]) for s in range(T)]
    ks = [torch.tanh(hs[s] @ P["Pk1"]) for s in range(T)]
    vs = [torch.tanh(hs[s] @ P["Pv1"]) for s in range(T)]
    Ws = []
    h_pre_prev = None
    for t in range(T):                       # 与冻结装置同步：先读、再写
        Ws.append(M._W(P["U1"], P["V1"]).detach().clone())
        k = torch.tanh((torch.zeros(D_H) if h_pre_prev is None else h_pre_prev) @ P["Pk1"])
        v = torch.tanh(hs[t] @ P["Pv1"])
        W = M._W(P["U1"], P["V1"])
        G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
        Uo, Vo = P["U1"], P["V1"]
        P["U1"] = Uo - M.ETA * (G @ Vo)
        P["V1"] = Vo - M.ETA * (G.t() @ Uo)
        h_pre_prev = hs[t]
    rows = []
    for t in range(1, T):
        g = vs[t - 1]                                        # 目标：第 $t-1$ 步写入的值
        q = ks[(t - 2) % T]                                  # oracle：$k(h_{t-2})=k(h_t)$（周期 2）
        y = Ws[t] @ q
        # 精确低秩记忆（把 $t$ 步之前写入的 (k,v) 对做 rank-2 最小二乘）的**上界**参考
        pairs = [(ks[(s - 1) % T], vs[s]) for s in range(1, t)]
        if len(pairs) >= 2:
            K = torch.stack([p[0] for p in pairs], 1)
            Vm = torch.stack([p[1] for p in pairs], 1)
            U_, S_, Vt = torch.linalg.svd(Vm @ K.t())
            Wstar = (U_[:, :2] * S_[:2]) @ Vt[:2]
            ystar = Wstar @ q
        else:
            ystar = torch.zeros(D_V)
        rows.append({"t": t, "cos_W": cos(y, g), "cos_star": cos(ystar, g),
                     "cos_q_vs_k": cos(q, ks[t - 1]),
                     "cos_y_vs_W0q": cos(y, (M.build(sd, "V0")["U1"] @ M.build(sd, "V0")["V1"].t()) @ q),
                     "dW_rel": float((Ws[t] - Ws[0]).norm() / Ws[0].norm())})
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    allr = [run(sd) for sd in SEEDS]
    agg = {}
    for t in range(1, T):
        cw = [r[t - 1]["cos_W"] for r in allr]
        cs = [r[t - 1]["cos_star"] for r in allr]
        cq = [r[t - 1]["cos_q_vs_k"] for r in allr]
        dr = [r[t - 1]["dW_rel"] for r in allr]
        agg[t] = {"cos_W": sum(cw) / len(cw), "cos_star": sum(cs) / len(cs),
                  "cos_q_vs_k": sum(cq) / len(cq), "dW_rel": sum(dr) / len(dr)}
        print(f"t={t}  cos(W_t q, g) {agg[t]['cos_W']:+8.4f}   "
              f"rank-2 上界 {agg[t]['cos_star']:+8.4f}   "
              f"cos(q,k_{t-1}) {agg[t]['cos_q_vs_k']:+8.4f}   ‖ΔW‖/‖W0‖ {agg[t]['dW_rel']:.3e}")
    m = sum(agg[t]["cos_W"] for t in agg) / len(agg)
    ms = sum(agg[t]["cos_star"] for t in agg) / len(agg)
    print(f"\n均值：cos_W {m:+.4f}   上界 {ms:+.4f}")
    print("判读：" + ("★ 连 rank-2 上界都 ≈0 ⇒ 目标不可由该记忆读出，必须先改装置" if abs(ms) < 0.15
                    else "上界可用，差距在写规则的存储效率"))
    json.dump({"per_seed": allr, "agg": {str(k): v for k, v in agg.items()}},
              open(f"{OUT}/diag32.json", "w"), ensure_ascii=False, indent=1)
