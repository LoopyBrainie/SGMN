#!/opt/pytorch/bin/python
r"""DIR-33 开卡前诊断（二）：**值饱和**才是 A33-6 的根因 —— 扫 $P_v$ 尺度。

A33-6 要防的是「当前输入就几乎定出目标」：$|\cos(g_{\rm even},g_{\rm odd})|$ 接近 1。
`diag33.py` 已排除输入共线（正交化后输入 |cos|=0），但目标仍到 0.889 ⇒ 剩余来源是
$\tanh$ 把两族的 $h P_v$ 同时推向饱和角 ⇒ 扫 $P_v$ 尺度找**构造上**就宽松的区域。

判据（本文件不冻结任何东西，只报数）：目标 |cos|_max ≤ 0.75 才算「有真实余量」。
"""
import json
import sys
import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
import R1_task33 as R
import R1_grad as M

OUT = "/home/ec2-user/sgmn-loop/dir33/out"
_SQ = M.D_H ** 0.5


def period2_ortho(sd, scaled=True):
    g = torch.Generator().manual_seed(int(sd) + 7777)
    x = torch.randn(2, R.D_H, generator=g)
    x[1] = x[1] - (x[1] @ x[0]) / (x[0] @ x[0]) * x[0]
    if scaled:
        x = x / x.norm(dim=1, keepdim=True) * _SQ
    return torch.stack([x[t % 2] for t in range(R.T)])


def measure(pv_scale, seeds):
    tgt, sep = [], []
    for sd in seeds:
        P = R.build(sd, "V0")
        P["Pv1"] = P["Pv1"] * pv_scale
        P["a"] = period2_ortho(sd)
        H = R.hs_of(P)
        ks = [torch.tanh(H[s] @ P["Pk1"]) for s in range(R.T)]
        g = R.targets(P)
        for t in range(2, R.T):
            sep.append(abs(float(ks[t] @ ks[t - 1] / (ks[t].norm() * ks[t - 1].norm()))))
        for i in range(len(g)):
            for j in range(len(g)):
                if (i - j) % 2 == 1:
                    tgt.append(abs(float(g[i] @ g[j] / (g[i].norm() * g[j].norm()))))
    return max(tgt), max(sep)


def gates_at(pv_scale, seeds):
    r"""P1/P2 与 R.gates 同式，但 $P_v$ 缩放 + 正交输入。"""
    p1, p2 = [], []
    for sd in seeds:
        P = R.build(sd, "V0")
        P["Pv1"] = P["Pv1"] * pv_scale
        P["a"] = period2_ortho(sd)
        H = R.hs_of(P)
        T = R.T
        ks = [torch.tanh(H[s] @ P["Pk1"]) for s in range(T)]
        vs = [torch.tanh(H[s] @ P["Pv1"]) for s in range(T)]
        Ws, hprev = [], None
        for t in range(T):
            Ws.append(M._W(P["U1"], P["V1"]).detach().clone())
            k = torch.tanh((torch.zeros(R.D_H) if hprev is None else hprev) @ P["Pk1"])
            v = torch.tanh(H[t] @ P["Pv1"])
            W = M._W(P["U1"], P["V1"])
            G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
            Uo, Vo = P["U1"], P["V1"]
            P["U1"] = Uo - M.ETA * (G @ Vo)
            P["V1"] = Vo - M.ETA * (G.t() @ Uo)
            hprev = H[t]
        cs = lambda u, v: float((u * v).sum() / (u.norm() * v.norm())) if float(u.norm()) > 0 and float(v.norm()) > 0 else 0.0
        for t in range(1, T):
            g = vs[t - 1]
            q = ks[(t - 2) % T]
            pp = [(ks[(s - 1) % T], vs[s]) for s in range(1, t)]
            if len(pp) >= 2:
                K = torch.stack([x[0] for x in pp], 1)
                Vm = torch.stack([x[1] for x in pp], 1)
                U_, S_, Vt = torch.linalg.svd(Vm @ K.t())
                Wst = (U_[:, :2] * S_[:2]) @ Vt[:2]
            else:
                Wst = torch.zeros(R.D_V, R.D_K)
            p1.append(cs(Wst @ q, g))
            p2.append(cs(Ws[t] @ q, g))
    return sum(p1) / len(p1), sum(p2) / len(p2)


def main():
    res = {}
    print(f"{'Pv尺度':>7}{'目标|cos|max':>14}{'键可分性':>10}{'P1':>9}{'P2':>9}{'真实余量':>10}")
    for s in (1.0, 0.7, 0.5, 0.35, 0.25):
        tm, sm = measure(s, R.SEEDS_F)
        p1, p2 = gates_at(s, R.SEEDS_F)
        res[str(s)] = {"tgt_max": tm, "sep_max": sm, "P1": p1, "P2": p2,
                       "margin_vs_075": 0.75 - tm}
        print(f"{s:7.2f}{tm:14.4f}{sm:10.4f}{p1:+9.4f}{p2:+9.4f}{0.75-tm:+10.4f}")
    json.dump(res, open(f"{OUT}/DIR33_diag_pv.json", "w"), indent=1)
    ok = [k for k, v in res.items() if v["tgt_max"] <= 0.75 and v["P1"] >= 0.5 and v["P2"] > 0.2]
    print(f"\n满足「目标|cos|≤0.75 且过 P1/P2」的 $P_v$ 尺度: {ok or '无'}")


if __name__ == "__main__":
    main()
