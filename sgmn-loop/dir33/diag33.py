#!/opt/pytorch/bin/python
r"""DIR-33 开卡前诊断（决定二十九，**不进判据**）：为什么 A33-6 与 A33-5 失败？

两个问题：
  ① A33-6 = 0.9666 > 0.9：两个奇偶目标 $g_{\rm even},g_{\rm odd}$ 近乎共线 ⇒ 当前输入就
     几乎定出目标，记忆无从体现。**根因待查**：是 $P_v$ 的抽样，还是两个奇偶输入本身太像？
  ② A33-5 = 1.4e-14：$T$ 由 8→16 后求和项从 7 增到 15，float64 的累积舍入随之变大。
     判断这是**舍入**而非实现漂移（DIR-32 漂移的签名是 $10^{+1}$ 量级）。

另外报三个候选输入构造下的 P1/P2/A33-6，供开卡冻结时选一个**构造上**过关的。
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
VARIANTS = ("draw", "ortho", "ortho_scaled")
_SQ = M.D_H ** 0.5


def period2_variant(sd, variant):
    g = torch.Generator().manual_seed(int(sd) + 7777)
    x = torch.randn(2, R.D_H, generator=g)
    if variant == "draw":
        a = torch.stack([x[t % 2] for t in range(R.T)])
        return a
    # 把 $x_1$ 对 $x_0$ 正交化（去掉投影），再按需恢复原尺度
    x = x.clone()
    x[1] = x[1] - (x[1] @ x[0]) / (x[0] @ x[0]) * x[0]
    if variant == "ortho_scaled":                 # 保持每个向量的原范数量级 ≈ √d_h
        x = x / x.norm(dim=1, keepdim=True) * _SQ
    return torch.stack([x[t % 2] for t in range(R.T)])


def stats(variant, seeds):
    sep, tgt, xcos = [], [], []
    for sd in seeds:
        P = R.build(sd, "V0")
        P["a"] = period2_variant(sd, variant)
        H = R.hs_of(P)
        ks = [torch.tanh(H[s] @ P["Pk1"]) for s in range(R.T)]
        g = R.targets(P)
        for t in range(2, R.T):
            sep.append(abs(float(ks[t] @ ks[t - 1] / (ks[t].norm() * ks[t - 1].norm()))))
        for i in range(len(g)):
            for j in range(len(g)):
                if (i - j) % 2 == 1:
                    tgt.append(abs(float(g[i] @ g[j] / (g[i].norm() * g[j].norm()))))
        x0, x1 = P["a"][0], P["a"][1]
        xcos.append(abs(float(x0 @ x1 / (x0.norm() * x1.norm()))))
    return {"sep_max": max(sep), "tgt_max": max(tgt), "x_cos_max": max(xcos)}


def main():
    res = {}
    for v in VARIANTS:
        st = stats(v, R.SEEDS_F)
        # P1/P2 需要按该构造重算 gates：直接复用 R.gates 的逻辑但换输入
        orig = R.period2
        R.period2 = lambda P, sd, _v=v: P.update({"a": period2_variant(sd, _v)})
        p1, p2 = R.gates()
        R.period2 = orig
        st.update({"P1": p1, "P2": p2})
        res[v] = st
        print(f"{v:14s} 奇偶目标|cos|max {st['tgt_max']:.4f} · 键可分性 {st['sep_max']:.4f} · "
              f"输入|cos|max {st['x_cos_max']:.4f} · P1 {p1:+.4f} · P2 {p2:+.4f}")
    json.dump(res, open(f"{OUT}/DIR33_diag_inputs.json", "w"), indent=1)
    ok6 = [v for v in VARIANTS if res[v]["tgt_max"] <= 0.9]
    ok2 = [v for v in VARIANTS if res[v]["P2"] > 0.2 and res[v]["P1"] >= 0.5]
    print(f"\n过 A33-6(≤0.9) 的构造: {ok6 or '无'}  ·  过 P1/P2 的构造: {ok2 or '无'}")
    print(f"交集（开卡候选）: {[v for v in ok6 if v in ok2] or '无'}")


if __name__ == "__main__":
    main()
