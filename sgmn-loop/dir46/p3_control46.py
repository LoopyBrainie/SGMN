#!/opt/pytorch/bin/python
r"""DIR-46 · 证明义务 **P-3**（反向对照）＋ 附带对照 **P-3b**（卡 §9；**均为诊断，不改判据**）。

- **P-3（冻结原文照做）**：`period2` 内容下把 `Pk1` 换成**随机正交版本**（`Pk1 @ R`，`R` 为随机正交 `16×16`）
  ⇒ 取回相似度 **是否**显著下降。
- **P-3b（本轮追加，如实披露）**：保持键不变，但**打乱 k↔v 配对**（第 `s` 个键去比第 `s+1` 个值）
  ⇒ `C` 必须**显著下降**（否则 `C` 根本不反映"键-值配对"，测量对象即失效）。

★ 本文件**不 import** `R1_task46`、**不 import** `probe46a`；测量层手写（与 `workex46.c_indep` 同式）。
"""
import json
import math
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M       # noqa: E402
import R1_task33 as R     # noqa: E402

OUT = os.path.join(ROOT, "dir46", "out")
SEEDS = [M.SEED0 + 23800 + i for i in range(12)]     # 与 Stage A / probe 同段（诊断用）
N_W, N_REP, D_H = 8, 16, R.D_H


def content(sd, n_w, mode="period2"):
    g = torch.Generator().manual_seed(int(sd) + 7777)
    x = torch.randn(n_w, D_H, generator=g)
    x = x / x.norm(dim=1, keepdim=True) * (D_H ** 0.5)
    return torch.stack([x[t % 2] for t in range(n_w)]) if mode == "period2" else x


def trace(sd, arm, n_w=N_W):
    r"""`arm ∈ {genuine, rot_pk, swap_v}`（`swap_v` 只改**读**时的配对，不改写入）。"""
    P = R.build(sd, "V0")
    Pk = P["Pk1"]
    if arm == "rot_pk":
        g = torch.Generator().manual_seed(int(sd) + 999)
        A = torch.randn(Pk.shape[1], Pk.shape[1], generator=g)
        Q, _ = torch.linalg.qr(A)                     # 随机正交 16×16（`QᵀQ=I`）
        Pk = Pk @ Q                                   # ★ 保留原范数谱 ⇒ 只换方向
    a = content(sd, n_w)
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(n_w)]
    zero = torch.zeros(D_H)
    K = [torch.tanh((zero if s == 0 else H[s - 1]) @ Pk) for s in range(n_w)]
    Vv = [torch.tanh(H[s] @ P["Pv1"]) for s in range(n_w)]
    U, V = P["U1"].clone(), P["V1"].clone()
    for s in range(n_w):
        k, v = K[s], Vv[s]
        for _ in range(N_REP):
            W = M._W(U, V)
            G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
            U = U - M.ETA * (G @ V)
            V = V - M.ETA * (G.t() @ U)
    W = M._W(U, V)
    if not bool(torch.isfinite(W).all()):
        return None
    cs = []
    for s in range(1, n_w):
        tgt = Vv[s + 1] if (arm == "swap_v" and s + 1 < n_w) else Vv[s]
        num = float(W @ K[s] @ tgt)
        den = float((W @ K[s]).norm() * tgt.norm())
        cs.append(num / den if den > 0 else float("nan"))
    cs = [c for c in cs if math.isfinite(c)]
    return (sum(cs) / len(cs)) if cs else None


def main():
    res = {"meta": {"n_w": N_W, "n": len(SEEDS), "seeds": SEEDS,
                    "note": "P-3 = 冻结原文；P-3b = 本轮附带的配对破坏对照（诊断，均不进判据）"}}
    arms = {}
    for arm in ("genuine", "rot_pk", "swap_v"):
        vals = [trace(sd, arm) for sd in SEEDS]
        good = [v for v in vals if v is not None]
        mu = sum(good) / len(good)
        s = math.sqrt(sum((x - mu) ** 2 for x in good) / (len(good) - 1)) if len(good) > 1 else float("nan")
        arms[arm] = {"mean": mu, "sd": s, "n": len(good), "n_cens": len(vals) - len(good),
                     "values": vals}
        print(f"  {arm:9s} C = {mu:+.5f}  (sd {s:.4f}, n={len(good)}, 删失 {len(vals)-len(good)})")

    def paired(a1, a2):
        d = [x - y for x, y in zip(arms[a1]["values"], arms[a2]["values"])
             if x is not None and y is not None]
        m = sum(d) / len(d)
        s = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
        se = s / math.sqrt(len(d))
        return {"mean_diff": m, "sd": s, "3se": 3 * se, "n": len(d),
                "significant": abs(m) >= 3 * se}

    res["P3_rot_pk"] = paired("genuine", "rot_pk")
    res["P3b_swap_v"] = paired("genuine", "swap_v")
    print(f"  P-3   genuine − rot_pk : {res['P3_rot_pk']['mean_diff']:+.5f}"
          f"  (3se {res['P3_rot_pk']['3se']:.5f}, n={res['P3_rot_pk']['n']})"
          f"  {'★显著' if res['P3_rot_pk']['significant'] else 'ns'}")
    print(f"  P-3b  genuine − swap_v : {res['P3b_swap_v']['mean_diff']:+.5f}"
          f"  (3se {res['P3b_swap_v']['3se']:.5f}, n={res['P3b_swap_v']['n']})"
          f"  {'★显著' if res['P3b_swap_v']['significant'] else 'ns'}")
    res["arms"] = arms
    p = os.path.join(OUT, "DIR46r10_p3.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("→", p)


if __name__ == "__main__":
    main()
