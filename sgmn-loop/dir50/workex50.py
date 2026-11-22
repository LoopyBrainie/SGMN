#!/opt/pytorch/bin/python
r"""DIR-50 · **工作例（独立第二链）**：**不 import `dev50.rollout`**，把两层接线 + 写动力学 + 读数 + 剂量构造
**另写一遍**（逐字沿用 DIR-49 的装置与 `dir31` 的接线），再与实现**逐位**比对。

★ 只共享**参数生成器**（`dir33.build`）与**外部库**（`R1_grad` 的 `_W`/常数）—— 动力学与读数路径**独立实现**。
用法：`workex50.py --n 3 --out dir50/out/DIR50_workex.json`
"""
import argparse
import json
import math
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir50"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task46 as T46      # noqa: E402
import dev50 as D            # noqa: E402

CTX, T_TGT, T_HIT, W_AFTER = D.CTX, D.T_TGT, D.T_HIT, D.W_AFTER
N_REP = D.N_REP
D_H, D_K = D.D_H, D.D_K
T_STEPS = D.T_STEPS
ATOL = 1e-9


def chain2(P, a, arm, dose):
    r"""**独立**的两层 rollout（第二链）。"""
    U = [P["U1"].clone(), P["U2"].clone()]
    Vm = [P["V1"].clone(), P["V2"].clone()]
    prev = [None, None]
    c0 = [None, None]
    curve = [[], []]
    tgt = [None, None]
    ov_k = None
    for t in range(T_STEPS):
        hp_prev = a[t]
        hpre = [None, None]
        hp = [None, None]
        for li, l in ((0, 1), (1, 2)):
            h = torch.tanh(hp_prev @ P["A%d" % l] + P["b%d" % l])
            hpre[li] = h
            csrc = a[t] if l == 1 else hp_prev
            c = csrc.detach()
            q = torch.tanh(c @ P["Wq%d" % l] + P["bq%d" % l])
            W = U[li] @ Vm[li].t()
            y = W.detach() @ q
            m = torch.tanh(torch.cat([c, y], 0) @ P["Wo%d" % l] + P["bo%d" % l])
            hp[li] = h + P["g%d" % l] * m
            hp_prev = hp[li]
        for li, l in ((0, 1), (1, 2)):
            hprev = prev[li]
            if hprev is None:
                hprev = torch.zeros(D_H)
            k = torch.tanh(hprev.detach() @ P["Pk%d" % l])
            v = torch.tanh(hpre[li].detach() @ P["Pv%d" % l])
            if t == T_HIT and arm != "none" and li == (0 if arm == "hit1" else 1):
                # ★ 独立实现的剂量构造：同样的规格（种子、正交化、范数匹配），独立写一遍
                ut = tgt[li][0] / float(tgt[li][0].norm())
                g = torch.Generator().manual_seed(1000 + 7919 * t + li)
                rr = torch.randn(D_K, generator=g, dtype=ut.dtype)
                rr = rr - (rr @ ut) / (ut @ ut) * ut
                rr = rr / float(rr.norm())
                kt = dose * ut + math.sqrt(max(0.0, 1.0 - dose * dose)) * rr
                kn = float(k.norm())
                k = kt * kn
                ov_k = abs(float(k @ tgt[li][0]) / (float(k.norm() * tgt[li][0].norm()) + 1e-300))
            # ★ 独立实现的写内循环：**逐字复现装置** `dir48.step_pair` 的语义 ——
            #   ★ 该步是**非同步**的（`V` 的更新用**新** `U`）；这与 v2.4 §3.3 的「同步步」**不同**，
            #   但**必须**与装置一致（否则第二链测的是另一个系统）。该偏差已在 D-14 量化（`η²` 标度）。
            for _ in range(N_REP):
                W = U[li] @ Vm[li].t()
                G = 1.0 * torch.outer(W @ k - v, k) + 0.05 * W
                U[li] = U[li] - 0.2 * (G @ Vm[li])            # 新 U
                Vm[li] = Vm[li] - 0.2 * (G.t() @ U[li])      # ★ 用**新** U（非同步步 = 装置语义）
            if t == T_TGT:
                tgt[li] = (k.clone(), v.clone())
        if t == T_TGT:
            for li in (0, 1):
                W = U[li] @ Vm[li].t()
                num = float(W @ tgt[li][0] @ tgt[li][1])
                den = float((W @ tgt[li][0]).norm() * tgt[li][1].norm())
                c0[li] = num / den if den > 0 else float("nan")
        elif t > T_TGT:
            d = t - T_TGT
            if d <= W_AFTER:
                for li in (0, 1):
                    W = U[li] @ Vm[li].t()
                    num = float(W @ tgt[li][0] @ tgt[li][1])
                    den = float((W @ tgt[li][0]).norm() * tgt[li][1].norm())
                    curve[li].append(num / den if den > 0 else float("nan"))
        prev = hpre
    return {"c0": c0, "curve": curve, "ov_k": ov_k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conds = [("none", None)] + [("hit1", d) for d in (0.0, 1.0)] + [("hit2", d) for d in (0.0, 1.0)]
    maxd = 0.0
    n_cmp = 0
    rows = []
    for i in range(a.n):
        sd = 31000 + i
        P = R.build(sd, "V0")
        cont = T46.content(sd, D.NMARK, "perstep")
        for arm, dose in conds:
            rr = D.rollout(P, cont, arm, dose)
            cc = chain2(P, cont, arm, dose)
            d = 0.0
            for li in (0, 1):
                x, y = rr["c0"][li], cc["c0"][li]
                if math.isnan(x) and math.isnan(y):
                    continue
                d = max(d, abs(x - y))
                for u, v in zip(rr["curve"][li], cc["curve"][li]):
                    if math.isnan(u) and math.isnan(v):
                        continue
                    d = max(d, abs(u - v))
            if rr["ov_k"] is not None and cc["ov_k"] is not None:
                d = max(d, abs(rr["ov_k"] - cc["ov_k"]))
            maxd = max(maxd, d)
            n_cmp += 1
            rows.append({"seed": sd, "arm": arm, "dose": dose, "max_abs_diff": d})
    out = {"n_inst": a.n, "n_cmp": n_cmp, "atol": ATOL, "max_abs_diff": maxd,
           "ok": maxd <= ATOL, "rows": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
