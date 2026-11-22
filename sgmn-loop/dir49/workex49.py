#!/opt/pytorch/bin/python
r"""DIR-49 · 工作例（**独立第二链**）：不用 `R1_task49` 的任何函数，从零手写同一批计算。

★ 目的：给卡 §3.3 与 `A49-2` 提供逐位对照。本文件**只**借用装置常量（`M.ALPHA/LAM/ETA`、`R.build`）与冻结常数；
   内容流、`K/V`、写循环、读数、正交构造**全部重写**。

用法：`workex49.py`（非零退出码 = 不一致）
"""
import math, sys
import torch
torch.set_default_dtype(torch.float64)

ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir49"):
    sys.path.insert(0, ROOT + "/" + _d)
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402

CTX, N_INT, N_AFTER = 3, 5, 7
N_REP, D_H, RHO_UNUSED = 16, 8, 0.5
NMARK = CTX + 1 + 32       # = 36（与实现同口径）
TAG_C = 7777


def indep_content(sd):
    g = torch.Generator().manual_seed(int(sd) + TAG_C)
    x = torch.randn(NMARK, D_H, generator=g)
    return x / x.norm(dim=1, keepdim=True) * math.sqrt(D_H)


def indep_rebuild(P, a):
    hs = []
    for t in range(a.shape[0]):
        hs.append(torch.tanh(a[t] @ P["A1"] + P["b1"]))
    ks, vs = [], []
    for t in range(a.shape[0]):
        src = torch.zeros(D_H) if t == 0 else hs[t - 1]
        ks.append(torch.tanh(src @ P["Pk1"]))
        vs.append(torch.tanh(hs[t] @ P["Pv1"]))
    return ks, vs


def indep_ortho(P, a, target_k, iters=300, restarts=4, seed=12345):
    A1, b1, Pk1 = P["A1"], P["b1"], P["Pk1"]
    base = a[CTX + 1].clone()
    best_x, best_ov = None, 9.9
    g = torch.Generator().manual_seed(seed)
    for rs in range(restarts):
        if rs == 0:
            x = base.clone().requires_grad_(True)
        else:
            x = (base + 0.5 * torch.randn(D_H, generator=g)).requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.05)
        for _ in range(iters):
            k5 = torch.tanh(torch.tanh(x @ A1 + b1) @ Pk1)
            loss = (k5 @ target_k) ** 2 / (float(k5.norm() * target_k.norm()) ** 2 + 1e-30)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                x.clamp_(-3 * math.sqrt(D_H), 3 * math.sqrt(D_H))
        with torch.no_grad():
            k5 = torch.tanh(torch.tanh(x @ A1 + b1) @ Pk1)
            ov = abs(float(k5 @ target_k) / (float(k5.norm() * target_k.norm()) + 1e-300))
            if ov < best_ov:
                best_ov, best_x = ov, x.detach().clone()
    return best_x, best_ov


def indep_run(sd, arm):
    P = R.build(sd, "V0")
    a0 = indep_content(sd)
    ks, vs = indep_rebuild(P, a0)
    U, V = P["U1"].clone(), P["V1"].clone()

    def w_of():
        return M._W(U, V)

    def advance(k, v):
        nonlocal U, V
        for _ in range(N_REP):
            Wm = w_of()
            GG = M.ALPHA * torch.outer(Wm @ k - v, k) + M.LAM * Wm
            U = U - M.ETA * (GG @ V)
            V = V - M.ETA * (GG.t() @ U)

    def rd():
        Wm = w_of()
        den = torch.linalg.norm(Wm @ ks[CTX]) * torch.linalg.norm(vs[CTX])
        num = (Wm @ ks[CTX]) @ vs[CTX]
        return float(num / den) if float(den) > 0 else float("nan")

    for t in range(CTX + 1):
        advance(ks[t], vs[t])
    c0 = rd()
    kt, vt = ks[CTX], vs[CTX]
    a = a0.clone()
    if arm == "ortho":
        x, _ = indep_ortho(P, a, kt)
        a[CTX + 1] = x
    else:
        w = {"d0.0": 0.0, "d0.5": 0.5, "d1.0": 1.0}[arm]
        mix = (1 - w) * a[CTX + 1].unsqueeze(0) + w * a[CTX - 1].unsqueeze(0)
        a[CTX + 1] = mix / mix.norm(dim=1, keepdim=True) * math.sqrt(D_H)
    kk, vv = indep_rebuild(P, a)
    ov_k = abs(float(kk[N_INT] @ kt) / (float(kk[N_INT].norm() * kt.norm())))
    ov_v = abs(float(vv[N_INT] @ vt) / (float(vv[N_INT].norm() * vt.norm())))
    advance(kk[N_INT], vv[N_INT])
    c1 = rd()
    for t in range(CTX + 3, CTX + 3 + N_AFTER):
        advance(kk[t], vv[t])
    cN = rd()
    return {"ov_k": ov_k, "ov_v": ov_v, "c0": c0, "c1": c1, "cN": cN,
            "rel1": (c1 / c0) if (math.isfinite(c0) and c0 > 0) else None,
            "relN": (cN / c0) if (math.isfinite(c0) and c0 > 0) else None,
            "k_tgt": [float(z) for z in kt], "v_tgt": [float(z) for z in vt]}


def main():
    import R1_task49 as T49
    seeds = [29400, 29401, 29402]
    worst = {"ov_k": 0.0, "ov_v": 0.0, "c0": 0.0, "c1": 0.0, "cN": 0.0}
    bad = 0
    for sd in seeds:
        for arm in T49.ARMS:
            ind = indep_run(sd, arm)
            got = T49.trace_one(sd, arm)
            for k in ("ov_k", "ov_v", "c0", "c1", "cN"):
                a_, b_ = ind[k], got[k]
                if a_ is None or b_ is None:
                    continue
                d = abs(a_ - b_)
                worst[k] = max(worst[k], d)
                if d > 1e-9:
                    bad += 1
                    print(f"✗ sd={sd} arm={arm} {k}: indep={a_!r} impl={b_!r} |Δ|={d:.3e}")
            for k in ("k_tgt", "v_tgt"):
                d = max(abs(x - y) for x, y in zip(ind[k], got[k]))
                if d > 0.0:
                    bad += 1
                    print(f"✗ sd={sd} arm={arm} {k} 非逐位相同 max|Δ|={d:.3e}")
        print(f"  sd={sd} 比对完（4 臂）", flush=True)
    print(f"\n工作例：{len(seeds)} 实例 × 4 臂；逐量 max|Δ| = "
          + ", ".join(f"{k}:{v:.2e}" for k, v in worst.items()) + f"；不一致 {bad}")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
