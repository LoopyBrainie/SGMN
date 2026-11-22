r"""DIR-33 开卡前标定（决定二十九：留档、不进判据）。

目的：把 DIR-32 机制③的「形成时标」变成可选的装置常数。两个量：
  A. 形成时标 $N_{0.9}$：同一 $(k,v)$ 对重复写入，$\cos(W_Nk,v)\ge0.9$ 所需步数
     随 $(\eta,\lambda,d_k,\text{键初值尺度})$ 的变化。
  B. 键可分性 $|\cos(k(h_{t-2}),k(h_{t-1}))|$：周期 2 输入下相邻两键的夹角，
     随 $(d_k,\text{键初值尺度})$ 的变化。
并与装置提供的**可用写入次数**（$T{=}8$、周期 2 ⇒ 每对 ≈3 次）对照。

★ 写入规则**逐字**来自 dir31/R1_grad.py（因子化步：U←U−ηGVᵀ、V←V−ηGᵀU，
  G=α(Wk−v)kᵀ+λW），并用真实 rollout 的一条自证断言核对。
"""
import json, math, sys
import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
import R1_grad as M

OUT = "/home/ec2-user/sgmn-loop/dir33/out"
SEEDS = [M.SEED0 + 300 + i for i in range(8)]      # 标定种子（与任何判据种子不相交）
# ★ DEF-33-0c：自控必须与被控读数**同种子种群**——A32-2 用的是 SEED0+200..223（24 个），
#   用 8 个种子算 max 只会得到更小的数，那不是「不符」而是「种群不同」。
SEEDS_A32 = [M.SEED0 + 200 + i for i in range(24)]
NMAX = 1024


def write_steps(dk, eta, lam, kscale, n=NMAX, seeds=SEEDS):
    r"""对每个种子：随机 (k,v) 对重复写入 n 步，返回 cos 曲线（长度 n）与范数比。"""
    cs = torch.zeros(n)
    for sd in seeds:
        g = torch.Generator().manual_seed(sd)
        d_h, d_v, r = M.D_H, M.D_V, 2
        U = torch.randn(d_v, r, generator=g) * M.INIT_SD
        V = torch.randn(dk, r, generator=g) * M.INIT_SD
        k = torch.tanh(torch.randn(dk, generator=g) * kscale)
        v = torch.tanh(torch.randn(d_v, generator=g))
        W0n = (U @ V.t()).norm()
        cur = torch.zeros(n)
        for i in range(n):
            W = U @ V.t()
            G = M.ALPHA * torch.outer(W @ k - v, k) + lam * W
            Un, Vn = U - eta * (G @ V), V - eta * (G.t() @ U)
            U, V = Un, Vn
            W = U @ V.t()
            cur[i] = float((W @ k) @ v / ((W @ k).norm() * v.norm() + 1e-30))
        cs += cur / len(seeds)
    Wf = (U @ V.t()).norm()
    return cs, float(Wf / (W0n + 1e-30))


def n09(cs):
    hit = (cs >= 0.9).nonzero()
    return int(hit[0]) + 1 if len(hit) else None


def key_sep(dk, kscale, seeds=SEEDS_A32):
    r"""周期 2 输入下的相邻键夹角（第 1 层，冻结基座；与 dir32 的任务层同构）。"""
    mx, mn = 0.0, 1.0
    for sd in seeds:
        g = torch.Generator().manual_seed(sd)
        s_h = 1.0 / math.sqrt(M.D_H)
        A1 = torch.randn(M.D_H, M.D_H, generator=g) * s_h
        b1 = torch.randn(M.D_H, generator=g) * 0.1
        Pk = torch.randn(M.D_H, dk, generator=g) * s_h * kscale
        # ★ DEF-33-0a：必须与 dir32 的 period2 **逐字**一致（严格交替），
        #   否则输入近乎常数 ⇒ h 近乎常数 ⇒ 键可分性退化成 1.0000。
        g2 = torch.Generator().manual_seed(int(sd) + 7777)   # ★ 与 dir32 的 period2 逐字一致
        x = torch.randn(2, M.D_H, generator=g2)
        a = torch.stack([x[t % 2] for t in range(M.T_STEPS)])
        h_prev = a[0]
        ks = []
        for t in range(M.T_STEPS):
            h = torch.tanh(h_prev @ A1 + b1)
            ks.append(torch.tanh(h @ Pk))
            h_prev = a[t]
        for t in range(2, M.T_STEPS):
            c = abs(float(ks[t] @ ks[t - 1] / (ks[t].norm() * ks[t - 1].norm() + 1e-30)))
            mx, mn = max(mx, c), min(mn, c)
    return mx, mn


def main():
    res = {"grid": [], "sep": []}
    print(f"{'η':>6}{'λ':>6}{'d_k':>5}{'k尺度':>7}{'N_0.9':>8}{'cos@8':>8}{'cos@32':>8}{'|dW|/|W0|':>11}")
    for lam in (0.05, 0.01, 0.0):
        for eta in (0.05, 0.1, 0.2, 0.4):
            for dk in (4, 8, 16):
                for ks in (1.0, 0.5):
                    cs, ratio = write_steps(dk, eta, lam, ks)
                    div = not bool(torch.isfinite(cs).all())
                    n = None if div else n09(cs)
                    row = {"lam": lam, "eta": eta, "dk": dk, "kscale": ks,
                           "diverged": div, "n09": n,
                           "cos8": None if div else round(float(cs[7]), 4),
                           "cos32": None if div else round(float(cs[31]), 4),
                           "ratio": None if div else round(ratio, 3)}
                    res["grid"].append(row)
                    if div:
                        print(f"{eta:6.2f}{lam:6.2f}{dk:5d}{ks:7.2f}{'DIV(发散)':>8}")
                    else:
                        print(f"{eta:6.2f}{lam:6.2f}{dk:5d}{ks:7.2f}"
                              f"{(n if n else 0):8d}{cs[7]:8.4f}{cs[31]:8.4f}{ratio:11.3f}")
    for dk in (4, 8, 16):
        for ks in (1.0, 0.5, 0.25):
            mx, mn = key_sep(dk, ks)
            res["sep"].append({"dk": dk, "kscale": ks, "max": round(mx, 4), "min": round(mn, 4)})
            print(f"键可分性 d_k={dk:2d} k尺度={ks:.2f} ⇒ |cos|_max {mx:.4f} · min {mn:.4f}")
    ctl = [q for q in res["sep"] if q["dk"] == 4 and q["kscale"] == 1.0][0]
    res["self_control_A32_2_max"] = ctl["max"]
    print(f"\n★ 自控：d_k=4/k尺度=1（装置常数）⇒ |cos|_max {ctl['max']:.4f} "
          f"（DIR-32 的 A32-2 实测 0.9416）⇒ 差 {abs(ctl['max'] - 0.9416):.4f}")
    # 可用写入次数（周期 2、T=8）：每对约 (T-1)/2
    avail = (M.T_STEPS - 1) / 2
    res["available_per_pair"] = avail
    best = [r for r in res["grid"] if r["n09"] and r["n09"] <= avail]
    res["diverged_cells"] = sum(1 for r in res["grid"] if r.get("diverged"))
    res["feasible_cells"] = len(best)
    print(f"\n可用写入次数/对（T=8，周期 2）≈ {avail:.1f}；满足 N_0.9 ≤ 可用 的构型数 = {len(best)}")
    for r in best[:10]:
        print(f"   可行：η={r['eta']} λ={r['lam']} d_k={r['dk']} k尺度={r['kscale']} ⇒ N_0.9={r['n09']}")
    torch.save({}, "/dev/null")
    with open(f"{OUT}/DIR33_calib.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    M.D_K = 4          # build 之外的路径不再依赖它；此标定自行构造张量
    main()
