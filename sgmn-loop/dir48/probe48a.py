#!/opt/pytorch/bin/python
r"""DIR-48 · 开卡前诊断 A：**信息寿命分解** —— 单对目标在写入流中的余弦半衰期，三臂拆「正则衰减 vs 新写入干扰」。

★ 本轮**测量形式的要害**（决定了能不能有干净的 `d`）：若一次写一整块（如 DIR-46/47 的 `S=8`），
   块内各槽位**被写时刻不同** ⇒ 同一名义 `d` 下它们的真实时距不同（混淆）。故本卡改为 **只放 1 个目标对**：
   前面 `ctx` 对上下文（只为让 `W` 非空）、目标对、之后 `W` 步"经历"。⇒ `d` = 目标对写入后**经过的新对数**，无混淆。

三臂（**唯一自变量 = 目标对写入后那些步"里面放了什么"**）：
  `quiet`  —— 什么都不做（只读数）⇒ **设计自检锚**：读数不得改变状态（`C(d)` 必须逐位相同）
  `decay`  —— 同样步数，但写入项 `α=0`（只有 `λW` 正则项）⇒ **纯衰减臂**，隔离 `τ_ref≈1/(ηλ)=100` 步
  `busy`   —— 同样步数，写入**新对**（`α=1`）⇒ `λ + α` 干扰

读数：`C(d) = cos(W k_target, v_target)`，`d = 0..W−1`；半衰期 `L := min{d : C(d) < ρ·C(0)}`（`ρ=0.5`），右删失于 `W`。

用法：`probe48a.py --n 60 --w 16 --out dir48/out/DIR48_probe_a.json`
"""
import argparse, json, math, os, sys
import torch
torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task46 as T46      # noqa: E402

CTX, W_WIN, NREP, D_H = 3, 16, 16, 8
RHO = 0.5
ARMS = ("quiet", "decay", "busy")


def stream(sd, npairs):
    P = R.build(sd, "V0")
    a = T46.content(sd, npairs, "perstep")
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(npairs)]
    zero = torch.zeros(D_H)
    K = [torch.tanh((zero if t == 0 else H[t - 1]) @ P["Pk1"]) for t in range(npairs)]
    V = [torch.tanh(H[t] @ P["Pv1"]) for t in range(npairs)]
    return P, K, V


def step_pair(U, Vm, k, v, alpha):
    for _ in range(NREP):
        Wm = M._W(U, Vm)
        G = alpha * torch.outer(Wm @ k - v, k) + M.LAM * Wm
        U = U - M.ETA * (G @ Vm)
        Vm = Vm - M.ETA * (G.t() @ U)
    return U, Vm


def cos_at(U, Vm, k, v):
    Wm = M._W(U, Vm)
    num = float(Wm @ k @ v)
    den = float((Wm @ k).norm() * v.norm())
    return num / den if den > 0 else float("nan")


def run(sd, arm, w):
    P, K, V = stream(sd, CTX + 1 + w)
    U, Vm = P["U1"].clone(), P["V1"].clone()
    for t in range(CTX + 1):                       # ★ 前缀：三臂逐位相同
        U, Vm = step_pair(U, Vm, K[t], V[t], 1.0)
    kt, vt = K[CTX], V[CTX]                        # ★ 目标对
    c = [cos_at(U, Vm, kt, vt)]
    for d in range(1, w):
        if arm == "busy":
            U, Vm = step_pair(U, Vm, K[CTX + d], V[CTX + d], 1.0)
        elif arm == "decay":
            U, Vm = step_pair(U, Vm, K[CTX + d], V[CTX + d], 0.0)   # α=0：只有 λW
        # quiet: 不推进（纯读数）
        c.append(cos_at(U, Vm, kt, vt))
    return c


def halflife(c, rho=RHO):
    c0 = c[0]
    if not math.isfinite(c0) or c0 <= 0:
        return None, c0
    thr = rho * c0
    for d in range(1, len(c)):
        if c[d] < thr:
            return d, c0
    return len(c), c0          # 右删失：L = W（censored）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--w", type=int, default=W_WIN)
    ap.add_argument("--seed0", type=int, default=26000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"n": a.n, "w": a.w, "seed0": a.seed0, "arms": {}}
    for arm in ARMS:
        Ls, c0s, curves, bad = [], [], [], 0
        for i in range(a.n):
            sd = a.seed0 + i
            c = run(sd, arm, a.w)
            L, c0 = halflife(c)
            curves.append(c); c0s.append(c0)
            if L is None:
                bad += 1
            else:
                Ls.append(L)
        ws = a.w
        n_cens = sum(1 for L in Ls if L >= ws)
        cross = (len(Ls) - n_cens) / a.n
        med_L = sorted(Ls)[len(Ls) // 2] if Ls else None
        mean_c = [sum(c[d] for c in curves) / len(curves) for d in range(ws)]
        out["arms"][arm] = {"n": a.n, "n_undef_c0": bad, "cens": n_cens / max(1, len(Ls)),
                            "cross_frac": cross, "med_L": med_L,
                            "mean_L": sum(Ls) / len(Ls) if Ls else None,
                            "mean_c0": sum(c0s) / len(c0s),
                            "min_c0": min(c0s), "max_c0": max(c0s),
                            "mean_curve": [round(x, 4) for x in mean_c]}
        print(f"[{arm:5s}] n={a.n} c0∈[{min(c0s):+.3f},{max(c0s):+.3f}] mean_c0={out['arms'][arm]['mean_c0']:+.4f} "
              f"cross={cross:.3f} cens={out['arms'][arm]['cens']:.3f} med_L={med_L} undef={bad}", flush=True)
        print("        curve:", " ".join(f"{x:+.3f}" for x in mean_c[:ws]), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
