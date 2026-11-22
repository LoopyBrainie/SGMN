#!/opt/pytorch/bin/python
r"""DIR-48 · 开卡前诊断 B：把窗口拉到 `W=32`（=5.12 τ_ref）量出**纯衰减寿命**，给"分解"两个数。

★ `probe48a` 已定：(1) `quiet` 臂逐位不变（读数不改状态，设计自检通过）；(2) `W=16` 时 `decay` 跨半 **0.167**、`busy` 跨半 **0.95**（`med_L=2`）。
★ 本探针要回答：**纯 λ 衰减的寿命到底是"16 以内/16 附近"还是"远大于 16"** —— 这决定卡的判据是用"窗口内跨半"还是"中位寿命"。
★ 另做两件事：① 把**非有限**实例单独计（`n_div`），与**右删失**严格分开；② 检查 `quiet` 臂 `C(d)` 与 `C(0)` **逐位相同**（设计自检的硬断言）。

用法：`probe48b.py --n 200 --w 32 --out dir48/out/DIR48_probe_b.json`
"""
import argparse, json, math, os, sys
import torch
torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir48"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import probe48a as PA        # noqa: E402

RHO = 0.5


def km_median(Ls, cens_flag, w):
    r"""Kaplan–Meier 中位数（**删失感知**）：返回 (med, reached)。`cens_flag[i]=True` 表示 `L=w` 且为右删失。"""
    pairs = sorted(zip(Ls, cens_flag), key=lambda x: x[0])
    n_at = len(pairs)
    surv, s = 1.0, None
    i = 0
    while i < n_at:
        t = pairs[i][0]
        d = 0
        c = 0
        while i < n_at and pairs[i][0] == t:
            if pairs[i][1]:
                c += 1
            else:
                d += 1
            i += 1
        if n_at > 0 and d:
            surv *= (1 - d / n_at)
        if surv <= 0.5 and s is None:
            s = t
        n_at -= (d + c)
        if n_at <= 0:
            break
    return (s, s is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--w", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=26000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W = a.w
    out = {"n": a.n, "w": W, "seed0": a.seed0, "tau_ref_steps": 1 / (0.2 * 0.05),
           "tau_ref_pairs": 1 / (0.2 * 0.05) / 16, "arms": {}}
    for arm in PA.ARMS:
        Ls, cs, div, quiet_ident, curves = [], [], 0, True, []
        for i in range(a.n):
            sd = a.seed0 + i
            c = PA.run(sd, arm, W)
            if not all(math.isfinite(x) for x in c):
                div += 1
                continue
            if arm == "quiet" and any(x != c[0] for x in c[1:]):
                quiet_ident = False
            L, c0 = PA.halflife(c)
            if L is None:
                div += 1
                continue
            Ls.append(L); cs.append(L >= W); curves.append(c)
        n_ok = len(Ls)
        n_cens = sum(1 for x in cs if x)
        med, reached = km_median(Ls, cs, W)
        mean_c = [sum(x[d] for x in curves) / len(curves) for d in range(W)] if curves else []
        out["arms"][arm] = {"n": a.n, "n_div": div, "n_use": n_ok,
                            "cens_frac": n_cens / n_ok if n_ok else None,
                            "cross_frac": (n_ok - n_cens) / n_ok if n_ok else None,
                            "cross_frac_at16": (sum(1 for L, cc in zip(Ls, cs) if (not cc) and L <= 15) / n_ok) if n_ok else None,
                            "km_median": med, "km_reached": reached,
                            "mean_c0": (sum(x[0] for x in curves) / len(curves)) if curves else None,
                            "half_c_at8": mean_c[8] if len(mean_c) > 8 else None,
                            "half_c_at16": mean_c[16] if len(mean_c) > 16 else None,
                            "half_c_at31": mean_c[31] if len(mean_c) > 31 else None,
                            "mean_curve": [round(x, 4) for x in mean_c],
                            "quiet_bit_identical": (quiet_ident if arm == "quiet" else None)}
        print(f"[{arm:5s}] n_use={n_ok} n_div={div} cens={out['arms'][arm]['cens_frac']:.3f} "
              f"cross={out['arms'][arm]['cross_frac']:.3f} (≤15: {out['arms'][arm]['cross_frac_at16']:.3f}) "
              f"KM_med={med}{'' if reached else '(未达)'} mean_c0={out['arms'][arm]['mean_c0']:+.4f}", flush=True)
        if arm == "quiet":
            print(f"        ★ quiet 逐位不变 = {quiet_ident}", flush=True)
        print("        c@0/8/16/31:", out["arms"][arm]["mean_c0"],
              out["arms"][arm]["half_c_at8"], out["arms"][arm]["half_c_at16"],
              out["arms"][arm]["half_c_at31"], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
