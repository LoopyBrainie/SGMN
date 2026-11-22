#!/opt/pytorch/bin/python
r"""DIR-48 · 实现：**信息寿命分解** —— 目标对在写入流中的余弦半衰期，三臂拆「`λ` 正则衰减 vs `α` 新写入干扰」。

卡：`revolve/DIR-48/DIRECTION_rev48_v10.md`（rev 1.0，md5 `2d928c9c24cda9ffe0e1e7e76fbcfdba`，**阈值已冻结**）。
★ **逐字派生**：装置（`dir31/R1_grad.py` + `dir33/R1_task33.py`）、内容生成器（`dir46/R1_task46.content`）、
   写入内层循环（`dir47/R1_task47.write_pair`）**一字不改**；唯一改动 = 把 `α` 提为**显式参数**（`decay` 臂传 `0.0`）。
★ **一实例 = 一个目标对 = 一条寿命** `L`（半衰期，`RHO=0.5·c(0)`，右删失记 `L=W` 且 `censored=True`）。

用法：`R1_task48.py --arm {quiet,decay,busy} --n 1000 --seed0 26200 --out PATH [--tag r10]`
"""
import argparse
import json
import math
import os
import sys
import time

import torch

torch.set_default_dtype(torch.float64)

ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task46 as T46      # noqa: E402
import R1_task47 as T47      # noqa: E402

# ══════ 冻结常数（卡 §2；**开卡后一字不改**）
CTX = 3
W_WIN = {"quiet": 16, "decay": 32, "busy": 8}     # ★ rev 1.1：busy 按发散预算定（DEF-48-2）
N_REP = T47.N_REP            # 16
RHO = 0.5
D_H = T47.D_H
ARMS = ("quiet", "decay", "busy")
CONTENT_TAG = 7777           # 逐字沿用 dir46
# ★★ `DEF-48-4`：`torch.randn(n, D_H, generator=g)` 的取值**依赖 `n`**（分块取样）⇒ 若按各臂窗口长度分别生成，
#    同种子下 `k/v` 会**不同**，"三臂用同一批 k/v"就不成立。⇒ **一律按最大窗口生成长度 `NMARK`**，再按需切片。
NMARK = CTX + 1 + max(W_WIN.values())      # = 36（quiet/busy 只用前 19/20 个，decay 全用）


def content(sd, npairs):
    r"""固定 `NMARK` 长度的内容流（**逐字**沿用 `dir46/R1_task46.content` 的 perstep 分支）。"""
    assert npairs <= NMARK, (npairs, NMARK)
    return T46.content(sd, NMARK, "perstep")[:npairs]


def build_KV(sd, npairs):
    r"""★ 内容长度恒为 `NMARK`，只切前 `npairs` 个 ⇒ 三臂前缀**逐位相同**（`A48-3`）。"""
    P = R.build(sd, "V0")
    a = content(sd, npairs)
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(npairs)]
    zero = torch.zeros(D_H)
    K = [torch.tanh((zero if t == 0 else H[t - 1]) @ P["Pk1"]) for t in range(npairs)]
    V = [torch.tanh(H[t] @ P["Pv1"]) for t in range(npairs)]
    return P, K, V


def step_pair(U, Vm, k, v, alpha):
    r"""一次「对」的写入 = `N_REP` 步（**逐字**抄 `R1_task47.write_pair`，仅 `α` 显式化）。"""
    for _ in range(N_REP):
        Wm = M._W(U, Vm)
        G = alpha * torch.outer(Wm @ k - v, k) + M.LAM * Wm
        U = U - M.ETA * (G @ Vm)
        Vm = Vm - M.ETA * (G.t() @ U)
    return U, Vm


def read_target(U, Vm, k, v):
    r"""纯读（**不得**改状态）：返回 `(cos, ‖W‖)`。"""
    Wm = M._W(U, Vm)
    num = float(Wm @ k @ v)
    den = float((Wm @ k).norm() * v.norm())
    return (num / den if den > 0 else float("nan")), float(Wm.norm())


def trace_one(sd, arm):
    assert arm in ARMS
    W = W_WIN[arm]
    P, K, V = build_KV(sd, CTX + W)
    U, Vm = P["U1"].clone(), P["V1"].clone()
    n_writes = 0
    for t in range(CTX + 1):                       # ★ 前缀 + 目标对：三臂**逐位相同**
        U, Vm = step_pair(U, Vm, K[t], V[t], 1.0)
        n_writes += 1
    kt, vt = K[CTX], V[CTX]
    c, wn = [], []
    cos0, w0 = read_target(U, Vm, kt, vt)
    c.append(cos0); wn.append(w0)
    for d in range(1, W):
        if arm == "decay":
            U, Vm = step_pair(U, Vm, K[CTX + d], V[CTX + d], 0.0)   # 同步数、α=0 ⇒ 只有 λW
            n_writes += 1
        elif arm == "busy":
            U, Vm = step_pair(U, Vm, K[CTX + d], V[CTX + d], 1.0)   # 新对
            n_writes += 1
        # quiet：不推进（纯读数）
        cc, ww = read_target(U, Vm, kt, vt)
        c.append(cc); wn.append(ww)

    finite = all(math.isfinite(x) for x in c)
    undef = (not finite) or (not math.isfinite(cos0)) or (cos0 <= 0)
    L, cens = None, None
    if not undef:
        thr = RHO * cos0
        L, cens = W, True
        for d in range(1, W):
            if c[d] < thr:
                L, cens = d, False
                break
    row = {"seed": sd, "arm": arm, "W": W, "n_writes": n_writes,
           "c": c, "wnorm": wn, "c0": cos0,
           "finite": finite, "undef": bool(undef), "L": L, "censored": cens}
    # ★ A48-1 / A48-5 逐实例当场断言
    assert len(c) == W and len(wn) == W, (sd, arm, len(c), W)
    assert n_writes == (CTX + 1 if arm == "quiet" else CTX + 1 + (W - 1)), (sd, arm, n_writes)
    if not undef:
        assert (cens is True) == (L == W), (sd, arm, L, cens)
        if L < W:
            assert c[L] < RHO * cos0 and all(c[d] >= RHO * cos0 for d in range(1, L)), (sd, arm, L)
    if arm == "quiet":
        # ★ `DEF-48-6`：`quiet` **也**可能 `undef` —— 发散来自前 4 次写（前缀含 `α=1`），与"读数"无关。
        #    故不能断言"quiet 永不 undef"；只能断言：**读数不改状态**（NaN 感知的逐位相同）。
        same = lambda a, b: (a == b) or (math.isnan(a) and math.isnan(b))
        assert all(same(x, c[0]) for x in c) and all(same(x, wn[0]) for x in wn), (sd, "quiet 状态被读数改动")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=26200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="r10")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(a.n):
            row = trace_one(a.seed0 + i, a.arm)
            row["tag"] = a.tag
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"[{a.arm}] {i+1}/{a.n}  {time.time()-t0:.1f}s", flush=True)
    print(f"DONE {a.arm} n={a.n} seed0={a.seed0} → {a.out}  墙钟 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
