#!/opt/pytorch/bin/python
r"""DIR-58 · 实现：**`S` 的遗忘** —— 常数寿命是否成立 / 两通道是否可分离
（卡 `revolve/DIR-58/DIRECTION_rev58_v10.md`，rev 1.0 冻结，md5 `efc4fce81837752b236d7e68014a9a16`）。

★ 逐字派生：装置（`dir31`/`dir33`/`dir46`/`dir47`/`dir48`/`dir51`）与唯一的写/衰减内层
  `R1_task48.step_pair(U, Vm, k, v, alpha)` **一字不改**；本文件只做**编排 + 读数**。
★ 唯一自变量（卡 §2.2）：在同一**总对数** $q$ 上，是否发生 **1 次** $\alpha{=}1$ 的异值写入：
    `D`（只衰减）：$q$ 次 `alpha=0`；
    `W`（1 次驱逐）：1 次 `alpha=1` 写 $(k_I,v_I)$ + $(q{-}1)$ 次 `alpha=0`（$q\ge1$；$q{=}0$ 不进 W，`DEF-58-1`）。
★ 唯一读出量（卡 §3）：$E=\|Wk-v\|$、$E_{\rm null}=\mathrm{median}_{5\ \text{未写键}}\|Wk'-v_P\|$、$S=E_{\rm null}-E$；
  `S_new` 用**同一口径**，把 $(k_I,v_I)$ 代进去（臂 D 里它是从未写过的零对照）。
★ `read_metrics()` 的源码里**不得出现 `cos`**（`A58-6` 源码断言；键的拒绝采样另在 `key_batch`，不属读数路径）。

用法：`R1_task58.py --arm {D,W} --n 96 --seed0 59500 --out PATH`
"""
import argparse
import hashlib
import json
import os
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir51"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task48 as T48      # noqa: E402
import R1_task51 as T51      # noqa: E402

ARMS = ("D", "W")
QS = {"D": (0, 1, 2, 4, 8), "W": (1, 2, 4, 8)}
OFF_K, OFF_V, OFF_R, OFF_I = 59501, 59502, 59503, 59504
N_UNWRITTEN = 5
COS_MAX = 0.30
DEVICE = ("dir31/R1_grad.py", "dir33/R1_task33.py", "dir46/R1_task46.py",
          "dir47/R1_task47.py", "dir48/R1_task48.py", "dir51/R1_task51.py")
DEVICE_MD5 = {
    "dir31/R1_grad.py": "8fa40f708600bec7e6c484727d2180c7",
    "dir33/R1_task33.py": "96ef89e87ec61556b52e0e464850087e",
    "dir46/R1_task46.py": "d66c849692d93bc5b24a91f9a445e0ef",
    "dir47/R1_task47.py": "fd03c70c6490cfd726489714b9c2c80f",
    "dir48/R1_task48.py": "bb6154687a48d4ea33f0ebd9bc0cb393",
    "dir51/R1_task51.py": "c03af6abc98a28c01e39ff68c6d0ee27",
}


def gen_keys(g, m, max_cos=COS_MAX):
    r"""`m` 把键，两两 $|\cos|\le$ `max_cos`（拒绝采样）。**不是**读数路径。"""
    ks = []
    while len(ks) < m:
        k = R.KEY_SCALE * T51.gen_vec(g, M.D_K)
        if all(abs(float((k @ q) / (k.norm() * q.norm()))) <= max_cos for q in ks):
            ks.append(k)
    return ks


def gen_value(g, v_ref, max_cos=COS_MAX):
    r"""与 `v_ref` 的 $|\cdot|$ 余弦 $\le$ `max_cos` 的单位值（拒绝采样）。**不是**读数路径。"""
    while True:
        v = T51.gen_vec(g, M.D_V)
        if abs(float(v @ v_ref)) <= max_cos:
            return v


def read_metrics(U, Vm, k_old, vP, k_un, k_new, v_new):
    r"""★ 唯一读数路径（`A58-6`：本函数源码中不得出现余弦算符字样）。"""
    W = M._W(U, Vm)
    e_list = [float((W @ k - vP).norm()) for k in k_un]
    e_null = statistics.median(e_list)
    e_old = float((W @ k_old - vP).norm())
    e_new = float((W @ k_new - v_new).norm())
    return {"E_old": e_old, "E_new": e_new, "E_null": e_null,
            "S_old": e_null - e_old, "S_new": e_null - e_new,
            "W_norm": float(W.norm()), "Wk_norm": float((W @ k_old).norm()),
            "E_un": e_list}


def seed_state(seed):
    r"""生成该种子的全部外生对象（两臂**逐位相同**，`A58-9/A58-13`）。"""
    P = R.build(seed, "V0")
    gk = torch.Generator().manual_seed(seed + OFF_K)
    gv = torch.Generator().manual_seed(seed + OFF_V)
    gr = torch.Generator().manual_seed(seed + OFF_R)
    gi = torch.Generator().manual_seed(seed + OFF_I)
    ks = gen_keys(gk, 4)
    vP = T51.gen_vec(gv, M.D_V)
    k_un = gen_keys(gr, N_UNWRITTEN)
    k_new = gen_keys(gi, 1)[0]
    v_new = gen_value(gi, vP)
    U0, V0 = T48.step_pair(P["U1"].clone(), P["V1"].clone(), ks[0], vP, 1.0)
    return {"ks": ks, "vP": vP, "k_un": k_un, "k_new": k_new, "v_new": v_new,
            "U0": U0, "V0": V0}


def sha16(t):
    return hashlib.sha256(t.detach().numpy().tobytes()).hexdigest()[:16]


def run_cells(st, arm):
    r"""**每格都从共同态 `(U0,V0)` 重起**；臂 W 在**格内**先施加那一次 $\alpha{=}1$ 写入（`A58-10` 按格计数）。"""
    rec = {"arm": arm, "common_sha16": sha16(st["U0"]) + ":" + sha16(st["V0"]), "cells": {}}
    for q in QS[arm]:
        U, Vm = st["U0"].clone(), st["V0"].clone()
        if arm == "W":
            U, Vm = T48.step_pair(U, Vm, st["k_new"], st["v_new"], 1.0)
        start = sha16(U) + ":" + sha16(Vm)
        for _ in range(q if arm == "D" else q - 1):
            U, Vm = T48.step_pair(U, Vm, st["ks"][0], st["vP"], 0.0)
        m = read_metrics(U, Vm, st["ks"][0], st["vP"], st["k_un"], st["k_new"], st["v_new"])
        m["q"] = q
        m["start_sha16"] = start
        rec["cells"]["q%d" % q] = m
    return rec


def run_seed(seed, arm):
    rec = run_cells(seed_state(seed), arm)
    rec["seed"] = seed
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed0", type=int, default=59500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [run_seed(a.seed0 + i, a.arm) for i in range(a.n)]
    res = {"arm": a.arm, "seed0": a.seed0, "n": a.n, "qs": list(QS[a.arm]),
           "n_rep": T48.N_REP, "eta": float(M.ETA), "lam": float(M.LAM),
           "d_k": M.D_K, "d_v": M.D_V, "key_scale": R.KEY_SCALE,
           "device_md5": DEVICE_MD5, "rows": rows}
    out = a.out or os.path.join(ROOT, "dir58", "out", "DIR58r10_%s.json" % a.arm)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("arm=%s n=%d seed0=%d qs=%s → %s" % (a.arm, a.n, a.seed0, QS[a.arm], out))


if __name__ == "__main__":
    main()