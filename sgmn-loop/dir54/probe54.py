#!/opt/pytorch/bin/python
r"""DIR-54 · **探针**（前提/可行性/效应量）——「挂载层集合」：同秩同因子集下，集中 vs 分摊

★ **这一步不是正式轮**：只用来（i）验证"零秩层 ≡ 未挂载"的力学、（ii）验证因子集在臂间**逐位匹配**、
  （iii）估计效应量与噪声（据此冻结阈值）。**探针不得产出判据结论**（纪律：smoke/probe 不判）。

唯一自变量：**两个 rank-1 关联因子的放置位置**（同秩 2、同一因子集）。
  none (0,0) · conc1 (2,0) · conc2 (0,2) · split (1,1) · both (2,2 自由预算参考)

派生关系：装置 `dir31/R1_grad.py`、任务层 `dir33/R1_task33.py`、读数 `dir40/R1_task40.trace40`
**逐字零改动**（只 import）；本文件只加"因子放置"这一层。
"""
import argparse
import json
import math
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
for p in ("/home/ec2-user/sgmn-loop/dir40", "/home/ec2-user/sgmn-loop/dir37",
          "/home/ec2-user/sgmn-loop/dir33", "/home/ec2-user/sgmn-loop/dir31",
          "/home/brra1n0/.sandbox/experiment/1107/sgmn-loop/dir40",
          "/home/brra1n0/.sandbox/experiment/1107/sgmn-loop/dir37",
          "/home/brra1n0/.sandbox/experiment/1107/sgmn-loop/dir33",
          "/home/brra1n0/.sandbox/experiment/1107/sgmn-loop/dir31"):
    sys.path.insert(0, p)
import R1_task33 as R          # 任务层（逐字未改）
import R1_grad as M            # 装置（逐字未改）
import R1_task40 as T40        # 读数（逐字未改；N_REP=16 的写协议）
import trace54 as TR           # 读数层：唯一差异 = 损失读第几层（见该文件头）

D_H, D_K, D_V = M.D_H, M.D_K, M.D_V
T = R.T
ARMS = ("none", "conc1", "conc2", "split", "both")
RANK = {"none": (0, 0), "conc1": (2, 0), "conc2": (0, 2), "split": (1, 1), "both": (2, 2)}

SEED0 = M.SEED0
P_TRAIN = [SEED0 + 71000 + i for i in range(96)]     # 71000..71095
P_EVAL = [SEED0 + 71300 + i for i in range(48)]      # 71300..71347（★ 与训练段**不相交**）


# ══════════════════════════════════════════ 因子放置（唯一自变量）
def _draw_pairs(sd, n=4):
    r"""同分布（`N(0, INIT_SD)`，与 `R.build` 的 `rn(..., scale=INIT_SD)` 逐字同分布）、确定性。"""
    g = torch.Generator().manual_seed(sd * 7919 + 13)
    out = []
    for _ in range(n):
        u = torch.randn(D_V, 1, generator=g) * M.INIT_SD
        v = torch.randn(D_K, 1, generator=g) * M.INIT_SD
        out.append((u.requires_grad_(True), v.requires_grad_(True)))
    return out


def _empty():
    return (torch.zeros(D_V, 0, requires_grad=True), torch.zeros(D_K, 0, requires_grad=True))


def mount(P, sd, arm):
    r"""把 `arm` 的因子放到指定层。`conc1/conc2/split` 用**同一对矩阵** `(a,b)`；`both` 再加 `(c,d)`。"""
    pairs = _draw_pairs(sd)
    a, b, c, d = pairs[0], pairs[1], pairs[2], pairs[3]
    z = _empty()
    rk = RANK[arm]
    put = {}
    if arm == "none":
        put[1], put[2] = z, _empty()
    elif arm == "conc1":
        put[1] = (torch.cat([a[0], b[0]], 1), torch.cat([a[1], b[1]], 1))
        put[2] = _empty()
    elif arm == "conc2":
        put[1] = _empty()
        put[2] = (torch.cat([a[0], b[0]], 1), torch.cat([a[1], b[1]], 1))
    elif arm == "split":
        put[1] = (a[0].clone(), a[1].clone())
        put[2] = (b[0].clone(), b[1].clone())
    elif arm == "both":
        put[1] = (torch.cat([a[0], b[0]], 1), torch.cat([a[1], b[1]], 1))
        put[2] = (torch.cat([c[0], d[0]], 1), torch.cat([c[1], d[1]], 1))
    for l in (1, 2):
        U, V = put[l]
        assert U.shape == (D_V, rk[l - 1]) and V.shape == (D_K, rk[l - 1]), (arm, l)
        P[f"U{l}"] = U.requires_grad_(True)
        P[f"V{l}"] = V.requires_grad_(True)
    return put


def factor_matrices(sd):
    r"""`(a,b)` 的两张 rank-1 矩阵，用于逐位匹配断言。"""
    pairs = _draw_pairs(sd)
    return [pairs[0][0] @ pairs[0][1].t(), pairs[1][0] @ pairs[1][1].t()], pairs


# ══════════════════════════════════════════ 头（8 个参数组，**臂间完全相同**）
def head_init():
    r"""两层读头全部可训：`g,Wo,bo`（两层）+ `Wq2,bq2`（层 2 的寻址控制器）。

    ★ 与 DIR-40/53 的差别（**必须在卡里登记**）：那不是"只训层 1 头"——层 2 若用**未训**的头读记忆，
      对"把记忆放到层 2"这一臂**系统性不公平** ⇒ 本方向必须两层都训，且臂间参数集**逐位相同**。
    """
    gin = torch.Generator().manual_seed(4242)
    def rn(*shape, scale):
        return (torch.randn(*shape, generator=gin) * scale).requires_grad_(True)
    phi = {}
    for l in (1, 2):
        phi[f"g{l}"] = torch.full((1,), M.GAMMA, requires_grad=True)
        phi[f"Wo{l}"] = rn(D_H + D_V, D_H, scale=1.0 / math.sqrt(D_H + D_V))
        phi[f"bo{l}"] = rn(D_H, scale=0.1)
    phi["Wq2"] = rn(D_H, D_K, scale=1.0 / math.sqrt(D_H))
    phi["bq2"] = rn(D_K, scale=0.1)
    return phi


HEAD_NAMES = ("g1", "Wo1", "bo1", "g2", "Wo2", "bo2", "Wq2", "bq2")
# 原生装置（dir31.rollout）两层都用**自己的**寻址头 ⇒ 该档把两层 `Wq,bq` 都训
NATIVE_NAMES = ("g1", "Wo1", "bo1", "Wq1", "bq1", "g2", "Wo2", "bo2", "Wq2", "bq2")


def head_init_native():
    phi = head_init()
    gin = torch.Generator().manual_seed(4242)
    # ★ 与 head_init 共用生成器会改变抽取顺序 ⇒ 这里用**独立**生成器（同分布、跨臂逐位相同）
    g2 = torch.Generator().manual_seed(4343)
    phi["Wq1"] = (torch.randn(D_H, D_K, generator=g2) / math.sqrt(D_H)).requires_grad_(True)
    phi["bq1"] = (torch.randn(D_K, generator=g2) * 0.1).requires_grad_(True)
    return phi


def trace_native(P):
    r"""**装置原生**损失（`dir31.rollout` 逐字，1 write/步/层，损失读第 2 层 vs `P["tgt"]`）。"""
    L, _, _ = M.rollout(P, "V0")
    return L


def train_native(arm, seeds, steps=250, lr=0.03):
    phi = head_init_native()
    opt = torch.optim.Adam([phi[n] for n in NATIVE_NAMES], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        for sd in seeds:
            L = trace_native(_mk(sd, phi, arm, train=True))
            (L / len(seeds)).backward()
        opt.step()
    return {k: v.detach().clone() for k, v in phi.items()}


def evaluate_native(arm, phi, seeds):
    Ls = []
    for sd in seeds:
        with torch.no_grad():
            Ls.append(float(trace_native(_mk(sd, phi, arm))))
    return Ls


def _mk(sd, phi, arm, train=False):
    P = R.build(sd, "V0")
    R.period2(P, sd)
    mount(P, sd, arm)
    P.update({k: (v if train else v.clone()) for k, v in phi.items()})
    return P


READOUT = {"v": "l2"}


def trace(P, sd, phi=None):
    L, _ = TR.trace54(P, T40.N_REP, "oracle", sd, phi or {}, readout=READOUT["v"])
    return L


def train(arm, seeds, steps=250, lr=0.03):
    phi = head_init()
    opt = torch.optim.Adam([phi[n] for n in HEAD_NAMES], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        for sd in seeds:
            L = trace(_mk(sd, phi, arm, train=True), sd, phi)
            (L / len(seeds)).backward()
        opt.step()
    return {k: v.detach().clone() for k, v in phi.items()}


def evaluate(arm, phi, seeds):
    Ls = []
    for sd in seeds:
        with torch.no_grad():
            Ls.append(float(trace(_mk(sd, phi, arm), sd)))
    return Ls


# ══════════════════════════════════════════ 不变量
def invariants():
    chk = {}
    sd = P_TRAIN[0]
    # ① 参数量匹配
    cnt = {a: sum(D_V * r + D_K * r for r in RANK[a]) for a in ARMS}
    chk["n_params"] = cnt
    chk["A54p-1_matched"] = bool(cnt["conc1"] == cnt["conc2"] == cnt["split"]
                                 and cnt["both"] == 2 * cnt["conc1"] and cnt["none"] == 0)
    # ② 因子集逐位匹配：(2,0) 的 W1 == a+b ；split 的两张 W 之和 == 同一矩阵（t=0）
    mm, _ = factor_matrices(sd)
    phi = head_init()
    Pc = _mk(sd, phi, "conc1")
    Wc = M._W(Pc["U1"], Pc["V1"])
    chk["A54p-2_factor_identity"] = float((Wc - (mm[0] + mm[1])).abs().max().detach())
    Ps = _mk(sd, phi, "split")
    Ws = M._W(Ps["U1"], Ps["V1"]) + M._W(Ps["U2"], Ps["V2"])
    chk["A54p-2_split_sum"] = float((Ws - (mm[0] + mm[1])).abs().max().detach())
    # ③ 零秩层在整段 rollout 后仍为 0（W ≡ 0）
    for arm, layer in (("conc1", 2), ("conc2", 1), ("none", 1), ("none", 2)):
        P = _mk(sd, phi, arm)
        M.rollout(P, "V0")
        w = float(M._W(P[f"U{layer}"], P[f"V{layer}"]).abs().max())
        chk[f"A54p-3_zero_{arm}_L{layer}"] = w
    chk["A54p-3_ok"] = bool(all(v == 0.0 for k, v in chk.items() if k.startswith("A54p-3_zero")))
    # ④ none ≡ no_memory=True（逐位）
    P1 = _mk(sd, phi, "none")
    P2 = R.build(sd, "V0"); R.period2(P2, sd); P2.update({k: v.clone() for k, v in phi.items()})
    L1, _, _ = M.rollout(P1, "V0")
    L2, _, _ = M.rollout(P2, "V0", no_memory=True)
    chk["A54p-4_none_eq_nomemory"] = float(abs(float(L1) - float(L2)))
    # ⑤ 初值尺度与装置默认可比（同秩 2 的 Frobenius 范数比）
    nb = []
    for s in P_TRAIN:
        Pd = R.build(s, "V0"); R.period2(Pd, s)
        nb.append(float((M._W(Pd["U1"], Pd["V1"])).norm()))
        Pc = _mk(s, head_init(), "conc1")
        nb.append(float((M._W(Pc["U1"], Pc["V1"])).norm()))
    dev, new = statistics.fmean(nb[0::2]), statistics.fmean(nb[1::2])
    chk["A54p-5_norm_device"], chk["A54p-5_norm_ours"] = dev, new
    chk["A54p-5_ratio"] = new / dev
    chk["A54p-5_ok"] = bool(0.6 <= new / dev <= 1.6)
    # ⑦ 读数层守卫：readout="l1" 必须与 trace40 **逐位相同**
    P = _mk(sd, phi, "split")
    L40, _ = T40.trace40(P, T40.N_REP, "oracle", sd, phi)
    P = _mk(sd, phi, "split")
    L54, _ = TR.trace54(P, T40.N_REP, "oracle", sd, phi, readout="l1")
    chk["A54p-7_l1_eq_trace40"] = float(abs(float(L40) - float(L54)))
    # ⑧ 读第 2 层时，层 2 的记忆**必须**改变损失（否则"挂载层"不可测 ⇒ 方向无前提）
    chk["A54p-8_l2_live"] = float(abs(
        float(TR.trace54(_mk(sd, phi, "conc2"), T40.N_REP, "oracle", sd, phi, readout="l2")[0])
        - float(TR.trace54(_mk(sd, phi, "none"), T40.N_REP, "oracle", sd, phi, readout="l2")[0])))
    chk["A54p-8_ok"] = bool(chk["A54p-8_l2_live"] > 0.0)
    # ⑥ 确定性：同臂两次逐位相同
    ph = head_init()
    e1 = evaluate("split", ph, P_EVAL[:2])
    e2 = evaluate("split", ph, P_EVAL[:2])
    chk["A54p-6_det"] = float(abs(e1[0] - e2[0]) + abs(e1[1] - e2[1]))
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("inv", "arm", "native"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--ntrain", type=int, default=8)
    ap.add_argument("--neval", type=int, default=12)
    ap.add_argument("--readout", choices=("l1", "l2"), default="l2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    READOUT["v"] = a.readout
    t0 = time.time()
    if a.mode == "inv":
        chk = invariants()
        print(json.dumps(chk, ensure_ascii=False, indent=1))
        if a.out:
            json.dump(chk, open(a.out, "w"), ensure_ascii=False, indent=1)
        return
    tr, ev = P_TRAIN[:a.ntrain], P_EVAL[:a.neval]
    # ★ DEF-54-1：`P_TRAIN` 只有 8 个时 `[:96]` 会**静默截断**为 8（本轮探针首次运行即踩到）
    assert len(tr) == a.ntrain and len(ev) == a.neval, (len(tr), a.ntrain, len(ev), a.neval)
    if a.mode == "native":
        phi = train_native(a.arm, tr, steps=a.steps)
        Ls = evaluate_native(a.arm, phi, ev)
        res = {"arm": a.arm, "ranks": RANK[a.arm], "readout": "native-l2", "steps": a.steps,
               "ntrain": len(tr), "neval": len(ev), "mean_L": statistics.fmean(Ls),
               "sd_L": statistics.stdev(Ls) if len(Ls) > 1 else 0.0, "L": Ls,
               "secs": time.time() - t0}
        print(json.dumps(res, ensure_ascii=False))
        if a.out:
            json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        return
    phi = train(a.arm, tr, steps=a.steps)
    Ls = evaluate(a.arm, phi, ev)
    res = {"arm": a.arm, "ranks": RANK[a.arm], "readout": a.readout, "steps": a.steps, "ntrain": len(tr),
           "neval": len(ev), "mean_L": statistics.fmean(Ls),
           "sd_L": statistics.stdev(Ls) if len(Ls) > 1 else 0.0, "L": Ls,
           "secs": time.time() - t0}
    print(json.dumps(res, ensure_ascii=False))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()