#!/opt/pytorch/bin/python
r"""DIR-54 · **读数层**（`trace54`）—— 与 `dir40/R1_task40.trace40` **逐字同体**，只把"损失读哪一层"变成参数。

★ 为什么需要它（探针事实，`dir54/out/probe54_*.json`）：`trace40` 的损失行是
  `((R.pout() @ hp[0] - gs[t-1])**2).sum()` ⇒ **只读第 1 层**（DIR-33 卡 §3.5 的**有意**选择：与目标同层）。
  于是**第 2 层的记忆对损失完全没有通路**：探针实测 `conc2 ≡ none`、`both ≡ conc1`（逐位相同，见
  `probe54_none.json` / `probe54_conc2.json`）。⇒ 要问「记忆该挂在哪一层」，必须让损失读**被问的那一层**。

★ 本文件对 `trace40` 的改动**只有一处**（其余逐字）：
    `readout == "l1"` ⇒ `((R.pout()  @ hp[0] - gs[t-1])**2).sum()`   ← 与 `trace40` **逐位相同**（断言 A54-7）
    `readout == "l2"` ⇒ `((pout2()   @ hp[1] - gs[t-1])**2).sum()`   ← 唯一变化：读第 2 层
  `pout2()` 是与 `R.pout()` **同分布、同尺度**（$1/\sqrt{D_H}$）的**固定**随机输出映射，跨臂共享、不训练。
"""
import math
import sys

import torch

for p in ("/home/ec2-user/sgmn-loop/dir40", "/home/ec2-user/sgmn-loop/dir37",
          "/home/ec2-user/sgmn-loop/dir33", "/home/ec2-user/sgmn-loop/dir31",
          "/home/brra1n0/.sandbox/experiment/1107/sgmn-loop/dir40",
          "/home/brra1n0/.sandbox/experiment/1107/sgmn-loop/dir37",
          "/home/brra1n0/.sandbox/experiment/1107/sgmn-loop/dir33",
          "/home/brra1n0/.sandbox/experiment/1107/sgmn-loop/dir31"):
    sys.path.insert(0, p)
import R1_task33 as R
import R1_grad as M

D_H, D_V, D_K = M.D_H, M.D_V, M.D_K
T = R.T
POUT2_SEED = 20261113
_P2 = {}


def pout2():
    r"""固定输出映射（`DIR-54` 装置 revision 的唯一新增常量）：$D_V\times D_H$，尺度 $1/\sqrt{D_H}$。"""
    if "m" not in _P2:
        g = torch.Generator().manual_seed(POUT2_SEED)
        _P2["m"] = torch.randn(D_V, D_H, generator=g) / math.sqrt(D_H)
    return _P2["m"]


def trace54(P, n_rep, qsrc, sd, phi=None, readout="l2", want_diag=False, shared_pk1=None):
    r"""`T40.trace40` 的逐字同体；**唯一差异** = 损失行按 `readout` 取第 1 / 第 2 层。"""
    if shared_pk1 is not None:
        P["Pk1"] = shared_pk1
    L = P["a"].new_zeros(())
    gs = R.targets(P)
    h_pre_prev = [None, None]
    q1s, k1s = [], []
    acc = {"q_norm": 0.0, "y_norm": 0.0, "m_abs": 0.0, "n": 0, "c_eq_h": 0.0}
    q_fp = 0.0
    wmax = 0.0
    for t in range(T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if li == 0:
                c = h
                acc["c_eq_h"] = max(acc["c_eq_h"], float((c - h).abs().max()))
                if qsrc == "oracle":
                    q = torch.tanh(h @ P["Pk1"])
                else:
                    q = torch.tanh(c @ P["Wq1"] + P["bq1"])
                q1s.append(q.detach().clone())
                q_fp = q_fp + float(q.sum())
            else:
                c = hp_prev.detach()
                q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            Wd = W.detach()
            y = Wd @ q
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            if want_diag and li == 0:
                acc["q_norm"] += float(q.norm())
                acc["y_norm"] += float(y.norm())
                acc["m_abs"] += float(m.abs().mean())
                acc["n"] += 1
            gam = P[f"g{l}"]
            hp[li] = h + gam * m
            hp_prev = hp[li]
        if t >= 1:
            if readout == "l1":                       # ← 与 trace40 逐字相同
                L = L + ((R.pout() @ hp[0] - gs[t - 1]) ** 2).sum()
            elif readout == "l2":                     # ← 唯一变化
                L = L + ((pout2() @ hp[1] - gs[t - 1]) ** 2).sum()
            else:
                raise ValueError(readout)
        upd = {}
        for li, l in enumerate((1, 2)):
            hprev = h_pre_prev[li]
            if hprev is None:
                hprev = torch.zeros(D_H)
            k = torch.tanh(hprev.detach() @ P[f"Pk{l}"])
            v = torch.tanh(h_pre[li].detach() @ P[f"Pv{l}"])
            if li == 0:
                k1s.append(k.detach().clone())
            Uo, Vo = P[f"U{l}"], P[f"V{l}"]
            for _ in range(n_rep):
                W = M._W(Uo, Vo)
                G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
                upd[f"U{l}"] = Uo - M.ETA * (G @ Vo)
                upd[f"V{l}"] = Vo - M.ETA * (G.t() @ Uo)
                Uo, Vo = upd[f"U{l}"], upd[f"V{l}"]
        upd = {kk: vv.detach() for kk, vv in upd.items()}
        P.update(upd)
        h_pre_prev = h_pre
        if want_diag:
            wmax = max(wmax, float(M._W(P["U1"], P["V1"]).norm()))
    diag = {"Wmax": wmax, "c_eq_h": acc["c_eq_h"], "q_fp": q_fp}
    if want_diag:
        n = max(acc["n"], 1)
        diag["q_norm"] = acc["q_norm"] / n
        diag["y_norm"] = acc["y_norm"] / n
        diag["m_abs"] = acc["m_abs"] / n
    return L, diag