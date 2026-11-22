#!/opt/pytorch/bin/python
r"""DIR-40 · **读控制器（共享键映射下）** —— 层 1 的「**读控制器是否被训练**」单一变量化 + 内容性对照。

卡：`revolve/DIR-40/DIRECTION_rev40_v10.md`（rev 1.0，阈值已冻结；md5 由 A40-4 守卫）。
派生自：`dir39/R1_task39.py`（**逐字**的 trace/初值/训练协议）、`dir38/R1_task38.py`、`dir37/R1_task37.py`。

★ **唯一自变量 = 层 1 的读控制器「训不训」**（四档；**`c` 四档逐字相同 = `h`** ⇒ 只换 `q`）：
  ① `oracle`   ：`q = tanh(h · Pk1_shared)`（**描述性上限**；头**本档自训**）
  ② `rand`     ：`q = tanh(h · Wq1 + bq1)`，`(Wq1,bq1)` **存在但不在优化器里**（**逐位冻结**），
                  头**本档自训** ⇒ **强基线**（头会适应冻结读源）
  ③ `learned`  ：`q = tanh(h · Wq1 + bq1)`，`(Wq1,bq1)` **被任务梯度训过**（与 ② **同一初值**），
                  头**与之联合训练** ⇒ 处理
  ④ `learned_rc`：③ 的**同一**训练参数，但**记忆内容被随机置换**（`R.read_W(...,"S",sd)`）⇒ 通用注入对照

★ **装置 revision（相对 DIR-39 的唯一装置改动）**：层 1 的写键映射 `Pk1` **跨实例共享**
  （分布与装置逐字一致：`randn(D_H,D_K)/√D_H × KEY_SCALE`，固定种子）⇒
  ⇒ 共享读控制器**才可能**命中键 ⇒ ⇒ 这才是 §8「读取负责学习如何使用关联」在**共享参数**下的最小版本。
★ 写侧逐字冻结在 DIR-38 的 `n_rep=16`；`η,α,λ,γ`、键尺度、`T=16` 全不变。

用法：stageA | smoke | formal --arm {oracle,rand,learned} | merge
"""
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir37")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir38")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir39")
import R1_task33 as R           # 任务层（逐字未改；import 时改写装置的有效常数）
import R1_grad as M             # 装置（逐字未改）
import R1_task37 as T37         # DIR-37 的测量层（_mk / 训练协议 / 迹线）
import R1_task38 as T38         # DIR-38 的 harness（n_rep 写侧 / 交叉锚）
import R1_task39 as T39         # DIR-39 的 trace / 初值（逐字派生源）

OUT = "/home/ec2-user/sgmn-loop/dir40/out"
D_H, D_V, D_K, T = R.D_H, R.D_V, R.D_K, R.T
SEED0 = M.SEED0

# ══════════ 种子段（卡 §5：全部新段；探针段 17000–17263 不得复用）
TRAIN = [SEED0 + 18000 + i for i in range(192)]     # 训练段 18000..18191
EVAL = [SEED0 + 18500 + i for i in range(400)]       # 评估段 18500..18899
SMOKE_TRAIN = [SEED0 + 18900 + i for i in range(4)]
SMOKE_EVAL = [SEED0 + 18910 + i for i in range(2)]

# ══════════ 冻结阈值（卡 §1.3；**卡冻结后填入**）
CARD_PATH = "/home/ec2-user/sgmn-loop/revolve/DIR-40/DIRECTION_rev40_v11.md"
CARD_MD5_EXPECTED = "3522789d19fcd21894db949ce9253364"
CARD_V10_PATH = "/home/ec2-user/sgmn-loop/revolve/DIR-40/DIRECTION_rev40_v10.md"
CARD_V10_MD5_EXPECTED = "07541c443fb971ef7fcaced9b201dad9"
MD5_DEVICE = "8fa40f708600bec7e6c484727d2180c7"
MD5_TASK = "96ef89e87ec61556b52e0e464850087e"
MD5_DIR38 = "d725da6d80bd072f9068a51857abf744"
MD5_DIR39 = "795563cb6212f228f8463ed418f8ce8c"
N_REP = 16
LEVELS = ("oracle", "rand", "learned", "learned_rc")
N_TRAIN = 192
STEPS, LR = 250, 0.03
SIG_K = 3.0
FRAC_HEADROOM = 0.20
SHARED_PK1_SEED = 20261112


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def init_params():
    r"""DIR-39 的初值生成器（**逐字**）：同一生成器 4242、同一抽取顺序 ⇒ 三臂逐位配对。"""
    return T39.init_params()


def shared_pk1():
    r"""装置 revision：层 1 的写键映射 `Pk1` **跨实例共享**（分布与装置逐字一致）。"""
    g = torch.Generator().manual_seed(SHARED_PK1_SEED)
    return (torch.randn(D_H, D_K, generator=g) / math.sqrt(D_H)) * R.KEY_SCALE


def trace40(P, n_rep, qsrc, sd, phi, want_diag=False, shared_pk1=None, rand_content=False):
    r"""`T39.trace39` 的**逐字派生**：只加（i）`shared_pk1` 钩子（ii）`rand_content` 钩子
    （iii）三个读数：`c_eq_h`（层 1 的 `c` 与 `h` 的**逐位**最大差 ⇒ 真实的单变量断言）、
    `q_fp`（层 1 查询的指纹 ⇒ 断言各档的 `q` **确实不同**）、`cos_q_k`（**余弦**寻址读数）。

    ★ `align_q_k` 仍按 DIR-39 的**投影比**口径给出（连续性），但**不得**当余弦用（D-63-CORR）。
    """
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
                c = h                                   # ★ **各档逐字相同**（oracle 的 c）
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
            if rand_content and li == 0:
                Wd = R.read_W(Wd, "S", sd)
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
            L = L + ((R.pout() @ hp[0] - gs[t - 1]) ** 2).sum()
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
        coss, alpr = [], []
        for t in range(1, len(q1s)):
            q, kp = q1s[t], k1s[t - 1]
            nq, nk = float(q.norm()), float(kp.norm())
            if nq > 1e-300 and nk > 1e-300:
                coss.append(float(q @ kp) / (nq * nk))      # ★ 余弦
                alpr.append(float(q @ kp) / nk)              # 投影比（DIR-39 口径，连续性）
        diag["cos_q_k"] = statistics.fmean(coss) if coss else float("nan")
        diag["align_q_k"] = statistics.fmean(alpr) if alpr else float("nan")
    return L, diag


def train_arm40(qsrc, seeds, shared_pk=None, train_ctrl=False, steps=STEPS, lr=LR):
    r"""**分臂联合训练**（卡 §1.1）：头 `(g1,Wo1,bo1)` 必训；控制器 `(Wq1,bq1)` **只在处理臂里训**。

    ★ 初值与 DIR-39 的 `init_params()` **逐字同源**（同一生成器 4242、同一抽取顺序）
      ⇒ 三臂的**头初值逐位相同**、`rand` 与 `learned` 的**控制器初值逐位相同** ⇒ 逐位配对。
    ★ `train_ctrl=False` ⇒ `(Wq1,bq1)` **存在但不在优化器里** ⇒ 训练后**逐位不变**（可检查的冻结）。
    """
    init = init_params()
    # ★ DEF-40-3：`init["Wq1"]` 与 `phi["Wq1"]` 是**同一个张量对象** ⇒ 训练后 `init` 也被更新
    #   ⇒ 「训后 vs 初值」的差**恒为 0**（空断言；与 DIR-39 的 DEF-39-5 同类）⇒ 必须**先快照**。
    init_snap = {k: v.detach().clone() for k, v in init.items()}
    phi = {"g1": init["g1"], "Wo1": init["Wo1"], "bo1": init["bo1"],
           "Wq1": init["Wq1"], "bq1": init["bq1"]}
    names = ["g1", "Wo1", "bo1"] + (["Wq1", "bq1"] if train_ctrl else [])
    opt = torch.optim.Adam([phi[n] for n in names], lr=lr)
    gn = {}
    for _ in range(steps):
        opt.zero_grad()
        for sd in seeds:
            P = T37._mk(sd, phi, train=True)
            L, _ = trace40(P, N_REP, qsrc, sd, phi, shared_pk1=shared_pk)
            (L / len(seeds)).backward()
        for n in names:
            if phi[n].grad is not None:
                gn[n] = max(gn.get(n, 0.0), float(phi[n].grad.norm()))
        opt.step()
    out = {k: v.detach().clone() for k, v in phi.items()}
    meta = {"gradnorms": gn,
            "dWq1_from_init": float((out["Wq1"] - init_snap["Wq1"]).norm()),
            "dbq1_from_init": float((out["bq1"] - init_snap["bq1"]).norm())}
    return out, meta


def eval_arm40(qsrc, phi, seeds, shared_pk=None, rand_content=False):
    Ls, coss, alprs, qn, yn, ma, cch, wmx = [], [], [], [], [], [], [], []
    for sd in seeds:
        P = T37._mk(sd, phi, train=False)
        L, dg = trace40(P, N_REP, qsrc, sd, phi, want_diag=True, shared_pk1=shared_pk,
                        rand_content=rand_content)
        Ls.append(float(L))
        coss.append(dg["cos_q_k"]); alprs.append(dg["align_q_k"])
        qn.append(dg["q_norm"]); yn.append(dg["y_norm"]); ma.append(dg["m_abs"])
        cch.append(dg["c_eq_h"])
        wmx.append(dg["Wmax"])
    return {"mean_L": statistics.fmean(Ls),
            "sd_L": statistics.stdev(Ls) if len(Ls) > 1 else 0.0,
            "L_series": Ls, "cos_q_k": statistics.fmean(coss),
            "align_q_k": statistics.fmean(alprs), "q_norm": statistics.fmean(qn),
            "y_norm": statistics.fmean(yn), "m_abs": statistics.fmean(ma),
            "c_eq_h_max": max(cch), "Wmax": max(wmx)}


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    se = sd / len(d) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": SIG_K * se, "n": len(d)}


# ══════════ Stage A（硬闸；**断言必须真的读被断言的对象** ⇒ 纪律 97）
def stage_a():
    g = {}
    md5_dev = _md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py")
    md5_task = _md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py")
    md5_d38 = _md5("/home/ec2-user/sgmn-loop/dir38/R1_task38.py")
    md5_d39 = _md5("/home/ec2-user/sgmn-loop/dir39/R1_task39.py")
    card_md5 = _md5(CARD_PATH)
    card_v10_md5 = _md5(CARD_V10_PATH)
    g["A40-3_device_md5"] = md5_dev
    g["A40-3_task_md5"] = md5_task
    g["A40-3_dir38_md5"] = md5_d38
    g["A40-3_dir39_md5"] = md5_d39
    g["A40-3_ok"] = (md5_dev == MD5_DEVICE and md5_task == MD5_TASK
                     and md5_d38 == MD5_DIR38 and md5_d39 == MD5_DIR39)
    g["A40-4_card_md5"] = card_md5
    g["A40-4_card_v10_md5"] = card_v10_md5
    g["A40-4_ok"] = (card_md5 == CARD_MD5_EXPECTED and card_v10_md5 == CARD_V10_MD5_EXPECTED)

    # A40-5：种子段不相交（含 DIR-40 探针段与全部历史段）
    hist = []
    for k, (a, b) in list(T37.HIST_RANGES.items()) + list(T37.OWN_SEGMENTS.items()):
        hist.extend(range(a, b + 1))
    hist.extend(range(12500, 13096))     # DIR-38 探针段
    hist.extend(range(13500, 13900))     # DIR-38 评估段
    hist.extend(range(14000, 14384))     # DIR-38 训练段
    hist.extend(range(14500, 14512))     # DIR-38 smoke
    hist.extend(range(15000, 15148))     # DIR-39 探针段
    hist.extend(range(15500, 15900))     # DIR-39 评估段
    hist.extend(range(16000, 16192))     # DIR-39 训练段
    hist.extend(range(16400, 16412))     # DIR-39 smoke
    hist.extend(range(17000, 17048))     # DIR-40 探针甲
    hist.extend(range(17100, 17164))     # DIR-40 探针乙训练
    hist.extend(range(17200, 17264))     # DIR-40 探针乙评估
    g["A40-5_train_hist"] = len(set(TRAIN) & set(hist))
    g["A40-5_train_eval"] = len(set(TRAIN) & set(EVAL))
    g["A40-5_eval_hist"] = len(set(EVAL) & set(hist))
    g["A40-5_ok"] = (g["A40-5_train_hist"] == 0 and g["A40-5_train_eval"] == 0
                     and g["A40-5_eval_hist"] == 0)

    # A40-1：n_rep=1 + `oracle` 读（**未共享** Pk1）⇒ 与 DIR-37 装置臂**逐位相同**
    phi_dev = T37.train_head("dev", SMOKE_TRAIN)
    errs = []
    for sd in SMOKE_EVAL:
        P = T37._mk(sd, phi_dev)
        Lref, _ = T37.trace(P, "dev", sd, phi_dev)
        P = T37._mk(sd, phi_dev)
        L40, _ = trace40(P, 1, "oracle", sd, phi_dev)
        errs.append(abs(float(Lref) - float(L40)))
    g["A40-1_max_abs_diff"] = max(errs)
    g["A40-1_ok"] = (max(errs) == 0.0)

    # A40-2：n_rep=16 + `oracle` 读 ⇒ 与 DIR-38 的 n_rep=16 臂**逐位相同**（跨实现交叉锚）
    errs38, w40, w38 = [], 0.0, 0.0
    for sd in SMOKE_EVAL:
        P = T38.T37._mk(sd, phi_dev)
        L38, dg38 = T38.trace38(P, 16, "grad", sd, phi_dev, want_diag=True)
        P = T38.T37._mk(sd, phi_dev)
        L40, dg40 = trace40(P, 16, "oracle", sd, phi_dev, want_diag=True)
        errs38.append(abs(float(L38) - float(L40)))
        w40 = max(w40, dg40["Wmax"]); w38 = max(w38, dg38["Wmax"])
    g["A40-2_max_abs_diff"] = max(errs38)
    g["A40-2_Wmax_40"] = w40
    g["A40-2_Wmax_38"] = w38
    g["A40-2_ok"] = (max(errs38) == 0.0 and w40 == w38)

    # A40-6：**真实的单变量断言**（纪律 97）：层 1 的 `c` 与 `h` 逐位相同；
    #   且各档的 `q` 关系**符合设计**：`oracle ≠ rand`（读源不同）、
    #   `rand ≠ learned`（**训过**的控制器不同）、`learned == learned_rc`（**同一控制器**，
    #   只有记忆内容不同 ⇒ 查询指纹**必须**逐位相同）。
    #   ★ DEF-40-4：rev 1.0 的 A40-6 要求「四档指纹互不相同」⇒ **错**：`learned_rc` 与 `learned`
    #   共用控制器 ⇒ 指纹**必然相同**；而 `rand`/`learned` 在**未训练**时也必然相同
    #   ⇒ ⇒ 必须在**训过之后**比。rev 1.1 修正（**判据阈值未动**）。
    spk = shared_pk1()
    phi_r5, meta_r5 = train_arm40("rand", SMOKE_TRAIN, shared_pk=spk, train_ctrl=False, steps=5)
    phi_l5, meta_l5 = train_arm40("learned", SMOKE_TRAIN, shared_pk=spk, train_ctrl=True, steps=5)
    cch, qfp = {}, {}
    for lv, (qq, pp) in (("oracle", ("oracle", phi_dev)),
                         ("rand", ("rand", phi_r5)),
                         ("learned", ("learned", phi_l5)),
                         ("learned_rc", ("learned", phi_l5))):
        P = T37._mk(SMOKE_EVAL[0], pp)
        _L, dg = trace40(P, N_REP, qq, SMOKE_EVAL[0], pp, want_diag=True,
                         shared_pk1=spk, rand_content=(lv == "learned_rc"))
        cch[lv] = dg["c_eq_h"]
        qfp[lv] = dg["q_fp"]
    g["A40-6_c_eq_h"] = cch
    g["A40-6_q_fp"] = qfp
    g["A40-6_ok"] = (all(v == 0.0 for v in cch.values())
                     and qfp["oracle"] != qfp["rand"]
                     and qfp["rand"] != qfp["learned"]
                     and qfp["learned"] == qfp["learned_rc"])

    # A40-7：冻结臂的控制器**逐位不变**（**真的比较训练后的张量**）；训练臂**确实被训动**且梯度非零有限
    phi_r, meta_r = phi_r5, meta_r5
    phi_l, meta_l = phi_l5, meta_l5
    g["A40-7_dWq1_frozen_arm"] = meta_r["dWq1_from_init"]
    g["A40-7_dWq1_trained_arm"] = meta_l["dWq1_from_init"]
    g["A40-7_gradnorm_Wq1"] = meta_l["gradnorms"].get("Wq1", 0.0)
    g["A40-7_ok"] = (meta_r["dWq1_from_init"] == 0.0 and meta_r["dbq1_from_init"] == 0.0
                     and meta_l["dWq1_from_init"] > 0.0
                     and meta_l["gradnorms"].get("Wq1", 0.0) > 0.0
                     and math.isfinite(meta_l["gradnorms"].get("Wq1", 0.0)))

    # A40-8：三臂的**头初值与控制器初值逐位相同**（初值确定 + 跨臂配对）
    i1, i2 = init_params(), init_params()
    g["A40-8_init_deterministic"] = all(float((i1[k] - i2[k]).abs().max()) == 0.0 for k in i1)
    g["A40-8_ok"] = g["A40-8_init_deterministic"]

    # A40-9：四档的**写侧相同** ⇒ `max‖W‖` 逐位一致
    wm = {}
    for lv in LEVELS:
        P = T37._mk(SMOKE_EVAL[0], phi_dev)
        _L, dg = trace40(P, N_REP, lv, SMOKE_EVAL[0], phi_dev, want_diag=True,
                         shared_pk1=spk, rand_content=(lv == "learned_rc"))
        wm[lv] = dg["Wmax"]
    g["A40-9_Wmax"] = wm
    g["A40-9_ok"] = all(abs(wm[lv] - wm["oracle"]) < 1e-12 for lv in LEVELS)

    # A40-10：随机内容对照**确实改了记忆**（内容被置换）
    P = T37._mk(SMOKE_EVAL[0], phi_dev)
    W = M._W(P["U1"], P["V1"]).detach()
    Wp = R.read_W(W, "S", SMOKE_EVAL[0])
    g["A40-10_dW_perm"] = float((Wp - W).norm())
    g["A40-10_ok"] = g["A40-10_dW_perm"] > 0.0

    # A40-11：共享 `Pk1` **真的共享**（跨实例逐位相同）且尺度与装置的逐实例分布一致
    pks = []
    for sd in SMOKE_EVAL:
        P = T37._mk(sd, phi_dev)
        trace40(P, 0, "oracle", sd, phi_dev, shared_pk1=spk)
        pks.append(P["Pk1"])
    same = all(float((p - pks[0]).abs().max()) == 0.0 for p in pks)
    per_inst = []
    for sd in SMOKE_EVAL:
        P = T37._mk(sd, phi_dev)
        per_inst.append(float(M._W(P["U1"], P["V1"]).norm()))  # 占位，下面用 Pk1 的范数
    norms_inst = []
    for sd in SMOKE_EVAL:
        P = T37._mk(sd, phi_dev)
        norms_inst.append(float(P["Pk1"].norm()))
    g["A40-11_shared_identical"] = bool(same)
    g["A40-11_norm_shared"] = float(spk.norm())
    g["A40-11_norm_inst_mean"] = statistics.fmean(norms_inst)
    g["A40-11_ok"] = bool(same)

    # A40-12：共享 `Pk1` 的**分布**与装置逐实例 `Pk1` 一致（同一生成方式：均值比在 [0.8, 1.25]）
    ratio = float(spk.norm()) / statistics.fmean(norms_inst)
    g["A40-12_norm_ratio"] = ratio
    g["A40-12_ok"] = (0.8 <= ratio <= 1.25)

    ok = all(v for k, v in g.items() if k.endswith("_ok"))
    print("Stage A：" + ("★ 全部通过" if ok else "★ 未通过"))
    for k in sorted(g):
        print(f"  {k:26s} {g[k]}")
    return g, ok


# ══════════ smoke（证明协议可执行；**永不产生判据裁决**）
def smoke():
    spk = shared_pk1()
    out = {"note": "smoke：只证明可执行，不给判据"}
    for name, (qsrc, tctrl) in (("oracle", ("oracle", False)),
                                ("rand", ("rand", False)),
                                ("learned", ("learned", True))):
        phi, meta = train_arm40(qsrc, SMOKE_TRAIN, shared_pk=spk, train_ctrl=tctrl, steps=3)
        e = eval_arm40(qsrc, phi, SMOKE_EVAL, shared_pk=spk)
        out[name] = {"mean_L": e["mean_L"], "cos_q_k": e["cos_q_k"], "Wmax": e["Wmax"],
                     "c_eq_h_max": e["c_eq_h_max"], "dWq1_from_init": meta["dWq1_from_init"]}
        print(f"  {name:8s} L={e['mean_L']:.6f} cos={e['cos_q_k']:.4f} Wmax={e['Wmax']:.4f} "
              f"c_eq_h={e['c_eq_h_max']:.1e} |Wq-init|={meta['dWq1_from_init']:.3e}", flush=True)
    print("smoke 完成（**无判据裁决**）")
    return out


# ══════════ 正式轮：单臂（三个臂在 trainer 上**并行**跑；`merge` 汇总裁决）
def formal_arm(arm):
    t0 = time.time()
    spk = shared_pk1()
    qsrc, tctrl = {"oracle": ("oracle", False), "rand": ("rand", False),
                   "learned": ("learned", True)}[arm]
    print(f"[{arm}] 训头{'(+控制器)' if tctrl else ''}（分臂联合训练，共享 Pk1，n_rep={N_REP}）"
          f"n={len(TRAIN)} ...", flush=True)
    phi, meta = train_arm40(qsrc, TRAIN, shared_pk=spk, train_ctrl=tctrl)
    print(f"[{arm}] 训练完成 {time.time()-t0:.0f}s  梯度 {meta['gradnorms']}", flush=True)
    arms = {}
    arms[arm] = eval_arm40(qsrc, phi, EVAL, shared_pk=spk)
    if arm == "learned":
        arms["learned_rc"] = eval_arm40(qsrc, phi, EVAL, shared_pk=spk, rand_content=True)
    for k, a in arms.items():
        print(f"[{arm}] 评估 {k:11s} L={a['mean_L']:.6f} (sd {a['sd_L']:.4f}) "
              f"cos={a['cos_q_k']:.4f} align={a['align_q_k']:.4f} |q|={a['q_norm']:.4f} "
              f"|y|={a['y_norm']:.4f} m̄={a['m_abs']:.4f}", flush=True)
    out = {"arm": arm, "n_train": len(TRAIN), "n_eval": len(EVAL),
           "train_seg": [18000, 18191], "eval_seg": [18500, 18899], "n_rep": N_REP,
           "shared_pk1_seed": SHARED_PK1_SEED,
           "arms": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_series")}
                    for k, v in arms.items()},
           "L_series": {k: v["L_series"] for k, v in arms.items()},
           "gradnorms": meta["gradnorms"],
           "dWq1_from_init": meta["dWq1_from_init"],
           "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(OUT, f"DIR40r10_arm_{arm}.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


# ══════════ 汇总裁决（卡 §2.1，**冻结**）
def merge():
    arms, series = {}, {}
    metas = {}
    for arm in ("oracle", "rand", "learned"):
        p = os.path.join(OUT, f"DIR40r10_arm_{arm}.json")
        d = json.load(open(p, encoding="utf-8"))
        for k, v in d["arms"].items():
            arms[k] = v
            series[k] = d["L_series"][k]
        metas[arm] = d
    R_head = arms["rand"]["mean_L"] - arms["oracle"]["mean_L"]
    numer = arms["rand"]["mean_L"] - arms["learned"]["mean_L"]
    pg = paired(series["rand"], series["learned"])
    pc = paired(series["learned_rc"], series["learned"])
    floor = FRAC_HEADROOM * R_head
    finite = all(math.isfinite(a["mean_L"]) for a in arms.values())
    out = {
        "n_train": len(TRAIN), "n_eval": len(EVAL),
        "train_seg": [18000, 18191], "eval_seg": [18500, 18899], "n_rep": N_REP,
        "sig_k": SIG_K, "frac_headroom": FRAC_HEADROOM,
        "arms": arms, "L_series": series,
        "headroom_R_rand_minus_oracle": R_head, "numer_rand_minus_learned": numer,
        "effect_floor": floor,
        "paired_rand_learned": pg, "paired_learnedrc_learned": pc,
        "gradnorms": {k: v["gradnorms"] for k, v in metas.items()},
        "dWq1_from_init": {k: v["dWq1_from_init"] for k, v in metas.items()},
        "secs": {k: v["secs"] for k, v in metas.items()},
        "diverge": bool(not finite),
    }
    if out["diverge"]:
        out["outcome"] = "DIVERGED"
        out["counted"] = False
        out["note"] = "存在非有限读数 ⇒ 该档整档作废（种子与填表全部留档）"
    elif R_head <= 0:
        out["outcome"] = "INDETERMINATE"
        out["counted"] = False
        out["note"] = "可寻址区间 `L_rand − L_oracle ≤ 0` ⇒ 「oracle 是上限」的前提不成立 ⇒ 不可判定（**不是失败**）"
    elif numer <= -pg["3se"]:
        out["outcome"] = "FALSIFIED-WORSE"
        out["counted"] = False
        out["note"] = "**训练过的读控制器显著地比冻结随机控制器更差** ⇒ 强负结果：机制必须查清"
    elif numer < pg["3se"]:
        out["outcome"] = "FALSIFIED-SIG"
        out["counted"] = False
        out["note"] = "配对显著性不过（learned 未显著优于 rand）⇒ H 未被确认"
    elif numer < floor:
        out["outcome"] = "FALSIFIED-EFFECT"
        out["counted"] = False
        out["note"] = "显著但效应量低于可寻址区间的 20% ⇒ H 未被确认"
    elif pc["mean"] < pc["3se"]:
        out["outcome"] = "GENERIC-INJECTION"
        out["counted"] = False
        out["note"] = "收益在**随机内容**下依然存在 ⇒ 是**通用注入**而非内容性收益 ⇒ 不计数"
    else:
        out["outcome"] = "PASS"
        out["counted"] = True
        out["note"] = "过线 + 显著 + 内容性 ⇒ H 被确认"
    with open(os.path.join(OUT, "DIR40r10_formal.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n══ DIR-40 正式轮（汇总裁决） ══")
    for k in ("oracle", "rand", "learned", "learned_rc"):
        print(f"  L[{k:11s}] = {arms[k]['mean_L']:.6f}   cos = {arms[k]['cos_q_k']:.4f}"
              f"   |q|={arms[k]['q_norm']:.4f}  |y|={arms[k]['y_norm']:.4f}"
              f"  m̄={arms[k]['m_abs']:.4f}")
    print(f"  可寻址区间 R = L(rand) − L(oracle) = {R_head:.6f}")
    print(f"  效应量 = L(rand) − L(learned) = {numer:.6f}   阈（20%·R）= {floor:.6f}")
    print(f"  配对 rand−learned: mean={pg['mean']:.6f} 3se={pg['3se']:.6f} n={pg['n']}")
    print(f"  内容性对照 learned_rc−learned: mean={pc['mean']:.6f} 3se={pc['3se']:.6f}")
    print(f"  ★ 结局 = {out['outcome']} · 计入 SUCCEEDED = {out['counted']}")
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "stageA"
    if what == "stageA":
        g, ok = stage_a()
        json.dump({"guards": g, "ok": ok},
                  open(os.path.join(OUT, "DIR40r10_stageA.json"), "w"), ensure_ascii=False, indent=1)
        sys.exit(0 if ok else 1)
    if what == "smoke":
        json.dump(smoke(), open(os.path.join(OUT, "DIR40r10_smoke.json"), "w"),
                  ensure_ascii=False, indent=1)
        sys.exit(0)
    if what == "formal":
        # ★ DEF-40-6：rev 1.1 的用法写的是 `formal --arm {oracle,rand,learned}`，
        #   但首版实现取的是 `sys.argv[2]` ⇒ 拿到的是字面量 `"--arm"` ⇒ `KeyError`。
        #   ⇒ 现在**两种写法都支持**（`formal --arm X` 与 `formal X`）。
        rest = [a for a in sys.argv[2:] if not a.startswith("--")]
        arm = rest[-1] if rest else "learned"
        formal_arm(arm)
        sys.exit(0)
    if what == "merge":
        merge()
        sys.exit(0)
    print("用法：stageA | smoke | formal --arm {oracle,rand,learned} | merge")
    sys.exit(2)
