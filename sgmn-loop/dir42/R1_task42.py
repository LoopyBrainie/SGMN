#!/opt/pytorch/bin/python
r"""DIR-42 · **正式轮**实现：2×2 的**第三格** ——（逐实例 `Pk1` × **每实例自带读器**）

卡：`revolve/DIR-42/DIRECTION_rev42_v10.md`（rev 1.0，**阈值与判据已冻结**；md5 由 `A42-3` 守卫）。

派生自 `dir41/R1_task41.py`（**逐字**：种子段结构 / 判定表骨架 / 配对统计 / `merge` 的裁决逻辑）
⇒ ★ **装置文件 `dir31`、任务层 `dir33`、`dir37/38/39/40/41` 一律不改**（`A42-1`）。

★ **唯一的装置 revision**：**读器的参数共享范围**
  （DIR-41 的 PI 格 = 逐实例 `Pk1` × **共享**读器；本卡 = 逐实例 `Pk1` × **每实例**读器）
  ⇒ ★ 头 `(g1,Wo1,bo1)` 与控制器 `(Wq1,bq1)` **都**变成**每实例一份**。

★ **时序留出**（本卡的必要配套）：训练目标**只**累加 `t∈[1,8]` 的损失项，
  评估**只**累加 `t∈[9,15]` 的损失项 ⇒ ⇒ eval 窗口的损失项**从未进入**训练目标
  ⇒ ⇒ 这是「**同实例的时序留出**」，**不是**跨实例留出（逐实例读器按定义无法跨实例留出）。

用法：stageA | smoke | formal --arm {oracle,rand,learned} --shard i --nshards 4 | merge
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
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir37")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir38")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir39")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir40")
import R1_task33 as R           # 任务层（逐字未改）
import R1_grad as M             # 装置（逐字未改）
import R1_task37 as T37         # 测量层（`_mk` 逐字未改）
import R1_task38 as T38         # DIR-38 的 harness（跨实现锚）
import R1_task40 as T40         # DIR-40 的测量层（`trace40` 逐字复用）

OUT = "/home/ec2-user/sgmn-loop/dir42/out"
D_H, D_V, D_K, T = R.D_H, R.D_V, R.D_K, R.T
SEED0 = M.SEED0

# ══════════ 冻结常量（卡 §3.2）
CARD_PATH = "/home/ec2-user/sgmn-loop/revolve/DIR-42/DIRECTION_rev42_v10.md"
CARD_MD5_EXPECTED = "5d684ee1cf8dff14ffc4a344293299fa"
MD5_DEVICE = "8fa40f708600bec7e6c484727d2180c7"
MD5_TASK = "96ef89e87ec61556b52e0e464850087e"
MD5_DIR37 = "9236ac0490be9c413f1e8af163289207"
MD5_DIR38 = "d725da6d80bd072f9068a51857abf744"
MD5_DIR39 = "795563cb6212f228f8463ed418f8ce8c"
MD5_DIR40 = "baab3566891591944efb60a5b6dc9623"
MD5_DIR41 = "af8f001fe4237b998830a430f9293379"
N_REP = 16
N_INST = 400                  # 每臂的实例数（每实例一份读器）
STEPS, LR = 250, 0.03
SIG_K = 3.0
FRAC_HEADROOM = 0.20
COS_SIGNATURE = 0.5           # 「对齐签名存在」的诊断阈（**不设硬闸**）
READER_SEED_OFFSET = 10_000_000
# ★ **行 schema 版本**：改行结构（增删字段）⇒ **必须** +1 ⇒ ⇒ 旧行自动作废
#   （★ 只靠「配置指纹」盖不住 schema 漂移 ⇒ ⇒ 指纹里必须有 schema 版本）
ROW_SCHEMA = 1

# ══════════ 种子段（卡 §6）：全部为 `SEED0 + offset`
INST_OFF = list(range(22500, 22500 + N_INST))          # 22500..22899
SMOKE_INST_OFF = list(range(22100, 22104))             # 22100..22103
TRAIN_WIN = (1, 8)             # 训练窗口（损失项下标，含端点）
EVAL_WIN = (9, 15)             # 评估窗口（损失项下标，含端点）

ARMS = ("oracle", "rand", "learned")


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def _card_guard(stage):
    r"""★ **冻结守卫**（**纵深防御**）：任何**能产出判据读数**的命令都必须过这道闸。

    - `CARD_MD5_EXPECTED` 仍是占位符 ⇒ 卡片**尚未冻结** ⇒ 拒绝执行；
    - 卡的实测 md5 与冻结值不符 ⇒ 卡片**在冻结后被改过** ⇒ 拒绝执行。

    ⇒ ★ 为什么 Stage A 里的 `A42-3` 不够：`A42-3` 只**报告**比对结果，而
      `dir42/run_formal42.sh` 是一个**可独立调用**的脚本 ⇒ ⇒ 若守卫只在 Stage A，
      就存在「绕过 Stage A 直接跑正式轮」的路径 ⇒ ⇒ 守卫下沉到 `formal_arm` 与 `merge`。
    ⇒ ★ 顺序不变：回填 §8 ⇒ **算卡 md5 并写进 `CARD_MD5_EXPECTED`** ⇒ 冻结 ⇒ 才可跑。
    """
    if CARD_MD5_EXPECTED == "__CARD_MD5__":
        raise SystemExit(f"★ [{stage}] 卡片尚未冻结（CARD_MD5_EXPECTED 仍是占位符）⇒ 拒绝执行")
    got = _md5(CARD_PATH)
    if got != CARD_MD5_EXPECTED:
        raise SystemExit(f"★ [{stage}] 卡片 md5 不符：冻结值 {CARD_MD5_EXPECTED} ≠ 实测 {got}"
                         f" ⇒ 拒绝执行（卡在冻结后被改过）")
    return got


# ══════════ 读器初值（**每实例**确定性；`rand` 与 `learned` 两臂**逐位同初值**）
def reader_init(sd):
    r"""`dir39.init_params()` 的**逐字派生**，生成器种子 = 实例种子 + `READER_SEED_OFFSET`。

    ★ 绘制顺序逐字相同（`Wo1` → `bo1` → `Wq1`；`bq1` 为零）。
    ★ 生成器流**独立于**装置的实例种子流 ⇒ 读器初值与该实例的 `Pk1` **无构造相关**。
    """
    gin = torch.Generator().manual_seed(int(sd) + READER_SEED_OFFSET)
    return {
        "g1": torch.full((1,), M.GAMMA, requires_grad=True),
        "Wo1": (torch.randn(D_H + D_V, D_H, generator=gin)
                / math.sqrt(D_H + D_V)).requires_grad_(True),
        "bo1": (torch.randn(D_H, generator=gin) * 0.1).requires_grad_(True),
        "Wq1": (torch.randn(D_H, D_K, generator=gin) * 0.354).requires_grad_(True),
        "bq1": torch.zeros(D_K, requires_grad=True),
    }


# ══════════ `trace40` 的**逐字派生**：唯一新增 = **损失窗口**
def trace42(P, n_rep, qsrc, sd, phi, want_diag=False, rand_content=False, win=None):
    r"""`dir40.trace40` 的**逐字派生**：只加（i）**损失窗口** `win=(lo,hi)`（含端点；`None` ⇒ 全部 `t≥1`）
    （ii）读器 `phi` **每实例一份**（`g1/Wo1/bo1` 头 ＋ `Wq1/bq1` 控制器）。

    ★ 轨迹**照常走完 `T` 步**（写侧不因窗口而改变）⇒ 窗口只决定**哪些损失项进入求和**
      ⇒ ⇒ eval 窗口的损失项**从未进入**训练目标（卡 §2.3）。
    ★ `qsrc="oracle"` ⇒ `q = tanh(h · Pk1)`（**该实例自己的** `Pk1`）；
      `"rand"` ⇒ `q = tanh(c · Wq1 + bq1)`（控制器**不在优化器里** ⇒ 逐位冻结）；
      `"learned"` ⇒ 同式，但控制器**被任务梯度训过**。
    """
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
        if t >= 1 and (win is None or (win[0] <= t <= win[1])):
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
                coss.append(float(q @ kp) / (nq * nk))
                alpr.append(float(q @ kp) / nk)
        diag["cos_q_k"] = statistics.fmean(coss) if coss else float("nan")
        diag["align_q_k"] = statistics.fmean(alpr) if alpr else float("nan")
    return L, diag


def train_reader42(qsrc, sd, steps=STEPS, train_win=TRAIN_WIN, lr=LR):
    r"""**每实例**读器训练：头 `(g1,Wo1,bo1)` 必训；控制器 `(Wq1,bq1)` **只在 `learned` 档训**。
    ⇒ `rand` 档的控制器**逐位冻结**（不在优化器里）⇒ 与 `learned` 档**同初值** ⇒ 逐位配对。
    ★ 训练损失**只**累加 `train_win` 窗口的损失项（卡 §2.3）。
    """
    phi = reader_init(sd)
    snap = {k: v.detach().clone() for k, v in phi.items()}
    names = ["g1", "Wo1", "bo1"] + (["Wq1", "bq1"] if qsrc == "learned" else [])
    opt = torch.optim.Adam([phi[n] for n in names], lr=lr)
    gn = {}
    for _ in range(steps):
        opt.zero_grad()
        P = T37._mk(sd, phi, train=True)
        L, _ = trace42(P, N_REP, qsrc, sd, phi, win=train_win)
        L.backward()
        for n in names:
            if phi[n].grad is not None:
                gn[n] = max(gn.get(n, 0.0), float(phi[n].grad.norm()))
        opt.step()
    out = {k: v.detach().clone() for k, v in phi.items()}
    meta = {"gradnorms": gn,
            "dWq1_from_init": float((out["Wq1"] - snap["Wq1"]).norm()),
            "dbq1_from_init": float((out["bq1"] - snap["bq1"]).norm())}
    return out, meta


def eval_reader42(qsrc, phi, sd, win=EVAL_WIN, rand_content=False):
    P = T37._mk(sd, phi, train=False)
    L, dg = trace42(P, N_REP, qsrc, sd, phi, want_diag=True, win=win,
                    rand_content=rand_content)
    return {"L": float(L), **dg}


def _fmean_safe(xs):
    """★ `statistics.fmean` 在混有 `inf`/`nan` 时**可能抛异常**（`fsum` 遇 `inf` 与 `-inf`）
    ⇒ ⇒ 统一先判有限：任一非有限 ⇒ 返回 `nan`（由 `merge` 的 `finite` 检查判 `DIVERGED`）。"""
    xs = list(xs)
    if not all(math.isfinite(x) for x in xs):
        return float("nan")
    return statistics.fmean(xs)


def paired(a, b):
    r"""配对差统计。★ **非有限输入不抛异常** ⇒ 返回非有限的 `se`/`3se`
    ⇒ ⇒ 由 `merge` 的 `finite` 检查判成 `DIVERGED`（卡 §4 第 2 支）。

    ★ **旧实现在这里 `statistics.stdev` 抛 `ValueError: inf or nan encountered in data`
    ⇒ ⇒ `merge` 直接崩栈 ⇒ ⇒ 「非有限 ⇒ `DIVERGED`」这条分支**形同虚设**
    （★ 由离线自检 `dir42/verify_merge42.py` 查出，**冻结前**已修）。
    """
    d = [x - y for x, y in zip(a, b)]
    m = _fmean_safe(d)
    if all(math.isfinite(x) for x in d):
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
    else:
        sd = float("nan")
    se = sd / len(d) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": SIG_K * se, "n": len(d)}


# ══════════ Stage A（硬闸；**断言必须真的读被断言的对象** ⇒ 纪律 97）
def _row_cos_stats(mats):
    cs = []
    for P in mats:
        for i in range(D_H):
            for j in range(i + 1, D_H):
                a, b = P[i], P[j]
                na, nb = float(a.norm()), float(b.norm())
                if na > 1e-300 and nb > 1e-300:
                    cs.append(float(a @ b) / (na * nb))
    return statistics.fmean(cs), (statistics.stdev(cs) if len(cs) > 1 else 0.0), len(cs)


def _all_pairs_differ(mats):
    r"""**枚举全部 `C(n,2)` 对**（纪律 102）⇒ 返回 (比较对数, 全异?)。"""
    pairs, diff, same = 0, 0, []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            pairs += 1
            if float((mats[i] - mats[j]).abs().max()) > 0.0:
                diff += 1
            else:
                same.append((i, j))
    return pairs, diff, same


def stage_a():
    g = {}
    # ── A42-1：装置/任务层/上游脚本 md5 未变
    md5s = {
        "dir31/R1_grad.py": MD5_DEVICE, "dir33/R1_task33.py": MD5_TASK,
        "dir37/R1_task37.py": MD5_DIR37, "dir38/R1_task38.py": MD5_DIR38,
        "dir39/R1_task39.py": MD5_DIR39, "dir40/R1_task40.py": MD5_DIR40,
        "dir41/R1_task41.py": MD5_DIR41,
    }
    got = {p: _md5("/home/ec2-user/sgmn-loop/" + p) for p in md5s}
    g["A42-1_md5"] = got
    g["A42-1_ok"] = all(got[p] == v for p, v in md5s.items())

    # ── A42-2：有效常数实测（纪律 94）
    eff = {"D_H": M.D_H, "D_K": M.D_K, "D_V": M.D_V, "R": M.R, "T_STEPS": M.T_STEPS,
           "ETA": M.ETA, "LAM": M.LAM, "ALPHA": M.ALPHA, "GAMMA": M.GAMMA,
           "KEY_SCALE": R.KEY_SCALE, "INIT_SD": M.INIT_SD, "SEED0": M.SEED0}
    want = {"D_H": 8, "D_K": 16, "D_V": 4, "R": 2, "T_STEPS": 16, "ETA": 0.2,
            "LAM": 0.05, "ALPHA": 1.0, "GAMMA": 0.5, "KEY_SCALE": 0.5,
            "INIT_SD": 0.5, "SEED0": 20261111}
    g["A42-2_effective"] = eff
    g["A42-2_ok"] = all(eff[k] == want[k] for k in want)

    # ── A42-3：卡 md5 守卫
    g["A42-3_card_md5"] = _md5(CARD_PATH)
    g["A42-3_ok"] = g["A42-3_card_md5"] == CARD_MD5_EXPECTED

    # ── A42-4：**offset 空间**的种子段隔离（纪律 97；修正 DIR-40 的空断言）
    hist = []
    for _k, (a, b) in list(T37.HIST_RANGES.items()) + list(T37.OWN_SEGMENTS.items()):
        hist.extend(range(a, b + 1))
    for a, b in [(12500, 13096), (13500, 13900), (14000, 14384), (14500, 14512),
                 (15000, 15148), (15500, 15900), (16000, 16192), (16400, 16412),
                 (17000, 17048), (17100, 17164), (17200, 17264),
                 (18000, 18191), (18500, 18899), (18900, 18911),   # DIR-40 正式轮
                 (20400, 20412), (20500, 20595), (20600, 20999),   # DIR-41 探针
                 (21000, 21095), (21100, 21111), (21500, 21899),   # DIR-41 正式轮
                 (22000, 22015)]:                                # DIR-42 探针
        hist.extend(range(a, b + 1))
    hist_set = set(hist)
    g["A42-4_hist_n"] = len(hist_set)
    g["A42-4_inst_hist"] = len(set(INST_OFF) & hist_set)
    g["A42-4_smoke_hist"] = len(set(SMOKE_INST_OFF) & hist_set)
    g["A42-4_inst_smoke"] = len(set(INST_OFF) & set(SMOKE_INST_OFF))
    g["A42-4_reader_seed_above_device"] = bool(
        min(SEED0 + o + READER_SEED_OFFSET for o in INST_OFF)
        > max(SEED0 + o for o in INST_OFF))
    g["A42-4_ok"] = (g["A42-4_inst_hist"] == 0 and g["A42-4_smoke_hist"] == 0
                     and g["A42-4_inst_smoke"] == 0
                     and g["A42-4_reader_seed_above_device"])

    # ── A42-5：逐位锚（DIR-37 装置臂）：`n_rep=1` + oracle + 装置自有 `Pk1` + **全窗口**
    phi_dev = T37.train_head("dev", [SEED0 + o for o in SMOKE_INST_OFF])
    errs = []
    for o in SMOKE_INST_OFF:
        sd = SEED0 + o
        P = T37._mk(sd, phi_dev)
        Lref, _ = T37.trace(P, "dev", sd, phi_dev)
        P = T37._mk(sd, phi_dev)
        L42, _ = trace42(P, 1, "oracle", sd, phi_dev, win=None)
        errs.append(abs(float(Lref) - float(L42)))
    g["A42-5_max_abs_diff"] = max(errs)
    g["A42-5_ok"] = (max(errs) == 0.0)

    # ── A42-6：跨实现锚（DIR-38 的 `n_rep=16` 臂）：`L` 与 `max‖W‖` 逐位相同
    errs38, w42, w38 = [], 0.0, 0.0
    for o in SMOKE_INST_OFF:
        sd = SEED0 + o
        P = T38.T37._mk(sd, phi_dev)
        L38, dg38 = T38.trace38(P, 16, "grad", sd, phi_dev, want_diag=True)
        P = T38.T37._mk(sd, phi_dev)
        L42, dg42 = trace42(P, 16, "oracle", sd, phi_dev, want_diag=True, win=None)
        errs38.append(abs(float(L38) - float(L42)))
        w42 = max(w42, dg42["Wmax"]); w38 = max(w38, dg38["Wmax"])
    g["A42-6_max_abs_diff"] = max(errs38)
    g["A42-6_Wmax_42"] = w42
    g["A42-6_Wmax_38"] = w38
    g["A42-6_ok"] = (max(errs38) == 0.0 and w42 == w38)

    # ── A42-7 / A42-8 / A42-9：单变量断言 + 指纹 + 快照基准
    cch, qfp, diag = {}, {}, {}
    for o in SMOKE_INST_OFF:
        sd = SEED0 + o
        spk = None                       # ★ 装置自有的逐实例 `Pk1`（本卡的第三格）
        phi_r, meta_r = train_reader42("rand", sd, steps=5)
        phi_l, meta_l = train_reader42("learned", sd, steps=5)
        for lv, qq, pp in (("oracle", "oracle", phi_dev),
                           ("rand", "rand", phi_r),
                           ("learned", "learned", phi_l),
                           ("learned_rc", "learned", phi_l)):
            P = T37._mk(sd, pp)
            _L, dg = trace42(P, N_REP, qq, sd, pp, want_diag=True,
                             rand_content=(lv == "learned_rc"))
            cch.setdefault(sd, {})[lv] = dg["c_eq_h"]
            qfp.setdefault(sd, {})[lv] = dg["q_fp"]
            diag.setdefault(sd, {})[lv] = dg["Wmax"]
        g.setdefault("A42-9_meta", {})[str(sd)] = {
            "rand_dWq1": meta_r["dWq1_from_init"], "rand_dbq1": meta_r["dbq1_from_init"],
            "learned_dWq1": meta_l["dWq1_from_init"], "learned_dbq1": meta_l["dbq1_from_init"],
            "gradnorm_Wq1": meta_l["gradnorms"].get("Wq1", 0.0),
            "gradnorm_bq1": meta_l["gradnorms"].get("bq1", 0.0)}
    g["A42-7_c_eq_h"] = cch
    g["A42-7_ok"] = all(v == 0.0 for d in cch.values() for v in d.values())

    fp_ok = True
    for sd, d in qfp.items():
        if not (d["learned"] == d["learned_rc"] and d["rand"] != d["learned"]
                and d["oracle"] != d["rand"] and d["oracle"] != d["learned"]):
            fp_ok = False
    g["A42-8_q_fp"] = qfp
    g["A42-8_ok"] = fp_ok

    mv_ok = True
    for sd, m in g["A42-9_meta"].items():
        if not (m["rand_dWq1"] == 0.0 and m["rand_dbq1"] == 0.0
                and m["learned_dWq1"] > 0.0 and m["gradnorm_Wq1"] > 0.0
                and math.isfinite(m["gradnorm_Wq1"])):
            mv_ok = False
    g["A42-9_ok"] = mv_ok

    # ── A42-10：内容置换生效
    P = T37._mk(SEED0 + SMOKE_INST_OFF[0], phi_dev)
    W = M._W(P["U1"], P["V1"]).detach()
    Wp = R.read_W(W, "S", SEED0 + SMOKE_INST_OFF[0])
    g["A42-10_dW_perm"] = float((Wp - W).norm())
    g["A42-10_ok"] = g["A42-10_dW_perm"] > 0.0

    # ── A42-11：结构（**全部 `C(n,2)` 对**，纪律 102）
    STRUCT = [SEED0 + 22500 + i for i in range(20)]
    pks, rds = [], []
    for sd in STRUCT:
        P = T37._mk(sd, phi_dev)
        pks.append(P["Pk1"])
        rds.append(reader_init(sd)["Wq1"].detach())
    npk, dpk, spk_ = _all_pairs_differ(pks)
    nrd, drd, srd = _all_pairs_differ(rds)
    g["A42-11_pk1"] = {"n": len(pks), "pairs_checked": npk, "pairs_differing": dpk,
                       "pairs_identical": spk_}
    g["A42-11_reader_init"] = {"n": len(rds), "pairs_checked": nrd, "pairs_differing": drd,
                               "pairs_identical": srd}
    g["A42-11_ok"] = (dpk == npk == nrd == drd and not spk_ and not srd)

    # ── A42-12：读器初值确定（同实例两次调用逐位相同）
    i1 = reader_init(SEED0 + SMOKE_INST_OFF[0])
    i2 = reader_init(SEED0 + SMOKE_INST_OFF[0])
    g["A42-12_init_deterministic"] = all(
        float((i1[k] - i2[k]).abs().max()) == 0.0 for k in i1)
    g["A42-12_ok"] = g["A42-12_init_deterministic"]

    # ── A42-13：**损失窗口的真实隔离**（真的读被断言的对象 ⇒ 纪律 97）
    #   (a) 训练/评估窗口的损失项下标集合**不相交**且**并集 = 全部 t∈[1,T-1]**
    #   (b) **实测**：`L_train + L_eval == L_full`（窗口是全体的一个划分）
    #   (c) `L_train != L_full` 且 `L_eval != L_full`（两个窗口都**真的**排除了一部分项）
    #   (d) 训练用的 `trace42(..., win=TRAIN_WIN)` 与评估用的 `trace42(..., win=EVAL_WIN)`
    #        **是同一条轨迹、不同的求和窗口** ⇒ ⇒ eval 窗口的项从未进入训练目标。
    sd0 = SEED0 + SMOKE_INST_OFF[0]
    phi0 = reader_init(sd0)
    L_full, _ = trace42(T37._mk(sd0, phi0, train=False), N_REP, "oracle", sd0, phi0, win=None)
    L_tr, _ = trace42(T37._mk(sd0, phi0, train=False), N_REP, "oracle", sd0, phi0, win=TRAIN_WIN)
    L_ev, _ = trace42(T37._mk(sd0, phi0, train=False), N_REP, "oracle", sd0, phi0, win=EVAL_WIN)
    g["A42-13_win_index_sets"] = {
        "train": [TRAIN_WIN[0], TRAIN_WIN[1]], "eval": [EVAL_WIN[0], EVAL_WIN[1]],
        "all_loss_terms": [1, M.T_STEPS - 1],
        "disjoint": bool(EVAL_WIN[0] > TRAIN_WIN[1]),
        "covers_all": bool(TRAIN_WIN[0] == 1 and EVAL_WIN[1] == M.T_STEPS - 1)}
    g["A42-13_sum_abs_diff"] = abs(float(L_full) - float(L_tr) - float(L_ev))
    g["A42-13_L_full"] = float(L_full)
    g["A42-13_L_train"] = float(L_tr)
    g["A42-13_L_eval"] = float(L_ev)
    g["A42-13_train_excludes"] = bool(float(L_tr) != float(L_full))
    g["A42-13_eval_excludes"] = bool(float(L_ev) != float(L_full))
    g["A42-13_ok"] = (g["A42-13_win_index_sets"]["disjoint"]
                      and g["A42-13_win_index_sets"]["covers_all"]
                      and g["A42-13_sum_abs_diff"] < 1e-9
                      and g["A42-13_train_excludes"] and g["A42-13_eval_excludes"])

    # ── A42-14：诊断读数（**不设硬闸**）
    g["A42-14_diag"] = diag

    # ── A42-15：★ **正式轮代码路径干跑**（1 个实例；临时 `OUT`；**不碰**正式落盘）
    #   ★ **为什么必须有**：本卡的 `trace42` 是**本卡自己的**派生（DIR-41 直接复用
    #     `T40.trace40` ⇒ 不需要裸 `T`）⇒ ⇒ **派生时漏掉一个模块级常量**（`T`）
    #     就会让正式轮**一开工就崩**，而：探针（另写的 `probe42.py`）**照不到**它、
    #     四份离线自检（只测 `merge`/指纹/卡↔代码）**也照不到**它、
    #     语法检查**更照不到**它 ⇒ ⇒ ★★ **「代码从未被执行过」是最贵的失败模式**。
    #   ⇒ ⇒ 故 Stage A 里**真跑一次**正式轮路径（`formal_arm`，`nshards=N_INST` ⇒ 只落到 1 个实例）。
    import tempfile as _tf
    _out_saved = OUT
    _tmp_out = _tf.mkdtemp(prefix="D42_dryrun_")
    globals()["OUT"] = _tmp_out
    try:
        _o = formal_arm("learned", shard=0, nshards=N_INST)      # ⇒ `idxs = [0]` ⇒ **只跑 1 个实例**
        g["A42-15_dryrun"] = {"ok": bool(_o and _o["n"] == 1), "n": (_o["n"] if _o else 0),
                              "L": (_o["rows"][0]["L"] if _o and _o["rows"] else None),
                              "secs": (_o.get("secs") if _o else None)}
    except Exception as _e:
        g["A42-15_dryrun"] = {"ok": False, "error": repr(_e)}
    finally:
        globals()["OUT"] = _out_saved
    g["A42-15_ok"] = bool(g["A42-15_dryrun"].get("ok"))

    ok = all(v for k, v in g.items() if k.endswith("_ok"))
    print("Stage A：" + ("★ 全部通过" if ok else "★ 未通过"))
    for k in sorted(g):
        print(f"  {k:26s} {g[k]}")
    return g, ok


# ══════════ smoke（证明协议可执行；**永不产生判据裁决**）
def smoke():
    out = {"note": "smoke：只证明可执行，不给判据"}
    for o in SMOKE_INST_OFF:
        sd = SEED0 + o
        for arm in ARMS:
            phi, meta = train_reader42(arm, sd, steps=3)
            e = eval_reader42(arm, phi, sd)
            e_in = eval_reader42(arm, phi, sd, win=None)
            out[f"{sd}/{arm}"] = {"L_eval": e["L"], "L_full_insample": e_in["L"],
                                  "cos_q_k": e["cos_q_k"], "Wmax": e["Wmax"],
                                  "c_eq_h_max": e["c_eq_h"],
                                  "dWq1_from_init": meta["dWq1_from_init"]}
            print(f"  {sd}/{arm:8s} L_eval={e['L']:.6f} L_full={e_in['L']:.6f} "
                  f"cos={e['cos_q_k']:.4f} Wmax={e['Wmax']:.4f} "
                  f"c_eq_h={e['c_eq_h']:.1e} |Wq-init|={meta['dWq1_from_init']:.3e}",
                  flush=True)
    print("smoke 完成（**无判据裁决**）")
    return out


# ══════════ 增量落盘（**纪律 103**：逐实例 `O_APPEND` ⇒ 可续跑）
def _jsonl_path(arm, shard):
    return os.path.join(OUT, f"DIR42r10_arm_{arm}_shard{shard}.jsonl")


def _row_cfg():
    """每行都带配置指纹 ⇒ ★ 配置一变，旧行**自动作废**（不得混用不同配置的读数）

    ★ 指纹必须盖住**三个维度**，缺一不可：
      ① **配置**（`steps/lr/n_rep/窗口/实例段`）—— 读数的数值依赖它；
      ② **schema 版本**（`schema`）—— ★ **行结构**变了（旧行缺字段 ⇒ `merge` 会 `KeyError`）
         ⇒ ⇒ 只盖配置**盖不住 schema 漂移** ⇒ ⇒ 改行结构必须 `ROW_SCHEMA += 1`；
      ③ **种子基址**（`seed0`）—— ★ 装置种子基址变了 ⇒ 同 `idx` 的种子不同 ⇒ ⇒ 旧行不得复用
         （虽然 `A42-1` 已拦装置变更，但这里是**纵深防御**）。
    """
    return {"steps": STEPS, "lr": LR, "n_rep": N_REP,
            "train_win": list(TRAIN_WIN), "eval_win": list(EVAL_WIN),
            "inst_off": [INST_OFF[0], INST_OFF[-1]],
            "schema": ROW_SCHEMA, "seed0": SEED0}


def _append_jsonl(path, row):
    """单次 `write()` 追加一行（POSIX 下小行的 `O_APPEND` 写是原子的）"""
    line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _read_jsonl(path):
    """读增量行；★ 容忍**最后一行被截断**（追加写被打断）⇒ 只丢该实例，不丢整片"""
    rows = []
    if not os.path.exists(path):
        return rows
    cfg = _row_cfg()
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("cfg") != cfg:
            continue
        rows.append(r)
    return rows


# ══════════ 正式轮：单臂分片（4 片并行）
def formal_arm(arm, shard=0, nshards=4):
    _card_guard(f"formal {arm}/shard{shard}")      # ★ 冻结守卫（先过闸，再算）
    t0 = time.time()
    idxs = [i for i in range(N_INST) if i % nshards == shard]
    os.makedirs(OUT, exist_ok=True)
    jpath = _jsonl_path(arm, shard)
    done = {r["idx"]: r for r in _read_jsonl(jpath)}
    reused = [si for si in idxs if si in done]
    if reused:
        print(f"[{arm}/shard{shard}] ★ 续跑：复用已落盘的 {len(reused)} 个实例读数", flush=True)
    rows = []
    for si in idxs:
        if si in done:
            rows.append(done[si])
            continue
        sd = SEED0 + INST_OFF[si]
        phi, meta = train_reader42(arm, sd, steps=STEPS)
        e = eval_reader42(arm, phi, sd, win=EVAL_WIN)
        e_seen = eval_reader42(arm, phi, sd, win=TRAIN_WIN)   # ★ **已见**窗口（样本内）
        e_full = eval_reader42(arm, phi, sd, win=None)        # 训练窗 + 评估窗（**混合**）
        row = {"idx": si, "seed": sd, "L": e["L"], "L_seen": e_seen["L"],
               "L_full": e_full["L"],
               "cos": e["cos_q_k"], "align": e["align_q_k"], "q_norm": e["q_norm"],
               "y_norm": e["y_norm"], "m_abs": e["m_abs"], "Wmax": e["Wmax"],
               "dWq1": meta["dWq1_from_init"], "gn_Wq1": meta["gradnorms"].get("Wq1", 0.0),
               "cfg": _row_cfg()}
        if arm == "learned":
            e_rc = eval_reader42(arm, phi, sd, win=EVAL_WIN, rand_content=True)
            row["L_rc"] = e_rc["L"]
        rows.append(row)
        _append_jsonl(jpath, row)
        if (len(rows) % 10) == 0:
            print(f"[{arm}/shard{shard}] {len(rows)}/{len(idxs)} "
                  f"L={row['L']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["idx"])
    out = {"arm": arm, "shard": shard, "nshards": nshards, "n": len(rows),
           "n_inst": N_INST, "n_rep": N_REP, "steps": STEPS, "lr": LR,
           "train_win": list(TRAIN_WIN), "eval_win": list(EVAL_WIN),
           "rows": rows, "secs": time.time() - t0,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(OUT, f"DIR42r10_arm_{arm}_shard{shard}.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[{arm}/shard{shard}] 完成 {len(rows)} 个实例，{time.time()-t0:.0f}s", flush=True)
    return out


# ══════════ 汇总裁决（卡 §3/§4，**冻结**）
def merge():
    r"""读 12 个分片 JSON ⇒ 按**冻结**判定表裁决。★ **不读探针文件**。"""
    _card_guard("merge")                            # ★ 冻结守卫（先过闸，再裁决）
    arms, rows_by_arm, missing, src_by_arm = {}, {}, [], {}
    cfg = _row_cfg()
    for arm in ARMS:
        rows, srcs = [], []
        for sh in range(4):
            jl = _read_jsonl(_jsonl_path(arm, sh))       # ★ 增量行（优先）
            jp = os.path.join(OUT, f"DIR42r10_arm_{arm}_shard{sh}.json")
            jr = []
            if os.path.exists(jp):
                try:
                    jr = json.load(open(jp, encoding="utf-8")).get("rows", [])
                except (json.JSONDecodeError, OSError):
                    jr = []
            # ★ 分片 JSON 里的行也过**配置指纹**（旧配置的读数不得混入）
            jr = [r for r in jr if r.get("cfg") == cfg]
            if not jl and not jr:
                missing.append(f"{arm}/shard{sh}")
                continue
            if len(jl) >= len(jr):
                rows.extend(jl); srcs.append(f"shard{sh}:jsonl")
            else:
                rows.extend(jr); srcs.append(f"shard{sh}:json")
        rows.sort(key=lambda r: r["idx"])
        rows_by_arm[arm] = rows
        src_by_arm[arm] = srcs
    out = {"n_inst": N_INST, "n_rep": N_REP, "steps": STEPS, "sig_k": SIG_K,
           "frac_headroom": FRAC_HEADROOM, "cos_signature": COS_SIGNATURE,
           "train_win": list(TRAIN_WIN), "eval_win": list(EVAL_WIN),
           "missing": missing, "source": src_by_arm}

    bad_n = [f"{a}/{r.get('idx')}" for a in ARMS for r in rows_by_arm.get(a, [])
             if not (isinstance(r.get("idx"), int) and 0 <= r["idx"] < len(INST_OFF))
             or r.get("seed") != SEED0 + INST_OFF[r["idx"]]]
    out["bad_n"] = bad_n
    if missing or any(len(rows_by_arm.get(a, [])) != N_INST for a in ARMS):
        out["outcome"] = "INCOMPLETE"
        out["counted"] = False
        print("outcome=INCOMPLETE（缺臂或 n 不符）")
        with open(os.path.join(OUT, "DIR42r10_formal.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        return out

    def series(arm, key):
        return [r[key] for r in rows_by_arm[arm]]
    Lo = series("oracle", "L")
    Lr = series("rand", "L")
    Ll = series("learned", "L")
    Lrc = series("learned", "L_rc")
    Lseen = series("learned", "L_seen")
    Lseen_r = series("rand", "L_seen")
    Lseen_o = series("oracle", "L_seen")
    Lfull = series("learned", "L_full")

    R = _fmean_safe([a - b for a, b in zip(Lr, Lo)])
    pd = paired(Lr, Ll)
    pc = paired(Lrc, Ll)
    pd_in = paired(Lseen_r, Lseen)
    floor = FRAC_HEADROOM * R
    p1 = pd["mean"] >= pd["3se"]
    p2 = pd["mean"] >= floor
    p3 = pc["mean"] >= pc["3se"]
    finite = all(math.isfinite(x) for x in
                 [R, pd["mean"], pd["se"], pc["mean"], pc["se"], floor])

    cos = {"oracle": _fmean_safe([r["cos"] for r in rows_by_arm["oracle"]]),
           "rand": _fmean_safe([r["cos"] for r in rows_by_arm["rand"]]),
           "learned": _fmean_safe([r["cos"] for r in rows_by_arm["learned"]])}
    abs_cos_learned = abs(cos["learned"])

    out["means"] = {"L_oracle": _fmean_safe(Lo), "L_rand": _fmean_safe(Lr),
                    "L_learned": _fmean_safe(Ll), "L_learned_rc": _fmean_safe(Lrc),
                    "L_seen_oracle": _fmean_safe(Lseen_o),
                    "L_seen_rand": _fmean_safe(Lseen_r),
                    "L_seen_learned": _fmean_safe(Lseen),
                    "L_full_learned": _fmean_safe(Lfull)}
    out["R"] = R
    out["floor_0.2R"] = floor
    out["delta_out"] = pd
    out["delta_in_sample"] = pd_in
    out["content_C"] = pc
    out["P1_signif"] = bool(p1)
    out["P2_floor"] = bool(p2)
    out["P3_content"] = bool(p3)
    out["cos"] = cos
    out["abs_cos_learned"] = abs_cos_learned
    out["alignment_signature_present"] = bool(abs_cos_learned >= COS_SIGNATURE)
    out["finite"] = bool(finite)

    if not finite:
        out["outcome"] = "DIVERGED"
        out["counted"] = False
    elif not p1:
        out["outcome"] = "NO-OUT-OF-SAMPLE-GAIN"
        out["counted"] = False
    elif not p3:
        out["outcome"] = "NOT-CONTENT-DEPENDENT"
        out["counted"] = False
    elif not p2:
        out["outcome"] = "BELOW-EFFECT-FLOOR"
        out["counted"] = False
    else:
        out["outcome"] = "PASS"
        out["counted"] = True
    print(f"outcome={out['outcome']}  counted={out['counted']}")
    print(f"  R={R:.6f}  floor={floor:.6f}")
    print(f"  Δ_out mean={pd['mean']:.6f} 3se={pd['3se']:.6f}  (P1={p1})")
    print(f"  Δ_in_sample mean={pd_in['mean']:.6f} 3se={pd_in['3se']:.6f}")
    print(f"  C content mean={pc['mean']:.6f} 3se={pc['3se']:.6f}  (P3={p3})")
    print(f"  cos oracle={cos['oracle']:.4f} rand={cos['rand']:.4f} "
          f"learned={cos['learned']:.4f} (|cos|={abs_cos_learned:.4f})")
    with open(os.path.join(OUT, "DIR42r10_formal.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "stageA":
        gg, ok = stage_a()
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "DIR42r10_stageA.json"), "w") as f:
            json.dump(gg, f, ensure_ascii=False, indent=1)
        raise SystemExit(0 if ok else 1)
    if cmd == "smoke":
        o = smoke()
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "DIR42r10_smoke.json"), "w") as f:
            json.dump(o, f, ensure_ascii=False, indent=1)
        raise SystemExit(0)
    if cmd == "formal":
        arm = sys.argv[sys.argv.index("--arm") + 1]
        shard = int(sys.argv[sys.argv.index("--shard") + 1]) if "--shard" in sys.argv else 0
        nsh = int(sys.argv[sys.argv.index("--nshards") + 1]) if "--nshards" in sys.argv else 4
        formal_arm(arm, shard, nsh)
        raise SystemExit(0)
    if cmd == "merge":
        merge()
        raise SystemExit(0)
    print(__doc__)
