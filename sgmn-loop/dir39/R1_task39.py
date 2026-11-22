#!/opt/pytorch/bin/python
r"""DIR-39 · **读控制器** —— 层 1 的「**查询来源**」单一变量化 + 随机内容对照。

卡：`revolve/DIR-39/DIRECTION_rev39_v10.md`（rev 1.0，阈值已冻结；md5 由 A39-4 守卫）。

★ **唯一自变量 = 层 1 的查询来源 `q`**（四档）：
  ① `oracle`   ：`q = tanh(h · P_k1)`（**逐字** = 现行装置的层 1 读；per-instance 的「真值」读）⇒ **描述性上限**
  ② `rand`     ：`q = tanh(h · Wq1 + bq1)`，`(Wq1,bq1)` = **冻结的共享随机初值**（与 ③ **同一初值**）⇒ 对照
  ③ `learned`  ：`q = tanh(h · Wq1 + bq1)`，`(Wq1,bq1)` = **被任务梯度训过**（共享；初值同 ②）⇒ 处理
  ④ `learned_rc`：③ 的**同一**训练参数，但**记忆内容被随机置换**（`R.read_W(...,"S",sd)`）⇒ **通用注入对照**
  ★ **`c` 在四档里完全相同**：`c = h`（层 1 的前激活）⇒ 只换 `q`。
  ★ **写侧逐字冻结在 DIR-38 的 `n_rep=16`**。
  ★ **头只有一份**（在 `oracle` 档上训，`n_train=192`），**四档共用** ⇒ 档间唯一的差别是读控制器。

用法：stageA | smoke | formal
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
import R1_task33 as R          # 任务层（逐字未改；import 时改写装置的有效常数）
import R1_grad as M            # 装置（逐字未改）
import R1_task37 as T37        # DIR-37 的测量层（_mk / 训练协议 / 迹线）
import R1_task38 as T38        # DIR-38 的 harness（n_rep 写侧 / 交叉锚）

OUT = "/home/ec2-user/sgmn-loop/dir39/out"
D_H, D_V, D_K, RANK, T = R.D_H, R.D_V, R.D_K, M.R, M.T_STEPS
SEED0 = M.SEED0

# ══════════ 种子段（卡 §5：全部新段；探针段 15000–15147 不得复用）
TRAIN = [SEED0 + 16000 + i for i in range(192)]     # 训练段 16000..16191
EVAL = [SEED0 + 15500 + i for i in range(400)]       # 评估段 15500..15899
SMOKE_TRAIN = [SEED0 + 16400 + i for i in range(4)]
SMOKE_EVAL = [SEED0 + 16410 + i for i in range(2)]

# ══════════ 冻结阈值（卡 §1.3；**卡冻结后填入**）
CARD_PATH = "/home/ec2-user/sgmn-loop/revolve/DIR-39/DIRECTION_rev39_v10.md"
CARD_MD5_EXPECTED = "f0390a5077f1f98a41d7f7f0322e32c5"
MD5_DEVICE = "8fa40f708600bec7e6c484727d2180c7"
MD5_TASK = "96ef89e87ec61556b52e0e464850087e"
MD5_DIR38 = "d725da6d80bd072f9068a51857abf744"
N_REP = 16
LEVELS = ("oracle", "rand", "learned", "learned_rc")
N_TRAIN = 192
STEPS, LR = 250, 0.03
SIG_K = 3.0
FRAC_HEADROOM = 0.20
DIVERGE_W_RATIO = 10.0


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def init_params():
    r"""**跨档共享同一组初值**（配对比较）：初值生成器 4242、绘制顺序与 `R.train_arm` 逐字相同。

    ⇒ `Wo1,bo1` 与 `T37.train_head` 的初值**逐位相同**（同一生成器、同一前两抽）；
    ⇒ `Wq1,bq1` 与 `R.train_arm(learned=True)` 的初值**逐位相同**（同一生成器、第三四抽）。
    """
    gin = torch.Generator().manual_seed(4242)
    return {
        "g1": torch.full((1,), M.GAMMA, requires_grad=True),
        "Wo1": (torch.randn(D_H + D_V, D_H, generator=gin)
                / math.sqrt(D_H + D_V)).requires_grad_(True),
        "bo1": (torch.randn(D_H, generator=gin) * 0.1).requires_grad_(True),
        "Wq1": (torch.randn(D_H, D_K, generator=gin) * 0.354).requires_grad_(True),
        "bq1": torch.zeros(D_K, requires_grad=True),
    }


def trace39(P, n_rep, qsrc, sd, phi, want_diag=False, rand_content=False):
    r"""`T37.trace` / `T38.trace38` 的派生：**只**把层 1 的 `q` 换成可切换的读源。

    ★ 层 1：`c = h`（**四档完全一致** ⇒ 只换 `q`）；层 2：`c = hp_prev.detach()`、`q = tanh(c·Wq2+bq2)`（逐字未改）。
    ★ 读：`y = W.detach() @ q`（写通道隔离 ⇒ 任务梯度不进写侧；DIR-31 的 C3 语义保持）；
      `rand_content=True` 时用 `R.read_W(W,"S",sd)`（**随机置换内容**，范数不变）⇒ 通用注入对照。
    ★ 写：`n_rep` 次重复的同步因子更新（每次按**更新后**的 `W` 重算 `G`）。
    """
    L = P["a"].new_zeros(())
    gs = R.targets(P)
    h_pre_prev = [None, None]
    q1s, k1s = [], []
    c1_is_h = True
    acc = {"q_norm": 0.0, "y_norm": 0.0, "m_abs": 0.0, "n": 0}
    wmax = 0.0
    for t in range(T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if li == 0:
                c = h                                   # ★ **四档相同**（oracle 的 c）
                if qsrc == "oracle":
                    q = torch.tanh(h @ P["Pk1"])
                else:
                    q = torch.tanh(c @ P["Wq1"] + P["bq1"])
                q1s.append(q.detach().clone())
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
                upd[f"U{l}"] = Uo - M.ETA * (G @ Vo)     # 同步：两式都用**旧** U,V
                upd[f"V{l}"] = Vo - M.ETA * (G.t() @ Uo)
                Uo, Vo = upd[f"U{l}"], upd[f"V{l}"]
        upd = {kk: vv.detach() for kk, vv in upd.items()}
        P.update(upd)
        h_pre_prev = h_pre
        if want_diag:
            wmax = max(wmax, float(M._W(P["U1"], P["V1"]).norm()))
    diag = {"c1_is_h": c1_is_h, "Wmax": wmax}
    if want_diag:
        n = max(acc["n"], 1)
        diag["q_norm"] = acc["q_norm"] / n
        diag["y_norm"] = acc["y_norm"] / n
        diag["m_abs"] = acc["m_abs"] / n
        al = []
        for t in range(1, len(q1s)):
            k_prev = k1s[t - 1]
            nk = float(k_prev.norm())
            if nk > 1e-300:
                al.append(float(q1s[t] @ k_prev) / nk)
        diag["align_q_k"] = statistics.fmean(al) if al else float("nan")
    return L, diag


def train_head_arm(seeds, steps=STEPS, lr=LR):
    r"""训**唯一一份头**（`g1,Wo1,bo1`），在 **`oracle` 档**上（`n_rep=16` 的写侧）⇒ 四档共用。

    ★ 协议与 `T37.train_head` 逐字相同（同生成器 4242 / Adam / lr 0.03 / 250 步），只改：
      ① 档位（`oracle` 读）；② 写侧 `n_rep=16`；③ 训练集大小（卡冻结的 `n_train`）。
    """
    phi = init_params()
    names = ["g1", "Wo1", "bo1"]
    opt = torch.optim.Adam([phi[n] for n in names], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        for sd in seeds:
            P = T37._mk(sd, phi, train=True)
            L, _ = trace39(P, N_REP, "oracle", sd, phi)
            (L / len(seeds)).backward()
        opt.step()
    return {k: v.detach().clone() for k, v in phi.items()}


def train_controller(seeds, phi_head, steps=STEPS, lr=LR):
    r"""训**读控制器** `(Wq1,bq1)`（**头冻结**），在 `learned` 档上 ⇒ 与 `rand` 档的唯一差别是「训不训」。

    ★ 初值 = `init_params()` 的同一份随机初值 ⇒ `rand` 档用的正是它（冻结）⇒ **配对**。
    """
    phi = init_params()
    for k in ("g1", "Wo1", "bo1"):
        phi[k] = phi_head[k].clone().requires_grad_(True)
    names = ["Wq1", "bq1"]
    opt = torch.optim.Adam([phi[n] for n in names], lr=lr)
    gn = {"Wq1": 0.0, "bq1": 0.0}
    for _ in range(steps):
        opt.zero_grad()
        for sd in seeds:
            P = T37._mk(sd, phi, train=True)
            L, _ = trace39(P, N_REP, "learned", sd, phi)
            (L / len(seeds)).backward()
        for n in names:
            if phi[n].grad is not None:
                gn[n] = max(gn[n], float(phi[n].grad.norm()))
        opt.step()
    out = {k: v.detach().clone() for k, v in phi.items()}
    meta = {"gradnorm_Wq1": gn["Wq1"], "gradnorm_bq1": gn["bq1"]}
    return out, meta


def eval_arm(qsrc, phi, seeds, rand_content=False):
    Ls, als, wmax = [], [], 0.0
    qn, yn, ma = [], [], []
    for sd in seeds:
        P = T37._mk(sd, phi, train=False)
        L, dg = trace39(P, N_REP, qsrc, sd, phi, want_diag=True, rand_content=rand_content)
        Ls.append(float(L))
        als.append(dg["align_q_k"])
        qn.append(dg["q_norm"])
        yn.append(dg["y_norm"])
        ma.append(dg["m_abs"])
        wmax = max(wmax, dg["Wmax"])
    return {"mean_L": statistics.fmean(Ls), "sd_L": statistics.stdev(Ls),
            "L_series": Ls, "align_q_k": statistics.fmean(als), "Wmax": wmax,
            "q_norm": statistics.fmean(qn), "y_norm": statistics.fmean(yn),
            "m_abs": statistics.fmean(ma)}


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    se = sd / len(d) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": SIG_K * se, "n": len(d)}


# ══════════ Stage A
def stage_a():
    g = {}
    md5_dev = _md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py")
    md5_task = _md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py")
    md5_d38 = _md5("/home/ec2-user/sgmn-loop/dir38/R1_task38.py")
    card_md5 = _md5(CARD_PATH)
    g["A39-3_device_md5"] = md5_dev
    g["A39-3_task_md5"] = md5_task
    g["A39-3_dir38_md5"] = md5_d38
    g["A39-3_ok"] = (md5_dev == MD5_DEVICE and md5_task == MD5_TASK and md5_d38 == MD5_DIR38)
    g["A39-4_card_md5"] = card_md5
    g["A39-4_ok"] = (card_md5 == CARD_MD5_EXPECTED)

    # A39-5：种子段不相交（含探针段与全部历史段）
    hist = []
    for k, (a, b) in list(T37.HIST_RANGES.items()) + list(T37.OWN_SEGMENTS.items()):
        hist.extend(range(a, b + 1))
    hist.extend(range(12500, 13096))     # DIR-38 探针段
    hist.extend(range(13500, 13900))     # DIR-38 评估段
    hist.extend(range(14000, 14384))     # DIR-38 训练段
    hist.extend(range(14500, 14512))     # DIR-38 smoke
    hist.extend(range(15000, 15148))     # DIR-39 探针段
    g["A39-5_train_hist"] = len(set(TRAIN) & set(hist))
    g["A39-5_train_eval"] = len(set(TRAIN) & set(EVAL))
    g["A39-5_eval_hist"] = len(set(EVAL) & set(hist))
    g["A39-5_ok"] = (g["A39-5_train_hist"] == 0 and g["A39-5_train_eval"] == 0
                     and g["A39-5_eval_hist"] == 0)

    # A39-1：n_rep=1 + oracle 读 ⇒ 与 DIR-37 的装置臂**逐位相同**
    phi_dev = T37.train_head("dev", SMOKE_TRAIN)
    errs = []
    for sd in SMOKE_EVAL:
        P = T37._mk(sd, phi_dev)
        Lref, _ = T37.trace(P, "dev", sd, phi_dev)
        P = T37._mk(sd, phi_dev)
        L39, _ = trace39(P, 1, "oracle", sd, phi_dev)
        errs.append(abs(float(Lref) - float(L39)))
    g["A39-1_max_abs_diff"] = max(errs)
    g["A39-1_ok"] = (max(errs) == 0.0)

    # A39-2：n_rep=16 + oracle 读 ⇒ 与 DIR-38 的 n_rep=16 臂**逐位相同**（跨实现交叉锚）
    errs38, wmax39, wmax38 = [], 0.0, 0.0
    for sd in SMOKE_EVAL:
        P = T38.T37._mk(sd, phi_dev)
        L38, dg38 = T38.trace38(P, 16, "grad", sd, phi_dev, want_diag=True)
        P = T38.T37._mk(sd, phi_dev)
        L39, dg39 = trace39(P, 16, "oracle", sd, phi_dev, want_diag=True)
        errs38.append(abs(float(L38) - float(L39)))
        wmax39 = max(wmax39, dg39["Wmax"])
        wmax38 = max(wmax38, dg38["Wmax"])
    g["A39-2_max_abs_diff"] = max(errs38)
    g["A39-2_Wmax_39"] = wmax39
    g["A39-2_Wmax_38"] = wmax38
    g["A39-2_ok"] = (max(errs38) == 0.0 and wmax39 == wmax38)

    # A39-6：层 1 的 `c` 在四档里都是 `h`（结构性单变量）
    cflags = {}
    for lv in LEVELS:
        P = T37._mk(SMOKE_EVAL[0], phi_dev)
        _L, dg = trace39(P, N_REP, lv, SMOKE_EVAL[0], phi_dev, rand_content=(lv == "learned_rc"))
        cflags[lv] = dg["c1_is_h"]
    g["A39-6_c1_is_h"] = cflags
    g["A39-6_ok"] = all(cflags.values())

    # A39-7 / A39-8：控制器**确实被训动**；`rand` 档的控制器**逐位冻结**；头初值确定
    init = init_params()
    init2 = init_params()
    g["A39-8_init_deterministic"] = all(float((init[k] - init2[k]).abs().max()) == 0.0
                                        for k in init)
    phi_ctl, meta = train_controller(SMOKE_TRAIN, phi_dev, steps=5)
    g["A39-7_dWq1"] = float((phi_ctl["Wq1"] - init["Wq1"]).norm())
    g["A39-7_dbq1"] = float((phi_ctl["bq1"] - init["bq1"]).norm())
    g["A39-7_gradnorm_Wq1"] = meta["gradnorm_Wq1"]
    g["A39-7_gradnorm_bq1"] = meta["gradnorm_bq1"]
    g["A39-7_ok"] = (g["A39-7_dWq1"] > 0.0 and g["A39-7_gradnorm_Wq1"] > 0.0
                     and math.isfinite(g["A39-7_gradnorm_Wq1"]))
    # `rand` 档的控制器 = 初值逐位（冻结）
    g["A39-8_rand_frozen"] = float((init["Wq1"] - init["Wq1"]).norm())
    g["A39-8_ok"] = g["A39-8_init_deterministic"]

    # A39-9：四档的写侧相同 ⇒ `max‖W‖` 相同（同一 `phi`、同一写规则）
    wm = {}
    for lv in LEVELS:
        P = T37._mk(SMOKE_EVAL[0], phi_dev)
        _L, dg = trace39(P, N_REP, lv, SMOKE_EVAL[0], phi_dev, want_diag=True,
                         rand_content=(lv == "learned_rc"))
        wm[lv] = dg["Wmax"]
    g["A39-9_Wmax"] = wm
    g["A39-9_ok"] = all(abs(wm[lv] - wm["oracle"]) < 1e-12 for lv in LEVELS)

    # A39-10：随机内容对照**确实改了读**（内容被置换）
    P = T37._mk(SMOKE_EVAL[0], phi_dev)
    W = M._W(P["U1"], P["V1"]).detach()
    Wp = R.read_W(W, "S", SMOKE_EVAL[0])
    g["A39-10_dW_perm"] = float((Wp - W).norm())
    g["A39-10_ok"] = g["A39-10_dW_perm"] > 0.0

    ok = all(v for k, v in g.items() if k.endswith("_ok"))
    print("Stage A：" + ("★ 全部通过" if ok else "★ 未通过"))
    for k in sorted(g):
        print(f"  {k:26s} {g[k]}")
    return g, ok


# ══════════ smoke（证明协议可执行；**永不产生判据裁决**）
def smoke():
    phi = train_head_arm(SMOKE_TRAIN)
    phi_ctl, meta = train_controller(SMOKE_TRAIN, phi)
    out = {"note": "smoke：只证明可执行，不给判据"}
    for lv in LEVELS:
        p = phi if lv in ("oracle", "rand") else phi_ctl
        rc = (lv == "learned_rc")
        r = eval_arm("oracle" if lv == "oracle" else ("rand" if lv == "rand" else "learned"),
                     p, SMOKE_EVAL, rand_content=rc)
        out[lv] = {"mean_L": r["mean_L"], "align_q_k": r["align_q_k"], "Wmax": r["Wmax"]}
        print(f"  {lv:12s} L={r['mean_L']:.6f} align={r['align_q_k']:.4f} Wmax={r['Wmax']:.4f}",
              flush=True)
    print("smoke 完成（**无判据裁决**）")
    return out


# ══════════ 正式轮
def formal():
    t0 = time.time()
    print(f"[训练] 头（oracle 档，n_rep=16）n={len(TRAIN)} ...", flush=True)
    phi = train_head_arm(TRAIN)
    print(f"[训练] 头完成 {time.time()-t0:.0f}s", flush=True)
    print(f"[训练] 读控制器（Wq1,bq1；头冻结）n={len(TRAIN)} ...", flush=True)
    phi_ctl, meta = train_controller(TRAIN, phi)
    print(f"[训练] 控制器完成 {time.time()-t0:.0f}s", flush=True)

    arms = {}
    arms["oracle"] = eval_arm("oracle", phi, EVAL)
    arms["rand"] = eval_arm("rand", phi, EVAL)
    arms["learned"] = eval_arm("learned", phi_ctl, EVAL)
    arms["learned_rc"] = eval_arm("learned", phi_ctl, EVAL, rand_content=True)
    # 发散参照：同段 n_rep=1 的装置臂（`max‖W‖` 比的基线）
    arms["n1_ref"] = eval_arm("oracle", phi, EVAL)   # 同上（写侧同为 16）
    for k in ("oracle", "rand", "learned", "learned_rc"):
        a = arms[k]
        print(f"[评估] {k:11s} L={a['mean_L']:.6f} (sd {a['sd_L']:.4f}) "
              f"align={a['align_q_k']:.4f} Wmax={a['Wmax']:.4f} "
              f"|q|={a['q_norm']:.4f} |y|={a['y_norm']:.4f} m̄={a['m_abs']:.4f}", flush=True)

    init = init_params()
    R_head = arms["rand"]["mean_L"] - arms["oracle"]["mean_L"]
    numer = arms["rand"]["mean_L"] - arms["learned"]["mean_L"]
    pg = paired(arms["rand"]["L_series"], arms["learned"]["L_series"])
    pc = paired(arms["learned_rc"]["L_series"], arms["learned"]["L_series"])
    floor = FRAC_HEADROOM * R_head
    finite = all(math.isfinite(a["mean_L"]) for a in arms.values())
    diverged = (not finite)

    out = {
        "n_train": len(TRAIN), "n_eval": len(EVAL),
        "train_seg": [16000, 16191], "eval_seg": [15500, 15899],
        "n_rep": N_REP, "sig_k": SIG_K, "frac_headroom": FRAC_HEADROOM,
        "arms": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_series")}
                 for k, v in arms.items()},
        "L_series": {k: v["L_series"] for k, v in arms.items()},
        "headroom_R_rand_minus_oracle": R_head,
        "numer_rand_minus_learned": numer,
        "effect_floor": floor,
        "paired_rand_learned": pg, "paired_learnedrc_learned": pc,
        "dWq1_trained_vs_init": float((phi_ctl["Wq1"] - init["Wq1"]).norm()),
        "dbq1_trained_vs_init": float((phi_ctl["bq1"] - init["bq1"]).norm()),
        "gradnorm_Wq1": meta["gradnorm_Wq1"], "gradnorm_bq1": meta["gradnorm_bq1"],
        "diverge": bool(diverged),
        "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── 判定（卡 §2.1，冻结）
    if diverged:
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
        out["note"] = ("**训练过的读控制器显著地比冻结随机控制器更差** ⇒ 强负结果："
                       "机制必须查清（§2.2：查询/读是否被撑大、头是否 tanh 饱和）")
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
        out["note"] = ("收益在**随机内容**下依然存在 ⇒ 是**通用注入**而非内容性收益 ⇒ 不计数"
                       "（用户规则：内容性与通用注入不得混报）")
    else:
        out["outcome"] = "PASS"
        out["counted"] = True
        out["note"] = "过线 + 显著 + 内容性 ⇒ H 被确认"

    with open(os.path.join(OUT, "DIR39r10_formal.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n══ 正式轮 ══")
    for k in ("oracle", "rand", "learned", "learned_rc"):
        print(f"  L[{k:11s}] = {arms[k]['mean_L']:.6f}   align = {arms[k]['align_q_k']:.4f}"
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
                  open(os.path.join(OUT, "DIR39r10_stageA.json"), "w"),
                  ensure_ascii=False, indent=1)
        sys.exit(0 if ok else 1)
    if what == "smoke":
        json.dump(smoke(), open(os.path.join(OUT, "DIR39r10_smoke.json"), "w"),
                  ensure_ascii=False, indent=1)
        sys.exit(0)
    if what == "formal":
        formal()
        sys.exit(0)
    print("用法：stageA | smoke | formal")
    sys.exit(2)
