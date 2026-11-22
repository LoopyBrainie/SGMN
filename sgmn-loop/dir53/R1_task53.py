#!/opt/pytorch/bin/python
r"""DIR-53 · **正式轮**实现：$P_k$ 的「**离线学习**」有没有独立收益？

卡：`revolve/DIR-53/DIRECTION_rev53_v12.md`（rev 1.2，**阈值与判据已冻结**；md5 由 `A53-2` 守卫）。

派生自 `dir41/R1_task41.py`（**逐字**：`init_params` / `train_arm40` / `eval_arm40` / `paired` /
种子段 / 输出格式 / 判定表骨架），而 `dir41` 又逐字派生自 `dir40/R1_task40.py`。
⇒ ★ **装置 `dir31`、任务层 `dir33`、测量层 `dir37`、协议层 `dir40` 一律不改**（`A53-1`）。

★ **唯一的自变量**：层 1 写键映射 `Pk1` 的**来源**（卡 §2.2）：
   `FIX`    = 固定随机（`randn(D_H,D_K)·(1/√D_H)·KEY_SCALE`，种子 `20262201+j−1`）
   `OFF`    = **同一初值**，用**写侧自检索目标**离线训练 250 步（每步行范数投影）
   `ORACLE` = 与 `FIX` **逐位相同**的 `Pk1`，但读源换成 `q = tanh(h·Pk1)`（描述性上限）

★ **五格 × 三臂 = 15 次头训练**（`P1..P5` × `FIX/OFF/ORACLE`）。

用法：stageA | smoke | formal --cell {P1..P5} --arm {FIX,OFF,ORACLE} | offcheck | merge
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
for _d in ("dir31", "dir33", "dir37", "dir38", "dir39", "dir40"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_d}")
import R1_task33 as R           # 任务层（逐字未改）
import R1_grad as M             # 装置（逐字未改）
import R1_task37 as T37         # 测量层（_mk / 训练协议 / 迹线）
import R1_task40 as T40         # ★ DIR-40 的测量层（**逐字复用**）

OUT = "/home/ec2-user/sgmn-loop/dir53/out"
D_H, D_V, D_K = R.D_H, R.D_V, R.D_K
T_STEPS = M.T_STEPS
SEED0 = M.SEED0

# ══════════ 冻结常量（卡 §3.2 / §7）
CARD_PATH = "/home/ec2-user/sgmn-loop/revolve/DIR-53/DIRECTION_rev53_v12.md"
CARD_MD5_EXPECTED = "c5fc757c7e462eb0b85f6844c1caa4c1"
MD5_DEVICE = "8fa40f708600bec7e6c484727d2180c7"
MD5_TASK = "96ef89e87ec61556b52e0e464850087e"
MD5_DIR37 = "9236ac0490be9c413f1e8af163289207"
MD5_DIR40 = "baab3566891591944efb60a5b6dc9623"
N_REP = 16
N_TRAIN = 96
N_EVAL = 400
STEPS, LR = 250, 0.03
SIG_K = 3.0
FRAC_HEADROOM = 0.20
PASS_RATE_MIN = 4
N_ROWS = D_H

# ══════════ 种子段（卡 §6；与 DIR-41 逐字相同 ⇒ 可比）
TRAIN_OFF = list(range(21000, 21000 + N_TRAIN))
EVAL_OFF = list(range(21500, 21500 + N_EVAL))
SMOKE_TRAIN_OFF = list(range(21100, 21104))
SMOKE_EVAL_OFF = list(range(21110, 21112))
TRAIN = [SEED0 + o for o in TRAIN_OFF]
EVAL = [SEED0 + o for o in EVAL_OFF]
SMOKE_TRAIN = [SEED0 + o for o in SMOKE_TRAIN_OFF]
SMOKE_EVAL = [SEED0 + o for o in SMOKE_EVAL_OFF]

# ══════════ 五格（卡 §2.3）
PK1_SEEDS = {"P1": 20262201, "P2": 20262202, "P3": 20262203,
             "P4": 20262204, "P5": 20262205}
CELLS = tuple(PK1_SEEDS)
ARMS = ("FIX", "OFF", "ORACLE")
LEVELS = ("FIX", "OFF", "OFF_rc", "ORACLE")


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ══════════ `Pk1` 初值的生成式（卡 §2.3：**与装置源码逐字同式**）
def pk1_for(cell):
    r"""`randn(D_H, D_K, generator=g) * (1/sqrt(D_H)) * KEY_SCALE`（先 `s_h` 后 `KEY_SCALE`）。

    ★ 与装置 `dir31/build`（`rn(D_H,D_K,scale=s_h)`）+ `dir33/build`（`*KEY_SCALE`）**逐字同式**。
    ★ **与 DIR-41 的 `shared_pk1_for` 同式、不同种子段**（`20262201..` vs `20262101..`）⇒ 边缘律相同、逐位不同。
    """
    g = torch.Generator().manual_seed(int(PK1_SEEDS[cell]))
    s_h = 1.0 / math.sqrt(D_H)
    return torch.randn(D_H, D_K, generator=g) * s_h * R.KEY_SCALE


# ══════════ 离线学习（卡 §2.2a，**冻结**；只用训练段 + 装置侧量，**不含任务损失**）
def offline_pairs(seeds, phi0):
    r"""训练段的写侧 `(h, v)` 对：与装置 `dir33` 的写步**逐字同源**。

    装置 `KS = [tanh(H[s] @ Pk1) for s in range(T)]`、`VS = [tanh(H[s] @ Pv1) for s in range(T)]`
    （`hs_of` 只依赖 `a, A1, b1` ⇒ **与 `phi` 无关**；这里仍显式传 `phi0` 以固定 `_mk` 的其余状态）。
    ⇒ 96 实例 × `T=16` 步 = **1536 对**。
    """
    Hp, Vp = [], []
    for sd in seeds:
        P = T37._mk(sd, phi0, train=True)
        H = R.hs_of(P)
        for s in range(T_STEPS):
            h = H[s].detach()
            Hp.append(h)
            Vp.append(torch.tanh(h @ P["Pv1"]))
    return torch.stack(Hp), torch.stack(Vp)


def offline_learn_pk1(pk0, seeds, steps=STEPS, lr=LR, phi0=None):
    r"""在训练段上用**写侧自检索目标**离线训练 `Pk1`（卡 §2.2a）。

    `W = Σ_j v_j k_jᵀ`、`y_i = W k_i`、`L_off = mean_i ‖y_i − v_i‖²`（与装置读写同式，无新超参）。
    ★ **只对 `Pk1` 求梯度**；每步后**按行**把 `‖P_k‖ᵢ` 投回初值行范数（`A53-6` 守卫）。
    ★ `meta` 里带 `pk0_sha1`/`pk_sha1`/`pk0_fro` ⇒ `A53-5` 与 `merge` 的跨臂同源守卫可**真读**。
    ★ 返回的 `meta` 里 `L_off_first` = 前 20 步均值、`L_off_last` = 后 20 步均值（**窗口不重叠**）。
    """
    if phi0 is None:
        phi0 = T40.init_params()
    Hm, Vm = offline_pairs(seeds, phi0)
    n0_row = pk0.norm(dim=1).clone()          # (D_H,)   仅用于报告/断言
    n0c = n0_row.unsqueeze(1)                 # (D_H,1)  供投影广播（DEF-53-3）
    pk = pk0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([pk], lr=lr)
    Ls, gn_max = [], 0.0
    for _ in range(steps):
        opt.zero_grad()
        K = torch.tanh(Hm @ pk)                 # (n_pairs, D_K)
        W = Vm.T @ K                            # (D_V, D_K)  ← 与装置写规则同式
        Y = (W @ K.T).T                         # (n_pairs, D_V) ← 与装置读规则同式
        L = ((Y - Vm) ** 2).mean()
        L.backward()
        gn_max = max(gn_max, float(pk.grad.norm()))
        Ls.append(float(L.detach()))
        opt.step()
        with torch.no_grad():                   # ★ 行范数投影：只改方向、不改尺度
            pk.mul_(n0c / pk.norm(dim=1, keepdim=True))
    with torch.no_grad():
        row_dev = (pk.norm(dim=1) - n0_row).abs()
    meta = {
        "L_off_first20": statistics.fmean(Ls[:20]),
        "L_off_last20": statistics.fmean(Ls[-20:]),
        "L_off_sd_first20": statistics.stdev(Ls[:20]) if len(Ls) >= 2 else 0.0,
        "L_off_sd_last20": statistics.stdev(Ls[-20:]) if len(Ls) >= 2 else 0.0,
        "L_off_se_window": math.sqrt((statistics.variance(Ls[:20]) / 20.0)
                                     + (statistics.variance(Ls[-20:]) / 20.0)),
        "grad_pk_max": gn_max, "grad_pk_last": float(pk.grad.norm()),
        "rel_dPk": float((pk.detach() - pk0).norm() / pk0.norm()),
        "row_norm_dev_max": float(row_dev.max()),
        "row_norm_dev_rel_max": float((row_dev / n0_row).max()),
        "n_pairs": int(Hm.shape[0]), "steps": steps, "lr": lr,
        "pk0_sha1": hashlib.sha1(pk0.detach().contiguous().numpy().tobytes()).hexdigest(),
        "pk_sha1": hashlib.sha1(pk.detach().contiguous().numpy().tobytes()).hexdigest(),
        "pk0_fro": float(pk0.norm()),
    }
    return pk.detach().clone(), meta


# ══════════ 正式轮：单格单臂（多格多臂在 trainer 上**并行**；`merge` 汇总裁决）
def formal_arm(cell, arm):
    r"""跑一格的一臂 ⇒ 写 `DIR53r12_{cell}_{arm}.json`（**只放读数，不放裁决**）。"""
    t0 = time.time()
    pk_fix = pk1_for(cell)
    off_meta = None
    if arm == "OFF":
        use_pk, off_meta = offline_learn_pk1(pk_fix, TRAIN)
        use_pk = use_pk.detach()
        print(f"[{cell}/OFF] 离线学完 {time.time()-t0:.0f}s  "
              f"L_off {off_meta['L_off_first20']:.4f}→{off_meta['L_off_last20']:.4f}  "
              f"‖ΔPk‖/‖Pk0‖={off_meta['rel_dPk']:.4f}  "
              f"行范数偏差 max={off_meta['row_norm_dev_max']:.2e}", flush=True)
    else:
        use_pk = pk_fix
    qsrc, tctrl = {"ORACLE": ("oracle", False),
                   "FIX": ("learned", True),
                   "OFF": ("learned", True)}[arm]
    print(f"[{cell}/{arm}] 训头{'(+控制器)' if tctrl else ''}（qsrc={qsrc}，n_rep={N_REP}）"
          f"n={len(TRAIN)} ...", flush=True)
    phi, meta = T40.train_arm40(qsrc, TRAIN, shared_pk=use_pk, train_ctrl=tctrl, steps=STEPS)
    print(f"[{cell}/{arm}] 训练完成 {time.time()-t0:.0f}s  梯度 {meta['gradnorms']}", flush=True)
    arms = {arm: T40.eval_arm40(qsrc, phi, EVAL, shared_pk=use_pk)}
    if arm == "OFF":
        arms["OFF_rc"] = T40.eval_arm40(qsrc, phi, EVAL, shared_pk=use_pk, rand_content=True)
    for k, a in arms.items():
        print(f"[{cell}/{arm}] 评估 {k:8s} L={a['mean_L']:.6f} (sd {a['sd_L']:.4f}) "
              f"cos={a['cos_q_k']:.4f} align={a['align_q_k']:.4f} |q|={a['q_norm']:.4f} "
              f"|y|={a['y_norm']:.4f} m̄={a['m_abs']:.4f} Wmax={a['Wmax']:.4f}", flush=True)
    out = {"cell": cell, "arm": arm, "n_train": len(TRAIN), "n_eval": len(EVAL),
           "train_off": [TRAIN_OFF[0], TRAIN_OFF[-1]],
           "eval_off": [EVAL_OFF[0], EVAL_OFF[-1]],
           "n_rep": N_REP, "steps": STEPS, "lr": LR,
           "pk1_seed": PK1_SEEDS[cell], "pk1_fro": float(use_pk.norm()),
           "pk1_sha1": hashlib.sha1(use_pk.numpy().tobytes()).hexdigest(),
           "rel_dPk_arm": (0.0 if arm in ("FIX", "ORACLE")
                           else float((use_pk - pk_fix).norm() / pk_fix.norm())),
           "arms": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_series")}
                    for k, v in arms.items()},
           "L_series": {k: v["L_series"] for k, v in arms.items()},
           "gradnorms": meta["gradnorms"],
           "dWq1_from_init": meta["dWq1_from_init"], "dbq1_from_init": meta["dbq1_from_init"],
           "off_meta": off_meta,
           "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"DIR53r12_{cell}_{arm}.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


# ══════════ Stage A（硬闸；断言必须真的读被断言的对象 ⇒ 纪律 97）
def stage_a():
    g = {}
    phi_dev = T40.init_params()

    # ── A53-1：装置/任务层/测量层/协议层 md5 未变
    md5s = {"dir31/R1_grad.py": MD5_DEVICE, "dir33/R1_task33.py": MD5_TASK,
            "dir37/R1_task37.py": MD5_DIR37, "dir40/R1_task40.py": MD5_DIR40}
    got = {p: _md5("/home/ec2-user/sgmn-loop/" + p) for p in md5s}
    g["A53-1_md5"] = got
    g["A53-1_ok"] = all(got[p] == v for p, v in md5s.items())

    # ── A53-2：卡 md5 守卫（rev 1.2）
    g["A53-2_card_md5"] = _md5(CARD_PATH)
    g["A53-2_ok"] = g["A53-2_card_md5"] == CARD_MD5_EXPECTED

    # ── A53-3：三臂的**头初值逐位相同**（`steps=0` ⇒ 直接回读初值）
    spk = pk1_for("P1")
    init_arms = {}
    for arm in ARMS:
        qsrc, tctrl = {"ORACLE": ("oracle", False), "FIX": ("learned", True),
                       "OFF": ("learned", True)}[arm]
        ph, _m = T40.train_arm40(qsrc, SMOKE_TRAIN, shared_pk=spk,
                                 train_ctrl=tctrl, steps=0)
        init_arms[arm] = ph
    same3 = all(float((init_arms["FIX"][k] - init_arms["ORACLE"][k]).abs().max()) == 0.0
                and float((init_arms["FIX"][k] - init_arms["OFF"][k]).abs().max()) == 0.0
                for k in init_arms["FIX"])
    i1, i2 = T40.init_params(), T40.init_params()
    g["A53-3_same_across_arms"] = bool(same3)
    g["A53-3_init_deterministic"] = all(
        float((i1[k] - i2[k]).abs().max()) == 0.0 for k in i1)
    g["A53-3_ok"] = bool(same3 and g["A53-3_init_deterministic"])

    # ── A53-4：`FIX` 与 `ORACLE` 的 `Pk1` **逐位相同**（**真断言**，非"同调用比同调用"）
    #   (i) 生成器**无共享状态**：同一格两次调用之间**插入另一格**的调用，结果必须逐位相同
    x4 = pk1_for("P1")
    _ = pk1_for("P3")                      # 干扰：若共用全局生成器/顺序不同，x4 与 y4 会不同
    y4 = pk1_for("P1")
    g["A53-4_interleaved_max_abs_diff"] = float((x4 - y4).abs().max())
    #   (ii) `trace40` 的钩子**真的**把本格矩阵写进实例（回读 `P["Pk1"]` 逐位比较）
    P = T37._mk(SMOKE_EVAL[0], phi_dev)
    T40.trace40(P, 0, "oracle", SMOKE_EVAL[0], phi_dev, shared_pk1=x4)
    g["A53-4_hook_max_abs_diff"] = float((P["Pk1"] - x4).abs().max())
    g["A53-4_ok"] = (g["A53-4_interleaved_max_abs_diff"] == 0.0
                     and g["A53-4_hook_max_abs_diff"] == 0.0)

    # ── A53-5：`OFF` 的**初值** `Pk0` 与 `FIX` 逐位相同（真跑离线学习，读它记录的 `pk0_sha1`）
    pk25 = pk1_for("P2")
    _, om5 = offline_learn_pk1(pk25, SMOKE_TRAIN, steps=3, lr=LR)
    g["A53-5_pk0_sha1"] = om5["pk0_sha1"]
    g["A53-5_fix_sha1"] = hashlib.sha1(pk25.detach().contiguous().numpy().tobytes()).hexdigest()
    g["A53-5_ok"] = (g["A53-5_pk0_sha1"] == g["A53-5_fix_sha1"])

    # ── A53-6：**行范数投影真的生效**（真跑 250 步，逐行 < 1e-12）
    pk0 = pk1_for("P3")
    pk_off, om = offline_learn_pk1(pk0, SMOKE_TRAIN, steps=STEPS, lr=LR)
    n0, n1 = pk0.norm(dim=1), pk_off.norm(dim=1)
    g["A53-6_row_dev_max"] = float((n1 - n0).abs().max())
    g["A53-6_rel_dPk"] = om["rel_dPk"]
    g["A53-6_ok"] = bool(g["A53-6_row_dev_max"] < 1e-12 and om["rel_dPk"] > 0.0)

    # ── A53-7：`FIX` 臂训练后 `Pk1` **逐位不变**（无原地改写）
    spk7 = pk1_for("P4")
    ref7 = spk7.detach().clone()
    T40.train_arm40("learned", SMOKE_TRAIN, shared_pk=spk7, train_ctrl=True, steps=3)
    g["A53-7_max_abs_diff"] = float((spk7 - ref7).abs().max())
    g["A53-7_ok"] = g["A53-7_max_abs_diff"] == 0.0

    # ── A53-8：5 张抽取两两不同（10 对）+ 各自与装置实例 `Pk1` 不同
    mats = {c: pk1_for(c) for c in CELLS}
    pairs = {}
    for a in CELLS:
        for b in CELLS:
            if a < b:
                pairs[f"{a}|{b}"] = bool(float((mats[a] - mats[b]).abs().max()) > 0.0)
    dev_pk = T37._mk(SMOKE_EVAL[0], phi_dev)["Pk1"]
    vs_dev = {c: bool(float((mats[c] - dev_pk).abs().max()) > 0.0) for c in CELLS}
    g["A53-8_pairs"] = pairs
    g["A53-8_vs_device_instance"] = vs_dev
    g["A53-8_ok"] = bool(all(pairs.values()) and all(vs_dev.values()))

    # ── A53-9：训练段/评估段种子隔离（**offset 空间**；含历史段）
    hist = []
    for _k, (a, b) in list(T37.HIST_RANGES.items()) + list(T37.OWN_SEGMENTS.items()):
        hist.extend(range(a, b + 1))
    for a, b in [(12500, 13096), (13500, 13900), (14000, 14384), (14500, 14512),
                 (15000, 15148), (15500, 15900), (16000, 16192), (16400, 16412),
                 (17000, 17048), (17100, 17164), (17200, 17264),
                 (18000, 18191), (18500, 18899), (18900, 18911),
                 (20400, 20412), (20500, 20595), (20600, 20999)]:
        hist.extend(range(a, b + 1))
    hs = set(hist)
    g["A53-9_train_hist"] = len(set(TRAIN_OFF) & hs)
    g["A53-9_eval_hist"] = len(set(EVAL_OFF) & hs)
    g["A53-9_smoke_hist"] = len(set(SMOKE_TRAIN_OFF + SMOKE_EVAL_OFF) & hs)
    g["A53-9_train_eval"] = len(set(TRAIN_OFF) & set(EVAL_OFF))
    g["A53-9_ok"] = (g["A53-9_train_hist"] == 0 and g["A53-9_eval_hist"] == 0
                     and g["A53-9_smoke_hist"] == 0 and g["A53-9_train_eval"] == 0)

    # ── A53-10：非有限检查（三臂在 1 个实例上各跑一次）
    fin = {}
    for arm in ARMS:
        qsrc = "oracle" if arm == "ORACLE" else "learned"
        P = T37._mk(SMOKE_EVAL[0], phi_dev)
        L, dg = T40.trace40(P, N_REP, qsrc, SMOKE_EVAL[0], phi_dev,
                            want_diag=True, shared_pk1=pk1_for("P5"))
        fin[arm] = {"L": float(L), "finite": bool(math.isfinite(float(L)))}
    g["A53-10_finite"] = fin
    g["A53-10_ok"] = all(v["finite"] for v in fin.values())

    # ── A53-11：**Freeze C 守卫** —— `∂L_task/∂P_k` 逐元素为 0
    pkl = pk1_for("P1").clone().requires_grad_(True)
    P = T37._mk(SMOKE_EVAL[0], phi_dev)
    L, _ = T40.trace40(P, N_REP, "learned", SMOKE_EVAL[0], phi_dev, shared_pk1=pkl)
    L.backward()
    gdn = 0.0 if pkl.grad is None else float(pkl.grad.abs().max())
    g["A53-11_task_grad_pk_max"] = gdn
    g["A53-11_grad_is_none"] = pkl.grad is None
    g["A53-11_ok"] = (gdn == 0.0)

    # ── A53-12：**离线目标真的可训**（`∂L_off/∂P_k ≠ 0` 且 250 步后 `‖ΔPk‖>0`）
    pkl2 = pk1_for("P2").clone().requires_grad_(True)
    Hm, Vm = offline_pairs(SMOKE_TRAIN, phi_dev)
    K = torch.tanh(Hm @ pkl2)
    W = Vm.T @ K
    Loff = (((W @ K.T).T - Vm) ** 2).mean()
    Loff.backward()
    gn_off = float(pkl2.grad.norm())
    g["A53-12_off_grad_norm"] = gn_off
    g["A53-12_n_pairs"] = int(Hm.shape[0])
    g["A53-12_ok"] = bool(gn_off > 0.0 and g["A53-6_rel_dPk"] > 0.0)

    ok = all(v for k, v in g.items() if k.endswith("_ok"))
    print("Stage A：" + ("★ 全部通过" if ok else "★ 未通过"))
    for k in sorted(g):
        print(f"  {k:30s} {g[k]}")
    return g, ok


# ══════════ smoke（证明协议可执行；**永不产生判据裁决**）
def smoke():
    out = {"note": "smoke：只证明可执行，不给判据"}
    for cell in ("P1",):
        pk_fix = pk1_for(cell)
        pk_off, om = offline_learn_pk1(pk_fix, SMOKE_TRAIN, steps=8, lr=LR)
        for arm in ARMS:
            use_pk = pk_off if arm == "OFF" else pk_fix
            qsrc, tctrl = {"ORACLE": ("oracle", False), "FIX": ("learned", True),
                           "OFF": ("learned", True)}[arm]
            phi, meta = T40.train_arm40(qsrc, SMOKE_TRAIN, shared_pk=use_pk,
                                        train_ctrl=tctrl, steps=3)
            e = T40.eval_arm40(qsrc, phi, SMOKE_EVAL, shared_pk=use_pk)
            out[f"{cell}/{arm}"] = {"mean_L": e["mean_L"], "cos_q_k": e["cos_q_k"],
                                    "Wmax": e["Wmax"], "c_eq_h_max": e["c_eq_h_max"],
                                    "dWq1_from_init": meta["dWq1_from_init"],
                                    "rel_dPk": (om["rel_dPk"] if arm == "OFF" else 0.0)}
            print(f"  {cell}/{arm:7s} L={e['mean_L']:.6f} cos={e['cos_q_k']:.4f} "
                  f"Wmax={e['Wmax']:.4f} c_eq_h={e['c_eq_h_max']:.1e} "
                  f"|Wq-init|={meta['dWq1_from_init']:.3e}", flush=True)
    print("smoke 完成（**无判据裁决**）")
    return out


# ══════════ 汇总裁决（卡 §3.3 / §4，**冻结**）
def merge():
    r"""读 15 个 `DIR53r12_*.json` ⇒ 按**冻结**判定表裁决。★ **不读探针文件**。"""
    arms, series, metas, missing, inert = {}, {}, {}, [], []
    for cell in CELLS:
        for arm in ARMS:
            p = os.path.join(OUT, f"DIR53r12_{cell}_{arm}.json")
            if not os.path.exists(p):
                missing.append(f"{cell}/{arm}")
                continue
            d = json.load(open(p, encoding="utf-8"))
            metas[f"{cell}/{arm}"] = d
            for k, v in d["arms"].items():
                arms[(cell, k)] = v
                series[(cell, k)] = d["L_series"][k]
            if arm == "OFF":
                rd = float(d.get("rel_dPk_arm", 0.0))
                rn = float((d.get("off_meta") or {}).get("row_norm_dev_max", 0.0))
                if not (rd > 0.0 and rn < 1e-12):
                    inert.append(f"{cell}/OFF(rel_dPk={rd:.3e},row_dev={rn:.2e})")

    out = {"n_train": N_TRAIN, "n_eval": N_EVAL, "n_rep": N_REP, "steps": STEPS,
           "sig_k": SIG_K, "frac_headroom": FRAC_HEADROOM,
           "pass_rate_min": PASS_RATE_MIN, "cells": {}, "missing": missing,
           "inert_arms": inert}

    bad_n = []
    for cell in CELLS:
        for arm in ARMS:
            d = metas.get(f"{cell}/{arm}")
            if d and (d.get("n_train") != N_TRAIN or d.get("n_eval") != N_EVAL
                      or d.get("n_rep") != N_REP or d.get("steps") != STEPS):
                bad_n.append(f"{cell}/{arm}")
    out["bad_n"] = bad_n

    # ★ 运行期同源守卫（A53-4 / A53-5 的跨臂版本，**不是判据**）：同格 `FIX` 与 `ORACLE` 的
    #   `pk1_sha1` 必须相等；`OFF` 的 `off_meta.pk0_sha1` 必须等于同格 `FIX` 的 `pk1_sha1`。
    sha_mismatch = []
    for cell in CELLS:
        df, do = metas.get(f"{cell}/FIX"), metas.get(f"{cell}/ORACLE")
        dd = metas.get(f"{cell}/OFF")
        if df and do and df.get("pk1_sha1") != do.get("pk1_sha1"):
            sha_mismatch.append(f"{cell}:FIX!=ORACLE")
        if df and dd:
            p0 = (dd.get("off_meta") or {}).get("pk0_sha1")
            if p0 and p0 != df.get("pk1_sha1"):
                sha_mismatch.append(f"{cell}:OFF.pk0!=FIX")
    out["sha_mismatch"] = sha_mismatch

    for cell in CELLS:
        if any((cell, lv) not in arms for lv in LEVELS):
            continue
        Lf = arms[(cell, "FIX")]["mean_L"]
        Lo = arms[(cell, "ORACLE")]["mean_L"]
        Loff = arms[(cell, "OFF")]["mean_L"]
        Lrc = arms[(cell, "OFF_rc")]["mean_L"]
        Rj = Lf - Lo
        pd = T40.paired(series[(cell, "FIX")], series[(cell, "OFF")])
        pc = T40.paired(series[(cell, "OFF_rc")], series[(cell, "OFF")])
        delta = Lf - Loff
        floor = FRAC_HEADROOM * Rj
        p1 = delta >= SIG_K * pd["se"]
        p2 = delta >= floor
        p3 = pc["mean"] >= SIG_K * pc["se"]
        om = metas.get(f"{cell}/OFF", {}).get("off_meta") or {}
        train_improved = bool(
            (om.get("L_off_first20", 0.0) - om.get("L_off_last20", 0.0))
            >= SIG_K * om.get("L_off_se_window", 0.0))
        out["cells"][cell] = {
            "L_FIX": Lf, "L_OFF": Loff, "L_OFF_rc": Lrc, "L_ORACLE": Lo,
            "R": Rj, "floor_0.20R": floor, "delta": delta,
            "paired_delta": pd, "paired_content": pc,
            "P1_significance": bool(p1), "P2_effect": bool(p2), "P3_content": bool(p3),
            "pass": bool(p1 and p2 and p3),
            "off_train_improved": train_improved,
            "off_L_first20": om.get("L_off_first20"), "off_L_last20": om.get("L_off_last20"),
            "off_rel_dPk": om.get("rel_dPk"),
            "cos": {lv: arms[(cell, lv)]["cos_q_k"] for lv in LEVELS},
            "Wmax": {lv: arms[(cell, lv)]["Wmax"] for lv in LEVELS},
        }

    passed = [c for c in CELLS if out["cells"].get(c, {}).get("pass")]
    out["pass_count"] = len(passed)
    out["pass_cells"] = passed
    out["overfit_only"] = [c for c in CELLS
                           if out["cells"].get(c, {}).get("off_train_improved")
                           and not out["cells"].get(c, {}).get("pass")]

    finite = all(math.isfinite(a["mean_L"]) for a in arms.values())
    ses = []
    for c in out["cells"].values():
        ses += [c["paired_delta"]["se"], c["paired_content"]["se"]]
    finite = finite and all(math.isfinite(s) for s in ses)

    if missing or bad_n or inert or sha_mismatch:
        out["outcome"] = "INCOMPLETE"
        out["counted"] = False
        out["note"] = ("缺臂 / n 不符 / `OFF` 臂未真的被训到（`rel_dPk=0` 或行范数投影失效）"
                       "⇒ 判据**从未被计算** ⇒ 整轮作废（**不是** FAIL）")
    elif not finite:
        out["outcome"] = "DIVERGED"
        out["counted"] = False
        out["note"] = "存在非有限读数 ⇒ 整轮作废"
    elif any(c["R"] <= 0 for c in out["cells"].values()):
        out["outcome"] = "INDETERMINATE"
        out["counted"] = False
        out["note"] = "任一格 R ≤ 0 ⇒ 「oracle 是上限」的前提不成立 ⇒ 不可判定（**不是失败**）"
    elif len(out["overfit_only"]) > 0 and out["pass_count"] < PASS_RATE_MIN:
        out["outcome"] = "OVERFIT-ONLY"
        out["counted"] = False
        out["note"] = ("离线目标在训练段显著下降，但评估段完整 PASS 数不足 ⇒ 收益不迁移"
                       "（`H53` 否证；归因 = 过拟合）")
    elif out["pass_count"] < PASS_RATE_MIN:
        out["outcome"] = "FIXED-SUFFICES"
        out["counted"] = False
        out["note"] = ("完整 PASS 数 < 4/5 ⇒ 在本玩具/本协议/本训练预算下，把 `Pk` 学出来换不到"
                       "超过 20% 可寻址区间的收益 ⇒ **固定随机 `Pk` 已足够**（`H53` 否证）"
                       "★ 不得写成「学习无用」「`Pk` 不可学」")
    else:
        out["outcome"] = "PASS"
        out["counted"] = True
        out["note"] = "5 张预注册独立抽取中至少 4 张通过三谓词 ⇒ 离线学习 `Pk` 有独立的内容性收益（`H53` 确认）"

    out["gradnorms"] = {k: v["gradnorms"] for k, v in metas.items()}
    out["dWq1_from_init"] = {k: v["dWq1_from_init"] for k, v in metas.items()}
    out["secs"] = {k: v["secs"] for k, v in metas.items()}
    out["pk1_seed"] = dict(PK1_SEEDS)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "DIR53r12_formal.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n══ DIR-53 正式轮（汇总裁决） ══")
    for cell in CELLS:
        c = out["cells"].get(cell)
        if not c:
            print(f"  [{cell}] 缺读数")
            continue
        print(f"\n── 格 {cell} ──")
        for lv in LEVELS:
            print(f"   L[{lv:8s}] = {c['L_' + lv]:12.6f}   cos = {c['cos'][lv]:+.4f}")
        print(f"   R = {c['R']:.6f}   地板 0.20R = {c['floor_0.20R']:.6f}   "
              f"Δ = {c['delta']:.6f}   3se = {c['paired_delta']['3se']:.6f}")
        print(f"   内容差 C = {c['paired_content']['mean']:.6f} "
              f"3se = {c['paired_content']['3se']:.6f}")
        print(f"   P1 显著 {c['P1_significance']} · P2 效应 {c['P2_effect']} · "
              f"P3 内容 {c['P3_content']}  ⇒ 该格 {'PASS' if c['pass'] else '不 PASS'}")
    print(f"\n  完整 PASS 数 = {out['pass_count']}/5  过拟合格 = {out['overfit_only']}")
    print(f"  ★ 结局 = {out['outcome']} · 计入 SUCCEEDED = {out['counted']}")
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "stageA"
    if what == "stageA":
        g, ok = stage_a()
        json.dump({"guards": g, "ok": ok},
                  open(os.path.join(OUT, "DIR53r12_stageA.json"), "w"),
                  ensure_ascii=False, indent=1)
        sys.exit(0 if ok else 1)
    if what == "smoke":
        json.dump(smoke(), open(os.path.join(OUT, "DIR53r12_smoke.json"), "w"),
                  ensure_ascii=False, indent=1)
        sys.exit(0)
    if what == "formal":
        rest = [a for a in sys.argv[2:] if not a.startswith("--")]
        formal_arm(rest[0], rest[1])
        sys.exit(0)
    if what == "offcheck":
        pk0 = pk1_for("P1")
        _, om = offline_learn_pk1(pk0, TRAIN, steps=STEPS, lr=LR)
        print(json.dumps(om, ensure_ascii=False, indent=1))
        sys.exit(0)
    if what == "merge":
        merge()
        sys.exit(0)
    print("用法：stageA | smoke | formal --cell {P1..P5} --arm {FIX,OFF,ORACLE} | "
          "offcheck | merge")
    sys.exit(2)