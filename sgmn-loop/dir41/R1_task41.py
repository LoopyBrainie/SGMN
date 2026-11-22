#!/opt/pytorch/bin/python
r"""DIR-41 · **正式轮**实现：键映射的「**共享性**」是不是读侧可学性的**前提**？

卡：`revolve/DIR-41/DIRECTION_rev41_v10.md`（rev 1.0，**阈值与判据已冻结**；md5 由 `A41-3` 守卫）。

派生自 `dir40/R1_task40.py`（**逐字**：trace / 初值 / 训练协议 / 评估口径 / `train_arm40` / `eval_arm40` / `paired`）
⇒ ★ **装置文件 `dir31`、任务层 `dir33`、`dir37/38/39/40` 一律不改**（`A41-1`）。

★ **唯一的装置 revision**：层 1 的写键映射 `Pk1` 的**共享性**
  （共享格 = 覆盖成一张固定矩阵；逐实例格 PI = 保留装置自己的逐实例 `Pk1`）
  ⇒ ★ **只改「是否跨实例共享」，不改分布**（`A41-13` 的 n=400 分布闸门核验）。

★ **六格**：`SH1..SH5`（5 张预注册独立抽取的共享 `Pk1`）· `PI`（逐实例）⇒ **18 次臂训练**（6 格 × 3 臂）。
★ **六格共用同一训练段与同一评估段** ⇒ 跨格只差 `Pk1` 的**结构**。

用法：stageA | smoke | formal --cell {SH1..SH5,PI} --arm {oracle,rand,learned} | merge
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
import R1_task33 as R           # 任务层（逐字未改；import 时改写装置的有效常数）
import R1_grad as M             # 装置（逐字未改）
import R1_task37 as T37         # 测量层（_mk / 训练协议 / 迹线）
import R1_task38 as T38         # DIR-38 的 harness（跨实现锚）
import R1_task40 as T40         # ★ DIR-40 的测量层（**逐字复用**）

OUT = "/home/ec2-user/sgmn-loop/dir41/out"
D_H, D_V, D_K = R.D_H, R.D_V, R.D_K
SEED0 = M.SEED0

# ══════════ 冻结常量（卡 §3.2）
CARD_PATH = "/home/ec2-user/sgmn-loop/revolve/DIR-41/DIRECTION_rev41_v10.md"
CARD_MD5_EXPECTED = "7d45fa28f7e22b0ac569296aa3739758"
MD5_DEVICE = "8fa40f708600bec7e6c484727d2180c7"
MD5_TASK = "96ef89e87ec61556b52e0e464850087e"
MD5_DIR37 = "9236ac0490be9c413f1e8af163289207"
MD5_DIR38 = "d725da6d80bd072f9068a51857abf744"
MD5_DIR39 = "795563cb6212f228f8463ed418f8ce8c"
MD5_DIR40 = "baab3566891591944efb60a5b6dc9623"
N_REP = 16
N_TRAIN = 96
N_EVAL = 400
STEPS, LR = 250, 0.03
SIG_K = 3.0
FRAC_HEADROOM = 0.20
PASS_RATE_MIN = 4          # 5 张共享抽取中至少 4 张
CONTRAST_MIN = 4           # 对比检验至少 4/5 张图显著

# ══════════ 种子段（卡 §6）：全部为 `SEED0 + offset`
TRAIN_OFF = list(range(21000, 21000 + N_TRAIN))          # 21000..21095
EVAL_OFF = list(range(21500, 21500 + N_EVAL))            # 21500..21899
SMOKE_TRAIN_OFF = list(range(21100, 21104))              # 21100..21103
SMOKE_EVAL_OFF = list(range(21110, 21112))              # 21110..21111
TRAIN = [SEED0 + o for o in TRAIN_OFF]
EVAL = [SEED0 + o for o in EVAL_OFF]
SMOKE_TRAIN = [SEED0 + o for o in SMOKE_TRAIN_OFF]
SMOKE_EVAL = [SEED0 + o for o in SMOKE_EVAL_OFF]

# ══════════ 六格（卡 §2.3）
SHARED_PK1_SEEDS = {"SH1": 20262101, "SH2": 20262102, "SH3": 20262103,
                    "SH4": 20262104, "SH5": 20262105}
SHARED_CELLS = tuple(SHARED_PK1_SEEDS)
CELLS = SHARED_CELLS + ("PI",)
ARMS = ("oracle", "rand", "learned")
LEVELS = ("oracle", "rand", "learned", "learned_rc")


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ══════════ `Pk1` 的生成式（卡 §2.2：**与装置源码逐字同式**）
def shared_pk1_for(seed):
    r"""共享抽取：`randn(D_H, D_K, generator=g) * (1/sqrt(D_H)) * KEY_SCALE`。

    ★ **与装置源码逐字同式**（卡 §2.2）：
      装置 `dir31/R1_grad.py::build`：`rn(*shape, scale) = randn(*shape, generator=g) * scale`，
      `s_h = 1.0 / math.sqrt(D_H)`，`P["Pk1"] = rn(D_H, D_K, scale=s_h)`
      ⇒ 任务层 `dir33/R1_task33.py::build`：`P["Pk1"] = P["Pk1"] * KEY_SCALE`（`KEY_SCALE = 0.5`）
      ⇒ 有效 `Pk1 = randn(D_H, D_K) * s_h * KEY_SCALE`。
    ★ **乘法次序与装置一致**（先 `s_h` 后 `KEY_SCALE`）⇒ 逐位等于「用同式同种子」的结果。
    ★ **不得**为了与装置某一次实例抽样逐位相同而改公式：装置的 `Pk1` 是该实例生成器流的第 3 笔抽取，
      而共享抽取用的是独立生成器的第 1 笔 ⇒ 「公式相同」≠「逐位相同」，**边缘律相同**才是本卡需要的性质。
    """
    g = torch.Generator().manual_seed(int(seed))
    s_h = 1.0 / math.sqrt(D_H)
    return torch.randn(D_H, D_K, generator=g) * s_h * R.KEY_SCALE


def cell_pk1(cell):
    r"""格 → `Pk1`：`PI` ⇒ `None`（**保留装置自己的逐实例 `Pk1`**）；`SH*` ⇒ 该格的共享图。"""
    if cell == "PI":
        return None
    return shared_pk1_for(SHARED_PK1_SEEDS[cell])


# ══════════ 正式轮：单格单臂（多格多臂在 trainer 上**并行**跑；`merge` 汇总裁决）
def formal_arm(cell, arm):
    r"""跑一格的一臂 ⇒ 写 `DIR41r10_cell_{cell}_{arm}.json`（**只放读数，不放裁决**）。"""
    t0 = time.time()
    spk = cell_pk1(cell)
    qsrc, tctrl = {"oracle": ("oracle", False), "rand": ("rand", False),
                   "learned": ("learned", True)}[arm]
    print(f"[{cell}/{arm}] 训头{'(+控制器)' if tctrl else ''}（分臂联合训练，"
          f"{'共享 Pk1' if spk is not None else '逐实例 Pk1'}，n_rep={N_REP}）"
          f"n={len(TRAIN)} ...", flush=True)
    phi, meta = T40.train_arm40(qsrc, TRAIN, shared_pk=spk, train_ctrl=tctrl, steps=STEPS)
    print(f"[{cell}/{arm}] 训练完成 {time.time()-t0:.0f}s  梯度 {meta['gradnorms']}", flush=True)
    arms = {}
    arms[arm] = T40.eval_arm40(qsrc, phi, EVAL, shared_pk=spk)
    if arm == "learned":
        arms["learned_rc"] = T40.eval_arm40(qsrc, phi, EVAL, shared_pk=spk, rand_content=True)
    for k, a in arms.items():
        print(f"[{cell}/{arm}] 评估 {k:11s} L={a['mean_L']:.6f} (sd {a['sd_L']:.4f}) "
              f"cos={a['cos_q_k']:.4f} align={a['align_q_k']:.4f} |q|={a['q_norm']:.4f} "
              f"|y|={a['y_norm']:.4f} m̄={a['m_abs']:.4f} Wmax={a['Wmax']:.4f}", flush=True)
    out = {"cell": cell, "arm": arm, "n_train": len(TRAIN), "n_eval": len(EVAL),
           "train_off": [TRAIN_OFF[0], TRAIN_OFF[-1]], "eval_off": [EVAL_OFF[0], EVAL_OFF[-1]],
           "n_rep": N_REP, "steps": STEPS, "lr": LR,
           "pk1_mode": "device-own-per-instance" if spk is None else "shared",
           "pk1_seed": (SHARED_PK1_SEEDS.get(cell) if spk is not None else "device-own"),
           "arms": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_series")}
                    for k, v in arms.items()},
           "L_series": {k: v["L_series"] for k, v in arms.items()},
           "gradnorms": meta["gradnorms"],
           "dWq1_from_init": meta["dWq1_from_init"], "dbq1_from_init": meta["dbq1_from_init"],
           "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"DIR41r10_cell_{cell}_{arm}.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


# ══════════ Stage A（硬闸；**断言必须真的读被断言的对象** ⇒ 纪律 97）
def _row_cos_stats(mats):
    r"""一批 `Pk1` 矩阵 ⇒ 行对余弦的均值与 sd（行独立 ⇒ `sd(cos) = 1/√D_K`）。"""
    cs = []
    for P in mats:
        for i in range(D_H):
            for j in range(i + 1, D_H):
                a, b = P[i], P[j]
                na, nb = float(a.norm()), float(b.norm())
                if na > 1e-300 and nb > 1e-300:
                    cs.append(float(a @ b) / (na * nb))
    return statistics.fmean(cs), (statistics.stdev(cs) if len(cs) > 1 else 0.0), len(cs)


def stage_a():
    g = {}
    # ── A41-1：装置/任务层/上游脚本 md5 未变
    md5s = {
        "dir31/R1_grad.py": MD5_DEVICE, "dir33/R1_task33.py": MD5_TASK,
        "dir37/R1_task37.py": MD5_DIR37, "dir38/R1_task38.py": MD5_DIR38,
        "dir39/R1_task39.py": MD5_DIR39, "dir40/R1_task40.py": MD5_DIR40,
    }
    got = {p: _md5("/home/ec2-user/sgmn-loop/" + p) for p in md5s}
    g["A41-1_md5"] = got
    g["A41-1_ok"] = all(got[p] == v for p, v in md5s.items())

    # ── A41-2：有效常数实测（纪律 94）
    eff = {"D_H": M.D_H, "D_K": M.D_K, "D_V": M.D_V, "R": M.R, "T_STEPS": M.T_STEPS,
           "ETA": M.ETA, "LAM": M.LAM, "ALPHA": M.ALPHA, "GAMMA": M.GAMMA,
           "KEY_SCALE": R.KEY_SCALE, "INIT_SD": M.INIT_SD, "SEED0": M.SEED0}
    want = {"D_H": 8, "D_K": 16, "D_V": 4, "R": 2, "T_STEPS": 16, "ETA": 0.2,
            "LAM": 0.05, "ALPHA": 1.0, "GAMMA": 0.5, "KEY_SCALE": 0.5,
            "INIT_SD": 0.5, "SEED0": 20261111}
    g["A41-2_effective"] = eff
    g["A41-2_ok"] = all(eff[k] == want[k] for k in want)

    # ── A41-3：卡 md5 守卫
    g["A41-3_card_md5"] = _md5(CARD_PATH)
    g["A41-3_ok"] = g["A41-3_card_md5"] == CARD_MD5_EXPECTED

    # ── A41-4：**offset 空间**的种子段隔离
    #   ★ 修正 DIR-40 的 `A40-5`：它拿**绝对种子**去和 **offset 区间**求交 ⇒ 交集恒为空 ⇒ 空断言。
    hist = []
    for _k, (a, b) in list(T37.HIST_RANGES.items()) + list(T37.OWN_SEGMENTS.items()):
        hist.extend(range(a, b + 1))
    for a, b in [(12500, 13096), (13500, 13900), (14000, 14384), (14500, 14512),
                 (15000, 15148), (15500, 15900), (16000, 16192), (16400, 16412),
                 (17000, 17048), (17100, 17164), (17200, 17264),
                 (18000, 18191), (18500, 18899), (18900, 18911),   # DIR-40 正式轮
                 (20400, 20412), (20500, 20595), (20600, 20999)]:  # DIR-41 探针
        hist.extend(range(a, b + 1))
    hist_set = set(hist)
    g["A41-4_hist_n"] = len(hist_set)
    g["A41-4_train_hist"] = len(set(TRAIN_OFF) & hist_set)
    g["A41-4_eval_hist"] = len(set(EVAL_OFF) & hist_set)
    g["A41-4_smoke_hist"] = len(set(SMOKE_TRAIN_OFF + SMOKE_EVAL_OFF) & hist_set)
    g["A41-4_train_eval"] = len(set(TRAIN_OFF) & set(EVAL_OFF))
    g["A41-4_ok"] = (g["A41-4_train_hist"] == 0 and g["A41-4_eval_hist"] == 0
                     and g["A41-4_smoke_hist"] == 0 and g["A41-4_train_eval"] == 0)

    # ── A41-5 / A41-6：跨实现逐位锚（DIR-37 / DIR-38）
    phi_dev = T37.train_head("dev", SMOKE_TRAIN)
    errs = []
    for sd in SMOKE_EVAL:
        P = T37._mk(sd, phi_dev)
        Lref, _ = T37.trace(P, "dev", sd, phi_dev)
        P = T37._mk(sd, phi_dev)
        L40, _ = T40.trace40(P, 1, "oracle", sd, phi_dev)
        errs.append(abs(float(Lref) - float(L40)))
    g["A41-5_max_abs_diff"] = max(errs)
    g["A41-5_ok"] = (max(errs) == 0.0)

    errs38, w40, w38 = [], 0.0, 0.0
    for sd in SMOKE_EVAL:
        P = T38.T37._mk(sd, phi_dev)
        L38, dg38 = T38.trace38(P, 16, "grad", sd, phi_dev, want_diag=True)
        P = T38.T37._mk(sd, phi_dev)
        L40, dg40 = T40.trace40(P, 16, "oracle", sd, phi_dev, want_diag=True)
        errs38.append(abs(float(L38) - float(L40)))
        w40 = max(w40, dg40["Wmax"]); w38 = max(w38, dg38["Wmax"])
    g["A41-6_max_abs_diff"] = max(errs38)
    g["A41-6_Wmax_41"] = w40
    g["A41-6_Wmax_38"] = w38
    g["A41-6_ok"] = (max(errs38) == 0.0 and w40 == w38)

    # ── A41-7：真实单变量断言（`c_eq_h`，逐位）
    # ── A41-8：指纹**设计关系**（逐对，纪律 99；**训练后**比）
    # ── A41-9：快照基准的冻结/训动（纪律 98）
    cch, qfp, diag = {}, {}, {}
    for cell in ("SH1", "PI"):
        spk = cell_pk1(cell)
        phi_r, meta_r = T40.train_arm40("rand", SMOKE_TRAIN, shared_pk=spk,
                                        train_ctrl=False, steps=5)
        phi_l, meta_l = T40.train_arm40("learned", SMOKE_TRAIN, shared_pk=spk,
                                        train_ctrl=True, steps=5)
        for lv, (qq, pp) in (("oracle", ("oracle", phi_dev)),
                             ("rand", ("rand", phi_r)),
                             ("learned", ("learned", phi_l)),
                             ("learned_rc", ("learned", phi_l))):
            P = T37._mk(SMOKE_EVAL[0], pp)
            _L, dg = T40.trace40(P, N_REP, qq, SMOKE_EVAL[0], pp, want_diag=True,
                                 shared_pk1=spk, rand_content=(lv == "learned_rc"))
            cch.setdefault(cell, {})[lv] = dg["c_eq_h"]
            qfp.setdefault(cell, {})[lv] = dg["q_fp"]
            diag.setdefault(cell, {})[lv] = dg["Wmax"]
        g.setdefault("A41-9_meta", {})[cell] = {
            "rand_dWq1": meta_r["dWq1_from_init"], "rand_dbq1": meta_r["dbq1_from_init"],
            "learned_dWq1": meta_l["dWq1_from_init"], "learned_dbq1": meta_l["dbq1_from_init"],
            "gradnorm_Wq1": meta_l["gradnorms"].get("Wq1", 0.0),
            "gradnorm_bq1": meta_l["gradnorms"].get("bq1", 0.0)}
    g["A41-7_c_eq_h"] = cch
    g["A41-7_ok"] = all(v == 0.0 for d in cch.values() for v in d.values())

    g["A41-8_q_fp"] = qfp
    fp_ok = True
    for cell, d in qfp.items():
        # ★ 硬断言（纪律 99）：`learned == learned_rc`（**同一控制器** ⇒ 指纹必然相同）
        #   与 `rand != learned`（**训练后**的控制器不同）
        if not (d["learned"] == d["learned_rc"] and d["rand"] != d["learned"]):
            fp_ok = False
    # ★ 诊断（**不设闸**）：`oracle != rand`、`oracle != learned`、跨格 `learned` 两两不等
    diag_fp = {}
    for cell, d in qfp.items():
        diag_fp[cell] = {"oracle_ne_rand": d["oracle"] != d["rand"],
                         "oracle_ne_learned": d["oracle"] != d["learned"],
                         "rand_ne_learned": d["rand"] != d["learned"],
                         "learned_eq_learned_rc": d["learned"] == d["learned_rc"]}
    g["A41-8_diag"] = diag_fp
    g["A41-8_ok"] = fp_ok

    g["A41-9_meta"] = g["A41-9_meta"]
    mv_ok = True
    for cell, m in g["A41-9_meta"].items():
        if not (m["rand_dWq1"] == 0.0 and m["rand_dbq1"] == 0.0
                and m["learned_dWq1"] > 0.0 and m["gradnorm_Wq1"] > 0.0
                and math.isfinite(m["gradnorm_Wq1"])):
            mv_ok = False
    g["A41-9_ok"] = mv_ok

    # ── A41-10：内容置换生效
    P = T37._mk(SMOKE_EVAL[0], phi_dev)
    W = M._W(P["U1"], P["V1"]).detach()
    Wp = R.read_W(W, "S", SMOKE_EVAL[0])
    g["A41-10_dW_perm"] = float((Wp - W).norm())
    g["A41-10_ok"] = g["A41-10_dW_perm"] > 0.0

    # ── A41-11：`Pk1` 结构（共享格跨实例逐位相同；跨格逐位不同；PI 跨实例逐位不同）
    #   ★ DEF-41-1（run1 拦下）：`pi_diff` 原来对**列表自身**求差（含 `pks[0]` 与自己比 ⇒ 恒为 0）
    #     ⇒ 断言恒假。修正为只比 `pi_pks[1:]`；**断言内容（卡 §7）一字未改**。
    #   ★ 另：结构检查改用 4 个实例（原为 2 个 ⇒ 只有 1 次跨实例比较，过于单薄）。
    STRUCT_SEEDS = EVAL[:4]
    struct = {}
    shared_same, pi_diff = True, True
    firsts = {}
    for cell in SHARED_CELLS:
        spk = cell_pk1(cell)
        pks = []
        for sd in STRUCT_SEEDS:
            P = T37._mk(sd, phi_dev)
            T40.trace40(P, 0, "oracle", sd, phi_dev, shared_pk1=spk)
            pks.append(P["Pk1"])
        same = all(float((p - pks[0]).abs().max()) == 0.0 for p in pks)
        shared_same = shared_same and same
        struct[cell] = {"shared_identical": bool(same), "fro": float(spk.norm()),
                        "n_instances": len(pks)}
        firsts[cell] = pks[0]
    pi_pks = []
    for sd in STRUCT_SEEDS:
        P = T37._mk(sd, phi_dev)
        pi_pks.append(P["Pk1"])
    pi_diff = all(float((p - pi_pks[0]).abs().max()) > 0.0 for p in pi_pks[1:])
    g["A41-11_pi_n_instances"] = len(pi_pks)
    across = {}
    for a in SHARED_CELLS:
        for b in SHARED_CELLS:
            if a < b:
                across[f"{a}|{b}"] = bool(float((firsts[a] - firsts[b]).abs().max()) > 0.0)
    g["A41-11_shared"] = struct
    g["A41-11_pi_instances_differ"] = bool(pi_diff)
    g["A41-11_across_cells_differ"] = across
    g["A41-11_ok"] = bool(shared_same and pi_diff and all(across.values()))

    # ── A41-12：初值确定（跨臂跨格同源）
    i1, i2 = T40.init_params(), T40.init_params()
    g["A41-12_init_deterministic"] = all(
        float((i1[k] - i2[k]).abs().max()) == 0.0 for k in i1)
    g["A41-12_ok"] = g["A41-12_init_deterministic"]

    # ── A41-13：`Pk1` 生成式的**分布核对**（n = 400；**只用理论边界**）
    dev_mats = []
    for sd in [SEED0 + 30000 + i for i in range(400)]:
        P = T37._mk(sd, phi_dev)
        dev_mats.append(P["Pk1"])
    sh_mats = []
    for base in SHARED_PK1_SEEDS.values():
        for k in range(80):
            sh_mats.append(shared_pk1_for(base + 1000 * k))
    dev_e = torch.cat([m.reshape(-1) for m in dev_mats]).tolist()
    sh_e = torch.cat([m.reshape(-1) for m in sh_mats]).tolist()
    dev_mean, sh_mean = statistics.fmean(dev_e), statistics.fmean(sh_e)
    dev_sd, sh_sd = statistics.stdev(dev_e), statistics.stdev(sh_e)
    dev_f = [float(m.norm()) for m in dev_mats]
    sh_f = [float(m.norm()) for m in sh_mats]
    dev_fm, sh_fm = statistics.fmean(dev_f), statistics.fmean(sh_f)
    dc, dcs, dn = _row_cos_stats(dev_mats)
    sc, scs, sn = _row_cos_stats(sh_mats)
    def _q(v, p):
        v = sorted(v)
        k = (len(v) - 1) * p
        f = int(math.floor(k)); c = min(f + 1, len(v) - 1)
        return v[f] + (v[c] - v[f]) * (k - f)
    def _ratio(a, b):
        return (a / b) if abs(b) > 1e-300 else float("inf")
    q05_dev, q05_sh = _q(dev_e, 0.05), _q(sh_e, 0.05)
    q95_dev, q95_sh = _q(dev_e, 0.95), _q(sh_e, 0.95)
    g["A41-13_n"] = {"dev_mats": len(dev_mats), "sh_mats": len(sh_mats),
                     "dev_entries": len(dev_e), "sh_entries": len(sh_e),
                     "row_cos_pairs": [dn, sn]}
    g["A41-13_mean"] = {"dev": dev_mean, "shared": sh_mean,
                        "abs_diff": abs(sh_mean - dev_mean)}
    g["A41-13_sd"] = {"dev": dev_sd, "shared": sh_sd, "ratio": _ratio(sh_sd, dev_sd)}
    g["A41-13_frob"] = {"dev": dev_fm, "shared": sh_fm, "ratio": _ratio(sh_fm, dev_fm)}
    g["A41-13_cos"] = {"dev_mean": dc, "shared_mean": sc,
                       "abs_diff": abs(sc - dc), "dev_sd": dcs, "shared_sd": scs,
                       "sd_ratio": _ratio(scs, dcs)}
    g["A41-13_q05"] = {"dev": q05_dev, "shared": q05_sh, "ratio": _ratio(q05_sh, q05_dev)}
    g["A41-13_q95"] = {"dev": q95_dev, "shared": q95_sh, "ratio": _ratio(q95_sh, q95_dev)}
    g["A41-13_ok"] = (g["A41-13_mean"]["abs_diff"] < 0.01
                      and 0.95 <= g["A41-13_sd"]["ratio"] <= 1.05
                      and 0.95 <= g["A41-13_frob"]["ratio"] <= 1.05
                      and g["A41-13_cos"]["abs_diff"] < 0.02
                      and 0.95 <= g["A41-13_cos"]["sd_ratio"] <= 1.05
                      and 0.95 <= g["A41-13_q05"]["ratio"] <= 1.05
                      and 0.95 <= g["A41-13_q95"]["ratio"] <= 1.05)

    # ── A41-14：诊断读数（**不设硬闸**）
    g["A41-14_Wmax_diag"] = diag

    ok = all(v for k, v in g.items() if k.endswith("_ok"))
    print("Stage A：" + ("★ 全部通过" if ok else "★ 未通过"))
    for k in sorted(g):
        print(f"  {k:26s} {g[k]}")
    return g, ok


# ══════════ smoke（证明协议可执行；**永不产生判据裁决**）
def smoke():
    out = {"note": "smoke：只证明可执行，不给判据"}
    for cell in CELLS:
        spk = cell_pk1(cell)
        for arm in ARMS:
            qsrc, tctrl = {"oracle": ("oracle", False), "rand": ("rand", False),
                           "learned": ("learned", True)}[arm]
            phi, meta = T40.train_arm40(qsrc, SMOKE_TRAIN, shared_pk=spk,
                                        train_ctrl=tctrl, steps=3)
            e = T40.eval_arm40(qsrc, phi, SMOKE_EVAL, shared_pk=spk)
            out[f"{cell}/{arm}"] = {"mean_L": e["mean_L"], "cos_q_k": e["cos_q_k"],
                                    "Wmax": e["Wmax"], "c_eq_h_max": e["c_eq_h_max"],
                                    "dWq1_from_init": meta["dWq1_from_init"]}
            print(f"  {cell}/{arm:8s} L={e['mean_L']:.6f} cos={e['cos_q_k']:.4f} "
                  f"Wmax={e['Wmax']:.4f} c_eq_h={e['c_eq_h_max']:.1e} "
                  f"|Wq-init|={meta['dWq1_from_init']:.3e}", flush=True)
    print("smoke 完成（**无判据裁决**）")
    return out


# ══════════ 汇总裁决（卡 §3/§4，**冻结**）
def merge():
    r"""读 18 个 `DIR41r10_cell_*.json` ⇒ 按**冻结**判定表裁决。★ **不读探针文件**。"""
    arms, series, metas, missing = {}, {}, {}, []
    for cell in CELLS:
        for arm in ARMS:
            p = os.path.join(OUT, f"DIR41r10_cell_{cell}_{arm}.json")
            if not os.path.exists(p):
                missing.append(f"{cell}/{arm}")
                continue
            d = json.load(open(p, encoding="utf-8"))
            metas[f"{cell}/{arm}"] = d
            for k, v in d["arms"].items():
                arms[(cell, k)] = v
                series[(cell, k)] = d["L_series"][k]
    out = {"n_train": N_TRAIN, "n_eval": N_EVAL, "n_rep": N_REP, "steps": STEPS,
           "sig_k": SIG_K, "frac_headroom": FRAC_HEADROOM,
           "pass_rate_min": PASS_RATE_MIN, "contrast_min": CONTRAST_MIN,
           "cells": {}, "missing": missing}

    # ── 第 1 支：缺臂 / n 不符
    bad_n = []
    for cell in CELLS:
        for arm in ARMS:
            d = metas.get(f"{cell}/{arm}")
            if d and (d.get("n_train") != N_TRAIN or d.get("n_eval") != N_EVAL
                      or d.get("n_rep") != N_REP or d.get("steps") != STEPS):
                bad_n.append(f"{cell}/{arm}")
    out["bad_n"] = bad_n

    # ── 逐格计算（P1/P2/P3）
    for cell in CELLS:
        if any((cell, lv) not in arms for lv in LEVELS):
            continue
        Lo = arms[(cell, "oracle")]["mean_L"]
        Lr = arms[(cell, "rand")]["mean_L"]
        Ll = arms[(cell, "learned")]["mean_L"]
        Lrc = arms[(cell, "learned_rc")]["mean_L"]
        Rj = Lr - Lo
        pd = T40.paired(series[(cell, "rand")], series[(cell, "learned")])
        pc = T40.paired(series[(cell, "learned_rc")], series[(cell, "learned")])
        delta = Lr - Ll
        floor = FRAC_HEADROOM * Rj
        p1 = delta >= SIG_K * pd["se"]
        p2 = delta >= floor
        p3 = pc["mean"] >= SIG_K * pc["se"]
        out["cells"][cell] = {
            "L_oracle": Lo, "L_rand": Lr, "L_learned": Ll, "L_learned_rc": Lrc,
            "R": Rj, "floor_0.20R": floor, "delta": delta,
            "paired_delta": pd, "paired_content": pc,
            "P1_significance": bool(p1), "P2_effect": bool(p2), "P3_content": bool(p3),
            "pass": bool(p1 and p2 and p3),
            "cos": {lv: arms[(cell, lv)]["cos_q_k"] for lv in LEVELS},
            "Wmax": {lv: arms[(cell, lv)]["Wmax"] for lv in LEVELS},
        }

    # ── 对比检验（卡 §3.4：**逐图**，逐实例配对）
    contrast = {}
    if ("PI", "rand") in series and ("PI", "learned") in series:
        d_pi = [a - b for a, b in zip(series[("PI", "rand")], series[("PI", "learned")])]
        for cell in SHARED_CELLS:
            if (cell, "rand") not in series or (cell, "learned") not in series:
                continue
            d_j = [a - b for a, b in zip(series[(cell, "rand")], series[(cell, "learned")])]
            c_j = [x - y for x, y in zip(d_j, d_pi)]
            st = T40.paired(c_j, [0.0] * len(c_j))
            contrast[cell] = {"mean": st["mean"], "sd": st["sd"], "se": st["se"],
                              "3se": st["3se"], "n": st["n"],
                              "pass": bool(st["mean"] >= st["3se"])}
    out["contrast"] = contrast

    shared_pass = [c for c in SHARED_CELLS if out["cells"].get(c, {}).get("pass")]
    out["shared_pass_count"] = len(shared_pass)
    out["shared_pass_cells"] = shared_pass
    pi = out["cells"].get("PI", {})
    out["pi_present"] = bool(pi)
    out["pi_full_pass"] = bool(pi.get("pass"))
    out["pi_sig_and_content"] = bool(pi.get("P1_significance") and pi.get("P3_content"))
    out["contrast_pass_count"] = sum(1 for v in contrast.values() if v["pass"])

    # ── 有限性 / 发散
    finite = all(math.isfinite(a["mean_L"]) for a in arms.values())
    ses = []
    for c in out["cells"].values():
        ses += [c["paired_delta"]["se"], c["paired_content"]["se"]]
    for v in contrast.values():
        ses += [v["se"]]
    finite = finite and all(math.isfinite(s) for s in ses)

    # ── 冻结判定表（**按序命中**）
    if missing or bad_n:
        out["outcome"] = "INCOMPLETE"
        out["counted"] = False
        out["note"] = "缺臂或 n 不符 ⇒ 整轮作废"
    elif not finite:
        out["outcome"] = "DIVERGED"
        out["counted"] = False
        out["note"] = "存在非有限读数 ⇒ 整轮作废"
    elif any(c["R"] <= 0 for c in out["cells"].values()):
        out["outcome"] = "INDETERMINATE"
        out["counted"] = False
        out["note"] = "任一格 R ≤ 0 ⇒ 「oracle 是上限」的前提不成立 ⇒ 不可判定（**不是失败**）"
    elif out["pi_sig_and_content"]:
        out["outcome"] = "NOT-NECESSARY"
        out["counted"] = False
        out["note"] = "PI 同时满足显著性与内容性 ⇒ 共享性**不是**前提 ⇒ H41 被否证"
    elif out["shared_pass_count"] < PASS_RATE_MIN:
        out["outcome"] = "SHARED-NOT-REPLICATED"
        out["counted"] = False
        out["note"] = "共享格完整 PASS 数不足 4/5 ⇒ **不得**写成「共享性不是前提」"
    elif out["contrast_pass_count"] < CONTRAST_MIN:
        out["outcome"] = "CONTRAST-FAIL"
        out["counted"] = False
        out["note"] = "对比检验通过数不足 4/5 ⇒ 共享效应未显著大于逐实例效应"
    else:
        out["outcome"] = "PASS"
        out["counted"] = True
        out["note"] = "5 张预注册独立抽取中至少 4 张通过，且 PI 未出现同类优势，且对比检验 ≥4/5 ⇒ H41 确认"
    out["gradnorms"] = {k: v["gradnorms"] for k, v in metas.items()}
    out["dWq1_from_init"] = {k: v["dWq1_from_init"] for k, v in metas.items()}
    out["secs"] = {k: v["secs"] for k, v in metas.items()}
    out["pk1_seed"] = dict(SHARED_PK1_SEEDS)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "DIR41r10_formal.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n══ DIR-41 正式轮（汇总裁决） ══")
    for cell in CELLS:
        c = out["cells"].get(cell)
        if not c:
            print(f"  [{cell}] 缺读数")
            continue
        print(f"\n── 格 {cell} ──")
        for lv in LEVELS:
            print(f"   L[{lv:11s}] = {c['L_' + lv]:12.6f}   cos = {c['cos'][lv]:+.4f}")
        print(f"   R = {c['R']:.6f}   地板 0.20R = {c['floor_0.20R']:.6f}   "
              f"Δ = {c['delta']:.6f}   3se = {c['paired_delta']['3se']:.6f}")
        print(f"   P1 显著 {c['P1_significance']} · P2 效应 {c['P2_effect']} · "
              f"P3 内容 {c['P3_content']}  ⇒ 该格 {'PASS' if c['pass'] else '不 PASS'}")
    if contrast:
        print("\n── 对比检验（逐图） ──")
        for cell, v in contrast.items():
            print(f"   {cell}: mean={v['mean']:.6f} 3se={v['3se']:.6f} ⇒ "
                  f"{'通过' if v['pass'] else '不通过'}")
    print(f"\n  共享格通过数 = {out['shared_pass_count']}/5   "
          f"对比检验通过数 = {out['contrast_pass_count']}/5")
    if pi:
        print(f"  PI: Δ={pi['delta']:.6f} 3se={pi['paired_delta']['3se']:.6f} "
              f"地板={pi['floor_0.20R']:.6f} 内容差={pi['paired_content']['mean']:.6f} "
              f"3se={pi['paired_content']['3se']:.6f} ⇒ 完整 PASS={pi['pass']}")
    print(f"  ★ 结局 = {out['outcome']} · 计入 SUCCEEDED = {out['counted']}")
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "stageA"
    if what == "stageA":
        g, ok = stage_a()
        json.dump({"guards": g, "ok": ok},
                  open(os.path.join(OUT, "DIR41r10_stageA.json"), "w"), ensure_ascii=False, indent=1)
        sys.exit(0 if ok else 1)
    if what == "smoke":
        json.dump(smoke(), open(os.path.join(OUT, "DIR41r10_smoke.json"), "w"),
                  ensure_ascii=False, indent=1)
        sys.exit(0)
    if what == "formal":
        rest = [a for a in sys.argv[2:] if not a.startswith("--")]
        cell, arm = rest[0], rest[1]
        formal_arm(cell, arm)
        sys.exit(0)
    if what == "merge":
        merge()
        sys.exit(0)
    print("用法：stageA | smoke | formal --cell {SH1..SH5,PI} --arm {oracle,rand,learned} | merge")
    sys.exit(2)
