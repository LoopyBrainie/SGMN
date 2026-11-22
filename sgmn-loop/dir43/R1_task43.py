#!/opt/pytorch/bin/python
r"""DIR-43 · **正式轮**实现：任务梯度的**下降方向**是否**指向寻址**？

卡：`revolve/DIR-43/DIRECTION_rev43_v10.md`（rev 1.0，**阈值与判据已冻结**；md5 由 `A43-3` 守卫）。

派生自 `dir42/R1_task42.py` —— ★ **直接 `import` 复用** `R1_task42.trace42` / `R1_task42.reader_init` /
`R1_task42.eval_reader42`（**不做二次派生** ⇒ 与 DIR-42 **位级同源**，DEF-43-7）
⇒ ★ 装置文件 `dir31`、任务层 `dir33`、`dir37/38/39/40/41/42` 一律不改（`A43-1`）。

★ **唯一的装置 revision**：**无**（沿用 DIR-42 的逐实例 `Pk1` × 每实例读器）。
★ **唯一新增 = 测量层**：
  ① `ρ(t) := cos(g_desc(t), g_addr)`，`g_desc = −∂L_train/∂Wq1`、`g_addr = ∂A/∂Wq1`（卡 §0.1）；
  ② `ρ_rand(t)`（随机正交键的同一对照）；
  ③ `direct` 臂（只优化对齐目标 `−A`，判据**非空**用）。

★ **符号约定（卡 §0.1，DEF-43-1）**：`ρ > 0` ⇒ 沿任务梯度的**下降方向**走，对齐变好 ⇒「指向寻址」；
  等价有限步判据：`sign(ΔL 沿 +ĝ_addr) = −sign(ρ)`（`A43-13` 真跑自检）。

用法：stageA | smoke | formal --arm {learned,rand,direct} --shard i --nshards 4 | merge
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
ROOT = "/home/ec2-user/sgmn-loop"
sys.path.insert(0, f"{ROOT}/dir42")
sys.path.insert(0, f"{ROOT}/dir31")
sys.path.insert(0, f"{ROOT}/dir33")
sys.path.insert(0, f"{ROOT}/dir37")
import R1_task42 as T42          # ★ 逐字复用：trace42 / reader_init / eval_reader42 / T37 / 训练协议
import R1_task33 as R            # 任务层（逐字未改）
import R1_grad as M              # 装置（逐字未改）

OUT = f"{ROOT}/dir43/out"
D_H, D_V, D_K, T = R.D_H, R.D_V, R.D_K, R.T
SEED0 = M.SEED0
T37 = T42.T37

# ══════════ 冻结常量（卡 §3.2）
CARD_PATH = f"{ROOT}/revolve/DIR-43/DIRECTION_rev43_v10.md"
CARD_MD5_EXPECTED = "9a9a8998a0dff9c3645fb6dc0ec981a1"
MD5_DEVICE = "8fa40f708600bec7e6c484727d2180c7"
MD5_TASK = "96ef89e87ec61556b52e0e464850087e"
MD5_DIR37 = "9236ac0490be9c413f1e8af163289207"
MD5_DIR38 = "d725da6d80bd072f9068a51857abf744"
MD5_DIR39 = "795563cb6212f228f8463ed418f8ce8c"
MD5_DIR40 = "baab3566891591944efb60a5b6dc9623"
MD5_DIR41 = "af8f001fe4237b998830a430f9293379"
MD5_DIR42 = "b500304b74928a791577e71e4e88eeeb"
N_REP = T42.N_REP
N_INST = 400                 # ★ 判据臂（learned/direct）的实例数（与 DIR-42 同）
N_CTRL = 100                 # ★ rand 对照臂（诊断，不进判据）
STEPS, LR = T42.STEPS, T42.LR
SIG_K = 3.0
COS_DIRECT = 0.9             # direct 臂的判据-非空阈
READER_SEED_OFFSET = 10_000_000
RAND_KEY_SEED_OFFSET = 55_555_555
ROW_SCHEMA = 1

# ══════════ 种子段（卡 §6）
INST_OFF = list(range(22500, 22500 + N_INST))      # 22500..22899（与 DIR-42 正式轮逐位相同）
CTRL_OFF = list(range(22500, 22500 + N_CTRL))      # 22500..22599（判据段的前 100，诊断对照）
SMOKE_INST_OFF = list(range(23000, 23004))         # 23000..23003
PROBE_INST_OFF = list(range(22000, 22064))         # 22000..22063（探针；与判据段不相交）
TRAIN_WIN = T42.TRAIN_WIN                          # (1, 8)
EVAL_WIN = T42.EVAL_WIN                            # (9, 15)
RHO_GRID = (0, 1, 2, 5, 10, 25, 50, 100, 250)      # ρ(t) 的诊断采样点
ARMS = ("learned", "rand", "direct")
N_EXPECT = {"learned": N_INST, "rand": N_CTRL, "direct": N_INST}


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def _card_guard(stage):
    r"""★ **冻结守卫**（**纵深防御**）：任何**能产出判据读数**的命令都必须过这道闸。

    - `CARD_MD5_EXPECTED` 仍是占位符 ⇒ 卡片**尚未冻结** ⇒ 拒绝执行；
    - 卡的实测 md5 与冻结值不符 ⇒ 卡片**在冻结后被改过** ⇒ 拒绝执行。
    ⇒ ★ 顺序：回填本卡 ⇒ **算卡 md5 并写进 `CARD_MD5_EXPECTED`** ⇒ 冻结 ⇒ 才可跑。
    """
    if CARD_MD5_EXPECTED == "__CARD_MD5__":
        raise SystemExit(f"★ [{stage}] 卡片尚未冻结（CARD_MD5_EXPECTED 仍是占位符）⇒ 拒绝执行")
    got = _md5(CARD_PATH)
    if got != CARD_MD5_EXPECTED:
        raise SystemExit(f"★ [{stage}] 卡片 md5 不符：冻结值 {CARD_MD5_EXPECTED} ≠ 实测 {got}"
                         f" ⇒ 拒绝执行（卡在冻结后被改过）")
    return got


# ══════════ 测量层（★ 唯一新增）：`q*`、`g_addr`、随机正交键
def ctx_of(P):
    r"""层-1 的**读上下文** `c(t) = tanh(a[t] @ A1 + b1)`（**与读器无关** ⇒ 是常量）。"""
    return [torch.tanh(P["a"][t] @ P["A1"] + P["b1"]) for t in range(T)]


def qstar_of(P, key=None):
    r"""**目标查询** `q*(t) = tanh(c(t) @ Pk1)`（`Pk1` 已是缩放后的装置量）。"""
    K = P["Pk1"] if key is None else key
    c = ctx_of(P)
    return [torch.tanh(c[t] @ K) for t in range(T)]


def align_grad(P, phi, W=None, b=None, key=None):
    r"""`g_addr := ∂A/∂Wq1`，`A := mean_{t=1..T-1} cos(q(t), q*(t))`，`q*` 视作**常量**。

    ★ `c` 与 `q*` 都是**常量** ⇒ ⇒ `A` 只是 `(Wq1, bq1)` 的函数 ⇒ **无需轨迹** ⇒ 极便宜。
    ★ 与 `trace42` 的诊断 `cos_q_k` **同口径**：`trace42` 配对 `q1s[t]` 与 `k1s[t-1]`，
      而 `k1s[t-1] = tanh(c(t-2) @ Pk1)` 且 `c` 周期 2 ⇒ ⇒ `k1s[t-1] ≡ q*(t)`（`A43-6` 实测守恒）。
    ★ 用**独立叶子**（`detach().clone()`）⇒ **不污染** `phi["Wq1"].grad`（`A43-5` 的位级锚靠这条）。
    """
    c, qs = ctx_of(P), qstar_of(P, key)
    Wd = phi["Wq1"].detach().clone() if W is None else W
    bd = phi["bq1"].detach().clone() if b is None else b
    Wl = Wd.clone().requires_grad_(True)
    cs = [torch.nn.functional.cosine_similarity(torch.tanh(c[t] @ Wl + bd), qs[t], dim=0)
          for t in range(1, T)]
    A = torch.stack(cs).mean()
    gW = torch.autograd.grad(A, [Wl])[0]
    return float(A.detach()), gW.detach().clone()


def rand_key(P, sd):
    r"""**特异性对照**用的「错误键」：`Pk_rand = Pk1 @ Q`（`Q` 正交随机）⇒ 保持尺度/谱，
    但与「正确键」无关 ⇒ ⇒ 公平对照。★ 种子流 = 实例种子 + `RAND_KEY_SEED_OFFSET`（卡 §6）。"""
    g = torch.Generator().manual_seed(int(sd) + RAND_KEY_SEED_OFFSET)
    A = torch.randn(D_K, D_K, generator=g, dtype=torch.float64)
    Q, Rr = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(Rr)).unsqueeze(0)
    return P["Pk1"] @ Q


def cos2(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float((a * b).sum()) / (na * nb) if na > 1e-300 and nb > 1e-300 else float("nan")


# ══════════ 训练：与 DIR-42 **位级同源**（`trace42` 直接复用），只加 ρ 采样
def train_reader43(qsrc, sd, steps=STEPS, train_win=TRAIN_WIN, lr=LR):
    r"""**每实例**读器训练（协议 **逐字 = DIR-42**）＋ 在 `RHO_GRID` 上采 `ρ(t)`。

    ★ 与 `R1_task42.train_reader42` 的**唯一差异**：
      ① 显式把 `phi["Wq1"].grad` / `phi["bq1"].grad` 每步清成 `None`
         （`learned` 臂里二者**在**优化器内 ⇒ `opt.zero_grad()` 已清 ⇒ **无影响**；
          `rand` 臂里二者**不在**优化器内 ⇒ 否则梯度会**跨步累加** ⇒ 必须显式清）；
      ② 在网格步**额外**存一份 `g_task` 与当时的 `(Wq1, bq1)`（**不**改变轨迹）。
    ⇒ ★ **`learned` 臂的终态与 DIR-42 逐位相同**（`A43-5` 实测）。
    """
    phi = T42.reader_init(sd)
    snap = {k: v.detach().clone() for k, v in phi.items()}
    names = ["g1", "Wo1", "bo1"] + (["Wq1", "bq1"] if qsrc == "learned" else [])
    opt = torch.optim.Adam([phi[n] for n in names], lr=lr)
    gn = {}
    grid = {}                      # step -> (g_task, Wq1, bq1)
    for step in range(steps):
        opt.zero_grad()
        phi["Wq1"].grad = None     # ★ 显式清（见 docstring ①）
        phi["bq1"].grad = None
        P = T37._mk(sd, phi, train=True)
        L, _ = T42.trace42(P, N_REP, qsrc, sd, phi, win=train_win)
        L.backward()
        if phi["Wq1"].grad is not None and step in RHO_GRID:
            grid[step] = (phi["Wq1"].grad.detach().clone(),
                          phi["Wq1"].detach().clone(),
                          phi["bq1"].detach().clone())
        for n in names:
            if phi[n].grad is not None:
                gn[n] = max(gn.get(n, 0.0), float(phi[n].grad.norm()))
        opt.step()

    # ── 终态（`ρ(STEPS)`）：在**训练完成后**的控制器上再取一次任务梯度
    opt.zero_grad()
    phi["Wq1"].grad = None
    phi["bq1"].grad = None
    P = T37._mk(sd, phi, train=True)
    L, _ = T42.trace42(P, N_REP, qsrc, sd, phi, win=train_win)
    L.backward()
    g_task = (phi["Wq1"].grad.detach().clone() if phi["Wq1"].grad is not None
              else torch.zeros_like(phi["Wq1"]))
    g_task_b = (phi["bq1"].grad.detach().clone() if phi["bq1"].grad is not None
                else torch.zeros_like(phi["bq1"]))
    grid[steps] = (g_task, phi["Wq1"].detach().clone(), phi["bq1"].detach().clone())

    # ── 由（常量）`q*` 与（存下的）`g_task` 算 ρ：`g_desc = −g_task`
    P0 = T37._mk(sd, None)         # 设备（`c`/`q*` 只依赖种子 ⇒ 与读器无关）
    krand = rand_key(P0, sd)
    rho_c, rhor_c = {}, {}
    for step, (gt, W, b) in sorted(grid.items()):
        _, g_addr = align_grad(P0, phi, W=W, b=b)
        _, g_addr_r = align_grad(P0, phi, W=W, b=b, key=krand)
        rho_c[str(step)] = cos2(-gt, g_addr)      # ★ 下降方向 vs 寻址方向
        rhor_c[str(step)] = cos2(-gt, g_addr_r)

    out = {k: v.detach().clone() for k, v in phi.items()}
    meta = {"gradnorms": gn,
            "dWq1_from_init": float((out["Wq1"] - snap["Wq1"]).norm()),
            "dbq1_from_init": float((out["bq1"] - snap["bq1"]).norm()),
            "rho_curve": rho_c, "rho_rand_curve": rhor_c,
            "rho_final": rho_c[str(steps)], "rho0": rho_c.get("0"),
            "rho_rand_final": rhor_c[str(steps)], "rho_rand0": rhor_c.get("0"),
            "gn_task_final": float(g_task.norm())}
    return out, meta


def direct43(sd, steps=STEPS, lr=LR):
    r"""**正控**（`direct` 臂）：**只**优化对齐目标 `−A`（只动 `Wq1,bq1`，**不走** `L`）。
    ⇒ `cos_end` := 优化完后 `min_{t=1..T-1} cos(q(t), q*(t))`（**逐实例的最差时刻**）。"""
    P = T37._mk(sd, None)
    phi = T42.reader_init(sd)
    c, qs = ctx_of(P), qstar_of(P)
    W = phi["Wq1"].detach().clone().requires_grad_(True)
    b = phi["bq1"].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        cs = [torch.nn.functional.cosine_similarity(torch.tanh(c[t] @ W + b), qs[t], dim=0)
              for t in range(1, T)]
        A = torch.stack(cs).mean()
        (-A).backward()
        opt.step()
    with torch.no_grad():
        cs = [float(torch.nn.functional.cosine_similarity(torch.tanh(c[t] @ W + b), qs[t], dim=0))
              for t in range(1, T)]
    return {"direct_cos_mean": statistics.fmean(cs), "direct_cos_min": min(cs),
            "direct_cos_max": max(cs),
            "direct_Wdist_Pk1": float((W.detach() - P["Pk1"]).norm())}


# ══════════ 统计（★ 非有限不抛异常 ⇒ 由 `merge` 判 `DIVERGED`，DEF-42-2 的教训）
def _fmean_safe(xs):
    xs = list(xs)
    if not xs or not all(math.isfinite(x) for x in xs):
        return float("nan")
    return statistics.fmean(xs)


def _stats(xs):
    """均值/sd/se/3se（★ 非有限 ⇒ `se` 非有限 ⇒ 不抛异常）。"""
    xs = list(xs)
    if not xs or not all(math.isfinite(x) for x in xs):
        return {"mean": float("nan"), "sd": float("nan"), "se": float("nan"),
                "3se": float("nan"), "n": len(xs)}
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    se = sd / len(xs) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": SIG_K * se, "n": len(xs)}


def paired(a, b):
    """配对差统计（`a − b`）。"""
    return _stats([x - y for x, y in zip(a, b)])


def decide(v):
    r"""★ **冻结判定表**（卡 §4）：纯函数 ⇒ ⇒ 可被 `A43-14` 的合成自检**逐支**触发。

    ★ **单一真相源**：`p1`/`p2`/`p3` 由本函数**自己**从统计量算出（**不**接受外部传入的布尔）
      ⇒ ⇒ 杜绝「传进来的布尔与统计量不自洽」导致**分支不可达**（DEF-43-8，由 `verify_merge43.py` 查出）。
    `v` 的键：`n_ok`（臂齐且 `n` 符）、`finite`（全有限）、
    `rho_mean`、`rho_se`（主判据：终态 `ρ`）、`rho0_mean`、`rho0_se`（次判据：初值 `ρ`）、
    `rho_diff_mean`、`rho_diff_se`（特异性）、`direct_mean`（`direct` 臂 `cos_end` 的均值）。
    ⇒ 返回 `{"outcome", "counted", "primary_label", "secondary_label", "p1", "p1b", "p2", "p3"}`。
    """
    if not v["n_ok"]:
        return {"outcome": "INCOMPLETE", "counted": False, "primary_label": "NA",
                "secondary_label": "NA", "p1": False, "p1b": False, "p2": False, "p3": False}
    if not v["finite"]:
        return {"outcome": "DIVERGED", "counted": False, "primary_label": "NA",
                "secondary_label": "NA", "p1": False, "p1b": False, "p2": False, "p3": False}
    # —— 三条谓词（**自己算**，不收外部布尔）
    p1 = v["rho_mean"] >= SIG_K * v["rho_se"]
    p1b = v["rho0_mean"] >= SIG_K * v["rho0_se"]
    p2 = v["rho_diff_mean"] >= SIG_K * v["rho_diff_se"]
    p3 = v["direct_mean"] >= COS_DIRECT
    # —— 主判据标签（终态 ρ）
    if v["rho_mean"] <= -SIG_K * v["rho_se"]:
        primary = "ANTI"
    elif not p1:
        primary = "NULL"
    else:
        primary = "POSITIVE"
    # —— 次判据标签（初值 ρ）
    if v["rho0_mean"] <= -SIG_K * v["rho0_se"]:
        secondary = "ANTI"
    elif not p1b:
        secondary = "NULL"
    else:
        secondary = "POSITIVE"
    if primary == "ANTI":
        out = "ANTI-ADDRESSING"
    elif primary == "NULL":
        out = "NOT-ADDRESSING"
    elif not p2:
        out = "NOT-SPECIFIC"
    elif not p3:
        out = "EMPTY-CRITERION"
    else:
        out = "PASS"
    return {"outcome": out, "counted": (out == "PASS"),
            "primary_label": primary, "secondary_label": secondary,
            "p1": bool(p1), "p1b": bool(p1b), "p2": bool(p2), "p3": bool(p3)}


def selftest_decide():
    r"""★ `A43-14`：判定表**七支**逐支**真触发**（纪律 104：分支存在 ≠ 分支可达）。"""
    base = {"n_ok": True, "finite": True,
            "rho_mean": 0.09, "rho_se": 0.03, "rho0_mean": 0.09, "rho0_se": 0.03,
            "rho_diff_mean": 0.10, "rho_diff_se": 0.032,
            "direct_mean": 0.995}
    cases = [
        ("INCOMPLETE", dict(base, n_ok=False)),
        ("DIVERGED", dict(base, finite=False)),
        ("ANTI-ADDRESSING", dict(base, rho_mean=-0.09)),
        ("NOT-ADDRESSING", dict(base, rho_mean=0.02)),
        ("NOT-SPECIFIC", dict(base, rho_diff_mean=0.02)),
        ("EMPTY-CRITERION", dict(base, direct_mean=0.85)),
        ("PASS", dict(base)),
    ]
    got = []
    for want, vv in cases:
        r = decide(vv)
        got.append({"want": want, "got": r["outcome"], "ok": r["outcome"] == want,
                    "counted": r["counted"]})
    ok = all(x["ok"] for x in got)
    # —— 次判据标签也要能分叉（NULL/POSITIVE/ANTI）
    lab = [decide(dict(base, rho0_mean=0.09))["secondary_label"],
           decide(dict(base, rho0_mean=0.01))["secondary_label"],
           decide(dict(base, rho0_mean=-0.09))["secondary_label"]]
    lab_ok = lab == ["POSITIVE", "NULL", "ANTI"]
    return {"cases": got, "ok": bool(ok and lab_ok), "secondary_labels": lab,
            "n_cases": len(got), "secondary_ok": lab_ok}


# ══════════ Stage A（硬闸；**断言必须真的读被断言的对象** ⇒ 纪律 97）
def _all_pairs_differ(mats):
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
    # ── A43-1：装置/任务层/上游脚本 md5 未变
    md5s = {
        "dir31/R1_grad.py": MD5_DEVICE, "dir33/R1_task33.py": MD5_TASK,
        "dir37/R1_task37.py": MD5_DIR37, "dir38/R1_task38.py": MD5_DIR38,
        "dir39/R1_task39.py": MD5_DIR39, "dir40/R1_task40.py": MD5_DIR40,
        "dir41/R1_task41.py": MD5_DIR41, "dir42/R1_task42.py": MD5_DIR42,
    }
    got = {p: _md5(f"{ROOT}/{p}") for p in md5s}
    g["A43-1_md5"] = got
    g["A43-1_ok"] = all(got[p] == v for p, v in md5s.items())

    # ── A43-2：有效常数实测（纪律 94）
    eff = {"D_H": M.D_H, "D_K": M.D_K, "D_V": M.D_V, "R": M.R, "T_STEPS": M.T_STEPS,
           "ETA": M.ETA, "LAM": M.LAM, "ALPHA": M.ALPHA, "GAMMA": M.GAMMA,
           "KEY_SCALE": R.KEY_SCALE, "INIT_SD": M.INIT_SD, "SEED0": M.SEED0}
    want = {"D_H": 8, "D_K": 16, "D_V": 4, "R": 2, "T_STEPS": 16, "ETA": 0.2,
            "LAM": 0.05, "ALPHA": 1.0, "GAMMA": 0.5, "KEY_SCALE": 0.5,
            "INIT_SD": 0.5, "SEED0": 20261111}
    g["A43-2_effective"] = eff
    g["A43-2_ok"] = all(eff[k] == want[k] for k in want)

    # ── A43-3：卡 md5 守卫
    g["A43-3_card_md5"] = _md5(CARD_PATH)
    g["A43-3_ok"] = g["A43-3_card_md5"] == CARD_MD5_EXPECTED

    # ── A43-4：offset 空间的种子段隔离（纪律 97）
    hist = []
    for _k, (a, b) in list(T37.HIST_RANGES.items()) + list(T37.OWN_SEGMENTS.items()):
        hist.extend(range(a, b + 1))
    for a, b in [(12500, 13096), (13500, 13900), (14000, 14384), (14500, 14512),
                 (15000, 15148), (15500, 15900), (16000, 16192), (16400, 16412),
                 (17000, 17048), (17100, 17164), (17200, 17264),
                 (18000, 18191), (18500, 18899), (18900, 18911),
                 (20400, 20412), (20500, 20595), (20600, 20999),
                 (21000, 21095), (21100, 21111), (21500, 21899),
                 (22000, 22015), (22100, 22103)]:     # DIR-42 探针 / smoke
        hist.extend(range(a, b + 1))
    hist_set = set(hist)
    dir42_formal = set(range(22500, 22900))
    g["A43-4_hist_n"] = len(hist_set)
    g["A43-4_formal_hist"] = len(set(INST_OFF) & hist_set)
    g["A43-4_formal_eq_dir42"] = (set(INST_OFF) == dir42_formal)
    g["A43-4_smoke_hist"] = len(set(SMOKE_INST_OFF) & hist_set)
    g["A43-4_smoke_formal"] = len(set(SMOKE_INST_OFF) & set(INST_OFF))
    g["A43-4_probe_formal"] = len(set(PROBE_INST_OFF) & set(INST_OFF))
    g["A43-4_probe_smoke"] = len(set(PROBE_INST_OFF) & set(SMOKE_INST_OFF))
    dev_max = max(SEED0 + o for o in INST_OFF)
    g["A43-4_reader_above_device"] = bool(
        min(SEED0 + o + READER_SEED_OFFSET for o in INST_OFF) > dev_max)
    g["A43-4_randkey_above_device"] = bool(
        min(SEED0 + o + RAND_KEY_SEED_OFFSET for o in INST_OFF) > dev_max)
    _rd_lo = min(SEED0 + o + READER_SEED_OFFSET for o in INST_OFF)
    _rd_hi = max(SEED0 + o + READER_SEED_OFFSET for o in INST_OFF)
    _rk_lo = min(SEED0 + o + RAND_KEY_SEED_OFFSET for o in INST_OFF)
    _rk_hi = max(SEED0 + o + RAND_KEY_SEED_OFFSET for o in INST_OFF)
    g["A43-4_streams_range"] = {"reader": [_rd_lo, _rd_hi], "randkey": [_rk_lo, _rk_hi]}
    g["A43-4_streams_disjoint"] = bool(_rd_hi < _rk_lo or _rk_hi < _rd_lo)
    g["A43-4_ok"] = (g["A43-4_formal_hist"] == 0 and g["A43-4_formal_eq_dir42"]
                     and g["A43-4_smoke_hist"] == 0 and g["A43-4_smoke_formal"] == 0
                     and g["A43-4_probe_formal"] == 0 and g["A43-4_probe_smoke"] == 0
                     and g["A43-4_reader_above_device"] and g["A43-4_randkey_above_device"]
                     and g["A43-4_streams_disjoint"])

    # ── A43-5：★ **逐位锚**（DIR-42 归档的 `learned` 行）：同实例、同初值、同协议 ⇒ 逐位相同
    import glob
    anchor_rows = {}
    for p in sorted(glob.glob(f"{ROOT}/dir42/out/DIR42r10_arm_learned_shard*.json")):
        try:
            for r in json.load(open(p, encoding="utf-8")).get("rows", []):
                anchor_rows[int(r["idx"])] = r
        except (json.JSONDecodeError, OSError):
            pass
    probe_idx = [i for i in (0, 1) if i in anchor_rows][:2]
    diffs = {}
    for i in probe_idx:
        sd = SEED0 + INST_OFF[i]
        phi, _meta = train_reader43("learned", sd, steps=STEPS)
        e = T42.eval_reader42("learned", phi, sd, win=EVAL_WIN)
        ref = anchor_rows[i]
        d = {"L": abs(e["L"] - ref["L"]), "cos": abs(e["cos_q_k"] - ref["cos"]),
             "dWq1": abs(_meta["dWq1_from_init"] - ref["dWq1"]),
             "gn_Wq1": abs(_meta["gradnorms"].get("Wq1", 0.0) - ref["gn_Wq1"])}
        diffs[str(i)] = d
    g["A43-5_anchor_diffs"] = diffs
    g["A43-5_n_anchor"] = len(diffs)
    g["A43-5_ok"] = (len(diffs) >= 1 and all(
        all(x == 0.0 for x in d.values()) for d in diffs.values()))

    # ── A43-6：构造事实实测（**在真实轨迹上**：理想控制器 ⇒ `cos_q_k` = 1.0）
    ideal = {}
    for o in SMOKE_INST_OFF[:2]:
        sd = SEED0 + o
        P = T37._mk(sd, None)
        phi = T42.reader_init(sd)
        phi2 = {k: v.detach().clone() for k, v in phi.items()}
        phi2["Wq1"] = P["Pk1"].detach().clone()
        phi2["bq1"] = torch.zeros(D_K, dtype=torch.float64)
        Pv = T37._mk(sd, phi2, train=True)
        _L, dg = T42.trace42(Pv, N_REP, "learned", sd, phi2, want_diag=True, win=TRAIN_WIN)
        ideal[str(sd)] = dg["cos_q_k"]
    g["A43-6_ideal_cos_q_k"] = ideal
    g["A43-6_ok"] = bool(ideal) and all(abs(v - 1.0) < 1e-9 for v in ideal.values())

    # ── A43-7 / A43-8：单变量 + 快照基准（纪律 98）
    meta_r, meta_l = {}, {}
    for o in SMOKE_INST_OFF:
        sd = SEED0 + o
        _pr, mr = train_reader43("rand", sd, steps=5)
        _pl, ml = train_reader43("learned", sd, steps=5)
        meta_r[str(sd)] = mr
        meta_l[str(sd)] = ml
    g["A43-9_meta"] = {"rand": meta_r, "learned": meta_l}
    g["A43-7_ok"] = all(mr["dWq1_from_init"] == 0.0 and ml["dWq1_from_init"] > 0.0
                        for mr, ml in zip(meta_r.values(), meta_l.values()))
    g["A43-8_ok"] = all(
        mr["dWq1_from_init"] == 0.0 and mr["dbq1_from_init"] == 0.0
        and ml["dWq1_from_init"] > 0.0 and math.isfinite(ml["gn_task_final"])
        and ml["gn_task_final"] > 0.0
        for mr, ml in zip(meta_r.values(), meta_l.values()))

    # ── A43-9：正控可达（`direct` 臂）
    dr = [direct43(SEED0 + o) for o in SMOKE_INST_OFF[:2]]
    g["A43-9_direct"] = dr
    g["A43-9_ok"] = bool(dr) and all(d["direct_cos_mean"] >= COS_DIRECT for d in dr)

    # ── A43-10：结构（**全部 `C(n,2)` 对**，纪律 102）
    STRUCT = [SEED0 + 22500 + i for i in range(20)]
    pks, rds = [], []
    for sd in STRUCT:
        P = T37._mk(sd, None)
        pks.append(P["Pk1"])
        rds.append(T42.reader_init(sd)["Wq1"].detach())
    npk, dpk, spk_ = _all_pairs_differ(pks)
    nrd, drd, srd = _all_pairs_differ(rds)
    g["A43-10_pk1"] = {"n": len(pks), "pairs_checked": npk, "pairs_differing": dpk,
                       "pairs_identical": spk_}
    g["A43-10_reader_init"] = {"n": len(rds), "pairs_checked": nrd, "pairs_differing": drd,
                               "pairs_identical": srd}
    g["A43-10_ok"] = (dpk == npk == nrd == drd and not spk_ and not srd)

    # ── A43-11：读器初值确定
    i1 = T42.reader_init(SEED0 + SMOKE_INST_OFF[0])
    i2 = T42.reader_init(SEED0 + SMOKE_INST_OFF[0])
    g["A43-11_init_deterministic"] = all(
        float((i1[k] - i2[k]).abs().max()) == 0.0 for k in i1)
    g["A43-11_ok"] = g["A43-11_init_deterministic"]

    # ── A43-12：构造事实实测（跨读器：`c`/`q*`/层-1 写入 与读器无关；任务梯度不进写入通道）
    s4 = {}
    for o in SMOKE_INST_OFF[:2]:
        sd = SEED0 + o
        P = T37._mk(sd, None)
        c, qs = ctx_of(P), qstar_of(P)
        phis = [T42.reader_init(sd),
                {k: v.detach().clone() for k, v in T42.reader_init(sd).items()}]
        phis[1]["Wq1"] = phis[1]["Wq1"] * 8.0
        phis[1]["bq1"] = torch.linspace(-1.0, 1.0, D_K, dtype=torch.float64)
        W1s = []
        for ph in phis:
            Pv = T37._mk(sd, ph, train=True)
            T42.trace42(Pv, N_REP, "learned", sd, ph, win=TRAIN_WIN)
            W1s.append(M._W(Pv["U1"], Pv["V1"]).detach().clone())
        Pv = T37._mk(sd, phis[0], train=True)
        L, _ = T42.trace42(Pv, N_REP, "learned", sd, phis[0], win=TRAIN_WIN)
        L.backward()
        gu = float(Pv["U1"].grad.norm()) if Pv["U1"].grad is not None else 0.0
        gv = float(Pv["V1"].grad.norm()) if Pv["V1"].grad is not None else 0.0
        phis[0]["Wq1"].grad = None
        s4[str(sd)] = {"layer1_write_reader_indep": float((W1s[0] - W1s[1]).abs().max()),
                       "grad_flow_written_channel": max(gu, gv)}
    g["A43-12_struct"] = s4
    g["A43-12_ok"] = bool(s4) and all(v["layer1_write_reader_indep"] == 0.0
                                      and v["grad_flow_written_channel"] == 0.0
                                      for v in s4.values())

    # ── A43-13：★ **`ρ` 的符号约定自检**（`sign(ΔL 沿 +ĝ_addr) = −sign(ρ)`）
    sgn = []
    for o in SMOKE_INST_OFF[:4]:
        sd = SEED0 + o
        P0 = T37._mk(sd, None)
        phi = T42.reader_init(sd)
        Pv = T37._mk(sd, phi, train=True)
        L, _ = T42.trace42(Pv, N_REP, "learned", sd, phi, win=TRAIN_WIN)
        L.backward()
        gt = phi["Wq1"].grad.detach().clone() if phi["Wq1"].grad is not None \
            else torch.zeros_like(phi["Wq1"])
        phi["Wq1"].grad = None
        _A, g_addr = align_grad(P0, phi)
        rho = cos2(-gt, g_addr)                       # ★ 下降方向 vs 寻址方向
        u = g_addr / g_addr.new_tensor(float(g_addr.norm()))
        rel = float(phi["Wq1"].norm())
        ph2 = {k: v.detach().clone() for k, v in phi.items()}
        ph2["Wq1"] = phi["Wq1"].detach() + 0.01 * rel * u
        P2 = T37._mk(sd, ph2, train=True)
        L2, _ = T42.trace42(P2, N_REP, "learned", sd, ph2, win=TRAIN_WIN)
        dL = float(L2) - float(L)
        sgn.append({"sd": int(sd), "rho": rho, "dL": dL,
                    "sign_ok": (rho * dL < 0.0) if (abs(rho) > 0.05 and abs(dL) > 1e-12) else None})
    g["A43-13_sign_check"] = sgn
    checked = [x for x in sgn if x["sign_ok"] is not None]
    g["A43-13_ok"] = bool(checked) and all(x["sign_ok"] for x in checked)

    # ── A43-14：判定表**七支**逐支合成自检
    st = selftest_decide()
    g["A43-14_selftest"] = st
    g["A43-14_ok"] = st["ok"]

    # ── A43-15：★ **正式轮代码路径干跑**（1 个实例；临时 `OUT`；**不碰**正式落盘）
    import tempfile as _tf
    _out_saved = OUT
    _tmp_out = _tf.mkdtemp(prefix="D43_dryrun_")
    globals()["OUT"] = _tmp_out
    try:
        _o = formal_arm("learned", shard=0, nshards=N_INST)
        g["A43-15_dryrun"] = {"ok": bool(_o and _o["n"] == 1), "n": (_o["n"] if _o else 0),
                              "rho_final": (_o["rows"][0]["rho_final"] if _o and _o["rows"] else None),
                              "secs": (_o.get("secs") if _o else None)}
    except Exception as _e:
        g["A43-15_dryrun"] = {"ok": False, "error": repr(_e)}
    finally:
        globals()["OUT"] = _out_saved
    g["A43-15_ok"] = bool(g["A43-15_dryrun"].get("ok"))

    ok = all(v for k, v in g.items() if k.endswith("_ok"))
    print("Stage A：" + ("★ 全部通过" if ok else "★ 未通过"))
    for k in sorted(g):
        if k.endswith("_ok") or k.endswith("_md5"):
            print(f"  {k:26s} {g[k]}")
    return g, ok


# ══════════ smoke（证明协议可执行；**永不产生判据裁决**）
def smoke():
    out = {"note": "smoke：只证明可执行，不给判据"}
    for o in SMOKE_INST_OFF:
        sd = SEED0 + o
        for arm in ("learned", "rand"):
            phi, meta = train_reader43(arm, sd, steps=3)
            e = T42.eval_reader42(arm, phi, sd, win=EVAL_WIN)
            out[f"{sd}/{arm}"] = {"L_eval": e["L"], "cos_q_k": e["cos_q_k"],
                                  "Wmax": e["Wmax"], "c_eq_h": e["c_eq_h"],
                                  "dWq1_from_init": meta["dWq1_from_init"],
                                  "rho_final": meta["rho_final"], "rho0": meta["rho0"],
                                  "rho_rand_final": meta["rho_rand_final"]}
            print(f"  {sd}/{arm:8s} L_eval={e['L']:.6f} cos={e['cos_q_k']:.4f} "
                  f"rho0={meta['rho0']:+.4f} rhoF={meta['rho_final']:+.4f} "
                  f"rhoR={meta['rho_rand_final']:+.4f} |Wq-init|={meta['dWq1_from_init']:.3e}",
                  flush=True)
        d = direct43(sd, steps=STEPS)
        out[f"{sd}/direct"] = d
        print(f"  {sd}/direct cos_mean={d['direct_cos_mean']:.6f} "
              f"cos_min={d['direct_cos_min']:.6f}", flush=True)
    print("smoke 完成（**无判据裁决**）")
    return out


# ══════════ 增量落盘（**纪律 103**：逐实例 `O_APPEND` ⇒ 可续跑）
def _jsonl_path(arm, shard):
    return os.path.join(OUT, f"DIR43r10_arm_{arm}_shard{shard}.jsonl")


def _row_cfg():
    """每行都带配置指纹 ⇒ ★ 配置一变，旧行**自动作废**（含 schema 版本与种子基址）。"""
    return {"steps": STEPS, "lr": LR, "n_rep": N_REP,
            "train_win": list(TRAIN_WIN), "eval_win": list(EVAL_WIN),
            "inst_off": [INST_OFF[0], INST_OFF[-1]], "rho_grid": list(RHO_GRID),
            "schema": ROW_SCHEMA, "seed0": SEED0}


def _append_jsonl(path, row):
    line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _read_jsonl(path):
    """读增量行；★ 容忍**最后一行被截断** ⇒ 只丢该实例，不丢整片。"""
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
    _card_guard(f"formal {arm}/shard{shard}")
    t0 = time.time()
    seg = INST_OFF if arm in ("learned", "direct") else CTRL_OFF
    n_exp = N_EXPECT[arm]
    idxs = [i for i in range(n_exp) if i % nshards == shard]
    os.makedirs(OUT, exist_ok=True)
    jpath = _jsonl_path(arm, shard)
    done = {r["idx"]: r for r in _read_jsonl(jpath)}
    if done:
        print(f"[{arm}/shard{shard}] ★ 续跑：复用已落盘的 {len(done)} 个实例读数", flush=True)
    rows = []
    for si in idxs:
        if si in done:
            rows.append(done[si])
            continue
        sd = SEED0 + seg[si]
        row = {"idx": si, "seed": sd, "arm": arm, "cfg": _row_cfg()}
        if arm == "direct":
            d = direct43(sd, steps=STEPS)
            row.update({"direct_cos_min": d["direct_cos_min"],
                        "direct_cos_mean": d["direct_cos_mean"],
                        "direct_cos_max": d["direct_cos_max"],
                        "direct_Wdist_Pk1": d["direct_Wdist_Pk1"]})
        else:
            phi, meta = train_reader43(arm, sd, steps=STEPS)
            e = T42.eval_reader42(arm, phi, sd, win=EVAL_WIN)
            e_seen = T42.eval_reader42(arm, phi, sd, win=TRAIN_WIN)
            e_full = T42.eval_reader42(arm, phi, sd, win=None)
            row.update({"L": e["L"], "L_seen": e_seen["L"], "L_full": e_full["L"],
                        "cos": e["cos_q_k"], "align": e["align_q_k"],
                        "q_norm": e["q_norm"], "y_norm": e["y_norm"], "m_abs": e["m_abs"],
                        "Wmax": e["Wmax"], "dWq1": meta["dWq1_from_init"],
                        "gn_Wq1": meta["gradnorms"].get("Wq1", 0.0),
                        "rho_final": meta["rho_final"], "rho0": meta["rho0"],
                        "rho_rand_final": meta["rho_rand_final"], "rho_rand0": meta["rho_rand0"],
                        "rho_curve": meta["rho_curve"], "rho_rand_curve": meta["rho_rand_curve"],
                        "gn_task_final": meta["gn_task_final"]})
        rows.append(row)
        _append_jsonl(jpath, row)
        if (len(rows) % 10) == 0:
            rr = row.get("rho_final")
            print(f"[{arm}/shard{shard}] {len(rows)}/{len(idxs)} "
                  f"rho={rr if rr is None else round(rr, 4)} ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["idx"])
    out = {"arm": arm, "shard": shard, "nshards": nshards, "n": len(rows),
           "n_inst": n_exp, "n_rep": N_REP, "steps": STEPS, "lr": LR,
           "train_win": list(TRAIN_WIN), "eval_win": list(EVAL_WIN),
           "rho_grid": list(RHO_GRID), "rows": rows, "secs": time.time() - t0,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(OUT, f"DIR43r10_arm_{arm}_shard{shard}.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[{arm}/shard{shard}] 完成 {len(rows)} 个实例，{time.time()-t0:.0f}s", flush=True)
    return out


# ══════════ 汇总裁决（卡 §3/§4，**冻结**）
def merge():
    r"""读各臂分片 ⇒ 按**冻结**判定表裁决。★ **不读探针文件**（卡 §4 第 1 支）。"""
    _card_guard("merge")
    cfg = _row_cfg()
    rows_by_arm, missing, src_by_arm, read_files = {}, [], {}, []
    for arm in ARMS:
        rows, srcs = [], []
        for sh in range(4):
            jl = _jsonl_path(arm, sh)
            read_files.append(jl)
            r1 = _read_jsonl(jl)
            jp = os.path.join(OUT, f"DIR43r10_arm_{arm}_shard{sh}.json")
            read_files.append(jp)
            r2 = []
            if os.path.exists(jp):
                try:
                    r2 = json.load(open(jp, encoding="utf-8")).get("rows", [])
                except (json.JSONDecodeError, OSError):
                    r2 = []
            r2 = [r for r in r2 if r.get("cfg") == cfg]
            if not r1 and not r2:
                missing.append(f"{arm}/shard{sh}")
                continue
            if len(r1) >= len(r2):
                rows.extend(r1); srcs.append(f"shard{sh}:jsonl")
            else:
                rows.extend(r2); srcs.append(f"shard{sh}:json")
        rows.sort(key=lambda r: r["idx"])
        rows_by_arm[arm] = rows
        src_by_arm[arm] = srcs
    out = {"n_inst": N_INST, "n_ctrl": N_CTRL, "n_rep": N_REP, "steps": STEPS,
           "sig_k": SIG_K, "cos_direct": COS_DIRECT, "rho_grid": list(RHO_GRID),
           "train_win": list(TRAIN_WIN), "eval_win": list(EVAL_WIN),
           "missing": missing, "source": src_by_arm,
           "read_files": read_files,
           "read_files_all_shards": all("DIR43r10_arm_" in p for p in read_files),
           "n_read_files": len(read_files)}

    n_ok = True
    for arm in ARMS:
        n_have = len(rows_by_arm.get(arm, []))
        if n_have != N_EXPECT[arm]:
            n_ok = False
            out.setdefault("n_mismatch", {})[arm] = {"have": n_have, "want": N_EXPECT[arm]}
    bad_n = []
    for arm in ("learned", "rand"):
        seg = INST_OFF if arm == "learned" else CTRL_OFF
        for r in rows_by_arm.get(arm, []):
            if not (isinstance(r.get("idx"), int) and 0 <= r["idx"] < len(seg)) \
                    or r.get("seed") != SEED0 + seg[r["idx"]]:
                bad_n.append(f"{arm}/{r.get('idx')}")
    out["bad_n"] = bad_n
    if bad_n:
        n_ok = False

    if not n_ok or missing:
        out["outcome"] = "INCOMPLETE"
        out["counted"] = False
        out["primary_label"] = "NA"
        out["secondary_label"] = "NA"
        print("outcome=INCOMPLETE（缺臂或 n 不符）")
        with open(os.path.join(OUT, "DIR43r10_formal.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        return out

    L = rows_by_arm["learned"]
    D = rows_by_arm["direct"]
    rho = [r["rho_final"] for r in L]
    rho0 = [r["rho0"] for r in L]
    rhor = [r["rho_rand_final"] for r in L]
    rhor0 = [r["rho_rand0"] for r in L]
    st_rho = _stats(rho)
    st_rho0 = _stats(rho0)
    st_diff = paired(rho, rhor)
    st_diff0 = paired(rho0, rhor0)
    dmin = [r["direct_cos_min"] for r in D]
    dmean = [r["direct_cos_mean"] for r in D]
    st_dmin = _stats(dmin)
    st_dmean = _stats(dmean)

    finite = all(math.isfinite(x) for x in
                 [st_rho["mean"], st_rho["se"], st_rho0["mean"], st_rho0["se"],
                  st_diff["mean"], st_diff["se"], st_dmin["mean"], st_dmean["mean"]])

    v = {"n_ok": True, "finite": bool(finite),
         "rho_mean": st_rho["mean"], "rho_se": st_rho["se"],
         "rho0_mean": st_rho0["mean"], "rho0_se": st_rho0["se"],
         "rho_diff_mean": st_diff["mean"], "rho_diff_se": st_diff["se"],
         "direct_mean": st_dmean["mean"], "direct_min": st_dmin["mean"]}
    dec = decide(v)
    p1, p1b, p2, p3 = dec["p1"], dec["p1b"], dec["p2"], dec["p3"]
    out.update({"rho_primary": st_rho, "rho_secondary": st_rho0,
                "rho_diff_primary": st_diff, "rho_diff_secondary": st_diff0,
                "direct_cos_min_stats": st_dmin, "direct_cos_mean_stats": st_dmean,
                "P1_signif": bool(p1), "P1b_signif_secondary": bool(p1b),
                "P2_specific": bool(p2), "P3_nonempty": bool(p3),
                "finite": bool(finite),
                "rho_negative_frac": sum(1 for x in rho if x < 0) / len(rho),
                "rho_curve_mean": {t: _fmean_safe([r["rho_curve"][t] for r in L
                                                    if t in r.get("rho_curve", {})])
                                   for t in map(str, RHO_GRID)},
                "outcome": dec["outcome"], "counted": dec["counted"],
                "primary_label": dec["primary_label"],
                "secondary_label": dec["secondary_label"]})
    print(f"outcome={out['outcome']}  counted={out['counted']}  "
          f"primary={out['primary_label']}  secondary={out['secondary_label']}")
    print(f"  ρ(终态) mean={st_rho['mean']:+.6f} 3se={st_rho['3se']:.6f} (P1={p1})")
    print(f"  ρ(初值) mean={st_rho0['mean']:+.6f} 3se={st_rho0['3se']:.6f} (P1'={p1b})")
    print(f"  ρ−ρ_rand mean={st_diff['mean']:+.6f} 3se={st_diff['3se']:.6f} (P2={p2})")
    print(f"  direct cos: mean={st_dmean['mean']:.6f} min_i={st_dmin['mean']:.6f} "
          f"(P3={p3}；阈值 {COS_DIRECT})")
    print(f"  ρ<0 的实例占比 = {out['rho_negative_frac']:.3f}")
    assert out["read_files_all_shards"], "★ merge 读到了非分片产物（卡 §4 第 1 支的隔离）"
    with open(os.path.join(OUT, "DIR43r10_formal.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "stageA":
        gg, ok = stage_a()
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "DIR43r10_stageA.json"), "w") as f:
            json.dump(gg, f, ensure_ascii=False, indent=1)
        raise SystemExit(0 if ok else 1)
    if cmd == "smoke":
        o = smoke()
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "DIR43r10_smoke.json"), "w") as f:
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
