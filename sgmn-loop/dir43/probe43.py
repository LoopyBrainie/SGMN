#!/opt/pytorch/bin/python
r"""DIR-43 · **开卡前探针**（**不进判据、不计 `SUCCEEDED`**）

方向（待冻结）：§7.2「**数值与结构选择**」行的「**查询坐标对齐方式**」
              ＋ §7.2「**集成接口验收**」行的「读取参数的非零有限梯度」的**实质验收**
              ⇒ DIR-42 只证明梯度**非零有限**，**没有**证明它**指向寻址**。

本探针只做两件事：
  (S) **结构核实** —— 读码得出的四个断言，逐条**在 trainer 上实测**；
  (M) **效应量估计** —— `ρ`（任务梯度 × 寻址方向 的夹角余弦）的均值/方差 ⇒ 用于**定 `N`**。

★ **探针不产生判据**：它既不冻结阈值，也不决定 H43 的真伪。
★ **派生**：`dir42/R1_task42.py`（**逐字**：读器初值 / 训练协议 / 损失窗口 / 装置常数）。

用法：python3 probe43.py [--n N] [--out PATH]
"""
import argparse
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
import R1_task42 as T42          # ★ 逐字复用：读器初值 / trace42 / 训练协议
import R1_task33 as R            # 任务层
import R1_grad as M              # 装置

D_H, D_V, D_K, T = R.D_H, R.D_V, R.D_K, R.T
N_REP = T42.N_REP
TRAIN_WIN, EVAL_WIN = T42.TRAIN_WIN, T42.EVAL_WIN
STEPS, LR = T42.STEPS, T42.LR

# ★ **探针段 = 22000..22063**：与 DIR-43 的**正式轮段**（22500..22899）**不相交**
#   ⇒ ⇒ 正式轮的 400 个实例**从未被探针预看** ⇒ 本轮是**盲测**（探针只定 `N`、确认判据非空）。
# ★ **run-1 的更正（留档）**：首轮探针**误用**了正式轮段的前 64 个实例（22500..22563）
#   ⇒ 会让正式轮变成「确认」而非盲测 ⇒ ⇒ 已改到 22000..22063 并**重跑**；
#   run-1 的读数**完整保留**（`dir43/out/DIR43_probe_run1*.json*`），**不删改、不追认**。
PROBE_N = 64
PROBE_INST_OFF = list(range(22000, 22000 + PROBE_N))


# ══════════════ 结构核实所需的**纯函数**（不依赖轨迹）
def ctx_of(P):
    r"""层-1 的**读上下文** `c(t)`：`tanh(a[t] @ A1 + b1)`。

    ★ 读码断言：**与读器无关**（`trace42` 里 li=0 时 `hp_prev` 被重置为 `a[t]`）
      ⇒ ⇒ `c`、`q*(t)`、层-1 的写入键/值**全是常量**。
    """
    return [torch.tanh(P["a"][t] @ P["A1"] + P["b1"]) for t in range(T)]


def qstar_of(P, key=None):
    r"""**目标查询** `q*(t) = tanh(c(t) @ Pk1)`（`Pk1` 已是**缩放后**的装置量）。"""
    K = P["Pk1"] if key is None else key
    c = ctx_of(P)
    return [torch.tanh(c[t] @ K) for t in range(T)]


def align_obj(P, phi, key=None):
    r"""`A(Wq1,bq1) := mean_{t=1..T-1} cos(q(t), q*(t))`，`q(t) = tanh(c(t) @ Wq1 + bq1)`。

    ★ `c` 与 `q*` 都是**常量** ⇒ ⇒ `A` 只是 `(Wq1, bq1)` 的函数 ⇒ **无需轨迹**。
    ★ 与 `trace42` 的诊断 `cos_q_k` **同口径**：`trace42` 配对 `q1s[t]` 与 `k1s[t-1]`；
      而 `k1s[t-1] = tanh(c(t-2) @ Pk1)` 且 `c` 周期 2 ⇒ ⇒ `k1s[t-1] ≡ q*(t)`。
    """
    c, qs = ctx_of(P), qstar_of(P, key)
    W = phi["Wq1"].detach().clone().requires_grad_(True)
    b = phi["bq1"].detach().clone().requires_grad_(True)
    cs = []
    for t in range(1, T):
        q = torch.tanh(c[t] @ W + b)
        cs.append(torch.nn.functional.cosine_similarity(q, qs[t], dim=0))
    A = torch.stack(cs).mean()
    return A, W, b


def addr_grad(P, phi, key=None):
    r"""寻址方向 `g_addr := ∂A/∂Wq1`（`bq1` 方向另记）。"""
    A, W, b = align_obj(P, phi, key)
    A.backward()
    return float(A), (W.grad if W.grad is not None else torch.zeros_like(W)), \
        (b.grad if b.grad is not None else torch.zeros_like(b))


def cos2(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float((a * b).sum()) / (na * nb) if na > 1e-300 and nb > 1e-300 else float("nan")


def rand_key(P, sd):
    r"""**特异性对照**用的「错误键」：`Pk_rand = Pk1 @ Q`（`Q` 正交随机）⇒ 保持尺度/谱，
    但与「正确键」无关 ⇒ ⇒ 公平对照。"""
    g = torch.Generator().manual_seed(int(sd) + 55_555_555)
    A = torch.randn(D_K, D_K, generator=g, dtype=torch.float64)
    Q, Rr = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(Rr)).unsqueeze(0)
    return P["Pk1"] @ Q


# ══════════════ (S) 结构核实
def structural_checks(sds):
    out = {}
    S = {"S1_qstar_eq_written_key": [], "S2_ideal_controller_cos": [],
         "S3_ctx_period2": [], "S4_layer1_write_reader_indep": [],
         "S5_grad_flow_written_channel": []}
    for sd in sds:
        P = T42.T37._mk(sd, None)
        phi = T42.reader_init(sd)
        c = ctx_of(P)
        qs = qstar_of(P)

        # —— S3：`c` 周期 2（`a[t]=x[t%2]` ⇒ `c[t] ≡ c[t-2]`）
        d3 = max(float((c[t] - c[t - 2]).abs().max()) for t in range(2, T))
        S["S3_ctx_period2"].append(d3)

        # —— S1：`q*(t) ≡ k(t−1)`（写入时刻 t−1 用的键）
        #     `k(s) = tanh(c(s-1) @ Pk1)` ⇒ `k(t-1) = tanh(c(t-2) @ Pk1)`；
        #     `c(t-2) ≡ c(t)` ⇒ ⇒ `k(t-1) ≡ q*(t)`。
        d1 = max(float((torch.tanh(c[t - 2] @ P["Pk1"]) - qs[t]).abs().max())
                 for t in range(1, T))
        S["S1_qstar_eq_written_key"].append(d1)

        # —— S2：理想控制器 `Wq1:=Pk1, bq1:=0` ⇒ `cos(q(t), q*(t)) = 1`
        phi2 = {k: v.detach().clone() for k, v in phi.items()}
        phi2["Wq1"] = P["Pk1"].detach().clone()
        phi2["bq1"] = torch.zeros(D_K, dtype=torch.float64)
        A2, _, _ = align_obj(P, phi2)
        S["S2_ideal_controller_cos"].append(float(A2.detach()))

        # —— S4：层-1 的记忆 `W1` 与读器**无关** ——
        #     `trace42` 的层-1 写入用 `h_pre_prev[0]=c(t-1)` 与 `h_pre[0]=c(t)`（**与读器无关**）
        #     ⇒ ⇒ 层-1 的 `U1/V1` 演化**不依赖读器** ⇒ ⇒ 用两个**差别很大**的读器各跑一遍，
        #     比较**最终**的 `W1`（`trace42` 就地更新 `P` ⇒ 跑完即可读）⇒ 应**逐位相同**。
        phis = [phi, {k: v.detach().clone() for k, v in phi.items()}]
        phis[1]["Wq1"] = phis[1]["Wq1"] * 8.0
        phis[1]["bq1"] = torch.linspace(-1.0, 1.0, D_K, dtype=torch.float64)
        W1s = []
        for ph in phis:
            Pv = T42.T37._mk(sd, ph, train=True)
            T42.trace42(Pv, N_REP, "learned", sd, ph, win=TRAIN_WIN)
            W1s.append(M._W(Pv["U1"], Pv["V1"]).detach().clone())
        S["S4_layer1_write_reader_indep"].append(
            float((W1s[0] - W1s[1]).abs().max()))

        # —— S5：**任务梯度不进入写入通道**（层-1 的 `U1/V1` 梯度应为 0）
        Pv = T42.T37._mk(sd, phi, train=True)
        L, _ = T42.trace42(Pv, N_REP, "learned", sd, phi, win=TRAIN_WIN)
        L.backward()
        gu = float(Pv["U1"].grad.norm()) if Pv["U1"].grad is not None else 0.0
        gv = float(Pv["V1"].grad.norm()) if Pv["V1"].grad is not None else 0.0
        S["S5_grad_flow_written_channel"].append(max(gu, gv))
        phi["Wq1"].grad = None
    for k, v in S.items():
        out[k] = {"max": max(v), "mean": statistics.fmean(v)}
    return out


# ══════════════ (M) 效应量
def measure(sds, eps_list, direct_steps):
    rows = []
    for sd in sds:
        P0 = T42.T37._mk(sd, None)
        phi = T42.reader_init(sd)
        # —— 任务梯度（训练窗口）
        Pv = T42.T37._mk(sd, phi, train=True)
        L, _ = T42.trace42(Pv, N_REP, "learned", sd, phi, win=TRAIN_WIN)
        L.backward()
        g_task = phi["Wq1"].grad.detach().clone()
        g_task_b = phi["bq1"].grad.detach().clone()
        # —— 寻址方向（正确键 / 随机键）
        A_ok, g_addr, g_addr_b = addr_grad(P0, phi)
        A_rd, g_addr_r, g_addr_r_b = addr_grad(P0, phi, key=rand_key(P0, sd))
        rho = cos2(g_task, g_addr)
        rho_rand = cos2(g_task, g_addr_r)
        # —— 有限步检查：沿 `ĝ_addr` 走一步，L 是否下降
        u = g_addr / g_task.new_tensor(float(g_addr.norm()))
        rel = float(phi["Wq1"].norm())
        dL = {}
        for eps in eps_list:
            ph2 = {k: v.detach().clone() for k, v in phi.items()}
            ph2["Wq1"] = phi["Wq1"].detach() + eps * rel * u
            ph2["bq1"] = phi["bq1"].detach() + eps * rel * u.new_tensor(0.0)
            P2 = T42.T37._mk(sd, ph2, train=True)
            L2, _ = T42.trace42(P2, N_REP, "learned", sd, ph2, win=TRAIN_WIN)
            dL[str(eps)] = float(L2) - float(L)
        rows.append({"sd": int(sd), "L": float(L),
                     "A_init": A_ok, "A_init_randkey": A_rd,
                     "rho": rho, "rho_rand": rho_rand,
                     "gn_task": float(g_task.norm()), "gn_addr": float(g_addr.norm()),
                     "gn_task_b": float(g_task_b.norm()),
                     "dL_eps": dL})
    return rows


def direct_reach(sd, steps=250, lr=0.03):
    r"""**正控**：直接优化对齐目标 `−A`（只动 `Wq1,bq1`）⇒ 看能否达到 `cos≈1`。
    ★ `A` 无需轨迹 ⇒ ⇒ 极便宜。"""
    P = T42.T37._mk(sd, None)
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
    return {"A_final": statistics.fmean(cs), "min_A": min(cs), "max_A": max(cs),
            "W_dist_to_Pk1": float((W.detach() - P["Pk1"]).norm())}


# ══════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=PROBE_N)
    ap.add_argument("--out", default=f"{ROOT}/dir43/out/DIR43_probe.json")
    ap.add_argument("--rows", default=f"{ROOT}/dir43/out/DIR43_probe_rows.jsonl")
    a = ap.parse_args()
    sds = PROBE_INST_OFF[:a.n]
    eps_list = [0.01, 0.03, 0.1, 0.3]
    t0 = time.time()

    print("[probe] 结构核实 …", flush=True)
    st = structural_checks(sds[:min(16, len(sds))])

    print("[probe] 效应量 …", flush=True)
    rows = measure(sds, eps_list, direct_steps=250)

    print("[probe] 正控（直接优化对齐） …", flush=True)
    dr = [direct_reach(sd) for sd in sds[:8]]

    rho = [r["rho"] for r in rows]
    rhor = [r["rho_rand"] for r in rows]
    d = [a_ - b_ for a_, b_ in zip(rho, rhor)]
    summ = {
        "n": len(rows),
        "rho_mean": statistics.fmean(rho), "rho_sd": statistics.stdev(rho),
        "rho_se": statistics.stdev(rho) / len(rho) ** 0.5,
        "rho_rand_mean": statistics.fmean(rhor),
        "rho_diff_mean": statistics.fmean(d),
        "rho_diff_sd": statistics.stdev(d),
        "rho_diff_se": statistics.stdev(d) / len(d) ** 0.5,
        "A_init_mean": statistics.fmean([r["A_init"] for r in rows]),
        "A_init_randkey_mean": statistics.fmean([r["A_init_randkey"] for r in rows]),
        "gn_task_mean": statistics.fmean([r["gn_task"] for r in rows]),
        "gn_addr_mean": statistics.fmean([r["gn_addr"] for r in rows]),
        "direct_A_final_mean": statistics.fmean([x["A_final"] for x in dr]),
        "direct_A_final_min": min(x["A_final"] for x in dr),
    }
    # —— 有限步检查：每个 ε 的平均 ΔL（负 = 沿寻址方向走，L 下降）
    for eps in eps_list:
        v = [r["dL_eps"][str(eps)] for r in rows]
        summ[f"dL_eps{eps}_mean"] = statistics.fmean(v)
        summ[f"dL_eps{eps}_sd"] = statistics.stdev(v)
        summ[f"dL_eps{eps}_negfrac"] = sum(1 for x in v if x < 0) / len(v)
    # —— 功效：给定当前效应，`3se` 需要多大 N 才能把它分出来
    eff = abs(summ["rho_mean"])
    summ["n_for_3se_at_probe_effect"] = (3.0 * summ["rho_sd"] / eff) ** 2 if eff > 0 else None

    pathlib.Path(os.path.dirname(a.out)).mkdir(parents=True, exist_ok=True)
    out = {"kind": "DIR43_probe", "n": len(rows), "inst_off": PROBE_INST_OFF[:a.n],
           "probe_inst_off": PROBE_INST_OFF[:a.n],
           "structural": st, "summary": summ, "direct": dr,
           "eps_list": eps_list, "wall_s": round(time.time() - t0, 2)}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(a.rows, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"structural": st, "summary": summ}, indent=2, ensure_ascii=False))
    print(f"[probe] {a.out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
