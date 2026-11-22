#!/opt/pytorch/bin/python
r"""DIR-39 · **探针**（开卡前 · **不进判据**）—— 层 1 的「**查询来源**」单一变量化。

★ 方向来源：框架 §8「写入负责形成关联，**读取负责学习如何使用关联**」+ §7.2「持久化与 LLM 协同」行的
  「前向任务收益」+「集成接口验收」行的「读取参数的非零有限梯度」；派生自 D-62 / D-62-CORR。
★ **本脚本只做便宜探针**：三档读源、同一写侧（`n_rep=16`）、小训练集与小评估集 ⇒ 给出**档位/量级/机制读数**，
  **不产生任何判据裁决**（判据阈值待探针之后由卡片冻结）。

**唯一自变量 = 层 1 的查询来源 `q`**（三档）：
  ① `oracle`：`q = tanh(h · P_k1)`（**逐字** = 现行装置的层 1 读；per-instance 的“真值”读）
  ② `rand`  ：`q = tanh(h · Wq1 + bq1)`，其中 `(Wq1,bq1)` = **冻结的共享随机初值**（与 ③ **同一初值**）
  ③ `learned`：`q = tanh(h · Wq1 + bq1)`，其中 `(Wq1,bq1)` = **被任务梯度训过**（共享；初值同 ②）

★ **`c` 在三档里完全相同**：`c = h`（层 1 的前激活，即 oracle 那一档的 `c`）⇒ 只换 `q`。
★ **写侧逐字冻结在 DIR-38 的 `n_rep=16`**（同一步内重复装置自己的同步因子更新 16 次）。
★ 每档**各自训一份头**（`g1,Wo1,bo1`；③ 额外训 `Wq1,bq1`）⇒ 三档都是「各自最优」的系统，
  **唯一被故意改变的**是读源（训练协议、种子、初值、优化器、步数、lr 全部相同）。

用法：probe
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
import R1_task33 as R          # 任务层（逐字未改）
import R1_grad as M            # 装置（逐字未改）
import R1_task37 as T37        # DIR-37 的测量层（_mk / 训练协议 / 迹线）

OUT = "/home/ec2-user/sgmn-loop/dir39/out"
D_H, D_V, D_K, RANK, T = R.D_H, R.D_V, R.D_K, M.R, M.T_STEPS
SEED0 = M.SEED0

# ══════════ 种子段（**全新**，与 DIR-31…38 的全部段零交集）
PROBE_TRAIN = [SEED0 + 15000 + i for i in range(64)]   # 15000..15063
PROBE_EVAL = [SEED0 + 15100 + i for i in range(48)]     # 15100..15147
N_REP = 16                                             # ★ 写侧冻结在 DIR-38 的设置
LEVELS = ("oracle", "rand", "learned")
STEPS, LR = 250, 0.03                                  # 与 dir33/dir37 的训练协议逐字相同


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


def trace39(P, n_rep, qsrc, sd, phi, want_diag=False):
    r"""`R1_task38.trace38` 的派生：**只**把层 1 的 `q` 换成可切换的读源，其余逐字相同。

    ★ 层 1：`c = h`（**三档完全一致** ⇒ 只换 `q`）；层 2：`c = hp_prev.detach()`、`q = tanh(c·Wq2+bq2)`（逐字未改）。
    ★ 读：`y = W.detach() @ q`（写通道隔离 ⇒ 任务梯度不进写侧；DIR-31 的 C3 语义保持）。
    ★ 写：`n_rep` 次重复的同步因子更新（每次按**更新后**的 `W` 重算 `G`）。
    """
    L = P["a"].new_zeros(())
    gs = R.targets(P)
    h_pre_prev = [None, None]
    q1s, k1s, c1s = [], [], []
    for t in range(T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if li == 0:
                c = h                                   # ★ **三档相同**（oracle 的 c）
                if qsrc == "oracle":
                    q = torch.tanh(h @ P["Pk1"])
                else:
                    q = torch.tanh(c @ P["Wq1"] + P["bq1"])
                q1s.append(q.detach().clone())
                c1s.append(c.detach().clone())
            else:
                c = hp_prev.detach()
                q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            y = W.detach() @ q
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            gam = P[f"g{l}"]
            hp[li] = h + gam * m
            hp_prev = hp[li]
        if t >= 1:
            L = L + ((R.pout() @ hp[0] - gs[t - 1]) ** 2).sum()
        # ── 写（冻结在 `n_rep` 次）
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
            if li == 0:
                pass
        upd = {kk: vv.detach() for kk, vv in upd.items()}
        P.update(upd)
        h_pre_prev = h_pre
    diag = {}
    if want_diag:
        # 机制读数：层 1 的查询与「上一步写入的键」的对齐 ⟨q, k⟩/‖k‖（t≥1 用 k_{t-1}）
        al = []
        for t in range(1, len(q1s)):
            k_prev = k1s[t - 1]
            nk = float(k_prev.norm())
            if nk > 1e-300:
                al.append(float((q1s[t] @ k_prev)) / nk)
        diag["align_q_k"] = statistics.fmean(al) if al else float("nan")
    return L, diag


def train_level(qsrc, seeds, steps=STEPS, lr=LR):
    r"""训一档：参数集按档变化，**其余协议逐字相同**（同生成器 4242 / Adam / lr / 步数 / 种子序）。

    ★ `oracle`/`rand` 只训头（`g1,Wo1,bo1`）；`learned` 额外训 `(Wq1,bq1)` ⇒ ③ 与 ② 的**唯一差别**是「训不训」。
    """
    phi = init_params()
    names = ["g1", "Wo1", "bo1"] + (["Wq1", "bq1"] if qsrc == "learned" else [])
    opt = torch.optim.Adam([phi[n] for n in names], lr=lr)
    gnorm = {"Wq1": 0.0, "bq1": 0.0}
    for _ in range(steps):
        opt.zero_grad()
        for sd in seeds:
            P = T37._mk(sd, phi, train=True)
            L, _ = trace39(P, N_REP, qsrc, sd, phi)
            (L / len(seeds)).backward()
        if qsrc == "learned":
            for n in ("Wq1", "bq1"):
                if phi[n].grad is not None:
                    gnorm[n] = max(gnorm[n], float(phi[n].grad.norm()))
        opt.step()
    out = {k: v.detach().clone() for k, v in phi.items()}
    meta = {"gradnorm_Wq1": gnorm["Wq1"], "gradnorm_bq1": gnorm["bq1"]}
    return out, meta


def eval_level(qsrc, phi, seeds):
    Ls, als = [], []
    for sd in seeds:
        P = T37._mk(sd, phi, train=False)
        L, dg = trace39(P, N_REP, qsrc, sd, phi, want_diag=True)
        Ls.append(float(L))
        als.append(dg["align_q_k"])
    return {"mean_L": statistics.fmean(Ls), "sd_L": statistics.stdev(Ls),
            "L_series": Ls, "align_q_k": statistics.fmean(als), "align_series": als}


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    se = sd / len(d) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": 3.0 * se, "n": len(d)}


def main():
    t0 = time.time()
    out = {"note": "DIR-39 探针（开卡前 · 不进判据 · 不给任何裁决）",
           "n_rep": N_REP, "train_seg": [15000, 15063], "eval_seg": [15100, 15147],
           "n_train": len(PROBE_TRAIN), "n_eval": len(PROBE_EVAL),
           "md5_device": _md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py"),
           "md5_task": _md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py")}

    # ── 自检 A：`oracle` 档 + n_rep=1 + DIR-37 的头协议 ⇒ 必须与 DIR-38 的 `n1` 臂**逐位相同**
    phi_dev = T37.train_head("dev", PROBE_TRAIN[:8])
    P = T37._mk(PROBE_EVAL[0], phi_dev)
    L_ref, _ = T37.trace(P, "dev", PROBE_EVAL[0], phi_dev)
    P = T37._mk(PROBE_EVAL[0], phi_dev)
    L_38, _ = trace39(P, 1, "oracle", PROBE_EVAL[0], phi_dev)
    out["anchor_nrep1_vs_dir37"] = abs(float(L_ref) - float(L_38))
    print(f"[自检] n_rep=1 的 oracle 档 vs DIR-37 装置臂：|Δ| = {out['anchor_nrep1_vs_dir37']:.3e}")

    # ── 三档：各自训一份
    phis, metas, evs = {}, {}, {}
    for lv in LEVELS:
        t1 = time.time()
        phis[lv], metas[lv] = train_level(lv, PROBE_TRAIN)
        evs[lv] = eval_level(lv, phis[lv], PROBE_EVAL)
        print(f"[{lv:8s}] 训练 {time.time()-t1:.0f}s · L={evs[lv]['mean_L']:.6f} "
              f"(sd {evs[lv]['sd_L']:.4f}) · align⟨q,k⟩={evs[lv]['align_q_k']:.4f}", flush=True)

    # ── 行为读数：③ 的读控制器确实被训动了；② 与 ③ 的初值逐位相同
    init = init_params()
    out["dWq1_learned_vs_init"] = float((phis["learned"]["Wq1"] - init["Wq1"]).norm())
    out["dWq1_rand_vs_init"] = float((phis["rand"]["Wq1"] - init["Wq1"]).norm())
    out["dWq1_learned_vs_rand"] = float((phis["learned"]["Wq1"] - phis["rand"]["Wq1"]).norm())
    out["dbq1_learned_vs_init"] = float((phis["learned"]["bq1"] - init["bq1"]).norm())
    out["gradnorm_Wq1_learned"] = metas["learned"]["gradnorm_Wq1"]
    out["gradnorm_bq1_learned"] = metas["learned"]["gradnorm_bq1"]
    out["dWo1_across_levels"] = float((phis["oracle"]["Wo1"] - phis["rand"]["Wo1"]).norm())

    # ── 配对读数（**量级**，不是裁决）
    out["paired_rand_minus_learned"] = paired(evs["rand"]["L_series"], evs["learned"]["L_series"])
    out["paired_rand_minus_oracle"] = paired(evs["rand"]["L_series"], evs["oracle"]["L_series"])
    out["paired_learned_minus_oracle"] = paired(evs["learned"]["L_series"], evs["oracle"]["L_series"])
    # 功效参考：n=400 时 3se = 3·sd/√400
    sd_rl = out["paired_rand_minus_learned"]["sd"]
    out["power_ref_3se_at_n400"] = 3.0 * sd_rl / math.sqrt(400)
    out["power_ref_detect_frac_of_range"] = (
        out["power_ref_3se_at_n400"] / out["paired_rand_minus_oracle"]["mean"]
        if out["paired_rand_minus_oracle"]["mean"] > 0 else None)
    out["arms"] = {lv: {"mean_L": evs[lv]["mean_L"], "sd_L": evs[lv]["sd_L"],
                         "align_q_k": evs[lv]["align_q_k"]} for lv in LEVELS}
    out["L_series"] = {lv: evs[lv]["L_series"] for lv in LEVELS}
    out["secs"] = time.time() - t0
    out["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "DIR39_probe.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n══ 探针读数（**不是判据**）══")
    for lv in LEVELS:
        print(f"  L[{lv:8s}] = {evs[lv]['mean_L']:.6f}   align = {evs[lv]['align_q_k']:.4f}")
    p = out["paired_rand_minus_learned"]
    print(f"  配对 rand−learned: mean={p['mean']:.6f} sd={p['sd']:.6f} 3se={p['3se']:.6f}")
    po = out["paired_rand_minus_oracle"]
    print(f"  配对 rand−oracle : mean={po['mean']:.6f} sd={po['sd']:.6f} 3se={po['3se']:.6f}")
    print(f"  n=400 的功效参考 3se ≈ {out['power_ref_3se_at_n400']:.6f}"
          f"（= 可寻址区间的 {out['power_ref_detect_frac_of_range']:.4f}）")
    print(f"  ‖ΔWq1‖（learned vs 初值）= {out['dWq1_learned_vs_init']:.6f}"
          f" · rand vs 初值 = {out['dWq1_rand_vs_init']:.6f}")
    print(f"  ‖∇Wq1‖ = {out['gradnorm_Wq1_learned']:.3e} · ‖∇bq1‖ = {out['gradnorm_bq1_learned']:.3e}")
    print(f"  耗时 {out['secs']:.0f}s")


if __name__ == "__main__":
    main()
