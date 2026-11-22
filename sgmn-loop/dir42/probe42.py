#!/opt/pytorch/bin/python
r"""DIR-42 · **开卡前探针**（**不进判据、不计 `SUCCEEDED`**）

目的（DIR-40 探针甲的未测项 + DIR-41 的收窄）：
  甲只测了**样本内**的「per-instance 读控制器」⇒ 48/48 过拟合 ⇒ **不能**当出样本结论；
  且甲**没有**内容置换对照 ⇒ 「损失下降」≠「学会使用关联」。
  ⇒ 本探针在**时序留出**（train 窗口 `t∈[1,8]`、eval 窗口 `t∈[9,15]`；eval 窗口的损失项
  **从未进入**训练目标）下重测第三格（逐实例 `Pk1` × **每实例自带读器**），
  并加上：① 内容置换对照 `learned_rc` ② 对齐读数 `cos⟨q,k⟩` ③ 与甲的**样本内**读数对照。

★ 本探针**只用装置**（`dir31`/`dir33`/`dir37`）⇒ **不改**任何上游文件。
★ 探针实例段：`SEED0+22000..22015`（16 个）⇒ 与全部历史段**不相交**（Stage A 再核）。
"""
import json
import math
import os
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir37")
import R1_task33 as R           # 任务层（逐字未改）
import R1_grad as M             # 装置（逐字未改）
import R1_task37 as T37         # 测量层（`_mk` / `trace` 逐字未改）

OUT = "/home/ec2-user/sgmn-loop/dir42/out"
D_H, D_K, D_V = R.D_H, R.D_K, R.D_V
SEED0 = M.SEED0
T = M.T_STEPS
N_REP = 16
LR = 0.03

# 探针的实例段（**与历史段不相交**）
PROBE_OFF = list(range(22000, 22008))          # 8 个
READER_SEED_OFFSET = 10_000_000                # ★ 读器初值的**独立**生成器流

# 训练/评估的损失窗口（t 为损失项下标：t∈1..T-1）
TRAIN_WIN = (1, 8)
EVAL_WIN = (9, 15)


# ══════════ 读器初值（**每实例**确定性；同一初值在 `rand` 与 `learned` 两臂间逐位配对）
def reader_init(sd):
    r"""`dir39.init_params()` 的**逐字派生**，但生成器种子 = 实例种子 + 独立偏移。

    ★ 绘制顺序与 `init_params()` **逐字相同**（`Wo1` → `bo1` → `Wq1`；`bq1` 为零）。
    ★ 生成器种子**独立于装置的实例种子流**（+10^7）⇒ 读器初值与该实例的 `Pk1` **无构造相关**。
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


# ══════════ `trace40` 的**逐字派生**：只加（i）**损失窗口**（ii）每实例读器（`phi` 即该实例的读器）
def trace42(P, n_rep, qsrc, sd, phi, want_diag=False, rand_content=False, win=None):
    r"""`dir40.trace40` 的**逐字派生**：唯一新增 = **损失窗口** `win=(lo,hi)`（含端点；
    `None` ⇒ 全部 `t≥1`）。轨迹**照常走完 `T` 步**（写侧不因窗口而改变）⇒ 窗口只决定
    **哪些损失项进入求和** ⇒ ⇒ eval 窗口的损失项**从未进入**训练目标。

    ★ 读器 = `phi`（**每实例一份**）：`g1/Wo1/bo1`（头）＋ `Wq1/bq1`（控制器）。
      `qsrc="oracle"` ⇒ `q = tanh(h · Pk1)`（**该实例自己的** `Pk1`）；
      `"rand"` ⇒ `q = tanh(c · Wq1 + bq1)`（控制器**不在优化器里** ⇒ 逐位冻结）；
      `"learned"` ⇒ 同上，但控制器**被任务梯度训过**。
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


def train_reader42(qsrc, sd, steps, train_win=TRAIN_WIN, lr=LR):
    r"""**每实例**读器训练：头 `(g1,Wo1,bo1)` 必训；控制器 `(Wq1,bq1)` **只在 `learned` 档训**。
    ⇒ `rand` 档的控制器**逐位冻结**（不在优化器里）⇒ 与 `learned` 档**同初值** ⇒ 逐位配对。
    ★ 训练损失**只**累加 `train_win` 窗口的损失项。
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
    return {"L": float(L), **{k: v for k, v in dg.items()}}


def _stat(d):
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if len(d) > 1 else 0.0
    se = sd / len(d) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": 3.0 * se, "n": len(d)}


def _one_job(job):
    r"""一个作业 = 一个 (配置, 臂, 实例) ⇒ 训练 + 三次评估。**逐实例独立** ⇒ 可并行。"""
    ci, arm, si, steps, tw = job
    sd = SEED0 + PROBE_OFF[si]
    phi, meta = train_reader42(arm, sd, steps, train_win=tw)
    e = eval_reader42(arm, phi, sd, win=EVAL_WIN)
    e_in = eval_reader42(arm, phi, sd, win=None)      # 样本内（对照甲）
    e_tr = eval_reader42(arm, phi, sd, win=tw)       # 训练窗口（诊断）
    row = {"seed": sd, "L_eval": e["L"], "L_full_insample": e_in["L"],
           "L_trainwin": e_tr["L"], "cos": e["cos_q_k"],
           "dWq1": meta["dWq1_from_init"], "gn_Wq1": meta["gradnorms"].get("Wq1", 0.0)}
    if arm == "learned":
        e_rc = eval_reader42(arm, phi, sd, win=EVAL_WIN, rand_content=True)
        row["L_eval_rc"] = e_rc["L"]
    return ci, arm, si, row


def _job_key(ci, arm, si):
    return f"c{ci}|{arm}|i{si}"


def _load_done_rows(path):
    r"""★ **增量落盘 + 可续跑**：已完成的作业行从 `jsonl` 读回 ⇒ 重跑时**跳过**。"""
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[(rec["ci"], rec["arm"], rec["si"])] = rec["row"]
    return done


def _run_job_and_persist(job, path):
    r"""跑一个作业并**立即**追加到 `jsonl`（⇒ 后段聚合的 bug 不会毁掉已算的读数）。

    ★ **多进程安全的追加**：用 `os.open(..., O_APPEND)` + **单次** `os.write` ⇒
      每行是一个原子追加（POSIX `O_APPEND`）⇒ 不会互相截断。
    """
    ci, arm, si, steps, tw = job
    row = _one_job(job)[3]
    line = (json.dumps({"ci": ci, "arm": arm, "si": si, "row": row},
                        ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o664)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    return ci, arm, si, row


def probe():
    t0 = time.time()
    seeds = [SEED0 + o for o in PROBE_OFF]
    out = {"note": "DIR-42 开卡前探针：不进判据、不计 SUCCEEDED",
           "probe_off": [PROBE_OFF[0], PROBE_OFF[-1]], "n_instances": len(seeds),
           "train_win": list(TRAIN_WIN), "eval_win": list(EVAL_WIN),
           "n_rep": N_REP, "lr": LR, "reader_seed_offset": READER_SEED_OFFSET,
           "configs": {}}
    configs = [(100, TRAIN_WIN), (250, TRAIN_WIN), (500, TRAIN_WIN)]
    ROWS = os.path.join(OUT, "DIR42_probe_rows.jsonl")
    os.makedirs(OUT, exist_ok=True)
    done = _load_done_rows(ROWS)
    jobs = [(ci, arm, si, steps, tw)
            for ci, (steps, tw) in enumerate(configs)
            for arm in ("oracle", "rand", "learned")
            for si in range(len(seeds))]
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"probe：{len(jobs)} 个作业待跑（已完成 {len(done)} 个）"
          f"（4 路并行，逐作业**增量落盘**）", flush=True)
    if jobs:
        import multiprocessing as mp
        with mp.Pool(4) as pool:
            results = pool.starmap(_run_job_and_persist, [(j, ROWS) for j in jobs])
        for ci, arm, si, row in results:
            done[(ci, arm, si)] = row
    else:
        results = [(ci, arm, si, done[(ci, arm, si)])
                   for ci in range(len(configs))
                   for arm in ("oracle", "rand", "learned") for si in range(len(seeds))]
    buckets = {}
    for ci, arm, si, row in results:
        buckets.setdefault((ci, arm), []).append(row)
    for ci, (steps, tw) in enumerate(configs):
        key = f"s{steps}_tw{tw[0]}-{tw[1]}"
        rec = {"steps": steps, "train_win": list(tw), "arms": {}}
        for arm in ("oracle", "rand", "learned"):
            rows = sorted(buckets.get((ci, arm), []), key=lambda r: r["seed"])
            Ls = [r["L_eval"] for r in rows]
            rec["arms"][arm] = {"mean_L_eval": statistics.fmean(Ls),
                                "mean_L_full_insample": statistics.fmean(
                                    [r["L_full_insample"] for r in rows]),
                                "mean_L_trainwin": statistics.fmean([r["L_trainwin"] for r in rows]),
                                "mean_cos": statistics.fmean([r["cos"] for r in rows]),
                                "rows": rows}
        # 配对统计
        Lr = {r["seed"]: r["L_eval"] for r in rec["arms"]["rand"]["rows"]}
        Ll = {r["seed"]: r["L_eval"] for r in rec["arms"]["learned"]["rows"]}
        Lo = {r["seed"]: r["L_eval"] for r in rec["arms"]["oracle"]["rows"]}
        seen_r = {r["seed"]: r["L_trainwin"] for r in rec["arms"]["rand"]["rows"]}
        seen_l = {r["seed"]: r["L_trainwin"] for r in rec["arms"]["learned"]["rows"]}
        R = statistics.fmean([Lr[s] - Lo[s] for s in seeds])
        d_rand = _stat([Lr[s] - Ll[s] for s in seeds])
        d_orc = _stat([Lo[s] - Ll[s] for s in seeds])
        d_seen = _stat([seen_r[s] - seen_l[s] for s in seeds])
        rc = [r["L_eval_rc"] - r["L_eval"] for r in rec["arms"]["learned"]["rows"]]
        cst = _stat(rc)
        rec["stats"] = {"R_oracle_rand": R,
                        "delta_rand_learned": d_rand, "delta_oracle_learned": d_orc,
                        "delta_seen_window": d_seen,
                        "content_C_learned_rc": cst,
                        "P1_signif": d_rand["mean"] >= d_rand["3se"],
                        "P2_floor_0.2R": d_rand["mean"] >= 0.20 * R,
                        "P3_content": cst["mean"] >= cst["3se"]}
        rec["secs"] = time.time() - t0
        out["configs"][key] = rec
        print(f"[{key}] " + " | ".join(
            f"{a}: L_eval={rec['arms'][a]['mean_L_eval']:.4f} cos={rec['arms'][a]['mean_cos']:+.3f}"
            for a in ("oracle", "rand", "learned")), flush=True)
        s = rec["stats"]
        print(f"    Δ(rand-learned) out={s['delta_rand_learned']['mean']:.4f} "
              f"3se={s['delta_rand_learned']['3se']:.4f} | 已见窗 Δ={s['delta_seen_window']['mean']:.4f} "
              f"| C(rc-learned)={s['content_C_learned_rc']['mean']:.4f} "
              f"3se={s['content_C_learned_rc']['3se']:.4f} | R={s['R_oracle_rand']:.4f}", flush=True)
    out["total_secs"] = time.time() - t0
    with open(os.path.join(OUT, "DIR42_probe.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"probe 完成：{out['total_secs']:.0f}s ⇒ {OUT}/DIR42_probe.json")
    return out


if __name__ == "__main__":
    probe()
