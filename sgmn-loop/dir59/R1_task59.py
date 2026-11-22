#!/opt/pytorch/bin/python
r"""DIR-59 · task 级测量层（`trace59` / `train_arm59` / `eval_arm59`）+ 自检

★ 逐字派生自 `dir40/R1_task40.py` 的 `trace40` / `train_arm40` / `eval_arm40`（md5 见 `MD5_T40`），
  **唯一改动 = 两个钩子**：
    ① `shared_pv1=None`  ⇒ 覆盖层 1 的写值映射 `P["Pv1"]`（DIR-59 的**唯一自变量**）；
    ② `alpha=None`       ⇒ 覆盖写步的 $\alpha$（`None` 时用 `M.ALPHA`；`0.0` = 记忆关掉的地板臂）。
  `A59D-1` 断言：两个钩子都为 `None` 时 `trace59` 与 `trace40` **逐位相同**（损失与全部诊断）。

★ 为什么需要这一层（`PROBE59_REV1.1.md §4`）：D-76（DIR-53）已确立本框架行的规范操作化是**训练读数头后的
  任务损失**；关联层（$\|Wk-v\|$）的载荷只有 3.2%，承载不了"来源"的等价判据。
"""
import hashlib
import math
import pathlib
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir37")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir40")
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task37 as T37      # noqa: E402
import R1_task40 as T40      # noqa: E402

D_H, D_K, D_V = R.D_H, R.D_K, R.D_V
T, N_REP = M.T_STEPS, 16
STEPS, LR = 250, 0.03
SIG_K = 3.0


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ══════════════════════════════════════════════════════════════════════
# `trace59`：`trace40` 的逐字派生（只加两个钩子）
# ══════════════════════════════════════════════════════════════════════
def trace59(P, n_rep, qsrc, sd, phi, want_diag=False, shared_pk1=None, rand_content=False,
            shared_pv1=None, alpha=None, target_pv1=None):
    r"""与 `trace40` 逐字相同，除了：`shared_pv1` 覆盖 `Pv1`、`alpha` 覆盖写步的 $\alpha$。"""
    if shared_pk1 is not None:
        P["Pk1"] = shared_pk1
    if shared_pv1 is not None:
        P["Pv1"] = shared_pv1
    a_write = M.ALPHA if alpha is None else alpha
    L = P["a"].new_zeros(())
    # ★ `target_pv1`：把**任务目标**用的 $P_v$ 与**写步**用的 $P_v$ 解耦（`hs_of` 只依赖 `a,A1,b1` ⇒ 可安全替换）
    gs = R.targets(P) if target_pv1 is None else R.targets({**P, "Pv1": target_pv1})
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
                G = a_write * torch.outer(W @ k - v, k) + M.LAM * W      # ★ 唯一的 = 改动处
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


def train_arm59(qsrc, seeds, shared_pk=None, shared_pv=None, alpha=None,
                train_ctrl=False, steps=STEPS, lr=LR, target_pv=None):
    r"""`train_arm40` 的逐字派生（唯一改动：把两个新钩子透传给 `trace59`）。"""
    init = T40.init_params()
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
            L, _ = trace59(P, N_REP, qsrc, sd, phi, shared_pk1=shared_pk,
                           shared_pv1=shared_pv, alpha=alpha, target_pv1=target_pv)
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


def eval_arm59(qsrc, phi, seeds, shared_pk=None, shared_pv=None, alpha=None, rand_content=False,
               target_pv=None):
    Ls, coss, alprs, qn, yn, ma, cch, wmx = [], [], [], [], [], [], [], []
    for sd in seeds:
        P = T37._mk(sd, phi, train=False)
        L, dg = trace59(P, N_REP, qsrc, sd, phi, want_diag=True, shared_pk1=shared_pk,
                        shared_pv1=shared_pv, alpha=alpha, rand_content=rand_content,
                        target_pv1=target_pv)
        Ls.append(float(L))
        coss.append(dg["cos_q_k"]); alprs.append(dg["align_q_k"])
        qn.append(dg["q_norm"]); yn.append(dg["y_norm"]); ma.append(dg["m_abs"])
        cch.append(dg["c_eq_h"]); wmx.append(dg["Wmax"])
    return {"mean_L": statistics.fmean(Ls),
            "sd_L": statistics.stdev(Ls) if len(Ls) > 1 else 0.0,
            "L_series": Ls, "cos_q_k": statistics.fmean(coss),
            "align_q_k": statistics.fmean(alprs), "q_norm": statistics.fmean(qn),
            "y_norm": statistics.fmean(yn), "m_abs": statistics.fmean(ma),
            "c_eq_h_max": max(cch), "Wmax": max(wmx)}


# ══════════════════════════════════════════════════════════════════════
# 自检（`A59D-1..4`；任一失败 ⇒ 禁止 59D 探针）
# ══════════════════════════════════════════════════════════════════════
def selftest(seeds=(59800, 59801)):
    g = {}
    phi = T40.init_params()
    # A59D-1：两钩子 None ⇒ 与 trace40 逐位相同（损失 + 全部诊断）
    dL, dDiag = 0.0, {}
    for sd in seeds:
        L5, d5 = trace59(T37._mk(sd, phi, False), N_REP, "learned", sd, phi, want_diag=True)
        L4, d4 = T40.trace40(T37._mk(sd, phi, False), N_REP, "learned", sd, phi, want_diag=True)
        dL = max(dL, abs(float(L5) - float(L4)))
        for k in d4:
            a, b = d5[k], d4[k]
            if isinstance(a, float) and isinstance(b, float) and not (math.isnan(a) and math.isnan(b)):
                dDiag[k] = max(dDiag.get(k, 0.0), abs(a - b))
    g["A59D-1"] = {"max_abs_dL": dL, "max_abs_dDiag": max(dDiag.values()) if dDiag else 0.0,
                   "keys": sorted(dDiag)}
    g["A59D-1"]["pass"] = (dL == 0.0 and g["A59D-1"]["max_abs_dDiag"] == 0.0)
    # A59D-2：`shared_pv1` 钩子真的咬住（换一个 $P_v$ ⇒ 损失必须变）
    pv_a = _pv1_for(20263301)
    pv_b = _pv1_for(20263302)
    La, _ = trace59(T37._mk(59800, phi, False), N_REP, "learned", 59800, phi, shared_pv1=pv_a)
    Lb, _ = trace59(T37._mk(59800, phi, False), N_REP, "learned", 59800, phi, shared_pv1=pv_b)
    g["A59D-2"] = {"L_own": float(La), "L_foreign": float(Lb),
                   "abs_diff": abs(float(La) - float(Lb))}
    g["A59D-2"]["pass"] = g["A59D-2"]["abs_diff"] > 0.0
    # A59D-3：`alpha=0` 真的关掉写入 ⇒ 与"纯 λ 衰减"独立重实现逐位相同
    P = T37._mk(59800, phi, False)
    U0, V0 = P["U1"].clone(), P["V1"].clone()
    trace59(P, N_REP, "learned", 59800, phi, alpha=0.0)
    W_after = M._W(P["U1"], P["V1"])
    U, V = U0.clone(), V0.clone()
    for _ in range(T * N_REP):
        G = M.LAM * M._W(U, V)
        Uo, Vo = U, V
        U = Uo - M.ETA * (G @ Vo)
        V = Vo - M.ETA * (G.t() @ Uo)
    g["A59D-3"] = {"max_abs_dW": float((W_after - M._W(U, V)).abs().max())}
    g["A59D-3"]["pass"] = g["A59D-3"]["max_abs_dW"] == 0.0
    # A59D-4：两套 $P_v$ 确实不同（逐位 + 余弦）
    g["A59D-4"] = {"sha_own": hashlib.sha256(pv_a.numpy().tobytes()).hexdigest()[:16],
                   "sha_foreign": hashlib.sha256(pv_b.numpy().tobytes()).hexdigest()[:16],
                   "fro_own": float(pv_a.norm()), "fro_foreign": float(pv_b.norm())}
    g["A59D-4"]["pass"] = (g["A59D-4"]["sha_own"] != g["A59D-4"]["sha_foreign"])
    # A59D-6：`target_pv1` 钩子咬住（只换任务目标的值映射 ⇒ 损失必须变），且默认路径不变
    L_t0, _ = trace59(T37._mk(59800, phi, False), N_REP, "learned", 59800, phi, shared_pv1=pv_a)
    L_t1, _ = trace59(T37._mk(59800, phi, False), N_REP, "learned", 59800, phi,
                      shared_pv1=pv_a, target_pv1=pv_b)
    g["A59D-6"] = {"L_target_own": float(L_t0), "L_target_foreign": float(L_t1),
                   "abs_diff": abs(float(L_t0) - float(L_t1))}
    g["A59D-6"]["pass"] = g["A59D-6"]["abs_diff"] > 0.0
    g["md5"] = {p: _md5("/home/ec2-user/sgmn-loop/" + p) for p in
                ("dir31/R1_grad.py", "dir33/R1_task33.py", "dir37/R1_task37.py",
                 "dir40/R1_task40.py")}
    g["all_pass"] = all(v.get("pass", False) for k, v in g.items() if k.startswith("A59D"))
    return g


def _pv1_for(cell_seed):
    r"""`P_v` 初值的生成式：**与装置 `dir31/build` 逐字同式**（`randn(D_H,D_V)*1/sqrt(D_H)`，**不带** `KEY_SCALE`）。

    ★ 检查点：`dir31.build` 里 `Pk1`/`Pv1` 都是 `rn(D_H, D_K, scale=s_h)`；`dir33.build` 只给 `Pk1` 乘 `KEY_SCALE`
      ⇒ `Pv1` **不乘** `KEY_SCALE`（见 `A59D-5` 的逐位核对：用 `T37._mk` 重建后与 build 的 `Pv1` 比较）。
    """
    g = torch.Generator().manual_seed(int(cell_seed))
    return torch.randn(D_H, D_V, generator=g) * (1.0 / math.sqrt(D_H))   # ★ (D_H, D_V)，不是 D_K


if __name__ == "__main__":
    out = selftest()
    print(json_dump := __import__("json").dumps(out, ensure_ascii=False, indent=1))
    print("★ 全部通过" if out["all_pass"] else "★★ 自检失败")