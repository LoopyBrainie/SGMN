#!/opt/pytorch/bin/python
r"""DIR-38 · **写规则的「一步够不够」** —— 同一步内写更新重复 `n_rep` 次的阶梯 + 岭不动点锚。

卡：`revolve/DIR-38/DIRECTION_rev38_v10.md`（rev 1.0，阈值已冻结；md5 由 A38-4 守卫）。

★ **唯一自变量 = `n_rep`**（每个在线时间步内，装置自己的**同步因子更新**原样重复的次数）。
  `η=0.05`、`α=1.0`、`λ=0.05`、`γ`、读公式 `y=Wq`、层结构、头的训练协议**全部不变**。
  ★ **不是把步长乘 16**：同一 `(k,v)` 只算一次，`G` 每次按**更新后的** `W` 重算，两式同步用**旧** `U,V`。

★ **装置与任务层逐字零改动**：只 import `dir31/R1_grad.py`（装置）、`dir33/R1_task33.py`（任务层）、
  `dir37/R1_task37.py`（DIR-37 的测量层：参考读 / 迹线 / 头训练协议）⇒ A38-2 用 md5 钉死。

★ **锚（描述性，不进判据）** `ridge`：把状态置为该写规则**当前一对**的岭不动点
  `W★ = v kᵀ/(‖k‖²+λ)`（⇒ `G(W★)=0`，因为 `α=1`）⇒ 分母量的是「**把这条规则走完**」的上限，
  **不是** `W_LS=V_mK⁺` 的上限。

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
import R1_task33 as R          # 任务层（逐字未改）
import R1_grad as M            # 装置（逐字未改）
import R1_task37 as T37        # DIR-37 的测量层（头的训练协议 / 迹线 / 参考读）

OUT = "/home/ec2-user/sgmn-loop/dir38/out"
D_H, D_V, D_K, RANK, T = R.D_H, R.D_V, R.D_K, M.R, M.T_STEPS
SEED0 = M.SEED0

# ══════════ 种子段（卡 §5：全部新段；不得复用探针段 12500–13095）
TRAIN = [SEED0 + 14000 + i for i in range(384)]     # 训练段 14000..14383
EVAL = [SEED0 + 13500 + i for i in range(400)]      # 评估段 13500..13899
SMOKE_TRAIN = [SEED0 + 14500 + i for i in range(4)]  # smoke（不进判据）
SMOKE_EVAL = [SEED0 + 14510 + i for i in range(2)]

# ══════════ 冻结阈值（卡 §1.3）
CARD_PATH = "/home/ec2-user/sgmn-loop/revolve/DIR-38/DIRECTION_rev38_v11.md"
CARD_MD5_EXPECTED = "98848374e1a367580d5c8c89eb6029a5"
MD5_DEVICE = "8fa40f708600bec7e6c484727d2180c7"
MD5_TASK = "96ef89e87ec61556b52e0e464850087e"
N_REPS = (1, 4, 16)
CL_THRESH = 0.50
SIG_K = 3.0
DIVERGE_W_RATIO = 10.0
REL_TOL = 1e-12
RIDGE_TOL = 1e-14


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def trace38(P, n_rep, mode, sd, phi, want_diag=False, ret_pairs=False):
    r"""`T37.trace` 的镜像：读**完全相同**（`y=Wq`），只多两样东西：
    ① 写更新在同一时间步内重复 `n_rep` 次（每次按更新后的 `W` 重算 `G`）；
    ② `ridge` 档把状态置为该规则的**当前一对**的岭不动点。

    ★ `n_rep=1` 且 `mode="grad"` ⇒ 与装置**逐位相同**（A38-1）。
    """
    L = P["a"].new_zeros(())
    gs = R.targets(P)
    h_pre_prev = [None, None]
    pairs = []
    queries = []
    diag = []
    wmax = 0.0
    for t in range(T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if li == 0:
                c, q = h, torch.tanh(h @ P["Pk1"])       # oracle 寻址（与装置逐字相同）
            else:
                c = hp_prev.detach()
                q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            y = W.detach() @ q                          # ★ 读：各臂完全相同
            if li == 0:
                queries.append(q.detach().clone())
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            gam = P[f"g{l}"]
            hp[li] = h + gam * m
            hp_prev = hp[li]
            if li == 0 and want_diag and pairs:
                k_or, v_or = pairs[-1][0], pairs[-1][1]
                nv, nk = float(v_or.norm()), float(k_or.norm())
                if nv > 1e-300 and nk > 1e-300:
                    diag.append(float((W.detach() @ k_or - v_or).norm()) / nv)
        if t >= 1:
            ell = ((R.pout() @ hp[0] - gs[t - 1]) ** 2).sum()
            L = L + ell
        # ── 写（唯一自变量在这里）
        upd = {}
        for li, l in enumerate((1, 2)):
            hprev = h_pre_prev[li]
            if hprev is None:
                hprev = torch.zeros(D_H)
            k = torch.tanh(hprev.detach() @ P[f"Pk{l}"])
            v = torch.tanh(h_pre[li].detach() @ P[f"Pv{l}"])
            if mode == "ridge":
                # ★ 该写规则当前一对的**岭不动点**：W★ = v kᵀ/(‖k‖²+λ) ⇒ G(W★)=0（α=1）
                s = float(k.norm() ** 2) + M.LAM
                Un = torch.zeros(D_V, RANK, dtype=P[f"U{l}"].dtype)
                Vn = torch.zeros(D_K, RANK, dtype=P[f"V{l}"].dtype)
                Un[:, 0] = v
                Vn[:, 0] = k / s
                upd[f"U{l}"], upd[f"V{l}"] = Un, Vn
            else:
                Uo, Vo = P[f"U{l}"], P[f"V{l}"]
                for _ in range(n_rep):
                    W = M._W(Uo, Vo)
                    G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
                    upd[f"U{l}"] = Uo - M.ETA * (G @ Vo)     # 同步：两式都用**旧** U,V
                    upd[f"V{l}"] = Vo - M.ETA * (G.t() @ Uo)
                    Uo, Vo = upd[f"U{l}"], upd[f"V{l}"]       # ★ 下一次按**更新后**的 W 重算 G
            if li == 0:
                pairs.append((k.detach().clone(), v.detach().clone()))
        upd = {kk: vv.detach() for kk, vv in upd.items()}
        P.update(upd)
        h_pre_prev = h_pre
        wmax = max(wmax, float(M._W(P["U1"], P["V1"]).norm()))
    dg = {"e_Wkey": (statistics.fmean(diag) if diag else float("nan")), "Wmax": wmax}
    if ret_pairs:
        dg["pairs"] = pairs
        dg["queries"] = queries
    return L, dg


def train_head(seeds):
    r"""头：**只训一份**，在装置臂（`n_rep=1`、装置读）上 ⇒ 协议逐字取 DIR-37 的 `train_head`。"""
    return T37.train_head("dev", seeds)


def eval_arm(phi, seeds, n_rep, mode):
    Ls, eW, wmax = [], [], 0.0
    for sd in seeds:
        P = T37._mk(sd, phi)
        L, dg = trace38(P, n_rep, mode, sd, phi, want_diag=True)
        Ls.append(float(L))
        eW.append(dg["e_Wkey"])
        wmax = max(wmax, dg["Wmax"])
    finite = all(math.isfinite(x) for x in Ls)
    return {"mean_L": statistics.fmean(Ls) if finite else float("nan"),
            "L_series": Ls, "e_Wkey": statistics.fmean(eW) if finite else float("nan"),
            "e_Wkey_series": eW, "Wmax": wmax, "finite": finite}


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
    card_md5 = _md5(CARD_PATH)
    g["A38-2_device_md5"] = md5_dev
    g["A38-2_task_md5"] = md5_task
    g["A38-4_card_md5"] = card_md5
    g["A38-2_ok"] = (md5_dev == MD5_DEVICE) and (md5_task == MD5_TASK)
    g["A38-4_ok"] = (card_md5 == CARD_MD5_EXPECTED)

    # A38-3：种子段不相交
    hist = []
    for k, (a, b) in list(T37.HIST_RANGES.items()) + list(T37.OWN_SEGMENTS.items()):
        hist.extend(range(a, b + 1))
    hist.extend(range(12500, 13096))          # 探针段（卡 §5 禁止复用）
    hist.extend(range(13500, 13900))          # 本卡评估段（对训练段而言）
    hist.extend(range(14500, 14512))          # smoke 段
    g["A38-3_train_hist_overlap"] = len(set(range(14000, 14384)) & set(hist))
    g["A38-3_train_eval_overlap"] = len(set(range(14000, 14384)) & set(range(13500, 13900)))
    g["A38-3_ok"] = (g["A38-3_train_hist_overlap"] == 0 and g["A38-3_train_eval_overlap"] == 0)

    # A38-1：n_rep=1 与 DIR-37 的装置臂逐位相同
    phi = train_head(SMOKE_TRAIN)
    errs = []
    for sd in SMOKE_EVAL:
        P = T37._mk(sd, phi)
        Lref, _ = T37.trace(P, "dev", sd, phi)
        P = T37._mk(sd, phi)
        L1, _ = trace38(P, 1, "grad", sd, phi)
        errs.append(abs(float(Lref) - float(L1)))
    g["A38-1_max_abs_diff"] = max(errs)
    g["A38-1_ok"] = (max(errs) == 0.0)

    # A38-5：n_rep=16 有限且 ‖W‖ 有界
    a1 = eval_arm(phi, SMOKE_EVAL, 1, "grad")
    a16 = eval_arm(phi, SMOKE_EVAL, 16, "grad")
    g["A38-5_finite16"] = bool(a16["finite"])
    g["A38-5_Wmax1"] = a1["Wmax"]
    g["A38-5_Wmax16"] = a16["Wmax"]
    g["A38-5_ratio"] = (a16["Wmax"] / a1["Wmax"]) if a1["Wmax"] > 0 else float("inf")
    g["A38-5_ok"] = bool(a16["finite"] and g["A38-5_ratio"] <= DIVERGE_W_RATIO)

    # A38-6：ridge 档满足不动点方程 G(W★)=0（手算工作例 §1.2）
    worst_G = 0.0
    worst_Wk = 0.0
    worst_read = 0.0
    for sd in SMOKE_EVAL:
        P = T37._mk(sd, phi)
        _L, dg = trace38(P, 1, "grad", sd, phi, ret_pairs=True)
        pairs, queries = dg["pairs"], dg["queries"]
        # 第一次**非退化**写（t=1 写下的对：k_1 = tanh(h_0 P_k)，v_1 = tanh(h_1 P_v)）
        k, v = pairs[1]
        s = float(k.norm() ** 2) + M.LAM
        Wstar = torch.outer(v, k) / s
        G = M.ALPHA * torch.outer(Wstar @ k - v, k) + M.LAM * Wstar
        worst_G = max(worst_G, float(G.norm()) / (float(v.norm()) + 1.0))
        want = v * (float(k.norm() ** 2) / s)
        worst_Wk = max(worst_Wk, float((Wstar @ k - want).norm()) / (float(v.norm()) + 1.0))
        # 构造事实：周期 2 ⇒ t≥2 的层 1 查询与写入时的键**逐位相同** ⇒ 读恒等式才成立
        for t in range(2, min(len(queries), len(pairs))):
            if float((queries[t] - pairs[t - 1][0]).norm()) == 0.0:
                ystar = Wstar @ queries[t]
                worst_read = max(worst_read, float((ystar - want).norm()) / (float(v.norm()) + 1.0))
                break
    g["A38-6_G_norm_rel"] = worst_G
    g["A38-6_Wstar_k_rel"] = worst_Wk
    g["A38-6_read_rel"] = worst_read
    g["A38-6_ok"] = (worst_G <= RIDGE_TOL) and (worst_Wk <= 1e-12) and (worst_read <= 1e-9)

    # A38-7：头的训练协议与 DIR-37 逐字相同（同函数 ⇒ 同一实现；此处记下 φ 的指纹）
    g["A38-7_phi_keys"] = sorted(phi.keys())
    g["A38-7_gamma"] = float(phi["g1"])
    g["A38-7_ok"] = (sorted(phi.keys()) == ["Wo1", "bo1", "g1"])

    # A38-8：写通道确实动了（**结构性**：终态两两不同 + 位移非零；**不含单调性**）
    # ★ rev 1.1：rev 1.0 的「‖ΔW‖ 随 n_rep 单调增」是错误前提（实测非单调）⇒ 降级为描述性读数
    dW = {}
    Wf = {}
    for nr in N_REPS:
        P = T37._mk(SMOKE_EVAL[0], phi)
        W0 = M._W(P["U1"], P["V1"]).clone()
        trace38(P, nr, "grad", SMOKE_EVAL[0], phi)
        W1 = M._W(P["U1"], P["V1"])
        dW[nr] = float((W1 - W0).norm())
        Wf[nr] = W1.clone()
    sep14 = float((Wf[1] - Wf[4]).norm())
    sep116 = float((Wf[1] - Wf[16]).norm())
    sep416 = float((Wf[4] - Wf[16]).norm())
    g["A38-8_dW"] = dW
    g["A38-8_sep_1_4"] = sep14
    g["A38-8_sep_1_16"] = sep116
    g["A38-8_sep_4_16"] = sep416
    g["A38-8_ok"] = (sep14 > 0.0 and sep116 > 0.0 and sep416 > 0.0
                     and all(v > 0.0 for v in dW.values()))

    ok = all(v for k, v in g.items() if k.endswith("_ok"))
    print("Stage A：" + ("★ 全部通过" if ok else "★ 未通过"))
    for k in sorted(g):
        print(f"  {k:26s} {g[k]}")
    return g, ok


# ══════════ smoke（证明协议可执行；**永不产生判据裁决**）
def smoke():
    phi = train_head(SMOKE_TRAIN)
    out = {"note": "smoke：只证明可执行，不给判据"}
    for tag, nr, mode in (("n1", 1, "grad"), ("n4", 4, "grad"), ("n16", 16, "grad"), ("ridge", 1, "ridge")):
        r = eval_arm(phi, SMOKE_EVAL, nr, mode)
        out[tag] = {"mean_L": r["mean_L"], "e_Wkey": r["e_Wkey"], "Wmax": r["Wmax"], "finite": r["finite"]}
        print(f"  {tag:6s} L={r['mean_L']:.6f} e_Wkey={r['e_Wkey']:.6f} Wmax={r['Wmax']:.4f}", flush=True)
    print("smoke 完成（**无判据裁决**）")
    return out


# ══════════ 正式轮
def formal():
    t0 = time.time()
    print(f"[训练] 头（装置臂 n_rep=1）n={len(TRAIN)} ...", flush=True)
    phi = train_head(TRAIN)
    print(f"[训练] 完成 {time.time()-t0:.0f}s", flush=True)

    arms = {}
    for nr in N_REPS:
        arms[f"n{nr}"] = eval_arm(phi, EVAL, nr, "grad")
        print(f"[评估] n_rep={nr:2d} L={arms[f'n{nr}']['mean_L']:.6f} "
              f"e_Wkey={arms[f'n{nr}']['e_Wkey']:.6f} Wmax={arms[f'n{nr}']['Wmax']:.4f} "
              f"finite={arms[f'n{nr}']['finite']}  ({time.time()-t0:.0f}s)", flush=True)
    arms["ridge"] = eval_arm(phi, EVAL, 1, "ridge")
    print(f"[评估] ridge    L={arms['ridge']['mean_L']:.6f} "
          f"e_Wkey={arms['ridge']['e_Wkey']:.6f} Wmax={arms['ridge']['Wmax']:.4f} "
          f"finite={arms['ridge']['finite']}  ({time.time()-t0:.0f}s)", flush=True)

    L1 = arms["n1"]["L_series"]
    L16 = arms["n16"]["L_series"]
    denom = arms["n1"]["mean_L"] - arms["ridge"]["mean_L"]
    numer = arms["n1"]["mean_L"] - arms["n16"]["mean_L"]
    cl = (numer / denom) if denom > 0 else None
    pg = paired(L1, L16)
    ratio_w = (arms["n16"]["Wmax"] / arms["n1"]["Wmax"]) if arms["n1"]["Wmax"] > 0 else float("inf")
    diverge = (not arms["n16"]["finite"]) or (ratio_w > DIVERGE_W_RATIO)

    ew_key = paired(arms["n1"]["e_Wkey_series"], arms["n16"]["e_Wkey_series"])
    out = {
        "n_train": len(TRAIN), "n_eval": len(EVAL),
        "train_seg": [14000, 14383], "eval_seg": [13500, 13899],
        "cl_threshold": CL_THRESH, "sig_k": SIG_K, "diverge_w_ratio": DIVERGE_W_RATIO,
        "arms": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_series")} for k, v in arms.items()},
        "L_series": {k: v["L_series"] for k, v in arms.items()},
        "e_Wkey_series": {k: v["e_Wkey_series"] for k, v in arms.items()},
        "denom_L1_minus_ridge": denom, "numer_L1_minus_L16": numer,
        "CL": cl, "paired_L1_L16": pg, "Wmax_ratio_16_over_1": ratio_w,
        "diverge": bool(diverge), "e_Wkey_paired": ew_key,
        "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── 判定（卡 §2.1，冻结）
    if diverge:
        out["outcome"] = "DIVERGED"
        out["counted"] = False
        out["note"] = "n_rep=16 发散 ⇒ 该档整档作废（种子与填表全部留档，不删不改）"
    elif denom <= 0:
        out["outcome"] = "INDETERMINATE"
        out["counted"] = False
        out["note"] = "分母 ≤ 0 ⇒ 不可判定（**不是失败**）"
    elif cl is not None and cl >= CL_THRESH and pg["mean"] >= pg["3se"]:
        out["outcome"] = "PASS"
        out["counted"] = True
        out["note"] = "闭合率过线且配对显著性通过 ⇒ H 被确认"
    elif cl is not None and cl >= CL_THRESH:
        out["outcome"] = "BELOW_SIG"
        out["counted"] = False
        out["note"] = "闭合率过线但配对显著性不过 ⇒ 不计数"
    elif ew_key["mean"] >= ew_key["3se"]:
        out["outcome"] = "EKEY_DOWN_LOSS_DOWN"
        out["counted"] = False
        out["note"] = "e_Wkey 已降而损失未降 ⇒ 冻结头用不了新的读；下一卡才重训头（本轮不重训）"
    else:
        out["outcome"] = "STABLE_BELOW"
        out["counted"] = False
        out["note"] = ("稳定但未过线 ⇒ **只授权**下一卡试更大的 n_rep 或更大的 η；"
                       "★ **不得**写成「速度不是主因、下一轮改形式」")

    with open(os.path.join(OUT, "DIR38r10_formal.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n══ 正式轮 ══")
    for k in ("n1", "n4", "n16", "ridge"):
        print(f"  L[{k:5s}] = {arms[k]['mean_L']:.6f}   e_Wkey = {arms[k]['e_Wkey']:.6f}   "
              f"Wmax = {arms[k]['Wmax']:.4f}")
    print(f"  分母 (L1−Lridge) = {denom:.6f} · 分子 (L1−L16) = {numer:.6f} · CL = {cl}")
    print(f"  配对 L1−L16: mean={pg['mean']:.6f} 3se={pg['3se']:.6f} n={pg['n']}")
    print(f"  e_Wkey 配对: mean={ew_key['mean']:.6f} 3se={ew_key['3se']:.6f}")
    print(f"  Wmax 比 (16/1) = {ratio_w:.4f} · 发散 = {diverge}")
    print(f"  ★ 结局 = {out['outcome']} · 计入 SUCCEEDED = {out['counted']}")
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "stageA"
    if what == "stageA":
        g, ok = stage_a()
        json.dump({"guards": g, "ok": ok}, open(os.path.join(OUT, "DIR38r10_stageA.json"), "w"),
                  ensure_ascii=False, indent=1)
        sys.exit(0 if ok else 1)
    if what == "smoke":
        json.dump(smoke(), open(os.path.join(OUT, "DIR38r10_smoke.json"), "w"), ensure_ascii=False, indent=1)
        sys.exit(0)
    if what == "formal":
        formal()
        sys.exit(0)
    print("用法：stageA | smoke | formal")
    sys.exit(2)
