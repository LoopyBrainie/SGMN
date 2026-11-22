#!/opt/pytorch/bin/python
r"""DIR-38 · **开卡前探针**（exploratory；**不进判据**）：把 DIR-37 的 `G_inst` 拆成
「**读法**（寻址/取读不对）」与「**$W$ 里没有信息**（写规则不足）」两部分。

★ **为什么这是探针、不是一轮**：Advisor（Cycle 46 收束）要求「**开卡前**用便宜探针分清，
  分清之前**不得**改 `dir31/R1_grad.py`、**不得**跑正式轮」⇒ 本文件是**卡之前的探索**，
  它的读数**只用于设计 DIR-38 的卡**（单一对照 + 阈值），**不计入 `SUCCEEDED`**、
  **不写证据行**。本项目的先例：DIR-33 的「开卡前标定」（决定二十九）。

★ **装置与任务层逐字零改动**：只 import `dir31/R1_grad.py`（装置）、`dir33/R1_task33.py`（任务层）、
  `dir37/R1_task37.py`（DIR-37 的测量层）⇒ 任务与装置在**代码对象层面**就是 DIR-33/35/37 的那些，
  不是拷贝（纪律 68）。探针守卫 G3 用 md5 把这一性质钉死。

★ **唯一自变量 = 读数来源**（层的 1 读；层 2 的读与写规则不变）。五个臂：
  `dev`    : y = W q            （装置读）
  `ref`    : y = W^{LS} q       （DIR-37 的参考读；$W^{LS}=V_mK^+$）
  `key`    : y = W k_{t-1}      （**装置的矩阵 + 理想键**）⇒ 「从装置的 $W$ 里能读出的最好结果」
  `refkey` : y = W^{LS} k_{t-1} （LS 矩阵 + 理想键）        ⇒ 「内容本身的最好结果」
  `heb`    : y = W^{heb} q      （**理想 Hebb 写**（$\mathrm{rank}_2(V_mK^\top)$，DIR-33 的 `Ts` 臂）
                                   + 装置的寻址）⇒ 「若写规则完美、寻址不变」
★ `key`/`refkey`/`heb` 三臂的**层 1 轨迹**（$h,k,v$）与 `dev` **逐位相同**（层 1 的 $h,k,v$ 只由输入决定）
  ⇒ 三臂是**干净的单变量替换**；损失只测层 1 的 $h^+$（`R.pout() @ hp[0]`）⇒ 读是唯一通道。

★ **两级读数**：
  ① **机制级（与头无关）**：逐 $t$ 的相对误差
       `e_dev`  = ‖W_t q_t − v_{t-1}‖/‖v_{t-1}‖      （装置读的总误差；≈ DIR-37 的 `rell` 0.7255）
       `e_ref`  = ‖W^{LS}_t q_t − v_{t-1}‖/‖v_{t-1}‖  （参考读的误差；≈ 0.0667）
       `e_addr` = ‖q_t − k_{t-1}‖/‖k_{t-1}‖          （**寻址错位**，键空间；与矩阵无关）
       `e_Wkey` = ‖W_t k_{t-1} − v_{t-1}‖/‖v_{t-1}‖  （**矩阵质量**：$W$ 在该键下存的值对不对）
       `e_Wh`   = ‖W_t q_t − W_t k_{t-1}‖/‖v_{t-1}‖  （寻址错位**经装置矩阵**的效果）
  ② **损失级（冻结头，与 DIR-37 的 $G_{\rm inst}$ 同口径）**：$L_{\rm dev},L_{\rm ref},L_{\rm key},
     L_{\rm refkey},L_{\rm heb}$（全部用**同一** $\phi_A$，即 DIR-37 的臂 B 口径：读变、头不变）。

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
import R1_task37 as T37        # DIR-37 的测量层（参考读 / 迹线 / 训练协议）

OUT = "/home/ec2-user/sgmn-loop/dir38/out"
D_H, D_V, D_K, T = R.D_H, R.D_V, R.D_K, R.T
SEED0 = M.SEED0

# ══════════ 种子段（纪律 70 + 85：全部新段；历史最大偏移 = 12000）
PROBE_TRAIN = [SEED0 + 13000 + i for i in range(96)]   # 训练段 13000..13095（96）
PROBE_EVAL = [SEED0 + 12500 + i for i in range(50)]    # 评估段 12500..12549（50）

# ══════════ 守卫（探针级；不是卡的 Stage A）
MD5_DEVICE = "8fa40f708600bec7e6c484727d2180c7"
MD5_TASK = "96ef89e87ec61556b52e0e464850087e"
ARMS = ("dev", "ref", "key", "refkey", "heb")


def _md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def _heb(pairs):
    r"""理想 Hebb 写：$\mathrm{rank}_2(V_mK^\top)$（DIR-33 的 `Ts` 臂，逐字同构）。"""
    if len(pairs) < 2:
        return torch.zeros(D_V, D_K)
    K = torch.stack([p[0] for p in pairs], 1)
    Vm = torch.stack([p[1] for p in pairs], 1)
    U_, S_, Vt = torch.linalg.svd(Vm @ K.t())
    return (U_[:, :2] * S_[:2]) @ Vt[:2]


def _read(kind, W, q, pairs, diag):
    r"""层 1 的读（唯一自变量）。返回 `y`；`diag` 非 None 时填机制级读数。"""
    Wd = W.detach()
    if kind == "dev":
        y = Wd @ q
    elif kind == "ref":
        Wr = T37.ref_W(pairs)
        y = (Wr @ q) if Wr is not None else torch.zeros(D_V)
    elif kind in ("key", "refkey", "heb"):
        if not pairs:                       # t=0：无已写下的配对 ⇒ 保持装置读
            y = Wd @ q
        else:
            k_or, v_or = pairs[-1][0], pairs[-1][1]
            if kind == "key":
                y = Wd @ k_or
            elif kind == "refkey":
                Wr = T37.ref_W(pairs)
                y = (Wr @ k_or) if Wr is not None else torch.zeros(D_V)
            else:
                y = _heb(pairs) @ q
    elif kind not in ("dev", "ref"):
        raise ValueError(kind)
    # ── 机制级诊断（**与臂无关**：层 1 的 $h,k,v$ 只由输入决定 ⇒ 在 `dev` 臂上测即可）
    if diag is not None and pairs:
        k_or, v_or = pairs[-1][0], pairs[-1][1]
        Wr = T37.ref_W(pairs)
        nv = float(v_or.norm())
        nk = float(k_or.norm())
        # ★ t=1 的键是**退化键**（$h_{-1}=0$ ⇒ $k_0=0$）⇒ 该步不计入键空间的诊断
        if nv > 1e-300 and nk > 1e-300:
            diag.append({
                "e_dev": float((Wd @ q - v_or).norm()) / nv,
                "e_ref": float((Wr @ q - v_or).norm()) / nv if Wr is not None else float("nan"),
                "e_addr": float((q - k_or).norm()) / nk,
                "e_Wkey": float((Wd @ k_or - v_or).norm()) / nv,
                "e_Wh": float((Wd @ q - Wd @ k_or).norm()) / nv,
            })
    return y


def trace2(P, kind, sd, phi, want_diag=False):
    r"""`T37.trace` 的**镜像**：层 1 的读可切换为五个臂之一，并可记录机制级诊断。

    ★ 必须与 `T37.trace` 在 `dev`/`ref` 两臂上**逐位相同**（探针守卫 G1/G2）⇒ 镜像保真。
    ★ 其余部分（层 1 的 $h,k,v$、层 2 的读、写入更新、损失的层 1 取法）逐字照抄 `T37.trace`。
    """
    L = P["a"].new_zeros(())
    gs = R.targets(P)
    h_pre_prev = [None, None]
    pairs = []
    recs = []
    diag = [] if want_diag else None
    for t in range(T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            if li == 0:
                c, q = h, torch.tanh(h @ P["Pk1"])       # oracle 寻址
            else:
                c = hp_prev.detach()
                q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            if li == 0:
                y = _read(kind, W, q, pairs, diag)
            else:
                y = W.detach() @ q
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            gam = P[f"g{l}"]
            hp[li] = h + gam * m
            hp_prev = hp[li]
        if t >= 1:
            ell = ((R.pout() @ hp[0] - gs[t - 1]) ** 2).sum()
            L = L + ell
            if want_diag:
                recs.append({"t": t, "ell": float(ell)})
        upd = {}
        for li, l in enumerate((1, 2)):
            hprev = h_pre_prev[li]
            if hprev is None:
                hprev = torch.zeros(D_H)
            k = torch.tanh(hprev.detach() @ P[f"Pk{l}"])
            v = torch.tanh(h_pre[li].detach() @ P[f"Pv{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
            Uo, Vo = P[f"U{l}"], P[f"V{l}"]
            upd[f"U{l}"] = Uo - M.ETA * (G @ Vo)                 # 同步：两式都用**旧** U,V
            upd[f"V{l}"] = Vo - M.ETA * (G.t() @ Uo)
            if li == 0:
                pairs.append((k.detach().clone(), v.detach().clone()))
        upd = {kk: vv.detach() for kk, vv in upd.items()}
        P.update(upd)
        h_pre_prev = h_pre
    return L, recs, diag


def main():
    os.makedirs(OUT, exist_ok=True)
    guards = {}

    # ── G3：装置 / 任务层 md5（逐字零改）
    md5_dev = _md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py")
    md5_task = _md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py")
    guards["G3_device_md5"] = md5_dev
    guards["G3_task_md5"] = md5_task
    guards["G3_ok"] = (md5_dev == MD5_DEVICE) and (md5_task == MD5_TASK)
    print(f"[G3] device={md5_dev} task={md5_task} ok={guards['G3_ok']}", flush=True)
    assert guards["G3_ok"], "装置/任务层 md5 变了 ⇒ 探针作废"

    # ── G4：种子段与历史不相交
    hist = set(T37.HIST_RANGES.keys()) | set(T37.OWN_SEGMENTS.keys())
    segs = {"probe38_train": (13000, 13095), "probe38_eval": (12500, 12549)}
    hist_vals = []
    for k, (a, b) in list(T37.HIST_RANGES.items()) + list(T37.OWN_SEGMENTS.items()):
        hist_vals.extend(range(a, b + 1))
    hist_set = set(hist_vals)
    overlap = [s for s, (a, b) in segs.items() for v in range(a, b + 1) if v in hist_set]
    guards["G4_seed_overlap"] = len(overlap)
    guards["G4_train_eval_overlap"] = len(set(range(13000, 13096)) & set(range(12500, 12550)))
    print(f"[G4] 与历史段交集={len(overlap)} 训练/评估交集={guards['G4_train_eval_overlap']}", flush=True)
    assert guards["G4_seed_overlap"] == 0 and guards["G4_train_eval_overlap"] == 0

    # ── 训练臂 A 的头（与 DIR-37 臂 A 逐字同一协议）；缓存 ⇒ 重跑不必重训
    phi_path = os.path.join(OUT, "DIR38_phiA.pt")
    if os.path.exists(phi_path):
        phiA = torch.load(phi_path)
        print(f"[训练] φ_A(dev) 读自缓存 {phi_path}", flush=True)
    else:
        t0 = time.time()
        phiA = T37.train_head("dev", PROBE_TRAIN)
        torch.save(phiA, phi_path)
        print(f"[训练] φ_A(dev) n={len(PROBE_TRAIN)} {time.time()-t0:.0f}s", flush=True)

    # ── G1/G2：镜像保真（dev/ref 两臂与 T37.trace 逐位相同）
    errs = {"dev": [], "ref": []}
    for sd in PROBE_EVAL[:12]:
        P = T37._mk(sd, phiA)
        Ld, _ = T37.trace(P, "dev", sd, phiA)
        P = T37._mk(sd, phiA)                 # ★ 每次迹线都要**新实例**（迹线会改写 $P$）
        Ld2, _, _ = trace2(P, "dev", sd, phiA)
        P = T37._mk(sd, phiA)
        Lr, _ = T37.trace(P, "ref", sd, phiA)
        P = T37._mk(sd, phiA)
        Lr2, _, _ = trace2(P, "ref", sd, phiA)
        errs["dev"].append(abs(float(Ld) - float(Ld2)))
        errs["ref"].append(abs(float(Lr) - float(Lr2)))
    guards["G1_dev_rel_max"] = max(errs["dev"])
    guards["G2_ref_rel_max"] = max(errs["ref"])
    print(f"[G1] dev 臂镜像相对差 max={guards['G1_dev_rel_max']:.3e} · "
          f"[G2] ref 臂镜像相对差 max={guards['G2_ref_rel_max']:.3e}", flush=True)
    assert guards["G1_dev_rel_max"] == 0.0 and guards["G2_ref_rel_max"] == 0.0

    # ── 五臂损失（同一冻结头 φ_A）+ 机制级诊断
    out = {"n_train": len(PROBE_TRAIN), "n_eval": len(PROBE_EVAL),
           "train_seg": [13000, 13095], "eval_seg": [12500, 12549], "arms": {}}
    t0 = time.time()
    for kind in ARMS:
        Ls = []
        for sd in PROBE_EVAL:
            P = T37._mk(sd, phiA)
            L, _, _ = trace2(P, kind, sd, phiA)
            Ls.append(float(L))
        out["arms"][kind] = {"mean_L": statistics.fmean(Ls), "L_series": Ls}
        print(f"[臂] {kind:7s} L={statistics.fmean(Ls):.6f}  ({time.time()-t0:.0f}s)", flush=True)

    # ── 机制级诊断（装置轨迹；与臂无关）
    diags = {"e_dev": [], "e_ref": [], "e_addr": [], "e_Wkey": [], "e_Wh": []}
    for sd in PROBE_EVAL:
        P = T37._mk(sd, phiA)
        _, _, dg = trace2(P, "dev", sd, phiA, want_diag=True)
        for kk in diags:
            diags[kk].append(statistics.fmean([d[kk] for d in dg]))
    out["diag"] = {kk: {"mean": statistics.fmean(vv), "sd": statistics.stdev(vv),
                        "series": vv} for kk, vv in diags.items()}

    # ── 损失级分解（冻结头口径；与 DIR-37 的 G_inst 同口径）
    L = {k: out["arms"][k]["mean_L"] for k in ARMS}
    out["L"] = L
    out["split"] = {
        "G_inst_dir37_like": L["dev"] - L["ref"],
        "addr_gain": L["dev"] - L["key"],              # 只修寻址（保留装置矩阵）
        "matrix_gain_at_ideal_addr": L["key"] - L["refkey"],   # 只修矩阵（保留理想键）
        "matrix_gain_at_dev_query": L["dev"] - L["ref"],      # DIR-37 的 G_inst
        "addr_gain_at_ref_matrix": L["ref"] - L["refkey"],    # 理想矩阵上的寻址增益
        "ideal_write_gain_at_dev_query": L["dev"] - L["heb"],
        "total_to_ideal": L["dev"] - L["refkey"],
    }
    # 加性恒等式（两项分解都精确）
    out["split"]["identity_order1"] = (
        (L["dev"] - L["key"]) + (L["key"] - L["refkey"]) - (L["dev"] - L["refkey"]))
    out["split"]["identity_order2"] = (
        (L["dev"] - L["ref"]) + (L["ref"] - L["refkey"]) - (L["dev"] - L["refkey"]))
    # 份额
    tot = out["split"]["total_to_ideal"]
    if tot > 0:
        out["share"] = {
            "addr_share_of_total": out["split"]["addr_gain"] / tot,
            "matrix_share_of_total": out["split"]["matrix_gain_at_ideal_addr"] / tot,
            "addr_share_of_Ginst": (out["split"]["addr_gain"] / out["split"]["G_inst_dir37_like"]
                                    if out["split"]["G_inst_dir37_like"] else None),
            "matrix_share_of_Ginst": (out["split"]["matrix_gain_at_ideal_addr"]
                                      / out["split"]["G_inst_dir37_like"]
                                      if out["split"]["G_inst_dir37_like"] else None),
        }
    d = out["diag"]
    out["diag_shares"] = {
        "e_addr_over_e_dev": d["e_addr"]["mean"] / d["e_dev"]["mean"] if d["e_dev"]["mean"] else None,
        "e_Wkey_over_e_dev": d["e_Wkey"]["mean"] / d["e_dev"]["mean"] if d["e_dev"]["mean"] else None,
    }

    out["guards"] = guards
    out["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(OUT, "DIR38_probe.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n══ 损失级（冻结头 φ_A，同一评估段）══")
    for k in ARMS:
        print(f"  L[{k:7s}] = {L[k]:.6f}")
    print("\n══ 分解 ══")
    for kk, vv in out["split"].items():
        print(f"  {kk:34s} = {vv:+.6f}")
    print("\n══ 份额 ══")
    for kk, vv in out.get("share", {}).items():
        print(f"  {kk:34s} = {vv:+.4f}" if vv is not None else f"  {kk:34s} = None")
    print("\n══ 机制级诊断（与头无关）══")
    for kk in diags:
        print(f"  {kk:8s} mean={d[kk]['mean']:.6f} sd={d[kk]['sd']:.6f}")
    for kk, vv in out["diag_shares"].items():
        print(f"  {kk:34s} = {vv:+.4f}" if vv is not None else f"  {kk:34s} = None")
    print("\n★ 探针完成（**不进判据**；只用于设计 DIR-38 的卡）")


if __name__ == "__main__":
    main()
