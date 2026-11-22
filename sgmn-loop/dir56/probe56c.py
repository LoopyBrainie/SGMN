#!/opt/pytorch/bin/python
r"""DIR-56 · **前提探针 C**：写侧自检索目标对 $P_v$ 是**可学**、**退化**、还是**只是记住**？

★ 探针纪律：**不设阈值、不写判据、不产出裁决读数**。
★ 三臂（都用 DIR-53 的**同一个**离线程序：`L_off = mean‖Y−V‖²`、`W = VᵀK`、`Y = (W Kᵀ)ᵀ`、250 步、Adam lr 0.03）：
  **C2a pooled(n=1536)**：把所有实例的 `(h,v)` 拼在一起学**一个** $P_v$（DIR-53 的共享口径，供对照）
  **C2b per-instance + 留出**：每个实例只用自己的 16 对，**8 训 / 8 测**（奇偶切分）⇒ 检验"学到映射"还是"记住这 8 对"
  **C3  per-instance + 行范数投影**（DIR-53 纪律：只改方向、不改尺度）⇒ 与 C2b 同切分
★ 评估量：`L_off_train` / `L_off_test`（留出）· `L_ideal = min_A mean‖YA − Vm‖²`（**全局一个**线性读出，`Vm` = 各实例真值）
  ⇒ `L_ideal` 必须在**所有实例拼接后**拟合一个 A，否则 16 行 4 参可精确插值（**首版就错在这里，已留档**）。
★ C4 Freeze C：写产物 detach ⇒ `∂L_task/∂P_v ≡ 0`。

用法：`/opt/pytorch/bin/python dir56/probe56c.py`
输出：`dir56/out/PROBE56C.json`
"""
import json
import os
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)
for _d in ("dir31", "dir33", "dir37", "dir38", "dir39", "dir40"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_d}")
import R1_task33 as R
import R1_task37 as T37
import R1_task40 as T40

OUT = "/home/ec2-user/sgmn-loop/dir56/out"
SEED0 = 20261111
SEEDS = [SEED0 + 21000 + i for i in range(32)]
STEPS, LR = 250, 0.03          # ★ 与 DIR-53 离线程序逐字相同


def pairs_of(H, pk, pv):
    K = torch.tanh(H @ pk)
    V = torch.tanh(H @ pv)
    return K, V


def readback(K, V):
    W = V.T @ K
    return (W @ K.T).T


def l_off(Y, V):
    return float(((Y - V) ** 2).mean())


def l_ideal(Y, Tgt):
    A = torch.linalg.lstsq(Y, Tgt).solution
    return float((((Y @ A) - Tgt) ** 2).mean())


def make_trainer(H):
    def _train(pv0, K, V_tgt, idx, steps, project):
        pv = pv0.detach().clone().requires_grad_(True)
        n0c = pv0.norm(dim=1).clone().unsqueeze(1)
        opt = torch.optim.Adam([pv], lr=LR)
        Ls = []
        for _ in range(steps):
            opt.zero_grad()
            Vt = torch.tanh(H[idx] @ pv)
            Y = readback(K[idx], Vt)
            L = ((Y - V_tgt[idx]) ** 2).mean()
            L.backward()
            Ls.append(float(L.detach()))
            opt.step()
            if project:
                with torch.no_grad():
                    pv.mul_(n0c / pv.norm(dim=1, keepdim=True))
        return pv, Ls
    return _train


def main():
    os.makedirs(OUT, exist_ok=True)
    phi0 = T40.init_params()
    inst = []
    for sd in SEEDS:
        P = T37._mk(sd, phi0, train=True)
        H = torch.stack(R.hs_of(P))
        K, V = pairs_of(H, P["Pk1"], P["Pv1"])
        inst.append({"H": H, "K": K, "V": V, "pv": P["Pv1"]})

    # ── 基线：真映射下的 L_off 与全局 L_ideal
    Y0 = torch.cat([readback(r["K"], r["V"]) for r in inst])
    Vm = torch.cat([r["V"] for r in inst])
    base = {"L_off_true": statistics.median([l_off(readback(r["K"], r["V"]), r["V"]) for r in inst]),
            "L_ideal_true": l_ideal(Y0, Vm)}

    # ── C2a pooled：一个共享 Pv 拟合全部 1536 对
    Hp = torch.cat([r["H"] for r in inst])
    Kp = torch.cat([r["K"] for r in inst])
    Vp = torch.cat([r["V"] for r in inst])
    pv_pool0 = inst[0]["pv"]
    pv_pool, Ls_pool = make_trainer(Hp)(pv_pool0, Kp, Vp,
                                        torch.arange(Hp.shape[0]), STEPS, project=False)
    with torch.no_grad():
        Yp = readback(torch.tanh(Hp @ pv_pool), Kp)
        c2a = {"n_pairs": int(Hp.shape[0]),
               "L_off_first20": statistics.fmean(Ls_pool[:20]),
               "L_off_last20": statistics.fmean(Ls_pool[-20:]),
               "pv_norm_ratio": float(pv_pool.norm() / pv_pool0.norm()),
               "L_ideal_before": base["L_ideal_true"], "L_ideal_after": l_ideal(Yp, Vp),
               "v_norm_mean_after": float(torch.tanh(Hp @ pv_pool).norm(dim=1).mean())}

    # ── C2b / C3：逐实例 8 训 / 8 测（奇偶切分）
    res_b, res_c = [], []
    for r in inst:
        H, K, V, pv0 = r["H"], r["K"], r["V"], r["pv"]
        idx_tr = torch.arange(0, H.shape[0], 2)
        idx_te = torch.arange(1, H.shape[0], 2)
        for project, bucket in ((False, res_b), (True, res_c)):
            pv, Ls = make_trainer(H)(pv0, K, V, idx_tr, STEPS, project)
            with torch.no_grad():
                Ytr = readback(K[idx_tr], torch.tanh(H[idx_tr] @ pv))
                Yte = readback(K[idx_te], torch.tanh(H[idx_te] @ pv))
                bucket.append({"L_train_first20": statistics.fmean(Ls[:20]),
                               "L_train_last20": statistics.fmean(Ls[-20:]),
                               "L_test_before": l_off(readback(K[idx_te], torch.tanh(H[idx_te] @ pv0)), V[idx_te]),
                               "L_test_after": l_off(Yte, V[idx_te]),
                               "pv_norm_ratio": float(pv.norm() / pv0.norm()),
                               "rel_dPv": float((pv - pv0).norm() / pv0.norm()),
                               "pv_trained": pv.detach()})

    def agg(rows, key):
        vals = [x[key] for x in rows]
        return {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}

    def summarize(rows):
        out = {k: agg(rows, k) for k in ("L_train_first20", "L_train_last20",
                                         "L_test_before", "L_test_after",
                                         "pv_norm_ratio", "rel_dPv")}
        out["n_train_gap_closed"] = sum(1 for x in rows
                                        if x["L_train_last20"] < 1e-6)
        out["n_test_worse_than_before"] = sum(1 for x in rows
                                             if x["L_test_after"] > x["L_test_before"] + 1e-12)
        # 全局 L_ideal：把各实例训练后的 Y 拼起来、**一个** A
        Yall = torch.cat([readback(r["K"], torch.tanh(r["H"] @ x["pv_trained"]))
                          for r, x in zip(inst, rows)])
        out["L_ideal_after_global"] = l_ideal(Yall, Vm)
        out["L_ideal_before_global"] = base["L_ideal_true"]
        for x in rows:
            x.pop("pv_trained")
        return out

    c2b, c3 = summarize(res_b), summarize(res_c)

    # ── C4 Freeze C
    r0 = inst[0]
    pv4 = r0["pv"].detach().clone().requires_grad_(True)
    l_on = ((readback(r0["K"], torch.tanh(r0["H"] @ pv4)) - r0["V"]) ** 2).mean()
    l_on.backward()
    g_on = float(pv4.grad.norm())
    pv5 = r0["pv"].detach().clone().requires_grad_(True)
    l5 = ((readback(r0["K"], torch.tanh(r0["H"] @ pv5).detach()) - r0["V"]) ** 2).mean()
    if l5.requires_grad:
        gg = torch.autograd.grad(l5, pv5, allow_unused=True)[0]
        g_off = 0.0 if gg is None else float(gg.norm())
        out_req = True
    else:
        g_off, out_req = 0.0, False
    c4 = {"task_grad_pv_writes_live": g_on, "task_grad_pv_writes_detached": g_off,
          "detached_loss_has_grad_fn": out_req}

    res = {"probe": "56c", "steps": STEPS, "lr": LR, "n_instances": len(inst), "baseline": base,
           "C2a_pooled": c2a, "C2b_per_instance_split": c2b, "C3_per_instance_split_projected": c3,
           "C4_freeze_c": c4}
    with open(os.path.join(OUT, "PROBE56C.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print("── 基线：L_off(true) 中位 %.3f；L_ideal(true) = %.5f"
          % (base["L_off_true"], base["L_ideal_true"]))
    print("── C2a pooled(n=%d)：L_off %.2f→%.5f；‖Pv‖ 比值 %.4f；|v|→%.4f；L_ideal %.5f→%.5f"
          % (c2a["n_pairs"], c2a["L_off_first20"], c2a["L_off_last20"], c2a["pv_norm_ratio"],
             c2a["v_norm_mean_after"], c2a["L_ideal_before"], c2a["L_ideal_after"]))
    for tag, s in (("C2b 逐实例(8训/8测)", c2b), ("C3  逐实例+行范数投影", c3)):
        print("── %s：L_train %.3f→%.5f（%d/32 归零）；L_test %.3f→%.3f（%d/32 变差）；"
              "‖Pv‖比值 %.4f；L_ideal %.5f→%.5f"
              % (tag, s["L_train_first20"]["median"], s["L_train_last20"]["median"],
                 s["n_train_gap_closed"], s["L_test_before"]["median"], s["L_test_after"]["median"],
                 s["n_test_worse_than_before"], s["pv_norm_ratio"]["median"],
                 s["L_ideal_before_global"], s["L_ideal_after_global"]))
    print("── C4 Freeze C：detach ⇒ %.3e；不 detach ⇒ %.2f"
          % (c4["task_grad_pv_writes_detached"], c4["task_grad_pv_writes_live"]))


if __name__ == "__main__":
    main()