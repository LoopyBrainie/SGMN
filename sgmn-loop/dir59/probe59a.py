#!/opt/pytorch/bin/python
r"""DIR-59 · **前提探针 A**：「$P_v$ 的其他来源」—— 写值空间 $\ne$ 任务值空间时，写入可读性与任务 payload

出处：`framework/SGMN_v2.4_架构设计.md` §7.2 第 **523** 行
      「写入映射与门控 | $P_k,P_v$ 的**固定、离线学习或其他来源**；$\alpha$ 的产生规则及内容选择收益」
（同一行里 $P_k$ 离线 = **D-76**，$P_v$ 离线 = **DIR-56**，$\alpha$ 规则与内容选择 = **D-33..D-39**；
 唯一未判分支 = **「其他来源」**。）

★ 为什么「其他来源」在本谱系里只有一个非退化读法（探针 §1 会给出数字）：
  DIR-56 已证 $v_{\rm write}\equiv g_{\rm target}$ **逐位相同**（同一个 $P_v$）⇒ "来源"若仍是本任务抽的，就只是
  **重抽一次随机映射**（= 种子噪声）；若来自别处 ⇒ **写值空间 $\ne$ 任务值空间**，这是唯一能把"来源"变成
  可测量差别的方式。本探针就测这一种。

★ 逐字来源：`dir33/R1_task33.py:106-165` `rollout_task`（唯一改动：目标 `gs` 取自 `P_own` 而非 `P_write`）。
  `A59p-1` 断言 OWN 臂与该函数的损失**逐位相同**（否则本文件的复制已漂移）。

纪律：**不设阈值、不写判据、不产出裁决读数**；只产装置事实与"能不能问"。
运行：`/opt/pytorch/bin/python dir59/probe59a.py`（trainer）· 产物 `dir59/out/PROBE59A.json`
"""
import hashlib
import json
import pathlib
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
import R1_grad as M          # noqa: E402  ★ 先 import 装置，再 import task 层（task 层会改 M 常数）
import R1_task33 as T33      # noqa: E402

D_H, D_K, D_V, T = T33.D_H, T33.D_K, T33.D_V, T33.T
OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir59/out")
SEEDS = list(range(59600, 59632))          # ★ 新段：与 400/500/59000/59500 各段均不相交
N_UN = 8                                    # 零对照键数（取自**相邻种子**的隐藏态，经本实例 $P_k$）


def md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ══════════════════════════════════════════════════════════════════════
# rollout：逐字抄 `dir33/R1_task33.py:106-165`，**唯一改动** = 目标来自 `P_own`
# ══════════════════════════════════════════════════════════════════════
def rollout_measure(P_write, P_own, sd, alpha=1.0):
    r"""返回 layer-1 的 $(K,V^{\rm wr},W_1,\|\Delta W\|)$；`P_write` 提供 $P_v$，`P_own` 提供目标 $g_t$。

    `alpha=0` ⇒ **无写入对照**（只保留 $\lambda$ 正则项，与 Q2 口径一致）。
    """
    P = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in P_write.items()}
    gs = T33.targets(P_own)                 # ★ 目标恒取 own
    hs_own = T33.hs_of(P_own)
    W0 = M._W(P["U1"], P["V1"]).detach().clone()
    K1, V1w = [], []
    h_pre_prev = [None, None]
    for t in range(T):
        hp_prev = P["a"][t]
        h_pre, hp = [None, None], [None, None]
        for li, l in enumerate((1, 2)):
            h = torch.tanh(hp_prev @ P[f"A{l}"] + P[f"b{l}"])
            h_pre[li] = h
            c = (P["a"][t] if l == 1 else hp_prev).detach()
            q = torch.tanh(c @ P[f"Wq{l}"] + P[f"bq{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            y = W.detach() @ q
            m = torch.tanh(torch.cat([c, y], dim=-1) @ P[f"Wo{l}"] + P[f"bo{l}"])
            hp[li] = h + P[f"g{l}"] * m
            hp_prev = hp[li]
        upd = {}
        for li, l in enumerate((1, 2)):
            hprev = h_pre_prev[li]
            if hprev is None:
                hprev = torch.zeros(D_H)
            k = torch.tanh(hprev.detach() @ P[f"Pk{l}"])
            v = torch.tanh(h_pre[li].detach() @ P[f"Pv{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            G = alpha * torch.outer(W @ k - v, k) + M.LAM * W
            Uo, Vo = P[f"U{l}"], P[f"V{l}"]
            upd[f"U{l}"] = Uo - M.ETA * (G @ Vo)
            upd[f"V{l}"] = Vo - M.ETA * (G.t() @ Uo)
            if l == 1:
                K1.append(k.detach().clone())
                V1w.append(v.detach().clone())
        upd = {kk: vv.detach() for kk, vv in upd.items()}
        P.update(upd)
        h_pre_prev = h_pre
    W1 = M._W(P["U1"], P["V1"]).detach()
    return {"K1": K1, "V1w": V1w, "W1": W1.clone(), "W0": W0,
            "dW": float((W1 - W0).norm()), "hs_own": hs_own, "gs": gs}


def null_keys(P_own, P_other):
    r"""零对照键：**相邻种子**的隐藏态经**本实例** $P_k$（= 同一键空间里从未写过的点）。"""
    H = T33.hs_of(P_other)
    idx = torch.linspace(0, T - 1, N_UN).round().long().tolist()
    return [torch.tanh(H[i] @ P_own["Pk1"]) for i in idx]


def measure(res, un_keys):
    r"""三组读数：内容保真（写到什么）、任务载荷（own 目标）、零对照。"""
    W = res["W1"]
    e_wr = [float((W @ k - v).norm()) for k, v in zip(res["K1"], res["V1w"])]
    e_own = [float((W @ k - g).norm()) for k, g in zip(res["K1"], res["gs"])]
    e_null = [float(torch.median(torch.stack([(W @ ku - g).norm() for ku in un_keys])))
              for g in res["gs"]]
    return {"E_written": statistics.fmean(e_wr), "E_own": statistics.fmean(e_own),
            "E_null": statistics.fmean(e_null),
            "W_norm": float(W.norm()), "W0_norm": float(res["W0"].norm()),
            "dW": res["dW"]}


def arm_values(P_own, P_other, mode):
    r"""要写进 `P["Pv1"]` 的那张映射。`mix` 用 $(1-s)P_v^{\rm own}+sP_v^{\rm other}$。"""
    if mode == "OWN":
        return P_own["Pv1"]
    if mode == "FOREIGN":
        return P_other["Pv1"]
    if mode.startswith("MIX"):
        s = float(mode[3:]) / 100.0
        return (1.0 - s) * P_own["Pv1"] + s * P_other["Pv1"]
    raise ValueError(mode)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODES = ["OWN", "MIX50", "FOREIGN"]
    rows = []
    for sd in SEEDS:
        P_own = T33.build(sd, "V0")
        P_other = T33.build(sd + 1, "V0")
        un = null_keys(P_own, P_other)
        row = {"seed": sd, "arms": {}}
        for mode in MODES:
            P_w = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in P_own.items()}
            P_w["Pv1"] = arm_values(P_own, P_other, mode)
            r = rollout_measure(P_w, P_own, sd)
            row["arms"][mode] = measure(r, un)
        # 无写入对照（α=0）：目标/键/值不变，只留 λ
        P_nw = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in P_own.items()}
        row["arms"]["NONE"] = measure(rollout_measure(P_nw, P_own, sd, alpha=0.0), un)
        # 几何分离度：写值 vs own 值 的逐对余弦
        Vown = [torch.tanh(h @ P_own["Pv1"]) for h in T33.hs_of(P_own)]
        Vfor = [torch.tanh(h @ P_other["Pv1"]) for h in T33.hs_of(P_own)]
        cs = [float(torch.dot(a, b) / (a.norm() * b.norm())) for a, b in zip(Vown, Vfor)]
        row["cos_v_own_foreign"] = statistics.fmean(cs)
        row["max_cos_v"] = max(abs(c) for c in cs)
        rows.append(row)
        print("  seed %d  E_own OWN/50/FOR/NONE = %.3f / %.3f / %.3f / %.3f | E_null %.3f | "
              "E_written FOR %.3f | cos(v,v') %.3f"
              % (sd, *(row["arms"][m]["E_own"] for m in ("OWN", "MIX50", "FOREIGN", "NONE")),
                 row["arms"]["NONE"]["E_own"], row["arms"]["FOREIGN"]["E_written"],
                 row["cos_v_own_foreign"]))

    # ── A59p-1：OWN 臂必须与 `T33.rollout_task` 的 W 轨迹**逐位相同**（复制未漂移）
    sd0 = SEEDS[0]
    P_ref = T33.build(sd0, "V0")
    T33.rollout_task(P_ref, "device", "T", sd0)          # ★ 原地更新 P_ref 的 U1/V1
    W_ref = M._W(P_ref["U1"], P_ref["V1"]).detach().clone()
    P_copy = T33.build(sd0, "V0")
    r = rollout_measure(P_copy, T33.build(sd0, "V0"), sd0)
    dguard = float((W_ref - r["W1"]).abs().max())
    guard = {"max_abs_dW": dguard, "bitwise": dguard == 0.0,
             "sha_ref": hashlib.sha256(W_ref.numpy().tobytes()).hexdigest()[:16],
             "sha_copy": hashlib.sha256(r["W1"].numpy().tobytes()).hexdigest()[:16],
             "md5_dir33": md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py"),
             "md5_dir31": md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py")}

    def med(mode, key):
        return statistics.median([r["arms"][mode][key] for r in rows])

    def sd_(vals):
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    summ = {}
    for mode in MODES + ["NONE"]:
        summ[mode] = {k: {"med": med(mode, k),
                          "sd": sd_([r["arms"][mode][k] for r in rows])}
                      for k in ("E_own", "E_written", "E_null", "dW", "W_norm", "W0_norm")}
    paired = {}
    for a, b in (("OWN", "NONE"), ("FOREIGN", "OWN"), ("FOREIGN", "NONE"), ("MIX50", "OWN")):
        d = [r["arms"][a]["E_own"] - r["arms"][b]["E_own"] for r in rows]
        paired["%s-%s" % (a, b)] = {
            "med": statistics.median(d), "sd": sd_(d),
            "se": sd_(d) / (len(d) ** 0.5),
            "n_neg": sum(1 for x in d if x < 0), "n": len(d)}

    out = {"seeds": SEEDS, "n_seeds": len(rows), "rows": rows, "summary": summ,
           "paired_E_own": paired, "guard": guard,
           "cos_v": {"med": statistics.median([r["cos_v_own_foreign"] for r in rows]),
                     "max_abs_med": statistics.median([r["max_cos_v"] for r in rows])}}
    (OUT / "PROBE59A.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding="utf-8")

    print("\n★ A59p-1 复制守卫：max|ΔW| = %.1e（bitwise=%s）sha %s / %s"
          % (guard["max_abs_dW"], guard["bitwise"], guard["sha_ref"], guard["sha_copy"]))
    print("★ 中位（32 种子）：")
    for mode in MODES + ["NONE"]:
        s = summ[mode]
        print("   %-8s E_own %.4f (sd %.4f) · E_written %.4f · E_null %.4f · dW %.3f · ‖W‖ %.3f"
              % (mode, s["E_own"]["med"], s["E_own"]["sd"], s["E_written"]["med"],
                 s["E_null"]["med"], s["dW"]["med"], s["W_norm"]["med"]))
    print("★ 配对差（E_own）：")
    for k, v in paired.items():
        print("   %-14s med %+0.4f · se %.4f · 负号 %d/%d" % (k, v["med"], v["se"], v["n_neg"], v["n"]))
    print("★ cos(v_own, v_foreign)：中位 %.4f · 逐实例最大 |cos| 的中位 %.4f"
          % (out["cos_v"]["med"], out["cos_v"]["max_abs_med"]))
    print("★ 产物 → dir59/out/PROBE59A.json")


if __name__ == "__main__":
    main()