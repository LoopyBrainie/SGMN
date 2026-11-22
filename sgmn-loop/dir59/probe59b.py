#!/opt/pytorch/bin/python
r"""DIR-59 · **前提探针 B**：把"写值对齐到同键目标"后再问「$P_v$ 的其他来源」

★ 探针 A（`probe59a.py`）的装置事实：在 dir33 任务层里，**写值与被读的目标本来就差一步** ——
  写 $(k,v)=(\tanh(h_{t-1}P_k),\ \tanh(h_tP_v))$，而键 $k$ 上的任务是 $g_t=\tanh(h_{t-1}P_v)$
  ⇒ 即使"同源"（OWN），写入对目标的贡献也只有 **6%**（$E_{\rm own}$ 0.8777 vs 无写 0.9632），
  于是"外源 vs 同源"的对比被这个**固有错位**压住（+0.0380 ± 0.0070）。
  本探针 B 把写值改成 $v=\tanh(h_{t-1}P_v)$（= 与目标同键对齐，`align=on`），问：
  **对齐之后，"来源"是否变成一个干净、可测的自变量？**

★ 只产装置事实：不设阈值、不写判据、不产出裁决读数。
★ 逐字来源：`probe59a.py` 的 `rollout_measure`（两个新开关 `align` / `npairs`）；
  `A59p-2` 断言 `align=off, npairs=T` 与探针 A 的 OWN 臂**逐位相同**。
运行：`/opt/pytorch/bin/python dir59/probe59b.py` · 产物 `dir59/out/PROBE59B.json`
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
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir59")
import R1_grad as M          # noqa: E402
import R1_task33 as T33      # noqa: E402
import probe59a as A         # noqa: E402

D_H, D_K, D_V, T = T33.D_H, T33.D_K, T33.D_V, T33.T
OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir59/out")
SEEDS = list(range(59600, 59632))
N_UN = 8


def rollout(P_write, P_own, alpha=1.0, align=False, npairs=T):
    r"""`align=True` ⇒ 写值取 $\tanh(h_{t-1}P_v)$（与目标同键）；`npairs` ⇒ 只写最后 `npairs` 对。"""
    P = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in P_write.items()}
    gs = T33.targets(P_own)
    W0 = M._W(P["U1"], P["V1"]).detach().clone()
    K1, V1w, G1 = [], [], []
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
            src = hprev.detach() if align else h_pre[li].detach()
            v = torch.tanh(src @ P[f"Pv{l}"])
            W = M._W(P[f"U{l}"], P[f"V{l}"])
            a_eff = alpha if t >= T - npairs else 0.0
            G = a_eff * torch.outer(W @ k - v, k) + M.LAM * W
            Uo, Vo = P[f"U{l}"], P[f"V{l}"]
            upd[f"U{l}"] = Uo - M.ETA * (G @ Vo)
            upd[f"V{l}"] = Vo - M.ETA * (G.t() @ Uo)
            if l == 1 and t >= T - npairs and t >= 1:      # ★ `gs` 只对 t=1..T-1 有定义
                K1.append(k.detach().clone())
                V1w.append(v.detach().clone())
                G1.append(gs[t - 1].detach().clone())
        upd = {kk: vv.detach() for kk, vv in upd.items()}
        P.update(upd)
        h_pre_prev = h_pre
    W1 = M._W(P["U1"], P["V1"]).detach()
    return {"K1": K1, "V1w": V1w, "W1": W1.clone(), "W0": W0,
            "dW": float((W1 - W0).norm()), "gs": G1}


def measure(res, un_keys):
    W = res["W1"]
    e_wr = [float((W @ k - v).norm()) for k, v in zip(res["K1"], res["V1w"])]
    e_own = [float((W @ k - g).norm()) for k, g in zip(res["K1"], res["gs"])]
    e_null = [float(torch.median(torch.stack([(W @ ku - g).norm() for ku in un_keys])))
              for g in res["gs"]]
    return {"E_written": statistics.fmean(e_wr), "E_own": statistics.fmean(e_own),
            "E_null": statistics.fmean(e_null), "dW": res["dW"],
            "W_norm": float(W.norm()), "W0_norm": float(res["W0"].norm())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sd in SEEDS:
        P_own = T33.build(sd, "V0")
        P_other = T33.build(sd + 1, "V0")
        un = A.null_keys(P_own, P_other)
        rec = {"seed": sd}
        for align in (False, True):
            for npairs in (T, 2):
                for src in ("OWN", "FOREIGN"):
                    P_w = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in P_own.items()}
                    if src == "FOREIGN":
                        P_w["Pv1"] = P_other["Pv1"]
                    tag = "%s_%s_n%d" % ("AL" if align else "MS", src, npairs)
                    rec[tag] = measure(rollout(P_w, P_own, align=align, npairs=npairs), un)
        P_nw = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in P_own.items()}
        rec["NONE"] = measure(rollout(P_nw, P_own, alpha=0.0, npairs=T), un)
        rows.append(rec)

    # ── A59p-2：align=off, npairs=T, OWN 必须与探针 A 的 OWN 逐位相同
    sd0 = SEEDS[0]
    P_own0 = T33.build(sd0, "V0")
    rA = A.rollout_measure({k: (v.clone() if torch.is_tensor(v) else v) for k, v in P_own0.items()},
                           T33.build(sd0, "V0"), sd0)
    rB = rollout({k: (v.clone() if torch.is_tensor(v) else v) for k, v in P_own0.items()},
                 T33.build(sd0, "V0"), align=False, npairs=T)
    d = float((rA["W1"] - rB["W1"]).abs().max())
    guard = {"max_abs_dW_vs_probeA": d, "bitwise": d == 0.0,
             "sha": hashlib.sha256(rB["W1"].numpy().tobytes()).hexdigest()[:16],
             "md5_dir33": A.md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py")}

    tags = [k for k in rows[0] if k != "seed"]
    summ = {t: {f: {"med": statistics.median([r[t][f] for r in rows]),
                    "sd": statistics.stdev([r[t][f] for r in rows])}
                for f in ("E_own", "E_written", "E_null", "dW", "W_norm")} for t in tags}
    paired = {}
    for a, b in (("AL_OWN_n%d" % T, "AL_FOREIGN_n%d" % T),
                 ("AL_OWN_n2", "AL_FOREIGN_n2"),
                 ("AL_OWN_n2", "NONE"), ("MS_OWN_n2", "MS_FOREIGN_n2")):
        d_ = [r[a]["E_own"] - r[b]["E_own"] for r in rows]
        paired["%s-%s" % (a, b)] = {"med": statistics.median(d_),
                                    "se": statistics.stdev(d_) / len(d_) ** 0.5,
                                    "sd": statistics.stdev(d_),
                                    "n_neg": sum(1 for x in d_ if x < 0), "n": len(d_)}

    out = {"seeds": SEEDS, "rows": rows, "summary": summ, "paired_E_own": paired, "guard": guard}
    (OUT / "PROBE59B.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding="utf-8")

    print("\n★ A59p-2 守卫：vs 探针A max|ΔW| = %.1e（bitwise=%s）" % (d, guard["bitwise"]))
    print("★ 中位（32 种子）· E_own / E_written / E_null")
    for t in ("NONE", "MS_OWN_n16", "MS_FOREIGN_n16", "AL_OWN_n16", "AL_FOREIGN_n16",
              "MS_OWN_n2", "MS_FOREIGN_n2", "AL_OWN_n2", "AL_FOREIGN_n2"):
        s = summ[t]
        print("   %-15s %8.4f / %8.4f / %8.4f   (dW %.3f)" %
              (t, s["E_own"]["med"], s["E_written"]["med"], s["E_null"]["med"], s["dW"]["med"]))
    print("★ 配对差（E_own）：")
    for k, v in paired.items():
        print("   %-26s med %+0.4f · se %.4f · 负号 %d/%d"
              % (k, v["med"], v["se"], v["n_neg"], v["n"]))
    print("★ 产物 → dir59/out/PROBE59B.json")


if __name__ == "__main__":
    main()