#!/opt/pytorch/bin/python
r"""DIR-59 · 探针 **C = A/B 的配对修正版**（`DEF-59-1`）

★ 缺陷（工作例 `workex59.py` 抓出来的）：`probe59a/b` 里 `K1/V1w` 收了 **16** 对（$t{=}0..15$），
  而 `gs` 只有 **15** 个（$t{=}1..15$） ⇒ `zip` 把 $(k_t,g_{t+1})$ 配在一起，
  $E_{\rm own}$ 实际测的是"**下一拍**目标"，与"同键目标"（$g_t=P_v(h_{t-1})$）错一拍。
  后果：**OWN 臂里** $E_{\rm own}$ 退化为存储保真（$g_{t+1}=P_v(h_t)=v_t$ 的同源版本），
  于是"载荷"与"存储"两个量在 OWN 臂里不可分；FOREIGN 臂的对比因此被污染。
★ 本探针只改配对：**$t\in\{1,\dots,15\}$ 的 $(k_t,g_t)$ 与 $(k_t,v_t^{\rm wr})$ 一起收**，$k_0$ 丢弃（它没有 $g_0$）。
  其余（装置、种子段、$\alpha$、零对照、臂定义）与 `probe59a/b` 逐字相同。
★ 只产装置事实：不设阈值、不写判据。
运行：`/opt/pytorch/bin/python dir59/probe59c_fixedpair.py` · 产物 `dir59/out/PROBE59C.json`
"""
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

T, D_H = T33.T, T33.D_H
OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir59/out")
SEEDS = list(range(59600, 59632))


def rollout(P_write, P_own, alpha=1.0, align=False, npairs=T):
    r"""同 `probe59b.rollout`，**唯一改动**：只收 $t\ge1$ 的 $(k,v,g)$，三者**同下标**。"""
    P = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in P_write.items()}
    gs = T33.targets(P_own)                      # 15 个：t=1..15
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
            if l == 1 and t >= 1:                # ★★ 修正点：只收 t>=1，与 gs 同下标
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
    assert len(res["K1"]) == len(res["V1w"]) == len(res["gs"]) == T - 1, "配对长度必须相等"
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

    tags = [k for k in rows[0] if k != "seed"]
    summ = {t: {f: {"med": statistics.median([r[t][f] for r in rows]),
                    "sd": statistics.stdev([r[t][f] for r in rows])}
                for f in ("E_own", "E_written", "E_null", "dW", "W_norm")} for t in tags}
    paired = {}

    def pair(a, b, key):
        d = [r[a][key] - r[b][key] for r in rows]
        return {"med": statistics.median(d), "se": statistics.stdev(d) / len(d) ** 0.5,
                "sd": statistics.stdev(d), "n_neg": sum(1 for x in d if x < 0), "n": len(d)}

    for a, b in (("MS_OWN_n16", "NONE"), ("MS_FOREIGN_n16", "MS_OWN_n16"),
                 ("MS_OWN_n16", "MS_FOREIGN_n16"),
                 ("AL_OWN_n16", "NONE"), ("AL_FOREIGN_n16", "AL_OWN_n16"),
                 ("AL_FOREIGN_n16", "NONE"), ("AL_OWN_n2", "AL_FOREIGN_n2")):
        for key in ("E_own", "E_written"):
            paired["%s|%s|%s" % (key, a, b)] = pair(a, b, key)

    out = {"seeds": SEEDS, "rows": rows, "summary": summ, "paired": paired,
           "fix": "DEF-59-1：K1/V1w/G1 只收 t>=1（同下标），k_0 丢弃"}
    (OUT / "PROBE59C.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding="utf-8")

    print("★ 配对修正版（32 种子）· E_own / E_written / E_null")
    for t in ("NONE", "MS_OWN_n16", "MS_FOREIGN_n16", "AL_OWN_n16", "AL_FOREIGN_n16",
              "MS_OWN_n2", "MS_FOREIGN_n2", "AL_OWN_n2", "AL_FOREIGN_n2"):
        s = summ[t]
        print("   %-15s %8.4f / %8.4f / %8.4f   (dW %.3f)"
              % (t, s["E_own"]["med"], s["E_written"]["med"], s["E_null"]["med"], s["dW"]["med"]))
    print("★ 配对差：")
    for k, v in paired.items():
        print("   %-42s med %+0.4f · se %.4f · 负号 %d/%d" % (k, v["med"], v["se"], v["n_neg"], v["n"]))
    print("★ 产物 → dir59/out/PROBE59C.json")


if __name__ == "__main__":
    main()