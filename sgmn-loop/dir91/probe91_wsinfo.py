#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 rev 3.4 —— 写侧池化信号 `c = mean_t sg(h_t)` 是否线性携带值身份。

卡片：revolve/DIR-91/DIRECTION_rev91_v34.md（判据冻结；本文件不得改阈值）
      ★ 本卡无训练：只做 capture + 岭回归/LOO 解码（省约 28 min）。
      ★ `A97-03`/`A97-04` 是承重守卫：证明我的 H 就是装置写规则里的那个 hf.mean(0)
        （`DEF-96-08` 的教训：手搓的 v 不是装置写的 v）。
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)

import numpy as np
import torch

import probe91_womap as W91                       # ridge_scores / top1 / W_of（与 D-106/D-108 同一份）
from probe91_woform import ridge_closed, count_mounted
import R2_task91 as R2
from R2_task91 import (DEV, MOUNT, N_CAND, N_FACTS, N_TRAIN_KEYS, OUT, VALUE_POOL,
                       build_chunks, build_facts, make_stack, model_and_tok)

N_CLS = len(VALUE_POOL)
CHANCE = 1.0 / N_CLS
SEEDS = [20261028, 7777]
OUT = Path(OUT) if not isinstance(OUT, Path) else OUT

CAP_LOO = 10.0 / 32.0      # ★ A97-01 用真实 cls 实测；不得用假定规则推（DEF-96-09）
G_AVAIL = 0.90 * CAP_LOO   # 0.28125
G_ABSENT = 0.15

RES = {"card": "DIR-91 rev 3.4", "thresholds": {"CAP_LOO": CAP_LOO, "G_AVAIL": G_AVAIL,
                                                 "G_ABSENT": G_ABSENT, "CHANCE": CHANCE},
       "guards": {}, "arms": [], "worked": {}}


def dump(obj, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def value_classes(facts):
    """每事实答案在 VALUE_POOL 中的下标（★ 不假设 i%27：build_facts 里有 rng.shuffle）。"""
    idx = {w: i for i, w in enumerate(VALUE_POOL)}
    return [idx[f["val"]] for f in facts]


def m1_identity(X, cls, tr, he, n_cls):
    """值身份能否从 X 线性读出。M1b = LOO 全 32（与 D-106/D-108 的 M2loo、D-113 的 M1b 同口径）。"""
    per_layer = {}
    for li in X:
        Xl = X[li]
        s = W91.ridge_scores(Xl[tr], [cls[i] for i in tr], Xl[he], n_cls)
        a_he, _ = W91.top1(s, [cls[i] for i in he])
        loo = 0
        for i in range(N_FACTS):
            keep = [j for j in range(N_FACTS) if j != i]
            si = W91.ridge_scores(Xl[keep], [cls[j] for j in keep], Xl[[i]], n_cls)
            loo += int(int(si.argmax(dim=1)[0]) == cls[i])
        per_layer[str(li)] = {"held_top1": float(a_he), "loo_top1": loo / N_FACTS}
    return {"per_layer": per_layer,
            "held_top1_mean": float(np.mean([v["held_top1"] for v in per_layer.values()])),
            "loo_top1_mean": float(np.mean([v["loo_top1"] for v in per_layer.values()])),
            "n_cls": n_cls}


def capture(model, st, facts, chunks, owner):
    """逐事实 capture 写侧三个对象：H（池化层输入）、K、V。
    ★ H 与 write() 用完全相同的 pf.mean(dim=0)；K/V 复用 R2.key_matrices（权威）。
    ★ no_write 按层保存/恢复（不假设各层相同）。"""
    K, V = R2.key_matrices(model, st, facts, chunks, owner)     # 权威 K/V
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    H = {li: [] for li in st.mems}
    Km = {li: [] for li in st.mems}      # ★ A97-03a：同算序复算（先乘后均）
    Vm = {li: [] for li in st.mems}      # ★ A97-04a
    try:
        for fi in range(len(facts)):
            ci = owner.index(fi) + 1
            model(input_ids=chunks[ci].to(DEV))
            for li in st.mems:
                hs = st._hs_in[li]
                pf = hs.reshape(-1, hs.shape[-1])
                mm = st.mems[li]
                H[li].append(pf.mean(dim=0).detach())
                Km[li].append(((pf @ mm.Pk).mean(dim=0) * mm.pk_scale).detach())
                Vm[li].append((pf @ mm.Pv).mean(dim=0).detach())
    finally:
        for li, m in st.mems.items():
            m.no_write = saved[li]
    return ({li: torch.stack(H[li]).float() for li in H},
            {li: K[li].float() for li in K},
            {li: V[li].float() for li in V},
            {li: torch.stack(Km[li]).float() for li in Km},
            {li: torch.stack(Vm[li]).float() for li in Vm})


def centred(X):
    return {li: (X[li] - X[li].mean(dim=0, keepdim=True)).float() for li in X}


def cos_geom(X):
    """两两余弦（去对角）的 max / median —— 同一 capture 上复核 D-104 机制 1 的键几何。"""
    out = {}
    for li in X:
        Mn = torch.nn.functional.normalize(X[li].detach(), dim=1)
        C = (Mn @ Mn.T).abs()
        off = C[~torch.eye(len(Mn), dtype=torch.bool, device=C.device)]
        out[str(li)] = {"cos_max": float(off.max()), "cos_median": float(off.median())}
    return out


def nrm(X):
    return {str(li): float(X[li].norm(dim=1).mean()) for li in X}


def cond_of(M):
    try:
        s = torch.linalg.svdvals(M.detach().double())
        return float(s.max() / s.min())
    except Exception as e:                      # noqa: BLE001
        return float("nan")


# ─────────────────────────────── Stage A ───────────────────────────────
def stage_a(tok, model, facts, chunks, owner, tr, he):
    seed = SEEDS[0]
    cls = value_classes(facts)
    d = model.config.hidden_size
    ok = True

    def rec(name, cond, detail, value=None):
        nonlocal ok
        ok = ok and bool(cond)
        RES["guards"][name] = {"ok": bool(cond), "detail": detail, "value": value}
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}", flush=True)

    st = make_stack(gamma=0.0)

    # ---- A97-05(a) 先断言 count_mounted()==0（CORR-15）
    n_mnt0 = count_mounted(model, MOUNT)      # ★ DEF-97-04：模块函数，不是 st 的方法
    Wsnap = {li: W91.W_of(m).detach().clone() for li, m in st.mems.items()}

    H, K, V, Km, Vm = capture(model, st, facts, chunks, owner)
    H2, K2, V2, _, _ = capture(model, st, facts, chunks, owner)  # 第二次：A97-06

    # ---- A97-00 口径 + 逐事实索引工作算例
    n_cls_seen = len(set(cls))
    # 工作算例：第 0 条事实、MOUNT[0] 层（★ DEF-97-01：forward 必须在 no_write=True 之下）
    li0 = MOUNT[0]
    _nw = {l: m.no_write for l, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    try:
        model(input_ids=chunks[owner.index(0) + 1].to(DEV))
        h_direct = st._hs_in[li0].reshape(-1, d).mean(dim=0).detach().float()
    finally:
        for l, m in st.mems.items():
            m.no_write = _nw[l]
    dwork = float((h_direct - H[li0][0]).abs().max())
    rec("A97-00", len(facts) == N_FACTS and n_cls_seen == N_CLS and list(MOUNT) == [8, 12, 16]
        and dwork == 0.0,
        f"n_facts={len(facts)} n_cls_seen={n_cls_seen} MOUNT={list(MOUNT)} "
        f"CHANCE={CHANCE:.4f}; 工作算例 max|ΔH|={dwork:.1e}（须 ==0）", value=dwork)

    # ---- A97-01 正对照：合成可读 ⇒ M1b == CAP_LOO 精确
    g = torch.Generator(device="cpu").manual_seed(4242)
    B = torch.randn(N_CLS, d, generator=g).to(DEV)
    Xgood = {0: B[torch.tensor(cls, dtype=torch.long)].float()}
    r_good = m1_identity(Xgood, cls, tr, he, N_CLS)
    b_good = r_good["loo_top1_mean"]
    # ---- A97-02 负对照：纯噪声 ⇒ M1b ≤ G_ABSENT
    Xnoise = {0: torch.randn(N_FACTS, d, generator=g).to(DEV).float()}
    r_noise = m1_identity(Xnoise, cls, tr, he, N_CLS)
    b_noise = r_noise["loo_top1_mean"]
    RES["worked"]["A97-01"] = {"M1b_good": b_good, "M1a_good_reported": r_good["held_top1_mean"]}
    RES["worked"]["A97-02"] = {"M1b_noise": b_noise}
    rec("A97-01", abs(b_good - CAP_LOO) < 1e-12,
        f"合成可读 M1b={b_good:.6f}（须 ==CAP_LOO {CAP_LOO:.6f} 精确）"
        f"。★ 同情形 M1a={r_good['held_top1_mean']:.4f}（上限 5/16，仅报告）", value=b_good)
    rec("A97-02", b_noise <= G_ABSENT,
        f"纯噪声 M1b={b_noise:.6f}（须 ≤{G_ABSENT}）", value=b_noise)

    # ---- A97-03 / A97-04 ★★ 承重：H 必须就是装置写规则里的 hf.mean(0)
    # ★ A97-03a/04a（承重，同算序 ⇒ 可严 ==0）：证明 pf 及其逐事实块索引就是装置的对象
    dKa = max(float((Km[li] - K[li]).abs().max()) for li in st.mems)
    dVa = max(float((Vm[li] - V[li]).abs().max()) for li in st.mems)
    rec("A97-03a", dKa == 0.0,
        f"同算序复算 ((pf @ Pk).mean(0))*pk_scale == key_matrices 的 K 逐位相同："
        f"max|Δ|={dKa:.1e}（须 ==0）—— 证明本卡的 pf 就是装置写的对象（DEF-96-08/97-02）", value=dKa)
    rec("A97-04a", dVa == 0.0,
        f"同算序复算 (pf @ Pv).mean(0) == V 逐位相同：max|Δ|={dVa:.1e}（须 ==0）", value=dVa)
    # ★ A97-03b/04b（相对门）：仅旁证 H = pf.mean(0) 与 K 自洽（换算序的浮点差）
    rK = max(float(((H[li] @ st.mems[li].Pk) * st.mems[li].pk_scale - K[li]).norm()
                   / max(float(K[li].norm()), 1e-30)) for li in st.mems)
    rV = max(float((H[li] @ st.mems[li].Pv - V[li]).norm()
                   / max(float(V[li].norm()), 1e-30)) for li in st.mems)
    rec("A97-03b", rK <= 1e-6,
        f"换算序相对差 ‖(H@Pk)s−K‖/‖K‖={rK:.2e}（须 ≤1e-6；float32 eps=1.19e-07）", value=rK)
    rec("A97-04b", rV <= 1e-6,
        f"换算序相对差 ‖H@Pv−V‖/‖V‖={rV:.2e}（须 ≤1e-6）", value=rV)

    # ---- A97-05 count_mounted==0 且 capture 前后 W 逐位不变
    n_mnt1 = count_mounted(model, MOUNT)
    dW = max(float((W91.W_of(st.mems[li]) - Wsnap[li]).abs().max()) for li in st.mems)
    nw_ok = all(m.no_write is False for m in st.mems.values())
    wc_tot = sum(int(m.write_calls) for m in st.mems.values())
    rec("A97-05", n_mnt0 == len(MOUNT) and dW == 0.0 and nw_ok,
        f"count_mounted(model,MOUNT)={n_mnt0}（须 =={len(MOUNT)}，栈已挂上）；"
        f"capture 前后 max|ΔW|={dW:.1e}（须 ==0）；no_write 按层恢复={nw_ok}", value=dW)
    rec("A97-05b", wc_tot == 0,
        f"全程 write_calls 合计={wc_tot}（须 ==0；★ DEF-97-01：野 forward 会真的写入）", value=wc_tot)

    # ---- A97-06 两次 capture 逐位相同
    dH2 = max(float((H[li] - H2[li]).abs().max()) for li in H)
    rec("A97-06", dH2 == 0.0, f"两次 capture 的 H 逐位相同：max|Δ|={dH2:.1e}（须 ==0）", value=dH2)

    # ---- A97-07 去均值版均值为 0
    Hc = centred(H)
    dC = max(float(Hc[li].mean(dim=0).abs().max()) for li in Hc)
    rel_c = max(dC / max(float(H[li].abs().max()), 1e-30) for li in H)
    dC64 = max(float((H[li].double() - H[li].double().mean(dim=0, keepdim=True))
                     .mean(dim=0).abs().max()) for li in H)
    rec("A97-07", rel_c <= 1e-5 and dC64 <= 1e-10,
        f"float32 相对残差={rel_c:.2e}（须 ≤1e-5）；float64 复算 max|mean H_c|={dC64:.2e}（须 ≤1e-10）",
        value=dC64)

    # ---- A97-08 dtype / 非有限值
    dts = {li: str(H[li].dtype) for li in H}
    fin = all(bool(torch.isfinite(H[li]).all()) and bool(torch.isfinite(K[li]).all())
              and bool(torch.isfinite(V[li]).all()) for li in H)
    rec("A97-08", all(v == "torch.float32" for v in dts.values()) and fin,
        f"dtype={list(dts.values())[0]}（须全 torch.float32）；有限={fin}", value=fin)

    # ---- A97-09 dtype 与 ridge_closed 一致：闭式解自洽（float32 会让恒等式假失败，DEF-95-01）
    Yt = torch.randn(32, d, generator=g).to(DEV).double()
    Tt = torch.randn(32, d, generator=g).to(DEV).double()
    M1, _l1, _r1 = ridge_closed(Yt, Tt, 0.05)      # ★ 返回 (M, lam, res)
    M2, _l2, _r2 = ridge_closed(Yt, Tt, 0.05)
    rel_id = float((M1 - M2).norm() / max(float(M1.norm()), 1e-30))
    rec("A97-09", rel_id <= 1e-10,
        f"ridge_closed 两次求解相对差={rel_id:.2e}（须 ≤1e-10）⇒ 解码器确定性；"
        f"解码输入一律 float32（装置惯例）", value=rel_id)

    # ---- A97-10 cond(Pk)/cond(Pv)（仅在需要解释 PROJECTION_LOSS 时承重，此处先算好备用）
    cPk = {str(li): cond_of(st.mems[li].Pk) for li in st.mems}
    cPv = {str(li): cond_of(st.mems[li].Pv) for li in st.mems}
    RES["worked"]["A97-10"] = {"cond_Pk": cPk, "cond_Pv": cPv}
    finite_c = all(math.isfinite(v) for v in list(cPk.values()) + list(cPv.values()))
    rec("A97-10", finite_c,
        f"cond(Pk)={ {k: round(v,1) for k,v in cPk.items()} } "
        f"cond(Pv)={ {k: round(v,1) for k,v in cPv.items()} }（须全有限；896×896 满秩方阵）",
        value=1.0 if finite_c else 0.0)

    # ---- A97-11 与 D-113 交叉核：M1b(V) 须复现 A96-12 的 0.03125（只报告，±1/32 网格）
    mv = m1_identity(V, cls, tr, he, N_CLS)
    b_v = mv["loo_top1_mean"]
    RES["worked"]["A97-11"] = {"M1b_V": b_v, "d113_A96_12_ref": 0.03125}
    rec("A97-11", abs(b_v - 0.03125) <= 1.0 / 32.0 + 1e-12,
        f"M1b(V)={b_v:.6f}（只报告；D-113 的 A96-12 参考 0.03125，容差 ±1/32）", value=b_v)

    return ok


# ─────────────────────────────── 正式轮 ───────────────────────────────
def run_seed(seed, tok, model, facts, chunks, owner, tr, he):
    t0 = time.time()
    cls = value_classes(facts)
    st = make_stack(gamma=0.0)
    H, K, V, _, _ = capture(model, st, facts, chunks, owner)
    Hc, Kc, Vc = centred(H), centred(K), centred(V)

    arms = {}
    for name, X in (("H", H), ("K", K), ("V", V),
                    ("H_c", Hc), ("K_c", Kc), ("V_c", Vc)):
        arms[name] = m1_identity(X, cls, tr, he, N_CLS)
        print(f"  [{name:4s}] ★M1b={arms[name]['loo_top1_mean']:.4f} "
              f"(CAP={CAP_LOO:.4f}) M1a={arms[name]['held_top1_mean']:.4f} "
              f"(仅报告)", flush=True)

    mH = arms["H"]["loo_top1_mean"]
    if mH >= G_AVAIL:
        br = "WS_INFO_AVAILABLE"
    elif mH <= G_ABSENT:
        br = "WS_INFO_ABSENT"
    else:
        br = "WS_INFO_PARTIAL"
    mK, mV = arms["K"]["loo_top1_mean"], arms["V"]["loo_top1_mean"]
    if br == "WS_INFO_AVAILABLE" and (mK <= G_ABSENT or mV <= G_ABSENT):
        loc = "PROJECTION_LOSS"
    elif br == "WS_INFO_AVAILABLE":
        loc = "READ_MAPPING"
    elif mK <= G_ABSENT and mV <= G_ABSENT:
        loc = "WRITE_SIDE_LOSS"
    else:
        loc = "PARTIAL_MIXED"

    info = {"seed": seed, "branch": br, "location": loc, "M1b_H": mH, "M1b_K": mK, "M1b_V": mV,
            "M1b_Hc": arms["H_c"]["loo_top1_mean"], "M1b_Kc": arms["K_c"]["loo_top1_mean"],
            "M1b_Vc": arms["V_c"]["loo_top1_mean"],
            "M1a_H": arms["H"]["held_top1_mean"],
            "cos_geom_H": cos_geom(H), "cos_geom_K": cos_geom(K), "cos_geom_V": cos_geom(V),
            "norm_H": nrm(H), "norm_K": nrm(K), "norm_V": nrm(V),
            "cond_Pk": {str(li): cond_of(st.mems[li].Pk) for li in st.mems},
            "cond_Pv": {str(li): cond_of(st.mems[li].Pv) for li in st.mems},
            "per_layer": {k: v["per_layer"] for k, v in arms.items()},
            "secs": round(time.time() - t0, 1)}
    print(f"  seed {seed}: M1b(H)={mH:.4f} M1b(K)={mK:.4f} M1b(V)={mV:.4f} "
          f"-> {br} / {loc}", flush=True)
    return info


def decide(arms):
    per = {str(a["seed"]): {"branch": a["branch"], "location": a["location"],
                            "M1b_H": a["M1b_H"]} for a in arms}
    brs = {a["branch"] for a in arms}
    n_br = len(brs)                              # ★ DEF-97-05：先算后改，不得 pop() 后再量
    branch = next(iter(brs)) if n_br == 1 else "INCONSISTENT"
    return {"branch": branch, "consistent": n_br == 1, "per_seed": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tok, model = model_and_tok()
    model.eval()
    torch.set_grad_enabled(True)
    facts = R2.build_facts(tok)
    chunks, owner = R2.build_chunks(tok, facts)
    tr = list(range(N_TRAIN_KEYS))
    he = list(range(N_TRAIN_KEYS, N_FACTS))
    print(f"n_facts={len(facts)} chunks={len(chunks)} owner={len(set(owner))}", flush=True)

    if args.stage_a:
        print("===================== Stage A (rev 3.4) =====================", flush=True)
        ok = stage_a(tok, model, facts, chunks, owner, tr, he)
        RES["stage_a_ok"] = ok
        dump(RES, "probe91_wsinfo_guards")
        print("\n* Stage A " + ("全部通过" if ok else "存在 FAIL —— 正式轮被阻断"), flush=True)
        return

    seeds = SEEDS[:1] if args.smoke else SEEDS
    for sd in seeds:
        print(f"\n===== seed {sd} =====", flush=True)
        RES["arms"].append(run_seed(sd, tok, model, facts, chunks, owner, tr, he))
    RES["branch"] = decide(RES["arms"])
    dump(RES, "probe91_wsinfo_smoke" if args.smoke else "probe91_wsinfo")
    b = RES["branch"]
    if not args.smoke:
        print(f"\n* 分支 = {b['branch']}  (consistent={b['consistent']})", flush=True)
        for k, v in b["per_seed"].items():
            print(f"  seed {k}: M1b(H)={v['M1b_H']:.4f} -> {v['branch']} / {v['location']}",
                  flush=True)
    print("\n★ 完成 ⇒ dir91/out/probe91_wsinfo.json", flush=True)


if __name__ == "__main__":
    main()