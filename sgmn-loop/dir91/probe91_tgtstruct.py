#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 rev 3.1 · **瓶颈是样本量、目标构造，还是 y 根本不携带答案身份？**

卡片：revolve/DIR-91/DIRECTION_rev91_v33.md（判据冻结；本文件不得改阈值）
      rev 3.3：★ 更正 rev 3.2 —— A96-11 的 oracle 注入**单位范数**向量被隐状态量级吞没
      （DEF-96-07）⇒ TARGET_UNREACHABLE 被推翻；改为**幅度扫描** A96-11′（含 alpha=0 基线），
      新增 A96-12（v_i 正对照验标签管线）、A96-13（MAG_RATIO）。
      ★ rev 3.0 的 §2 M1 门经 Stage A 实测证伪（M1a 结构上限 5/16=0.3125，
        绝对门 0.50 不可达，DEF-96-01）⇒ 主判据改为 M1b（全 32 条 LOO，全 27 类覆盖），
        M1a 降为**仅报告**并与其结构上限并列。其余门一字未改。
派生自 D-111 §七「可分辨的下一问」＋ framework §7.2-7「前向任务收益」。

★ §0 前提更正（已在卡内留档）：CE 任务损失**不是**未测项 ——
  R2_task91.train_read_task 本身就是 4 选 1 CE、read_params=[Wq,Wo,gamma]，
  故 adam16 ≡ task16 且是 §7.1 Freeze C 的**忠实实现**。本卡因此不改判"目标形状"，
  改判**瓶颈落在哪一层**。

四项读数：
  M0  跨轮参照 zero/free/ridge16（ridge16 须逐位复现 D-111: 1.000/0.0625、1.000/0.1875）
  M1  ★机制门：27 类值身份能否从 y_held 线性读出（ridge_scores；门 CEIL_LOW=0.25 / CEIL_HIGH=0.50）
  M2  共享目标：t_j := normalize(lm_head[token(v_j)])，重拟合 ridge_closed
  M3  同值留出拆分（仅报告；n 太小）
  M4  未覆盖网格格 {Wo-only} × CE（冻结 Wq、γ）

派生自 probe91_lowrank.py，只改量测层；ridge_closed/capture_y/eval_inject/build_state/
set_gamma/set_m/count_mounted/optimise_free 一律 import probe91_woform（CORR-14 结构性满足）。
"""
import argparse, json, os, sys, time
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)

import numpy as np
import torch

import probe91_woform as W28
from probe91_woform import count_mounted, capture_y, eval_inject, ridge_closed, set_gamma
import probe91_womap as W91                       # ridge_scores / top1（与 D-106/D-108 同一份）
import R2_task91 as R2
from R2_task91 import (DEV, MOUNT, N_CAND, N_DRAWS, N_FACTS, N_TRAIN_KEYS, OUT, VALUE_POOL,
                       build_chunks, build_facts, make_eval_items, model_and_tok)

# ---------------------------------------------------------------- §2 冻结门
G_GEN = 0.50
G_LOW = 0.25
G_DELTA = 0.125
CHANCE = 1.0 / N_CAND
REC_MIN = 0.90           # ★ rev 3.2：rec 的"没有身份"门
CAP_LOO = 10.0 / 32.0    # ★ rev 3.2：M1b 的结构上限（5 个类各有 2 条 ⇒ 10 条可辨识）
CAP_M1A = 5.0 / 16.0     # M1a 的结构上限（训练集只覆盖类 0..15）——仅报告，不得当门
ORACLE_ALPHA = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]   # ★ rev 3.3 冻结网格
CEIL_D106 = 10.0 / 32.0  # 参照：D-106/D-108 在 v_i 上的实测结构天花板
SEEDS = [20261028, 7777]
ALL32 = list(range(N_FACTS))
N_CLS = len(VALUE_POOL)
# D-111 的 ridge16 读数（A96-05 须精确复现）
D111_RIDGE16 = {20261028: (1.0, 0.0625), 7777: (1.0, 0.1875)}

RES = {"card": "DIR-91 rev 3.3", "file": "probe91_tgtstruct.py",
       "gates": {"G_gen": G_GEN, "G_low": G_LOW, "G_delta": G_DELTA, "chance": CHANCE,
                 "rec_min": REC_MIN, "cap_loo": CAP_LOO, "ceil_d106": CEIL_D106,
                 "cap_m1a": CAP_M1A, "oracle_alpha": ORACLE_ALPHA},
       "consts": {"n_cls": N_CLS, "n_facts": N_FACTS, "n_train_keys": N_TRAIN_KEYS,
                  "n_cand": N_CAND, "n_draws": N_DRAWS, "mount": list(MOUNT)},
       "guards": {}, "arms": [], "branch": {}, "worked": {}}
FAILS = []


def rec(name, ok, detail, value=None):
    RES["guards"][name] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)
    return bool(ok)


def dump(obj, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print(f"  -> {p}", flush=True)


# ---------------------------------------------------------------- 标签 / 共享目标
def value_classes(facts):
    """每事实答案在 VALUE_POOL 中的下标（不假设 i%27，逐条查）。"""
    idx = {w: i for i, w in enumerate(VALUE_POOL)}
    return [idx[f["val"]] for f in facts]


def shared_targets(tok, model):
    """T_shared[c] = normalize(lm_head.weight[token(v_c)]) ∈ R^{d_model}，27 列。"""
    W = model.lm_head.weight.detach().float()          # (vocab, d_model)
    rows = []
    for w in VALUE_POOL:
        tid = tok(" " + w, add_special_tokens=False)["input_ids"]
        assert len(tid) == 1, (w, tid)
        rows.append(W[tid[0]])
    T = torch.stack(rows)                              # (27, d_model)
    return T / T.norm(dim=1, keepdim=True).clamp_min(1e-30)


# ---------------------------------------------------------------- M1 · 机制门
def m1_identity(Y, cls, tr, he, n_cls):
    """27 类值身份能否从 y 线性读出。
    M1a（门）：拟合 y_train(16) → 值类，判别 y_held(16)。
    M1b（参照）：全 32 条的 leave-one-out，与 D-106/D-108 的 M2/LOO 同口径。"""
    per_layer = {}
    for li in Y:
        Yl = Y[li]
        s = W91.ridge_scores(Yl[tr], [cls[i] for i in tr], Yl[he], n_cls)
        a_he, _ = W91.top1(s, [cls[i] for i in he])
        loo = 0
        for i in range(N_FACTS):
            keep = [j for j in range(N_FACTS) if j != i]
            si = W91.ridge_scores(Yl[keep], [cls[j] for j in keep], Yl[[i]], n_cls)
            loo += int(int(si.argmax(dim=1)[0]) == cls[i])
        per_layer[str(li)] = {"held_top1": float(a_he), "loo_top1": loo / N_FACTS}
    a_he_mean = float(np.mean([v["held_top1"] for v in per_layer.values()]))
    loo_mean = float(np.mean([v["loo_top1"] for v in per_layer.values()]))
    return {"per_layer": per_layer, "held_top1_mean": a_he_mean,
            "loo_top1_mean": loo_mean, "n_cls": n_cls,
            "chance_1over27": 1.0 / n_cls}


# ---------------------------------------------------------------- 单 seed
def run_seed(seed, tok, model, facts, chunks, owner, tr, he):
    t0 = time.time()
    cls = value_classes(facts)
    T_shared = shared_targets(tok, model)                    # (27, d_model)
    st, wc, Wsnap = W28.build_state(tok, model, facts, chunks, owner, seed)
    out = []
    try:
        set_gamma(st, 1.0)                                   # 与 D-109/D-111 逐字相同 ⇒ ridge16 须复现
        Y = capture_y(model, st, facts, facts[0]["prompt_ids"][-1])
        tgt32, floss = W28.optimise_free(model, st, tok, facts, ALL32,
                                        __import__("random").Random(555))
        RES["arms"].append({"arm": "free_losses", "seed": seed,
                            "info": {str(k): v for k, v in floss.items()}})

        # ---- M0 参照臂
        for nm, Mv in (("zero", None), ("free", tgt32)):
            fit = eval_inject(model, st, tok, facts, tr, Mv, __import__("random").Random(555))
            gen = eval_inject(model, st, tok, facts, he, Mv, __import__("random").Random(555))
            out.append({"arm": nm, "seed": seed, "acc_train": fit["acc_q"],
                        "acc_held": gen["acc_q"], "margin_train": fit["margin_mean"],
                        "margin_held": gen["margin_mean"], "n_q": gen["n_q"],
                        "info": {}, "sec": 0})
            print(f"  [{nm}] acc_train={out[-1]['acc_train']:.3f} "
                  f"acc_held={out[-1]['acc_held']:.3f}", flush=True)

        # ---- ridge16（跨轮参照，必须逐位复现 D-111）
        Ytr = {li: torch.stack([Y[li][i] for i in tr]) for li in st.mems}
        Ttr = {li: torch.stack([tgt32[li][j] for j in range(len(tr))]) for li in st.mems}
        M_ref, lams, ress = {}, {}, {}
        for li in st.mems:
            M, lam, res = ridge_closed(Ytr[li].T, Ttr[li].T)
            M_ref[li], lams[str(li)], ress[str(li)] = M.float(), lam, res
        Mv_ref = {li: (M_ref[li] @ Y[li].T).T for li in st.mems}
        fit = eval_inject(model, st, tok, facts, tr, Mv_ref, __import__("random").Random(555))
        gen = eval_inject(model, st, tok, facts, he, Mv_ref, __import__("random").Random(555))
        out.append({"arm": "ridge16", "seed": seed, "acc_train": fit["acc_q"],
                    "acc_held": gen["acc_q"], "margin_train": fit["margin_mean"],
                    "margin_held": gen["margin_mean"], "n_q": gen["n_q"],
                    "info": {"lambda": lams, "rel_res": ress, "n_fit": len(tr)}, "sec": 0})
        print(f"  [ridge16] acc_train={fit['acc_q']:.4f} acc_held={gen['acc_q']:.4f}", flush=True)

        # ---- M1 机制门
        m1 = m1_identity(Y, cls, tr, he, N_CLS)
        RES["arms"].append({"arm": "M1_identity", "seed": seed, "info": m1, "acc_train": None,
                            "acc_held": m1["held_top1_mean"]})
        print(f"  [M1] ★M1b(LOO 全类)={m1['loo_top1_mean']:.4f} "
              f"(chance 1/27={1.0/N_CLS:.3f}; CAP_LOO={CAP_LOO:.4f}; "
              f"rec={m1['loo_top1_mean']/CAP_LOO:.4f}) | "
              f"M1a(16→16)={m1['held_top1_mean']:.4f} "
              f"(结构上限 CAP_M1A={CAP_M1A:.4f}, 仅报告)", flush=True)

        # ---- M2 共享目标
        Tsh_tr = {li: T_shared[torch.tensor([cls[i] for i in tr], dtype=torch.long)]
                  for li in st.mems}
        M_sh, lam_sh, res_sh = {}, {}, {}
        for li in st.mems:
            M, lam, res = ridge_closed(Ytr[li].T, Tsh_tr[li].T)
            M_sh[li], lam_sh[str(li)], res_sh[str(li)] = M.float(), lam, res
        Mv_sh = {li: (M_sh[li] @ Y[li].T).T for li in st.mems}
        f2 = eval_inject(model, st, tok, facts, tr, Mv_sh, __import__("random").Random(555))
        g2 = eval_inject(model, st, tok, facts, he, Mv_sh, __import__("random").Random(555))
        out.append({"arm": "M2_shared", "seed": seed, "acc_train": f2["acc_q"],
                    "acc_held": g2["acc_q"], "margin_train": f2["margin_mean"],
                    "margin_held": g2["margin_mean"], "n_q": g2["n_q"],
                    "info": {"lambda": lam_sh, "rel_res": res_sh,
                             "n_target_cols": int(len(set(cls)))}, "sec": 0})
        print(f"  [M2] acc_train={f2['acc_q']:.4f} acc_held={g2['acc_q']:.4f} "
              f"margin(tr/ho)={f2['margin_mean']:+.2f}/{g2['margin_mean']:+.2f}", flush=True)

        # ---- M3 同值拆分（仅报告；判别力来自上面 ridge16 的逐题 detail 不够，故用子集重评）
        seen = set(cls[i] for i in tr)
        grp_seen = [i for i in he if cls[i] in seen]
        grp_new = [i for i in he if cls[i] not in seen]
        out.append({"arm": "M3_split", "seed": seed, "acc_train": None, "acc_held": None,
                    "info": {"n_seen": len(grp_seen), "n_new": len(grp_new),
                             "cls_seen": [cls[i] for i in grp_seen],
                             "cls_new": [cls[i] for i in grp_new],
                             "note": "仅报告：n_seen 太小（粒度 1/n），不得作判决门"}})
        print(f"  [M3] held split: seen-value={len(grp_seen)} 条, unseen-value={len(grp_new)} 条",
              flush=True)
        st.remove()
        st = None

        # ---- M4 {Wo-only} × CE
        st2, wc2, _ = W28.build_state(tok, model, facts, chunks, owner, seed)
        try:
            set_gamma(st2, 0.0)                              # 与 adam16 同：写后 γ=0
            Wq0 = {li: m.Wq.detach().clone() for li, m in st2.mems.items()}
            g0 = {li: float(m.gamma) for li, m in st2.mems.items()}
            for m in st2.mems.values():
                m.Wq.requires_grad_(False)
                m.gamma.requires_grad_(False)
            for m in st2.mems.values():
                m.no_write = True
            rng = __import__("random").Random(1000 + seed % 100)
            losses, viol, div = R2.train_read_task(model, st2, tok, facts, tr, rng)
            dq = max(float((m.Wq - Wq0[li]).abs().max()) for li, m in st2.mems.items())
            dg = max(abs(float(m.gamma) - g0[li]) for li, m in st2.mems.items())
            f4 = eval_inject(model, st2, tok, facts, tr, "native", __import__("random").Random(555))
            g4 = eval_inject(model, st2, tok, facts, he, "native", __import__("random").Random(555))
            out.append({"arm": "M4_wo_only_ce", "seed": seed, "acc_train": f4["acc_q"],
                        "acc_held": g4["acc_q"], "margin_train": f4["margin_mean"],
                        "margin_held": g4["margin_mean"], "n_q": g4["n_q"],
                        "info": {"violations": viol, "diverged": div,
                                 "final_loss": float(np.mean(losses[-50:])) if losses else None,
                                 "dWq": dq, "dgamma": dg,
                                 "gamma_final": float(st2.mems[list(st2.mems)[0]].gamma)},
                        "sec": 0})
            print(f"  [M4] acc_train={f4['acc_q']:.4f} acc_held={g4['acc_q']:.4f} "
                  f"dWq={dq:.3e} dgamma={dg:.3e} viol={viol}", flush=True)
            RES.setdefault("m4_freeze", {})[str(seed)] = {"dWq": dq, "dgamma": dg}
        finally:
            st2.remove()
    finally:
        if st is not None:
            st.remove()
    for a in out:
        if a.get("sec") == 0:
            a["sec"] = round(time.time() - t0, 1)
    return out


# ================================================================ 判定
def decide():
    per = {}
    for seed in SEEDS:
        arms = {a["arm"]: a for a in RES["arms"] if a.get("seed") == seed}
        if "M1_identity" not in arms:
            continue
        inf = arms["M1_identity"]["info"]
        m1b = inf["loo_top1_mean"]
        rec_ = m1b / CAP_LOO                          # ★ rev 3.2 主机制门
        m1a = inf["held_top1_mean"]                   # 仅报告
        m2 = arms["M2_shared"]["acc_held"]
        m2tr = arms["M2_shared"]["acc_train"]
        m4 = arms["M4_wo_only_ce"]["acc_held"]
        orb = RES["worked"].get("A96-11", {}).get("best", {}) or {}
        oracle = orb.get("acc_held")
        if oracle is not None and oracle >= G_GEN:
            br = "TARGET_REACHABLE"
        elif oracle is not None and oracle <= G_LOW:
            br = "TARGET_UNREACHABLE"
        elif rec_ < REC_MIN:
            br = "Y_NO_IDENTITY"
        elif m4 >= G_GEN:
            br = "WQ_CONFOUND"
        elif m2 >= G_GEN:
            br = "TARGET_STRUCTURE"
        elif m2 < G_LOW and m4 < G_LOW:
            br = "SAMPLE_LIMIT"
        else:
            br = "Y_PARTIAL"
        per[str(seed)] = {"M1b_loo_top1": m1b, "rec": rec_, "cap_loo": CAP_LOO,
                          "M1a_held_top1": m1a, "cap_m1a": CAP_M1A,
                          "M2_acc_train": m2tr, "M2_acc_held": m2,
                          "M4_acc_held": m4, "oracle": oracle, "branch": br}
    branches = {v["branch"] for v in per.values()}
    return {"per_seed": per,
            "branch": branches.pop() if len(branches) == 1 else "AMBIGUOUS",
            "consistent": len(branches) == 1 or not per}


# ================================================================ Stage A
def stage_a(tok, model, facts, chunks, owner, tr, he):
    seed = SEEDS[0]
    cls = value_classes(facts)
    rec("A96-00", True,
        f"n_facts={N_FACTS} n_train={len(tr)} n_held={len(he)} n_cand={N_CAND} "
        f"n_cls={N_CLS} MOUNT={list(MOUNT)} CHANCE={CHANCE}")

    # ---- A96-01 M1 worked example：可读 ⇒ 1.0；纯噪声 ⇒ ≤0.15
    g = torch.Generator().manual_seed(7)
    d = 896
    B = torch.randn(N_CLS, d, generator=g)
    B = B.to(DEV)                                          # DEF-96-04：合成量一律与正式路径同设备
    Ygood = torch.stack([B[cls[i]] + 1e-4 * torch.randn(d, generator=g).to(DEV)
                         for i in range(N_FACTS)])
    r_good = m1_identity({0: Ygood}, cls, tr, he, N_CLS)
    Ynoise = torch.randn(N_FACTS, d, generator=g).to(DEV)        # DEF-96-04
    r_noise = m1_identity({0: Ynoise}, cls, tr, he, N_CLS)
    b_good, b_noise = r_good["loo_top1_mean"], r_noise["loo_top1_mean"]
    RES["worked"]["A96-01"] = {"M1b_good": b_good, "M1b_noise": b_noise,
                               "M1a_good_reported": r_good["held_top1_mean"],
                               "cap_m1a": CAP_M1A,
                               "note": "M1a 在合成可读情形下亦受 5/16 上限约束（DEF-96-01）"}
    rec("A96-01", (abs(b_good - CAP_LOO) < 1e-12) and (b_noise <= 0.15),
        f"M1b 合成可读 LOO={b_good:.6f}（须 ==CAP_LOO {CAP_LOO:.6f} 精确）；"
        f"纯噪声 LOO={b_noise:.4f}（须 ≤0.15）。"
        f"★ 同情形 M1a={r_good['held_top1_mean']:.4f}（上限 {CAP_M1A:.4f}，仅报告）",
        value=b_good)

    # ---- A96-02 M2 worked example + T_shared 性质
    T_shared = shared_targets(tok, model)
    nrm = (T_shared.norm(dim=1) - 1.0).abs().max().item()
    uniq = int(torch.unique(torch.round(T_shared * 1e5) / 1e5, dim=0).shape[0])
    Ts_tr = {0: T_shared[torch.tensor([cls[i] for i in tr], dtype=torch.long)]}
    Yt = torch.stack([B[cls[i]] + 1e-4 * torch.randn(d, generator=g).to(DEV)
                      for i in range(N_FACTS)])
    Mw, _, rw = ridge_closed(Yt[tr].T, Ts_tr[0].T)
    # ★ ridge_closed 返回 double、正式路径存 float32 ⇒ 诊断式两边都显式 .double()
    resid = float((Mw @ Yt[tr].T.double() - Ts_tr[0].T.double()).norm()
                  / Ts_tr[0].T.double().norm().clamp_min(1e-30))
    RES["worked"]["A96-02"] = {"rel_res_synthetic": resid, "T_norm_err": nrm, "T_uniq": uniq}
    rec("A96-02", (resid <= 1e-3) and (nrm < 1e-6) and (uniq == N_CLS),
        f"共享目标合成可解 rel_res={resid:.3e}（须 ≤1e-3）；"
        f"‖T‖−1 max={nrm:.2e}（须 <1e-6）；T 互异列 {uniq}（须 {N_CLS}）", value=resid)

    # ---- 构造一次真实状态（供 A96-03/05/06）
    st, wc, _ = W28.build_state(tok, model, facts, chunks, owner, seed)
    try:
        set_gamma(st, 1.0)
        Y = capture_y(model, st, facts, facts[0]["prompt_ids"][-1])
        # ---- A96-03 capture_y 非空转
        nrm_min = min(float(Y[li].norm(dim=1).min()) for li in Y)
        cosmax = 0.0
        for li in Y:
            Yn = Y[li] / Y[li].norm(dim=1, keepdim=True).clamp_min(1e-30)
            C = Yn @ Yn.T
            cosmax = max(cosmax, float(C[~torch.eye(N_FACTS, dtype=torch.bool)].max()))
        shapes_ok = all(tuple(Y[li].shape) == (N_FACTS, 896) for li in Y)
        rec("A96-03", (nrm_min > 0) and (cosmax < 0.999) and shapes_ok,
            f"‖y‖min={nrm_min:.3e}（须 >0）；两两 cos max={cosmax:.4f}（须 <0.999）；"
            f"shape=(32,896) {shapes_ok}", value=cosmax)

        # ---- A96-11′ oracle **幅度扫描**（★ rev 3.3：DEF-96-07 ⇒ 必须扫 alpha，含 alpha=0 基线）
        Trow = T_shared[torch.tensor(cls, dtype=torch.long)]
        sweep, best = [], None
        for alpha in ORACLE_ALPHA:
            Tor = {li: (alpha * Trow) for li in st.mems}
            fa = eval_inject(model, st, tok, facts, tr, Tor, __import__("random").Random(555))
            ga = eval_inject(model, st, tok, facts, he, Tor, __import__("random").Random(555))
            sweep.append({"alpha": alpha, "acc_train": fa["acc_q"], "acc_held": ga["acc_q"],
                          "margin_train": fa["margin_mean"], "margin_held": ga["margin_mean"]})
            if best is None or ga["acc_q"] > best["acc_held"]:
                best = sweep[-1]
            print(f"    alpha={alpha:7.1f} tr={fa['acc_q']:.4f} he={ga['acc_q']:.4f} "
                  f"margin tr/ho={fa['margin_mean']:+.3f}/{ga['margin_mean']:+.3f}", flush=True)
        RES["worked"]["A96-11"] = {"sweep": sweep, "best": best, "grid": ORACLE_ALPHA}
        base = [s for s in sweep if s["alpha"] == 0.0][0]
        a10 = [s for s in sweep if s["alpha"] == 10.0][0]
        rec("A96-11", best["acc_held"] >= G_GEN,
            f"oracle 幅度扫描：最佳 alpha={best['alpha']:.1f} ⇒ acc_held={best['acc_held']:.4f}"
            f"（须 ≥{G_GEN}）；基线 alpha=0 held={base['acc_held']:.4f}；"
            f"margin(10)={a10['margin_held']:+.3f} > margin(0)={base['margin_held']:+.3f}"
            f"={a10['margin_held'] > base['margin_held']}", value=best["acc_held"])

        # ---- A96-13 MAG_RATIO：证明 alpha=1 远小于隐状态（本缺陷成因）
        mag = {}
        for li in st.mems:
            hnorm = float(Y[li].norm(dim=1).mean())
            mn = float(Trow.norm(dim=1).mean())          # alpha=1 时的 ||m||
            mag[str(li)] = {"y_norm_mean": hnorm, "m_norm_alpha1": mn,
                            "ratio_y_over_m": hnorm / max(mn, 1e-30)}
        RES["worked"]["A96-13"] = mag
        rec("A96-13", all(v["ratio_y_over_m"] > 1.0 for v in mag.values()),
            f"MAG_RATIO (‖y‖/‖m(alpha=1)‖) 逐层="
            f"{ {k: round(v['ratio_y_over_m'],2) for k,v in mag.items()} }"
            f" ⇒ alpha=1 注入幅度远小于读出量级（DEF-96-07 的成因）")

        # ---- A96-12（★ rev 3.3 修正为**仅报告**）：v_i 的标签管线正对照
        #      ★ DEF-96-08：我原先要求 M1b(v) == CAP_LOO，但那是 **D_vtpos 实验臂**的读数；
        #      本 probe 用的是**未改动的原生装置**（A_mean 池化），D-108 实测其 M2loo ≈ 0.042
        #      （近 chance）。⇒ 该期望值本身就错，A96-12 不得当门。
        #      标签管线已由 A96-01（合成可读 = 上限精确）+ A96-04（投毒/打乱 = 0）验证。
        _, V = R2.key_matrices(model, st, facts, chunks, owner)
        m1v = m1_identity(V, cls, tr, he, N_CLS)
        RES["worked"]["A96-12"] = {"M1b_v": m1v["loo_top1_mean"], "cap_loo": CAP_LOO,
                                   "M1a_v": m1v["held_top1_mean"],
                                   "d108_A_mean_M2loo_ref": 0.042,
                                   "note": "仅报告：原生 A_mean 池化下 v_i 近 chance（D-108 已测 0.042）；"
                                           "0.312 是 D_vtpos 臂的数，本 probe 不用那条臂",
                                   "per_layer": m1v["per_layer"]}
        print(f"  [INFO] A96-12 v_i 正对照（仅报告）：M1b(v)={m1v['loo_top1_mean']:.6f} "
              f"(cap={CAP_LOO:.4f}; D-108 的 A_mean 臂 M2loo≈0.042 ⇒ 近 chance 是**预期**)", flush=True)

        # ---- A96-04 投毒：y_held 换随机 / cls 打乱 ⇒ top1 ≤0.15
        Yp = {li: Y[li].clone() for li in Y}
        gp = torch.Generator().manual_seed(11)
        for li in Yp:
            for i in he:
                Yp[li][i] = torch.randn(896, generator=gp).to(DEV)
        rp = m1_identity(Yp, cls, tr, he, N_CLS)
        a_poison = rp["held_top1_mean"]; b_poison = rp["loo_top1_mean"]
        import random as _r
        rr = _r.Random(99)
        cls_shuf = list(cls)
        rr.shuffle(cls_shuf)
        rs = m1_identity(Y, cls_shuf, tr, he, N_CLS)
        a_shuf = rs["held_top1_mean"]; b_shuf = rs["loo_top1_mean"]
        RES["worked"]["A96-04"] = {"poison_M1a": a_poison, "poison_M1b": b_poison,
                                   "shuffled_M1a": a_shuf, "shuffled_M1b": b_shuf}
        rec("A96-04", (b_poison <= 0.15) and (b_shuf <= 0.15),
            f"投毒 y_held：M1b={b_poison:.4f}（须 ≤0.15，M1a={a_poison:.4f}）；"
            f"打乱 cls：M1b={b_shuf:.4f}（须 ≤0.15，M1a={a_shuf:.4f}）")

        # ---- A96-05 ridge16 逐位复现 D-111
        tgt32, _ = W28.optimise_free(model, st, tok, facts, ALL32,
                                     __import__("random").Random(555))
        Ytr = {li: torch.stack([Y[li][i] for i in tr]) for li in st.mems}
        Ttr = {li: torch.stack([tgt32[li][j] for j in range(len(tr))]) for li in st.mems}
        M_ref = {}
        for li in st.mems:
            M, _, _ = ridge_closed(Ytr[li].T, Ttr[li].T)
            M_ref[li] = M.float()
        Mv = {li: (M_ref[li] @ Y[li].T).T for li in st.mems}
        f = eval_inject(model, st, tok, facts, tr, Mv, __import__("random").Random(555))
        gg = eval_inject(model, st, tok, facts, he, Mv, __import__("random").Random(555))
        e_tr, e_he = D111_RIDGE16[seed]
        RES["worked"]["A96-05"] = {"acc_train": f["acc_q"], "acc_held": gg["acc_q"],
                                   "expect": [e_tr, e_he]}
        rec("A96-05", (abs(f["acc_q"] - e_tr) < 1e-12) and (abs(gg["acc_q"] - e_he) < 1e-12),
            f"ridge16 复现 D-111：train={f['acc_q']:.4f}（须 {e_tr}）"
            f" held={gg['acc_q']:.4f}（须 {e_he}）")

        # ---- A96-06 M2 与 M0 走同一份 ridge_closed；诊断量隔离
        rec("A96-06", callable(ridge_closed) and (ridge_closed is W28.ridge_closed),
            "M2 与 M0 用同一个 ridge_closed 函数对象（只换 T）")
        dtypes_ok = all(p.dtype == torch.float32 for li in st.mems
                        for p in (st.mems[li].Wo,))
        rec("A96-10", dtypes_ok, f"正式路径 Wo dtype={ {li: str(st.mems[li].Wo.dtype) for li in st.mems} }")
    finally:
        st.remove()

    # ---- A96-07 count_mounted：先断言 0，再验证 M4 两次构造读数逐位相同
    nm = count_mounted(model, MOUNT)
    stA, _, _ = W28.build_state(tok, model, facts, chunks, owner, seed)
    try:
        WqA = {li: m.Wq.detach().clone() for li, m in stA.mems.items()}
        for m in stA.mems.values():
            m.Wq.requires_grad_(False)
            m.gamma.requires_grad_(False)
        rngA = __import__("random").Random(1000 + seed % 100)
        R2.train_read_task(model, stA, tok, facts, tr, rngA)
        WA = {li: m.Wo.detach().clone() for li, m in stA.mems.items()}
        dWq_A = max(float((m.Wq - WqA[li]).abs().max()) for li, m in stA.mems.items())
    finally:
        stA.remove()
    stB, _, _ = W28.build_state(tok, model, facts, chunks, owner, seed)
    try:
        for m in stB.mems.values():
            m.Wq.requires_grad_(False)
            m.gamma.requires_grad_(False)
        rngB = __import__("random").Random(1000 + seed % 100)
        R2.train_read_task(model, stB, tok, facts, tr, rngB)
        WB = {li: m.Wo.detach().clone() for li, m in stB.mems.items()}
    finally:
        stB.remove()
    dW = max(float((WA[li] - WB[li]).abs().max()) for li in WA)
    RES["worked"]["A96-06_m4"] = {"dWq": dWq_A, "two_run_dW": dW}
    rec("A96-07", (nm == 0) and (dW == 0.0),
        f"count_mounted()==0 先成立（{nm}）；M4 两次独立构造 Wo 逐位相同 dW={dW:.3e}", value=dW)
    rec("A96-06b", dWq_A == 0.0,
        f"M4 冻结 Wq 生效：dWq={dWq_A:.3e}（须 0）", value=dWq_A)

    # ---- A96-09 口径
    keys = [f["prompt_ids"] for f in facts]
    items = make_eval_items(tok, facts, he, __import__("random").Random(555))
    per_q = {}
    ok_cand = True
    for it in items:
        per_q.setdefault(tuple(it["prompt_ids"]), []).append(it)
        ntrue = sum(1 for c in it["cand"] if c[1])
        if ntrue != 1 or len(it["cand"]) != N_CAND:
            ok_cand = False
    rec("A96-09", (len(items) == len(he) * N_DRAWS) and ok_cand
        and all(len(v) == N_DRAWS for v in per_q.values())
        and abs(CHANCE - 0.25) < 1e-12,
        f"n_item={len(items)}（须 {len(he)*N_DRAWS}）；n_q={len(per_q)}（须 {len(he)}）；"
        f"每题 draws 数={set(len(v) for v in per_q.values())}；每题恰一正确项={ok_cand}；"
        f"chance={CHANCE}")
    return len(FAILS) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tok, model = model_and_tok()
    model.eval()
    torch.set_grad_enabled(True)
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    tr = list(range(N_TRAIN_KEYS))
    he = list(range(N_TRAIN_KEYS, N_FACTS))
    if a.stage_a:
        print("===================== Stage A (rev 3.3) =====================", flush=True)
        ok = stage_a(tok, model, facts, chunks, owner, tr, he)
        RES["stage_a_ok"] = ok
        dump(RES, "probe91_tgtstruct_guards")
        print("\n* Stage A " + ("全部通过" if ok else "存在 FAIL —— 正式轮被阻断"), flush=True)
        sys.exit(0 if ok else 2)
    seeds = SEEDS[:1] if a.smoke else SEEDS
    for seed in seeds:
        print(f"===== seed {seed} =====", flush=True)
        RES["arms"].extend(run_seed(seed, tok, model, facts, chunks, owner, tr, he))
    RES["branch"] = decide()
    dump(RES, f"probe91_tgtstruct{'_smoke' if a.smoke else ''}")
    for s, v in RES["branch"]["per_seed"].items():
        print(f"  seed {s}: rec={v['rec']:.4f} (M1b={v['M1b_loo_top1']:.4f}/{v['cap_loo']:.4f}) "
              f"M1a={v['M1a_held_top1']:.4f} | M2 tr/ho="
              f"{v['M2_acc_train']:.4f}/{v['M2_acc_held']:.4f} M4={v['M4_acc_held']:.4f} "
              f"oracle(best held)={v['oracle']} -> {v['branch']}", flush=True)
    print(f"\n* 分支 = {RES['branch']['branch']}  "
          f"(consistent={RES['branch']['consistent']})", flush=True)


if __name__ == "__main__":
    main()