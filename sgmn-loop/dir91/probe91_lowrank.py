#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 rev 2.9 · **瓶颈是参数太多还是样本太少？低秩映射 M = A B^T 能否只用 16 条样本泛化**

卡片：revolve/DIR-91/DIRECTION_rev91_v29.md（判据冻结；本文件不得改阈值）
派生自 D-109 §七「可分辨的下一问」。

唯一独立变量 = **M 的秩上界 r**（r ∈ {1,2,4,8,12,16}）。其余一切逐字继承 probe91_woform.py：
  * 直接 import probe91_woform，**复用同一份** ridge_closed / eval_inject / capture_y /
    build_state / set_gamma / set_m / count_mounted / optimise_free ⇒ CORR-14「必须走正式轮那条
    代码路径」是结构性满足的（不存在第二份实现可以对不上）。
  * 写侧一律 A_mean、**写阶段 γ=0**、写后 W 冻结、所有臂共用一次写 ⇒ 各臂 W 逐位相同。
  * 目标向量 t_j 来自 optimise_free（Random(555)、STEPS_FREE=120、LR_FREE=0.05）⇒ 与 D-109 逐字相同。

★ 秩约束岭回归用**精确闭式解**，不用 ALS：
    min ||M Y − T||_F² + λ||M||_F²   s.t. rank(M) ≤ r
  令 Y = UΣVᵀ、G = TVΣ、W̃* = G(Σ²+λI)^{-1/2}、W̃_r = trunc_r(W̃*)、M_r = W̃_r(Σ²+λI)^{-1/2}Uᵀ。
  r ≥ rank(Y) 时代数恒等于 ridge_closed 的 M_λ ⇒ **r=16 是"约束不起作用"的精确控制臂**（A95-02）。
  用 ALS 会引入局部最优，"低秩失败"就与"参数个数不是瓶颈"无法区分（CORR-13）。
"""
import argparse, json, os, sys, time
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)

import numpy as np
import torch

import sgmn_module
from sgmn_module import LayerMem
import probe91_woform as W28
from probe91_woform import count_mounted, capture_y, eval_inject, ridge_closed, set_gamma, set_m
import R2_task91 as R2
from R2_task91 import (ALPHA, B as N_CHUNKS, CHUNK, DEV, ETA_WRITE, LAM, MOUNT, N_CAND,
                       N_DRAWS, N_FACTS, N_TRAIN_KEYS, OUT, READ_LR, READ_STEPS,
                       acc_by_question, build_chunks, build_facts, make_eval_items,
                       make_stack, model_and_tok, score_and_accuracy)

# ---------------------------------------------------------------- §2 冻结
G_GEN = 0.50
G_LOW = 0.25
G_DELTA = 0.125
CHANCE = 1.0 / N_CAND
SEEDS = [20261028, 7777]
R_GRID = [1, 2, 4, 8, 12, 16]
R_WITNESS = [1, 2, 4, 8]           # 只有这些 r 可以充当 PARAM_LIMIT 的证人（r=12 是桥、r=16 是控制）
R_CONTROL = 16
RES_NULL_RATIO = 0.90
RIDGE_REL = W28.RIDGE_REL
STEPS_FREE, LR_FREE = W28.STEPS_FREE, W28.LR_FREE
ALL32 = list(range(N_FACTS))

RES = {"card": "DIR-91 rev 2.9", "file": "probe91_lowrank.py",
       "gates": {"G_gen": G_GEN, "G_low": G_LOW, "G_delta": G_DELTA, "chance": CHANCE,
                 "res_null_ratio": RES_NULL_RATIO},
       "consts": {"r_grid": R_GRID, "r_witness": R_WITNESS, "r_control": R_CONTROL,
                  "ridge_rel": RIDGE_REL, "steps_free": STEPS_FREE, "lr_free": LR_FREE,
                  "n_train_keys": N_TRAIN_KEYS, "n_facts": N_FACTS, "n_cand": N_CAND,
                  "n_draws": N_DRAWS, "n_chunk": N_CHUNKS, "mount": list(MOUNT)},
       "guards": {}, "arms": [], "branch": {}, "worked": {}}
FAILS = []


def rec(name, ok, detail, value=None):
    RES["guards"][name] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"{'[PASS]' if ok else '[FAIL]'} {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def dump(obj, name):
    p = OUT / f"{name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print(f"  -> {p}", flush=True)


# ================================================================ 秩约束岭（精确闭式）
def lowrank_closed(Y, T, r, rel=RIDGE_REL, check_rank=False):
    """min ||M Y − T||_F² + λ||M||_F²  s.t. rank(M) ≤ r  ——  精确闭式解。

    Y: (d_v, n) 输入（每列一个事实的 y）; T: (d_model, n) 目标。
    λ 与 ridge_closed 逐字相同：λ = rel·tr(YYᵀ)/n。
    返回 (M, lam, rel_res, q, r_eff, rank_num)。
    """
    Yd, Td = Y.double(), T.double()
    n = Yd.shape[1]
    A = Yd @ Yd.T
    lam = rel * float(torch.diagonal(A).sum()) / max(n, 1)      # 与 ridge_closed 逐字相同
    U, S, Vh = torch.linalg.svd(Yd, full_matrices=False)
    if S.numel():
        keep = S > (S.max() * 1e-12)
        U, S, V = U[:, keep], S[keep], Vh.T[:, keep]
    else:
        V = Vh.T
    q = int(S.numel())
    G = Td @ V @ torch.diag(S)                                  # (d_model, q)
    Li = torch.diag(1.0 / torch.sqrt(S ** 2 + lam))             # (Σ²+λI)^{-1/2}
    Wt = G @ Li                                                 # 无约束最优 W̃*
    r_eff = min(int(r), q)
    if r_eff < q:
        Uw, Sw, Vwh = torch.linalg.svd(Wt, full_matrices=False)
        Wt = (Uw[:, :r_eff] * Sw[:r_eff]) @ Vwh[:r_eff]         # trunc_r
    W = Wt @ Li
    M = W @ U.T
    res = float((M @ Yd - Td).norm() / Td.norm().clamp_min(1e-30))
    rk = None
    if check_rank:
        sv = torch.linalg.svdvals(M)
        rk = int((sv > 1e-9 * sv[0]).sum()) if sv.numel() else 0
    return M, lam, res, q, r_eff, rk


def relF(A, B):
    return float((A - B).norm() / B.norm().clamp_min(1e-30))


# ================================================================ Stage A
def _worked():
    """A95-00：known-answer worked example（§3）。λ=0 以便精确断言。"""
    g = torch.Generator().manual_seed(20261028)
    d_v, d_m, n = 3, 2, 4
    Y = torch.randn(d_v, n, generator=g, dtype=torch.float64)
    u1 = torch.tensor([1.0, 0.5], dtype=torch.float64)
    v1 = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64)
    C1 = torch.outer(u1, v1)                                     # rank 1
    T1 = C1 @ Y
    M1, _, res1, q1, re1, rk1 = lowrank_closed(Y, T1, 1, rel=0.0, check_rank=True)
    M1b, _, res1b, *_ = lowrank_closed(Y, T1, q1, rel=0.0, check_rank=False)
    u2 = torch.tensor([0.2, -1.3], dtype=torch.float64)
    v2 = torch.tensor([-1.0, 0.4, 0.6], dtype=torch.float64)
    C2 = C1 + torch.outer(u2, v2)                                # rank 2
    T2 = C2 @ Y
    _, _, res2_r1, _, _, _ = lowrank_closed(Y, T2, 1, rel=0.0, check_rank=False)
    _, _, res2_r2, _, _, _ = lowrank_closed(Y, T2, 2, rel=0.0, check_rank=False)
    RES["worked"] = {"rank1_target_res_r1": res1, "rank1_target_rank_M": rk1,
                     "rank1_target_res_fullrank": res1b,
                     "rank2_target_res_r1": res2_r1, "rank2_target_res_r2": res2_r2}
    ok = (res1 < 1e-10 and rk1 == 1 and res1b < 1e-10
          and res2_r1 > 0.2 and res2_r2 < 1e-10)
    rec("A95-00", ok,
        f"worked: rank1 目标下 r=1 残差={res1:.2e}(须<1e-10)、rank(M)={rk1}(须=1)；"
        f"rank2 目标下 r=1 残差={res2_r1:.3f}(须>0.2)、r=2 残差={res2_r2:.2e}(须<1e-10)",
        RES["worked"])


def _algebra():
    """A95-01/02/03/04/08/11：合成数据上的秩代数与确定性。"""
    g = torch.Generator().manual_seed(7777)
    d_v, d_m, n = 24, 18, 8
    Y = torch.randn(d_v, n, generator=g, dtype=torch.float64)
    T = torch.randn(d_m, n, generator=g, dtype=torch.float64)
    # ★ DEF-95-01：必须喂 float64 —— ridge_closed 内部 .double()，先降成 float32 会把
    #   A95-02 的恒等式退化到 ~1e-8（精度假失败），而不是代数错。
    M_lam, lam_ref, res_ref = ridge_closed(Y, T)
    M_lam, lam_ref = M_lam.double(), float(lam_ref)

    # A95-02：r ≥ rank(Y) ⇒ 与 ridge_closed 恒等（承重）
    gaps, ranks = [], []
    for r in [n, n + 4]:
        M, lam, res, q, re_, rk = lowrank_closed(Y, T, r, check_rank=True)
        gaps.append(relF(M, M_lam)); ranks.append((r, q, re_, rk))
    rec("A95-02", max(gaps) < 1e-10,
        f"r≥rank(Y) 时 rel(M_r, M_λ)={max(gaps):.2e}(须<1e-10) ⇒ 闭式解与 ridge_closed 恒等；"
        f"(r,q,r_eff,rank(M))={ranks}", {"gaps": gaps, "ranks": ranks})

    # A95-01：rank(M_r) == min(r, rank(M_λ))
    ok1, det1 = True, []
    for r in [1, 2, 3, 4, 6, 8]:
        M, lam, res, q, re_, rk = lowrank_closed(Y, T, r, check_rank=True)
        want = min(r, int((torch.linalg.svdvals(M_lam) > 1e-9 * torch.linalg.svdvals(M_lam)[0]).sum()))
        det1.append({"r": r, "rank_num": rk, "want": want, "r_eff": re_})
        ok1 = ok1 and (rk == want)
    rec("A95-01", ok1, f"逐 r 断言 rank(M_r)==min(r,rank(M_λ))：{det1}", {"rows": det1})

    # A95-03：rel_res 与 ‖M_r−M_λ‖_F 随 r 单调不增
    rs, ds = [], []
    for r in [1, 2, 3, 4, 6, 8, 12, 16]:
        M, lam, res, *_ = lowrank_closed(Y, T, r)
        rs.append(res); ds.append(relF(M, M_lam))
    mono_r = all(rs[i + 1] <= rs[i] + 1e-12 for i in range(len(rs) - 1))
    mono_d = all(ds[i + 1] <= ds[i] + 1e-12 for i in range(len(ds) - 1))
    rec("A95-03", mono_r and mono_d,
        f"rel_res 随 r 单调不增={mono_r}；‖M_r−M_λ‖_F 单调不增={mono_d}；"
        f"rel_res={[round(v,6) for v in rs]}",
        {"rel_res": rs, "dist_to_full": ds})

    # A95-04：λ 与 ridge_closed 逐位相同
    lams = [lowrank_closed(Y, T, r)[1] for r in R_GRID]
    rec("A95-04", all(v == lam_ref for v in lams),
        f"λ(每个 r) 与 ridge_closed 的 λ 逐位相同={all(v == lam_ref for v in lams)}；λ={lam_ref:.12e}",
        {"lam_ref": lam_ref, "lams": lams})

    # A95-08：两次拟合逐位相同（先断言 count_mounted==0 —— 由调用方保证）
    Ma, *_ = lowrank_closed(Y, T, 4)
    Mb, *_ = lowrank_closed(Y, T, 4)
    rec("A95-08", bool((Ma == Mb).all()),
        f"同参数两次闭式拟合 M 逐位相同={(Ma == Mb).all()}", None)

    # A95-11：λ=0 且 r ≥ n ⇒ 与最小二乘 lstsq 同解
    M0, _, _, _, re0, _ = lowrank_closed(Y, T, n, rel=0.0)
    lstsq = torch.linalg.lstsq((Y.T), (T.T)).solution.T   # M Y = T  ⇒  Yᵀ Mᵀ = Tᵀ
    rec("A95-11", relF(M0, lstsq) < 1e-8,
        f"λ=0、r≥n 时 rel(M_r, lstsq)={relF(M0, lstsq):.2e}(须<1e-8，纯代数)", None)

    # A95-05：无泄漏 —— 把留出列投毒后重拟合，M 逐位不变；而"错用全部列"会变
    tr_c, he_c = list(range(0, 5)), list(range(5, 8))
    T_all = T.clone()
    T_a = T_all[:, tr_c].contiguous()
    T_all2 = T_all.clone()
    T_all2[:, he_c] = torch.randn(d_m, len(he_c), generator=g, dtype=torch.float64) * 7.0
    T_b = T_all2[:, tr_c].contiguous()
    M_leak_a, *_ = lowrank_closed(Y[:, tr_c].contiguous(), T_a, 3)
    M_leak_b, *_ = lowrank_closed(Y[:, tr_c].contiguous(), T_b, 3)
    M_all, *_ = lowrank_closed(Y, T_all2, 3)
    same = bool((M_leak_a == M_leak_b).all())
    differs = relF(M_all, M_leak_a) > 1e-3
    rec("A95-05", same and differs,
        f"投毒留出列后重拟合 M 逐位不变={same}(须True)；而把全部 8 列都放进拟合会变 "
        f"(rel={relF(M_all, M_leak_a):.2e}，须>1e-3) ⇒ 守卫有牙齿",
        {"same": same, "leak_rel": relF(M_all, M_leak_a)})


def _row_injection(model, tok, facts, tr):
    """A95-06（CORR-15）：读出位注入**逐行**断言，同时覆盖 B=1 与 B=4。"""
    st = make_stack(gamma=0.0, no_write=True, seed=SEEDS[0])
    try:
        p = facts[0]["prompt_ids"]
        cands = [facts[0]["prompt_ids"][-1]] * 1
        # --- B=1：无 override ⇒ 与原生逐位 0.0
        seq1 = torch.tensor([list(p) + cands], dtype=torch.long, device=DEV)
        LayerMem.read = W28._ORIG_READ
        lg_native = model(input_ids=seq1).logits.float().clone()
        LayerMem.read = W28._read_r28
        set_m(st, None)
        lg_patched = model(input_ids=seq1).logits.float().clone()
        d_native = float((lg_patched - lg_native).abs().max())
        # --- override 逐行检查（B=1 与 B=4）
        d_li = list(st.mems)[0]
        mo = (torch.arange(st.mems[d_li].d_model, device=DEV, dtype=torch.float32) % 7 - 3) * 0.25
        res_rows = {}
        for Bn in (1, 4):
            cs = [facts[0]["prompt_ids"][-1]] * Bn
            seq = torch.tensor([list(p) + [c] for c in cs], dtype=torch.long, device=DEV)
            set_m(st, {li: mo for li in st.mems})
            model(input_ids=seq)
            inj = st.mems[d_li].injection.float()
            L = seq.shape[1]
            rows = [float((inj[b, L - 2] - mo).abs().max()) for b in range(Bn)]
            spread = float((inj[:, L - 2] - inj[0, L - 2]).abs().max())
            # 非读出位不得被 override 泄漏
            other = float((inj[:, L - 4] - mo).abs().max())
            res_rows[f"B{Bn}"] = {"row_dev": rows, "spread": spread, "other_dev": other}
        set_m(st, None)
        okp = (d_native == 0.0
               and all(v < 1e-6 for v in res_rows["B1"]["row_dev"])
               and res_rows["B1"]["spread"] == 0.0
               and all(v < 1e-6 for v in res_rows["B4"]["row_dev"])
               and res_rows["B4"]["spread"] == 0.0
               and all(res_rows[k]["other_dev"] > 1e-3 for k in res_rows))
        rec("A95-06", okp,
            f"无 override 与原生逐位差={d_native:.1e}(须=0)；B=1 逐行偏差={res_rows['B1']['row_dev']}、"
            f"B=4 逐行偏差={[round(v,2) for v in res_rows['B4']['row_dev']]}、"
            f"B=4 行间 spread={res_rows['B4']['spread']:.1e}(须=0)；"
            f"非读出位未被泄漏(other_dev>1e-3)={[round(res_rows[k]['other_dev'],3) for k in res_rows]}",
            res_rows)
    finally:
        set_m(st, None)
        LayerMem.read = W28._read_r28
        st.remove()


def _mount_count():
    """A95-07（CORR-15/DEF-94-07）：count_mounted 0→len(MOUNT)→0。"""
    n0 = count_mounted(MODEL_REF[0], MOUNT)
    st = make_stack(gamma=0.0, no_write=True, seed=SEEDS[0])
    n1 = count_mounted(MODEL_REF[0], MOUNT)
    st.remove()
    n2 = count_mounted(MODEL_REF[0], MOUNT)
    rec("A95-07", n0 == 0 and n1 == len(MOUNT) and n2 == 0,
        f"count_mounted: 建前={n0}(须0)、挂上={n1}(须{len(MOUNT)})、卸后={n2}(须0) ⇒ 不得有两个 stack 并存",
        {"before": n0, "mounted": n1, "after": n2})


def _real_algebra(tok, model, facts, chunks, owner, tr):
    """A95-10：真实数据代数（T:=Y 自映射）。rank(M_λ) ≤ n_fit 且 r=n_fit 时与 ridge_closed 恒等。"""
    st, wc, Wsnap = W28.build_state(tok, model, facts, chunks, owner, SEEDS[0])
    try:
        set_gamma(st, 1.0)
        Y = capture_y(model, st, facts, facts[0]["prompt_ids"][-1])
        rows, ok = [], True
        for li in st.mems:
            Yl = torch.stack([Y[li][i] for i in tr])            # (16, d)
            M_lam, lam, res = ridge_closed(Yl.T, Yl.T)
            M_lam = M_lam.double()
            sv = torch.linalg.svdvals(M_lam)
            rk_lam = int((sv > 1e-9 * sv[0]).sum())
            M_r, _, _, q, re_, rk_r = lowrank_closed(Yl.T, Yl.T, len(tr), check_rank=True)
            g = relF(M_r, M_lam)
            rows.append({"layer": li, "rank(M_lam)": rk_lam, "q": q, "rank(M_r)": rk_r,
                         "rel_gap": g})
            ok = ok and (rk_lam <= len(tr)) and (g < 1e-10)
        brf = ", ".join("L%d:rank=%d,gap=%.1e" % (x["layer"], x["rank(M_lam)"], x["rel_gap"])
                        for x in rows)
        rec("A95-10", ok,
            f"真实 Y 上 rank(M_λ) ≤ n_fit({len(tr)}) 且 r=n_fit 时 rel(M_r,M_λ)<1e-10 "
            f"[{brf}]", {"rows": rows, "write_calls": wc})
    finally:
        st.remove()


MODEL_REF = [None]


def stage_a(tok, model, facts, chunks, owner, tr, he):
    MODEL_REF[0] = model
    RES["config"] = {"mount": list(MOUNT), "chunk": CHUNK, "n_chunk": N_CHUNKS,
                     "n_facts": N_FACTS, "n_train_keys": N_TRAIN_KEYS,
                     "r_grid": R_GRID, "steps_free": STEPS_FREE, "lr_free": LR_FREE}
    rec("A95-00a", True,
        f"MOUNT={list(MOUNT)} CHUNK={CHUNK} B={N_CHUNKS} n_q={N_TRAIN_KEYS} "
        f"r_grid={R_GRID} chance={CHANCE:.3f}", RES["config"])
    _worked()
    _algebra()
    _mount_count()
    _row_injection(model, tok, facts, tr)
    _real_algebra(tok, model, facts, chunks, owner, tr)
    # A95-09：口径（与 D-105/108/109 同口径）
    items = make_eval_items(tok, facts, tr, __import__("random").Random(555))
    n_q = len({it["key_idx"] for it in items})
    has_correct = all(sum(1 for c in it["cand"] if c[1]) == 1 for it in items)
    n_cand_ok = all(len(it["cand"]) == N_CAND for it in items)
    # 逐题 draws 数必须都等于 N_DRAWS（acc_by_question 是"同题全对才算命中"）
    from collections import Counter
    per_q = Counter(it["key_idx"] for it in items)
    draws_ok = all(v == N_DRAWS for v in per_q.values())
    # 干扰项互不相同且都 ≠ 正确值（口径正确性）
    dist_ok = True
    for it in items:
        vals = [c[0] for c in it["cand"]]
        if len({tuple(v) for v in vals}) != N_CAND:
            dist_ok = False
        correct_ids = [tuple(c[0]) for c in it["cand"] if c[1]][0]
        if any(tuple(c[0]) == correct_ids for c in it["cand"] if not c[1]):
            dist_ok = False
    rec("A95-09", n_q == N_TRAIN_KEYS and len(items) == N_TRAIN_KEYS * N_DRAWS
        and has_correct and n_cand_ok and draws_ok and dist_ok
        and abs(CHANCE - 0.25) < 1e-12,
        f"n_q={n_q}(须{N_TRAIN_KEYS}) n_item={len(items)}(须{N_TRAIN_KEYS*N_DRAWS}) "
        f"n_cand={N_CAND} 每题恰一正确项={has_correct} 每题draws={N_DRAWS}({draws_ok}) "
        f"候选互异且干扰≠正确={dist_ok} chance={CHANCE:.3f}(须0.25)",
        {"n_q": n_q, "n_item": len(items), "draws_ok": draws_ok, "dist_ok": dist_ok})
    return FAILS == []


# ================================================================ 正式轮
def run_seed(seed, tok, model, facts, chunks, owner, tr, he):
    t0 = time.time()
    st, wc, Wsnap = W28.build_state(tok, model, facts, chunks, owner, seed)
    out = []
    try:
        set_gamma(st, 1.0)                       # 与 D-109 逐字相同的时序 ⇒ ridge16 须复现
        Y = capture_y(model, st, facts, facts[0]["prompt_ids"][-1])
        tgt32, floss = W28.optimise_free(model, st, tok, facts, ALL32,
                                        __import__("random").Random(555))
        RES["arms"].append({"arm": "free_losses", "seed": seed,
                            "info": {str(k): v for k, v in floss.items()}})
        # ---- 参照臂：zero / free / ridge16（跨轮复现 D-109）
        for nm, Mv in (("zero", None), ("free", tgt32)):
            fit = eval_inject(model, st, tok, facts, tr, Mv, __import__("random").Random(555))
            gen = eval_inject(model, st, tok, facts, he, Mv, __import__("random").Random(555))
            out.append({"arm": nm, "seed": seed, "acc_train": fit["acc_q"],
                        "acc_held": gen["acc_q"], "margin_train": fit["margin_mean"],
                        "margin_held": gen["margin_mean"], "n_q": gen["n_q"],
                        "info": {}, "sec": 0})
            print(f"  [{nm}] acc_train={out[-1]['acc_train']:.3f} "
                  f"acc_held={out[-1]['acc_held']:.3f} margin(tr/ho)="
                  f"{out[-1]['margin_train']:+.2f}/{out[-1]['margin_held']:+.2f}", flush=True)
        # ---- ridge16（跨轮参照，必须逐位复现 D-109 的 1.000 / 0.062 | 0.188）
        Ytr = {li: torch.stack([Y[li][i] for i in tr]) for li in st.mems}
        Ttr = {li: torch.stack([tgt32[li][j] for j in range(len(tr))]) for li in st.mems}
        M_ref, lams, ress = {}, {}, {}
        for li in st.mems:
            M, lam, res = ridge_closed(Ytr[li].T, Ttr[li].T)
            M_ref[li], lams[str(li)], ress[str(li)] = M.float(), lam, res
        Mv = {li: (M_ref[li] @ Y[li].T).T for li in st.mems}
        fit = eval_inject(model, st, tok, facts, tr, Mv, __import__("random").Random(555))
        gen = eval_inject(model, st, tok, facts, he, Mv, __import__("random").Random(555))
        out.append({"arm": "ridge16", "seed": seed, "acc_train": fit["acc_q"],
                    "acc_held": gen["acc_q"], "margin_train": fit["margin_mean"],
                    "margin_held": gen["margin_mean"], "n_q": gen["n_q"],
                    "info": {"lambda": lams, "rel_res": ress, "n_fit": len(tr)}, "sec": 0})
        print(f"  [ridge16] acc_train={fit['acc_q']:.3f} acc_held={gen['acc_q']:.3f} "
              f"margin(tr/ho)={fit['margin_mean']:+.2f}/{gen['margin_mean']:+.2f} "
              f"n_fit={len(tr)}", flush=True)
        # ---- 主判据：低秩臂
        The = {li: torch.stack([tgt32[li][j] for j in he]) for li in st.mems}
        base_held_res = None
        for r in R_GRID:
            Ms, res_tr, res_he, info = {}, {}, {}, {}
            for li in st.mems:
                M, lam, res, q, re_, _ = lowrank_closed(Ytr[li].T, Ttr[li].T, r)
                Ms[li] = M.float()
                res_tr[str(li)] = res
                T_he = The[li].double()
                # ★ DEF-95-03：诊断用 double 副本；Ms[li] 必须仍是 float32（正式路径与 ridge16 同 dtype）
                res_he[str(li)] = float((M.double() @ Y[li][he].T.double() - T_he.T).norm()
                                        / T_he.norm().clamp_min(1e-30))
                info[str(li)] = {"q": q, "r_eff": re_, "lam": lam,
                                 "rank_M": min(int(r), q)}
            Mv = {li: (Ms[li] @ Y[li].T).T for li in st.mems}
            f2 = eval_inject(model, st, tok, facts, tr, Mv, __import__("random").Random(555))
            g2 = eval_inject(model, st, tok, facts, he, Mv, __import__("random").Random(555))
            arm = {"arm": f"lr{r}", "seed": seed, "acc_train": f2["acc_q"],
                   "acc_held": g2["acc_q"], "margin_train": f2["margin_mean"],
                   "margin_held": g2["margin_mean"], "n_q": g2["n_q"],
                   "info": {"per_layer": info, "rel_res_train": res_tr,
                            "rel_res_held": res_he,
                            "n_param_equiv": {}}, "sec": 0}
            arm["info"]["n_param_equiv"] = {str(li): 2 * int(Ytr[li].shape[1]) * int(r)
                                            for li in st.mems}
            out.append(arm)
            print(f"  [lr{r:<2}] acc_train={arm['acc_train']:.3f} "
                  f"acc_held={arm['acc_held']:.3f} margin(tr/ho)="
                  f"{arm['margin_train']:+.2f}/{arm['margin_held']:+.2f} "
                  f"rel_res_he={[round(v,4) for v in res_he.values()]}", flush=True)
        # ---- adam16 参照（D-109 已定 0.875/0.000、0.812/0.000）
        st.remove()
        st = None
        out.append(W28.run_adam("adam16", tok, model, facts, chunks, owner, tr, he, seed))
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
    for s in SEEDS:
        d = {a["arm"]: a for a in RES["arms"] if a.get("seed") == s and "acc_held" in a}
        if not all(k in d for k in ("ridge16", "free")) or not any(
                k.startswith("lr") for k in d):
            continue
        # 1) INSTRUMENT_INVALID 抢先
        r16 = d.get(f"lr{R_CONTROL}")
        inval = None
        if r16 is None:
            inval = "lr16 缺失"
        else:
            if abs(r16["acc_train"] - d["ridge16"]["acc_train"]) > G_DELTA:
                inval = f"lr16 acc_train {r16['acc_train']:.3f} vs ridge16 {d['ridge16']['acc_train']:.3f}"
            if abs(r16["acc_held"] - d["ridge16"]["acc_held"]) > G_DELTA:
                inval = (f"lr16 acc_held {r16['acc_held']:.3f} vs "
                         f"ridge16 {d['ridge16']['acc_held']:.3f}")
        # 2) 拟合窗口与证人
        fit_rs = [r for r in R_WITNESS if d.get(f"lr{r}")
                  and d[f"lr{r}"]["acc_train"] >= G_GEN]
        win_rs = [r for r in R_WITNESS if d.get(f"lr{r}")
                  and d[f"lr{r}"]["acc_train"] >= G_GEN
                  and d[f"lr{r}"]["acc_held"] >= G_GEN]
        held_of_fit = {r: d[f"lr{r}"]["acc_held"] for r in fit_rs}
        br12 = d.get("lr12")
        if inval:
            b = "INSTRUMENT_INVALID"
        elif not fit_rs:
            b = "RANK_TOO_SMALL"
        elif win_rs:
            b = "PARAM_LIMIT"
        elif all(v <= G_LOW for v in held_of_fit.values()):
            b = "SAMPLE_LIMIT"
        else:
            b = "PARTIAL"
        per[str(s)] = {"ridge16_tr": d["ridge16"]["acc_train"],
                       "ridge16_he": d["ridge16"]["acc_held"],
                       "free_he": d["free"]["acc_held"],
                       "lr16_he": r16["acc_held"] if r16 else None,
                       "fit_window_r": fit_rs, "held_of_fit": held_of_fit,
                       "lr12_he": br12["acc_held"] if br12 else None,
                       "branch": b, "inval": inval}
    bs = {v["branch"] for v in per.values()}
    branch = bs.pop() if len(bs) == 1 else "INCONCLUSIVE"
    bridge = (branch == "RANK_TOO_SMALL"
              and all((v["lr12_he"] is not None and v["lr12_he"] >= G_GEN) for v in per.values()))
    return {"per_seed": per, "branch": branch, "bridge_only": bool(bridge)}


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
        print("===================== Stage A (rev 2.9) =====================", flush=True)
        ok = stage_a(tok, model, facts, chunks, owner, tr, he)
        RES["stage_a_ok"] = ok
        dump(RES, "probe91_lowrank_guards")
        print("\n* Stage A " + ("全部通过" if ok else "存在 FAIL —— 正式轮被阻断"), flush=True)
        sys.exit(0 if ok else 2)
    seeds = SEEDS[:1] if a.smoke else SEEDS
    for seed in seeds:
        print(f"===== seed {seed} =====", flush=True)
        RES["arms"].extend(run_seed(seed, tok, model, facts, chunks, owner, tr, he))
    RES["branch"] = decide()
    dump(RES, f"probe91_lowrank{'_smoke' if a.smoke else ''}")
    for s, v in RES["branch"]["per_seed"].items():
        print(f"  seed {s}: ridge16={v['ridge16_he']:.3f} lr16={v['lr16_he']} "
              f"fit_window={v['fit_window_r']} held_of_fit={v['held_of_fit']} "
              f"-> {v['branch']}", flush=True)
    print(f"\n* 分支 = {RES['branch']['branch']}  "
          f"(bridge_only={RES['branch']['bridge_only']})", flush=True)


if __name__ == "__main__":
    main()
