#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-91 rev 3.5 · **写侧 `k` 池化（line 86）是不是残余瓶颈？**

卡片：revolve/DIR-91/DIRECTION_rev91_v35.md（判据冻结；本文件不得改阈值）
派生自 D-106（probe91_valread.py，rev 2.7）—— **逐字复用其 `v_rule` / `value_positions`**，
唯一新增的独立变量 = `sgmn_module.py:86` 那一行 `k` 的取值方式（D-106 从未动过它）。

五臂（唯一独立变量 = 池化规则**对** `(k @ line86, v @ line87)`；其余全部原样）：
  K_mean_V_mean  k=pk_raw.mean(0)*s     v=zv.mean(0)          无 oracle（原生，须复现 D-106 A_mean）
  K_mean_V_vt    k=pk_raw.mean(0)*s     v=zv[idx].mean(0)      v 位（须复现 D-106 D_vtpos）
  K_vt_V_mean    k=pk_raw[idx].mean(0)*s v=zv.mean(0)         k 位   ★全新
  K_vt_V_vt      k=pk_raw[idx].mean(0)*s v=zv[idx].mean(0)    k+v 位 ★全新
  K_last_V_mean  k=pk_raw[-1]*s          v=zv.mean(0)         无 oracle ★全新（唯一可实现）

★ CORR-19 规则 ③：K_mean 必须保持原式 `pk_raw.mean(dim=0) * s`（先均后乘），
  不得写成 `(pf@s).mean`；K_vt/K_last 是另一算序 ⇒ 只与自身直接复算比，绝不与 K_mean 比。
★ CORR-14：Stage A 含「臂 × 独立变量取值」表（5 臂 × {k,v} = 10 格）并逐格断言，
  且走正式轮那条代码路径（同一个 `_write_patched`）。
★ DEF-93-04：正式轮必须在写阶段前后**显式安装/恢复** patch；A98-05/06 断言各臂 k 互不相等，
  防止「patch 未生效却五臂读数逐位相同」。
"""
import argparse, json, math, os, sys, time
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)

import numpy as np
import torch

import sgmn_module
from sgmn_module import LayerMem, sg
import probe91_womap as W91
import probe91_valread as V          # ★ v_rule / value_positions 逐字复用（不重写）
import probe91_woform as WF         # ★ count_mounted 取 woform 版（收**层下标**；V 版收层对象）
from R2_task91 import (ALPHA, B as N_CHUNKS, CHUNK, DEV, ETA_WRITE, LAM, MOUNT, N_CAND,
                       N_DRAWS, N_FACTS, N_TRAIN_KEYS, OUT, READ_LR, READ_STEPS,
                       acc_by_question, build_chunks, build_facts, make_eval_items,
                       make_stack, model_and_tok, nll_of_candidate, read_params,
                       score_and_accuracy, train_read_task, write_params)

CLS_N = 27                                  # 值池大小（D-114 口径）

RES = {"card": "DIR-91 rev 3.5", "file": "probe91_kpool.py",
       "alpha": ALPHA, "lam": LAM, "eta": ETA_WRITE, "read_steps": READ_STEPS,
       "read_lr": READ_LR, "n_chunk": N_CHUNKS, "n_facts": N_FACTS,
       "n_train_keys": N_TRAIN_KEYS, "n_cand": N_CAND, "n_draws": N_DRAWS,
       "guards": {}, "stage_a": [], "arms": [], "worked": {}}
FAILS = []

G_GEN, G_LOW, G_DELTA = 0.50, 0.25, 0.125
G_PARTIAL = G_LOW + G_DELTA
G_ABSENT, G_AVAIL = 0.15, 0.28125
COS_SEP = 0.90
COS_NATIVE = 0.99

ARMS = ["K_mean_V_mean", "K_mean_V_vt", "K_vt_V_mean", "K_vt_V_vt", "K_last_V_mean"]
NEW_ARMS = ["K_vt_V_mean", "K_vt_V_vt", "K_last_V_mean"]
ORACLE_ARMS = {"K_mean_V_vt", "K_vt_V_mean", "K_vt_V_vt"}   # 任何用到 vt 位的臂

# 每条臂的 (k 规则, v 规则)
RULE = {
    "K_mean_V_mean": ("mean", "mean"),
    "K_mean_V_vt":   ("mean", "vt"),
    "K_vt_V_mean":   ("vt",   "mean"),
    "K_vt_V_vt":     ("vt",   "vt"),
    "K_last_V_mean": ("last", "mean"),
}

ARM = {"name": "K_mean_V_mean"}
VPOS = {"V": None}
REC = {}                       # patch 记录当次实际使用的 (k, v, pk_raw, pf, zv, idx)


def rec(name, ok, detail, value=None):
    RES["guards"][name] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def dump(obj, name):
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  ⇒ {p}", flush=True)


# ==================================================================== k 规则（新增）
def k_rule(mode, pk_raw, kp_scale, idx):
    """把 pk_raw=(N,d_k) 收成一个 (d_k,)。唯一新增独立变量。

    ★ CORR-19 规则 ③：mean 分支必须与原生逐位一致（先 mean 后乘）。
    """
    if mode == "mean":
        return pk_raw.mean(dim=0) * kp_scale                     # ★ 原生算序，逐位相同
    if mode == "vt":
        assert idx is not None and len(idx) > 0, "k=vt 需要值 token 位"
        j = torch.as_tensor(idx, device=pk_raw.device, dtype=torch.long)
        return pk_raw.index_select(0, j).mean(dim=0) * kp_scale
    if mode == "last":
        return pk_raw[-1] * kp_scale
    raise ValueError(mode)


def v_rule_mode(mode, zv, kp, k, idx):
    """v 侧：mean 用原生算序；vt 逐字复用 V.v_rule 的 D_vtpos 分支。"""
    if mode == "mean":
        return zv.mean(dim=0)                                    # ★ 原生算序，逐位相同
    if mode == "vt":
        return V.v_rule("D_vtpos", zv, kp, k, idx)               # ★ 逐字复用，不重写
    raise ValueError(mode)


# ==================================================================== 写侧 patch
_ORIG_WRITE = LayerMem.write


def _write_patched(self, hs_in, drop_sg=False):
    """与 sgmn_module.py:79-104 逐字相同，唯一差别 = line 86 的 `k` 与 line 87 的 `v`。"""
    if self.prev_hs is None or self.no_write:
        return
    self.write_calls += 1
    pf = self.prev_hs.reshape(-1, self.prev_hs.shape[-1])
    hf = hs_in.reshape(-1, hs_in.shape[-1])
    pk_raw = sg(pf @ self.Pk, drop_sg)
    kmode, vmode = RULE[ARM["name"]]
    k = k_rule(kmode, pk_raw, self.pk_scale, VPOS["V"])
    kp = pk_raw * self.pk_scale                                  # 仅供 v_rule 的 attn 分支（本卡不用）
    zv = sg(hf @ self.Pv, drop_sg)
    v = v_rule_mode(vmode, zv, kp, k, VPOS["V"])
    # ★ 供 A98-00/01 的逐格断言：记录当次实际输入与输出（同算序复算用）
    REC.clear()
    REC.update({"arm": ARM["name"], "k": k.detach().clone(), "v": v.detach().clone(),
                "pk_raw": pk_raw.detach().clone(), "pf": pf.detach().clone(),
                "zv": zv.detach().clone(),
                "idx": list(VPOS["V"]) if VPOS["V"] else None,
                "scale": float(self.pk_scale)})
    W = (self.W0 + self.U @ self.V.T) if self.use_factorized else self.Wfull
    G = self.alpha * torch.outer(W @ k - v, k) + self.lam * (W - self.W0)
    with torch.no_grad():
        if self.use_factorized:
            self.U.copy_(self.U - self.eta * (G @ self.V))
            self.V.copy_(self.V - self.eta * (G.T @ self.U))
        else:
            self.Wfull.copy_(self.Wfull - self.eta * G)
    self.steps += 1


def install_write():
    LayerMem.write = _write_patched


def restore_write():
    LayerMem.write = _ORIG_WRITE


def write_phase_arm(model, st, chunks, owner, vpos_by_fact):
    """= R2_task91.write_phase 逐字；差别 = 每块前设 VPOS（供 vt 臂取值位）。"""
    kmode, vmode = RULE[ARM["name"]]
    need_vt = ("vt" in (kmode, vmode))
    for ci, ids in enumerate(chunks):
        VPOS["V"] = vpos_by_fact.get(owner[ci]) if need_vt else None
        model(input_ids=ids.to(DEV))
    VPOS["V"] = None
    return {li: m.write_calls for li, m in st.mems.items()}


def key_matrices_arm(model, st, facts, chunks, owner, vpos_by_fact):
    """= R2_task91.key_matrices，但 K/V 用**本臂的规则**（与写规则同源）。"""
    saved = {li: m.no_write for li, m in st.mems.items()}
    for m in st.mems.values():
        m.no_write = True
    K = {li: [] for li in st.mems}
    Vv = {li: [] for li in st.mems}
    try:
        for fi in range(len(facts)):
            ci = owner.index(fi) + 1
            VPOS["V"] = vpos_by_fact.get(fi) if ("vt" in RULE[ARM["name"]]) else None
            model(input_ids=chunks[ci].to(DEV))
            for li, m in st.mems.items():
                hs = st._hs_in[li]
                pf = hs.reshape(-1, hs.shape[-1])
                pk_raw = sg(pf @ m.Pk)
                kmode, vmode = RULE[ARM["name"]]
                K[li].append(k_rule(kmode, pk_raw, m.pk_scale, VPOS["V"]).detach())
                zv = sg(pf @ m.Pv)
                Vv[li].append(v_rule_mode(vmode, zv, pk_raw * m.pk_scale, None, VPOS["V"]).detach())
    finally:
        VPOS["V"] = None
        for li, m in st.mems.items():
            m.no_write = saved[li]
    return ({li: torch.stack(K[li]).float() for li in K},
            {li: torch.stack(Vv[li]).float() for li in Vv})


# ==================================================================== Stage A
def stage_a(tok, model, facts, chunks, owner, vpos):
    print("\n" + "=" * 21 + " Stage A (rev 3.5) " + "=" * 21, flush=True)
    vp_by_fact = {i: vpos[i]["V"] for i in range(len(facts))}

    # ---- A98-00 口径
    ok00 = (len(facts) == 32 and CHUNK == 64 and NCHK_eq() and MOUNT == [8, 12, 16]
            and N_CAND == 4 and CLS_N == 27 and len(set(f["val"] for f in facts)) == 27)
    rec("A98-00", ok00,
        f"n_facts={len(facts)} n_cls={CLS_N} N_CAND={N_CAND} chance={1/N_CAND:.4f} "
        f"B={N_CHUNKS} CHUNK={CHUNK} MOUNT={MOUNT} G_GEN={G_GEN} G_LOW={G_LOW} G_DELTA={G_DELTA}")

    # ---- A98-04 值 token 位定位（复现 D-106 日志原行）
    sizes = sorted({len(v["V"]) for v in vpos})
    pers = sorted({v["period"] for v in vpos})
    rec("A98-04", sizes == [2] and 31 <= pers[0] and pers[-1] <= 36,
        f"|V|∈[{sizes[0]},{sizes[-1]}]、period∈[{pers[0]},{pers[-1]}]"
        f"（D-106 日志原行：|V|∈[2,2]、period∈[31,36]）")

    # ---- A98-01/02/03/05/06 臂 × 独立变量表（10 格逐格断言，同算序复算）
    st = make_stack(gamma=0.0, no_write=False, seed=20261028)
    n_before = WF.count_mounted(model, MOUNT)     # ★ 收层下标（D-114 同源）
    rec("A98-07a", n_before == 3,
        f"建栈后 count_mounted(model,MOUNT)={n_before}（须 ==3）")
    grid, kvecs, vvecs = {}, {}, {}
    install_write()
    try:
        # 用第 0 号事实的第一块跑一次 write，取 REC
        for arm in ARMS:
            ARM["name"] = arm
            kmode, vmode = RULE[arm]
            VPOS["V"] = vpos[0]["V"] if "vt" in (kmode, vmode) else None
            # 需要 prev_hs：先一次 no_write 的 forward 建立 prev_hs
            saved = {li: m.no_write for li, m in st.mems.items()}
            for m in st.mems.values():
                m.no_write = True
            try:
                model(input_ids=chunks[0].to(DEV))
            finally:
                for li, m in st.mems.items():
                    m.no_write = saved[li]
            li = MOUNT[0]
            st.mems[li].write(st._hs_in[li], drop_sg=False)
            r = dict(REC)
            # 同算序复算
            k_re = k_rule(kmode, r["pk_raw"], r["scale"], r["idx"])
            v_re = v_rule_mode(vmode, r["zv"], r["pk_raw"] * r["scale"], r["k"], r["idx"])
            dk = float((k_re - r["k"]).abs().max())
            dv = float((v_re - r["v"]).abs().max())
            grid[arm] = {"d_k": dk, "d_v": dv, "k_norm": float(r["k"].norm()),
                         "v_norm": float(r["v"].norm())}
            kvecs[arm] = r["k"].clone()
            vvecs[arm] = r["v"].clone()
            rec(f"A98-01_{arm}", dk == 0.0 and dv == 0.0,
                f"同算序复算逐位相同：k max|Δ|={dk:.1e}、v max|Δ|={dv:.1e}（须 ==0）；"
                f"‖k‖={float(r['k'].norm()):.2f} ‖v‖={float(r['v'].norm()):.2f}", value=dk)
        ARM["name"] = "K_mean_V_mean"

        # A98-02：K_mean 与**未 patch 的原生 write** 逐位相同
        li = MOUNT[0]
        for m in st.mems.values():
            m.no_write = True
        try:
            model(input_ids=chunks[0].to(DEV))
        finally:
            for m in st.mems.values():
                m.no_write = False
        st.mems[li].write(st._hs_in[li], drop_sg=False)
        k_patched_mean = REC["k"].clone()
        pf = REC["pf"].clone(); sc = REC["scale"]
        k_native = (sg(pf @ st.mems[li].Pk).mean(dim=0)) * sc        # 原生式，逐字照抄 line 86
        d02 = float((k_patched_mean - k_native).abs().max())
        rec("A98-02", d02 == 0.0,
            f"K_mean 与原生 line86 式 `(sg(pf@Pk).mean(0))*s` 逐位相同：max|Δ|={d02:.1e}（须 ==0）",
            value=d02)
        # A98-03：v 侧复用 V.v_rule 的 D_vtpos 分支
        ARM["name"] = "K_mean_V_vt"
        VPOS["V"] = vpos[0]["V"]
        for m in st.mems.values():
            m.no_write = True
        try:
            model(input_ids=chunks[0].to(DEV))
        finally:
            for m in st.mems.values():
                m.no_write = False
        st.mems[li].write(st._hs_in[li], drop_sg=False)
        v_mine = REC["v"].clone()
        v_ref = V.v_rule("D_vtpos", REC["zv"], REC["pk_raw"] * REC["scale"], REC["k"], REC["idx"])
        d03 = float((v_mine - v_ref).abs().max())
        rec("A98-03", d03 == 0.0,
            f"v 侧逐字复用 V.v_rule(D_vtpos) 逐位相同：max|Δ|={d03:.1e}（须 ==0）", value=d03)
        ARM["name"] = "K_mean_V_mean"
        VPOS["V"] = None

        # A98-05/06：非退化 —— 3 个不同 k 规则必须给出 3 个不同向量
        kk = {"mean": kvecs["K_mean_V_mean"], "vt": kvecs["K_vt_V_mean"],
              "last": kvecs["K_last_V_mean"]}
        d_mv = float((kk["mean"] - kk["vt"]).abs().max())
        d_ml = float((kk["mean"] - kk["last"]).abs().max())
        d_vl = float((kk["vt"] - kk["last"]).abs().max())
        rec("A98-05", d_mv > 0 and d_ml > 0 and d_vl > 0,
            f"三 k 规则两两不同：|mean−vt|={d_mv:.3e} |mean−last|={d_ml:.3e} "
            f"|vt−last|={d_vl:.3e}（须全 >0；防 DEF-93-04 空转）",
            value=min(d_mv, d_ml, d_vl))
        vv = {"mean": vvecs["K_mean_V_mean"], "vt": vvecs["K_mean_V_vt"]}
        d_v = float((vv["mean"] - vv["vt"]).abs().max())
        rec("A98-06", d_v > 0,
            f"两 v 规则不同：|mean−vt|={d_v:.3e}（须 >0）", value=d_v)
    finally:
        restore_write()
        st.remove()
        ARM["name"] = "K_mean_V_mean"
        VPOS["V"] = None

    # ---- A98-08/09/10：确定性 + dtype/有限 + write_calls 跨臂一致（5 臂各一次）
    def cap2(arm, seed):
        ARM["name"] = arm
        sty = make_stack(gamma=0.0, no_write=False, seed=seed)
        install_write()
        try:
            wc = write_phase_arm(model, sty, chunks, owner, vp_by_fact)
            Ka, Va = key_matrices_arm(model, sty, facts, chunks, owner, vp_by_fact)
            wc2 = {li: m.write_calls for li, m in sty.mems.items()}
            Kb, Vb = key_matrices_arm(model, sty, facts, chunks, owner, vp_by_fact)
            return wc, wc2, Ka, Va, Kb, Vb
        finally:
            restore_write()
            sty.remove()
            ARM["name"] = "K_mean_V_mean"
            VPOS["V"] = None

    wcs = {}
    for arm in ARMS:
        wc, wc2, Ka, Va, Kb, Vb = cap2(arm, 7777)
        wcs[arm] = wc
        d_det = max(float((Ka[li] - Kb[li]).abs().max()) for li in Ka)
        d_detv = max(float((Va[li] - Vb[li]).abs().max()) for li in Va)
        rec(f"A98-08_{arm}", d_det == 0.0 and d_detv == 0.0,
            f"两次 capture 逐位相同：K max|Δ|={d_det:.1e}、V max|Δ|={d_detv:.1e}（须 ==0）", value=d_det)
        dts = {str(torch.float32)} == {str(x.dtype) for x in list(Ka.values()) + list(Va.values())}
        fin = all(torch.isfinite(x).all() for x in list(Ka.values()) + list(Va.values()))
        wc_ok = all(wc2[li] == wc[li] for li in wc)      # capture 阶段不得新增写入
        rec(f"A98-09_{arm}", dts and fin and wc_ok,
            f"dtype 全 float32={dts}；有限={fin}；capture 阶段 write_calls 无新增={wc_ok}")
        RES["stage_a"].append({"arm": arm, "write_calls": {str(k): v for k, v in wc.items()}})

    # A98-10：池化规则**不得改变写入步数** ⇒ 五臂 write_calls 必须逐层相同
    flat = {arm: {str(k): int(v) for k, v in d.items()} for arm, d in wcs.items()}
    same = all(flat[a] == flat[ARMS[0]] for a in ARMS)
    nw = sorted(set(flat[ARMS[0]].values()))
    rec("A98-10", same,
        f"五臂 write_calls 逐层相同={same}；值={flat[ARMS[0]]}"
        f"（池化规则不得改变写入步数；首块 prev_hs=None 故非 {N_CHUNKS}）",
        value=float(sum(nw)))

    RES["worked"]["grid"] = grid
    return len(FAILS) == 0


def NCHK_eq():
    return N_CHUNKS == 600


# ==================================================================== 一臂（正式轮）
def run_arm(arm, tok, model, facts, chunks, owner, vp_by_fact, tr, he, seed, cls, ncls):
    ARM["name"] = arm
    st = make_stack(gamma=0.0, no_write=False, seed=seed)
    t0 = time.time()
    install_write()                       # ★ DEF-93-04：正式轮必须安装写侧 patch
    try:
        wc = write_phase_arm(model, st, chunks, owner, vp_by_fact)
        K, Vv = key_matrices_arm(model, st, facts, chunks, owner, vp_by_fact)
        wc_after_cap = {li: m.write_calls for li, m in st.mems.items()}
        rng = __import__("random").Random(1000 + seed % 100)
        losses, viol, div = train_read_task(model, st, tok, facts, tr, rng)
        fit = W91.eval_keys(model, st, tok, facts, tr, None, False, None)
        gen = W91.eval_keys(model, st, tok, facts, he, None, False, None)
        model(input_ids=chunks[0].to(DEV))
        g = float(st.mems[list(st.mems)[0]].gamma)

        # M2loo（值身份，与 D-106/D-114 同口径）
        m2 = {}
        for li in st.mems:
            Vl = Vv[li].float()
            s = W91.ridge_scores(Vl[tr], [cls[i] for i in tr], Vl[he], ncls)
            a_ho, _ = W91.top1(s, [cls[i] for i in he])
            loo = 0
            for i in range(N_FACTS):
                keep = [j for j in range(N_FACTS) if j != i]
                si = W91.ridge_scores(Vl[keep], [cls[j] for j in keep], Vl[[i]], ncls)
                loo += int(int(si.argmax(dim=1)[0]) == cls[i])
            m2[li] = {"held_top1": a_ho, "loo_top1": loo / N_FACTS}

        # M1b(K)：键的值身份（同一把解码器，D-114）
        m1 = {}
        for li in st.mems:
            Kl = K[li].float()
            loo = 0
            for i in range(N_FACTS):
                keep = [j for j in range(N_FACTS) if j != i]
                si = W91.ridge_scores(Kl[keep], [cls[j] for j in keep], Kl[[i]], ncls)
                loo += int(int(si.argmax(dim=1)[0]) == cls[i])
            m1[li] = loo / N_FACTS

        # 键几何：cos(k_i,k_j) 中位/max
        geo = {}
        for li in st.mems:
            Kl = K[li].float()
            n = Kl / Kl.norm(dim=1, keepdim=True).clamp_min(1e-12)
            C = n @ n.T
            iu = torch.triu_indices(Kl.shape[0], Kl.shape[0], offset=1)
            vals = C[iu[0], iu[1]]
            geo[li] = {"cos_median": float(vals.median()), "cos_max": float(vals.max()),
                       "k_norm_mean": float(Kl.norm(dim=1).mean())}

        a = {"arm": arm, "seed": seed, "gamma_final": g,
             "acc_train": fit["acc_q"], "acc_held": gen["acc_q"],
             "margin_train": fit["margin_mean"], "margin_held": gen["margin_mean"],
             "violations": viol, "diverged": div,
             "final_loss": float(np.mean(losses[-50:])) if losses else None,
             "m2": {str(k): v for k, v in m2.items()},
             "m1k": {str(k): v for k, v in m1.items()},
             "geo": {str(k): v for k, v in geo.items()},
             "write_calls": wc, "write_calls_after_capture": wc_after_cap,
             "sec": round(time.time() - t0, 1)}
        print(f"  [{arm}] acc_held={a['acc_held']:.4f} acc_train={a['acc_train']:.4f} "
              f"margin(ho)={a['margin_held']:+.2f} M2loo={np.mean([v['loo_top1'] for v in m2.values()]):.3f} "
              f"M1b(K)={np.mean(list(m1.values())):.4f} "
              f"cos_med={np.mean([v['cos_median'] for v in geo.values()]):.4f} "
              f"γ={g:+.4f} viol={viol} div={div} {a['sec']}s", flush=True)
        return a
    finally:
        restore_write()
        st.remove()
        ARM["name"] = "K_mean_V_mean"
        VPOS["V"] = None


# ==================================================================== 判决
def judge(arms):
    by = {(a["seed"], a["arm"]): a for a in arms}
    seeds = sorted({a["seed"] for a in arms})
    per = {}
    for s in seeds:
        per[s] = {a: {"acc_held": by[(s, a)]["acc_held"],
                      "M2loo": float(np.mean([v["loo_top1"] for v in by[(s, a)]["m2"].values()])),
                      "M1bK": float(np.mean(list(by[(s, a)]["m1k"].values()))),
                      "cos_med": float(np.mean([v["cos_median"] for v in by[(s, a)]["geo"].values()])),
                      "cos_max": float(np.max([v["cos_max"] for v in by[(s, a)]["geo"].values()]))}
                  for a in ARMS}

    def _br(d):
        """**逐 seed** 按冻结判决表取第一条满足者。"""
        if d["K_last_V_mean"]["acc_held"] >= G_GEN:
            return "IMPLEMENTABLE_FIX"
        if any(d[a]["acc_held"] >= G_GEN for a in ("K_vt_V_mean", "K_vt_V_vt")):
            return "KEY_POOL_BOTTLENECK"
        if (all(d[a]["acc_held"] <= G_LOW for a in NEW_ARMS)
                and all(d[a]["cos_med"] <= COS_SEP for a in NEW_ARMS)
                and all(d[a]["M1bK"] >= G_AVAIL for a in NEW_ARMS)):
            return "SEPARABLE_BUT_UNREADABLE"
        if all(d[a]["cos_med"] >= COS_NATIVE for a in NEW_ARMS):
            return "KEY_NOT_SEPARABLE"
        return "INCONCLUSIVE"

    brs = {s: _br(per[s]) for s in seeds}
    vals = set(brs.values())
    n_br = len(vals)
    # ★ DEF-97-05：先算 len 后取支，不得 pop() 后再量；一致性必须由**逐 seed 支**得出
    br = next(iter(vals)) if n_br == 1 else "INCONCLUSIVE"
    h1_refuted = (br in ("KEY_NOT_SEPARABLE", "SEPARABLE_BUT_UNREADABLE"))
    return {"per_seed": {str(k): v for k, v in per.items()},
            "per_seed_branch": {str(k): v for k, v in brs.items()},
            "branch": br, "consistent": n_br == 1,
            "H1_confirmed": br in ("KEY_POOL_BOTTLENECK", "IMPLEMENTABLE_FIX"),
            "H1_refuted": bool(h1_refuted),
            "thresholds": {"G_GEN": G_GEN, "G_LOW": G_LOW, "G_DELTA": G_DELTA,
                           "chance": 1.0 / N_CAND, "G_ABSENT": G_ABSENT,
                           "G_AVAIL": G_AVAIL, "COS_SEP": COS_SEP}}


def cross_round(arms):
    """A98-11：跨轮复现 D-106（承重）。不复现 ⇒ patch 与 D-106 不同源，正式轮作废。"""
    by = {(a["seed"], a["arm"]): a for a in arms}
    seeds = sorted({a["seed"] for a in arms})
    TOL = 1.0 / 32.0
    ref = {  # D-106 实测（out/probe91_valread.log）
        "K_mean_V_mean": {"acc_held": (0.0, 0.0), "M2loo": (0.042,)},
        "K_mean_V_vt":   {"acc_held": (0.0, 0.0), "M2loo": (0.312,)},
    }
    out, ok = {}, True
    for arm, r in ref.items():
        ah = tuple(by[(s, arm)]["acc_held"] for s in seeds)
        m2 = float(np.mean([np.mean([v["loo_top1"] for v in by[(s, arm)]["m2"].values()])
                            for s in seeds]))
        oka = all(abs(a - b) <= TOL for a, b in zip(ah, r["acc_held"])) if len(ah) == len(r["acc_held"]) else False
        okm = abs(m2 - r["M2loo"][0]) <= TOL
        out[arm] = {"acc_held_got": ah, "acc_held_ref": r["acc_held"], "acc_held_ok": oka,
                    "M2loo_got": m2, "M2loo_ref": r["M2loo"][0], "M2loo_ok": okm}
        ok = ok and oka and okm
    rec("A98-11", ok,
        f"跨轮复现 D-106：{ {k: (v['acc_held_got'], v['acc_held_ok'], round(v['M2loo_got'],4), v['M2loo_ok']) for k, v in out.items()} }"
        f"（容差 ±1/32）", value=1.0 if ok else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", action="store_true")
    ap.add_argument("--seeds", type=str, default="20261028,7777")
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    args = ap.parse_args()

    tok, model = model_and_tok()
    model.eval()
    torch.set_grad_enabled(True)
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    vpos = V.value_positions(tok, facts)
    vp_by_fact = {i: vpos[i]["V"] for i in range(len(facts))}
    tr = list(range(N_TRAIN_KEYS)); he = list(range(N_TRAIN_KEYS, N_FACTS))
    cls = [V_POOL_INDEX(f) for f in facts]

    print(f"n_facts={len(facts)} chunks={len(chunks)} owner={len(set(owner))}", flush=True)
    ok = stage_a(tok, model, facts, chunks, owner, vpos)
    RES["stage_a_ok"] = bool(ok)
    if not ok:
        print(f"\n★ Stage A 阻断正式轮（FAIL={FAILS}）", flush=True)
        dump(RES, "probe91_kpool_guards")
        return 2
    if args.stage_a:
        print("* Stage A 全部通过", flush=True)
        dump(RES, "probe91_kpool_guards")
        return 0

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    sel = [x for x in args.arms.split(",") if x.strip()]
    print("\n" + "=" * 21 + " 正式轮 " + "=" * 21, flush=True)
    for seed in seeds:
        print(f"\n===== seed {seed} =====", flush=True)
        for arm in sel:
            a = run_arm(arm, tok, model, facts, chunks, owner, vp_by_fact, tr, he, seed, cls, CLS_N)
            RES["arms"].append(a)
        dump(RES, f"probe91_kpool_s{seed}")

    RES["cross_round"] = cross_round(RES["arms"])
    j = judge(RES["arms"])
    RES["judgement"] = j
    print("\n" + "=" * 70, flush=True)
    for s in sorted({a["seed"] for a in RES["arms"]}):
        row = {a: round(next(x for x in RES["arms"] if x["seed"] == s and x["arm"] == a)["acc_held"], 4)
               for a in sel}
        print(f"  seed {s} acc_held: {row}", flush=True)
    print(f"  ★ 判决 = {j['branch']}  (consistent={j['consistent']}, "
          f"H1_confirmed={j['H1_confirmed']}, H1_refuted={j['H1_refuted']})", flush=True)
    dump(RES, "probe91_kpool")
    return 0


def V_POOL_INDEX(f):
    """值 -> 值池下标（与 D-106/D-114 同一 cls 口径）。"""
    from R2_task91 import VALUE_POOL
    return VALUE_POOL.index(f["val"])


if __name__ == "__main__":
    sys.exit(main())
