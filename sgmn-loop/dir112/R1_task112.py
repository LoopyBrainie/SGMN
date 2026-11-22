#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-111 · R1 —— 两通路复合是「串联最弱环」还是「可加求和」？**内点预注册复现**。

卡片：`revolve/DIR-111/DIRECTION_rev111_v10.md`（rev 1.0，**冻结**）

★ 派生方式：从 `dir109/R1_task109.py` **逐字派生**（`cp` 后只改四处）。
  **不改**：`run_cell` / `measure_state` / `state_scalars` / `struct_stats` /
  `install_capture` / `snapshot` / `train_read_sel` / `params_sel` / `set_grad_gamma`
  与一切装置常量（`READ_STEPS`/`READ_LR`/`CLIP`/`N_FACTS`/…），全部逐字继承。
  **只改**：`GAMMAS`（3 个新 γ）· `SEEDS`（3 个新 seed）· `RES["dir"]/card` ·
  `stage_a()`（DIR-111 的守卫清单）· `verdict()`（内点形状判据）。

★★ `DIR-110` §7 的「8/8 最弱环」证据**已被本卡 §0.2 机械否决**（`COR-111-01`）：
   那 8 行**全部**满足 `δ_Q+δ_O > 1`，即求和模型预测**不可容许值**（`δ_B ≤ 1` 是恒等约束）
   ⇒ 「`MAX` 赢」由**天花板强制**，不是串联复合的证据。加内点门后合格 **0/8**。
   ⇒ 本卡改在**内点**（`δ_Q+δ_O ≤ 0.95`）检验，故必须有**新格**。

★★ **`READ_LR` 必须是 dir90 系的 `1e-2`**（`dir90/R1_task90.py:47`），**不得**改成 dir91 系的 `1e-3`。
   `D-138`/`D-139` 的「`1e-2` 应为 `1e-3`」更正**只适用于 `DIR-105`（dir91 系）**。
"""
import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dir90"))
sys.path.insert(0, str(_ROOT / "dir104"))

import R1_task90 as D90                                    # noqa: E402
import R1_task104 as D104                                 # noqa: E402
from sgmn_module import sg                                # noqa: E402

OUT = _HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
D90.OUT = OUT                     # ★ 必须在本 import 之后覆盖（D104 也设过 D90.OUT）

# ---------------------------------------------------------------- 冻结阈值（卡片 §3.2）
TAU_ATTR = 0.10
# ★ 同名别名：`DIR-110`/`DIR-111` 的卡里叫 `TAU_ARM`，`DIR-109` 叫 `TAU_ATTR` —— **同一个值 0.10**，
#   不是新阈值，只为了与本卡文字一致（`A111-10` 的冻结枚举也用它）。
TAU_ARM = TAU_ATTR
RATIO = 2.0
TAU_GAIN = 10.0
PROBE_RTOL = 1e-5
W_DRIFT_TOL = 1e-8
# ★★ 本卡（`DIR-112` rev 1.0）**只测一个 γ** —— 由 `PREMISE-CHECK.md` 的功率分析强制：
#    5 γ × 3 seed 的信噪比只有 1.6–1.7（同 γ 内 seed 就翻号，`COR-112-01`），
#    故把预算换成 1 γ × 8 seed 的**符号检验**（`P(≥7/8|H0)=0.0352`）。卡片 §2.3。
GAMMA112 = 0.0010
GAMMAS = (GAMMA112,)
# ★ 8 个**新** seed（与 DIR-111 的 20261118/20261125/14 及全仓库历史 seed 零碰撞）—— 卡片 §2.2 冻结
SEEDS = (20261208, 20261224, 20270105, 20270119, 20270202, 20270216, 20270302, 20270316)

# ---- 本卡判据的冻结阈值（卡片 §2.3）----
MIN_NEG = 7              # 每个量各自的负号 seed 数下限（`≥7/8` ⇒ 单侧 p=0.0352）
H0_P = 0.0352            # 单侧二项 P(≥7/8)，写死供复核
NULL_EPS = 1e-9          # `1 − X_NONE ≤ NULL_EPS` ⇒ 该 seed 不可用
N_SEEDS = 8
# 以下三项**继承自 DIR-111 但本卡不作判据**，保留仅为让 `shape_*`/`dir110_ceiling_audit` 可复用
CEIL_MAX = 0.95
MIN_NQ = 3
RATIO_MAX = 0.90
ARMS = ("T-NONE", "T-Q", "T-O", "T-BOTH")
WHICH = {"T-NONE": "none", "T-Q": "q", "T-O": "o", "T-BOTH": "both"}
L16 = "16"                        # 判据 B 的口径层（κ 最大层）

RES = {"dir": "DIR-112", "round": "R1", "model": D90.MODEL, "device": D90.DEV,
       "card": "revolve/DIR-112/DIRECTION_rev112_v10.md",
       "min_neg": MIN_NEG, "h0_p": H0_P, "null_eps": NULL_EPS, "n_seeds": N_SEEDS,
       "derived_from": "dir111/R1_task111.py",
       "tau_attr": TAU_ATTR, "ratio": RATIO, "tau_gain": TAU_GAIN,
       "probe_rtol": PROBE_RTOL, "w_drift_tol": W_DRIFT_TOL,
       "gammas": list(GAMMAS), "seeds": list(SEEDS), "arms": list(ARMS),
       "read_lr": float(D90.READ_LR), "read_steps": D90.READ_STEPS,
       "device_md5": {}, "stageA": [], "cells": {}, "crit": {}}

_CAP = {}


def rec(name, ok, detail, value=None, blocking=True):
    RES["stageA"].append({"name": name, "ok": bool(ok), "detail": detail,
                          "value": value, "blocking": bool(blocking)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return bool(ok)


def dump(o, n):
    D90.dump(o, n)


# ------------------------------------------------------------------ 可训练参数集（自变量）
def params_sel(st, which):
    """`which="both"` 的顺序**必须**与 `D104.params_nogamma` 逐项同序同物（`A109-1b`）。"""
    assert which in ("none", "q", "o", "both"), which
    ps = []
    for m in st.mems.values():
        if which in ("q", "both"):
            ps.append(m.Wq)
        if which in ("o", "both"):
            ps.append(m.Wo)
    return ps


def set_grad_gamma(st, on):
    for m in st.mems.values():
        m.gamma.requires_grad_(bool(on))


def gammas(st):
    return {str(li): float(m.gamma.detach()) for li, m in st.mems.items()}


# ------------------------------------------------------------------ 训练（单侧臂：逐行复刻）
def train_read_sel(model, st, tok, facts, train_keys, rng, which):
    """`which="both"` ⇒ **直接**调 `D104.train_read_frozen`（同代码路径 ⇒ `A109-1` 构造性成立）。
    其余 ⇒ 逐行复刻 `R1_task104.py:110-146`，**唯一**差别是优化器参数集。"""
    if which == "both":
        return D104.train_read_frozen(model, st, tok, facts, train_keys, rng)

    import torch.nn.functional as F
    ps = params_sel(st, which)
    assert len(ps) > 0
    for m in st.mems.values():
        m.no_write = True
    set_grad_gamma(st, False)
    opt = torch.optim.Adam(ps, lr=D90.READ_LR)
    batches = D104.make_batches(tok, facts, train_keys, rng)
    losses, viol, diverged, gtrace = [], 0, False, []
    for step in range(D90.READ_STEPS):
        prompt_ids, ids, tgt = batches[step % len(batches)]
        P = len(prompt_ids)
        seq = torch.tensor([prompt_ids + [c] for c in ids], dtype=torch.long, device=D90.DEV)
        opt.zero_grad(set_to_none=True)
        lg = model(input_ids=seq).logits.float()
        idx = torch.arange(len(ids), device=D90.DEV)
        pick = lg[idx, P - 1, torch.tensor(ids, device=D90.DEV)].unsqueeze(0)
        loss = F.cross_entropy(pick, torch.tensor([tgt], device=D90.DEV))
        loss.backward()
        if step % 50 == 0:
            # 写入参数不得有梯度（契约；与 D104 同口径）
            for p in D90.write_params(st):
                if p.grad is not None and float(p.grad.abs().max()) != 0.0:
                    viol += 1
            gtrace.append(gammas(st))
        torch.nn.utils.clip_grad_norm_([p for p in ps if p.grad is not None], D90.CLIP)
        opt.step()
        losses.append(float(loss))
        if not all(bool(torch.isfinite(D90.W_of(m)).all()) for m in st.mems.values()):
            diverged = True
            break
    for m in st.mems.values():
        m.no_write = False
    set_grad_gamma(st, True)
    return losses, viol, diverged, gtrace


# ------------------------------------------------------------------ 状态读数
def snapshot(st):
    return {li: {"Wq": m.Wq.detach().clone(), "Wo": m.Wo.detach().clone(),
                 "W": D90.W_of(m).clone()} for li, m in st.mems.items()}


def state_scalars(st, snap):
    """位移 `ρ`（★ `D-138`/`D-139` 纪律：只用位移，不用梯度幅度）与读路径增益。"""
    out = {}
    for li, m in st.mems.items():
        s = snap[li]
        W = D90.W_of(m)
        nq0, no0 = float(s["Wq"].norm()), float(s["Wo"].norm())
        Wq, Wo = m.Wq.detach(), m.Wo.detach()
        g0 = float((s["Wo"] @ s["W"] @ s["Wq"]).norm())
        g1 = float((Wo @ W @ Wq).norm())
        out[str(li)] = {
            "rho_q": float((Wq - s["Wq"]).norm()) / max(nq0, 1e-12),
            "rho_o": float((Wo - s["Wo"]).norm()) / max(no0, 1e-12),
            "dW_fro": float((W - s["W"]).norm()),
            "gain": g1 / max(g0, 1e-12),
            "gain_abs": g1,
            "nWq0": nq0, "nWo0": no0,
            "nWq": float(Wq.norm()), "nWo": float(Wo.norm()),
        }
    return out


def struct_stats(vecs):
    """题间可分辨性：`cos_mean`（越大越同向）、`rho`（越小越同向；各向同性 ≈√2）、
    `rank_eff = (Σσ)²/Σσ²`（秩 r 且奇异值等 ⇒ r）。"""
    M = torch.stack(vecs).float()
    n = M.shape[0]
    Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-12)
    G = Mn @ Mn.T
    off = G[~torch.eye(n, dtype=torch.bool, device=G.device)]
    d = torch.cdist(M, M)
    iu = torch.triu_indices(n, n, 1)
    sv = torch.linalg.svdvals(M.double())
    return {"cos_mean": float(off.mean()),
            "rho": float(d[iu[0], iu[1]].mean()) / max(float(M.norm(dim=1).mean()), 1e-12),
            "rank_eff": float((sv.sum() ** 2) / (sv ** 2).sum())}


def install_capture(st):
    """捕获 `c / q / m`，并**重算** `m` 做自洽守卫（`A109-3`）。"""
    for li, m in st.mems.items():
        orig = m.read

        def _read(hs_in, drop_sg=False, _m=m, _orig=orig, _li=li):
            out = _orig(hs_in, drop_sg)
            with torch.no_grad():
                c = sg(hs_in, drop_sg)
                sh = hs_in.shape
                cf = c.reshape(-1, c.shape[-1])
                q = (_m.Wq @ cf.T).T
                W = _m.Wfull if not _m.use_factorized else (_m.W0 + _m.U @ _m.V.T)
                y = (sg(W, drop_sg) @ q.T).T
                mo = (_m.Wo @ y.T).T
                q3 = q.reshape(sh[0], sh[1], -1)
                m3 = mo.reshape(sh)
                den = max(float(out.abs().max()), 1e-12)
                err = float((m3 - out).abs().max()) / den
            _CAP[_li] = {"c": c.detach().float()[0, -1, :].clone(),
                         "q": q3.detach().float()[0, -1, :].clone(),
                         "m": out.detach().float()[0, -1, :].clone(),
                         "hnorm": float(hs_in.detach().float().norm(dim=-1)[0, -1]),
                         "rel_err": err}
            return out
        m.read = _read


@torch.no_grad()
def measure_state(model, st, items):
    """四臂**全测**（`A109-8`：`DIR-104` 的未训练臂从未读过状态）。"""
    _CAP.clear()
    install_capture(st)
    try:
        acc = {li: [] for li in st.mems}
        for it in items:
            j = next(k for k, (_, ok) in enumerate(it["cand"]) if ok)
            D90.nll_of_candidate(model, st, it["prompt_ids"], it["cand"][j][0])
            for li, d in _CAP.items():
                acc[li].append(d)
    finally:
        D104.uninstall_probe(st, "read")
    relerr = max(d["rel_err"] for rows in acc.values() for d in rows)
    out = {}
    for li, rows in acc.items():
        g = float(st.mems[li].gamma.detach())
        kap = float(np.mean([float(d["m"].norm()) * g for d in rows])) / \
            max(float(np.mean([d["hnorm"] for d in rows])), 1e-12)
        out[str(li)] = {"gamma": g, "kappa": kap,
                        "c": struct_stats([d["c"] for d in rows]),
                        "q": struct_stats([d["q"] for d in rows]),
                        "m": struct_stats([d["m"] for d in rows])}
    return out, relerr


# ------------------------------------------------------------------ 单格
def run_cell(arm, gamma0, seed, tok, model, facts, chunks, owner, train_keys, eval_keys,
             keep_S=True):
    t0 = time.time()
    which = WHICH[arm]
    st = D90.make_stack(gamma=gamma0, no_write=True, seed=seed)   # 四臂全部 no_write
    snap = snapshot(st)
    losses, viol, diverged = [], None, False
    if which != "none":
        losses, viol, diverged, _gt = train_read_sel(
            model, st, tok, facts, train_keys, random.Random(1000 + seed % 100), which)
    # ★★★ `DEF-109-02`（= `DEF-95-01` 重蹈）：`D104.train_read_frozen` 末尾把 `no_write`
    #      **还原为 False**；不重封 ⇒ 评测期间写入规则会真的跑 ⇒ `W` 被改写、激活溢到 inf。
    #      `T-NONE` 不训练所以从不被解封 —— 这正是 `A109-2` 过而 `A109-1` 挂的原因。
    for m in st.mems.values():
        m.no_write = True
    assert all(m.no_write for m in st.mems.values()), "评测前必须重封 no_write"
    wc0 = {li: m.write_calls for li, m in st.mems.items()}
    items = eval_items_cache["items"]
    scal = state_scalars(st, snap)
    state, relerr = measure_state(model, st, items)
    tgt = D104.targets_of(items)
    S = D104.score_matrix(model, st, items)
    acc_i = D104.acc_item_of(S, tgt)
    acc_q, nq = D104.acc_q_of(S, tgt, items)
    ok = D104.item_ok(S, tgt)
    wc1 = {li: m.write_calls for li, m in st.mems.items()}
    eval_writes = int(sum(wc1[li] - wc0[li] for li in wc0))
    finite_params = all(bool(torch.isfinite(m.Wq).all()) and bool(torch.isfinite(m.Wo).all())
                        for m in st.mems.values())

    out = {"arm": arm, "which": which, "gamma0": gamma0, "seed": seed,
           "acc_item": acc_i, "acc_q": acc_q, "n_q": nq, "n_item": len(items),
           "final_loss": float(np.mean(losses[-50:])) if losses else None,
           "gamma_final": gammas(st), "violations": viol, "diverged": diverged,
           "eval_writes": eval_writes, "finite_params": finite_params,
           "state": state, "scalars": scal, "probe_rel_err": relerr,
           "item_ok": [int(x) for x in ok], "sec": round(time.time() - t0, 1)}
    if keep_S:
        out["S"] = [[round(float(x), 4) for x in row] for row in S]
        out["tgt"] = [int(x) for x in tgt]
    print(f"  [{arm} g={gamma0} seed={seed}] acc_item={acc_i:.4f} "
          f"rho_q={scal[L16]['rho_q']:.4f} rho_o={scal[L16]['rho_o']:.4f} "
          f"gain16={scal[L16]['gain']:.3e} kappa16={state[L16]['kappa']:.3f} "
          f"{'div ' if diverged else ''}{out['sec']}s", flush=True)
    st.remove()
    return out


eval_items_cache = {}


# ------------------------------------------------------------------ 判据函数（纯函数，`A109-5` 验算）
def classify(DQ, DO, DB):
    """卡片 §2.3 的冻结求值顺序（互斥且穷尽）。"""
    I = DB - DQ - DO
    if DB < TAU_ATTR:
        return "C109-NO-DAMAGE", I
    if DQ >= TAU_ATTR and DO >= TAU_ATTR and abs(I) < TAU_ATTR:
        return "C109-BOTH-ADDITIVE", I
    if DQ >= TAU_ATTR and DQ >= RATIO * DO and abs(I) < TAU_ATTR:
        return "C109-WQ-DOMINANT", I
    if DO >= TAU_ATTR and DO >= RATIO * DQ and abs(I) < TAU_ATTR:
        return "C109-WO-DOMINANT", I
    if DQ < TAU_ATTR and DO < TAU_ATTR and DB >= TAU_ATTR:
        return "C109-MULTIPLICATIVE", I
    return "C109-NO-CLEAN-ATTRIBUTION", I


# ------------------------------------------------------------------ Stage A

# ==================== 内点形状判据（卡片 §2.1–§2.3，纯函数） ====================
L8 = "8"
OBJS_V = ("m",)                     # ★ 只判 m（q 由 A111-7 证明 delta_O == 0，无信息）
METS_V = ("rank_eff", "cos_mean")   # 2 个量
Q_CAND = [(o, mt, g) for o in OBJS_V for mt in METS_V for g in GAMMAS]   # 6 个候选（冻结枚举）


def _cells_have(cells, arm, g, seeds):
    return [cells[f"{arm}|{g}|{s}"] for s in seeds if f"{arm}|{g}|{s}" in cells]


def _X(cells, arm, g, L, o, mt, seeds=None):
    """三 seed 均值（`seeds` 给定时只用那些 seed）。缺格返回 None。"""
    ss = list(SEEDS) if seeds is None else list(seeds)
    have = _cells_have(cells, arm, g, ss)
    if len(have) != len(ss):
        return None
    return float(np.mean([c["state"][L][o][mt] for c in have]))


def shape_point(cells, g, L, o, mt, seeds=None):
    """归一化损伤 delta 与两模型预测（卡片 §2.1/§2.2 的逐字实现）。"""
    xn = _X(cells, "T-NONE", g, L, o, mt, seeds)
    if xn is None:
        return {"gamma": g, "layer": L, "obj": o, "met": mt, "missing": True}
    span = 1.0 - xn
    if abs(span) < 1e-9:
        return {"gamma": g, "layer": L, "obj": o, "met": mt, "skip": True,
                "why": "span=1-X_NONE=0"}
    xq = _X(cells, "T-Q", g, L, o, mt, seeds)
    xo = _X(cells, "T-O", g, L, o, mt, seeds)
    xb = _X(cells, "T-BOTH", g, L, o, mt, seeds)
    if None in (xq, xo, xb):
        return {"gamma": g, "layer": L, "obj": o, "met": mt, "missing": True}
    dq, do, db = (xq - xn) / span, (xo - xn) / span, (xb - xn) / span
    mp, sp = max(dq, do), dq + do
    me, se = abs(db - mp), abs(db - sp)
    ratio = (me / se) if se > 1e-12 else float("nan")
    # ---- 内点门（卡片 §2.2，三条同时满足） ----
    why = []
    if abs(dq) < TAU_ARM:
        why.append(f"|dQ|={abs(dq):.3f}<{TAU_ARM}")
    if abs(do) < TAU_ARM:
        why.append(f"|dO|={abs(do):.3f}<{TAU_ARM}")
    if sp > CEIL_MAX:
        why.append(f"dQ+dO={sp:.3f}>{CEIL_MAX}(sum 越界)")
    return {"gamma": g, "layer": L, "obj": o, "met": mt,
            "X_NONE": xn, "X_Q": xq, "X_O": xo, "X_BOTH": xb, "span": span,
            "dQ": dq, "dO": do, "dB": db, "max_pred": mp, "sum_pred": sp,
            "max_err": me, "sum_err": se, "ratio": ratio,
            "eligible": not why, "why_not": why}


def shape_verdict(cells, seeds=None):
    """卡片 §2.3 的唯一判定路径。返回 (code, detail)。"""
    pts = [shape_point(cells, g, L8, o, mt, seeds) for (o, mt, g) in Q_CAND]
    ok = [p for p in pts if p.get("eligible")]
    n_q = len(ok)
    detail = {"points": pts, "n_candidates": len(pts), "n_eligible": n_q,
              "ceil_max": CEIL_MAX, "tau_arm": TAU_ARM, "min_nq": MIN_NQ}
    if n_q < MIN_NQ:
        return "C111-NO-INTERIOR-POINT", detail
    losers = [p for p in ok if not (p["max_err"] < p["sum_err"])]
    ratios = sorted(p["ratio"] for p in ok)
    med = float(np.median(ratios))
    detail.update({"median_ratio": med, "ratios": ratios,
                   "sum_model_wins": [f"{p['obj']}.{p['met']}@g{p['gamma']}" for p in losers]})
    if losers:
        return "C111-FALSIFIED", detail
    if med <= RATIO_MAX:
        return "C111-WEAKEST-LINK-CONFIRMED", detail
    return "C111-WEAKEST-LINK-WEAK", detail


def dir110_ceiling_audit():
    """卡片 §0.2 / `A111-8`：从 DIR-110 冻结 json 复算上限审计（零新格）。"""
    f = _ROOT / "dir110" / "out" / "verdict110.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    rows = []
    for p in d.get("points", []):
        if p.get("skip") or p.get("obj") != "m":
            continue
        rows.append({"key": f"{p['met']}@L{p['layer']}",
                     "gamma": float(p["gamma"]),
                     "dQ": float(p["dQ"]), "dO": float(p["dO"]),
                     "dB": float(p["dB"]),
                     "sum": float(p["dQ"]) + float(p["dO"]),
                     "max_err": float(p["max_err"]), "sum_err": float(p["sum_err"])})
    n_all = len(rows)
    n_over = sum(1 for r in rows if r["sum"] > 1.0)
    n_int = sum(1 for r in rows if r["sum"] <= CEIL_MAX)
    return {"rows": rows, "n": n_all, "n_over_1": n_over, "n_interior": n_int}


def delta_of(cells, seed, mt, gamma=None):
    """★ 卡片 §1 的逐 seed 配对归一化损伤。返回 (Δ, δ_Q, δ_O, δ_B) 或 None（不可用）。

    ★★ 必须**逐 seed 配对**（同 seed 的四臂相减）：`DIR-111` 的 `_X` 是**跨 seed 平均**
       `X` 之后才算 δ，那会掩盖 seed 间的符号翻转 —— 正是 `COR-112-01` 的教训。
    """
    g = GAMMA112 if gamma is None else gamma
    xn = _X(cells, "T-NONE", g, L8, "m", mt, seeds=[seed])
    if xn is None:
        return None
    if abs(1.0 - xn) <= NULL_EPS:          # 归一化分母退化 ⇒ 该 seed 不可用
        return None
    d = {}
    for arm, tag in (("T-Q", "Q"), ("T-O", "O"), ("T-BOTH", "B")):
        xa = _X(cells, arm, g, L8, "m", mt, seeds=[seed])
        if xa is None:
            return None
        d[tag] = (xa - xn) / (1.0 - xn)
    delta = d["B"] - max(d["Q"], d["O"])
    return delta, d["Q"], d["O"], d["B"]


def sign_verdict(cells, seeds=None):
    """卡片 §1.4 的冻结判决（**穷尽且互斥**，按顺序求值）。纯函数。"""
    ss = list(SEEDS) if seeds is None else list(seeds)
    per = {}
    for mt in METS_V:
        rows = []
        for sd in ss:
            got = delta_of(cells, sd, mt)
            rows.append({"seed": sd, "ok": got is not None,
                         "delta": None if got is None else got[0],
                         "dQ": None if got is None else got[1],
                         "dO": None if got is None else got[2],
                         "dB": None if got is None else got[3]})
        usable = [r for r in rows if r["ok"]]
        n_neg = sum(1 for r in usable if r["delta"] < 0)
        per[mt] = {"rows": rows, "n_usable": len(usable), "n_neg": n_neg}
    ok = {mt: (per[mt]["n_usable"] == len(ss) and per[mt]["n_neg"] >= MIN_NEG)
          for mt in METS_V}
    if any(per[mt]["n_usable"] < len(ss) for mt in METS_V):
        code = "C112-INSUFFICIENT"
    elif all(ok[mt] for mt in METS_V):
        code = "C112-ANTAGONISM-CONFIRMED"
    elif any(ok[mt] for mt in METS_V):
        code = "C112-PARTIAL"
    else:
        code = "C112-FALSIFIED"
    return code, per


def delta_selftest():
    """`A112-7`：`Δ` 的恒等式自检（纯合成数，不碰模型）。"""
    import numpy as _np
    bad = []
    # 直接验算 Δ 的定义，不经 cells（避免夹具开销）
    cases = [((0.2, 0.2, 0.1), -0.1),
             ((0.2, 0.2, 0.3), +0.1),
             ((0.1, 0.5, 0.4), -0.1),      # max 取 δ_O=0.5 ⇒ 0.4-0.5
             ((0.5, 0.1, 0.4), -0.1),      # max 取 δ_Q=0.5 ⇒ 0.4-0.5
             ((0.3, 0.7, 0.7), +0.0),      # 恰等于 max
             # ★ 全负：`max(-0.1,-0.2) = -0.1`（取**较不负**的一侧）⇒ Δ = -0.3 + 0.1 = **-0.2**
             #   （我首次写这个算例时写成了 -0.1，是**期望值错**，不是代码错 —— 见测试首跑的记录）
             ((-0.1, -0.2, -0.3), -0.2)]
    for (dq, do, db), want in cases:
        got = db - max(dq, do)
        if abs(got - want) > 1e-12:
            bad.append({"dQ": dq, "dO": do, "dB": db, "want": want, "got": got})
    return bad


def stage_a(tok, model, facts, chunks, owner, train_keys, eval_keys, items):
    """`DIR-112` 的 Stage A 硬门（卡片 §3）。任一 `blocking` 失败 ⇒ 阻断正式轮。"""
    def rec(name, ok, detail, blocking=True):
        RES["stageA"].append({"name": name, "ok": bool(ok), "detail": detail,
                              "blocking": bool(blocking)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        return bool(ok)

    # ---------- A112-0 装置 4 文件 md5 ----------
    want = {"sgmn_module.py": "80e334a5a6751a07d9e03162ebf17ff0"}
    RES["device_md5"] = {}
    dev_bad = {}
    for rel in ("../sgmn_module.py", "../dir90/R1_task90.py", "../dir104/R1_task104.py",
                "../dir111/R1_task111.py"):
        q = (_HERE / rel).resolve()
        if not q.exists():
            dev_bad[rel] = "缺失"
            continue
        RES["device_md5"][rel] = md5(q)
    if want["sgmn_module.py"] != RES["device_md5"].get("../sgmn_module.py"):
        dev_bad["sgmn_module.py"] = (RES["device_md5"].get("../sgmn_module.py"), want["sgmn_module.py"])
    rec("A112-0", not dev_bad, f"4 个装置文件 md5（不合 {dev_bad}）；自身 md5={md5(pathlib.Path(__file__))}")

    # ---------- A112-1 数据结构 ----------
    rec("A112-1", isinstance(RES["cells"], dict),
        f"RES['cells'] 是 dict（必须是 dict；键 'arm|gamma|seed'）")

    # ---------- A112-11 冻结枚举 ----------
    enum = {"GAMMA112": (GAMMA112, 0.0010), "len(SEEDS)": (len(SEEDS), 8),
            "MIN_NEG": (MIN_NEG, 7), "H0_P": (H0_P, 0.0352), "NULL_EPS": (NULL_EPS, 1e-9),
            "len(GAMMAS)": (len(GAMMAS), 1)}
    enum_bad = {k: v for k, v in enum.items() if v[0] != v[1]}
    rec("A112-11", not enum_bad, f"冻结枚举 {enum} ⇒ 不合 {enum_bad}")

    # ---------- A112-2 / A112-4 未训练臂与单通路隔离（用 1 个 seed 实跑 2 格） ----------
    s0 = SEEDS[0]
    c_none = run_cell("T-NONE", GAMMA112, s0, tok, model, facts, chunks, owner,
                      train_keys, eval_keys)
    RES["cells"][f"T-NONE|{GAMMA112}|{s0}"] = c_none
    sc = c_none["scalars"][L16]
    rec("A112-2", abs(sc["rho_q"]) < 1e-9 and abs(sc["rho_o"]) < 1e-9 and abs(sc["gain"] - 1.0) < 1e-9,
        f"T-NONE: rho_q={sc['rho_q']:.3e} rho_o={sc['rho_o']:.3e} gain={sc['gain']:.12f}（须 0/0/1）")

    # ---------- A112-3 ★ T-NONE 的 state 逐位 γ-不变 ----------
    c_none2 = run_cell("T-NONE", 0.0100, s0, tok, model, facts, chunks, owner,
                       train_keys, eval_keys)
    diffs = {mt: c_none["state"][L8]["m"][mt] - c_none2["state"][L8]["m"][mt]
             for mt in METS_V}
    rec("A112-3", all(abs(v) == 0.0 for v in diffs.values()),
        f"T-NONE state 在 γ=0.0010 vs 0.0100 逐位差 {diffs}（须全 0.0）")

    # ---------- A112-4 单通路隔离 ----------
    c_q = run_cell("T-Q", GAMMA112, s0, tok, model, facts, chunks, owner, train_keys, eval_keys)
    c_o = run_cell("T-O", GAMMA112, s0, tok, model, facts, chunks, owner, train_keys, eval_keys)
    for k, v in (("T-Q", c_q), ("T-O", c_o)):
        RES["cells"][f"{k}|{GAMMA112}|{s0}"] = v
    sq, so = c_q["scalars"][L16], c_o["scalars"][L16]
    rec("A112-4", abs(sq["rho_o"]) < 1e-9 and abs(so["rho_q"]) < 1e-9,
        f"T-Q rho_o={sq['rho_o']:.3e} / T-O rho_q={so['rho_q']:.3e}（须各为 0）")

    # ---------- A112-5 评测期零写入 + 参数有限 ----------
    allc = list(RES["cells"].values())
    bad5 = [k for k, c in RES["cells"].items()
            if c.get("eval_writes", 0) != 0 or not c.get("finite_params", False)]
    rec("A112-5", not bad5, f"格数 {len(allc)}；eval_writes!=0 或 finite_params=False 的格 {bad5}")

    # ---------- A112-6 判据纯逻辑自测（8 个合成场景） ----------
    def mk(per_seed_vals):
        """用合成 X 造 cells：X_NONE=0.5，X_arm 由目标 δ 反解 ⇒ δ 精确可控。"""
        cc = {}
        xn = 0.5
        for sd, (dq, do, db) in per_seed_vals.items():
            cc[f"T-NONE|{GAMMA112}|{sd}"] = {"state": {L8: {"m": {"rank_eff": xn, "cos_mean": xn}}}}
            for arm, val in (("T-Q", dq), ("T-O", do), ("T-BOTH", db)):
                cc[f"{arm}|{GAMMA112}|{sd}"] = {
                    "state": {L8: {"m": {"rank_eff": xn + val * (1 - xn),
                                        "cos_mean": xn + val * (1 - xn)}}}}
        return cc
    ss8 = list(range(9001, 9009))
    sc6 = []
    # a) 8/8 负 ⇒ CONFIRMED
    c = mk({sd: (0.2, 0.3, 0.05) for sd in ss8})
    sc6.append(("8/8负→CONFIRMED", sign_verdict(c, ss8)[0], "C112-ANTAGONISM-CONFIRMED"))
    # b) 7/8 负 ⇒ CONFIRMED（边界内侧）
    d = {sd: (0.2, 0.3, 0.05) for sd in ss8}; d[ss8[0]] = (0.2, 0.05, 0.5)
    sc6.append(("7/8负→CONFIRMED", sign_verdict(mk(d), ss8)[0], "C112-ANTAGONISM-CONFIRMED"))
    # c) 6/8 负 ⇒ FALSIFIED（边界外侧）
    d = {sd: (0.2, 0.3, 0.05) for sd in ss8}
    for sd in ss8[:2]: d[sd] = (0.2, 0.05, 0.5)
    sc6.append(("6/8负→FALSIFIED", sign_verdict(mk(d), ss8)[0], "C112-FALSIFIED"))
    # d) 全正 ⇒ FALSIFIED
    c = mk({sd: (0.2, 0.1, 0.5) for sd in ss8})
    sc6.append(("8/8正→FALSIFIED", sign_verdict(c, ss8)[0], "C112-FALSIFIED"))
    # e) 恰等于 max ⇒ Δ=0 ⇒ 不算负 ⇒ FALSIFIED
    c = mk({sd: (0.3, 0.3, 0.3) for sd in ss8})
    sc6.append(("Δ=0不算负→FALSIFIED", sign_verdict(c, ss8)[0], "C112-FALSIFIED"))
    # f) 缺一个 seed 的格 ⇒ INSUFFICIENT
    c = mk({sd: (0.2, 0.3, 0.05) for sd in ss8}); del c[f"T-BOTH|{GAMMA112}|{ss8[0]}"]
    sc6.append(("缺格→INSUFFICIENT", sign_verdict(c, ss8)[0], "C112-INSUFFICIENT"))
    # g) 分母退化（X_NONE=1.0）⇒ INSUFFICIENT
    c = mk({sd: (0.2, 0.3, 0.05) for sd in ss8})
    c[f"T-NONE|{GAMMA112}|{ss8[0]}"]["state"][L8]["m"]["rank_eff"] = 1.0
    sc6.append(("分母退化→INSUFFICIENT", sign_verdict(c, ss8)[0], "C112-INSUFFICIENT"))
    # h) 单量成立 ⇒ PARTIAL（rank_eff 全负、cos_mean 全正）
    c = mk({sd: (0.2, 0.3, 0.05) for sd in ss8})
    for sd in ss8:
        c[f"T-Q|{GAMMA112}|{sd}"]["state"][L8]["m"]["cos_mean"] = 0.5 + 0.2 * 0.5
        c[f"T-O|{GAMMA112}|{sd}"]["state"][L8]["m"]["cos_mean"] = 0.5 + 0.1 * 0.5
        c[f"T-BOTH|{GAMMA112}|{sd}"]["state"][L8]["m"]["cos_mean"] = 0.5 + 0.5 * 0.5
    sc6.append(("单量→PARTIAL", sign_verdict(c, ss8)[0], "C112-PARTIAL"))
    bad6 = [(n, g, w) for n, g, w in sc6 if g != w]
    rec("A112-6", not bad6, f"8 个合成场景全对（{len(sc6)} 个）；失败 {bad6}")

    # ---------- A112-7 Δ 恒等式 ----------
    bad7 = delta_selftest()
    rec("A112-7", not bad7, f"Δ 恒等式 6 个算例；失败 {bad7}")

    # ---------- A112-8 样本外纪律：cells 的 seed 全在 SEEDS 内 ----------
    foreign = sorted({k.split("|")[2] for k in RES["cells"]} - {str(s) for s in SEEDS})
    rec("A112-8", not foreign, f"cells 中的 seed 全在 SEEDS112 内；外来 seed {foreign}")

    # ---------- A112-9 复核 DIR-111 的 6/6 ----------
    src = (_ROOT / "dir111" / "out" / "stage_formal.json")
    if src.exists():
        old = json.loads(src.read_text(encoding="utf-8"))["cells"]
        rows = {}
        for mt in METS_V:
            ds = []
            for sd in (20261118, 20261125, 14):
                got = delta_of(old, sd, mt)      # 用**同一函数**，只是换数据源
                ds.append(None if got is None else round(got[0], 6))
            rows[mt] = ds
        ok9 = all(all(x is not None and x < 0 for x in v) for v in rows.values())
    else:
        rows, ok9 = {"err": "缺 dir111/out/stage_formal.json"}, False
    rec("A112-9", ok9, f"DIR-111 γ=0.001 的逐 seed Δ 须全负：{rows}")

    RES["plan_n"] = N_SEEDS * len(ARMS)          # ★ 32（T-NONE 每 seed 一格，跨 γ 共用）
    return all(g["ok"] for g in RES["stageA"] if g["blocking"])


def md5(p):

    import hashlib
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ 判据
def verdict():
    """卡片 §1.4 的判决 + 报告块（report-only）。"""
    cells = RES["cells"]
    code, per = sign_verdict(cells)

    # 报告块①：逐 seed 明细已在 per 里；汇总 mean/SD
    rep = {}
    for mt in METS_V:
        vals = [r["delta"] for r in per[mt]["rows"] if r["ok"]]
        dq = [r["dQ"] for r in per[mt]["rows"] if r["ok"]]
        do = [r["dO"] for r in per[mt]["rows"] if r["ok"]]
        db = [r["dB"] for r in per[mt]["rows"] if r["ok"]]
        rep[mt] = {
            "n_usable": per[mt]["n_usable"], "n_neg": per[mt]["n_neg"],
            "delta_mean": float(np.mean(vals)) if vals else None,
            "delta_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "dQ_mean": float(np.mean(dq)) if dq else None,
            "dO_mean": float(np.mean(do)) if do else None,
            "dB_mean": float(np.mean(db)) if db else None,
            # 报告块②：与 DIR-111 同 γ 的并排对照（**只报告差距，不判**）
            "dir111_delta_mean": {"rank_eff": -0.0932, "cos_mean": -0.3626}[mt],
        }
        rep[mt]["gap_vs_dir111"] = (None if rep[mt]["delta_mean"] is None
                                    else rep[mt]["delta_mean"] - rep[mt]["dir111_delta_mean"])

    # 设备健康（acc_item 不进判据，DEF-111-05）
    accs = {}
    for a in ARMS:
        have = [c for k, c in cells.items() if k.startswith(f"{a}|{GAMMA112}|")
                and c.get("acc_item") is not None]
        accs[a] = float(np.mean([c["acc_item"] for c in have])) if have else None

    # δ 的 Q/O/B 三臂原值（供后续功率估算）
    return {"verdict": code,
            "gamma": GAMMA112, "n_seeds": N_SEEDS, "min_neg": MIN_NEG, "h0_p": H0_P,
            "null_eps": NULL_EPS,
            "per_metric": per, "report": rep,
            "acc_item_report_only": accs,
            "reason": "; ".join("%s: %d/%d 负" % (mt, per[mt]["n_neg"], per[mt]["n_usable"])
                                for mt in METS_V)}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["A", "smoke", "formal", "adjudicate", "all"])
    ap.add_argument("--seeds", default="")
    ap.add_argument("--from", dest="src", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(SEEDS)

    if a.stage == "adjudicate":
        src = pathlib.Path(a.src) if a.src else (OUT / "stage_formal.json")
        RES.update(json.loads(src.read_text(encoding="utf-8")))
        crit = verdict()
        RES["crit"] = crit
        dump(RES, "stage_verdict.json")
        print("\n[判决]", crit["verdict"], "·", crit.get("reason"), flush=True)
        for mt in METS_V:
            r = crit["report"][mt]
            print("  %-9s 负号 %d/%d  Δ均值 %s  SD %s  (DIR-111 同 γ: %s ⇒ 差 %s)" % (
                mt, r["n_neg"], r["n_usable"],
                ("%+.4f" % r["delta_mean"]) if r["delta_mean"] is not None else "n/a",
                ("%.4f" % r["delta_sd"]) if r["delta_sd"] is not None else "n/a",
                ("%+.4f" % r["dir111_delta_mean"]),
                ("%+.4f" % r["gap_vs_dir111"]) if r["gap_vs_dir111"] is not None else "n/a"),
                flush=True)
        return 0

    tok, model = D90.model_and_tok()
    facts = D90.build_facts(tok)
    chunks, owner = D90.build_chunks(tok, facts)
    order = list(range(D90.N_FACTS))
    random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:D90.N_TRAIN_KEYS], order[D90.N_TRAIN_KEYS:]
    items = D90.make_eval_items(tok, facts, eval_keys, random.Random(555))
    eval_items_cache["items"] = items
    print(f"事实 {len(facts)}；STEPS={D90.READ_STEPS} LR={D90.READ_LR}；"
          f"eval items={len(items)}；γ={GAMMAS}", flush=True)

    if a.stage in ("A", "all"):
        ok = stage_a(tok, model, facts, chunks, owner, train_keys, eval_keys, items)
        dump(RES, "stage_stageA.json")
        if not ok:
            print("★ Stage A 未通过 ⇒ 正式轮阻断（Round standard 3）", flush=True)
            return 1
    if a.stage == "A":
        return 0

    if a.stage == "smoke":
        plan = [(arm, GAMMAS[0], seeds[0]) for arm in ARMS]
        for arm, g, s in plan:
            RES["cells"][f"{arm}|{g}|{s}"] = run_cell(
                arm, g, s, tok, model, facts, chunks, owner, train_keys, eval_keys)
        dump(RES, "stage_smoke.json")
        print("\n★ smoke 完成（不出判据）", flush=True)
        return 0

    # ★★ 正式轮前置：**必须读回 Stage A 的结论**（否则单跑 `--stage formal` 会绕过硬门）
    sa = OUT / "stage_stageA.json"
    if not sa.exists():
        print("★ 缺 stage_stageA.json ⇒ 不得进正式轮（Round standard 3）", flush=True)
        return 1
    prev = json.loads(sa.read_text(encoding="utf-8"))
    RES["stageA"] = prev.get("stageA", [])
    RES["device_md5"] = prev.get("device_md5", {})
    bad = [x for x in RES["stageA"] if x.get("blocking") and not x.get("ok")]
    if bad:
        print(f"★ Stage A 有 {len(bad)} 项阻断失败 ⇒ 阻断正式轮：{bad[:3]}", flush=True)
        return 1
    # ★★ `DEF-112-03`：继承来的格必须**标注来源**，否则"这份数据是哪种进程史产出的"
    #    从产物本身读不出来（`DIR-111` 的教训：γ=0.001 的 6 格其实来自 Stage A 进程）。
    for k, v in prev.get("cells", {}).items():
        if k not in RES["cells"]:
            v = dict(v)
            v["src_stage"] = "stageA"
            RES["cells"][k] = v
    print(f"Stage A 门通过；继承 {len(prev.get('cells', {}))} 格（已标 src_stage='stageA'）", flush=True)

    plan = [(arm, g, s) for g in GAMMAS for arm in ARMS for s in seeds]
    RES["plan_n"] = len(plan)
    print(f"格计划 {len(plan)} 格", flush=True)
    for arm, g, s in plan:
        k = f"{arm}|{g}|{s}"
        if k in RES["cells"]:
            print(f"  跳过（已有）{k}", flush=True)
            continue
        c = run_cell(arm, g, s, tok, model, facts, chunks, owner, train_keys, eval_keys)
        c["src_stage"] = "formal"
        RES["cells"][k] = c
        dump(RES, "stage_formal.json")
    crit = verdict()
    RES["crit"] = crit
    dump(RES, "stage_formal.json")
    dump(RES, "stage_verdict.json")
    print("\n=== 判据 ===", flush=True)
    print(json.dumps(crit, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())