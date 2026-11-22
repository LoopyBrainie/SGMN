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
# ★★ 本卡（`DIR-113` rev 1.0）**只测一个 γ** —— 同 `DIR-112`（`COR-112-01`：γ 轴信噪比不足）。
GAMMA113 = 0.0010
GAMMAS = (GAMMA113,)
# ★ 12 个**新** seed（与 DIR-111 的 20261118/20261125/14、DIR-112 的 8 个及全仓库历史 seed 零碰撞）
#   —— 卡片 §2.2 冻结；前置检查五逐 seed `grep -rlw` 机械核过（12/12 命中=0）。
SEEDS = (20270401, 20270415, 20270429, 20270513, 20270527, 20270610,
         20270624, 20270708, 20270722, 20270805, 20270819, 20270902)

# ---- 本卡判据的冻结阈值（卡片 §2.3）----
MIN_NEG_A = 9            # 结构面：`#{Δ<0} ≥ 9/12`（卡片 §2.3）
ALPHA_B = 0.05           # 任务面：簇级单样本 t **单侧** p 阈值
NULL_EPS = 1e-9          # `1 − X_NONE ≤ NULL_EPS` ⇒ 该 seed 不可用
N_SEEDS = 12
N_ITEM = 48              # `acc_item` 的项数（前置检查一：48 项共享同一 W ⇒ **非独立**）
# ★ `DEF-112-03` 继承：每格标来源阶段（`--stage formal` 会 setdefault 继承 Stage A 的格，
#   不标来源则"正式轮"数据混两种进程史而读不出来）。由 main() 在跑前设置。
STAGE_TAG = ["unknown"]
# 以下三项**继承自 DIR-111 但本卡不作判据**，保留仅为让 `shape_*`/`dir110_ceiling_audit` 可复用
CEIL_MAX = 0.95
MIN_NQ = 3
RATIO_MAX = 0.90
ARMS = ("T-NONE", "T-Q", "T-O", "T-BOTH")
WHICH = {"T-NONE": "none", "T-Q": "q", "T-O": "o", "T-BOTH": "both"}
L16 = "16"                        # 判据 B 的口径层（κ 最大层）

RES = {"dir": "DIR-113", "round": "R1", "model": D90.MODEL, "device": D90.DEV,
       "card": "revolve/DIR-113/DIRECTION_rev113_v10.md",
       "min_neg_a": MIN_NEG_A, "alpha_b": ALPHA_B, "null_eps": NULL_EPS, "n_seeds": N_SEEDS,
       "derived_from": "dir112/R1_task112.py",
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
           "src_stage": STAGE_TAG[0],
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
# ★★★ 只取 `rank_eff` —— 前置检查二：`cos_mean` 是它的**镜像**
#   （`T-BOTH` 内 ρ=−1.0000、全 32 格 −0.8724，且代数必然：有效秩 vs 行归一化 Gram 非对角均值）。
#   两个镜像量相加**不产生第二个证据**（`DEF-112-04`）。`cos_mean` 只作报告块。
METS_V = ("rank_eff",)
METS_REPORT = ("rank_eff", "cos_mean")   # 报告块用（不进判据）
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
    g = GAMMA113 if gamma is None else gamma
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


def _t_sf(t, df):
    """单侧 t 生存函数 `P(T > t)`。优先 scipy；缺则用正态近似并**显式标记**。"""
    try:
        from scipy import stats as _st
        return float(_st.t.sf(t, df)), "scipy"
    except Exception:
        import math as _m
        return 0.5 * _m.erfc(t / _m.sqrt(2.0)), "normal-approx(缺 scipy)"


from math import comb as _comb


def cluster_stats(cells, seeds=None):
    """★★★ 卡片 §1.1 的**簇级**任务面统计（**独立单位 = 一个训练好的格，不是 item**）。

    `d(s) = acc_item_{T-BOTH}(s) − max(acc_item_{T-Q}(s), acc_item_{T-O}(s))`
    ⇒ 每 seed **一个**观测，`n = 可用 seed 数`。簇级单样本 t 检验，单侧 `H1: mean d < 0`。

    ★★★ 为什么不按 item 池化（前置检查一）：48 个 item **共享同一个训练好的 `W`**，
       正确性通过 `W` 相关 ⇒ 把 `(seed, item)` 当独立会把 n 放大 8 倍。
       实测 `DIR-112`：item 池化给单侧 `p = 5.98e-04`，而簇级只给 `p ≈ 0.044` ⇒ **虚高约 30 倍**。
       这正是 `DEF-112-04` 的同一族错误（同一个检验数两遍）。**item 级只作报告块，永不进判据。**
    """
    import math as _m
    ss = list(SEEDS) if seeds is None else list(seeds)
    rows = []
    for sd in ss:
        acc = {}
        for arm in ("T-Q", "T-O", "T-BOTH"):
            c = cells.get(f"{arm}|{GAMMA113}|{sd}")
            acc[arm] = None if c is None else c.get("acc_item")
        if any(acc[a] is None for a in acc):
            rows.append({"seed": sd, "ok": False, "d": None, "acc": acc})
            continue
        mx = max(acc["T-Q"], acc["T-O"])
        rows.append({"seed": sd, "ok": True, "d": acc["T-BOTH"] - mx, "acc": acc,
                     "max_single": mx})
    usable = [r for r in rows if r["ok"]]
    n = len(usable)
    out = {"rows": rows, "n_usable": n, "n_expected": len(ss)}
    if n < 2:
        out.update({"mean_d": None, "sd_d": None, "se_d": None,
                    "t": None, "df": None, "p_one_sided": None, "src": None})
        return out
    ds = [r["d"] for r in usable]
    mean = sum(ds) / n
    var = sum((x - mean) ** 2 for x in ds) / (n - 1)
    sd = _m.sqrt(var)
    se = sd / _m.sqrt(n)
    df = n - 1
    # ★★ `DEF-113-01`：`SE == 0`（所有 seed 的 d 完全相同）时 t 检验**未定义**。
    #    **不得**静默地让 B 失败（那会把"无法计算"混成"未成立"，`loop.md` 明确禁止）。
    #    改：退回**精确符号检验**（单侧二项，`H1: 负号占多数`），并在 `src` 里写明走了哪条路。
    n_neg = sum(1 for x in ds if x < 0)
    n_pos = sum(1 for x in ds if x > 0)
    n_zero = sum(1 for x in ds if x == 0)
    if se > 0:
        t = mean / se
        p = _t_sf(-t, df)[0]          # 单侧 H1: mean<0 ⇒ p = P(T <= t) = sf(-t)
        src = _t_sf(-1.0, 1)[1]
    else:
        t = None
        # 精确符号检验（只数非零）。全零 ⇒ p=1.0（无信息）。
        nz = n_neg + n_pos
        if nz == 0:
            p, src = 1.0, "sign-exact(se=0, 全零 ⇒ 无信息)"
        else:
            p = sum(_comb(nz, i) for i in range(n_neg, nz + 1)) / 2 ** nz
            src = "sign-exact(se=0 ⇒ t 未定义，退回符号检验)"
    out.update({"mean_d": mean, "sd_d": sd, "se_d": se, "t": t, "df": df,
                "p_one_sided": p, "src": src,
                "n_neg": n_neg, "n_pos": n_pos, "n_zero": n_zero})
    return out


def itempool_report(cells, seeds=None):
    """★ 报告块（**永不进判据**）：item 级配对不一致对计数。

    明确标注**反保守**：48 项共享同一 `W` ⇒ 独立单位是格不是 item。
    记录它是为了(a)复现 `DIR-112` 的对照、(b)防止后续 cycle 误引其 p 值。
    """
    ss = list(SEEDS) if seeds is None else list(seeds)
    win = lose = tot = 0
    for sd in ss:
        ok = {}
        for arm in ("T-Q", "T-O", "T-BOTH"):
            c = cells.get(f"{arm}|{GAMMA113}|{sd}")
            ok[arm] = None if c is None else c.get("item_ok")
        if any(ok[a] is None for a in ok):
            continue
        q, o, b = ok["T-Q"], ok["T-O"], ok["T-BOTH"]
        if not (len(q) == len(o) == len(b)):
            continue
        for i in range(len(b)):
            mx = max(q[i], o[i])
            tot += 1
            if b[i] == 1 and mx == 0:
                win += 1
            elif b[i] == 0 and mx == 1:
                lose += 1
    return {"win": win, "lose": lose, "n_pairs": tot,
            "discordant": win + lose,
            "discordant_rate": (None if tot == 0 else (win + lose) / tot),
            "NOTE": "★ 反保守：48 项共享同一 W，独立单位是格不是 item ⇒ 其 p 值**不得**作判据或引用为证据"}


def joint_verdict(cells, seeds=None):
    """卡片 §1.4 的冻结判决（**穷尽且互斥**，按顺序求值）。纯函数。

    A 结构面：`#{Δ<0} ≥ 9/12`（只 `rank_eff`）
    B 任务面：`mean d < 0` **且** 单侧 `p ≤ 0.05`（簇级 t，df=11）
    """
    ss = list(SEEDS) if seeds is None else list(seeds)
    mt = METS_V[0]                                  # 只 rank_eff
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
    struct = {"metric": mt, "rows": rows, "n_usable": len(usable),
              "n_neg": n_neg, "n_expected": len(ss), "min_neg": MIN_NEG_A}
    task = cluster_stats(cells, ss)

    # 可用 seed 判定：结构面必须 12 个都能算 δ 且 12 格 acc 齐
    insufficient = (struct["n_usable"] < len(ss)) or (task["n_usable"] < len(ss))
    A = (not insufficient) and (n_neg >= MIN_NEG_A)
    B = (not insufficient) and (task["mean_d"] is not None
                               and task["mean_d"] < 0
                               and task["p_one_sided"] is not None
                               and task["p_one_sided"] <= ALPHA_B)
    if insufficient:
        code = "C113-INSUFFICIENT"
    elif A and B:
        code = "C113-ANTAGONISM-COEXISTS-WITH-TASK-HARM"
    elif A and not B:
        code = "C113-NO-TASK-HARM-DETECTED"
    elif (not A) and B:
        code = "C113-TASK-HARM-WITHOUT-STRUCT-REPLICATION"
    else:
        code = "C113-FALSIFIED"
    return code, {"struct": struct, "task": task, "A": bool(A), "B": bool(B)}


def delta_selftest():
    """`A113-9` 用的 `Δ` 恒等式自检（纯合成数，不碰模型）。"""
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

    # ---------- A113-0 装置 4 文件 md5 ----------
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
    rec("A113-0", not dev_bad, f"4 个装置文件 md5（不合 {dev_bad}）；自身 md5={md5(pathlib.Path(__file__))}")

    # ---------- A113-1 数据结构 ----------
    rec("A113-1", isinstance(RES["cells"], dict),
        f"RES['cells'] 是 dict（必须是 dict；键 'arm|gamma|seed'）")

    # ---------- A113-11 冻结枚举 ----------
    enum = {"GAMMA113": (GAMMA113, 0.0010), "len(SEEDS)": (len(SEEDS), 12),
            "MIN_NEG_A": (MIN_NEG_A, 9), "ALPHA_B": (ALPHA_B, 0.05),
            "NULL_EPS": (NULL_EPS, 1e-9), "N_ITEM": (N_ITEM, 48),
            "len(GAMMAS)": (len(GAMMAS), 1), "len(METS_V)": (len(METS_V), 1)}
    enum_bad = {k: v for k, v in enum.items() if v[0] != v[1]}
    rec("A113-11", not enum_bad, f"冻结枚举 {enum} ⇒ 不合 {enum_bad}")

    # ---------- A113-2 未训练臂（用 1 个 seed 实跑） ----------
    s0 = SEEDS[0]
    c_none = run_cell("T-NONE", GAMMA113, s0, tok, model, facts, chunks, owner,
                      train_keys, eval_keys)
    RES["cells"][f"T-NONE|{GAMMA113}|{s0}"] = c_none
    sc = c_none["scalars"][L16]
    rec("A113-2", abs(sc["rho_q"]) < 1e-9 and abs(sc["rho_o"]) < 1e-9 and abs(sc["gain"] - 1.0) < 1e-9,
        f"T-NONE: rho_q={sc['rho_q']:.3e} rho_o={sc['rho_o']:.3e} gain={sc['gain']:.12f}（须 0/0/1）")

    # ---------- A113-3 ★ T-NONE 的 state 逐位 γ-不变 ----------
    c_none2 = run_cell("T-NONE", 0.0100, s0, tok, model, facts, chunks, owner,
                       train_keys, eval_keys)
    diffs = {mt: c_none["state"][L8]["m"][mt] - c_none2["state"][L8]["m"][mt]
             for mt in METS_V}
    rec("A113-3", all(abs(v) == 0.0 for v in diffs.values()),
        f"T-NONE state 在 γ=0.0010 vs 0.0100 逐位差 {diffs}（须全 0.0）")

    # ---------- A113-2b 单通路隔离 ----------
    c_q = run_cell("T-Q", GAMMA113, s0, tok, model, facts, chunks, owner, train_keys, eval_keys)
    c_o = run_cell("T-O", GAMMA113, s0, tok, model, facts, chunks, owner, train_keys, eval_keys)
    # ★★ `A113-5` 的教训：判据要**端到端算一次**才叫"可计算"。
    #    `DIR-112` 的 Stage A 只跑 3 格、**没跑 T-BOTH** ⇒ 它的"判据可计算"从未在门里被真正验证。
    #    本卡在门里把 `s0` 的**四臂跑齐**，使 `δ`/`d` 真的算出来。
    c_b = run_cell("T-BOTH", GAMMA113, s0, tok, model, facts, chunks, owner, train_keys, eval_keys)
    for k, v in (("T-Q", c_q), ("T-O", c_o), ("T-BOTH", c_b)):
        RES["cells"][f"{k}|{GAMMA113}|{s0}"] = v
    sq, so = c_q["scalars"][L16], c_o["scalars"][L16]
    rec("A113-2b", abs(sq["rho_o"]) < 1e-9 and abs(so["rho_q"]) < 1e-9,
        f"T-Q rho_o={sq['rho_o']:.3e} / T-O rho_q={so['rho_q']:.3e}（须各为 0）")

    # ---------- A113-2c 评测期零写入 + 参数有限 ----------
    allc = list(RES["cells"].values())
    bad5 = [k for k, c in RES["cells"].items()
            if c.get("eval_writes", 0) != 0 or not c.get("finite_params", False)]
    rec("A113-2c", not bad5, f"格数 {len(allc)}；eval_writes!=0 或 finite_params=False 的格 {bad5}")

    # ---------- A113-4 ★ item_ok 长度 48 且 acc_item == mean(item_ok) ----------
    bad_item = []
    for k, c in RES["cells"].items():
        io_ = c.get("item_ok")
        if io_ is None or len(io_) != N_ITEM:
            bad_item.append((k, "len!=48"))
            continue
        ai = c.get("acc_item")
        if ai is None or abs(float(ai) - sum(io_) / len(io_)) > 1e-12:
            bad_item.append((k, "acc_item!=mean"))
    rec("A113-4", not bad_item, f"item_ok 长度恒 48 且 acc_item==mean(item_ok)；不合 {bad_item}")

    # ---------- A113-5 ★ δ 与 d 可计算（分母不退化、四臂 acc 齐） ----------
    bad_delta = []
    s0 = SEEDS[0]
    # (a) s0 的四臂必须齐（否则判据不可能端到端算一次）
    miss = [a for a in ARMS if f"{a}|{GAMMA113}|{s0}" not in RES["cells"]]
    if miss:
        bad_delta.append((s0, f"四臂不全，缺 {miss}"))
    # (b) δ 必须真的算出来
    got = delta_of(RES["cells"], s0, METS_V[0])
    if got is None:
        bad_delta.append((s0, "δ 不可算"))
    # (c) 分母不得退化
    cn = RES["cells"].get(f"T-NONE|{GAMMA113}|{s0}")
    if cn is not None:
        xn = cn["state"][L8]["m"]["rank_eff"]
        if abs(1.0 - xn) <= NULL_EPS:
            bad_delta.append((s0, f"1-rank_eff_NONE={1.0 - xn:.3e} <= NULL_EPS"))
    # (d) d 必须真的算出来（单 seed 时 n=1 ⇒ 只有 d 无 t，这是合法的）
    tl = cluster_stats(RES["cells"], [s0])
    if tl["n_usable"] < 1:
        bad_delta.append((s0, "d 不可算（acc 缺）"))
    if tl["n_usable"] == 1 and tl["rows"][0]["d"] is None:
        bad_delta.append((s0, "d 为 None"))
    dv = None if tl["n_usable"] < 1 else tl["rows"][0]["d"]
    rec("A113-5", not bad_delta,
        f"δ/d **端到端算一次**（s0={s0}）：δ={None if got is None else round(got[0], 6)}，"
        f"d={None if dv is None else round(dv, 6)}；不合 {bad_delta}")

    # ---------- A113-6 ★★ 簇级统计自洽（纯数学，不碰模型） ----------
    bad6 = []
    # (a) d(s) 恒为 1/48 的整数倍
    for r in tl["rows"]:
        if r["ok"]:
            step = r["d"] * N_ITEM
            if abs(step - round(step)) > 1e-9:
                bad6.append(("d 非 1/48 整数倍", r["seed"], r["d"]))
    # (b) mean_d 与逐 seed 算术平均一致（合成数据）
    # 用合成 cells 验 cluster_stats 的 t/p 数值正确性
    def mkacc(seed, aq, ao, ab):
        cc = {}
        for arm, v in (("T-Q", aq), ("T-O", ao), ("T-BOTH", ab)):
            cc[f"{arm}|{GAMMA113}|{seed}"] = {"acc_item": v}
        return cc
    syn = {}
    vals = [(0.10, 0.20, 0.05), (0.10, 0.20, 0.08), (0.10, 0.20, 0.12),
            (0.10, 0.20, 0.06), (0.10, 0.20, 0.09), (0.10, 0.20, 0.11)]
    syn_seeds = list(range(9001, 9001 + len(vals)))
    for sd, (aq, ao, ab) in zip(syn_seeds, vals):
        syn.update(mkacc(sd, aq, ao, ab))
    st = cluster_stats(syn, syn_seeds)
    d_manual = [ab - max(aq, ao) for aq, ao, ab in vals]
    mean_m = sum(d_manual) / len(d_manual)
    if abs(st["mean_d"] - mean_m) > 1e-12:
        bad6.append(("mean_d 与手算不一致", st["mean_d"], mean_m))
    if st["df"] != len(vals) - 1:
        bad6.append(("df 错", st["df"], len(vals) - 1))
    # t 手算：t = mean / (sd/sqrt(n))
    import math as _m
    sd_m = _m.sqrt(sum((x - mean_m) ** 2 for x in d_manual) / (len(d_manual) - 1))
    t_m = mean_m / (sd_m / _m.sqrt(len(d_manual)))
    if abs(st["t"] - t_m) > 1e-9:
        bad6.append(("t 与手算不一致", st["t"], t_m))
    # p 单调性：mean 越负，p 越小
    p_neg = st["p_one_sided"]
    syn2 = {}
    vals2 = [(aq, ao, ab - 0.05) for aq, ao, ab in vals]      # 把 BOTH 再拉低 ⇒ 更负
    for sd, (aq, ao, ab) in zip(syn_seeds, [(a, o, b) for a, o, b in vals2]):
        syn2.update(mkacc(sd, aq, ao, ab))
    p_more = cluster_stats(syn2, syn_seeds)["p_one_sided"]
    if not (p_more < p_neg):
        bad6.append(("p 未随效应变负而下降", p_more, p_neg))
    rec("A113-6", not bad6, f"簇级统计自洽（d 粒度 1/48、mean/t/p 与手算一致、p 单调）；不合 {bad6}")

    # ---------- A113-7 src_stage 字段 ----------
    miss7 = [k for k, c in RES["cells"].items() if c.get("src_stage") not in ("stageA", "formal")]
    rec("A113-7", not miss7, f"每格 src_stage ∈ {{stageA,formal}}（DEF-112-03 继承）；缺 {miss7}")

    # ---------- A113-8 报告块：item 池化（★ 非阻断，只记录） ----------
    ip = itempool_report(RES["cells"])
    rec("A113-8", True, f"item 池化报告块（★ 反保守、**不进判据**）：{ip}", blocking=False)

    # ---------- A113-9 ★★ 判决码 5 分支纯逻辑自测 + Δ/簇级恒等式 ----------
    def mk(per_seed, acc=None):
        """合成 cells：X_NONE=0.5，X_arm 由目标 δ 反解 ⇒ δ 精确可控；acc 单独给。"""
        cc = {}
        xn = 0.5
        for sd, (dq, do, db) in per_seed.items():
            cc[f"T-NONE|{GAMMA113}|{sd}"] = {"state": {L8: {"m": {"rank_eff": xn}}},
                                             "acc_item": 0.29, "item_ok": [0] * N_ITEM}
            for arm, val in (("T-Q", dq), ("T-O", do), ("T-BOTH", db)):
                cc[f"{arm}|{GAMMA113}|{sd}"] = {
                    "state": {L8: {"m": {"rank_eff": xn + val * (1 - xn)}}},
                    "acc_item": (acc or {}).get(sd, {}).get(arm, 0.29),
                    "item_ok": [0] * N_ITEM}
        return cc
    ss12 = list(range(9101, 9113))
    negA = {sd: (0.2, 0.3, 0.05) for sd in ss12}        # Δ<0 全 12
    posA = {sd: (0.2, 0.05, 0.5) for sd in ss12}        # Δ>0 全 12
    # 任务面：让 BOTH 显著低于 max(单臂)
    accHarm = {sd: {"T-Q": 0.20, "T-O": 0.18, "T-BOTH": 0.02} for sd in ss12}
    accSame = {sd: {"T-Q": 0.20, "T-O": 0.18, "T-BOTH": 0.25} for sd in ss12}
    # 「结构 9/12 负」：前 9 负、后 3 正
    mixA = dict(negA); 
    for sd in ss12[9:]: mixA[sd] = (0.2, 0.05, 0.5)
    mixA8 = dict(negA)
    for sd in ss12[8:]: mixA8[sd] = (0.2, 0.05, 0.5)     # 只 8 负 ⇒ A 不成立
    cases = [
        ("A且B→COEXISTS", mk(negA, accHarm), ss12, "C113-ANTAGONISM-COEXISTS-WITH-TASK-HARM"),
        ("A成立B不成立→NO-TASK-HARM", mk(negA, accSame), ss12, "C113-NO-TASK-HARM-DETECTED"),
        ("A不成立B成立→TASK-HARM-WITHOUT", mk(mixA8, accHarm), ss12, "C113-TASK-HARM-WITHOUT-STRUCT-REPLICATION"),
        ("都不成立→FALSIFIED", mk(mixA8, accSame), ss12, "C113-FALSIFIED"),
        ("9/12 恰达标→A成立", mk(mixA, accHarm), ss12, "C113-ANTAGONISM-COEXISTS-WITH-TASK-HARM"),
    ]
    # 缺格 ⇒ INSUFFICIENT
    cm = mk(negA, accHarm); del cm[f"T-BOTH|{GAMMA113}|{ss12[0]}"]
    cases.append(("缺格→INSUFFICIENT", cm, ss12, "C113-INSUFFICIENT"))
    # Δ=0 不算负（12 个全 Δ=0 ⇒ 0 负 ⇒ A 不成立）
    eqA = {sd: (0.3, 0.3, 0.3) for sd in ss12}
    cases.append(("Δ=0不算负", mk(eqA, accHarm), ss12, "C113-TASK-HARM-WITHOUT-STRUCT-REPLICATION"))
    bad9 = [(n, joint_verdict(c, s_)[0], w) for n, c, s_, w in cases
            if joint_verdict(c, s_)[0] != w]
    # Δ 恒等式（沿用 DIR-112 的 6 个算例 + 全负算例）
    for (dq, do, db), want in [((0.2, 0.2, 0.1), -0.1), ((0.2, 0.2, 0.3), +0.1),
                               ((0.1, 0.5, 0.4), -0.1), ((0.5, 0.1, 0.4), -0.1),
                               ((0.3, 0.7, 0.7), +0.0), ((-0.1, -0.2, -0.3), -0.2)]:
        if abs((db - max(dq, do)) - want) > 1e-12:
            bad9.append(("Δ 恒等式", (dq, do, db), want))
    rec("A113-9", not bad9, f"判决码 {len(cases)} 分支 + Δ 恒等式 6 例；失败 {bad9}")

    RES["plan_n"] = N_SEEDS * len(ARMS)          # ★ 32（T-NONE 每 seed 一格，跨 γ 共用）
    return all(g["ok"] for g in RES["stageA"] if g["blocking"])


def md5(p):

    import hashlib
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ 判据
def verdict():
    """卡片 §1.4 的判决 + 报告块（report-only）。"""
    cells = RES["cells"]
    code, per = joint_verdict(cells)
    st, tk = per["struct"], per["task"]

    # 报告块：cos_mean 的同一判据（★ 只报告，**不进判据** —— 它是 rank_eff 的镜像）
    rep_cos = None
    if "cos_mean" in METS_REPORT:
        rows = []
        for sd in SEEDS:
            cn = cells.get(f"T-NONE|{GAMMA113}|{sd}")
            if cn is None:
                continue
            xn = cn["state"][L8]["m"]["cos_mean"]
            if abs(1.0 - xn) <= NULL_EPS:
                continue
            d = {}
            for arm, tag in (("T-Q", "Q"), ("T-O", "O"), ("T-BOTH", "B")):
                c = cells.get(f"{arm}|{GAMMA113}|{sd}")
                if c is None:
                    d = None
                    break
                d[tag] = (c["state"][L8]["m"]["cos_mean"] - xn) / (1.0 - xn)
            if d:
                rows.append(d["B"] - max(d["Q"], d["O"]))
        rep_cos = {"n": len(rows), "n_neg": sum(1 for x in rows if x < 0),
                   "NOTE": "★ cos_mean 是 rank_eff 的镜像（ρ=−1.0000）⇒ 只报告、不进判据、不与 A 相加"}

    # 设备健康（acc_item 绝对值**不进判据**，DEF-111-05）
    accs = {}
    for a in ARMS:
        have = [c for k, c in cells.items() if k.startswith(f"{a}|{GAMMA113}|")
                and c.get("acc_item") is not None]
        accs[a] = float(np.mean([c["acc_item"] for c in have])) if have else None

    mde = None
    if tk["sd_d"] is not None and tk["n_usable"] >= 2:
        # MDE = (t_crit(0.05, df) + z(0.80)) * SD / sqrt(n)  —— 预注册于卡片 §1.3 / 前置检查四
        from scipy import stats as _st
        tc = float(_st.t.ppf(1 - ALPHA_B, tk["df"]))
        mde = (tc + 0.8416) * tk["sd_d"] / (tk["n_usable"] ** 0.5)

    return {"verdict": code,
            "gamma": GAMMA113, "n_seeds": N_SEEDS, "min_neg_a": MIN_NEG_A, "alpha_b": ALPHA_B,
            "null_eps": NULL_EPS, "n_item": N_ITEM,
            "A_struct": per["A"], "B_task": per["B"],
            "struct": {k: v for k, v in st.items()},
            "task": {k: v for k, v in tk.items()},
            "itempool_report_only": itempool_report(cells),
            "cos_mean_report_only": rep_cos,
            "acc_item_report_only": accs,
            "mde_delta": mde,
            "reason": ("A: %d/%d 负（需≥%d）· B: mean d=%s 单侧 p=%s (df=%s) ⇒ A=%s B=%s" % (
                st["n_neg"], st["n_usable"], MIN_NEG_A,
                ("%+.5f" % tk["mean_d"]) if tk["mean_d"] is not None else "n/a",
                ("%.5f" % tk["p_one_sided"]) if tk["p_one_sided"] is not None else "n/a",
                tk["df"], per["A"], per["B"]))}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["A", "smoke", "formal", "cell", "merge", "adjudicate", "all"])
    ap.add_argument("--seeds", default="")
    ap.add_argument("--arm", default="")          # --stage cell 用
    ap.add_argument("--out", default="")          # --stage cell 用
    ap.add_argument("--from", dest="src", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(SEEDS)

    # ★ `DEF-112-03`：每格标来源阶段
    STAGE_TAG[0] = {"A": "stageA", "all": "stageA", "smoke": "stageA",
                    "formal": "formal", "cell": "formal"}.get(a.stage, "unknown")

    if a.stage == "adjudicate":
        src = pathlib.Path(a.src) if a.src else (OUT / "stage_formal.json")
        RES.update(json.loads(src.read_text(encoding="utf-8")))
        crit = verdict()
        RES["crit"] = crit
        dump(RES, "stage_verdict.json")
        print("\n[判决]", crit["verdict"], flush=True)
        print("  ·", crit.get("reason"), flush=True)
        st, tk = crit["struct"], crit["task"]
        print("  A 结构面(rank_eff): %d/%d 负（需 ≥%d）⇒ A=%s" % (
            st["n_neg"], st["n_usable"], MIN_NEG_A, crit["A_struct"]), flush=True)
        print("  B 任务面(簇级): mean d=%s  SD=%s  SE=%s  t=%s  df=%s  单侧 p=%s ⇒ B=%s" % (
            ("%+.5f" % tk["mean_d"]) if tk["mean_d"] is not None else "n/a",
            ("%.5f" % tk["sd_d"]) if tk["sd_d"] is not None else "n/a",
            ("%.5f" % tk["se_d"]) if tk["se_d"] is not None else "n/a",
            ("%+.4f" % tk["t"]) if tk["t"] is not None else "n/a",
            tk["df"],
            ("%.5f" % tk["p_one_sided"]) if tk["p_one_sided"] is not None else "n/a",
            crit["B_task"]), flush=True)
        print("  MDE(|Δ| 可检出下限) = %s" % (
            ("%.5f" % crit["mde_delta"]) if crit["mde_delta"] is not None else "n/a"), flush=True)
        print("  逐 seed d:", flush=True)
        for r in tk["rows"]:
            print("     %-9s d=%s  acc(BOTH)=%s max_single=%s" % (
                r["seed"],
                ("%+.5f" % r["d"]) if r.get("d") is not None else "n/a",
                ("%.4f" % r["acc"]["T-BOTH"]) if r["acc"]["T-BOTH"] is not None else "n/a",
                ("%.4f" % r["max_single"]) if r.get("max_single") is not None else "n/a"),
                flush=True)
        ip = crit["itempool_report_only"]
        print("  ★ 报告块（反保守、不进判据）item 池化: 胜=%d 负=%d / %d 对（不一致率 %s）" % (
            ip["win"], ip["lose"], ip["n_pairs"],
            ("%.4f" % ip["discordant_rate"]) if ip["discordant_rate"] is not None else "n/a"),
            flush=True)
        cc = crit["cos_mean_report_only"]
        if cc:
            print("  ★ 报告块（镜像、不进判据）cos_mean: %d/%d 负" % (cc["n_neg"], cc["n"]), flush=True)
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

    # ================= `--stage cell`：**单格单进程**（卡片 §4「一格一进程」） =================
    # 为什么：`DEF-105-02` 指同进程内多格可能因果历史污染。本装置实测无影响（2880 字段不等=0），
    # 但 `DEF-112-02` 记了协议不一致的隐患 ⇒ 本卡按卡片文字**逐格起独立进程**，不留例外。
    if a.stage == "cell":
        if not a.arm or not a.out:
            print("★ --stage cell 需要 --arm 与 --out", flush=True)
            return 2
        sd = seeds[0]
        # ★ 单格也过同一道 Stage A 硬门（读回结论），不得绕过
        sa0 = OUT / "stage_stageA.json"
        if not sa0.exists():
            print("★ 缺 stage_stageA.json ⇒ 不得进正式轮", flush=True)
            return 1
        pv = json.loads(sa0.read_text(encoding="utf-8"))
        bd = [x for x in pv.get("stageA", []) if x.get("blocking") and not x.get("ok")]
        if bd:
            print(f"★ Stage A 有 {len(bd)} 项阻断失败 ⇒ 阻断：{bd[:3]}", flush=True)
            return 1
        c = run_cell(a.arm, GAMMA113, sd, tok, model, facts, chunks, owner,
                     train_keys, eval_keys)
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(
            json.dumps({"arm": a.arm, "gamma": GAMMA113, "seed": sd,
                        "cell": c, "script_md5": md5(pathlib.Path(__file__))},
                       ensure_ascii=False), encoding="utf-8")
        print(f"  ✓ 落盘 {a.out}", flush=True)
        return 0

    # ================= `--stage merge`：把 iso 单格合并成 stage_formal.json =================
    if a.stage == "merge":
        isod = OUT / "iso"
        files = sorted(isod.glob("*.json"))
        if not files:
            print("★ out/iso 下没有单格结果", flush=True)
            return 1
        RES["cells"] = {}
        for f in files:
            o = json.loads(f.read_text(encoding="utf-8"))
            RES["cells"][f"{o['arm']}|{o['gamma']}|{o['seed']}"] = o["cell"]
        RES["plan_n"] = N_SEEDS * len(ARMS)
        RES["src_stage"] = "one-process-per-cell (run_formal113.sh)"
        dump(RES, "stage_formal.json")
        cnt = {}
        for c in RES["cells"].values():
            cnt[c.get("src_stage")] = cnt.get(c.get("src_stage"), 0) + 1
        print(f"✓ 合并 {len(files)} 个单格 ⇒ {len(RES['cells'])} 格；src_stage={cnt}", flush=True)
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