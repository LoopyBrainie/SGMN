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
# ★ 三个**新** γ（`DIR-109` 的 0.005/0.05 未复用）—— 卡片 §2.4 冻结
GAMMAS = (0.0010, 0.0025, 0.0100)
# ★ 三个**新** seed（与 DIR-109 的 20261028/20261104/7 无交集）—— 卡片 §3 冻结
SEEDS = (20261118, 20261125, 14)

# ---- 内点形状判据的冻结阈值（卡片 §2.2 / §2.3）
CEIL_MAX = 0.95          # 合格 ⟺ δ_Q + δ_O ≤ CEIL_MAX（δ_B 上界 1.0，留 0.05 余量）
MIN_NQ = 3               # 合格裁决点少于 3 个 ⇒ C111-NO-INTERIOR-POINT
RATIO_MAX = 0.90         # median(MAX/SUM) ≤ 0.90 才算 CONFIRMED（否则 WEAK）
ARMS = ("T-NONE", "T-Q", "T-O", "T-BOTH")
WHICH = {"T-NONE": "none", "T-Q": "q", "T-O": "o", "T-BOTH": "both"}
L16 = "16"                        # 判据 B 的口径层（κ 最大层）

RES = {"dir": "DIR-111", "round": "R1", "model": D90.MODEL, "device": D90.DEV,
       "card": "revolve/DIR-111/DIRECTION_rev111_v10.md",
       "ceil_max": CEIL_MAX, "min_nq": MIN_NQ, "ratio_max": RATIO_MAX,
       "derived_from": "dir109/R1_task109.py md5=5c4506b82100bbef05cddbdb664d5111",
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


def stage_a(tok, model, facts, chunks, owner, train_keys, eval_keys, items):
    RES["device_md5"] = {
        "sgmn_module.py": md5(_ROOT / "sgmn_module.py"),
        "dir90/R1_task90.py": md5(_ROOT / "dir90" / "R1_task90.py"),
        "dir104/R1_task104.py": md5(_ROOT / "dir104" / "R1_task104.py"),
        "dir109/R1_task109.py": md5(_ROOT / "dir109" / "R1_task109.py"),
        "dir111/R1_task111.py": md5(_HERE / "R1_task111.py")}
    # ★ 自身 md5 无法自指冻结 ⇒ 只冻结 4 个装置文件的 md5，自身 md5 仅记录（下方说明）
    want = {"sgmn_module.py": "80e334a5a6751a07d9e03162ebf17ff0",
            "dir90/R1_task90.py": "2fb3b65360736e971da14e3d1c4da2a6",
            "dir104/R1_task104.py": "89fd053699c031e94a42db653c7ad22b",
            "dir109/R1_task109.py": "5c4506b82100bbef05cddbdb664d5111"}
    bad_md5 = {k: RES["device_md5"][k] for k, v in want.items() if RES["device_md5"][k] != v}
    rec("A111-0", not bad_md5,
        f"4 个装置文件 md5 与卡片一致（不合 {bad_md5}）；自身 md5={RES['device_md5']['dir111/R1_task111.py'][:16]}",
        bad_md5 or None)

    # ---- A111-1：cells 容器形状（DEF-110-06：DIR-109 的 cells 是 dict 不是 list）
    rec("A111-1", isinstance(RES["cells"], dict),
        f"RES['cells'] 是 {type(RES['cells']).__name__}（必须是 dict；键 'arm|gamma|seed'）")

    # ---- A111-8：§0.2 的上限审计可复算（零新格，先于任何新格）
    au = dir110_ceiling_audit()
    ok8 = (au is not None and au["n"] == 8 and au["n_over_1"] == 8 and au["n_interior"] == 0)
    rec("A111-8", ok8,
        (f"DIR-110 §7 的 {au['n']} 行中 sum>1 的行数={au['n_over_1']}（须 8）、"
         f"内点(sum<={CEIL_MAX})行数={au['n_interior']}（须 0）⇒ COR-111-01 成立")
        if au else "缺 dir110/out/verdict110.json")

    # ---- A111-5：§2.5 算例逐位（用 DIR-110 冻结的 m.cos_mean@L8 g=0.05）
    if au:
        row = next((r for r in au["rows"]
                    if r["key"] == "cos_mean@L8" and abs(r["gamma"] - 0.05) < 1e-12), None)
        if row is None:
            rec("A111-5", False, "§2.5 算例行（m.cos_mean@L8 g=0.05）未在 DIR-110 数据中找到")
        else:
            # 从冻结的 dQ/dO/dB 重算两模型预测与两个误差，再与冻结的 max_err/sum_err 对账
            mp, sp = max(row["dQ"], row["dO"]), row["sum"]
            me_re = abs(row["dB"] - mp)
            se_re = abs(row["dB"] - sp)
            me_ok = abs(row["max_err"] - me_re) < 1e-6      # 算术自洽（不是与硬编码对账）
            se_ok = abs(row["sum_err"] - se_re) < 1e-6
            mag_ok = abs(me_re - 0.002) < 5e-4 and abs(se_re - 0.425) < 5e-4   # 卡片 §2.5 冻结值
            inel = sp > CEIL_MAX                            # 门必须拒它
            rec("A111-5", me_ok and se_ok and mag_ok and inel,
                f"重算 MAXerr={me_re:.4f}(卡片 0.002,{mag_ok}) SUMerr={se_re:.4f}(卡片 0.425) "
                f"ratio={me_re / max(se_re, 1e-12):.4f} dQ+dO={sp:.4f}>CEIL_MAX?{inel} ⇒ 判不合格")

    # ---- A111-5b：三个合成算例 + 穷尽性（纯函数，不碰装置）
    syn = [((0.30, 0.50, 0.55), True, True),    # (a) 合格 ∧ 最弱环赢
           ((0.30, 0.50, 0.78), True, False),   # (b) 合格 ∧ 求和赢
           ((0.60, 0.50, 0.90), False, None)]   # (c) 不合格（门起作用）
    bad_syn = []
    for (dq, do, db), want_elig, want_max in syn:
        mp, sp = max(dq, do), dq + do
        elig = (abs(dq) >= TAU_ARM and abs(do) >= TAU_ARM and sp <= CEIL_MAX)
        maxwin = abs(db - mp) < abs(db - sp)
        if elig != want_elig or (want_max is not None and maxwin != want_max):
            bad_syn.append(((dq, do, db), elig, maxwin))
    rec("A111-5b", not bad_syn,
        f"3 个合成算例全对（合格/门/胜负）；失败 {bad_syn}", bad_syn or None)

    # ---- A111-9：rank_eff 解析核（继承 A109-5c）
    re_cases = []
    for r in (1, 4, 16):
        M = torch.zeros(16, 64)
        M[:r, :r] = torch.eye(r) * 3.0
        re_cases.append(abs(struct_stats([M[i] for i in range(16)])["rank_eff"] - r) < 1e-6)
    rec("A111-9", all(re_cases), f"rank_eff 在 r=1/4/16 上等于 r（{re_cases}）")

    # ---- A111-10：6 个候选由冻结枚举产生、与读数无关；plan_n=36
    plan = [(arm, g, s) for g in GAMMAS for arm in ARMS for s in SEEDS]
    rec("A111-10",
        len(Q_CAND) == 6 and len(plan) == 36
        and Q_CAND == [(o, mt, g) for o in OBJS_V for mt in METS_V for g in GAMMAS],
        f"候选 {len(Q_CAND)} 个（须 6）= 1 量 m × 2 量 × 3 γ；plan_n={len(plan)}（须 36），"
        f"无按结果删点")

    # ---- 跑 6 格：γ=GAMMAS[0]，T-NONE × 3 seed + T-Q/T-O/T-BOTH × seed[0]
    g0 = GAMMAS[0]
    got = {}
    for s in SEEDS:
        c = run_cell("T-NONE", g0, s, tok, model, facts, chunks, owner, train_keys, eval_keys)
        got[("T-NONE", s)] = c
        RES["cells"][f"T-NONE|{g0}|{s}"] = c
    for arm in ("T-Q", "T-O", "T-BOTH"):
        c = run_cell(arm, g0, SEEDS[0], tok, model, facts, chunks, owner, train_keys, eval_keys)
        got[(arm, SEEDS[0])] = c
        RES["cells"][f"{arm}|{g0}|{SEEDS[0]}"] = c

    # ---- A111-2：T-NONE 的 rho=0 ∧ gain=1 精确
    bad = []
    for s in SEEDS:
        for li, v in got[("T-NONE", s)]["scalars"].items():
            if not (v["rho_q"] < 1e-12 and v["rho_o"] < 1e-12 and abs(v["gain"] - 1.0) < 1e-12):
                bad.append((s, li, v["rho_q"], v["rho_o"], v["gain"]))
    rec("A111-2", not bad, f"T-NONE: rho_q=rho_o=0 ∧ gain=1 精确（违例 {bad[:3]}）", bad[:6] or None)

    # ---- A111-3：单侧训练不得动另一侧
    bad = []
    for arm, key in (("T-Q", "rho_o"), ("T-O", "rho_q")):
        for li, v in got[(arm, SEEDS[0])]["scalars"].items():
            if v[key] > 1e-12:
                bad.append((arm, li, key, v[key]))
    rec("A111-3", not bad, f"T-Q 的 rho_o=0 ∧ T-O 的 rho_q=0 精确（违例 {bad[:3]}）", bad[:6] or None)

    # ---- A111-4：评测期零写入（DEF-109-02）
    bad = [(k, v["eval_writes"]) for k, v in got.items() if v["eval_writes"] != 0]
    rec("A111-4", not bad,
        f"评测期写入次数恒为 0（违例 {bad[:4]}）；另核 finite_params="
        f"{all(v['finite_params'] for v in got.values())}", bad[:8] or None)

    # ---- A111-6：c@L8 四臂逐位相同（装置内建对照）
    g = g0
    worst6, det6 = 0.0, {}
    for mt in ("rank_eff", "cos_mean"):
        vs = {arm: got[(arm, SEEDS[0]) if arm != "T-NONE" else ("T-NONE", SEEDS[0])]["state"][L8]["c"][mt]
              for arm in ARMS}
        rng = max(vs.values()) - min(vs.values())
        det6[mt] = round(rng, 12)
        worst6 = max(worst6, rng)
    rec("A111-6", worst6 < 1e-6,
        f"c@L8 跨四臂 max-min = {worst6:.3e} < 1e-6（逐量 {det6}）⇒ L8 在任何注入之前", det6)

    # ---- A111-7：delta_O(q@L8) == 0 是架构恒等式（同 seed 对比）
    bad7, det7 = [], {}
    for mt in ("rank_eff", "cos_mean"):
        xn = got[("T-NONE", SEEDS[0])]["state"][L8]["q"][mt]
        xo = got[("T-O", SEEDS[0])]["state"][L8]["q"][mt]
        span = 1.0 - xn
        d_o = (xo - xn) / span if abs(span) > 1e-9 else float("nan")
        det7[mt] = round(d_o, 12)
        if not (abs(d_o) < 1e-9):
            bad7.append((mt, d_o))
    rec("A111-7", not bad7,
        f"delta_O(q@L8) = {det7} ⇒ |.|<1e-9（架构恒等式 q=Wq c、Wo 在下游）⇒ 排除 q 有依据", det7)

    return all(x["ok"] for x in RES["stageA"] if x["blocking"])


def md5(p):

    import hashlib
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ 判据
def verdict():
    cells = RES["cells"]
    code, det = shape_verdict(cells)
    # L16 报告块（DEF-110-04：只报告，不进判决）
    l16 = [shape_point(cells, g, L16, o, mt) for (o, mt, g) in Q_CAND]
    # c@L8 四臂同一性（每个 γ 各核一次；报告块）
    ident = {}
    for g in GAMMAS:
        worst = 0.0
        for mt in ("rank_eff", "cos_mean"):
            vs = [_X(cells, a, g, L8, "c", mt) for a in ARMS]
            if None in vs:
                continue
            worst = max(worst, max(vs) - min(vs))
        ident[str(g)] = worst
    # 设备健康记录（acc_item 已知会撞 1/3 退化点，不进判据）
    acc = {str(g): {a: _X(cells, a, g, L8, "m", "rank_eff") for a in ARMS} for g in GAMMAS}
    accs = {}
    for g in GAMMAS:
        accs[str(g)] = {}
        for a in ARMS:
            have = _cells_have(cells, a, g, SEEDS)
            if len(have) == len(SEEDS):
                accs[str(g)][a] = float(np.mean([c["acc_item"] for c in have]))
    return {"verdict": code,
            "n_eligible": det["n_eligible"], "n_candidates": det["n_candidates"],
            "min_nq": MIN_NQ, "ceil_max": CEIL_MAX, "tau_arm": TAU_ARM,
            "ratio_max": RATIO_MAX,
            "median_ratio": det.get("median_ratio"),
            "ratios": det.get("ratios"),
            "sum_model_wins": det.get("sum_model_wins"),
            "points_L8": det["points"],
            "points_L16_report_only": l16,
            "c_L8_identity_max_minus_min": ident,
            "acc_item_report_only": accs,
            "ceiling_audit_dir110": dir110_ceiling_audit(),
            "reason": ("qualified=%d/%d; median(MAX/SUM)=%s" % (
                det["n_eligible"], det["n_candidates"],
                ("%.3f" % det["median_ratio"]) if det.get("median_ratio") is not None else "n/a"))}


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
        RES["stageA"] = [g for g in RES["stageA"] if not g["name"].startswith("A111-0")] + \
                        [{"name": "A111-0", "ok": True, "detail": "adjudicate 复用",
                          "blocking": True}]
        crit = verdict()
        RES["crit"] = crit
        dump(RES, "stage_verdict.json")
        print("\n[判决]", crit["verdict"], "·", crit.get("reason"), flush=True)
        print("[判据B]", crit.get("code_B"), flush=True)
        print("[矩阵]", json.dumps(crit.get("rows", {}), ensure_ascii=False), flush=True)
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
    for k, v in prev.get("cells", {}).items():
        RES["cells"].setdefault(k, v)
    print(f"Stage A 门通过；继承 {len(prev.get('cells', {}))} 格", flush=True)

    plan = [(arm, g, s) for g in GAMMAS for arm in ARMS for s in seeds]
    RES["plan_n"] = len(plan)
    print(f"格计划 {len(plan)} 格", flush=True)
    for arm, g, s in plan:
        k = f"{arm}|{g}|{s}"
        if k in RES["cells"]:
            print(f"  跳过（已有）{k}", flush=True)
            continue
        RES["cells"][k] = run_cell(arm, g, s, tok, model, facts, chunks, owner,
                                   train_keys, eval_keys)
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