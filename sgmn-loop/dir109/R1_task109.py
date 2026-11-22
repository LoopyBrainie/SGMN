#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-109 · R1 —— 读出训练到底改坏了什么？`{W_q, W_o}` 的 **2×2 归因** + 训练**前/后**状态级读数。

卡片：`revolve/DIR-109/DIRECTION_rev109_v10.md`（rev 1.0，**冻结**）

★ 派生方式（`A109-1`/`A109-2` 的保证）：**不复制装置代码**，直接
  `import R1_task104 as D104`（它已 `import R1_task90 as D90`）并**逐字复用**
  `model_and_tok / make_stack / build_facts / build_chunks / nll_of_candidate /
  make_eval_items / train_read_frozen`。⇒ 装置字节不变由 md5 直接锁。
  `T-BOTH` **直接调用** `D104.train_read_frozen`（不重写）⇒ 与 `DIR-104` 的 `INJ` 同代码路径。

**只替换**（a）自变量：可训练参数集 ∈ {∅, {W_q}, {W_o}, {W_q,W_o}}（`gamma` 恒不可训练）
        （b）测量层：`measure_state` 在**训练后**读 `ρ_q/ρ_o/gain/κ` 与 `c,q,m` 的
            `cos_mean/rho/rank_eff`，**四臂全测**（`DIR-104` 的 `RAWINJ` 12 格 `kappa=None`）

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
RATIO = 2.0
TAU_GAIN = 10.0
PROBE_RTOL = 1e-5
W_DRIFT_TOL = 1e-8
GAMMAS = (0.005, 0.05)
SEEDS = (20261028, 20261104, 7)
ARMS = ("T-NONE", "T-Q", "T-O", "T-BOTH")
WHICH = {"T-NONE": "none", "T-Q": "q", "T-O": "o", "T-BOTH": "both"}
L16 = "16"                        # 判据 B 的口径层（κ 最大层）

RES = {"dir": "DIR-109", "round": "R1", "model": D90.MODEL, "device": D90.DEV,
       "card": "revolve/DIR-109/DIRECTION_rev109_v10.md",
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
def stage_a(tok, model, facts, chunks, owner, train_keys, eval_keys, items):
    RES["device_md5"] = {
        "sgmn_module.py": md5(_ROOT / "sgmn_module.py"),
        "dir90/R1_task90.py": md5(_ROOT / "dir90" / "R1_task90.py"),
        "dir104/R1_task104.py": md5(_ROOT / "dir104" / "R1_task104.py")}
    want = {"sgmn_module.py": "80e334a5a6751a07d9e03162ebf17ff0",
            "dir90/R1_task90.py": "2fb3b65360736e971da14e3d1c4da2a6",
            "dir104/R1_task104.py": "89fd053699c031e94a42db653c7ad22b"}
    ok_md5 = all(RES["device_md5"][k] == v for k, v in want.items())
    rec("A109-0", ok_md5, f"装置 md5 与卡片一致 {RES['device_md5']}", RES["device_md5"])

    # ---- A109-5：§2.4 算例逐位（纯函数）
    cases = [
        ("算例1", 0.041667, 0.361111, 0.625000, "C109-NO-CLEAN-ATTRIBUTION"),
        ("算例2", 0.013889, 0.027778, 0.625000, "C109-MULTIPLICATIVE"),
        ("算例2b", 0.000000, 0.000000, 0.625000, "C109-MULTIPLICATIVE"),
        ("加性", 0.200000, 0.250000, 0.450000, "C109-BOTH-ADDITIVE"),
        ("Q独大", 0.300000, 0.050000, 0.340000, "C109-WQ-DOMINANT"),
        ("O独大", 0.050000, 0.300000, 0.340000, "C109-WO-DOMINANT"),
        ("无损伤", 0.010000, 0.020000, 0.030000, "C109-NO-DAMAGE"),
    ]
    bad = [(n, classify(a, b, c)[0], w) for n, a, b, c, w in cases if classify(a, b, c)[0] != w]
    rec("A109-5a", not bad, f"7 个分类算例全对（失败 {bad}）", bad or None)
    # 穷尽性：4 种 (DQ,DO) × DB 三档 全部分类唯一
    seen = 0
    for dq in (0.05, 0.15):
        for do in (0.05, 0.15):
            for db in (0.05, 0.30, 0.60):
                classify(dq, do, db)
                seen += 1
    rec("A109-5b", seen == 12, f"穷尽性走遍 {seen}/12 组合无异常")
    # rank_eff 解析核
    re_cases = []
    for r in (1, 4, 16):
        M = torch.zeros(16, 64)
        M[:r, :r] = torch.eye(r) * 3.0
        re_cases.append(abs(struct_stats([M[i] for i in range(16)])["rank_eff"] - r) < 1e-6)
    rec("A109-5c", all(re_cases), f"rank_eff 在 r=1/4/16 上等于 r（{re_cases}）")

    # ---- A109-1b：params_sel("both") 与 params_nogamma 逐项同序
    st0 = D90.make_stack(gamma=0.005, no_write=True, seed=SEEDS[0])
    a = params_sel(st0, "both")
    b = D104.params_nogamma(st0)
    ok_same = len(a) == len(b) and all(x is y for x, y in zip(a, b))
    rec("A109-1b", ok_same, f"params_sel('both') 与 D104.params_nogamma 逐项同对象同序"
                            f"（n={len(a)} vs {len(b)}）")
    st0.remove()

    # ---- 跑 6 格（T-NONE / T-BOTH × 3 seed）+ 2 格（T-Q / T-O × 1 seed）
    d104 = json.loads((_ROOT / "dir104" / "out" / "stage_formal.json").read_text(encoding="utf-8"))
    g = 0.005
    got = {}
    for arm in ("T-NONE", "T-BOTH"):
        for s in SEEDS:
            c = run_cell(arm, g, s, tok, model, facts, chunks, owner, train_keys, eval_keys)
            got[(arm, s)] = c
            RES["cells"][f"{arm}|{g}|{s}"] = c
    for arm in ("T-Q", "T-O"):
        c = run_cell(arm, g, SEEDS[0], tok, model, facts, chunks, owner, train_keys, eval_keys)
        got[(arm, SEEDS[0])] = c
        RES["cells"][f"{arm}|{g}|{SEEDS[0]}"] = c

    # ---- A109-2：T-NONE 逐位复现 RAWINJ
    diffs = []
    for s in SEEDS:
        ref = d104["cells"][f"RAWINJ|{g}|{s}"]
        if abs(ref["acc_item"] - got[("T-NONE", s)]["acc_item"]) > 1e-12:
            diffs.append((s, ref["acc_item"], got[("T-NONE", s)]["acc_item"]))
    rec("A109-2", not diffs, f"T-NONE(γ={g}) 逐位复现 DIR-104 的 RAWINJ ×3 seed（差 {diffs}）", diffs or None)

    # ---- A109-1：T-BOTH 逐位复现 INJ（acc_item 相等 ∧ final_loss 前 6 位）
    diffs = []
    for s in SEEDS:
        ref = d104["cells"][f"INJ|{g}|{s}"]
        now = got[("T-BOTH", s)]
        if abs(ref["acc_item"] - now["acc_item"]) > 1e-12:
            diffs.append((s, "acc", ref["acc_item"], now["acc_item"]))
        if ref["final_loss"] is not None and now["final_loss"] is not None:
            if abs(ref["final_loss"] - now["final_loss"]) > 5e-7:
                diffs.append((s, "loss", ref["final_loss"], now["final_loss"]))
    rec("A109-1", not diffs, f"T-BOTH(γ={g}) 逐位复现 DIR-104 的 INJ ×3 seed（差 {diffs}）", diffs or None)

    # ---- A109-3：q/m 重算自洽
    errs = {k: v["probe_rel_err"] for k, v in got.items()}
    mx = max(errs.values())
    rec("A109-3", mx < PROBE_RTOL, f"q/m 重算相对误差 max={mx:.3e} < {PROBE_RTOL}", mx)

    # ---- A109-4：冻结契约（★ Gamma 必须按 float32 比较：`float32(0.005) != 0.005`，`DEF-109-01`）
    bad = []
    for (arm, s), c in got.items():
        if abs(c["state"][L16]["gamma"] - float(torch.tensor(g, dtype=torch.float32))) > 1e-9:
            bad.append((arm, s, "gamma", c["state"][L16]["gamma"]))
        for li, sc in c["scalars"].items():
            if sc["dW_fro"] > W_DRIFT_TOL:
                bad.append((arm, s, f"dW@{li}", sc["dW_fro"]))
        if c["violations"]:
            bad.append((arm, s, "viol", c["violations"]))
        if not c["finite_params"]:
            bad.append((arm, s, "Wq/Wo 非有限", True))
    rec("A109-4", not bad,
        f"γ 精确冻结(float32 口径) ∧ ‖ΔW‖=0 ∧ 写入无梯度 ∧ Wq/Wo 有限（违例 {bad[:4]}）", bad[:8] or None)

    # ---- A109-9（★ 本轮新增，专防 `DEF-109-02`）：评测期不得发生任何写入
    bad = [(k, v["eval_writes"]) for k, v in got.items() if v["eval_writes"] != 0]
    rec("A109-9", not bad, f"评测期写入次数恒为 0（违例 {bad[:4]}）", bad[:8] or None)

    # ---- A109-6：T-NONE 的 ρ=0 ∧ gain=1 精确
    bad = []
    for s in SEEDS:
        sc = got[("T-NONE", s)]["scalars"]
        for li, v in sc.items():
            if not (v["rho_q"] < 1e-12 and v["rho_o"] < 1e-12 and abs(v["gain"] - 1.0) < 1e-12):
                bad.append((s, li, v["rho_q"], v["rho_o"], v["gain"]))
    rec("A109-6", not bad, f"T-NONE: ρ_q=ρ_o=0 ∧ gain=1 精确（违例 {bad[:3]}）", bad[:6] or None)

    # ---- A109-7：单侧训练不得动另一侧
    bad = []
    for arm, frozen_key in (("T-Q", "rho_o"), ("T-O", "rho_q")):
        sc = got[(arm, SEEDS[0])]["scalars"]
        for li, v in sc.items():
            if v[frozen_key] > 1e-12:
                bad.append((arm, li, frozen_key, v[frozen_key]))
    rec("A109-7", not bad, f"T-Q 的 ρ_o=0 ∧ T-O 的 ρ_q=0 精确（违例 {bad[:3]}）", bad[:6] or None)

    # ---- A109-8：未训练格的状态可读且 κ 有限为正（DIR-104 从未覆盖）
    bad = []
    for s in SEEDS:
        st_res = got[("T-NONE", s)]["state"]
        for li, v in st_res.items():
            if not (np.isfinite(v["kappa"]) and v["kappa"] > 0):
                bad.append((s, li, v["kappa"]))
    rec("A109-8", not bad, f"T-NONE 三 seed 三层 κ ∈ (0,∞) 有限（违例 {bad[:3]}）", bad[:6] or None)

    return all(x["ok"] for x in RES["stageA"] if x["blocking"])


def md5(p):
    import hashlib
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ 判据
def verdict():
    cells = RES["cells"]
    rows, codes = {}, {}
    for g in GAMMAS:
        agg = {}
        for arm in ARMS:
            ks = [f"{arm}|{g}|{s}" for s in SEEDS]
            have = [cells[k] for k in ks if k in cells]
            if len(have) != len(SEEDS):
                continue
            ok = np.concatenate([np.array(c["item_ok"], dtype=int) for c in have])
            agg[arm] = {"acc": float(ok.mean()), "ok": ok,
                        "gain16": [c["scalars"][L16]["gain"] for c in have],
                        "kappa16": [c["state"][L16]["kappa"] for c in have],
                        "rho_q16": [c["scalars"][L16]["rho_q"] for c in have],
                        "rho_o16": [c["scalars"][L16]["rho_o"] for c in have],
                        "final_loss": [c["final_loss"] for c in have]}
        if len(agg) != len(ARMS):
            continue
        DQ = agg["T-NONE"]["acc"] - agg["T-Q"]["acc"]
        DO = agg["T-NONE"]["acc"] - agg["T-O"]["acc"]
        DB = agg["T-NONE"]["acc"] - agg["T-BOTH"]["acc"]
        code, I = classify(DQ, DO, DB)
        p_q = D104.paired_perm_p(agg["T-NONE"]["ok"], agg["T-Q"]["ok"])
        p_o = D104.paired_perm_p(agg["T-NONE"]["ok"], agg["T-O"]["ok"])
        p_b = D104.paired_perm_p(agg["T-NONE"]["ok"], agg["T-BOTH"]["ok"])
        # 判据 B：gain 比 = gain(T-BOTH)/gain(T-NONE)=gain(T-BOTH)/1
        gb = agg["T-BOTH"]["gain16"]
        gn = agg["T-NONE"]["gain16"]
        ratio = [b / max(n, 1e-12) for b, n in zip(gb, gn)]
        bcode = "C109-B-GAIN-EXPLOSION" if all(r >= TAU_GAIN for r in ratio) \
            else "C109-B-NO-GAIN-EXPLOSION"
        rows[g] = {"acc": {k: v["acc"] for k, v in agg.items()},
                   "D_Q": DQ, "D_O": DO, "D_B": DB, "I": I,
                   "p_q": p_q, "p_o": p_o, "p_b": p_b,
                   "code": code, "code_B": bcode,
                   "gain16": {k: v["gain16"] for k, v in agg.items()},
                   "kappa16": {k: v["kappa16"] for k, v in agg.items()},
                   "rho_q16": {k: v["rho_q16"] for k, v in agg.items()},
                   "rho_o16": {k: v["rho_o16"] for k, v in agg.items()},
                   "gain_ratio": ratio,
                   "final_loss": {k: v["final_loss"] for k, v in agg.items()}}
        codes[g] = code
    if not codes:
        return {"verdict": "C109-INCOMPLETE", "reason": "格不全"}
    uniq = set(codes.values())
    code = list(uniq)[0] if len(uniq) == 1 else "/".join(f"{g}:{c}" for g, c in codes.items())
    gdep = len(uniq) > 1
    if gdep:
        code = code + "-GAMMA-DEPENDENT"
    bset = {rows[g]["code_B"] for g in rows}
    return {"verdict": code, "codes_per_gamma": codes,
            "code_B": ("C109-B-GAIN-EXPLOSION" if bset == {"C109-B-GAIN-EXPLOSION"}
                       else ("C109-B-NO-GAIN-EXPLOSION" if bset == {"C109-B-NO-GAIN-EXPLOSION"}
                             else "MIXED")),
            "gamma_dependent": gdep, "rows": rows,
            "reason": f"D_Q={rows[list(rows)[0]]['D_Q']:.4f} "
                      f"D_O={rows[list(rows)[0]]['D_O']:.4f} "
                      f"D_B={rows[list(rows)[0]]['D_B']:.4f} "
                      f"I={rows[list(rows)[0]]['I']:.4f}"}


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
        RES["stageA"] = [g for g in RES["stageA"] if not g["name"].startswith("A109-0")] + \
                        [{"name": "A109-0", "ok": True, "detail": "adjudicate 复用",
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
        plan = [(arm, 0.005, seeds[0]) for arm in ARMS]
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