#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-91 rev 1.7/1.8 · 闭式 W* 端到端 4 选 1（task: DIR-91-rev2.3）

守卫 A91-22/23/24/25/26（另调 A91-19/20/21）。三臂同读侧预算、gamma=0：
  W0 / real（599 次写入）/ Wstar（直接置为装置自身目标的精确不动点）
裁决：eval_keys 上的按题准确率 acc_q（n=16，4 选 1，chance=0.25）。
"""
import argparse, json, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from R2_task91 import (ALPHA, DEV, ETA_WRITE, GAMMA0, LAM, N_CAND, N_DRAWS, N_FACTS,
                       N_TRAIN_KEYS, OUT, READ_STEPS, W_of, acc_by_question,
                       build_chunks, build_facts, key_matrices, make_eval_items,
                       make_stack, model_and_tok, nll_of_candidate, score_and_accuracy,
                       train_read_task, write_phase)

RES = {"card": "DIR-91 rev 1.8", "task": "DIR-91-rev2.3",
       "alpha": ALPHA, "lam": LAM, "eta": ETA_WRITE, "read_steps": READ_STEPS,
       "n_train_keys": N_TRAIN_KEYS, "n_cand": N_CAND, "n_draws": N_DRAWS,
       "guards": {}, "arms": []}
FAILS = []


def rec(name, ok, detail, value=None):
    RES["guards"][name] = {"ok": bool(ok), "detail": detail, "value": value}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------ 闭式解
def closed_form(K, V, W0, alpha=ALPHA, lam=LAM):
    """W* = (alpha V^T K + lam W0)(alpha K^T K + lam I)^{-1}，float64、CPU。"""
    K64, V64, W064 = K.double().cpu(), V.double().cpu(), W0.double().cpu()
    dk = K64.shape[1]
    A = alpha * (K64.T @ K64) + lam * torch.eye(dk, dtype=torch.float64)
    B = alpha * (V64.T @ K64) + lam * W064
    return torch.linalg.solve(A.T, B.T).T          # (d_v, d_k) float64


def stationarity(K, V, W0, Ws, alpha=ALPHA, lam=LAM):
    """相对不动点残差 ||alpha(W* K^T K - V^T K) + lam(W*-W0)||_F / ||lam W0||_F。"""
    K64, V64, W064, Ws64 = K.double().cpu(), V.double().cpu(), W0.double().cpu(), Ws.double().cpu()
    R = alpha * (Ws64 @ (K64.T @ K64) - (V64.T @ K64)) + lam * (Ws64 - W064)
    den = max(float(lam * W064.norm()), 1e-30)
    return float(R.norm()) / den


def objective(K, V, W0, W, alpha=ALPHA, lam=LAM):
    """L(W) = alpha/2 ||W K^T - V^T||_F^2 + lam/2 ||W-W0||_F^2（float64）。"""
    K64, V64, W064, W64 = K.double().cpu(), V.double().cpu(), W0.double().cpu(), W.double().cpu()
    return 0.5 * alpha * float((W64 @ K64.T - V64.T).norm()) ** 2 \
        + 0.5 * lam * float((W64 - W064).norm()) ** 2


def simulate_epochs(K, V, W0, checkpoints, alpha=ALPHA, lam=LAM, eta=ETA_WRITE):
    """装置**精确逐对**更新式在冻结 (K,V) 上顺序多轮（事实级 32 对，轮转顺序）的模拟。
    ★ 理想化口径：不是装置真实的 599 个块级对；仅回答"同一目标再给预算能到多近"。"""
    Kd, Vd, W0d = K.double(), V.double(), W0.double()
    n = Kd.shape[0]
    W = W0d.clone()
    out = {}
    ep = 0
    for target in sorted(checkpoints):
        while ep < target:
            for t in range(n):
                k, v = Kd[t], Vd[t]
                G = alpha * torch.outer(W @ k - v, k) + lam * (W - W0d)
                W = W - eta * G
            ep += 1
        out[target] = rel_of(K, V, W, torch.float64)
    return out


def rel_of(K, V, W, dt=torch.float32):
    """与 storage_gamma0 同口径：逐对相对误差的**中位数**。全部在 CPU 上算（32×d，可忽略）。"""
    Kk = K.detach().to("cpu").to(dt)
    Vv = V.detach().to("cpu").to(dt)
    Ww = W.detach().to("cpu").to(dt)
    r = ((Kk @ Ww.T) - Vv).norm(dim=1) / Vv.norm(dim=1).clamp_min(1e-30)
    return float(r.median())


# ------------------------------------------------------------------ guards
def run_guards(tok, model, facts, chunks, owner, seed):
    print(f"\n===== Stage A 守卫（seed={seed}）=====", flush=True)
    # ★★ 两个 SGMNStack **不得同时挂载**在同一个 model 上：否则 st_w 的 forward hook
    # 会在 st_nw 的 key_matrices 前向里继续写入 ⇒ 写入数 631 而非 599（本轮实测到此错，
    # 由 A91-20 捕获）。装置 storage_gamma0 也是**顺序**建栈，此处照此办理。
    st_nw = make_stack(gamma=0.0, no_write=True, seed=seed)
    try:
        K_nw, V_nw = key_matrices(model, st_nw, facts, chunks, owner)
    finally:
        st_nw.remove()

    st_w = make_stack(gamma=0.0, no_write=False, seed=seed)
    try:
        write_phase(model, st_w, chunks)
        K_w, V_w = key_matrices(model, st_w, facts, chunks, owner)
        n_pair = int(K_nw[list(K_nw)[0]].shape[0])
        RES["n_pairs"] = n_pair
        RES["d_k"] = int(K_nw[list(K_nw)[0]].shape[1])

        # A91-26 前提：gamma=0 下 k,v 不漂移（逐位）
        d = {li: float(max((K_w[li] - K_nw[li]).abs().max(),
                           (V_w[li] - V_nw[li]).abs().max())) for li in K_nw}
        rec("A91-26", max(d.values()) == 0.0,
            f"γ=0 下「无写」与「写后」的 K,V 逐位相同（逐层 max|Δ|={d}，须全为 0）", value=d)

        bad = [li for li, m in st_w.mems.items() if m.use_factorized]
        wc = {li: int(m.write_calls) for li, m in st_w.mems.items()}
        exp = len(chunks) - 1
        rec("A91-20b", (not bad) and all(v == exp for v in wc.values()),
            f"use_factorized 全 False（违例 {bad}）；写入数={wc}，期望={exp}（=块数-1）", value=wc)

        per = {}
        for li, m in st_w.mems.items():
            K, V, W0 = K_w[li], V_w[li], m.W0.detach()
            Ws64 = closed_form(K, V, W0)                      # CPU float64
            Ws32 = Ws64.to(DEV).float()
            res = stationarity(K, V, W0, Ws64)
            rel64 = rel_of(K, V, Ws64, torch.float64)         # float64 解的精度（基线）
            rel32 = rel_of(K, V, Ws32, torch.float32)         # float32 落地的精度
            # A91-24（rev 1.9）：**逐对**方向与稳定性（装置真实动力学），并模拟预算
            K64 = K.detach().double()
            V64 = V.detach().double()
            W064 = W0.detach().double()
            n = K64.shape[0]
            dec_min = 1e9
            eta_margin_min = 1e9
            for tt in range(n):
                k, v = K64[tt], V64[tt]
                Lt0 = 0.5 * ALPHA * float((W064 @ k - v).norm()) ** 2
                G_t = ALPHA * torch.outer(W064 @ k - v, k)
                Wnew = W064 - ETA_WRITE * G_t
                Lt1 = 0.5 * ALPHA * float((Wnew @ k - v).norm()) ** 2 \
                    + 0.5 * LAM * float((Wnew - W064).norm()) ** 2
                dec_min = min(dec_min, Lt0 - Lt1)          # >0 表示降低
                eta_margin_min = min(eta_margin_min,
                                     (2.0 / (ALPHA * float(k.norm()) ** 2 + LAM)) / ETA_WRITE)
            cp = [0, 1, 2, 4, 8, 19, 64]
            sim = simulate_epochs(K, V, W0, cp)
            per[li] = {"res": res, "rel64": rel64, "rel32": rel32,
                       "rel_W0": rel_of(K, V, W064, torch.float64),
                       "rel_real": rel_of(K, V, m.Wfull),
                       "norm_Ws": float(Ws64.norm()), "max_Ws": float(Ws64.abs().max()),
                       "dW": float((Ws64 - W0.detach().cpu().double()).norm()),
                       "pair_dec_min": dec_min, "eta_margin": eta_margin_min,
                       "sim_rel": {str(k): v for k, v in sim.items()},
                       "ratio_rel": rel32 / max(rel64, 1e-30)}
        RES["per_layer"] = {str(k): v for k, v in per.items()}

        rmax = max(v["res"] for v in per.values())
        rec("A91-22", rmax <= 1e-10,
            f"W* 是装置自身目标的**精确不动点**：逐层相对残差 max={rmax:.3e}（须 ≤1e-10，float64）",
            value={str(k): v["res"] for k, v in per.items()})

        fid = max(v["ratio_rel"] for v in per.values())
        absmax = max(v["rel32"] for v in per.values())
        rec("A91-23", fid <= 1.05 and absmax <= 1.5e-3,
            f"float32 落地保真：max(rel32/rel64)={fid:.4f}（须 ≤1.05）；"
            f"max rel32={absmax:.3e}（须 ≤1.5e-3，地板 8.03e-04）",
            value={str(k): v["rel32"] for k, v in per.items()})

        dec = all(v["pair_dec_min"] > 0 for v in per.values())
        eta_ok = all(v["eta_margin"] > 1.0 for v in per.values())
        rec("A91-24", dec and eta_ok,
            f"**逐对**方向：全部 32 对均降低（最小降低={min(v['pair_dec_min'] for v in per.values()):.3e}，"
            f"须>0）={dec}；**逐对**稳定：η 相对 2/(α|k|²+λ) 的最小余量="
            f"{min(v['eta_margin'] for v in per.values()):.2f}×（须>1）={eta_ok}；"
            f"**报告** 模拟 rel 随轮数={{{', '.join(f"{k}: {x:.2e}" for k, x in list(per.values())[0]['sim_rel'].items())}}}",
            value={str(k): v["sim_rel"] for k, v in per.items()})

        print("\n  逐层读数（rel 为中位数口径）：", flush=True)
        for li, v in per.items():
            print(f"    L{li}: rel(W0)={v['rel_W0']:.4f}  rel(real)={v['rel_real']:.4f}  "
                  f"rel(W*)={v['rel32']:.3e}  |W*|={v['norm_Ws']:.3f}  "
                  f"η余量={v['eta_margin']:.2f}x  sim(19轮)={v['sim_rel']['19']:.2e}", flush=True)
    finally:
        st_w.remove()


# ------------------------------------------------------------------ 三臂
def run_arm(name, tok, model, facts, chunks, train_keys, eval_keys, seed, Wstar, K, V):
    st = make_stack(gamma=0.0, no_write=(name != "real"), seed=seed)
    t0 = time.time()
    try:
        if name == "Wstar":
            for li, m in st.mems.items():
                with torch.no_grad():
                    m.Wfull.copy_(Wstar[li])
        wc = write_phase(model, st, chunks) if name == "real" else {"n/a": 0}
        losses, viol, div = train_read_task(model, st, tok, facts, train_keys,
                                           random.Random(1000 + seed % 100))
        for m in st.mems.values():
            m.no_write = True
        items = make_eval_items(tok, facts, eval_keys, random.Random(555))
        scorer = lambda p, cs: [nll_of_candidate(model, st, p, c) for c in cs]   # noqa: E731
        acc, _, det = score_and_accuracy(items, scorer)
        acc_q, nq = acc_by_question(items, det)
        margins = []
        for it, dd in zip(items, det):
            nlls = np.array(dd["nll"], dtype=float)
            ok_idx = [j for j, c in enumerate(it["cand"]) if c[1]]
            bad = [j for j in range(len(nlls)) if j not in ok_idx]
            margins.append(float(min(nlls[b] for b in bad) - nlls[ok_idx[0]]))
        out = {"arm": name, "seed": seed, "acc_item": acc, "acc_q": acc_q, "n_q": nq,
               "n_item": len(items), "margin_mean": float(np.mean(margins)),
               "margin_pos_frac": float(np.mean([m > 0 for m in margins])),
               "violations": viol, "diverged": div,
               "final_loss": float(np.mean(losses[-50:])) if losses else None,
               "read_steps": READ_STEPS,
               "rel": {str(li): rel_of(K[li], V[li], W_of(m)) for li, m in st.mems.items()},
               "w_norm": {str(li): float(W_of(m).norm()) for li, m in st.mems.items()},
               "write_calls": wc if isinstance(wc, dict) else {"n/a": 0},
               "sec": round(time.time() - t0, 1)}
        print(f"  [{name} seed={seed}] acc_q={acc_q:.3f} acc_item={acc:.4f} "
              f"n_q={nq} margin={out['margin_mean']:+.4f} "
              f"pos={out['margin_pos_frac']:.2f} loss={out['final_loss']} "
              f"div={div} {out['sec']}s", flush=True)
        return out
    finally:
        st.remove()


_KV = {}


_KV = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="guards", choices=["guards", "main", "all"])
    ap.add_argument("--seeds", default="20261028,7777")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    tok, model = model_and_tok()
    facts = build_facts(tok)
    chunks, owner = build_chunks(tok, facts)
    order = list(range(N_FACTS))
    random.Random(777).shuffle(order)
    train_keys, eval_keys = order[:N_TRAIN_KEYS], order[N_TRAIN_KEYS:]
    print(f"事实 {len(facts)}；块 {len(chunks)}；训练键 {len(train_keys)}；评测键 {len(eval_keys)}；"
          f"α={ALPHA} λ={LAM} η={ETA_WRITE}", flush=True)
    RES["train_keys"] = train_keys
    RES["eval_keys"] = eval_keys

    if a.stage in ("guards", "all"):
        run_guards(tok, model, facts, chunks, owner, seeds[0])
        if FAILS:
            print(f"\n★ Stage A FAIL: {FAILS} ⇒ 阻断正式轮", flush=True)
            (OUT / "probe91_wstar_guards.json").write_text(
                json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
            return 2

    if a.stage in ("main", "all"):
        for seed in seeds:
            print(f"\n===== seed {seed} =====", flush=True)
            st = make_stack(gamma=0.0, no_write=False, seed=seed)
            try:
                write_phase(model, st, chunks)
                K, V = key_matrices(model, st, facts, chunks, owner)
                Wstar = {}
                for li, m in st.mems.items():
                    Ws64 = closed_form(K[li], V[li], m.W0.detach())
                    Wstar[li] = Ws64.to(DEV).float()
                    RES.setdefault("wstar_norm", {})[f"{seed}_{li}"] = {
                        "norm": float(Ws64.norm()), "max": float(Ws64.abs().max()),
                        "rel32": rel_of(K[li], V[li], Wstar[li])}
            finally:
                st.remove()

            n_items = len(make_eval_items(tok, facts, eval_keys, random.Random(555)))
            for name in ("W0", "real", "Wstar"):
                RES["arms"].append(run_arm(name, tok, model, facts, chunks,
                                           train_keys, eval_keys, seed, Wstar, K, V))
            # A91-25 臂间可比（同评测项数、同读步数）
            got = [x for x in RES["arms"] if x["seed"] == seed]
            same = all(x["n_item"] == n_items and x["read_steps"] == READ_STEPS for x in got)
            rec(f"A91-25@{seed}", same and len(got) == 3,
                f"三臂仅初值 W 不同：同 seed/同读批/同 READ_STEPS={READ_STEPS}/"
                f"同评测项 n_item={n_items}（臂数 {len(got)}）", value=n_items)

    (OUT / "probe91_wstar.json").write_text(json.dumps(RES, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print("\n★ 完成 ⇒ dir91/out/probe91_wstar.json", flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
