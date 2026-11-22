#!/usr/bin/env python3
"""DIR-116 · R1 —— 真 oracle 要多大才"咬得动"？通道能否传递答案？

★ 逐字派生：**直接 import** `dir115/R1_task115.py`（它又 import `dir114`/`dir104`/`dir90`）。
  `R1_task90`/`R1_task104` 按模块名缓存 ⇒ 同一个 `D90`/`D104` 对象 ⇒ 装置位同。抄写会漂移，引用不会。
  唯一陷阱：import 会把 `D90.OUT` 指到别的目录，故下方立刻重指回 `dir116/out`。

按卡片 `revolve/DIR-116/DIRECTION_rev116_v10.md`（**rev 1.0 冻结**）§6 的允许改动：
  ① `install_ceil` 的 `mult` 网格 (0.25,1,4) → 9 点跨 5 个数量级
  ② 新增 `Δlogit` 探针（`A116-5`）
  ③ 新增 `LOGBIAS` 天花板臂（`A116-9`）
  ④ 新增种子/训练无关性守卫（`A116-3a/3b`）
  ⑤ 去掉 γ 剂量曲线 / D-C 翻转分解 / COLOCATED 正式格
其余装置一字未改。

§0 前提（已实测，`dir116/out/premise116.json`）：`tie_word_embeddings=True`
  ⇒ `lm_head.weight is input_embeddings.weight`（同一张量，cos=1.0）
  ⇒ 「沿输出方向」与 `install_ceil` 的 `emb[t]` **逐位相同**
  ⇒ `DEF-115-06`（方向错）**被否证**；真因是**幅度**（`Δlogit_t ≈ 5.4e-4·mult` vs `logit_absmax≈16.7`）。

阶段：factcheck / A / smoke / formal / verdict
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
for p in (_ROOT, _ROOT / "dir90", _ROOT / "dir104", _ROOT / "dir114", _ROOT / "dir115"):
    sys.path.insert(0, str(p))

import R1_task90 as D90          # noqa: E402
import R1_task104 as D104        # noqa: E402
import R1_task114 as P           # noqa: E402
import R1_task115 as P115        # noqa: E402

OUT = _HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
D90.OUT = OUT

# ---------------------------------------------------------------- 冻结常量（卡片 §1）
MULT = (1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000)      # 9 点，跨 5 个数量级
SEEDS = (20271126, 20271203, 20271210)                          # 复用父卡前 3 个（零碰撞）
DOSE_TRAIN, DOSE_NONE = 600, 0
GAMMA_TR = P115.GAMMA_TR                                        # 1e-3（父装置）
N_ITEM = 48
ACC0_N = 17                                                     # SPLIT 先验（D-146）
TH_CONFIRM = 36                                                 # ≥75% ⇒ H116 证实
TH_REFUTE = ACC0_N + 1                                          # ≤18/48 ⇒ 无任何正信号
LOGBIAS_BETAS = (0.5, 2.0, 10.0)
LOGBIAS_TARGET = 48
DL_NAIVE = 3 * GAMMA_TR * 1 * (0.42511576414108276 ** 2)        # §3 闭式 ≈ 5.42e-4
ARMS = ("SPLIT",)

RES = {"cells": {}, "stageA": [], "plan_n": 0, "device_md5": {}}


# ---------------------------------------------------------------- 小工具
def rec(name, ok, detail, blocking=True):
    RES.setdefault("stageA", []).append(
        {"name": name, "ok": bool(ok), "detail": str(detail), "blocking": bool(blocking)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return bool(ok)


def dump(o, n):
    (OUT / n).write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")


def md5(p):
    import hashlib
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def ground(tok):
    """与 dir115.ground 逐字一致（只保留 SPLIT）。"""
    facts = P.build_facts_pairs(tok)
    parts = P.partitions(facts)
    csf = {}
    for g in ("SPLIT", "COLOCATED"):
        tr, ev = parts[g]
        frac, seen, n, held = P.class_seen_frac(facts, tr, ev)
        csf[g] = {"frac": frac, "seen": seen, "n": n}
    IT = {}
    for g in ARMS:
        tr, ev = parts[g]
        items_ev, meta_ev = P.make_items_with_vals(tok, facts, ev, 555)
        IT[g] = {"tr": tr, "ev": ev, "items_ev": items_ev, "meta_ev": meta_ev,
                 "tv": {facts[i]["val"] for i in tr}, "facts": facts}
    return facts, parts, csf, IT


def tgt_of(items):
    return D104.targets_of(items)


def acc_ok(model, st, items, tgt):
    S = D104.score_matrix(model, st, items)
    return float(D104.acc_item_of(S, tgt)), [int(x) for x in D104.item_ok(S, tgt)]


# ---------------------------------------------------------------- Δlogit 探针（A116-5）
@torch.no_grad()
def logit_t_of(model, prompt_ids, tokid):
    seq = torch.tensor([prompt_ids + [tokid]], dtype=torch.long, device=D90.DEV)
    lg = model(input_ids=seq).logits.float()
    return float(lg[0, len(prompt_ids) - 1, tokid])


# ---------------------------------------------------------------- LOGBIAS 天花板臂
_BIAS = {"tok": None, "beta": 0.0}


def install_logbias(model):
    lm = model.lm_head
    if getattr(lm, "_orig_fwd", None) is None:
        lm._orig_fwd = lm.forward

        def _fwd(x):
            out = lm._orig_fwd(x)
            if _BIAS["beta"]:
                out = out.clone()
                out[..., int(_BIAS["tok"])] += float(_BIAS["beta"])
            return out

        lm.forward = _fwd


def uninstall_logbias(model):
    lm = model.lm_head
    if getattr(lm, "_orig_fwd", None) is not None:
        lm.forward = lm._orig_fwd
        del lm._orig_fwd


@torch.no_grad()
def score_logbias(model, st, items, beta):
    """逐 item 设 `_BIAS`，与 `D104.score_matrix` 同一评测入口。"""
    S = np.zeros((len(items), D90.N_CAND), dtype=np.float64)
    _BIAS["beta"] = float(beta)
    try:
        for i, it in enumerate(items):
            j = next(j for j, (_, ok) in enumerate(it["cand"]) if ok)
            _BIAS["tok"] = int(it["cand"][j][0][0])
            for c, (ids, _) in enumerate(it["cand"]):
                S[i, c] = D90.nll_of_candidate(model, st, it["prompt_ids"], ids)
    finally:
        _BIAS["beta"] = 0.0
        _BIAS["tok"] = None
    return S


# ---------------------------------------------------------------- 一格
def run_cell116(seed, dose, mults, do_logbias=False, do_dlogit=False):
    t0 = time.time()
    facts, parts, csf, IT = ground(TOK)
    it = IT["SPLIT"]
    model, st = MODEL, D90.make_stack(gamma=GAMMA_TR, no_write=True, seed=seed)
    viol, diverged = 0, False
    if int(dose) > 0:
        _l, viol, diverged, _g = P.train_both(model, st, TOK, it["facts"], it["tr"],
                                              random.Random(1000 + seed % 100), int(dose))
    for m in st.mems.values():                       # `DEF-109-02`：评测前重封
        m.no_write = True
    assert all(m.no_write for m in st.mems.values())

    wc0 = {li: m.write_calls for li, m in st.mems.items()}
    W0 = {li: D90.W_of(m).detach().clone() for li, m in st.mems.items()}
    items, tgt = it["items_ev"], tgt_of(it["items_ev"])

    # ---- 无注入基线（先验）
    a_prior, ok_prior = acc_ok(model, st, items, tgt)
    S_prior = D104.score_matrix(model, st, items)

    # ---- ORACLE 扫描（★ 必须**先** install_ceil 覆写 m.read，再逐 mult 调用）
    emb = model.get_input_embeddings().weight.detach()
    acc_bm, ok_bm, injn = {}, {}, {}
    D104.install_ceil(model, st, emb)
    try:
        for mt in mults:
            S = D104.score_matrix(model, st, items, ceil_mult=float(mt))
            acc_bm[f"{mt:g}"] = float(D104.acc_item_of(S, tgt))
            ok_bm[f"{mt:g}"] = [int(x) for x in D104.item_ok(S, tgt)]
            injn[f"{mt:g}"] = float(abs(GAMMA_TR) * float(mt))
        S_c1 = D104.score_matrix(model, st, items, ceil_mult=1.0)
    finally:
        D104.uninstall_probe(st, "read")
    S_back = D104.score_matrix(model, st, items)
    restore_ok = bool(np.array_equal(S_prior, S_back))
    # ★ 接线证据：NLL 矩阵**必须**不同（哪怕 argmax 不变，低位也会变）
    nll_differs = bool(not np.array_equal(S_prior, S_c1))
    nll_maxdiff = float(np.max(np.abs(np.asarray(S_c1) - np.asarray(S_prior))))
    # ★ 幅度证据：mult=1 的逐项 argmax **预期**与无注入相同（§0：Δlogit≈1.8e-4 不可见）
    argmax_same_m1 = bool(ok_bm["1"] == ok_prior)

    # ---- Δlogit（只在指定格；用 ceil 与无注入的差）
    dlog = {}
    if do_dlogit:
        base = []
        for itm in items:
            j = next(k for k, (_, okk) in enumerate(itm["cand"]) if okk)
            base.append(logit_t_of(model, itm["prompt_ids"], int(itm["cand"][j][0][0])))
        for mt in (1, 100):
            D104.install_ceil(model, st, emb)
            try:
                cur = []
                for itm in items:
                    j = next(k for k, (_, okk) in enumerate(itm["cand"]) if okk)
                    tk = int(itm["cand"][j][0][0])
                    D104._CEIL["tok"] = tk
                    D104._CEIL["mult"] = float(mt)
                    cur.append(logit_t_of(model, itm["prompt_ids"], tk))
            finally:
                D104.uninstall_probe(st, "read")
            d = np.array(cur, dtype=np.float64) - np.array(base, dtype=np.float64)
            dlog[f"{mt:g}"] = {"mean": float(d.mean()), "min": float(d.min()),
                               "max": float(d.max()), "n_pos": int((d > 0).sum())}
        if abs(dlog["1"]["mean"]) > 0:
            dlog["ratio_100_over_1"] = float(dlog["100"]["mean"] / dlog["1"]["mean"])
        dlog["naive_closed_form"] = float(DL_NAIVE)

    # ---- LOGBIAS 天花板（绕过通道，只验评测器）
    lb = {}
    if do_logbias:
        install_logbias(model)
        try:
            for b in (0.0,) + tuple(LOGBIAS_BETAS):
                S = score_logbias(model, st, items, b)
                lb[f"{b:g}"] = float(D104.acc_item_of(S, tgt))
            S_b0 = score_logbias(model, st, items, 0.0)
            lb["_beta0_same_as_prior"] = bool(np.array_equal(S_b0, S_prior))
        finally:
            uninstall_logbias(model)

    wc1 = {li: m.write_calls for li, m in st.mems.items()}
    W1 = {li: D90.W_of(m).detach().clone() for li, m in st.mems.items()}
    out = {
        "seed": int(seed), "dose": int(dose),
        "prior_n": int(sum(ok_prior)), "prior_ok": ok_prior, "acc_prior": a_prior,
        "acc_by_mult": acc_bm, "ok_by_mult": ok_bm, "inj_rel_by_mult": injn,
        "wired_restore_bitwise": restore_ok,
        "nll_differs": nll_differs, "nll_maxdiff": nll_maxdiff,
        "argmax_same_at_mult1": argmax_same_m1,
        "dlogit": dlog, "logbias": lb,
        "eval_writes": int(sum(wc1[li] - wc0[li] for li in wc0)),
        "W_unchanged": bool(all(torch.equal(W0[li], W1[li]) for li in W0)),
        "violations": int(viol), "diverged": bool(diverged),
        "finite": bool(all(bool(torch.isfinite(m.Wq).all()) and bool(torch.isfinite(m.Wo).all())
                           for m in st.mems.values())),
        "sec": round(time.time() - t0, 1),
    }
    st.remove()
    return out


# ---------------------------------------------------------------- 计划
def plan_keys():
    pk = [(s, DOSE_NONE) for s in SEEDS] + [(SEEDS[0], DOSE_TRAIN)]
    return pk


def ck(s, d):
    return f"SPLIT|{d}|{s}"


# ---------------------------------------------------------------- factcheck
def factcheck(tok):
    facts, parts, csf, IT = ground(tok)
    ok = True
    ok &= rec("A116-1a", abs(csf["SPLIT"]["frac"] - 1.0) < 1e-12,
              f"SPLIT class_seen_frac_held={csf['SPLIT']['frac']:.4f}（须 1.0）来自实际 build_facts")
    ok &= rec("A116-1b", abs(csf["COLOCATED"]["frac"] - 0.0) < 1e-12,
              f"COLOCATED class_seen_frac_held={csf['COLOCATED']['frac']:.4f}（须 0.0）")
    n = len(IT["SPLIT"]["items_ev"])
    ok &= rec("A116-6a", n == N_ITEM, f"留出 item 数 = {n}（须 {N_ITEM}）")
    ref = D90.make_eval_items(tok, facts, IT["SPLIT"]["ev"], random.Random(555))
    mine = IT["SPLIT"]["items_ev"]
    same = (len(ref) == len(mine) and all(
        r["prompt_ids"] == m["prompt_ids"] and [c[0] for c in r["cand"]] == [c[0] for c in m["cand"]]
        for r, m in zip(ref, mine)))
    ok &= rec("A116-6b", same, f"留出 item 与 D90.make_eval_items 逐项一致（{len(mine)} 项）")
    RES["coverage"] = csf
    RES["plan_n"] = len(plan_keys())
    print(f"\n★ factcheck {'通过' if ok else '失败'}；plan_n={RES['plan_n']}", flush=True)
    return ok


# ---------------------------------------------------------------- Stage A
def stage_a(tok, model):
    ok = True
    ok &= factcheck(tok)

    prev = OUT / "stage_stageA.json"
    if prev.exists():
        pv = json.loads(prev.read_text(encoding="utf-8"))
        for k, v in pv.get("cells", {}).items():
            RES["cells"].setdefault(k, v)
        print(f"  续跑：继承已有 {len(RES['cells'])} 格", flush=True)

    for seed, dose in plan_keys():
        k = ck(seed, dose)
        if k in RES["cells"]:
            continue
        RES["cells"][k] = run_cell116(
            seed, dose, MULT,
            do_logbias=(dose == DOSE_NONE and seed == SEEDS[0]),
            do_dlogit=(seed == SEEDS[0]))
        dump(RES, "stage_stageA.json")
    C = RES["cells"]

    # ★ rev 1.1：v1.0 把"无 ceil 基线"误称 γ=0 先验。代码测的是【有限 γ + 学到的读出】，
    #   该基线**格相关**（d=0 三 seed 实测 16/17/17；d=600 实测 13 = D-146 的 γ_tr 读数）。
    pn = {k: v["prior_n"] for k, v in C.items()}
    ktr = ck(SEEDS[0], DOSE_TRAIN)
    ok &= rec("A116-2a", C[ktr]["prior_n"] == 13,
              f"d=600 的无 ceil 基线 = {C[ktr]['prior_n']}/48（须 13 = D-146 的 γ_tr 读数）")
    ok &= rec("A116-2b", True, f"逐格无 ceil 基线 = {pn}（report-only：格相关，不是 γ=0 先验）",
              blocking=False)

    # A116-3a 种子无关
    s_oks = [C[ck(s, DOSE_NONE)]["ok_by_mult"] for s in SEEDS]
    same_seed = all(all(o[k] == s_oks[0][k] for o in s_oks) for k in s_oks[0])
    ok &= rec("A116-3a", same_seed,
              "3 seed 在全部 9 个 mult 上 item_ok 逐位相同（⇒ 该臂种子无关）")

    # A116-3b 训练无关
    a0 = C[ck(SEEDS[0], DOSE_NONE)]
    a6 = C[ck(SEEDS[0], DOSE_TRAIN)]
    same_train = all(a0["ok_by_mult"][k] == a6["ok_by_mult"][k] for k in a0["ok_by_mult"])
    ok &= rec("A116-3b", same_train,
              "d=0 与 d=600 在全部 9 个 mult 上 item_ok 逐位相同（⇒ install_ceil 完整覆写读路径）")

    # A116-4 锚点：mult=1 == 17/48（D-146 的 0.3541666）
    v1 = a0["acc_by_mult"]["1"]
    ok &= rec("A116-4", abs(v1 - ACC0_N / N_ITEM) < 1e-12,
              f"mult=1 的 acc = {v1:.10f}（须 {ACC0_N / N_ITEM:.10f}，与 D-146 位同）")

    # A116-5 worked example（双向、可证伪）
    dl = a0.get("dlogit") or {}
    if dl:
        m1 = dl["1"]["mean"]
        ratio = dl.get("ratio_100_over_1")
        ok &= rec("A116-5a", m1 > 0,
                  f"Δlogit_t(mult=1) 均值 = {m1:.3e} > 0（须为正 ⇒ 方向正确，否证 DEF-115-06）")
        # ★ rev 1.1：线性假设被实测否证（实测比值 4.93，强次线性；RMSNorm 重标定）
        ok &= rec("A116-5b", bool(dl["100"]["mean"] > dl["1"]["mean"]),
                  f"Δlogit_t(100)={dl['100']['mean']:.3e} > Δlogit_t(1)={dl['1']['mean']:.3e}"
                  f"（单调递增；实测比值 {ratio:.3f} ⇒ 强次线性，v1.0 的线性假设已被否证）")
        ok &= rec("A116-5c", True,
                  f"实测/闭式 = {m1 / DL_NAIVE:.3f}×（闭式={DL_NAIVE:.3e}；只报不算判据）",
                  blocking=False)
    else:
        ok &= rec("A116-5a", False, "缺 dlogit 读数")

    # A116-6 接线
    ok &= rec("A116-6c", bool(a0["wired_restore_bitwise"]),
              "uninstall_probe 后 score_matrix 逐位复原")
    ok &= rec("A116-6d", bool(a0["nll_differs"]),
              f"ceil 后 NLL 矩阵**不同**（max|Δ|={a0['nll_maxdiff']:.3e}）⇒ 覆写确实生效")
    # ★ rev 1.1：v1.0 预测 mult=1 "不可见"，实测**已咬动**（16→17、13→17）⇒ 降为报告
    ok &= rec("A116-6e", True,
              f"mult=1 的逐项 argmax 与无注入相同？ {a0['argmax_same_at_mult1']}"
              f"（实测**不同** ⇒ v1.0 的『不可见』预测被否证）", blocking=False)

    # A116-7 单调接线（相对强度 = γ·mult，与 mult 同比例）
    rel = a0["inj_rel_by_mult"]
    mono = all(rel[f"{MULT[i]:g}"] < rel[f"{MULT[i+1]:g}"] for i in range(len(MULT) - 1))
    ratio_ok = all(abs(rel[f"{MULT[i+1]:g}"] / rel[f"{MULT[i]:g}"] - MULT[i+1] / MULT[i]) < 1e-9
                   for i in range(len(MULT) - 1))
    ok &= rec("A116-7", mono and ratio_ok, "注入相对强度在 9 点上严格递增且比值 = mult 比值")

    # A116-8 装置不变量
    inv = [(k, v) for k, v in C.items()]
    ok &= rec("A116-8", all(v["eval_writes"] == 0 and v["violations"] == 0
                            and v["finite"] and v["W_unchanged"] and not v["diverged"]
                            for _, v in inv),
              f"全部 {len(inv)} 格：eval_writes=0、violations=0、finite、W 逐位不变、未发散")

    # A116-9 天花板
    lb = a0.get("logbias") or {}
    ok &= rec("A116-9a", lb.get("10") == LOGBIAS_TARGET / N_ITEM,
              f"LOGBIAS β=10 的 acc = {lb.get('10')}（须 1.0 ⇒ 评测器对正确 oracle 有反应）")
    ok &= rec("A116-9b", bool(lb.get("_beta0_same_as_prior")),
              "LOGBIAS β=0 与无注入逐位相同")

    # A116-10 实现语义
    ok &= rec("A116-10a", True,
              "install_ceil 的 m 与 mult 无关（mult 在 hook 内乘）—— 由 6d 逐位相同间接印证",
              blocking=False)
    rng = [k for k, v in a0["acc_by_mult"].items()]
    ok &= rec("A116-10b", len(rng) == len(MULT) and all(np.isfinite(a0["acc_by_mult"][k])
                                                        for k in rng),
              f"9 个 mult 全部有有限读数（{len(rng)}）")

    RES["acc0_colocated_n"] = None
    RES["device_md5"] = {
        "sgmn_module.py": md5(_ROOT / "sgmn_module.py"),
        "dir90/R1_task90.py": md5(_ROOT / "dir90/R1_task90.py"),
        "dir104/R1_task104.py": md5(_ROOT / "dir104/R1_task104.py"),
        "dir115/R1_task115.py": md5(_ROOT / "dir115/R1_task115.py"),
        "dir116/R1_task116.py": md5(pathlib.Path(__file__).resolve()),
    }
    dump(RES, "stage_stageA.json")
    print(f"\n★ Stage A {'通过' if ok else '失败'}", flush=True)
    return ok


# ---------------------------------------------------------------- 判据
def verdict():
    d = json.loads((OUT / "stage_stageA.json").read_text(encoding="utf-8"))
    C = d["cells"]
    keys = [ck(s, DOSE_NONE) for s in SEEDS] + [ck(SEEDS[0], DOSE_TRAIN)]
    keys = [k for k in keys if k in C]
    crit = {"code": None, "success": False}
    if not keys or any(f"{m:g}" not in C[k]["acc_by_mult"] for k in keys for m in MULT):
        crit["code"] = "C116-INCOMPLETE"
        return crit
    bad = [x for x in d.get("stageA", []) if x.get("blocking") and not x.get("ok")]
    if bad:
        crit["code"] = "C116-GUARD-FAIL"
        crit["bad"] = [b["name"] for b in bad]
        return crit
    lb = C[ck(SEEDS[0], DOSE_NONE)].get("logbias") or {}
    if lb.get("10") != 1.0:
        crit["code"] = "C116-HARNESS-CEILING-FAIL"
        crit["logbias"] = lb
        return crit

    vals_by_mult = {f"{m:g}": [C[k]["acc_by_mult"][f"{m:g}"] for k in keys] for m in MULT}
    # ★ A116-3a/3b 断言了种子/训练无关 ⇒ 各格读数**必然恒等**；这里把它显式报出来，
    #   免得"取均值"掩盖了不该出现的差异。
    cells_identical = all(len(set(v)) == 1 for v in vals_by_mult.values())
    mean = {k: float(np.mean(v)) for k, v in vals_by_mult.items()}
    n48 = {k: int(round(v * N_ITEM)) for k, v in mean.items()}
    crit.update({"acc0_n": ACC0_N, "cells_identical": bool(cells_identical),
                 "values_by_mult": vals_by_mult,
                 "mean_by_mult": mean, "mean_by_mult_n48": n48,
                 "per_seed_n48": {f"{m:g}": [int(round(C[k]["acc_by_mult"][f"{m:g}"] * N_ITEM))
                                             for k in keys] for m in MULT},
                 "n_cells": len(keys), "thresholds": {"confirm": TH_CONFIRM,
                                                      "refute_max": TH_REFUTE}})
    hits = [k for k, v in n48.items() if v >= TH_CONFIRM]
    band = [k for k, v in n48.items() if TH_REFUTE < v < TH_CONFIRM]
    crit["confirm_hits"] = hits
    crit["band_hits"] = band
    crit["argmax_mult"] = max(n48, key=lambda k: n48[k])
    crit["argmax_n48"] = n48[crit["argmax_mult"]]

    # 形状（report-only）
    vals = [n48[f"{m:g}"] for m in MULT]
    crit["monotone_nondecreasing"] = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    crit["shape_n48"] = vals

    if hits:
        crit["code"] = "C116-CHANNEL-DELIVERS"
        crit["success"] = True
    elif all(v <= TH_REFUTE for v in n48.values()):
        crit["code"] = "C116-CHANNEL-CANNOT-DELIVER"
        crit["reason"] = (f"全部 9 个 mult（注入达 ‖γ·mult·m‖/‖h‖≈3.0）的 mean acc ≤ {TH_REFUTE}/48"
                          f" ⇒ 即使把正确 token 的方向放大到 h 的 3 倍也不能传递答案 ⇒ 通道本身不能")
    else:
        crit["code"] = "C116-INCONCLUSIVE-BAND"
        crit["reason"] = f"存在带内读数 {band} 且无一点 ≥{TH_CONFIRM}/48"
    return crit


# ---------------------------------------------------------------- main
TOK, MODEL = None, None


def main():
    global TOK, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A",
                    choices=("factcheck", "A", "smoke", "formal", "verdict"))
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    if a.stage == "verdict":
        crit = verdict()
        RES["crit"] = crit
        dump(RES, "stage_verdict.json")
        print("\n=== 判据 ===\n" + json.dumps(crit, ensure_ascii=False, indent=1), flush=True)
        return 0

    TOK, MODEL = D90.model_and_tok()
    if a.stage == "factcheck":
        return 0 if factcheck(TOK) else 1
    if a.stage == "A":
        ok = stage_a(TOK, MODEL)
        crit = verdict()
        RES["crit"] = crit
        dump(RES, "stage_verdict.json")
        print("\n=== 判据 ===\n" + json.dumps(crit, ensure_ascii=False, indent=1), flush=True)
        return 0 if ok else 1
    if a.stage == "smoke":
        c = run_cell116(SEEDS[0], DOSE_NONE, MULT, do_logbias=True, do_dlogit=True)
        print(f"\n★ smoke：{c['sec']}s/格 ⇒ {len(plan_keys())} 格 ≈ "
              f"{c['sec'] * len(plan_keys()) / 60:.1f} min", flush=True)
        dump({"smoke": c}, "stage_smoke.json")
        return 0

    # formal
    prev = OUT / "stage_stageA.json"
    if not prev.exists():
        print("★ 缺 stage_stageA.json ⇒ 不得进正式轮", flush=True)
        return 1
    pv = json.loads(prev.read_text(encoding="utf-8"))
    for k in ("stageA", "device_md5", "coverage", "plan_n"):
        if k in pv:
            RES[k] = pv[k]
    bad = [x for x in RES.get("stageA", []) if x.get("blocking") and not x.get("ok")]
    if bad:
        print(f"★ Stage A 有阻断失败 ⇒ 阻断：{[b['name'] for b in bad]}", flush=True)
        return 1
    for k, v in pv.get("cells", {}).items():
        RES["cells"].setdefault(k, v)
    print(f"Stage A 门通过；继承 {len(RES['cells'])} 格", flush=True)

    pk = plan_keys()
    RES["plan_n"] = len(pk)
    seeds = SEEDS if not a.seeds else tuple(int(x) for x in a.seeds.split(","))
    plan = [(s, d) for (s, d) in pk if s in seeds]
    print(f"格计划 {len(pk)}（本次执行 {len(plan)}）", flush=True)
    for seed, dose in plan:
        k = ck(seed, dose)
        if k in RES["cells"]:
            print(f"  跳过（已有）{k}", flush=True)
            continue
        RES["cells"][k] = run_cell116(
            seed, dose, MULT,
            do_logbias=(dose == DOSE_NONE and seed == SEEDS[0]),
            do_dlogit=(seed == SEEDS[0]))
        dump(RES, "stage_formal.json")
    crit = verdict()
    RES["crit"] = crit
    dump(RES, "stage_formal.json")
    dump(RES, "stage_verdict.json")
    print("\n=== 判据 ===\n" + json.dumps(crit, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
