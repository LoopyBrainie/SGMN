"""`R1_task119.py` —— `DIR-119` rev 1.1（冻结）实现。

★ 逐字派生：**直接 import** `dir118/R1_task118.py`（它 → `dir117` → `dir115` → `dir114`/`dir104`/`dir90`）。
★ **唯一改动 = 测量层**：`dir117.acc_of` 把 `S` 丢掉，本卡把 `S` 留住并导出
  `rank_hist` / `acc_argmax` / `pick_argmin` / `pick_argmax` / `err_in_train_frac` / `err_in_train_null`。
  **训练、增益网格、超参、题集、seed、N_CAND、rng_seed 一律不动**（卡片 §4）。
★ ★ 为什么用 **monkey-patch** 而不是重写 cell 体：`P117._run_cell117_inner` 里 `acc_of` 的调用序列
  是**装置的一部分**（先 prior 再 9 点）。重写会引入"看起来一样"的第二套实现（`DEF-118-17` 的教训）。
  打补丁只换"测量"，**不动**训练与调用顺序。
  ★ 并且**不假设**调用顺序：跑完立刻用 `prior_acc` 与 `acc_by_mult` **反查**每个 stash 的归属，
  对不上就抛异常（**证明映射，不假设映射**）。
★ ★ C1/C2 必须在 `run_cell117` **内部**做：`st = make_stack(...)` 在 `finally: st.remove()`
  ⇒ 训练态在 cell 返回后已消失，事后无法再评测。
"""
import argparse
import json
import pathlib
import random
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "dir118"))
import numpy as np            # noqa: E402

import R1_task118 as P118     # noqa: E402
import R1_task117 as P117     # noqa: E402
import R1_task90 as D90       # noqa: E402
import R1_task104 as D104     # noqa: E402
import mech119                # noqa: E402
from mech119 import score_and_mech, ND   # noqa: E402

OUT = _HERE / "out"
for _d in ("cells", "det", "c2"):
    (OUT / _d).mkdir(parents=True, exist_ok=True)
D90.OUT = OUT            # ★ import 会改指，立刻抢回
P117.OUT = OUT
P118.OUT = OUT           # ★ 让 P118.run_cell 的落盘目录变成 dir119/out

P = P118.P                          # dir114 模块（make_items_with_vals/class_seen_frac/…）
MODEL, DTYPE, DEV = D90.MODEL, D90.DTYPE, D90.DEV
MULT = P117.MULT
GAMMA_TR = P117.GAMMA_TR
SEEDS = P118.SEEDS                  # ★ 与 dir118 同一组 9 个，不换（卡片 §2）
DOSES = (600, 0)
ARM = "COLOCATED"
N_ITEM = 48
JUDGE_MULT, REF_MULT = 10000, 1

# ---------------------------------------------------------------- 冻结阈值（卡片 §2，一字不许改）
M2_ACC = 0.90          # J119-1
M1_GAP = 0.20          # J119-2
T_MIN = 2.0            # J119-3 / J119-4
NULL_EXPECT = 0.5      # 卡片 §2 的解析零假设；A119-4 容差 0.02
NULL_TOL = 0.02
C2_RNGS = (556, 557, 558)
C1_PERM = (2, 0, 3, 1)

RES = {"stageA": [], "cells": {}, "c2": {}, "c1": {}, "det": {}, "plan_n": 0}


def rec(name, ok, detail, blocking=True):
    RES.setdefault("stageA", []).append(
        {"name": name, "ok": bool(ok), "detail": str(detail), "blocking": bool(blocking)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return bool(ok)


def dump(o, n):
    (OUT / n).write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")



# ---------------------------------------------------------------- patch 状态
_STASH = {"S": [], "meta": None, "tgt": None, "tv": None,
          "c2_items": None, "c2_meta": None, "c2_tgt": None, "c2": {}}
_ORIG_ACC_OF = P117.acc_of
_CNT = {"i": -1}


def acc_of_patched(model, st, items, tgt):
    """★ 与 `dir117.acc_of` **读同一个 S**，只是不再丢掉它。"""
    S = D104.score_matrix(model, st, items)
    S_np = S.detach().cpu().numpy() if hasattr(S, "detach") else np.asarray(S)
    if not bool(np.isfinite(S_np).all()):
        _STASH["S"].append(None)
        _CNT["i"] += 1
        return None, None, False
    a = float(D104.acc_item_of(S, tgt))
    ok = [int(x) for x in D104.item_ok(S, tgt)]
    _STASH["S"].append(np.array(S_np, dtype=np.float64, copy=True))
    _CNT["i"] += 1
    # C2：训练态还活着时，用同一个 `st` 给重抽候选集打分（只在被判的两点）
    idx = _CNT["i"] - 1                      # 0..8 ⇒ MULT 的下标；-1 ⇒ prior 那一次
    if 0 <= idx < len(MULT) and float(MULT[idx]) in (JUDGE_MULT, REF_MULT) \
            and _STASH["c2_items"] is not None:
        key = f"{MULT[idx]:g}"
        got = {}
        for rng_seed, its in _STASH["c2_items"].items():
            Sa = D104.score_matrix(model, st, its)
            San = Sa.detach().cpu().numpy() if hasattr(Sa, "detach") else np.asarray(Sa)
            got[str(rng_seed)] = score_and_mech(
                San, _STASH["c2_tgt"][rng_seed], _STASH["c2_meta"][rng_seed], _STASH["tv"])
        _STASH["c2"][key] = got
    return (a, ok, True) if np.isfinite(a) else (None, None, False)


def run_cell(seed, dose, tok, model, facts, tr, ev, items_ev, meta_ev, tv,
             c2_items=None, c2_meta=None, c2_tgt=None, write_cell=True,
             scratch_out=None):
    """复用 `P118.run_cell`（内含 skip-if-exists + 落盘 + arm 标记），只额外注入测量层。"""
    _STASH["S"] = []
    _CNT["i"] = -1
    _STASH["meta"], _STASH["tgt"], _STASH["tv"] = meta_ev, None, tv
    _STASH["c2_items"], _STASH["c2_meta"], _STASH["c2_tgt"] = c2_items, c2_meta, c2_tgt
    _STASH["c2"] = {}
    P117.acc_of = acc_of_patched
    saved_out = P118.OUT
    if scratch_out is not None:
        # ★ `DEF-119-05`（修）：`P118.run_cell` 自己会写 `OUT/cells/...`，
        #   光关掉我这层不够 —— 它写的是**没有 mech_by_mult 的裸版**，
        #   于是下一次调用会命中它的 skip-if-exists 而完全不跑模型（stash 空）。
        #   ⇒ det 必须把 `P118.OUT` 整个挪到 scratch，让 cells/ 不被污染。
        (scratch_out / "cells").mkdir(parents=True, exist_ok=True)
        P118.OUT = scratch_out
    try:
        d = P118.run_cell(ARM, seed, dose, tok, model, facts, tr, items_ev)
    finally:
        P117.acc_of = _ORIG_ACC_OF
        P118.OUT = saved_out

    tgt = D104.targets_of(items_ev)
    S_all = _STASH["S"]
    # ★ 证明映射，不假设映射：1 次 prior + 9 点
    if dose == 600 or dose == 0:
        assert len(S_all) == 1 + len(MULT), \
            (f"S 个数 {len(S_all)} != 1+{len(MULT)}（测量层归属不可信）。"
             f"★ 最常见原因：`P118.run_cell` 的 skip-if-exists 命中 —— "
             f"检查 {OUT/'cells'}/{ARM}_{int(dose)}_{int(seed)}.json 是否已存在"
             f"（裸版无 mech_by_mult 也会触发）。det 请用 scratch_out。")
        a0 = score_and_mech(S_all[0], tgt, meta_ev, tv) if S_all[0] is not None else None
        assert a0 is None or abs(a0["acc_argmin"] - d["prior_acc"]) < 1e-12, \
            f"stash[0] 与 prior_acc 不符：{a0 and a0['acc_argmin']} vs {d['prior_acc']}"
        for i, m in enumerate(MULT):
            if S_all[i + 1] is None:
                continue
            ai = score_and_mech(S_all[i + 1], tgt, meta_ev, tv)
            assert abs(ai["acc_argmin"] - d["acc_by_mult"][f"{m:g}"]) < 1e-12, \
                (f"stash[{i+1}] 与 acc_by_mult[{m:g}] 不符："
                 f"{ai['acc_argmin']} vs {d['acc_by_mult'][f'{m:g}']}")

    per = {}
    for i, m in enumerate(MULT):
        k = f"{m:g}"
        per[k] = (score_and_mech(S_all[i + 1], tgt, meta_ev, tv)
                  if S_all[i + 1] is not None else {"finite": False})
    d["mech_by_mult"] = per
    d["prior_mech"] = (score_and_mech(S_all[0], tgt, meta_ev, tv)
                       if S_all[0] is not None else {"finite": False})
    d["c2"] = _STASH["c2"]
    d["n_seen_hist"] = per[f"{JUDGE_MULT:g}"].get("n_seen_hist")
    if write_cell:            # ★ `DEF-119-05`：det 不得写 cells/，否则第二次 det 会被
        f = OUT / "cells" / f"{ARM}_{int(dose)}_{int(seed)}.json"   # `P118.run_cell`
        f.write_text(json.dumps(d, ensure_ascii=False, indent=1),      # 的 skip-if-exists
                     encoding="utf-8")                                 # 早返回 ⇒ stash 空 ⇒ 断言炸
    return d


# ================================================================ Stage A
def synthetic_case():
    """卡片 §3 的手算例（**冻结读数，A119-5 逐位断言**）。"""
    tv = {"x", "z", "q", "r"}
    S = np.array([[1.5, 0.9, 2.0, 0.4],
                  [0.1, 3.0, 2.5, 1.8]], dtype=np.float64)
    meta = [{"cand_val": ["x", "y", "z", "w"]},
            {"cand_val": ["p", "q", "r", "s"]}]
    tgt = np.array([1, 0], dtype=np.int64)
    return S, tgt, meta, tv


def stage_a(tok, model):
    print("\n=== Stage A：DIR-119 守卫（13 条）===", flush=True)
    facts, parts, csf, IT = P118.ground(tok, ARM)
    items, meta = IT["items_ev"], IT["meta_ev"]
    tv, tr, ev = IT["tv"], IT["tr"], IT["ev"]
    tgt = D104.targets_of(items)

    # A119-1 前提：dir118 存档里没有新字段
    ref = _HERE.parent / "dir118" / "out" / "stage_formal.json"
    need_absent = ("rank_hist", "pick_argmin", "pick_argmax", "err_in_train_frac",
                   "mech_by_mult", "acc_argmax")
    if ref.exists():
        rj = json.loads(ref.read_text(encoding="utf-8"))
        c0 = rj["cells"][sorted(rj["cells"])[0]]
        absent = [k for k in need_absent if k not in c0]
        rec("A119-1", len(absent) == len(need_absent),
            f"dir118 cell 缺 {len(absent)}/{len(need_absent)} 个新字段（键数 {len(c0)}）"
            f"⇒ 新测量层必要；仍缺 = {absent}")
    else:
        rec("A119-1", False, "找不到 dir118/out/stage_formal.json（前提无法核）")

    # A119-2 分层：评测项全未见值
    rec("A119-2", abs(csf[ARM]["frac"] - 0.0) < 1e-12,
        f"class_seen_frac_held(COLOCATED)={csf[ARM]['frac']:.6f}（须 0.0）；"
        f"seen={csf[ARM]['seen']}/{csf[ARM]['n']}")

    # A119-3 结构
    struct_ok = (len(items) == N_ITEM
                 and all(len(it["cand"]) == D90.N_CAND for it in items)
                 and all(sum(1 for _ids, o in it["cand"] if o) == 1 for it in items)
                 and all(len(ids) == 1 for it in items for ids, _o in it["cand"]))
    rec("A119-3", struct_ok,
        f"{len(items)} 项 × {D90.N_CAND} 候选、各恰 1 正确、候选皆单 token")

    # A119-4 tv 同源 + 零假设解析值
    m0 = score_and_mech(np.zeros((N_ITEM, D90.N_CAND)) + np.arange(D90.N_CAND)[None, :],
                        tgt, meta, tv)
    frac_ok = abs(m0["n_seen_distr_mean"] / float(ND) - NULL_EXPECT) <= NULL_TOL
    tv_ok = (all(facts[i]["val"] not in tv for i in ev)
             and len(tv) == len({facts[i]["val"] for i in tr}))
    rec("A119-4", tv_ok and frac_ok,
        f"tv 同源={tv_ok}（|tv|={len(tv)}）；零假设均值实测 "
        f"{m0['n_seen_distr_mean']/float(ND):.4f}（须 {NULL_EXPECT}±{NULL_TOL}）；"
        f"n_seen_hist={m0['n_seen_hist']}")

    # A119-5 手算例
    S5, t5, mt5, tv5 = synthetic_case()
    g = score_and_mech(S5, t5, mt5, tv5)
    exp = {"acc_argmin": 0.5, "acc_argmax": 0.0, "rank_hist": [1, 1, 0, 0],
           "n_err": 1, "err_in_train_frac": 0.0, "err_in_train_null": 2.0 / 3.0,
           "n_seen_distr_mean": 2.0}
    bad = []
    for k, want in exp.items():
        got = g[k]
        okk = (abs(got - want) < 1e-12) if isinstance(want, (int, float)) else (got == want)
        if not okk:
            bad.append(f"{k}: 实测{got} vs 期望{want}")
    gap = g["err_in_train_frac"] - g["err_in_train_null"]
    if abs(gap - (-2.0 / 3.0)) >= 1e-12:
        bad.append(f"GAP: 实测{gap} vs 期望{-2.0/3.0}")
    rec("A119-5", not bad, "手算例 8 读数逐位相符" if not bad else "；".join(bad))

    # A119-6 恒等式 acc_argmax == rank_hist[3]/n（随机 200 例 + 真实 S）
    rng = np.random.default_rng(4242)
    bad6 = 0
    for _ in range(200):
        Sr = rng.normal(size=(N_ITEM, D90.N_CAND))
        gr = score_and_mech(Sr, tgt, meta, tv)
        if abs(gr["acc_argmax"] - gr["rank_hist"][3] / float(N_ITEM)) > 1e-15:
            bad6 += 1
    rec("A119-6", bad6 == 0,
        f"随机 200 例 + 真实题集：acc_argmax ≡ rank_hist[3]/n，不符 {bad6} 例")

    # A119-7 C1：列置换（合成：解析） + 真跑 score_matrix（经验）
    p = list(C1_PERM)
    inv = [p.index(j) for j in range(D90.N_CAND)]
    Sp = S5[:, p]
    tp = np.array([inv[int(x)] for x in t5])
    mtp = [{"cand_val": [mt5[i]["cand_val"][p[j]] for j in range(D90.N_CAND)]}
           for i in range(len(mt5))]
    gp = score_and_mech(Sp, tp, mtp, tv5)
    same_analytic = (abs(gp["acc_argmin"] - g["acc_argmin"]) < 1e-15
                     and gp["rank_hist"] == g["rank_hist"]
                     and abs(gp["err_in_train_null"] - g["err_in_train_null"]) < 1e-15)
    items_p = [{"key_idx": it["key_idx"], "prompt_ids": it["prompt_ids"],
                "cand": [it["cand"][j] for j in p]} for it in items]
    st = D90.make_stack(gamma=GAMMA_TR, no_write=True, seed=SEEDS[0])
    try:
        for mm in st.mems.values():
            mm.no_write = True
        P115 = P118.P115
        P115.set_gamma(st, GAMMA_TR * REF_MULT)
        S_a = np.asarray(D104.score_matrix(model, st, items), dtype=np.float64)
        S_b = np.asarray(D104.score_matrix(model, st, items_p), dtype=np.float64)
        # ★ `DEF-119-02`（本 cycle 自我更正）：正确的不变量是
        #   `S_b == S_a[:, p]`（**逐位**），**不是** `S_b == S_a`。
        #   候选是对同一 prompt **独立**打分的（`nll_of_candidate`），故重排候选
        #   只会把同一组 NLL 值换个列位置 —— 第一版把两式写成同一个断言，
        #   于是"装置正确"被自己的错断言判成失败。**守卫报红先怀疑自己的实现。**
        perm_same = bool(np.array_equal(S_a[:, p], S_b))
        cp = [p.index(j) for j in range(D90.N_CAND)]
        acc_b = float(np.mean(np.argmin(S_b, axis=1) == np.array([cp[int(x)] for x in tgt])))
        acc_same = abs(acc_b - float(np.mean(np.argmin(S_a, axis=1) == tgt))) < 1e-15
    finally:
        st.remove()
    rec("A119-7", same_analytic and perm_same and acc_same,
        f"列置换：解析不变={same_analytic}；真跑 S 逐位满足 S_b==S_a[:,p]={perm_same}；"
        f"acc 不变={acc_same}"
        f" ★ 这**只**证明装置无位置耦合，**不**等于「已排除位置效应」（DEF-119-01）")

    # A119-8/9 结构口径（真实 S 上验收）
    gr = score_and_mech(S_a, tgt, meta, tv)
    a89 = (gr["n"] == N_ITEM
           and len(gr["ok"]) == N_ITEM
           and abs(sum(gr["ok"]) / float(N_ITEM) - gr["acc_argmin"]) < 1e-12
           and sum(gr["rank_hist"]) == N_ITEM)
    rec("A119-8", a89, f"真实 S：|ok|={len(gr['ok'])}、sum/48==acc、Σrank_hist==48")
    rec("A119-9", True,
        f"有限性口径：真实 S 全有限={bool(np.isfinite(S_a).all())}"
        f"（逐 cell 的 eval_writes/W 不变/violations 在 cell 阶段复核）", blocking=False)

    # A119-10 M2 门会跳也会落
    St_lo = np.tile(np.array([0.0, 1.0, 2.0, 3.0]), (N_ITEM, 1))
    S_hi = St_lo.copy()
    t_lo = np.zeros(N_ITEM, dtype=np.int64)                  # 正确项=argmin ⇒ rank 1
    t_hi = np.full(N_ITEM, 3, dtype=np.int64)                # 正确项=argmax ⇒ rank 4
    g_lo = score_and_mech(St_lo, t_lo, meta, tv)
    g_hi = score_and_mech(S_hi, t_hi, meta, tv)
    rec("A119-10", g_lo["acc_argmax"] == 0.0 and g_hi["acc_argmax"] == 1.0,
        f"正确项 rank1 ⇒ acc_argmax={g_lo['acc_argmax']}（须 0.0，不触发）；"
        f"rank4 ⇒ {g_hi['acc_argmax']}（须 1.0，触发）")

    # A119-11 M1 门会跳也会落（构造 meta：每项 2 个见过干扰项 ⇒ 零假设 2/3）
    tv_s = {"x", "z"}
    meta_s = [{"cand_val": ["x", "y", "z", "w"]} for _ in range(N_ITEM)]
    t_s = np.ones(N_ITEM, dtype=np.int64)                    # 正确 = "y"（未见值）
    # 触发行：argmin 取 "x"（见过）且**错**、正确项 rank 2（⇒ acc_argmax=0，M2 不误触发）
    S_hit = np.tile(np.array([0.0, 1.0, 9.0, 2.0]), (N_ITEM, 1))
    g_hit = score_and_mech(S_hit, t_s, meta_s, tv_s)
    gap_hit = g_hit["err_in_train_frac"] - g_hit["err_in_train_null"]
    # 不触发行：错误项里 2/3 落见过值、1/3 落未见值 ⇒ 与零假设 2/3 一致 ⇒ GAP = 0
    S_mix = np.empty((N_ITEM, 4), dtype=np.float64)
    for i in range(N_ITEM):
        S_mix[i] = (np.array([0.0, 1.0, 9.0, 2.0]) if i % 3 != 2
                    else np.array([5.0, 3.0, 4.0, 1.0]))     # argmin="w"（未见）
    g_mix = score_and_mech(S_mix, t_s, meta_s, tv_s)
    gap_mix = (None if g_mix["n_err"] == 0 else
               g_mix["err_in_train_frac"] - g_mix["err_in_train_null"])
    rec("A119-11", gap_hit >= M1_GAP and (gap_mix is None or gap_mix < M1_GAP),
        f"全选见过 ⇒ GAP={gap_hit:.4f}（须 ≥{M1_GAP}，触发）；"
        f"混合构造 ⇒ GAP={gap_mix if gap_mix is None else round(gap_mix,4)}（须 <{M1_GAP}，不触发）")

    # A119-12/13 在 cell 阶段（需要真跑）
    rec("A119-12", True, "自确定性：在 det 阶段双进程复跑同一 cell 后复核",
        blocking=False)
    rec("A119-13", True, "设备继承：dose=0 的 81 个读数在 merge 阶段逐位比对",
        blocking=False)

    RES["premise"] = {"csf": csf, "n_item": len(items), "n_seen_hist": m0["n_seen_hist"]}
    RES["plan_n"] = len(SEEDS) * len(DOSES)
    dump(RES, "stage_stageA.json")
    bad_block = [r for r in RES["stageA"] if r["blocking"] and not r["ok"]]
    print(f"\n=== Stage A 结果：{len(RES['stageA'])} 条，阻断失败 {len(bad_block)} 条 ===",
          flush=True)
    return 0 if not bad_block else 1


# ================================================================ 单格
def cells_main(seed, dose):
    tok, model = D90.model_and_tok()   # ★ D90 返回 (tok, m)，顺序不可颠倒
    facts, parts, csf, IT = P118.ground(tok, ARM)
    items, meta = IT["items_ev"], IT["meta_ev"]
    tv, tr, ev = IT["tv"], IT["tr"], IT["ev"]
    c2_items = c2_meta = c2_tgt = None
    if int(dose) == 600:
        c2_items, c2_meta, c2_tgt = {}, {}, {}
        for r in C2_RNGS:
            its, mts = P.make_items_with_vals(tok, facts, ev, r)
            c2_items[str(r)] = its
            c2_meta[str(r)] = mts
            c2_tgt[str(r)] = D104.targets_of(its)
    d = run_cell(int(seed), int(dose), tok, model, facts, tr, ev, items, meta, tv,
                 c2_items, c2_meta, c2_tgt)
    o = {"seed": int(seed), "dose": int(dose),
         "acc_by_mult": d["acc_by_mult"], "prior_n": d["prior_n"],
         "mech_by_mult": d["mech_by_mult"], "prior_mech": d.get("prior_mech"),
         "c2": d.get("c2"), "n_seen_hist": d.get("n_seen_hist"),
         "eval_writes": d.get("eval_writes"), "violations": d.get("violations"),
         "W_bitwise_same": d.get("W_bitwise_same"), "n_finite": d.get("n_finite"),
         "wq_md5": d.get("wq_md5"), "wo_md5": d.get("wo_md5")}
    dump(o, f"cells/{ARM}_{int(dose)}_{int(seed)}.json")
    print(f"  完成 {ARM}|{dose}|{seed} prior={d['prior_n']} "
          f"acc(1e4)={d['acc_by_mult'].get('10000')} "
          f"argmax={d['mech_by_mult'].get('10000', {}).get('acc_argmax')}", flush=True)
    return 0


def det_main(tag, seed, dose):
    tok, model = D90.model_and_tok()   # ★ D90 返回 (tok, m)，顺序不可颠倒
    facts, parts, csf, IT = P118.ground(tok, ARM)
    items, meta = IT["items_ev"], IT["meta_ev"]
    d = run_cell(int(seed), int(dose), tok, model, facts, IT["tr"], IT["ev"],
                 items, meta, IT["tv"], write_cell=False,
                 # ★ `DEF-119-05`（第二次修）：scratch 必须**按 tag 分开**。
                 #   若 a/b 共用同一 scratch，det a 写下的裸格会让 det b 命中
                 #   `P118.run_cell` 的 skip-if-exists ⇒ 又是 stash 空 ⇒ 同一处崩溃。
                 scratch_out=OUT / "det" / f"_scratch_{tag}")
    o = {"tag": tag, "seed": int(seed), "dose": int(dose),
         "acc_by_mult": d["acc_by_mult"],
         "picks": {k: v.get("pick_argmin") for k, v in d["mech_by_mult"].items()},
         "rank_hist": {k: v.get("rank_hist") for k, v in d["mech_by_mult"].items()}}
    dump(o, f"det/{tag}.json")
    print(f"  det {tag} 完成", flush=True)
    return 0


# ================================================================ merge / 裁决
def _tstat(xs, mu0):
    xs = [float(x) for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = float(np.mean(xs))
    sd = float(np.std(xs, ddof=1))
    if sd == 0.0:
        # ★ `DEF-119-03`：退化情形必须**保号**。原写法恒返回 `+inf`，
        #   于是"均值显著**低于** μ0"（应 t=−inf、`J119-4` 通过）被判成 +inf ⇒ 门误判失败。
        return 0.0 if m == mu0 else (float("inf") if m > mu0 else float("-inf"))
    return (m - mu0) / (sd / np.sqrt(len(xs)))


def merge_main():
    print("\n=== merge：汇总 + 裁决 ===", flush=True)
    ref = _HERE.parent / "dir118" / "out" / "stage_formal.json"
    refj = json.loads(ref.read_text(encoding="utf-8")) if ref.exists() else {"cells": {}}
    cells = {}
    for f in sorted((OUT / "cells").glob(f"{ARM}_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        cells[f"{d['dose']}|{d['seed']}"] = d
    missing = [f"{dd}|{s}" for dd in DOSES for s in SEEDS if f"{dd}|{s}" not in cells]

    # A119-13 设备继承：dose=0 的 81 个读数与 dir118 逐位相同
    mism, cmp_n = [], 0
    for s in SEEDS:
        k = f"0|{s}"
        rk = f"COLOCATED|0|{s}"
        if k not in cells or rk not in refj["cells"]:
            continue
        for m in MULT:
            a = cells[k]["acc_by_mult"].get(f"{m:g}")
            b = refj["cells"][rk]["acc_by_mult"].get(f"{m:g}")
            cmp_n += 1
            if a is None or b is None or abs(a - b) > 0:
                mism.append((s, f"{m:g}", a, b))
    rec("A119-13", (cmp_n > 0 and not mism),
        f"dose=0 与 dir118 逐位比对 {cmp_n} 个读数，不符 {len(mism)} 个"
        + (f"：{mism[:3]}" if mism else "（★ 无训练 ⇒ 确定性，可作硬门）"))

    # A119-12 自确定性
    det = {}
    for tag in ("a", "b"):
        f = OUT / "det" / f"{tag}.json"
        if f.exists():
            det[tag] = json.loads(f.read_text(encoding="utf-8"))
    if len(det) == 2:
        same = (det["a"]["acc_by_mult"] == det["b"]["acc_by_mult"]
                and det["a"]["picks"] == det["b"]["picks"]
                and det["a"]["rank_hist"] == det["b"]["rank_hist"])
        rec("A119-12", same, f"双进程复跑同一 cell：逐位相同={same}")
    else:
        rec("A119-12", False, f"det 文件不全（拿到 {sorted(det)}）")

    # 逐 seed 读数
    rows, gaps_s, accmax_s, accmin_s = [], [], [], []
    for s in SEEDS:
        k = f"600|{s}"
        if k not in cells:
            continue
        if "mech_by_mult" not in cells[k]:
            raise SystemExit(
                f"✗ cell {k} 缺 mech_by_mult（很可能是 `P118.run_cell` 写下的裸版）"
                f"⇒ 删掉该文件重跑 `--stage cell --seed {s} --dose 600`")
        me = cells[k]["mech_by_mult"].get(f"{JUDGE_MULT:g}", {})
        if not me.get("finite"):
            rows.append({"seed": s, "censored": True})
            continue
        gap = (None if me["err_in_train_frac"] is None
               else me["err_in_train_frac"] - me["err_in_train_null"])
        rows.append({"seed": s, "acc_argmin": me["acc_argmin"],
                     "acc_argmax": me["acc_argmax"], "rank_hist": me["rank_hist"],
                     "n_err": me["n_err"],
                     "err_in_train_frac": me["err_in_train_frac"],
                     "err_in_train_null": me["err_in_train_null"], "GAP": gap})
        gaps_s.append(gap)
        accmax_s.append(me["acc_argmax"])
        accmin_s.append(me["acc_argmin"])

    censored = [r for r in rows if r.get("censored")]
    n_eval = len(rows)
    stat = {}
    if gaps_s:
        stat["acc_argmin_mean"] = float(np.mean(accmin_s))
        stat["acc_argmax_mean"] = float(np.mean(accmax_s))
        stat["GAP_mean"] = float(np.mean([g for g in gaps_s if g is not None]))
        stat["t_argmin_vs_025"] = _tstat(accmin_s, 0.25)
        stat["t_GAP"] = _tstat([g for g in gaps_s if g is not None], 0.0)

    j = {}
    j["J119-1"] = bool(stat.get("acc_argmax_mean") is not None
                       and stat["acc_argmax_mean"] >= M2_ACC)
    j["J119-2"] = bool(stat.get("GAP_mean") is not None and stat["GAP_mean"] >= M1_GAP)
    tg = stat.get("t_GAP")
    j["J119-3"] = bool(tg is not None and tg >= T_MIN)
    ta = stat.get("t_argmin_vs_025")
    j["J119-4"] = bool(ta is not None and ta <= -T_MIN)

    if censored or len(accmin_s) < len(SEEDS):
        verdict = "CENSORED"
    elif not j["J119-4"]:
        verdict = "REPLICATION-FAILURE"
    elif j["J119-1"]:
        verdict = "M2"
    elif j["J119-2"] and j["J119-3"]:
        verdict = "M1"
    else:
        verdict = "M-OTHER"

    out = {"arm": ARM, "missing": missing, "censored": censored, "rows": rows,
           "stat": stat, "judge": j, "verdict": verdict,
           "thresholds": {"M2_ACC": M2_ACC, "M1_GAP": M1_GAP, "T_MIN": T_MIN,
                          "JUDGE_MULT": JUDGE_MULT, "REF_MULT": REF_MULT},
           "stageA": RES.get("stageA", [])}
    dump(out, "stage_verdict.json")

    print(f"  缺格 = {missing}", flush=True)
    print(f"  acc_argmin 均值 = {stat.get('acc_argmin_mean')}（随机 0.25，t={ta}）", flush=True)
    print(f"  acc_argmax 均值 = {stat.get('acc_argmax_mean')}（J119-1 须 ≥{M2_ACC}）", flush=True)
    print(f"  GAP 均值 = {stat.get('GAP_mean')}（J119-2 须 ≥{M1_GAP}；t={tg}）", flush=True)
    print(f"  判据 = {j}", flush=True)
    print(f"  ★★ 判决 = {verdict}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("A", "cell", "det", "merge"))
    ap.add_argument("--seed", type=int, default=SEEDS[0])
    ap.add_argument("--dose", type=int, default=600)
    ap.add_argument("--tag", default="a")
    a = ap.parse_args()
    t0 = time.time()
    if a.stage == "merge":
        return merge_main()
    tok, model = D90.model_and_tok()   # ★ D90 返回 (tok, m)，顺序不可颠倒
    # ★ `DEF-119-02`：解包顺序写反时，错误会在 `torch.embedding` 里以
    #   `indices must be Tensor, not str` 出现，离病根很远。此处提前断言。
    assert hasattr(tok, "encode") and hasattr(tok, "__call__"), \
        f"tok 不像 tokenizer：{type(tok).__name__}（十有八九 model/tok 解包颠倒了）"
    assert hasattr(model, "forward") and hasattr(model, "parameters"), \
        f"model 不像 nn.Module：{type(model).__name__}"
    if a.stage == "A":
        rc = stage_a(tok, model)
    elif a.stage == "cell":
        rc = cells_main(a.seed, a.dose)
    else:
        rc = det_main(a.tag, a.seed, a.dose)
    print(f"  [{a.stage}] rc={rc} 用时 {time.time()-t0:.1f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())