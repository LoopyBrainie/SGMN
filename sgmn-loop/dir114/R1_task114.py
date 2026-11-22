#!/usr/bin/env python3
"""DIR-114 · R1 —— 「训练即损伤」的归因：留出答案值**覆盖**是成因吗？

★ 逐字派生自 `dir113/R1_task113.py`（→ `dir104/R1_task104.py` → `dir90/R1_task90.py`）。
  按卡片 `revolve/DIR-114/DIRECTION_rev114_v11.md`（**rev 1.1 冻结**）§10 的允许改动范围：
    ① 值分配 / **三臂划分**（`COLOCATED` / `SPLIT` / `LEGACY`）
    ② `READ_STEPS` 剂量轴 {0,100,600}
    ③ 新增 `acc_train` 与 `INJ-OFF`（γ=0）参考格
    ④ **去掉** `rank_eff` / `cos_mean` / 复合臂（本轮只跑 `T-BOTH` 的读侧训练）
  除此之外的装置（`make_stack` / `train_read_frozen` / `score_matrix` / 评测入口）**一字未改**。

★ `READ_LR = D90.READ_LR = 1e-2`（`COR-114-02`：`rev 1.0` 曾误写 1e-3，已由 `rev 1.1` 更正）。

阶段（`--stage`）：
  factcheck  —— 只查事实/划分/覆盖度 + `A114-2`（**不加载模型**，秒级）
  A          —— Stage A 硬门（`A114-1..5,7,8,9`；`A114-6` 按卡片 §5 在裁决期作闸门）
  smoke      —— 一格测时（**不出判据**）
  formal     —— 正式轮 54 格（可断点续跑）
  verdict    —— 只重算判据（不跑格）
"""
import argparse
import hashlib
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

import R1_task90 as D90                                     # noqa: E402
import R1_task104 as D104                                   # noqa: E402

OUT = _HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
D90.OUT = OUT

# ---------------------------------------------------------------- 冻结常量（卡片 rev 1.1）
GAMMA114 = 0.0010
GAMMA_INJ_OFF = 0.0
DOSES = (0, 100, 600)                 # 卡片 §3.2
LEGACY_DOSES = (0, 600)               # 卡片 §3.1（桥接臂只跑两端）
GRP_PRIMARY = ("COLOCATED", "SPLIT")
GRP_ALL = ("COLOCATED", "SPLIT", "LEGACY", "INJ-OFF")

SEEDS = (20271126, 20271203, 20271210, 20271217, 20271224, 20271231)   # 卡片 §3.3（零碰撞已核）
N_SEEDS = len(SEEDS)
N_VAL = 16                            # 16 个值 × 2 条事实（卡片 §3.1）
N_FACT = 2 * N_VAL                    # 32

# ---- 卡片 §2.3 冻结阈值 ----
GATE_FIT = 0.50                       # ★ rev 1.2：**report-only** 参考（借自 DIR-106 的「训练 batch」口径）
GATE_FIT_REL = 1.0 / 16.0             # ★ rev 1.2 唯一判据性学习阈值：「acc_train(600) − acc_train(0) ≥ 1 题」
TOL_ITEM = 1.0 / 48.0                 # damage「消失」= acc_held(600) ≥ acc_held(0) − 1/48
FALSIFY_ITEM = 2.0 / 48.0             # 否证线：damage ≤ −2/48 才算「仍损伤」
MIN_SEEDS = 6                         # 一致性 6/6 ⇒ p = 0.0156（卡片 §2.4）
# ★ 预注册分辨率（卡片 §2.5）：由 D-144 的 SD(damage)=0.0360 算得，写死以免事后调整
MDE_PAIRED_N6 = 0.0420
MDE_DID_N6 = 0.0594
REF_DAMAGE_D144 = -0.1892
N_ITEM = 48

STAGE_TAG = ["unknown"]

RES = {"dir": "DIR-114", "round": "R1", "model": D90.MODEL, "device": D90.DEV,
       "card": "revolve/DIR-114/DIRECTION_rev114_v11.md",
       "card_superseded": ["revolve/DIR-114/DIRECTION_rev114_v10.md"],
       "derived_from": "dir113/R1_task113.py",
       "gamma": GAMMA114, "gamma_inj_off": GAMMA_INJ_OFF,
       "doses": list(DOSES), "legacy_doses": list(LEGACY_DOSES),
       "seeds": list(SEEDS), "n_val": N_VAL, "n_fact": N_FACT,
       "gate_fit_report_only": GATE_FIT, "gate_fit_rel": GATE_FIT_REL,
       "tol_item": TOL_ITEM, "falsify_item": FALSIFY_ITEM,
       "min_seeds": MIN_SEEDS,
       "mde_paired_n6": MDE_PAIRED_N6, "mde_did_n6": MDE_DID_N6,
       "ref_damage_d144": REF_DAMAGE_D144,
       "read_lr": float(D90.READ_LR), "read_steps_default": D90.READ_STEPS,
       "device_md5": {}, "stageA": [], "coverage": {}, "partitions": {},
       "cells": {}, "crit": {}}


def rec(name, ok, detail, value=None, blocking=True):
    RES["stageA"].append({"name": name, "ok": bool(ok), "detail": str(detail),
                          "value": value, "blocking": bool(blocking)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}"
          + (f"  (value={value})" if value is not None else ""), flush=True)
    return bool(ok)


def dump(o, n):
    (OUT / n).write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")


def md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ================================================================ 事实与划分
def build_facts_pairs(tok, seed=9091):
    """**16 个值 × 2 条事实 = 32 条**（值各出现恰好 2 次）。卡片 §3.1。

    ★ 与 `D90.build_facts` 的**唯一**结构差别是值分配（27 值/32 事实 → 16 值/各 2 次），
      键生成方式、句子模板、token 化路径**逐字沿用** `D90.build_facts`。
      `A114-2` 保证 `COLOCATED` 与 `SPLIT` 用的是**同一批事实**（只有划分不同）。
    """
    assert len(D90.VALUE_POOL) >= N_VAL, len(D90.VALUE_POOL)
    vals16 = list(D90.VALUE_POOL[:N_VAL])
    rng = random.Random(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    keys, seen = [], set()
    while len(keys) < N_FACT:
        k = "".join(rng.choice(alphabet) for _ in range(5))
        if k in seen:
            continue
        seen.add(k)
        keys.append(k)
    vals = vals16 * 2                 # 每个值恰好两次
    rng.shuffle(vals)                 # ⇒ 同值两条不相邻、位置随机
    facts = [{"key": k, "val": v} for k, v in zip(keys, vals)]
    for f in facts:
        f["sent"] = f"The access code for item {f['key']} is {f['val']}."
        f["prompt"] = f"The access code for item {f['key']} is"
        f["val_ids"] = tok(" " + f["val"], add_special_tokens=False)["input_ids"]
        f["prompt_ids"] = tok(f["prompt"], add_special_tokens=False)["input_ids"]
    return facts


def partitions(facts, legacy_seed=777, shuffle_seed=777):
    """三臂划分。**同一 `facts` 对象**，只有 `(train_keys, eval_keys)` 不同。

    `COLOCATED`：每个值的**两条**都进同一侧（8 值→训练 16 条；8 值→留出 16 条）⇒ 覆盖 0
    `SPLIT`    ：每个值的**各一条**进各侧 ⇒ 覆盖 1.0
    `LEGACY`   ：`D90.build_facts` 的 27 值/32 事实 + `Random(777)`（**由调用方传原 facts**）
    """
    by_val = {}
    for i, f in enumerate(facts):
        by_val.setdefault(f["val"], []).append(i)
    assert all(len(v) == 2 for v in by_val.values()), \
        {k: len(v) for k, v in by_val.items() if len(v) != 2}
    ordered = sorted(by_val)                       # 确定性
    rng = random.Random(shuffle_seed)
    vs = ordered[:]
    rng.shuffle(vs)                                # 哪 8 个值整体进训练侧
    train_side, held_side = set(vs[:N_VAL // 2]), set(vs[N_VAL // 2:])

    # COLOCATED：整值进同一侧
    tr_c = [i for v in ordered if v in train_side for i in by_val[v]]
    ev_c = [i for v in ordered if v in held_side for i in by_val[v]]

    # SPLIT：每值各一条（哪一条进训练侧随机，但确定性）
    rng2 = random.Random(shuffle_seed + 1)
    tr_s, ev_s = [], []
    for v in ordered:
        a, b = by_val[v]
        if rng2.random() < 0.5:
            tr_s.append(a); ev_s.append(b)
        else:
            tr_s.append(b); ev_s.append(a)

    assert len(tr_c) == len(ev_c) == len(tr_s) == len(ev_s) == N_VAL, \
        (len(tr_c), len(ev_c), len(tr_s), len(ev_s))
    assert len(set(tr_c) & set(ev_c)) == 0 and len(set(tr_s) & set(ev_s)) == 0
    return {"COLOCATED": (tr_c, ev_c), "SPLIT": (tr_s, ev_s)}


def legacy_partition(n_facts=D90.N_FACTS, n_train=D90.N_TRAIN_KEYS, seed=777):
    order = list(range(n_facts))
    random.Random(seed).shuffle(order)
    return order[:n_train], order[n_train:]


def class_seen_frac(facts, train_keys, eval_keys):
    """★ `A114-1` 的唯一实现：**从实际 facts 对象算**，不许重实现（`COR-114-01`）。
    返回 (frac, n_held_seen, n_held, n_distinct_held_values_unseen)。"""
    tv = {facts[i]["val"] for i in train_keys}
    seen = sum(1 for i in eval_keys if facts[i]["val"] in tv)
    hv = {facts[i]["val"] for i in eval_keys}
    return (seen / len(eval_keys), seen, len(eval_keys),
            len(hv - tv))


# ================================================================ 装置（逐字沿用）
def gammas(st):
    return {li: float(m.gamma) for li, m in st.mems.items()}


def set_grad_gamma(st, on):
    for m in st.mems.values():
        m.gamma.requires_grad_(bool(on))


def params_nogamma(st):
    ps = []
    for m in st.mems.values():
        ps += [m.Wq, m.Wo]
    return ps


def train_both(model, st, tok, facts, train_keys, rng, dose):
    """`which="both"` ⇒ **直接**调 `D104.train_read_frozen`（同代码路径）。
    唯一改动：把 `D90.READ_STEPS` 临时设成 `dose`（剂量轴 = 唯一自变量之一）。"""
    prev = D90.READ_STEPS
    D90.READ_STEPS = int(dose)
    try:
        return D104.train_read_frozen(model, st, tok, facts, train_keys, rng)
    finally:
        D90.READ_STEPS = prev


def _margin_and_rank(S, tgt, meta, train_vals):
    """机制块（卡片 §6，report-only）。
    margin = min_wrong NLL − correct NLL（>0 好，<0 「自信地错」）；rank 1 = 最优。
    ★ 「错误落入训练值」的**零假设按每一项自己的 3 个干扰项算**（卡片 §6-3）：
       `n_train_distr/3`，**不是** `16/27` 这种常数。"""
    n = len(tgt)
    margins, ranks, err_in, err_null, n_err = [], [], [], [], 0
    for i in range(n):
        row = S[i]
        c = int(tgt[i])
        cand_val = meta[i]["cand_val"]
        wrong = [row[j] for j in range(len(row)) if j != c]
        margins.append(min(wrong) - row[c])
        order = np.argsort(row)
        ranks.append(int(np.where(order == c)[0][0]) + 1)
        n_train_distr = sum(1 for j in range(len(cand_val)) if j != c and cand_val[j] in train_vals)
        if int(np.argmin(row)) != c:                      # 错了
            n_err += 1
            pred_val = cand_val[int(np.argmin(row))]
            err_in.append(1 if pred_val in train_vals else 0)
            err_null.append(n_train_distr / 3.0)
    rank_hist = [ranks.count(k) for k in (1, 2, 3, 4)]
    return {"margin_mean": float(np.mean(margins)) if margins else None,
            "rank_hist": rank_hist,
            "n_err": n_err,
            "err_in_train_frac": (float(np.mean(err_in)) if n_err else None),
            "err_in_train_null": (float(np.mean(err_null)) if n_err else None)}


def eval_set(model, st, items, meta, train_vals):
    """唯一评测入口（`D104.score_matrix`）⇒ acc_item / acc_q / item_ok / 机制块。"""
    tgt = D104.targets_of(items)
    S = D104.score_matrix(model, st, items)
    acc_item = D104.acc_item_of(S, tgt)
    acc_q, nq = D104.acc_q_of(S, tgt, items)
    ok = D104.item_ok(S, tgt)
    mech = _margin_and_rank(S, tgt, meta, train_vals)
    return {"acc_item": float(acc_item), "acc_q": float(acc_q), "n_q": int(nq),
            "item_ok": [int(x) for x in ok], "mech": mech}


def make_items_with_vals(tok, facts, keys, rng_seed):
    """★ 自带候选值标签（`D90.make_eval_items` 不返回值串，机制块需要）。
    逐字复刻 `D90.make_eval_items` 的抽样逻辑（同 rng、同 `sorted(set(others))`）。"""
    rng = random.Random(rng_seed)
    items, meta = [], []
    for ki in keys:
        f = facts[ki]
        correct = f["val"]
        others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != correct]
        pool = sorted(set(others))
        for _ in range(D90.N_DRAWS):
            distr = rng.sample(pool, D90.N_CAND - 1)
            cands = [correct] + distr
            rng.shuffle(cands)
            items.append({"key_idx": ki,
                          "prompt_ids": f["prompt_ids"],
                          "cand": [(tok(" " + c, add_special_tokens=False)["input_ids"],
                                    c == correct) for c in cands]})
            meta.append({"cand_val": cands, "is_correct_val": [c == correct for c in cands]})
    return items, meta


def run_cell(grp, dose, seed, tok, model, facts, tr, ev, items_ev, meta_ev,
             items_tr, meta_tr, train_vals):
    t0 = time.time()
    if grp == "INJ-OFF":
        st = D90.make_stack(gamma=GAMMA_INJ_OFF, no_write=True, seed=seed)
        losses, viol, diverged = [], 0, False
    else:
        st = D90.make_stack(gamma=GAMMA114, no_write=True, seed=seed)
        losses, viol, diverged = [], 0, False
        if int(dose) > 0:
            losses, viol, diverged, _gt = train_both(
                model, st, tok, facts, tr, random.Random(1000 + seed % 100), dose)
    # ★ `DEF-109-02`/`DEF-95-01`：训练末尾会把 no_write 还原为 False ⇒ 评测前必须重封
    for m in st.mems.values():
        m.no_write = True
    assert all(m.no_write for m in st.mems.values()), "评测前必须重封 no_write"
    wc0 = {li: m.write_calls for li, m in st.mems.items()}

    held = eval_set(model, st, items_ev, meta_ev, train_vals)
    tra = eval_set(model, st, items_tr, meta_tr, train_vals)

    wc1 = {li: m.write_calls for li, m in st.mems.items()}
    eval_writes = int(sum(wc1[li] - wc0[li] for li in wc0))
    finite = all(bool(torch.isfinite(m.Wq).all()) and bool(torch.isfinite(m.Wo).all())
                 for m in st.mems.values())
    out = {"grp": grp, "dose": int(dose), "seed": seed, "src_stage": STAGE_TAG[0],
           "acc_held": held["acc_item"], "acc_held_q": held["acc_q"],
           "acc_train": tra["acc_q"], "acc_train_item": tra["acc_item"],
           "n_q_train": tra["n_q"], "n_held_item": len(items_ev),
           "final_loss": float(np.mean(losses[-50:])) if losses else None,
           "gamma_final": gammas(st), "violations": viol, "diverged": bool(diverged),
           "eval_writes": eval_writes, "finite_params": finite,
           "mech_held": held["mech"], "mech_train": tra["mech"],
           "item_ok_held": held["item_ok"],
           "sec": round(time.time() - t0, 1)}
    print(f"  [{grp} d={dose} seed={seed}] acc_held={out['acc_held']:.4f} "
          f"acc_train(q)={out['acc_train']:.4f} "
          f"loss={out['final_loss'] if out['final_loss'] is None else round(out['final_loss'], 4)} "
          f"{'DIV ' if diverged else ''}{out['sec']}s", flush=True)
    st.remove()
    return out


# ================================================================ 计划
def plan_keys():
    out = []
    for s in SEEDS:
        for g in GRP_PRIMARY:
            for d in DOSES:
                out.append((g, d, s))
        for d in LEGACY_DOSES:
            out.append(("LEGACY", d, s))
        out.append(("INJ-OFF", 0, s))
    return out


def ck(g, d, s):
    return f"{g}|{d}|{s}"


# ================================================================ 事实自检（不加载模型）
def factcheck(tok):
    print("=== 事实与划分自检（不加载模型）===", flush=True)
    facts = build_facts_pairs(tok)
    parts = partitions(facts)
    ltr, lev = legacy_partition()
    RES["partitions"] = {}
    allok = True
    for g, (tr, ev) in list(parts.items()):
        fr, seen, n, unseen = class_seen_frac(facts, tr, ev)
        RES["partitions"][g] = {"n_train": len(tr), "n_eval": len(ev),
                                "class_seen_frac_held": fr, "seen": seen, "n_held": n,
                                "unseen_held_values": unseen}
        print(f"  {g:10s} train={len(tr)} held={len(ev)} "
              f"class_seen_frac_held={fr:.4f} ({seen}/{n}) 未见值={unseen}", flush=True)
    # LEGACY 用**自己的** facts
    lfacts = D90.build_facts(tok)
    fr, seen, n, unseen = class_seen_frac(lfacts, ltr, lev)
    RES["partitions"]["LEGACY"] = {"n_train": len(ltr), "n_eval": len(lev),
                                   "class_seen_frac_held": fr, "seen": seen, "n_held": n,
                                   "unseen_held_values": unseen}
    print(f"  {'LEGACY':10s} train={len(ltr)} held={len(lev)} "
          f"class_seen_frac_held={fr:.4f} ({seen}/{n}) 未见值={unseen}", flush=True)

    # ---- A114-1 ----
    exp = {"COLOCATED": 0.0, "SPLIT": 1.0}
    ok1 = True
    for g, e in exp.items():
        got = RES["partitions"][g]["class_seen_frac_held"]
        ok1 &= abs(got - e) < 1e-12
    ok1 &= RES["partitions"]["LEGACY"]["seen"] == 1 and RES["partitions"]["LEGACY"]["n_held"] == 16
    rec("A114-1", ok1,
        f"覆盖度从实际 facts 算：COLOCATED=0/16, SPLIT=16/16, LEGACY=1/16 "
        f"(实测 {RES['partitions']['COLOCATED']['seen']}/16, "
        f"{RES['partitions']['SPLIT']['seen']}/16, "
        f"{RES['partitions']['LEGACY']['seen']}/16)", value=ok1)
    allok &= ok1

    # ---- A114-2：只差划分 ----
    trc, evc = parts["COLOCATED"]
    trs, evs = parts["SPLIT"]
    vm_c = sorted(facts[i]["val"] for i in evc)
    vm_s = sorted(facts[i]["val"] for i in evs)
    diffs = []
    # 值多重集：**全体 32 条**事实完全相同（同一对象）⇒ 天然相同，仍机械核
    if sorted(f["val"] for f in facts) != sorted(f["val"] for f in facts):
        diffs.append("值多重集不同(不应发生)")
    if len(set(f["val"] for f in facts)) != N_VAL:
        diffs.append(f"类数 != {N_VAL}")
    if len(facts) != N_FACT:
        diffs.append(f"事实数 != {N_FACT}")
    # 两臂的留出侧必须**大小相同**（16 vs 16）
    if len(evc) != len(evs):
        diffs.append(f"留出侧大小不同 {len(evc)} vs {len(evs)}")
    # 每臂的**留出值集**各自应与「训练值集」呈预期的交叠结构
    if set(facts[i]["val"] for i in evc) & set(facts[i]["val"] for i in trc):
        diffs.append("COLOCATED 留出值与训练值有交（应无交）")
    if set(facts[i]["val"] for i in evs) != set(facts[i]["val"] for i in trs):
        diffs.append("SPLIT 留出值集 != 训练值集（应相等）")

    # ★ 测量层等价性：自带的 `make_items_with_vals` 必须**逐项复刻** `D90.make_eval_items`
    #   （否则「测量层未改」就是空话）。比 prompt_ids 与 4 个候选 token 序列。
    mine_ev, _meta_ev = make_items_with_vals(tok, facts, evc, 555)
    ref_ev = D90.make_eval_items(tok, facts, evc, random.Random(555))
    if len(mine_ev) != len(ref_ev):
        diffs.append(f"item 数 {len(mine_ev)} vs D90 {len(ref_ev)}")
    else:
        for x, y in zip(mine_ev, ref_ev):
            if x["key_idx"] != y["key_idx"]:
                diffs.append("key_idx 不同"); break
            if [c[0] for c in x["cand"]] != [c[0] for c in y["cand"]]:
                diffs.append("候选 token 序列不同"); break
            if [c[1] for c in x["cand"]] != [c[1] for c in y["cand"]]:
                diffs.append("正确位标不同"); break
            if x["prompt_ids"] != y["prompt_ids"]:
                diffs.append("prompt_ids 不同"); break
    ok2 = not diffs
    rec("A114-2", ok2,
        f"COLOCATED 与 SPLIT 同事实、同值多重集({len(set(f['val'] for f in facts))} 值)、"
        f"同 item 生成路径（与 D90.make_eval_items 逐项一致，{len(mine_ev)} 项）；"
        f"差异清单={diffs or '空'}", value=diffs or None)
    allok &= ok2

    # ---- A114-3：d=0 逐位复算（需模型，在 Stage A 内做；见 stage_a）----
    rec("A114-3", True, "d=0 逐位复算在 Stage A 内执行（此处只排期，不阻断）",
        value=None, blocking=False)
    return allok


# ================================================================ Stage A
def stage_a(tok, model):
    print("\n=== Stage A 硬门 ===", flush=True)
    facts = build_facts_pairs(tok)
    parts = partitions(facts)
    lfacts = D90.build_facts(tok)
    ltr, lev = legacy_partition()
    chunks, owner = D90.build_chunks(tok, facts)          # ★ 与划分无关（只依赖 facts）
    s0 = SEEDS[0]
    train_vals = {facts[i]["val"] for i in parts["COLOCATED"][0]}

    # A114-1 / A114-2 已在 factcheck 记过，这里不重复跑（沿用结论）
    ok_mech = factcheck(tok)

    # ---- 取 COLOCATED 的 d=0 / d=600 两格（同时充当 smoke 与 A114-4/5/7 的载体）----
    g = "COLOCATED"
    tr, ev = parts[g]
    items_ev, meta_ev = make_items_with_vals(tok, facts, ev, 555)
    items_tr, meta_tr = make_items_with_vals(tok, facts, tr, 777)
    tv = {facts[i]["val"] for i in tr}

    print(f"  取 2 格（{g} d=0 / d=600, seed={s0}）用于 A114-4/5/7", flush=True)
    c0 = run_cell(g, 0, s0, tok, model, facts, tr, ev, items_ev, meta_ev,
                  items_tr, meta_tr, tv)
    c0["src_stage"] = "stageA"
    RES["cells"][ck(g, 0, s0)] = c0
    c6 = run_cell(g, 600, s0, tok, model, facts, tr, ev, items_ev, meta_ev,
                  items_tr, meta_tr, tv)
    c6["src_stage"] = "stageA"
    RES["cells"][ck(g, 600, s0)] = c6

    # ---- A114-3：d=0 逐位复算（同 seed 另建一栈）----
    c0b = run_cell(g, 0, s0, tok, model, facts, tr, ev, items_ev, meta_ev,
                   items_tr, meta_tr, tv)
    same = (c0b["item_ok_held"] == c0["item_ok_held"])
    rec("A114-3", same,
        f"d=0 的 item_ok 向量逐位复算（同 seed 另建一栈）："
        f"{sum(x == y for x, y in zip(c0['item_ok_held'], c0b['item_ok_held']))}/{len(c0['item_ok_held'])} 位相同",
        value=same)

    # ---- A114-4：no_write 重封 + eval_writes == 0 ----
    ok4 = (c0["eval_writes"] == 0 and c6["eval_writes"] == 0)
    rec("A114-4", ok4, f"评测期写入 eval_writes: d0={c0['eval_writes']} d600={c6['eval_writes']}",
        value=ok4)

    # ---- A114-5：damage 与 acc_train 可计算（DEF-113-02 继承：端到端验，不只验"格子在"）----
    have0 = ck(g, 0, s0) in RES["cells"] and ck(g, 600, s0) in RES["cells"]
    nonnull = all(RES["cells"][ck(g, d, s0)]["acc_held"] is not None and
                  RES["cells"][ck(g, d, s0)]["acc_train"] is not None
                  for d in (0, 600))
    dmg = c6["acc_held"] - c0["acc_held"] if have0 and nonnull else None
    ok5 = bool(have0 and nonnull and dmg is not None)
    rec("A114-5", ok5,
        f"COLOCATED d=0/d=600 齐全且 acc_held/acc_train 非 None；"
        f"damage(seed {s0}) = {dmg:+.4f} = {dmg * 48:+.1f}/48" if ok5 else "缺格或 None",
        value=None if dmg is None else round(float(dmg), 4))

    # ---- A114-7：acc_train 自洽 ----
    ok7 = True
    det7 = []
    for d, c in ((0, c0), (600, c6)):
        det7.append(f"d={d}: acc_train={c['acc_train']:.4f} acc_held={c['acc_held']:.4f} "
                    f"acc_train_item={c['acc_train_item']:.4f}")
    # d=0 时应≈chance 量级（不要求精确，只要求"未学"）
    # ★ rev 1.2：此处只是「d=0 未学」的宽松自洽检查（report-only 阈值可用），不是判据
    ok7 &= (c0["acc_train"] is not None and c0["acc_train"] <= GATE_FIT)
    # d=600 时应明显 > 0（与 final_loss → 0 自洽）；★ 这里不预设 0.5（那是判据，不是守卫）
    ok7 &= (c6["acc_train"] is not None and c6["acc_train"] > 0.0)
    rec("A114-7", ok7, "; ".join(det7), value=ok7)

    # ---- A114-8：一次一栈 / finite / 不 diverged ----
    ok8 = all(c["finite_params"] and not c["diverged"] for c in (c0, c6))
    rec("A114-8", ok8, f"finite_params 全 True 且无 diverged: "
                       f"{[ (c['finite_params'], c['diverged']) for c in (c0,c6) ]}", value=ok8)

    # ---- A114-9：plan_n ----
    pk = plan_keys()
    RES["plan_n"] = len(pk)
    miss = [k for k in (ck(g_, d_, s_) for g_, d_, s_ in pk) if k not in RES["cells"]]
    rec("A114-9", True,
        f"plan_n = {len(pk)}（预期 54）；此刻尚缺 {len(miss)} 格（正式轮补齐；缺格 ⇒ INCOMPLETE）",
        value=len(pk))

    # ---- 设备 md5 ----
    RES["device_md5"] = {
        "dir90/R1_task90.py": md5(_ROOT / "dir90" / "R1_task90.py"),
        "dir104/R1_task104.py": md5(_ROOT / "dir104" / "R1_task104.py"),
        "sgmn_module.py": md5(_ROOT / "sgmn_module.py"),
        "dir113/R1_task113.py": md5(_ROOT / "dir113" / "R1_task113.py"),
    }
    print(f"  device md5: {RES['device_md5']}", flush=True)
    return ok_mech and ok4 and ok5 and ok7 and ok8


# ================================================================ 裁决
def verdict():
    cells = RES["cells"]
    pk = plan_keys()
    RES["plan_n"] = len(pk)
    miss = [ck(g, d, s) for g, d, s in pk if ck(g, d, s) not in cells]
    out = {"plan_n": len(pk), "missing": miss[:20], "n_missing": len(miss)}

    # ---- A114-9：缺格 ⇒ INCOMPLETE（优先于全部判决码）----
    if miss:
        out["code"] = "INCOMPLETE"
        out["reason"] = f"plan_n={len(pk)} 但缺 {len(miss)} 格"
        return out

    def dmg(grp, s):
        a = cells[ck(grp, 600, s)]["acc_held"]
        b = cells[ck(grp, 0, s)]["acc_held"]
        if a is None or b is None:
            return None
        return float(a - b)

    D = {g: {s: dmg(g, s) for s in SEEDS} for g in GRP_PRIMARY}
    AT = {g: {s: cells[ck(g, 600, s)]["acc_train"] for s in SEEDS} for g in GRP_PRIMARY}
    AT0 = {g: {s: cells[ck(g, 0, s)]["acc_train"] for s in SEEDS} for g in GRP_PRIMARY}
    out["damage"] = {g: {str(s): (None if v[s] is None else round(v[s], 6))
                         for s in SEEDS} for g, v in D.items()}
    out["acc_train_d600"] = {g: {str(s): (None if AT[g][s] is None else round(AT[g][s], 6))
                                 for s in SEEDS} for g in GRP_PRIMARY}
    out["acc_held"] = {g: {str(s): {d: cells[ck(g, d, s)]["acc_held"] for d in DOSES}
                           for s in SEEDS} for g in GRP_PRIMARY}

    # ---- A114-6：现象必须在装置里现形（卡片 §5 顺序 2）----
    col = D["COLOCATED"]
    ok6 = all(col[s] is not None and col[s] <= -FALSIFY_ITEM for s in SEEDS)
    n_ok6 = sum(1 for s in SEEDS if col[s] is not None and col[s] <= -FALSIFY_ITEM)
    out["A114-6"] = {"ok": ok6, "n_seeds": n_ok6, "of": len(SEEDS),
                     "note": f"COLOCATED damage ≤ −2/48 的 seed 数 = {n_ok6}/{len(SEEDS)}"}
    if not ok6:
        out["code"] = "C114-DEVICE-DRIFT"
        out["reason"] = (f"COLOCATED 只有 {n_ok6}/{len(SEEDS)} 个 seed 再现「训练即损伤」"
                         f"（阈值 damage ≤ −2/48）⇒ 新装置没带上原现象，无法归因")
        return out

    # ---- 主判据 ----
    spl = D["SPLIT"]
    gone = {s: (spl[s] is not None and spl[s] > -FALSIFY_ITEM) for s in SEEDS}   # > −2/48 ⇒ 消失
    # ★ rev 1.2：相对口径（同 seed 配对、同 W 初值）—— 只测「是否仍在学」，不偏袒任何损伤分支
    fit = {s: (AT["SPLIT"][s] is not None and AT0["SPLIT"][s] is not None
               and (AT["SPLIT"][s] - AT0["SPLIT"][s]) >= GATE_FIT_REL - 1e-12)
           for s in SEEDS}
    n_gone, n_fit = sum(gone.values()), sum(fit.values())
    out["SPLIT_damage_gone"] = {"n": n_gone, "of": len(SEEDS),
                                "per_seed": {str(s): gone[s] for s in SEEDS}}
    out["SPLIT_fit"] = {"n": n_fit, "of": len(SEEDS),
                        "per_seed": {str(s): fit[s] for s in SEEDS},
                        "gate_rel": GATE_FIT_REL, "gate_abs_report_only": GATE_FIT,
                        "delta_per_seed": {str(s): (None if AT["SPLIT"][s] is None or AT0["SPLIT"][s] is None
                                                    else round(AT["SPLIT"][s] - AT0["SPLIT"][s], 6))
                                            for s in SEEDS}}

    # 观测的差中差（SPLIT 的 damage 减去 COLOCATED 的 damage）
    did = [spl[s] - col[s] for s in SEEDS if spl[s] is not None and col[s] is not None]
    out["did"] = {"mean": float(np.mean(did)) if did else None,
                  "per_seed": [round(float(x), 6) for x in did],
                  "mde_n6": MDE_DID_N6,
                  "expected_if_full": abs(REF_DAMAGE_D144)}
    if did:
        out["did"]["ratio_to_mde"] = float(abs(np.mean(did)) / MDE_DID_N6)

    # —— 顺序 3：损伤消失 且 仍在学（6/6）
    if n_gone == len(SEEDS) and n_fit == len(SEEDS):
        out["code"] = "C114-COVERAGE-CAUSAL"
        out["reason"] = ("SPLIT（覆盖 1.0）下满剂量 damage 6/6 seed 均已消失（> −2/48），"
                         f"且 acc_train 相对自身未训练基线提升 ≥ {GATE_FIT_REL:.4f}（6/6）"
                         " ⇒ 留出答案值覆盖是「训练即损伤」的成因")
        out["success"] = True
        return out
    # —— 顺序 4：SPLIT 仍损伤（≥1 seed）
    still = [s for s in SEEDS if spl[s] is not None and spl[s] <= -FALSIFY_ITEM]
    if still:
        out["code"] = "C114-COVERAGE-NOT-CAUSAL"
        out["reason"] = (f"COLOCATED 6/6 再现损伤，但 SPLIT 仍有 {len(still)}/{len(SEEDS)} 个 seed "
                         f"damage ≤ −2/48 ⇒ 覆盖不是（全部）成因（否证 F1）")
        out["success"] = False
        return out
    # —— 顺序 5：损伤消失但不再学（≥1 seed）
    notfit = [s for s in SEEDS if not fit[s]]
    out["code"] = "C114-FIX-DEGENERATES"
    out["reason"] = (f"SPLIT 下 damage 已消失，但 acc_train 相对提升 < {GATE_FIT_REL:.4f} 出现在 "
                     f"{len(notfit)}/{len(SEEDS)} 个 seed ⇒ 「不损伤」只由「不学习」换来（否证 F2，rev 1.2 口径）")
    out["success"] = False
    return out


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="factcheck",
                    choices=["factcheck", "A", "smoke", "formal", "verdict"])
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--only", default=None, help="只跑某组，如 COLOCATED")
    a = ap.parse_args()

    if a.stage == "factcheck":
        tok, _ = None, None
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(D90.MODEL)
        ok = factcheck(tok)
        dump(RES, "stage_factcheck.json")
        print(f"\n★ factcheck {'通过' if ok else '失败'}", flush=True)
        return 0 if ok else 1

    tok, model = D90.model_and_tok()
    print(f"模型 {D90.MODEL} on {D90.DEV}；LR={D90.READ_LR}；γ={GAMMA114}", flush=True)

    if a.stage == "A":
        STAGE_TAG[0] = "stageA"
        ok = stage_a(tok, model)
        dump(RES, "stage_stageA.json")
        print(f"\n★ Stage A {'通过' if ok else '未通过 ⇒ 正式轮阻断'}", flush=True)
        return 0 if ok else 1

    # 以下各阶段都需要 A 的结论（Round standard 3）
    sa = OUT / "stage_stageA.json"
    if not sa.exists():
        print("★ 缺 stage_stageA.json ⇒ 不得进正式轮", flush=True)
        return 1
    prev = json.loads(sa.read_text(encoding="utf-8"))
    RES["stageA"] = prev.get("stageA", [])
    RES["device_md5"] = prev.get("device_md5", {})
    RES["partitions"] = prev.get("partitions", {})
    RES["plan_n"] = prev.get("plan_n")
    bad = [x for x in RES["stageA"] if x.get("blocking") and not x.get("ok")]
    if bad:
        print(f"★ Stage A 有 {len(bad)} 项阻断失败 ⇒ 阻断：{bad[:3]}", flush=True)
        return 1
    for k, v in prev.get("cells", {}).items():
        v = dict(v); v["src_stage"] = "stageA"; RES["cells"].setdefault(k, v)
    print(f"Stage A 门通过；继承 {len(prev.get('cells', {}))} 格", flush=True)

    facts = build_facts_pairs(tok)
    parts = partitions(facts)
    lfacts = D90.build_facts(tok)
    ltr, lev = legacy_partition()
    chunks, owner = D90.build_chunks(tok, facts)
    # 每臂的 items 只依赖 (facts, keys, rng) ⇒ 建一次
    IT = {}
    for g in GRP_PRIMARY:
        tr, ev = parts[g]
        IT[g] = {"tr": tr, "ev": ev,
                 "items_ev": make_items_with_vals(tok, facts, ev, 555),
                 "items_tr": make_items_with_vals(tok, facts, tr, 777),
                 "tv": {facts[i]["val"] for i in tr},
                 "facts": facts}
    IT["LEGACY"] = {"tr": ltr, "ev": lev,
                    "items_ev": make_items_with_vals(tok, lfacts, lev, 555),
                    "items_tr": make_items_with_vals(tok, lfacts, ltr, 777),
                    "tv": {lfacts[i]["val"] for i in ltr},
                    "facts": lfacts}
    # INJ-OFF 用 SPLIT 的划分（覆盖已修好的那一侧）—— 卡片 §3.4，report-only
    IT["INJ-OFF"] = IT["SPLIT"]

    if a.stage == "smoke":
        g, d, s = "COLOCATED", 600, SEEDS[0]
        print(f"smoke：1 格（{g} d={d} seed={s}）测时 —— 不出判据", flush=True)
        c = run_cell(g, d, s, tok, model, IT[g]["facts"], IT[g]["tr"], IT[g]["ev"],
                     IT[g]["items_ev"][0], IT[g]["items_ev"][1],
                     IT[g]["items_tr"][0], IT[g]["items_tr"][1], IT[g]["tv"])
        print(f"\n★ smoke 完成：{c['sec']}s/格 ⇒ 54 格 ≈ {c['sec'] * 54 / 60:.1f} min", flush=True)
        dump({"smoke": c}, "stage_smoke.json")
        return 0

    if a.stage == "verdict":
        crit = verdict(); RES["crit"] = crit
        dump(RES, "stage_verdict.json")
        print("\n=== 判据 ===\n" + json.dumps(crit, ensure_ascii=False, indent=1), flush=True)
        return 0

    # ---- formal ----
    STAGE_TAG[0] = "formal"
    pk = plan_keys()
    RES["plan_n"] = len(pk)
    seeds = SEEDS if not a.seeds else tuple(int(x) for x in a.seeds.split(","))
    plan = [(g, d, s) for (g, d, s) in pk if s in seeds and (not a.only or g == a.only)]
    print(f"格计划 {len(pk)} 格（本次执行 {len(plan)} 格）", flush=True)
    for g, d, s in plan:
        k = ck(g, d, s)
        if k in RES["cells"]:
            print(f"  跳过（已有）{k}", flush=True)
            continue
        it = IT[g]
        c = run_cell(g, d, s, tok, model, it["facts"], it["tr"], it["ev"],
                     it["items_ev"][0], it["items_ev"][1],
                     it["items_tr"][0], it["items_tr"][1], it["tv"])
        c["src_stage"] = "formal"
        RES["cells"][k] = c
        dump(RES, "stage_formal.json")
    crit = verdict(); RES["crit"] = crit
    dump(RES, "stage_formal.json")
    dump(RES, "stage_verdict.json")
    print("\n=== 判据 ===\n" + json.dumps(crit, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())