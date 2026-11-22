#!/usr/bin/env python3
"""DIR-115 · R1 —— 残余损伤的机制：注入项对**未见键**有可用正信号吗？（增益剂量曲线）

★ 逐字派生：**直接 import** `dir114/R1_task114.py`（`P`），复用其
  `build_facts_pairs` / `partitions` / `class_seen_frac` / `make_items_with_vals` / `train_both`。
  ★ 为何是 import 而不是抄一遍：`R1_task90`/`R1_task104` 按模块名缓存 ⇒ 两者拿到**同一个**
  `D90`/`D104` 对象 ⇒ `READ_STEPS`/`READ_LR`/`make_stack`/`train_read_frozen`/`score_matrix`
  与父装置**位同**。抄写会漂移，引用不会。唯一陷阱：import `P` 会把 `D90.OUT` 指到 `dir114/out`，
  故下方立刻重指回 `dir115/out`。

按卡片 `revolve/DIR-115/DIRECTION_rev115_v10.md`（**rev 1.0 冻结**）§10 的允许改动范围：
  ① **新增 γ 覆写层**（8 点网格，评测期覆写，不重训）
  ② **新增翻转分解** `D(γ)` / `C(γ)` + 恒等式断言
  ③ **新增 ORACLE 接线守卫**（`install_ceil`，只验接线、不断言效应量）
  ④ **去掉** `LEGACY` 桥接臂与 `acc_train`/`mech` 块（本卡不判学习/覆盖，父卡已判）
除此之外装置一字未改。

阶段（`--stage`）：
  factcheck  —— 只查事实/划分/覆盖度 + item 同一（**不加载模型**，秒级）
  A          —— Stage A 硬门（`A115-1..10`）
  smoke      —— 一格测时（**不出判据**）
  formal     —— 正式轮 18 格（可断点续跑）
  verdict    —— 只重算判据（不跑格）
"""
import argparse
import json
import random
import sys
import pathlib
import time

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _ROOT / "dir90", _ROOT / "dir104", _ROOT / "dir114"):
    sys.path.insert(0, str(p))

import R1_task90 as D90                                     # noqa: E402
import R1_task104 as D104                                   # noqa: E402
import R1_task114 as P                                      # noqa: E402  ← 父装置（逐字复用）

OUT = _HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
D90.OUT = OUT                                               # ★ 覆盖 P 在 import 时的重指
P.OUT = OUT

# ---------------------------------------------------------------- 冻结常量（卡片 §2.3 / §3.2）
GAMMA_TR = 0.0010                     # 父装置的增益；实测**未被训练**（卡片 §3.4）
GRID = (0.0, 6.25e-05, 1.25e-04, 2.5e-04, 5e-04, 1.0e-03, 2.0e-03, 4.0e-03)   # 卡片 §3.2
N_GRID = len(GRID)
SEEDS = (20271126, 20271203, 20271210, 20271217, 20271224, 20271231)          # 同父卡（重训即守卫）
N_SEEDS = len(SEEDS)

ARM_PRIMARY = "SPLIT"                 # 只有它覆盖 1.0 ⇒ 只有它能判（卡片 §3.5）
ARMS_TRAINED = ("SPLIT", "COLOCATED")
DOSE_TRAIN = 600
DOSE_NONE = 0

# ---- 卡片 §2.3 冻结阈值 ----
ACC0_SPLIT_N = 17                     # SPLIT 的 γ=0 先验正确项数（= INJ-OFF 实测）
N_ITEM = 48
TOL_CONFIRM = 1.0 / 48.0              # 确认容差：mean6 acc <= ACC0 + 1/48 = 18/48
F1_A_N = 18                           # 否证 (a)：6/6 seed acc >= 18/48
F1_B_N = 19                           # 否证 (b)：mean6 >= 19/48
ORACLE_MULTS = (0.25, 1.0, 4.0)

# ---- 父装置在 γ=γ_tr, d=600 的实测值（A115-3 位同复现目标）----
EXPECT_TR = {"SPLIT": (13, 15, 11, 17, 12, 10), "COLOCATED": (2, 1, 1, 4, 6, 1)}

RES = {"card": "DIRECTION_rev115_v10.md", "grid": list(GRID), "seeds": list(SEEDS),
       "gamma_tr": GAMMA_TR, "cells": {}, "plan_n": None}


def dump(o, name):
    (OUT / name).write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")


def rec(name, ok, detail, value=None, blocking=True):
    RES.setdefault("stageA", []).append(
        {"name": name, "ok": bool(ok), "detail": detail, "value": value, "blocking": bool(blocking)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return bool(ok)


def ck(arm, dose, seed):
    return f"{arm}|{int(dose)}|{seed}"


def plan_keys():
    out = []
    for arm in ARMS_TRAINED:
        for s in SEEDS:
            out.append((arm, DOSE_TRAIN, s))
    for s in SEEDS:                       # ★ 关键对照：SPLIT 未训练（卡片 §3.1）
        out.append((ARM_PRIMARY, DOSE_NONE, s))
    return out


# ================================================================ γ 覆写层
def set_gamma(st, g):
    """评测期覆写注入增益。★ 装置里 γ 只有一处前向用处（`sgmn_module.py:161`）。"""
    for m in st.mems.values():
        with torch.no_grad():
            m.gamma.fill_(float(g))


def read_gamma(st):
    return {li: float(m.gamma) for li, m in st.mems.items()}


def acc_and_ok(model, st, items, tgt):
    """唯一评测入口（`D104.score_matrix` ⇒ `acc_item_of` / `item_ok`）。不含 mech（省时）。"""
    S = D104.score_matrix(model, st, items)
    return float(D104.acc_item_of(S, tgt)), D104.item_ok(S, tgt)


# ================================================================ 翻转分解（报告层）
def flips(prior_ok, cur_ok):
    """D = 先验对→现在错 ; C = 先验错→现在对。恒等式: acc*48 == prior_n - D + C。"""
    pc = [i for i, v in enumerate(prior_ok) if v == 1]
    pw = [i for i, v in enumerate(prior_ok) if v == 0]
    D = sum(1 for i in pc if cur_ok[i] == 0)
    C = sum(1 for i in pw if cur_ok[i] == 1)
    return int(D), int(C)


# ================================================================ 一格
def run_cell115(arm, dose, seed, tok, model, facts, tr, ev, items_ev, train_vals,
                want_oracle=False):
    t0 = time.time()
    st = D90.make_stack(gamma=GAMMA_TR, no_write=True, seed=seed)
    losses, viol, diverged = [], 0, False
    if int(dose) > 0:
        losses, viol, diverged, _gt = P.train_both(
            model, st, tok, facts, tr, random.Random(1000 + seed % 100), dose)
    # ★ `DEF-109-02`：训练末尾会把 no_write 还原 ⇒ 评测前必须重封（父卡同款）
    for m in st.mems.values():
        m.no_write = True
    assert all(m.no_write for m in st.mems.values()), "评测前必须重封 no_write"

    wc0 = {li: m.write_calls for li, m in st.mems.items()}
    W0 = {li: D90.W_of(m).detach().clone() for li, m in st.mems.items()}
    tgt = D104.targets_of(items_ev)
    g0 = read_gamma(st)

    acc_by_gamma, ok_by_gamma, inj_norm = {}, {}, {}
    for g in GRID:
        set_gamma(st, g)
        a, ok = acc_and_ok(model, st, items_ev, tgt)
        acc_by_gamma[f"{g:.5e}"] = a
        ok_by_gamma[f"{g:.5e}"] = [int(x) for x in ok]
        inj_norm[f"{g:.5e}"] = float(abs(g))     # ‖γ·m‖ ∝ |γ|·‖m‖；‖m‖ 单独记见下
    # γ=0 的注入项必须是**精确 0**（卡片 §2.4 / A115-2）
    zero_key = f"{0.0:.5e}"
    ok_prior = ok_by_gamma[zero_key]
    prior_n = int(sum(ok_prior))

    D_by_gamma, C_by_gamma, ident_ok = {}, {}, True
    for g in GRID:
        k = f"{g:.5e}"
        D, C = flips(ok_prior, ok_by_gamma[k])
        D_by_gamma[k], C_by_gamma[k] = D, C
        if prior_n - D + C != int(sum(ok_by_gamma[k])):
            ident_ok = False

    # ---- ORACLE 接线（只对 SPLIT d=600；卡片 §6-3 / A115-10）----
    oracle = None
    if want_oracle:
        emb = model.get_input_embeddings().weight.detach()
        set_gamma(st, GAMMA_TR)
        a_norm, _ok_norm = acc_and_ok(model, st, items_ev, tgt)
        S_norm = D104.score_matrix(model, st, items_ev)
        D104.install_ceil(model, st, emb)
        oa, onrm = {}, {}
        assert all(it["cand"] for it in items_ev)
        for mult in ORACLE_MULTS:
            ao, _o = 0, None
            S = D104.score_matrix(model, st, items_ev, ceil_mult=mult)
            ao = float(D104.acc_item_of(S, tgt))
            oa[f"{mult:g}"] = ao
            # 有效注入强度 = γ·mult·‖emb[t]‖，逐项取正确候选
            nrm = []
            for it in items_ev:
                j = next(j for j, (_, isc) in enumerate(it["cand"]) if isc)
                tokid = int(it["cand"][j][0][0])
                nrm.append(float(torch.linalg.norm(emb[tokid])) * abs(GAMMA_TR) * mult)
            onrm[f"{mult:g}"] = float(np.mean(nrm))
        S_oracle_m1 = D104.score_matrix(model, st, items_ev, ceil_mult=1.0)
        D104.uninstall_probe(st, "read")
        S_back = D104.score_matrix(model, st, items_ev)
        oracle = {"acc": oa, "inj_norm": onrm,
                  "wired_differs": bool(not np.array_equal(S_norm, S_oracle_m1)),
                  "restore_bitwise": bool(np.array_equal(S_norm, S_back)),
                  "acc_normal": a_norm}

    wc1 = {li: m.write_calls for li, m in st.mems.items()}
    eval_writes = int(sum(wc1[li] - wc0[li] for li in wc0))
    finite = all(bool(torch.isfinite(m.Wq).all()) and bool(torch.isfinite(m.Wo).all())
                 for m in st.mems.values())
    W_same = {li: bool(torch.equal(W0[li], D90.W_of(m).detach())) for li, m in st.mems.items()}

    out = {"arm": arm, "dose": int(dose), "seed": seed,
           "acc_by_gamma": acc_by_gamma, "item_ok_by_gamma": ok_by_gamma,
           "D_by_gamma": D_by_gamma, "C_by_gamma": C_by_gamma,
           "prior_n": prior_n, "ident_ok": bool(ident_ok),
           "gamma_after_writes": read_gamma(st), "gamma_before": g0,
           "final_loss": float(np.mean(losses[-50:])) if losses else None,
           "violations": viol, "diverged": bool(diverged),
           "eval_writes": eval_writes, "finite_params": finite, "W_unchanged": W_same,
           "oracle": oracle, "n_item": len(items_ev),
           "sec": round(time.time() - t0, 1)}
    print(f"  [{arm} d={dose} seed={seed}] " +
          " ".join(f"g={float(k):.2e}:{v:.4f}" for k, v in acc_by_gamma.items()) +
          f"  (prior {prior_n}/48) {out['sec']}s", flush=True)
    st.remove()
    return out


# ================================================================ 事实/划分
def ground(tok):
    facts = P.build_facts_pairs(tok)
    parts = P.partitions(facts)
    csf = {}
    for g in ARMS_TRAINED:
        tr, ev = parts[g]
        frac, seen, n, held = P.class_seen_frac(facts, tr, ev)
        csf[g] = {"frac": frac, "seen": seen, "n": n}
    IT = {}
    for g in ARMS_TRAINED:
        tr, ev = parts[g]
        items_ev, meta_ev = P.make_items_with_vals(tok, facts, ev, 555)
        IT[g] = {"tr": tr, "ev": ev, "items_ev": items_ev, "meta_ev": meta_ev,
                 "tv": {facts[i]["val"] for i in tr}, "facts": facts}
    return facts, parts, csf, IT


def factcheck(tok):
    facts, parts, csf, IT = ground(tok)
    ok = True
    ok &= rec("A115-1a", abs(csf["SPLIT"]["frac"] - 1.0) < 1e-12,
              f"SPLIT class_seen_frac_held={csf['SPLIT']['frac']:.4f}（须 1.0）"
              f" 来自实际 build_facts 输出")
    ok &= rec("A115-1b", abs(csf["COLOCATED"]["frac"] - 0.0) < 1e-12,
              f"COLOCATED class_seen_frac_held={csf['COLOCATED']['frac']:.4f}（须 0.0）")
    n_ev = {g: len(IT[g]["items_ev"]) for g in ARMS_TRAINED}
    ok &= rec("A115-6a", all(v == N_ITEM for v in n_ev.values()),
              f"各臂留出 item 数 = {n_ev}（须 {N_ITEM}）")
    # item 同一：与父卡 D90.make_eval_items 逐项核
    ref = D90.make_eval_items(tok, facts, IT["SPLIT"]["ev"], random.Random(555))
    mine = IT["SPLIT"]["items_ev"]
    same = (len(ref) == len(mine) and all(
        r["prompt_ids"] == m["prompt_ids"]
        and [c[0] for c in r["cand"]] == [c[0] for c in m["cand"]]
        for r, m in zip(ref, mine)))
    ok &= rec("A115-6b", same, f"SPLIT 留出 item 与 D90.make_eval_items 逐项一致（{len(mine)} 项）")
    RES["coverage"] = csf
    RES["plan_n"] = len(plan_keys())
    print(f"\n★ factcheck {'通过' if ok else '失败'}；plan_n={RES['plan_n']}", flush=True)
    return ok


# ================================================================ Stage A
def stage_a(tok, model):
    facts, parts, csf, IT = ground(tok)
    ok = True
    ok &= factcheck(tok)

    # ★ `DEF-115-04`：Stage A 必须可续跑 —— 否则"只重评守卫"会重训 12 格（≈25 min）
    prev = OUT / "stage_stageA.json"
    if prev.exists():
        pv = json.loads(prev.read_text(encoding="utf-8"))
        for k, v in pv.get("cells", {}).items():
            RES["cells"].setdefault(k, v)
        print(f"  续跑：继承已有 {len(RES['cells'])} 格", flush=True)

    # 需要真跑的格：12 训练格 + 6 未训练格（A115-2/3/5 需要全 seed）
    RES["plan_n"] = len(plan_keys())
    for arm, dose, seed in plan_keys():
        k = ck(arm, dose, seed)
        if k in RES["cells"]:
            continue
        it = IT[arm]
        RES["cells"][k] = run_cell115(
            arm, dose, seed, tok, model, it["facts"], it["tr"], it["ev"],
            it["items_ev"], it["tv"], want_oracle=(arm == "SPLIT" and dose == 600))
        dump(RES, "stage_stageA.json")
    C = RES["cells"]
    zero_key = f"{0.0:.5e}"

    # A115-2 锚点：SPLIT γ=0 ⇒ 17/48 且 6/6 逐项相同
    zs = [C[ck("SPLIT", d, s)]["item_ok_by_gamma"][zero_key] for d in (DOSE_NONE,) for s in SEEDS]
    zs += [C[ck("SPLIT", DOSE_TRAIN, s)]["item_ok_by_gamma"][zero_key] for s in SEEDS]
    pn = [int(sum(z)) for z in zs]
    ok &= rec("A115-2a", all(n == ACC0_SPLIT_N for n in pn),
              f"SPLIT γ=0 先验正确项数 = {pn}（须全为 {ACC0_SPLIT_N}）")
    ok &= rec("A115-2b", all(z == zs[0] for z in zs),
              "SPLIT γ=0 的 item_ok 在 6 seed × {d=0,d=600} 共 12 格上逐项相同"
              "（机械证明 γ=0 把注入压成精确 0）")

    # A115-5：COLOCATED 自己的 ACC0，不得搬用 17/48
    cz = [C[ck("COLOCATED", DOSE_TRAIN, s)]["item_ok_by_gamma"][zero_key] for s in SEEDS]
    cpn = [int(sum(z)) for z in cz]
    ok &= rec("A115-5", all(z == cz[0] for z in cz) and all(n == cpn[0] for n in cpn),
              f"COLOCATED 自有 γ=0 锚点 = {cpn[0]}/48（6/6 逐项相同）；与 17/48 "
              f"{'相同（可疑，须复核）' if cpn[0] == ACC0_SPLIT_N else '不同 ⇒ 锚点划分相关，未跨臂搬用'}")
    RES["acc0_colocated_n"] = cpn[0]

    # A115-3 位同复现：γ=γ_tr, d=600 逐 seed 等于父卡
    gtr = f"{GAMMA_TR:.5e}"
    rep = {}
    for arm in ARMS_TRAINED:
        got = tuple(int(round(C[ck(arm, DOSE_TRAIN, s)]["acc_by_gamma"][gtr] * N_ITEM)) for s in SEEDS)
        rep[arm] = {"expect": EXPECT_TR[arm], "got": got, "ok": got == EXPECT_TR[arm]}
    ok &= rec("A115-3", all(v["ok"] for v in rep.values()),
              f"位同复现父卡：{ {k: (v['got'], v['ok']) for k, v in rep.items()} }")
    RES["reproduce"] = rep

    # A115-4 γ 覆写生效
    cell = C[ck("SPLIT", DOSE_TRAIN, SEEDS[0])]
    diffs = [abs(cell["acc_by_gamma"][f"{g:.5e}"] - cell["acc_by_gamma"][zero_key]) for g in GRID[1:]]
    ok &= rec("A115-4b", any(d > 0 for d in diffs),
              f"γ>0 与 γ=0 的 acc 有差异（max Δ={max(diffs):.4f}）")
    # ★ `DEF-115-03`：必须读 `gamma_before`（训练后、**任何覆写之前**），
    #    不是 `gamma_after_writes`（那是本卡自己扫完后的最后一个网格值 4e-3）。
    gvals = set()
    for k in C:
        gvals.update(round(float(v), 12) for v in C[k]["gamma_before"].values())
    ok &= rec("A115-4c", len(gvals) == 1 and abs(next(iter(gvals)) - GAMMA_TR) < 1e-9,
              f"训练后（覆写前）γ 的唯一取值集 = {gvals}（须恒为 {GAMMA_TR}）"
              f" ⇒ γ 确未被训练，网格是绝对的"
              f"；对照：覆写后为 {sorted({round(float(v),6) for k in C for v in C[k]['gamma_after_writes'].values()})}")

    # A115-7 恒等式
    ident = all(c["ident_ok"] for c in C.values())
    ok &= rec("A115-7", ident, "全部格全部 γ 满足 acc·48 == prior_n − D + C")

    # A115-8 装置不变量
    inv = all(c["eval_writes"] == 0 and c["violations"] == 0 and c["finite_params"]
              and not c["diverged"] and all(c["W_unchanged"].values()) for c in C.values())
    bad = [k for k, c in C.items() if not (c["eval_writes"] == 0 and c["violations"] == 0
                                          and c["finite_params"] and not c["diverged"]
                                          and all(c["W_unchanged"].values()))]
    ok &= rec("A115-8", inv, f"eval_writes==0 / violations==0 / finite / 未发散 / W 逐位不变；坏格={bad[:3]}")

    # A115-9 确定性（同一格同一 γ 连测两次）
    it = IT["SPLIT"]
    st = D90.make_stack(gamma=GAMMA_TR, no_write=True, seed=SEEDS[0])
    for m in st.mems.values():
        m.no_write = True
    tgt = D104.targets_of(it["items_ev"])
    set_gamma(st, GAMMA_TR)
    a1, o1 = acc_and_ok(model, st, it["items_ev"], tgt)
    a2, o2 = acc_and_ok(model, st, it["items_ev"], tgt)
    st.remove()
    ok &= rec("A115-9", a1 == a2 and np.array_equal(o1, o2),
              f"同格同 γ 连测两次逐位相同（acc={a1:.4f}）")

    # A115-10 ORACLE 接线
    o = C[ck("SPLIT", DOSE_TRAIN, SEEDS[0])]["oracle"]
    ratios = [o["inj_norm"]["4"] / o["inj_norm"]["0.25"]] if o else [0]
    ok &= rec("A115-10a", bool(o and o["wired_differs"]), "install_ceil 后 score_matrix 与正常路径逐位不同")
    ok &= rec("A115-10b", abs(ratios[0] - 16.0) < 1e-6, f"有效注入范数 mult 1/4→4 之比 = {ratios[0]:.6f}（须 16）")
    ok &= rec("A115-10c", bool(o and o["restore_bitwise"]), "uninstall_probe 后逐位复原正常路径")

    RES["device_md5"] = {
        "sgmn_module.py": P.md5(_ROOT / "sgmn_module.py"),
        "dir90/R1_task90.py": P.md5(_ROOT / "dir90/R1_task90.py"),
        "dir104/R1_task104.py": P.md5(_ROOT / "dir104/R1_task104.py"),
        "dir114/R1_task114.py": P.md5(_ROOT / "dir114/R1_task114.py"),
        "dir115/R1_task115.py": P.md5(_HERE / "R1_task115.py"),
    }
    dump(RES, "stage_stageA.json")
    return bool(ok)


# ================================================================ 判决
def verdict():
    C = RES["cells"]
    need = [ck(a, d, s) for a, d, s in plan_keys()]
    missing = [k for k in need if k not in C]
    out = {"plan_n": len(need), "n_have": len(need) - len(missing), "n_missing": len(missing)}
    # 每格 8 个 γ 齐全？
    short = [k for k in need if k in C and len(C[k]["acc_by_gamma"]) != N_GRID]
    out["n_short"] = len(short)
    if missing or short:
        out["code"] = "C115-INCOMPLETE"
        out["reason"] = f"缺格 {missing[:3]}；γ 读数不全 {short[:3]} ⇒ 不得记 PASS/FAIL"
        out["success"] = False
        return out
    bad = [x for x in RES.get("stageA", []) if x.get("blocking") and not x.get("ok")]
    if bad:
        out["code"] = "C115-DEVICE-DRIFT"
        out["reason"] = f"Stage A 阻断项失败：{[b['name'] for b in bad]}"
        out["success"] = False
        return out

    def acc(arm, dose, s, g):
        return C[ck(arm, dose, s)]["acc_by_gamma"][f"{g:.5e}"]

    # 主判据只在 SPLIT d=600
    mean, per_seed, Dm, Cm = {}, {}, {}, {}
    for g in GRID[1:]:
        vals = [acc("SPLIT", DOSE_TRAIN, s, g) * N_ITEM for s in SEEDS]
        mean[g] = float(np.mean(vals))
        per_seed[g] = [float(v) for v in vals]
        Dm[g] = float(np.mean([C[ck("SPLIT", DOSE_TRAIN, s)]["D_by_gamma"][f"{g:.5e}"] for s in SEEDS]))
        Cm[g] = float(np.mean([C[ck("SPLIT", DOSE_TRAIN, s)]["C_by_gamma"][f"{g:.5e}"] for s in SEEDS]))
    acc0_n = float(C[ck("SPLIT", DOSE_TRAIN, SEEDS[0])]["prior_n"])
    out["acc0_n"] = acc0_n
    out["mean_by_gamma_n48"] = {f"{g:.2e}": round(mean[g], 4) for g in GRID[1:]}
    out["per_seed_by_gamma_n48"] = {f"{g:.2e}": [round(x, 2) for x in per_seed[g]] for g in GRID[1:]}
    out["D_mean"] = {f"{g:.2e}": round(Dm[g], 3) for g in GRID[1:]}
    out["C_mean"] = {f"{g:.2e}": round(Cm[g], 3) for g in GRID[1:]}

    # F1：存在 γ 使 6/6 seed ≥18/48 且 mean ≥19/48
    f1_hits = []
    band = []
    for g in GRID[1:]:
        all6 = all(v >= F1_A_N for v in per_seed[g])
        if all6 and mean[g] >= F1_B_N:
            f1_hits.append(g)
        if (F1_A_N - 1e-9) <= mean[g] < F1_B_N:
            band.append(g)
    out["F1_hits"] = [f"{g:.2e}" for g in f1_hits]
    out["band_gammas"] = [f"{g:.2e}" for g in band]
    out["confirm_all_mean_le_18"] = all(mean[g] <= 18.0 + 1e-9 for g in GRID[1:])
    # 单调性（report-only，卡片 §6-1；★ 不作判据，见 §8 草案 B）
    seq = [mean[g] for g in GRID[1:]]
    out["monotone_nonincreasing"] = all(seq[i + 1] <= seq[i] + 1e-9 for i in range(len(seq) - 1))
    out["mean_argmax_gamma"] = f"{GRID[1:][int(np.argmax(seq))]:.2e}"

    # 报告：d=0 对照 + 曲线形状
    out["d0_vs_d600"] = {}
    for g in GRID[1:]:
        v0 = [acc("SPLIT", DOSE_NONE, s, g) * N_ITEM for s in SEEDS]
        out["d0_vs_d600"][f"{g:.2e}"] = {"d0_mean": round(float(np.mean(v0)), 3),
                                         "d600_mean": round(mean[g], 3)}
    out["d600_vs_d0_gap"] = {f"{g:.2e}": round(mean[g] - float(np.mean(
        [acc("SPLIT", DOSE_NONE, s, g) * N_ITEM for s in SEEDS])), 3) for g in GRID[1:]}
    out["colocated_report_only"] = {
        f"{g:.2e}": {"mean": round(float(np.mean([acc("COLOCATED", DOSE_TRAIN, s, g) * N_ITEM for s in SEEDS])), 3)}
        for g in GRID[1:]}
    oc = C[ck("SPLIT", DOSE_TRAIN, SEEDS[0])]["oracle"]
    if oc:
        out["oracle_report_only"] = {"acc_by_mult": oc["acc"],
                                     "acc_normal_at_gtr": round(oc["acc_normal"], 4)}

    if f1_hits:
        out["code"] = "C115-INJECTION-HAS-SIGNAL"
        out["reason"] = (f"F1 命中：γ={[f'{g:.2e}' for g in f1_hits]} 处 6/6 seed acc≥18/48 且 mean≥19/48 "
                         f"⇒ 存在可用正信号 ⇒ H115 否证")
        out["success"] = False
    elif all(mean[g] <= 18.0 + 1e-9 for g in GRID[1:]):
        out["code"] = "C115-INJECTION-NO-POSITIVE-SIGNAL"
        out["reason"] = (f"全部 7 个 γ∈(0,4e-3] 的 mean6 acc ≤ 18/48（= 先验 {acc0_n:.0f}/48 + 1 项容差）"
                         f" ⇒ 注入项对未见键无可用正信号 ⇒ H115 证实")
        out["success"] = True
    else:
        out["code"] = "C115-INCONCLUSIVE-BAND"
        out["reason"] = f"有 γ 落在 [18/48,19/48) 但不满足 F1：{[f'{g:.2e}' for g in band]} ⇒ 无判决"
        out["success"] = False
    return out


# ================================================================ 入口
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="factcheck",
                    choices=["factcheck", "A", "smoke", "formal", "verdict"])
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--only", default=None, help="只跑某臂，如 SPLIT / COLOCATED")
    a = ap.parse_args()

    if a.stage == "factcheck":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(D90.MODEL)
        ok = factcheck(tok)
        dump(RES, "stage_factcheck.json")
        return 0 if ok else 1

    tok, model = D90.model_and_tok()
    print(f"模型 {D90.MODEL} on {D90.DEV}；LR={D90.READ_LR}；γ_tr={GAMMA_TR}；网格 {N_GRID} 点", flush=True)

    if a.stage == "A":
        ok = stage_a(tok, model)
        print(f"\n★ Stage A {'通过' if ok else '未通过 ⇒ 正式轮阻断'}", flush=True)
        return 0 if ok else 1

    sa = OUT / "stage_stageA.json"
    if not sa.exists():
        print("★ 缺 stage_stageA.json ⇒ 不得进正式轮", flush=True)
        return 1
    prev = json.loads(sa.read_text(encoding="utf-8"))
    for k in ("stageA", "device_md5", "coverage", "reproduce", "acc0_colocated_n"):
        if k in prev:
            RES[k] = prev[k]
    for k, v in prev.get("cells", {}).items():
        RES["cells"].setdefault(k, v)
    bad = [x for x in RES.get("stageA", []) if x.get("blocking") and not x.get("ok")]
    if bad:
        print(f"★ Stage A 有 {len(bad)} 项阻断失败 ⇒ 阻断：{[b['name'] for b in bad]}", flush=True)
        return 1
    print(f"Stage A 门通过；继承 {len(RES['cells'])} 格", flush=True)

    facts, parts, csf, IT = ground(tok)

    if a.stage == "smoke":
        arm, dose, seed = "SPLIT", DOSE_TRAIN, SEEDS[0]
        it = IT[arm]
        c = run_cell115(arm, dose, seed, tok, model, it["facts"], it["tr"], it["ev"],
                        it["items_ev"], it["tv"], want_oracle=True)
        print(f"\n★ smoke：{c['sec']}s/格 ⇒ 18 格 ≈ {c['sec'] * 18 / 60:.1f} min", flush=True)
        dump({"smoke": c}, "stage_smoke.json")
        return 0

    if a.stage == "verdict":
        crit = verdict()
        RES["crit"] = crit
        dump(RES, "stage_verdict.json")
        print("\n=== 判据 ===\n" + json.dumps(crit, ensure_ascii=False, indent=1), flush=True)
        return 0

    pk = plan_keys()
    RES["plan_n"] = len(pk)
    seeds = SEEDS if not a.seeds else tuple(int(x) for x in a.seeds.split(","))
    plan = [(g, d, s) for (g, d, s) in pk if s in seeds and (not a.only or g == a.only)]
    print(f"格计划 {len(pk)} 格（本次执行 {len(plan)} 格）", flush=True)
    for arm, dose, seed in plan:
        k = ck(arm, dose, seed)
        if k in RES["cells"]:
            print(f"  跳过（已有）{k}", flush=True)
            continue
        it = IT[arm]
        RES["cells"][k] = run_cell115(arm, dose, seed, tok, model, it["facts"], it["tr"],
                                      it["ev"], it["items_ev"], it["tv"],
                                      want_oracle=(arm == "SPLIT" and dose == DOSE_TRAIN))
        dump(RES, "stage_formal.json")
    crit = verdict()
    RES["crit"] = crit
    dump(RES, "stage_formal.json")
    dump(RES, "stage_verdict.json")
    print("\n=== 判据 ===\n" + json.dumps(crit, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())