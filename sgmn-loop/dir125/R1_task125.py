#!/usr/bin/env python3
"""`R1_task125.py` —— `DIR-125` rev 1.0（冻结）实现。

★ 逐字派生：**直接 import** `dir119/R1_task119.py`（它 → `dir118` → `dir117` → … → `dir90`）。
★ **声明的改动只有卡片 §7 那 6 条**，其余（训练、增益网格、超参、题集、seed、
  `N_CAND`、`rng_seed`、测量层的 `score_and_mech` 调用序列）**一字不动**：
    ①② `DOSES` 换成新剂量 + `c2` 在所有剂量构建（去 `dose==600` scope）
    ③ stash↔mult 映射断言**无条件执行**（去 `dose in (0,600)` 守卫）
    ④ `OUT` 改指 `dir125/out`（`DEF-106-06` 的正面预防）
    ⑤ 求解器放宽域（在 `mech125` 里，不在本文件）
    ⑥ `dose ∈ {0,600}` 的既有 cell **只读引用**

★ 为什么 `run_cell` / `cells_main` 要**复制**而不是 monkey-patch：
  `c2` 的构建条件与映射断言的守卫都**写在函数体内**，无法从外部打补丁；
  而重写训练部分是**禁止的**（`DEF-118-17` 的教训）。故这里只复制**这两处守卫**所在的
  两个函数，且逐行对照 `dir119`，改动处均标 `# ★ 卡片 §7-#`。

用法：
  --stage a        Stage A 硬门（不需要新剂量数据）
  --stage smoke    冒烟：跑 1 个新剂量格，验 `c2` 非空 + 结构常量 + 自确定性
  --stage cell --dose D --seed S   单格（一格一进程，`DEF-105-02`）
  --stage formal   汇总 + 裁决
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_R = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_R / "dir119"))

import numpy as np                                     # noqa: E402

import R1_task119 as P119                              # noqa: E402
import mech125 as M                                    # noqa: E402

P118 = P119.P118
P117 = P119.P117
D90 = P119.D90
D104 = P119.D104
P = P119.P
MULT = P119.MULT
GAMMA_TR = P119.GAMMA_TR
SEEDS = P119.SEEDS
ARM = P119.ARM
JUDGE_MULT, REF_MULT = P119.JUDGE_MULT, P119.REF_MULT
N_ITEM = P119.N_ITEM
C2_RNGS = P119.C2_RNGS

# ★ 卡片 §7-1：新剂量
DOSES = tuple(M.DOSE_NEW) + (M.DOSE_PILOT,)        # (150, 300, 450, 25)
DOSES_MAIN = tuple(M.DOSE_NEW)                     # 判据剂量
SMOKE_DOSE, SMOKE_SEED = M.DOSE_PILOT, SEEDS[0]

# ★ 卡片 §7-4：OUT 改指
OUT = _HERE / "out"
for _d in ("cells", "det", "c2"):
    (OUT / _d).mkdir(parents=True, exist_ok=True)
D90.OUT = OUT
P117.OUT = OUT
P118.OUT = OUT
P119.OUT = OUT                                     # `P119.run_cell` 里的 `OUT` 也指过去

RES = {"stageA": [], "cells": {}, "smoke": {}, "verdict": {}}


def rec(name, ok, detail, blocking=True):
    RES.setdefault("stageA", []).append(
        {"name": name, "ok": bool(ok), "detail": str(detail), "blocking": bool(blocking)})
    print("  [%s] %s: %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    return bool(ok)


def dump(o, n):
    (OUT / n).write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")


def md5_cells(dirpath):
    """`dir119/out/cells/*.json` 的 md5 清单（`A125-15`：只读引用、前后不变）。"""
    out = {}
    p = pathlib.Path(dirpath)
    if not p.is_dir():
        return out
    for f in sorted(p.glob("*.json")):
        out[f.name] = hashlib.md5(f.read_bytes()).hexdigest()
    return out


# ================================================================ cell（复制自 dir119，改 2 处）
def run_cell(seed, dose, tok, model, facts, tr, ev, items_ev, meta_ev, tv,
             c2_items=None, c2_meta=None, c2_tgt=None, write_cell=True,
             scratch_out=None):
    """复用 `P118.run_cell`，只把 **c2 与映射断言**解除 dose 限制（卡片 §7-2/§7-3）。"""
    P119._STASH["S"] = []
    P119._CNT["i"] = -1
    P119._STASH["meta"] = meta_ev
    P119._STASH["tgt"] = None
    P119._STASH["tv"] = tv
    P119._STASH["c2_items"] = c2_items
    P119._STASH["c2_meta"] = c2_meta
    P119._STASH["c2_tgt"] = c2_tgt
    P119._STASH["c2"] = {}
    P117.acc_of = P119.acc_of_patched
    saved_out = P118.OUT
    if scratch_out is not None:
        (pathlib.Path(scratch_out) / "cells").mkdir(parents=True, exist_ok=True)
        P118.OUT = pathlib.Path(scratch_out)
    try:
        d = P118.run_cell(ARM, seed, dose, tok, model, facts, tr, items_ev)
    finally:
        P117.acc_of = P119._ORIG_ACC_OF
        P118.OUT = saved_out

    tgt = D104.targets_of(items_ev)
    S_all = P119._STASH["S"]
    # ★ 卡片 §7-3：映射断言**无条件执行**（原来被 `if dose == 600 or dose == 0:` 包住）
    assert len(S_all) == 1 + len(MULT), (
        "S 个数 %d != 1+%d（测量层归属不可信）。★ 最常见原因：`P118.run_cell` 的 "
        "skip-if-exists 命中 —— 检查 %s 是否已存在" % (len(S_all), len(MULT),
                                                   OUT / "cells" / ("%s_%d_%d.json" % (ARM, int(dose), int(seed)))))
    a0 = P119.score_and_mech(S_all[0], tgt, meta_ev, tv) if S_all[0] is not None else None
    assert a0 is None or abs(a0["acc_argmin"] - d["prior_acc"]) < 1e-12, \
        "stash[0] 与 prior_acc 不符：%s vs %s" % (a0 and a0["acc_argmin"], d["prior_acc"])
    for i, m in enumerate(MULT):
        if S_all[i + 1] is None:
            continue
        ai = P119.score_and_mech(S_all[i + 1], tgt, meta_ev, tv)
        assert abs(ai["acc_argmin"] - d["acc_by_mult"]["%g" % m]) < 1e-12, (
            "stash[%d] 与 acc_by_mult[%g] 不符：%s vs %s"
            % (i + 1, m, ai["acc_argmin"], d["acc_by_mult"]["%g" % m]))

    per = {}
    for i, m in enumerate(MULT):
        k = "%g" % m
        per[k] = (P119.score_and_mech(S_all[i + 1], tgt, meta_ev, tv)
                  if S_all[i + 1] is not None else {"finite": False})
    d["mech_by_mult"] = per
    d["prior_mech"] = (P119.score_and_mech(S_all[0], tgt, meta_ev, tv)
                       if S_all[0] is not None else {"finite": False})
    d["c2"] = P119._STASH["c2"]
    d["n_seen_hist"] = per["%g" % JUDGE_MULT].get("n_seen_hist")
    if write_cell:
        f = OUT / "cells" / ("%s_%d_%d.json" % (ARM, int(dose), int(seed)))
        f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def build_c2(tok, facts, ev):
    """★ 卡片 §7-2：`c2` 对**所有**剂量都构建（原来只在 `dose==600`）。"""
    c2_items, c2_meta, c2_tgt = {}, {}, {}
    for r in C2_RNGS:
        its, mts = P.make_items_with_vals(tok, facts, ev, r)
        c2_items[str(r)] = its
        c2_meta[str(r)] = mts
        c2_tgt[str(r)] = D104.targets_of(its)
    return c2_items, c2_meta, c2_tgt


def cells_main(seed, dose):
    tok, model = D90.model_and_tok()          # ★ `D90` 返回 (tok, model)，顺序不可颠倒
    facts, parts, csf, IT = P118.ground(tok, ARM)
    items, meta = IT["items_ev"], IT["meta_ev"]
    tv, tr, ev = IT["tv"], IT["tr"], IT["ev"]
    c2_items, c2_meta, c2_tgt = build_c2(tok, facts, ev)
    d = run_cell(int(seed), int(dose), tok, model, facts, tr, ev, items, meta, tv,
                 c2_items, c2_meta, c2_tgt)
    o = {"seed": int(seed), "dose": int(dose),
         "acc_by_mult": d["acc_by_mult"], "prior_n": d["prior_n"],
         "mech_by_mult": d["mech_by_mult"], "prior_mech": d.get("prior_mech"),
         "c2": d.get("c2"), "n_seen_hist": d.get("n_seen_hist"),
         "eval_writes": d.get("eval_writes"), "violations": d.get("violations"),
         "W_bitwise_same": d.get("W_bitwise_same"), "n_finite": d.get("n_finite"),
         "wq_md5": d.get("wq_md5"), "wo_md5": d.get("wo_md5"),
         "card": "DIR-125/v10"}
    dump(o, "cells/%s_%d_%d.json" % (ARM, int(dose), int(seed)))
    print("  完成 %s|%d|%d prior=%s acc(1e4)=%s"
          % (ARM, dose, seed, d["prior_n"], d["acc_by_mult"].get("10000")), flush=True)
    return o


# ================================================================ Stage A
def stage_a():
    print("=" * 100)
    print("Stage A —— 硬门（不需要新剂量数据）")
    print("=" * 100)

    # A125-1 工作例逐位
    u35, cl35 = M.solve_u(35, 23, 20, return_clamped=True)
    th35 = M.theta_of(35, 23, 20)
    th_old = 23 * M.P1(1.0) / 35.0
    u22 = M.solve_u(22, 23, 20)
    u225 = M.solve_u(22.5, 23, 20)
    ok = (abs(u35 - 0.2065603008) <= 1e-9 and abs(th35 - 0.5956264661) <= 1e-9
          and abs(th_old - 0.4380952381) <= 1e-9
          and abs((th35 - th_old) - 0.1575312280) <= 1e-9
          and abs(u22 - 1.0) <= 1e-9 and abs(u225 - 0.94898202) <= 1e-8 and cl35 is False)
    rec("A125-1", ok, "工作例 6 读数逐位（u=%.10f θ=%.10f θ_old=%.10f Δθ=%+.10f）"
        % (u35, th35, th_old, th35 - th_old))

    # A125-2 单调性 + 边界
    us = [M.solve_u(m, 23, 20) for m in range(1, 43)]
    mono = all(us[i] > us[i + 1] for i in range(len(us) - 1))
    ue, ce = M.solve_u(43, 23, 20, return_clamped=True)
    uo, co = M.solve_u(44, 23, 20, return_clamped=True)
    rec("A125-2", mono and ue == 0.0 and ce and uo == 0.0 and co,
        "m=1..42 严格递减=%s；m≥n1+n2 ⇒ (0.0, clamped=True)" % mono)

    # A125-3 旧域不变性（回归守卫）：dose=600 的 J124-1 必须逐位不变
    ctx600 = M.build_ctx(600)
    st600 = M.strata_of(ctx600)
    obs600 = M.stratum_obs(ctx600, st600)
    r600 = M.strat_test(obs600)
    hi_old = 68 * M.P1(1.0) + 99 * M.P2(1.0)
    no_clamp = all((o["m"] == 0) or not M.solve_u(o["m"], o["N1"], o["N2"],
                                                  return_clamped=True)[1] for o in obs600)
    ok3 = (r600["executable"] and r600["S"] == 109 and r600["m"] == 133
           and abs(r600["p_two"] - 0.000366782) <= 1e-9
           and r600["direction"] == "high" and r600["clamped_layers"] == 0 and no_clamp)
    rec("A125-3", ok3, "放宽域后 J124-1 逐位不变：S=%s m=%s p_two=%.9f clamped=0（hi_old=%.4f，逐层 m_s≤49）"
        % (r600["S"], r600["m"], r600["p_two"], hi_old))

    # A125-4 池化反例
    u68 = M.solve_u(133, 68, 99)
    th68 = M.theta_of(133, 68, 99)
    th_old68 = 68 * M.P1(1.0) / 133.0
    rec("A125-4", abs(u68 - 0.1963949625) <= 1e-9 and abs(th68 - 0.4655612531) <= 1e-9
        and abs(th_old68 - 0.3408521303) <= 1e-9 and 133.0 >= hi_old,
        "池化 (68,99,133)：u=%.10f θ=%.10f vs θ_old=%.10f ⇒ 「分层」与「放宽域」是两条独立修正"
        % (u68, th68, th_old68))

    # A125-7 映射断言**无条件**（结构性文本断言：不得再被 dose 守卫包住）
    # ★ A125-7 用 **AST 结构检查**，不是文本扫描（`DEF-124-05` 的教训：文本会命中
    #   我自己描述改动的注释）。要求：该 assert 语句在 `run_cell` 体内**没有被任何
    #   `if` 包住**（即它所在的父链里没有 `ast.If`）。
    import ast as _ast
    tree = _ast.parse((_HERE / "R1_task125.py").read_text(encoding="utf-8"))
    found = []          # [(func_name, depth_of_if, src_line)]

    def _walk(node, chain, fname):
        for ch in _ast.iter_child_nodes(node):
            nm = getattr(ch, "name", None) or fname
            if isinstance(ch, _ast.Assert):
                txt = _ast.unparse(ch.test)
                if "len(S_all) == 1 + len(MULT)" in txt.replace(" ", " "):
                    found.append((fname, sum(1 for x in chain if isinstance(x, _ast.If)), ch.lineno))
            _walk(ch, chain + [ch], nm)

    _walk(tree, [], "<module>")
    in_if = [f for f in found if f[1] > 0]
    rec("A125-7", bool(found) and not in_if,
        "映射断言在 AST 里找到 %d 处；被 `if` 包住的 %d 处 ⇒ 无条件执行"
        % (len(found), len(in_if)))

    # A125-8 手算例（DIR-124 A124-6 冻结值）
    wex = M.strat_test([{"key": "A", "N1": 4, "N2": 5, "e1": 3, "e2": 0},
                        {"key": "B", "N1": 6, "N2": 3, "e1": 1, "e2": 0}])
    ok8w = (wex["S"] == 4 and abs(wex["p_two"] - 0.5737934260) <= 1e-9
            and abs(wex["p_upper"] - 0.2598205554) <= 1e-9 and wex["direction"] == "high")
    rec("A125-8", ok8w, "D-124 冻结两层工作例：S=%s p_two=%.10f p_upper=%.10f dir=%s"
        % (wex["S"], wex["p_two"], wex["p_upper"], wex["direction"]))

    # A125-9 卷积归一 + 期望
    ms, ths = [3, 5, 0, 7], [0.4, 0.55, 0.5, 0.3]
    p = M.convolve_null(ms, ths)
    exp = sum(m * t for m, t in zip(ms, ths))
    mean = float((np.arange(len(p)) * p).sum())
    rec("A125-9", abs(p.sum() - 1.0) <= 1e-12 and abs(mean - exp) <= 1e-9,
        "Σpmf=%.15f E[S]=%.10f" % (p.sum(), mean))

    # A125-10 合成池上 reject 两态都出现
    rng = np.random.default_rng(11)
    seen = set()
    for k in range(60):
        rows = []
        for _ in range(9):
            e1 = int(rng.binomial(23, 0.16))
            e2 = int(rng.binomial(20, 0.04 * (1 + 3 * (k % 2))))
            rows.append({"key": "s%d" % _, "N1": 23, "N2": 20, "e1": e1, "e2": e2})
        r = M.strat_test(rows)
        if r["executable"]:
            seen.add(bool(r["reject"]))
    rec("A125-10", seen == {True, False}, "合成池 60 次：reject 出现 %r ⇒ 判据非空转" % sorted(seen))

    # A125-11 功效/水平（机器算，27 层）
    strata27 = [{"key": "d%d,%s" % (d, s["key"]), "N1": s["N1"], "N2": s["N2"]}
                for d in DOSES_MAIN for s in st600]
    lvl = M.level_for_strata(strata27, u=9.2, reps=20000)
    pw = M.power_for_strata(strata27, u=9.2, ratio=1.96, reps=20000)
    mc_se = (0.05 * 0.95 / 20000) ** 0.5
    rec("A125-11", lvl <= 0.05 + 3 * mc_se and pw >= M.POWER_MIN,
        "27 层：水平 %.4f ≤ %.4f；功效(1.96) %.4f ≥ %.2f"
        % (lvl, 0.05 + 3 * mc_se, pw, M.POWER_MIN))
    RES["power27"] = {"level": lvl, "power_1_96": pw}

    # A125-12 判据输入无 e3（结构守卫）
    ctx0 = M.build_ctx(0)
    st0 = M.strata_of(ctx0)
    obs0 = M.stratum_obs(ctx0, st0)
    uses_e3 = any("e3" in r or len(r) > 6 for r in obs0)
    rec("A125-12", not uses_e3 and all(r["N1"] == 23 and r["N2"] == 20 for r in obs0),
        "dose=0 的 9 层全是 (N1,N2)=(23,20) 且只含 e1/e2 ⇒ k=3 层未被使用")

    # A125-14 路径断言：新格必须落在 dir125/out/cells
    target = OUT / "cells" / ("%s_%d_%d.json" % (ARM, SMOKE_DOSE, SMOKE_SEED))
    ok14 = str(target).find(str(_HERE / "out" / "cells")) >= 0 and "dir119" not in str(target)
    rec("A125-14", ok14, "新格路径 = %s（含 dir125/out/cells、不含 dir119）" % target)

    # A125-15 归档只读基线（正式轮后再比一次）
    base = md5_cells(M.DIR119_CELLS)
    RES["dir119_md5_before"] = base
    rec("A125-15", len(base) == 18, "dir119 归档 18 个 cell 的 md5 基线已记录")

    blocking = [c for c in RES["stageA"] if c["blocking"] and not c["ok"]]
    print()
    print("%d / %d PASS；阻断项 %d"
          % (sum(1 for c in RES["stageA"] if c["ok"]), len(RES["stageA"]), len(blocking)))
    dump(RES, "stage_stageA.json")
    return 0 if not blocking else 1


# ================================================================ smoke
def stage_smoke():
    print("=" * 100)
    print("Smoke —— 1 个新剂量格（dose=%d, seed=%d）验 c2 非空 / 结构常量 / 自确定性"
          % (SMOKE_DOSE, SMOKE_SEED))
    print("=" * 100)
    o = cells_main(SMOKE_SEED, SMOKE_DOSE)
    c2 = o.get("c2") or {}
    has = ("10000" in c2) and all(str(r) in c2["10000"] for r in C2_RNGS)
    shapes = all(c2["10000"][str(r)].get("n") == N_ITEM
                 and len(c2["10000"][str(r)].get("pick_argmin") or []) == N_ITEM
                 for r in C2_RNGS) if has else False
    rec("A125-5", has and shapes,
        "dose=%d 的 c2 含 %r 三键、每键 n=48 且 pick_argmin 长 48（否则 DEF-125-02 复发）"
        % (SMOKE_DOSE, sorted(c2.get("10000", {}).keys())))

    ctx = M.build_ctx(SMOKE_DOSE)
    st = M.strata_of(ctx)
    ok6 = (ctx["rngs"] == M.ALL_RNGS
           and all(s["N1"] == M.STRUCT_N1_PER_SEED and s["N2"] == M.STRUCT_N2_PER_SEED
                   for s in st))
    rec("A125-6", ok6, "新剂量 4-qset 可用=%r；每层 (N1,N2)=(%s,%s) 与 dose=600 逐位相同"
        % (ctx["rngs"], st[0]["N1"] if st else "?", st[0]["N2"] if st else "?"))

    # A125-13 自确定性：同 cell 再跑一次（scratch 隔离），读数逐位相同
    scratch = OUT / "det" / ("smoke_%d_%d" % (SMOKE_DOSE, SMOKE_SEED))
    tok, model = D90.model_and_tok()
    facts, parts, csf, IT = P118.ground(tok, ARM)
    tv, tr, ev = IT["tv"], IT["tr"], IT["ev"]
    ci, cm, ct = build_c2(tok, facts, ev)
    d2 = run_cell(SMOKE_SEED, SMOKE_DOSE, tok, model, facts, tr, ev,
                  IT["items_ev"], IT["meta_ev"], tv, ci, cm, ct,
                  write_cell=False, scratch_out=scratch)
    same_acc = (d2["acc_by_mult"] == o["acc_by_mult"])
    same_pick = all((d2["mech_by_mult"].get(k) or {}).get("pick_argmin")
                    == (o["mech_by_mult"].get(k) or {}).get("pick_argmin")
                    for k in d2["mech_by_mult"])
    rec("A125-13", same_acc and same_pick,
        "同 cell 两进程：acc_by_mult 逐位相同=%s；9 个 mult 的 pick_argmin 逐位相同=%s"
        % (same_acc, same_pick))

    RES["smoke"] = {"dose": SMOKE_DOSE, "seed": SMOKE_SEED,
                    "acc_by_mult": o["acc_by_mult"],
                    "c2_keys": sorted(c2.get("10000", {}).keys()),
                    "rngs": list(ctx["rngs"])}
    smoke_ids = {"A125-5", "A125-6", "A125-13"}
    blocking = [c for c in RES["stageA"]
                if c["name"] in smoke_ids and c["blocking"] and not c["ok"]]
    print()
    print("%d / %d PASS；阻断项 %d"
          % (sum(1 for c in RES["stageA"] if c["ok"]), len(RES["stageA"]), len(blocking)))
    dump(RES, "stage_smoke.json")
    return 0 if not blocking else 1


# ================================================================ formal
def stage_formal():
    print("=" * 100)
    print("Formal —— 汇总 + 裁决（DIR-125）")
    print("=" * 100)

    # ---- R125-5 / 夹紧计数 + 逐剂量读数
    per_dose = {}
    for d in M.DOSE_ALL:
        try:
            ctx = M.build_ctx(d)
        except RuntimeError as e:
            print("  ⚠ dose=%d 不可读：%s" % (d, e))
            continue
        st = M.strata_of(ctx)
        obs = M.stratum_obs(ctx, st)
        r = M.strat_test(obs)
        wh = M.w_hat(obs)
        ci = M.ratio_ci(obs)                      # ★ report-only，不改判据
        per_dose[d] = {"strat": r, "what": wh, "ci": ci, "ns": [o["m"] for o in obs],
                       "m_tot": sum(o["m"] for o in obs),
                       "expect": r["expect"], "rngs": list(ctx["rngs"])}
        print("  dose=%-4d Σm=%-4d S=%-4s E[S]=%.4f p_two=%-12.6g dir=%-5s clamped=%d "
              "ratio=%s"
              % (d, per_dose[d]["m_tot"], r["S"], r["expect"], r["p_two"],
                 r["direction"], r["clamped_layers"],
                 ("%.4f [%.3f, %.3f]" % (wh["ratio"], ci["lo"], ci["hi"]))
              if wh["ratio"] is not None else "NA"))

    # ---- J125-1：池化 3 个判据剂量 × 9 seed = 27 层
    pooled = []
    for d in DOSES_MAIN:
        if d not in per_dose:
            continue
        ctx = M.build_ctx(d)
        st = M.strata_of(ctx, dose_label=d)
        pooled += M.stratum_obs(ctx, st)
    j1 = M.strat_test(pooled)
    j1_ok = bool(j1["executable"] and j1["reject"] and j1["direction"] == "high")

    def _f(v, fmt="%.9f"):
        return (fmt % v) if isinstance(v, (int, float)) else str(v)

    print("\n  J125-1（27 层，双侧）: S=%s m=%s E[S]=%s p_two=%s dir=%s clamped=%s ⇒ %s"
          % (j1["S"], j1["m"], _f(j1["expect"], "%.4f"), _f(j1["p_two"]),
             j1["direction"], j1["clamped_layers"],
             "拒绝 high" if j1_ok else "未达判据（或不可执行：%s）" % j1.get("reason")))

    # ---- J125-2：每个判据剂量的 ratio > 1
    j2_detail = {}
    j2_ok = True
    for d in DOSES_MAIN:
        wh = per_dose.get(d, {}).get("what", {})
        rat = wh.get("ratio")
        ok = rat is not None and rat > 1.0
        j2_detail[str(d)] = {"ratio": rat, "ok": ok}
        j2_ok = j2_ok and ok
        print("  J125-2 dose=%-4d ratio=%s ⇒ %s"
              % (d, ("%.4f" % rat) if rat is not None else "NA",
                 ">1 ✓" if ok else "未达（须 >1）"))

    # ---- J125-3：dose=0 重读（放宽域）
    r0 = per_dose.get(0, {}).get("strat", {})
    j3_reject_high = bool(r0.get("executable") and r0.get("reject")
                          and r0.get("direction") == "high")
    print("  J125-3 dose=0（放宽域重读）: p_two=%s dir=%s clamped=%s ⇒ %s"
          % (r0.get("p_two"), r0.get("direction"), r0.get("clamped_layers"),
             "拒绝 high ⇒ 支(b) 否证" if j3_reject_high else "未拒 high ⇒ 支(b) 未被否证"))

    # ---- 完备性
    missing = [d for d in DOSES_MAIN if d not in per_dose]
    n_expected = 9 * len(DOSES)
    n_have = 0
    for d in DOSES:
        if d in per_dose:
            n_have += len(M.load_cells(d))
    complete = (not missing) and n_have >= n_expected

    # ---- 判决表（卡片 §5，5 支）
    clamped_any = any(per_dose[d]["strat"]["clamped_layers"] > 0
                      for d in DOSES_MAIN if d in per_dose)
    if not complete or not j1["executable"]:
        row, verdict, attribution = 5, "INCOMPLETE_OR_INEXECUTABLE", "不可判"
    elif clamped_any:
        row, verdict, attribution = 5, "CLAMPED_LAYER_IN_CRITERION", "不可判"
    elif j1_ok and j2_ok and not j3_reject_high:
        row, verdict, attribution = 1, "H125_CONFIRMED_LEARNED", "学习产物"
    elif j1_ok and j2_ok and j3_reject_high:
        row, verdict, attribution = 2, "H124_CONFOUNDED_PRIOR", "先验（与学习无关）"
    elif j1_ok and not j2_ok:
        row, verdict, attribution = 3, "H125_PARTIAL_DOSE_SPECIFIC", "不可判"
    else:
        row, verdict, attribution = 4, "H125_FALSIFIED", "不可判"

    print("\n" + "=" * 100)
    print("★★ 判决：第 %d 行 → %s（归因：%s）" % (row, verdict, attribution))
    print("=" * 100)

    # ---- 归档只读复核
    # ★ 基线在 `stage_stageA.json` 里（formal 是**独立进程**，`RES` 是空的）。
    #   第一版直接读 `RES["dir119_md5_before"]` ⇒ 拿到 `{}` ⇒ 恒判 False（假 FAIL，
    #   与 `DEF-124-06`/`DEF-123-02` 同族：断言的方向/来源写错）。
    after = md5_cells(M.DIR119_CELLS)
    before = RES.get("dir119_md5_before")
    if not before:
        sa = OUT / "stage_stageA.json"
        if sa.is_file():
            before = (json.loads(sa.read_text(encoding="utf-8"))
                      .get("dir119_md5_before") or {})
    unchanged = bool(before) and (before == after)
    print("  dir119 归档：基线 %d 个 cell、现状 %d 个、逐位不变=%s"
          % (len(before or {}), len(after), unchanged))

    RES["verdict"] = {
        "row": row, "verdict": verdict, "attribution": attribution,
        "complete": complete, "n_cells": n_have, "n_expected": n_expected,
        "missing_doses": missing,
        "J125_1": {k: j1[k] for k in ("S", "m", "expect", "p_two", "p_upper", "p_lower",
                                      "direction", "strata_used", "clamped_layers",
                                      "reject", "executable")},
        "J125_1_ok": j1_ok,
        "J125_2": j2_detail, "J125_2_ok": j2_ok,
        "J125_3": {"p_two": r0.get("p_two"), "direction": r0.get("direction"),
                   "clamped_layers": r0.get("clamped_layers"),
                   "S": r0.get("S"), "m": r0.get("m"), "expect": r0.get("expect"),
                   "reject_high": j3_reject_high},
        "per_dose": {str(d): {"m_tot": v["m_tot"], "ns": v["ns"],
                              "S": v["strat"]["S"], "E_S": v["strat"]["expect"],
                              "p_two": v["strat"]["p_two"],
                              "direction": v["strat"]["direction"],
                              "clamped": v["strat"]["clamped_layers"],
                              "ratio": v["what"]["ratio"],
                              "ratio_ci": v.get("ci"),
                              "w1": v["what"]["w1"], "w2": v["what"]["w2"],
                              "N1": v["what"]["N1"], "N2": v["what"]["N2"],
                              "e1": v["what"]["e1"], "e2": v["what"]["e2"]}
                          for d, v in per_dose.items()},
        "dir119_archive_unchanged": unchanged,
        "power27": RES.get("power27"),
    }
    dump(RES, "formal.json")
    print("\n→ out/formal.json ；dir119 归档不变 = %s" % unchanged)
    return 0 if (complete and not missing) else 1


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("a", "smoke", "cell", "formal"))
    ap.add_argument("--seed", type=int, default=SEEDS[0])
    ap.add_argument("--dose", type=int, default=DOSES[0])
    a = ap.parse_args()
    t0 = time.time()
    if a.stage == "a":
        rc = stage_a()
    elif a.stage == "smoke":
        rc = stage_smoke()
    elif a.stage == "cell":
        if a.dose not in DOSES:
            print("拒绝：dose=%d 不在 %r（dose∈{0,600} 是冻结归档，只读引用）"
                  % (a.dose, DOSES))
            return 2
        cells_main(a.seed, a.dose)
        rc = 0
    else:
        rc = stage_formal()
    print("\n[%.1fs] rc=%d" % (time.time() - t0, rc), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())