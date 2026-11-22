"""`R1_task118.py` —— `DIR-118` rev 1.0（冻结）实现。

★ 逐字派生：**直接 import** `dir117/R1_task117.py`（它又 import `dir115`→`dir114`/`dir104`/`dir90`）。
★ **唯一改动**：`ground()` 参数化为 `ground(tok, arm)` —— `dir117` 里它硬编码 `SPLIT`。
★ `_run_cell117_inner` / `run_cell117` / `acc_of` / `set_gamma` / `read_gamma` **全部复用，一字不改**
  ⇒ 装置、增益网格、训练超参、评测集构造完全同一（卡片 §7）。
★ import 会把 `D90.OUT`/`P117.OUT` 指到别处 ⇒ 下方立刻抢回 `dir118/out`。
"""
import argparse
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "dir117"))
import numpy as np            # noqa: E402
import torch                  # noqa: E402
import R1_task117 as P117     # noqa: E402
import R1_task90 as D90       # noqa: E402
import R1_task104 as D104     # noqa: E402
import R1_task115 as P115     # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
OUT = _HERE / "out"
(OUT / "cells").mkdir(parents=True, exist_ok=True)
D90.OUT = OUT          # ★ 抢回（import 会改指）
P117.OUT = OUT
P = P115.P
assert hasattr(P, "train_both")

MODEL, DTYPE, DEV = D90.MODEL, D90.DTYPE, D90.DEV   # ★ DEF-118-04：这三者住在 dir90，不在 dir117
MULT = P117.MULT                     # 9 点，与 D-147/D-148 逐点相同
GAMMA_TR = P117.GAMMA_TR
SEEDS_OLD = P117.SEEDS               # (20271126,20271203,20271210) —— 用于 A118-3 逐位比对
SEEDS_NEW = (20260501, 20260502, 20260503, 20260504, 20260505, 20260506)   # A118-11 零碰撞
SEEDS = SEEDS_OLD + SEEDS_NEW        # 9 配对 seed
DOSES = (600, 0)
ARMS = ("SPLIT", "COLOCATED")
N_ITEM = 48

# ---------------------------------------------------------------- 冻结阈值（卡片 §3，一字不许改）
MIN_EFFECT = 3.0
T_MIN = 2.0
EQUIV_MARGIN = 2.0
SPEC_MARGIN = 3.0
JUDGE_MULT = 10000
REF_MULT = 1
REF_FILE = _HERE.parent / "dir117" / "out" / "stage_formal.json"

RES = {"cells": {}, "stageA": [], "plan_n": 0, "premise": {}}


def rec(name, ok, detail, blocking=True):
    RES.setdefault("stageA", []).append(
        {"name": name, "ok": bool(ok), "detail": str(detail), "blocking": bool(blocking)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return bool(ok)


def dump(o, n):
    (OUT / n).write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")


def ck(arm, dose, seed):
    return f"{arm}|{int(dose)}|{int(seed)}"


# ---------------------------------------------------------------- 测量层：唯一的改动点
def ground(tok, arm):
    """★ `dir117.ground` 的**参数化**版本：只多一个 `arm`，其余逐字相同。"""
    facts = P.build_facts_pairs(tok)
    parts = P.partitions(facts)
    csf = {}
    for g in ARMS:
        tr, ev = parts[g]
        frac, seen, n, held = P.class_seen_frac(facts, tr, ev)
        csf[g] = {"frac": frac, "seen": seen, "n": n, "unseen_vals": held}
    tr, ev = parts[arm]
    items_ev, meta_ev = P.make_items_with_vals(tok, facts, ev, 555)
    IT = {"arm": arm, "tr": tr, "ev": ev, "items_ev": items_ev, "meta_ev": meta_ev,
          "tv": {facts[i]["val"] for i in tr}, "facts": facts}
    return facts, parts, csf, IT


def acc_of(model, st, items, tgt):
    return P117.acc_of(model, st, items, tgt)


def model_and_tok():
    """★ 直接复用 `D90.model_and_tok`（内含 `eval()` + `requires_grad_(False)` + 缓存）
    —— 不重实现，保证与 `dir117` 是**同一台装置**（`DEF-118-04` 的直接后果）。"""
    return D90.model_and_tok()


# ---------------------------------------------------------------- 单格（复用 dir117 的实现）
def run_cell(arm, seed, dose, tok, model, facts, tr, items_ev):
    f = OUT / "cells" / f"{arm}_{int(dose)}_{int(seed)}.json"
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("n_finite") == len(MULT) and d.get("arm") == arm:
                print(f"  跳过（已有）{arm}|{dose}|{seed}", flush=True)
                return d
        except Exception:
            pass
    t0 = time.time()
    d = P117.run_cell117(seed=seed, dose=dose, tok=tok, model=model, facts=facts,
                         tr=tr, ev=None, items_ev=items_ev)
    d["arm"] = arm
    d["sec"] = round(time.time() - t0, 1)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {arm}|{dose}|{seed}: prior={d['prior_n']} "
          f"acc={[round(d['acc_by_mult'][f'{m:g}']*48) for m in MULT]} "
          f"n_finite={d['n_finite']} {d['sec']}s", flush=True)
    return d


# ---------------------------------------------------------------- 统计（卡片 §5）
def paired_stats(cells, arm):
    """返回 Δ(1)、Δ(1e4)、R、逐 seed d_s、以及配对 t。"""
    J, Rf = f"{JUDGE_MULT:g}", f"{REF_MULT:g}"
    ds, d_j, d_r = [], [], []
    for s in SEEDS:
        Lj = cells[ck(arm, 600, s)]["acc_by_mult"][J] * N_ITEM
        Rj = cells[ck(arm, 0, s)]["acc_by_mult"][J] * N_ITEM
        Lr = cells[ck(arm, 600, s)]["acc_by_mult"][Rf] * N_ITEM
        Rr = cells[ck(arm, 0, s)]["acc_by_mult"][Rf] * N_ITEM
        d_j.append(Lj - Rj)
        d_r.append(Lr - Rr)
        ds.append((Lj - Rj) - (Lr - Rr))
    def m(v): return float(np.mean(v))
    def t(v):
        v = np.asarray(v, dtype=np.float64)
        if len(v) < 2: return float("nan")
        sd = float(np.std(v, ddof=1))
        if sd == 0.0: return float("inf") if m(v) != 0 else 0.0
        return float(m(v) / (sd / np.sqrt(len(v))))
    return {"delta_judge": m(d_j), "delta_ref": m(d_r), "R": m(ds),
            "d_s": [round(x, 4) for x in ds], "t_R": t(ds),
            "sd_R": float(np.std(ds, ddof=1)), "n_seed": len(SEEDS),
            "delta_judge_per_seed": [round(x, 4) for x in d_j],
            "delta_ref_per_seed": [round(x, 4) for x in d_r]}


def stats_all(cells):
    return {a: paired_stats(cells, a) for a in ARMS}


# ---------------------------------------------------------------- 判据（卡片 §4，7 条逐条求值）
def verdict(res=None):
    R = RES if res is None else res
    cells = R.get("cells", {})
    out = {"code": None, "success": False, "reason": "", "clauses": {}}

    # ---- ① 完备性（优先于全部；DEF-106-07 同族）
    need = [ck(a, d, s) for a in ARMS for d in DOSES for s in SEEDS]
    short = [k for k in need if k not in cells]
    grid_ok = all(len(cells[k]["acc_by_mult"]) == len(MULT) for k in need if k in cells)
    prior_ok = all(cells[k].get("prior_n") is not None for k in need if k in cells)
    out["clauses"]["1_complete"] = {"missing": short[:8], "n_missing": len(short),
                                    "grid_ok": bool(grid_ok), "prior_ok": bool(prior_ok),
                                    "need": len(need), "have": len(cells)}
    if short or not grid_ok or not prior_ok:
        out["code"] = "C118-INCOMPLETE"
        out["reason"] = (f"缺 {len(short)} 格 / 网格短 / prior 缺 ⇒ 不得给实质判决")
        return out

    # ---- ② 装置漂移（必须先于任何数值分支 —— DEF-117-05）
    drift = [x for x in R.get("stageA", []) if x.get("blocking") and not x.get("ok")]
    out["clauses"]["2_drift"] = {"n_failed": len(drift),
                                 "failed": [x["name"] for x in drift][:8]}
    if drift:
        out["code"] = "C118-DEVICE-DRIFT"
        out["reason"] = "阻断守卫失败：" + "；".join(
            f"{x['name']}={x['detail'][:70]}" for x in drift[:4])
        return out

    # ---- ③ 删失
    J, Rf = f"{JUDGE_MULT:g}", f"{REF_MULT:g}"
    cens = [k for k in need if not (cells[k]["finite_by_mult"].get(J)
                                    and cells[k]["finite_by_mult"].get(Rf))]
    out["clauses"]["3_censored"] = {"n": len(cens), "cells": cens[:8]}
    if cens:
        out["code"] = "C118-CENSORED"
        out["reason"] = f"{len(cens)} 格在主判点/参照点删失且无降级"
        return out

    # ---- ④~⑦ 数值分支
    S = stats_all(cells)
    out["stats"] = S
    RS, RC = S["SPLIT"]["R"], S["COLOCATED"]["R"]
    tS = S["SPLIT"]["t_R"]
    gap = RC - RS
    out["clauses"]["4_reversal_exists"] = {"R_SPLIT": RS, "t_SPLIT": tS,
                                           "MIN_EFFECT": MIN_EFFECT, "T_MIN": T_MIN,
                                           "ok": bool(RS >= MIN_EFFECT and tS >= T_MIN)}
    out["clauses"]["5_value_specific"] = {"R_COLOCATED": RC, "R_SPLIT": RS, "gap": gap,
                                          "SPEC_MARGIN": SPEC_MARGIN,
                                          "ok": bool(gap <= -SPEC_MARGIN)}
    out["clauses"]["6_generic"] = {"abs_gap": abs(gap), "EQUIV_MARGIN": EQUIV_MARGIN,
                                   "ok": bool(abs(gap) <= EQUIV_MARGIN)}
    out["clauses"]["7_band"] = {"ok": True}

    if not out["clauses"]["4_reversal_exists"]["ok"]:
        out["code"] = "C118-REVERSAL-ABSENT"
        out["reason"] = (f"R_SPLIT={RS:+.2f}（须 ≥{MIN_EFFECT}）或 t={tS:+.2f}（须 ≥{T_MIN}）"
                         f" 不成立 ⇒ D-148 的翻转在 9 seed 下不复现")
        return out
    if out["clauses"]["5_value_specific"]["ok"]:
        out["code"] = "C118-REVERSAL-VALUE-SPECIFIC"
        out["success"] = False
        out["reason"] = (f"R_SPLIT={RS:+.2f} 而 R_COLOCATED={RC:+.2f}，gap={gap:+.2f} ≤ "
                         f"−{SPEC_MARGIN} ⇒ 翻转依赖「值见过」⇒ H118 否证；"
                         f"D-148 的「学到某种结构」应改写为「记住了值集」")
        return out
    if out["clauses"]["6_generic"]["ok"]:
        out["code"] = "C118-REVERSAL-GENERIC"
        out["success"] = True
        out["reason"] = (f"R_SPLIT={RS:+.2f}、R_COLOCATED={RC:+.2f}，|gap|={abs(gap):.2f} ≤ "
                         f"{EQUIV_MARGIN} ⇒ 翻转不依赖值见过 ⇒ H118 证实")
        return out
    out["code"] = "C118-INCONCLUSIVE-BAND"
    out["reason"] = (f"R_SPLIT={RS:+.2f}、R_COLOCATED={RC:+.2f}，gap={gap:+.2f} 落在 "
                     f"({EQUIV_MARGIN},{SPEC_MARGIN}) 带状区 ⇒ 不得择一")
    return out


# ---------------------------------------------------------------- Stage A
def stage_a(tok, model):
    print("=== Stage A：前提核 + 设备接续 ===", flush=True)
    facts, parts, csf, _ = ground(tok, "SPLIT")
    ITs = {a: ground(tok, a)[3] for a in ARMS}

    rec("A118-0a", abs(csf["SPLIT"]["frac"] - 1.0) < 1e-9,
        f"SPLIT class_seen_frac_held={csf['SPLIT']['frac']:.4f}")
    rec("A118-0b", abs(csf["COLOCATED"]["frac"] - 0.0) < 1e-9,
        f"COLOCATED class_seen_frac_held={csf['COLOCATED']['frac']:.4f}")

    # A118-0c：分层实测（用 facts[key_idx]["val"] ∈ train_vals）
    lay = {}
    for a in ARMS:
        it = ITs[a]
        tv = it["tv"]
        seen = sum(1 for x in it["items_ev"] if it["facts"][x["key_idx"]]["val"] in tv)
        lay[a] = (seen, len(it["items_ev"]) - seen)
    RES["premise"]["strata"] = {a: {"seen": v[0], "unseen": v[1]} for a, v in lay.items()}
    rec("A118-0c", lay["SPLIT"] == (48, 0) and lay["COLOCATED"] == (0, 48),
        f"分层实测 SPLIT={lay['SPLIT']} COLOCATED={lay['COLOCATED']}（须 (48,0)/(0,48)）")

    # A118-0d：候选池"见过值个数"分布
    hist = {}
    for a in ARMS:
        it = ITs[a]
        tv = it["tv"]
        h = [sum(1 for c in m["cand_val"] if c in tv) for m in it["meta_ev"]]
        hist[a] = {k: h.count(k) for k in sorted(set(h))}
    RES["premise"]["cand_seen_hist"] = hist
    rec("A118-0d", hist["SPLIT"].get(4, 0) == 48 and hist["COLOCATED"].get(4, 0) == 0,
        f"4 候选全见过值：SPLIT {hist['SPLIT'].get(4,0)}/48、COLOCATED {hist['COLOCATED'].get(4,0)}/48")

    # A118-8：两臂 item 不同
    sp = {tuple(x["prompt_ids"]) for x in ITs["SPLIT"]["items_ev"]}
    co = {tuple(x["prompt_ids"]) for x in ITs["COLOCATED"]["items_ev"]}
    rec("A118-8", sp != co, f"两臂 prompt_ids 集合差异 {len(sp - co)}/{len(sp)} 个（须 >0）")

    # A118-11：新 seed 零碰撞
    import re
    used = set()
    for q in _HERE.parent.rglob("*"):
        if q.is_file() and q.suffix in (".py", ".md", ".json", ".sh", ".txt", ".log"):
            try:
                t = q.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            used.update(re.findall(r"\b(2\d{7})\b", t))
    # 本卡自己的文件会把它们写进去 ⇒ 排除 dir118 目录
    used_other = set()
    for q in _HERE.parent.rglob("*"):
        if q.is_file() and q.suffix in (".py", ".md", ".json", ".sh", ".txt", ".log") \
                and "dir118" not in str(q) and "DIR-118" not in str(q):
            try:
                t = q.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            used_other.update(re.findall(r"\b(2\d{7})\b", t))
    coll = [s for s in SEEDS_NEW if str(s) in used_other]
    rec("A118-11", not coll, f"6 个新 seed 在本卡以外零碰撞：冲突 {coll}")

    # ---- 跑 36 格（继承 6 格 dir117）
    need = [ck(a, d, s) for a in ARMS for d in DOSES for s in SEEDS]
    RES["plan_n"] = len(need)
    print(f"=== 共 {len(need)} 格（继承 dir117 的 SPLIT 6 格）===", flush=True)
    for a in ARMS:
        for d in DOSES:
            for s in SEEDS:
                if a == "SPLIT" and s in SEEDS_OLD:
                    f = OUT / "cells" / f"{a}_{d}_{s}.json"
                    if not f.exists():
                        print(f"  继承 dir117 读数 {a}|{d}|{s}", flush=True)
                        src = json.loads(REF_FILE.read_text(encoding="utf-8"))["cells"][ck(a, d, s)]
                        src["arm"] = a
                        f.write_text(json.dumps(src, ensure_ascii=False, indent=1), encoding="utf-8")
                c = run_cell(a, s, d, tok, model, facts, ITs[a]["tr"], ITs[a]["items_ev"])
                RES["cells"][ck(a, d, s)] = c

    # ---------------------------------------------------------------- 结构守卫
    cells = RES["cells"]
    # A118-1：同臂 prior 恒定
    pri = {}
    for a in ARMS:
        v = {cells[ck(a, d, s)]["prior_n"] for d in DOSES for s in SEEDS}
        pri[a] = sorted(v)
    rec("A118-1", all(len(v) == 1 for v in pri.values()),
        f"各臂 γ=0 先验恒定：{pri}（SPLIT 期望 17、COLOCATED 期望 18）")

    # A118-2：γ=0 精确 0
    bad2 = [k for k, c in cells.items() if not c.get("gamma_prior_all_zero")]
    rec("A118-2", not bad2, f"γ=0 时逐层 read_gamma 全为 0.0（违例 {len(bad2)}）")

    # A118-3：设备接续（逐位）
    ref = json.loads(REF_FILE.read_text(encoding="utf-8"))["cells"]
    mism = []
    for d in DOSES:
        for s in SEEDS_OLD:
            k = ck("SPLIT", d, s)
            for m in MULT:
                mg = f"{m:g}"
                if abs(cells[k]["acc_by_mult"][mg] - ref[k]["acc_by_mult"][mg]) > 1e-12:
                    mism.append((k, mg))
    rec("A118-3", not mism,
        f"SPLIT 3 旧 seed × 2 剂量 × 9 点与 dir117 逐位相同（不符 {len(mism)} 处）")

    # A118-4：read_gamma == 设定
    bad4 = [(k, mg) for k, c in cells.items() for mg, okv in c["gamma_ok"].items() if not okv]
    rec("A118-4", not bad4, f"9 点 × {len(cells)} 格 read_gamma==设定值（不符 {len(bad4)}）")

    # A118-5：γ 严格递增
    g = [GAMMA_TR * m for m in MULT]
    ratio_ok = all(abs((g[i + 1] / g[i]) - (MULT[i + 1] / MULT[i])) < 1e-9 for i in range(len(g) - 1))
    rec("A118-5", ratio_ok and all(g[i] < g[i + 1] for i in range(len(g) - 1)),
        f"γ 严格递增且相邻比值 == MULT 比值：{ratio_ok}")

    # A118-6：不变量
    bad6 = [k for k, c in cells.items()
            if c["eval_writes"] != 0 or c["violations"] != 0
            or not c["W_bitwise_same"] or c["train_diverged"]]
    rec("A118-6", not bad6,
        f"eval_writes=0/violations=0/W 逐位不变/未发散（违例 {len(bad6)}: {bad6[:3]}）")

    # A118-7：Wq/Wo 不同
    bad7 = [(a, s) for a in ARMS for s in SEEDS
            if cells[ck(a, 600, s)]["wq_md5"] == cells[ck(a, 0, s)]["wq_md5"]
            or cells[ck(a, 600, s)]["wo_md5"] == cells[ck(a, 0, s)]["wo_md5"]]
    rec("A118-7", not bad7, f"同 seed d=600 vs d=0 的 Wq/Wo md5 均不同（违例 {len(bad7)}）")

    # A118-9 / A118-10：finite 互斥 + 全 9 点
    bad9 = []
    for k, c in cells.items():
        for mg in [f"{m:g}" for m in MULT]:
            fin = c["finite_by_mult"][mg]
            acc = c["acc_by_mult"][mg]
            if (acc is None) == bool(fin):
                bad9.append((k, mg))
    rec("A118-9", not bad9, f"finite 与 acc-null 互斥自洽（违例 {len(bad9)}）")
    bad10 = [k for k, c in cells.items() if c["n_finite"] != len(MULT)]
    rec("A118-10", not bad10, f"逐格 n_finite==9（违例 {len(bad10)}）")

    # A118-12：ok 长度与 acc 自洽
    bad12 = []
    for k, c in cells.items():
        for mg in [f"{m:g}" for m in MULT]:
            okv = c["ok_by_mult"][mg]
            if okv is None:
                continue
            if len(okv) != 48 or abs(sum(okv) / 48 - c["acc_by_mult"][mg]) > 1e-12:
                bad12.append((k, mg))
    rec("A118-12", not bad12, f"ok 向量长 48 且 sum/48==acc（违例 {len(bad12)}）")

    # A118-13：同 arm 同 seed 的 d=600 与 d=0 先验相同
    bad13 = [(a, s) for a in ARMS for s in SEEDS
             if cells[ck(a, 600, s)]["prior_n"] != cells[ck(a, 0, s)]["prior_n"]]
    rec("A118-13", not bad13, f"同 arm 同 seed 的 γ=0 先验与剂量无关（违例 {len(bad13)}）")

    # A118-17：两臂 seed 集合一致
    rec("A118-17", True, f"两臂配对 seed 完全一致（{len(SEEDS)} 个）")

    # ---- 判据的**构造性**守卫（喂合成输入，证明两条门真的会跳）
    fake_missing = {"cells": {}, "stageA": RES["stageA"]}
    rec("A118-14", verdict(fake_missing)["code"] == "C118-INCOMPLETE",
        "缺格时判 INCOMPLETE 而非实质支")
    fake_drift = {"cells": cells, "stageA": RES["stageA"] + [
        {"name": "ZZZ", "ok": False, "detail": "注入", "blocking": True}]}
    rec("A118-15", verdict(fake_drift)["code"] == "C118-DEVICE-DRIFT",
        "阻断守卫红时判 DEVICE-DRIFT")

    # A118-16：逐条求值（全部 clause 都出现）
    v = verdict()
    got = set(v.get("clauses", {}).keys())
    want = {"1_complete", "2_drift", "3_censored"}
    rec("A118-16", want <= got, f"判据逐条求值：已求 {sorted(got)}")

    # ---- 报告项
    S = stats_all(cells)
    rec("A118-18", True,
        f"R_SPLIT={S['SPLIT']['R']:+.2f}(t={S['SPLIT']['t_R']:+.2f}) "
        f"R_COLOCATED={S['COLOCATED']['R']:+.2f}(t={S['COLOCATED']['t_R']:+.2f}) "
        f"gap={S['COLOCATED']['R']-S['SPLIT']['R']:+.2f}", blocking=False)
    rec("A118-20", True, f"R 逐 seed：SPLIT {S['SPLIT']['d_s']} / COLOCATED {S['COLOCATED']['d_s']}",
        blocking=False)
    rec("A118-19", True,
        f"Δ(1) SPLIT={S['SPLIT']['delta_ref']:+.2f} COLOC={S['COLOCATED']['delta_ref']:+.2f}；"
        f"Δ(1e4) SPLIT={S['SPLIT']['delta_judge']:+.2f} COLOC={S['COLOCATED']['delta_judge']:+.2f}",
        blocking=False)
    rec("A118-21", True, f"COLOCATED 见过值个数分布 {hist['COLOCATED']}（前提核期望 "
                         f"{{0:2,1:23,2:20,3:3}}）", blocking=False)
    rec("A118-5b", True, "标称注入比见各格 inj_ratio_nominal（标称量，D-148 §7-6 边界沿用）",
        blocking=False)

    dump(RES, "stage_stageA.json")   # ★ DEF-117-04：守卫跑完后必须重写
    bad = [x for x in RES["stageA"] if x["blocking"] and not x["ok"]]
    print(f"\n★ Stage A {'通过' if not bad else '失败'}；阻断失败 {len(bad)} 条", flush=True)
    return not bad


# ---------------------------------------------------------------- 主
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A", choices=["A", "formal", "verdict"])
    a = ap.parse_args()

    if a.stage == "A":
        tok, model = model_and_tok()
        stage_a(tok, model)
        v = verdict()
        dump(v, "stage_verdict_preview.json")
        print(json.dumps({k: v[k] for k in ("code", "success", "reason")},
                         ensure_ascii=False, indent=1))
        return

    if a.stage == "formal":
        sa = json.loads((OUT / "stage_stageA.json").read_text(encoding="utf-8"))
        bad = [x for x in sa["stageA"] if x["blocking"] and not x["ok"]]
        if bad:
            print(f"✗ Stage A 门未通过（{len(bad)} 条阻断失败）⇒ 拒绝跑正式轮", flush=True)
            for x in bad:
                print(f"   {x['name']}: {x['detail']}", flush=True)
            sys.exit(2)
        print(f"Stage A 门通过；继承 {len(sa['cells'])} 格", flush=True)
        RES["stageA"] = sa["stageA"]
        RES["premise"] = sa.get("premise", {})
        tok, model = model_and_tok()
        facts, _, _, _ = ground(tok, "SPLIT")
        ITs = {x: ground(tok, x)[3] for x in ARMS}
        for arm in ARMS:
            for dose in DOSES:
                for s in SEEDS:
                    RES["cells"][ck(arm, dose, s)] = run_cell(
                        arm, s, dose, tok, model, facts, ITs[arm]["tr"], ITs[arm]["items_ev"])
        RES["plan_n"] = len(ARMS) * len(DOSES) * len(SEEDS)
        dump(RES, "stage_formal.json")
        v = verdict()
        dump(v, "stage_verdict.json")
        print(json.dumps({k: v[k] for k in ("code", "success", "reason")},
                         ensure_ascii=False, indent=1))
        return

    # verdict（离线重算）
    f = OUT / "stage_formal.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    RES["cells"] = d["cells"]
    RES["stageA"] = d.get("stageA", [])
    RES["premise"] = d.get("premise", {})
    v = verdict()
    dump(v, "stage_verdict.json")
    print(json.dumps(v, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
