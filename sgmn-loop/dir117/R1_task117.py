#!/usr/bin/env python3
"""DIR-117 · R1 —— 大幅增益能否救回"学到的"读出？（补 `D-147` §8-2 的范围缺口）

★ 逐字派生：**直接 import** `dir115/R1_task115.py`（它又 import `dir114`/`dir104`/`dir90`）。
  按模块名缓存 ⇒ 同一个 `D90`/`D104` 对象 ⇒ 装置位同。抄写会漂移，引用不会。
  唯一陷阱：import 会把 `D90.OUT` 指到别的目录，故下方立刻重指回 `dir117/out`。

按卡片 `revolve/DIR-117/DIRECTION_rev117_v11.md`（**rev 1.1 生效**）§7 的允许改动：
  ① `GRID` → `GAMMA_TR × MULT`（9 点，跨 5 个数量级，**不含 0**；`prior` 单独用 γ=0 测）
  ② 新增 `finite / diverged` 删失簿记（`A117-9/10`）
  ③ 新增 `learned` vs `random` 的四路对照与 `A117-7`（`Wq/Wo` 确实不同）
  ④ 判决块 `C117-*`
  ⑤ `A117-3`：从 `dir115/out/stage_formal.json` **运行时**读父卡 `γ=1e-3` 读数逐 seed 位同核对
其余装置一字未改（`set_gamma`/`acc_and_ok`/`train_both`/重封 `no_write`/`W` 不变量全部复用 `P115`）。

★ rev 1.1 更正：`A117-5` 原写「注入相对强度比值 == mult 比值」在机械上不可良定义
  （`m = Wo·sg(W)·Wq·sg(c)`、`c = sg(h)`，而 `h` 随注入改变 ⇒ `m`/`h` 都在动，比值必然不等于 mult 比值）。
  故收窄为**纯算术**断言（γ == GAMMA_TR×MULT、严格递增、比值精确），物理比值降为报告 `A117-5b` 并标注**标称**。
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
import R1_task115 as P115        # noqa: E402

OUT = _HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
D90.OUT = OUT                     # ★ import 会改指，立刻抢回

P = P115.P                        # = R1_task114（训练/事实/物品构造）
assert hasattr(P, "train_both"), "P115.P 必须是 R1_task114"

# ---------------------------------------------------------------- 冻结常量（卡片 §2 / rev 1.1 §R2）
MULT = (1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000)   # 与 D-147 逐点相同
GAMMA_TR = P115.GAMMA_TR                                       # 1e-3
GRID = tuple(GAMMA_TR * m for m in MULT)
SEEDS = (20271126, 20271203, 20271210)                         # 与 D-147 / D-146 同 3 seed
DOSES = (600, 0)                                               # d=600 训练 / d=0 未训练(=random 对照)
N_ITEM = 48
ACC0_N = 17                                                    # 无注入先验
TH_RESCUE = 36                                                 # 可用门槛（= D-147 的 TH_CONFIRM）
TH_NOSIGNAL = 18                                               # 逐 seed 容差上限
JUDGE_MULT = 10000                                             # 主判点（= D-147 oracle 48/48 处）
REF_FILE = _ROOT / "dir115" / "out" / "stage_formal.json"       # A117-3 的父卡读数

# D-147 前提读数（标称比值用；卡片 §R2 A117-5b）
M0_NORMS = (14.29, 15.59, 19.28)     # L8/L12/L16 ‖m‖
H0_NORMS = (13.85, 15.73, 19.52)     # 同层 ‖h‖

RES = {"cells": {}, "stageA": [], "plan_n": 0}


# ---------------------------------------------------------------- 小工具
def rec(name, ok, detail, blocking=True):
    RES.setdefault("stageA", []).append(
        {"name": name, "ok": bool(ok), "detail": str(detail), "blocking": bool(blocking)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return bool(ok)


def dump(o, n):
    (OUT / n).write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")


def ck(dose, seed):
    return f"SPLIT|{int(dose)}|{int(seed)}"


def plan_keys():
    return [ck(d, s) for d in DOSES for s in SEEDS]


def _w_md5(t):
    import hashlib
    return hashlib.md5(t.detach().cpu().numpy().tobytes()).hexdigest()


def ground(tok):
    """与 dir116.ground 逐字一致（只保留 SPLIT）。"""
    facts = P.build_facts_pairs(tok)
    parts = P.partitions(facts)
    csf = {}
    for g in ("SPLIT", "COLOCATED"):
        tr, ev = parts[g]
        frac, seen, n, held = P.class_seen_frac(facts, tr, ev)
        csf[g] = {"frac": frac, "seen": seen, "n": n}
    tr, ev = parts["SPLIT"]
    items_ev, meta_ev = P.make_items_with_vals(tok, facts, ev, 555)
    IT = {"SPLIT": {"tr": tr, "ev": ev, "items_ev": items_ev, "meta_ev": meta_ev,
                    "tv": {facts[i]["val"] for i in tr}, "facts": facts}}
    return facts, parts, csf, IT


def acc_of(model, st, items, tgt):
    """返回 (acc, ok, finite)。★ finite=False ⇒ 该点**删失**，不得记作任一侧读数（卡片 §2-3）。"""
    S = D104.score_matrix(model, st, items)
    # ★ `DEF-117-02`：`D104.score_matrix` 返回 **numpy**，不是 torch.Tensor ⇒ 用 np.isfinite
    S_np = S.detach().cpu().numpy() if hasattr(S, "detach") else np.asarray(S)
    if not bool(np.isfinite(S_np).all()):
        return None, None, False
    a = float(D104.acc_item_of(S, tgt))
    ok = [int(x) for x in D104.item_ok(S, tgt)]
    return (a, ok, True) if np.isfinite(a) else (None, None, False)


# ---------------------------------------------------------------- 单格
def _run_cell117_inner(seed, dose, tok, model, facts, tr, ev, items_ev, st):
    t0 = time.time()
    losses, viol, diverged = [], 0, False
    if int(dose) > 0:
        losses, viol, diverged, _gt = P.train_both(
            model, st, tok, facts, tr, random.Random(1000 + seed % 100), int(dose))
    # ★ `DEF-109-02`：训练末尾会把 no_write 还原 ⇒ 评测前必须重封
    for m in st.mems.values():
        m.no_write = True
    assert all(m.no_write for m in st.mems.values()), "评测前必须重封 no_write"

    wc0 = {li: m.write_calls for li, m in st.mems.items()}
    W0 = {li: D90.W_of(m).detach().clone() for li, m in st.mems.items()}
    tgt = D104.targets_of(items_ev)

    # ---- 先验：γ = 0（注入项必须精确为 0）
    P115.set_gamma(st, 0.0)
    # ★ `DEF-117-03`：`P115.read_gamma` 返回 **{层: float} 字典**，不是标量 ⇒ 必须逐层比对
    g_prior = P115.read_gamma(st)
    a0, ok0, f0 = acc_of(model, st, items_ev, tgt)
    prior_n = int(sum(ok0)) if ok0 else None

    # ---- 9 点网格
    acc, okg, finite, gamma_ok = {}, {}, {}, {}
    for mult in MULT:
        g = GAMMA_TR * mult
        P115.set_gamma(st, g)
        got = P115.read_gamma(st)
        # ★ γ 存在 **float32** tensor 里（0.3 等不可精确表示）⇒ 用 float32 相称的容差
        gamma_ok[f"{mult:g}"] = bool(all(abs(v - g) <= 1e-5 * max(1.0, abs(g))
                                        for v in got.values()))
        a, ok, fin = acc_of(model, st, items_ev, tgt)
        k = f"{mult:g}"
        acc[k], okg[k], finite[k] = a, ok, bool(fin)

    # ---- 不变量：W 逐位不变、写入为 0
    W_same = all(torch.equal(W0[li], D90.W_of(m)) for li, m in st.mems.items())
    wc_after = {li: m.write_calls for li, m in st.mems.items()}

    return {
        "seed": int(seed), "dose": int(dose),
        "prior_n": prior_n, "prior_acc": a0, "gamma_prior": {k: v for k, v in g_prior.items()},
        "gamma_prior_all_zero": bool(all(abs(v) == 0.0 for v in g_prior.values())),
        "acc_by_mult": acc, "ok_by_mult": okg, "finite_by_mult": finite,
        "gamma_ok": gamma_ok,
        "n_finite": int(sum(1 for v in finite.values() if v)),
        "diverged_mult": [k for k, v in finite.items() if not v],
        "inj_ratio_nominal": {f"{m:g}": float(GAMMA_TR * m * np.mean(M0_NORMS) / np.mean(H0_NORMS))
                              for m in MULT},
        "train_losses_tail": [float(x) for x in losses[-3:]] if losses else [],
        "violations": int(viol), "train_diverged": bool(diverged),
        "eval_writes": int(sum(wc_after.values()) - sum(wc0.values())),
        "W_bitwise_same": bool(W_same),
        "wq_md5": _w_md5(st.mems[sorted(st.mems)[0]].Wq),
        "wo_md5": _w_md5(st.mems[sorted(st.mems)[0]].Wo),
        "sec": round(time.time() - t0, 1),
    }


def run_cell117(seed, dose, tok, model, facts, tr, ev, items_ev):
    """★ `DEF-117-06`：`D90.make_stack` 会把 forward hook **挂在 model 上且不自动摘除**，
    而 `nll_of_candidate` 完全不用 `st`（只跑 `model(seq)`）⇒ 残留 hook 会**持续注入**，
    污染后续每一格的读数（实测：第 1 格 prior=17 正确，第 2 格起被前一格的 γ=1e-3 拖到 13）。
    ⇒ hook 的生命周期必须由 `try/finally` 拥有，异常时也不能泄漏。"""
    st = D90.make_stack(gamma=GAMMA_TR, no_write=True, seed=seed)
    try:
        return _run_cell117_inner(seed, dose, tok, model, facts, tr, ev, items_ev, st)
    finally:
        st.remove()


# ---------------------------------------------------------------- Stage A
def stage_a(tok, model):
    facts, parts, csf, IT = ground(tok)
    it = IT["SPLIT"]
    RES["coverage"] = csf
    RES["plan_n"] = len(plan_keys())
    ok = True
    print(f"\n=== Stage A：{RES['plan_n']} 格 ===", flush=True)

    # A117-1 划分覆盖（装置接续）
    rec("A117-0a", abs(csf["SPLIT"]["frac"] - 1.0) < 1e-9,
        f"SPLIT class_seen_frac_held={csf['SPLIT']['frac']:.4f}（须 1.0）", blocking=False)

    # A117-8 留出 item 数与逐项同一
    mine = it["items_ev"]
    ref = D90.make_eval_items(tok, facts, it["ev"], random.Random(555))
    same = (len(ref) == len(mine) and all(
        r["prompt_ids"] == m["prompt_ids"]
        and [c[0] for c in r["cand"]] == [c[0] for c in m["cand"]]
        for r, m in zip(ref, mine)))
    ok &= rec("A117-8", len(mine) == N_ITEM and same,
              f"留出 item 数 = {len(mine)}（须 {N_ITEM}），与 D90.make_eval_items 逐项一致={same}")

    # 跑 6 格
    for dose in DOSES:
        for seed in SEEDS:
            k = ck(dose, seed)
            print(f"  跑 {k} …", flush=True)
            RES["cells"][k] = run_cell117(seed, dose, tok, model, it["facts"], it["tr"],
                                          it["ev"], it["items_ev"])
            c = RES["cells"][k]
            print(f"    prior={c['prior_n']}  9 点 acc="
                  f"{[None if c['acc_by_mult'][f'{m:g}'] is None else round(c['acc_by_mult'][f'{m:g}']*48) for m in MULT]}"
                  f"  n_finite={c['n_finite']}  {c['sec']}s", flush=True)
            dump(RES, "stage_stageA.json")

    C = RES["cells"]

    # A117-1 先验 = 17（6 格）
    pn = {k: v["prior_n"] for k, v in C.items()}
    ok &= rec("A117-1", all(x == ACC0_N for x in pn.values()),
              f"γ=0 先验正确项数 = {pn}（须全为 {ACC0_N}）")

    # A117-2 γ=0 使注入精确为 0
    ok &= rec("A117-2", all(v["gamma_prior_all_zero"] for v in C.values()),
              f"γ=0 时 read_gamma 逐层全为 0.0 ⇒ 注入项精确 0")

    # A117-3 mult=1 与父卡 γ=1e-3 逐 seed 位同
    ref_ok, ref_detail = False, "父卡文件缺失 ⇒ 装置接续未验证"
    if REF_FILE.exists():
        rf = json.loads(REF_FILE.read_text(encoding="utf-8")).get("cells", {})
        pairs, miss = [], []
        for dose in DOSES:
            for seed in SEEDS:
                rk = ck(dose, seed)
                rv = rf.get(rk, {}).get("acc_by_gamma", {}).get("1.00000e-03")
                mine_v = C[rk]["acc_by_mult"]["1"]
                if rv is None or mine_v is None:
                    miss.append(rk)
                else:
                    pairs.append((rk, round(rv * N_ITEM), round(mine_v * N_ITEM)))
        ref_ok = (not miss) and all(a == b for _, a, b in pairs)
        ref_detail = (f"mult=1 vs 父卡 γ=1e-3 逐 seed（父,本卡）={[(k, a, b) for k, a, b in pairs]}"
                      f"{'；缺 ' + str(miss) if miss else ''}")
    ok &= rec("A117-3", ref_ok, ref_detail, blocking=REF_FILE.exists())

    # A117-4 set_gamma 生效（9 点 × 6 格）
    allg = all(all(v["gamma_ok"].values()) for v in C.values())
    ok &= rec("A117-4", allg, f"9 点 × {len(C)} 格 read_gamma 全部回报 == 设定值")

    # A117-5（rev 1.1 收窄）：γ 精确 == GAMMA_TR×MULT、严格递增、相邻比值 == MULT 比值
    gs = [GAMMA_TR * m for m in MULT]
    ratios_ok = all(abs(gs[i + 1] / gs[i] - MULT[i + 1] / MULT[i]) < 1e-9 for i in range(len(gs) - 1))
    inc = all(gs[i + 1] > gs[i] for i in range(len(gs) - 1))
    ok5 = rec("A117-5", inc and ratios_ok,
              f"γ 严格递增={inc}，相邻比值 == MULT 比值={ratios_ok}（γ = {['%.3g' % g for g in gs]}）")

    # A117-5b 标称注入强度（报告）
    r0 = C[ck(DOSES[0], SEEDS[0])]["inj_ratio_nominal"]
    rec("A117-5b", True, f"标称 ‖γ·m‖/‖h‖ 逐点 = {({k: '%.2e' % v for k, v in r0.items()})}"
                         f"（★ 标称：m₀/h₀ 取自 γ=0，不含 h 的反馈）", blocking=False)

    # A117-6 读侧不变量
    ok &= rec("A117-6", all(v["eval_writes"] == 0 and v["violations"] == 0
                            and v["W_bitwise_same"] and not v["train_diverged"] for v in C.values()),
              f"6 格：eval_writes=0、violations=0、W 逐位不变、未发散 ⇒ "
              f"{all(v['eval_writes'] == 0 for v in C.values())}")

    # A117-7 learned 与 random 的 Wq/Wo 确实不同
    differ = True
    for seed in SEEDS:
        a = C[ck(600, seed)]
        b = C[ck(0, seed)]
        if a["wq_md5"] == b["wq_md5"] or a["wo_md5"] == b["wo_md5"]:
            differ = False
    ok &= rec("A117-7", differ, "同 seed 的 d=600 与 d=0 的 Wq/Wo md5 均不同 ⇒ random 对照真是随机")

    # A117-9 删失簿记完整
    ok &= rec("A117-9", all(len(v["acc_by_mult"]) == len(MULT) and len(v["finite_by_mult"]) == len(MULT)
                            and all((v["finite_by_mult"][k] and v["acc_by_mult"][k] is not None)
                                    or (not v["finite_by_mult"][k] and v["acc_by_mult"][k] is None)
                                    for k in v["acc_by_mult"])
                            for v in C.values()),
              f"6 格 × 9 点全部有 finite 布尔，且非有限点的 acc 必为 null（互斥自洽）")

    # A117-10 / A117-11 报告
    rec("A117-10", True, f"逐格有限点个数 = { {k: v['n_finite'] for k, v in C.items()} }；"
                         f"diverged 点 = { {k: v['diverged_mult'] for k, v in C.items() if v['diverged_mult']} }",
        blocking=False)
    for dose in DOSES:
        for seed in SEEDS:
            c = C[ck(dose, seed)]
            rec("A117-11", True,
                f"{ck(dose, seed)} 全形（项/48）= "
                f"{[('X' if c['acc_by_mult'][f'{m:g}'] is None else round(c['acc_by_mult'][f'{m:g}'] * 48)) for m in MULT]}",
                blocking=False)

    # A117-12 四路对照（in 主判点；报告）
    rec("A117-12", True, "四路对照见判据块 four_way（报告）", blocking=False)

    dump(RES, "stage_stageA.json")     # ★ `DEF-117-04`：守卫跑完后必须重写，否则文件里只有前 2 条
    bad = [x for x in RES["stageA"] if x["blocking"] and not x["ok"]]
    print(f"\n★ Stage A {'通过' if not bad else '失败'}；阻断失败 {len(bad)} 条", flush=True)
    return not bad


# ---------------------------------------------------------------- 判据
def verdict():
    C = RES["cells"]
    out = {"code": None, "success": False}
    need = [ck(d, s) for d in DOSES for s in SEEDS]
    short = [k for k in need if k not in C or len(C[k]["acc_by_mult"]) != len(MULT)
             or C[k]["prior_n"] is None]
    tr_keys = [ck(600, s) for s in SEEDS]
    out["n_cells"] = len(C)
    out["plan_n"] = RES.get("plan_n", 0)
    out["prior_n"] = {k: C[k]["prior_n"] for k in C} if C else {}
    out["thresholds"] = {"acc0": ACC0_N, "rescue": TH_RESCUE, "nosignal": TH_NOSIGNAL,
                         "judge_mult": JUDGE_MULT}

    if short:
        out["code"] = "C117-INCOMPLETE"
        out["success"] = False
        out["reason"] = f"缺格/网格短/读数不全：{short}"
        return out

    # ★ §5-2 装置漂移门（`DEF-117-05`：首版漏了这一条，导致在漂移的装置上照样给出实质判决）
    drift = [x for x in RES.get("stageA", []) if x.get("blocking") and not x.get("ok")]
    if drift:
        out["code"] = "C117-DEVICE-DRIFT"
        out["success"] = False
        out["reason"] = ("阻断守卫失败 ⇒ 不得给判决：" +
                         "；".join(f"{x['name']}={x['detail'][:80]}" for x in drift))
        return out

    # 主判点与删失降级（预先指定）
    def fin(k, m):
        return C[k]["finite_by_mult"][f"{m:g}"] and C[k]["acc_by_mult"][f"{m:g}"] is not None

    judge = JUDGE_MULT
    degraded, note = False, ""
    if not all(fin(k, JUDGE_MULT) for k in tr_keys):
        cand = [m for m in sorted(MULT, reverse=True) if all(fin(k, m) for k in tr_keys)]
        note = (f"mult={JUDGE_MULT} 在训练格上非有限 ⇒ 降级；合法降级点={cand}")
        if len(cand) < 3:
            out["code"] = "C117-CENSORED"
            out["success"] = False
            out["reason"] = f"主判点删失且合法降级点 <3 ⇒ 判据不得计算。{note}"
            return out
        judge = cand[0]
        degraded = True

    per = [round(C[k]["acc_by_mult"][f"{judge:g}"] * N_ITEM) for k in tr_keys]
    mean = float(np.mean(per))
    mx = max(per)
    out.update({"judge_mult": judge, "judge_degraded": degraded, "judge_note": note,
                "per_seed_n48": per, "mean_n48": round(mean, 3), "max_n48": mx})

    # 曲线全形（训练格均值）
    out["shape_mean_n48"] = {f"{m:g}": (round(float(np.mean(
        [C[k]["acc_by_mult"][f"{m:g}"] * N_ITEM for k in tr_keys if fin(k, m)])), 3)
        if any(fin(k, m) for k in tr_keys) else None) for m in MULT}

    # 四路对照（同 mult=judge）：oracle 引 D-147
    rand_per = [round(C[ck(0, s)]["acc_by_mult"][f"{judge:g}"] * N_ITEM)
                if fin(ck(0, s), judge) else None for s in SEEDS]
    out["four_way"] = {"mult": judge,
                       "oracle_n48": 48, "oracle_src": "D-147（mult=1e4；judge 相同则直接可比）",
                       "learned_n48_mean": round(mean, 3), "learned_per_seed": per,
                       "random_n48_mean": (round(float(np.mean([x for x in rand_per if x is not None])), 3)
                                           if any(x is not None for x in rand_per) else None),
                       "random_per_seed": rand_per,
                       "prior_n48": ACC0_N}

    if mean >= TH_RESCUE:
        out["code"] = "C117-GAIN-RESCUES"
        out["success"] = False        # H117 被否证
        out["reason"] = (f"主判点 mult={judge} 训练格均值 {mean:.1f}/48 ≥ {TH_RESCUE} "
                         f"⇒ 增益可救 ⇒ H117 被否证，DIR-115 的结论须更正为『仅低幅段成立』")
        return out
    if TH_NOSIGNAL < mx < TH_RESCUE:
        out["code"] = "C117-INCONCLUSIVE-BAND"
        out["success"] = False
        out["reason"] = (f"主判点 mult={judge} 有 seed 达 {mx}/48（> {TH_NOSIGNAL}）但均值 {mean:.1f} "
                         f"未达 {TH_RESCUE} ⇒ 带内，无判决")
        return out
    if mean <= ACC0_N and mx <= TH_NOSIGNAL:
        out["code"] = "C117-GAIN-CANNOT-RESCUE"
        out["success"] = True         # H117 证实
        out["reason"] = (f"主判点 mult={judge} 训练格均值 {mean:.1f}/48 ≤ 先验 {ACC0_N} 且逐 seed 最大 "
                         f"{mx}/48 ≤ {TH_NOSIGNAL} ⇒ 在 D-147 的同一增益下学到的读出仍不高于先验 "
                         f"⇒ 瓶颈是方向，不是增益（H117 证实）"
                         + (f"；★ 已降级：{note}" if degraded else ""))
        return out
    out["code"] = "C117-INCONCLUSIVE-BAND"
    out["success"] = False
    out["reason"] = f"均值 {mean:.1f} ≤ 先验但有 seed {mx}/48 ≥ 19 ⇒ 带内（卡片 §5-7）"
    return out


# ---------------------------------------------------------------- main
TOK, MODEL = None, None


def main():
    global TOK, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A", choices=("A", "smoke", "formal", "verdict"))
    a = ap.parse_args()

    if a.stage == "verdict":
        crit = verdict()
        RES["crit"] = crit
        dump(RES, "stage_verdict.json")
        print("\n=== 判据 ===\n" + json.dumps(crit, ensure_ascii=False, indent=1), flush=True)
        return 0

    TOK, MODEL = D90.model_and_tok()

    if a.stage == "A":
        ok = stage_a(TOK, MODEL)
        crit = verdict()
        RES["crit"] = crit
        dump(RES, "stage_verdict.json")
        print("\n=== 判据 ===\n" + json.dumps(crit, ensure_ascii=False, indent=1), flush=True)
        return 0 if ok else 1

    if a.stage == "smoke":
        facts, parts, csf, IT = ground(TOK)
        it = IT["SPLIT"]
        c = run_cell117(SEEDS[0], 600, TOK, MODEL, it["facts"], it["tr"], it["ev"], it["items_ev"])
        dump({"smoke": c}, "stage_smoke.json")
        print(f"\n★ smoke：{c['sec']}s/格 ⇒ {len(plan_keys())} 格 ≈ "
              f"{c['sec'] * len(plan_keys()) / 60:.1f} min", flush=True)
        return 0

    # formal：先把 Stage A 的读数全部继承过来
    sa = OUT / "stage_stageA.json"
    if not sa.exists():
        print("★ 缺 stage_stageA.json ⇒ 不得进正式轮", flush=True)
        return 2
    prev = json.loads(sa.read_text(encoding="utf-8"))
    for k, v in prev.get("cells", {}).items():
        RES["cells"].setdefault(k, v)
    RES["stageA"] = prev.get("stageA", [])
    RES["plan_n"] = prev.get("plan_n", 0)
    RES["coverage"] = prev.get("coverage", {})
    bad = [x for x in RES["stageA"] if x["blocking"] and not x["ok"]]
    if bad:
        print(f"★ Stage A 有 {len(bad)} 条阻断失败 ⇒ 不得进正式轮", flush=True)
        return 2
    print(f"Stage A 门通过；继承 {len(RES['cells'])} 格", flush=True)

    facts, parts, csf, IT = ground(TOK)
    it = IT["SPLIT"]
    for dose in DOSES:
        for seed in SEEDS:
            k = ck(dose, seed)
            if k in RES["cells"]:
                print(f"  跳过（已有）{k}", flush=True)
                continue
            RES["cells"][k] = run_cell117(seed, dose, TOK, MODEL, it["facts"], it["tr"],
                                          it["ev"], it["items_ev"])
            dump(RES, "stage_formal.json")
    dump(RES, "stage_formal.json")

    crit = verdict()
    RES["crit"] = crit
    dump(RES, "stage_verdict.json")
    print("\n=== 判据 ===\n" + json.dumps(crit, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())