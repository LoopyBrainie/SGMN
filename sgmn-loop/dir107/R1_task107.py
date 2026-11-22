#!/usr/bin/env python3
"""DIR-107 · 在**有效答案类覆盖度**下重测 `§7.2-7b` 的**样本量轴**。

卡片：`revolve/DIR-107/DIRECTION_rev107_v10.md`（rev 1.0 冻结）。

派生关系（逐字，只改 §4.1 的**唯一自变量**）：
- **值分配** ← `dir93/R1_task93.py:build_facts` 的类覆盖方案
  （`_C = VALUE_POOL[:SPLIT_CLASSES]`、`vals = [_C[i % SPLIT_CLASSES] ...]`、
  `assert len(set(vals)) == SPLIT_CLASSES`）。**一字未改。**
- **测量层** ← `dir106/R1_task106.py:run_cell`（**直接调用，不改**）。
- **唯一改动** ← 顺序生成 `_make_order_class`（顶替 `dir106.make_order`）。

★ 为什么必须改顺序：`dir106.make_order` 是**均匀随机置换**，随机取 16/32 条
  **不保证覆盖全 16 类**（期望只覆盖 ≈12 类）⇒ 若照抄，`class_seen_frac_held`
  会退回 <1 且随 `split` 抖动 ⇒ 又变成一条被钉住的度量（`P107-0` 审计 / `C-137`）。

★ `_make_order_class` 的**结构**（`A107-8` 要求与 `n_train` 无关）：
  `order = [每类第 1 条（按类的置换序）] + [每类第 2 条（同一置换序）]`
  ⇒ 对**任何** `n ∈ {16,20,24,28}`，`order[:n]` 都含全部 16 类（n≥16）
  ⇒ `class_seen_frac_held ≡ 1.0`（冻结前已实测 3 seed × 4 split × 4 n = 48 组合，min=max=1.0）。
"""
import argparse
import json
import os
import random
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ★ 必须把**项目根**也放进 sys.path：`sgmn_module.py` 在根目录，而直接跑
#   `dir107/R1_task107.py` 时 `sys.path[0]` 是 `dir107/` ⇒ 否则 `import sgmn_module` 失败。
#   （`R2_task91.py:29` 会插根，但那要等它被 import，而本文件先 import `sgmn_module`。）
for _d in ("", "dir91", "dir100", "dir106"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from sgmn_module import LayerMem
from R2_task91 import N_CAND, N_FACTS, build_chunks, model_and_tok

import R1_task106 as D106          # ★ 测量层（不改）
import R2_task91 as R91

# ★ 捕获「未被任何 patch 污染」的原始方法，供 `A107-11` 判泄漏
ORIG_WRITE_LOCAL = LayerMem.write
ORIG_READ_LOCAL = LayerMem.read

# ---------------------------------------------------------------- 冻结常量
CARD = "DIR-107 rev 1.2"
CHANCE = 1.0 / N_CAND                              # 0.25
GATE_FIT = 0.50                                    # 沿用 D-91 冻结值
GATE_GEN = 0.50                                    # 沿用 D-91 冻结值
N_TRAIN_SET = (16, 20, 24, 28)                     # 主扫描
SPLIT_CLASSES = 16                                 # ★ dir93:61
N_SPLIT = 4
SEEDS = (20261028, 7777, 60013)

# ★ 题数账目：4 split × 3 seed × (32 - n)
EXPECT_NQ_MAIN = {n: N_SPLIT * len(SEEDS) * (N_FACTS - n) for n in N_TRAIN_SET}
EXPECT_CELLS = {"main": N_SPLIT * len(SEEDS) * len(N_TRAIN_SET),
                "shuf": 1 * len(SEEDS) * len(N_TRAIN_SET),
                "w0": 1 * len(SEEDS) * len(N_TRAIN_SET)}

# ★ rev 1.2 的装置忠实性锚：`D-136`/`A106-2` 在 n=16 的 `acc_train`（逐位同源）
D136_ACC_TRAIN_N16 = 0.9375
# ★ `D-121` 的 n=16 `acc_held`：**只作登记**（rev 1.2 §12.1：其模板与本卡不同，不可复现）
D121_ACC_HELD_N16 = 0.0625
ANCHOR_TOL = 0.0625

OUT = __import__("pathlib").Path(__file__).resolve().parent / "out"

RES = {"card": CARD, "stage": None, "cells": [], "guards": {}}
FAILS = []


# ---------------------------------------------------------------- 录
def rec(cid, ok, detail, value=None):
    RES["guards"][cid] = {"ok": bool(ok), "detail": detail, "value": value}
    print(("  PASS  " if ok else "  FAIL  ") + cid + " | " + str(detail), flush=True)
    if not ok:
        FAILS.append(cid)


def rec_only(cid, detail, value=None):
    RES["guards"][cid] = {"ok": None, "detail": detail, "value": value}
    print("  (rec) " + cid + " | " + str(detail), flush=True)


# ---------------------------------------------------------------- 值分配（★ 抄 dir93）
def build_facts_class(tok):
    """★ rev 1.1（`DEF-107-02` 修正）：**事实构造逐字用 `R2_task91.build_facts`**（= `dir106`
    装置，V3 多样模板），**只把 `val` 换成 `dir93` 的类覆盖赋值**并重算派生字段。

    rev 1.0 曾把 `dir93` 的**整段** `build_facts` 抄来，连**单一句子模板**一起继承
    ⇒ 偷偷引入第二个自变量 ⇒ `acc_train` 0.4375 < `GATE_FIT`（`dir106` 同点 0.9375）
    ⇒ `A107-10` FAIL。见 `DIRECTION_rev107_v11.md` §9.2。
    """
    facts = R91.build_facts(tok)                     # ★ 键/句子/块与 dir106 逐字同源
    # ★★★ 类覆盖赋值（只抄 dir93 的这 4 行）
    _C = R91.VALUE_POOL[:SPLIT_CLASSES]
    vals = [_C[i % SPLIT_CLASSES] for i in range(N_FACTS)]
    assert len(set(vals)) == SPLIT_CLASSES, (len(set(vals)), SPLIT_CLASSES)
    for i, f in enumerate(facts):
        f["val"] = vals[i]                           # ★ 本卡唯一改动
        f["sent"] = R91.sentence_of(i, f)            # 用新 val 重算（按 i 确定、吃 f["val"]）
        f["prompt"] = f["sent"].split(" " + f["val"] + ",")[0]
        assert f["sent"].startswith(f["prompt"] + " " + f["val"]), (f["sent"], f["prompt"])
        f["val_ids"] = tok(" " + f["val"], add_special_tokens=False)["input_ids"]
        f["prompt_ids"] = tok(f["prompt"], add_special_tokens=False)["input_ids"]
    return facts


# ---------------------------------------------------------------- 顺序（★ 本卡唯一改动）
def _make_order_class(seed, split):
    """类分层置换：`[每类第1条…] + [每类第2条…]`（★ 与 `n_train` 无关，`A107-8`）。"""
    g = {c: [] for c in range(SPLIT_CLASSES)}
    _vals = [R91.VALUE_POOL[i % SPLIT_CLASSES] for i in range(N_FACTS)]
    _uniq = sorted(set(_vals))
    for i, v in enumerate(_vals):
        g[_uniq.index(v)].append(i)
    perm = list(range(SPLIT_CLASSES))
    random.Random(seed * 100 + split).shuffle(perm)
    first = [g[c][0] for c in perm]
    second = [g[c][1] for c in perm]
    order = first + second
    assert sorted(order) == list(range(N_FACTS)), "顺序必须是 0..31 的置换"
    return order


def _old_vals():
    """复现**旧**分配（`i%27`+shuffle），供 `A107-5` 负向对照（抄 dir93:_old_vals）。"""
    rng = random.Random(9091)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    seen, n = set(), 0
    while n < N_FACTS:
        k = "".join(rng.choice(alphabet) for _ in range(5))
        if k in seen:
            continue
        seen.add(k)
        n += 1
    v = [R91.VALUE_POOL[i % len(R91.VALUE_POOL)] for i in range(N_FACTS)]
    rng.shuffle(v)
    return v


def _split_ceiling(vals, n_train):
    """(训练覆盖类数, 留出覆盖数, 上界) —— 纯算术（抄 dir93:_split_ceiling，加 n 参数）。"""
    cls = [sorted(set(vals)).index(v) for v in vals]
    tr, he = list(range(n_train)), list(range(n_train, N_FACTS))
    trc = {cls[i] for i in tr}
    cov = sum(1 for i in he if cls[i] in trc)
    return len(trc), cov, (cov / len(he) if he else None)


# ---------------------------------------------------------------- 单格
def run_one(model, tok, facts, chunks, owner, seed, split, n_train, arm, stage):
    """★ 调用 `dir106.run_cell`（测量层不改），前后做守卫与补丁一致性检查。"""
    # ★★ 关键：`dir106.run_cell` 的 finally 只在 `ORIG_WRITE is not None` 时还原
    #    `LayerMem.write/read`。这些是 **dir106 的模块全局**，只在 `dir106.main()` 里设置。
    #    直接调用会留下 monkey-patch ⇒ 必须自己先设好（否则泄漏到后续格）。
    t0 = time.time()
    D106.ORIG_WRITE = LayerMem.write
    D106.ORIG_READ = LayerMem.read
    w_before, r_before = LayerMem.write, LayerMem.read
    try:
        cell = D106.run_cell(model, tok, facts, chunks, owner, seed, split, n_train, arm, stage)
    finally:
        leak = (LayerMem.write is not w_before) or (LayerMem.read is not r_before)
        if leak:
            LayerMem.write, LayerMem.read = w_before, r_before
    cell["leak_restored"] = bool(leak)
    cell["card"] = CARD
    cell["sec"] = round(time.time() - t0, 1)
    return cell


# ---------------------------------------------------------------- Stage A
def stage_a(model, tok, facts, chunks, owner):
    print("\n===== Stage A 守卫（DIR-107 rev 1.2）=====\n", flush=True)

    # ---- A107-2：补丁生效 + 四个 n 上覆盖度 = 1.0
    D106.make_order = _make_order_class
    patched = D106.make_order is _make_order_class
    covs = []
    for n in N_TRAIN_SET:
        for sd in SEEDS:
            for sp in range(N_SPLIT):
                order = _make_order_class(sd, sp)
                tr, he = order[:n], order[n:]
                trc = {facts[i]["val"] for i in tr}
                covs.append(sum(1 for i in he if facts[i]["val"] in trc) / len(he))
                assert len(tr) == n and len(he) == N_FACTS - n
    rec("A107-2", patched and min(covs) == 1.0 and max(covs) == 1.0,
        "make_order 已替换=%s；class_seen_frac_held 48 组合 min=%.4f max=%.4f" % (patched, min(covs), max(covs)),
        {"min": min(covs), "max": max(covs), "n": len(covs)})

    # ---- A107-8：order 与 n_train 无关；且各 n 前缀的**类集合嵌套**（剂量轴干净）
    nested_ok, n16 = True, []
    for sd in SEEDS:
        for sp in range(N_SPLIT):
            o = _make_order_class(sd, sp)
            chain = [{facts[i]["val"] for i in o[:m]} for m in N_TRAIN_SET]
            nested_ok &= all(chain[k] <= chain[k + 1] for k in range(len(chain) - 1))
            nested_ok &= (len(chain[0]) == SPLIT_CLASSES)
            n16.append(len(chain[0]))
    rec("A107-8", nested_ok,
        "同一 (seed,split) 的 4 个 n 前缀类集合嵌套、order[:16] 覆盖 %d 类（须 %d）⇒ 与 n_train 无关"
        % (min(n16), SPLIT_CLASSES), None)

    # ---- A107-5：负向对照（旧分配在同一 n=16 上 = 0.0625）
    ov = _old_vals()
    _, cov_o, ceil_o = _split_ceiling(ov, 16)
    rec("A107-5", abs(ceil_o - 0.0625) < 1e-12,
        "旧分配 i%%27+shuffle 在 n=16 上留出覆盖 %d/16 ⇒ 上界 %.4f（须 0.0625）" % (cov_o, ceil_o), ceil_o)

    # ---- A107-6：阈值可达性（机会带全部 < 0.50）
    bands = {n: CHANCE + 2.0 * (CHANCE * (1 - CHANCE) / EXPECT_NQ_MAIN[n]) ** 0.5 for n in N_TRAIN_SET}
    rec("A107-6", all(bands[n] < GATE_GEN for n in N_TRAIN_SET),
        "CHANCE+2SE0 全部 < %.2f（n=28 最紧 = %.4f）" % (GATE_GEN, bands[28]),
        {str(k): round(v, 4) for k, v in bands.items()})

    # ---- A107-7：题数账目
    rec("A107-7", EXPECT_NQ_MAIN == {16: 192, 20: 144, 24: 96, 28: 48}
        and EXPECT_CELLS == {"main": 48, "shuf": 12, "w0": 12},
        "EXPECT_NQ_MAIN=%s；EXPECT_CELLS=%s" % (EXPECT_NQ_MAIN, EXPECT_CELLS), None)

    # ---- A107-1a/1b/1c：跑 n=16 一格（split0, seed 20261028）
    print("\n  [A107-1] 跑 n=16 单格（split=0, seed=20261028）…", flush=True)
    c = run_one(model, tok, facts, chunks, owner, 20261028, 0, 16, "main", "stageA")
    RES["cells"] = [c]
    a_tr, a_held = c.get("acc_train"), c.get("acc_held")

    # A107-1a：装置忠实性 —— acc_train 复现 D-136/A106-2 的 n=16 读数 0.9375
    rec("A107-1a", a_tr is not None and abs(a_tr - D136_ACC_TRAIN_N16) <= 0.0625,
        "n=16 acc_train=%.4f vs D-136/A106-2 锚 %.4f（容差 ±0.0625）★ rev 1.2 新锚"
        % (a_tr if a_tr is not None else -1, D136_ACC_TRAIN_N16), a_tr)
    # A107-1b：可解释性 —— acc_held ≥ 0 且严格小于 acc_train（装置有判别力）
    rec("A107-1b", a_held is not None and a_held >= 0 and a_tr is not None and a_held < a_tr,
        "n=16 acc_held=%.4f 须 ≥0 且 < acc_train %.4f" % (a_held if a_held is not None else -1,
                                                         a_tr if a_tr is not None else -1), a_held)
    # A107-1c：与 D-121 之差 —— **只登记、不判定**（rev 1.2 §12.3）
    rec_only("A107-1c",
             "n=16 acc_held=%.4f vs D-121 的 %.4f；差 %+.4f（★ D-121 用 dir93 统一模板，不可复现，只登记）"
             % (a_held if a_held is not None else -1, D121_ACC_HELD_N16,
                (a_held or 0) - D121_ACC_HELD_N16),
             a_held)
    rec("A107-10", (a_tr or 0) >= GATE_FIT,
        "n=16 acc_train=%.4f 须 ≥ %.2f（★ 锚在**最易**端，DEF-106-05）" % (a_tr or -1, GATE_FIT),
        a_tr)

    # ---- A107-3：逐格覆盖度（本格）
    csf = c.get("class_seen_frac_held")
    rec("A107-3", csf is not None and csf >= 0.95, "本格 class_seen_frac_held=%.4f 须 ≥0.95" % (csf or -1), csf)

    # ---- A107-4：W_drift / 基座指纹
    drift = c.get("W_drift")
    fp_ok = c.get("base_fp_before") == c.get("base_fp_after")
    rec("A107-4", (drift == 0.0) and fp_ok,
        "W_drift=%s（须 0.0）；base_fp 不变=%s（%s）" % (drift, fp_ok, c.get("base_fp_after")),
        {"W_drift": drift, "fp": c.get("base_fp_after")})

    # ---- A107-11：monkey-patch 不泄漏（DEF-107-01）
    ok11 = (LayerMem.write is ORIG_WRITE_LOCAL) and (LayerMem.read is ORIG_READ_LOCAL)
    rec("A107-11", ok11,
        "LayerMem.write/read 已还原为原方法（leak_restored=%s）" % c.get("leak_restored"),
        c.get("leak_restored"))


# ---------------------------------------------------------------- 判决
def pooled_acc(cells, n, arm="main", key="acc_held"):
    """按 `n_held` 加权池化某臂在某 `n` 上的**答案选择准确率**。

    ★ `DEF-107-09` 修正：本函数原先**硬写** `arm == "main"` 且读 **`acc_item_held`**，
    而卡片 §3 冻结的测量对象是 **`acc_held`**（`= acc_q`，即 `D-121`/`D-136` 引用的那个量）
    ⇒ 旧实现在 `detail[n]["acc_held"]` 里装的其实是**逐项**准确率，属**字段误标**；
    且因硬写 `main`，`shuf`/`w0` 对照臂全部返回 `None`（`aux` 全 `null`）。
    现将 `arm`/`key` 参数化：判据用 `acc_held`，对照臂与 `acc_item_held` 作为辅报。
    """
    tot = num = 0.0
    for c in cells:
        if c.get("arm") != arm or c.get("n_train") != n:
            continue
        nq = c.get("n_held", 0) or 0
        a = c.get(key)
        if a is None or nq == 0:
            continue
        tot += a * nq
        num += nq
    return (tot / num) if num else None, int(num)


def se_band(nq):
    """★ 只用 p=CHANCE，与观测值无关（DEF-106-10）。"""
    return CHANCE + 2.0 * (CHANCE * (1 - CHANCE) / max(1, nq)) ** 0.5


def verdict_main(path):
    cells = RES.get("cells") or []
    seen = {"main": 0, "shuf": 0, "w0": 0}
    for c in cells:
        if c.get("arm") in seen:
            seen[c["arm"]] += 1
    v = {"card": CARD, "n_cells": len(cells), "cells_by_arm": seen,
         "expect_cells": EXPECT_CELLS, "guards": RES["guards"]}

    # A107-9 完备性（优先于其余全部）
    if any(seen[k] != EXPECT_CELLS[k] for k in seen):
        v.update({"verdict": "INCOMPLETE", "why": "格数不符 %s vs %s" % (seen, EXPECT_CELLS)})
        _write(path, v)
        return v

    accs, bands, det = {}, {}, {}
    for n in N_TRAIN_SET:
        a, nq = pooled_acc(cells, n)
        if a is None or nq != EXPECT_NQ_MAIN[n]:
            v.update({"verdict": "INCOMPLETE", "why": "n=%d 池化缺失或题数 %s != %d" % (n, nq, EXPECT_NQ_MAIN[n])})
            _write(path, v)
            return v
        accs[n], bands[n] = a, se_band(EXPECT_NQ_MAIN[n])
        det[n] = {"acc_held": a, "nq": nq, "band": bands[n]}

    # A107-3 逐格覆盖度 → 作废不合格格
    bad = [c for c in cells if (c.get("class_seen_frac_held") is not None
                                and c["class_seen_frac_held"] < 0.95)]
    if bad:
        v.update({"verdict": "INCOMPLETE",
                  "why": "%d 格 class_seen_frac_held<0.95 ⇒ 作废" % len(bad)})
        _write(path, v)
        return v

    m1 = accs[28] >= GATE_GEN
    m2 = all(accs[N_TRAIN_SET[i]] < accs[N_TRAIN_SET[i + 1]] for i in range(len(N_TRAIN_SET) - 1))
    f_a = all(accs[n] <= bands[n] for n in N_TRAIN_SET) and (not m1)
    f_b = all(accs[N_TRAIN_SET[i]] > accs[N_TRAIN_SET[i + 1]] for i in range(len(N_TRAIN_SET) - 1))

    # 必须同时报出（§2.4）；★ `DEF-107-09`：对照臂需传 arm，否则恒为 None
    aux = {}
    for arm in ("shuf", "w0"):
        aux[arm] = {str(n): pooled_acc(cells, n, arm=arm)[0] for n in N_TRAIN_SET}
    # ★ 辅报：逐项准确率（旧实现的判据量，现仅作参考，不作判据）
    aux["acc_item_held"] = {str(n): pooled_acc(cells, n, key="acc_item_held")[0]
                             for n in N_TRAIN_SET}
    aux["acc_train"] = {}
    for n in N_TRAIN_SET:
        vals = [c.get("acc_train") for c in cells
                if c.get("arm") == "main" and c.get("n_train") == n and c.get("acc_train") is not None]
        aux["acc_train"][str(n)] = float(np.mean(vals)) if vals else None
    aux["W_drift_max"] = max([c.get("W_drift") or 0.0 for c in cells] or [0.0])

    if m1 and m2:
        branch = "C107-SAMPLE-LIMITED"
    elif f_a or f_b:
        branch = "C107-NOT-SAMPLE"
    else:
        branch = "C107-PARTIAL"

    v.update({"verdict": branch, "acc_held": accs, "bands": bands, "detail": det,
              "M107_1": m1, "M107_2": m2, "F107_a": f_a, "F107_b": f_b, "aux": aux})
    _write(path, v)
    return v


def _write(path, v):
    p = OUT / os.path.basename(path or "stage_verdict.json")
    if not str(p).endswith(".json"):
        p = OUT / "stage_verdict.json"
    p.write_text(json.dumps(v, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n[判决] %s ⇒ %s" % (p, v.get("verdict")), flush=True)
    for k in ("acc_held", "bands", "M107_1", "M107_2", "F107_a", "F107_b", "why"):
        if k in v:
            print("   %-8s %s" % (k, v[k]), flush=True)


# ---------------------------------------------------------------- 驱动
def _setup():
    tok, model = model_and_tok()               # ★ 返回顺序 (tok, model)
    facts = build_facts_class(tok)             # ★ 本卡的值分配
    chunks, owner = build_chunks(tok, facts)
    return model, tok, facts, chunks, owner


def _dump(name):
    p = OUT / ("%s.json" % name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("[完成] → %s" % p, flush=True)


def _load_cells(cd):
    import pathlib
    p = pathlib.Path(cd)
    if not p.is_absolute():
        p = OUT.parent / p
    cells = []
    for f in sorted(p.glob("*.json")):
        cells.extend(json.loads(f.read_text(encoding="utf-8")).get("cells", []))
    cells.sort(key=lambda c: (str(c["arm"]), c["n_train"], c["split"], c["seed"]))
    print("[merge] 载入 %d 格（%s）" % (len(cells), p), flush=True)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stageA",
                    choices=["stageA", "smoke", "formal", "merge", "adjudicate"])
    ap.add_argument("--arm", default="main", choices=["main", "w0", "shuf"])
    ap.add_argument("--n-train", type=int, default=-1)
    ap.add_argument("--split", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--cells-dir", default="out/cells")
    ap.add_argument("--from", dest="from_json", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RES["stage"] = a.stage

    if a.stage == "merge":
        RES["cells"] = _load_cells(a.cells_dir)
        _dump("stage_formal")
        verdict_main(a.from_json or "stage_formal.json")
        return
    if a.stage == "adjudicate":
        RES["cells"] = _load_cells(a.cells_dir)
        verdict_main(a.from_json or "stage_formal.json")
        return

    model, tok, facts, chunks, owner = _setup()
    print("[装置] 事实 %d；块 %d；值类 %d；候选 %d；chance=%.2f"
          % (len(facts), len(chunks), len({f["val"] for f in facts}), N_CAND, CHANCE), flush=True)

    if a.stage == "stageA":
        stage_a(model, tok, facts, chunks, owner)
        _dump("stage_stageA")
        print("\n★ Stage A %s" % ("全部通过" if not FAILS else "有 FAIL：%s" % FAILS), flush=True)
        raise SystemExit(0 if not FAILS else 1)

    # ★ formal/smoke：一格一进程（DEF-105-02：同配置不同前序历史不可比）
    D106.make_order = _make_order_class            # ★ 补丁（本卡唯一改动）
    n_train = a.n_train if a.n_train > 0 else N_TRAIN_SET[0]
    split = a.split if a.split >= 0 else 0
    seed = a.seed if a.seed > 0 else SEEDS[0]
    cell = run_one(model, tok, facts, chunks, owner, seed, split, n_train, a.arm, a.stage)
    RES["cells"] = [cell]
    print("[格] arm=%s n=%d split=%d seed=%d acc_train=%.4f acc_held=%s csf=%s %ss"
          % (cell["arm"], cell["n_train"], cell["split"], cell["seed"],
             cell.get("acc_train") or -1, cell.get("acc_held"),
             cell.get("class_seen_frac_held"), cell["sec"]), flush=True)
    _dump("cell_%s_%d_%d_%d" % (cell["arm"], cell["n_train"], cell["split"], cell["seed"]))


if __name__ == "__main__":
    main()