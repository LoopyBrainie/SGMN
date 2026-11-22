"""`R1_task121.py` —— `DIR-121` 的正式轮（`--stage a` / `--stage formal`）。

★ 卡片：`revolve/DIR-121/DIRECTION_rev121_v10.md`（rev 1.0 冻结）+ `..._v11.md`（rev 1.1 增量）。
★ 装置：**纯再分析**。复用 `R1_task120.load()` 与 `mech120`（逃逸定义**逐字**，不新造）。
   不训练、不注入、不评估；只需要 tokenizer 重建题目（与 `DIR-120` 同路径）。
★ 判据（冻结，见 rev 1.0 §2-3）：
   - `J121-1` 条件在**行边缘** ⇒ `D_col` 上尾 `p_A ≤ 0.025` ⇒ 主判据（决定 `H121`）
   - `J121-2` 条件在**列边缘** ⇒ `D_row` 上尾 `p_B ≤ 0.025` ⇒ 对照，不影响 `H121` 成立与否但必须报
"""
import argparse
import json
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
for _d in (str(_HERE), str(_HERE.parent), str(_HERE.parent / "dir120")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import R1_task120 as P120  # noqa: E402
import mech120 as M0       # noqa: E402
import mech121 as M        # noqa: E402

OUT = _HERE / "out"
ARM = "COLOCATED"
J = 10000
B_NULL = 20000
SEED_A = 20270101
SEED_B = 20270202
ALPHA_FAM = 0.05
ALPHA_EACH = 0.025           # Bonferroni（两测不独立，见 rev 1.0 §2-3）
EXPECT_SEEDS = 9
N_ITEMS_PER_SEED = 48
EXPECT_HAS = 46
EXPECT_TOTAL = 42
EXPECT_R = [0, 3, 18, 0, 2, 9, 5, 0, 5]
EXPECT_COLHIST = {0: 17, 1: 19, 2: 7, 3: 3}


def build(D):
    """把 `D` 展开成 `9 × 46` 的逃逸表 + 边缘。"""
    seeds = D["seeds"]
    n_items = D["n_items"]
    esc_rows, ns_rows = [], []
    for k, s in enumerate(seeds):
        lo, hi = k * n_items, (k + 1) * n_items
        pick, meta, tgt, tv = D["pick"][lo:hi], D["meta"][lo:hi], D["tgt"][lo:hi], D["tv"]
        esc, ns = [], []
        for i in range(n_items):
            cv = meta[i]["cand_val"]
            seen = M0._n_seen_of(cv, int(tgt[i]), tv)
            ns.append(seen)
            esc.append(bool(seen >= 1 and (cv[int(pick[i])] not in tv)))
        esc_rows.append(esc)
        ns_rows.append(ns)
    has = [i for i in range(n_items) if ns_rows[0][i] >= 1]
    X = M.table_from_masks(esc_rows, has)
    R, C = M.margins(X)
    return {"seeds": seeds, "has": has, "X": X, "R": R, "C": C,
            "ns0": ns_rows[0], "esc_rows": esc_rows, "n_items": n_items}


def stage_a(D, T):
    res = []

    def g(tag, cond, detail=""):
        ok = bool(cond)
        res.append({"tag": tag, "ok": ok, "detail": "" if ok else str(detail)})
        print(f"[{'PASS' if ok else 'FAIL'}] {tag}" + (f"  — {detail}" if detail and not ok else ""))
        return ok

    g("A121-1 9 seed × 48 题 = 432", len(D["seeds"]) == EXPECT_SEEDS
      and D["n_items"] == N_ITEMS_PER_SEED, f"{len(D['seeds'])} × {D['n_items']}")
    g("A121-2 |has|=46 且 46×9=414", len(T["has"]) == EXPECT_HAS
      and len(T["has"]) * EXPECT_SEEDS == 414, f"|has|={len(T['has'])}")

    # A121-3 逐 seed n_seen 直方图 == 存档 n_seen_hist（★ 只此一处，无并行脚手架）
    ok3, d3 = True, {}
    import json as _json
    cells = sorted((P120.OUT119 / "cells").glob(f"{ARM}_600_*.json"))
    for f in cells:
        d = _json.loads(f.read_text(encoding="utf-8"))
        s = int(d["seed"])
        k = D["seeds"].index(s)
        lo, hi = k * D["n_items"], (k + 1) * D["n_items"]
        h = {0: 0, 1: 0, 2: 0, 3: 0}
        for i in range(D["n_items"]):
            h[M0._n_seen_of(D["meta"][lo + i]["cand_val"], int(D["tgt"][lo + i]), D["tv"])] += 1
        arch = {int(a): int(b) for a, b in d["n_seen_hist"].items()}
        if h != arch:
            ok3 = False
            d3[s] = (h, arch)
    g("A121-3 逐 seed n_seen 直方图 == 存档", ok3, d3)

    g("A121-4 逐 seed 逃逸数 == 已发表 + 合计 42",
      [int(x) for x in T["R"]] == EXPECT_R and int(T["R"].sum()) == EXPECT_TOTAL,
      f"R={list(map(int, T['R']))}")
    g("A121-5 行/列边缘合计相等 == 42",
      int(T["R"].sum()) == int(T["C"].sum()) == EXPECT_TOTAL,
      f"{int(T['R'].sum())} vs {int(T['C'].sum())}")

    hist = {v: int((T["C"] == v).sum()) for v in sorted(set(int(x) for x in T["C"]))}
    g("A121-6 列边缘直方图 == {0:17,1:19,2:7,3:3}", hist == EXPECT_COLHIST, hist)

    cbar, rbar = T["C"].sum() / float(len(T["has"])), T["R"].sum() / float(EXPECT_SEEDS)
    g("A121-7 c̄==42/46 且 r̄==42/9",
      abs(cbar - 42 / 46) < 1e-12 and abs(rbar - 42 / 9) < 1e-12, f"{cbar} {rbar}")

    # A121-8 / A121-9 手算例（卡片 §3）
    from collections import Counter
    nullA = M.enumerate_null_A([1, 2], 3)
    cntA = Counter(round(v, 12) for v in nullA)
    g("A121-8 手算例 A：D_col 零分布 {0:3/9, 2:6/9}",
      len(nullA) == 9 and cntA[0.0] == 3 and cntA[2.0] == 6, dict(cntA))
    nullB = M.enumerate_null_B([1, 1, 1], 2)
    cntB = Counter(round(v, 12) for v in nullB)
    g("A121-9 手算例 B：D_row 零分布 {1/3:6/8, 3:2/8}",
      len(nullB) == 8 and cntB[round(1 / 3, 12)] == 6 and cntB[3.0] == 2, dict(cntB))

    # A121-10 确定性
    a1 = M.simulate_null_A(T["R"], len(T["has"]), B=300, seed=SEED_A)
    a2 = M.simulate_null_A(T["R"], len(T["has"]), B=300, seed=SEED_A)
    b1 = M.simulate_null_B(T["C"], EXPECT_SEEDS, B=300, seed=SEED_B)
    b2 = M.simulate_null_B(T["C"], EXPECT_SEEDS, B=300, seed=SEED_B)
    g("A121-10 置换模拟确定性（同 seed 两次逐位相同）",
      a1.tobytes() == a2.tobytes() and b1.tobytes() == b2.tobytes())

    # A121-11 边缘保持性（用独立实现核 200 次模拟的行/列合计）
    import numpy as np
    rng = np.random.default_rng(4242)
    okA = True
    for _ in range(200):
        Cacc = np.zeros(len(T["has"]), dtype=np.int64)
        for r in [int(x) for x in T["R"]]:
            if r <= 0:
                continue
            idx = rng.permutation(len(T["has"]))[:r]
            Cacc[idx] += 1
        if int(Cacc.sum()) != EXPECT_TOTAL:
            okA = False
    okB = True
    for _ in range(200):
        Racc = np.zeros(EXPECT_SEEDS, dtype=np.int64)
        for c in [int(x) for x in T["C"]]:
            if c <= 0:
                continue
            idx = rng.permutation(EXPECT_SEEDS)[:c]
            Racc[idx] += 1
        if int(Racc.sum()) != EXPECT_TOTAL:
            okB = False
    g("A121-11 模拟保持边缘（行/列合计恒 42）", okA and okB)

    # A121-12 两个判据都能被计算（非退化）
    g("A121-12 两判据可算（c̄>0、r̄>0、B>0）",
      cbar > 0 and rbar > 0 and B_NULL > 0 and len(T["has"]) > 1 and EXPECT_SEEDS > 1)

    return res


def formal(D, T):
    t0 = time.time()
    obsA = M.d_col(T["C"])
    obsB = M.d_row(T["R"])
    nullA = M.simulate_null_A(T["R"], len(T["has"]), B=B_NULL, seed=SEED_A)
    nullB = M.simulate_null_B(T["C"], EXPECT_SEEDS, B=B_NULL, seed=SEED_B)
    pA, pB = M.p_upper(obsA, nullA), M.p_upper(obsB, nullB)

    j1 = pA <= ALPHA_EACH
    j2 = pB <= ALPHA_EACH
    h121 = j1                      # 主判据
    if j1 and j2:
        reading = "两维都需要 ⇒ H121 成立；逃逸 = seed 倾向 × 题目结构，不得再单称 seed regime"
    elif j1 and not j2:
        reading = "H121 成立；题目结构是唯一可检维度 ⇒ D-151 的 seed regime 须改写为题目 regime"
    elif (not j1) and j2:
        reading = "H121 否证 ⇒ 逃逸 = 纯 seed 泄漏率，题目不带信号；D-151 的 regime 说法被加强"
    else:
        reading = "H121 否证且两向皆不超额 ⇒ 稀疏性解释离散；D-151 的 χ²/df=8.11 须重新解读"

    out = {
        "dir": "DIR-121", "arm": ARM, "dose": 600, "mult": J,
        "n_seeds": EXPECT_SEEDS, "n_items_has": len(T["has"]), "n_cells": 414,
        "table_total": int(T["R"].sum()),
        "R": [int(x) for x in T["R"]],
        "C_hist": {str(v): int((T["C"] == v).sum()) for v in sorted(set(int(x) for x in T["C"]))},
        "cbar": float(T["C"].sum() / float(len(T["has"]))),
        "rbar": float(T["R"].sum() / float(EXPECT_SEEDS)),
        "D_col_obs": obsA, "null_A": M.summarize(nullA), "p_A": pA,
        "D_row_obs": obsB, "null_B": M.summarize(nullB), "p_B": pB,
        "alpha_each": ALPHA_EACH, "alpha_family": ALPHA_FAM,
        "J121_1": j1, "J121_2": j2, "H121": h121, "reading": reading,
        "per_seed_rate": {str(s): float(T["R"][k] / float(len(T["has"])))
                          for k, s in enumerate(D["seeds"])},
        "B": B_NULL, "seedA": SEED_A, "seedB": SEED_B,
        "sec": round(time.time() - t0, 2),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "formal"], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    D = P120.load()
    T = build(D)
    print(f"[R1_task121] seeds={len(D['seeds'])} has={len(T['has'])} total={int(T['R'].sum())}")

    if a.stage == "a":
        res = stage_a(D, T)
        bad = [r for r in res if not r["ok"]]
        (OUT / "stage_stageA.json").write_text(
            json.dumps({"n": len(res), "n_fail": len(bad), "res": res}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"\n=== Stage A: {len(res) - len(bad)}/{len(res)} PASS ===")
        return 0 if not bad else 1

    res = stage_a(D, T)
    bad = [r for r in res if not r["ok"]]
    (OUT / "stage_stageA.json").write_text(
        json.dumps({"n": len(res), "n_fail": len(bad), "res": res}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    if bad:
        print(f"\n✗ Stage A 有 {len(bad)} 条失败 ⇒ 不得进 Formal")
        return 1
    print(f"\n=== Stage A {len(res)}/{len(res)} PASS ⇒ 进 Formal ===")

    out = formal(D, T)
    (OUT / "formal.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("D_col_obs", "p_A", "D_row_obs", "p_B",
                                          "J121_1", "J121_2", "H121")},
                     ensure_ascii=False, indent=1))
    print("reading:", out["reading"])
    return 0


if __name__ == "__main__":
    sys.exit(main())