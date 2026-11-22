#!/opt/pytorch/bin/python
"""DIR-107 · **事后零假设诊断**（POST-HOC，**不参与冻结判据**）

★★ 目的：`R1_task107.py` 冻结的 `se_band()` 用的是 `p = CHANCE = 0.25`。
但 `acc_held` = `acc_q`，其定义（`R2_task91.py:272 acc_by_question`）是
「**同题 N_DRAWS 次全对**才算命中」，而 `N_DRAWS = 3`、`N_CAND = 4`
⇒ 纯随机预测器下 `P(acc_q 命中) = (1/4)^3 = 1/64 = 0.015625`，**不是 0.25**。

⇒ 冻结带对 **`acc_item_held`**（逐项、单次抽样、chance 0.25）是**对的**，
   对 **`acc_held`** 则**偏高约 6–24 倍**（`DEF-107-10`）。

★ 本脚本**只做报告**：
  - **不改**任何冻结阈值，**不改** `R1_task107.py` 的判决路径，
  - 不产出 `VERDICT` 分支，只在 `stage_formal.json` 上并列算出两种零假设下的带。

冻结判据的分支仍以 `R1_task107.py --stage merge` 的输出为准（`C107-NOT-SAMPLE`）。
本文件回答的是另一个问题：**「acc_held 是否在随机之上？」** —— 冻结带答不了这个问题。
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
CHANCE = 0.25          # 单次抽样、4 选 1
N_DRAWS = 3            # 与 R2_task91.N_DRAWS 一致
P0_ACCQ = CHANCE ** N_DRAWS   # = 1/64 = 0.015625，acc_q 的真实零假设
GATE_GEN = 0.50
NS = (16, 20, 24, 28)


def band(p0, nq):
    """p0 + 2·SE0：只用 p0，与观测值无关（沿用 def_106_10 的原则）。"""
    return p0 + 2.0 * (p0 * (1.0 - p0) / max(1, nq)) ** 0.5


def pooled(cells, n, key, arm="main"):
    tot = num = 0.0
    for c in cells:
        if c.get("arm") != arm or c.get("n_train") != n:
            continue
        nq = c.get("n_held") or 0
        a = c.get(key)
        if a is None or nq == 0:
            continue
        tot += a * nq
        num += nq
    return (tot / num if num else None), int(num)


def load_cells():
    """从 `out/cells/*.json` 逐格载入。

    ★ `DEF-107-11`：`merge` 先 `_dump("stage_formal")` 写含 `cells` 的 RES，
    随后 `verdict_main` → `_write(path)` 用**同一个 basename** 把判决对象写回
    ⇒ `stage_formal.json` 被**覆盖成判决**、`cells` 丢失。故此处直接读逐格文件。
    """
    cells = []
    for f in sorted((HERE / "out" / "cells").glob("*.json")):
        cells.extend(json.loads(f.read_text(encoding="utf-8")).get("cells", []))
    return cells


def main():
    cells = load_cells()
    v = json.loads((HERE / "out" / "stage_formal.json").read_text(encoding="utf-8"))
    print("=" * 92)
    print("DIR-107 事后零假设诊断（POST-HOC · 不参与冻结判据）")
    print("  载入 %d 格；冻结判决（%s）= %s"
          % (len(cells), v.get("card"), v.get("verdict")))
    print("  N_DRAWS=%d  N_CAND=%d ⇒ acc_q 零假设 p0 = (1/4)^%d = %.6f"
          % (N_DRAWS, 4, N_DRAWS, P0_ACCQ))
    print("  冻结带用 p0 = CHANCE = %.2f（对 acc_item 正确、对 acc_q 偏高）" % CHANCE)
    print("=" * 92)
    rows = []
    for n in NS:
        aq, nq = pooled(cells, n, "acc_held")
        ai, _ = pooled(cells, n, "acc_item_held")
        tr, _ = pooled(cells, n, "acc_train")
        rows.append({
            "n": n, "nq": nq,
            "aq": aq, "ai": ai, "tr": tr,
            "band_frozen": band(CHANCE, nq),
            "band_correct": band(P0_ACCQ, nq),
        })

    print("\n[A] acc_held（= acc_q，3 抽全对，真实 p0=1/64）")
    print("  %3s %5s %8s %12s %12s %10s" % ("n", "nq", "acc_held", "band(冻结 p=.25)", "band(正确 p=1/64)", "vs 正确带"))
    for r in rows:
        rel = "高于" if r["aq"] > r["band_correct"] else "不高于"
        print("  %3d %5d %8.4f %12.4f %14.4f %10s"
              % (r["n"], r["nq"], r["aq"], r["band_frozen"], r["band_correct"], rel))

    print("\n[B] acc_item_held（逐项、单次抽样，真实 p0=0.25 ⇒ 冻结带本就正确）")
    print("  %3s %8s %12s %10s" % ("n", "acc_item", "band(p=.25)", "vs 带"))
    for r in rows:
        rel = "高于" if r["ai"] > r["band_frozen"] else "不高于"
        print("  %3d %8.4f %12.4f %10s" % (r["n"], r["ai"], r["band_frozen"], rel))

    print("\n[C] 拟合（acc_train）—— 排除「没拟合」这一混淆")
    for r in rows:
        print("  n=%2d acc_train=%.4f %s GATE_FIT=%.2f"
              % (r["n"], r["tr"], "≥" if r["tr"] >= 0.50 else "<", 0.50))

    print("\n[D] 单调性（两种零假设下都看）")
    aq = [r["aq"] for r in rows]
    ai = [r["ai"] for r in rows]
    print("  acc_held      严格递减 = %s  %s" % (
        all(aq[i] > aq[i + 1] for i in range(3)), ["%.4f" % x for x in aq]))
    print("  acc_item_held 严格递减 = %s  %s" % (
        all(ai[i] > ai[i + 1] for i in range(3)), ["%.4f" % x for x in ai]))

    print("\n[E] 对照臂（区分 内容性收益 与 通用注入收益）")
    for arm in ("main", "shuf", "w0"):
        vals = {}
        for n in NS:
            a, _ = pooled(cells, n, "acc_held", arm=arm)
            vals[n] = a
        print("  %-4s %s" % (arm, {n: (round(vals[n], 4) if vals[n] is not None else None) for n in NS}))

    print("\n" + "=" * 92)
    print("★ 冻结判据分支（R1_task107.py --stage merge） = %s" % v.get("verdict"))
    print("★ 本诊断结论只用于 VERDICT.md 的『机制』与『证据边界』，不改分支。")
    print("=" * 92)

    (HERE / "out" / "diag_null.json").write_text(
        json.dumps({"p0_accq": P0_ACCQ, "chance": CHANCE, "n_draws": N_DRAWS,
                    "rows": rows, "frozen_verdict": v.get("verdict")},
                   ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()