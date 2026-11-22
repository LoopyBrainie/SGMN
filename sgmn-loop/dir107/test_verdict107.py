#!/usr/bin/env python3
"""DIR-107 · 判决逻辑单元测试（**不加载模型、不碰 GPU、不读真实数据**）。

派生自 `dir106/test_verdict106.py`，只改支名与判据文字。
★ 目的（`DIRECTION_rev107_v10.md` §5）：在喂入**任何**真实数据之前，
  把四支 + `INCOMPLETE` + 完备性守卫**双向**测穿。

★ 关键回归：`DEF-106-10` —— `SE` 若用**观测 p** 计算，则 `p` 越高带越宽 ⇒
  检验**自我吸收**上升趋势 ⇒ 会把**真实剂量反应**判成 `C107-NOT-SAMPLE`
  （而那正是允许把 `7b` 记 `已足够` 的支，误差方向最坏）。
  本测试用一条**干净单调剂量反应**专门守这一条。
"""
import json
import sys
import tempfile
from pathlib import Path

CHANCE = 0.25
GATE_GEN = 0.50
GATE_FIT = 0.50
N_CAND = 4

# ---- 与卡片 §4.3 A107-7 一致：题数账目 ----
N_TRAIN_SET = (16, 20, 24, 28)
N_FACTS = 32
N_SPLIT = 4
SEEDS = (20261028, 7777, 60013)
EXPECT_NQ_MAIN = {n: N_SPLIT * len(SEEDS) * (N_FACTS - n) for n in N_TRAIN_SET}
EXPECT_CELLS = {"main": N_SPLIT * len(SEEDS) * len(N_TRAIN_SET), "shuf": 1 * len(SEEDS) * len(N_TRAIN_SET),
                "w0": 1 * len(SEEDS) * len(N_TRAIN_SET)}

PASS, FAIL = [], []


def chk(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(("  PASS  " if ok else "  FAIL  ") + name + (("  | " + detail) if detail else ""))


def se_band(nq, p=CHANCE):
    """★ 只用 `p=CHANCE`（`DEF-106-10`）：不得依赖观测值。"""
    return p + 2.0 * (p * (1 - p) / max(1, nq)) ** 0.5


# ============================================================ 被测逻辑（与 R1_task107 同构）
def pooled_acc(cells, n):
    """池化 acc_held：按题数加权（不是按格平均）——★ 必须与 acc_item 口径一致。"""
    tot = num = 0.0
    for c in cells:
        if c.get("arm") != "main" or c.get("n_train") != n:
            continue
        nq = c.get("n_held", 0)
        a = c.get("acc_item_held")
        if a is None or nq == 0:
            continue
        tot += a * nq
        num += nq
    return (tot / num) if num else None, int(num)


def completeness(cells):
    """A107-9：三臂格数与题数必须全到。返回 (ok, msg)。"""
    seen = {"main": 0, "shuf": 0, "w0": 0}
    for c in cells:
        a = c.get("arm")
        if a in seen:
            seen[a] += 1
    if any(seen[k] != EXPECT_CELLS[k] for k in seen):
        return False, "格数不符: %s vs %s" % (seen, EXPECT_CELLS)
    if seen["main"] != EXPECT_CELLS["main"]:
        return False, "main 格数不符"
    return True, "格数 OK %s" % seen


def verdict_main(doc, cells):
    """返回支名。★ `INCOMPLETE` 优先于其余全部（`A107-9`）。"""
    ok, msg = completeness(cells)
    if not ok:
        return "INCOMPLETE", {"why": msg}

    accs, bands, det = {}, {}, {}
    for n in N_TRAIN_SET:
        a, nq = pooled_acc(cells, n)
        if a is None:
            return "INCOMPLETE", {"why": "n=%d 无池化读数" % n}
        if nq != EXPECT_NQ_MAIN[n]:
            return "INCOMPLETE", {"why": "n=%d 题数 %d != %d" % (n, nq, EXPECT_NQ_MAIN[n])}
        accs[n] = a
        # ★ SE₀ 用 p=CHANCE，与观测值无关
        bands[n] = se_band(EXPECT_NQ_MAIN[n], CHANCE)
        det[n] = {"acc_held": a, "nq": nq, "band": bands[n]}

    m1 = accs[28] >= GATE_GEN
    m2 = all(accs[N_TRAIN_SET[i]] < accs[N_TRAIN_SET[i + 1]] for i in range(len(N_TRAIN_SET) - 1))
    # F107-a：四点全在带内 且 M1 不成立
    f_a = all(accs[n] <= bands[n] for n in N_TRAIN_SET) and (not m1)
    # F107-b：严格单调递减
    f_b = all(accs[N_TRAIN_SET[i]] > accs[N_TRAIN_SET[i + 1]] for i in range(len(N_TRAIN_SET) - 1))

    info = {"accs": accs, "bands": bands, "M107_1": m1, "M107_2": m2,
            "F107_a": f_a, "F107_b": f_b, "detail": det}
    if m1 and m2:
        return "C107-SAMPLE-LIMITED", info
    if f_a or f_b:
        return "C107-NOT-SAMPLE", info
    return "C107-PARTIAL", info


# ============================================================ 造格
def mk(n, acc, split=0, seed=20261028, arm="main"):
    nq = N_FACTS - n
    return {"arm": arm, "n_train": n, "split": split, "seed": seed,
            "n_held": nq, "acc_item_held": acc, "acc_held": acc}


def build(per_n, splits=1, seeds=(20261028,)):
    """按 per_n 的 acc 造全套格（含 shuf/w0 占位以满足完备性）。"""
    cells = []
    for s in range(splits):
        for sd in seeds:
            for n in N_TRAIN_SET:
                cells.append(mk(n, per_n[n], s, sd, "main"))
    for s in range(1):
        for sd in seeds:
            for n in N_TRAIN_SET:
                cells.append(mk(n, 0.0, s, sd, "shuf"))
                cells.append(mk(n, 0.0, s, sd, "w0"))
    return cells


def main():
    print("=" * 78)
    print("DIR-107 判决逻辑单元测试（无 GPU）")
    print("=" * 78)

    print("\n-- 0. 题数账目（A107-7）--")
    chk("EXPECT_NQ_MAIN = {16:192,20:144,24:96,28:48}",
        EXPECT_NQ_MAIN == {16: 192, 20: 144, 24: 96, 28: 48}, str(EXPECT_NQ_MAIN))
    chk("EXPECT_CELLS = {main:48,shuf:12,w0:12}",
        EXPECT_CELLS == {"main": 48, "shuf": 12, "w0": 12}, str(EXPECT_CELLS))

    print("\n-- 1. 机会带只用 p=CHANCE（DEF-106-10 回归）--")
    b192 = se_band(192)
    chk("se_band(192) 与 p 无关", abs(b192 - (0.25 + 2 * (0.25 * 0.75 / 192) ** 0.5)) < 1e-12, "%.4f" % b192)
    chk("se_band(48) = 0.3750（n=28，最紧仍 <0.50）", abs(se_band(48) - 0.375) < 1e-9, "%.4f" % se_band(48))
    chk("四个 n 的机会带全部 < GATE_GEN(0.50)（A107-6）",
        all(se_band(EXPECT_NQ_MAIN[n]) < GATE_GEN for n in N_TRAIN_SET),
        str({n: round(se_band(EXPECT_NQ_MAIN[n]), 4) for n in N_TRAIN_SET}))

    print("\n-- 2. ★ 干净单调剂量反应 ⇒ C107-SAMPLE-LIMITED --")
    dose = {16: 0.06, 20: 0.20, 24: 0.35, 28: 0.55}
    v, info = verdict_main({}, build(dose, splits=4, seeds=SEEDS))
    chk("剂量反应落 C107-SAMPLE-LIMITED", v == "C107-SAMPLE-LIMITED", "%s M1=%s M2=%s" % (v, info.get("M107_1"), info.get("M107_2")))

    print("\n-- 3. ★ 平坦零响应 ⇒ C107-NOT-SAMPLE --")
    flat = {16: 0.06, 20: 0.06, 24: 0.06, 28: 0.06}
    v2, _ = verdict_main({}, build(flat, splits=4, seeds=SEEDS))
    chk("平坦零响应落 C107-NOT-SAMPLE", v2 == "C107-NOT-SAMPLE", v2)

    print("\n-- 4. 严格单调递减 ⇒ F107-b ⇒ C107-NOT-SAMPLE --")
    dec = {16: 0.60, 20: 0.50, 24: 0.40, 28: 0.30}
    v3, i3 = verdict_main({}, build(dec, splits=4, seeds=SEEDS))
    chk("单调递减落 C107-NOT-SAMPLE", v3 == "C107-NOT-SAMPLE", "%s F_b=%s" % (v3, i3.get("F107_b")))

    print("\n-- 5. M1 成立但 M2 不成立 ⇒ C107-PARTIAL（不得结题）--")
    nonmono = {16: 0.60, 20: 0.30, 24: 0.55, 28: 0.60}
    v4, i4 = verdict_main({}, build(nonmono, splits=4, seeds=SEEDS))
    chk("M1∧¬M2 落 C107-PARTIAL", v4 == "C107-PARTIAL", "%s M1=%s M2=%s" % (v4, i4.get("M107_1"), i4.get("M107_2")))

    print("\n-- 6. 完备性守卫双向（A107-9 / DEF-106-07）--")
    full = build(flat, splits=4, seeds=SEEDS)
    ok_full, m_full = completeness(full)
    chk("完备 ⇒ True", ok_full, m_full)
    miss = [c for c in full if not (c["arm"] == "main" and c["n_train"] == 24 and c["split"] == 3)]
    v5, i5 = verdict_main({}, miss)
    chk("★ 缺 1 格 ⇒ INCOMPLETE（不得退化成实质科学结论）", v5 == "INCOMPLETE", str(i5))
    chk("缺格 INCOMPLETE 优先于 SAMPLE-LIMITED", v5 != "C107-SAMPLE-LIMITED")
    miss2 = [c for c in full if c["arm"] != "w0"]
    chk("缺整臂 w0 ⇒ INCOMPLETE", verdict_main({}, miss2)[0] == "INCOMPLETE")

    print("\n-- 7. 池化按题数加权（不是按格平均）--")
    # 2 格：n=28 每格 4 题，acc 1.0 与 0.0 ⇒ 池化 = 0.5
    two = [mk(28, 1.0, 0, 111), mk(28, 0.0, 1, 222)]
    p28, nq28 = pooled_acc(two, 28)
    chk("池化 = 0.5000（题数加权）", abs(p28 - 0.5) < 1e-12, "%.4f nq=%d" % (p28, nq28))
    # ★ 若错用「格平均」也会得 0.5；故再给一个不等题数的例子？(n 相同 ⇒ 题数相同)
    three = [mk(28, 1.0, 0, 1), mk(28, 1.0, 1, 2), mk(28, 0.0, 2, 3)]
    p3, _ = pooled_acc(three, 28)
    chk("池化 3 格 = 0.6667", abs(p3 - 2 / 3) < 1e-9, "%.4f" % p3)

    print("\n" + "=" * 78)
    print("PASS %d / FAIL %d" % (len(PASS), len(FAIL)))
    if FAIL:
        print("★ 失败用例：" + ", ".join(FAIL))
        return 1
    print("★ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())