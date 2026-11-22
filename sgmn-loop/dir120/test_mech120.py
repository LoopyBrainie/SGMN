"""`test_mech120.py` —— `DIR-120` 测量层的自测（**本地无 GPU / 无 torch 可跑**）。

    /usr/bin/python3 test_mech120.py             # 正常跑，应全部 PASS、exit 0
    /usr/bin/python3 test_mech120.py --inject 3  # 注入第 3 条，应恰 1 条 FAIL、exit 1

★★ 纪律 `DEF-118-17`：**每一条都必须能红**。`--inject n` 故意把第 n 条的期望值改错，
  若它仍然 PASS ⇒ 说明这条断言**没有真的在测东西**，必须重写。

★ 手算例（卡片 §3）在这里**逐位**断言：它本身就是被测对象，不是"示例"。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

import mech120 as M

# ---------------------------------------------------------------- 卡片 §3 手算例（冻结）
TV = {"a", "d"}
CAND = [["a", "b", "c", "d"],      # i0: n_seen(j!=1) = 2 (a, d)
        ["p", "q", "r", "s"],      # i1: n_seen = 0
        ["a", "e", "f", "g"],      # i2: n_seen = 1 (a)
        ["a", "b", "c", "d"]]      # i3: n_seen(j!=1) = 3 (a, c, d)
META = [{"cand_val": c} for c in CAND]
TGT = [1, 0, 1, 1]
PICK = [0, 0, 1, 3]                # i0→a(见) i1→p(未见) i2→e(未见) i3→d(见)
PRIOR_PICK = [0, 0, 0, 0]          # prior 只在 i1 上答对


def _part():
    return M.partition(PICK, META, TV, TGT)


def _null_seen_stable():
    a, obs = M.null_top3(PICK, META, TV, "seen", B=200, seed=20261122)
    b, obs2 = M.null_top3(PICK, META, TV, "seen", B=200, seed=20261122)
    return (a.shape == b.shape) and bool(np.array_equal(a, b)) and (obs == obs2)


def _null_unseen_stable():
    a, obs = M.null_top3(PICK, META, TV, "unseen", B=200, seed=20261122)
    b, obs2 = M.null_top3(PICK, META, TV, "unseen", B=200, seed=20261122)
    return (a.shape == b.shape) and bool(np.array_equal(a, b)) and (obs == obs2)


#: ★ ⑮⑯ 的专用池。两处都必须绕开，否则断言没有分辨力：
#:   ① 不能用 `TV={a,d}` —— 只有 2 个值时 top-3 占比**恒为 1.0**，观测与零假设永远相等；
#:   ② 填充值必须**每项互不共享** —— 若各项候选高度重叠，零假设**自己也**能抽到 top3=1.0
#:      ⇒ `q99 == 1.0` ⇒ 观测即使 100% 集中也判不出来（实测踩过）。
POOL = [f"v{i}" for i in range(32)]
TVBIG = set(POOL)


def _items(picked):
    """每项 = 被选值 + 3 个**只在本项出现**的填充值 ⇒ 零假设必然摊开。"""
    out, k = [], 0
    fill = [v for v in POOL if v not in set(picked)]
    for v in picked:
        out.append({"cand_val": [v] + fill[k:k + 3]})
        k += 3
    return out


def _null_detects_concentration():
    """★ 8 项见侧选择**全落在同一个值**上时，置换零假设必须判它**超出** `q99`。"""
    picked = [POOL[0]] * 8      # 8 项**全落在同一个值**上
    meta = _items(picked)
    pick = [m["cand_val"].index(v) for m, v in zip(meta, picked)]
    arr, obs = M.null_top3(pick, meta, TVBIG, "seen", B=600, seed=20261122)
    return obs > M.null_summary(arr, obs)["q99"]


def _null_not_flagged_when_uniform():
    """★ 反向：见侧选择**摊到 8 个互不相同的值**上时，零假设**不应**判超。

    （防"零假设永远报显著"这种把判据废掉的实现。）
    """
    picked = POOL[:8]          # 8 项摊到 8 个互不相同的值上
    meta = _items(picked)
    pick = [m["cand_val"].index(v) for m, v in zip(meta, picked)]
    arr, obs = M.null_top3(pick, meta, TVBIG, "seen", B=600, seed=20261122)
    return obs <= M.null_summary(arr, obs)["q99"]


def _identity_random():
    """★ 随机 200 组上断言 `E_total == E_n0 + E_esc_ok`。

    构造必须守住前提：**正确值 ∉ tv**，且 `tv` 里的值只作干扰项。
    """
    import random
    rng = random.Random(20261122)
    seen_pool = ["a", "d"]
    unseen_pool = ["b", "c", "e", "f", "g", "p", "q", "r", "s"]
    bad = 0
    for _ in range(200):
        n = rng.choice([6, 12, 48])
        meta, tgt, pick = [], [], []
        for _i in range(n):
            tv_local = set(seen_pool)
            # 正确值必为未见值
            correct = rng.choice(unseen_pool)
            rest = [v for v in (seen_pool + unseen_pool) if v != correct]
            rng.shuffle(rest)
            cand = [correct] + rest[:3]
            rng.shuffle(cand)
            t = cand.index(correct)
            meta.append({"cand_val": cand})
            tgt.append(t)
            pick.append(rng.randrange(4))
        p = M.partition(pick, meta, tv_local, tgt)
        if p["E_total"] != p["E_n0"] + p["E_esc_ok"]:
            bad += 1
    return bad


TESTS = [
    # ── §3 手算例（12 个冻结读数）────────────────────────────────────
    ("① `n_has_seen == 3`（i1 的 `n_seen=0` 不算）", lambda: _part()["n_has_seen"], 3),
    ("② `escape == 1`（只有 i2：有见过候选却选了未见值）", lambda: _part()["escape"], 1),
    ("③ `r == 1/3`（分母是 3，**不是** 4）",
     lambda: round(_part()["r"], 12), round(1.0 / 3.0, 12)),
    ("④ `E_total == 2`（i1、i2 答对）", lambda: _part()["E_total"], 2),
    ("⑤ `E_n0 == 1`（i1 是 `n_seen=0` 项答对）", lambda: _part()["E_n0"], 1),
    ("⑥ `E_esc_ok == 1`（i2 是「逃逸且正确」＝残差）", lambda: _part()["E_esc_ok"], 1),
    ("⑦ ★ 恒等式 `E_total == E_n0 + E_esc_ok`（§0-3，计数恒等）",
     lambda: _part()["E_total"] - (_part()["E_n0"] + _part()["E_esc_ok"]), 0),
    ("⑧ `n11 == 1`（只有 i1 在 prior 与本次都答对）",
     lambda: M.n11_of(PICK, PRIOR_PICK, TGT), 1),
    ("⑨ `q == 1.0`（逃逸项 1 个、其中答对 1 个）", lambda: M.q_of(_part()), 1.0),
    ("⑩ `q0 == 1/3`（★ 总体是**逃逸项**：只有 i2 逃逸，`|U|=3` ⇒ 1/3；rev1.1 更正）",
     lambda: round(M.q0_of(_part(), META, TV, TGT), 9), round(1.0 / 3.0, 9)),
    ("⑪ `top3_seen == 1.0`（见侧选 {a,d} 各 1 次）", lambda: M.top3_share(_part()["seen_vals"]), 1.0),
    ("⑫ `top3_unseen == 1.0`（未见侧选 {p,e} 各 1 次）",
     lambda: M.top3_share(_part()["unseen_vals"]), 1.0),

    # ── 零假设的性质 ────────────────────────────────────────────────
    ("⑬ 置换零假设**同种子两次逐位相同**（见侧）", _null_seen_stable, True),
    ("⑭ 置换零假设**同种子两次逐位相同**（未见侧）", _null_unseen_stable, True),
    ("⑮ ★ 零假设**能**判出极端集中（100% 落同一值 ⇒ 超 `q99`）",
     _null_detects_concentration, True),
    ("⑯ ★ 零假设**不会**乱报显著（选择摊开时 ⇒ 不超 `q99`）",
     _null_not_flagged_when_uniform, True),

    # ── `top3_share` 的边界 ─────────────────────────────────────────
    ("⑰ `top3_share` 空列表 ⇒ None（不是 0，避免「没数据」被当成「不集中」）",
     lambda: M.top3_share([]), None),
    ("⑱ `top3_share` 单一值 N 次 ⇒ 1.0", lambda: M.top3_share(["x"] * 7), 1.0),
    ("⑲ `top3_share` 4 个互不相同 ⇒ 3/4（top-3 是 3 个单次）",
     lambda: M.top3_share(["a", "b", "c", "d"]), 0.75),

    # ── 恒等式的**普适性**（随机数据上也必须成立）────────────────────
    ("⑳ ★ 恒等式在随机数据上任意 200 组都成立（正确值必未见 ⇒ 选见过必错）",
     _identity_random, 0),

    # ── 分母语义 ────────────────────────────────────────────────────
    ("㉑ `r` 的分母是 `n_has_seen` 而非 `n`（全无见过候选时 `r` 为 None）",
     lambda: M.partition([0], [{"cand_val": ["p", "q", "r", "s"]}], TV, [0])["r"], None),
]




def main():
    inject = None
    if "--inject" in sys.argv:
        inject = int(sys.argv[sys.argv.index("--inject") + 1])
    print("mech120 自测（DIR-120 §3 手算例 + 零假设性质 + 恒等式普适性）")
    print("-" * 84)
    fails = 0
    for i, (tag, fn, want) in enumerate(TESTS, 1):
        if inject == i:
            want = ("__INJECTED__",) if not isinstance(want, tuple) else ("__INJECTED__",)
        try:
            got = fn()
        except Exception as e:                                    # noqa: BLE001
            got = f"<EXC {type(e).__name__}: {e}>"
        good = (got == want)
        if not good:
            fails += 1
        print(f"  [{'PASS' if good else 'FAIL'}] {tag}: 实测={got!r} 期望={want!r}")
    print("-" * 84)
    print(f"断言失败条数 = {fails}")
    print("★ 全部 PASS" if fails == 0 else "✗ 有 FAIL")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())