"""`test_mech122.py` —— `DIR-122` 测量层自测（**纯 numpy，本地可跑，不需要 GPU/torch**）。

★ 纪律来源（本 loop 反复踩过）：
  - `DEF-120-01` / `DEF-121-01`：手算例只写在文档里 ⇒ 一路带进正式跑才被发现例外
    ⇒ **手算例必须被断言**。本文件把卡片 §3-1（`Γ=6`）与 §3-2（`Γ=−2`）逐位断言。
  - `DEF-120-03`：零假设若「不可能失败」就不算检验 ⇒ 额外断言**零分布有分辨力**。
  - `DEF-120-04` / `DEF-121-02`：断言目标一律是**标量/列表**；**不留空转占位脚手架**。
  - `DEF-122-01`：断言形状必须与被测对象逐字对齐（稠密 vs 稀疏直方图）。

跑法：`/usr/bin/python3 test_mech122.py`（**必须全 PASS**，`rc=0`）。
每条都支持 `--inject <编号>` 强制失败，用来证明断言**能红**（不是空转）。
"""
import sys

import numpy as np

import mech122 as M

FAILED = []
PASSED = []

_CALLS = {"n": 0}
INJECT = None
for _i, _a in enumerate(sys.argv):
    if _a == "--inject":
        INJECT = int(sys.argv[_i + 1])


def chk(tag, cond, detail=""):
    """★ `--inject k` 强制第 k 条**执行**的检查变红（证明每条断言都能失败，不是空转）。"""
    _CALLS["n"] += 1
    ok = bool(cond)
    if INJECT is not None and _CALLS["n"] == INJECT:
        ok = False
        detail = f"injected failure on check #{INJECT}"
    (PASSED if ok else FAILED).append(tag)
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}" + (f"  — {detail}" if detail and not ok else ""))
    return ok


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


# ================================================================ 手算例 A（卡片 §3-1）
def case_A():
    E = np.array([[0, 0, 0],
                  [1, 1, 1],
                  [2, 2, 2]], dtype=np.int64)
    Z = E - E.mean(axis=0, keepdims=True)
    chk("T122-1a 手算例 A 列均值 == 1", all(close(v, 1.0) for v in E.mean(axis=0)),
        f"got {E.mean(axis=0)}")
    chk("T122-1b 手算例 A 的 Z == [[-1,-1,-1],[0,0,0],[1,1,1]]",
        Z.tolist() == [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], f"got {Z.tolist()}")
    chk("T122-1c 手算例 A：Γ == 6（逐位）", close(M.gamma_stat(E), 6.0), f"got {M.gamma_stat(E)}")


# ================================================================ 手算例 B（卡片 §3-2）
def case_B():
    E = np.array([[0, 2, 0],
                  [1, 1, 1],
                  [2, 0, 2]], dtype=np.int64)
    Z = E - E.mean(axis=0, keepdims=True)
    chk("T122-2a 手算例 B 的 Z == [[-1,1,-1],[0,0,0],[1,-1,1]]",
        Z.tolist() == [[-1.0, 1.0, -1.0], [0.0, 0.0, 0.0], [1.0, -1.0, 1.0]], f"got {Z.tolist()}")
    chk("T122-2b 手算例 B：Γ == -2（逐位）", close(M.gamma_stat(E), -2.0), f"got {M.gamma_stat(E)}")
    # ★ 有符号 ⇒ 上尾检验有方向、不是恒正（DEF-120-03 同型防护）
    chk("T122-2c Γ 能取负值（上尾检验有方向）", M.gamma_stat(E) < 0, f"got {M.gamma_stat(E)}")


# ================================================================ D_row 回归（DIR-121 冻结值）
def case_drow_regression():
    # DIR-121 手算例 A 的 R（`test_mech121.T121-1f`：D_row == 1/3）
    R = np.array([1, 2], dtype=np.int64)
    chk("T122-3a D_row 回归：R=[1,2] ⇒ 1/3", close(M.mech121.d_row(R), 1.0 / 3.0),
        f"got {M.mech121.d_row(R)}")
    # DIR-121 手算例 B 的 R（= [1,2] 同值；另取 R=[0,3] 独立核一遍公式）
    chk("T122-3b D_row 公式：R=[0,6] ⇒ (9+9)/3 = 6",
        close(M.mech121.d_row(np.array([0, 6], dtype=np.int64)), 6.0),
        f"got {M.mech121.d_row(np.array([0, 6], dtype=np.int64))}")


# ================================================================ Γ 置换零分布的性质
def case_gamma_null():
    E = np.array([[0, 0, 0],
                  [1, 1, 1],
                  [2, 2, 2]], dtype=np.int64)
    g_obs = M.gamma_stat(E)
    null = M.simulate_null_gamma(E, B=3000, seed=999)
    chk("T122-4a Γ 零分布长度 == B", len(null) == 3000, f"got {len(null)}")
    # ★ 手算（卡片未写，此处补足）：三列 Z 全为 [-1,0,1]。
    #   任一对列内积 Σ_s Z_a(s)Z_b(s) = 2 ⟺ 两列置换**完全相同**；否则最大只有 1。
    #   ⇒ Γ = 6 ⟺ 三列置换完全一致。3 列的置换组合共 (3!)^3 = 216 种，其中一致的 3! = 6 种
    #   ⇒ **P(Γ ≥ 6) = 6/216 = 1/36 ≈ 0.027778**。
    #   （首版写「p ≤ 0.01」是**随手阈值**，与精确值不符 ⇒ 假 FAIL；`DEF-122-03`。）
    p = M.mech121.p_upper(g_obs, null)
    chk("T122-4b 完全对齐 ⇒ p ≈ 1/36（手算 6/216）", abs(p - 1.0 / 36.0) <= 0.012,
        f"got p={p}, 期望 ≈ {1.0/36.0:.6f}（B=3000 的 3σ ≈ 0.009）")
    # ★ 零分布有分辨力：最大值 >= 观测（否则上尾退化成恒 0）
    chk("T122-4c 零分布有分辨力（max >= 观测）", float(null.max()) >= g_obs,
        f"max={null.max()}, obs={g_obs}")
    # ★ 零分布均值约 0（伪对齐不该有系统正偏）
    chk("T122-4d 零分布均值 ≈ 0（|<mean>| < 0.5）", abs(float(null.mean())) < 0.5,
        f"mean={null.mean()}")
    # 确定性
    null2 = M.simulate_null_gamma(E, B=3000, seed=999)
    chk("T122-4e 置换确定性（同 seed 逐位相同）", np.array_equal(null, null2), "mismatch")

    # 反对齐的观测应 p 很大（≥0.5）
    Eb = np.array([[0, 2, 0], [1, 1, 1], [2, 0, 2]], dtype=np.int64)
    pb = M.mech121.p_upper(M.gamma_stat(Eb), null)
    chk("T122-4f 反对齐 ⇒ p 很大（≥ 0.5）", pb >= 0.5, f"got p={pb}, g={M.gamma_stat(Eb)}")


# ================================================================ 精确枚举 vs 蒙特卡洛
def _enumerate_gamma_all(E):
    """**全枚举** `(S!)^J` 种逐列置换下的 `Γ`（小例子用：S=3,J=3 ⇒ 216 种）。"""
    from itertools import permutations
    E = np.asarray(E, dtype=float)
    S, J = E.shape
    Z = E - E.mean(axis=0, keepdims=True)
    perms = list(permutations(range(S)))
    out = []
    for combo in __import__("itertools").product(perms, repeat=J):
        Zp = np.column_stack([Z[list(combo[j]), j] for j in range(J)])
        g = 0.0
        for j in range(J):
            for k in range(j + 1, J):
                g += float((Zp[:, j] * Zp[:, k]).sum())
        out.append(g)
    return np.array(out)


def case_exact_vs_mc():
    """★ 参照实现 = **全枚举**（不是"再写一遍模拟"）。首版拿 4000 vs 500 个样本的**最大值**
    互比，纯抽样噪声 ⇒ 无意义断言（`DEF-122-03`）。"""
    E = np.array([[0, 2, 0],
                  [1, 1, 1],
                  [2, 0, 2]], dtype=np.int64)
    exact = _enumerate_gamma_all(E)
    chk("T122-5a 全枚举组合数 == 216（= (3!)^3）", exact.size == 216, f"got {exact.size}")
    chk("T122-5b 精确零分布均值 == 0", abs(float(exact.mean())) < 1e-12, f"mean={exact.mean()}")

    # 完全对齐的例子：精确 p 应为 6/216 = 1/36
    Ea = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int64)
    ex_a = _enumerate_gamma_all(Ea)
    g_a = M.gamma_stat(Ea)
    p_exact = float((ex_a >= g_a - 1e-12).sum()) / ex_a.size
    chk("T122-5c 精确 p(完全对齐) == 1/36（6/216）", abs(p_exact - 1.0 / 36.0) < 1e-12,
        f"got {p_exact}")

    # 蒙特卡洛必须落在精确值附近（3σ）
    B = 40000
    mc = M.simulate_null_gamma(Ea, B=B, seed=20261231)
    p_mc = M.mech121.p_upper(g_a, mc)
    se = (p_exact * (1 - p_exact) / B) ** 0.5
    chk("T122-5d 蒙特卡洛 p 与精确 p 一致（3σ 内）", abs(p_mc - p_exact) <= 3 * se + 1e-12,
        f"mc={p_mc}, exact={p_exact}, 3se={3*se:.6f}")
    chk("T122-5e 蒙特卡洛均值与精确均值一致（3σ 内）",
        abs(float(mc.mean()) - float(ex_a.mean())) <= 3 * float(ex_a.std()) / (B ** 0.5) + 1e-12,
        f"mc_mean={mc.mean()}, exact_mean={ex_a.mean()}")

    # 反对齐的例子：精确 p 应很大
    Eb = np.array([[0, 2, 0], [1, 1, 1], [2, 0, 2]], dtype=np.int64)
    ex_b = _enumerate_gamma_all(Eb)
    p_exact_b = float((ex_b >= M.gamma_stat(Eb) - 1e-12).sum()) / ex_b.size
    chk("T122-5f 精确 p(反对齐) ≥ 0.5", p_exact_b >= 0.5, f"got {p_exact_b}")

    # 模拟不改动输入
    E_copy = E.copy()
    M.simulate_null_gamma(E, B=50, seed=11)
    chk("T122-5g 模拟不改动输入 E", np.array_equal(E, E_copy), "input mutated")


# ================================================================ rates 的 None 语义
def case_rates():
    E = np.array([[0], [3]], dtype=np.int64)
    d = np.array([2], dtype=np.int64)
    r = M.rates(E, d)
    chk("T122-6a rates 正常：E=[0,3], d=2 ⇒ [0.0, 1.5]", r == [[0.0], [1.5]], f"got {r}")
    r0 = M.rates(E, np.array([0], dtype=np.int64))
    chk("T122-6b d == 0 ⇒ None（不得当零逃逸）", r0 == [[None], [None]], f"got {r0}")


# ================================================================ 稠密直方图（DEF-122-01）
def case_hist():
    h = M.hist_of([0, 1, 2])
    chk("T122-7a hist_of 稠密 4 箱（含 0 计数）",
        h == {"0": 1, "1": 1, "2": 1, "3": 0}, f"got {h}")
    chk("T122-7b norm_hist 补齐缺箱", M.norm_hist({"0": 3}) == {"0": 3, "1": 0, "2": 0, "3": 0},
        f"got {M.norm_hist({'0': 3})}")
    chk("T122-7c norm_hist 对已稠密输入幂等",
        M.norm_hist({"0": 1, "1": 1, "2": 1, "3": 0}) == {"0": 1, "1": 1, "2": 1, "3": 0},
        "not idempotent")


# ================================================================ 重建的关键不变量
def case_recon():
    facts, tr, ev, tv, items, ns = M.build_all()
    _cnt = {}
    for f in facts:
        _cnt[f["val"]] = _cnt.get(f["val"], 0) + 1
    chk("T122-8a facts == 32 条且每值恰 2 次（16 个值）",
        len(facts) == 32 and set(_cnt.values()) == {2} and len(_cnt) == 16,
        f"条数={len(facts)}, 值的种数={len(_cnt)}, 计数集合={sorted(set(_cnt.values()))}")
    chk("T122-8b 训练侧与留出侧值集不交", len(tv) == 8 and not (tv & {facts[i]["val"] for i in ev}),
        "两侧值有交")
    chk("T122-8c 4 个 rng 的项数均为 48", all(len(v) == 48 for v in items.values()),
        f"{ {k: len(v) for k, v in items.items()} }")
    chk("T122-8d 每项 4 个互不相同的候选值",
        all(len(set(it["cand_val"])) == 4 for v in items.values() for it in v), "有重复候选")
    chk("T122-8e tgt_i 指向正确值",
        all(it["cand_val"][it["tgt_i"]] == facts[it["key_idx"]]["val"]
            for v in items.values() for it in v), "tgt_i 错位")
    # 主读物的两个已发表常数（重建护栏）
    chk("T122-8f 主读物 n_seen_hist == {0:2,1:23,2:20,3:3}",
        M.hist_of(ns[M.MAIN_RNG]) == {"0": 2, "1": 23, "2": 20, "3": 3},
        f"got {M.hist_of(ns[M.MAIN_RNG])}")
    # 可逃逸项数（卡片 §1-3 冻结值）
    d = [sum(1 for x in ns[r] if x >= 1) for r in M.C2_RNGS]
    chk("T122-8g c2 三个题集的可逃逸项数 == [45, 44, 45]", d == [45, 44, 45], f"got {d}")


def main():
    case_A()
    case_B()
    case_drow_regression()
    case_gamma_null()
    case_exact_vs_mc()
    case_rates()
    case_hist()
    case_recon()
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} PASS")
    if FAILED:
        print("FAILED:", FAILED)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())