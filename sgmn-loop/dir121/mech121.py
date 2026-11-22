"""`mech121.py` —— `DIR-121` 的**测量层**（纯函数 + numpy，无 torch）。

★ 抽出来的唯一原因（与 `mech120.py` 同）：卡片 §3 的**手算例必须能本地无 GPU 断言**。
★ 抽出来**不产生第二套逻辑**：逃逸定义逐字复用 `mech120.partition` / `mech120._n_seen_of`；
  `R1_task121.py` 与 `test_mech121.py` 都 import 这里。

★★ 本模块的两个统计量（卡片 §2-3，已冻结）：

  ① `D_col`（`J121-1`，测 **item 结构**）：条件在**行边缘** `R`（逐 seed 逃逸数固定），
     每 seed 从 `n_items` 题中**均匀无放回**抽 `R_s` 题 ⇒ 得到列边缘 `C` 的零分布，
     统计量 `D_col = Σ_i (C_i − c̄)² / c̄`，`c̄ = ΣR / n_items`。
     ★ 上尾（`≥`）才是备择方向：「超额集中」。

  ② `D_row`（`J121-2`，测 **seed 结构**）：条件在**列边缘** `C`（逐题逃逸数固定），
     每题把 `C_i` 次逃逸**均匀分配给 `C_i` 个不同 seed** ⇒ 得到行边缘 `R` 的零分布，
     统计量 `D_row = Σ_s (R_s − r̄)² / r̄`，`r̄ = ΣC / n_seeds`。

★ 两测**不独立**（同一张表的两个条件视图）⇒ 族内 `α=0.05` 各 `0.025`（`DEF-112-04` 口径）。
"""
import itertools

import numpy as np


# ----------------------------------------------------------------- 表的构造
def table_from_masks(esc_rows, has_idx):
    """`esc_rows`：逐 seed 的布尔列表；`has_idx`：保留的题号。返回 `int8` 矩阵 `(n_seed, n_has)`。"""
    return np.array([[1 if row[i] else 0 for i in has_idx] for row in esc_rows], dtype=np.int64)


def margins(X):
    """返回 `(R, C)`：行边缘与列边缘。"""
    X = np.asarray(X, dtype=np.int64)
    return X.sum(axis=1), X.sum(axis=0)


# ----------------------------------------------------------------- 统计量
def d_col(C):
    """`D_col = Σ_i (C_i − c̄)² / c̄`（`c̄ = ΣC / n_items`）。"""
    C = np.asarray(C, dtype=float)
    cbar = C.sum() / C.size
    return float(((C - cbar) ** 2).sum() / cbar)


def d_row(R):
    """`D_row = Σ_s (R_s − r̄)² / r̄`（`r̄ = ΣR / n_seeds`）。"""
    R = np.asarray(R, dtype=float)
    rbar = R.sum() / R.size
    return float(((R - rbar) ** 2).sum() / rbar)


# ----------------------------------------------------------------- 精确枚举（手算例 / 测试用）
def enumerate_null_A(R, n_items):
    """条件在行边缘 `R` 的**完全枚举**零分布（只用于极小表）。返回 `D_col` 的列表（含重复）。"""
    R = [int(x) for x in R]
    per = [list(itertools.combinations(range(n_items), r)) for r in R]
    out = []
    for combo in itertools.product(*per):
        C = np.zeros(n_items, dtype=np.int64)
        for idx in combo:
            for i in idx:
                C[i] += 1
        out.append(d_col(C))
    return out


def enumerate_null_B(C, n_seeds):
    """条件在列边缘 `C` 的**完全枚举**零分布（只用于极小表）。返回 `D_row` 的列表（含重复）。"""
    C = [int(x) for x in C]
    per = [list(itertools.combinations(range(n_seeds), c)) for c in C]
    out = []
    for combo in itertools.product(*per):
        R = np.zeros(n_seeds, dtype=np.int64)
        for idx in combo:
            for s in idx:
                R[s] += 1
        out.append(d_row(R))
    return out


# ----------------------------------------------------------------- 模拟（正式轮用）
def simulate_null_A(R, n_items, B=20000, seed=20270101):
    """条件在行边缘 `R`：每 seed 无放回抽 `R_s` 题。返回 `(B,)` 的 `D_col` 数组。"""
    R = np.asarray(R, dtype=np.int64)
    rng = np.random.default_rng(seed)
    Cn = np.zeros((B, n_items), dtype=np.int64)
    rows = np.arange(B)[:, None]
    for r in R:
        r = int(r)
        if r <= 0:
            continue
        if r >= n_items:          # 退化：全选
            Cn += 1
            continue
        # 逐行"无放回抽 r 个"：对随机数取前 r 小的下标（等价于均匀抽子集）
        idx = np.argpartition(rng.random((B, n_items)), r - 1, axis=1)[:, :r]
        Cn[rows, idx] += 1
    cbar = Cn.sum(axis=1, keepdims=True) / float(n_items)
    return (((Cn - cbar) ** 2).sum(axis=1) / cbar.ravel()).astype(float)


def simulate_null_B(C, n_seeds, B=20000, seed=20270202):
    """条件在列边缘 `C`：每题把 `C_i` 次逃逸分给 `C_i` 个不同 seed。返回 `(B,)` 的 `D_row` 数组。"""
    C = np.asarray(C, dtype=np.int64)
    rng = np.random.default_rng(seed)
    Rn = np.zeros((B, n_seeds), dtype=np.int64)
    rows = np.arange(B)[:, None]
    for c in C:
        c = int(c)
        if c <= 0:
            continue
        if c >= n_seeds:
            Rn += 1
            continue
        idx = np.argpartition(rng.random((B, n_seeds)), c - 1, axis=1)[:, :c]
        Rn[rows, idx] += 1
    rbar = Rn.sum(axis=1, keepdims=True) / float(n_seeds)
    return (((Rn - rbar) ** 2).sum(axis=1) / rbar.ravel()).astype(float)


# ----------------------------------------------------------------- p 值
def p_upper(obs, null):
    """单侧上尾 `p = P(null ≥ obs)`（含等号；`+1` 平滑不用 —— 用全枚举/大 `B`）。"""
    null = np.asarray(null, dtype=float)
    return float((null >= obs).sum() / null.size) if null.size else None


def summarize(null):
    """零分布的分位数摘要（供判决书引用，不进判据）。"""
    q = np.quantile(np.asarray(null, dtype=float), [0.5, 0.9, 0.95, 0.99])
    return {"n": int(np.asarray(null).size), "p50": float(q[0]), "p90": float(q[1]),
            "q95": float(q[2]), "q99": float(q[3])}