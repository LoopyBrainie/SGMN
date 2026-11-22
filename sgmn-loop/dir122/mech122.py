"""`mech122.py` —— `DIR-122` 的**测量层**（纯函数；`numpy` + 标准库，**不 import torch**）。

★★ 只有一处是新的：`gamma_stat` / `simulate_null_gamma`（`J122-2`）。
   `D_row` 及其置换零分布**逐字复用** `dir121/mech121.py`（`import mech121`），**不另写一套**。

== 为什么整个测量层不需要 tokenizer / GPU ==
`dir114.build_facts_pairs` / `partitions` / `make_items_with_vals` 的**值层**只由
`random.Random(9091 / 777 / rng_seed)` 驱动；`tok` 只用于 `prompt_ids` / `val_ids`。
逃逸判定只需要 `cand_val` / `tgt_i` / `tv` ⇒ 可**纯本地重建**。

== 独立单位 ==
`48 项 = 16 key × 3 draw`（`D90.N_DRAWS = 3`）⇒ 逐项**不独立**，key 才是单位。
本模块的判据只在**项级边缘**上做置换（给定边际后重排 seed 标签），因此**不**依赖逐项独立性。

== 口径 ==
逃逸 = 有见过候选（`n_seen ≥ 1`）却选了未见值；分母 = `#(n_seen ≥ 1)`（逐字沿用
`dir120.mech120._n_seen_of` / `D-150` / `D-151` / `D-152`）。
"""
import os
import random
import sys

import numpy as np

# ---- 逐字复用 DIR-121 的 D_row 与置换零分布（不重写）----
_DIR121 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dir121")
if _DIR121 not in sys.path:
    sys.path.insert(0, _DIR121)
import mech121  # noqa: E402

# ---- 常量：逐字取自 dir90/R1_task90.py 与 dir114/R1_task114.py（`precheck122.S0` 核对来源）----
VALUE_POOL = ("teal amber violet olive rust crimson azure beige coral ivory lemon "
              "peach plum rose ruby tan wheat panda yak copper silver golden iron "
              "bronze jade pearl quartz").split()
N_CAND, N_DRAWS = 4, 3
N_VAL, N_FACT = 16, 32
FACT_SEED, SHUFFLE_SEED = 9091, 777
MAIN_RNG = 555
C2_RNGS = (556, 557, 558)

# ---- 判据统计量的冻结参数（卡片 §2）----
B_PERM = 20000
BI_SEED_J1 = 20261123          # `J122-1` 三个题集共用
BI_SEED_J2 = 20261124          # `J122-2`
ALPHA_FAMILY = 0.05
ALPHA_J1_TOTAL = 0.025         # 族内 0.025 → 三个题集 Bonferroni 各 0.025/3
ALPHA_J1_EACH = ALPHA_J1_TOTAL / 3.0
ALPHA_J2 = 0.025


# ================================================================ 重建（值层）
def build_facts():
    """逐字复刻 `dir114.build_facts_pairs`（去 `tok`：只留 `key` / `val`）。"""
    assert len(VALUE_POOL) >= N_VAL, len(VALUE_POOL)
    vals16 = list(VALUE_POOL[:N_VAL])
    rng = random.Random(FACT_SEED)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    keys, seen = [], set()
    while len(keys) < N_FACT:
        k = "".join(rng.choice(alphabet) for _ in range(5))
        if k in seen:
            continue
        seen.add(k)
        keys.append(k)
    vals = vals16 * 2
    rng.shuffle(vals)                      # ★ 同一个 rng 续用（逐字复刻）
    return [{"key": k, "val": v} for k, v in zip(keys, vals)]


def partitions_colocated(facts):
    """逐字复刻 `dir114.partitions` 的 `COLOCATED` 支。"""
    by_val = {}
    for i, f in enumerate(facts):
        by_val.setdefault(f["val"], []).append(i)
    assert all(len(v) == 2 for v in by_val.values()), "每个值须恰 2 条"
    ordered = sorted(by_val)
    rng = random.Random(SHUFFLE_SEED)
    vs = ordered[:]
    rng.shuffle(vs)
    train_side, held_side = set(vs[:N_VAL // 2]), set(vs[N_VAL // 2:])
    tr = [i for v in ordered if v in train_side for i in by_val[v]]
    ev = [i for v in ordered if v in held_side for i in by_val[v]]
    return tr, ev


def make_items(facts, keys, rng_seed):
    """逐字复刻 `dir114.make_items_with_vals` 的**值层**。

    返回 `[{"key_idx", "cand_val", "tgt_i"}]`；`tgt_i` = 正确值在 `cand_val` 中的**唯一**位置
    （4 个候选值互不相同 ⇒ 位置唯一，与 `is_correct_val` 等价）。
    """
    rng = random.Random(rng_seed)
    out = []
    for ki in keys:
        correct = facts[ki]["val"]
        others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != correct]
        pool = sorted(set(others))
        for _ in range(N_DRAWS):
            distr = rng.sample(pool, N_CAND - 1)
            cands = [correct] + distr
            rng.shuffle(cands)
            assert cands.count(correct) == 1, "正确值须唯一"
            out.append({"key_idx": ki, "cand_val": cands, "tgt_i": cands.index(correct)})
    return out


def n_seen_vec(items, tv):
    """逐项 `n_seen`（干扰项里见过值的个数；**不含正确槽**）。"""
    return [sum(1 for j, v in enumerate(it["cand_val"]) if j != it["tgt_i"] and v in tv)
            for it in items]


def escape_mask(items, n_seen, picks, tv):
    """逐项逃逸布尔：`n_seen ≥ 1` 且**选中的值 ∉ tv**。"""
    m = []
    for it, ns, p in zip(items, n_seen, picks):
        pv = it["cand_val"][int(p)]
        m.append(bool(ns >= 1) and (pv not in tv))
    return m


def hist_of(vec, n_bins=N_CAND):
    """★ **稠密**直方图（`n_bins` 箱全写，含 0 计数）—— 存档 `n_seen_hist` 是稠密的，
    比较必须同形状（`DEF-122-01`：稀疏直方图曾造成**假 FAIL**）。

    ★★ `n_bins` **必须按被测量的取值范围选**：`n_seen` ∈ `0..3`（= `N_CAND`），
       而**列边缘** `C_i`（同一项被几个 seed 逃逸）∈ `0..n_seeds` = `0..9`
       ⇒ 对 `C` 必须传 `n_bins=n_seeds+1`（`DEF-122-04`：首版复用 4 箱 ⇒ `KeyError: '4'`）。
    """
    h = {str(k): 0 for k in range(n_bins)}
    for v in vec:
        h[str(v)] += 1
    return h


def norm_hist(h, n_bins=N_CAND):
    """把存档（可能缺零箱）直方图归一成稠密 `n_bins` 箱。"""
    return {str(k): int(h.get(str(k), 0)) for k in range(n_bins)}


def build_all():
    """一次性构造：`(facts, tr, ev, tv, {rng: items}, {rng: n_seen})`。"""
    facts = build_facts()
    tr, ev = partitions_colocated(facts)
    tv = {facts[i]["val"] for i in tr}
    items = {r: make_items(facts, ev, r) for r in (MAIN_RNG,) + C2_RNGS}
    ns = {r: n_seen_vec(items[r], tv) for r in items}
    return facts, tr, ev, tv, items, ns


# ================================================================ 统计量
def escape_matrix(cells, items_q, ns_q, tv):
    """`E[s][j]`（逐 seed、逐题集的逃逸数）与分母 `d[j]`。

    `cells`：按 seed 排序的存档 dict 列表；`j` 顺序 = `items_q` 的顺序。
    """
    rows = []
    for c in cells:
        rows.append([int(sum(escape_mask(items_q[j], ns_q[j], c["picks"][j], tv)))
                     for j in range(len(items_q))])
    E = np.array(rows, dtype=np.int64)
    d = np.array([int(sum(1 for v in ns_q[j] if v >= 1)) for j in range(len(items_q))],
                 dtype=np.int64)
    return E, d


def rates(E, d):
    """`r[s][j] = E/d`；`d == 0` ⇒ `None`（★ 不得把「无机会」当「零逃逸」）。"""
    E = np.asarray(E, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    out = []
    for s in range(E.shape[0]):
        out.append([(None if d[j] == 0 else float(E[s, j]) / float(d[j]))
                    for j in range(E.shape[1])])
    return out


def gamma_stat(E):
    """`Γ = Σ_{j<k} Σ_s Z_sj Z_sk`，`Z_sj = E_sj − ē_j`（列内中心化）。"""
    E = np.asarray(E, dtype=float)
    Z = E - E.mean(axis=0, keepdims=True)
    J = Z.shape[1]
    g = 0.0
    for j in range(J):
        for k in range(j + 1, J):
            g += float((Z[:, j] * Z[:, k]).sum())
    return g


def simulate_null_gamma(E, B=B_PERM, seed=BI_SEED_J2):
    """零分布：**在每个题集内独立置换 seed 标签**（各列边际保留，只摧毁跨题集对应）。"""
    E = np.asarray(E, dtype=np.int64)
    S, J = E.shape
    rng = np.random.default_rng(seed)
    Zp = []
    for j in range(J):
        col = E[:, j].astype(float)
        Z = col - col.mean()                       # 置换不改列均值 ⇒ 先中心化等价
        idx = np.argsort(rng.random((B, S)), axis=1)
        Zp.append(Z[idx])                          # (B, S)
    g = np.zeros(B, dtype=float)
    for j in range(J):
        for k in range(j + 1, J):
            g += (Zp[j] * Zp[k]).sum(axis=1)
    return g


def d_row_per_qset(E):
    """逐题集的 `D_row`（**直接调用 `mech121.d_row`**，不重写）。"""
    E = np.asarray(E, dtype=np.int64)
    return [mech121.d_row(E[:, j]) for j in range(E.shape[1])]


def null_d_row_per_qset(C, n_seeds, B=B_PERM, seed=BI_SEED_J1):
    """逐题集的 `D_row` 零分布（**直接调用 `mech121.simulate_null_B`**）。"""
    return [mech121.simulate_null_B(C[:, j], n_seeds, B=B, seed=seed)
            for j in range(C.shape[1])]