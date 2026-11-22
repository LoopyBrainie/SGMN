"""`mech120.py` —— `DIR-120` 的**测量层**（纯函数，只依赖 numpy）。

★ 抽出来的唯一原因：手算例（卡片 §3）必须能在**本地无 GPU/无 torch** 处自测
  （round standard 要求「测量对象须对一个具体手算例验证」；`handoff` 纪律要求
  「手算例必须写成可断言的自测 —— 手算例本身就是被测对象」）。
★ 抽出来**不产生第二套逻辑**：`R1_task120.py` import 这里，`test_mech120.py` 也 import 这里。

★★ 本模块只有两个概念，务必分清（`DIR-120` §0-2 / §0-3）：

  ① **逃逸（escape）**：某项**有**见过候选（`n_seen ≥ 1`）却**选了未见值**。
     `D-150` §7-3 用的分母是 432，但那 432 里有 **18 项根本没有见过候选**（`n_seen=0`）
     ⇒ 那些项**不可能**选出见过值。正确分母是 `#(n_seen ≥ 1)` = **414**。
     ⇒ 「见到就一定要选」的硬规则要求 `r = 0`；实测 `r = 0.1014` ⇒ `M1` 是**加成**，不是硬规则。

  ② **残差 ≡ 逃逸且正确**：正确值**必为未见值**（构造前提）⇒ 选见过值**必错** ⇒
     在 `n_seen ≥ 1` 的项上，"答对"**等价于**"逃逸且选中那个未见正确值"。
     故恒等式 `E_total == E_n0 + E_esc_ok`（`A120-5` 在真实数据上断言它）。
     ★ 这个恒等是**计数**的、不是模型近似 —— 这是本卡相对 `D-150` §3-3 "1.04% 期望上限" 的实质推进。

★ 口径逐字沿用 `mech119.score_and_mech`：判定用 `argmin`/`pick_argmin`；
  `n_seen` **只数干扰项**（`j != 正确槽`），因为正确值本就不在 `tv` 里。
"""
import numpy as np

N_CAND = 4


def _n_seen_of(cand_val, tgt_i, tv):
    """干扰项里"见过值"的个数（★ 不含正确槽 —— 正确值本就不 ∈ tv，构造前提）。"""
    return sum(1 for j, v in enumerate(cand_val) if j != tgt_i and v in tv)


def partition(pick, meta, tv, tgt):
    """把一次读数（432 项或 48 项）切成 `DIR-120` 需要的计数。

    返回 dict（**全为计数/比例，无浮点判据**）：
      - `escape` / `r`           : 逃逸项数与逃逸率（分母 = `n_has_seen`）
      - `E_total` / `E_n0` / `E_esc_ok` : 总正确数、`n_seen=0` 项上的正确数、逃逸且正确数
      - `n_has_seen`             : `n_seen ≥ 1` 的项数（= 414）
      - `seen_vals` / `unseen_vals` : 见侧/未见侧的**被选中值列表**（用于值级集中度）
      - `escape_mask` / `ok_mask`   : 逐项布尔（供 `q0` 与 `n11` 使用）
    """
    pick = [int(x) for x in pick]
    tgt = [int(x) for x in tgt]
    n = len(pick)
    assert len(tgt) == n and len(meta) == n, (len(pick), len(tgt), len(meta))

    escape_mask, ok_mask, has_seen_mask = [], [], []
    seen_vals, unseen_vals, n_seen_list, picked_seen_list = [], [], [], []
    E_esc_ok = E_n0 = 0
    for i in range(n):
        cv = meta[i]["cand_val"]
        ns = _n_seen_of(cv, tgt[i], tv)
        has = ns >= 1
        pv = cv[pick[i]]
        p_seen = pv in tv
        ok = pick[i] == tgt[i]
        n_seen_list.append(ns)
        picked_seen_list.append(p_seen)
        has_seen_mask.append(has)
        escape_mask.append(has and not p_seen)
        ok_mask.append(ok)
        (seen_vals if p_seen else unseen_vals).append(pv)
        if ok:
            if has:
                E_esc_ok += 1
            else:
                E_n0 += 1

    escape = int(sum(escape_mask))
    n_has_seen = int(sum(has_seen_mask))
    return {
        "n": n,
        "escape": escape,
        "n_has_seen": n_has_seen,
        "r": (escape / float(n_has_seen)) if n_has_seen else None,
        "E_total": int(sum(ok_mask)),
        "E_n0": E_n0,
        "E_esc_ok": E_esc_ok,
        "seen_vals": seen_vals,
        "unseen_vals": unseen_vals,
        "escape_mask": escape_mask,
        "ok_mask": ok_mask,
        "has_seen_mask": has_seen_mask,
        "n_seen_list": n_seen_list,
        "picked_seen_list": picked_seen_list,
    }


def _top3_from_counts(counts, total):
    """`top3` 的**唯一**实现（`top3_share` 与零假设都调它 ⇒ 不产生第二套逻辑）。"""
    if not total:
        return None
    return sum(sorted(counts.values())[-3:]) / float(total)


def top3_share(vals):
    """`C3` 口径的集中度：最高频 3 个值合计 / 总次数。空列表返回 `None`。

    ★ 与 `D-150` §7-3 的口径**逐字一致**（那边报 0.4560），便于直接对读。
    """
    if not vals:
        return None
    import collections
    return _top3_from_counts(collections.Counter(vals), len(vals))


def q0_of(part, meta, tv, tgt):
    """逃逸子集上的**候选内均匀**参照 `q0 = mean(1/|U_i|)`，`|U_i| = 4 − n_seen_i`。

    ★ `|U_i|` 是该项里"未见候选"的个数（含正确项）：项内均匀乱选时命中正确项的概率 = `1/|U_i|`。
      只在**逃逸项**上取均值 ⇒ 与 `q` 同分母，可直接比较。
    """
    vals = []
    for i, esc in enumerate(part["escape_mask"]):
        if not esc:
            continue
        cv = meta[i]["cand_val"]
        ns = _n_seen_of(cv, int(tgt[i]), tv)
        vals.append(1.0 / float(N_CAND - ns))
    return (sum(vals) / len(vals)) if vals else None


def q_of(part):
    """`q = E_esc_ok / escape`（逃逸项里最终答对的比例）。"""
    return (part["E_esc_ok"] / float(part["escape"])) if part["escape"] else None


def n11_of(pick, prior_pick, tgt):
    """`n11 = #(正确@本次 ∧ 正确@prior)` —— `W⁰` 残留**最多**能解释的项数（计数上界）。"""
    return int(sum(1 for a, b, t in zip(pick, prior_pick, tgt)
                   if int(a) == int(t) and int(b) == int(t)))


# ---------------------------------------------------------------- 零假设（状态保持置换）
def null_top3(pick, meta, tv, arm, B=20000, seed=20261122):
    """**状态保持置换零假设**下 `arm ∈ {"seen","unseen"}` 一侧的 top-3 占比分布。

    ★ 定义（卡片 §2 冻结）：对每一项，**条件在"该项选了哪种状态"**上，
      在该项**与所选同状态**的候选里**均匀重抽**。
      ⇒ 保持：每项选了 seen/unseen 这件事、每个值在该项中是否可用、各侧总次数；
      ⇒ 摧毁：**同状态内按值身份**的偏好（这正是 `C3` 次级集中要检验的东西）。
    ★ 为什么不跨状态置换：那会把 `M1`（见过/未见这一层）本身也抹掉 ⇒ 零假设变成"没有 M1"，
      于是 `C3` 的次级集中**永远测不出来**。必须条件在状态上。
    """
    assert arm in ("seen", "unseen")
    want_seen = arm == "seen"
    idx = [i for i in range(len(pick)) if (meta[i]["cand_val"][int(pick[i])] in tv) == want_seen]
    if not idx:
        return np.array([]), None
    # 每项的可选槽位：与观测选中项**同状态**的槽位
    pools = []
    for i in idx:
        cv = meta[i]["cand_val"]
        pools.append([j for j in range(len(cv)) if (cv[j] in tv) == want_seen])
    rng = np.random.default_rng(seed)
    vals_per_item = [meta[i]["cand_val"] for i in idx]
    total = len(idx)
    out = np.empty(B, dtype=np.float64)
    for b in range(B):
        counts = {}
        for vi, pool in zip(vals_per_item, pools):
            v = vi[pool[int(rng.integers(len(pool)))]]
            counts[v] = counts.get(v, 0) + 1
        out[b] = _top3_from_counts(counts, total)
    obs = top3_share([meta[i]["cand_val"][int(pick[i])] for i in idx])
    return out, obs


def null_summary(null_arr, obs):
    """零假设摘要：`q99` 与 `mean`（`α=0.01` 单侧）。"""
    if null_arr is None or len(null_arr) == 0:
        return {"q99": None, "mean": None, "obs": obs}
    return {"q99": float(np.quantile(null_arr, 0.99)),
            "mean": float(null_arr.mean()), "obs": obs}