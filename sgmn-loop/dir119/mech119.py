"""`mech119.py` —— `DIR-119` 的**测量层**（纯函数，只依赖 numpy）。

★ 从 `R1_task119.py` 抽出来的唯一原因：**手算例要能在本地无 GPU/无 torch 处自测**
  （round standard 要求「测量对象须对一个具体手算例验证」；`handoff` 纪律要求
  「手算例必须写成可断言的自测 —— 手算例本身就是被测对象」）。
★ 抽出来**不产生第二套逻辑**：`R1_task119.py` **import 这里**，`test_mech119.py` 也 import 这里。
★ 口径逐字对齐 `dir114._margin_and_rank`：
  - 判定用 `argmin`（候选 NLL 越低越好）
  - `rank` = 正确项在 `argsort(row)` 中的位置 +1 ⇒ rank 1 = 最被偏好、rank 4 = 最不被偏好
  - `n_seen_distr` 只数 `j != 正确下标` 的见过值 ⇒ `err_null = n_seen_distr / (N_CAND−1)`
    （**条件在"错了"上** ⇒ 只在 3 个干扰项里挑，故除 3 不除 4）
"""
import numpy as np

N_CAND = 4
#: `err_in_train_null` 的分母 = 干扰项个数（条件在"错了"上 ⇒ 只在 N_CAND−1 个干扰里挑）
ND = N_CAND - 1


def score_and_mech(S, tgt, meta, tv):
    """把一次 `score_matrix` 的 `S`（48×4）**全部**榨干。纯函数、无模型依赖 ⇒ 可喂合成数据自测。

    ★ 口径逐字对齐 `dir114._margin_and_rank`：
      - 判定用 `argmin`（NLL 越低越好）
      - `rank` = 正确项在 `argsort(row)` 中的位置 +1 ⇒ rank 1 = 最被偏好、rank 4 = 最不被偏好
      - `n_seen_distr` 只数 `j != 正确下标` 的见过值 ⇒ `err_null = n_seen_distr / (N_CAND−1)`
    """
    S = np.asarray(S, dtype=np.float64)
    n = int(S.shape[0])
    finite = bool(np.isfinite(S).all())
    if not finite:
        return {"finite": False, "n": n}
    tgt = np.asarray(tgt, dtype=np.int64)
    pick_min = np.argmin(S, axis=1)
    pick_max = np.argmax(S, axis=1)
    okv = (pick_min == tgt)
    order = np.argsort(S, axis=1, kind="stable")
    ranks = np.array([int(np.where(order[i] == tgt[i])[0][0]) + 1 for i in range(n)],
                     dtype=np.int64)
    rank_hist = [int((ranks == k).sum()) for k in (1, 2, 3, 4)]
    margins = []
    for i in range(n):
        wrong = [S[i, j] for j in range(S.shape[1]) if j != int(tgt[i])]
        margins.append(min(wrong) - S[i, int(tgt[i])])

    err_idx = [i for i in range(n) if not okv[i]]
    n_err = len(err_idx)
    err_in, err_null = [], []
    n_seen_all = []
    for i in range(n):
        cv = meta[i]["cand_val"]
        c = int(tgt[i])
        n_seen = sum(1 for j in range(len(cv)) if j != c and cv[j] in tv)
        n_seen_all.append(n_seen)
        if not okv[i]:
            err_in.append(1 if cv[int(pick_min[i])] in tv else 0)
            err_null.append(n_seen / float(ND))
    return {
        "finite": True, "n": n,
        "acc_argmin": float(okv.mean()),
        "acc_argmax": float((pick_max == tgt).mean()),
        "ok": [int(x) for x in okv],
        "rank_hist": rank_hist,
        "margin_mean": float(np.mean(margins)),
        "pick_argmin": [int(x) for x in pick_min],
        "pick_argmax": [int(x) for x in pick_max],
        "n_err": n_err,
        "err_in_train_frac": (float(np.mean(err_in)) if n_err else None),
        "err_in_train_null": (float(np.mean(err_null)) if n_err else None),
        "n_seen_distr_mean": float(np.mean(n_seen_all)),
        "n_seen_hist": {str(k): sum(1 for v in n_seen_all if v == k) for k in range(ND + 1)},
    }
