#!/usr/bin/env python
"""`P118-2` 的**可行性核**：`SPLIT` 到底能不能改用「混入未见干扰值」这把尺子？两把尺子的精确零期望各是多少？

★ 纪律：**一律读装置自己的字段**，不重实现（`COR-114-01`）。
   - `ground(tok, arm)` → `(facts, parts, csf, IT)`，其中 `IT["tv"]` = 训练值集、`IT["ev"]` = 评测键、
     `csf[arm]["unseen_vals"]` = **装置自己算的**"训练里没出现过的值"。
   - `P.make_items_with_vals` 用 `sorted(set(others))` 建 pool ⇒ pool 与臂无关，臂只决定 `tr`/`ev`。
★ 不加载模型、不训练、不产生任何判据判决；只做**结构与算术**。
"""
import sys
import json
import pathlib
from collections import Counter

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "dir117"))

import R1_task118 as R18      # noqa: E402

OUT = _HERE / "out"
ARMS = R18.ARMS
N_CAND = None


def main():
    tok, model = R18.model_and_tok()
    _, parts, csf, _ = R18.ground(tok, "SPLIT")
    facts = R18.P.build_facts_pairs(tok)
    n_facts = len(facts)
    vals = [f["val"] for f in facts]
    mv = Counter(vals)
    pool_all = sorted(mv)
    n_vals = len(pool_all)
    e_rand = sum(m * m for m in mv.values()) / n_facts
    global N_CAND
    N_CAND = R18.P.D90.N_CAND

    rep = {"n_facts": n_facts, "n_distinct_vals": n_vals,
           "mult_hist": {int(k): v for k, v in sorted(Counter(mv.values()).items())},
           "m_v_hist": {str(k): int(v) for k, v in sorted(mv.items())},
           "E_random_Nhit": e_rand,
           "E_random_Nhit_frac": e_rand / n_facts,
           "const_argmax_ceiling": max(mv.values()),
           "N_CAND": N_CAND, "arms": {}}

    print(f"facts={n_facts}  distinct_vals={n_vals}  m_v 分布={rep['mult_hist']}")
    print(f"E_random(N_hit^v) = Σm_v²/N = {sum(m*m for m in mv.values())}/{n_facts} = {e_rand}")
    print(f"常数 argmax 的天花板 = max m_v = {max(mv.values())}")
    print()

    for arm in ARMS:
        tr, ev = parts[arm]
        tv = {facts[i]["val"] for i in tr}
        n_unseen_dev = int(csf[arm]["unseen_vals"])   # ★ 装置返回的是**计数**，不是集合
        unseen = {v for v in pool_all if v not in tv}   # 集合由 tv/pool 派生（可核：|unseen| 须 == n_unseen_dev）
        # pool（装置口径）：某 item 的 pool = 其它 facts 的值去掉正确值
        avail_unseen_pool = pool_all_unseen = [v for v in pool_all if v not in tv]
        rep["arms"][arm] = {
            "n_tr": len(tr), "n_ev": len(ev), "n_train_vals": len(tv),
            "n_unseen_vals": len(unseen),
            "n_unseen_dev_field": n_unseen_dev,
            "unseen_vals": sorted(unseen),
            "pool_size_for_item": None,
        }
        # 逐 item 检查：若要求「全部 N_CAND-1 个干扰项都是未见值」，能做多少 item？
        ok_all_unseen = 0
        ok_ge1_unseen = 0
        pool_sizes = []
        for ki in ev:
            correct = facts[ki]["val"]
            others = [g["val"] for j, g in enumerate(facts) if j != ki and g["val"] != correct]
            pool = sorted(set(others))
            pool_sizes.append(len(pool))
            pu = [v for v in pool if v not in tv]
            if len(pu) >= N_CAND - 1:
                ok_all_unseen += 1
            if len(pu) >= 1:
                ok_ge1_unseen += 1
        rep["arms"][arm].update({
            "pool_size_min": min(pool_sizes), "pool_size_max": max(pool_sizes),
            "n_ev_items": len(ev),
            "items_can_have_all_unseen_distractors": ok_all_unseen,
            "items_can_have_ge1_unseen_distractor": ok_ge1_unseen,
        })
        print(f"-- {arm} --")
        assert len(unseen) == n_unseen_dev, (arm, len(unseen), n_unseen_dev)
        print(f"   训练事实 {len(tr)} / 评测事实 {len(ev)}；训练值 {len(tv)}/{n_vals} 个不同值"
              f" ⇒ **未见值 {len(unseen)} 个** {sorted(unseen)}")
        print(f"   每题 pool 大小 {min(pool_sizes)}..{max(pool_sizes)}")
        print(f"   要求全部 {N_CAND-1} 个干扰项都是未见值：可做 **{ok_all_unseen}/{len(ev)}** item")
        print(f"   要求至少 1 个干扰项是未见值：可做 **{ok_ge1_unseen}/{len(ev)}** item")
        print()

    # 功效：fix ① 的零假设仍是恰好 1/4（4 选 1，恰一个正确）；变的是**可分辨性**
    n_item = 48
    p0, p1 = 0.25, 0.40
    import math
    se = math.sqrt(p1 * (1 - p1) / n_item)
    rep["power"] = {"n_item": n_item, "p0": p0, "p1": p1,
                    "se_at_p1": se, "z_vs_p0": (p1 - p0) / se,
                    "n_for_3se_abs": None}
    # 3se 判别 0.25 vs 0.40 所需 n（单臂，绝对差）
    n_need = None
    for n in range(8, 4000):
        s = math.sqrt(p1 * (1 - p1) / n)
        if (p1 - p0) / s >= 3.0:
            n_need = n
            break
    rep["power"]["n_for_3se_abs"] = n_need
    print(f"功效：n={n_item} 下 0.25→0.40 的 z = {rep['power']['z_vs_p0']:.2f}；"
          f"达 3se 需 n={n_need}")

    (OUT / "premise_p118_2.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n⇒ 落盘 dir118/out/premise_p118_2.json")


if __name__ == "__main__":
    main()