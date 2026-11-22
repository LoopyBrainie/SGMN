#!/usr/bin/env python
"""`P118-1` 诊断 B —— **前提核**（不跑模型前向，只建词表与 item 列表）。

要证的三个前提（缺一，下面的算术推论就不成立）：
  P1. 候选顺序经 `rng.shuffle` ⇒ **正确候选的位置均匀分布**（否则"固定位置picker=25%"不成立）。
  P2. `COLOCATED` 的留出 item：正确答案是**未见值**，而 3 个干扰项**都是见过值**。
  P3. `SPLIT` 的留出 item：4 个候选**全是**见过值（`DEF-118-01`）。

推论（若 P1–P3 成立）：`COLOCATED` 训练后 5.3% **低于**随机 25%，
  ⇒ 不可能是"固定位置/构造工件"（那只能给 25%）
  ⇒ 模型的**选择依赖候选内容**，且**系统性回避正确项**
  ⇒ 与"训练把读出拉向**训练见过的值**"一致（θ ≈ 0.79，见下）。
"""
import json
import pathlib
import sys
import collections

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dir90"))
sys.path.insert(0, str(HERE.parent / "dir114"))

import R1_task90 as D90            # noqa: E402
import R1_task114 as P             # noqa: E402

ARMS = ("SPLIT", "COLOCATED")
N_ITEM = 48


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(D90.MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    facts = P.build_facts_pairs(tok)
    parts = P.partitions(facts)
    out = {}
    for arm in ARMS:
        tr, ev = parts[arm]
        items, meta = P.make_items_with_vals(tok, facts, ev, 555)
        train_vals = {facts[i]["val"] for i in tr}

        pos = collections.Counter()
        n_seen_cand = collections.Counter()
        correct_seen = 0
        for it, m in zip(items, meta):
            # ★ `DEF-118-16`：此处原写法 `for k, ok in enumerate(it["cand"])` 是**错的** ——
            #   `it["cand"]` 的元素是 `(ids, flag)` **元组**，`ok` 拿到的是元组本身（恒真）
            #   ⇒ `next(...)` 恒返回 0 ⇒ 伪造出"正确答案恒在 index 0"的假发现。必须解包。
            j = next(k for k, (_, flag) in enumerate(it["cand"]) if flag)
            assert sum(1 for _, fl in it["cand"] if fl) == 1, "正确候选必须恰好一个"
            pos[j] += 1
            cv = m["cand_val"]
            assert cv[j] == facts[it["key_idx"]]["val"], "位置 j 的值必须等于 fact 的正确值"
            seen_flags = [v in train_vals for v in cv]
            n_seen_cand[sum(seen_flags)] += 1
            if seen_flags[j]:
                correct_seen += 1

        n = len(items)
        print(f"\n{'='*70}\n### {arm}   留出 item 数 = {n}\n{'='*70}")
        print(f"P1 正确候选位置直方图：{dict(sorted(pos.items()))}")
        nz = [pos.get(j, 0) for j in range(4)]
        print(f"   ⇒ 均匀？各位置计数 {nz}；最大单点占比 = {max(nz)/n:.3f}"
              f"（固定位置 picker 的上限准确率 = {max(nz)}/{n} = {max(nz)/n:.3f}）")
        print(f"P2/P3 『每题候选里含几个见过值』直方图：{dict(sorted(n_seen_cand.items()))}")
        print(f"   ⇒ 正确答案本身是见过值的题数 = {correct_seen}/{n}"
              f"（SPLIT 应 = {n}；COLOCATED 应 = 0）")
        # ★ 用**装置自己的**函数复核"目标值是否见过"（`COR-114-01`：不许重实现）
        fr, seen, nh, nun = P.class_seen_frac(facts, tr, ev)
        print(f"   [装置口径] class_seen_frac: frac={fr:.4f} seen={seen}/{nh} "
              f"unseen_distinct={nun}（COLOCATED 应 0/48；SPLIT 应 48/48）")

        # θ：若模型在"偏好见过值"下完全回避正确项，则 acc = 0.25*(1-theta)
        acc = 2.56 / 48 if arm == "COLOCATED" else 16.67 / 48
        print(f"   观测 acc = {acc:.4f} ⇒ 若 acc = 0.25*(1-theta)，则 theta = {1-acc/0.25:.3f}")

        out[arm] = {"n": n, "pos_hist": dict(sorted(pos.items())),
                    "n_seen_cand_hist": {str(k): v for k, v in sorted(n_seen_cand.items())},
                    "correct_is_seen": correct_seen,
                    "fixed_pos_max_acc": max(nz) / n}

    (HERE / "out" / "premise_p118_1.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n✓ 已写 out/premise_p118_1.json")

    # ---- 结论性断言（前提核的标准形态：不满足就大声报）
    print(f"\n{'='*70}\n前提核结论\n{'='*70}")
    ok1 = all(abs(v / 48 - 0.25) < 0.16 for v in out["COLOCATED"]["pos_hist"].values())
    print(f"P1 位置大致均匀（各位置占比与 0.25 差 < 0.16）：{'PASS' if ok1 else 'FAIL'}")
    print(f"P2 COLOCATED 正确答案是见过值的题数 == 0："
          f"{'PASS' if out['COLOCATED']['correct_is_seen'] == 0 else 'FAIL'}")
    print(f"P3 SPLIT 正确答案是见过值的题数 == 48："
          f"{'PASS' if out['SPLIT']['correct_is_seen'] == 48 else 'FAIL'}")


if __name__ == "__main__":
    main()