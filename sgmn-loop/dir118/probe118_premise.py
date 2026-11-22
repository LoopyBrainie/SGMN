"""`probe118_premise.py` —— `DIR-118` 前提核（**开卡之前**跑，只读 tokenizer / 不建模型、不用 GPU）。

问题：`P117-1` 原写「按 `it['val'] ∈ train_vals` 把 48 项分成两层」。开卡前必须先验：
① 这个字段存在吗？② 两层**非空**吗？③ 候选池的"值是否见过"结构在两臂里各是什么样？

★ 为什么必须先验：`DIR-116` 的 `P115-1` 就是被前提核推翻的（`tie_word_embeddings`），
   若照原文开卡会做出一张**空卡**。
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "dir114"))
import R1_task114 as P   # noqa: E402
from transformers import AutoTokenizer   # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = pathlib.Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
facts = P.build_facts_pairs(tok)
parts = P.partitions(facts)

print(f"事实总数 = {len(facts)}；值种数 = {len({f['val'] for f in facts})}")
print(f"item 的键集合 = {sorted(facts[0].keys())}")
print(f"★ `it['val']` 这个字段存在吗？ ⇒ {'val' in facts[0]}（事实层）")
print()

res = {}
for arm in ("SPLIT", "COLOCATED"):
    tr, ev = parts[arm]
    frac, seen, n, unseen_vals = P.class_seen_frac(facts, tr, ev)
    tv = {facts[i]["val"] for i in tr}
    items, meta = P.make_items_with_vals(tok, facts, ev, 555)

    # ① item 行里有没有 val 字段
    item_has_val = "val" in items[0]
    # ② 按「该 item 的正确值是否在训练值集」分层
    seen_items = sum(1 for it in items if facts[it["key_idx"]]["val"] in tv)
    unseen_items = len(items) - seen_items
    # ③ 候选池结构：每题 4 个候选里有几个是"训练里见过的值"
    cand_seen_hist = []
    for m in meta:
        cand_seen_hist.append(sum(1 for c in m["cand_val"] if c in tv))
    all_seen = sum(1 for h in cand_seen_hist if h == 4)
    none_seen = sum(1 for h in cand_seen_hist if h == 0)

    res[arm] = {
        "class_seen_frac_held": frac, "held_seen": seen, "n_held": n,
        "held_distinct_values_unseen": unseen_vals,
        "n_items": len(items), "item_has_val_field": item_has_val,
        "items_correct_val_seen": seen_items, "items_correct_val_unseen": unseen_items,
        "items_all4_candidates_seen": all_seen, "items_0_candidates_seen": none_seen,
        "cand_seen_count_hist": {k: cand_seen_hist.count(k) for k in sorted(set(cand_seen_hist))},
        "train_vals": len(tv),
    }
    print(f"=== {arm} ===")
    print(f"  tr={len(tr)} ev={len(ev)}  class_seen_frac_held={frac:.4f} ({seen}/{n})")
    print(f"  留出集里**未见值**的种数 = {unseen_vals}")
    print(f"  ★ item 行里有 'val' 字段？ {item_has_val}")
    print(f"  ★ 分层：正确值**见过** {seen_items} 项 / **未见过** {unseen_items} 项  (共 {len(items)})")
    print(f"  ★ 候选池：4 个候选**全是**见过值 {all_seen} 项；**全非**见过值 {none_seen} 项")
    print(f"     每题见过值个数分布 = {res[arm]['cand_seen_count_hist']}")
    print()

(OUT / "premise118.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"✓ 落盘 {OUT / 'premise118.json'}")

# ---------------------------------------------------------------- 判定
sp, co = res["SPLIT"], res["COLOCATED"]
print("\n================ 前提核判定 ================")
ok1 = (sp["items_correct_val_unseen"] == 0) and (co["items_correct_val_seen"] == 0)
print(f"① 两臂恰好是互补的两层（SPLIT 全见过 / COLOCATED 全未见过）⇒ {ok1}")
print(f"   SPLIT: seen={sp['items_correct_val_seen']} unseen={sp['items_correct_val_unseen']}")
print(f"   COLOC : seen={co['items_correct_val_seen']} unseen={co['items_correct_val_unseen']}")
ok2 = not sp["item_has_val_field"]
print(f"② `it['val']` 不存在 ⇒ {ok2}（必须用 `facts[it['key_idx']]['val']`）")
ok3 = sp["items_all4_candidates_seen"] == sp["n_items"]
print(f"③ SPLIT 的 4 个候选**全部**是见过值 ⇒ {ok3}（⇒ 该臂无法区分"
      f"「偏好正确的见过值」与「偏好任意见过值」）")
print(f"\n⇒ `P117-1` 原文的 item 内分层在 SPLIT 上是 (48, 0) —— {'**退化**' if ok1 else '非退化'}")
