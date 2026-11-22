"""`diag_c3.py` —— `DIR-119` 的 `C3` 词面偏置诊断（report-only）。

★ 只需 tokenizer（**不加载模型、不重训**）：`make_items_with_vals` 的题目与候选是**确定性**的
  （`rng_seed=555`、同 `ev` keys、同 `N_DRAWS`），故可从训练存档的 `pick_argmin` 反查出
  **被选中的是哪个 token / 哪个值**，再检验"选中是否集中在少数 token 上"。
★ 关心的问题：若"低于随机"只是**词面偏置**（某几个 token 天生 NLL 低），则选中会集中在**少数 token**
  上、且与"是否见过"**无关**。若相反（选中随"见过"走、token 分散），则 M1 站得住。
"""
import json
import pathlib
import sys
import collections

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "dir118"))

import R1_task118 as P118          # noqa: E402
import R1_task90 as D90            # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "out"
ARM = "COLOCATED"
J = "10000"


def main():
    tok, _model = D90.model_and_tok()
    facts, parts, csf, IT = P118.ground(tok, ARM)
    items, meta, tv = IT["items_ev"], IT["meta_ev"], IT["tv"]
    tokid = [[int(c[0][0]) for c in it["cand"]] for it in items]

    all_picks = collections.Counter()
    seen_pick = collections.Counter()
    tok_pick = collections.Counter()
    n_pick = 0
    seen_all = 0
    for f in sorted((OUT / "cells").glob(f"COLOCATED_600_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        me = (d.get("mech_by_mult") or {}).get(J)
        if not me:
            continue
        picks = me["pick_argmin"]
        for i, k in enumerate(picks):
            cv = meta[i]["cand_val"]
            v = cv[k]
            all_picks[v] += 1
            tok_pick[tokid[i][k]] += 1
            n_pick += 1
            if v in tv:
                seen_pick[v] += 1
                seen_all += 1

    n_val = len(all_picks)
    top_val = all_picks.most_common(3)
    top_tok = tok_pick.most_common(3)
    print(f"样本：{n_pick} 次选择（9 seed × 48 项，dose=600，mult={J}）")
    print(f"被选中过的不同值数 = {n_val}；不同 token 数 = {len(tok_pick)}")
    print(f"选中项的『见过』占比 = {seen_all/n_pick:.4f}（含正确项；"
          f"★ 对照：错误项的 GAP 见 VERDICT §3）")
    print(f"最高频值（值:次数）= {top_val}")
    print(f"最高频 token（token:次数）= {top_tok}")
    tot = sum(all_picks.values())
    print(f"前 3 个值合计占比 = {sum(c for _v, c in top_val)/tot:.4f}")
    print(f"前 3 个 token 合计占比 = {sum(c for _t, c in top_tok)/tot:.4f}")
    print(f"★ 均匀假设下，16 个值里前 3 名合计 ≈ {3/16:.4f}；"
          f"若选中高度集中于少数 token ⇒ 词面偏置是竞争解释")
    print(f"『见过值』被选中次数合计 = {sum(seen_pick.values())} / {n_pick}"
          f"；『见过值』在候选池中的出现次数见 n_seen_hist={IT and None}")
    print(f"见过值 top3 = {seen_pick.most_common(3)}")


if __name__ == "__main__":
    main()
