"""`precheck120.py` —— `DIR-120` 的**开卡前结构核**（只看"测量对象存不存在"，**不看判据读数**）。

★★ 纪律（本文件的存在理由）：`DIRECTION` 卡冻结阈值之后**不得**再改；因此开卡前若偷看
  判据数字，就等于用结果调阈值（`DIR-51` 立的规矩：`n=40` 探针只验可行性，不得反推阈值）。
  所以本探针**只打印**三类东西：
  ① 键存不存在（布尔）；
  ② **已发表**的常数（`D-150` 里已有的 18/48、`n_seen_hist`、值数）；
  ③ 重建题目是否与存档**逐值相同**（这是前提，不是判据）。
  ★ 它**不打印** 2×2 交叉表、不打印任何本卡判据统计量 —— 那些只在 `R1_task120.py` 跑。
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "dir119"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "dir118"))

import R1_task118 as P118          # noqa: E402
import R1_task90 as D90            # noqa: E402
import R1_task104 as D104          # noqa: E402

OUT119 = pathlib.Path(__file__).resolve().parent.parent / "dir119" / "out"
ARM = "COLOCATED"
J = "10000"          # ★ 键格式是 f"{m:g}"（已核：cells 的 acc_by_mult 键为 "1"/"10"/…/"10000"）


def main():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")

    cells = sorted((OUT119 / "cells").glob(f"{ARM}_600_*.json"))
    chk("cells(600) 9 个", len(cells) == 9, f"实测 {len(cells)}")

    d0 = json.loads(cells[0].read_text(encoding="utf-8"))
    chk("cell 含 prior_mech", "prior_mech" in d0 and d0["prior_mech"].get("finite") is True)
    chk(f"cell 含 mech_by_mult[{J}]", J in (d0.get("mech_by_mult") or {}))
    me = d0["mech_by_mult"][J]
    chk("mech 含 pick_argmin(48)", len(me.get("pick_argmin") or []) == 48)
    chk("mech 含 ok(48)", len(me.get("ok") or []) == 48)

    priors = [d0["prior_mech"]["acc_argmin"]]
    priors += [json.loads(f.read_text(encoding="utf-8"))["prior_mech"]["acc_argmin"]
               for f in cells[1:]]
    chk("prior_mech 全部 == 18/48（已发表常数）",
        all(abs(x - 18.0 / 48.0) < 1e-12 for x in priors),
        f"{sorted(set(round(x, 6) for x in priors))}")

    hists = [json.loads(f.read_text(encoding="utf-8"))["n_seen_hist"] for f in cells]
    chk("n_seen_hist 全部 == {0:2,1:23,2:20,3:3}（已发表常数）",
        all(h == {"0": 2, "1": 23, "2": 20, "3": 3} for h in hists),
        f"{hists[0]}")

    tok, _model = D90.model_and_tok()
    facts, _parts, _csf, IT = P118.ground(tok, ARM)
    items, meta, tv = IT["items_ev"], IT["meta_ev"], IT["tv"]
    tgt = D104.targets_of(items)
    n_seen = [sum(1 for j, v in enumerate(meta[i]["cand_val"]) if j != int(tgt[i]) and v in tv)
              for i in range(len(items))]
    got = {str(k): sum(1 for x in n_seen if x == k) for k in range(4)}
    chk("重建题目复现存档 n_seen_hist（逐值）", got == {"0": 2, "1": 23, "2": 20, "3": 3}, f"{got}")
    chk("重建 48 项 / 每项 4 候选", len(items) == 48 and all(len(m["cand_val"]) == 4 for m in meta))
    chk("tgt 合法", len(tgt) == 48 and all(0 <= int(t) < 4 for t in tgt))

    allvals = sorted({v for m in meta for v in m["cand_val"]})
    correct_vals = sorted({meta[i]["cand_val"][int(tgt[i])] for i in range(48)})
    chk("候选值全集 / 正确值集合可枚举", len(allvals) > 0 and len(correct_vals) > 0,
        f"候选值 {len(allvals)} 个、正确值 {len(correct_vals)} 个、tv {len(tv)} 个")
    chk("正确值全部 ∉ tv（构造前提）", all(v not in tv for v in correct_vals))

    print(f"\n★ 结构核：{'全部 PASS' if ok else '有 FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())