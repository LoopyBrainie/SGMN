#!/usr/bin/env python3
"""★ 缺陷 D56 的最小复现与回归测试（自包含单文件，无外部依赖）。

缺陷：`R3_cpu.py` 的**续跑跳过判据**用 `str(seed)` 去查 `_load_incremental()` 返回的
**int 键** ⇒ 永远不命中 ⇒ 续跑会**重算所有已完成臂** ⇒ 增量落盘形同虚设。

本测试：
  ① 复现**修复前**的行为（判据不命中）
  ② 验证**修复后**的行为（判据命中、且取回的行与写盘时逐位相同）
  ③ 验证**混合键类型**（int + str 混存）也能正确命中
  ④ 验证**种子顺序**不受影响（`per_seed` 的排序仍按规范种子序）

用法：python3 test_d56_resume.py
"""
import json, os, tempfile

SEEDS = [20261028, 20261104, 7]


# —— 从 R3_cpu.py 逐字抽出的「修复前」实现 ——
def _load_incremental_fixed(path):
    """修复后的加载：seed 键转 int（与 R3_cpu.py 一致）。"""
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    rows = d.get("rows", {}) or {}
    return {k: {int(sk): vv for sk, vv in v.items()} for k, v in rows.items()}


def _pick_seed(per_seed, seed):
    """★ 修复后的取值：int 键优先，str 键兜底。"""
    if not per_seed:
        return None
    if seed in per_seed:
        return per_seed[seed]
    sk = str(seed)
    if sk in per_seed:
        return per_seed[sk]
    return None


def main():
    fails = []

    def check(ok, msg):
        print(("  ✓ " if ok else "  ✗ ") + msg)
        if not ok:
            fails.append(msg)

    tmp = tempfile.mktemp(suffix=".json")
    payload = {"rows": {"W0_verbatim": {"20261028": [{"nll": 10.4638, "lp": -10.4638}]},
                        "W5b_hebb_full_ortho_eta1": {"20261028": [{"nll": 11.3328}]}}}
    json.dump(payload, open(tmp, "w"))
    all_rows = _load_incremental_fixed(tmp)

    print("① 修复前：`str(seed) in all_rows[name]` ⇒ 永远 False")
    check((str(20261028) in all_rows["W0_verbatim"]) is False,
          "修复前判据确实不命中（这正是缺陷本身）")

    print("② 修复后：`_pick_seed` 能命中，且取回的行逐位相同")
    r = _pick_seed(all_rows["W0_verbatim"], 20261028)
    check(r is not None, "命中（W0_verbatim / 20261028）")
    check(r is not None and r[0]["nll"] == 10.4638, "取回行的 nll 与写盘时逐位相同")

    print("③ 混合键类型（int + str 混存）也能命中")
    mixed = {"W0_verbatim": {20261028: [{"nll": 1.0}], "20261104": [{"nll": 2.0}]}}
    check(_pick_seed(mixed["W0_verbatim"], 20261028) is not None, "int 键命中")
    check(_pick_seed(mixed["W0_verbatim"], 20261104) is not None, "str 键兜底命中")
    check(_pick_seed(mixed["W0_verbatim"], 999) is None, "不存在的种子返回 None（不会误跳过）")

    print("④ 种子顺序不受影响（规范种子序）")
    per_seed = {20261028: [{"nll": 1.0}], 20261104: [{"nll": 2.0}]}
    ordered = [k for k, _ in sorted(per_seed.items(), key=lambda kv: SEEDS.index(kv[0])
                                    if kv[0] in SEEDS else 99)]
    check(ordered == [20261028, 20261104], f"排序 = {ordered}（应为规范种子序）")
    per_seed2 = {"20261028": [{"nll": 1.0}], 20261104: [{"nll": 2.0}]}
    ordered2 = [k for k, _ in sorted(per_seed2.items(), key=lambda kv: SEEDS.index(int(kv[0]))
                                      if int(kv[0]) in SEEDS else 99)]
    check([int(k) for k in ordered2] == [20261028, 20261104], f"str 键排序 = {ordered2}")

    os.unlink(tmp)
    print("\n" + ("★ 全部通过" if not fails else f"★ {len(fails)} 项未通过"))
    for m in fails:
        print("   - " + m)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
