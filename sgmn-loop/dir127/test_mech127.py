#!/usr/bin/env python3
"""`DIR-127` 测量层自测（`T1…T17`）。★ 每条都必须**能红**（`DEF-118-17`）。

用法：
    python test_mech127.py            # 全跑
    python test_mech127.py --inject N # 人为破坏第 N 条 ⇒ 必须**恰好 1 红**
"""
import sys

sys.path.insert(0, "dir127")
sys.path.insert(0, "dir126")
sys.path.insert(0, ".")
import mech127 as MM  # noqa: E402
import mech126 as M126  # noqa: E402

M = MM.M
TESTS = []
_INJ = [0]


def test(fn):
    TESTS.append(fn)
    return fn


def _injecting(idx):
    """`--inject idx` ⇒ 第 `idx` 条测试返回**假**（其余仍真）。

    ★ 第一版写成 `_INJ[0] == 0 or _INJ[0] == idx` ⇒ 注入 `idx` 时该条**仍返回真**
      ⇒ 扫 `1..17` 得到「0/17 恰好 1 红」。这是「断言极性写反」的第 2 次
      （`DEF-126-03` 同族）。⇒ 正确极性：**正常为真，注入目标为假**。
    """
    return _INJ[0] != idx


# ---------------------------------------------------------------- T1
@test
def t1_subsets():
    ss = MM.all_subsets()
    by_k = {}
    for s in ss:
        by_k[len(s)] = by_k.get(len(s), 0) + 1
    ok = (len(ss) == 15 and by_k == {1: 4, 2: 6, 3: 4, 4: 1})
    return ok and _injecting(1), "15 个子集，k 分布 %s" % by_k


# ---------------------------------------------------------------- T2
@test
def t2_availability():
    s0, a0, c0 = MM.subsets_for(0)
    s600, a600, _c = MM.subsets_for(600)
    ok = (len(s0) == 1 and a0 == (555,) and len(s600) == 15 and len(a600) == 4)
    return ok and _injecting(2), "dose=0 → %d 子集/%s；dose=600 → %d 子集/%d 题集" % (
        len(s0), a0, len(s600), len(a600))


# ---------------------------------------------------------------- T3
@test
def t3_two_path():
    """生产路径 vs 独立复算（`A127-4` 的自测版；取 3 个结构抽样）。"""
    bad = 0
    for d, sub in ((600, (557,)), (300, (555, 558)), (25, (555, 556, 557, 558))):
        base = M126.build_ctx(d)
        cells, items, ns, tv = base["cells"], base["items"], base["ns"], base["tv"]
        r = MM.reading(d, sub)
        n1 = sum(sum(1 for v in ns[x] if v == 1) for x in sub)
        n2 = sum(sum(1 for v in ns[x] if v == 2) for x in sub)
        e1 = e2 = 0
        for c in cells:
            for x in sub:
                dd = M.M3.counts_by_stratum(items[x], ns[x], c["picks"][x], tv)
                e1 += dd[1]
                e2 += dd[2]
        if (r["N1"], r["N2"], r["e1"], r["e2"]) != (n1, n2, e1, e2):
            bad += 1
    return bad == 0 and _injecting(3), "3 个结构两路一致，差 %d" % bad


# ---------------------------------------------------------------- T4
@test
def t4_classify_tags():
    """7 个标签都可被 `classify` 产生（构造性）。"""
    base = {"n_cells": 9, "ratio": 0.5, "frac": 0.0, "m_tot": 10}
    cases = {
        "SAME": dict(base, ratio=1.5, frac=0.0),
        "FA":   dict(base, ratio=0.5, frac=0.0),
        "FB":   dict(base, ratio=1.5, frac=0.5),
        "NA":   dict(base, ratio=None),
        "NB":   dict(base, m_tot=0),
        "NC":   dict(base, n_cells=8),
        "E1":   dict(base, ratio=1.0),
    }
    got = {k: MM.classify(v) for k, v in cases.items()}
    ok = all(got[k] == k for k in cases)
    return ok and _injecting(4), "7 个标签全部可达：%s" % sorted(got.values())


# ---------------------------------------------------------------- T5
@test
def t5_frac_reuse():
    """`reading` 的 `frac` 必须**逐位等于** `M126.frac_low_u` 对同一 `obs` 的输出。"""
    d, sub = 450, (555, 557)
    ctx = dict(M126.build_ctx(d))
    ctx["rngs"] = tuple(sub)
    st = M.strata_of(ctx)
    obs = M.stratum_obs(ctx, st)
    fr = M126.frac_low_u(obs)
    r = MM.reading(d, sub)
    ok = (r["frac"] == fr["frac"] and r["m_lo"] == fr["m_lo"] and r["m_tot"] == fr["m_tot"])
    return ok and _injecting(5), "frac=%.10f m_lo/m_tot=%d/%d" % (fr["frac"], fr["m_lo"], fr["m_tot"])


# ---------------------------------------------------------------- T6
@test
def t6_structure_pure():
    """★ `N₁/N₂` **只由结构决定**：换 `rngs` 后与 `picks` 的内容无关。

    做法：把 cell 的 `picks` 全部换成别的值（只改观测、不改 `n_seen`），
    `N₁/N₂` 必须**逐位不变**；`e1/e2` 则必然变。
    """
    d, sub = 300, (555, 557)
    r1 = MM.reading(d, sub)
    base = M126.build_ctx(d)
    fake = []
    for c in base["cells"]:
        c2 = dict(c)
        c2["picks"] = {k: [0] * len(v) for k, v in c["picks"].items()}
        fake.append(c2)
    saved = M126.build_ctx
    try:
        M126.build_ctx = lambda dd: dict(base, cells=fake)
        MM._supports.cache.clear()
        r2 = MM.reading(d, sub)
    finally:
        M126.build_ctx = saved
        MM._supports.cache.clear()
    ok = (r1["N1"], r1["N2"]) == (r2["N1"], r2["N2"])
    return ok and _injecting(6), "N1/N2 不变 (%d,%d)；e1/e2 变为 (%d,%d)" % (
        r2["N1"], r2["N2"], r2["e1"], r2["e2"])


# ---------------------------------------------------------------- T7
@test
def t7_gate_formula():
    ok = MM.gate_of(68, 99) == (2 * 68 + 99) / 3.0 and abs(MM.gate_of(9, 29) - 15.666666666666666) < 1e-12
    return ok and _injecting(7), "gate(68,99)=%.6f gate(9,29)=%.10f" % (
        MM.gate_of(68, 99), MM.gate_of(9, 29))


# ---------------------------------------------------------------- T8
@test
def t8_cache_not_mutated():
    """★ `reading` 只做**浅拷贝**并改 `rngs` ⇒ 不得污染 `_supports` 的缓存本体。"""
    d = 375
    ctx0 = MM._supports(d)
    r = MM.reading(d, (557,))
    ctx1 = MM._supports(d)
    ok = (tuple(ctx0["rngs"]) == tuple(ctx1["rngs"])) and r["k"] == 1
    return ok and _injecting(8), "缓存 rngs 未被污染 = %s" % (tuple(ctx1["rngs"]),)


# ---------------------------------------------------------------- T9
@test
def t9_row_count():
    rs = MM.all_readings()
    ok = len(rs) == 121
    return ok and _injecting(9), "读数行数 = %d（须 121）" % len(rs)


# ---------------------------------------------------------------- T10
@test
def t10_old_vs_new_split():
    """9 个全子集 + 112 个新结构 = 121。"""
    n_full = n_new = 0
    for d in MM.DOSE_ALL9:
        subs, avail, _c = MM.subsets_for(d)
        full = tuple(sorted(avail))
        for s in subs:
            if tuple(s) == full:
                n_full += 1
            else:
                n_new += 1
    ok = (n_full, n_new) == (9, 112)
    return ok and _injecting(10), "全子集 %d + 新结构 %d = %d" % (n_full, n_new, n_full + n_new)


# ---------------------------------------------------------------- T11
@test
def t11_alias_chain():
    """★★ `DEF-126-07` 的教训：**钉住整条别名链**，不只信一层。"""
    ok = (MM.M is M126.M) and (MM.M126 is M126) and hasattr(MM.M, "M3") \
        and tuple(MM.ALL_RNGS) == (555, 556, 557, 558) and MM.N_QS == 4 \
        and tuple(MM.DOSE_ALL9) == (0, 25, 150, 300, 350, 375, 400, 450, 600)
    return ok and _injecting(11), "M→mech125 · M126→mech126 · M.M3 存在 · 常量全对"


# ---------------------------------------------------------------- T12
@test
def t12_idempotent():
    """同一结构算两次必须逐位相同（无隐藏状态）。"""
    a = MM.reading(400, (556, 557))
    b = MM.reading(400, (556, 557))
    keys = ("N1", "N2", "frac", "ratio", "m_sum", "m_max", "gate", "e1", "e2")
    ok = all(a[k] == b[k] for k in keys)
    return ok and _injecting(12), "9 个字段两次相同"


# ---------------------------------------------------------------- T13
@test
def t13_detail_len():
    """逐层明细的长度必须 = cell 数（9 层），且 `m_s` 自洽。"""
    r = MM.reading(300, (555, 556, 557, 558))
    ok = (len(r["detail"]) == 9 and sum(d["m"] for d in r["detail"]) == r["m_sum"]
          and max(d["m"] for d in r["detail"]) == r["m_max"])
    return ok and _injecting(13), "9 层，Σm=%d，max m_s=%d" % (r["m_sum"], r["m_max"])


# ---------------------------------------------------------------- T14
@test
def t14_predict_sign():
    ok = (M126.predict_sign(0.0) == 1 and M126.predict_sign(0.5) == -1
          and M126.predict_sign(None) is None
          and M126.sign_of_ratio(1.5) == 1 and M126.sign_of_ratio(0.5) == -1
          and M126.sign_of_ratio(1.0) == 0 and M126.sign_of_ratio(None) is None)
    return ok and _injecting(14), "predict_sign / sign_of_ratio 边界全对"


# ---------------------------------------------------------------- T15
@test
def t15_frac_bounds():
    """`frac` 必须落在 `[0,1]`，且 `m_lo ≤ m_tot`。"""
    bad = 0
    for r in MM.all_readings():
        if r["frac"] is None:
            continue
        if not (0.0 <= r["frac"] <= 1.0) or r["m_lo"] > r["m_tot"]:
            bad += 1
    return bad == 0 and _injecting(15), "121 个读数 frac 全在 [0,1]，违例 %d" % bad


# ---------------------------------------------------------------- T16
@test
def t16_gate_matches_struct():
    """读数的 `gate` 必须等于由该结构 `(N₁,N₂)` 派生的闸门。"""
    bad = 0
    for r in MM.all_readings():
        if abs(r["gate"] - MM.gate_of(r["N1"], r["N2"])) > 1e-12:
            bad += 1
    return bad == 0 and _injecting(16), "121 个 gate 自洽，违例 %d" % bad


# ---------------------------------------------------------------- T17
@test
def t17_subset_is_subset():
    """子集读数的 `N₁/N₂` 必须 ≤ 全子集的对应值（单调性）。"""
    bad = 0
    for d in MM.DOSE_ALL9:
        subs, avail, _c = MM.subsets_for(d)
        full = tuple(sorted(avail))
        rf = MM.reading(d, full)
        for s in subs:
            rs = MM.reading(d, s)
            if rs["N1"] > rf["N1"] or rs["N2"] > rf["N2"] or rs["m_sum"] > rf["m_sum"]:
                if tuple(s) != full:
                    bad += 1
    return bad == 0 and _injecting(17), "子集 N₁/N₂/Σm 均 ≤ 全子集，违例 %d" % bad


def main():
    if "--inject" in sys.argv:
        _INJ[0] = int(sys.argv[sys.argv.index("--inject") + 1])
    reds = []
    print("=" * 88)
    print("DIR-127 测量层自测（--inject %s）" % (_INJ[0] or "off"))
    print("=" * 88)
    for i, fn in enumerate(TESTS, 1):
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, "异常：%s: %s" % (type(e).__name__, e)
        print("  T%-3d %-10s %s" % (i, "PASS" if ok else "**FAIL**", msg))
        if not ok:
            reds.append(i)
    print("-" * 88)
    if _INJ[0]:
        good = (len(reds) == 1 and reds[0] == _INJ[0])
        print("★ --inject %d ⇒ 红 %s ⇒ %s" % (_INJ[0], reds, "恰好 1 红（正确）" if good else "**异常**"))
        return 0 if good else 1
    print("★ %d/%d PASS" % (len(TESTS) - len(reds), len(TESTS)))
    return 0 if not reds else 1


if __name__ == "__main__":
    sys.exit(main())
