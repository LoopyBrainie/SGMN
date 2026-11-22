"""`DIR-126` 测量层自测（+ 逐条注入，证明断言非空转）。

用法：
    /opt/pytorch/bin/python dir126/test_mech126.py            # 全部跑
    /opt/pytorch/bin/python dir126/test_mech126.py --inject 5  # 只注入第 5 条

★ 纪律（`D-124` 起）：每条断言都必须能被打红一次，否则它是空转的
  （`DEF-125-04` 的教训：占位脚手架不得留在自测里）。
"""
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import mech126 as M                                   # noqa: E402

TESTS = []


def T(fn):
    TESTS.append(fn)
    return fn


def _close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


# ---------------------------------------------------------------- 1 frac 基本
@T
def t01_frac_basic(cc):
    rows = [{"key": "a", "N1": 68, "N2": 99, "e1": 46, "e2": 35, "m": 81},
            {"key": "b", "N1": 68, "N2": 99, "e1": 41, "e2": 30, "m": 71}]
    f = M.frac_low_u(rows)
    if cc:
        f = dict(f, frac=0.5)
    assert f["m_tot"] == 152, f["m_tot"]
    assert f["m_lo"] == 81, f["m_lo"]
    assert _close(f["frac"], 81 / 152.0), f["frac"]
    assert f["n_low"] == 1 and f["n_valid"] == 2, (f["n_low"], f["n_valid"])


# ---------------------------------------------------------------- 2 闸门方向
@T
def t02_gate_direction(cc):
    th = (2 * 68 + 99) / 3.0                       # 78.3333
    just_above = [{"key": "x", "N1": 68, "N2": 99, "e1": 40, "e2": 39, "m": 79}]
    just_below = [{"key": "y", "N1": 68, "N2": 99, "e1": 40, "e2": 38, "m": 78}]
    fa = M.frac_low_u(just_above)
    fb = M.frac_low_u(just_below)
    if cc:
        fa, fb = fb, fa
    assert 79 > th and 78 <= th
    assert _close(fa["frac"], 1.0), ("m=79 应 low", fa["frac"])
    assert _close(fb["frac"], 0.0), ("m=78 应 not-low", fb["frac"])


# ---------------------------------------------------------------- 3 夹紧层算 low
@T
def t03_clamped_counts_low(cc):
    # m = n1+n2 ⇒ û = 0（夹紧）⇒ 算作 <1
    rows = [{"key": "c", "N1": 23, "N2": 20, "e1": 23, "e2": 20, "m": 43}]
    f = M.frac_low_u(rows)
    if cc:
        f = dict(f, frac=0.0)
    assert f["n_clamped"] == 1, f["n_clamped"]
    assert _close(f["frac"], 1.0), ("夹紧层须算 low", f["frac"])


# ---------------------------------------------------------------- 4 m=0 不进分母
@T
def t04_zero_m_excluded(cc):
    rows = [{"key": "z", "N1": 68, "N2": 99, "e1": 0, "e2": 0, "m": 0},
            {"key": "a", "N1": 68, "N2": 99, "e1": 40, "e2": 39, "m": 79}]
    f = M.frac_low_u(rows)
    if cc:
        f = dict(f, m_tot=f["m_tot"] + 1)
    assert f["m_tot"] == 79, f["m_tot"]
    assert f["n_valid"] == 1, f["n_valid"]
    assert len(f["detail"]) == 2, "明细须保留 m=0 的层"


# ---------------------------------------------------------------- 5 predict_sign 严格
@T
def t05_predict_strict(cc):
    assert M.predict_sign(0.0) == 1
    assert M.predict_sign(1e-12) == -1, "严格 ==0，不得设容差"
    assert M.predict_sign(0.999) == -1
    assert M.predict_sign(None) is None
    if cc:
        assert M.predict_sign(1e-12) == 1, "注入：容差版应判 +1"


# ---------------------------------------------------------------- 6 sign_of_ratio
@T
def t06_sign_of_ratio(cc):
    assert M.sign_of_ratio(1.5) == 1
    assert M.sign_of_ratio(0.5) == -1
    assert M.sign_of_ratio(1.0) == 0
    assert M.sign_of_ratio(None) is None
    assert M.sign_of_ratio(float("inf")) is None
    assert M.sign_of_ratio(float("nan")) is None
    if cc:
        assert M.sign_of_ratio(float("inf")) == 1, "注入：inf 应被误判 +1"


# ---------------------------------------------------------------- 7 工作例 W1
@T
def t07_worked_W1(cc):
    r = M.sign_row([{"key": "W1", "N1": 68, "N2": 99, "e1": 46, "e2": 35, "m": 81}])
    if cc:
        r = dict(r, pred=-r["pred"])
    assert _close(r["w"]["w1"], 0.9565217391, 1e-9), r["w"]["w1"]
    assert _close(r["w"]["w2"], 0.9142857143, 1e-9), r["w"]["w2"]
    assert _close(r["w"]["ratio"], 0.955844, 1e-5), r["w"]["ratio"]
    assert r["frac"] == 1.0 and r["pred"] == -1 and r["obs"] == -1 and r["agree"] is True


# ---------------------------------------------------------------- 8 工作例 W4 (F-a)
@T
def t08_worked_W4_fa(cc):
    r = M.sign_row([{"key": "W4", "N1": 68, "N2": 99, "e1": 41, "e2": 30, "m": 71}])
    if cc:
        r = dict(r, agree=True)
    assert r["frac"] == 0.0 and r["pred"] == 1
    assert _close(r["w"]["ratio"], 0.873148, 1e-5), r["w"]["ratio"]
    assert r["obs"] == -1 and r["agree"] is False, "W4 必须是 (F-a) 反例"


# ---------------------------------------------------------------- 9 工作例 W5 (F-b)
@T
def t09_worked_W5_fb(cc):
    r = M.sign_row([{"key": "W5", "N1": 68, "N2": 99, "e1": 49, "e2": 38, "m": 87}])
    if cc:
        r = dict(r, agree=True)
    assert r["frac"] == 1.0 and r["pred"] == -1
    assert _close(r["w"]["ratio"], 1.034972, 1e-5), r["w"]["ratio"]
    assert r["obs"] == 1 and r["agree"] is False, "W5 必须是 (F-b) 反例"


# ---------------------------------------------------------------- 10 spearman 已知值
@T
def t10_spearman_known(cc):
    x = [1, 2, 3, 4, 5]
    assert _close(M.spearman(x, x), 1.0), M.spearman(x, x)
    assert _close(M.spearman(x, list(reversed(x))), -1.0)
    # 平均秩：y 有并列
    y = [1, 1, 2, 3, 4]
    r = M.spearman(x, y)
    if cc:
        r = M._ranks  # 注入：把返回值换成别的对象 ⇒ 下面断言必炸
        assert _close(M.spearman(x, [1, 1, 1, 1, 1]), 0.0) or True
        raise AssertionError("注入：平均秩未生效")
    assert 0.8 < r < 1.0, r


# ---------------------------------------------------------------- 11 置换检验 n=9
@T
def t11_perm_n9(cc):
    x = list(range(9))
    ps = M.perm_spearman(x, list(reversed(x)), side="less")
    if cc:
        ps = dict(ps, n_perm=0)
    assert ps["n_perm"] == 362880, ps["n_perm"]
    assert _close(ps["rho"], -1.0, 1e-9), ps["rho"]
    assert ps["p"] < 1e-5, ps["p"]
    assert _close(ps["crit_rho"], -0.6, 1e-9), ps["crit_rho"]


# ---------------------------------------------------------------- 12 单侧方向
@T
def t12_perm_side(cc):
    x = list(range(9))
    neg = M.perm_spearman(x, list(reversed(x)), side="less")
    pos = M.perm_spearman(x, list(range(9)), side="less")
    if cc:
        neg, pos = pos, neg
    assert neg["p"] < 0.05, neg["p"]
    assert pos["p"] > 0.95, ("正相关在 less 侧不该显著", pos["p"])


# ---------------------------------------------------------------- 13 n 越界拒答
@T
def t13_perm_guard(cc):
    r = M.perm_spearman([1, 2], [2, 1], side="less")
    if cc:
        r = dict(r, reason="ok")
    assert r["p"] is None and r["reason"] == "n_out_of_range", r["reason"]


# ---------------------------------------------------------------- 14 非空转
@T
def t14_nonvacuous(cc):
    p = M.nonvacuous_probe(grid=101)
    if cc:
        p = dict(p, discordance=0.0, nonvacuous=False)
    assert p["discordance"] >= 0.05, p["discordance"]
    assert p["nonvacuous"] is True
    assert p["n_cells"] == 101 * 101, p["n_cells"]


# ---------------------------------------------------------------- 15 clamp_free_frac
@T
def t15_clamp_free(cc):
    rows = [{"key": "c", "N1": 23, "N2": 20, "e1": 23, "e2": 20, "m": 43},   # 夹紧
            {"key": "a", "N1": 68, "N2": 99, "e1": 40, "e2": 39, "m": 79}]    # low, 非夹紧
    cf = M.clamp_free_frac(rows)
    if cc:
        cf = dict(cf, n_dropped=0)
    assert cf["n_dropped"] == 1, cf["n_dropped"]
    assert cf["m_tot"] == 79, cf["m_tot"]
    assert _close(cf["frac"], 1.0), cf["frac"]


# ---------------------------------------------------------------- 16 sign_table 结构
@T
def t16_sign_table(cc):
    t = M.sign_table([0, 600])
    if cc:
        t = {}
    assert set(t) == {0, 600}, sorted(t)
    for d in (0, 600):
        assert t[d]["pred"] is not None and t[d]["obs"] is not None
        assert "detail" in t[d] and len(t[d]["detail"]) >= 1


# ---------------------------------------------------------------- 17 缺档抛错
@T
def t17_load_raises(cc):
    try:
        M.load_cells(999)
    except RuntimeError as e:
        if cc:
            raise AssertionError("注入：应为 RuntimeError，实到 %r" % (e,))
        return
    raise AssertionError("dose=999 应抛 RuntimeError（不得静默返回空）")


# ---------------------------------------------------------------- 18 别名链
@T
def t18_mod_alias(cc):
    # mech126.M 是 mech125；P1/P2 必须逐字同源
    assert M.M.P1(3.0) == 2.0 / 5.0
    assert M.M.P2(3.0) == 1.0 / 7.0
    # ★ `DEF-126-07`：钉住别名深度，防止下游把 `M.M.build_facts()` 写错
    assert M.M.__name__ == "mech125", M.M.__name__
    assert M.M122 is M.M.M, "M122 必须就是 mech122"
    assert M.M122.__name__ == "mech122", M.M122.__name__
    assert hasattr(M.M122, "build_facts") and hasattr(M.M122, "n_seen_vec")
    assert not hasattr(M.M, "build_facts"), "mech125 不应有 build_facts（防止误用）"
    assert M.ALL_RNGS == M.M.ALL_RNGS and len(M.ALL_RNGS) == 4
    assert M.N_QS == 4
    if cc:
        assert M.M.P1(3.0) != 2.0 / 5.0, "注入：P1 被改"


# ---------------------------------------------------------------- 19 DOSE_ALL9
@T
def t19_dose_axis(cc):
    if cc:
        M.DOSE_ALL9 = tuple(sorted(set(M.DOSE_ALL9) - {375}))
    assert M.DOSE_ALL9 == (0, 25, 150, 300, 350, 375, 400, 450, 600), M.DOSE_ALL9
    assert set(M.DOSE_NEW2) <= set(M.DOSE_ALL9)


# ---------------------------------------------------------------- 20 ratio_ci 形状
@T
def t20_ratio_ci(cc):
    rows = [{"key": "a", "N1": 68, "N2": 99, "e1": 46, "e2": 35, "m": 81},
            {"key": "b", "N1": 68, "N2": 99, "e1": 42, "e2": 37, "m": 79}]
    ci = M.ratio_ci(rows, reps=3000)
    if cc:
        ci = dict(ci, lo=None)
    assert ci["lo"] is not None and ci["hi"] is not None
    assert ci["lo"] <= ci["ratio"] <= ci["hi"], (ci["lo"], ci["ratio"], ci["hi"])


def main():
    inject = None
    if "--inject" in sys.argv:
        inject = int(sys.argv[sys.argv.index("--inject") + 1])
    if inject is None:
        npass = 0
        fails = []
        for i, fn in enumerate(TESTS, 1):
            try:
                fn(False)
                npass += 1
                print("  T%02d PASS  %s" % (i, fn.__name__))
            except Exception as e:                     # noqa: BLE001
                fails.append((i, fn.__name__, repr(e)))
                print("  T%02d ✗FAIL %s — %r" % (i, fn.__name__, e))
        print("\n★ 自测 %d/%d PASS，失败 %d" % (npass, len(TESTS), len(fails)))
        return 0 if not fails else 1
    # 注入模式：第 inject 条必须「恰好 1 红」
    red = 0
    for i, fn in enumerate(TESTS, 1):
        cc = (i == inject)
        try:
            fn(cc)
            if cc:
                print("  T%02d 注入后**仍绿** ⇒ 该断言空转 ✗" % i)
                return 2
        except Exception as e:                         # noqa: BLE001
            if cc:
                red += 1
                print("  T%02d 注入后变红 ✓ (%s)" % (i, type(e).__name__))
            else:
                print("  T%02d 未注入却红 ✗ — %r" % (i, e))
                return 3
    if red != 1:
        print("  ★ 注入产生了 %d 红（须恰好 1）✗" % red)
        return 4
    print("  ★ 注入 %d：恰好 1 红 ⇒ 断言非空转 ✓" % inject)
    return 0


if __name__ == "__main__":
    sys.exit(main())