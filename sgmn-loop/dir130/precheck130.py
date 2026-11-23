"""`DIR-130` 前置核 / Stage A 硬门 / 自测 / 工作例 / formal 驱动。

用法（**全部在 trainer 上跑**）：
    /opt/pytorch/bin/python precheck130.py --stage precheck   # Stage A 硬门 + 自测 + 工作例
    /opt/pytorch/bin/python precheck130.py --stage smoke      # 只验管线形状，不出判决
    /opt/pytorch/bin/python precheck130.py --stage formal     # 上面全过之后再跑，打印全部判据读数
    /opt/pytorch/bin/python precheck130.py --inject K          # 让第 K 条自测必定变红

★ 纪律（`DEF-129-02`）：`--stage precheck` **只打印结构事实 + 单一结构的可算性 + 逻辑 level**；
  **不打印**属于判据的**聚合读数**（`p₃₅₀`/`p₃₀₀`/逐 `k` 曲线/余量极值）—— 那是 `--stage formal` 的事。
"""
import argparse
import collections
import contextlib
import io
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import mech130 as M                                              # noqa: E402


# ------------------------------------------------------------------ 夹具（自测用）
def _r(frac, k=9, m_tot=10, m_lo=0):
    return {"frac": frac, "k": k, "m_tot": m_tot, "m_lo": m_lo}


def _lay(m, u, clamped=False):
    return {"m": m, "u": u, "clamped": clamped}


def fx_mixed():
    """1 正 / 1 零 / 1 不可判 / 1 正 ⇒ `p = 2/3`、`n_comp=3`。"""
    return [_r(0.5), _r(0.0), _r(None, m_tot=0), _r(0.25)]


def fx_all_none():
    return [_r(None, m_tot=0) for _ in range(4)]


def fx_span_k():
    """`k = 1..9` 各一条，用于 `C5` 的形状核。"""
    return [_r(0.1 * k, k=k) for k in range(1, 10)]


def fx_reading(dose, sub, frac, m_tot, n_low):
    return {"dose": dose, "sub": tuple(sub), "k": 9, "frac": frac,
            "m_tot": m_tot, "n_low": n_low}


# ------------------------------------------------------------------ 自测
def _tests(inject=0):
    """返回 `[(name, fn), ...]`；`fn()` 返回 `(ok, msg)`。`inject==k` 时把第 k 条的期望翻反。"""
    T = []

    def t1():
        d = M.p_flip(fx_mixed())
        exp = {"p": 2.0 / 3.0, "n_pos": 2, "n_comp": 3, "n_uncomp": 1}
        ok = (abs(d["p"] - exp["p"]) < 1e-15 and d["n_pos"] == exp["n_pos"]
              and d["n_comp"] == exp["n_comp"] and d["n_uncomp"] == exp["n_uncomp"])
        if inject == 1:
            ok = not ok
        return (ok, f"p={d['p']!r} pos={d['n_pos']} comp={d['n_comp']} uncomp={d['n_uncomp']}")
    T.append(("T1 p_flip 排除 None 且 p=2/3", t1))

    def t2():
        d = M.p_flip(fx_all_none())
        ok = (d["p"] is None and d["n_pos"] == 0 and d["n_comp"] == 0 and d["n_uncomp"] == 4)
        if inject == 2:
            ok = not ok
        return (ok, f"p={d['p']} comp={d['n_comp']} uncomp={d['n_uncomp']}")
    T.append(("T2 全不可判 ⇒ p=None 不除零", t2))

    def t3():
        # ★ 判决表 5 行逐行
        got = [M.decide(False, None, None),
               M.decide(True, 0.0, 1.0),
               M.decide(True, 0.0, 0.0),
               M.decide(True, 1.0, 1.0),
               M.decide(True, 1.0, 0.0)]
        exp = list(M.VERDICTS)
        ok = (got == exp)
        if inject == 3:
            ok = not ok
        return (ok, f"got={[g.split('-', 1)[1] for g in got]}")
    T.append(("T3 判决表 5 行逐行命中", t3))

    def t4():
        # ★ C0 不过 ⇒ 无论 C2/C3 如何都必须是 PROTOCOL-INVALID
        got = {M.decide(False, a, b) for a in (0.0, 1.0, None) for b in (0.0, 1.0, None)}
        ok = (got == {M.VERDICTS[0]})
        if inject == 4:
            ok = not ok
        return (ok, f"unique={len(got)} 全为 PROTOCOL-INVALID={ok}")
    T.append(("T4 C0 阻断优先于一切", t4))

    def t5():
        n = len(list(M.cell_subsets()))
        ks = collections.Counter(k for k, _ in M.cell_subsets())
        ok = (n == 511 and ks[1] == 9 and ks[9] == 1 and sum(ks.values()) == 511)
        if inject == 5:
            ok = not ok
        return (ok, f"n={n} k1={ks[1]} k9={ks[9]}")
    T.append(("T5 cell 子集恰 511 个", t5))

    def t6():
        got = M.k_counts()
        exp = [9, 36, 84, 126, 126, 84, 36, 9, 1]
        ok = (got == exp and sum(got) == 511)
        if inject == 6:
            ok = not ok
        return (ok, f"{got}")
    T.append(("T6 C(9,k) = 9,36,84,126,126,84,36,9,1", t6))

    def t7():
        subs = M.all_subsets()
        by_k = collections.Counter(len(s) for s in subs)
        ok = (len(subs) == 15 == 2 ** 4 - 1 and (by_k[1], by_k[2], by_k[3], by_k[4]) == (4, 6, 4, 1))
        if inject == 7:
            ok = not ok
        return (ok, f"n={len(subs)} k={dict(by_k)}")
    T.append(("T7 题集恰 15 个 = 2^4-1", t7))

    def t8():
        # ★ 复现核**能抓**差异（构造性见证）
        rows = [fx_reading(350, (555,), 0.0, 100, 0)]
        good = [fx_reading(350, (555,), 0.0, 100, 0)]
        bad = [fx_reading(350, (555,), 0.25, 100, 0)]
        r_ok = M.reproduction(rows, good)
        r_bad = M.reproduction(rows, bad)
        ok = (r_ok["ok"] is True and r_ok["n_compared"] == 1
              and r_bad["ok"] is False and r_bad["n_bad"] == 1)
        if inject == 8:
            ok = not ok
        return (ok, f"good_ok={r_ok['ok']} bad_ok={r_bad['ok']} bad_n={r_bad['n_bad']}")
    T.append(("T8 复现核能抓 frac 差异", t8))

    def t9():
        rows = fx_span_k()
        pk = M.per_k(rows)
        ok = (sorted(pk) == list(range(1, 10))
              and sum(v["n_rows"] for v in pk.values()) == len(rows))
        if inject == 9:
            ok = not ok
        return (ok, f"keys={sorted(pk)} 覆盖={sum(v['n_rows'] for v in pk.values())}/{len(rows)}")
    T.append(("T9 per_k 覆盖 k=1..9 且不丢行", t9))

    def t10():
        rows = [{"k": 9, "detail": [_lay(5, 0.0, True), _lay(3, 1.5), _lay(0, None)]}]
        d = M.u_margin(rows)
        ok = (d["umin"] == 0.0 and d["n_low_layers"] == 1 and d["n_clamped"] == 1
              and d["n_layers"] == 2 and d["margin_below"] == 1.0)
        if inject == 10:
            ok = not ok
        return (ok, f"umin={d['umin']} nlow={d['n_low_layers']} ncl={d['n_clamped']}")
    T.append(("T10 u_margin 计夹紧且忽略 m=0 层", t10))

    def t11():
        # ★ 阈值边界：≤ 0.05 与 ≥ 0.95 都算**成立**
        ok = (M.decide(True, M.P350_MAX, M.P300_MIN) == M.VERDICTS[1]
              and M.decide(True, M.P350_MAX + 1e-12, M.P300_MIN) == M.VERDICTS[3]
              and M.decide(True, M.P350_MAX, M.P300_MIN - 1e-12) == M.VERDICTS[2])
        if inject == 11:
            ok = not ok
        return (ok, f"边界 <=/>= 判成立={ok}")
    T.append(("T11 阈值边界 ≤0.05 / ≥0.95 算成立", t11))

    def t12():
        # ★ 严格 >0：frac 恰为 0.0 **不**计入正侧
        d = M.p_flip([_r(0.0), _r(0.0), _r(1e-300)])
        ok = (d["p"] == 1.0 / 3.0 and d["n_pos"] == 1)
        if inject == 12:
            ok = not ok
        return (ok, f"p={d['p']!r} pos={d['n_pos']} （1e-300 算正、0.0 不算）")
    T.append(("T12 frac>0 严格、1e-300 算正", t12))

    def t13():
        # ★ S9 检测器非空转：干净串 → []，四个禁用词逐个 → 命中
        clean = "Stage A 全过; 结构事实; 工作例 frac=0.0"
        hits = {w: _s9_detect("x" + w + "y") for w in S9_FORBIDDEN}
        ok = (_s9_detect(clean) == [] and all(len(v) == 1 and v[0] == w for w, v in hits.items()))
        if inject == 13:
            ok = not ok
        return (ok, f"干净串命中={_s9_detect(clean)} 逐词命中={all(bool(v) for v in hits.values())}")
    T.append(("T13 S9 检测器非空转（干净串不误报）", t13))

    return T


def run_selftests(inject=0):
    T = _tests(inject)
    bad = []
    for name, fn in T:
        try:
            ok, msg = fn()
        except Exception as exc:                      # noqa: BLE001
            ok, msg = False, f"异常 {type(exc).__name__}: {exc}"
        print(f"  {'✓' if ok else '✗'} {name}: {msg}")
        if not ok:
            bad.append(name)
    return bad


# ------------------------------------------------------------------ 全域读数（Stage A 与 formal 共用）
_CACHE = {}


def _sweeps():
    """两剂量的全域枚举，**只做一遍**（Stage A 的 `S5`/`S8` 与 formal 共用同一批读数）。"""
    if "rows" not in _CACHE:
        subs = M.all_subsets()
        _CACHE["rows_a"], _CACHE["meta_a"] = M.sweep(M.DOSE_A, subs)
        _CACHE["rows_b"], _CACHE["meta_b"] = M.sweep(M.DOSE_B, subs)
        _CACHE["subs"] = subs
    return _CACHE["rows_a"], _CACHE["rows_b"]


def _cells(dose):
    return M._base_ctx(dose)["cells"]


def _write_mode_opens(src):
    """★ 用 **AST** 找真正的写模式 `open()` 调用（`DEF-130-02`：文本扫描会被**文档字符串**污染）。

    判据：`mode` 参数（第 2 位置或 `mode=`）为常量字符串且含 `w/a/x/+` ⇒ 记为写；
    `mode` 缺省（⇒ `"r"`）或为 `"r"/"rb"` ⇒ 不算。动态 `mode` 变量 ⇒ 记为可疑（保守）。
    """
    import ast
    found = []
    write_chars = set("wax+")
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Name) and fn.id == "open"):
            continue
        mode_node = node.args[1] if len(node.args) > 1 else None
        for kw in node.keywords:
            if kw.arg == "mode":
                mode_node = kw.value
        if mode_node is None:
            continue                                     # 缺省 = 只读
        if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
            if write_chars & set(mode_node.value):
                found.append({"line": node.lineno, "mode": mode_node.value})
        else:
            found.append({"line": node.lineno, "mode": "<dynamic>"})
    return found


# ------------------------------------------------------------------ Stage A 守卫
def stage_a():
    """`S1`..`S8`（卡片 §5）。返回 `[(id, ok, detail), ...]`。"""
    out = []
    rows_a, rows_b = _sweeps()
    rnd = random.Random(1302026)

    # --- S1 结构：cells / C(9,k) / 题集
    ca, cb = _cells(M.DOSE_A), _cells(M.DOSE_B)
    sa = tuple(sorted(c["seed"] for c in ca))
    sb = tuple(sorted(c["seed"] for c in cb))
    kc = M.k_counts()
    subs = _CACHE["subs"]
    bk = collections.Counter(len(s) for s in subs)
    ok_S1 = (len(ca) == 9 and len(cb) == 9
             and len(set(sa)) == 9 and len(set(sb)) == 9          # 逐层 = 逐 seed 的前提
             and sa == sb == M.SEEDS_ARCHIVED                      # 两剂量同 seed 集，且与归档常量一致
             and kc == [9, 36, 84, 126, 126, 84, 36, 9, 1] and sum(kc) == 511
             and len(subs) == 15 and (bk[1], bk[2], bk[3], bk[4]) == (4, 6, 4, 1))
    out.append(("S1", ok_S1,
                f"cells={len(ca)}/{len(cb)} 互异seed={len(set(sa))}/{len(set(sb))} "
                f"两剂量同集={sa == sb} 与归档常量一致={sa == M.SEEDS_ARCHIVED} "
                f"ΣC(9,k)={sum(kc)} subs={len(subs)} k={dict(bk)}"))

    # --- S2 (F1) N1/N2 与 cells 无关
    bad2 = 0
    for dose in (M.DOSE_A, M.DOSE_B):
        ref = {}
        for _ in range(20):
            k = rnd.randint(1, 9)
            idxs = tuple(sorted(rnd.sample(range(9), k)))
            for sub in subs:
                r = M.reading_cells(dose, sub, idxs)
                key = (dose, tuple(sub))
                ref.setdefault(key, (r["N1"], r["N2"]))
                if ref[key] != (r["N1"], r["N2"]):
                    bad2 += 1
    out.append(("S2", bad2 == 0, f"(F1) N1/N2 跨随机 C 不一致次数={bad2}（须 0）"))

    # --- S3 (C0-1) k=9 复现 readings.json 的 30 行
    rep = M.reproduction(rows_a + rows_b, M.load_readings())
    ok_S3 = (rep["ok"] and rep["n_compared"] == 30)
    out.append(("S3", ok_S3,
                f"(C0-1) k=9 复现 {rep['n_compared']}/30 行，不一致={rep['n_bad']}"
                + ("" if ok_S3 else f" 例={rep['bad'][:1]}")))

    # --- S4 (F2) 加性 + 单调
    bad4 = 0
    single = {}
    for dose in (M.DOSE_A, M.DOSE_B):
        for i in range(9):
            for sub in subs:
                single[(dose, tuple(sub), i)] = M.reading_cells(dose, sub, (i,))["m_tot"]
    for dose in (M.DOSE_A, M.DOSE_B):
        for _ in range(50):
            k = rnd.randint(2, 9)
            idxs = tuple(sorted(rnd.sample(range(9), k)))
            for sub in subs:
                tot = M.reading_cells(dose, sub, idxs)["m_tot"]
                if tot != sum(single[(dose, tuple(sub), i)] for i in idxs):
                    bad4 += 1
                if k > 1:
                    drop = idxs[0]
                    rest = tuple(i for i in idxs if i != drop)
                    if tot > M.reading_cells(dose, sub, rest)["m_tot"] + single[(dose, tuple(sub), drop)]:
                        bad4 += 1
    out.append(("S4", bad4 == 0, f"(F2) 加性/单调违例={bad4}（须 0）"))

    # --- S5 frac None ⟺ m_tot==0；frac==0.0 ⟺ m_lo==0（两剂量全域）
    # ★ v1.3：跨读数的**计数**走 `M.count_over`（返回带身份的 `Agg` ⇒ precheck 渲染即失败）。
    #   ★ 这里**不再打印** None/零 的个数 —— 那正是 `P133-8` 实测的泄漏面
    #   （`None=270` / `零=11092` = 两个 per-dose 计数的**池化** ⇒ 可界住 p ⇒ 可判掉分支）。
    bad5 = 0
    for r in rows_a + rows_b:
        is_none = r["frac"] is None
        if is_none != (r["m_tot"] == 0):
            bad5 += 1
        if not is_none:
            if (r["frac"] == 0.0) != (r["m_lo"] == 0):
                bad5 += 1
    n_none = M.count_over(rows_a + rows_b, lambda r: r["frac"] is None, "n_none")
    n_zero = M.count_over(rows_a + rows_b, lambda r: r["frac"] is not None and r["frac"] == 0.0,
                           "n_zero")
    out.append(("S5", bad5 == 0, f"违例={bad5}（须 0）"))

    # --- S6 mech130 无写模式 open
    src = open(os.path.join(HERE, "mech130.py"), "r", encoding="utf-8").read()
    writes = _write_mode_opens(src)
    out.append(("S6", len(writes) == 0,
                f"mech130 写模式 open 调用={len(writes)}（须 0，AST 判定）{writes[:2]}"))

    # --- S7 NC 标签不得用作过滤：k=1 时 n_cells==1 仍计入
    # ★ v1.3：只报**布尔**，不报 `n_comp` 的值 —— `n_comp` 是跨读数聚合（`Agg`），
    #   在 precheck 阶段**渲染即失败** ⇒ 它根本不该出现在输出里。
    one = [r for r in rows_b if r["k"] == 1]
    d1 = M.p_flip(one)
    uses_nc = ('"NC"' in src) or ("'NC'" in src) or ("verdict_of" in src)
    counted = bool(d1["n_comp"] > 0)
    ok_S7 = (len(one) == 9 * 15 and one[0]["n_cells"] == 1 and one[0]["k"] == 1
             and counted and not uses_nc)
    out.append(("S7", ok_S7,
                f"k=1 行数={len(one)}（须 9×15=135）n_cells={one[0]['n_cells']} "
                f"计入>0={counted} 源码用 NC 标签={uses_nc}（须 False）"))

    # --- S8 总行数 == 15330
    n_tot = len(rows_a) + len(rows_b)
    ok_S8 = (n_tot == M.NOMINAL_TOTAL == 15330
             and len(rows_a) == M.NOMINAL_PER_DOSE and len(rows_b) == M.NOMINAL_PER_DOSE)
    out.append(("S8", ok_S8, f"总行数={n_tot} 每剂量={len(rows_a)}/{len(rows_b)} 标称={M.NOMINAL_TOTAL}"))
    return out


#: `S9` 的禁用词 —— 这些只许出现在 formal 段
S9_FORBIDDEN = ("p₃₅₀ =", "p₃₀₀ =", "★ 判决:", "C130-FRAC", "C130-PROTOCOL")


def _s9_detect(text):
    """★ `S9` 检测器：返回命中的禁用词（空 ⇒ 静默）。

    单列成函数，专为**可自测**（`T13` 用构造性见证证明它不是空转）。
    """
    return [w for w in S9_FORBIDDEN if w in text]


def _precheck_subprocess():
    """跑 `--stage precheck` 子进程，返回 `(rc, stdout, stderr)`。"""
    import subprocess
    try:
        p = subprocess.run([sys.executable, os.path.join(HERE, "precheck130.py"),
                            "--stage", "precheck"],
                           capture_output=True, text=True, timeout=1700)
    except Exception as exc:                          # noqa: BLE001
        return (None, "", f"__EXC__ {type(exc).__name__}: {exc}")
    return (p.returncode, (p.stdout or ""), (p.stderr or ""))


def s9_precheck_silent():
    """`S9`：★ `--stage precheck` **不得**产出任何 `C2`/`C3` 读数或判决行（阶段纪律）。

    做法：把 `--stage precheck` 作为**子进程**跑一遍，扫它的 stdout/stderr。
    ★ 不递归：`precheck` 分支在 Stage A 之后直接 `return`，不会再进 smoke/formal。
    """
    rc, so, se = _precheck_subprocess()
    bad = _s9_detect(so + se)
    ok = (not bad) and (rc == 0)
    return ("S9", ok, f"precheck 子进程 rc={rc} 命中判据读数={bad}（须 []）"), so


# ------------------------------------------------------------------ `S10`（`v1.3`）：数字来源审计
def _num_tokens(text):
    """抽取**ASCII** 数字 token。

    ★ 必须用 `[0-9]` 而不是 `\\d` —— 后者会匹配 `p₃₅₀` 的 Unicode 下标数字（`₃₅₀`）。
    """
    return re.findall(r"[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?", text)


def _structural_numbers():
    """`v1.3` §6 的 (a) 类：**结构常量**（由 `ns`/题集/子集结构决定，与读数无关）。"""
    S = {float(x) for x in range(0, 10)}
    S |= {float(M.N_CELL_SUBSETS), float(M.NOMINAL_PER_DOSE), float(M.NOMINAL_TOTAL)}
    S |= {float(x) for x in M.k_counts()}
    S |= {float(2 ** 4 - 1), float(M.N_CELLS * 15), float(2 * 15)}
    S |= {float(M.DOSE_A), float(M.DOSE_B)}
    S |= {float(M.CARD_ID)}                        # ★ 卡号（标题行 `DIR-130 …`）
    S |= {float(x) for x in M.M.ALL_RNGS}          # ★ `M` 是 mech130；`M.M` 才是 mech125
    S |= {float(s) for s in M.SEEDS_ARCHIVED}
    return S


def _single_structure_numbers():
    """`v1.3` §6 的 (ii) 类：**单结构量** —— 审计器**重算**同一条读数取得。"""
    r = M.reading_cells(M.DOSE_B, (555,), (0,))
    return {float(r["m_tot"]), float(r["n_low"]), float(r["n_cells"]),
            float(r["N1"]), float(r["N2"]), float(r["frac"])}


def _fixture_numbers():
    """`v1.3` §6 的 (iii) 类：**合成夹具**的值 —— 审计器**重跑**自测取得。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_selftests(0)
    return {float(t) for t in _num_tokens(buf.getvalue())}


def s10_number_audit(text):
    """★ `S10`（`v1.3`）：precheck 输出的**数字来源审计**。

    规则：输出中出现的**每一个**数字，必须可溯源到
      (a) **结构常量**（与读数无关）；或
      (b) **单结构量**（**一条**读数的 `m_tot`/`n_low`/`frac`/`n_cells`/`N1`/`N2`）；或
      (c) **合成夹具**的值（自测消息里的数字）。
    ⇒ 任何**跨读数聚合**（`p`/`n_pos`/`n_comp`/`n_uncomp`/`comp_frac`/逐 `k`/池化/部分和）
      既不在 (a) 也不在 (b)/(c) ⇒ **必被抓出**。

    ★ 与**载体守卫**互补：载体拦「走了 `Agg` 出口」的；本条拦「绕过出口裸算的」。
      ★ `S10` 是**第二道网**：它的白名单故意包含 `0..9`（小整数）——
        若某条跨读数聚合计数恰好是个位数，载体仍会拦住它（它在 precheck 根本渲染不出来）；
        `S10` 拦的是**载体没覆盖到**的写法。
    """
    nums = {float(t) for t in _num_tokens(text)}
    allowed = _structural_numbers() | _single_structure_numbers() | _fixture_numbers()
    bad = sorted(nums - allowed)
    return ("S10", not bad,
            f"输出数字去重 {len(nums)} 个；未溯源 {len(bad)} 个（须 0）{bad[:6]}")


# ------------------------------------------------------------------ 工作例
def worked_example():
    """卡片 §6：`dose=350`、`sub={555}`、`C={20260501}`（`k=1`）—— 只证"单一结构可算"。"""
    r = M.reading_cells(M.DOSE_B, (555,), (0,))
    print("\n【工作例 §6】dose=350, sub={555}, C={20260501}（k=1）")
    print(f"  m_tot = {r['m_tot']}  n_low = {r['n_low']}  n_cells = {r['n_cells']}")
    print(f"  N1 = {r['N1']}  N2 = {r['N2']}   （★ 结构量，与 cells 无关 —— (F1)）")
    print(f"  frac = {r['frac']!r}")
    print("  ★ p₃₅₀/p₃₀₀ 属 C2/C3，按纪律**只在 --stage formal 打印**")
    return r


# ------------------------------------------------------------------ 主流程
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="precheck", choices=["precheck", "smoke", "formal"])
    ap.add_argument("--inject", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    # ★ `v1.3`：阶段必须在任何读数/打印之前定下 —— `precheck`/`smoke` 下真实读数的
    #   跨读数聚合量**渲染即失败**（`Agg` 载体）；只有 `formal` 可打印。
    M.set_stage(a.stage)

    nT = len(_tests(0))

    if a.inject:
        # ★ 注入模式只为验"自测非空转"，**不跑 Stage A**（Stage A 与注入无关，且跑 13 遍太贵）
        print(f"自测 T1..T{nT}（★ 注入模式，期望恰好 1 红）")
        bad = run_selftests(a.inject)
        ok = (len(bad) == 1)
        print(f"  ⇒ 红 {len(bad)} 条: {bad}")
        print(f"\n★ 注入 #{a.inject}: {'通过（该测试非空转）' if ok else '失败（非恰好 1 红）'}")
        return 0 if ok else 2

    print("=" * 78)
    print("DIR-130 Stage A 守卫核（只打结构事实；不打 C2/C3 聚合读数）")
    print("=" * 78)
    ga = stage_a()
    for sid, ok, detail in ga:
        print(f"  {'✓' if ok else '✗'} {sid}: {detail}")

    print(f"\n自测 T1..T{nT}")
    bad = run_selftests(0)
    print(f"  ⇒ 自测 {nT - len(bad)}/{nT} 过，红 {len(bad)} 条: {bad}")

    worked_example()

    sbad = [sid for sid, ok, _ in ga if not ok]
    if sbad:
        print(f"\n★ Stage A 失败: {sbad} ⇒ 阻断 formal")
        return 1
    print("\n★ Stage A 全过")

    rows_a, rows_b = _sweeps()

    # ★ DEF-130-04：`--stage precheck` 必须在**这里**停住 —— 否则会穿透到 formal
    #   而把 C2/C3 读数与判决行打出来（违反本卡 §6 的 `DEF-129-02` 纪律）。
    if a.stage == "precheck":
        print("  （--stage precheck 到此为止：C2/C3 与判决只在 --stage formal 产出）")
        return 0

    if a.stage == "smoke":
        print("\n【S9/S10 · `--stage precheck` 的阶段纪律守卫（`v1.3`）】")
        (sid9, ok9, d9), pre_out = s9_precheck_silent()
        print(f"  {'✓' if ok9 else '✗'} {sid9}: {d9}")
        sid10, ok10, d10 = s10_number_audit(pre_out)
        print(f"  {'✓' if ok10 else '✗'} {sid10}: {d10}")
        if not (ok9 and ok10):
            print("\n★ S9/S10 失败 ⇒ 阻断 formal")
            return 1
        # ★ 落盘**被审计过的那一份** precheck 输出（归档产物与被审计文本逐字一致）
        os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
        with open(os.path.join(HERE, "out", "precheck130_v13.log"), "w", encoding="utf-8") as fh:
            fh.write(pre_out)
        res = M.analyse(rows_a=rows_a, rows_b=rows_b)
        need = {"c0", "c2_p350", "c3_p300", "c4_u350", "c5_per_k_350", "verdict"}
        shape = {
            "六个顶层键齐": need <= set(res),
            "C0 有 ok/复现": {"ok", "reproduction_ok"} <= set(res["c0"]),
            "C2 有 p/thr": {"p", "thr_max", "holds"} <= set(res["c2_p350"]),
            "C3 有 p/thr": {"p", "thr_min", "holds"} <= set(res["c3_p300"]),
            "C5 覆盖 k=1..9": len(res["c5_per_k_350"]) == 9,
            "总行数 15330": res["n_rows_total"] == 15330,
            "verdict 在冻结枚举里": res["verdict"] in M.VERDICTS,
        }
        print("\n【SMOKE —— 只验管线形状，不产出判决】")
        for k, v in shape.items():
            print(f"  {'✓' if v else '✗'} {k}")
        print("  ★ smoke **不打印** p₃₅₀/p₃₀₀ 的数值、不打印判决行（按纪律）")
        ok = all(shape.values())
        if a.out:
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump(M.plainify({"shape": shape, "ok": ok,
                                      "meta_a": res["meta_a"], "meta_b": res["meta_b"]}), fh,
                          ensure_ascii=False, indent=1)
        return 0 if ok else 1

    # ---------------- formal
    res = M.analyse(rows_a=rows_a, rows_b=rows_b)
    print("\n" + "=" * 78)
    print("DIR-130 FORMAL —— 判据读数")
    print("=" * 78)

    c0 = res["c0"]
    print("\n【C0 · 协议有效性（阻断项）】")
    print(f"  复现 k=9 vs readings.json: {c0['reproduction']['n_compared']}/30 行, "
          f"不一致 {c0['reproduction']['n_bad']} ⇒ ok={c0['reproduction_ok']}")
    print(f"  可算基数: dose=300 {c0['comp_frac_a']:.6f} · dose=350 {c0['comp_frac_b']:.6f} "
          f"（须 ≥ {c0['min_computable']}）")
    print(f"  ⇒ C0 {'成立' if c0['ok'] else '不成立'}")

    c2, c3 = res["c2_p350"], res["c3_p300"]
    print("\n【C2 · dose=350 零侧稳定性（主判据）】")
    print(f"  p₃₅₀ = {c2['p']!r}  （{c2['n_pos']}/{c2['n_comp']}，不可判 {c2['n_uncomp']}）")
    print(f"  阈值 p₃₅₀ ≤ {c2['thr_max']} ⇒ {'成立' if c2['holds'] else '不成立'}")

    print("\n【C3 · dose=300 正侧稳定性】")
    print(f"  p₃₀₀ = {c3['p']!r}  （{c3['n_pos']}/{c3['n_comp']}，不可判 {c3['n_uncomp']}）")
    print(f"  阈值 p₃₀₀ ≥ {c3['thr_min']} ⇒ {'成立' if c3['holds'] else '不成立'}")

    print("\n【C4 · û 余量（报告，不参与判决）】")
    for tag, d in (("350 全域", res["c4_u350"]), ("350 k=9", res["c4_u350_k9"]),
                   ("300 全域", res["c4_u300"]), ("300 k=9", res["c4_u300_k9"])):
        print(f"  {tag}: umin={d['umin']!r} umax={d['umax']!r} "
              f"u_maxlow={d['u_maxlow']!r} 余量上={d['margin_above']!r} 余量下={d['margin_below']!r} "
              f"层数={d['n_layers']} 低层={d['n_low_layers']} 夹紧={d['n_clamped']}")

    print("\n【C5 · 逐 k 曲线（报告，不参与判决）】")
    print("   k | p₃₅₀     状态 | p₃₀₀     状态")
    for k in range(1, 10):
        a3, a0 = res["c5_per_k_350"][k], res["c5_per_k_300"][k]
        print(f"  {k:2d} | {a3['p']!r:>20} n={a3['n_comp']:5d} | {a0['p']!r:>20} n={a0['n_comp']:5d}")

    print(f"\n  ★ 总读数 {res['n_rows_total']} 条（标称 {M.NOMINAL_TOTAL}）")
    print("\n" + "=" * 78)
    print(f"★ 判决: {res['verdict']}")
    print("=" * 78)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(M.plainify(res), fh, ensure_ascii=False, indent=1, default=str)
        print(f"  （已写 {a.out}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())