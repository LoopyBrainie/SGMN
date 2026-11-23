"""`DIR-130` `v1.3` 的**守卫验证探针**（`P133-9`）。

★ 本探针**只做验证**：不产生新读数、不改任何冻结文件、不动判据与阈值。

验证三件事（卡片 `v1.3` §6-6 的验收 `V1`/`V2`/`V5`）：

* **`V1`（形式无关）**：6 条被禁形式**全部**被 `Agg` 载体拦下 —— 渲染即抛 `AggPrintBlocked`。
  ★ 关键性质：**拦的是「量」不是「字」** ⇒ 与写法（带空格/无空格/换措辞）**无关**。
* **`V2`（零误伤）**：合法结构输出（**单结构量** + **合成夹具**）在 `precheck` 阶段**照常渲染**。
* **`V5`（分支不可判定）**：由 `v1.3` 的 precheck 输出**推不出**任何判据分支的成立与否；
  并用 `v1.2` 的**归档**输出作**对照**（同一套可行域计算 ⇒ 旧输出**能**判掉 `C3` ⇒ 泄漏是实的）。

用法（**在 trainer 上跑**）：
    /opt/pytorch/bin/python probes/probe130_v13_guard.py --out out/probe130_v13_guard.json
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(DIR)
for _d in (DIR, os.path.join(ROOT, "dir126"), os.path.join(ROOT, "dir127"), ROOT):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import mech130 as M                                              # noqa: E402
import precheck130 as P                                          # noqa: E402

NEW_LOG = os.path.join(DIR, "out", "precheck130_v13.log")
OLD_LOG = os.path.join(DIR, "out", "precheck130.log")


# ------------------------------------------------------------------ V1：被禁形式**全部**被拦
def _render_attempts(q):
    """对一个量尝试**各种渲染写法**；返回 `(成功渲染的写法列表, 是否被拦)`。"""
    rendered = []
    blocked = False
    trials = [
        ("repr", lambda: repr(q)),
        ("str", lambda: str(q)),
        ("format(空)", lambda: format(q, "")),
        ("format(.6f)", lambda: format(q, ".6f")),
        ("f-string 带空格", lambda: f"p₃₅₀ = {q}"),
        ("f-string 无空格", lambda: f"p₃₅₀={q}"),
    ]
    for name, fn in trials:
        try:
            fn()
            rendered.append(name)
        except M.AggPrintBlocked:
            blocked = True
    return rendered, blocked


def v1_forbidden_forms(rows_a, rows_b):
    """6 条被禁形式 ⇒ 每个都必须被载体拦下（且**所有**写法都拦）。"""
    p350 = M.p_flip(rows_b)
    pk350 = M.per_k(rows_b)
    pooled_zero = M.count_over(rows_a + rows_b,
                               lambda r: r["frac"] is not None and r["frac"] == 0.0, "n_zero")
    pooled_none = M.count_over(rows_a + rows_b, lambda r: r["frac"] is None, "n_none")
    forms = [
        ("① 判据终值（带空格写法）", p350["p"]),
        ("② 判据终值（无空格写法）", p350["p"]),
        ("③ 判据的分子（跨读数计数）", p350["n_pos"]),
        ("④ 判据的分母（跨读数计数）", p350["n_comp"]),
        ("⑤ 逐 k 的计数（分层的跨读数聚合）", pk350[3]["n_comp"]),
        ("⑥ 池化计数（`v1.2` 实际泄漏的那两个）", pooled_zero),
    ]
    out = []
    for name, q in forms:
        rendered, blocked = _render_attempts(q)
        out.append({"form": name, "tag": getattr(q, "tag", None),
                    "origin": getattr(q, "origin", None),
                    "blocked": blocked, "rendered": rendered})
    # 第七个：池化的 None 计数（`v1.2` 的 `None=270`）
    rendered, blocked = _render_attempts(pooled_none)
    out.append({"form": "⑦ 池化计数（`v1.2` 的 `None=270`）", "tag": pooled_none.tag,
                "origin": pooled_none.origin, "blocked": blocked, "rendered": rendered})
    return out


# ------------------------------------------------------------------ V2：合法输出零误伤
def v2_legal_outputs():
    r = M.reading_cells(M.DOSE_B, (555,), (0,))
    legal = []
    for key in ("m_tot", "n_low", "n_cells", "N1", "N2", "frac"):
        q = r[key]
        try:
            legal.append({"kind": f"单结构量 · {key}", "value": repr(q), "ok": True})
        except M.AggPrintBlocked:
            legal.append({"kind": f"单结构量 · {key}", "value": None, "ok": False})
    # 合成夹具（`origin="fixture"`）必须**可渲染** —— 自测的期望值靠它
    fx = M.p_flip([{"frac": 0.5, "k": 9, "m_tot": 10, "m_lo": 1, "origin": "fixture"},
                   {"frac": 0.0, "k": 9, "m_tot": 10, "m_lo": 0, "origin": "fixture"}])
    for key in ("p", "n_pos", "n_comp"):
        try:
            legal.append({"kind": f"合成夹具 · {key}", "value": repr(fx[key]), "ok": True})
        except M.AggPrintBlocked:
            legal.append({"kind": f"合成夹具 · {key}", "value": None, "ok": False})
    return legal


# ------------------------------------------------------------------ V5：分支是否可由 precheck 单独判定
def feasible_p_interval(n_zero_pool, n_none_pool, nominal=7665, dose=300):
    """给定**池化**的 `(n_zero, n_none)` 与每剂量标称行数 ⇒ `p(dose)` 的可行区间。

    约束（只用 precheck 里出现过的量）：
      * `n_zero(d) + n_none(d) <= nominal`（每剂量）；
      * `n_zero(300) + n_zero(350) = n_zero_pool`（池化）；
      * `n_none(300) + n_none(350) = n_none_pool`（池化）；
      * `n_zero(d) >= 0`、`n_none(d) >= 0`。
    ⇒ 枚举 `a = n_none(dose)` 的全部整数取值 ⇒ 闭区间。
    """
    ps = []
    for a in range(0, n_none_pool + 1):
        n_none_d = a
        n_none_o = n_none_pool - a
        if n_none_o < 0:
            continue
        lo = max(0, n_zero_pool - (nominal - n_none_o))
        hi = min(nominal - n_none_d, n_zero_pool)
        if lo > hi:
            continue
        n_comp = nominal - n_none_d
        if n_comp <= 0:
            continue
        ps.append(1.0 - lo / float(n_comp))
        ps.append(1.0 - hi / float(n_comp))
    return (min(ps), max(ps)) if ps else None


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def v5_decidability(new_text, old_text):
    """① 新输出：**无**跨读数聚合 ⇒ `p` 无约束 ⇒ 任何分支都不可判定；
    ② 旧输出（`v1.2` 归档）：由 `(零=11092, None=270)` **能**界住 `p₃₀₀` ⇒ `C3` 可判定。"""
    # --- 新输出：先过 `S10`（数字来源审计）
    sid, ok10, d10 = P.s10_number_audit(new_text)
    # 唯一的读数相关信息 = **一条**读数（白名单 (ii)）⇒ 对 `p` 的约束 ≤ 1/nominal
    bound_one = 1.0 / float(M.NOMINAL_PER_DOSE)
    new_iv = [0.0, 1.0 - bound_one]
    new_decides = {
        "C2（p₃₅₀ ≤ 0.05）": not (new_iv[0] <= M.P350_MAX <= new_iv[1]),
        "C3（p₃₀₀ ≥ 0.95）": not (new_iv[0] <= M.P300_MIN <= new_iv[1]),
    }
    # --- 旧输出：从文本里取那两个池化计数
    def _grab(pat):
        m = re.search(pat, old_text)
        return int(m.group(1)) if m else None
    nz = _grab(r"零=(\d+)")
    nn = _grab(r"None=(\d+)")
    old_iv = feasible_p_interval(nz, nn) if (nz is not None and nn is not None) else None
    old_decides = {}
    if old_iv:
        old_decides = {
            "C2（p₃₅₀ ≤ 0.05）": not (old_iv[0] <= M.P350_MAX <= old_iv[1]),
            "C3（p₃₀₀ ≥ 0.95）": not (old_iv[0] <= M.P300_MIN <= old_iv[1]),
        }
    return {"S10": {"ok": ok10, "detail": d10},
            "new": {"interval_p300": new_iv, "decides": new_decides,
                    "any_decidable": any(new_decides.values())},
            "old": {"n_zero_pooled": nz, "n_none_pooled": nn,
                    "interval_p300": old_iv, "decides": old_decides,
                    "any_decidable": any(old_decides.values())}}


# ------------------------------------------------------------------ 主流程
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    # ★ 探针自己跑在 `precheck` 阶段 ⇒ 载体生效（`formal` 下渲染不受限，验证就失去意义）
    M.set_stage("precheck")
    P.M.set_stage("precheck")

    print("=" * 78)
    print("DIR-130 v1.3 守卫验证探针（V1 / V2 / V5）")
    print("=" * 78)

    subs = M.all_subsets()
    rows_a, _ = M.sweep(M.DOSE_A, subs)
    rows_b, _ = M.sweep(M.DOSE_B, subs)
    print(f"  （全域枚举：每剂量 {len(rows_a)} 行，共 {len(rows_a) + len(rows_b)} 行）")

    print("\n【V1 · 6 条被禁形式全部被拦（形式无关）】")
    v1 = v1_forbidden_forms(rows_a, rows_b)
    for it in v1:
        ok = it["blocked"] and not it["rendered"]
        print(f"  {'✓' if ok else '✗'} {it['form']}: tag={it['tag']} "
              f"origin={it['origin']} 被拦={it['blocked']} 渲染成功={it['rendered']}")
    v1_ok = all(it["blocked"] and not it["rendered"] for it in v1)

    print("\n【V2 · 合法结构输出零误伤】")
    v2 = v2_legal_outputs()
    for it in v2:
        print(f"  {'✓' if it['ok'] else '✗'} {it['kind']}: {it['value']!r}")
    v2_ok = all(it["ok"] for it in v2)

    print("\n【V5 · 分支是否可由 precheck 单独判定】")
    new_text = _read(NEW_LOG)
    old_text = _read(OLD_LOG)
    v5 = v5_decidability(new_text, old_text)
    print(f"  {'✓' if v5['S10']['ok'] else '✗'} S10: {v5['S10']['detail']}")
    iv_new = v5["new"]["interval_p300"]
    print(f"  新输出（v1.3）: p₃₀₀ 可行区间 = [{iv_new[0]!r}, {iv_new[1]!r}] "
          f"⇒ 可判定的分支 = {v5['new']['decides']}")
    iv_old = v5["old"]["interval_p300"]
    print(f"  旧输出（v1.2）: 池化 零={v5['old']['n_zero_pooled']} None={v5['old']['n_none_pooled']}"
          f" ⇒ p₃₀₀ 可行区间 = [{iv_old[0]!r}, {iv_old[1]!r}]")
    print(f"               ⇒ 可判定的分支 = {v5['old']['decides']}")
    v5_ok = (v5["S10"]["ok"] and not v5["new"]["any_decidable"] and v5["old"]["any_decidable"])

    res = {"v1": {"ok": v1_ok, "forms": v1},
           "v2": {"ok": v2_ok, "items": v2},
           "v5": dict(v5, ok=v5_ok),
           "ok": bool(v1_ok and v2_ok and v5_ok)}
    print(f"\n  ⇒ V1 {'✓' if v1_ok else '✗'} · V2 {'✓' if v2_ok else '✗'} "
          f"· V5 {'✓' if v5_ok else '✗'}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1, default=str)
        print(f"  （已写 {a.out}）")
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
