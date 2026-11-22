#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-112 · 判据的**纯逻辑**自测（不训练、不建模型栈，只调纯函数）。

对应卡片 `A112-6` / `A112-7`。★ 教训 `DEF-111-02`/`DEF-111-03`：
夹具必须**累加**而非赋值、**不得**用 `dict(前一场景)` 继承，否则残留项会让期望值算错。
本文件的每个场景都**从零构造**。

用法：/opt/pytorch/bin/python test_sign112.py
"""
import sys

import R1_task112 as T

FAIL = []
G = T.GAMMA112
L8 = T.L8
METS = T.METS_V


def mk(per_seed, xnone=0.5):
    """由目标 δ 反解合成 cells：X_NONE=xnone，X_arm = xnone + δ·(1−xnone)。

    ⇒ 这样 δ 是**精确**的，判据的任何偏差都会暴露。
    ★ 必须逐 (arm, seed) **累加**进同一个 state 字典（`DEF-111-02`）。
    """
    acc = {}
    for sd, vals in per_seed.items():
        for arm, dv in vals.items():
            k = f"{arm}|{G}|{sd}"
            node = acc.setdefault(k, {L8: {"m": {}}})
            for mt in METS:
                node[L8]["m"][mt] = xnone + dv[mt] * (1.0 - xnone)
    return {k: {"state": v} for k, v in acc.items()}


def tri(dq, do, db):
    """同一 δ 同时用于两个量（除非另行覆盖）。"""
    return {"T-NONE": {"rank_eff": 0.0, "cos_mean": 0.0},
            "T-Q": {"rank_eff": dq, "cos_mean": dq},
            "T-O": {"rank_eff": do, "cos_mean": do},
            "T-BOTH": {"rank_eff": db, "cos_mean": db}}


def check(name, cells, seeds, want):
    code, per = T.sign_verdict(cells, seeds)
    ok = code == want
    print(f"  {'PASS' if ok else '**FAIL**'}  {name}  => {code} (want {want})")
    if not ok:
        FAIL.append((name, code, want))
    return code, per


SS = list(range(9001, 9009))

print("候选 seed 数 =", len(SS), " MIN_NEG =", T.MIN_NEG)

# 1) 8/8 负 ⇒ CONFIRMED
c = mk({sd: tri(0.2, 0.3, 0.05) for sd in SS})
check("场景1 8/8 负→CONFIRMED", c, SS, "C112-ANTAGONISM-CONFIRMED")

# 2) 7/8 负 ⇒ CONFIRMED（边界内侧，留一格余量）
d = {sd: tri(0.2, 0.3, 0.05) for sd in SS}
d[SS[0]] = tri(0.2, 0.05, 0.5)          # 该 seed 的 δ_B 高于 max ⇒ Δ>0
check("场景2 7/8 负→CONFIRMED", mk(d), SS, "C112-ANTAGONISM-CONFIRMED")

# 3) 6/8 负 ⇒ FALSIFIED（边界外侧）
d = {sd: tri(0.2, 0.3, 0.05) for sd in SS}
for sd in SS[:2]:
    d[sd] = tri(0.2, 0.05, 0.5)
check("场景3 6/8 负→FALSIFIED", mk(d), SS, "C112-FALSIFIED")

# 4) 全正 ⇒ FALSIFIED
c = mk({sd: tri(0.2, 0.1, 0.5) for sd in SS})
check("场景4 8/8 正→FALSIFIED", c, SS, "C112-FALSIFIED")

# 5) Δ 恰为 0 不算负 ⇒ FALSIFIED
c = mk({sd: tri(0.3, 0.3, 0.3) for sd in SS})
code, per = check("场景5 Δ=0 不算负→FALSIFIED", c, SS, "C112-FALSIFIED")
if per["rank_eff"]["n_neg"] != 0:
    FAIL.append(("场景5 n_neg", per["rank_eff"]["n_neg"], 0))

# 6) 缺一格 ⇒ INSUFFICIENT
c = mk({sd: tri(0.2, 0.3, 0.05) for sd in SS})
del c[f"T-BOTH|{G}|{SS[0]}"]
check("场景6 缺一格→INSUFFICIENT", c, SS, "C112-INSUFFICIENT")

# 7) 分母退化（X_NONE=1.0）⇒ 该 seed 不可用 ⇒ INSUFFICIENT
c = mk({sd: tri(0.2, 0.3, 0.05) for sd in SS})
c[f"T-NONE|{G}|{SS[0]}"]["state"][L8]["m"]["rank_eff"] = 1.0
check("场景7 分母退化→INSUFFICIENT", c, SS, "C112-INSUFFICIENT")

# 8) 单量成立 ⇒ PARTIAL（rank_eff 全负、cos_mean 全正）
c = mk({sd: tri(0.2, 0.3, 0.05) for sd in SS})
xn = 0.5
for sd in SS:
    for arm, dv in (("T-Q", 0.2), ("T-O", 0.1), ("T-BOTH", 0.5)):
        c[f"{arm}|{G}|{sd}"]["state"][L8]["m"]["cos_mean"] = xn + dv * (1 - xn)
check("场景8 单量→PARTIAL", c, SS, "C112-PARTIAL")

# 9) Δ 恒等式（A112-7）
print("\n  --- A112-7 Δ 恒等式 ---")
bad = T.delta_selftest()
print(f"  {'PASS' if not bad else '**FAIL**'}  Δ 恒等式 6 个算例；失败 {bad}")
if bad:
    FAIL.append(("A112-7", bad, None))

# 10) delta_of 的精确性：δ 反解后必须原值返回
print("\n  --- delta_of 精确性 ---")
c = mk({SS[0]: tri(0.234, 0.567, 0.111)})
for mt in METS:
    got = T.delta_of(c, SS[0], mt)
    want = (0.234, 0.567, 0.111)
    ok = got is not None and all(abs(g - w) < 1e-12 for g, w in zip((got[1], got[2], got[3]), want))
    print(f"  {'PASS' if ok else '**FAIL**'}  {mt}: δ_Q/δ_O/δ_B 回读 {None if got is None else tuple(round(x,12) for x in got[1:])} (want {want})")
    if not ok:
        FAIL.append((f"delta_of {mt}", got, want))
    # Δ 也要对
    ok2 = abs(got[0] - (0.111 - 0.567)) < 1e-12
    print(f"  {'PASS' if ok2 else '**FAIL**'}  {mt}: Δ = {got[0]:+.12f} (want {0.111-0.567:+.12f})")
    if not ok2:
        FAIL.append((f"Δ {mt}", got[0], 0.111 - 0.567))

# 11) max 选对边：δ_Q 更大时 Δ 用 δ_Q
c = mk({SS[0]: tri(0.80, 0.10, 0.50)})
got = T.delta_of(c, SS[0], "rank_eff")
ok = abs(got[0] - (0.50 - 0.80)) < 1e-12
print(f"\n  {'PASS' if ok else '**FAIL**'}  max 选 δ_Q: Δ = {got[0]:+.12f} (want {0.50-0.80:+.12f})")
if not ok:
    FAIL.append(("max 选 δ_Q", got[0], 0.50 - 0.80))

print()
if FAIL:
    print(f"★ 失败 {len(FAIL)} 项：")
    for f in FAIL[:10]:
        print("   ", f)
    sys.exit(1)
print("★ 全部通过")