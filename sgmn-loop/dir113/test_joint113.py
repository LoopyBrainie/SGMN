#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-113 · 判据的**纯逻辑**自测（不训练、不建模型栈，只调纯函数）。

对应卡片 `A113-9`。★ 教训 `DEF-111-02`/`DEF-111-03`：夹具必须**累加**而非赋值、
**不得**用 `dict(前一场景)` 继承，否则残留项会让期望值算错。本文件每个场景**从零构造**。
★ 教训 `DEF-112-05`：自测的**期望值**本身要手算核对（上轮我把 `max(-0.1,-0.2)` 写错）。
★ 教训 `DEF-112-07`：自测必须能**变红** —— 本文件末尾有一段「注入已知错误值」的活性验证。

用法：/opt/pytorch/bin/python test_joint113.py
"""
import sys

import R1_task113 as T

FAIL = []
G = T.GAMMA113
L8 = T.L8
MT = T.METS_V[0]          # 只 rank_eff
N = T.N_ITEM


def mk(per_seed, acc=None, nitem=N):
    """由目标 δ 反解合成 cells：X_NONE=0.5，X_arm = 0.5 + δ·(1−0.5) ⇒ δ **精确**。

    `acc[seed] = {"T-Q":.., "T-O":.., "T-BOTH":..}`；不给则该 seed 的 acc 全为 0.29。
    """
    cells = {}
    xn = 0.5
    for sd, (dq, do, db) in per_seed.items():
        a = (acc or {}).get(sd) or {}
        cells[f"T-NONE|{G}|{sd}"] = {
            "state": {L8: {"m": {MT: xn}}},
            "acc_item": 0.29, "item_ok": [0] * nitem, "src_stage": "stageA"}
        for arm, dv in (("T-Q", dq), ("T-O", do), ("T-BOTH", db)):
            ai = a.get(arm, 0.29)
            cells[f"{arm}|{G}|{sd}"] = {
                "state": {L8: {"m": {MT: xn + dv * (1.0 - xn)}}},
                "acc_item": ai,
                "item_ok": [1 if i < round(ai * nitem) else 0 for i in range(nitem)],
                "src_stage": "stageA"}
    return cells


def expect(name, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{name}: got={got} want={want}")
    if not ok:
        FAIL.append((name, got, want))


SS = list(range(9201, 9213))            # 12 个合成 seed

# ============================ 1) 五码判决（卡片 §1.4） ============================
print("\n=== 1) 判决码 5 分支 + 边界 ===")
NEG_A = {sd: (0.2, 0.3, 0.05) for sd in SS}          # Δ<0 全 12
POS_A = {sd: (0.2, 0.05, 0.5) for sd in SS}          # Δ>0 全 12
# ★ acc 必须是 1/48 的整数倍（真实 acc_item = k/48），且 **d 必须逐 seed 有变异**
#   （若 12 个 d 完全相同 ⇒ SE=0 ⇒ t 未定义；见 `DEF-113-01`）
_KQ, _KO, _KB = 10, 9, 1                     # /48 ⇒ 0.2083 / 0.1875 / 0.0208 ⇒ d≈-0.1875
ACC_HARM = {sd: {"T-Q": (_KQ + (i % 3)) / N, "T-O": (_KO + (i % 2)) / N,
                 "T-BOTH": (_KB + (i % 4)) / N} for i, sd in enumerate(SS)}
ACC_SAME = {sd: {"T-Q": (_KQ + (i % 3)) / N, "T-O": (_KO + (i % 2)) / N,
                 "T-BOTH": (14 + (i % 4)) / N} for i, sd in enumerate(SS)}
MIX_A9 = dict(NEG_A); [MIX_A9.__setitem__(sd, POS_A[sd]) for sd in SS[9:]]   # 9 负
MIX_A8 = dict(NEG_A); [MIX_A8.__setitem__(sd, POS_A[sd]) for sd in SS[8:]]   # 8 负

expect("A且B ⇒ COEXISTS", T.joint_verdict(mk(NEG_A, ACC_HARM), SS)[0],
       "C113-ANTAGONISM-COEXISTS-WITH-TASK-HARM")
expect("A成立B不成立 ⇒ NO-TASK-HARM", T.joint_verdict(mk(NEG_A, ACC_SAME), SS)[0],
       "C113-NO-TASK-HARM-DETECTED")
expect("A不成立B成立 ⇒ WITHOUT-STRUCT", T.joint_verdict(mk(MIX_A8, ACC_HARM), SS)[0],
       "C113-TASK-HARM-WITHOUT-STRUCT-REPLICATION")
expect("都不成立 ⇒ FALSIFIED", T.joint_verdict(mk(MIX_A8, ACC_SAME), SS)[0], "C113-FALSIFIED")
expect("恰 9/12 负 ⇒ A 成立（边界内侧）", T.joint_verdict(mk(MIX_A9, ACC_HARM), SS)[0],
       "C113-ANTAGONISM-COEXISTS-WITH-TASK-HARM")

c = mk(NEG_A, ACC_HARM); del c[f"T-BOTH|{G}|{SS[0]}"]
expect("缺一格 ⇒ INSUFFICIENT", T.joint_verdict(c, SS)[0], "C113-INSUFFICIENT")

EQ_A = {sd: (0.3, 0.3, 0.3) for sd in SS}      # Δ=0，不算负
expect("Δ=0 不算负（0 负）", T.joint_verdict(mk(EQ_A, ACC_HARM), SS)[0],
       "C113-TASK-HARM-WITHOUT-STRUCT-REPLICATION")

# ============================ 2) Δ 恒等式（手算期望） ============================
print("\n=== 2) Δ = δ_B − max(δ_Q,δ_O) 恒等式（期望逐条手算） ===")
# (dq, do, db) -> Δ ；★ 手算：max 取**较不负**的一侧
for (dq, do, db), want in [((0.2, 0.2, 0.1), -0.1),    # 0.1-0.2
                           ((0.2, 0.2, 0.3), +0.1),    # 0.3-0.2
                           ((0.1, 0.5, 0.4), -0.1),    # 0.4-0.5（max=δO）
                           ((0.5, 0.1, 0.4), -0.1),    # 0.4-0.5（max=δQ）
                           ((0.3, 0.7, 0.7), +0.0),    # 0.7-0.7
                           ((-0.1, -0.2, -0.3), -0.2)]:  # max(-0.1,-0.2)=-0.1 ⇒ -0.3+0.1
    got = db - max(dq, do)
    expect(f"Δ(dQ={dq},dO={do},dB={db})", round(got, 12), round(want, 12))

# ============================ 3) 簇级统计数值 ============================
print("\n=== 3) cluster_stats 数值（与手算对照） ===")
vals = [(5/N, 10/N, 2/N), (5/N, 10/N, 4/N), (5/N, 10/N, 6/N),
        (5/N, 10/N, 3/N), (5/N, 10/N, 5/N), (5/N, 10/N, 7/N)]
ss6 = SS[:6]
acc6 = {sd: {"T-Q": aq, "T-O": ao, "T-BOTH": ab} for sd, (aq, ao, ab) in zip(ss6, vals)}
c6 = mk({sd: (0.2, 0.3, 0.05) for sd in ss6}, acc6)
st = T.cluster_stats(c6, ss6)
d_manual = [ab - max(aq, ao) for aq, ao, ab in vals]        # [-0.15,-0.12,-0.08,-0.14,-0.11,-0.09]
expect("d 逐 seed 手算", [round(r["d"], 10) for r in st["rows"]], [round(x, 10) for x in d_manual])
import math as _m
mean_m = sum(d_manual) / len(d_manual)
sd_m = _m.sqrt(sum((x - mean_m) ** 2 for x in d_manual) / (len(d_manual) - 1))
t_m = mean_m / (sd_m / _m.sqrt(len(d_manual)))
expect("mean_d", round(st["mean_d"], 10), round(mean_m, 10))
expect("sd_d", round(st["sd_d"], 10), round(sd_m, 10))
expect("df", st["df"], len(d_manual) - 1)
expect("t", round(st["t"], 9), round(t_m, 9))
# p 单侧：H1 mean<0，t 为负 ⇒ p 应 < 0.5
ok = st["p_one_sided"] is not None and st["p_one_sided"] < 0.5
expect("p 单侧 < 0.5（t<0）", ok, True)

# d 恒为 1/48 整数倍
ok = all(abs(r["d"] * N - round(r["d"] * N)) < 1e-9 for r in st["rows"] if r["ok"])
expect("d 恒为 1/48 整数倍", ok, True)

print("\n=== 3b) DEF-113-01：SE=0 时退回精确符号检验（不得静默失败） ===")
acc_flat = {sd: {"T-Q": 10/N, "T-O": 9/N, "T-BOTH": 1/N} for sd in SS}   # d 全同 ⇒ SE=0
stf = T.cluster_stats(mk(NEG_A, acc_flat), SS)
expect("SE=0 ⇒ se_d == 0", stf["se_d"], 0.0)
expect("SE=0 ⇒ t 为 None（未定义，不伪装）", stf["t"], None)
expect("SE=0 ⇒ p 有值（符号检验退路）", stf["p_one_sided"] is not None, True)
expect("SE=0 且 d 全负 ⇒ p 很小", (stf["p_one_sided"] or 1.0) < 0.01, True)
expect("SE=0 ⇒ src 标明退路", "sign-exact" in (stf["src"] or ""), True)

# ============================ 4) 只 rank_eff，不重复计数 ============================
print("\n=== 4) 判据只用 rank_eff（DEF-112-04 不重复计数） ===")
expect("METS_V 长度 = 1", len(T.METS_V), 1)
expect("METS_V 内容 = rank_eff", list(T.METS_V), ["rank_eff"])
# 即便 cos_mean 全正，也不影响判据
c = mk(NEG_A, ACC_HARM)
for sd in SS:
    for arm in ("T-NONE", "T-Q", "T-O", "T-BOTH"):
        c[f"{arm}|{G}|{sd}"]["state"][L8]["m"]["cos_mean"] = 0.99
expect("cos_mean 全正不影响判决", T.joint_verdict(c, SS)[0],
       "C113-ANTAGONISM-COEXISTS-WITH-TASK-HARM")

# ============================ 5) ★ 活性验证（DEF-112-07 的教训） ============================
print("\n=== 5) 活性验证：注入已知错误 ⇒ 必须变红 ===")
# (a) 把 B 的效应反向（BOTH 最高）⇒ 必须落到 NO-TASK-HARM（不是 COEXISTS）
c_bad = mk(NEG_A, {sd: {"T-Q": 0.02, "T-O": 0.03, "T-BOTH": 0.30} for sd in SS})
got = T.joint_verdict(c_bad, SS)[0]
expect("BOTH 最高 ⇒ 不得判 COEXISTS", got != "C113-ANTAGONISM-COEXISTS-WITH-TASK-HARM", True)
expect("BOTH 最高 ⇒ NO-TASK-HARM", got, "C113-NO-TASK-HARM-DETECTED")
# (b) 结构面反向 ⇒ 不得判 COEXISTS
c_bad2 = mk(POS_A, ACC_HARM)
got2 = T.joint_verdict(c_bad2, SS)[0]
expect("结构面全正 ⇒ 不得判 COEXISTS", got2 != "C113-ANTAGONISM-COEXISTS-WITH-TASK-HARM", True)
# (c) 手算期望值自检：确认 max(-0.1,-0.2) = -0.1（上轮写错过）
expect("max(-0.1,-0.2) 手算", max(-0.1, -0.2), -0.1)
expect("Δ(全负算例) 手算", round(-0.3 - max(-0.1, -0.2), 12), -0.2)

# ============================ 汇总 ============================
print("\n" + "=" * 60)
if FAIL:
    print(f"★ {len(FAIL)} 项失败：")
    for n, g, w in FAIL:
        print(f"   - {n}: got={g} want={w}")
    sys.exit(1)
print("★ 全部通过")
sys.exit(0)