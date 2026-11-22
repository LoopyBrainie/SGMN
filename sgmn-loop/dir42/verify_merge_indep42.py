#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIR-42 · ★ **独立重算**（**不经 `merge()`**）：从 12 个分片 JSON 直接重算判定表

★ **为什么**：卡 §3/§4 的判定逻辑**只有一份**（在 `merge()` 里）⇒ ⇒ 若只有 `merge()`
  一个执行者，则「判定逻辑写错了」这件事**查不出来**（自证）⇒ ⇒ 必须有**第二份实现**
  从**原始读数**重算 ⇒ ⇒ 两份实现**独立地**给出同一结局 ⇒ ⇒ 判定才可信。

★ 本脚本：**不导入装置、不训练、不 import `R1_task42`** ⇒ ⇒ 完全独立于被测实现
  ⇒ ⇒ 只吃**分片 JSON 的读数** ⇒ ⇒ 复算判据。

判定表（卡 §4，冻结）：
  第 1 支  缺臂 / n 不符                     ⇒ `INCOMPLETE`
  第 2 支  任一损失或 se 非有限               ⇒ `DIVERGED`
  第 3 支  `Δ_out` 的 mean < 3se              ⇒ `NO-OUT-OF-SAMPLE-GAIN`
  第 4 支  `C` 的 mean < 3se                 ⇒ `NOT-CONTENT-DEPENDENT`
  第 5 支  `Δ_out` 的 mean < 0.20·R          ⇒ `BELOW-EFFECT-FLOOR`
  第 6 支  以上都过                          ⇒ `PASS_42`（`counted = True`）

用法（在 trainer 或本地均可跑；只读 JSON）：  python3 dir42/verify_merge_indep42.py
"""
import glob
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SIG_K = 3.0
FRAC_HEADROOM = 0.20


def main():
    fails = []

    def check(cond, msg):
        print(("  ✓ " if cond else "  ✗ ") + msg)
        if not cond:
            fails.append(msg)

    # ── 读 12 个分片 JSON（**不用** `merge()`，也**不**读 `.jsonl`）
    files = sorted(glob.glob(os.path.join(OUT, "DIR42r10_arm_*_shard*.json")))
    check(len(files) == 12, f"分片 JSON = 12 个（实测 {len(files)}）")
    if len(files) != 12:
        print("★ 独立重算 **中止**（分片不齐）")
        return 1

    arms = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        arms.setdefault(d["arm"], []).extend(d["rows"])
    for a in arms:
        arms[a].sort(key=lambda r: r["idx"])
    n = {a: len(v) for a, v in arms.items()}
    check(all(v == 400 for v in n.values()), f"每臂 400 个实例（实测 {n}）")

    # ── 与增量落盘的 `.jsonl` 交叉核对（落盘 ⇒ 分片 JSON ⇒ 同一批读数）
    for a, v in arms.items():
        jl = os.path.join(OUT, f"DIR42r10_arm_{a}_shard0.jsonl")
        if not os.path.exists(jl):
            continue
        rows = [json.loads(l) for l in open(jl, encoding="utf-8") if l.strip()]
        check(len(rows) == len([r for r in v if r["idx"] % 4 == 0]),
              f"{a} shard0 的 `.jsonl` 行数 == 分片 JSON 中的 shard0 行数（{len(rows)}）")

    Lo = [r["L"] for r in arms["oracle"]]
    Lr = [r["L"] for r in arms["rand"]]
    Ll = [r["L"] for r in arms["learned"]]
    Lrc = [r["L_rc"] for r in arms["learned"] if "L_rc" in r]
    Lseen = [r["L_seen"] for r in arms["learned"]]
    Lseen_r = [r["L_seen"] for r in arms["rand"]]
    Lseen_o = [r["L_seen"] for r in arms["oracle"]]
    Lfull = [r["L_full"] for r in arms["learned"]]
    cos = {a: statistics.fmean([r["cos"] for r in arms[a]]) for a in arms}

    def paired(a, b):
        d = [x - y for x, y in zip(a, b)]
        m = statistics.fmean(d)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        se = sd / len(d) ** 0.5
        return {"mean": m, "sd": sd, "se": se, "3se": SIG_K * se, "n": len(d)}

    R = statistics.fmean([a - b for a, b in zip(Lr, Lo)])
    pd = paired(Lr, Ll)                       # Δ_out
    pc = paired(Lrc, Ll)                      # 内容性 C
    pd_in = paired(Lseen_r, Lseen)            # 已见窗 Δ
    floor = FRAC_HEADROOM * R
    finite = all(math.isfinite(x) for x in
                 [R, pd["mean"], pd["se"], pc["mean"], pc["se"], floor])

    # ── 按**冻结**的六支判定
    if not all(len(v) == 400 for v in arms.values()):
        outcome = "INCOMPLETE"
    elif not finite:
        outcome = "DIVERGED"
    elif pd["mean"] < pd["3se"]:
        outcome = "NO-OUT-OF-SAMPLE-GAIN"
    elif pc["mean"] < pc["3se"]:
        outcome = "NOT-CONTENT-DEPENDENT"
    elif pd["mean"] < floor:
        outcome = "BELOW-EFFECT-FLOOR"
    else:
        outcome = "PASS_42"

    print("\n★ 独立重算的读数（**不经 `merge()`**）")
    print(f"  L_oracle = {statistics.fmean(Lo):.6f}")
    print(f"  L_rand   = {statistics.fmean(Lr):.6f}")
    print(f"  L_learned= {statistics.fmean(Ll):.6f}")
    print(f"  L_learned_rc = {statistics.fmean(Lrc):.6f}")
    print(f"  R = {R:.6f}   地板 0.20R = {floor:.6f}")
    print(f"  Δ_out  mean = {pd['mean']:.6f}  3se = {pd['3se']:.6f}  (n={pd['n']})")
    print(f"  C      mean = {pc['mean']:.6f}  3se = {pc['3se']:.6f}")
    print(f"  Δ_seen mean = {pd_in['mean']:.6f}  3se = {pd_in['3se']:.6f}")
    print(f"  L_seen: oracle={statistics.fmean(Lseen_o):.6f} "
          f"rand={statistics.fmean(Lseen_r):.6f} learned={statistics.fmean(Lseen):.6f}")
    print(f"  L_full(learned) = {statistics.fmean(Lfull):.6f}")
    print(f"  cos: {json.dumps({k: round(v, 6) for k, v in cos.items()})}")
    print(f"  ⇒ 独立重算结局 = **{outcome}**")

    # ── 与 `merge()` 的裁决**逐项**比对
    mf = os.path.join(OUT, "DIR42r10_formal.json")
    if os.path.exists(mf):
        m = json.load(open(mf, encoding="utf-8"))
        print("\n★ 与 `merge()` 的裁决比对")
        check(m.get("outcome") == outcome,
              f"结局一致：`merge()` = {m.get('outcome')} vs 独立重算 = {outcome}")
        for k, v in (("R", R), ("floor_0.2R", floor)):
            check(abs(m.get(k, float('nan')) - v) < 1e-12,
                  f"{k} 一致：`merge()` = {m.get(k)} vs 独立 = {v}")
        for k, mine in (("delta_out", pd), ("content_C", pc), ("delta_in_sample", pd_in)):
            theirs = m.get(k, {})
            same = all(abs(theirs.get(kk, float('nan')) - vv) < 1e-12
                       for kk, vv in mine.items())
            check(same, f"{k} 逐项一致（mean/3se/n …）")
        check(m.get("counted") is (outcome == "PASS_42"),
              f"`counted` 一致：`merge()` = {m.get('counted')}，"
              f"独立重算 = {outcome == 'PASS_42'}")
    else:
        check(False, f"找不到 `{mf}`（`merge()` 还没跑？）")

    print("\n★ 独立重算 " + ("与 `merge()` 完全一致" if not fails else "**与 `merge()` 不一致**"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
