#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-42 · 机制诊断读数（**不进判据**）：从正式轮的分片 JSON 汇总诊断量。

★ 本脚本**只读** `dir42/out/DIR42r10_arm_*_shard*.json` ⇒ ⇒ 不训练、不导入装置。
★ 输出用于裁决书的「机制」段 ⇒ ⇒ **不是判据**（判据只有 `merge` 的判定表）。
"""
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

files = sorted(glob.glob(os.path.join(OUT, "DIR42r10_arm_*_shard*.json")))
arms = {}
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    arms.setdefault(d["arm"], []).extend(d["rows"])
for a in arms:
    arms[a].sort(key=lambda r: r["idx"])

print("=== 每臂诊断均值（400 实例）===")
for a in ("oracle", "rand", "learned"):
    R = arms[a]
    print(f"{a:8s} L={statistics.fmean([r['L'] for r in R]):.6f} "
          f"q_norm={statistics.fmean([r['q_norm'] for r in R]):.4f} "
          f"y_norm={statistics.fmean([r['y_norm'] for r in R]):.4f} "
          f"Wmax={statistics.fmean([r['Wmax'] for r in R]):.4f} "
          f"dWq1={statistics.fmean([r['dWq1'] for r in R]):.4f} "
          f"|cos|={abs(statistics.fmean([r['cos'] for r in R])):.6f}")

Lr = {r["idx"]: r["L"] for r in arms["rand"]}
Ll = {r["idx"]: r["L"] for r in arms["learned"]}
Lo = {r["idx"]: r["L"] for r in arms["oracle"]}
print()
print(f"learned 优于 rand 的实例数 : {sum(1 for i in Lr if Ll[i] < Lr[i])}/400")
print(f"oracle  优于 rand 的实例数 : {sum(1 for i in Lr if Lo[i] < Lr[i])}/400")
print(f"learned 优于 oracle 的实例数: {sum(1 for i in Lr if Ll[i] < Lo[i])}/400")
print(f"learned 的 L > rand 的 L 的中位数比: "
      f"{statistics.median([Ll[i] / Lr[i] for i in Lr]):.4f}")

# 分位数
for a in ("oracle", "rand", "learned"):
    v = sorted(r["L"] for r in arms[a])
    print(f"{a:8s} L 分位: p10={v[40]:.6f} p50={v[200]:.6f} p90={v[360]:.6f}")
