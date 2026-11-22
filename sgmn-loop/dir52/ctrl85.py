#!/opt/pytorch/bin/python
r"""`DEF-52-5` 的**复现对照**（Advisor Gap-1）：证明"现在盘上的装置"能逐位复现原写手的行。

为什么必须先做：补算的 4 个种子是**新进程**跑的，它加载的是**当前盘上**的 `R1_task52.py`，
而原写手跑的是 10 小时前**内存里**的那份代码；若期间代码/环境有漂移，补算行与其余 196 行就不同源。

预注册判据（**先冻结、后运行**）：
  · 拿一个**已恢复**的种子（62048，在快照 009 里）重算到 scratch；
  · **逐字段全等**（`json` 对象 `==`，含浮点逐位）⇒ `PASS`；任一字段不等 ⇒ `FAIL`；
  · 只打印 `PASS/FAIL` 与**差异字段名**，**不打印任何数值**（保持裁决盲态）。
  · `PASS` 才允许把这 4 个补算行并入；`FAIL` ⇒ 不并入，按 196 行如实披露或新开 revision。

用法： /opt/pytorch/bin/python dir52/ctrl85.py
"""
import json
import os
import sys

SNAP = "dir52/out/FDSNAP/009"
CTRL = "dir52/out/CTRL/ctrl_62048.jsonl"
SEED = 62048


def row_from(path, seed):
    if not os.path.isfile(path):
        return None
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if s:
            r = json.loads(s)
            if int(r.get("seed", -1)) == seed:
                return r
    return None


def main():
    ref = row_from(os.path.join(SNAP, "shard0.jsonl"), SEED)   # 原写手那行（fd 抢救）
    new = row_from(CTRL, SEED)                                  # 当前装置重算
    if ref is None or new is None:
        print(f"FAIL 缺数据: ref={ref is not None} new={new is not None}")
        return 1
    if ref == new:
        print(f"PASS seed {SEED}: 当前装置与原写手行逐字段全等（{len(ref)} 个字段）")
        return 0
    keys = sorted(set(ref) | set(new))
    diff = [k for k in keys if ref.get(k) != new.get(k)]
    print(f"FAIL seed {SEED}: {len(diff)} 个字段不等 → {diff}")
    print("  ★ 装置漂移 ⇒ 补算行不得并入；按 196 行处置或新开 revision")
    return 1


if __name__ == "__main__":
    sys.exit(main())