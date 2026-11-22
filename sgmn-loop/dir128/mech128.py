#!/usr/bin/env python3
"""`DIR-128` 测量层 —— 「把 `ratio` 的**加权规则轴**与**题集抽样轴**拆开」的读数。

★★ 本模块**不训练、不写 cell、不动任何归档**：全部输入是
   `dir128/out/tensor128.json`（`probe128_dump.py` 由 `mech125/126/127` 机械 dump，
   md5 `6794994174c9218d0a166ce75eea45ce`，**trainer 与本地逐位相同**）。

★ 三条轴（卡片 §1-2 冻结）：
    轴 1 = **加权规则**（未平滑 plug-in）`A0..A4`  ← **唯一判据轴**
    轴 2 = **平滑**（Jeffreys `+0.5`）`A5..A7`      ← 报告
    轴 3 = **功能形式**（logit 层）`L0..L2`         ← 报告
    pair 级率族 `P0..P4`（分母 `n≈9–23`）           ← 报告（分辨率下限）

★ 卡片 §2-2 的解析引理 `D-128-L1`：**全部 `|S|=1` 结构上 `A0..A4` 代数恒等**。
  本模块不"实现"这条引理，而是让它**自然涌现**（各 `A_k` 独立算）——
  `precheck` 的 `S3` 再断言涌现结果逐位相等。
"""
import itertools
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TENSOR = os.path.join(HERE, "out", "tensor128.json")
LOG4 = math.log(4.0)

#: 轴 1 判据族（**冻结**，卡片 §1-3）
AXIS1 = ("A0", "A1", "A2", "A3", "A4")
#: 轴 2 报告族（Jeffreys 平滑）
AXIS2 = ("A5", "A6", "A7")
#: 轴 3 报告族（logit 层）
AXIS3 = ("L0", "L1", "L2")
#: pair 级率族（报告）
PAIRF = ("P0", "P1", "P2", "P3", "P4")

__all__ = ["load_tensor", "dose_view", "all_doses", "all_subsets", "ratio_of",
           "safe_logit", "rules", "tension", "sign_of", "AXIS1", "AXIS2",
           "AXIS3", "PAIRF", "LOG4"]


def load_tensor(path=TENSOR):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dose_view(T, d):
    """把张量里一个剂量规范化成 `{nc, avail, per_rng}`（`rules()` 的唯一入参形状）。"""
    d = str(int(d))
    ent = T["doses"][d]
    return {"nc": int(T["cells_per_dose"][d]),
            "avail": tuple(ent["avail"]),
            "per_rng": ent["per_rng"]}


def all_doses(T):
    return sorted(int(k) for k in T["doses"])


def all_subsets(avail):
    """非空子集，按 `(k, 字典序)`。"""
    av = sorted(avail)
    out = []
    for k in range(1, len(av) + 1):
        out.extend(itertools.combinations(av, k))
    return out


def ratio_of(p1, p2):
    """`ratio = p1(1-p2)/[4 p2(1-p1)]`（与 `mech125.w_hat` 逐字同源）。不可算 ⇒ `None`。"""
    if p1 is None or p2 is None:
        return None
    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        return None
    return (p1 * (1.0 - p2)) / (4.0 * p2 * (1.0 - p1))


def safe_logit(p):
    if p is None or not (0.0 < p < 1.0):
        return None
    return math.log(p / (1.0 - p))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _wmean(vals, wts):
    tot = sum(w for v, w in zip(vals, wts) if v is not None)
    if tot <= 0:
        return None
    return sum(v * w for v, w in zip(vals, wts) if v is not None) / tot


def _counts(dose_t, S):
    """把子集 `S` 上的计数摊平出来（唯一次要的簿记）。"""
    per = dose_t["per_rng"]
    nc = dose_t["nc"]
    n1 = sum(per[r]["n1"] for r in S)
    n2 = sum(per[r]["n2"] for r in S)
    e1c = [sum(per[r]["e1"][c] for r in S) for c in range(nc)]
    e2c = [sum(per[r]["e2"][c] for r in S) for c in range(nc)]
    return per, nc, n1, n2, e1c, e2c


def rules(dose_t, S):
    """返回 `{规则名: ratio 或 None}`。**全部规则在同一个 `p` 估计量上**（卡片 §1-1）。

    ★ `A0..A4` 是**判据族**；`A5..A7`/`L*`/`P*` **只报不判**。
    """
    per, nc, n1, n2, e1c, e2c = _counts(dose_t, S)
    if n1 <= 0 or n2 <= 0:
        return {k: None for k in AXIS1 + AXIS2 + AXIS3 + PAIRF}

    # 逐 rng 的池化率
    P1r = [sum(per[r]["e1"]) / (nc * per[r]["n1"]) if per[r]["n1"] else None for r in S]
    P2r = [sum(per[r]["e2"]) / (nc * per[r]["n2"]) if per[r]["n2"] else None for r in S]
    # 逐 cell 的率（分母 N 与 cell 无关）
    C1 = [c / float(n1) for c in e1c]
    C2 = [c / float(n2) for c in e2c]
    # pair 级率
    PA1 = [per[r]["e1"][c] / float(per[r]["n1"]) for r in S for c in range(nc)]
    PA2 = [per[r]["e2"][c] / float(per[r]["n2"]) for r in S for c in range(nc)]
    w1 = [per[r]["n1"] for r in S]
    w2 = [per[r]["n2"] for r in S]

    o = {}
    # ---------------- 轴 1（判据）
    o["A0"] = ratio_of(sum(e1c) / float(nc * n1), sum(e2c) / float(nc * n2))
    o["A1"] = ratio_of(_mean(P1r), _mean(P2r))
    o["A2"] = ratio_of(_mean(C1), _mean(C2))
    o["A3"] = ratio_of(_wmean(P1r, w2), _wmean(P2r, w2))
    o["A4"] = ratio_of(_wmean(P1r, w1), _wmean(P2r, w1))
    # ---------------- 轴 2（平滑，只报）
    o["A5"] = ratio_of(_mean([(c + 0.5) / (n1 + 1.0) for c in e1c]),
                       _mean([(c + 0.5) / (n2 + 1.0) for c in e2c]))
    o["A6"] = ratio_of(_mean([(sum(per[r]["e1"]) + 0.5) / (nc * per[r]["n1"] + 1.0) for r in S]),
                       _mean([(sum(per[r]["e2"]) + 0.5) / (nc * per[r]["n2"] + 1.0) for r in S]))
    o["A7"] = ratio_of((sum(e1c) + 0.5) / (nc * n1 + 1.0), (sum(e2c) + 0.5) / (nc * n2 + 1.0))
    # ---------------- 轴 3（logit 层，只报）★ 与 `ratio = exp(G)/4` 对应
    def _from_gap(g):
        return None if g is None else math.exp(g) / 4.0
    o["L0"] = _from_gap(_mean([safe_logit(C1[c]) - safe_logit(C2[c])
                               for c in range(nc)
                               if safe_logit(C1[c]) is not None and safe_logit(C2[c]) is not None]))
    o["L1"] = _from_gap(_mean([safe_logit(a) - safe_logit(b)
                               for a, b in zip(P1r, P2r)
                               if safe_logit(a) is not None and safe_logit(b) is not None]))
    o["L2"] = _from_gap(_mean([safe_logit(a) - safe_logit(b)
                               for a, b in zip(PA1, PA2)
                               if safe_logit(a) is not None and safe_logit(b) is not None]))
    # ---------------- pair 级率族（只报；分辨率下限）
    def _med(xs):
        xs = sorted(x for x in xs if x is not None)
        if not xs:
            return None
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    o["P0"] = ratio_of(_mean(PA1), _mean(PA2))
    o["P1"] = ratio_of(_med(PA1), _med(PA2))
    h1 = (len([x for x in PA1 if x > 0]) / sum(1.0 / x for x in PA1 if x > 0)) if any(x > 0 for x in PA1) else None
    h2 = (len([x for x in PA2 if x > 0]) / sum(1.0 / x for x in PA2 if x > 0)) if any(x > 0 for x in PA2) else None
    o["P2"] = ratio_of(h1, h2)
    pr = [ratio_of(a, b) for a, b in zip(PA1, PA2)]
    o["P3"] = _med(pr)
    o["P4"] = _med([ratio_of(a, b) for a, b in zip(P1r, P2r)])
    return o


def tension(dose_t, S, names):
    """给定规则族，返回 `(max/min, argmax, argmin, 可用条数)`。不可判 ⇒ `(None, None, None, n)`。"""
    o = rules(dose_t, S)
    got = [(o[n], n) for n in names if o.get(n) is not None and o[n] > 0]
    if len(got) < 2:
        return None, None, None, len(got)
    hi = max(got)
    lo = min(got)
    return hi[0] / lo[0], hi[1], lo[1], len(got)


def sign_of(x):
    """`+1` / `-1` / `0`（`x==1.0` ⇒ `0`，卡片 §2-4 的 `E1`）。"""
    if x is None:
        return None
    if x == 1.0:
        return 0
    return 1 if x > 1.0 else -1