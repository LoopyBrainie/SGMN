#!/usr/bin/env python3
"""`DIR-128` 前置探针（**开卡前**）—— 候选「合并不变量」电池。

每个候选都在同一张 `tensor128.json` 上纯算术算出，口径差别只在
① **聚合层级**（pair / cell / rng / pool）② **权重**（`N` 加权 / 等权）。
判据量统一取 **`log(ratio) - log 4`** 的符号（`ratio>1 ⟺ >0`），
其中 `ratio = p1(1-p2)/[4 p2(1-p1)]`（与 `mech125.w_hat` 逐字同源）。
"""
import itertools
import json
import math

T = json.load(open("dir128/out/tensor128.json"))
D = T["doses"]
DOSES = sorted(int(k) for k in D)
LOG4 = math.log(4.0)


def safe_logit(p):
    if p is None or p <= 0.0 or p >= 1.0:
        return None
    return math.log(p / (1.0 - p))


def ratio_of(p1, p2):
    if p1 is None or p2 is None or p1 <= 0 or p1 >= 1 or p2 <= 0 or p2 >= 1:
        return None
    return (p1 * (1.0 - p2)) / (4.0 * p2 * (1.0 - p1))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def subsets(d):
    av = D[str(d)]["avail"]
    out = []
    for k in range(1, len(av) + 1):
        out.extend(tuple(sorted(c)) for c in itertools.combinations(av, k))
    return out


def cells(d):
    return range(T["cells_per_dose"][str(d)])


def K(d, S):
    """返回该 (dose, 子集) 上全部候选量的 `log(ratio)-log4`（或 None）。"""
    per = D[str(d)]["per_rng"]
    n1S = sum(per[r]["n1"] for r in S)
    n2S = sum(per[r]["n2"] for r in S)
    CL = list(cells(d))
    e1S = {c: sum(per[r]["e1"][c] for r in S) for c in CL}
    e2S = {c: sum(per[r]["e2"][c] for r in S) for c in CL}
    eps1, eps2 = sum(e1S.values()), sum(e2S.values())

    o = {}
    # --- K0 基线：N 加权池化（= mech125.w_hat 逐字）
    o["K0_ratio_pool"] = ratio_of(eps1 / (len(CL) * n1S), eps2 / (len(CL) * n2S))
    # --- K1 rng 等权（先算逐 rng 池化率，再等权平均）
    P1r = [sum(per[r]["e1"]) / (len(CL) * per[r]["n1"]) for r in S]
    P2r = [sum(per[r]["e2"]) / (len(CL) * per[r]["n2"]) for r in S]
    o["K1_ratio_rng_eq"] = ratio_of(mean(P1r), mean(P2r))
    # --- K2 pair 等权（9×|S| 个 (cell,rng) 全等权）
    pr = [(per[r]["e1"][c] / per[r]["n1"], per[r]["e2"][c] / per[r]["n2"])
          for r in S for c in CL]
    o["K2_ratio_pair_eq"] = ratio_of(mean(x for x, _ in pr), mean(y for _, y in pr))
    # --- K3 cell 等权 logit 差（层=cell，N 用 S 的和）
    g = [x - y for x, y in ((safe_logit(e1S[c] / (n1S * 1.0)), safe_logit(e2S[c] / (n2S * 1.0)))
                            for c in CL) if x is not None and y is not None]
    o["K3_gap_cell_eq"] = (math.exp(mean(g)) / 4.0) if g else None
    # --- K4 pair 等权 logit 差
    g2 = [a - b for a, b in ((safe_logit(x), safe_logit(y)) for x, y in pr)
          if a is not None and b is not None]
    o["K4_gap_pair_eq"] = (math.exp(mean(g2)) / 4.0) if g2 else None
    # --- K5 pair 投票（>log4 的比例；换算成「等效 ratio」便于同尺度比较）
    o["K5_vote_pair"] = None
    if g2:
        o["K5_vote_pair"] = 1.0 + 3.0 * (sum(1 for x in g2 if x > LOG4) / len(g2))
    # --- K6 逐 rng ratio 的中位数
    o["K6_ratio_rng_med"] = median([ratio_of(a, b) for a, b in zip(P1r, P2r)])
    # --- K7 逐 rng ratio 的几何平均
    rr = [ratio_of(a, b) for a, b in zip(P1r, P2r)]
    rr = [x for x in rr if x and x > 0]
    o["K7_ratio_rng_geo"] = math.exp(mean([math.log(x) for x in rr])) if rr else None
    # --- K8 逐 rng ratio 的等权算术平均
    o["K8_ratio_rng_mean"] = mean(rr)
    # --- K9 逐 pair ratio 的中位数
    prr = [ratio_of(x, y) for x, y in pr]
    o["K9_ratio_pair_med"] = median(prr)
    # --- 记录不可判计数
    o["_undef"] = sum(1 for k, v in o.items() if k != "_undef" and v is None)
    o["_npair"] = len(pr)
    return o


KEYS = ["K0_ratio_pool", "K1_ratio_rng_eq", "K2_ratio_pair_eq", "K3_gap_cell_eq",
        "K4_gap_pair_eq", "K5_vote_pair", "K6_ratio_rng_med", "K7_ratio_rng_geo",
        "K8_ratio_rng_mean", "K9_ratio_pair_med"]

print("=" * 108)
print("逐剂量 · 子集内 max/min（合并不变性）—— 越接近 1.0 越不变")
print("=" * 108)
hdr = "dose  nsub " + " ".join("%-17s" % k.split("_", 1)[1] for k in KEYS)
print(hdr)
within = {k: [] for k in KEYS}
for d in DOSES:
    Ss = subsets(d)
    vals = {k: [] for k in KEYS}
    for S in Ss:
        o = K(d, S)
        for k in KEYS:
            if o[k] is not None and o[k] > 0:
                vals[k].append(o[k])
    row = "%-5d %-4d " % (d, len(Ss))
    for k in KEYS:
        v = vals[k]
        if len(v) >= 2:
            r = max(v) / min(v)
            within[k].append(r)
            row += "%-17s " % ("%.3f× (n=%d)" % (r, len(v)))
        else:
            row += "%-17s " % ("—" if not v else "n=1")
    print(row)

print()
print("=" * 108)
print("跨剂量分离（用**全子集**读数）与总结")
print("=" * 108)
print("%-20s %-10s %-12s %-12s %-10s" % ("候选", "max内", "跨剂量", "分离SEP", "≤1.10×?"))
for k in KEYS:
    full = []
    for d in DOSES:
        av = tuple(sorted(D[str(d)]["avail"]))
        v = K(d, av)[k]
        full.append(v if (v and v > 0) else None)
    fv = [x for x in full if x]
    between = (max(fv) / min(fv)) if len(fv) >= 2 else None
    w = max(within[k]) if within[k] else None
    sep = (between / w) if (between and w) else None
    print("%-20s %-10s %-12s %-12s %-10s" % (
        k,
        ("%.3f×" % w) if w else "—",
        ("%.3f×" % between) if between else "—",
        ("%.2f" % sep) if sep else "—",
        ("YES" if (w and w <= 1.10) else "no")))
print()
print("★ `ratio` 在 D-158 的记录值 = 2.654×（合并口径变动）；上表 K0 应为同一量级。")
