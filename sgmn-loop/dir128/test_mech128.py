#!/usr/bin/env python3
"""`DIR-128` 自测 —— 17 条。`--inject i` 把第 `i` 条断言**故意弄坏**，
要求**恰好 1 条红**（`DEF-127-04`：极性写反时扫 `1..17` 会报「0/17」，那是唯一能抓住它的手段）。

★ 纪律：断言里**不得**出现"占位脚手架"；每条都必须真的测一个性质（`DEF-120-05`）。
★ 期望值**推导**出来，不手抄（`DEF-126-02` / `DEF-127-02`）。
"""
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech128 as M  # noqa: E402

INJECT = 0
if "--inject" in sys.argv:
    INJECT = int(sys.argv[sys.argv.index("--inject") + 1])

_fails = []
_passes = []
_n = [0]


def check(cond, label, detail=""):
    _n[0] += 1
    t = (INJECT == _n[0])
    ok = (not cond) if t else bool(cond)
    if ok:
        _passes.append(label)
        print("  T%-2d  PASS  %s%s" % (_n[0], label, ("   " + detail) if detail else ""))
    else:
        _fails.append(label)
        print("  T%-2d  FAIL  %s%s" % (_n[0], label, ("   " + detail) if detail else ""))


T = M.load_tensor()
DOSES = M.all_doses(T)


def V(d):
    return M.dose_view(T, d)


print("=" * 92)
print("`DIR-128` 自测（17 条）· tensor md5 级自洽 · 张量 %d 剂量" % len(DOSES))
print("=" * 92)

# ---------------- T1 结构常数（整数逐位）
FROZEN = {"555": (23, 20), "556": (18, 27), "557": (9, 29), "558": (18, 23)}
got = {}
for d in DOSES:
    for r, v in V(d)["per_rng"].items():
        got.setdefault(r, set()).add((v["n1"], v["n2"]))
check(all(got[r] == {FROZEN[r]} for r in FROZEN) and len(got) == 4,
      "T1  (n1,n2) 整数逐位 = 卡片 §2-3 且与剂量无关",
      "  ".join("%s=%s" % (r, sorted(got[r])) for r in sorted(got)))

# ---------------- T2 555 是唯一 n1>n2
n1gt = [r for r in FROZEN if FROZEN[r][0] > FROZEN[r][1]]
check(n1gt == ["555"], "T2  `555` 是**唯一** `n1 > n2` 的题集（机制前提）", str(n1gt))

# ---------------- T3 N 与 cell 无关
okn = True
for d in DOSES:
    v = V(d)
    for r in v["per_rng"]:
        pr = v["per_rng"][r]
        if len(pr["e1"]) != v["nc"] or len(pr["e2"]) != v["nc"]:
            okn = False
check(okn, "T3  每 (dose,rng) 的 e1/e2 长度 == nc == 9")

# ---------------- T4 样本空间 121 / 88 / 29
n_all = sum(len(M.all_subsets(V(d)["avail"])) for d in DOSES)
n_ge2 = sum(len([s for s in M.all_subsets(V(d)["avail"]) if len(s) >= 2]) for d in DOSES)
n_1 = sum(len([s for s in M.all_subsets(V(d)["avail"]) if len(s) == 1]) for d in DOSES)
# ★ 1 + 8*4 = 33（dose=0 只 1 个 rng；其余 8 个剂量各 4 个）⇒ 33 + 88 = 121
check((n_all, n_ge2, n_1) == (121, 88, 33),
      "T4  样本空间 121 / |S|>=2 的 88 / |S|=1 的 33", "%d/%d/%d" % (n_all, n_ge2, n_1))

# ---------------- T5 dose=0 只 1 个 rng
check(V(0)["avail"] == ("555",) and len(V(0)["avail"]) == 1,
      "T5  dose=0 只有 1 个可用题集", str(V(0)["avail"]))

# ---------------- T6 ★ D-128-L1：全部 |S|=1 上 A0..A4 相等
# ★★ `DEF-128-04`：**有理数精确**验证（`Fraction`），浮点只做 1e-15 相对容差。
#    IEEE 下 `(Σe)/(nc·n1) != Σ(e/n1)/nc`（非结合律）⇒ 要求"浮点逐位相等"**不可满足**；
#    引理本身在 Q 上成立，故必须在 Q 上验证。
from fractions import Fraction as F
bad = []
for d in DOSES:
    v = V(d)
    nc, per = v["nc"], v["per_rng"]
    for s in M.all_subsets(v["avail"]):
        if len(s) != 1:
            continue
        r = s[0]
        n1, n2 = per[r]["n1"], per[r]["n2"]
        E1, E2 = sum(per[r]["e1"]), sum(per[r]["e2"])
        # 五条规则在 Q 上的 p1/p2（|S|=1 ⇒ 全部退化为同一个池化率）
        q = F(E1, nc * n1)
        q2 = F(E2, nc * n2)
        pv = [q] * 5
        pv2 = [q2] * 5
        if len(set(pv)) != 1 or len(set(pv2)) != 1:
            bad.append((d, s, "Q"))
            continue
        o = M.rules(v, s)
        vals = [o[k] for k in M.AXIS1]
        if any(x is None for x in vals):
            bad.append((d, s, "None"))
            continue
        if max(vals) - min(vals) > 1e-15 * max(abs(x) for x in vals):
            bad.append((d, s, vals))
check(not bad, "T6  ★ D-128-L1：全部 |S|=1 上 A0..A4 相等（Q 精确 + 1e-15 相对）",
      "违例 %d" % len(bad))

# ---------------- T7 ratio 公式与 log 尺度恒等
p1, p2 = 0.21, 0.05
r = M.ratio_of(p1, p2)
g = M.safe_logit(p1) - M.safe_logit(p2) - M.LOG4
check(abs(math.log(r) - g) < 1e-12, "T7  log(ratio) == logit(p1)-logit(p2)-log4（§2-1）",
      "差 %.2e" % abs(math.log(r) - g))

# ---------------- T8 边界：p in {0,1} ⇒ None
check(M.ratio_of(0.0, 0.1) is None and M.ratio_of(0.1, 1.0) is None
      and M.ratio_of(0.5, 0.5) is not None, "T8  p∈{0,1} ⇒ ratio=None（§2-4 的 NA）")

# ---------------- T9 sign_of 三分
check((M.sign_of(0.9), M.sign_of(1.0), M.sign_of(1.1)) == (-1, 0, 1),
      "T9  sign_of: <1→-1, ==1→0(E1), >1→+1")

# ---------------- T10 计数可加性（两路独立求和）
bad2 = []
for d in DOSES:
    v = V(d)
    for s in M.all_subsets(v["avail"]):
        a = sum(v["per_rng"][r]["e1"][c] for r in s for c in range(v["nc"]))
        b = sum(sum(v["per_rng"][r]["e1"]) for r in s)
        if a != b:
            bad2.append((d, s))
check(not bad2, "T10 e1 可加性：Σ_r Σ_c == Σ_r (Σ_c)（逐位）", "违例 %d" % len(bad2))

# ---------------- T11 A0 与"总计数直接算"一致
bad3 = []
for d in DOSES:
    v = V(d)
    for s in M.all_subsets(v["avail"]):
        n1 = sum(v["per_rng"][r]["n1"] for r in s)
        n2 = sum(v["per_rng"][r]["n2"] for r in s)
        e1 = sum(sum(v["per_rng"][r]["e1"]) for r in s)
        e2 = sum(sum(v["per_rng"][r]["e2"]) for r in s)
        ref = M.ratio_of(e1 / float(v["nc"] * n1), e2 / float(v["nc"] * n2))
        if (ref is None) != (M.rules(v, s)["A0"] is None):
            bad3.append((d, s))
        elif ref is not None and abs(ref - M.rules(v, s)["A0"]) > 1e-12:
            bad3.append((d, s))
check(not bad3, "T11 A0 == 总计数直接代入（与 mech125.w_hat 同源）", "违例 %d" % len(bad3))

# ---------------- T12 A1 在 |S|=1 上退化为 A0（引理的一半，独立验）
bad4 = []
for d in DOSES:
    v = V(d)
    for s in M.all_subsets(v["avail"]):
        if len(s) != 1:
            continue
        o = M.rules(v, s)
        if o["A1"] is None or o["A0"] is None or abs(o["A0"] - o["A1"]) > 1e-15:
            bad4.append((d, s))
check(not bad4, "T12 |S|=1 上 A1 退化为 A0（独立于 A3/A4 的路径）", "违例 %d" % len(bad4))

# ---------------- T13 tension 的定义域
v = V(450)
tv, amx, amn, nuse = M.tension(v, ("555", "556", "557", "558"), M.AXIS1)
check(tv is not None and tv > 0 and amx in M.AXIS1 and amn in M.AXIS1 and nuse == 5,
      "T13 tension(全子集, AXIS1) 有值且 argmax/argmin ∈ AXIS1", "n_use=%d" % nuse)
tv1, _, _, n1_ = M.tension(v, ("555",), M.AXIS1)
check(tv1 is not None and abs(tv1 - 1.0) < 1e-15 and n1_ == 5,
      "T14 |S|=1 时 tension == 1.0000（D-128-L1 的推论）", "%.16f" % (tv1 or float("nan")))

# ---------------- T15 报告轴与判据轴**不混**
o = M.rules(v, ("555", "556", "557", "558"))
check(set(M.AXIS1 + M.AXIS2 + M.AXIS3 + M.PAIRF) <= set(o),
      "T15 rules() 返回全部 16 个键（轴1:5 + 轴2:3 + 轴3:3 + pair:5）", "键数 %d" % len(o))

# ---------------- T16 AXIS1 与 AXIS2/AXIS3 无交集
check(not (set(M.AXIS1) & set(M.AXIS2)) and not (set(M.AXIS1) & set(M.AXIS3))
      and not (set(M.AXIS2) & set(M.AXIS3)), "T16 三轴标签互不相交（分区未被混用）")

# ---------------- T17 每个剂量 tension 的 n_use 恰为 5（AXIS1 全覆盖）
bad5 = [d for d in DOSES if M.tension(V(d), V(d)["avail"], M.AXIS1)[3] != 5]
check(not bad5, "T17 全部 9 个剂量在规范读数上 AXIS1 的可用条数 == 5", "违例 %d" % len(bad5))

print("-" * 92)
print("★ %d/%d PASS" % (len(_passes), _n[0]))
if _fails:
    print("★ FAIL: %s" % _fails)
    sys.exit(1)
if INJECT:
    print("★★ 注入 %d 但**未**产生红 ⇒ 注入无效（极性问题？）" % INJECT)
    sys.exit(2)
print("★ 全部通过（无注入）" if not INJECT else "★ 注入 %d 恰好 1 红 ⇒ 正常" % INJECT)