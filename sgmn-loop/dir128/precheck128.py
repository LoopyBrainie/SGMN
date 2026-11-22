#!/usr/bin/env python3
"""`DIR-128` 结构核（`S1…S7`）—— **盲态**：绝不打印任何 `(dose, S)` 的 `ratio`，也绝不打印 `J128-*` 结局。

★ 纪律（`DEF-127-03` 的修法）：本脚本**不调用** `tension()`（它会给出 `max/min`，即判据结局的原料）。
  只断言四类东西：① 结构量 ② 已发表常数 ③ 管线一致性（只报"是否相同"）④ 只读性。
★ `S6` 必须触及两个 `ratio` 值以复现 `D-158` —— 但它报的是**"与冻结值是否相容"**（布尔），
  且这两个值**已在 `D-158` 公开**，故不构成泄漏。
"""
import hashlib
import json
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mech128 as M  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CELL_DIRS = [os.path.join(ROOT, "dir119", "out", "cells"),
             os.path.join(ROOT, "dir125", "out", "cells"),
             os.path.join(ROOT, "dir126", "out", "cells")]

_fail = []
_res = []


def gate(name, cond, detail=""):
    _res.append({"id": name, "pass": bool(cond), "detail": detail})
    if not cond:
        _fail.append(name)
    print("  %-8s %-6s %s" % (name, "PASS" if cond else "★FAIL", detail))


def md5_dir(d):
    """目录内全部文件的 md5 汇总（只读）。"""
    if not os.path.isdir(d):
        return None
    h = hashlib.md5()
    for fn in sorted(os.listdir(d)):
        fp = os.path.join(d, fn)
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                h.update(fn.encode() + b"\0" + f.read())
    return h.hexdigest()


T = M.load_tensor()
DOSES = M.all_doses(T)
FROZEN_N = {"555": (23, 20), "556": (18, 27), "557": (9, 29), "558": (18, 23)}

print("=" * 96)
print("`DIR-128` 结构核（S1–S7）· 盲态 · 张量 %d 剂量" % len(DOSES))
print("=" * 96)

# ---------------- S1 张量结构
seen = {}
bad = []
for d in DOSES:
    v = M.dose_view(T, d)
    for r, pr in v["per_rng"].items():
        seen.setdefault(r, set()).add((pr["n1"], pr["n2"]))
    if v["nc"] != 9:
        bad.append(("nc", d))
ok = all(seen.get(r) == {FROZEN_N[r]} for r in FROZEN_N) and len(seen) == 4 and not bad
gate("S1", ok, "(n₁,n₂)=%s  nc=%d  违例=%d"
     % ({r: sorted(seen[r])[0] for r in sorted(seen) if seen[r]}, 9, len(bad)))

# ---------------- S2 N 与 cell 无关（张量层已断言；此处再验长度与唯一性）
bad2 = []
for d in DOSES:
    v = M.dose_view(T, d)
    for r, pr in v["per_rng"].items():
        if len(pr["e1"]) != v["nc"] or len(pr["e2"]) != v["nc"]:
            bad2.append((d, r))
        if any(x < 0 for x in pr["e1"] + pr["e2"]):
            bad2.append((d, r, "neg"))
gate("S2", not bad2, "全部 (dose,rng) 的 e 长度==nc 且非负  违例=%d" % len(bad2))

# ---------------- S3 ★ D-128-L1（有理数精确 + 浮点相对 1e-15）
bad3 = []
n_one = 0
for d in DOSES:
    v = M.dose_view(T, d)
    nc, per = v["nc"], v["per_rng"]
    for s in M.all_subsets(v["avail"]):
        if len(s) != 1:
            continue
        n_one += 1
        r = s[0]
        # |S|=1 时五条规则的 p₁ 在 Q 上全等于 E1/(nc·n1)
        qp1 = F(sum(per[r]["e1"]), nc * per[r]["n1"])
        qp2 = F(sum(per[r]["e2"]), nc * per[r]["n2"])
        o = M.rules(v, s)
        vals = [o[k] for k in M.AXIS1]
        if any(x is None for x in vals):
            bad3.append((d, s, "None"))
            continue
        ref = M.ratio_of(float(qp1), float(qp2))
        if ref is None:
            continue
        if max(vals) - min(vals) > 1e-15 * max(abs(x) for x in vals):
            bad3.append((d, s, "spread"))
        if abs(vals[0] - ref) > 1e-15 * abs(ref):
            bad3.append((d, s, "ref"))
gate("S3", not bad3 and n_one == 33,
     "|S|=1 结构 %d 个（须 33）；A0..A4 在 Q 上精确相等且浮点相对 <=1e-15；违例=%d" % (n_one, len(bad3)))

# ---------------- S4 计数可加性（121 个结构，两路）
bad4 = []
n_struct = 0
for d in DOSES:
    v = M.dose_view(T, d)
    for s in M.all_subsets(v["avail"]):
        n_struct += 1
        for key in ("e1", "e2"):
            a = sum(v["per_rng"][r][key][c] for r in s for c in range(v["nc"]))
            b = sum(sum(v["per_rng"][r][key]) for r in s)
            if a != b:
                bad4.append((d, s, key))
gate("S4", not bad4 and n_struct == 121,
     "结构 %d 个（须 121）；两路求和逐位相同；违例=%d" % (n_struct, len(bad4)))

# ---------------- S5 只读性（三个归档 cell 目录 md5）
m5 = {}
for d in CELL_DIRS:
    m5[os.path.basename(os.path.dirname(os.path.dirname(d)))] = md5_dir(d)
present = {k: v for k, v in m5.items() if v}
gate("S5", len(present) >= 2, "cell 目录 md5 记录：%s" % json.dumps(present, sort_keys=True))

# ---------------- S6 与 D-157/D-158 的已发表读数相容（只报布尔）
PUB = {"dose=450/(555,)": (0.836648, 0.8366), "dose=450/(557,)": (2.220149, 2.2201)}
ok6 = []
v450 = M.dose_view(T, 450)
for lab, s in (("dose=450/(555,)", ("555",)), ("dose=450/(557,)", ("557",))):
    got = M.rules(v450, s)["A0"]
    exp_pub, exp_round = PUB[lab]
    ok6.append(got is not None and abs(got - exp_pub) < 1e-5)
gate("S6", all(ok6), "D-158 的两个单题集读数在 1e-5 内复现（只报布尔，不打印值）：%s" % ok6)

# ---------------- S7 样本空间
n_1 = sum(len([s for s in M.all_subsets(M.dose_view(T, d)["avail"]) if len(s) == 1]) for d in DOSES)
n_2 = sum(len([s for s in M.all_subsets(M.dose_view(T, d)["avail"]) if len(s) >= 2]) for d in DOSES)
n_a = sum(len(M.all_subsets(M.dose_view(T, d)["avail"])) for d in DOSES)
gate("S7", (n_a, n_2, n_1) == (121, 88, 33),
     "121 / |S|>=2 的 88 / |S|=1 的 33（33+88=121）  实测 %d/%d/%d" % (n_a, n_2, n_1))

print("-" * 96)
verdict = "PASS" if not _fail else "FAIL(%s)" % _fail
print("★ 结构核 **%s** ⇒ %s" % (verdict, "允许进 Stage A / formal" if not _fail
                                else "**阻断** formal，必须先修"))
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "precheck128.json"), "w", encoding="utf-8") as f:
    json.dump({"gates": _res, "verdict": verdict, "tensor_md5": md5_dir(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out"))}, f, indent=1, ensure_ascii=False)
sys.exit(0 if not _fail else 1)