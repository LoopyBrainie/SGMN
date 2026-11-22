# -*- coding: utf-8 -*-
"""DIR-111 纯逻辑自测（不加载模型）：把 R1_task111.py 的判据区段 exec 出来，喂合成 cells。"""
import pathlib
import numpy as np

src = pathlib.Path(__file__).resolve().parent.joinpath("R1_task111.py").read_text(encoding="utf-8")
i = src.index('L8 = "8"')
j = src.index("def dir110_ceiling_audit")
G = {"np": np, "json": __import__("json"), "pathlib": pathlib,
     "_ROOT": pathlib.Path(__file__).resolve().parent.parent,
     "GAMMAS": (0.0010, 0.0025, 0.0100), "SEEDS": (11, 22, 33),
     "TAU_ARM": 0.10, "CEIL_MAX": 0.95, "MIN_NQ": 3, "RATIO_MAX": 0.90}
exec(src[i:j], G)
shape_verdict = G["shape_verdict"]
Q_CAND = G["Q_CAND"]
print("候选数 =", len(Q_CAND))

GAM = (0.0010, 0.0025, 0.0100)
FAIL = []


def cells_from(spec):
    """★ 必须先累加再成格：同一 (arm,g,s) 承载**多个**量，直接赋值会互相覆盖。"""
    acc = {}
    for (g, o, mt), vals in spec.items():
        for arm, val in zip(("T-NONE", "T-Q", "T-O", "T-BOTH"), vals):
            for s in (11, 22, 33):
                k = f"{arm}|{g}|{s}"
                node = acc.setdefault(k, {"8": {}, "16": {}})
                for L in ("8", "16"):
                    node[L].setdefault(o, {})[mt] = val
    return {k: {"state": v} for k, v in acc.items()}


def check(name, spec, want):
    code, det = shape_verdict(cells_from(spec))
    ok = code == want
    print(("  PASS  " if ok else "  **FAIL** ") + name + f"  => {code} (want {want}) n_q={det['n_eligible']}"
          + (f" median={det['median_ratio']:.3f}" if det.get("median_ratio") is not None else ""))
    if not ok:
        FAIL.append((name, code, want))
        for p in det["points"]:
            print("        ", p["obj"], p["met"], "g=%s" % p["gamma"],
                  "dQ=%.3f dO=%.3f dB=%.3f sum=%.3f Me=%.4f Se=%.4f r=%.3f elig=%s %s"
                  % (p["dQ"], p["dO"], p["dB"], p["sum_pred"], p["max_err"], p["sum_err"],
                     p["ratio"], p["eligible"], p["why_not"]))
    return code, det


# 1) 全内点、最弱环全赢 ⇒ CONFIRMED（§2.5(a) 型）
s1 = {}
for g in GAM:
    s1[(g, "m", "rank_eff")] = (0.20, 0.48, 0.68, 0.66)   # dQ=.35 dO=.60 sum=.95
    s1[(g, "m", "cos_mean")] = (0.20, 0.44, 0.60, 0.60)   # dQ=.30 dO=.50 sum=.80
check("场景1 内点全赢→CONFIRMED", s1, "C111-WEAKEST-LINK-CONFIRMED")

# 2) 一处求和赢 ⇒ FALSIFIED（§2.5(b) 型）
s2 = dict(s1)
s2[(0.0010, "m", "cos_mean")] = (0.20, 0.44, 0.60, 0.78)
check("场景2 一处求和赢→FALSIFIED", s2, "C111-FALSIFIED")

# 3) 全越界 ⇒ NO-INTERIOR-POINT
s3 = {}
for g in GAM:
    s3[(g, "m", "rank_eff")] = (0.20, 0.60, 0.75, 0.95)
    s3[(g, "m", "cos_mean")] = (0.20, 0.65, 0.80, 0.97)
check("场景3 全越界→NO-INTERIOR", s3, "C111-NO-INTERIOR-POINT")

# 4) 全赢但近乎打平（median ratio > 0.90）⇒ WEAK
db = (0.5 - 0.95 * 0.9) / (1 - 0.95)
s4 = {}
for g in GAM:
    s4[(g, "m", "rank_eff")] = (0.20, 0.52, 0.60, db)
    s4[(g, "m", "cos_mean")] = (0.20, 0.52, 0.60, db)
check("场景4 全赢但打平→WEAK (dB=%.4f)" % db, s4, "C111-WEAKEST-LINK-WEAK")

# 5) 恰 3 个合格（= MIN_NQ）⇒ 可判（★ 不用 dict(s3) 继承，避免残留项）
s5 = {}
for g in GAM:
    s5[(g, "m", "rank_eff")] = (0.20, 0.48, 0.68, 0.66)   # 合格（sum=.95）
    s5[(g, "m", "cos_mean")] = (0.20, 0.60, 0.80, 0.95)   # 不合格（sum=1.20 越界）
check("场景5 恰 3 合格→可判", s5, "C111-WEAKEST-LINK-CONFIRMED")
if shape_verdict(cells_from(s5))[1]["n_eligible"] != 3:
    FAIL.append(("场景5 n_q", shape_verdict(cells_from(s5))[1]["n_eligible"], 3))

# 6) 仅 2 个合格 ⇒ NO-INTERIOR（MIN_NQ=3 起作用）
s6 = {}
for g in GAM:
    s6[(g, "m", "rank_eff")] = (0.20, 0.48, 0.68, 0.66)
    s6[(g, "m", "cos_mean")] = (0.20, 0.60, 0.80, 0.95)
s6[(0.0100, "m", "rank_eff")] = (0.20, 0.60, 0.80, 0.95)  # 第三个 rank_eff 也越界 ⇒ 只剩 2 个
check("场景6 仅 2 合格→NO-INTERIOR", s6, "C111-NO-INTERIOR-POINT")
if shape_verdict(cells_from(s6))[1]["n_eligible"] != 2:
    FAIL.append(("场景6 n_q", shape_verdict(cells_from(s6))[1]["n_eligible"], 2))

# 7) 单臂未达 TAU_ARM ⇒ 不合格
s7 = {}
for g in GAM:
    s7[(g, "m", "rank_eff")] = (0.20, 0.42, 0.60, 0.60)   # dQ=.275 dO=.50 合格
    s7[(g, "m", "cos_mean")] = (0.20, 0.24, 0.60, 0.60)   # dQ=.05 < .10 不合格
code7, det7 = check("场景7 单臂未达门→3 合格", s7, "C111-WEAKEST-LINK-CONFIRMED")
if det7["n_eligible"] != 3:
    FAIL.append(("场景7 n_q", det7["n_eligible"], 3))

print("\n" + ("★ 全部通过" if not FAIL else ("★ 失败 %d 项：%s" % (len(FAIL), FAIL))))
raise SystemExit(0 if not FAIL else 1)
