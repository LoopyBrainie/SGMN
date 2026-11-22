#!/opt/pytorch/bin/python
r"""DIR-50 · **独立裁决器**：**不 import 实现**（只读 5 臂 jsonl + 卡），**自己**从 `c0/curve/med` 重算全部判据量，
逐条转写卡 rev 1.0 §5 判定表。

用法：`adjudicate50.py --out dir50/out/DIR50r10_adjudication.json`
"""
import argparse
import glob
import json
import math
import os

ROOT = "/home/ec2-user/sgmn-loop"
# ══════ 冻结阈值（卡 §2；**与卡一致，裁决器不得自定**）
ADJ_MIN = None  # 未用
DELTA_MOD_MAX = 0.10        # P1：两模态的直接伤害之差上限
RHO_CASCADE_MIN = 0.10      # P2：跨模态**读出**位移下限（通道打开）
CASCADE_RATIO_MAX = 0.10    # P3：级联 / 直接伤害 的比值上限
N_BAD_MAX = 0.10
SIG_K = 3
N_INST = 4000
SEED_LO, SEED_HI = 33000, 36999
ARMS = ("none", "hit1", "hit2")
DOSES = ("0.0", "1.0")
MED_STEP = 1                # med[] 索引 1 ⇒ step T_HIT+1 = 5（**第一个**受级联影响的步）


def load(arm, dose):
    rows = {}
    tag = f"arm_{arm}" + ("" if dose is None else f"_{dose}")
    for f in sorted(glob.glob(os.path.join(ROOT, f"dir50/out/DIR50r10_{tag}_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows[int(r["seed"])] = r
    return rows


def se3(v):
    n = len(v)
    if n < 2:
        return None
    m = sum(v) / n
    return SIG_K * ((sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5) / math.sqrt(n)


def snan(x, y):
    """NaN-aware 的逐位相同（`NaN == NaN` 在 Python 里为 False）。"""
    return (x == y) or (isinstance(x, float) and isinstance(y, float)
                        and math.isnan(x) and math.isnan(y))


def med(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {("none", None): load("none", None)}
    for arm in ("hit1", "hit2"):
        for dose in DOSES:
            R[(arm, dose)] = load(arm, dose)
    viol = []
    tab = {}

    # ── 独立重算 `rel`：rel_l(d) = c_l(d) / c0_l
    def rel(r, li, d):
        c0 = r["c0"][li]
        if not (math.isfinite(c0) and c0 > 0):
            return None
        c = r["curve"][li][d - 1]
        if not math.isfinite(c):
            return None
        return c / c0

    # ① INCOMPLETE
    inc = []
    for key, rows in R.items():
        if len(rows) != N_INST:
            inc.append(f"{key} 行数 {len(rows)} != {N_INST}")
        if sorted(rows) != list(range(SEED_LO, SEED_HI + 1)):
            inc.append(f"{key} 种子集合不符")
        for sd, r in rows.items():
            if r.get("n_writes") != 2 * 12:
                inc.append(f"{key} seed={sd} 写入数 {r.get('n_writes')} != 24")
                break
    # 结构不变量：`hit2` 的层 1 与 `none` 逐位相同；`hit1` 的层 2 在 d=1 与 `none` 逐位相同
    inv_h2 = 0
    inv_c1 = 0
    for dose in DOSES:
        for sd, r in R[("hit2", dose)].items():
            b = R[("none", None)].get(sd)
            if b is None:
                continue
            if not (snan(r["c0"][0], b["c0"][0]) and
                    all(snan(x, y) for x, y in zip(r["curve"][0], b["curve"][0]))):
                inv_h2 += 1
    for dose in DOSES:
        for sd, r in R[("hit1", dose)].items():
            b = R[("none", None)].get(sd)
            if b is None:
                continue
            if not (snan(r["c0"][1], b["c0"][1]) and snan(r["curve"][1][0], b["curve"][1][0])):
                inv_c1 += 1
    if inv_h2:
        inc.append(f"`hit2` 的层 1 与 `none` 不逐位相同：{inv_h2} 个种子")
    if inv_c1:
        inc.append(f"`hit1` 的层 2 在 d=1 与 `none` 不逐位相同：{inv_c1} 个种子")
    tab["1_INCOMPLETE"] = {"hit": bool(inc), "detail": inc[:8],
                            "inv_hit2_L1_identical": inv_h2 == 0,
                            "inv_hit1_L2_d1_identical": inv_c1 == 0}

    # ② DIVERGED
    nbad = {}
    for key, rows in R.items():
        nb = 0
        for r in rows.values():
            # ★ 判据量口径（卡 rev 1.1）：判据量集合 = {rel_1(1), rel_2(1), rel_2(2)}
            if any(rel(r, li, d) is None for li, d in ((0, 1), (1, 1), (1, 2))):
                nb += 1
        nbad[f"{key[0]}|{key[1]}"] = nb / max(1, len(rows))
    nbad_max = max(nbad.values())
    tab["2_DIVERGED"] = {"hit": nbad_max > N_BAD_MAX, "n_bad_max": nbad_max,
                         "N_BAD_MAX": N_BAD_MAX, "per_arm": nbad}

    # ── 判据量
    def paired(arm, dose, li, d):
        """同实例配对差：rel_l(d)|arm − rel_l(d)|none。"""
        out = []
        for sd, r in R[(arm, dose)].items():
            b = R[("none", None)].get(sd)
            if b is None:
                continue
            x, y = rel(r, li, d), rel(b, li, d)
            if x is not None and y is not None:
                out.append(x - y)
        return out

    # P1：直接伤害（d=1）在两模态间之差 —— ★ `se` 取**逐实例跨模态配对差**
    def paired_delta(arm_a, dose_a, li_a, arm_b, dose_b, li_b, d):
        """逐实例：Δ_i := [rel_a(d)|arm_a − rel_a(d)|none] − [rel_b(d)|arm_b − rel_b(d)|none]。"""
        out = []
        for sd, r in R[(arm_a, dose_a)].items():
            rb, bb = R[(arm_b, dose_b)].get(sd), R[("none", None)].get(sd)
            if rb is None or bb is None:
                continue
            xa, xb = rel(r, li_a, d), rel(bb, li_a, d)
            ya, yb = rel(rb, li_b, d), rel(bb, li_b, d)
            if None not in (xa, xb, ya, yb):
                out.append((xa - xb) - (ya - yb))
        return out

    per_dose = {}
    for dose in DOSES:
        dir1 = paired("hit1", dose, 0, 1)
        dir2 = paired("hit2", dose, 1, 1)
        D1 = sum(dir1) / len(dir1) if dir1 else None
        D2 = sum(dir2) / len(dir2) if dir2 else None
        dl = paired_delta("hit1", dose, 0, "hit2", dose, 1, 1)
        DM = abs(sum(dl) / len(dl)) if dl else None
        per_dose[dose] = {"DIRECT_1": D1, "se1": se3(dir1), "DIRECT_2": D2, "se2": se3(dir2),
                          "DELTA_MOD": DM, "se_delta": se3(dl), "n1": len(dir1), "n2": len(dir2),
                          "n_delta": len(dl)}
    dir1 = paired("hit1", "1.0", 0, 1)
    dir2 = paired("hit2", "1.0", 1, 1)
    DIRECT_1 = sum(dir1) / len(dir1) if dir1 else None
    DIRECT_2 = sum(dir2) / len(dir2) if dir2 else None
    se1, se2 = se3(dir1), se3(dir2)
    DELTA_MOD = per_dose["1.0"]["DELTA_MOD"]
    se_delta = per_dose["1.0"]["se_delta"]
    # P2：跨模态**读出**位移（上游被命中 ⇒ 层 1 的读出 `y_1` 偏移）—— 中位数
    rho_by_dose = {}
    rho_vals = []
    for dose in DOSES:
        rv = []
        for sd, r in R[("hit1", dose)].items():
            b = R[("none", None)].get(sd)
            if b is None:
                continue
            yh, yn = r["med"]["y1"][MED_STEP], b["med"]["y1"][MED_STEP]
            # ★ 数值闸门：发散实例的巨值会让平方和溢出 ⇒ 先剔除非有限/超量级分量
            if not all(math.isfinite(z) for z in list(yh) + list(yn)):
                continue
            if max(abs(z) for z in list(yh) + list(yn)) > 1e100:
                continue
            den = math.sqrt(sum(z * z for z in yn))
            if not (den > 0 and math.isfinite(den)):
                continue
            num = math.sqrt(sum((x - y) ** 2 for x, y in zip(yh, yn)))
            v = num / den
            if math.isfinite(v):
                rv.append(v)
        rho_by_dose[dose] = {"median": med(rv), "se": se3(rv), "n": len(rv)}
        rho_vals.extend(rv)
    RHO_Y1 = rho_by_dose["1.0"]["median"]
    # P3：级联 / 直接伤害 的比值
    casc = paired("hit1", "1.0", 1, 2)          # 层 2 在 d=2（第一个受级联影响的步）
    CASC = sum(casc) / len(casc) if casc else None
    se_casc = se3(casc)
    DIRECT = abs(DIRECT_2) if DIRECT_2 is not None else None
    RATIO = (abs(CASC) / DIRECT) if (CASC is not None and DIRECT) else None

    p1_ok = all(per_dose[dd]["DELTA_MOD"] is not None and per_dose[dd]["DELTA_MOD"] <= DELTA_MOD_MAX
                for dd in DOSES)
    p2_ok = RHO_Y1 is not None and RHO_Y1 >= RHO_CASCADE_MIN
    p3_ok = RATIO is not None and RATIO <= CASCADE_RATIO_MAX

    tab["3_MODALITY-DEPENDENT"] = {"hit": not p1_ok, "DELTA_MOD": DELTA_MOD, "se_delta": se_delta,
                                   "DELTA_MOD_MAX": DELTA_MOD_MAX,
                                   "DIRECT_1": DIRECT_1, "DIRECT_2": DIRECT_2, "se1": se1, "se2": se2,
                                   "per_dose": per_dose}
    tab["4_NO-CROSS-CHANNEL"] = {"hit": not p2_ok, "RHO_Y1": RHO_Y1, "RHO_CASCADE_MIN": RHO_CASCADE_MIN,
                                 "n": len(rho_vals)}
    tab["5_FULL-TRANSFER"] = {"hit": not p3_ok, "CASC": CASC, "se_casc": se_casc,
                              "DIRECT": DIRECT, "RATIO": RATIO, "CASCADE_RATIO_MAX": CASCADE_RATIO_MAX,
                              "n_casc": len(casc)}
    # ⑥ MARGINAL：任一判据量与其阈值相距 < 3se
    marg = []
    for dd in DOSES:
        pd_ = per_dose[dd]
        if pd_["DELTA_MOD"] is not None and pd_["se_delta"] is not None:
            if abs(pd_["DELTA_MOD"] - DELTA_MOD_MAX) < pd_["se_delta"]:
                marg.append(f"DELTA_MOD[{dd}] 距上限 {abs(pd_['DELTA_MOD'] - DELTA_MOD_MAX):.4f}"
                            f" < 3se {pd_['se_delta']:.4f}")
    if RHO_Y1 is not None and rho_vals:
        sv = se3(rho_vals)
        if sv is not None and abs(RHO_Y1 - RHO_CASCADE_MIN) < sv:
            marg.append(f"RHO_Y1 距下限 {abs(RHO_Y1 - RHO_CASCADE_MIN):.4f} < 3se {sv:.4f}")
    if RATIO is not None and se_casc is not None and DIRECT:
        sr = se_casc / DIRECT
        if abs(RATIO - CASCADE_RATIO_MAX) < sr:
            marg.append(f"RATIO 距上限 {abs(RATIO - CASCADE_RATIO_MAX):.4f} < 3se {sr:.4f}")
    tab["6_MARGINAL"] = {"hit": bool(marg), "detail": marg}

    order = ["1_INCOMPLETE", "2_DIVERGED", "3_MODALITY-DEPENDENT", "4_NO-CROSS-CHANNEL",
             "5_FULL-TRANSFER", "6_MARGINAL"]
    verdict, rn = "PASS", 7
    for i, k in enumerate(order):
        if tab[k]["hit"]:
            verdict, rn = k.split("_", 1)[1], i + 1
            break

    out = {"card": "DIR-50 rev 1.1 (md5 0be3668d2a6755adf0361b770b6d5005)",
           "verdict": verdict, "rule_hit": {"n": rn, "name": verdict},
           "H50": {"P1_modality_independent": p1_ok, "P2_channel_open": p2_ok,
                   "P3_memory_isolation": p3_ok},
           "ADJ": {"DIRECT_1": DIRECT_1, "se1": se1, "DIRECT_2": DIRECT_2, "se2": se2,
                   "DELTA_MOD": DELTA_MOD, "se_delta": se_delta,
                   "n1": len(dir1), "n2": len(dir2), "per_dose": per_dose},
           "CASCADE": {"CASC_d2": CASC, "se_casc": se_casc, "DIRECT": DIRECT, "RATIO": RATIO,
                       "n": len(casc),
                       "CASC_by_dose": {dose: (sum(paired("hit1", dose, 1, 2)) / len(paired("hit1", dose, 1, 2))
                                               if paired("hit1", dose, 1, 2) else None)
                                        for dose in DOSES},
                       "CASC_by_d": {str(d): (sum(paired("hit1", "1.0", 1, d)) / len(paired("hit1", "1.0", 1, d))
                                             if paired("hit1", "1.0", 1, d) else None)
                                     for d in range(1, 8)}},
           "CHANNEL": {"RHO_Y1_median": RHO_Y1, "n": len(rho_vals),
                       "RHO_Y1_by_dose": rho_by_dose},
           "table": tab, "n_violations": len(viol),
           "frozen": {"DELTA_MOD_MAX": DELTA_MOD_MAX, "RHO_CASCADE_MIN": RHO_CASCADE_MIN,
                      "CASCADE_RATIO_MAX": CASCADE_RATIO_MAX, "N_BAD_MAX": N_BAD_MAX,
                      "SIG_K": SIG_K, "N_INST": N_INST, "SEED": [SEED_LO, SEED_HI]}}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
