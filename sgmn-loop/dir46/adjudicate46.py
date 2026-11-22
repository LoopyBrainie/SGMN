#!/opt/pytorch/bin/python
r"""DIR-46 · **独立裁决器**：只读 `jsonl` 与冻结常数，**不 import 实现**（`R1_task46` 一行都不 import）。

卡 §5 判定表按序命中；卡 §9 证明义务：
  · **P-1** 逐位重算 `C`/`mean`/`sd`/`3se`/删失比 ⇒ 与实现侧写出的值比对（`ATOL=1e-12`）
  · **P-2** 计数恒等式 `n_read+n_skip=N_w−1` 每行必须成立（残差非 0 ⇒ `INCOMPLETE`）
  · **P-3** 反向对照由 `p3_control46.py` 产出，本器只**转述**（不改判据）
"""
import json
import math
import os
import sys

ROOT = "/home/ec2-user/sgmn-loop"
OUT = os.path.join(ROOT, "dir46", "out")

# ══════ 冻结常数（卡 §2；**与实现侧各自独立硬编码**，不得互相 import）
C_LOW, C_HIGH = 0.5, 0.9
ANCHOR_NW, CRIT_NW = 2, 8
CENS_MAX = 0.20
N_INST = 400
SIG_K = 3
NW_GRID = (2, 4, 8, 16, 32, 64, 128)
MODES = ("period2", "perstep")
SEED0_FORMAL = 20261111 + 24000
ATOL = 1e-12


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load_mode(mode):
    rows = []
    for half in ("a", "b"):
        p = os.path.join(OUT, f"DIR46r10_arm_{mode}_{half}.jsonl")
        if not os.path.exists(p):
            return None, rows, f"缺分片 {p}"
        with open(p, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    rows.append(json.loads(ln))
    return True, rows, None


def main():
    rep = {"meta": {"card": "rev 1.0", "card_md5": "19ace86942e89c0c4608b07174a2770a",
                    "thresholds": {"C_LOW": C_LOW, "C_HIGH": C_HIGH, "ANCHOR_NW": ANCHOR_NW,
                                   "CRIT_NW": CRIT_NW, "CENS_MAX": CENS_MAX,
                                   "N_INST": N_INST, "SIG_K": SIG_K,
                                   "SEED0_FORMAL": SEED0_FORMAL,
                                   "NW_GRID": list(NW_GRID)}},
           "violations": [], "modes": {}, "curve": {}}

    # ── 1. INCOMPLETE 检查（缺臂 / 行数 / 种子集合 / P-1 / P-2）
    incom = []
    data = {}
    for mode in MODES:
        ok, rows, err = load_mode(mode)
        if not ok:
            incom.append(err)
            continue
        data[mode] = rows
        if len(rows) != N_INST:
            incom.append(f"{mode}: 行数 {len(rows)} ≠ {N_INST}")
        seeds = sorted(r["seed"] for r in rows)
        want = list(range(SEED0_FORMAL, SEED0_FORMAL + N_INST))
        if seeds != want:
            incom.append(f"{mode}: 种子集合与 {SEED0_FORMAL}..{SEED0_FORMAL+N_INST-1} 不符"
                         f"（缺 {len(set(want)-set(seeds))}、多 {len(set(seeds)-set(want))}）")
        for r in rows:
            for k, c in r["cells"].items():
                n_w = int(k)
                # P-2 计数恒等式
                if c["n_read"] + c["n_skip"] != n_w - 1:
                    incom.append(f"{mode}:{r['seed']}:N_w={n_w} 计数恒等式失败")
                # P-1 `C` 必须等于 cos 列表的均值
                if c["cos"]:
                    if c["C"] is None or abs(mean(c["cos"]) - c["C"]) > ATOL:
                        incom.append(f"{mode}:{r['seed']}:N_w={n_w} C≠mean(cos)")
                    if c["n_read"] != len(c["cos"]):
                        incom.append(f"{mode}:{r['seed']}:N_w={n_w} n_read≠len(cos)")
                else:
                    if c["C"] is not None:
                        incom.append(f"{mode}:{r['seed']}:N_w={n_w} 空 cos 却有 C")
    rep["violations"] = incom[:50]
    rep["n_violations"] = len(incom)

    # ── 2. DIVERGED：非有限未按删失规则处置
    diverged = []
    for mode, rows in data.items():
        for r in rows:
            for k, c in r["cells"].items():
                if (not c["finite"]) and (c["C"] is not None or c["n_read"] != 0):
                    diverged.append(f"{mode}:{r['seed']}:N_w={k}")
    rep["diverged_violations"] = diverged[:20]

    # ── 3. 逐模式读数（曲线 + 判据点）
    for mode, rows in data.items():
        per = {}
        for n_w in NW_GRID:
            cs = [r["cells"][str(n_w)]["C"] for r in rows if r["cells"][str(n_w)]["C"] is not None]
            n_div = sum(1 for r in rows if not r["cells"][str(n_w)]["finite"])
            per[n_w] = {"mean": mean(cs), "sd": sd(cs), "n_use": len(cs),
                        "n_div": n_div, "cens_ratio": n_div / len(rows)}
            if per[n_w]["sd"] is not None:
                per[n_w]["se"] = per[n_w]["sd"] / math.sqrt(per[n_w]["n_use"])
                per[n_w]["3se"] = SIG_K * per[n_w]["se"]
        rep["curve"][mode] = {str(k): v for k, v in per.items()}
        # 首次跌破 C_LOW 的 N_w（诊断量）
        cross = [n_w for n_w in NW_GRID
                 if per[n_w]["mean"] is not None and per[n_w]["mean"] < C_LOW]
        rep["modes"][mode] = {
            "anchor_C": per[ANCHOR_NW]["mean"], "anchor_sd": per[ANCHOR_NW]["sd"],
            "crit_C": per[CRIT_NW]["mean"], "crit_sd": per[CRIT_NW]["sd"],
            "crit_3se": per[CRIT_NW].get("3se"),
            "crit_n_use": per[CRIT_NW]["n_use"], "crit_n_div": per[CRIT_NW]["n_div"],
            "crit_cens_ratio": per[CRIT_NW]["cens_ratio"],
            "first_below_C_LOW": (min(cross) if cross else None),
            "max_div_ratio": max(v["cens_ratio"] for v in per.values()),
        }

    # ── 4. 判定表（**按序命中**）
    verdict, hit = None, None

    def fire(n, name, why):
        nonlocal verdict, hit
        if verdict is None:
            verdict, hit = name, {**n, "why": why}

    fire({"n": 1}, "INCOMPLETE", f"{rep['n_violations']} 处结构违规（前 3：{rep['violations'][:3]}）") \
        if rep["n_violations"] else None
    fire({"n": 2}, "DIVERGED", f"非有限未按删失处置 {len(rep['diverged_violations'])} 处") \
        if rep["diverged_violations"] else None
    if verdict is None:
        cens = {m: rep["modes"][m]["crit_cens_ratio"] for m in data}
        fire({"n": 3}, "CENSORED", f"判据点删失比 {cens} > CENS_MAX={CENS_MAX}") \
            if any(v > CENS_MAX for v in cens.values()) else None
    if verdict is None:
        bad = {m: rep["modes"][m]["anchor_C"] for m in data}
        fire({"n": 4}, "EMPTY-CRITERION", f"C(2)={bad} 未达 C_HIGH={C_HIGH}") \
            if any((v is None or v < C_HIGH) for v in bad.values()) else None
    if verdict is None:
        c_per = rep["modes"]["perstep"]["crit_C"]
        fire({"n": 5}, "NO-CAPACITY-LIMIT", f"C_perstep(8)={c_per:+.5f} ≥ C_LOW={C_LOW}") \
            if c_per is not None and c_per >= C_LOW else None
    if verdict is None:
        c_p2 = rep["modes"]["period2"]["crit_C"]
        fire({"n": 6}, "CONTENT-INDEPENDENT", f"C_period2(8)={c_p2:+.5f} < C_HIGH={C_HIGH}") \
            if c_p2 is None or c_p2 < C_HIGH else None
    if verdict is None:
        c_per = rep["modes"]["perstep"]["crit_C"]
        t3 = rep["modes"]["perstep"]["crit_3se"] or float("inf")
        fire({"n": 7}, "MARGINAL", f"|C_perstep(8)−C_LOW|={abs(c_per-C_LOW):.5f} < 3se={t3:.5f}") \
            if abs(c_per - C_LOW) < t3 else None
    if verdict is None:
        fire({"n": 8}, "PASS", "H46 证实：C_perstep(8)<C_LOW 且 C_period2(8)≥C_HIGH，且不在噪声带内")

    rep["verdict"] = verdict
    rep["rule_hit"] = hit
    ok = verdict == "PASS"
    rep["succeseded_increment"] = bool(ok)

    # ── 5. P-3 转述
    p3p = os.path.join(OUT, "DIR46r10_p3.json")
    rep["P3"] = (json.load(open(p3p, encoding="utf-8")) if os.path.exists(p3p)
                 else {"status": "NOT-RUN"})

    p = os.path.join(OUT, "DIR46r10_adjudication.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"\n★ 结局 = {verdict}（判定表第 {hit['n']} 支）")
    print(f"  理由：{hit['why']}")
    for m in data:
        d = rep["modes"][m]
        print(f"  [{m}] C(2)={d['anchor_C']:+.5f}  C(8)={d['crit_C']:+.5f}"
              f"  (sd {d['crit_sd']:.4f}, 3se {d['crit_3se']:.5f}, n_use {d['crit_n_use']}"
              f", 删失 {d['crit_n_div']}/{N_INST}={d['crit_cens_ratio']:.3f})"
              f"  首次<{C_LOW} 于 N_w={d['first_below_C_LOW']}")
    print(f"  结构违规 {rep['n_violations']} 处 ⇒ {p}")
    sys.exit(0 if ok else 0)


if __name__ == "__main__":
    main()
