#!/opt/pytorch/bin/python
r"""DIR-43 · **独立裁决器**（不 import 实现）—— 从**原始 jsonl** 重算冻结判定表（卡 §4）。

★ **为什么要独立**：`merge()` 的聚合与判定和实现同源 ⇒ 实现若有错，裁决器跟着错。
  本文件**只读** 12 个分片 jsonl，**自己**重算统计量与**按序命中**卡 §4 的七支判定表，
  再与 `DIR43r10_formal.json` **逐字段对照**。

★ **只读 / 不训练**。★ **不提前裁决**：任一臂的实例数不足 ⇒ 打印「尚未跑完」并**拒绝出裁决**
  （判据「从未被计算」≠ PASS/FAIL）。

用法（**在 trainer 上**）：
    /opt/pytorch/bin/python dir43/adjudicate43.py
"""
import json
import math
import os
import statistics

ROOT = "/home/ec2-user/sgmn-loop"
OUT = f"{ROOT}/dir43/out"
CARD = f"{ROOT}/revolve/DIR-43/DIRECTION_rev43_v10.md"
FORMAL = f"{OUT}/DIR43r10_formal.json"

# ══════════════════════════ 冻结常数（**逐字抄自卡 §3.2**）
CARD_MD5 = "9a9a8998a0dff9c3645fb6dc0ec981a1"      # rev 1.0 冻结卡
SIG_K = 3.0
COS_DIRECT = 0.9
N_INST, N_CTRL = 400, 100
ARMS = ("learned", "rand", "direct")
N_EXPECT = {"learned": N_INST, "rand": N_CTRL, "direct": N_INST}
INST_OFF0, INST_OFFN = 22500, 22900                # 判据臂实例段 22500..22899
CTRL_OFF0, CTRL_OFFN = 22500, 22600                # rand 对照臂 22500..22599（判据段前 100）
SHARDS = 4


def md5(path):
    import hashlib
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def stats(xs):
    """均值/sd/se/3se（**独立重写**；非有限 ⇒ NaN，不抛异常）。"""
    xs = list(xs)
    if not xs or not all(math.isfinite(x) for x in xs):
        return {"mean": float("nan"), "sd": float("nan"), "se": float("nan"),
                "3se": float("nan"), "n": len(xs)}
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    se = sd / len(xs) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": SIG_K * se, "n": len(xs)}


def paired(a, b):
    return stats([x - y for x, y in zip(a, b)])


def decide(v):
    r"""★ **卡 §4 的七支判定表**（**逐字转写**；**按序命中**；只有第 7 支 `counted=true`）。

    ★ 与实现分开写 ⇒ 同一组数若给出不同结局 ⇒ 立刻暴露（对照见 `ADJ-43-4`）。
    """
    # —— 三条谓词（**自己从统计量算**，卡 §3.2）
    p1 = v["rho_mean"] >= SIG_K * v["rho_se"]
    p1b = v["rho0_mean"] >= SIG_K * v["rho0_se"]
    p2 = v["rho_diff_mean"] >= SIG_K * v["rho_diff_se"]
    p3 = v["direct_mean"] >= COS_DIRECT
    # —— 主/次标签（终态/初值）
    if v["rho_mean"] <= -SIG_K * v["rho_se"]:
        primary = "ANTI"
    elif not p1:
        primary = "NULL"
    else:
        primary = "POSITIVE"
    if v["rho0_mean"] <= -SIG_K * v["rho0_se"]:
        secondary = "ANTI"
    elif not p1b:
        secondary = "NULL"
    else:
        secondary = "POSITIVE"
    # —— **按序命中**（卡 §4 的 #1..#7）
    if not v["n_ok"]:
        out = "INCOMPLETE"
    elif not v["finite"]:
        out = "DIVERGED"
    elif primary == "ANTI":
        out = "ANTI-ADDRESSING"
    elif primary == "NULL":
        out = "NOT-ADDRESSING"
    elif not p2:
        out = "NOT-SPECIFIC"
    elif not p3:
        out = "EMPTY-CRITERION"
    else:
        out = "PASS"
    return {"outcome": out, "counted": (out == "PASS"), "primary_label": primary,
            "secondary_label": secondary, "p1": bool(p1), "p1b": bool(p1b),
            "p2": bool(p2), "p3": bool(p3)}


def main():
    rep = {"checks": [], "fails": []}

    def chk(cid, ok, note=""):
        rep["checks"].append({"id": cid, "ok": bool(ok), "note": note})
        if not ok:
            rep["fails"].append(f"{cid}: {note}")
        print(f"  [{cid}] {'OK' if ok else '★ FAIL'}  {note}")

    print("=== 0. 固定项 ===")
    chk("ADJ-43-1", md5(CARD) == CARD_MD5, f"卡 md5 == {CARD_MD5[:8]}…（rev 1.0 冻结）")

    # ══════════════════════════ 1. 读分片（**只读 jsonl**，不读探针产物）
    rows_by_arm, read_files = {}, []
    for arm in ARMS:
        rows = []
        for sh in range(SHARDS):
            p = f"{OUT}/DIR43r10_arm_{arm}_shard{sh}.jsonl"
            read_files.append(p)
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln:
                        rows.append(json.loads(ln))
        # 去重（同一 idx 只取第一条）⇒ 续跑可能重复追加
        seen, uniq = set(), []
        for r in sorted(rows, key=lambda r: r.get("idx", -1)):
            if r.get("idx") in seen:
                continue
            seen.add(r.get("idx"))
            uniq.append(r)
        rows_by_arm[arm] = uniq
        print(f"  [{arm}] {len(uniq)} / {N_EXPECT[arm]}")
    rep["read_files"] = read_files

    print("\n=== 1. 完整性（卡 §4 第 1 支）===")
    n_ok = True
    for arm in ARMS:
        h = len(rows_by_arm[arm])
        if h != N_EXPECT[arm]:
            n_ok = False
            print(f"    ★ {arm}: {h} ≠ {N_EXPECT[arm]}")
    chk("ADJ-43-2", n_ok, "三臂实例数齐（learned/direct=400、rand=100）")
    if not n_ok:
        print("★ **尚未跑完** ⇒ **不裁决**（判据「从未被计算」不得记 PASS/FAIL）")
        rep["verdict"] = "NOT-COMPUTED"
        rep["n_have"] = {a: len(rows_by_arm[a]) for a in ARMS}
        json.dump(rep, open(f"{OUT}/DIR43r10_adjudication.json", "w"), indent=1)
        return 2

    # seed↔idx 一致性（**从行里反推 SEED0**，不外引）
    L = rows_by_arm["learned"]
    R = rows_by_arm["rand"]
    D = rows_by_arm["direct"]
    s0 = {r["seed"] - r["idx"] for r in L}      # ⇒ 恒等于 SEED0 + INST_OFF0
    seed0 = (next(iter(s0)) - INST_OFF0) if len(s0) == 1 else None
    chk("ADJ-43-3", seed0 is not None and len(s0) == 1,
        f"learned 臂反推 SEED0 = {seed0}（唯一）")
    bad = []
    for arm, off0, offn in (("learned", INST_OFF0, INST_OFFN), ("rand", CTRL_OFF0, CTRL_OFFN),
                            ("direct", INST_OFF0, INST_OFFN)):
        for r in rows_by_arm[arm]:
            if not (isinstance(r.get("idx"), int) and 0 <= r["idx"] < (offn - off0)) \
                    or r.get("seed") != seed0 + off0 + r["idx"]:
                bad.append(f"{arm}/{r.get('idx')}")
    chk("ADJ-43-3b", not bad, f"seed↔idx 一致（坏 {len(bad)}）")

    print("\n=== 2. 统计量（独立重算）===")
    rho = [r["rho_final"] for r in L]
    rho0 = [r["rho0"] for r in L]
    rhor = [r["rho_rand_final"] for r in L]
    rhor0 = [r["rho_rand0"] for r in L]
    st_rho, st_rho0 = stats(rho), stats(rho0)
    st_diff, st_diff0 = paired(rho, rhor), paired(rho0, rhor0)
    st_dmin = stats([r["direct_cos_min"] for r in D])
    st_dmean = stats([r["direct_cos_mean"] for r in D])
    finite = all(math.isfinite(x) for x in
                 [st_rho["mean"], st_rho["se"], st_rho0["mean"], st_rho0["se"],
                  st_diff["mean"], st_diff["se"], st_dmin["mean"], st_dmean["mean"]])
    print(f"  ρ(终态) mean={st_rho['mean']:+.6f} 3se={st_rho['3se']:.6f} (n={st_rho['n']})")
    print(f"  ρ(初值) mean={st_rho0['mean']:+.6f} 3se={st_rho0['3se']:.6f}")
    print(f"  ρ−ρ_rand mean={st_diff['mean']:+.6f} 3se={st_diff['3se']:.6f}")
    print(f"  direct cos: mean={st_dmean['mean']:.6f}  min_i={st_dmin['mean']:.6f}  (阈 {COS_DIRECT})")
    print(f"  ρ<0 的实例占比 = {sum(1 for x in rho if x < 0) / len(rho):.3f}")

    print("\n=== 3. 卡 §4 判定表（按序命中）===")
    v = {"n_ok": True, "finite": bool(finite),
         "rho_mean": st_rho["mean"], "rho_se": st_rho["se"],
         "rho0_mean": st_rho0["mean"], "rho0_se": st_rho0["se"],
         "rho_diff_mean": st_diff["mean"], "rho_diff_se": st_diff["se"],
         "direct_mean": st_dmean["mean"], "direct_min": st_dmin["mean"]}
    dec = decide(v)
    print(f"  ★ outcome = **{dec['outcome']}**  counted={dec['counted']}  "
          f"primary={dec['primary_label']}  secondary={dec['secondary_label']}")
    print(f"    P1={dec['p1']}  P1'={dec['p1b']}  P2={dec['p2']}  P3={dec['p3']}")

    print("\n=== 4. 与实现的 merge 输出逐字段对照 ===")
    xcheck = {}
    if os.path.exists(FORMAL):
        iv = json.load(open(FORMAL))
        pairs = [("outcome", dec["outcome"], iv.get("outcome")),
                 ("counted", dec["counted"], iv.get("counted")),
                 ("primary_label", dec["primary_label"], iv.get("primary_label")),
                 ("secondary_label", dec["secondary_label"], iv.get("secondary_label")),
                 ("P1", dec["p1"], iv.get("P1_signif")),
                 ("P1b", dec["p1b"], iv.get("P1b_signif_secondary")),
                 ("P2", dec["p2"], iv.get("P2_specific")),
                 ("P3", dec["p3"], iv.get("P3_nonempty")),
                 ("rho_mean", st_rho["mean"], iv.get("rho_primary", {}).get("mean")),
                 ("rho_3se", st_rho["3se"], iv.get("rho_primary", {}).get("3se")),
                 ("rho_diff_mean", st_diff["mean"], iv.get("rho_diff_primary", {}).get("mean")),
                 ("direct_mean", st_dmean["mean"], iv.get("direct_cos_mean_stats", {}).get("mean"))]
        for name, mine, theirs in pairs:
            ok = (abs(float(mine) - float(theirs)) < 1e-12) if isinstance(mine, float) else (mine == theirs)
            xcheck[name] = {"mine": mine, "impl": theirs, "same": bool(ok)}
            print(f"  {name:16s} 裁决器={mine}  实现={theirs}  {'OK' if ok else '★ 不一致'}")
        chk("ADJ-43-4", all(x["same"] for x in xcheck.values()),
            f"与 merge 逐字段对照 {sum(x['same'] for x in xcheck.values())}/{len(xcheck)} 一致")
    else:
        print("  （merge 尚未运行 ⇒ 本轮不做对照）")

    rep.update({"outcome": dec["outcome"], "counted": dec["counted"],
                "primary_label": dec["primary_label"], "secondary_label": dec["secondary_label"],
                "P1": dec["p1"], "P1b": dec["p1b"], "P2": dec["p2"], "P3": dec["p3"],
                "rho_primary": st_rho, "rho_secondary": st_rho0,
                "rho_diff_primary": st_diff, "rho_diff_secondary": st_diff0,
                "direct_cos_min_stats": st_dmin, "direct_cos_mean_stats": st_dmean,
                "rho_negative_frac": sum(1 for x in rho if x < 0) / len(rho),
                "seed0": seed0, "impl_xcheck": xcheck})
    json.dump(rep, open(f"{OUT}/DIR43r10_adjudication.json", "w"), indent=1)
    print(f"\n已写 {OUT}/DIR43r10_adjudication.json")
    print("=" * 66)
    if rep["fails"]:
        print("★ 裁决器自身有 FAIL：")
        for f in rep["fails"]:
            print("   -", f)
        return 1
    print(f"★ 裁决器完成：outcome = {dec['outcome']}（counted={dec['counted']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
