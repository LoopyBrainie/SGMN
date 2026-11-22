#!/opt/pytorch/bin/python
r"""DIR-26-S′ · **独立裁决器**（不 import 实现）—— 从**持久行文件**重新算出冻结判据与通道。

★ **为什么要独立**：实现的 `crit()` 与裁决器**同源** ⇒ 若实现里有错，裁决器跟着错 ⇒ ⇒
  本文件**只读** `DIR26S_rows.json`（持久工件）+ D-43 归档 + **冻结卡**，**自己**重算
  §1.2 的通道表、§2.1 的阈值、§4 的护栏 ⇒ ⇒ 再与实现的 `DIR26S_verdict.json` **逐字段对照**。

★ **只读 / 不训练**：读 JSON，算统计量，写一份 `DIR26S_adjudication.json`。
★ **不提前裁决**：配对数 < `n_use` ⇒ **拒绝出裁决**（打印「尚未跑完」）⇒ 纪律：判据「从未被计算」≠ PASS/FAIL。

用法（**在 trainer 上**）：
    /opt/pytorch/bin/python dir26s/adjudicate26s.py
"""
import hashlib
import json
import math
import os

import numpy as np

ROOT = "/home/ec2-user/sgmn-loop"
OUT = f"{ROOT}/dir26s/out"
ROWS = f"{OUT}/DIR26S_rows.json"
POWER = f"{OUT}/DIR26S_power.json"
IMPL_VERDICT = f"{OUT}/DIR26S_verdict.json"
D43 = f"{ROOT}/dir26k/out/DIR26Kr10_formal.json"
CARD = f"{ROOT}/revolve/DIR-26S/DIRECTION_rev26s_v13.md"
IMPL = f"{ROOT}/dir26s/R1_dose_ar1_v2.py"

# ══════════════════════════ 冻结常数（**逐字抄自卡 rev 2.0 §1.1/§2.1/§5.2/§5.4/§5.5**）
CARD_MD5 = "d17a9e65e4d1a78a3c9257d724efba12"      # rev 2.0 冻结卡的 md5
DELTA_STAR = 0.00162                                # 卡 §1.1
GO_TOL = 0.00081                                    # 卡 §1.1/§2.1：δ*/2
SE_K = 3.0                                          # 卡 §1.2：3se
EDGE_TOL = 3e-4                                     # 卡 §4：A26S-3
RHO_HI, RHO_LO = 0.85, 0.10                         # 卡 §4：A26S-4
NU_CAP_MAX = 0.10                                   # 卡 §4：A26S-11
N_KV, N_READ = 60, 20
# ★ 卡 §5.5 的「`R1_gating12.SEEDS` 的前 30 个」（冻结、不得替换）；
#   本表由**D-43 归档**的 seed 字段反查核对（见 `ADJ-2`）⇒ 抄错就会当场暴露。
SEEDS30 = [20261028, 20261104, 7, 20261111, 20261118, 20261125, 20261202, 20261209, 20261216,
            20261223, 20261230, 20270106, 20270113, 20270120, 20270127, 20270203, 20270210,
            20270217, 20270224, 20270303, 20270310, 20270317, 20270324, 20270331, 20270407,
            20270414, 20270421, 20270428, 20270505, 20270512]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def eff(a, b):
    """配对差统计量（**独立重写**，不 import 实现的 `_eff`）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    mean = float(np.mean(d))
    return {"mean": mean, "sd": sd, "se": se, "k": SE_K * se, "n": n,
            "sig": bool(np.isfinite(se) and abs(mean) >= SE_K * se),
            "vals": [round(float(x), 8) for x in d]}


def main():
    rep = {"checks": [], "fails": []}

    def chk(cid, ok, note=""):
        rep["checks"].append({"id": cid, "ok": bool(ok), "note": note})
        if not ok:
            rep["fails"].append(f"{cid}: {note}")
        print(f"  [{cid}] {'OK' if ok else '★ FAIL'}  {note}")

    print("=== 0. 固定项（卡 / 实现指纹）===")
    chk("ADJ-1", md5(CARD) == CARD_MD5, f"卡 md5 == {CARD_MD5[:8]}…（rev 2.0 冻结）")
    rep["impl_md5"] = md5(IMPL)

    by = {}
    for r in json.load(open(ROWS))["rows"]:
        by[(float(r["phi"]), int(r["seed"]))] = r
    print(f"  已落盘行 {len(by)}")

    go = json.load(open(POWER)) if os.path.exists(POWER) else {}
    n_use = int(go.get("n_use") or 0)

    # ── 种子表核对：实现用的前 30 个种子（从归档 D-43 的 seed 字段反查）必须等于 SEEDS30
    d43 = json.load(open(D43))
    ref = {(int(r["n_kv"]), int(r["n_read"]), int(r["seed"])): r
           for r in d43["rows"] if int(r["n_kv"]) == N_KV}
    d43_seeds = sorted({s for (k, n, s) in ref if n == N_READ})
    chk("ADJ-2", d43_seeds == sorted(SEEDS30),
        f"D-43 归档的 30 个种子 == 裁决器种子表（{len(d43_seeds)} 个，**按值比较、不看顺序**）")

    # ── 配对（同一 seed、同一 cnt0，只换 φ）
    have0 = sorted(s for (p, s) in by if p == 0.0)
    have9 = sorted(s for (p, s) in by if p == 0.9)
    paired = sorted(set(have0) & set(have9))
    rep["n_rows"] = {"phi0": len(have0), "phi9": len(have9), "paired": len(paired), "n_use": n_use}

    print("\n=== 1. 跑完没有？===")
    if not paired or (n_use and len(paired) < n_use):
        print(f"★ **尚未跑完**（配对 {len(paired)} / 需要 {n_use}）⇒ **不裁决**"
              f"（判据「从未被计算」不得记 PASS/FAIL）")
        rep["verdict"] = "NOT-COMPUTED"
        json.dump(rep, open(f"{OUT}/DIR26S_adjudication.json", "w"), indent=1)
        return 2

    print("\n=== 2. 判据（卡 §1.2 主判据：|ΔG| ≥ 3se ⇒ DOS；< 3se ⇒ INV）===")
    dG = eff([by[(0.9, s)]["G_O"] for s in paired], [by[(0.0, s)]["G_O"] for s in paired])
    dU = eff([by[(0.9, s)]["U0_O"] for s in paired], [by[(0.0, s)]["U0_O"] for s in paired])
    dR = eff([by[(0.9, s)]["R_O"] for s in paired], [by[(0.0, s)]["R_O"] for s in paired])
    print(f"  ΔG  = {dG['mean']:+.6f}  sd={dG['sd']:.6f}  se={dG['se']:.6f}  3se={dG['k']:.6f}  "
          f"n={dG['n']}  ⇒ |ΔG| {'≥' if dG['sig'] else '<'} 3se")
    print(f"  ΔμU0= {dU['mean']:+.6f}  3se={dU['k']:.6f}  ⇒ sig={dU['sig']}")
    print(f"  ΔRμ = {dR['mean']:+.6f}  3se={dR['k']:.6f}  ⇒ sig={dR['sig']}")

    print("\n=== 3. 护栏（卡 §4；任一 FAIL ⇒ IND，不判）===")
    # A26S-6 位级锚点（原 30 种子 × {G_O,U0_O,R_O} = 90 值对，零容差）
    ncmp, bad = 0, []
    for s in SEEDS30:
        r = by.get((0.0, int(s)))
        if r is None:
            bad.append(f"{s}:missing")
            continue
        R = ref[(N_KV, N_READ, int(s))]
        for f in ("G_O", "U0_O", "R_O"):
            ncmp += 1
            if r[f] != R[f]:
                bad.append(f"{s}.{f}: {r[f]!r} != {R[f]!r}")
    chk("A26S-6", (not bad and ncmp == 3 * len(SEEDS30)), f"位级锚点 {ncmp} 值对，坏 {len(bad)}")
    # A26S-3 边缘矩
    edge_max, edge_worst = 0.0, None
    for s in paired[:10]:
        d0, d9 = by[(0.0, s)]["diag"], by[(0.9, s)]["diag"]
        for f in ("k2", "cosp", "k4"):
            v = abs(d9["km"][f] - d0["km"][f]) / max(abs(d0["km"][f]), 1e-30)
            if v > edge_max:
                edge_max, edge_worst = v, f"{s}.{f}"
    chk("A26S-3", edge_max <= EDGE_TOL, f"边缘矩最大相对差 {edge_max:.3e} ≤ {EDGE_TOL:g}（最坏 {edge_worst}）")
    # A26S-4 ρ̄_u 带
    ru0 = [by[(0.0, s)]["diag"]["rho_u"] for s in paired]
    ru9 = [by[(0.9, s)]["diag"]["rho_u"] for s in paired]
    chk("A26S-4", (min(ru9) >= RHO_HI and max(ru0) <= RHO_LO),
        f"ρ̄_u: φ=0 max {max(ru0):.4f} ≤ {RHO_LO}；φ=0.9 min {min(ru9):.4f} ≥ {RHO_HI}")
    # A26S-8/9/10 恒等 + 配额 + 镜像 + 调度
    ident = all(by[k]["arms"][t]["sum_ok"] and by[k]["arms"][t]["quota_ok"]
                for k in by for t in ("Fu", "Uu", "Ku"))
    chk("A26S-8/9", ident, "三臂 sum_ok ∧ quota_ok 全真")
    chk("A26S-10", all(by[k]["mirror_ok"] and by[k]["sched_dist"] > 0 for k in by),
        "mirror_ok ∧ sched_dist>0 全真")
    # 卡 §6-2：φ=0.9 的配额必须来自 φ=0 的 B 臂
    chk("A26S-7", all(by[(0.9, s)]["cnt_source"] == "frozen_cnt0" for s in have9)
        and all(by[(0.0, s)]["cnt_source"] == "B@this_phi" for s in have0),
        "cnt0 来源：φ=0.9 ⇒ frozen_cnt0；φ=0 ⇒ B@this_phi")
    # 口径 O 一致性
    chk("A26S-12", all(by[k]["arms"][t]["mu_ret1_consistent"] for k in by for t in ("Fu", "Uu", "Ku")),
        "μ 与 ret1 一致（口径 O）全真")
    # ν 上限
    chk("A26S-11", all(by[k]["nu_cap"] <= NU_CAP_MAX for k in by), f"ν_cap ≤ {NU_CAP_MAX} 全真")

    guards_ok = all(c["ok"] for c in rep["checks"] if c["id"].startswith("A26S"))

    print("\n=== 4. 通道（卡 §1.2 顺序表）===")
    if not go.get("go"):
        chan = "INFEASIBLE-POWER"
    elif not guards_ok:
        chan = "IND"
    elif dG["sig"]:
        chan = "DOS"
    else:
        chan = "INV"
    print(f"  ★ 通道 = **{chan}**（Go={go.get('go')} guards={guards_ok} |ΔG|≥3se={dG['sig']}）")
    labels = {"RATIO-REPORT": bool(dR["sig"]),
              "SCALE-REPORT": bool((not dR["sig"]) and dU["sig"])}
    print(f"  报告标签（非通道、不判）：{labels}")

    print("\n=== 5. 透明度（卡 §5.4 DEF-26S-15：全 n 对的 sd 反算 Go 条件）===")
    go_back = SE_K * dG["sd"] / math.sqrt(dG["n"])
    print(f"  用全 {dG['n']} 对的 sd(ΔG)={dG['sd']:.6f} 反算：3se={go_back:.6f} vs 门限 {GO_TOL}"
          f" ⇒ {'**仍成立**' if go_back <= GO_TOL else '**不再成立**'}")
    print(f"  （这是**报告字段**、不是通道定义：Go 是跑前功效门、主判据是跑后检验，两者不互相追溯改写）")

    print("\n=== 6. 水平表（卡 §2 M-2）===")
    lv = {}
    for p, tag in ((0.0, "phi0"), (0.9, "phi9")):
        G_ = [by[(p, s)]["G_O"] for s in paired]
        U_ = [by[(p, s)]["U0_O"] for s in paired]
        R_ = [by[(p, s)]["R_O"] for s in paired]
        lv[tag] = {"G_mean": float(np.mean(G_)), "G_sd": float(np.std(G_, ddof=1)),
                   "U0_mean": float(np.mean(U_)), "R_mean": float(np.mean(R_)),
                   "G_min": float(np.min(G_)), "G_max": float(np.max(G_)),
                   "rho_u_mean": float(np.mean([by[(p, s)]["diag"]["rho_u"] for s in paired])),
                   "qbar_mean": float(np.mean([by[(p, s)]["qbar"] for s in paired]))}
        print(f"  {tag}: Ḡ={lv[tag]['G_mean']:+.6f} (sd {lv[tag]['G_sd']:.6f})  "
              f"μ̄U0={lv[tag]['U0_mean']:+.6f}  R̄μ={lv[tag]['R_mean']:+.4f}  "
              f"ρ̄u={lv[tag]['rho_u_mean']:.4f}  q̄={lv[tag]['qbar_mean']:.2f}")

    def by_id(cid):
        return next(c["ok"] for c in rep["checks"] if c["id"] == cid)

    print("\n=== 7. 与实现的 verdict.json 逐字段对照 ===")
    xcheck = {}
    if os.path.exists(IMPL_VERDICT):
        iv = json.load(open(IMPL_VERDICT))
        pairs = [("channel", chan, iv.get("channel")),
                 ("dG.mean", dG["mean"], iv.get("dG", {}).get("mean")),
                 ("dG.k", dG["k"], iv.get("dG", {}).get("k")),
                 ("dG.n", dG["n"], iv.get("dG", {}).get("n")),
                 ("dG.sig", dG["sig"], iv.get("dG", {}).get("sig")),
                 ("dU0.k", dU["k"], iv.get("dU0", {}).get("k")),
                 ("dR.k", dR["k"], iv.get("dR", {}).get("k")),
                 ("guards.a_anchor", by_id("A26S-6"), iv.get("a_anchor", {}).get("ok")),
                 ("guards.c_rho", next(c["ok"] for c in rep["checks"] if c["id"] == "A26S-4"),
                  iv.get("c_rho", {}).get("ok"))]
        for name, mine, theirs in pairs:
            ok = (abs(mine - theirs) < 1e-12) if isinstance(mine, float) else (mine == theirs)
            xcheck[name] = {"mine": mine, "impl": theirs, "same": bool(ok)}
            print(f"  {name:16s} 裁决器={mine}  实现={theirs}  {'OK' if ok else '★ 不一致'}")
        chk("ADJ-3", all(v["same"] for v in xcheck.values()),
            f"与实现逐字段对照 {sum(v['same'] for v in xcheck.values())}/{len(xcheck)} 一致")
    else:
        print("  （实现的 verdict.json 尚未生成 ⇒ 本轮不做对照）")

    rep.update({"channel": chan, "guards_ok": bool(guards_ok), "dG": dG, "dU0": dU, "dR": dR,
                "levels": lv, "report_labels": labels,
                "go_back_from_full_n": {"se3": go_back, "tol": GO_TOL,
                                        "still_holds": bool(go_back <= GO_TOL)},
                "impl_xcheck": xcheck, "paired_seeds": paired, "go_file": go})
    json.dump(rep, open(f"{OUT}/DIR26S_adjudication.json", "w"), indent=1)
    print(f"\n已写 {OUT}/DIR26S_adjudication.json")
    print("=" * 66)
    if rep["fails"]:
        print("★ 裁决器自身有 FAIL：")
        for f in rep["fails"]:
            print("   -", f)
        return 1
    print(f"★ 裁决器完成：通道 = {chan}（护栏 {guards_ok}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
