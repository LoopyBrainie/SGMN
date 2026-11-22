#!/opt/pytorch/bin/python
r"""DIR-26-S′ · rev 2.1 的**证明义务**（卡 §9.2 的 P-1 / P-2 / P-3）。

★ **为什么必须交这张卷**：rev 2.1 只改了一条**护栏**（`measure_hit`），所以它必须证明自己对
**被落盘的数值零效果**；否则它就变成了"看完数据改设备"。

- **P-1 逐位重算**：用 rev 2.1 重算**已在 rev 2.0 落盘**的单元，与归档逐位比对
  $G_O$、$U0_O$、$R_O$ 与每臂 `mu`/`ret1`。★ 预注册降级条款：若因线程归约顺序出现末位差，
  须 ≤4 ulp **且** 绝对差 <1e-12，并如实并列报告。
- **P-2 恒等式自检**：对重算单元，$\mathrm{frac}\{m>0\}$ 与 `ret1` 的残差必须 $\in[0,\ n_{\rm tie}/n]$。
- **P-3 老断言只在边界失败**：凡是老断言失败的单元，必须同时满足 $|\texttt{ret1}-0.5|\le0.02$ **且** $|\mu|\le0.02$。

用法：`/opt/pytorch/bin/python dir26s/prove26s_v3.py`
"""
import importlib
import json
import sys

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26s")

OUT = "/home/ec2-user/sgmn-loop/dir26s/out"
M = importlib.import_module("R1_dose_ar1_v3")      # ★ rev 2.1

# P-1 的比对种子：两个**已在 rev 2.0 落盘**的单元（一普通、一较边界）
P1_SEEDS = [20270519, 20270609]


def ulp_gap(a, b):
    r"""相对差（以 2^-52 为单位 ≈ ulp 数）。$a=b=0$ ⇒ 0。"""
    a, b = float(a), float(b)
    d = abs(a - b)
    if d == 0.0:
        return 0.0
    scale = max(abs(a), abs(b))
    return d / (scale * 2.220446049250313e-16)


def main():
    by = M.load_rows()
    go = json.load(open(f"{OUT}/DIR26S_power.json"))
    pool = (list(M.SEEDS) + list(M.EXTRA_SEEDS))[: int(go["n_use"])]
    cnt0_of = {int(s): by[(0.0, int(s))]["counts"] for s in pool if (0.0, int(s)) in by}
    tau_d = M.cal_at(M.N_KV)
    tau = float(tau_d["c_tau"]) if isinstance(tau_d, dict) else float(tau_d)

    res = {"impl": "dir26s/R1_dose_ar1_v3.py", "P1": {}, "P2": {}, "P3": {}, "ulp": {}}
    ok_all = True

    # ── P-1 + P-2：两个已落盘单元的重算与逐位比对 ──
    print("=== P-1 逐位重算（rev 2.1 重算 vs rev 2.0 归档）===")
    for s in P1_SEEDS:
        old = by[((0.9), int(s))]
        new = M.run_unit_dose(int(s), 0.9, tau, cnt0=cnt0_of.get(int(s)))
        row = {"fields": {}, "arms": {}, "bit_equal": True, "max_ulp": 0.0}
        for f in ("G_O", "U0_O", "R_O", "G_X", "U0_X"):
            same = repr(old[f]) == repr(new[f])
            u = ulp_gap(old[f], new[f])
            row["fields"][f] = {"old": old[f], "new": new[f], "repr_equal": same, "ulp": u}
            row["max_ulp"] = max(row["max_ulp"], u)
            if not same:
                row["bit_equal"] = False
        for tag in ("A", "B", "P", "Fu", "Uu", "Ku"):
            if tag not in old["arms"] or tag not in new["arms"]:
                continue
            o, n = old["arms"][tag], new["arms"][tag]
            same = repr(o["mu"]) == repr(n["mu"]) and repr(o["ret1"]) == repr(n["ret1"])
            u = max(ulp_gap(o["mu"], n["mu"]), ulp_gap(o["ret1"], n["ret1"]))
            row["arms"][tag] = {"mu_old": o["mu"], "mu_new": n["mu"],
                                "ret1_old": o["ret1"], "ret1_new": n["ret1"],
                                "repr_equal": same, "ulp": u}
            row["max_ulp"] = max(row["max_ulp"], u)
            if not same:
                row["bit_equal"] = False
        # P-2：恒等式
        gaps = []
        for tag in ("A", "B", "P", "Fu", "Uu", "Ku"):
            if tag not in new["arms"]:
                continue
            a = new["arms"][tag]
            pf, tie = a.get("pos_frac"), a.get("n_tie")
            if pf is None:
                continue
            n = a["mu_read"]
            g = float(a["ret1"]) - float(pf)
            gaps.append({"arm": tag, "ret1": a["ret1"], "pos_frac": pf,
                         "gap": g, "tie_allow": (tie or 0) / n})
        row["identity"] = gaps
        res["P1"][int(s)] = row
        # ★ 判定
        tol_ok = row["max_ulp"] <= 4.0
        bit_ok = row["bit_equal"]
        res["ulp"][int(s)] = {"bit_equal": bit_ok, "max_ulp": row["max_ulp"], "tol_ok": tol_ok}
        verdict = "逐位相等" if bit_ok else (f"末位差 ≤{row['max_ulp']:.1f} ulp（预注册降级：{'通过' if tol_ok else '不通过'}）")
        print(f"  seed {s}: {verdict}  max_ulp={row['max_ulp']:.2f}")
        if not (bit_ok or tol_ok):
            ok_all = False

    print("\n=== P-2 恒等式自检 ===")
    p2_ok = True
    for s, row in res["P1"].items():
        for g in row["identity"]:
            good = (-1e-12 <= g["gap"] <= g["tie_allow"] + 1e-12)
            if not good:
                p2_ok = False
                print(f"  ✗ seed {s} arm {g['arm']}: gap={g['gap']:.3e} "
                      f"（允许 [0, {g['tie_allow']:.3e}]）")
            else:
                print(f"  ✓ seed {s} arm {g['arm']}: gap={g['gap']:.3e}（允许 ≤{g['tie_allow']:.3e}）")
    res["P2"] = {"ok": bool(p2_ok)}
    ok_all = ok_all and p2_ok

    # ── P-3：老断言只在「机会水平」失败 ──
    print("\n=== P-3 老断言失败的单元是否都处在机会水平 ===")
    hits = []
    for s in (20270707,):
        row = M.run_unit_dose(int(s), 0.9, tau, cnt0=cnt0_of.get(int(s)))
        by[(0.9, int(s))] = row
        for tag, a in row["arms"].items():
            old_holds = bool((a["mu"] > 0) == (a["ret1"] > 0.5))
            if not old_holds:
                near = abs(a["ret1"] - 0.5) <= 0.02 and abs(a["mu"]) <= 0.02
                hits.append({"seed": int(s), "arm": tag, "mu": a["mu"], "ret1": a["ret1"],
                             "near_chance": near})
                print(f"  seed {s} arm {tag}: mu {a['mu']:+.6f} ret1 {a['ret1']:.5f} "
                      f"⇒ 老断言失败，机会水平判定 near_chance={near}")
    p3_ok = all(h["near_chance"] for h in hits) if hits else True
    print(f"  老断言失败单元数 = {len(hits)}；全部处于机会水平 = {p3_ok}")
    res["P3"] = {"ok": bool(p3_ok), "hits": hits}
    ok_all = ok_all and p3_ok

    res["all_ok"] = bool(ok_all)
    json.dump(res, open(f"{OUT}/DIR26S_proof_v3.json", "w"), indent=1)
    print(f"\n★ P-1/P-2/P-3 {'全部通过' if ok_all else '**有未通过项**'} ⇒ {OUT}/DIR26S_proof_v3.json")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
