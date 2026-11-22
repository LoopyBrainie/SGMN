"""`R1_task120.py` —— `DIR-120` 的 Stage A 与正式跑（**纯再分析，不训练、不必加载模型**）。

    /opt/pytorch/bin/python R1_task120.py --stage a     # Stage A（8 条守卫，硬门）
    /opt/pytorch/bin/python R1_task120.py --stage formal # 正式跑（判据 J120-1..5）

★ 读数**只读** `dir119/out/cells/COLOCATED_600_*.json`（不改动它）。
★ 需要 tokenizer（重建题目以反查值身份）⇒ 必须用 `D90.model_and_tok()`；但**不需要**模型前向。
★ 一个结论对应一个产物：`out/stage_stageA.json`、`out/formal.json`。
"""
import argparse
import json
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "dir119"))
sys.path.insert(0, str(_HERE.parent / "dir118"))

import R1_task118 as P118          # noqa: E402
import R1_task90 as D90            # noqa: E402
import R1_task104 as D104          # noqa: E402
import mech120 as M                # noqa: E402

OUT119 = _HERE.parent / "dir119" / "out"
OUT = _HERE / "out"
ARM = "COLOCATED"
J = "10000"
B_NULL = 20000
SEED_NULL = 20261122
SEEDS_EXPECT = 9


def load():
    """读 9 个 cell + 重建题目。返回 (seeds, pick, prior_pick, ok, meta, tv, tgt, per_seed)。"""
    cells = sorted((OUT119 / "cells").glob(f"{ARM}_600_*.json"))
    seeds, pick, prior_pick, ok, per_seed = [], [], [], [], {}
    for f in cells:
        d = json.loads(f.read_text(encoding="utf-8"))
        me = (d.get("mech_by_mult") or {}).get(J)
        if not me or not me.get("finite"):
            raise SystemExit(f"cell {f.name} 缺 mech_by_mult[{J}] 或非有限 ⇒ 测量对象不存在")
        s = int(d["seed"])
        seeds.append(s)
        pick.extend(int(x) for x in me["pick_argmin"])
        ok.extend(int(x) for x in me["ok"])
        prior_pick.extend(int(x) for x in (d["prior_mech"] or {}).get("pick_argmin", []))
        per_seed[s] = {"pick": [int(x) for x in me["pick_argmin"]],
                       "ok": [int(x) for x in me["ok"]],
                       "prior_acc": float(d["prior_mech"]["acc_argmin"]),
                       "acc_by_mult_J": float(d["acc_by_mult"][J])}
    tok, _model = D90.model_and_tok()
    _facts, _parts, _csf, IT = P118.ground(tok, ARM)
    items, meta, tv = IT["items_ev"], IT["meta_ev"], IT["tv"]
    tgt = [int(x) for x in D104.targets_of(items)]
    # 9 个 seed 共用同一份 items_ev ⇒ 逐 seed 展开成 432 项
    n = len(items)
    meta432 = [meta[i % n] for i in range(len(pick))]
    tgt432 = [tgt[i % n] for i in range(len(pick))]
    return {"seeds": seeds, "pick": pick, "prior_pick": prior_pick, "ok": ok,
            "meta": meta432, "tgt": tgt432, "tv": tv,
            "per_seed": per_seed, "n_items": n}


def stage_a(D):
    """8 条守卫（卡片 §7）。任一失败 ⇒ 不得进 Formal。"""
    res = []

    def g(tag, cond, detail=""):
        res.append({"tag": tag, "pass": bool(cond), "detail": detail})
        print(f"  [{'PASS' if cond else 'FAIL'}] {tag}{('  ' + detail) if detail else ''}")

    n_expect = SEEDS_EXPECT * D["n_items"]
    g("A120-1 9 个 cell × 48 项，`pick_argmin`/`ok` 长度齐全",
      len(D["seeds"]) == SEEDS_EXPECT and len(D["pick"]) == n_expect
      and len(D["prior_pick"]) == n_expect and len(D["ok"]) == n_expect,
      f"seeds={len(D['seeds'])} n={len(D['pick'])}")

    # 逐项算 n_seen 直方图（只用第一份 items 的一次）
    hist = {str(k): 0 for k in range(4)}
    for i in range(D["n_items"]):
        hist[str(M._n_seen_of(D["meta"][i]["cand_val"], D["tgt"][i], D["tv"]))] += 1
    g("A120-2a 重建题目的 n_seen_hist == {0:2,1:23,2:20,3:3}",
      hist == {"0": 2, "1": 23, "2": 20, "3": 3}, str(hist))
    correct_vals = {D["meta"][i]["cand_val"][D["tgt"][i]] for i in range(D["n_items"])}
    g("A120-2b 8 个正确值全部 ∉ tv", all(v not in D["tv"] for v in correct_vals),
      f"{len(correct_vals)} 个正确值")

    g("A120-3 全部 seed 的 prior_mech.acc_argmin == 18/48",
      all(abs(v["prior_acc"] - 18.0 / 48.0) < 1e-12 for v in D["per_seed"].values()),
      str(sorted({round(v['prior_acc'], 6) for v in D['per_seed'].values()})))

    bad = []
    for s, v in D["per_seed"].items():
        if abs(sum(v["ok"]) / float(len(v["ok"])) - v["acc_by_mult_J"]) > 1e-12:
            bad.append(s)
    g("A120-4 `mean(ok)` == `acc_by_mult[10000]`（逐 seed 全等）", not bad, f"不符 seed={bad}")

    part = M.partition(D["pick"], D["meta"], D["tv"], D["tgt"])
    g("A120-5 ★ 恒等式 `E_total == E_n0 + E_esc_ok`（真实数据，整数恒等）",
      part["E_total"] == part["E_n0"] + part["E_esc_ok"],
      f"{part['E_total']} = {part['E_n0']} + {part['E_esc_ok']}")

    hand = M.partition([0, 0, 1, 3],
                       [{"cand_val": c} for c in
                        [["a", "b", "c", "d"], ["p", "q", "r", "s"],
                         ["a", "e", "f", "g"], ["a", "b", "c", "d"]]],
                       {"a", "d"}, [1, 0, 1, 1])
    q0_hand = M.q0_of(hand, [{"cand_val": c} for c in
                             [["a", "b", "c", "d"], ["p", "q", "r", "s"],
                              ["a", "e", "f", "g"], ["a", "b", "c", "d"]]],
                      {"a", "d"}, [1, 0, 1, 1])
    hand_ok = (hand["n_has_seen"] == 3 and hand["escape"] == 1
               and abs(hand["r"] - 1 / 3) < 1e-12 and hand["E_total"] == 2
               and hand["E_n0"] == 1 and hand["E_esc_ok"] == 1
               and M.n11_of([0, 0, 1, 3], [0, 0, 0, 0], [1, 0, 1, 1]) == 1
               and abs(M.q_of(hand) - 1.0) < 1e-12
               and abs(q0_hand - 1 / 3) < 1e-12
               and M.top3_share(hand["seen_vals"]) == 1.0
               and M.top3_share(hand["unseen_vals"]) == 1.0)
    g("A120-6 §3 手算例 11 个读数逐位相等（含 rev1.1 的 q0=1/3）", hand_ok)

    a1, o1 = M.null_top3(D["pick"], D["meta"], D["tv"], "seen", B=200, seed=SEED_NULL)
    a2, o2 = M.null_top3(D["pick"], D["meta"], D["tv"], "seen", B=200, seed=SEED_NULL)
    g("A120-7 置换零假设同种子两次逐位相同", (a1.shape == a2.shape) and bool((a1 == a2).all()) and o1 == o2)

    n_seen_pick = len(part["seen_vals"])
    g("A120-8 `#(选中见过值) == 372`（D-150 §7-3 已发表常数）", n_seen_pick == 372,
      f"实测 {n_seen_pick}")

    allok = all(r["pass"] for r in res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage_stageA.json").write_text(
        json.dumps({"arm": ARM, "mult": J, "guards": res, "all_pass": allok,
                    "n": part["n"], "n_has_seen": part["n_has_seen"],
                    "escape": part["escape"], "E_total": part["E_total"],
                    "E_n0": part["E_n0"], "E_esc_ok": part["E_esc_ok"],
                    "seen_picks": n_seen_pick},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n★ Stage A：{'全部 PASS' if allok else '有 FAIL ⇒ 不得进 Formal'}")
    return allok


def formal(D):
    t0 = time.time()
    part = M.partition(D["pick"], D["meta"], D["tv"], D["tgt"])
    n11 = M.n11_of(D["pick"], D["prior_pick"], D["tgt"])
    q, q0 = M.q_of(part), M.q0_of(part, D["meta"], D["tv"], D["tgt"])

    n_seen_arm, obs_seen = M.null_top3(D["pick"], D["meta"], D["tv"], "seen", B=B_NULL, seed=SEED_NULL)
    n_uns_arm, obs_uns = M.null_top3(D["pick"], D["meta"], D["tv"], "unseen", B=B_NULL, seed=SEED_NULL)
    s_seen = M.null_summary(n_seen_arm, obs_seen)
    s_uns = M.null_summary(n_uns_arm, obs_uns)

    j1 = part["r"] is not None and part["r"] >= 0.05
    j2 = part["E_total"] > 0 and (n11 / float(part["E_total"])) < 0.50
    j3 = (s_seen["q99"] is not None and obs_seen is not None and obs_seen > s_seen["q99"])
    j4 = (s_uns["q99"] is not None and obs_uns is not None and obs_uns > s_uns["q99"])
    j5 = (q is not None and q0 is not None and q >= 2.0 * q0)

    # C120-3 奇偶核（report-only）：前 4 / 后 5 个 seed 各算一遍归因与见侧集中
    par = {}
    for tag, ss in (("first4", D["seeds"][:4]), ("last5", D["seeds"][4:])):
        idxs = [k * D["n_items"] + i for k, s in enumerate(D["seeds"]) if s in ss
                for i in range(D["n_items"])]
        pk = [D["pick"][k] for k in idxs]
        pp = [D["prior_pick"][k] for k in idxs]
        tg = [D["tgt"][k] for k in idxs]
        me = [D["meta"][k] for k in idxs]
        p = M.partition(pk, me, D["tv"], tg)
        arr, ob = M.null_top3(pk, me, D["tv"], "seen", B=4000, seed=SEED_NULL)
        sm = M.null_summary(arr, ob)
        par[tag] = {"seeds": len(ss), "E_total": p["E_total"], "n11": M.n11_of(pk, pp, tg),
                    "n11_over_E": (M.n11_of(pk, pp, tg) / p["E_total"]) if p["E_total"] else None,
                    "r": p["r"], "top3_seen": ob, "q99_seen": sm["q99"],
                    "seen_gt_q99": bool(ob is not None and sm["q99"] is not None and ob > sm["q99"])}

    # report-only：16 个值的选中次数全表 + 逐 seed 分布
    import collections
    all_picks = collections.Counter()
    for i in range(part["n"]):
        all_picks[D["meta"][i]["cand_val"][D["pick"][i]]] += 1
    per_seed = {}
    for k, s in enumerate(D["seeds"]):
        sl = slice(k * D["n_items"], (k + 1) * D["n_items"])
        pk = D["pick"][sl]
        pp = D["prior_pick"][sl]
        tg = D["tgt"][sl]
        me = D["meta"][sl]
        p = M.partition(pk, me, D["tv"], tg)
        per_seed[str(s)] = {"escape": p["escape"], "E_total": p["E_total"],
                            "E_esc_ok": p["E_esc_ok"], "E_n0": p["E_n0"],
                            "r": p["r"], "n11": M.n11_of(pk, pp, tg)}

    out = {
        "arm": ARM, "mult": J, "n": part["n"], "n_has_seen": part["n_has_seen"],
        "partition": {k: part[k] for k in ("escape", "n_has_seen", "r", "E_total",
                                           "E_n0", "E_esc_ok")},
        "residual_share_of_432": part["E_esc_ok"] / float(part["n"]),
        "n11": n11, "n11_over_E_total": (n11 / float(part["E_total"])) if part["E_total"] else None,
        "q": q, "q0": q0, "q_over_q0": (q / q0) if (q and q0) else None,
        "seen_arm": {"obs": obs_seen, "null_mean": s_seen["mean"], "q99": s_seen["q99"], "B": B_NULL},
        "unseen_arm": {"obs": obs_uns, "null_mean": s_uns["mean"], "q99": s_uns["q99"], "B": B_NULL},
        "criteria": {"J120-1_r>=0.05": j1, "J120-2_n11/E<0.50": j2,
                     "J120-3_seen>q99": j3, "J120-4_unseen>q99": j4, "J120-5_q>=2q0": j5},
        "H120_verdict": bool(j1 and j2 and j3 and j5),
        "parity_C120-3": par,
        "value_counts": {v: c for v, c in all_picks.most_common()},
        "per_seed": per_seed,
        "r_with_432_denominator_C120-4": part["escape"] / float(part["n"]),
        "seed_null": SEED_NULL, "sec": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "formal.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n样本 {part['n']} 项（9 seed × 48），`n_has_seen`={part['n_has_seen']}")
    print(f"分区：逃逸 {part['escape']}（r={part['r']:.4f}）· E_total={part['E_total']}"
          f" = E_n0 {part['E_n0']} + E_esc_ok {part['E_esc_ok']}（残差占 432 的 "
          f"{out['residual_share_of_432']:.4f}）")
    print(f"n11={n11} ⇒ n11/E_total={out['n11_over_E_total']:.4f}")
    print(f"见侧 top3 = {obs_seen}（零假设 mean {s_seen['mean']:.4f}，q99 {s_seen['q99']:.4f}）")
    print(f"未见侧 top3 = {obs_uns}（零假设 mean {s_uns['mean']:.4f}，q99 {s_uns['q99']:.4f}）")
    print(f"q={q} q0={q0} ratio={out['q_over_q0']}")
    print(f"判据：J1={j1} J2={j2} J3={j3} J4={j4} J5={j5}")
    print(f"★ H120 = {out['H120_verdict']}")
    print(f"（耗时 {out['sec']}s）")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "formal"], required=True)
    a = ap.parse_args()
    D = load()
    if a.stage == "a":
        return 0 if stage_a(D) else 1
    formal(D)
    return 0


if __name__ == "__main__":
    sys.exit(main())