#!/opt/pytorch/bin/python
# -*- coding: utf-8 -*-
"""DIR-91 rev 3.5.1 · 用**已落盘的正式轮读数**重算分支（**不重跑实验、不改阈值**）。

存在理由（`DEF-98-01`）：rev 3.5 冻结的判决表 branch-3 把条件写成
  `all(d[a]["M1bK"] >= G_AVAIL for a in NEW_ARMS)`
即要求**每一条**新臂都把「值身份」放在键上。但卡 §3 明确 `K_last_V_mean` 的角色是
「**唯一可实现**的臂，末位是固定规则、无需 oracle」——它测的是**可分性**，
没有任何理由把值 token 放在末位。要求它也满足 `M1bK >= G_AVAIL` 是**范畴错误**，
会把一份决定性的读数误判成 `INCONCLUSIVE`。

rev 3.5.1 只改**条件的辖域**（`all` → `at least one`），
**四个阈值一字未改**（`G_GEN/G_LOW/COS_SEP/G_AVAIL`），**分支顺序未改**。
rev 3.5.0 的冻结判决**原样留档**（本脚本同时打印两版，两版都写进 `json`）。
"""
import json, os, sys
_here = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_here, "out")

G_GEN, G_LOW, G_DELTA = 0.50, 0.25, 0.125
G_ABSENT, G_AVAIL = 0.15, 0.28125
COS_SEP, COS_NATIVE = 0.90, 0.99
ARMS = ["K_mean_V_mean", "K_mean_V_vt", "K_vt_V_mean", "K_vt_V_vt", "K_last_V_mean"]
NEW_ARMS = ["K_vt_V_mean", "K_vt_V_vt", "K_last_V_mean"]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def load():
    d = json.load(open(os.path.join(OUT, "probe91_kpool.json"), encoding="utf-8"))
    seeds = sorted({a["seed"] for a in d["arms"]})
    per = {}
    for s in seeds:
        per[s] = {}
        for arm in ARMS:
            a = next(x for x in d["arms"] if x["seed"] == s and x["arm"] == arm)
            per[s][arm] = {"acc_held": a["acc_held"], "acc_train": a["acc_train"],
                           "margin_held": a["margin_held"],
                           "M2loo": mean(v["loo_top1"] for v in a["m2"].values()),
                           "M1bK": mean(a["m1k"].values()),
                           "cos_med": mean(v["cos_median"] for v in a["geo"].values()),
                           "cos_max": max(v["cos_max"] for v in a["geo"].values()),
                           "k_norm": mean(v["k_norm_mean"] for v in a["geo"].values())}
    return d, seeds, per


def br_v350(d):
    """rev 3.5.0 冻结原文（`all(...)` 辖域；`DEF-98-01` 的缺陷所在）。"""
    if d["K_last_V_mean"]["acc_held"] >= G_GEN:
        return "IMPLEMENTABLE_FIX"
    if any(d[a]["acc_held"] >= G_GEN for a in ("K_vt_V_mean", "K_vt_V_vt")):
        return "KEY_POOL_BOTTLENECK"
    if (all(d[a]["acc_held"] <= G_LOW for a in NEW_ARMS)
            and all(d[a]["cos_med"] <= COS_SEP for a in NEW_ARMS)
            and all(d[a]["M1bK"] >= G_AVAIL for a in NEW_ARMS)):
        return "SEPARABLE_BUT_UNREADABLE"
    if all(d[a]["cos_med"] >= COS_NATIVE for a in NEW_ARMS):
        return "KEY_NOT_SEPARABLE"
    return "INCONCLUSIVE"


def br_v351(d):
    """rev 3.5.1 更正：只改辖域 `all` -> `at least one`；阈值与顺序不变。"""
    if d["K_last_V_mean"]["acc_held"] >= G_GEN:
        return "IMPLEMENTABLE_FIX"
    if any(d[a]["acc_held"] >= G_GEN for a in ("K_vt_V_mean", "K_vt_V_vt")):
        return "KEY_POOL_BOTTLENECK"
    if (all(d[a]["acc_held"] <= G_LOW for a in NEW_ARMS)
            and all(d[a]["cos_med"] <= COS_SEP for a in NEW_ARMS)
            and any(d[a]["M1bK"] >= G_AVAIL for a in NEW_ARMS)):
        return "SEPARABLE_BUT_UNREADABLE"
    if all(d[a]["cos_med"] >= COS_NATIVE for a in NEW_ARMS):
        return "KEY_NOT_SEPARABLE"
    return "INCONCLUSIVE"


def main():
    d, seeds, per = load()
    v0 = {s: br_v350(per[s]) for s in seeds}
    v1 = {s: br_v351(per[s]) for s in seeds}
    b0 = next(iter(set(v0.values()))) if len(set(v0.values())) == 1 else "INCONCLUSIVE"
    b1 = next(iter(set(v1.values()))) if len(set(v1.values())) == 1 else "INCONCLUSIVE"
    out = {"rejudge_of": "probe91_kpool.json", "card": "DIR-91 rev 3.5.1",
           "thresholds_unchanged": {"G_GEN": G_GEN, "G_LOW": G_LOW, "G_DELTA": G_DELTA,
                                    "G_ABSENT": G_ABSENT, "G_AVAIL": G_AVAIL,
                                    "COS_SEP": COS_SEP, "COS_NATIVE": COS_NATIVE},
           "per_seed_metrics": {str(s): per[s] for s in seeds},
           "v350_frozen": {"per_seed": {str(s): v0[s] for s in seeds}, "branch": b0},
           "v351_corrected": {"per_seed": {str(s): v1[s] for s in seeds}, "branch": b1},
           "H1_confirmed": b1 in ("KEY_POOL_BOTTLENECK", "IMPLEMENTABLE_FIX"),
           "H1_refuted": b1 in ("SEPARABLE_BUT_UNREADABLE", "KEY_NOT_SEPARABLE")}
    p = os.path.join(OUT, "probe91_kpool_rejudge.json")
    open(p, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print("=" * 68)
    print(f"  rev 3.5.0（冻结原文，辖域缺陷 DEF-98-01）: {b0}   per_seed={ {str(k): v for k, v in v0.items()} }")
    print(f"  rev 3.5.1（只改辖域 all->any，阈值不变）  : {b1}   per_seed={ {str(k): v for k, v in v1.items()} }")
    print(f"  H1_confirmed={out['H1_confirmed']}  H1_refuted={out['H1_refuted']}")
    print(f"  ⇒ {p}")
    for s in seeds:
        print(f"\n  --- seed {s} ---")
        print(f"  {'arm':16s}{'acc_held':>9s}{'acc_tr':>8s}{'marg_ho':>9s}{'M2loo':>7s}{'M1bK':>8s}{'cos_med':>9s}")
        for arm in ARMS:
            m = per[s][arm]
            print(f"  {arm:16s}{m['acc_held']:9.4f}{m['acc_train']:8.4f}{m['margin_held']:9.2f}"
                  f"{m['M2loo']:7.4f}{m['M1bK']:8.4f}{m['cos_med']:9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
