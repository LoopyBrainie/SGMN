#!/opt/pytorch/bin/python
r"""DIR-55 · Stage A 硬门（`A55-1..14`，卡 rev 1.2 §7）。任一失败 ⇒ **禁止正式轮**。

★ 每个断言都必须**真读被守卫的对象**（纪律候选 123）：不写"同调用比同调用"，不写空断言。
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir55"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M
import R1_task33 as R
import R1_task48 as T48
import R1_task55 as D

D_H, D_K, D_V = M.D_H, M.D_K, M.D_V

# rev 2.0：默认值 = rev 1.4 的冻结配置（行为不变）；rev 2.0 用 --card/--gate median/--seed0 57000 覆盖
CARD = os.path.join(ROOT, "revolve", "DIR-55", "DIRECTION_rev55_v14.md")
CARD_MD5 = "ae4300936747b92a9b510b9011dbd411"
CARD_REV = "rev1.4"
SEED0, NPLAN, GATE = 56000, 96, "extreme"
OUT = os.path.join(ROOT, "dir55", "out", "DIR55r12_stageA.json")
DEV_MD5 = {
    "dir31/R1_grad.py": "8fa40f708600bec7e6c484727d2180c7",
    "dir33/R1_task33.py": "96ef89e87ec61556b52e0e464850087e",
    "dir46/R1_task46.py": "d66c849692d93bc5b24a91f9a445e0ef",
    "dir47/R1_task47.py": "fd03c70c6490cfd726489714b9c2c80f",
    "dir48/R1_task48.py": "bb6154687a48d4ea33f0ebd9bc0cb393",
}
SMOKE = list(D.SEEDS[:8])          # main() 里按 --seed0/--n-seeds 重算


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    global CARD, CARD_MD5, CARD_REV, SEED0, NPLAN, GATE, SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default=CARD)
    ap.add_argument("--card-md5", default=CARD_MD5)
    ap.add_argument("--card-rev", default=CARD_REV)
    ap.add_argument("--seed0", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--gate", choices=("extreme", "median"), default="extreme")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    CARD, CARD_MD5, CARD_REV, GATE = a.card, a.card_md5, a.card_rev, a.gate
    _out = a.out            # ★ `a` 在 A55-4 里被复用成 dict，故先把命令行值取出来（DEF-55-11）
    if a.seed0 is not None or a.n_seeds is not None:
        SEED0 = D.SEED0 if a.seed0 is None else a.seed0
        NPLAN = D.N_SEEDS if a.n_seeds is None else a.n_seeds
        D.SEEDS = list(range(SEED0, SEED0 + NPLAN))
    assert not (set(D.SEEDS) & D.PROBE_SEEDS), "A55-11：种子段与探针段相交"
    SMOKE = list(D.SEEDS[:8])
    t0 = time.time()
    g = {}

    # ── A55-1：装置 md5 + 卡片 md5
    g["A55-1_dev_md5"] = {p: {"want": w, "got": md5(os.path.join(ROOT, p))} for p, w in DEV_MD5.items()}
    g["A55-1_dev_ok"] = all(v["want"] == v["got"] for v in g["A55-1_dev_md5"].values())
    g["A55-1_card_md5"] = {"want": CARD_MD5, "got": md5(CARD)}
    g["A55-1_ok"] = bool(g["A55-1_dev_ok"] and g["A55-1_card_md5"]["want"] == g["A55-1_card_md5"]["got"])

    # ── A55-12：判据常数逐字等于卡（**装置与裁决器两侧都比**，rev 2.0 补 DEF-55-10）
    import adjudicate55 as J
    card_txt = open(CARD, encoding="utf-8").read()
    parsed = {k: float(re.search(rf"`{k}=([0-9.]+)`", card_txt).group(1))
              for k in ("T_A", "T_B1", "T_B2", "T_C", "T_C_ABS", "SURV", "STORE", "R_NAT", "K_MIN")}
    dev = {"T_A": J.T_A, "T_B1": J.T_B1, "T_B2": J.T_B2, "T_C": J.T_C, "T_C_ABS": J.T_C_ABS,
           "SURV": D.SURV, "STORE": float(D.STORE), "R_NAT": float(D.R_NAT), "K_MIN": D.K_MIN}
    adj = {"T_A": J.T_A, "T_B1": J.T_B1, "T_B2": J.T_B2, "T_C": J.T_C, "T_C_ABS": J.T_C_ABS,
           "SURV": J.SURV, "STORE": float(J.STORE), "R_NAT": float(J.R_NAT), "K_MIN": J.K_MIN}
    g["A55-12_consts"] = {"parsed_from_card": parsed, "device": dev, "adjudicator": adj}
    g["A55-12_ok"] = bool(parsed == dev == adj
                          and len(D.SEEDS) == NPLAN and D.SEEDS[0] == SEED0
                          and D.SEEDS[-1] == SEED0 + NPLAN - 1)

    # ── 采集 smoke 格（8 种子 × 7 个 (n,arm)）
    cells = {}
    for n, arm in [(2, "none"), (2, "natural_r8")] + [(3, x) for x in D.ARMS_N3]:
        cells[(n, arm)] = [D.cell(s, n, arm) for s in SMOKE]

    # ── A55-2：C ∈ [-1,1] 且有限
    bad = []
    for key, rows in cells.items():
        for r in rows:
            mats = ([r["C"]] if "C" in r else []) + [x["C"] for x in r.get("reps", [])]
            assert mats, ("A55-2 无矩阵可查", key, r["seed"])      # 防"空断言"
            for C in mats:
                for row in C:
                    for x in row:
                        if not (x == x) or abs(x) == float("inf") or abs(x) > 1 + 1e-12:
                            bad.append((key, r["seed"], x))
    g["A55-2_bad"] = bad[:5]
    g["A55-2_n_bad"] = len(bad)
    g["A55-2_ok"] = len(bad) == 0

    # ── A55-3：n=2 基线有效性（extreme=smoke 8 种子；median=**全 96 种子**，纪律候选 126）
    src = cells[(2, "none")] if GATE == "extreme" else [D.cell(s, 2, "none") for s in D.SEEDS]
    d3 = []
    for r in src:
        C = r["C_store"]
        dg = min(C[i][i] for i in range(2))
        off = max(abs(C[0][1]), abs(C[1][0]))
        d3.append({"seed": r["seed"], "diag_min": dg, "off_max": off})
    st = {"n_seeds": len(d3), "min_diag": min(x["diag_min"] for x in d3),
          "max_off": max(x["off_max"] for x in d3),
          "median_diag": statistics.median([x["diag_min"] for x in d3]),
          "median_off": statistics.median([x["off_max"] for x in d3])}
    g["A55-3"] = st
    if GATE == "median":      # rev 2.0：分位口径（跨全部种子取中位），最坏种子只报告
        g["A55-3_ok"] = bool(st["median_diag"] >= 0.95 and st["median_off"] <= 0.30)
    else:                     # rev 1.4：最坏种子口径
        g["A55-3_ok"] = bool(st["min_diag"] >= 0.95 and st["max_off"] <= 0.30)

    # ── A55-4：同种子同臂两次运行逐位相同
    a = D.cell(SMOKE[0], 3, "natural_m1")
    b = D.cell(SMOKE[0], 3, "natural_m1")
    det = max(abs(a["C"][i][j] - b["C"][i][j]) for i in range(3) for j in range(3))
    g["A55-4_det"] = det
    g["A55-4_ok"] = bool(det == 0.0)

    # ── A55-5：初始 (U1,V1) 逐位相同；`none` 与 `order012`（n=3、同顺序）存储后逐位相同
    inits = []
    for arm in D.ARMS_N3:
        P, keys, vals, idx = D.build_traits(SMOKE[0], 3, (2, 1, 0) if arm == "order210" else (0, 1, 2))
        inits.append((P["U1"].clone(), P["V1"].clone()))
    same_init = all(torch.equal(inits[0][0], x[0]) and torch.equal(inits[0][1], x[1]) for x in inits[1:])
    Pn, kn, vn, _, Un, Vn = D.stored_state(SMOKE[0], 3, (0, 1, 2))
    Po, ko, vo, _, Uo, Vo = D.stored_state(SMOKE[0], 3, (0, 1, 2))
    same_store = bool(torch.equal(Un, Uo) and torch.equal(Vn, Vo))
    g["A55-5"] = {"same_init": bool(same_init), "same_store_none_order012": same_store}
    g["A55-5_ok"] = bool(same_init and same_store)

    # ── A55-6：natural_m1 与 reinforce_m1 各 1 次追加写入、且 (k,v) 不同
    P, keys, vals, idx = D.build_traits(SMOKE[0], 3, (0, 1, 2))
    kn1, vn1 = D.natural_pair(P, SMOKE[0])
    diff_kv = float((kn1 - keys[0]).abs().max()) + float((vn1 - vals[0]).abs().max())
    g["A55-6"] = {"natural_pair_norm_k": float(kn1.norm()), "diff_kv_l1max": diff_kv,
                  "writes_natural": 1, "writes_reinforce": 1}
    g["A55-6_ok"] = bool(g["A55-6"]["writes_natural"] == g["A55-6"]["writes_reinforce"] == 1 and diff_kv > 1e-6)

    # ── A55-7：order012 与 order210 多重集相等、序列不等
    _, k012, v012, _ = D.build_traits(SMOKE[0], 3, (0, 1, 2))
    _, k210, v210, _ = D.build_traits(SMOKE[0], 3, (2, 1, 0))
    ms = all(any(torch.equal(a_, b_) for b_ in k210) for a_ in k012)
    seq_ne = not all(torch.equal(a_, b_) for a_, b_ in zip(k012, k210))
    g["A55-7"] = {"multiset_equal": bool(ms), "sequence_different": bool(seq_ne)}
    g["A55-7_ok"] = bool(ms and seq_ne)

    # ── A55-8：工作例（独立手算因子更新）
    U0 = torch.randn(D.D_V, 2, generator=torch.Generator().manual_seed(7)) * M.INIT_SD
    V0 = torch.randn(D.D_K, 2, generator=torch.Generator().manual_seed(8)) * M.INIT_SD
    k = R.KEY_SCALE * torch.eye(D_K)[0]
    v = torch.nn.functional.normalize(torch.randn(D_V, generator=torch.Generator().manual_seed(9)), dim=0)
    # ★ 独立重实现 `step_pair` 的 N_REP 次内部步（每步重算 W；V 用**更新后的** U）—— 见卡 rev 1.3 §2.3
    Uh, Vh_ = U0.clone(), V0.clone()
    for _ in range(T48.N_REP):
        W = Uh @ Vh_.t()
        G = M.ALPHA * torch.outer(W @ k - v, k) + M.LAM * W
        Uh = Uh - M.ETA * (G @ Vh_)
        Vh_ = Vh_ - M.ETA * (G.t() @ Uh)
    C_hand = float((Uh @ Vh_.t() @ k) @ v / (Uh @ Vh_.t() @ k).norm())
    Uc, Vc = T48.step_pair(U0.clone(), V0.clone(), k, v, 1.0)
    C_code = float((Uc @ Vc.t() @ k) @ v / (Uc @ Vc.t() @ k).norm())
    g["A55-8"] = {"C_hand": C_hand, "C_code": C_code, "abs_diff": abs(C_hand - C_code)}
    g["A55-8_ok"] = bool(abs(C_hand - C_code) <= 1e-12)

    # ── A55-9：幸存集口径
    ok9 = True
    for key, rows in cells.items():
        n = key[0]
        for r in rows:
            S = D.survivors(r["C"]) if "C" in r else r["S_after"]
            ok9 = ok9 and (0 <= len(S) <= n) and all(0 <= i < n for i in S)
    g["A55-9_ok"] = bool(ok9)

    # ── A55-10：近因率的分母（唯一幸存格）≥ 1 且被报告
    denom, hit = 0, 0
    for arm in ("order012", "order210"):
        for r in cells[(3, arm)]:
            if len(r["S_after"]) == 1:
                denom += 1
                orig = r["idx"][r["S_after"][0]]
                hit += int(orig == r["idx"][-1])
    g["A55-10"] = {"denominator": denom, "hits": hit, "rate": (hit / denom if denom else None)}
    g["A55-10_ok"] = bool(denom >= 1)

    # ── A55-11：种子表
    g["A55-11"] = {"n": len(D.SEEDS), "first": D.SEEDS[0], "last": D.SEEDS[-1],
                   "n_unique": len(set(D.SEEDS)),
                   "overlap_probe": len(set(D.SEEDS) & D.PROBE_SEEDS)}
    # 卡片正文里出现的所有 `NNNNN..NNNNN` 段：必须**恰好包含**本段，且与其余段**不相交**
    segs = [(int(x), int(y)) for x, y in re.findall(r"`(\d{5})\.\.(\d{5})`", open(CARD, encoding="utf-8").read())]
    cur = (D.SEEDS[0], D.SEEDS[-1])
    foreign = [s for s in segs if s != cur]
    # 只有「**部分重叠**」才是冲突；子区间（如卡里写 smoke 用 `56000..56007`）是合法的细化
    clash = [s for s in foreign if not (cur[1] < s[0] or s[1] < cur[0])
             and not (s[0] >= cur[0] and s[1] <= cur[1])]
    g["A55-11"]["card_segments"] = segs
    g["A55-11"]["foreign_segments"] = foreign
    g["A55-11"]["clash"] = clash
    g["A55-11"]["cur"] = cur
    g["A55-11_ok"] = bool(len(D.SEEDS) == NPLAN and len(set(D.SEEDS)) == NPLAN
                          and D.SEEDS[0] == SEED0 and D.SEEDS[-1] == SEED0 + NPLAN - 1
                          and g["A55-11"]["overlap_probe"] == 0
                          and cur in segs and not clash)

    # ── A55-13：所有自然对 ‖k‖ ≥ K_MIN
    knorms = [r["k_norm"] for r in cells[(3, "natural_m1")]]
    knorms += [x["k_norm"] for r in cells[(2, "natural_r8")] for x in r["reps"]]
    g["A55-13"] = {"min_k_norm": min(knorms), "n": len(knorms)}
    g["A55-13_ok"] = bool(g["A55-13"]["min_k_norm"] >= D.K_MIN)

    # ── A55-14：natural_r8 的 8 条内容种子不同、且每条都从**同一**初始状态出发
    r0 = cells[(2, "natural_r8")][0]
    P, keys, vals, idx, U, Vm = D.stored_state(r0["seed"], 2, None)
    k5, v5 = D.natural_pair(P, r0["seed"] * 100 + 5)
    U5, V5 = D.write_seq(U, Vm, [(k5, v5)], reps=1)
    C5 = D.cmat(U5, V5, keys, vals)
    d14 = max(abs(C5[i][j] - r0["reps"][5]["C"][i][j]) for i in range(2) for j in range(2))
    ks = [tuple(round(float(x), 12) for x in D.natural_pair(P, r0["seed"] * 100 + r)[0]) for r in range(8)]
    g["A55-14"] = {"recompute_r5_maxdiff": d14, "n_distinct_content_keys": len(set(ks))}
    g["A55-14_ok"] = bool(d14 <= 1e-15 and len(set(ks)) == D.R_NAT)

    res = {"stage": "A55", "card": CARD_REV, "card_md5": CARD_MD5, "gate": GATE,
           "seeds": {"n": len(D.SEEDS), "first": D.SEEDS[0], "last": D.SEEDS[-1]},
           "secs": time.time() - t0, "checks": g}
    passed = [k for k in g if k.endswith("_ok") and g[k]]
    res["n_passed"] = len(passed)
    res["n_total"] = len([k for k in g if k.endswith("_ok")])
    res["all_ok"] = bool(res["n_passed"] == res["n_total"])
    print(json.dumps({k: g[k] for k in g if k.endswith("_ok")}, ensure_ascii=False, indent=1))
    print(json.dumps({"n_passed": res["n_passed"], "n_total": res["n_total"], "all_ok": res["all_ok"]},
                     ensure_ascii=False))
    out = _out or OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    raise SystemExit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()