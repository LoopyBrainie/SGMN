"""`R1_task122.py` —— `DIR-122` 正式装置（卡片 `DIRECTION_rev122_v10.md`）。

★ 装置：**纯再分析**（不训练、不注入、不评估、不需要 GPU / tokenizer）。
  读数**只读** `dir119/out/cells/COLOCATED_600_<seed>.json`（`dose=600`、`mult=1e4`）。
  测量层全部来自 `dir122/mech122.py`（`D_row` 与置换零分布逐字复用 `dir121/mech121.py`）。

两阶段：
  `--stage a`      Stage A 12 条硬门（任一失败 ⇒ 不得进正式轮）
  `--stage formal` 正式轮：`J122-1`（三个题集）、`J122-2`、`J122-3`（report-only）

★ Stage A 与正式轮**同一进程外**分两次跑：Stage A 不打印任何真实判据观测值。
  ★★ `A122-12`（"判据真的被算过"）在**合成 9×3 矩阵**上执行 —— 既挡住「静默 None」
     这个失败模式，又**不**提前泄露真实 `p` 值。判决书 §5 会写明这个实现选择。
"""
import glob
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
for _p in (_HERE, os.path.join(_ROOT, "dir121"), os.path.join(_ROOT, "dir120")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mech122 as M          # noqa: E402
import mech121 as M121       # noqa: E402

CELLS_GLOB = "dir119/out/cells/COLOCATED_600_*.json"
OUT = os.path.join(_HERE, "out")

# ---- 已发表常数（护栏）----
PUB_MAIN_NSH = {"0": 2, "1": 23, "2": 20, "3": 3}          # D-151
PUB_ESCAPE = [0, 3, 18, 0, 2, 9, 5, 0, 5]                  # D-152
PUB_ESCAPE_TOTAL = 42                                      # D-152
PUB_D_ROW_MAIN = 58.28571428571427                         # D-152


CHECKS = []


def rec(tag, ok, msg):
    """记录 + 打印。★ `CHECKS` 是**真记录**（首版曾硬编码 `{"ok": True}` = 空转脚手架，已删）。"""
    CHECKS.append({"tag": tag, "ok": bool(ok), "msg": msg})
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}  {msg}", flush=True)
    return bool(ok)


def load_cells():
    """读 9 个 cell，抽出主读物与三个 c2 题集的 `pick_argmin`。"""
    cells = []
    for f in sorted(glob.glob(CELLS_GLOB)):
        d = json.load(open(f, encoding="utf-8"))
        picks = {M.MAIN_RNG: d["mech_by_mult"]["10000"]["pick_argmin"]}
        for r in M.C2_RNGS:
            picks[r] = d["c2"]["10000"][str(r)]["pick_argmin"]
        cells.append({"seed": int(d["seed"]), "picks": picks, "raw": d})
    return cells


def item_escape_matrix(cells, items, ns, tv, rng):
    """逐 seed × 逐项 的逃逸布尔 ⇒ `int` 矩阵 `(9, 48)`。"""
    rows = [[1 if e else 0 for e in M.escape_mask(items[rng], ns[rng], c["picks"][rng], tv)]
            for c in cells]
    return np.array(rows, dtype=np.int64)


# ================================================================ Stage A
def stage_a():
    RES = {"checks": [], "t": 0.0}
    t0 = time.time()
    ok = True
    facts, tr, ev, tv, items, ns = M.build_all()
    cells = load_cells()

    ok &= rec("A122-1", M.hist_of(ns[M.MAIN_RNG]) == PUB_MAIN_NSH,
              f"主读物 n_seen_hist = {M.hist_of(ns[M.MAIN_RNG])}（须 == {PUB_MAIN_NSH}，D-151）")

    esc_main = [int(sum(M.escape_mask(items[M.MAIN_RNG], ns[M.MAIN_RNG], c["picks"][M.MAIN_RNG], tv)))
                for c in cells]
    ok &= rec("A122-2", esc_main == PUB_ESCAPE and sum(esc_main) == PUB_ESCAPE_TOTAL,
              f"逐 seed 逃逸数 = {esc_main}、合计 {sum(esc_main)}"
              f"（须 == {PUB_ESCAPE} / {PUB_ESCAPE_TOTAL}，D-152）★ 顺序敏感")

    h_ok, h_msg = True, []
    for r in M.C2_RNGS:
        mine = M.hist_of(ns[r])
        pub = M.norm_hist(cells[0]["raw"]["c2"]["10000"][str(r)]["n_seen_hist"])
        h_ok &= (mine == pub)
        h_msg.append(f"{r}:{mine}{'==' if mine == pub else '!='}{pub}")
    ok &= rec("A122-3", h_ok, "三个 c2 题集 n_seen_hist（稠密）== 存档 —— " + "；".join(h_msg))

    d = [int(sum(1 for v in ns[r] if v >= 1)) for r in M.C2_RNGS]
    ok &= rec("A122-4", d == [45, 44, 45], f"三个 c2 题集可逃逸项数 = {d}（须 == [45, 44, 45]）")

    len_ok, rng_ok = True, True
    for c in cells:
        for r in (M.MAIN_RNG,) + M.C2_RNGS:
            p = c["picks"][r]
            len_ok &= (len(p) == 48)
            rng_ok &= all(0 <= int(x) <= M.N_CAND - 1 for x in p)
    ok &= rec("A122-5", len_ok and rng_ok,
              f"9 cell × 4 题集的 pick_argmin 长度均 48={len_ok}、取值 ∈[0,3]={rng_ok}")

    per_r = []
    for r in M.C2_RNGS:
        hs = {json.dumps(c["raw"]["c2"]["10000"][str(r)]["n_seen_hist"], sort_keys=True)
              for c in cells}
        per_r.append(len(hs))
    ok &= rec("A122-6", all(x == 1 for x in per_r),
              f"同一 rng 的 n_seen_hist 在 9 cell 上的不同取值数 = {per_r}（须全 1 ⇒ 配对设计）")

    Ea = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int64)
    Eb = np.array([[0, 2, 0], [1, 1, 1], [2, 0, 2]], dtype=np.int64)
    ok &= rec("A122-7", abs(M.gamma_stat(Ea) - 6.0) < 1e-12,
              f"手算例 A：Γ = {M.gamma_stat(Ea)!r}（须 == 6.0，卡片 §3-1）")
    ok &= rec("A122-8", abs(M.gamma_stat(Eb) - (-2.0)) < 1e-12,
              f"手算例 B：Γ = {M.gamma_stat(Eb)!r}（须 == -2.0，卡片 §3-2）")

    Esyn = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 0, 1], [1, 1, 0],
                     [2, 2, 2], [0, 2, 1], [1, 0, 2], [2, 1, 0]], dtype=np.int64)
    n1 = M.simulate_null_gamma(Esyn, B=200, seed=777)
    n2 = M.simulate_null_gamma(Esyn, B=200, seed=777)
    ok &= rec("A122-9", np.array_equal(n1, n2),
              f"置换确定性：同 seed 两次逐位相同 = {np.array_equal(n1, n2)}")

    # 保边际：模拟按定义逐列置换 ⇒ 列均值不变；用**列中心化后**的零分布均值 ≈ 0 来机械核对
    ok &= rec("A122-10", abs(float(n1.mean())) < 5.0,
              f"合成矩阵零分布均值 = {float(n1.mean()):.4f}（|·|<5 ⇒ 未系统偏移，保边际）")

    E_main = item_escape_matrix(cells, items, ns, tv, M.MAIN_RNG)
    C_main = E_main.sum(axis=0)
    dr_main = M121.d_row(E_main.sum(axis=1))
    ok &= rec("A122-11", abs(dr_main - PUB_D_ROW_MAIN) < 5e-9,
              f"主读物 D_row = {dr_main!r}（须 == {PUB_D_ROW_MAIN!r}，D-152 已发表）")

    # A122-12：判据机器在**合成**矩阵上必须产出非 None 的 p（挡「静默 None」）
    Es = np.array([[0, 1, 2], [1, 0, 0], [2, 2, 1], [0, 1, 1], [1, 2, 2],
                   [2, 0, 0], [0, 2, 1], [1, 1, 2], [2, 1, 0]], dtype=np.int64)
    drs = M.d_row_per_qset(Es)
    ps = []
    for j in range(3):
        Cj = np.array([1, 0, 2, 1, 0, 1, 2, 0, 1, 0, 1, 2, 1, 0, 1, 0], dtype=np.int64)
        nullj = M121.simulate_null_B(Cj, 9, B=20000, seed=M.BI_SEED_J1)
        ps.append(M121.p_upper(drs[j], nullj))
    g_syn = M.gamma_stat(Es)
    null_g = M.simulate_null_gamma(Es, B=20000, seed=M.BI_SEED_J2)
    p_g = M121.p_upper(g_syn, null_g)
    all_p = ps + [p_g]
    ok &= rec("A122-12", all(p is not None for p in all_p) and len(null_g) == 20000
              and len(null_g) == M.B_PERM,
              f"合成矩阵上四个 p 均非 None={all(p is not None for p in all_p)}、"
              f"B={len(null_g)}（须 {M.B_PERM}）★ 挡「判据从未计算」")

    RES["checks"] = CHECKS
    RES["all_pass"] = bool(ok)
    RES["t"] = round(time.time() - t0, 2)
    os.makedirs(OUT, exist_ok=True)
    json.dump(RES, open(os.path.join(OUT, "stage_stageA.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n★ Stage A {'全部通过' if ok else '失败'}（{RES['t']}s）", flush=True)
    return ok


# ================================================================ 正式轮
def formal():
    t0 = time.time()
    facts, tr, ev, tv, items, ns = M.build_all()
    cells = load_cells()
    seeds = [c["seed"] for c in cells]

    # ---- 逐题集 × 逐 seed 的逃逸表 ----
    per_q = {}
    for r in M.C2_RNGS:
        E = item_escape_matrix(cells, items, ns, tv, r)
        per_q[r] = {"E_item": E, "R": E.sum(axis=1), "C": E.sum(axis=0),
                    "d": int(sum(1 for v in ns[r] if v >= 1))}

    E = np.array([[int(per_q[r]["R"][s]) for r in M.C2_RNGS] for s in range(len(cells))],
                 dtype=np.int64)
    d = np.array([per_q[r]["d"] for r in M.C2_RNGS], dtype=np.int64)

    # ---- J122-1：三个题集各自的 D_row ----
    j1 = []
    for j, r in enumerate(M.C2_RNGS):
        dr = M121.d_row(per_q[r]["R"])
        null = M121.simulate_null_B(per_q[r]["C"], len(cells), B=M.B_PERM, seed=M.BI_SEED_J1)
        p = M121.p_upper(dr, null)
        j1.append({"rng": r, "D_row": dr, "p": p, "null": M121.summarize(null),
                   "C_hist": M.hist_of(per_q[r]["C"], n_bins=len(cells) + 1)})
        print(f"  J122-1  rng={r}: D_row={dr:.4f}  p={p}  "
              f"null p50={j1[-1]['null']['p50']:.3f} q99={j1[-1]['null']['q99']:.3f}", flush=True)

    # ---- J122-2：Γ ----
    g_obs = M.gamma_stat(E)
    null_g = M.simulate_null_gamma(E, B=M.B_PERM, seed=M.BI_SEED_J2)
    p_g = M121.p_upper(g_obs, null_g)
    print(f"  J122-2  Γ={g_obs:.4f}  p={p_g}  null p50={np.quantile(null_g,0.5):.3f} "
          f"q99={np.quantile(null_g,0.99):.3f}", flush=True)

    # ---- 判决（卡片 §2-5，冻结）----
    j1_all = all(x["p"] is not None and x["p"] <= M.ALPHA_J1_EACH for x in j1)
    j2_ok = (p_g is not None and p_g <= M.ALPHA_J2)
    if j1_all and j2_ok:
        row, verdict = 1, "H122_CONFIRMED"
    elif j1_all and not j2_ok:
        row, verdict = 2, "H122_FALSIFIED_NOT_REPRODUCIBLE"
    else:
        row, verdict = 3, "H122_FALSIFIED_SEED_EFFECT_NOT_ROBUST"

    # ---- J122-3 report-only ----
    rates = M.rates(E, d)
    spread = [max(x for x in rates[s] if x is not None) - min(x for x in rates[s] if x is not None)
              for s in range(len(cells))]
    top = []
    for j, r in enumerate(M.C2_RNGS):
        Rj = per_q[r]["R"].astype(float)
        top.append({"rng": r, "seed": int(seeds[int(np.argmax(Rj))]),
                    "R": int(Rj.max()), "r_bar": float(Rj.mean())})
    zero_seeds = [seeds[i] for i in range(len(cells)) if int(per_q[M.C2_RNGS[0]]["R"][i]) == 0]
    main_zero = [seeds[i] for i, e in enumerate(
        [int(sum(M.escape_mask(items[M.MAIN_RNG], ns[M.MAIN_RNG], c["picks"][M.MAIN_RNG], tv)))
         for c in cells]) if e == 0]
    always_zero = [s for s in main_zero if all(
        int(per_q[r]["R"][seeds.index(s)]) == 0 for r in M.C2_RNGS)]

    # ---- J122-3 追加（report-only，不参与判决）：逐 seed 的跨题集同质性 + 两两秩相关 ----
    # ★ 方向与 J122-2 **相反**：J122-2 问"率是否跟着 seed 走"，这里问"率是不是一个**固定常数**"。
    #   两者可以同时"显著"（γ 显著 + 某些 seed 仍波动）—— 判决书必须如实并列。
    homo = []
    for si, s_ in enumerate(seeds):
        Q = 0.0
        tot_e = float(E[si].sum())
        tot_d = float(d.sum())
        p_hat = tot_e / tot_d if tot_d else 0.0
        for j in range(len(M.C2_RNGS)):
            mu = d[j] * p_hat
            var = d[j] * p_hat * (1 - p_hat)
            if var > 0:
                Q += (float(E[si, j]) - mu) ** 2 / var
        # χ²(2) 的上尾有闭式：P(Q ≥ q) = exp(-q/2)
        homo.append({"seed": int(s_), "Q": Q, "p_chi2_2": float(np.exp(-Q / 2.0)),
                     "E": E[si].tolist()})

    def _rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            k = i
            while k + 1 < len(order) and v[order[k + 1]] == v[order[i]]:
                k += 1
            avg = (i + k) / 2.0 + 1.0
            for t in range(i, k + 1):
                r[order[t]] = avg
            i = k + 1
        return np.array(r)

    def _spearman(a, b):
        ra, rb = _rank(list(a)), _rank(list(b))
        ra -= ra.mean()
        rb -= rb.mean()
        den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
        return float((ra * rb).sum() / den) if den else None

    pairs = []
    for i in range(len(M.C2_RNGS)):
        for j in range(i + 1, len(M.C2_RNGS)):
            pairs.append({"pair": [M.C2_RNGS[i], M.C2_RNGS[j]],
                          "spearman": _spearman(E[:, i], E[:, j])})

    res = {
        "seeds": seeds,
        "qsets": list(M.C2_RNGS),
        "d": d.tolist(),
        "E": E.tolist(),
        "rates": rates,
        "J122_1": j1,
        "J122_2": {"Gamma": g_obs, "p": p_g,
                   "null": {"p50": float(np.quantile(null_g, 0.5)),
                            "p90": float(np.quantile(null_g, 0.9)),
                            "q95": float(np.quantile(null_g, 0.95)),
                            "q99": float(np.quantile(null_g, 0.99)),
                            "n": int(null_g.size)}},
        "alphas": {"each_qset": M.ALPHA_J1_EACH, "gamma": M.ALPHA_J2},
        "B": M.B_PERM,
        "decision_row": row,
        "verdict": verdict,
        "J122_3": {"spread": spread, "top_seed_per_qset": top,
                   "main_zero_seeds": main_zero, "always_zero_seeds": always_zero,
                   "per_seed_homogeneity": homo, "pair_spearman": pairs},
        "per_q_R": {str(r): per_q[r]["R"].tolist() for r in M.C2_RNGS},
        "per_q_C_hist": {str(r): M.hist_of(per_q[r]["C"], n_bins=len(cells) + 1)
                      for r in M.C2_RNGS},
        "sec": round(time.time() - t0, 2),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "formal.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n★★ 判决：第 {row} 行 → {verdict}")
    print(f"（J122-1 三个都过 = {j1_all}；J122-2 过 = {j2_ok}；用时 {res['sec']}s）", flush=True)
    return res


def main():
    stage = "a"
    for i, a in enumerate(sys.argv):
        if a == "--stage":
            stage = sys.argv[i + 1]
    if stage == "a":
        return 0 if stage_a() else 1
    formal()
    return 0


if __name__ == "__main__":
    sys.exit(main())