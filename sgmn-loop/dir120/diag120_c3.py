"""`diag120_c3.py` —— `report-only` 诊断：把 `D-150` §7-3 的 `C3` 读数**逐字复现**，
再用**可用性感知**的零假设重判它。**不进判据**（判据已由 `R1_task120.py` 冻结判完）。

★ 起因（必须写明）：`R1_task120.py` 的正式跑给出
  见侧 top3 = **0.5296**（零假设 mean 0.5085 / q99 **0.5376**）⇒ **未超** `q99`，
  而 `D-150` §7-3 用 **0.4560** 声称"见过值内部还存在次级词面偏好（某些见过值被偏得更多）"。
  **两个数不是同一个统计量**：0.4560 是**全部 432 次选择、跨 16 个值**的 top-3；
  我这边的 0.5296 是**只看 372 次见侧选择、跨 8 个见过值**的 top-3。
  ⇒ 本脚本同时算**两个统计量 × 两种零假设**，把"哪个数在哪种零假设下成立"钉死。
"""
import json
import pathlib
import sys
import collections

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "dir119"))
sys.path.insert(0, str(_HERE.parent / "dir118"))

import numpy as np                 # noqa: E402
import R1_task118 as P118          # noqa: E402
import R1_task90 as D90            # noqa: E402
import R1_task104 as D104          # noqa: E402
import mech120 as M                # noqa: E402

OUT119 = _HERE.parent / "dir119" / "out"
ARM = "COLOCATED"
J = "10000"
B = 20000
SEED = 20261122


def main():
    seeds, pick = [], []
    for f in sorted((OUT119 / "cells").glob(f"{ARM}_600_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        seeds.append(int(d["seed"]))
        pick.extend(int(x) for x in d["mech_by_mult"][J]["pick_argmin"])
    tok, _m = D90.model_and_tok()
    _f, _p, _c, IT = P118.ground(tok, ARM)
    items, meta, tv = IT["items_ev"], IT["meta_ev"], IT["tv"]
    tgt = [int(x) for x in D104.targets_of(items)]
    n = len(items)
    meta432 = [meta[i % n] for i in range(len(pick))]
    tgt432 = [tgt[i % n] for i in range(len(pick))]

    all_vals = [meta432[i]["cand_val"][pick[i]] for i in range(len(pick))]
    seen_vals = [v for v in all_vals if v in tv]

    # ---- 统计量 A：D-150 §7-3 的原口径（全部 432 次、跨 16 值）----
    obsA = M.top3_share(all_vals)
    # ---- 统计量 B：本卡 J120-3 的口径（只看见侧 372 次、跨 8 个见过值）----
    obsB = M.top3_share(seen_vals)
    print(f"统计量 A（全部 {len(all_vals)} 次、16 值）top3 = {obsA:.4f}"
          f"   ★ D-150 §7-3 发表值 = 0.4560 ⇒ {'复现一致' if abs(obsA - 0.4560) < 5e-4 else '不一致!'}")
    print(f"统计量 B（见侧 {len(seen_vals)} 次、8 个见过值）top3 = {obsB:.4f}")

    # ---- 可用性感知的零假设：条件在"每项选了哪种状态"，同状态内均匀重抽 ----
    want = [meta432[i]["cand_val"][pick[i]] in tv for i in range(len(pick))]
    pools = []
    for i in range(len(pick)):
        cv = meta432[i]["cand_val"]
        pools.append([j for j in range(len(cv)) if (cv[j] in tv) == want[i]])
    rng = np.random.default_rng(SEED)
    A = np.empty(B); Bs = np.empty(B)
    for b in range(B):
        vals = [meta432[i]["cand_val"][pools[i][int(rng.integers(len(pools[i])))]]
                for i in range(len(pick))]
        A[b] = M.top3_share(vals)
        Bs[b] = M.top3_share([v for v in vals if v in tv])

    def rep(name, obs, arr):
        q95, q99 = float(np.quantile(arr, 0.95)), float(np.quantile(arr, 0.99))
        p = float((arr >= obs).mean())
        print(f"  {name}: obs={obs:.4f}  null mean={arr.mean():.4f} q95={q95:.4f} q99={q99:.4f}"
              f"  p(单侧)={p:.4f}  ⇒ {'超 q99' if obs > q99 else ('超 q95' if obs > q95 else '未超 q95')}")
        return {"obs": obs, "null_mean": float(arr.mean()), "q95": q95, "q99": q99, "p": p}

    print("\n★ 可用性感知零假设（B=%d，seed=%d，条件在每项所选状态上）" % (B, SEED))
    rA = rep("统计量 A（D-150 原口径）", obsA, A)
    rB = rep("统计量 B（本卡 J120-3）  ", obsB, Bs)

    # ---- 朴素零假设（D-150 §7-3 当时用的）：忽略可用性，只在 8 个见过值里均匀 ----
    print("\n★ 对照：朴素零假设（忽略可用性，只在 8 个见过值里均匀）—— 这是 D-150 §7-3 用的那种")
    naive = np.empty(4000)
    for b in range(len(naive)):
        vals = rng.choice(np.array(sorted(tv)), size=len(all_vals), replace=True)
        naive[b] = M.top3_share(list(vals))
    print(f"  统计量 A 在朴素零假设下：mean={naive.mean():.4f} q95={np.quantile(naive,0.95):.4f}"
          f" ⇒ obs 0.4560 {'看似' if obsA > np.quantile(naive,0.95) else '不'}超 q95")
    print("  ★ 但朴素零假设把 60 次'未见值'选择也当成见过值来算 ⇒ 系统性**压低**零假设分布 ⇒ **假阳性**。")

    # ---- 残差分解：盲逃逸能否精确解释那 16 项 ----
    part = M.partition(pick, meta432, tv, tgt432)
    q0 = M.q0_of(part, meta432, tv, tgt432)
    print("\n★ 残差分解（卡片 §0-3 的恒等式 + 盲逃逸预测）")
    print(f"  逃逸 {part['escape']} 项 × 盲选命中率 q0={q0:.6f} = **{part['escape'] * q0:.2f}** 项"
          f"   实测 E_esc_ok = **{part['E_esc_ok']}** 项")
    print(f"  ⇒ 差 {part['E_esc_ok'] - part['escape'] * q0:+.2f} 项"
          f"（二项 sd ≈ {np.sqrt(part['escape'] * q0 * (1 - q0)):.2f}）")
    print(f"  残差占比 = E_esc_ok/432 = {part['E_esc_ok'] / part['n']:.4f}（D-150 说 ≈0.036，复现一致）")

    # ---- 逐 seed 稳定性（J120-1 的逃逸率是否由单个 seed 拉动）----
    print("\n★ 逐 seed（逃逸率 / 残差 / n11）")
    per = {}
    for k, s in enumerate(seeds):
        sl = slice(k * n, (k + 1) * n)
        p = M.partition(pick[sl], meta432[sl], tv, tgt432[sl])
        per[str(s)] = {"escape": p["escape"], "E_esc_ok": p["E_esc_ok"], "r": p["r"]}
        print(f"  seed {s}: 逃逸 {p['escape']:2d}/{p['n_has_seen']} (r={p['r']:.4f})"
              f" · 残差 {p['E_esc_ok']} · E_total {p['E_total']}")

    # ---- 可用性解释：每个值在各状态下的"可用次数" ----
    avail = collections.Counter()
    for i in range(len(pick)):
        for j, v in enumerate(meta432[i]["cand_val"]):
            if j != tgt432[i] or True:
                avail[(v, v in tv)] += 1
    print("\n★ 可用性与选中次数（解释「为什么某些见过值被选得多」）")
    cnt = collections.Counter(all_vals)
    rows = sorted(((v, avail[(v, v in tv)], cnt.get(v, 0), v in tv) for v in set(all_vals)),
                  key=lambda r: -r[2])
    for v, av, c, sv in rows:
        exp = av * (len(seen_vals) / float(sum(avail[(x, True)] for x in tv))) if sv else None
        print(f"  {v:>8s} {'见过' if sv else '未见'} 可用 {av:4d} 选中 {c:3d}"
              + (f"  （按可用性预期 ≈ {exp:.1f}）" if exp else ""))

    out = {"stat_A_obs": obsA, "stat_B_obs": obsB, "null_A": rA, "null_B": rB,
           "naive_A_mean": float(naive.mean()),
           "naive_A_q95": float(np.quantile(naive, 0.95)),
           "escape": part["escape"], "q0": q0,
           "pred_from_blind_escape": part["escape"] * q0,
           "E_esc_ok": part["E_esc_ok"], "per_seed": per,
           "B": B, "seed": SEED}
    (_HERE / "out").mkdir(parents=True, exist_ok=True)
    (_HERE / "out" / "diag_c3.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n⇒ 已写 out/diag_c3.json")


if __name__ == "__main__":
    main()