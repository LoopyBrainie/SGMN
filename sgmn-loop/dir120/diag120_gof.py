"""`diag120_gof.py` —— `report-only`：`J120-3/4` 的统计量**被可用性混淆**，改用恰当的拟合检验。

★ 为什么必须补这一步（写进 VERDICT）：
  `top3` 占比在**可用性不均**时几乎被可用性决定 —— 本装置里 `ruby`/`lemon`/`ivory`
  各可用 **108** 次，而 `tan`/`amber` 只 **45** 次 ⇒ 零假设下 top3 就已 ≈**0.51**，
  「还能测出多少偏好」的余地被压得极小。`D-150` §7-3 当时对的是**忽略可用性**的朴素零假设（≈0.32/0.44），
  那个零假设系统性**偏低** ⇒ 制造了 p≈0.059 的"看似显著"。
  ⇒ 用**精确的可用性加权期望**做 χ² 拟合检验（每项贡献 1/n_同状态候选数），才是对"值级偏好"的正确检验。

★ 另补：**逐 seed 逃逸数是否过度离散**。若"逃逸"是每项独立同概率 `p`，则 46 项的逃逸数
  应近似二项；实测 3 个 seed 恰为 **0**、一个到 **18** ⇒ 若离散度远超二项，则
  "`M1` 是常数概率加成"这个说法**必须**被限定为"seed 级regime"（写进边界）。
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
ARM, J = "COLOCATED", "10000"


def chi2_gof(obs_counter, exp_map):
    """χ² 拟合：返回 (chi2, df, p, 每值明细)。`p` 用 χ² 尾概率（无 scipy 依赖 ⇒ 用 Wilson–Hilferty 近似）。"""
    keys = [k for k in exp_map if exp_map[k] > 1e-9]
    chi2 = sum((obs_counter.get(k, 0) - exp_map[k]) ** 2 / exp_map[k] for k in keys)
    df = len(keys) - 1
    # Wilson–Hilferty：χ²_df ≈ df·(1 - 2/(9df) + z·sqrt(2/(9df)))³
    if df <= 0:
        return chi2, df, None, {}
    z = (((chi2 / df)) ** (1.0 / 3.0) - (1 - 2.0 / (9 * df))) / np.sqrt(2.0 / (9 * df))
    from math import erfc, sqrt
    p = 0.5 * erfc(z / sqrt(2.0))
    detail = {k: {"obs": obs_counter.get(k, 0), "exp": exp_map[k],
                  "z": (obs_counter.get(k, 0) - exp_map[k]) / np.sqrt(exp_map[k])} for k in keys}
    return chi2, df, float(p), detail


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

    obs_seen, obs_unseen = collections.Counter(), collections.Counter()
    exp_seen, exp_unseen = collections.Counter(), collections.Counter()
    for i in range(len(pick)):
        cv = meta432[i]["cand_val"]
        seen_slots = [j for j in range(len(cv)) if cv[j] in tv]
        uns_slots = [j for j in range(len(cv)) if cv[j] not in tv]
        pv = cv[pick[i]]
        picked_seen = pv in tv
        (obs_seen if picked_seen else obs_unseen)[pv] += 1
        # ★★ 期望必须**条件在"该项确实在该侧做出了选择"**上 —— 否则期望总数（414/424）
        #    与观测总数（372/60）对不上，χ² 全部虚高（★ 我第一版就踩了这个坑：
        #    unseen 侧 8 个 z 全是 −4.5…−7.5 = 期望总量错了 7 倍，不是真效应）。
        if picked_seen:
            for j in seen_slots:
                exp_seen[cv[j]] += 1.0 / len(seen_slots)
        else:
            for j in uns_slots:
                exp_unseen[cv[j]] += 1.0 / len(uns_slots)

    out = {}
    for tag, obs, exp in (("seen", obs_seen, exp_seen), ("unseen", obs_unseen, exp_unseen)):
        chi2, df, p, detail = chi2_gof(obs, exp)
        print(f"\n★ {tag}：共 {sum(obs.values())} 次选择，{len(detail)} 个值")
        print(f"  χ² = {chi2:.3f}, df = {df}, p = {p:.4f}"
              f"  ⇒ {'拒绝均匀（存在值级偏好）' if (p is not None and p < 0.01) else '不拒绝（可用性已足够解释）'}")
        for k, v in sorted(detail.items(), key=lambda kv: -abs(kv[1]["z"])):
            print(f"    {k:>8s} obs {v['obs']:3d} exp {v['exp']:7.2f}  z = {v['z']:+.2f}")
        out[tag] = {"chi2": chi2, "df": df, "p": p, "detail": {k: {kk: (None if vv is None else float(vv))
                                                                  for kk, vv in v.items()}
                                                              for k, v in detail.items()}}

    # ---- ★ 精确置换：对 χ² 统计量本身用同一个状态保持零假设（无分布假设）----
    #   ★ 动机：unseen 侧期望计数低到 3.67 ⇒ 渐近 χ² 可能偏松；而 `J120-4`（top3）又不显著
    #     ⇒ 两个检验冲突时**不能挑一个**，必须用**同一个零假设**对 **χ² 本身**做精确置换。
    B = 20000
    rng = np.random.default_rng(20261122)
    idx_uns = [i for i in range(len(pick)) if meta432[i]["cand_val"][pick[i]] not in tv]
    idx_seen = [i for i in range(len(pick)) if meta432[i]["cand_val"][pick[i]] in tv]
    pools_uns = [[j for j, v in enumerate(meta432[i]["cand_val"]) if v not in tv] for i in idx_uns]
    pools_seen = [[j for j, v in enumerate(meta432[i]["cand_val"]) if v in tv] for i in idx_seen]

    def chi2_under_null(idx, pools, obs):
        ex = collections.Counter()
        for i, pool in zip(idx, pools):
            for j in pool:
                ex[meta432[i]["cand_val"][j]] += 1.0 / len(pool)
        keys = [k for k in ex if ex[k] > 1e-9]
        return sum((obs.get(k, 0) - ex[k]) ** 2 / ex[k] for k in keys)

    exact = {}
    for tag, idx, pools, obs in (("seen", idx_seen, pools_seen, obs_seen),
                                 ("unseen", idx_uns, pools_uns, obs_unseen)):
        chi2_obs = chi2_under_null(idx, pools, obs)
        null = np.empty(B)
        for b in range(B):
            c = collections.Counter()
            for i, pool in zip(idx, pools):
                c[meta432[i]["cand_val"][pool[int(rng.integers(len(pool)))]]] += 1
            null[b] = chi2_under_null(idx, pools, c)
        p_ex = float((null >= chi2_obs).mean())
        print(f"\n★ {tag} 的 **精确置换** p（χ² 统计量，B={B}）：χ²_obs={chi2_obs:.3f}"
              f"  null mean={null.mean():.3f} q95={np.quantile(null,0.95):.3f}"
              f" q99={np.quantile(null,0.99):.3f}  ⇒ p_exact={p_ex:.4f}")
        exact[tag] = {"chi2_obs": chi2_obs, "null_mean": float(null.mean()),
                      "q95": float(np.quantile(null, 0.95)), "q99": float(np.quantile(null, 0.99)),
                      "p_exact": p_ex, "B": B}
        print(f"    判定（α=0.01）：{'★ 存在值级偏好' if p_ex < 0.01 else '不拒绝均匀'}")
    out["exact_permutation"] = exact

    # ---- 逃逸数的跨 seed 过度离散 ----
    part = M.partition(pick, meta432, tv, tgt432)
    per = []
    for k in range(len(seeds)):
        sl = slice(k * n, (k + 1) * n)
        p = M.partition(pick[sl], meta432[sl], tv, tgt432[sl])
        per.append({"seed": seeds[k], "escape": p["escape"], "n": p["n_has_seen"],
                    "r": p["r"], "E_esc_ok": p["E_esc_ok"]})
    es = np.array([x["escape"] for x in per], dtype=float)
    pool = es.sum() / float(sum(x["n"] for x in per))
    exp_bin = np.array([pool * x["n"] for x in per])
    chi2_disp = float((((es - exp_bin) ** 2) / (exp_bin * (1 - pool))).sum())
    df_disp = len(es) - 1
    print(f"\n★ 逃逸数的跨 seed 过度离散（H0：每项独立同概率 p={pool:.4f}）")
    print(f"  逐 seed 逃逸数 = {[int(x) for x in es]}（n=46/seed）")
    print(f"  二项零假设下 sd ≈ {np.sqrt(np.mean([pool*(1-pool)*x['n'] for x in per])):.2f}；"
          f"实测极差 0…{int(es.max())}")
    print(f"  Pearson χ² = {chi2_disp:.2f}, df = {df_disp} ⇒ χ²/df = {chi2_disp/df_disp:.2f}"
          f"  ⇒ {'★ 显著过离散 ⇒ 逃逸不是常数概率，是 seed 级 regime' if chi2_disp/df_disp > 2.0 else '未见过离散'}")

    out["escape_per_seed"] = per
    out["pooled_p"] = pool
    out["dispersion_chi2"] = chi2_disp
    out["dispersion_df"] = df_disp
    out["dispersion_ratio"] = chi2_disp / df_disp
    (_HERE / "out").mkdir(parents=True, exist_ok=True)
    (_HERE / "out" / "diag_gof.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n⇒ 已写 out/diag_gof.json")


if __name__ == "__main__":
    main()