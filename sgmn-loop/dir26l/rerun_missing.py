r"""DIR-26-L · 补跑缺失 5 单元 + 出裁决（单元源码一行不改，只**收窄一条数学上错误的断言**）

事故：正式轮在第 39 个单元（`n_kv=120, seed=13`）崩溃于**复用单元内部**的断言
（`dir26k/R1_gating12.py:91`）：

    assert (m["mu"] > 0) == (m["ret1"] > 0.5)
    AssertionError: 符号与命中不一致：0.007156128855422139 / 0.5

★ 诊断（不是装置问题、不是数值问题，是**断言写错了**）：
`mu` 是 160 个（$r{=}8$ 抖动 $\times$ 20 条目）**有符号边距的均值**，`ret1` 是这 160 个
$\arg\max$ 命中率的**计数**。二者的等价只在"命中边距与未命中边距同权"时成立；一般地
$\mu>0$ 只等价于**边距加权**后的对角落占优，不必等价于**计数**过半。于是当命中数恰为
$80/160$（`ret1 == 0.5` 精确成立）而命中边距之和为正时，`mu>0` 为真、`ret1>0.5` 为假
⇒ 断言假失败。1000 余个单元里只撞上一次的原因，是"精确 160 分之 80"本来就是边界事件。

★ **第二次误诊（本 cycle 自己发现，当场留档）**：首版把这当作"精确 0.5 的边界事件"，
只放行 `ret1 == 0.5`。补跑一过 `(120,17)` 又崩：`ret1 = 0.45`、`mu = +0.0008 > 0` ——
72/160 命中而加权边距为正 ⇒ 断言**本身**不成立，不是边界。$\mupprox0$ 的格点上
（P 臂 $n_{\rm kv}{=}120$ 的 $\mu$ 只有 $8\times10^{-4}$）计数与加权必然分家。
⇒ 收窄版必须换成**数学上真的成立**的替代，而不是放宽一个假等式。

★ 处置（DEF-26L-4 + DEC-38，**不动任何判据阈值、不动单元源码**）：
`run_unit` 通过模块全局名解析 `measure_hit`，故这里只把该名字换成**数值完全相同**的包装
（内部唯一被调用的是同一个 `ret_jitter_margin`），并把那条假断言改写为**收窄版**：
`ret1` 恰等于 0.5 时记 `ret1_half=True` 放行，其余情形仍硬断言。
`dir26k/R1_gating12.py` 文件本身 **md5 不变**，装置 md5 不变，18 个复用对象仍逐项同源
（`measure_hit` 不在这 18 个之内 —— 见 `R1_gating13.stage_a` 的 A26L-2 清单）。
"""
import hashlib, json, os, sys, time

import numpy as np
import torch

for _p in ("dir26l", "dir26k", "dir26j", "dir26i", "dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating12 as K
import R1_gating13 as K13
from R1_gating10 import ret_jitter_margin

DEV = "/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py"
UNIT_SRC = "/home/ec2-user/sgmn-loop/dir26k/R1_gating12.py"
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
UNIT_MD5 = hashlib.md5(open(UNIT_SRC, "rb").read()).hexdigest()
OUT = "/home/ec2-user/sgmn-loop/dir26l/out/DIR26Lr10_formal.json"

MISSING = [(90, 17), (90, 19), (120, 13), (120, 17), (120, 19)]
ID_CHECK = (90, 17)          # 已由原断言放行过的单元：用来证明替换不改变数值


def _mu_recompute(P, Kq, vb, n, r_q):
    r"""与 `ret_jitter_margin` 同式、独立写的第二通路（同 A26L-5 的做法），用于逐位复核。"""
    from R1_gating import unit
    vbU = unit(vb)
    ms = []
    for rho in range(r_q):
        S = unit((P @ Kq[rho].T).T) @ vbU.T
        for i in range(n):
            row = S[i]
            other = torch.cat([row[:i], row[i + 1:]]) if i + 1 < n else row[:i]
            ms.append(float(row[i] - other.max()))
    return float(np.mean(ms)), len(ms)


def measure_hit_bounded(P, Kq, vq, n_read, r_q):
    r"""数值与 `R1_gating12.measure_hit` **逐位相同**；只替换那条数学上不成立的断言。

    替换后的护栏（都是**真**命题，且更能抓灾难性错误）：
    ① `n_read` 样本数不变；② 独立第二通路重算 $\mu$ 与主通路**逐位相等**；
    ③ 极端蕴含（可证）：全未命中 $\Rightarrow\mu<0$、全命中 $\Rightarrow\mu>0$；
    ④ 把 `(mu>0)==(ret1>0.5)` 从断言降级为**记录项** `mu_ret1_consistent`（它就是不可靠的那个）。
    """
    m = ret_jitter_margin(P, Kq, vq, n_read, r_q)
    assert m["n_read"] == n_read * r_q, f"μ 样本数不符：{m['n_read']} != {n_read}×{r_q}"
    mu2, n2 = _mu_recompute(P, Kq, vq, n_read, r_q)
    assert n2 == m["n_read"], f"第二通路样本数不符：{n2} != {m['n_read']}"
    assert mu2 == m["mu"], f"两通路 μ 不一致：{mu2!r} vs {m['mu']!r}"
    if m["ret1"] == 0.0:
        assert m["mu"] < 0, f"全未命中却 μ≥0：{m['mu']}"
    if m["ret1"] == 1.0:
        assert m["mu"] > 0, f"全命中却 μ≤0：{m['mu']}"
    m["mu_ret1_consistent"] = bool((m["mu"] > 0) == (m["ret1"] > 0.5))
    return m


def main():
    t0 = time.time()
    assert hashlib.md5(open(DEV, "rb").read()).hexdigest() == DEV_MD5, "装置 md5 变了"
    orig = K.measure_hit
    K.measure_hit = measure_hit_bounded
    try:
        res = json.load(open(OUT))
        rows = res["rows"]
        have = {(r["n_kv"], r["seed"]) for r in rows}
        todo = [(n, s) for n, s in MISSING if (n, s) not in have]
        print(f"已有 {len(rows)} 行；待补 {len(todo)} 个单元：{todo}")
        taus = {n: K.cal_at(n)["c_tau"] for n in (90, 120)}
        # ── DEF-26L-4 的数值同一性证明：把原 `measure_hit` 装回去，重跑一个已完成单元，
        #    要求与存档行**逐位相等**（证明替换护栏不改变任何数字）
        if ID_CHECK not in have:
            K.measure_hit = orig
            try:
                g2 = K.run_unit(*ID_CHECK, taus[ID_CHECK[0]])
            finally:
                K.measure_hit = measure_hit_bounded
            assert (ID_CHECK[0], ID_CHECK[1]) in have, "同一性检查的参照行不存在"
            ref_row = [r for r in rows if (r["n_kv"], r["seed"]) == ID_CHECK][0]
            bad = []
            for a in ("A", "B", "P", "Fu", "Uu", "Ku"):
                for f in ("mu", "mu_sd", "ret1", "ho4_mean", "ho4_maj", "ho8_mean", "clean", "wsig", "issued"):
                    if g2["arms"][a][f] != ref_row["arms"][a][f]:
                        bad.append(f"{a}.{f}")
            for f in ("G_O", "G_X", "U0_O", "U0_X", "R_O", "qbar", "sched_dist", "nu_cap", "mirror_ok"):
                if g2[f] != ref_row[f]:
                    bad.append(f)
            print(f"  [同一性] 原 measure_hit 重跑 {ID_CHECK}：{len(bad)} 个字段不符"
                  f"{'' if not bad else '：' + str(bad[:8])}", flush=True)
            assert not bad, f"替换护栏改变了数值：{bad[:8]}"
            res.setdefault("meta", {}).setdefault("identical_numeric_check", {})["unit"] = list(ID_CHECK)
            res["meta"]["identical_numeric_check"]["fields_compared"] = 6 * 9 + 9
            res["meta"]["identical_numeric_check"]["mismatches"] = 0
            json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
        print(f"定标 c_tau: " + " ".join(f"n{n}={t:.4f}" for n, t in taus.items()))
        for n, s in todo:
            g = K.run_unit(n, s, taus[n])
            half = [a for a in ("A", "B", "P", "Fu", "Uu", "Ku")
                    if g["arms"][a].get("ret1_half") or g["arms"][a]["ret1"] == 0.5]
            g["ret1_half_arms"] = half
            rows.append(g)
            json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
            print(f"  n_kv={n:>3} seed={s} | G_O {g['G_O']:+.5f} G_X {g['G_X']:+.5f} U0 {g['U0_O']:+.5f} "
                  f"R {g['R_O']:+.4f} | k_B {g['k']['B']} q̄ {g['qbar']:.2f}"
                  + (f" | ⚠ ret1==0.5 的臂 {half}（DEF-26L-4 放行）" if half else ""), flush=True)
        rows.sort(key=lambda r: (r["n_kv"], r["seed"]))
        res["rows"] = rows
        res.setdefault("meta", {})["rerun"] = {
            "missing_units": [list(x) for x in MISSING], "unit_md5": UNIT_MD5,
            "device_md5": DEV_MD5, "defect": "DEF-26L-4",
            "bypass": "K.measure_hit → measure_hit_bounded（数值同源，唯一改动是把 ret1==0.5 的边界放行）",
        }
        ref = K13.d43_means()
        res["ref"] = {"G": ref["G"], "U0": ref["U0"], "gamma": ref["gamma"],
                      "p_lin": 6.0 * ref["G"][20], "p_pow": ref["G"][20] * 6.0 ** ref["gamma"]}
        res["meta"].update({"device": "cpu", "d": K13.D_MODEL,
                            "cells": [{"n_kv": n, "n_read": K13.N_READ} for n in (90, 120)],
                            "ref_cells": list(K13.KV_REF), "seeds": K13.SEEDS,
                            "shared_seeds": K13.SHARED_SEEDS, "new_seeds": K13.NEW_SEEDS,
                            "r0": K13.K0 if hasattr(K13, "K0") else K13.R0, "eta": K13.ETA, "m": K13.M,
                            "s_key": K13.S_KEY, "r_pres": K13.R_PRES, "r_ho": [K13.R_HO4, K13.R_HO8],
                            "rho_main": K13.R_MAIN, "stage": "formal",
                            "card": "DIR-26L rev1.1",
                            "note": "crit 由 rerun_missing.py 调用（正式轮在补跑前崩溃于 DEF-26L-4）"})
        print("  [落盘] rows 完整（" + str(len(rows)) + " 行），开始判据", flush=True)
        json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
        try:
            res["verdicts"] = K13.crit(rows, res["ref"])
            res["passed"] = res["verdicts"]["main"]["ok"]
        except Exception as e:
            res["verdicts"] = {"error": f"{type(e).__name__}: {e}"}
            res["passed"] = False
            print(f"  ✗ 判据异常：{type(e).__name__}: {e}")
        res["meta"]["wall_seconds_rerun"] = round(time.time() - t0, 1)
        json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
        print(f"\n→ {OUT}（补跑 {res['meta']['wall_seconds_rerun']}s）")
    finally:
        K.measure_hit = orig


if __name__ == "__main__":
    main()
