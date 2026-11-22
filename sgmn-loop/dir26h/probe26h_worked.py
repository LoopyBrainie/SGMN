"""DIR-26-H · probe26h_worked.py —— 卡片 §3 的**唯一算例核对**（$N{=}10$, seed 20261028）。

核对项（逐位，全部打印）：
  $c_\\tau$、$k_B$、$\\sum_iq_i$、$\\bar q$、$\\max q_i$、$q_i$ 的取值分布、
  `round_quota(q,w)` 在 $q\\in[0..\\max q]$ 上的三元组、$\\lvert\\mathrm{keep}_F\\rvert=\\lvert\\mathrm{keep}_K\\rvert=k_B$、
  实测剖面 $F$ 与 $K$ 的严格镜像、$K$ 臂发放数。
核对失败 ⇒ 不得跑正式轮（卡片 §3）。
"""
import collections
import sys

import numpy as np

for _p in ("dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

from R1_gating import D_MODEL, N_DIST, M, build_stream, drive, make_core
from R1_gating2 import item_counts
import R1_gating5 as G5
from R1_gating5 import calibrate_w, theta_of
from R1_gating6 import _pos_idxs                     # noqa: F401
import R1_gating7 as G7

R_PRES, R_MAIN, R0 = 4, 0.40, 8
N, SEED = 10, 20261028


def main():
    ok = True
    D = build_stream(SEED, D_MODEL, N, R_PRES, theta_of(0.9))
    G5.N_DIST = N
    c = calibrate_w(R0, R_MAIN, R_PRES, theta_of(0.9))
    G5.N_DIST = N_DIST
    tau = c["c_tau"]
    print(f"算例：N={N} seed={SEED} r={R0} R={R_PRES} M={M} T={R_PRES * N * 2 * M}")
    print(f"  c_tau = {tau:+.4f}   skip_rate {c['skip_rate']:.3f} gap {c['gap']:+.4f} rep {c['skip_frac_repeat']:.3f}")
    print(f"  A26H-3 复位：R1_gating5.N_DIST == 35 ? {G5.N_DIST == N_DIST}")

    core = make_core(D_MODEL, SEED, R0)
    kB, log = drive(core, D, "B", tau=tau)
    keepB = {t for t, e in enumerate(log) if e["issued"]}
    cnt = item_counts(keepB, N)
    qmax = max(cnt)
    print(f"  k_B = {kB}   Σq_i = {sum(cnt)}   q̄ = {kB / N:.2f}   max q_i = {qmax}")
    print(f"  q_i 分布 = {dict(sorted(collections.Counter(cnt).items()))}")
    print(f"  A26H-7 载荷：q̄∈[9.1,11.6] ? {9.1 <= kB / N <= 11.6}   逐条目 Σq_i==k_B ? {sum(cnt) == kB}")

    print("  round_quota(q, w) 全表：")
    tbl_ok = True
    for q in range(0, qmax + 1):
        row = {pk: G7.round_quota(q, G7.PROFILE[pk]) for pk in ("F", "U", "K")}
        for pk, v in row.items():
            tbl_ok = tbl_ok and sum(v) == q and all(0 <= x <= 2 * M for x in v)
        mir = row["K"] == row["F"][::-1]
        tbl_ok = tbl_ok and mir
        print(f"    q={q:>2}  F{tuple(row['F'])}  U0{tuple(row['U'])}  K{tuple(row['K'])}  镜像 {mir}")
    print(f"  ★ 全表：Σ次数==q、每轮 ≤2M、K 为 F 的反演 —— 全部成立 ? {tbl_ok}")
    ok = ok and tbl_ok

    prof, keeps = {}, {}
    for pk in ("F", "U", "K"):
        keep, per_item = G7.build_keep(N, R_PRES, cnt, G7.PROFILE[pk])
        keeps[pk] = keep
        prof[pk] = G7.realized_profile(keep, N, R_PRES)
        quota_ok = all(sum(per_item[i]) == cnt[i] for i in range(N))
        c2 = make_core(D_MODEL, SEED, R0)
        kk, _ = drive(c2, D, "R", keep=keep)
        print(f"  {pk}u: |keep| {len(keep):>3} 发放 {kk:>3}  ==k_B {kk == kB == len(keep)} "
              f"逐条目配额 {quota_ok} 实测剖面 {[round(x, 3) for x in prof[pk]]}")
        ok = ok and kk == kB == len(keep) and quota_ok
    mir = all(abs(prof["F"][r] - prof["K"][R_PRES - 1 - r]) < 1e-12 for r in range(R_PRES))
    print(f"  ★ 实测剖面 F 与 K 严格镜像 ? {mir}   （F{ [round(x, 3) for x in prof['F']] } vs K{ [round(x, 3) for x in prof['K']] }）")
    print(f"  ★ |keep_F| == |keep_K| ? {len(keeps['F']) == len(keeps['K'])}")
    ok = ok and mir and len(keeps["F"]) == len(keeps["K"])
    print(f"\n{'★ 算例核对通过：可以开跑正式轮' if ok else '✗ 算例核对失败：不得跑正式轮'}")


if __name__ == "__main__":
    main()
