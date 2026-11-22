"""DIR-26-F · 事后诊断（**报告项，非判据**）：四臂把写分配到了哪些轮次与哪个成员。

动机：E 臂用"B 的 $c_{now}$ 升序"选轮次，而 $\\rho{=}0$ 时 $W{=}0\\Rightarrow c_{now}\\equiv0$ ⇒ E 会**系统性地优先选首轮**。
本探针把这一构造后果量化（$\\rho$ 分布、成员 a/b 分布），用于 VERDICT 的机制段。
"""
import sys
import numpy as np
import torch

for _p in ("dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

from R1_gating import D_MODEL, N_DIST, M, build_stream, drive, make_core
from R1_gating2 import keep_positions, item_counts
from R1_gating5 import calibrate_w, theta_of
from R1_gating6 import R_PRES, R_MAIN, SEEDS, build_even, build_uneven

seed = SEEDS[0]
tau = calibrate_w(8, R_MAIN, R_PRES, theta_of(0.9))["c_tau"]
D = build_stream(seed, D_MODEL, N_DIST, R_PRES, theta_of(0.9))
cB = make_core(D_MODEL, seed, 8)
kB, logB = drive(cB, D, "B", tau=tau)
keep_B = {t for t, e in enumerate(logB) if e["issued"]}
cnt = item_counts(keep_B, N_DIST)
keeps = {"A": set(range(R_PRES * N_DIST * 2 * M)), "B": keep_B,
         "E": build_even(N_DIST, R_PRES, kB, logB),
         "P": keep_positions(N_DIST, R_PRES, kB),
         "U": build_uneven(N_DIST, R_PRES, cnt)}


def by_round(keep):
    h = [0] * R_PRES
    for t in keep:
        h[(t // (2 * M)) // N_DIST] += 1
    return [x / N_DIST for x in h]


def by_member(keep):
    h = [0, 0]
    for t in keep:
        h[((t // M) % 2)] += 1
    return [x / N_DIST for x in h]


print(f"k_B={kB}  每条目机会数={2 * R_PRES * M}  逐条目计数极差 B={max(cnt) - min(cnt)}")
print("臂   | 每条目写入数按轮次 ρ=0..3        | 成员 a / b      | 总发放")
for a, k in keeps.items():
    print(f"  {a} | " + " ".join(f"{x:5.2f}" for x in by_round(k))
          + "       | " + " ".join(f"{x:6.2f}" for x in by_member(k))
          + f"  | {len(k)}")
print("\n首轮占比：E %.2f  P %.2f  U %.2f  B %.2f"
      % tuple(by_round(keeps[a])[0] / max(sum(by_round(keeps[a])), 1e-9) for a in "EPUB"))
