"""DIR-26-G · 工作算例（卡片 §2，**先跑即核，不是判据**）

核四件事（真装置 $d{=}16$/$N{=}8$ 与真实 $k_B$ 比例两路）：
  (i)   $q_i{=}10$ 的逐轮配额逐位 = F (4,3,2,1) / U0 (2,3,3,2) / K (1,2,3,4)；
  (ii)  $n_\\rho(K)=n_{3-\\rho}(F)$ 逐条目逐位（反演精确）；
  (iii) $\\sum_i\\sum_\\rho n_{i\\rho}=k_B$ 且逐条目 $\\sum_\\rho n_{i\\rho}=q_i$；
  (iv)  实测剖面：F 严格递减、K 严格递增、U0 极差 $\\le0.5$（每条目每轮写入数）。
"""
import sys

import numpy as np

for _p in ("dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

from R1_gating import D_MODEL, N_DIST, build_stream, drive, make_core
from R1_gating2 import item_counts
from R1_gating5 import calibrate_w, theta_of
from R1_gating7 import (R_PRES, R_MAIN, PROFILE, W_F, W_K, build_keep, round_quota,
                        realized_profile, even_counts, jaccard, SEEDS)

print("── (i) 解析配额（含 cap=4 的溢出修复）──")
for q in (16, 15, 13, 10, 9, 8):
    print(f"  q={q}: F={round_quota(q, W_F)} U0={round_quota(q, PROFILE['U'])} K={round_quota(q, W_K)}")

print("\n── (ii) 反演与 (iii) 计数恒等（真实 $k_B$）──")
tau = calibrate_w(8, R_MAIN, R_PRES, theta_of(0.9))["c_tau"]
D = build_stream(SEEDS[0], D_MODEL, N_DIST, R_PRES, theta_of(0.9))
cB = make_core(D_MODEL, SEEDS[0], 8)
kB, logB = drive(cB, D, "B", tau=tau)
cnt_u = item_counts({t for t, e in enumerate(logB) if e["issued"]}, N_DIST)
cnt_e = even_counts(kB, N_DIST)
print(f"  k_B={kB} 不均档逐条目计数 min/max = {min(cnt_u)}/{max(cnt_u)}；均档 = {sorted(set(cnt_e))}")
ok_rev = all(round_quota(cnt_u[i], W_K) == round_quota(cnt_u[i], W_F)[::-1] for i in range(N_DIST))
print(f"  n_ρ(K)=n_(R−1−ρ)(F) 逐条目：{ok_rev}")
for cm, cnts in (("u", cnt_u), ("e", cnt_e)):
    for pk in ("F", "U", "K"):
        keep, per_item = build_keep(N_DIST, R_PRES, cnts, PROFILE[pk])
        quota_ok = all(sum(per_item[i]) == cnts[i] for i in range(N_DIST))
        print(f"  {pk}{cm}: |keep|={len(keep)} Σ计数={sum(item_counts(keep, N_DIST))}=k_B={kB} "
              f"逐条目 Σn=q_i={quota_ok} 实测剖面={['%.2f' % x for x in realized_profile(keep, N_DIST, R_PRES)]}")

print("\n── Jaccard（同计数档内三档剖面互为不同集合）──")
ku = {pk: build_keep(N_DIST, R_PRES, cnt_u, PROFILE[pk])[0] for pk in ("F", "U", "K")}
for a, b in (("F", "U"), ("U", "K"), ("F", "K")):
    print(f"  J({a},{b}) = {jaccard(ku[a], ku[b]):.3f}")

print("\n── (i′) 最小装置 $d{=}16$、$N{=}8$ 上的同一组检查 ──")
D16 = build_stream(SEEDS[0], 16, 8, R_PRES, theta_of(0.9))
c16 = make_core(16, SEEDS[0], 4)
k16, log16 = drive(c16, D16, "B", tau=calibrate_w(4, R_MAIN, R_PRES, theta_of(0.9))["c_tau"])
cnt16 = item_counts({t for t, e in enumerate(log16) if e["issued"]}, 8)
print(f"  k_B(d=16,N=8)={k16}；逐条目计数={cnt16}")
ok = True
for pk in ("F", "U", "K"):
    keep, per_item = build_keep(8, R_PRES, cnt16, PROFILE[pk])
    ok = ok and len(keep) == k16 and all(sum(per_item[i]) == cnt16[i] for i in range(8))
    print(f"  {pk}: |keep|={len(keep)} 剖面={['%.2f' % x for x in realized_profile(keep, 8, R_PRES)]}")
print(f"  ⇒ 最小装置上三档恒等：{ok}")
