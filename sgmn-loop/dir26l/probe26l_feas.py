r"""DIR-26-L · probe26l_feas.py —— $n_{\rm kv}\in\{90,120\}$ 的**可行性探针**（冻结前，单种子 1907，只定位不作判据）

回答四件事（全部是"能不能跑/能不能判"，不是结论）：
  ① 两个新格点的 `calibrate_w` 是否收敛，单元耗时多少（决定种子预算）；
  ② $\bar q$ 的分布会不会跌到"写不动"（$q_i$ 太小 ⇒ 日程表达塌掉 ⇒ 剂量因装置而非饱和下降）；
  ③ $\nu_{\rm cap}$（$q_i{=}4R$ 触顶比例）是否仍可忽略；
  ④ 单种子的 $G$ 与两条预测的距离：**线性外推** $p_{\rm lin}=6G_{20}$ 与**两点幂律** $p_{\rm pow}=G_{20}\cdot6^{0.432}$。

★ 探针读数**不进判据、不定阈值**（纪律 (i)）；探针种子 **1907 不入正式轮种子表**，以免污染"全新种子"。
"""
import sys, time
import numpy as np

for _p in ("dir26k", "dir26j", "dir26i", "dir26h", "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating12 as K
from R1_gating12 import cal_at, DEV_MD5, N_READ
from R1_gating import D_MODEL, N_DIST, build_stream
from R1_gating5 import theta_of
from R1_gating7 import md5, PROFILE, build_keep
from R1_gating2 import item_counts

PROBE_SEED = 1907
D43 = "/home/ec2-user/sgmn-loop/dir26k/out/DIR26Kr10_formal.json"
G20, G35, G60, GAM = 0.0062888, 0.0078997, 0.0101107, 0.4325


def main():
    print("=== ① 装置与标定 ===")
    print(f"  装置 md5 OK ? {md5('/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py') == DEV_MD5}")
    taus = {}
    for n in (60, 90, 120):
        t0 = time.time()
        t = cal_at(n)
        taus[n] = t["c_tau"]
        print(f"  n_kv={n:>3}  c_tau={t['c_tau']:.4f}  skip={t['skip_rate']:.3f}  gap={t['gap']:+.3f}  "
              f"rep={t['skip_frac_repeat']:.3f}  gate_ok={t['gate_ok']}  ({time.time()-t0:.1f}s)")
    print(f"  复位 N_DIST==35 ? {K.G5.N_DIST == N_DIST}")

    print("\n=== ② ③ ④ 单元：耗时 / 计数分布 / 单种子 G ===")
    pred = {"lin": G20 * 6.0, "pow": G20 * 6.0 ** GAM}
    print(f"  参照（D-43, 30 种子）：G20 {G20:+.5f} G35 {G35:+.5f} G60 {G60:+.5f} γ̂ {GAM:.3f}")
    print(f"  两条预测：p_lin(120) {pred['lin']:+.5f}  p_pow(120) {pred['pow']:+.5f}")
    for n in (60, 90, 120):
        t0 = time.time()
        g = K.run_unit(n, PROBE_SEED, taus[n])
        dt = time.time() - t0
        cnt = np.array(g["counts"]); rd = cnt[:N_READ]
        print(f"  n_kv={n:>3} {dt:6.1f}s k_B={g['k']['B']:>4} q̄={g['qbar']:.3f} q_max={g['qmax']:>2} "
              f"q_min={cnt.min():>2} ν_cap={g['nu_cap']:.3f} | G_O {g['G_O']:+.5f} G_X {g['G_X']:+.5f} "
              f"U0 {g['U0_O']:+.5f} R {g['R_O']:+.4f} | sched {g['sched_dist']:.3f}")
        print(f"        读集 20 条目的 q_i: min {rd.min()} median {int(np.median(rd))} max {rd.max()} "
              f"| 计数 p10 {np.percentile(cnt,10):.0f} p50 {np.percentile(cnt,50):.0f} p90 {np.percentile(cnt,90):.0f}")
        print(f"        剖面 F {[round(x,3) for x in g['prof']['Fu']]} vs K {[round(x,3) for x in g['prof']['Ku']]} "
              f"| 六臂 μ: " + " ".join(f"{t} {g['arms'][t]['mu']:+.4f}" for t in ("A", "B", "P", "Fu", "Uu", "Ku")))
        if n == 120:
            print(f"        ⇒ 单种子：G120 {g['G_O']:+.5f} 对 p_lin {pred['lin']:+.5f}（差 {g['G_O']-pred['lin']:+.5f}）、"
                  f"对 p_pow {pred['pow']:+.5f}（差 {g['G_O']-pred['pow']:+.5f}）")

    print("\n=== 预算（按实测线性外推）===")
    print("  单种子 6 臂 ≈ (n60)t60 + (n90)t90 + (n120)t120；正式轮只跑 90/120 两个新格点。")


if __name__ == "__main__":
    main()
