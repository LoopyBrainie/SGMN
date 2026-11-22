"""DIR-29 · 正式轮后诊断（**不改变任何判据**）：两件事。

① **同值之谜**：`fac(σ0=0.5, γ=0)` 与 `full(γ=0.5)` 的 η_c 都报 0.18799491692492837。
   查两者在 40 点网格上的**过零区间是否相同**（若是 ⇒ 二分路径相同 ⇒ 返回同一中点，
   属**二分中点的机械巧合**，而非两个不同过程真的相等）。
② **公平机制读数**：正式轮的 k2sq 报告格把两臂都放在 η=0.25，而 fac 臂的门槛只有
   0.177 ⇒ 在 0.25 上它**早就过了门槛并早死** ⇒ 两臂的有效输入膨胀不可比。
   这里改在**各臂自己的** η_ref=0.25·η_c(0) 上测 E[k2²]（仍是报告量）。
"""
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir29")
import numpy as np
from R1_tangent import find_eta_c, run, top_median

# ① 过零区间
for tag, arm, g, sig0 in (("fac(s0=0.5,g=0)", "fac", 0.0, 0.5), ("full(g=0.5)", "full", 0.5, 1.0)):
    grid = np.exp(np.linspace(np.log(1e-3), np.log(1.2), 40))
    _, rg = top_median(arm, grid, g, sig0=sig0)
    meds = np.median(rg["lam_top"], axis=0)
    idx = [j for j in range(39) if meds[j] <= 0.0 and meds[j + 1] > 0.0]
    j = idx[0] if idx else None
    print(f"{tag}: 过零区间 index={j} bracket=[{grid[j]:.6f},{grid[j+1]:.6f}] "
          f"meds[j]={meds[j]:+.6f} meds[j+1]={meds[j+1]:+.6f}", flush=True)
    if j is not None:
        print(f"   相邻中位 λ̂: {[round(float(x),6) for x in meds[max(0,j-2):j+3]]}", flush=True)

# ② 各臂自己的 η_ref 上的有效输入二阶矩
print()
for arm in ("full", "fac"):
    ec0, _, _ = find_eta_c(arm, 0.0)
    ref = 0.25 * ec0
    for g in (0.0, 0.5):
        r = run(arm, ref, g)
        print(f"{arm}: η_c(0)={ec0:.6f}  η_ref={ref:.6f}  γ={g}: "
              f"median E[k2²]={float(np.median(r['k2sq'])):.4f}  "
              f"截断列={r['n_trunc']}  median λ̂_top={float(np.median(r['lam_top'])):+.6f}", flush=True)
