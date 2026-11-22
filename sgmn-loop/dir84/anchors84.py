"""DIR-84 开卡前算例核对（`DEF-64-1` / 轮次标准 1(c) 的"一个具体算例"要求）。

目的：证明**绝对**口径 `v0 + d_abs*delta_hat` 与 DIR-83 的**相对**口径
`v0 + eps*||v0||*delta_hat` 在 `d_abs = eps*||v0||` 时是**同一条轨迹**
（t_nan 必须逐位相同）。若不成立，说明新测量层不等价，不得开卡。
"""
import sys, os, json, time
for q in ("dir70", "dir72", "dir73", "dir78", "dir79", "dir80", "dir81", "dir82", "dir83", "dir84"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np
import dev83, dev84

# DIR-83 formal83.json 的冻结读数（eps=3e-2 与 eps=1e-1，seed 7777）
D83 = {
    3.0: dict(nv0_w=None, eps=[(0.03, True, None), (0.1, False, 13)], d=[(0.1314, True), (0.4381, False)]),
    10.0: dict(nv0_w=None, eps=[(0.03, True, None), (0.1, False, 7)], d=[(0.2032, True), (0.6774, False)]),
    20.0: dict(nv0_w=None, eps=[(0.03, False, 51), (0.1, False, 5)], d=[(0.2749, False), (0.9165, False)]),
    30.0: dict(nv0_w=None, eps=[(0.03, False, 8), (0.1, False, 4)], d=[(0.3315, False), (1.1050, False)]),
}

res = {"gamma": [], "checks": []}
ok_all = True
t0 = time.time()
for g in dev84.GAM:
    e = dev83.eta_mid(g)
    nv0 = dev84.nv0_of(g)
    w1s = dev84.w1star_of(g)
    fz = dev83.lookup(g)
    row = dict(g=g, eta_mid=e, nv0=nv0, W1_star=w1s, W1_frozen=fz["W"],
               W1_rel_diff=abs(w1s - fz["W"]) / fz["W"])
    # 1) ||W1*|| 与冻结值一致（A2 的前置）
    if row["W1_rel_diff"] > 1e-6:  # DEF-84-1: FROZEN["W"] 是 6 位截断值，不能用 1e-9
        ok_all = False
        res["checks"].append(dict(what="W1_star vs frozen", g=g, ok=False, rel=row["W1_rel_diff"]))
    # 2) 绝对口径 = 相对口径（同一条轨迹）
    rr = []
    for (eps, fin83, tnan83) in D83[g]["eps"]:
        d_abs = eps * nv0
        r = dev84.traj_abs(g, e, d_abs, seed=7777)
        same_fin = (r["escaped"] is (not fin83))
        same_t = (r["t_nan"] == tnan83) if (not fin83) else (r["t_nan"] is None)
        rec = dict(g=g, eps=eps, d_abs=d_abs, escaped=r["escaped"], t_nan=r["t_nan"],
                   t_div=r["t_div"], mx_rel=r["mx_rel"], d_final=r["d_final"],
                   same_finite=same_fin, same_t_nan=same_t)
        if not (same_fin and same_t):
            ok_all = False
        rr.append(rec)
        res["checks"].append(dict(what="absolute==relative", **rec))
    row["anchors"] = rr
    res["gamma"].append(row)
    print("γ=%-4g nv0=%.6f ||W1*||=%.6f (冻结 %.6f, rel %.2e)" %
          (g, nv0, w1s, fz["W"], row["W1_rel_diff"]))
    for rec in rr:
        print("   eps=%-5g d_abs=%.6f escaped=%-5s t_nan=%-5s t_div=%-4s mx_rel=%.3e d_final=%.3e  [finite=%s t=%s]" %
              (rec["eps"], rec["d_abs"], rec["escaped"], rec["t_nan"], rec["t_div"],
               rec["mx_rel"], rec["d_final"], rec["same_finite"], rec["same_t_nan"]))

# 3) 计时：一次 2e5 步未逃逸运行的代价（预算用）
tt = time.time()
_ = dev84.traj_abs(10.0, dev83.eta_mid(10.0), 0.20, seed=7777)
one_run = time.time() - tt
res["timing_one_run_2e5"] = one_run
res["elapsed"] = time.time() - t0
res["allok"] = ok_all
json.dump(res, open(os.path.join(dev84.OUT84, "anchors84.json"), "w"), indent=1)
print("\n一次 2e5 未逃逸运行: %.2f s  ⇒ 3 方向 × 4 γ × ~8 次二分 ≈ %.1f min" %
      (one_run, 3 * 4 * 8 * one_run * 0.55 / 60))
print("allok =", ok_all, "| elapsed %.1f s" % res["elapsed"])