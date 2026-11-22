r"""DIR-30 · probe30_cross.py —— **判后补充报告**：定位对称尺度轴上 R 跨越 R_full-0.10 的交叉点。

★ **这是判后（post-verdict）的补充报告**，不改正式轮的任何一个读数、不改任何阈值：
  正式轮的判据（振幅 ≥ 0.15）已在 formal 跑完时判定；本探针只把**交叉尺度**从
  「夹在 σ0 ∈ (1,2)」细化到更窄的区间。

用法：/opt/pytorch/bin/python dir30/probe30_cross.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import R1_init as D

OUT = "/home/ec2-user/sgmn-loop/dir30/out"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()

rows = []
for s0 in (1.0, 1.25, 1.5, 1.75, 2.0):
    e0, _, i0 = D.find_eta_c_col("fac", 0.0, init="init-A", sig0=s0)
    e1, _, i1 = D.find_eta_c_col("fac", D.GAMMA_CONF, init="init-A", sig0=s0)
    R = None if (e0 is None or e1 is None) else e1 / e0
    rows.append({"sigma0": s0, "eta_c_g0": e0, "eta_c_g05": e1, "R": R,
                 "censored_max": max(i0["censored_frac"], i1["censored_frac"])})
    print(f"  sym σ0={s0}: η_c(0)={e0} η_c(0.5)={e1} R={R}", flush=True)

target = D.RES_R_FULL_MINUS_GAP  if hasattr(D, "RES_R_FULL_MINUS_GAP") else 0.28108836861596265
below = [r["sigma0"] for r in rows if r["R"] is not None and r["R"] <= target]
res = {"note": "判后补充报告：交叉尺度定位（不改正式轮读数）",
       "target_R_full_minus_gap": target, "rows": rows,
       "sigma_below_target": below,
       "sigma_star_bracket": ([max(x for x in below), 2.0] if below else None),
       "secs": round(time.time() - T0, 1)}
json.dump(res, open(os.path.join(OUT, "DIR30r10_cross.json"), "w"),
          ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
