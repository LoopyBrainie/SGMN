"""DIR-83 事后摘要（**只读**已有产物：fp npz + formal83.json，不跑新轨迹）。
用途：检验「吸引域半径是**绝对** O(1) 量」而非「相对固定」这一读法。
"""
import json, os, sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir83")
import numpy as np
import dev83 as dv

OUT = dv.OUT
F = json.load(open(os.path.join(OUT, "formal83.json")))
print("γ     ‖W1*‖     ‖v0‖     ε=3e-2→绝对  结果       ε=1e-1→绝对  结果")
print("-" * 92)
pts = []
for g in dv.GAM:
    z = np.load(os.path.join(OUT, "fp_target_g%g.npz" % g))
    xc = (z["U1"], z["V1"], z["U2"], z["V2"])
    nv0 = float(np.linalg.norm(np.concatenate([b.ravel() for b in xc])))
    r = F["rows"][str(int(g)) if str(int(g)) in F["rows"] else str(g)]
    e_s, e_b = r["eps_small"], r["eps_big"]
    a_s, a_b = e_s["eps"] * nv0, e_b["eps"] * nv0
    print("%-5g %-9.4f %-8.4f %-13.4f %-9s %-12.4f %-9s" % (
        g, r["W_ref"], nv0, a_s, "收敛" if e_s["finite"] else "发散@%d" % e_s["t_nan"],
        a_b, "收敛" if e_b["finite"] else "发散@%d" % e_b["t_nan"]))
    pts.append(dict(g=g, nv0=nv0, a_small=a_s, small_ok=e_s["finite"],
                    a_big=a_b, big_ok=e_b["finite"], t_nan_small=e_s["t_nan"]))
    # 对照臂
print()
print("④ 阴性对照（η=0.5ρ, ε=1）的**绝对**扰动 = ‖v0‖：")
for g, p in zip(dv.GAM, pts):
    r = F["rows"][str(int(g)) if str(int(g)) in F["rows"] else str(g)]
    c = r["ctrl"]
    print("    γ=%-4g 绝对扰动=%-9.4f finite=%-5s t_nan=%s" % (g, p["nv0"], c["finite"], c["t_nan"]))
print()
# 绝对口径的边界括号
conv = [p["a_small"] for p in pts if p["small_ok"]]
dive = [p["a_small"] for p in pts if not p["small_ok"]]
print("绝对口径 R_abs 的括号：收敛侧最大 = %s ；发散侧最小 = %s" %
      (("%.4f" % max(conv)) if conv else "无", ("%.4f" % min(dive)) if dive else "无"))
if conv and dive:
    print("⇒ R_abs ∈ [%.4f, %.4f)" % (max(conv), min(dive)))
print()
print("相对口径 ε*（= ε 阈值）在 γ 上：3→收敛于 3e-2，10→收敛于 3e-2，20/30→3e-2 已发散")
print("相对口径若恒定：ε* 应与 γ 无关 —— 实测不成立；绝对口径若恒定：R_abs 应有公共区间 —— 见上。")
json.dump(dict(points=pts, conv_max=max(conv) if conv else None,
               div_min=min(dive) if dive else None), open(os.path.join(OUT, "digest83.json"), "w"),
          ensure_ascii=False, indent=1)
print("\nDIGEST83 DONE")