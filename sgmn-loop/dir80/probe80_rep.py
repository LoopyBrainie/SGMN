"""DIR-80 装置探针（非判据）：keys γ=10 上"代表点不同 ⇒ ρ_λ 相差 1.77×"是
(a) 不动点真的随 η_warm 变，还是 (b) 有一个非解析的第二不动点，还是 (c) 数值伪像？
同时核 D-91 的代数预言 u1*(keys,10)=0.0950。"""
import sys
for q in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79", "dir80"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np, ref70, warm78, jac80

C, MODE = 6.0, "keys"
print("pred_keys(10) =", ref70.pred_keys(C, 10.0), flush=True)
print("pred_keys(3)  =", ref70.pred_keys(C, 3.0), flush=True)
for g in (10.0, 3.0):
    for eta_w in ([0.0045, 0.0248, 0.026] if g == 10.0 else [0.06, 0.15]):
        x, info = warm78.warm_fp(MODE, g, eta_w)
        u1 = float((x[0] @ x[1].T)[0, 0]); u2 = float((x[2] @ x[3].T)[0, 0])
        rho, dg = jac80.rho_scan(x, MODE, g, lo=1e-6, hi=1.0)
        print("γ=%-4g eta_w=%.4g  u1=%.6f u2=%.6f  fp_res=%.1e steps=%d  rho=%.6f (nev=%d)"
              % (g, eta_w, u1, u2, info["res"], info["steps"], rho, dg.get("nev", -1)), flush=True)
