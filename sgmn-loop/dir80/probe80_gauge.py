"""DIR-80 装置探针（非判据）：
(1) W* = U V^T 是否与 eta_warm 无关、而 (U,V) 因子分解是否随 eta_warm 变（⇒ D-99 的 gauge）。
(2) gap(eta_warm/eta_th) 的敏感性曲线，keys 与 target 各 4 个代表点规则。
   这条曲线是解释判据 ① 的关键：若 gap 随规则单调变，则 gap 不是动力学量。
"""
import sys, json
for q in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79", "dir80"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np, ref70, warm78, jac80

C, OUT = 6.0, "/home/ec2-user/sgmn-loop/dir80/out/probe80_gauge.json"
res = {}

def wdiff(x, y):
    return float(np.max(np.abs((x[0] @ x[1].T) - (y[0] @ y[1].T))))

def facdiff(x, y):
    return max(float(np.max(np.abs(a - b))) for a, b in zip(x, y))

# ---- (1) 因子分解 vs W*：keys γ=10 三点 ----
A = {}
for eta_w in (0.0045, 0.0248, 0.026):
    x, info = warm78.warm_fp("keys", 10.0, eta_w)
    rho, dg = jac80.rho_scan(x, "keys", 10.0, lo=1e-6, hi=1.0)
    A["%.4g" % eta_w] = dict(eta_w=eta_w, u1=float((x[0] @ x[1].T)[0, 0]), rho=rho,
                             steps=info["steps"], x=x)
res["gauge_keys10"] = {k: dict(eta_w=v["eta_w"], u1=v["u1"], rho=v["rho"], steps=v["steps"])
                       for k, v in A.items()}
ks = list(A)
base = A[ks[0]]["x"]
res["gauge_keys10"]["_diag"] = {k: dict(dW1=wdiff(base, A[k]["x"]), dFac=facdiff(base, A[k]["x"])) for k in ks}
print("(1) keys@10  基=η_w%.4g" % A[ks[0]]["eta_w"], flush=True)
for k in ks:
    d = res["gauge_keys10"]["_diag"][k]
    print("    η_w=%-7s u1=%.6f rho=%.6f  max|ΔW1|=%.2e  max|Δ因子|=%.2e"
          % (k, A[k]["u1"], A[k]["rho"], d["dW1"], d["dFac"]), flush=True)

# ---- (2) 敏感性曲线：两模式 × γ=10（eta_th 用已测的 N=2e4 括号）
ETA_TH = {("keys", 10.0): 0.027032409992084817, ("target", 10.0): 0.007303405}
for mode in ("keys", "target"):
    g, eth = 10.0, ETA_TH[(mode, 10.0)]
    rows = []
    for frac in (0.1, 0.3, 0.6, 0.9):
        eta_w = frac * eth
        x, info = warm78.warm_fp(mode, g, eta_w)
        rho, dg = jac80.rho_scan(x, mode, g, lo=1e-6, hi=1.0)
        rows.append(dict(frac=frac, eta_w=eta_w, rho=rho, gap=rho / eth - 1,
                         steps=info["steps"], res=info["res"]))
        print("(2) %-6s frac=%.1f eta_w=%.4g rho=%.6f gap=%.4f steps=%d"
              % (mode, frac, eta_w, rho, rho / eth - 1, info["steps"]), flush=True)
    res["sens_" + mode] = rows
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print("已写", OUT, flush=True)
