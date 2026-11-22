"""DIR-72 · 代数预言（**先于开卡**，只算特征多项式，不跑任何轨迹）。
矩阵空间线性化在 P-方向上的 2x2 块（c=D=6 ⇒ cb=1, λ=0.05）：
  M11 = cb+λ,  M12 = −γ·cb,  M21 = 2·cb·u1·(u2−1),  M22 = cb·u1² + λ
（推导：C = cb·I、B = cb·P（因键 = √c·e_i）、C2 = cb·u1²·P、B2 = P·C2。
  δC2 = cb(δW1 P + P δW1ᵀ)；在 P-方向上 δW = aP ⇒ δC2 = 2cb·a·P。）
预言：η*_pred = 2/λ_max(M)；同时给出补空间特征值 cb+λ ⇒ η*=2/(cb+λ)。
"""
import sys, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, ref70
C_ = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
cb, lam = C_ / ref70.D, ref70.LAM
rows = []
for mode, g in (("target", 3), ("target", 10), ("target", 30), ("keys", 3), ("keys", 10)):
    p = ref70.pred_target(C_, g) if mode == "target" else ref70.pred_keys(C_, g)
    u1, u2 = p["u1"], p["u2"]
    M = np.array([[cb + lam, -g * cb], [2 * cb * u1 * (u2 - 1.0), cb * u1 * u1 + lam]])
    ev = np.linalg.eigvals(M)
    lmax = float(max(ev.real))
    rows.append(dict(mode=mode, gamma=g, u1=u1, u2=u2, M=M.tolist(), ev=ev.real.tolist(),
                     lmax=lmax, eta_pred=2.0 / lmax, C_pred=(2.0 / lmax) * g * g,
                     eta_pred_complement=2.0 / (cb + lam)))
out = dict(cb=cb, lam=lam, rows=rows)
json.dump(out, open("/home/ec2-user/sgmn-loop/dir72/out/algebra72_v11.json", "w"), indent=1)
print("mode γ      u1*       u2*      λmax(M)   η*_pred    C_pred")
for r in rows:
    print("%-6s %-4g %9.5f %9.6f %9.4f %10.6f %8.4f" % (r["mode"], r["gamma"], r["u1"], r["u2"], r["lmax"], r["eta_pred"], r["C_pred"]))
print("补空间: η*_pred = 2/(cb+λ) = %.6f" % (2.0 / (cb + lam))); print("toy(1D 因子图) 解析: η* = 1/b = %.6f  ← 与矩阵空间 2/(cb+λ) 差 2 倍" % (1.0 / cb))
