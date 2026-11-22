r"""DIR-63 Stage A 守卫 A63-1…8（rev 1.0 §5；任一 FAIL 不得进正式轮）"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import R1_task63 as R
from ref63 import (D, LAM, ETA, KEYS, P, P0, v0, v1, J_layer, Jd_analytic,
                   Jc_analytic, generic)

RES = []
e = lambda i: np.eye(D)[:, i:i + 1]


def chk(tag, ok, detail):
    RES.append({"guard": tag, "ok": bool(ok), "detail": detail})
    print("%-70s %s | %s" % (tag, "PASS" if ok else "**FAIL**", detail))


# A63-1 代数自检：解析 vs 通用矩阵式
worst = 0.0
for thd in (0, 30, 45, 60, 90):
    th = np.deg2rad(thd)
    g = generic(th)
    worst = max(worst, abs(g["J1"] - Jc_analytic(th)), abs(g["Jd"] - Jd_analytic()),
                abs(g["J0"] - LAM / (1 + LAM)))
chk("A63-1 代数自检（解析 vs 矩阵式，5 档）", worst <= 1e-12, "worst=%.2e" % worst)

# A63-2 键集归一
Cx = sum(k @ k.T for k in KEYS) / len(KEYS)
chk("A63-2 键集归一 C_x=I", float(np.max(np.abs(Cx - np.eye(D)))) <= 1e-14,
    "max|C_x-I|=%.2e" % float(np.max(np.abs(Cx - np.eye(D)))))

# A63-3 单层装置锚（N=2e4）
o = R.run_direct(20261119, n=200000)
rel = abs(o["J"] - Jd_analytic()) / Jd_analytic()
chk("A63-3 单层装置锚（N=2e5, η=0.05, 批梯度）", rel <= 1e-6, "J=%.9f rel=%.2e" % (o["J"], rel))

# A63-4 秩不绑
g0 = generic(0.0); g90 = generic(np.pi / 2)
chk("A63-4 秩不绑（θ=0 两层秩=2）", g0["r0"] == 2 and g0["r1"] == 2,
    "rank(W0*,W1*(0))=%d,%d ；记录 rank(W1*(90°))=%d" % (g0["r0"], g0["r1"], g90["r1"]))

# A63-5 确定性
a = R.run_stack(0.0, 20261119, n=3000); b = R.run_stack(0.0, 20261119, n=3000)
chk("A63-5 确定性（两次逐位相同）", np.array_equal(a["W0"], b["W0"]) and np.array_equal(a["W1"], b["W1"]),
    "max|ΔW|=%.1e" % max(float(np.max(np.abs(a["W0"] - b["W0"]))), float(np.max(np.abs(a["W1"] - b["W1"])))))

# A63-6 直连基线
chk("A63-6 直连基线 λ/(1+λ)", abs(generic(0.0)["Jd"] - LAM / (1 + LAM)) <= 1e-12,
    "Jd=%.12f  λ/(1+λ)=%.12f" % (generic(0.0)["Jd"], LAM / (1 + LAM)))

# A63-7 陷阱继承（两层零初始化仍冻结，批梯度）
C0 = sum(k @ k.T for k in KEYS) / len(KEYS)
B0 = sum(v0(k, 0.0) @ k.T for k in KEYS) / len(KEYS)
U0 = np.zeros((D, 2)); V0 = np.zeros((D, 2)); U1 = np.zeros((D, 2)); V1 = np.zeros((D, 2))
dmax = 0.0
for t in range(2000):
    W0 = U0 @ V0.T
    G0 = (W0 @ C0 - B0) + LAM * W0
    U0 = U0 - ETA * (G0 @ V0); V0 = V0 - ETA * (G0.T @ U0)
    W0 = U0 @ V0.T
    hs = [W0 @ k for k in KEYS]
    C1 = sum(h @ h.T for h in hs) / len(hs)
    B1 = sum(v1(KEYS[i]) @ hs[i].T for i in range(D)) / len(hs)
    W1 = U1 @ V1.T
    G1 = (W1 @ C1 - B1) + LAM * W1
    U1 = U1 - ETA * (G1 @ V1); V1 = V1 - ETA * (G1.T @ U1)
    dmax = max(dmax, float(np.linalg.norm(U0 @ V0.T)), float(np.linalg.norm(U1 @ V1.T)))
chk("A63-7 零初始化陷阱继承（须冻结）", dmax <= 1e-12, "max||W_l||=%.2e" % dmax)

# A63-8 阈值分辨力标定
d = [Jc_analytic(np.deg2rad(t)) - Jd_analytic() for t in (0, 30, 45, 60, 90)]
gaps = [abs(d[i + 1] - d[i]) for i in range(len(d) - 1)]
ratio = min(gaps) / 1e-6
chk("A63-8 阈值分辨力（min 相邻间隔 / 1e-6 ≥ 1e3）", ratio >= 1e3,
    "ΔJ=%s  min_gap=%.4f  ratio=%.0f" % (["%.6f" % x for x in d], min(gaps), ratio))

out = {"guards": RES, "n_pass": sum(1 for x in RES if x["ok"]), "n": len(RES)}
if len(sys.argv) > 2 and sys.argv[1] == "--out":
    import json, os as _os
    _os.makedirs(_os.path.dirname(sys.argv[2]), exist_ok=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
print("\n★ Stage A: %d/%d PASS" % (out["n_pass"], out["n"]))
