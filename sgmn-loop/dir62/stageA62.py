r"""DIR-62 Stage A 守卫 A62-1…8（卡 rev 1.0 §5；任一 FAIL 不得进正式轮）"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import R1_preload62 as R
from ref62 import D, LAM, k, v, preload, J, closed, svd_r

RES = []
E = lambda i: np.eye(D)[:, i:i + 1]
C = k @ k.T
B = v @ k.T
RS = R.RSYM


def chk(tag, ok, detail):
    RES.append({"guard": tag, "ok": bool(ok), "detail": detail})
    print("%-72s %s | %s" % (tag, "PASS" if ok else "**FAIL**", detail))


def gradfun(W, W0):
    return (W @ k - v) @ k.T + LAM * (W - W0)


W0s = {t: preload(p) for t, p in R.REG.items()}

# ---- A62-1 平衡点锚 ----
worst = 0.0
for t, W0 in W0s.items():
    Ws = closed(W0)
    r1 = float(np.linalg.norm(Ws @ (C + LAM * np.eye(D)) - (B + LAM * W0)))
    r2 = float(np.linalg.norm(gradfun(Ws, W0)))
    worst = max(worst, r1, r2)
    print("   %-7s ||W*(C+λI)-(B+λW0)||=%.2e  ||∇J(W*)||=%.2e" % (t, r1, r2))
chk("A62-1 平衡点锚", worst <= 1e-10, "worst=%.3e" % worst)

# ---- A62-2 有限差分梯度锚 ----
rng = np.random.default_rng(7)
W = rng.normal(size=(D, D)); Dm = rng.normal(size=(D, D)); h = 1e-6
W0 = W0s["WEAK"]
fd = (J(W + h * Dm, W0) - J(W - h * Dm, W0)) / (2 * h)
an = float(np.sum(gradfun(W, W0) * Dm))
chk("A62-2 有限差分梯度锚", abs(fd - an) / abs(an) <= 1e-6, "rel=%.3e" % (abs(fd - an) / abs(an)))

# ---- A62-3 §5.1 恒等式（ΔW 预算） ----
def ident_D(W0, r):
    Ws = closed(W0)
    Mb = Ws - W0
    sig = np.linalg.svd(Mb @ RS, compute_uv=False)
    L = 0.5 * float(np.sum(sig[r:] ** 2))
    Mr = svd_r(Mb @ RS, r) @ np.linalg.inv(RS)
    return abs(J(W0 + Mr, W0) - J(Ws, W0) - L), L, float(np.linalg.matrix_rank(Mb, tol=1e-12))

worst = 0.0; det = []
for t, W0 in W0s.items():
    for r in (1, 2, 3):
        e, L, rk = ident_D(W0, r)
        worst = max(worst, e); det.append("%s/r%d:%.1e(L=%.3e,rank=%d)" % (t, r, e, L, rk))
chk("A62-3 §5.1 恒等式（ΔW 预算）", worst <= 1e-12, " ".join(det))

# ---- A62-4 W 预算恒等式 ----
def ident_W(W0, r):
    Ws = closed(W0)
    sig = np.linalg.svd(Ws @ RS, compute_uv=False)
    L = 0.5 * float(np.sum(sig[r:] ** 2))
    Wr = svd_r(Ws @ RS, r) @ np.linalg.inv(RS)
    return abs(J(Wr, W0) - J(Ws, W0) - L), L

worst = 0.0; det = []
for t, W0 in W0s.items():
    for r in (1, 2, 3):
        e, L = ident_W(W0, r)
        worst = max(worst, e); det.append("%s/r%d:%.1e(L=%.4f)" % (t, r, e, L))
chk("A62-4 W 预算恒等式", worst <= 1e-12, " ".join(det))

# ---- A62-5 rank M̄ = 1 与 L_A ≡ 0 ----
rk = {t: float(np.linalg.matrix_rank(closed(W0) - W0, tol=1e-12)) for t, W0 in W0s.items()}
LA = {t: ident_D(W0, 1)[1] for t, W0 in W0s.items()}
chk("A62-5 rank M̄=1 且 L_A(r=1)≤1e-30（数值零）",
    all(r == 1.0 for r in rk.values()) and all(x <= 1e-30 for x in LA.values()),
    "rank=%s  L_A=%s" % (rk, LA))

# ---- A62-6 敏感性负对照（键不再正交 ⇒ 存活律必须失效） ----
def g1_of(eps):
    kP = E(2) + eps * E(0)
    pairs = [(1.40, kP, E(0)), (0.50, E(3), E(1)), (0.15, E(4), E(4))]
    W4 = svd_r(closed(preload(pairs)) @ RS, 4) @ np.linalg.inv(RS)
    return float((W4 @ kP).T @ E(0)), (abs(float((W4 @ kP).T @ E(0)) - 1.40))
g_small, d_small = g1_of(0.01)
g_big, d_big = g1_of(0.30)
ratio = d_big / max(d_small, 1e-300)
chk("A62-6 敏感性负对照（非正交 ⇒ 失效）", d_big > 1e-2 and ratio >= 100,
    "dev(0.01)=%.3e dev(0.30)=%.3e ratio=%.0f gain_1(0.30)=%.6f" % (d_small, d_big, ratio, g_big))

# ---- A62-7 确定性 ----
Wa = R.run(None, None, W0s["WEAK"], 2, "W", 20261118)
Wb = R.run(None, None, W0s["WEAK"], 2, "W", 20261118)
chk("A62-7 确定性（两次逐位相同）", np.array_equal(Wa, Wb), "max|ΔW|=%.1e" % float(np.max(np.abs(Wa - Wb))))

# ---- A62-8 DIR-61 陷阱继承（零初始化仍冻结） ----
W0z = W0s["WEAK"]
U = np.zeros((D, 2)); V = np.zeros((D, 2)); dmax = 0.0
for _ in range(2000):
    Wt = U @ V.T
    G = (Wt @ k - v) @ k.T + LAM * (Wt - W0z)
    U = U - R.ETA * (G @ V); V = V - R.ETA * (G.T @ U)
    dmax = max(dmax, float(np.linalg.norm(U @ V.T)))
chk("A62-8 零初始化陷阱继承（须冻结）", dmax <= 1e-12, "max||W_t||=%.3e" % dmax)

# ---- 附：正式 JSON 的 sens 交叉核对 ----
fj = "dir62/out/DIR62r10_formal.json"
if os.path.exists(fj):
    s = json.load(open(fj)).get("sens", {})
    print("   [交叉] formal sens: g_ref=%.6f dev=%.3e" % (s.get("g_ref", np.nan), s.get("dev", np.nan)))

out = {"guards": RES, "n_pass": sum(1 for x in RES if x["ok"]), "n": len(RES)}
if len(sys.argv) > 2 and sys.argv[1] == "--out":
    os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
print("\n★ Stage A: %d/%d PASS" % (out["n_pass"], out["n"]))
