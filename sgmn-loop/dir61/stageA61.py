r"""DIR-61 Stage A（卡 §5，rev 1.0 原样；A61-4 预期 FAIL = DEF-61-2 的证据）"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esc61
import R1_init61 as R

RES = []


def chk(tag, ok, detail):
    RES.append({"guard": tag, "ok": bool(ok), "detail": detail})
    print("%-78s %s" % (tag + " | " + detail, "PASS" if ok else "**FAIL**"))


# A61-1 代数锚
d, r, K, V = esc61.device("d4r1")
C = K @ K.T; B = V @ K.T
exp_C = np.zeros((4, 4)); exp_C[:2, :2] = 1.0
exp_B = np.zeros((4, 4)); exp_B[2:, :2] = 1.0
chk("A61-1 代数锚 C,B", np.array_equal(C, exp_C) and np.array_equal(B, exp_B),
    "C: %s  B: %s" % (np.count_nonzero(C), np.count_nonzero(B)))

# A61-2 有限差分梯度锚（RAND t=0）
U0, V0 = R.arms()["RAND"][:2]
W0 = U0 @ V0.T
G = (W0 @ R.K - R.V) @ R.K.T + R.LAM * (W0 - W0)
rng = np.random.default_rng(7)
Dm = rng.normal(size=(4, 4)); h = 1e-6
fd = (R.J(W0 + h * Dm, W0) - R.J(W0 - h * Dm, W0)) / (2 * h)
an = float(np.sum(G * Dm))
rel = abs(fd - an) / max(abs(an), 1e-300)
chk("A61-2 有限差分梯度锚", rel <= 1e-6, "rel=%.3e (h=%g)" % (rel, h))

# A61-3 逐臂对齐量
A = R.arms()
frozen = {"ZERO", "ORTHO"}
bad = []
for name, (U, Vm, pred) in A.items():
    ktv = float(np.abs(R.K.T @ Vm).max()); utv = float(np.abs(U.T @ R.V).max())
    if name in frozen:                        # 两侧**同时**对齐
        ok = ktv <= 1e-15 and utv <= 1e-15
    elif name == "VNULL":                     # 只对齐键侧
        ok = ktv <= 1e-15 and utv >= 1e-2
    elif name == "UNULL":                     # 只对齐值侧
        ok = ktv >= 1e-2 and utv <= 1e-15
    else:                                     # DEADBLK/RAND：两侧都不全对齐
        ok = ktv >= 1e-2 and utv >= 1e-2
    if not ok:
        bad.append(name)
chk("A61-3 逐臂对齐量逐位核验", not bad,
    "冻结支 ok；未对齐臂 ok" if not bad else "不符: %s" % bad)

# A61-4′ 敏感性负对照（rev 1.1，替代 rev 1.0 的 A61-4；见 DEF-61-2）
U0, V0 = A["RAND"][:2]
Wa = R.run(U0, V0, n=200)["W_N"]; Wb = R.run(U0, V0, n=200, sync=False)["W_N"]
gap = float(np.linalg.norm(Wa - Wb))
fz = {}
for nm in ("ZERO", "ORTHO"):
    Uz, Vz = A[nm][:2]
    fz[nm] = (R.run(Uz, Vz, n=200)["dW_max"], R.run(Uz, Vz, n=200, sync=False)["dW_max"])
chk("A61-4′ 敏感性负对照（RAND 同步≠异步；ZERO/ORTHO 两版皆冻结）",
    gap > 1e-6 and all(max(v) <= 1e-12 for v in fz.values()),
    "RAND |Wa-Wb|=%.3e; ZERO(sync,async)=(%.1e,%.1e); ORTHO=(%.1e,%.1e)"
    % (gap, fz["ZERO"][0], fz["ZERO"][1], fz["ORTHO"][0], fz["ORTHO"][1]))

# A61-5 算例复现
t_half = esc61.run_fac("d2r1", math.pi / 2)["tau"]
t_small = esc61.run_fac("d2r1", 1e-4)["tau"]
chk("A61-5 算例回归 τ(π/2)=7±1, τ(1e-4)=357±5%",
    t_half is not None and abs(t_half - 7) <= 1 and t_small is not None and abs(t_small - 357) <= 0.05 * 357,
    "τ(π/2)=%s τ(1e-4)=%s" % (t_half, t_small))

# A61-6 全矩阵 6 步锚
tf = esc61.run_full(0.0)
chk("A61-6 全矩阵对照 6±1 步", tf is not None and abs(tf - 6) <= 1, "τ_full=%s" % tf)

# A61-7 删失语义
rc = esc61.run_fac("d2r1", 1e-4, nmax=100)
chk("A61-7 删失语义（Nmax=100 ⇒ CENSORED）", bool(rc["censored"]) and rc["tau"] is None,
    "censored=%s tau=%s" % (rc["censored"], rc["tau"]))

# A61-8 确定性
r1 = [esc61.run_fac("d2r1", th)["tau"] for th in (1e-4, 1e-2, 0.3)]
r2 = [esc61.run_fac("d2r1", th)["tau"] for th in (1e-4, 1e-2, 0.3)]
chk("A61-8 确定性（两次逐位相同）", r1 == r2, "%s vs %s" % (r1, r2))

out = {"guards": RES, "n_pass": sum(1 for x in RES if x["ok"]), "n": len(RES)}
if len(sys.argv) > 2 and sys.argv[1] == "--out":
    os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
print("\n★ Stage A: %d/%d PASS" % (out["n_pass"], out["n"]))
