"""DIR-81 测量层：`dir80/jac80.py` 逐字派生。
**唯一改动** = 新增 `warm_from_init(...)`：从**该格自己的 `init` 抽签**出发迭代 `step72.step`
至收敛，得到"这个幅度汇到哪里"的不动点。
理由：`dir78/warm78.warm_fp` 从 `near73.freeze_fp`（**解析构造**）出发，与 `init` **无关**
⇒ 若复用它算 rho_lambda，判据 ② 会变成同义反复（纪律 158）。见卡 §2 判据 ② 口径。
`flat/unflat/step48/jac/lmax/rho_scan/rho_nn/supjt` 全部沿用 jac80 的实现（import，不复制），
只有 `rho_scan` 是 jac80 的那一份（含 `(b-a)<=rel` 的 DEF-79-4 修正）。
"""
import sys
for q in ("dir69", "dir70", "dir72", "dir73", "dir78", "dir79", "dir80", "dir81"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np
import jac80, step72, ref70

flat, unflat, step48, jac, lmax = jac80.flat, jac80.unflat, jac80.step48, jac80.jac, jac80.lmax
rho_scan, rho_nn, supjt = jac80.rho_scan, jac80.rho_nn, jac80.supjt

D, R = int(getattr(ref70, "D", 6)), int(getattr(ref70, "R", 2))


def draw_init(init, seed=20261118):
    """**逐字**复刻 `ref70.run_stack` 的抽签顺序：U1,V1,U2,V2 各 (D,R)，再乘 init。"""
    rng = np.random.default_rng(seed)
    U1 = rng.normal(size=(D, R)) * init
    V1 = rng.normal(size=(D, R)) * init
    U2 = rng.normal(size=(D, R)) * init
    V2 = rng.normal(size=(D, R)) * init
    return U1, V1, U2, V2


def warm_from_init(mode, g, eta_warm, init, seed=20261118, tol=1e-15, nmax=2_000_000, c=6.0):
    """从该格自己的 init 出发，在 eta_warm 上迭代 step72.step 至
    ‖step(x)-x‖/‖x‖ <= tol。返回 (x*, info)。只迭代、不改装置。"""
    U1, V1, U2, V2 = (np.array(X, float, copy=True) for X in draw_init(init, seed))
    res, k = float("inf"), -1
    for k in range(nmax):
        n1, n2, n3, n4 = step72.step(U1, V1, U2, V2, eta_warm, mode, g, c=c)
        num = max(np.linalg.norm(n1 - U1), np.linalg.norm(n2 - V1),
                  np.linalg.norm(n3 - U2), np.linalg.norm(n4 - V2))
        den = max(np.linalg.norm(U1), np.linalg.norm(V1), np.linalg.norm(U2), np.linalg.norm(V2))
        U1, V1, U2, V2 = n1, n2, n3, n4
        res = num / den
        if res <= tol:
            break
    x = (U1, V1, U2, V2)
    v = flat(x)
    res_fp = float(np.linalg.norm(step48(v, eta_warm, mode, g, c) - v)) / float(np.linalg.norm(v))
    W1, W2 = U1 @ V1.T, U2 @ V2.T
    pk = ref70.pred_keys(c, g) if mode == "keys" else ref70.pred_target(c, g)
    return x, dict(res=res, res_fp=res_fp, steps=k + 1, converged=bool(res <= tol),
                   normU1=float(np.linalg.norm(U1)), normV1=float(np.linalg.norm(V1)),
                   normU2=float(np.linalg.norm(U2)), normV2=float(np.linalg.norm(V2)),
                   normW1=float(np.linalg.norm(W1)), normW2=float(np.linalg.norm(W2)),
                   u1_warm=float(W1[0, 0]), u2_warm=float(W2[0, 0]),
                   u1_pred=float(pk["u1"]), u2_pred=float(pk["u2"]),
                   normW1_init=float(np.linalg.norm(draw_init(init, seed)[0] @ draw_init(init, seed)[1].T)))