"""DIR-84 共享测量层：**绝对**口径吸引域半径的二分测量。

与 DIR-83 的**唯一**差别：扰动写成 `v0 + d_abs * delta_hat`（绝对），
而非 `v0 + eps*||v0|| * delta_hat`（相对 `||v0||`）。
装置一字未改：只读沿用 jac79.step48 / jac81.flat|unflat / jac82.gfix_qr / dev83 的冻结常量
与固定点缓存（`dir83/out/fp_target_g*.npz`）。

`DEF-83-5` 的教训：阈值不得定义在一个随自变量缩放的量上。
这里 d_abs 是 flat 向量（4*D*R = 48 维）上的欧氏范数，与 γ 无关。
"""
import sys, os
for q in ("dir70", "dir72", "dir73", "dir78", "dir79", "dir80", "dir81", "dir82", "dir83", "dir84"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np
import jac79, jac81
import dev83

OUT84 = "/home/ec2-user/sgmn-loop/dir84/out"
os.makedirs(OUT84, exist_ok=True)

N_STEPS = 200000          # 与 DIR-83 同口径（可比性），卡 §3 冻结
DIV_ABS = 1e3             # **只作诊断**，不是逃逸判据
SEEDS = (7777, 7778, 7779)  # 三个固定方向 δ̂，全 γ 共用
GAM = dev83.GAM
C = 6.0


def v0_of(g):
    """规范代表下的固定点 flat 向量（与 DIR-83 完全同源）。"""
    return dev83.flat(dev83.load_fp(g))


def nv0_of(g):
    return float(np.linalg.norm(v0_of(g)))


def w1star_of(g):
    """||W_1^*||_F，用于一致性核对（应等于 dev83.FROZEN 的 W）。"""
    w1, _ = dev83.wnorm(dev83.load_fp(g))
    return w1


def traj_abs(g, eta, d_abs, N=N_STEPS, seed=SEEDS[0]):
    """从 `x* + d_abs*delta_hat` 出发，最多 N 步。

    逃逸判据 = **仅非有限**（卡 §3 冻结；不用 1e3 阈值 —— DIR-83 的 t_div
    与 t_nan 只差 1–3 步且 mx_rel 达 1e147，说明它们不是"变大"而是溢出）。
    """
    v0 = v0_of(g)
    d = dev83.delta_hat(v0.size, seed)
    v = v0 + d_abs * d
    w1r, _ = dev83.wnorm(dev83.unflat(v0))
    mx, t_nan, t_div = 0.0, None, None
    for t in range(1, N + 1):
        vn = jac79.step48(v, eta, "target", g, c=C)
        if not np.isfinite(vn).all():
            t_nan = t
            break
        v = vn
        w1, w2 = dev83.wnorm(dev83.unflat(v))
        if w1 > mx:
            mx = w1
        if t_div is None and (w1 > DIV_ABS or w2 > DIV_ABS):
            t_div = t
    # 末态到 W_1^* 的相对距离（诊断：区分"回到不动点"与"仍在漂移"）
    if t_nan is None:
        w1f, _ = dev83.wnorm(dev83.unflat(v))
        d_final = abs(w1f - w1r) / w1r
    else:
        w1f, d_final = float("inf"), float("inf")
    return dict(d_abs=float(d_abs), eta=float(eta), seed=int(seed), N=N,
                escaped=bool(t_nan is not None), t_nan=t_nan, t_div=t_div,
                mx_rel=float(max(mx, w1r) / w1r), w_ref=w1r,
                w_final=float(w1f), d_final=float(d_final),
                steps=int(t_nan or N))


def escaped(t_nan):
    """逃逸判据：**只认非有限**（卡 §3 冻结）。1e3 阈值只作诊断，不得进此函数。"""
    return t_nan is not None


def bisect_pred(pred, lo, hi, rel):
    """在 [lo,hi] 上对单调谓词二分。返回最后一个 False 点与首个 True 点。

    A4/A5 用它测**真正的二分代码路径**（不是副本）。
    端点不满足时如实记 bound_hit，不做外推、不补点。
    """
    tr = []
    e_lo = pred(lo)
    tr.append((lo, e_lo))
    if e_lo:
        return dict(R=None, bracket=None, bound_hit="lo", evals=1, trace=tr)
    e_hi = pred(hi)
    tr.append((hi, e_hi))
    if not e_hi:
        return dict(R=None, bracket=None, bound_hit="hi", evals=2, trace=tr)
    n = 3
    while (hi - lo) / max(1e-12, 0.5 * (hi + lo)) > rel:
        mid = 0.5 * (lo + hi)
        if pred(mid):
            hi = mid
        else:
            lo = mid
        n += 1
    return dict(R=0.5 * (lo + hi), lo=lo, hi=hi, bracket=[lo, hi],
                bound_hit=None, evals=n, trace=tr)


def bisect_radius(g, eta, lo=0.02, hi=2.0, rel=0.03, seed=SEEDS[0], N=N_STEPS):
    """绝对口径吸引域半径的二分。谓词 = 该 d_abs 是否在 N 步内出现非有限值。"""
    tr = []

    def pred(d):
        r = traj_abs(g, eta, d, N=N, seed=seed)
        tr.append((d, r["escaped"], r["t_nan"], r["d_final"], r["mx_rel"], r["steps"]))
        return r["escaped"]

    out = bisect_pred(pred, lo, hi, rel)
    out["trace"] = tr                     # 富记录（含 t_nan / d_final / mx_rel）
    out["mono_ok"] = mono_ok([(d, e) for d, e, *_ in tr])
    out.update(g=g, eta=eta, seed=seed, N=N, lo_win=lo, hi_win=hi, rel=rel)
    return out


def mono_ok(trace):
    """二分路径上的点必须构成上区间：d1<d2 且 d2 未逃逸 ⇒ d1 也未逃逸。

    trace 元素可以是 `(d, escaped)` 或 `(d, escaped, *rest)`。
    """
    s = sorted((t[0], t[1]) for t in trace)
    seen_escape = False
    for d, e in s:
        if e:
            seen_escape = True
        elif seen_escape:
            return False
    return True