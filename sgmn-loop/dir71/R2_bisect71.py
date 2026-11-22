r"""DIR-71 量测层：**对数二分**，逐字派生 `dir69/ref69.py::bisect`。
与 ref69 的唯一差别：(a) 加 `mode` 参数（`ref70.run_stack` 已支持）(b) 返回最终括号 [lo,hi]、`n_eval`、完整 `hist`
(c) 精度条件改成**区间比** `exp(b-a) <= 1+rel`（ref69 用的是"相对对数宽度"，语义略有不同 —— 见 DEF-71-1）。
装置 `dir70/ref70.py` **原样未改**。
"""
import math
import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import ref70

C, N, SD, A = 6.0, 200000, 20261118, 0.1

def bisect(mode, gamma, c=6.0, n=200000, rel=0.02, lo=None, hi=None, seed=SD, init=A, verbose=True):
    a, b = math.log(lo), math.log(hi)
    fa = ref70.run_stack(c, gamma, lo, n, seed=seed, init=init, mode=mode)["div"]
    fb = ref70.run_stack(c, gamma, hi, n, seed=seed, init=init, mode=mode)["div"]
    hist = [(float(lo), bool(fa)), (float(hi), bool(fb))]; ne = 2
    if verbose: print("  [%s γ=%g] 端检 lo=%.6g div=%s / hi=%.6g div=%s" % (mode, gamma, lo, fa, hi, fb), flush=True)
    if fa or not fb:
        return dict(mode=mode, gamma=gamma, eta=float("nan"), lo=float("nan"), hi=float("nan"),
                    ok=False, note="两端同号", n_eval=ne, hist=hist)
    for _ in range(80):
        if (b - a) <= math.log(1.0 + rel): break
        m = 0.5 * (a + b)
        d = ref70.run_stack(c, gamma, math.exp(m), n, seed=seed, init=init, mode=mode)["div"]
        ne += 1; hist.append((float(math.exp(m)), bool(d)))
        if verbose: print("   试 η=%.6g div=%s" % (math.exp(m), d), flush=True)
        if d: b = m
        else: a = m
    return dict(mode=mode, gamma=gamma, eta=float(math.exp(0.5 * (a + b))),
                lo=float(math.exp(a)), hi=float(math.exp(b)), ok=True, note="", n_eval=ne, hist=hist)

def grid(mode, gamma, pts, c=6.0, n=200000, seed=SD, init=A):
    cells = {}
    for eta in pts:
        o = ref70.run_stack(c, gamma, eta, n, seed=seed, init=init, mode=mode)
        cells["%.0e" % eta] = dict(eta=float(eta), div=bool(o["div"]), steps=int(o["steps"]),
                                  u1=(float(o["u1"]) if abs(o["u1"]) < 1e30 else None))
        print("  [%s γ=%g] η=%.0e div=%s u1=%s" % (mode, gamma, eta, o["div"], cells["%.0e" % eta]["u1"]), flush=True)
    return cells
