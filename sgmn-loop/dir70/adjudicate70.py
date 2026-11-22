"""DIR-70 裁决器。读法约定（**写于 keys 数据存在之前**，不得事后更改）：
 · η*_top 用「网格中最大的有界 η 与紧随其后的最小发散 η」构成的括号；**拟合取几何中点**。
 · 判据 ② 用**保守读法**：keys 模式在 η=0.05 处**有界**即算 η*_top ≥ 0.05（不需要中点外推）。
 · 指数 = 3 点 (log γ, log η*_top_mid) 的最小二乘斜率。
"""
import json, math, sys, os
G = [1.0, 3.0, 10.0]; GRID = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 1.0, 3.0]
ROOT = "/home/ec2-user/sgmn-loop/dir70/out/grid"
def load(mode, g):
    p = os.path.join(ROOT, "%s_g%g.json" % (mode, g))
    if not os.path.exists(p): return None
    return json.load(open(p))["cells"]
def bracket(cells):
    """返回 (最大的有界 η, 其后的最小发散 η)；若首点即发散返回 (0, grid[0])。"""
    last_ok = None
    for e in GRID:
        k = "%.0e" % e
        if k not in cells: return None
        if cells[k]["div"]:
            return (last_ok, e)
        last_ok = e
    return (last_ok, None)   # 全网格有界 ⇒ 上缘 > max(grid)
def adm(mode):
    out = {}
    for g in G:
        c = load(mode, g)
        if c is None: return None
        b = bracket(c)
        if b is None: return None
        out[g] = dict(lo=b[0], hi=b[1], cells=c)
    return out
def slope(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
def analyze(mode):
    a = adm(mode)
    if a is None: return None
    xs, ys, rows = [], [], []
    for g in G:
        lo, hi = a[g]["lo"], a[g]["hi"]
        if lo is None: mid = hi / math.sqrt(3.0); note = "首点即发散"
        elif hi is None: mid = lo * math.sqrt(3.0); note = "上缘>max(grid)"
        else: mid = math.sqrt(lo*hi); note = ""
        xs.append(math.log(g)); ys.append(math.log(mid))
        rows.append((g, lo, hi, mid, note))
    return dict(slope=slope(xs, ys), rows=rows, adm=a)
res = {}
for m in ("target", "keys"):
    r = analyze(m)
    if r is None: print("✗ %s 数据不齐" % m); sys.exit(2)
    res[m] = r
print("=" * 78)
for m in ("target", "keys"):
    print("[%s] 拟合指数 = %.2f" % (m, res[m]["slope"]))
    for g, lo, hi, mid, note in res[m]["rows"]:
        print("   γ=%-3g 有界≤%-9s 发散≥%-9s mid=%.4g %s" % (g, lo, hi, mid, note))
d = abs(res["target"]["slope"] - res["keys"]["slope"])
c1 = d >= 0.7
k10 = res["keys"]["adm"][10.0]["cells"]
k05 = k10.get("5e-02", {}).get("div")
c2 = (k05 is False)
print("-" * 78)
print("判据① 指数差 = |%.2f - %.2f| = %.2f  (阈值 ≥0.7)  ⇒ %s" % (res["target"]["slope"], res["keys"]["slope"], d, "PASS" if c1 else "FAIL"))
print("判据② keys η*_top(10) ≥ 0.05（保守读法：η=0.05 有界）      ⇒ %s  (η=0.05 div=%s)" % ("PASS" if c2 else "FAIL", k05))
for m in ("target", "keys"):
    us = []
    for g, lo, hi, mid, note in res[m]["rows"]:
        c = res[m]["adm"][g]["cells"]
        kk = "%.0e" % lo if lo else None
        u = c[kk]["u1"] if (kk in c) else None
        us.append("%s@γ=%g" % (("-" if u is None else ("%.4f" % u)), g))
    print("判据③（报告）%s 不动点 u1(最大有界 η) = %s" % (m, " · ".join(us)))
verdict = "SUCCEEDED" if (c1 and c2) else ("PARTIAL" if (c1 or c2) else "FALSIFIED")
print("★ 裁决 = %s" % verdict)
json.dump(dict(slope={m: res[m]["slope"] for m in res}, diff=d, c1=c1, c2=c2, verdict=verdict,
               rows={m: [[g, lo, hi, mid] for g, lo, hi, mid, _ in res[m]["rows"]] for m in res}),
          open("/home/ec2-user/sgmn-loop/dir70/out/verdict70.json", "w"), indent=1)
