"""DIR-84 Stage A 守卫（硬门禁）。A1–A9 全过才放行正式轮。

A4/A5 测的是 `dev84.bisect_pred` —— 正式轮实际调用的那份代码，不是副本。
A6 用真实锚点的 `mx_rel` 分布检验「逃逸判据不是阈值驱动」。
A9 在真装置上跑一次完整二分（其读数**同时**作为正式轮 (γ=10, seed 7777) 的一格，不浪费）。
"""
import sys, os, json, time
for q in ("dir70", "dir72", "dir73", "dir78", "dir79", "dir80", "dir81", "dir82", "dir83", "dir84"):
    sys.path.insert(0, "/home/ec2-user/sgmn-loop/" + q)
import numpy as np
import dev83, dev84

R = {}
def rec(k, ok, **kw):
    R[k] = dict(ok=bool(ok), **kw)
    print("  %-3s %s %s" % (k, "PASS" if ok else "FAIL", kw))
    return ok

t0 = time.time()
ok_all = True

# ---------- A1: 固定点是固定点（N=2000，只需证明不移动） ----------
a1 = []
for g in dev84.GAM:
    r = dev84.traj_abs(g, dev83.eta_mid(g), 0.0, N=2000, seed=7777)
    a1.append(dict(g=g, escaped=r["escaped"], d_final=r["d_final"]))
ok = all((not r["escaped"]) and r["d_final"] <= 1e-12 for r in a1)
ok_all &= rec("A1", ok, d_final=[r["d_final"] for r in a1], N=2000)

# ---------- A2: ||W1*|| 与 formal83 的 W_ref 一致（tol 1e-6，DEF-84-1） ----------
f83 = json.load(open("/home/ec2-user/sgmn-loop/dir83/out/formal83.json"))
a2 = []
for g in dev84.GAM:
    w = dev84.w1star_of(g)
    wref = f83["rows"][str(g)]["W_ref"]
    a2.append(dict(g=g, w=w, wref=wref, rel=abs(w - wref) / wref))
ok = all(r["rel"] <= 1e-6 for r in a2)
ok_all &= rec("A2", ok, rel=[r["rel"] for r in a2])

# ---------- A3 + A6: 8 锚点（绝对=相对）与 mx_rel 分离 ----------
anc = json.load(open("/home/ec2-user/sgmn-loop/dir84/out/anchors84.json"))
an = [c for c in anc["checks"] if c["what"] == "absolute==relative"]
ok = len(an) == 8 and all(c["same_finite"] and c["same_t_nan"] for c in an)
ok_all &= rec("A3", ok, n=len(an),
              t_nan=[c["t_nan"] for c in an], escaped=[c["escaped"] for c in an])

fin = [c["mx_rel"] for c in an if not c["escaped"]]
div = [c["mx_rel"] for c in an if c["escaped"]]
mx_fin, mn_div = (max(fin) if fin else None), (min(div) if div else None)
ok = (mx_fin is not None and mn_div is not None and mx_fin <= 1.10 and mn_div >= 1e30
      and dev84.escaped(None) is False and dev84.escaped(5) is True)
ok_all &= rec("A6", ok, max_mx_rel_unstuck=mx_fin, min_mx_rel_escaped=mn_div)

# ---------- A4: 二分器自检（合成单调谓词） ----------
b = dev84.bisect_pred(lambda d: d > 0.5, 0.02, 2.0, 0.03)
ok = b["bound_hit"] is None and abs(b["R"] - 0.5) / 0.5 <= 0.03
ok_all &= rec("A4", ok, R=b["R"], bracket=b["bracket"], evals=b["evals"], bound_hit=b["bound_hit"])

# ---------- A5: bound_hit 检测 ----------
b_hi = dev84.bisect_pred(lambda d: False, 0.02, 2.0, 0.03)
b_lo = dev84.bisect_pred(lambda d: True, 0.02, 2.0, 0.03)
ok = b_hi["bound_hit"] == "hi" and b_lo["bound_hit"] == "lo" and b_hi["R"] is None and b_lo["R"] is None
ok_all &= rec("A5", ok, hi=b_hi["bound_hit"], lo=b_lo["bound_hit"])

# ---------- A7: mono_ok 自检 ----------
ok = (dev84.mono_ok([(0.1, False), (0.3, True), (0.5, False)]) is False
      and dev84.mono_ok([(0.1, False), (0.3, False), (0.5, True)]) is True)
ok_all &= rec("A7", ok)

# ---------- A8: 三方向是单位向量且两两近正交 ----------
d = [dev83.delta_hat(48, s) for s in dev84.SEEDS]
nrm = [float(np.linalg.norm(x)) for x in d]
cos = [float(abs(d[i] @ d[j])) for i in range(3) for j in range(i + 1, 3)]
ok = all(abs(x - 1.0) <= 1e-12 for x in nrm) and all(c <= 0.3 for c in cos)
ok_all &= rec("A8", ok, norms=nrm, abs_cos=cos)

# ---------- A9: 真装置抽检（γ=10, seed 7777） ----------
bs = dev84.bisect_radius(10.0, dev83.eta_mid(10.0), seed=7777)
lo_s, hi_s = bs["bracket"] if bs["bracket"] else (None, None)
# rev 1.1 (DEF-84-3): 只用**单侧**条件 —— 已知未逃逸点须仍在未逃逸侧、
# 已知逃逸点须仍在逃逸侧。不得对边界**位置**下断言（rev 1.0 的错误）。
ok = (bs["bound_hit"] is None and bs["mono_ok"]
      and lo_s is not None and lo_s >= 0.450317
      and hi_s <= 1.501058)
ok_all &= rec("A9", ok, R=bs["R"], bracket=bs["bracket"], evals=bs["evals"],
              mono_ok=bs["mono_ok"], bound_hit=bs["bound_hit"])

R["A9_cell"] = {k: v for k, v in bs.items() if k != "trace"}
R["allok"] = bool(ok_all)
R["elapsed"] = time.time() - t0
json.dump(R, open(os.path.join(dev84.OUT84, "stageA84.json"), "w"), indent=1)
print("\nStage A allok =", ok_all, "| elapsed %.1f s" % R["elapsed"])
print("A9 单元（并入正式轮）: R=%.6f bracket=%s" % (bs["R"], bs["bracket"]) if bs["R"] else "A9 bound_hit")