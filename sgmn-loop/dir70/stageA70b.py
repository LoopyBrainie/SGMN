import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import ref70 as S
C, N, Sd, A = 6.0, 200000, 20261118, 0.1
ok = True
def chk(n, c, m=""):
    global ok; ok = ok and bool(c); print("%-46s %s | %s" % (n, "PASS" if c else "FAIL", m), flush=True)
bad = []
for g in (1.0, 3.0):
    o = S.run_stack(C, g, 1e-2, N, seed=Sd, init=A, mode="target"); p = S.pred_target(C, g)
    fin = np.isfinite(o["u1"]) and not o["div"]; rel = abs(o["u1"] - p["u1"]) / p["u1"] if fin else float("inf")
    print("   target γ=%-3g u1=%-9.5f pred=%-9.5f rel=%.2e finite=%s" % (g, o["u1"], p["u1"], rel, fin), flush=True)
    if (not fin) or rel > 0.01: bad.append(("target", g))
for g in (1.0, 3.0, 10.0):
    o = S.run_stack(C, g, 0.05, N, seed=Sd, init=A, mode="keys"); p = S.pred_keys(C, g)
    fin = np.isfinite(o["u1"]) and not o["div"]; rel = abs(o["u1"] - p["u1"]) / p["u1"] if fin else float("inf")
    print("   keys   γ=%-3g u1=%-9.5f pred=%-9.5f rel=%.2e finite=%s" % (g, o["u1"], p["u1"], rel, fin), flush=True)
    if (not fin) or rel > 0.01: bad.append(("keys", g))
chk("A70-2 不动点核对(target×2 + keys×3)", not bad, "全部 ≤1%%" if not bad else "失败 %s" % bad)
ot = S.run_stack(C, 3.0, 1e-2, N, seed=Sd, init=A, mode="target")
ok_ = S.run_stack(C, 3.0, 1e-2, N, seed=Sd, init=A, mode="keys")
relm = float(np.linalg.norm(ok_["W1"] - ot["W1"]) / np.linalg.norm(ot["W1"])) if np.isfinite(ot["W1"]).all() and np.isfinite(ok_["W1"]).all() else float("inf")
chk("A70-3 自变量生效(两模式 W1 相对差>0.1)", relm > 0.1, "相对差=%.3f (target u1=%.4f / keys u1=%.4f)" % (relm, ot["u1"], ok_["u1"]))
print("\n★ StageA-70b: %d/2" % (2 if ok else -1), flush=True)
