import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import ref70 as S
C, N, Sd, A = 6.0, 200000, 20261118, 0.1
ok = True
def chk(n, c, m=""):
    global ok; ok = ok and bool(c); print("%-46s %s | %s" % (n, "PASS" if c else "FAIL", m), flush=True)
bad = []; rows = []
for g, e in ((1.0, 1e-2), (3.0, 1e-2)):
    o = S.run_stack(C, g, e, N, seed=Sd, init=A, mode="target"); p = S.pred_target(C, g)
    fin = np.isfinite(o["u1"]) and not o["div"]; rel = abs(o["u1"] - p["u1"]) / p["u1"] if fin else float("inf")
    rows.append(("target", g, e, o["u1"], p["u1"], rel, fin))
    if (not fin) or rel > 0.01: bad.append(("target", g))
for g, e in ((1.0, 0.05), (3.0, 0.05), (10.0, 1e-3)):
    o = S.run_stack(C, g, e, N, seed=Sd, init=A, mode="keys"); p = S.pred_keys(C, g)
    fin = np.isfinite(o["u1"]) and not o["div"]; rel = abs(o["u1"] - p["u1"]) / p["u1"] if fin else float("inf")
    rows.append(("keys", g, e, o["u1"], p["u1"], rel, fin))
    if (not fin) or rel > 0.01: bad.append(("keys", g))
for m, g, e, u, p, rel, fin in rows:
    print("   %-6s γ=%-3g η=%-7g u1=%-10.5f pred=%-10.5f rel=%.2e finite=%s" % (m, g, e, u, p, rel, fin), flush=True)
chk("A70-2′ 不动点核对(逐 γ 选 η)", not bad, "全部 ≤1%%" if not bad else "失败 %s" % bad)
print("\n★ StageA-70b(rev1.1): %d/1" % (1 if ok else -1), flush=True)
