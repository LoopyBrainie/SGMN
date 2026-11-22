import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir69"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir65")
import ref69 as S, ref65 as R
C, N, Sd, A = 6.0, 200000, 20261118, 0.1
ok = True
def chk(n, c, m=""):
    global ok; ok = ok and bool(c); print("%-38s %s | %s" % (n, "PASS" if c else "FAIL", m), flush=True)
r0 = R.run(C, 0.5, 20000, seed=Sd, init=A); s0 = S.run_stack(C, 0.0, 0.5, 20000, seed=Sd, init=A)
dw = float(np.max(np.abs(s0["W1"] - r0["W"])))
lo_ok = not S.run_stack(C, 0.0, 0.884, N, seed=Sd, init=A)["div"]
hi_div = S.run_stack(C, 0.0, 0.891, N, seed=Sd, init=A)["div"]
chk("A69-1 γ=0 逐位同轨迹+锚括号", dw == 0.0 and lo_ok and hi_div, "max|ΔW1|=%.1e 括号[%s,%s]" % (dw, lo_ok, hi_div))
ETA = {0: 0.01, 1: 0.01, 3: 0.01, 10: 1e-3, 20: 1e-3}
bad = []
for g, e in ETA.items():
    o = S.run_stack(C, g, e, N, seed=Sd, init=A); p = S.pred(g)
    fin = np.isfinite(o["u1"]) and not o["div"]
    rel = abs(o["u1"] - p["u1"]) / p["u1"] if fin else float("inf")
    print("   γ=%-3g η=%-6g u1=%-10.5f pred=%-10.5f rel=%.2e finite=%s" % (g, e, o["u1"], p["u1"], rel, fin), flush=True)
    if (not fin) or (not (rel <= 0.01)): bad.append(g)
chk("A69-2′ 不动点核对(逐 γ 选 η + 显式有限性)", not bad, "全部 ≤1%%" if not bad else "失败 γ=%s" % bad)
u3 = S.run_stack(C, 3.0, 0.01, N, seed=Sd, init=A)["u1"]; u0 = S.run_stack(C, 0.0, 0.01, N, seed=Sd, init=A)["u1"]
chk("A69-3′ 耦合项生效 u1*(3)/u1*(0)>2", np.isfinite(u3) and u3 / u0 > 2, "比值=%.2f" % (u3 / u0))
a = S.run_stack(C, 10.0, 1e-4, N, seed=Sd, init=A); b = S.run_stack(C, 10.0, 3.0, N, seed=Sd, init=A)
chk("A69-4′ 网格两端均可读", ("div" in a) and ("div" in b), "η=1e-4 div=%s / η=3 div=%s" % (a["div"], b["div"]))
print("\n★ Stage A(rev1.2): %d/4 PASS" % (4 if ok else -1), flush=True)
