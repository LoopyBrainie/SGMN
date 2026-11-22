import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir69"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir65")
import ref69 as S, ref65 as R
C, N, Sd, A = 6.0, 200000, 20261118, 0.1
ok = True
def chk(n, c, m=""):
    global ok; ok = ok and c; print("%-34s %s | %s" % (n, "PASS" if c else "FAIL", m))
r0 = R.run(C, 0.5, 20000, seed=Sd, init=A)          # ref65 原装置
s0 = S.run_stack(C, 0.0, 0.5, 20000, seed=Sd, init=A)  # 耦合装置 γ=0
dw = float(np.max(np.abs(s0["W1"] - r0["W"])))
# 锚检用括号法（阈值 |η*-0.88751|<=0.005 不变；[0.884,0.891] 宽度 0.79% 即可保证）
lo_ok = not S.run_stack(C, 0.0, 0.884, N, seed=Sd, init=A)["div"]
hi_div = S.run_stack(C, 0.0, 0.891, N, seed=Sd, init=A)["div"]
chk("A69-1 γ=0 逐位同轨迹 + 锚括号", dw == 0.0 and lo_ok and hi_div,
    "max|ΔW1|=%.1e 括号[0.884 有界=%s, 0.891 发散=%s] 锚 0.88751" % (dw, lo_ok, hi_div))
bad = []
for g in (0, 1, 3, 10, 20):
    o = S.run_stack(C, g, 0.01, N, seed=Sd, init=A); p = S.pred(g)
    if abs(o["u1"] - p["u1"]) / p["u1"] > 0.01: bad.append((g, round(o["u1"], 4), round(p["u1"], 4)))
chk("A69-2 不动点代数核对（5 个 γ）", not bad, "全部 ≤1%" if not bad else str(bad))
u10 = S.run_stack(C, 10.0, 0.01, N, seed=Sd, init=A)["u1"]
u0 = S.run_stack(C, 0.0, 0.01, N, seed=Sd, init=A)["u1"]
chk("A69-3 耦合项生效(u1*(10)/u1*(0)>5)", u10 / u0 > 5, "比值=%.2f (%.3f/%.3f)" % (u10 / u0, u10, u0))
chk("A69-4 二分可执行(γ=10:0.01 有界/3.0 发散)",
    (not S.run_stack(C, 10.0, 0.01, N, seed=Sd, init=A)["div"]) and S.run_stack(C, 10.0, 3.0, N, seed=Sd, init=A)["div"])
print("\n★ Stage A: %d/4 PASS" % (4 if ok else -1))
