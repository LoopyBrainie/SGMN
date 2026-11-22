import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir65")
import ref70 as S, ref65 as R
C, N, Sd, A = 6.0, 200000, 20261118, 0.1
ok = True
def chk(n, c, m=""):
    global ok; ok = ok and bool(c); print("%-46s %s | %s" % (n, "PASS" if c else "FAIL", m), flush=True)
r0 = R.run(C, 0.5, 20000, seed=Sd, init=A)
for mode in ("target", "keys"):
    s0 = S.run_stack(C, 0.0, 0.5, 20000, seed=Sd, init=A, mode=mode)
    dw = float(np.max(np.abs(s0["W1"] - r0["W"])))
    chk("A70-1 γ=0 逐位复现 ref65 [%s]" % mode, dw == 0.0, "max|ΔW1|=%.1e" % dw)
lo = not S.run_stack(C, 0.0, 0.884, N, seed=Sd, init=A, mode="keys")["div"]
hi = S.run_stack(C, 0.0, 0.891, N, seed=Sd, init=A, mode="keys")["div"]
chk("A70-1b 锚括号 [0.884,0.891]", lo and hi, "有界=%s 发散=%s" % (lo, hi))
a = S.run_stack(C, 10.0, 1e-3, N, seed=Sd, init=A, mode="keys")
b = S.run_stack(C, 10.0, 3.0, N, seed=Sd, init=A, mode="keys")
chk("A70-4 网格可执行(keys γ=10 两端)", (not a["div"]) and b["div"],
    "η=1e-3 有界=%s u1=%.4f / η=3 发散=%s" % (not a["div"], a["u1"], b["div"]))
print("\n★ StageA-70a: %d/3" % (3 if ok else -1), flush=True)
