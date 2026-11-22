import sys, numpy as np
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir68")
import ref68 as R
A, N, C, S = 0.1, 200000, 6.0, 20261118
ok = True
def chk(n, c, m=""):
    global ok; ok = ok and c; print("%-42s %s | %s" % (n, "PASS" if c else "FAIL", m))
uP, vP = R.init_dir("PAR", A, S); uM, vM = R.init_dir("MIX", A, S)
r1 = abs(np.linalg.norm(uP) - np.linalg.norm(uM)) / np.linalg.norm(uM)
r2 = abs(np.linalg.norm(vP) - np.linalg.norm(vM)) / np.linalg.norm(vM)
chk("A68-1 配平正确（范数同 + rows3-6=0）", max(r1, r2) <= 1e-12 and np.all(uP[2:] == 0) and np.all(vP[2:] == 0),
    "relU=%.1e relV=%.1e ‖U0‖=%.6f" % (r1, r2, np.linalg.norm(uP)))
o = R.run_dir("ORTH", C, 2.0, A, N, S)
chk("A68-2 ORTH 有界可测（max|W|≤1e-12, rev1.1）", (o["maxW"] <= 1e-12) and (not o["div"]), "max|W|=%.1e div=%s" % (o["maxW"], o["div"]))
b = R.bisect_dir("MIX", C, A, N, seed=S)
chk("A68-3 装置锚 MIX a=0.1 seed181118", abs(b["eta"] - 0.88751) <= 0.005, "η*=%.5f (DIR-67: 0.88751)" % b["eta"])
chk("A68-4 二分可执行（PAR 0.3 有界/2.0 发散）", (not R.run_dir("PAR", C, 0.3, A, N, S)["div"]) and R.run_dir("PAR", C, 2.0, A, N, S)["div"])
print("\n★ Stage A: %d/4 PASS" % (4 if ok else -1))
