import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir64")
import ref64 as R
print("格式          init=1e-3    init=1e-2    init=0.1     init=0.3     预测")
pred = {"fac_batch": 1.0, "fac_cycle": 0.1814}
for fmt in ("fac_batch", "fac_cycle"):
    row = [R.bisect_init(fmt, 6.0, ini) for ini in (1e-3, 1e-2, 0.1, 0.3)]
    print("%-13s %s  %s" % (fmt, "  ".join("%-11.5f" % r for r in row), pred[fmt]))
