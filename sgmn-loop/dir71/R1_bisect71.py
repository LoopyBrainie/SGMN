import sys, os, json
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir71")
import R2_bisect71 as B
# 起点括号全部取自 DIR-70 已落盘网格（卡片 §6）
BR = {("target", 3): (0.03, 0.1), ("target", 10): (3e-3, 1e-2), ("target", 30): (3e-4, 1e-3),
      ("keys", 3): (0.1, 0.3), ("keys", 10): (1e-2, 3e-2)}
OUT = "/home/ec2-user/sgmn-loop/dir71/out/grid"
def main(mode, gamma):
    os.makedirs(OUT, exist_ok=True)
    lo, hi = BR[(mode, int(gamma))]
    r = B.bisect(mode, gamma, lo=lo, hi=hi)
    fn = os.path.join(OUT, "bisect_%s_g%g.json" % (mode, gamma))
    json.dump(r, open(fn, "w"), indent=1)
    print("DONE %s γ=%g η*=%.6g [%.6g,%.6g] ok=%s n_eval=%d" % (mode, gamma, r["eta"], r["lo"], r["hi"], r["ok"], r["n_eval"]), flush=True)
if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
