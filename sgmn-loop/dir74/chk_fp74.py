import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir73"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, near73, ref70
for mode, g in (("keys", 10.0), ("target", 30.0)):
    xs, fp = near73.freeze_fp(mode, g, 0.3 * near73.eta_lin(mode, g))
    W1 = xs[0] @ xs[1].T; W2 = xs[2] @ xs[3].T
    p = ref70.pred_target(6.0, g) if mode == "target" else ref70.pred_keys(6.0, g)
    print(mode, g, "u1*=%.6f u2*=%.6f" % (p["u1"], p["u2"]))
    print("  W1 diag:", np.round(np.diag(W1), 6), " W2 diag:", np.round(np.diag(W2), 6))
    print("  max|W1-W1^T|=%.1e max|W2-W2^T|=%.1e" % (abs(W1 - W1.T).max(), abs(W2 - W2.T).max()))
    print("  offdiag max=%.1e" % max(abs(W1 - np.diag(np.diag(W1))).max(), abs(W2 - np.diag(np.diag(W2))).max()))
    print("  fp_res=%.1e" % fp.get("res", float("nan")))
