import sys
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir72"); sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir70")
import numpy as np, ref70, step72, spec72
mode, gamma, etaref = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
eta_f = 1e-3
fp = step72.fp_factors(mode, gamma, eta_warm=0.3 * etaref)
xs = spec72.pack(fp["U1"], fp["V1"], fp["U2"], fp["V2"])
J = spec72.jac_factor(xs, mode, gamma, eta_f)
lm = (np.linalg.eigvals(J) - 1.0) / eta_f
print("FP res=%.2e steps=%d" % (fp["res"], fp["steps"]))
print("dim=%d  zeros(|λJ-1|<1e-8)=%d  Re>0: %d  Re<0: %d" % (lm.size, int(np.sum(np.abs(np.linalg.eigvals(J)-1)<1e-8)), int(np.sum(np.real(lm)>0)), int(np.sum(np.real(lm)<0))))
pos = lm[np.real(lm) > 0]
if pos.size:
    print("Re>0 谱（按 Re 降序，前 6）:", [complex(v) for v in pos[np.argsort(-np.real(pos))][:6]])
neg = lm[np.real(lm) < 0]
b = 2.0 * (-np.real(neg)) / (np.abs(neg) ** 2)
idx = np.argsort(b)[:6]
print("最小的 6 个 2|a|/|λ|²  [bound, λ]:")
for i in idx:
    print("   %.9f   %s" % (float(b[i]), complex(neg[i])))
# 矩阵空间同样诊断
p = ref70.pred_target(6.0, gamma) if mode == "target" else ref70.pred_keys(6.0, gamma)
W1s, W2s = p["u1"] * ref70.P, p["u2"] * ref70.P
Jf = spec72.jac_field(W1s, W2s, mode, gamma, eta_f)
lmf = (np.linalg.eigvals(Jf) - 1.0) / eta_f
print("矩阵空间: dim=%d Re>0: %d Re<0: %d  min-bound=%.9f" % (lmf.size, int(np.sum(np.real(lmf)>0)), int(np.sum(np.real(lmf)<0)),
      2.0*np.min((-np.real(lmf))[np.real(lmf)<0]/(np.abs(lmf[np.real(lmf)<0])**2)) if np.any(np.real(lmf)<0) else float('inf')))
# 2x2 块直接读数（在 (aP,bP) 两个方向上作用）
def apply_block(J, idxa, idxb, h):
    pass
