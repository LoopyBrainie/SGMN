import sys, math, torch
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")
from sgmn_module_hebb_v7 import HebbianCore
ALPHA, LAM, ETA, M, NP, D = 1.0, 0.05, 0.1, 2, 35, 896
g = torch.Generator().manual_seed(20261028)
c = HebbianCore(D, D, ALPHA, LAM, ETA, g, "cpu", factorized=True, n_entries=1024, d_model_sig=8)
U = torch.randn(D, D, generator=g) / math.sqrt(D); V = torch.randn(D, D, generator=g) / math.sqrt(D)
U[:, 8:] = 0; V[:, 8:] = 0
c.U.copy_(U); c.V.copy_(V)
K = torch.eye(D)
va = torch.nn.functional.normalize(torch.randn(NP, D, generator=g), dim=-1)
r = torch.randn(NP, D, generator=g)
vb = torch.nn.functional.normalize(r - (r * va).sum(-1, keepdim=True) * va, dim=-1)
for i in range(NP):
    for v in (va[i], vb[i]):
        for _ in range(M):
            c.write(K[i], v)
W = c._W().detach(); t = va[0]
W64 = W.double(); t64 = t.double()
P64 = W64 @ torch.linalg.pinv(W64, rtol=1e-6)
for drv in ("gels", "gelsy", "gelsd"):
    try:
        q = torch.linalg.lstsq(W64, t64.unsqueeze(1), driver=drv).solution[:, 0]
        print(f"drv={drv:6s} ||q||={float(q.norm()):.3e} ||Wq||={float((W64@q).norm()):.3e}"
              f" resid_vs_P={float((W64@q - P64@t64).norm()):.3e}")
    except Exception as e:
        print(drv, "ERR", e)
q32 = torch.linalg.lstsq(W, t.unsqueeze(1)).solution[:, 0]
print(f"float32 lstsq: ||Wq||={float((W@q32).norm()):.3e} resid_vs_P64={float((W64@q32.double() - P64@t64).norm()):.3e}")
print(f"sigma_min/max kept: {float(torch.linalg.svdvals(W64)[-1]):.2e}")
