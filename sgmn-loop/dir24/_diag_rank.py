"""诊断 A24-2 / A24-4 的两条 FAIL：真秩亏 vs float32 容差。"""
import sys, math, torch
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")
from sgmn_module_hebb_v7 import HebbianCore
ALPHA, LAM, ETA, M, NP, D = 1.0, 0.05, 0.1, 2, 35, 896

def run(r0, seed=20261028, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    c = HebbianCore(D, D, ALPHA, LAM, ETA, g, "cpu", factorized=True, n_entries=1024, d_model_sig=8)
    c.U = c.U.to(dtype); c.V = c.V.to(dtype); c.W0 = c.W0.to(dtype)
    U = (torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype); V = (torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype)
    U[:, r0:] = 0; V[:, r0:] = 0
    c.U.copy_(U); c.V.copy_(V)
    K = torch.eye(D, dtype=dtype)
    va = torch.nn.functional.normalize(torch.randn(NP, D, generator=g), dim=-1).to(dtype)
    r = torch.randn(NP, D, generator=g).to(dtype)
    vb = torch.nn.functional.normalize(r - (r * va).sum(-1, keepdim=True) * va, dim=-1)
    for i in range(NP):
        for v in (va[i], vb[i]):
            for _ in range(M):
                c.write(K[i], v)
    W = c._W().detach()
    sv = torch.linalg.svdvals(W)
    rk = int(torch.linalg.matrix_rank(W).item())
    # 相对容差下的秩
    rk_rel = int((sv / sv[0] > 1e-6).sum().item())
    print(f"r0={r0:3d} dtype={str(dtype)[6:]:7s} rank_default={rk:3d} rank_rel1e-6={rk_rel:3d}"
          f" smax={float(sv[0]):.3e} smin={float(sv[-1]):.3e} 5 smallest={[f'{float(x):.2e}' for x in sv[-5:]]}")
    return W, sv

for r0 in (8, 70, 896):
    run(r0)
run(896, dtype=torch.float64)
