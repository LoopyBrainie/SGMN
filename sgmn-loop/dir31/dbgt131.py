#!/opt/pytorch/bin/python
"""定位 A31-3：T=1 上 autograd 与 FD 的具体数值与收敛性。"""
import torch
import R1_grad as M

for ts in (1, 2):
    M.T_STEPS = ts
    for e in ((0, 0), (3, 5), (7, 2)):
        an, fd, rel = M.fd_grad_autograd(t_steps=ts, entry=e)
        print(f"T={ts} Wo1[{e[0]},{e[1]}]  an={an:+.8e}  fd={fd:+.8e}  rel={rel:.3e}")
    # FD 随 h 的收敛
    for h in (1e-4, 1e-6, 1e-8):
        P = M.build(M.SEED0, "V0")
        L, _, _ = M.rollout(P, "V0")
        g = torch.autograd.grad(L, [P["Wo1"]], allow_unused=True)[0]
        Qp = M.build(M.SEED0, "V0"); Qp["Wo1"].data[0, 0] += h
        Qm = M.build(M.SEED0, "V0"); Qm["Wo1"].data[0, 0] -= h
        Lp, _, _ = M.rollout(Qp, "V0"); Lm, _, _ = M.rollout(Qm, "V0")
        print(f"    h={h:.0e}  an={float(g[0,0]):+.10e}  fd={float((Lp-Lm)/(2*h)):+.10e}")
