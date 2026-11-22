#!/opt/pytorch/bin/python
"""定位 FD 与 autograd 不符的根因。"""
import torch
import R1_grad as M

# ① build 可复现性
L1, _, _ = M.rollout(M.build(M.SEED0, "V0"), "V0")
L2, _, _ = M.rollout(M.build(M.SEED0, "V0"), "V0")
print(f"build 可复现: L1={float(L1):.12f} L2={float(L2):.12f} 相同={float(L1)==float(L2)}")

# ② autograd
P = M.build(M.SEED0, "V0")
L, _, _ = M.rollout(P, "V0")
gW = torch.autograd.grad(L, [P["Wo1"]], allow_unused=True, retain_graph=True)[0]
gg = torch.autograd.grad(L, [P["g1"]], allow_unused=True)[0]
print(f"autograd dL/dg1 = {float(gg[0]):+.10e}")

# ③ FD 随 h 的收敛（同一点）
for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
    Qp = M.build(M.SEED0, "V0"); Qp["Wo1"].data[0, 0] += h
    Qm = M.build(M.SEED0, "V0"); Qm["Wo1"].data[0, 0] -= h
    Lp, _, _ = M.rollout(Qp, "V0"); Lm, _, _ = M.rollout(Qm, "V0")
    print(f"  h={h:.0e}  FD={float(Lp-Lm)/(2*h):+.10e}")
# γ1 的 FD
for h in (1e-5, 1e-6):
    Qp = M.build(M.SEED0, "V0"); Qp["g1"].data[0] += h
    Qm = M.build(M.SEED0, "V0"); Qm["g1"].data[0] -= h
    Lp, _, _ = M.rollout(Qp, "V0"); Lm, _, _ = M.rollout(Qm, "V0")
    print(f"  dL/dg1  h={h:.0e} FD={float(Lp-Lm)/(2*h):+.10e}")
