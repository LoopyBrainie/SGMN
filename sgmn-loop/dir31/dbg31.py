#!/opt/pytorch/bin/python
"""逐键诊读：V0/V3 的 10 个读取参数梯度分别是什么。"""
import R1_grad as M
for v in ("V0", "V3", "V2"):
    r = M.probe(M.SEED0, v)
    print(f"--- {v}  rho={r['rho_mem']:.5f}  g_in={r['ain']['max']:.3e}")
    for k in M.READ_KEYS:
        d = r["read"][k]
        print(f"    {k:5s} none={d['none']!s:5s} max={d['max']:.4e} finite={d['finite']}")
