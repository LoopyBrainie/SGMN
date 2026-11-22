#!/opt/pytorch/bin/python
import R1_grad as M
for v in ("V0", "V4"):
    r = M.probe(M.SEED0, v)
    n_none = sum(1 for k in M.READ_KEYS if r["read"][k]["none"])
    print(f"{v}: 读取参数中 .grad is None 的个数 = {n_none}/10 ; ρ={r['rho_mem']:.5f} ; g_in={r['ain']['max']:.4e}")
    if v == "V4":
        print("    示例 g2:", {k: r["read"]["g2"][k] for k in ("none", "max", "finite")})
