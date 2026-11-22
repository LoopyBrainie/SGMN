#!/opt/pytorch/bin/python
r"""DIR-31 阈值标定第二轮：在**选定构型**（sd=0.5, η=0.05, λ=0.05）上读 12 个实例的
四项判据量，供卡片 §1.1 冻结阈值（阈值取实测下界的 1/4~1/10，失败变体落在 0）。"""
import statistics as st
import R1_grad as M
print(f"构型 sd={M.INIT_SD} eta={M.ETA} lam={M.LAM} gamma={M.GAMMA} T={M.T_STEPS}")
for v in M.VARIANTS:
    rs = [M.probe(M.SEED0 + i, v) for i in range(12)]
    q = lambda k: (min(r[k] for r in rs), st.median([r[k] for r in rs]))
    g_in = (min(r["ain"]["max"] for r in rs), st.median([r["ain"]["max"] for r in rs]))
    print(f"{v:3s} rho {q('rho_mem')[0]:.5f}/{q('rho_mem')[1]:.5f}  "
          f"gread {q('gread_min')[0]:.3e}/{q('gread_min')[1]:.3e}  "
          f"gwrite {max(r['gwrite_max'] for r in rs):.3e}  "
          f"gbase {max(r['gbase_max'] for r in rs):.3e}  "
          f"g_in {g_in[0]:.3e}/{g_in[1]:.3e}  act {q('alive_act')[0]:.4f}/{q('alive_act')[1]:.4f}  "
          f"wnone={all(r['gwrite_none_all'] for r in rs)} bnone={all(r['gbase_none_all'] for r in rs)} "
          f"fin={all(r['gread_finite_all'] for r in rs)} "
          f"bmov={max(r['optmoved']['base_moved'] for r in rs):.2e} rmov={min(r['optmoved']['read_moved'] for r in rs):.2e}")
