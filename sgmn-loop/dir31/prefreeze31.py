#!/opt/pytorch/bin/python
r"""DIR-31 **冻结前**标定（不是判据计算）：24 个实例的下界 + 两项自校验的可行性。
输出用于把卡片 §1.1 的四个阈值与 §5 的 FD 容差写死。"""
import statistics as st, torch
import R1_grad as M

# ── ① 24 实例的判据量下界（V0）──
rs = [M.probe(M.SEED0 + i, "V0") for i in range(M.N_INST)]
print(f"N={M.N_INST}  ρ_mem min/med = {min(r['rho_mem'] for r in rs):.5f} / {st.median([r['rho_mem'] for r in rs]):.5f}")
print(f"        gread min       = {min(r['gread_min'] for r in rs):.4e}")
print(f"        g_in  min       = {min(r['ain']['max'] for r in rs):.4e}")
print(f"        act   min/med   = {min(r['alive_act'] for r in rs):.5f} / {st.median([r['alive_act'] for r in rs]):.5f}")
print(f"        结构: wnone={all(r['gwrite_none_all'] for r in rs)} bnone={all(r['gbase_none_all'] for r in rs)} "
      f"fin={all(r['gread_finite_all'] for r in rs)}")

# ── ② 确定性：同种子两次必须逐位相同 ──
a, b = M.probe(M.SEED0, "V0"), M.probe(M.SEED0, "V0")
print(f"确定性 L 逐位相同 = {a['L'] == b['L']}  (Δ={abs(a['L']-b['L']):.3e})")

# ── ③ 有限差分 vs autograd（中心差分，float64）──
P = M.build(M.SEED0, "V0")
leaf = {k: P[k] for k in M.READ_KEYS}
L, _, _ = M.rollout(P, "V0")
g = torch.autograd.grad(L, [leaf["Wo1"]], allow_unused=True)[0]
h = 1e-6
idx = [(0, 0), (3, 5), (7, 2)]
worst = 0.0
for i, j in idx:
    Q = M.build(M.SEED0, "V0"); Q["Wo1"].data[i, j] += h
    Lp, _, _ = M.rollout(Q, "V0")
    Q = M.build(M.SEED0, "V0"); Q["Wo1"].data[i, j] -= h
    Lm, _, _ = M.rollout(Q, "V0")
    fd = (float(Lp) - float(Lm)) / (2 * h)
    an = float(g[i, j])
    rel = abs(fd - an) / max(abs(an), 1e-12)
    worst = max(worst, rel)
    print(f"  Wo1[{i},{j}] autograd={an:+.8e} fd={fd:+.8e} rel={rel:.2e}")
print(f"有限差分最大相对差 = {worst:.3e}  (拟冻结 FD_TOL=1e-6 ⇒ {'PASS' if worst < 1e-6 else 'FAIL'})")
