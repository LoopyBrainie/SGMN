#!/opt/pytorch/bin/python
r"""DIR-31 构型标定（**开卡前**）：找写通道真正活着的 $(\sigma_{uv},\eta,\lambda)$ 区间。
不进判据；只用于把卡片 §3 的装置常数选在「记忆确实改变输出」的可用点上。"""
import itertools, statistics as st
import R1_grad as M

print(f"{'sd':>5s} {'eta':>6s} {'lam':>6s} {'|W|':>9s} {'dW':>9s} {'act':>7s} {'rho_mem':>8s} {'L':>10s} {'ok':>4s}")
rows = []
for sd, eta, lam in itertools.product((0.1, 0.3, 0.5, 1.0), (0.01, 0.05, 0.1, 0.2), (0.05, 0.01)):
    M.INIT_SD, M.ETA, M.LAM = sd, eta, lam
    try:
        rs = [M.probe(M.SEED0 + i, "V0") for i in range(3)]
    except Exception as e:
        print(f"{sd:5.2f} {eta:6.2f} {lam:6.2f}  EXC {type(e).__name__}: {e}")
        continue
    rho = min(r["rho_mem"] for r in rs)
    L = min(r["L"] for r in rs)
    dW = statistics_median = st.median([r["alive_dW"] for r in rs])
    W = st.median([r["Wmax"] for r in rs]) if "Wmax" in rs[0] else float("nan")
    ok = all(r["gread_finite_all"] and r["gwrite_none_all"] and r["gbase_none_all"] for r in rs)
    act = dW / (W + 1e-12)
    rows.append((rho, sd, eta, lam, act))
    print(f"{sd:5.2f} {eta:6.2f} {lam:6.2f} {W:9.3e} {dW:9.3e} {act:7.4f} {rho:8.5f} {L:10.3f} {str(ok)[:4]:>4s}")
rows.sort(reverse=True)
print("\n按 rho_mem 排序（前 6）：")
for rho, sd, eta, lam, act in rows[:6]:
    print(f"  rho={rho:.5f}  sd={sd} eta={eta} lam={lam} act={act:.4f}")
