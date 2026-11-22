r"""四种格式的操作域边界闭式（开卡前标定；用于 rev 1.1）"""
import numpy as np
D, LAM = 6, 0.05
def mat_batch(c): return 2 / (c / D + LAM)
def fac_batch(c): return D / c
def mat_cycle(c):
    """逐键矩阵：方向 j 的每周期因子 = |1−η(c+λ)|·(1−ηλ)^{D−1} = 1（η(c+λ)>2 支）"""
    f = lambda e: abs(1 - e * (c + LAM)) * (1 - e * LAM) ** (D - 1) - 1
    lo0, hi0 = 2.0001 / (c + LAM), 0.999 / LAM
    grid = np.linspace(lo0, hi0, 40001)          # f 不单调：取**第一个**由负转正处
    vals = [f(e) for e in grid]
    idx = next((i for i in range(1, len(vals)) if vals[i] > 0 >= vals[i - 1]), None)
    if idx is None: return float("nan")
    lo, hi = grid[idx - 1], grid[idx]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)
print("  c    mat_batch    fac_batch=D/c   mat_cycle    （三者比）")
for c in (2.0, 3.0, 6.0, 12.0, 24.0):
    print("%-6.1f %-13.4f %-15.4f %-12.4f mat_cyc/mat_batch=%.3f"
          % (c, mat_batch(c), fac_batch(c), mat_cycle(c), mat_cycle(c) / mat_batch(c)))
