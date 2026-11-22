"""DIR-24 工作算例之二：写序系数 $c_a,c_b$ 的闭式（$k=e_1$、单位 $v$、$v_a\\perp v_b$）。

闭式：只更新第 $i$ 列 ⇒ $c\\leftarrow(1-\\eta(\\alpha+\\lambda))c+\\eta\\alpha v$。
$a,a,b,b$ 两次写 ⇒ $c_a=(1-\\eta(\\alpha+\\lambda))^3\\eta\\alpha$，$c_b=\\eta\\alpha\\cdot(1+(1-\\eta(\\alpha+\\lambda))(1-\\eta(\\alpha+\\lambda)))$。
"""
import math, sys, torch
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")
from sgmn_module_hebb_v7 import HebbianCore

for D in (896, 64):
    ALPHA, LAM, ETA, MW = 1.0, 0.05, 0.1, 2
    g = torch.Generator().manual_seed(3)
    c = HebbianCore(D, D, ALPHA, LAM, ETA, g, "cpu", factorized=False, n_entries=8, d_model_sig=8)
    c.W.zero_(); c.W0.zero_()
    va = torch.zeros(D); va[0] = 1.0
    vb = torch.zeros(D); vb[1] = 1.0
    k = torch.zeros(D); k[0] = 1.0
    for mbr, v in ((0, va), (1, vb)):
        for _ in range(MW):
            c.write(k, v)
    col = c.W[:, 0]
    gam = 1 - ETA * (ALPHA + LAM)
    ca_an = ETA * ALPHA * (1 + gam) * gam ** 2      # a,a,b,b：a 系数被后两次 b 写各衰减一次
    cb_an = ETA * ALPHA * (1 + gam)                  # b 系数：最后两次写恰好是 b
    print(f"d={D:4d}  device c_a={float(col @ va):.6f}  closed={ca_an:.6f}"
          f"  device c_b={float(col @ vb):.6f}  closed={cb_an:.6f}"
          f"  ratio c_b/c_a device={float(col @ vb)/float(col @ va):.4f}")
