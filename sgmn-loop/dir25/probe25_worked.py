"""DIR-25 §0 门禁算例：验证「容量瓶颈由读口径决定」这一测量对象可用。
两个读口径：(i) oracle 最小二乘（D-31 口径，信息上界）；(ii) cosine-argmax（D-28 口径）。
预测：(1) oracle 读在 r=d 时 margin=±1，与 N 无关（衰减不进入可分性）；
      (2) argmax 读在 r=d 时命中随 N 单调下降（衰减/幅度竞争进入）。
"""
import sys, math, torch
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")
from sgmn_module_hebb_v7 import HebbianCore

D, ALPHA, LAM, ETA, M = 896, 1.0, 0.05, 0.1, 2

def build(N, seed):
    g = torch.Generator().manual_seed(seed)
    c = HebbianCore(D, D, ALPHA, LAM, ETA, g, "cpu", factorized=True, n_entries=1024, d_model_sig=8)
    U = torch.randn(D, D, generator=g) / math.sqrt(D); V = torch.randn(D, D, generator=g) / math.sqrt(D)
    c.U.copy_(U); c.V.copy_(V)                      # r0 = d：不施秩约束 ⇒ 纯衰减对照
    K = torch.zeros(N, D); va, vb = [], []
    r = torch.randn(N, D, generator=g)
    base = torch.nn.functional.normalize(torch.randn(N, D, generator=g), dim=-1)
    for i in range(N):
        K[i, i] = 1.0                               # 标准基键（对间精确正交）
        a = base[i]
        x = r[i] - (r[i] @ a) * a
        b = torch.nn.functional.normalize(x, dim=-1)
        va.append(a); vb.append(b)
        for v in (a, b):
            for _ in range(M):
                c.write(K[i], v)
    return c, K, torch.stack(va), torch.stack(vb)

def unit(x): return x / x.norm()

for N in (2, 35, 140, 280):
    c, K, va, vb = build(N, 20261028)
    W = c._W().detach()
    # (i) oracle 读
    mg = []
    for i in (0, N // 2, N - 1):
        t = va[i]
        q = torch.linalg.lstsq(W, t.unsqueeze(1)).solution[:, 0]
        y = c.read(q.unsqueeze(0))[0]
        mg.append(float(y @ unit(va[i]) - y @ unit(vb[i])))
    # (ii) 内容通道（D-28 口径）：y = W k_i，不给解方程 ⇒ 幅度/衰减进入
    fid, nrm_i = [], []
    for i in (0, N // 2, N - 1):
        y = W @ K[i]
        fid.append(float(torch.nn.functional.cosine_similarity(y.unsqueeze(0), va[i].unsqueeze(0))))
        nrm_i.append(float(y.norm()))
        pred = (1 - ETA * LAM) ** (2 * (N - 1 - i))       # 衰减预测：每个后续写步 ×(1-ηλ)
        print(f"      i={i:4d}: cos(Wk,v)={fid[-1]:+.4f} ||Wk||={nrm_i[-1]:.4e} 预测(1-ηλ)^2(N-i)={pred:.4f}")
    print(f"N={N:4d} oracle |margin|(i=0,mid,last)={[f'{abs(m):.4f}' for m in mg]}"
          f" | 内容通道 cos(i=0,mid,last)={[f'{c:+.4f}' for c in fid]}")
