"""DIR-24 工作算例（MVP 标准 §1(c) 要求）：**只核测量对象与闭式**，不产判据读数。

核对两件事：
  W1  秩-$r_0$ 支撑的不变性：$U,V$ 只在首 $r_0$ 列非零 ⇒ 140 次写后 $\mathrm{rank}(W)\le r_0$（精确不变）。
  W2  读的闭式：$y=\mathrm{Proj}_{\mathrm{span}\{u_k\}}(t)$（最小二乘读的目标是 $t$ 本身）
      ⇒ margin $=\mathrm{Proj}(t)\cdot(\hat v_a-\hat v_b)$；并核对 $r_0=0$（全矩阵参考步）下的结构性 17/35。
装置：`sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，**零改动**）。
"""
import math, json, sys, torch
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")
from sgmn_module_hebb_v7 import HebbianCore

D, NP, ALPHA, LAM, ETA, MW = 64, 12, 1.0, 0.05, 0.1, 2   # 缩小尺度：d=64, 12 对（工作算例用，不是判据）

def unit(x): return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)

def pairs(seed):
    g = torch.Generator().manual_seed(seed)
    va = unit(torch.randn(NP, D, generator=g)); r = torch.randn(NP, D, generator=g)
    vb = unit(r - (r * va).sum(-1, keepdim=True) * va)
    return va, vb

def make(seed, r0):
    """r0=None ⇒ 全矩阵参考步（$W^0=0$）；r0=int ⇒ 因子步 + 首 r0 列支撑。"""
    g = torch.Generator().manual_seed(seed + 7)
    if r0 is None:
        c = HebbianCore(D, D, ALPHA, LAM, ETA, g, "cpu", factorized=False, n_entries=64, d_model_sig=8)
        c.W.zero_(); c.W0.zero_(); return c
    c = HebbianCore(D, D, ALPHA, LAM, ETA, g, "cpu", factorized=True, n_entries=64, d_model_sig=8)
    U = torch.randn(D, D, generator=g) / math.sqrt(D); V = torch.randn(D, D, generator=g) / math.sqrt(D)
    U[:, r0:] = 0.0; V[:, r0:] = 0.0            # ★ 秩约束只由**初值支撑**施加，装置零改动
    c.U.copy_(U); c.V.copy_(V); return c

def run(seed, r0):
    va, vb = pairs(seed); c = make(seed, r0)
    for i in range(NP):
        for mbr, v in ((0, va[i]), (1, vb[i])):
            k = torch.zeros(D); k[i] = 1.0
            for _ in range(MW): c.write(k, v)
    W = c._W().detach()
    rank = int(torch.linalg.matrix_rank(W).item())
    P = W @ torch.linalg.pinv(W)          # 到 range(W) 的投影
    ms, resid = [], 0.0
    for i in range(NP):
        for mbr, t in ((0, va[i]), (1, vb[i])):
            q = torch.linalg.lstsq(W, t.unsqueeze(1)).solution[:, 0]
            y = W @ q
            resid = max(resid, float((y - P @ t).norm()))     # W2：y 必须等于 Proj(t)
            m = float(y @ unit(va[i]) - y @ unit(vb[i]))
            ms.append((m, mbr))
    acc = sum(1.0 for m, mbr in ms if (m > 0) == (mbr == 0)) / len(ms)
    return {"r0": r0, "rank_W": rank, "acc": acc, "proj_resid_max": resid,
            "margin_mean": sum(m for m, _ in ms) / len(ms),
            "acc_a_only": sum(1.0 for m, mbr in ms if mbr == 0 and m > 0) / NP,
            "acc_b_only": sum(1.0 for m, mbr in ms if mbr == 1 and m < 0) / NP}

out = []
for seed in (1, 2):
    for r0 in (None, 1, 2, 8, 24, 64):
        out.append({"seed": seed, **run(seed, r0)})
for row in out:
    print(json.dumps(row))
print("SUMMARY")
for r0 in (None, 1, 2, 8, 24, 64):
    rs = [r for r in out if r["r0"] == r0]
    print(f"  r0={r0}  rank_W={[r['rank_W'] for r in rs]}  acc={[round(r['acc'],3) for r in rs]}"
          f"  resid_max={max(r['proj_resid_max'] for r in rs):.2e}"
          f"  acc_a={[round(r['acc_a_only'],3) for r in rs]}  acc_b={[round(r['acc_b_only'],3) for r in rs]}")
