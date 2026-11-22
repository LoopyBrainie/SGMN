"""DIR-26 · probe26_worked.py —— 卡片 §2 的**工作算例**（验证测量对象，不产生判据）

算例（手算可核，$d=16$、$r_0=4$、$N_{\\text{dist}}=4$、$R=4$、$\\theta=25.84^\\circ$）：
  · 每条目 4 次呈现，键 $k_{i,\\rho}=\\sqrt5\\,\\mathrm{norm}(\\cos\\theta\\,e_i+\\sin\\theta\\,u_{i,\\rho})$，$u\\perp e_i$ ⇒ 同条目**余弦恰为 $\\cos\\theta=0.9$**
  · **塑性预算** $r_0=4$（只开 $U,V$ 的前 4 列，DEC-15；$W^0$ 仍满秩——★ 第一版算例把这点写错了）⇒ 更新最多占 **4 个方向**
  · 臂 A（$\\alpha\\equiv1$，全写）vs 臂 D（只写每条目首现，$\\alpha\\in\\{0,1\\}$，跳过其余）
  · 测量：**distinct-item oracle 命中** $A_{\\text{oracle}}=\\frac1N\\sum_i \\mathbf 1[\\operatorname{sign}(\\text{margin}_i)\\text{正确}]$，
    $q_i^*=\\arg\\min_q\\lVert Wq-t_i\\rVert$、$y=Wq_i^*$、$\\text{margin}_i=y\\cdot\\mathrm{unit}(v^a_i)-y\\cdot\\mathrm{unit}(v^b_i)$
手算预期：A 把塑性预算花在**早期条目的 4 个抖动方向**上（$\\cos=0.9$ ⇒ 4 个方向线性无关）⇒ 后 2 个条目无处表示 ⇒ $A\\approx0.5$；
  D 只花 1 个方向/条目 ⇒ 4 条目正好用满 $r_0=4$ ⇒ $A\\approx1.0$。**若实测不是这个图案，说明测量对象写错了，卡片不得冻结。**
"""
import json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")
from sgmn_module_hebb_v7 import HebbianCore

D_MODEL = 16
S_KEY = 5.0
ETA = 0.1
LAM = 0.05
M = 2
R_PRES = 4
THETA = math.radians(25.8419)          # cos = 0.9000
N_DIST = 4
R0 = 4
SEED = 20261028


def unit(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-30)


def make_core(d, seed, r0, alpha):
    g = torch.Generator().manual_seed(seed)
    c = HebbianCore(d, d, alpha, LAM, ETA, g, "cpu", factorized=True,
                    n_entries=4096, d_model_sig=8)
    U = (torch.randn(d, d, generator=g) / math.sqrt(d)).to("cpu")
    V = (torch.randn(d, d, generator=g) / math.sqrt(d)).to("cpu")
    U[:, r0:] = 0.0
    V[:, r0:] = 0.0
    c.U.copy_(U); c.V.copy_(V)
    return c


def oracle_read(W, va, vb, target_a, n):
    ok, mgs = [], []
    for i in range(n):
        t = va[i] if bool(target_a[i]) else vb[i]
        q = torch.linalg.lstsq(W, t.unsqueeze(1)).solution[:, 0]
        y = W @ q
        mg = float(y @ unit(va[i]) - y @ unit(vb[i]))
        mgs.append(mg)
        ok.append((mg > 0) if bool(target_a[i]) else (mg < 0))
    return float(np.mean(ok)), mgs


def build_stream(seed, d, n, r_pres, theta):
    """呈现流：按「每条目连写 r_pres 次」交错（i=0..n-1 各写一次为一轮，共 r_pres 轮）。"""
    g = torch.Generator().manual_seed(seed + 1)
    E = torch.zeros(n, d)
    for i in range(n):
        E[i, i] = 1.0
    U = unit(torch.randn(n, r_pres, d, generator=g))              # 抖动方向（一般不与 e_i 平行）
    K = torch.zeros(r_pres, n, d)                                  # [round, item, d]
    for rho in range(r_pres):
        for i in range(n):
            u = unit(U[i, rho] - (U[i, rho] @ E[i]) * E[i]) if abs(float(U[i, rho] @ E[i])) < 0.999 \
                else unit(torch.randn(d, generator=g))
            K[rho, i] = math.sqrt(S_KEY) * unit(math.cos(theta) * E[i] + math.sin(theta) * u)
    va = unit(torch.randn(n, d, generator=g))
    rr = torch.randn(n, d, generator=g)
    vb = unit(rr - (rr * va).sum(-1, keepdim=True) * va)
    target_a = torch.arange(n) % 2 == 0
    return {"K": K, "va": va, "vb": vb, "target_a": target_a, "n": n, "r_pres": r_pres}


def plastic_read(W, W0, K, vb, n):
    """★ 塑性子空间读：$c_i=\\cos((W-W^0)k_i,\\ v_i)$，命中 = $c_i>0.5$。
    为什么不用 oracle 读（本算例的第一发现）：$W^0$ 满秩 ⇒ $q^*=\\mathrm{lstsq}(W,t)$ 精确解出 $t$ ⇒
    $\\mathrm{margin}=t\\cdot u_a-t\\cdot u_b$ 与记忆内容**无关** ⇒ 恒 1.000（这正是 D-32 中它对 $N,r$ 免疫的原因）。
    $\\cos$ 归一 ⇒ 对「均匀幅度变化」（= 框架 §3.5 的速率效应）不敏感 ⇒ 适合判「独立于速率效应的选择收益」。"""
    Wp = W - W0
    hit = []
    for i in range(n):
        c = float(torch.nn.functional.cosine_similarity((Wp @ K[n - 1, i]).unsqueeze(0),
                                                        vb[i].unsqueeze(0)).item())
        hit.append(c > 0.5)
    return float(np.mean(hit)), hit


def retrieval_read(W, W0, K, vb, n):
    """★ 检索@1（本算例的第三发现）：$\\hat j=\\arg\\max_j\\cos((W-W^0)k_i,v_j)$，命中 = $\\hat j=i$。
    为什么它是对的测量对象：$\\arg\\max$ 对**总塑性质量**（SNR）与**均匀幅度**（速率效应）都不敏感，
    只对「质量落在哪个条目上」敏感 ⇒ 正是「内容选择」要测的东西。"""
    Wp = W - W0
    Y = Wp @ K[n - 1].T                                  # (d,n)：塑性部分对末次键的响应
    S = torch.nn.functional.normalize(Y.T, dim=-1) @ torch.nn.functional.normalize(vb, dim=-1).T
    hits = [int(torch.argmax(S[i]).item()) == i for i in range(n)]
    return float(np.mean(hits)), hits


def run(arm, alpha_const=None, tau=None):
    D = build_stream(SEED, D_MODEL, N_DIST, R_PRES, THETA)
    a = 1.0 if alpha_const is None else alpha_const
    core = make_core(D_MODEL, SEED, R0, a)
    issued = 0
    detail = []
    for rho in range(R_PRES):
        for i in range(N_DIST):
            k = D["K"][rho, i]
            for v in (D["va"][i], D["vb"][i]):
                for _ in range(M):
                    W = core._W().detach()
                    if arm == "A":
                        go = True
                    elif arm == "D":
                        go = (rho == 0)                      # 只写首现（选择上界）
                    elif arm == "B":                         # ★ 内容门控（状态相关）：已能预测的就跳过
                        c_now = float(torch.nn.functional.cosine_similarity(
                            (W @ k).unsqueeze(0), v.unsqueeze(0)).item())
                        go = not (c_now > tau)               # tau = c_τ
                    else:                                    # C：恒定门控（等写量，速率效应控制）
                        go = True
                    if go:
                        core.write(k, v)
                        issued += 1
                    detail.append({"round": rho, "item": i, "issued": bool(go),
                                   "resid": float((W @ k - v).norm())})
    W = core._W().detach()
    acc, mgs = oracle_read(W, D["va"], D["vb"], D["target_a"], N_DIST)
    a_plas, hits = plastic_read(W, core.W0, D["K"], D["vb"], N_DIST)
    a_ret, ret_hits = retrieval_read(W, core.W0, D["K"], D["vb"], N_DIST)
    tot = R_PRES * N_DIST * 2 * M
    return {"arm": arm, "acc": acc, "margins": [round(x, 4) for x in mgs],
            "A_plas": a_plas, "plas_hits": hits,
            "A_ret": a_ret, "ret_hits": ret_hits,
            "issued": issued, "total_presented": tot, "alpha_bar": issued / tot,
            "rank_W": int(torch.linalg.matrix_rank(W, rtol=1e-12).item())}


if __name__ == "__main__":
    out = {}
    # 先跑 B 拿到其实际发放率 ⇒ C 用 α≈发放率（等写量、只差「分布」）
    out["B"] = run("B", tau=0.28)
    a_bar = out["B"]["alpha_bar"]
    for arm in ("A", "D"):
        pass
    out["C"] = run("C", alpha_const=a_bar)
    for arm in ("A", "D"):
        out[arm] = run(arm)
        print(f"  臂 {arm}: A_oracle={out[arm]['acc']:.4f}（恒 1.0 = 退化）· A_plas={out[arm]['A_plas']:.4f} · **A_ret={out[arm]['A_ret']:.4f}** "
              f"hits={out[arm]['plas_hits']} issued={out[arm]['issued']}/{out[arm]['total_presented']} "
              f"rank={out[arm]['rank_W']}")
    for arm in ("B", "C"):
        print(f"  臂 {arm}: A_plas={out[arm]['A_plas']:.4f} · **A_ret={out[arm]['A_ret']:.4f}** ret_hits={out[arm]['ret_hits']} "
              f"issued={out[arm]['issued']}/{out[arm]['total_presented']} ᾱ={out[arm]['alpha_bar']:.4f}")
    # 残差尺度（用于卡片 τ 的标定与 A26 门控分离断言）
    D = build_stream(SEED, D_MODEL, N_DIST, R_PRES, THETA)
    core = make_core(D_MODEL, SEED, R0, 1.0)
    res_first, res_repeat = [], []
    for rho in range(R_PRES):
        for i in range(N_DIST):
            k = D["K"][rho, i]
            for v in (D["va"][i], D["vb"][i]):
                for _ in range(M):
                    W = core._W().detach()
                    r = float((W @ k - v).norm())
                    (res_first if rho == 0 else res_repeat).append(r)
                    core.write(k, v)
    out["resid"] = {"first_median": float(np.median(res_first)),
                    "repeat_median": float(np.median(res_repeat)),
                    "first_max": float(np.max(res_first)), "repeat_min": float(np.min(res_repeat))}
    print(f"  残差：首现中位 {out['resid']['first_median']:.4f} / 重复中位 {out['resid']['repeat_median']:.4f}")
    # ── 三种「内容新颖性」口径的首现/重复分离度（卡片 §2 的 τ 标定依据）
    D = build_stream(SEED, D_MODEL, N_DIST, R_PRES, THETA)
    core = make_core(D_MODEL, SEED, R0, 1.0)
    rec = {"resid": ([], []), "plastic": ([], []), "cos": ([], [])}
    for rho in range(R_PRES):
        for i in range(N_DIST):
            k = D["K"][rho, i]
            for v in (D["va"][i], D["vb"][i]):
                for _ in range(M):
                    W = core._W().detach()
                    Wp = (core._W() - core.W0).detach()
                    tgt = 0 if rho == 0 else 1
                    rec["resid"][tgt].append(float((W @ k - v).norm()))
                    rec["plastic"][tgt].append(float((Wp @ k - v).norm()))
                    rec["cos"][tgt].append(float(torch.nn.functional.cosine_similarity(
                        (W @ k).unsqueeze(0), v.unsqueeze(0)).item()))
                    core.write(k, v)
    sep = {}
    for name, (f_, r_) in rec.items():
        sep[name] = {"first_median": float(np.median(f_)), "repeat_median": float(np.median(r_)),
                     "ratio": float(np.median(f_) / max(np.median(r_), 1e-12))}
        print(f"  {name:8s}: 首现中位 {sep[name]['first_median']:.4f} / 重复中位 {sep[name]['repeat_median']:.4f} "
              f"⇒ 分离 {sep[name]['ratio']:.2f}×")
    out["sep"] = sep
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "probe26_worked.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    print("→", p)
