"""DIR-26 续 · probe26_P.py —— P1–P4 四条测量对象路径的**最小算例比较**（不产生判据）

背景：`dir26/WORKED-EXAMPLES.md` 已证明三种读口径不可用（求解读退化 / 塑性绝对读被总质量支配 / 检索@1 在 N=4 天花板）。

本探针的判据只有一条（"测量对象可用"）：在**等写量**对照下，指标能把 A/B/C/D 四臂按预测分出层次，且
  · (i)  对**均匀幅度**（$\\alpha$ 缩放 = 速率效应）**不敏感**  —— 检验：$\\mathrm{ret1}(C)\\approx\\mathrm{ret1}(A)$ 属预期，
  · (ii) 对**分布**（哪条目吃质量）**敏感** —— 检验：$\\mathrm{ret1}(B)>\\mathrm{ret1}(C)$，
  · (iii) 不被总质量支配 —— 检验：四臂指标与 $\\lVert P\\rVert$（总塑性范数）**不**同序。

四臂（$m=2$、$R=4$ 次呈现、抖动 $\\cos\\theta=0.9$）：
  A $\\alpha\\equiv1$ 全写 · B 内容门控（$\\cos(W_tk_t,v_t)>c_\\tau$ 则跳过） · C 恒定 $\\alpha\\equiv\\bar\\alpha$（$\\bar\\alpha$=B 的发放率，**等写量**）
  D 只写每条目首现（位置选择，非内容选择）
  ★ R **随机门控**（本探针新增的关键控制臂）：与 B **逐位等发放数**（同 k、同写量），但**不看内容**随机跳过
     ⇒ B vs R 把"内容"从"次数/质量"里彻底分离出来；B vs C 分离"集中 vs 摊薄"（等质量）；R vs C 分离"少写 vs 摊薄"
指标（全部**不含解方程**，只用框架 §2.2 的 $y=\\mathrm{sg}(W)q$ 与 $P=UV^\\top$）：
  M1 `ret1_P`   ：$Pk_i$ 与 $\\{v_j\\}$ 的 $\\cos$ 矩阵上 **argmax 命中率**（对塑性均匀缩放严格不变 ⇒ 对速率效应免疫）
  M2 `rowm_P`   ：行归一化检索裕度 $\\frac1N\\sum_i \\frac{s_{ii}-\\max_{j\\ne i}s_{ij}}{\\lVert Pk_i\\rVert}$（连续版，避免天花板）
  M3 `cosP`     ：$\\cos(Pk_i,v_i)$ 均值（参照：已验证被总质量支配）
  M4 `tik`      ：Tikhonov 求解读 $q=(W^\\top W+\\gamma I)^{-1}W^\\top t$（$\\gamma$ 按 $\\operatorname{tr}(W^\\top W)/d$ 标定），同时报 $\\kappa(W)$ 与解残差
  M5 `oracle`   ：$\\mathrm{lstsq}$（参照：已验证退化）
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
THETA = math.radians(25.8419)          # cos = 0.9
R0 = 4
SEED = 20261028
N_GRID = (4, 8, 12)


def unit(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-30)


def make_core(d, seed, r0, alpha):
    g = torch.Generator().manual_seed(seed)
    c = HebbianCore(d, d, alpha, LAM, ETA, g, "cpu", factorized=True,
                    n_entries=8192, d_model_sig=8)
    U = (torch.randn(d, d, generator=g) / math.sqrt(d))
    V = (torch.randn(d, d, generator=g) / math.sqrt(d))
    U[:, r0:] = 0.0
    V[:, r0:] = 0.0
    c.U.copy_(U); c.V.copy_(V)
    return c


def build_stream(seed, d, n, r_pres, theta):
    g = torch.Generator().manual_seed(seed + 1)
    E = torch.zeros(n, d)
    for i in range(n):
        E[i, i] = 1.0
    U = unit(torch.randn(n, r_pres, d, generator=g))
    K = torch.zeros(r_pres, n, d)
    for rho in range(r_pres):
        for i in range(n):
            u = U[i, rho] - (U[i, rho] @ E[i]) * E[i]
            u = unit(u) if float(u.norm()) > 1e-6 else unit(torch.randn(d, generator=g))
            K[rho, i] = math.sqrt(S_KEY) * unit(math.cos(theta) * E[i] + math.sin(theta) * u)
    va = unit(torch.randn(n, d, generator=g))
    rr = torch.randn(n, d, generator=g)
    vb = unit(rr - (rr * va).sum(-1, keepdim=True) * va)
    proto = math.sqrt(S_KEY) * E.clone()                  # 原型键：无抖动，$\\cos(k_{i,\\rho},proto_i)=\\cos\\theta$
    return {"K": K, "va": va, "vb": vb, "proto": proto,
            "target_a": torch.arange(n) % 2 == 0, "n": n}


# ── 指标层（不含解方程）
def metrics(W, W0, K, vb, n, proto):
    P = W - W0
    Y = P @ proto.T                                      # (d, n) 塑性部分对**原型键**的响应
    S = unit(Y.T) @ unit(vb).T                            # (n, n) cos(P k_i, v_j)
    ret1 = float(np.mean([int(torch.argmax(S[i]).item()) == i for i in range(n)]))
    rowm = float(np.mean([(float(S[i, i]) - float(torch.max(torch.cat([S[i, :i], S[i, i + 1:]]))))
                          / max(float((P @ proto[i]).norm()), 1e-9) for i in range(n)]))
    cosp = float(np.mean([float(S[i, i]) for i in range(n)]))
    kappa = float(torch.linalg.cond(W).item())
    return {"ret1_P": ret1, "rowm_P": rowm, "cosP": cosp, "kappa_W": kappa,
            "P_norm": float(P.norm()), "S": S}


def tik_read(W, va, vb, target_a, gamma):
    d = W.shape[0]
    I = torch.eye(d)
    A = W.T @ W + gamma * I
    ok, res = [], []
    for i in range(len(va)):
        t = va[i] if bool(target_a[i]) else vb[i]
        q = torch.linalg.solve(A, W.T @ t)
        y = W @ q
        r = float((W @ q - t).norm()) / max(float(t.norm()), 1e-9)
        res.append(r)
        mg = float(y @ unit(va[i]) - y @ unit(vb[i]))
        ok.append((mg > 0) if bool(target_a[i]) else (mg < 0))
    return float(np.mean(ok)), float(np.mean(res))


def oracle_read(W, va, vb, target_a):
    ok = []
    for i in range(len(va)):
        t = va[i] if bool(target_a[i]) else vb[i]
        q = torch.linalg.lstsq(W, t.unsqueeze(1)).solution[:, 0]
        y = W @ q
        mg = float(y @ unit(va[i]) - y @ unit(vb[i]))
        ok.append((mg > 0) if bool(target_a[i]) else (mg < 0))
    return float(np.mean(ok))


def c_tau_pilot(n, d):
    """门控阈值标定：在 pilot 流上取「首现 cos」与「重复 cos」中位数的**中点**（规则冻结，阈值由 pilot 定）。"""
    D = build_stream(SEED, d, n, R_PRES, THETA)
    core = make_core(d, SEED, R0, 1.0)
    first, rep = [], []
    for rho in range(R_PRES):
        for i in range(n):
            k = D["K"][rho, i]
            for v in (D["va"][i], D["vb"][i]):
                for _ in range(M):
                    c = float(torch.nn.functional.cosine_similarity(
                        (core._W() @ k).unsqueeze(0), v.unsqueeze(0)).item())
                    (first if rho == 0 else rep).append(c)
                    core.write(k, v)
    lo, hi = float(np.median(first)), float(np.median(rep))
    return 0.5 * (lo + hi), {"first_median": lo, "repeat_median": hi}


def run(n, arm, alpha_const=None, tau=None, gamma_rel=1e-2, keep_seed=None, n_keep=None):
    """keep_seed/n_keep 供 R 臂使用：在 total 个写位置上**随机**保留 n_keep 个（与 B 的发放数逐位相同）。"""
    keep = None
    if arm == "R":
        gk = torch.Generator().manual_seed(int(keep_seed))
        perm = torch.randperm(R_PRES * n * 2 * M, generator=gk)
        keep = set(perm[:int(n_keep)].tolist())
    D = build_stream(SEED, D_MODEL, n, R_PRES, THETA)
    a = 1.0 if alpha_const is None else alpha_const
    core = make_core(D_MODEL, SEED, R0, a)
    issued = 0
    for rho in range(R_PRES):
        for i in range(n):
            k = D["K"][rho, i]
            for v in (D["va"][i], D["vb"][i]):
                for _ in range(M):
                    t_idx = ((rho * n + i) * 2 + (0 if bool(v is D["va"][i]) else 1)) * M + _
                    if arm == "A" or arm == "C":
                        go = True
                    elif arm == "D":
                        go = (rho == 0)
                    elif arm == "R":
                        go = (t_idx in keep)
                    else:                                   # B：内容门控
                        c_now = float(torch.nn.functional.cosine_similarity(
                            (core._W() @ k).unsqueeze(0), v.unsqueeze(0)).item())
                        go = not (c_now > tau)
                    if go:
                        core.write(k, v)
                        issued += 1
    W = core._W().detach()
    mg = metrics(W, core.W0, D["K"], D["vb"], n, D["proto"])
    gamma = gamma_rel * float((W.T @ W).trace()) / W.shape[0]
    tik, tik_res = tik_read(W, D["va"], D["vb"], D["target_a"], gamma)
    n_ok = int(round(mg["ret1_P"] * n))
    out = {"N": n, "arm": arm, "issued": issued, "total": R_PRES * n * 2 * M, "n_correct": n_ok,
           "eff": n_ok / max(issued, 1),
           "alpha_bar_issue": issued / (R_PRES * n * 2 * M),
           "ret1_P": mg["ret1_P"], "rowm_P": mg["rowm_P"], "cosP": mg["cosP"],
           "kappa_W": mg["kappa_W"], "P_norm": mg["P_norm"],
           "tik": tik, "tik_res": tik_res, "oracle": oracle_read(W, D["va"], D["vb"], D["target_a"])}
    return out


if __name__ == "__main__":
    rows = []
    for n in N_GRID:
        tau, cal = c_tau_pilot(n, D_MODEL)
        print(f"\n=== N={n} · c_τ={tau:.4f} (首现中位 {cal['first_median']:.4f} / 重复中位 {cal['repeat_median']:.4f}) ===")
        rows_b = run(n, "B", tau=tau)
        a_bar = rows_b["alpha_bar_issue"]
        rows_r = run(n, "R", keep_seed=SEED + 7, n_keep=rows_b["issued"])
        arms = [("A", run(n, "A")), ("B", rows_b), ("C", run(n, "C", alpha_const=a_bar)),
                ("D", run(n, "D")), ("R", rows_r)]
        hdr = f"{'臂':<3}{'发放':>7}{'命中':>5}{'效率':>8}{'ret1_P':>8}{'rowm_P':>9}{'cosP':>8}{'tik':>7}{'oracle':>8}{'|P|':>8}{'kappa':>10}"
        print(hdr)
        for name, r in arms:
            print(f"{name:<3}{r['issued']:>4}/{r['total']:<3}{r['n_correct']:>5}{r['eff']:>8.3f}"
                  f"{r['ret1_P']:>8.3f}{r['rowm_P']:>9.3f}"
                  f"{r['cosP']:>8.3f}{r['tik']:>7.3f}{r['oracle']:>8.3f}"
                  f"{r['P_norm']:>8.3f}{r['kappa_W']:>10.2e}")
            rows.append({**r, "c_tau": tau, **cal})
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "probe26_P.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rows, open(p, "w"), indent=1, ensure_ascii=False)
    print("\n→", p)
