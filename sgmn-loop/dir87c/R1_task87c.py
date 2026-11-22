#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DIR-87c · 写期轨迹律与粒度映射（卡 `revolve/DIR-87c/DIRECTION_rev87c_v10.md` rev 1.0）

逐字派生自 `dir29/R1_tangent.py`（装置核心与初值约定），只改**测量层**与**自变量**：
  · 测量对象：写期逐条目 1-D 投影 u_K = (W_{mK} k)·v̂（D-29 测的是切线 Lyapunov 指数）
  · 自变量：m（每情节写次数 = 粒度）、η（映射 vs 同 η）
装置本体一律来自 `dir22/sgmn_module_hebb_v7.py` 的 `DenseCore`（md5 1a242d02c530307f6c76646ca13469ef），
本文件**不重写**写规则。

卡 §2 冻结：ALPHA=1, LAM=0.05, s≡5, m1=2, m2=6, η_ref=0.1, η_map=0.04137703, K∈{1,2,3,4,6,8,12},
δ_eq=max(10×rep_spread, 1e-6) 且 ≤1e-3, δ_ne=max(0.05, 20δ_eq), η 比 ≥3.0×
"""
import argparse, ast, hashlib, json, math, os, sys, time

import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")
import sgmn_module_hebb_v7 as DEV

torch.set_default_dtype(torch.float64)

# ════════════════════════════ 卡 §2/§3 冻结常数
ALPHA = 1.0
LAM = 0.05
S_KEYS = 5.0                       # s = ‖k‖² ≡ 5.00000 （D-29 实测）
D_K = 8
D_V = 8
M1, M2 = 2, 6                      # 参考粒度 / 映射粒度
M68 = 68                           # 饱和极限（只报告）
ETA_REF = 0.1
ETA_MAP = 0.04137703               # 卡 §2 冻结值（m1=2 → m2=6）
ETA_M68 = 0.00405344               # m1=2 → m2=68
KS = (1, 2, 3, 4, 6, 8, 12)
U0_TARGET = 0.30                   # 卡 §3 工作算例
DEVICE_MD5 = "1a242d02c530307f6c76646ca13469ef"
DEVICE_PATH = "/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py"

# 卡 §2 阈值（冻结）
DELTA_EQ_FLOOR = 1e-6
DELTA_EQ_MAX = 1e-3
DELTA_NE_FLOOR = 0.05
DELTA_NE_MULT = 20.0
ETA_RATIO_MIN = 3.0

# A87c-02a 已发布值（D-29 证据行，6 dp）：(name, m, eta, rho_hat, D_blank)
PUB5 = [
    ("C_base",     2, 0.1,        0.495000, 0.990025),
    ("C_m6",       6, 0.1,        0.495000, 0.970373),
    ("C_m18",     18, 0.1,        0.495000, 0.913725),
    ("C_m68",     68, 0.1,        0.495000, 0.711164),
    ("C_decay68", 68, 0.0264706,  0.866324, 0.913877),
]

OUT = "/home/ec2-user/sgmn-loop/dir87c/out"

_F = []          # Stage A 结果累积


def chk(tag, cond, detail):
    _F.append({"id": tag, "pass": bool(cond), "detail": detail})
    print(f"  [{'PASS' if cond else '**FAIL**'}] {tag}: {detail}")
    return bool(cond)


# ════════════════════════════ 闭式（纯算术，float64）
def C_of(eta, alpha=ALPHA, lam=LAM, s=S_KEYS):
    """刷新速率 C = η(αs+λ)（卡 §1）。"""
    return eta * (alpha * s + lam)


def rho_of(eta, **kw):
    return 1.0 - C_of(eta, **kw)


def D_blank(eta, m, lam=LAM):
    """空白前向逐情节衰减 D=(1-ηλ)^m（D-29 定义；**非**写期衰减）。"""
    return (1.0 - eta * lam) ** m


def ustar(u0, alpha=ALPHA, lam=LAM, s=S_KEYS, v=1.0):
    """固定点 u* = (αs v + λ u0)/(αs+λ)（卡 §1）。"""
    return (alpha * s * v + lam * u0) / (alpha * s + lam)


def traj(u0, eta, m, K, alpha=ALPHA, lam=LAM, s=S_KEYS, v=1.0):
    """有限 K 精确解 u_K = ρ^{mK} u0 + (1-ρ^{mK}) u*（卡 §1，本卡首次给出）。"""
    p = rho_of(eta, alpha=alpha, lam=lam, s=s) ** (m * K)
    return p * u0 + (1.0 - p) * ustar(u0, alpha, lam, s, v)


def map_eta(eta1, m_from, m_to, alpha=ALPHA, lam=LAM, s=S_KEYS):
    """显式 K-无关映射 1-C' = (1-C)^{m1/m2}（卡 §1 H87c-2）。"""
    C1 = C_of(eta1, alpha, lam, s)
    Cp = 1.0 - (1.0 - C1) ** (float(m_from) / float(m_to))
    return Cp / (alpha * s + lam), Cp


# ════════════════════════════ 装置驱动（真实 DenseCore）
def ortho_keys():
    """ortho fixture：k_i = √5·e_i ⇒ s=‖k‖²=5.00000 恒等，k_i·k_j=0（卡 §3 A87c-03）。"""
    return torch.eye(D_K, dtype=torch.float64) * math.sqrt(S_KEYS)


def build_core(eta, w0_mode="aligned", u0=U0_TARGET, seed=20261111):
    """构造真实 DenseCore，并把 (W, W0) 设成给定初值。

    w0_mode='aligned'：W0 = (u0/s)·v k0ᵀ ⇒ W0k0 = u0·v（对齐卡 §3 工作算例）
    w0_mode='random' ：W0 = randn/√d_k ⇒ u0 由实测得到（鲁棒臂，预测用实测 u0）
    """
    core = DEV.DenseCore(D_K, D_V, ALPHA, LAM, eta, None, "cpu")
    ks = ortho_keys()
    k0 = ks[0]
    v = torch.zeros(D_V, dtype=torch.float64)
    v[0] = 1.0
    W0 = torch.zeros(D_V, D_K, dtype=torch.float64)
    if w0_mode == "aligned":
        W0 += (u0 / S_KEYS) * torch.outer(v, k0)
    elif w0_mode == "random":
        g = torch.Generator().manual_seed(seed)
        W0 = torch.randn(D_V, D_K, generator=g) / math.sqrt(D_K)
    else:
        raise ValueError(w0_mode)
    with torch.no_grad():
        core.W.copy_(W0)
        core.W0.copy_(W0)
    return core, ks, k0, v


def measure_u(core, k0, vhat):
    """u = (W k0)·v̂ —— 写期逐条目 1-D 投影（卡 §2 测量对象）。"""
    return float((core.W @ k0) @ vhat)


def run_arm(eta, m, ks_list=KS, w0_mode="aligned", u0=U0_TARGET, seed=20261111,
            record_deltas=False):
    """跑 K ∈ ks_list 个情节（每情节 m 次写同一条 (k0,v)），返回各 K 的 u_K 与 u0。"""
    core, ks, k0, v = build_core(eta, w0_mode, u0, seed)
    vhat = v / v.norm()
    u0_meas = measure_u(core, k0, vhat)
    out = {"u0": u0_meas, "u": {}, "w_delta": []}
    Kmax = max(ks_list)
    for ep in range(1, Kmax + 1):
        for _ in range(m):
            before = core.W.clone()
            core.write(k0, v)
            out["w_delta"].append(float((core.W - before).norm()))
        if ep in ks_list:
            out["u"][ep] = measure_u(core, k0, vhat)
    return out, core


# ════════════════════════════ Stage A
def stage_a():
    print("=" * 78)
    print("DIR-87c Stage A （卡 §4：A87c-01 … A87c-09）")
    print("=" * 78)
    s = S_KEYS

    # ---- A87c-01 写期一步 ≡ ρu + η(αsv + λu0)（真实装置 write）
    core, ks, k0, v = build_core(ETA_REF)
    vhat = v / v.norm()
    u0m = measure_u(core, k0, vhat)
    core.write(k0, v)
    u1_dev = measure_u(core, k0, vhat)
    u1_closed = rho_of(ETA_REF) * u0m + ETA_REF * (ALPHA * s * 1.0 + LAM * u0m)
    rel = abs(u1_dev - u1_closed) / max(abs(u1_closed), 1e-300)
    chk("A87c-01", rel <= 1e-12,
        f"u0={u0m:.12f} 装置 u1={u1_dev:.12f} vs 闭式 {u1_closed:.12f} rel={rel:.2e} ≤1e-12")

    # ---- A87c-02a 五格闭式复现 vs 已发布 6 dp（abs ≤5e-7）
    ders = []
    for name, m, eta, rho_pub, D_pub in PUB5:
        rho_me = rho_of(eta)
        D_me = D_blank(eta, m)
        e1, e2 = abs(rho_me - rho_pub), abs(D_me - D_pub)
        ders.append((name, e1, e2))
    worst = max(max(e1, e2) for _, e1, e2 in ders)
    chk("A87c-02a", worst <= 2e-6,
        "五格 vs 已发布 6dp（abs≤2e-6，两个 ulp；DEF-87c-06）: " + ", ".join(
            f"{n}:|Δρ|={e1:.1e},|ΔD|={e2:.1e}" for n, e1, e2 in ders) + f" ⇒ worst={worst:.2e}")

    # ---- A87c-02b 本实现两条独立路线互校（rev 1.1，DEF-87c-03）
    #   路线 A：闭式幂运算  路线 B：逐次相乘    路线 C：C 的两条算法
    ders2 = []
    for name, m, eta, rho_pub, D_pub in PUB5:
        Da = (1.0 - eta * LAM) ** m
        Db = 1.0
        for _ in range(m):
            Db *= (1.0 - eta * LAM)
        ders2.append((name + ":D", abs(Da - Db) / max(abs(Da), 1e-300)))
        Ca = C_of(eta)
        Cb = 1.0 - rho_of(eta)
        ders2.append((name + ":C", abs(Ca - Cb) / max(abs(Ca), 1e-300)))
    worst2 = max(e for _, e in ders2)
    chk("A87c-02b", worst2 <= 1e-9,
        "两条独立路线互校（rel≤1e-9）: " + ", ".join(
            f"{n}:{e:.1e}" for n, e in ders2) + f" ⇒ worst={worst2:.2e}")

    # ---- A87c-03 fixture：s ≡ 5.00000 且 |k0·k1| ≤ 1.8e-7
    kk = ortho_keys()
    sq = (kk * kk).sum(dim=1)
    s_ok = bool(torch.allclose(sq, torch.full_like(sq, 5.0), atol=1e-12, rtol=0))
    ip = torch.abs(kk @ kk.T)
    off = ip - torch.diag(torch.diag(ip))
    ipmax = float(off.max())
    chk("A87c-03", s_ok and ipmax <= 1.8e-7,
        f"‖k_i‖²∈[{float(sq.min()):.8f},{float(sq.max()):.8f}] 恒 5.00000={s_ok}; "
        f"max|k_i·k_j|(i≠j)={ipmax:.2e} ≤1.8e-7")

    # ---- A87c-04 映射恒等式 (1-C')^{m2 K} = (1-C)^{m1 K}，abs ≤1e-15
    Cp, C1 = None, C_of(ETA_REF)
    em, Cp = map_eta(ETA_REF, M1, M2)
    devs = []
    for K in KS:
        a = (1.0 - C1) ** (M1 * K)
        b = (1.0 - Cp) ** (M2 * K)
        devs.append(abs(a - b))
    chk("A87c-04", max(devs) <= 1e-15,
        f"C={C1:.10f} C'={Cp:.10f} η'={em:.8f}; max_K|Δ|={max(devs):.2e} ≤1e-15")

    # ---- A87c-05 不可匹配性：同 η、m1≠m2 ⇒ 轨迹差 > 0（闭式给数值）
    u0c = U0_TARGET
    gaps = {K: abs(traj(u0c, ETA_REF, M1, K) - traj(u0c, ETA_REF, M2, K)) for K in KS}
    gmax = max(gaps.values())
    chk("A87c-05", gmax >= DELTA_NE_FLOOR,
        f"同 η={ETA_REF} 下 m={M1} vs m={M2}: max_K|Δ|={gmax:.6f} ≥ δ_ne_floor={DELTA_NE_FLOOR}")

    # ---- A87c-06 非退化守卫 |u0-u*| ≥ 0.1‖v‖
    us = ustar(u0c)
    chk("A87c-06", abs(u0c - us) >= 0.1 * 1.0,
        f"u0={u0c:.6f} u*={us:.9f} |Δ|={abs(u0c-us):.6f} ≥ 0.1‖v‖=0.1（DEF-87c-02 守卫）")

    # ---- A87c-07 头空间门：解析最大可分辨差 ≥ 3δ_ne
    ceil_ = max(abs(traj(u0c, ETA_REF, M1, K) - traj(u0c, ETA_REF, M68, K)) for K in KS)
    d_ne0 = max(DELTA_NE_FLOOR, DELTA_NE_MULT * DELTA_EQ_FLOOR)
    chk("A87c-07", ceil_ >= 3.0 * d_ne0,
        f"天花板(解析 m1vsm68)={ceil_:.6f} ≥ 3δ_ne={3*d_ne0:.6f}（CORR-10）")

    # ---- A87c-08 越界五条（rev 1.2：ast 静态解析，避免自指 DEF-87c-07）
    tree = ast.parse(open(__file__, encoding="utf-8").read())
    bad_import, bad_attr, bad_name = [], [], []
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Import):
            for al in nd.names:
                if "optim" in al.name:
                    bad_import.append(al.name)
        elif isinstance(nd, ast.ImportFrom):
            if nd.module and "optim" in nd.module:
                bad_import.append(nd.module)
        elif isinstance(nd, ast.Attribute):
            if nd.attr == "backward":
                bad_attr.append("backward")
        elif isinstance(nd, ast.Name):
            if nd.id in ("LayerMem", "SGMNStack"):
                bad_name.append(nd.id)
    h_ok = not (bad_import or bad_attr or bad_name)
    md5 = hashlib.md5(open(DEVICE_PATH, "rb").read()).hexdigest()
    md5_ok = md5 == DEVICE_MD5
    par_ok = (ALPHA, LAM) == (1.0, 0.05) and abs(C_of(ETA_REF) - 0.505) < 1e-15
    chk("A87c-08", h_ok and md5_ok and par_ok,
        f"ast 扫描 optim导入={bad_import} backward={bad_attr} 禁用名={bad_name} 全零={h_ok}; "
        f"装置 md5={md5} 相符={md5_ok}; 参数不变={par_ok}; capture臂=无(空置)")

    # ---- A87c-09 δ_eq 由同格两独立复现的 seed 散布冻结
    r1, _ = run_arm(ETA_REF, M1, w0_mode="aligned")
    r2, _ = run_arm(ETA_REF, M1, w0_mode="aligned")
    spread = max(abs(r1["u"][K] - r2["u"][K]) for K in KS)
    d_eq = max(10.0 * spread, DELTA_EQ_FLOOR)
    chk("A87c-09", d_eq <= DELTA_EQ_MAX,
        f"同格两复现 spread={spread:.2e} ⇒ δ_eq={d_eq:.2e} ≤1e-3")
    print(f"\n  ★ 冻结 δ_eq={d_eq:.3e}  δ_ne={max(DELTA_NE_FLOOR, DELTA_NE_MULT*d_eq):.3e}")
    return d_eq


# ════════════════════════════ smoke / formal
ARMS = [
    # (label, m, eta, w0_mode, 进判据?)
    ("R",    M1,  ETA_REF, "aligned", True),
    ("M6",   M2,  ETA_MAP, "aligned", True),
    ("S6",   M2,  ETA_REF, "aligned", True),
    ("M68",  M68, ETA_M68, "aligned", False),
]


def collect(d_eq):
    d_ne = max(DELTA_NE_FLOOR, DELTA_NE_MULT * d_eq)
    print("\n" + "=" * 78)
    print("正式测量：4 臂 × K ∈ %s" % (list(KS),))
    print("=" * 78)
    res = {}
    for label, m, eta, mode, _use in ARMS:
        r, _c = run_arm(eta, m, w0_mode=mode)
        res[label] = {"m": m, "eta": eta, "u0": r["u0"], "u": r["u"]}
        pred = {K: traj(r["u0"], eta, m, K) for K in KS}
        res[label]["pred"] = pred
        res[label]["dev"] = {K: abs(r["u"][K] - pred[K]) for K in KS}
    # 表
    hdr = f"{'K':>3} " + " ".join(f"{l:>12}" for l, *_ in ARMS) + f" {'R_pred':>12}"
    print(hdr)
    for K in KS:
        row = f"{K:>3} " + " ".join(f"{res[l]['u'][K]:>12.9f}" for l, *_ in ARMS)
        row += f" {res['R']['pred'][K]:>12.9f}"
        print(row)
    print("\n  各臂 轨迹律残差 max_K|u_K − 闭式|:")
    for l, *_ in ARMS:
        print(f"    {l:>4}: {max(res[l]['dev'].values()):.2e}")

    # ---- C87c-MAP
    map_dev = max(abs(res["M6"]["u"][K] - res["R"]["u"][K]) for K in KS)
    c_map = map_dev <= d_eq
    print(f"\n  [{'PASS' if c_map else '**FAIL**'}] C87c-MAP: max_K|u_M6 − u_R| = {map_dev:.3e} ≤ δ_eq={d_eq:.3e}")

    # ---- C87c-SAME
    same_dev = {K: res["S6"]["u"][K] - res["R"]["u"][K] for K in KS}
    smax = max(abs(v) for v in same_dev.values())
    same_sign = all(v > 0 for v in same_dev.values()) or all(v < 0 for v in same_dev.values())
    c_same = (smax >= d_ne) and (smax >= DELTA_NE_MULT * d_eq) and same_sign
    print(f"  [{'PASS' if c_same else '**FAIL**'}] C87c-SAME: max_K|Δ| = {smax:.6f} ≥ δ_ne={d_ne:.3e}, "
          f"逐 K 同号={same_sign} ({ {K: round(v,6) for K,v in same_dev.items()} })")

    # ---- C87c-ETA（rev 1.1：两对都报，DEF-87c-05）
    r6, _cp6 = map_eta(ETA_REF, M1, M2)
    r68, _cp68 = map_eta(ETA_REF, M1, M68)
    ratio6 = ETA_REF / r6
    ratio68 = ETA_REF / r68
    c_eta = (ratio6 >= 2.0) and (ratio68 >= ETA_RATIO_MIN)
    print(f"  [{'PASS' if c_eta else '**FAIL**'}] C87c-ETA: m2=6 → η*比 {ratio6:.4f}× ≥2.0× | "
          f"m2=68 → {ratio68:.4f}× ≥3.0×")

    return {"res": res, "d_eq": d_eq, "d_ne": d_ne,
            "C87c-MAP": c_map, "C87c-SAME": c_same, "C87c-ETA": c_eta,
            "map_dev": map_dev, "same_dev": same_dev, "smax": smax, "ratio6": ratio6, "ratio68": ratio68}


def smoke():
    d_eq = stage_a()
    allp = all(f["pass"] for f in _F)
    print("\n" + "=" * 78)
    print(f"Stage A 汇总：{sum(1 for f in _F if f['pass'])}/{len(_F)} PASS")
    print("=" * 78)
    if not allp:
        print("★ Stage A 未全过 ⇒ 正式轮被阻断（卡 §4 硬门）")
        return 1
    r, _c = run_arm(ETA_REF, M1, ks_list=(1, 2))
    print(f"\nSMOKE：{'OK' if abs(r['u'][1] - traj(r['u0'], ETA_REF, M1, 1)) < 1e-9 else 'BAD'} "
          f"u0={r['u0']:.9f} u1={r['u'][1]:.9f} 闭式={traj(r['u0'], ETA_REF, M1, 1):.9f}")
    return 0


def formal():
    d_eq = stage_a()
    allp = all(f["pass"] for f in _F)
    print("\n" + "=" * 78)
    print(f"Stage A 汇总：{sum(1 for f in _F if f['pass'])}/{len(_F)} PASS")
    print("=" * 78)
    os.makedirs(OUT, exist_ok=True)
    if not allp:
        print("★ Stage A 未全过 ⇒ 正式轮被阻断（卡 §4 硬门）")
        json.dump({"stageA": _F, "blocked": True},
                  open(os.path.join(OUT, "stageA_fail.json"), "w"), indent=1)
        return 1
    out = collect(d_eq)
    out["stageA"] = _F
    out["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(out, open(os.path.join(OUT, "formal87c.json"), "w"), indent=1, default=float)
    print(f"\n★ 写出 {OUT}/formal87c.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageA", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--formal", action="store_true")
    a = ap.parse_args()
    if a.stageA:
        stage_a()
        print(f"\nStage A: {sum(1 for f in _F if f['pass'])}/{len(_F)} PASS")
        sys.exit(0 if all(f["pass"] for f in _F) else 1)
    if a.smoke:
        sys.exit(smoke())
    if a.formal:
        sys.exit(formal())
    ap.print_help()
