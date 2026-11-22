#!/usr/bin/env python3
"""DIR-89 · 方向卡 rev 1.1 · §7.2-4c：token/segment 到在线步的映射 —— 两粒度各自定位 η* 是否同值。

逐字派生自 dir87c/R1_task87c.py（装置驱动/测量层/fixture 全部照搬），
只改【测量层】(resid 归一到 u*) 与【自变量】(等 token 预算 T + 聚合约定 conv)。
装置 dir22/sgmn_module_hebb_v7.py 零改动（A89-01 md5 断言）。
"""
import argparse, ast, hashlib, json, math, os, sys, time

import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")
import sgmn_module_hebb_v7 as DEV

torch.set_default_dtype(torch.float64)

# ════════════════════════════ 卡 rev 1.1 冻结常数
ALPHA = 1.0
LAM = 0.05
S_KEYS = 5.0
D_K = 8
D_V = 8
M1, M2 = 2, 6
T_BUDGET = 198.0          # = 2·99 = 6·33 ⇒ 两粒度都整除（卡 §2.4）
THETA = 0.05              # 目标残差（卡 §2.4）
U0_TARGET = 0.30
B_BOUND = 1e-12           # 卡 §5 等价界（由条件数推导）
DEVICE_MD5 = "1a242d02c530307f6c76646ca13469ef"
DEVICE_PATH = "/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py"
OUT = "/home/ec2-user/sgmn-loop/dir89/out"
CONVS = ("repeat", "mean", "sum")
# 卡 rev 1.1 §3 冻结预测比值
PRED = {"repeat": 1.0, "mean": 0.343469293, "sum": 3.084371386}
# 卡 §5 C89-GAP
GAP_MIN = 0.3
# 根查找规格（卡 §2.4）
BR_LO = 1e-6
BR_FRAC = 0.99          # 上界 = BR_FRAC/(α s_eff + λ)：C<1 的稳定边界内（DEF-89-05）
RTOL, MAXIT = 1e-14, 200
C_GRID = (1e-6, 1e-4, 1e-3, 0.01, 0.03, 0.06)   # A89-06 横坐标（DEF-89-06）

_F = []


def chk(tag, cond, detail):
    _F.append({"id": tag, "pass": bool(cond), "detail": detail})
    print(f"  [{'PASS' if cond else '**FAIL**'}] {tag}: {detail}")


# ════════════════════════════ 装置驱动（逐字照搬 dir87c）
def ortho_keys():
    return torch.eye(D_K, dtype=torch.float64) * math.sqrt(S_KEYS)


def build_core(eta, w0_mode="aligned", u0=U0_TARGET, seed=20261111):
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
    return float((core.W @ k0) @ vhat)


def s_eff_of(conv, m):
    """卡 §2.3：s_eff 随约定变（sum ⇒ m²s）。"""
    if conv in ("repeat", "mean"):
        return S_KEYS
    return m * m * S_KEYS


def u_star_of(conv, m, u0):
    """卡 §2.1 的不动点（随约定变；sum 的 m²s 进入分子分母）。"""
    se = s_eff_of(conv, m)
    return (ALPHA * se * 1.0 + LAM * u0) / (ALPHA * se + LAM)


def nwrites_of(conv, m, T):
    """卡 §3：总写次数 —— repeat 为 T，mean/sum 为 T/m（DEF-89-03 的核心）。"""
    return T if conv == "repeat" else T / m


def run_conv(conv, eta, m, T, w0_mode="aligned", u0=U0_TARGET, seed=20261111):
    """跑完 T 个 token，返回 (u_measured, u0_measured, u_star)。"""
    core, ks, k0, v = build_core(eta, w0_mode, u0, seed)
    vhat = v / v.norm()
    u0m = measure_u(core, k0, vhat)
    K = int(round(T / m))
    if conv == "repeat":
        for _ in range(K):
            for _ in range(m):
                core.write(k0, v)
    elif conv == "mean":
        for _ in range(K):
            core.write(k0, v)          # 相同键的均值 = k0
    elif conv == "sum":
        kk, vv = m * k0, m * v
        for _ in range(K):
            core.write(kk, vv)
    else:
        raise ValueError(conv)
    u = measure_u(core, k0, vhat)
    return u, u0m, u_star_of(conv, m, u0m)


def resid_of(conv, eta, m, T= T_BUDGET, **kw):
    """卡 §2.4 观测量：归一化偏差 (u − u*)/(u0 − u*)，闭式 = (1−C)^{总写次数}。"""
    u, u0m, us = run_conv(conv, eta, m, T, **kw)
    return (u - us) / (u0m - us)


def C_of(conv, eta, m):
    return eta * (ALPHA * s_eff_of(conv, m) + LAM)


def closed_resid(conv, eta, m, T=T_BUDGET):
    return (1.0 - C_of(conv, eta, m)) ** nwrites_of(conv, m, T)


def eta_star_closed(conv, m, T=T_BUDGET, theta=THETA):
    """卡 §3：由 (1−C)^{nw} = θ 反解 η*。"""
    nw = nwrites_of(conv, m, T)
    C = 1.0 - theta ** (1.0 / nw)
    return C / (ALPHA * s_eff_of(conv, m) + LAM)


def br_hi(conv, m):
    """卡 rev 1.2 §2.4：每臂各自的上界 —— 落在 C<1 的稳定边界内。"""
    return BR_FRAC / (ALPHA * s_eff_of(conv, m) + LAM)


def find_eta(conv, m, T=T_BUDGET, theta=THETA, lo=BR_LO, hi=None,
             tol=RTOL, maxit=MAXIT):
    """卡 §2.4 冻结根查找：二分，resid 对 η 严格递减。"""
    if hi is None:
        hi = br_hi(conv, m)
    flo, fhi = resid_of(conv, lo, m, T), resid_of(conv, hi, m, T)
    if not (flo > theta > fhi):
        return None, {"lo": flo, "hi": fhi, "reason": "bracket-no-sign-change"}
    it = 0
    while it < maxit:
        mid = math.sqrt(lo * hi) if lo > 0 else 0.5 * (lo + hi)
        fm = resid_of(conv, mid, m, T)
        if abs(fm - theta) <= tol or (hi - lo) <= 1e-18:
            return mid, {"iters": it + 1, "resid": fm, "width": hi - lo}
        if fm > theta:
            lo = mid
        else:
            hi = mid
        it += 1
    return 0.5 * (lo + hi), {"iters": it, "resid": resid_of(conv, 0.5 * (lo + hi), m, T),
                             "width": hi - lo, "reason": "maxit"}


# ════════════════════════════ Stage A
def stage_a():
    print("=" * 78)
    print("DIR-89 Stage A （卡 rev 1.3 §4：A89-01 … A89-10）")
    print("=" * 78)

    # A89-01 装置 md5
    md5 = hashlib.md5(open(DEVICE_PATH, "rb").read()).hexdigest()
    chk("A89-01", md5 == DEVICE_MD5, f"装置 md5={md5} 相符={md5 == DEVICE_MD5}")

    # A89-02 write 无 m 参数 + eta 可逐次改写
    src = open(__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    dsig = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "write"]
    core_sig = None
    dsrc = open(DEVICE_PATH, encoding="utf-8").read()
    dtree = ast.parse(dsrc)
    for cls in [n for n in ast.walk(dtree) if isinstance(n, ast.ClassDef) and n.name == "DenseCore"]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "write"]:
            core_sig = [a.arg for a in fn.args.args]
    no_m = core_sig is not None and "m" not in core_sig
    c0, ks, k0, v = build_core(0.1)
    c0.eta = 0.2
    eta_mut = abs(c0.eta - 0.2) < 1e-18
    chk("A89-02", no_m and eta_mut,
        f"DenseCore.write 参数={core_sig} 无m={no_m}; eta 可改写={eta_mut}")

    # A89-03 ortho fixture
    ks = ortho_keys()
    ss = [(k @ k).item() for k in ks]
    off = max(abs(float(ks[i] @ ks[j])) for i in range(D_K) for j in range(D_K) if i != j)
    chk("A89-03", max(abs(s - S_KEYS) for s in ss) <= 1e-12 and off <= 1e-12,
        f"s∈[{min(ss):.8f},{max(ss):.8f}] |s−5|≤1e-12={max(abs(s-S_KEYS) for s in ss)<=1e-12}; max|k_i·k_j|={off:.2e}")

    # A89-04 单写闭式
    e = 0.1
    c1, ks1, k1, v1 = build_core(e)
    vh = v1 / v1.norm()
    u0m = measure_u(c1, k1, vh)
    c1.write(k1, v1)
    u1 = measure_u(c1, k1, vh)
    C = C_of("repeat", e, 2)
    us = u_star_of("repeat", 2, u0m)
    pred = us + (u0m - us) * (1 - C)
    rel4 = abs(u1 - pred) / max(abs(pred), 1e-300)
    chk("A89-04", rel4 <= 1e-12, f"u0={u0m:.12f} 装置u1={u1:.12f} 闭式={pred:.12f} rel={rel4:.2e}")

    # A89-05 指数语义：mK vs K 必须可区分，且实测跟 mK
    m_, K_ = 6, 2
    u_, u0_, us_ = run_conv("repeat", e, m_, m_ * K_)
    obs = (u_ - us_) / (u0_ - us_)
    c_mK = (1 - C) ** (m_ * K_)
    c_K = (1 - C) ** K_
    r_mK = abs(obs - c_mK) / max(abs(c_mK), 1e-300)
    r_K = abs(obs - c_K) / max(abs(c_K), 1e-300)
    chk("A89-05", r_mK <= 1e-12 and r_K > 1e-3,
        f"观测={obs:.12e} vs (1−C)^mK={c_mK:.12e} rel={r_mK:.2e}; vs (1−C)^K={c_K:.12e} rel={r_K:.2e}(>1e-3)")

    # A89-06 根查找单调性
    mono, det6 = True, []
    for cv in CONVS:
        for m in (M1, M2):
            # ★ DEF-89-06：横坐标取在 C 上，保证纵坐标不下溢（两两可分辨）
            denom = ALPHA * s_eff_of(cv, m) + LAM
            seq = [resid_of(cv, Cv / denom, m) for Cv in C_GRID]
            dec = all(seq[i + 1] < seq[i] for i in range(len(seq) - 1))
            dist = all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))
            mono = mono and dec and dist
            det6.append(f"{cv}/m{m}:{'ok' if (dec and dist) else 'NO'}")
    chk("A89-06", mono, f"六臂各自可行区间内 resid 严格递减: {' '.join(det6)}")

    # A89-07 六臂括号异号
    nb, det = 0, []
    for cv in CONVS:
        for m in (M1, M2):
            flo = resid_of(cv, BR_LO, m)
            fhi = resid_of(cv, br_hi(cv, m), m)
            ok = flo > THETA > fhi
            nb += ok
            det.append(f"{cv}/m{m}:{'ok' if ok else 'NO'}")
    chk("A89-07", nb == 6, f"括号异号 {nb}/6 [{' '.join(det)}]")

    # A89-08 自校：§3 的 η* 直接代入 ⇒ 残差 = θ
    w8 = 0.0
    det8 = []
    for cv in CONVS:
        for m in (M1, M2):
            e_star = eta_star_closed(cv, m)
            obs = resid_of(cv, e_star, m)
            w8 = max(w8, abs(obs - THETA))
            det8.append(f"{cv}/m{m}:{obs:.9f}")
    chk("A89-08", w8 <= 1e-12, f"六臂 |resid−θ| 最大={w8:.2e} ≤1e-12 [{', '.join(det8)}]")

    # A89-09 越界五条
    bad_imp, bad_attr, bad_name = [], [], []
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Import):
            for al in nd.names:
                if "optim" in al.name:
                    bad_imp.append(al.name)
        elif isinstance(nd, ast.ImportFrom):
            if nd.module and "optim" in nd.module:
                bad_imp.append(nd.module)
        elif isinstance(nd, ast.Attribute):
            if nd.attr == "backward":
                bad_attr.append("backward")
        elif isinstance(nd, ast.Name):
            if nd.id in ("LayerMem", "SGMNStack"):
                bad_name.append(nd.id)
    par_ok = (ALPHA, LAM) == (1.0, 0.05)
    chk("A89-09", not (bad_imp or bad_attr or bad_name) and par_ok and md5 == DEVICE_MD5,
        f"ast: optim={bad_imp} backward={bad_attr} 禁用名={bad_name} 全零={not(bad_imp or bad_attr or bad_name)}; "
        f"参数不变={par_ok}; 装置未写={md5 == DEVICE_MD5}")

    # A89-10 不比对已发布 6dp 值
    # ★ DEF-89-04：字面量必须由片段拼接，不得以完整形式出现在源里（否则扫到自己）
    banned = tuple(a + b for a, b in (("0.866", "324"), ("0.970", "373"), ("0.913", "877"),
                                      ("0.495", "000"), ("0.642", "453")))
    hit = [b for b in banned if b in src]
    chk("A89-10", not hit, f"源中无已发布 6dp 字面量 命中={hit or '无'}")

    print()
    npass = sum(1 for x in _F if x["pass"])
    print(f"Stage A 汇总：{npass}/{len(_F)} PASS")
    return npass == len(_F)


# ════════════════════════════ 正式轮
def formal():
    t0 = time.time()
    print("=" * 78)
    print("正式轮：三约定 × 两粒度，各自独立根查找定位 η*（等 token T=198, θ=0.05）")
    print("=" * 78)
    got, res = {}, {}
    for cv in CONVS:
        got[cv], res[cv] = {}, {}
        for m in (M1, M2):
            e_star, info = find_eta(cv, m)
            if e_star is None:
                print(f"  {cv}/m{m}: 根查找未收敛（F3）: {info}")
                got[cv][m] = None
                continue
            obs = resid_of(cv, e_star, m)
            got[cv][m] = {"eta": e_star, "resid": obs, **info}
            res[cv][m] = obs
            print(f"  {cv:6s} m={m}: η*={e_star:.9e} resid={obs:.12f} "
                  f"iters={info.get('iters')} 宽={info.get('width'):.2e}")

    print()
    print("  比值 η*_2/η*_6（闭式 vs 实测 vs 冻结预测）:")
    ratios = {}
    for cv in CONVS:
        a, b = got[cv].get(M1), got[cv].get(M2)
        cf = eta_star_closed(cv, M1) / eta_star_closed(cv, M2)
        if a and b:
            ratios[cv] = a["eta"] / b["eta"]
            rel = abs(ratios[cv] - cf) / abs(cf)
            print(f"    {cv:6s} 闭式={cf:.9f} 实测={ratios[cv]:.9f} "
                  f"冻结预测={PRED[cv]:.9f} |实测−闭式|/闭式={rel:.2e}")
        else:
            ratios[cv] = None
            print(f"    {cv:6s} 闭式={cf:.9f} 实测=**未计算**(F3)")

    print()
    ok_tok = ratios.get("repeat") is not None and abs(ratios["repeat"] - 1.0) <= B_BOUND
    print(f"  [{'PASS' if ok_tok else '**FAIL**'}] C89-TOK: repeat 比值={ratios.get('repeat')} "
          f"|r−1|≤{B_BOUND:g}")

    ok_seg = True
    det_seg = []
    for cv in ("mean", "sum"):
        if ratios.get(cv) is None:
            ok_seg = False
            det_seg.append(f"{cv}:未计算")
            continue
        cf = eta_star_closed(cv, M1) / eta_star_closed(cv, M2)
        rel = abs(ratios[cv] - cf) / abs(cf)
        ok_seg = ok_seg and (rel <= B_BOUND)
        det_seg.append(f"{cv}:{ratios[cv]:.9f} vs {cf:.9f} rel={rel:.2e}")
    print(f"  [{'PASS' if ok_seg else '**FAIL**'}] C89-SEG: {' | '.join(det_seg)} ≤{B_BOUND:g}")

    allr = [v for v in res.values() for v in v.values()]
    worst_r = max(abs(x - THETA) for x in allr) if allr else float("inf")
    ok_res = len(allr) == 6 and worst_r <= 1e-12
    print(f"  [{'PASS' if ok_res else '**FAIL**'}] C89-RESID: {len(allr)}/6 有读数，"
          f"最大|resid−θ|={worst_r:.2e} ≤1e-12")

    gap_ok, gap_det = True, []
    for cv in ("mean", "sum"):
        if ratios.get(cv) is None:
            gap_ok = False
            continue
        g = abs(ratios[cv] - ratios["repeat"])
        gap_det.append(f"{cv}:|{ratios[cv]:.6f}−{ratios['repeat']:.6f}|={g:.6f}")
        gap_ok = gap_ok and (g >= GAP_MIN)
    print(f"  [{'PASS' if gap_ok else '**FAIL**'}] C89-GAP: {' | '.join(gap_det)} ≥{GAP_MIN:g}")

    os.makedirs(OUT, exist_ok=True)
    payload = {"card": "rev 1.1", "T": T_BUDGET, "theta": THETA, "B": B_BOUND,
               "pred": PRED, "eta_star": {c: {str(m): got[c].get(m) for m in (M1, M2)} for c in CONVS},
               "ratio": ratios,
               "crit": {"C89-TOK": ok_tok, "C89-SEG": ok_seg, "C89-RESID": ok_res, "C89-GAP": gap_ok},
               "wall_s": round(time.time() - t0, 2)}
    with open(os.path.join(OUT, "formal89.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\n★ 写出 {OUT}/formal89.json  ({payload['wall_s']}s)")
    return ok_tok and ok_seg and ok_res and gap_ok


def smoke():
    """只证明协议可执行；不产出任何判据裁决。"""
    print("=" * 78)
    print("SMOKE：仅验通路（不产出判据）")
    print("=" * 78)
    ok = stage_a()
    if not ok:
        print("SMOKE：BAD（Stage A 未过）")
        return False
    u, u0m, us = run_conv("repeat", 0.1, 2, 4.0)
    print(f"SMOKE：OK u0={u0m:.9f} u*={us:.9f} u_T={u:.9f} 闭式残差={closed_resid('repeat',0.1,2,4.0):.9f}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageA", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--formal", action="store_true")
    a = ap.parse_args()
    if a.stageA:
        sys.exit(0 if stage_a() else 1)
    elif a.smoke:
        sys.exit(0 if smoke() else 1)
    elif a.formal:
        sys.exit(0 if formal() else 1)
    else:
        ap.print_help()
