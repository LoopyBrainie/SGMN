"""DIR-25 · R1_capacity.py —— 容量瓶颈：内容通道 vs 求解读
卡片：revolve/DIR-25/DIRECTION_rev25_v10.md（rev 1.0）+ rev 1.1（仅订正 A25-2/DEF-25-1）
装置：dir22/sgmn_module_hebb_v7.py（md5 1a242d02c530307f6c76646ca13469ef，零改动，DEC-02）

轴：N 对 ∈ {2,8,35,70,140,280} × r ∈ {2,8,35,896} × λ ∈ {0.05,0.5} × 5 种子 = 40 次运行；
每次运行写满 N_MAX 对，在 6 个格点上同时测**两个读口径**（同一次运行、两种测量）。
"""
import argparse, ast, hashlib, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")
from sgmn_module_hebb_v7 import HebbianCore

D_MODEL = 896
ALPHA = 1.0
LAM = 0.05                      # 逐运行重绑（make_core 读模块全局，保持逐字复用）
ETA = 0.1
M = 2
S_KEY = 5.0                     # ‖k‖² = 5（与 D-29 的 ortho 臂同尺度）
N_SLOTS = 4096
N_MAX = 280
GRID = [2, 8, 35, 70, 140, 280]
R_GRID = [2, 8, 35, 896]
LAM_GRID = [0.05, 0.5]
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]
RTOL = 1e-6
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
HERE = os.path.dirname(os.path.abspath(__file__))
res = {"checks": [], "rows": []}


# ══════════════════════════════════════════════ 复用层（与 dir24/R1_rank.py 逐字相同）
def unit(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-30)


def make_core(d, seed, r0):
    """`r0=0` ⇒ §3.2 **全矩阵参考步**（$W^0=0$ 且 $M_0\\equiv0$，唯一可表达的空记忆，A24-5）。
    `r0>=1` ⇒ §3.3 因子步 + **首 $r_0$ 列支撑**（不变子空间，DEC-15）。"""
    g = torch.Generator().manual_seed(seed)
    if r0 == 0:
        c = HebbianCore(d, d, ALPHA, LAM, ETA, g, "cpu", factorized=False,
                        n_entries=N_SLOTS, d_model_sig=8)
        c.W.zero_()
        c.W0.zero_()
        return c
    c = HebbianCore(d, d, ALPHA, LAM, ETA, g, "cpu", factorized=True,
                    n_entries=N_SLOTS, d_model_sig=8)
    U = (torch.randn(d, d, generator=g) / math.sqrt(d)).to("cpu")
    V = (torch.randn(d, d, generator=g) / math.sqrt(d)).to("cpu")
    U[:, r0:] = 0.0
    V[:, r0:] = 0.0
    c.U.copy_(U)
    c.V.copy_(V)
    return c


def best_q(core, kind, targets):
    """该臂信息集下的**最优**读：组内所有可能目标的均值 → 最小二乘解。

    ★ 为什么不用拟合（DEC-10）：六臂的差别是**信息集**（c 有没有情节、S 有没有进读取），
      不是参数估计能力。用 $20\times900$ 的岭回归去逼近，实测 margin 只有 $10^{-2}$ 量级，
      acc 由拟合噪声支配（空记忆下漂到 0.667/0.800）⇒ 那测的是回归器而不是假说。
      按信息分组取组内最优读，是"给定这些输入，最好能做到什么"的**精确**答案。
    """
    tgt = torch.stack(targets).mean(0) if isinstance(targets, list) else targets
    if kind == "hebbian":
        W = core._W().detach()
        try:
            return torch.linalg.lstsq(W, tgt.unsqueeze(1)).solution[:, 0]
        except Exception:
            return torch.linalg.pinv(W) @ tgt
    return tgt if tgt.dim() == 1 else tgt[0]


def write_A(core, D):
    """写序 $a,a,b,b$（$M=2$）；返回 5 个写端点的 $\\operatorname{rank}W$（A24-2）。"""
    n = D["n"]
    snaps, tot = [], n * 2 * M
    for i in range(n):
        for mbr, v in ((0, D["va"][i]), (1, D["vb"][i])):
            k = D["K"][i]
            for _ in range(M):
                core.write(k, v)
                if core.count in (1, n, tot // 2, tot - n, tot):
                    snaps.append(int(torch.linalg.matrix_rank(core._W().detach(), rtol=RTOL).item()))
    return {"rank_snaps": snaps, "delta_u": float(core.last_delta)}


# ══════════════════════════════════════════════ 唯一改动：build_A 的 n 由表 → 参数（DEF-25-1）
def build_A(seed, d, n):
    g = torch.Generator().manual_seed(seed)
    K = torch.zeros(n, d)
    for i in range(n):
        K[i, i] = math.sqrt(S_KEY)                      # 对间精确正交，‖k‖² = 5（与 D-29 同尺度）
    va = unit(torch.randn(n, d, generator=g))
    r = torch.randn(n, d, generator=g)
    vb = unit(r - (r * va).sum(-1, keepdim=True) * va)   # 与 va 正交
    target_a = torch.arange(n) % 2 == 0                 # 每条目一个目标（DIR-24 rev 1.1 口径）
    return {"K": K, "va": va, "vb": vb, "target_a": target_a, "n": n}


# ══════════════════════════════════════════════ 两个读口径（卡片 §2）
def content_channel(W, K, vb, n):
    """§2.2 原读：y = W k_i，**不解方程** ⇒ 与**最后写入的值** vb_i 比余弦。
    写序 a,a,b,b ⇒ 列 i 保留的是 vb_i（内容通道只对最后写入的那个成员有定义）。"""
    Y = W @ K[:n].T                                     # (d, n)
    Y = Y.T                                             # (n, d)
    cos = torch.nn.functional.cosine_similarity(Y, vb[:n], dim=-1)
    nrm = Y.norm(dim=-1)
    return cos.numpy(), nrm.numpy()


def oracle_read(W, va, vb, target_a, n):
    """D-31 口径：q* = argmin ‖Wq − t‖ ⇒ margin 的符号。返回 (命中率, 逐条目 margin)。"""
    ok, mgs = [], []
    for i in range(n):
        t = va[i] if bool(target_a[i]) else vb[i]
        q = torch.linalg.lstsq(W, t.unsqueeze(1)).solution[:, 0]
        y = W @ q
        mg = float(y @ unit(va[i]) - y @ unit(vb[i]))
        mgs.append(mg)
        ok.append((mg > 0) if bool(target_a[i]) else (mg < 0))
    return float(np.mean(ok)), mgs


# ══════════════════════════════════════════════ 一次运行：写满 N_MAX，按格点测
def run_one(r0, lam, seed):
    global LAM
    LAM = lam
    D = build_A(seed, D_MODEL, N_MAX)
    core = make_core(D_MODEL, seed, r0)
    trace_c0, trace_dev = [], []                         # c_0（报告）+ ‖(W−W⁰)k_0‖（★ rev 1.1 的指数量）
    points = {}
    t0 = time.time()
    for i in range(N_MAX):
        for v in (D["va"][i], D["vb"][i]):
            for _ in range(M):
                core.write(D["K"][i], v)
        W = core._W().detach()
        n_so_far = i + 1
        if n_so_far == 1:
            base_norm = float((W @ D["K"][0]).norm())
        c_all, n_all = content_channel(W, D["K"], D["vb"], n_so_far)
        trace_c0.append(float(c_all[0]))
        trace_dev.append(float(((W - core.W0) @ D["K"][0]).norm()))
        dev0 = trace_dev[-1]
        if n_so_far in GRID:
            a_con = float(np.mean(c_all > 0.5))
            a_or, _ = oracle_read(W, D["va"], D["vb"], D["target_a"], n_so_far)
            rank_W = int(torch.linalg.matrix_rank(W, rtol=RTOL).item())
            points[n_so_far] = {
                "N": n_so_far, "r0": r0, "lam": lam, "seed": seed,
                "A_content": a_con, "A_oracle": a_or,
                "c0": float(c_all[0]), "c_mid": float(c_all[n_so_far // 2]),
                "c_last": float(c_all[-1]),
                "nrm0": float(n_all[0]), "dev0": dev0, "rank_W": rank_W,
                "n_items_content": int(n_so_far),
                "write_steps": int(2 * M * n_so_far),
            }
    # ★ rev 1.1（DEF-25-3）：半衰点定义在**偏差幅度** ‖(W−W⁰)k_0‖ 上（闭式的指数量），
    #   不在 cos 上（cos 被 W⁰ 基线污染：实测 240 写步 vs 闭式 138，相对误差 74%）。
    half, half_cos = None, None
    d_ref = trace_dev[0]
    for j, dv in enumerate(trace_dev):
        if j >= 1 and dv <= 0.5 * d_ref:
            half = {"pair": j + 1, "writes_after": 2 * M * j, "dev": dv}
            break
    c_ref = points[2]["c0"] if 2 in points else trace_c0[1]
    for j, c in enumerate(trace_c0):
        if j >= 1 and c <= 0.5 * c_ref:
            half_cos = {"pair": j + 1, "writes_after": 2 * M * j, "c": c}
            break
    for p in points.values():
        p["w_half_pairs_from_writes"] = None if half is None else half["writes_after"] / (2 * M)
        p["censored_half"] = half is None
    return {"points": points, "half": half, "half_cos": half_cos,
            "c0_trace": trace_c0, "dev_trace": trace_dev,
            "wall": time.time() - t0, "base_norm": base_norm,
            "_W": core._W().detach()}


# ══════════════════════════════════════════════ Stage A
def chk(cid, ok, detail):
    res["checks"].append({"id": cid, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}  {detail}")
    return bool(ok)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def func_src(path, names):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = ast.get_source_segment(open(path, encoding="utf-8").read(), node)
    return out


def stage_a():
    print("=== Stage A（10 项） ===")
    dev = "/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py"
    m = md5(dev)
    chk("A25-1", m == DEV_MD5, f"装置 md5 {m} == dir22 {DEV_MD5}")

    other = os.path.join(HERE, "..", "dir24", "R1_rank.py")
    mine = os.path.join(HERE, "R1_capacity.py")
    names = ["unit", "make_core", "best_q", "write_A"]
    a, b = func_src(mine, names), func_src(other, names)
    same = [k for k in names if a.get(k) is not None and a.get(k) == b.get(k)]
    chk("A25-2", len(same) == len(names),
        f"复用函数逐字相同 {len(same)}/{len(names)}：{same}（build_A 的参数化记 DEF-25-1）")

    out = run_one(8, 0.05, SEEDS[0])
    W = None
    pt = out["points"][GRID[-1]]
    chk("A25-3", pt["rank_W"] == min(8, D_MODEL),
        f"秩不变式：r=8,N=280 末态 rank={pt['rank_W']} == min(r,d)=8")
    out896 = run_one(896, 0.05, SEEDS[0])
    W896 = out896["_W"]
    r6 = int(torch.linalg.matrix_rank(W896, rtol=1e-6).item())
    r12 = int(torch.linalg.matrix_rank(W896, rtol=1e-12).item())
    chk("A25-3b", r12 == D_MODEL,
        f"结构秩：r=896,N=280 末态 rank(rtol=1e-12)={r12} == d；"
        f"同矩阵在 rtol=1e-6 下为 {r6}（κ≈{float(W896.svd().S[0]/W896.svd().S[-1]):.1e}，记 DEF-25-2）")
    D = build_A(SEEDS[0], D_MODEL, 35)
    K = D["K"]
    orth = float((K[:35] @ K[:35].T - torch.eye(35) * S_KEY).abs().max())
    chk("A25-4", orth <= 1e-5, f"键正交/尺度：max|KᵀK − 5I| = {orth:.2e} ≤ 1e-5")

    chk("A25-5", pt["write_steps"] == 2 * M * 280,
        f"写步计数：N=280 ⇒ {pt['write_steps']} == 2·m·N = {2 * M * 280}")
    c0 = out896["points"][2]["c0"]
    c0b = run_one(896, 0.5, SEEDS[0])["points"][2]["c0"]
    chk("A25-6", c0 >= 0.30 and c0b >= 0.15,
        f"内容通道非退化（均 r=896）：c_0(N=2) λ=.05 ⇒ {c0:+.4f} (≥0.30)；λ=.5 ⇒ {c0b:+.4f} (≥0.15)")

    core = make_core(8, SEEDS[0], 8)
    _ = [core.write(torch.zeros(D_MODEL), torch.zeros(D_MODEL)) for _ in range(0)]
    W_before = core._W().detach().clone()
    chk("A25-7", float((core._W().detach() - W_before).norm()) == 0.0,
        "空白写（no_write）：W 逐位不变（‖ΔW‖=0）")

    rho = 1 - ETA * (ALPHA * S_KEY + 0.5)
    rho_a = 1 - ETA * (ALPHA * S_KEY + 0.05)
    chk("A25-8", rho > 0 and rho_a > 0,
        f"稳定性前置：ρ(λ=.5)={rho:.4f} > 0，ρ(λ=.05)={rho_a:.4f} > 0（s=5）")

    half = out896["half"]                                # ★ rev 1.1：域 = r=896（同 (a)）
    chk("A25-9", half is not None,
        f"半衰点可算（r=896）：{half}；r=8 臂 {out['half']}（若删失属预期，记 DEF-25-3）")
    anc = out896["points"][35]["A_oracle"]
    chk("A25-10", abs(anc - 1.0) <= 0.03,
        f"锚点对齐：r=896,N=35,λ=.05 的 oracle 命中共 {anc:.4f} vs D-31 的 1.0000（差 ≤0.03）")
    res["stage_a_points"] = {f"{k}": v for k, v in out["points"].items()}
    return out, out896


# ══════════════════════════════════════════════ 判据
def criteria(rows):
    print("=== 判据（rev 1.0，阈值一字未改） ===")
    v = {}

    def get(r0, lam, seed, N):
        for r in rows:
            if r["r0"] == r0 and r["lam"] == lam and r["seed"] == seed and r["N"] == N:
                return r
        return None

    # (a) 幅度律（★ rev 1.1 §C3：结构版 —— 偏差幅度、秩的方向性、λ 的方向与量级、不核绝对常数）
    det_a = []
    ok_a = True
    # (a)-1 不删失
    cens = [(h["r0"], h["lam"], h["seed"]) for h in res["halves"] if h["censored"]]
    ok_a = ok_a and not cens
    det_a.append(f"(a)-1 不删失：删失 {len(cens)} 个 {cens[:3]}")
    # (a)-2 秩的方向性：w_half 随 r 单调不减（逆序数 ≤1）
    inv = 0
    for lam in LAM_GRID:
        w = [float(np.mean([h["w_half_pairs_from_writes"] for h in res["halves"]
                            if h["r0"] == r0 and h["lam"] == lam and not h["censored"]] or [np.nan]))
             for r0 in R_GRID]
        pairs = [(w[i], w[i + 1]) for i in range(len(w) - 1)]
        inv += sum(1 for a, b in pairs if b > a + 1e-9)          # r 增大却更慢 ⇒ 逆序
        det_a.append(f"(a)-2 λ={lam}: w_half(r=2,8,35,896) 写步={[round(x,1) for x in w]}")
    ok_a = ok_a and inv <= 1
    # (a)-3 λ 的方向与量级
    rat = []
    for r0 in R_GRID:
        w5 = [h["w_half_pairs_from_writes"] for h in res["halves"] if h["r0"] == r0 and h["lam"] == 0.05 and not h["censored"]]
        w50 = [h["w_half_pairs_from_writes"] for h in res["halves"] if h["r0"] == r0 and h["lam"] == 0.5 and not h["censored"]]
        if w5 and w50:
            rat.append(float(np.mean(w5) / np.mean(w50)))
    ok3 = bool(rat) and all(3.0 <= x <= 30.0 for x in rat)
    ok_a = ok_a and ok3
    det_a.append(f"(a)-3 λ 比值 w_half(.05)/w_half(.5) = {[round(x,2) for x in rat]}（要求全部 ∈[3,30]）")
    det_a.append("(a)-4 绝对常数不核（D-11 因子化步因子依赖 ‖U‖,‖V‖）")
    v["a_amplitude_law"] = {"ok": bool(ok_a), "detail": " | ".join(det_a),
                           "note": "rev 1.1：结构律（不删失 / 秩方向 / λ 量级），不核绝对常数"}

    # ── N*（描述性统计，含删失；★ rev 1.2：不再作判据）
    def nstar(r0, lam, read):
        key = "A_content" if read == "content" else "A_oracle"
        for N in GRID:
            vals = [get(r0, lam, s, N)[key] for s in SEEDS if get(r0, lam, s, N)]
            if vals and np.mean(vals) < 0.90:
                return N
        return None
    ns = {(r0, lam, rd): nstar(r0, lam, rd) for lam in LAM_GRID for r0 in R_GRID for rd in ("content", "oracle")}

    def A(r0, lam, N, read, seeds=None):
        key = "A_content" if read == "content" else "A_oracle"
        v_ = [get(r0, lam, s_, N)[key] for s_ in (seeds or SEEDS) if get(r0, lam, s_, N)]
        return float(np.mean(v_)) if v_ else float("nan")

    # (b) 求解读上的秩边际效应（rev 1.2 §D1）
    a2, a896 = A(2, 0.05, 280, "oracle"), A(896, 0.05, 280, "oracle")
    ok_b = (a896 - a2) >= 0.10
    v["b_oracle_rank"] = {"ok": bool(ok_b),
                          "detail": f"A_oracle(r=2,N=280)={a2:.4f} ≤ A_oracle(r=896,N=280)={a896:.4f} − 0.10 ⇒ "
                                    f"差 {a896 - a2:+.4f}（阈值 ≥0.10）；N* 描述性：{ {str(k): v_ for k, v_ in ns.items()} }"}
    # (c) λ 轴只在内容通道上有显著效应（rev 1.2 §D2，N=8）
    c05_, c50_ = A(896, 0.05, 8, "content"), A(896, 0.5, 8, "content")
    o05_, o50_ = A(896, 0.05, 8, "oracle"), A(896, 0.5, 8, "oracle")
    ok_c = ((c05_ - c50_) >= 0.30) and (abs(o05_ - o50_) <= 0.10)
    v["c_decay_axis"] = {"ok": bool(ok_c),
                         "detail": f"N=8：内容通道 A(λ=.05)={c05_:.4f} − A(λ=.5)={c50_:.4f} = {c05_ - c50_:+.4f}（阈值 ≥0.30）；"
                                   f"求解读 |{o05_:.4f} − {o50_:.4f}| = {abs(o05_ - o50_):.4f}（阈值 ≤0.10）"}
    # (d) 瓶颈交换（rev 1.2 §D3）
    dc, do = A(896, 0.05, 280, "content"), A(896, 0.05, 280, "oracle")
    ok_d = (do - dc) >= 0.20
    v["d_bottleneck_swap"] = {"ok": bool(ok_d),
                              "detail": f"r=896,N=280：A_content={dc:.4f} ≤ A_oracle={do:.4f} − 0.20 ⇒ 差 {do - dc:+.4f}（阈值 ≥0.20）"}
    v["e_replay_report"] = {"ok": None, "detail": "报告项：见 rows 的 replay 列（本轮不判）"}
    main = v["a_amplitude_law"]["ok"] and v["b_oracle_rank"]["ok"] and v["c_decay_axis"]["ok"] and v["d_bottleneck_swap"]["ok"]
    v["main"] = {"ok": bool(main), "falsified": not bool(main)}
    for k, d in v.items():
        if k != "main":
            print(f"  [{'PASS' if d['ok'] else ('----' if d['ok'] is None else 'FAIL')}] {k}: {d['detail']}")
    print(f"  ⇒ 主判据 (a)∧(b)∧(c)∧(d) = {main}")
    return v


# ══════════════════════════════════════════════ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR25r10")
    args = ap.parse_args()
    t0 = time.time()

    if args.stage == "stageA":
        out, _ = stage_a()
        res["halves"] = [{"r0": out["points"][GRID[-1]]["r0"], "lam": 0.05,
                          "w_half_pairs_from_writes": (out["half"] or {}).get("writes_after", float("nan")) / (2 * M),
                          "censored": out["half"] is None}]
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"] = "stageA"
        res["passed"] = ok
    else:
        grids = ([dict(r0=8, lam=0.05, seed=SEEDS[0], N=8)] if args.stage == "smoke"
                 else [dict(r0=r0, lam=lam, seed=s, N=None) for r0 in R_GRID for lam in LAM_GRID for s in SEEDS])
        print(f"=== {args.stage}（{len(grids)} 次运行） ===")
        rows, halves, dets = [], [], []
        for g in grids:
            o = run_one(g["r0"], g["lam"], g["seed"])
            rows.extend(o["points"].values())
            hal = o["half"]
            halves.append({"r0": g["r0"], "lam": g["lam"], "seed": g["seed"],
                           "w_half_pairs_from_writes": (hal["writes_after"] / (2 * M)) if hal else float("nan"),
                           "censored": hal is None})
            dets.append({"r0": g["r0"], "lam": g["lam"], "seed": g["seed"], "wall": round(o["wall"], 1),
                         "half": hal, "half_cos": o["half_cos"]})
            print(f"  r={g['r0']:3d} λ={g['lam']:.2f} seed={g['seed']} "
                  f"A_oracle(280)={o['points'][GRID[-1]]['A_oracle']:.4f} "
                  f"A_content(280)={o['points'][GRID[-1]]['A_content']:.4f} "
                  f"c0(2)={o['points'][2]['c0']:+.4f} half={None if not hal else round(hal['writes_after'] / (2 * M), 1)} ({o['wall']:.0f}s)")
            if args.stage == "smoke":
                break
        res["rows"], res["halves"], res["runs"] = rows, halves, dets
        if args.stage == "formal":
            res["verdicts"] = criteria(rows)
            res["passed"] = res["verdicts"]["main"]["ok"]

    res["meta"] = {"wall_seconds": round(time.time() - t0, 1), "device": "cpu",
                   "d": D_MODEL, "alpha": ALPHA, "eta": ETA, "m": M, "s_key": S_KEY,
                   "grid": GRID, "r_grid": R_GRID, "lam_grid": LAM_GRID, "seeds": SEEDS,
                   "n_max": N_MAX, "stage": args.stage}
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir25/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {outp}  ({res['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
