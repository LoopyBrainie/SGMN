r"""DIR-26-N · probe26n_feas.py —— $\alpha$ 的**产生规则**是否可行（**开卡前**的可行性 + 功效核算）

议题：v2.4 §7.2 第 523 行「$\alpha$ 的产生规则」。当前 $\alpha$ 的阈值 $\tau$ 是**外部标定**的：
      `R1_gating5.calibrate_w` 在 pilot 种子上取**无门控写流的 $\cos$ 分位数** $\mathrm{quantile}(c,1-\rho)$，
      $\rho=R_{\rm MAIN}=0.40$ ⇒ 跳过率≈0.40。**本探针问三件事**（结果只定卡片，不出裁决）：

① **可执行性**：两条"无外部常数"的规则能不能真的产生门控？
   - 规则 **H**（`hist-median`）：逐条目维护该条目**已发生**的 $\cos$ 历史（同条目、两支、全部轮次，
     含被跳过的那些机会——历史只看观测不看是否写），**跳过 iff** $c_t>$ 历史中位数且历史长度 $\ge2$，
     否则写；观测随后入历史。⇒ 无任何"阈值常数"，只有"中位数"这一分布型选择。
   - 规则 **J**（`objective`）：$J(W)=\tfrac12\|Wk-v\|^2+\tfrac\lambda2\|W-W^0\|^2$，
     **跳过 iff** $\Delta J=J(W')-J(W)\ge0$，其中 $W'$ 是**装置自己的**同步因子步
     $U'=U-\eta GV,\;V'=V-\eta G^\top U$（用旧 $U,V$）⇒ $\alpha$ 由局部目标的**符号**产生。
② **退化风险**（决定卡片怎么冻结）：两条规则的跳过率、以及规则 **J** 的 $\Delta J<0$ 占比；
   若规则在 $m$ 两档上都"全写"或"全跳"，该规则在本装置上**不可判**（PREMISE-ABSENT），
   必须在**开卡前**知道，而不是开跑后才发现判据不可达（DEC-37/D-44 §1.1-4 的教训）。
③ **功效核算**：用 D-45 的 30 种子表算 $3\mathrm{se}$：判"不掉基线"和"保留冗余免疫"各需要多大 $\Delta$。

★ 本探针**不出任何裁决**（smoke 级用途），只落 `counts/issued/ret1/μ/门控率/一致性` 与一张功效表。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

for _p in ("dir26n", "dir26m", "dir26l", "dir26k", "dir26j", "dir26i", "dir26h", "dir26g",
           "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating12 as K
import R1_gating14 as M14
from R1_gating import D_MODEL, M, drive, make_core
from R1_gating5 import theta_of
from R1_gating12 import (R_PRES, R_HO4, R_HO8, cal_at, measure_prec, read_stream)
from rerun_missing import measure_hit_bounded

N_READ = 20
N_KV = 60
D45 = "/home/ec2-user/sgmn-loop/dir26m/out/DIR26Mr10_formal.json"
OUT = "/home/ec2-user/sgmn-loop/dir26n/out"


# ══════════════════════════════════════════════ 两条"产生规则"
def _J(core, Up, Vp, k, v):
    r"""$J$ 的**精确**值：数据项用 $W'k=W^0k+U'(V'^\top k)$，正则项用 $\|U'^\top V'\|_F^2=\|U'V'^\top\|_F^2$。"""
    rv = (core.W0 @ k + Up @ (Vp.T @ k)) - v
    # ★ DEF-26N-1（探针自查抓到的错误恒等式）：$\|U'V'^\top\|_F^2\ne\|U'^\top V'\|_F^2$；
    #   正确的便宜式是 $\|AB^\top\|_F^2=\mathrm{tr}((A^\top A)(B^\top B))$（两个 $r\times r$）。
    reg = float(((Up.T @ Up) * (Vp.T @ Vp)).sum())
    return 0.5 * float(rv @ rv) + 0.5 * core.lam * reg


def dj_exact(core, k, v):
    r"""$\Delta J$（装置**同步因子步**下的精确差值，不近似、不假定 $W'\approx W-\eta G$）。"""
    W = core._W()
    Gm = core.alpha * torch.outer(W @ k - v, k) + core.lam * (W - core.W0)
    Up = core.U - core.eta * (Gm @ core.V)
    Vp = core.V - core.eta * (Gm.T @ core.U)
    j0 = _J(core, core.U, core.V, k, v)
    j1 = _J(core, Up, Vp, k, v)
    return j1 - j0, j0, j1


def drive_rule(core, D, mode, tau=None, hist_min=2):
    r"""**逐字复制** `R1_gating.drive` 的控制流，只把 `arm=="B"` 的门控换成可插拔规则。

    mode `A`：固定 $\tau$（= 原 `drive(..., "B", tau=tau)`，Stage A 须逐位相等）；
    mode `H`：历史中位数规则；mode `J`：局部目标下降规则。
    返回 `issued, log`（`log` 与 `R1_gating.drive` 同形，另加 `dj`/`hist_n`/`thr` 供诊断）。
    """
    n, R = D["n"], D["r_pres"]
    issued, log = 0, []
    hist = [[] for _ in range(int(n))]          # 逐条目 cos 历史（规则 H）
    rlast = [None] * int(n)                     # 逐条目"上次写后"的残差范数（规则 R）
    wcos = []                                   # 已写机会的 cos（规则 W：写流自洽均值）
    dj_pos = dj_n = 0
    for rho in range(R):
        for i in range(n):
            k = D["K"][rho, i]
            for mbr, v in ((0, D["va"][i]), (1, D["vb"][i])):
                for m in range(M):
                    t_idx = G.stream_idx(rho, i, mbr, m, n)
                    c_now = float(torch.nn.functional.cosine_similarity(
                        (core._W() @ k).unsqueeze(0), v.unsqueeze(0)).item())
                    dj = float("nan")
                    if mode == "A":
                        go = not (tau is not None and c_now > tau)
                    elif mode == "H":
                        h = hist[i]
                        thr = float(np.median(h)) if len(h) >= hist_min else None
                        go = not (thr is not None and c_now > thr)
                    elif mode == "W":
                        thr = float(np.mean(wcos)) if len(wcos) >= 4 else None
                        go = not (thr is not None and c_now > thr)
                    elif mode == "R":
                        rn = float((core._W() @ k - v).norm())
                        thr = rlast[i]
                        go = not (thr is not None and rn < thr)
                    elif mode == "J":
                        dj, _j0, _j1 = dj_exact(core, k, v)
                        dj_n += 1
                        dj_pos += int(dj < 0)
                        go = not (dj >= 0.0)          # 只在目标严格下降时写
                    else:
                        raise ValueError(mode)
                    log.append({"round": rho, "item": i, "cos": c_now, "issued": bool(go),
                                "t": t_idx, "dj": dj, "hist_n": len(hist[i])})
                    hist[i].append(c_now)
                    if go:
                        core.write(k, v)
                        issued += 1
                        if mode == "W":
                            wcos.append(c_now)
                        if mode == "R":
                            rlast[i] = float((core._W() @ k - v).norm())
    return issued, log, {"dj_neg": dj_pos, "dj_n": dj_n}


# ══════════════════════════════════════════════ 工作例（测量对象自检）
def worked_example(core_factory, D, n_chk=6):
    r"""`dj_exact` 必须与**显式构造** $W'=W^0+U'V'^\top$ 的暴力 $J$ 差一致（相对误差 $\le10^{-9}$）。"""
    core = core_factory()
    rows, worst = [], 0.0
    for rho in range(1):
        for i in range(min(n_chk, int(D["n"]))):
            k = D["K"][rho, i]
            for mbr, v in ((0, D["va"][i]), (1, D["vb"][i])):
                dj, j0, j1 = dj_exact(core, k, v)
                W = core._W()
                Gm = core.alpha * torch.outer(W @ k - v, k) + core.lam * (W - core.W0)
                Up = core.U - core.eta * (Gm @ core.V)
                Vp = core.V - core.eta * (Gm.T @ core.U)
                Wp = core.W0 + Up @ Vp.T
                brute0 = 0.5 * float(((W @ k - v) ** 2).sum()) + 0.5 * core.lam * float(((W - core.W0) ** 2).sum())
                brute1 = 0.5 * float(((Wp @ k - v) ** 2).sum()) + 0.5 * core.lam * float(((Wp - core.W0) ** 2).sum())
                rel = abs((j1 - j0) - (brute1 - brute0)) / max(abs(brute1 - brute0), 1e-12)
                worst = max(worst, rel)
                rows.append({"i": i, "mbr": mbr, "dj": dj, "dj_brute": brute1 - brute0, "rel": rel})
                core.write(k, v)                      # 让状态动起来（不是零状态自检）
    return {"rows": rows, "worst_rel": worst}


# ══════════════════════════════════════════════ 单元
def probe_unit(seed, b, tau, n_kv=N_KV):
    D = M14.build_stream_b(b, seed, D_MODEL, n_kv, R_PRES, theta_of(0.9))
    assert int(D["n"]) == n_kv, f"写流 N 不符：{D['n']} != {n_kv}"
    RS8, RS4 = read_stream(seed, R_HO8), read_stream(seed, R_HO4)
    vb = D["vb"][:N_READ]
    out = {"seed": seed, "b": int(b), "m": N_READ + int(b), "n_kv": n_kv,
           "T": R_PRES * n_kv * 2 * M, "arms": {}}

    def rec(tag, core, issued, gate):
        W = core._W().detach()
        P = W - core.W0
        m8 = measure_hit_bounded(P, RS8["K"], vb, N_READ, R_HO8)
        p4 = measure_prec(P, RS4["K"], vb, N_READ, R_HO4)
        out["arms"][tag] = {"issued": int(issued), "ret1": m8["ret1"], "mu": m8["mu"],
                            "clean": float(G.metrics_plas(W, core.W0, D["proto"][:N_READ], vb, N_READ)["ret1"]),
                            "ho4": p4["ho_mean"], **gate}

    def gate_stats(log, issued):
        tot = len(log)
        sk = [e for e in log if not e["issued"]]
        return {"gate_rate": len(sk) / tot,
                "skip_frac_repeat": (float(np.mean([e["round"] > 0 for e in sk])) if sk else float("nan")),
                "qbar": issued / n_kv}

    c = make_core(D_MODEL, seed, K.R0); kA, _ = drive(c, D, "A"); rec("A0", c, kA, {})
    gated = {}
    for mode in ("A", "H", "J", "W", "R"):
        c = make_core(D_MODEL, seed, K.R0)
        kg, log, diag = drive_rule(c, D, mode, tau=tau)
        st = gate_stats(log, kg)
        if diag["dj_n"]:
            st["dj_pos_frac"] = diag["dj_neg"] / diag["dj_n"]      # ΔJ<0 的占比（"该写"的比例）
        rec(mode, c, kg, st)
        gated[mode] = log
    # 决策一致性：规则 vs 固定 τ（逐机会比对）
    for mode in ("H", "J", "W", "R"):
        a, g = gated["A"], gated[mode]
        agree = float(np.mean([(e1["issued"] == e2["issued"]) for e1, e2 in zip(a, g)]))
        both = float(np.mean([(e1["issued"] and e2["issued"]) for e1, e2 in zip(a, g)]))
        neither = float(np.mean([(not e1["issued"] and not e2["issued"]) for e1, e2 in zip(a, g)]))
        out["arms"][mode]["agree_vs_tau"] = agree
        out["arms"][mode]["both_write"] = both
        out["arms"][mode]["neither_write"] = neither
    return out


def power_table():
    r"""功效核算：从 D-45 的 30 种子表直接算 $3\mathrm{se}$（判"不掉基线"/"保留免疫"的可达性）。"""
    d = json.load(open(D45, encoding="utf-8"))
    rows = d["rows"]
    tab = {}
    for b in (40, 4):
        sub = [r for r in rows if r["b"] == b]
        cell = {}
        for arm in ("A", "B"):
            v = np.array([r["arms"][arm]["ret1"] for r in sub], float)
            cell[arm] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                         "3se": float(K.SE_K * v.std(ddof=1) / np.sqrt(len(v)))}
        dif = np.array([r["arms"]["B"]["ret1"] - r["arms"]["A"]["ret1"] for r in sub], float)
        cell["B-A"] = {"mean": float(dif.mean()), "sd": float(dif.std(ddof=1)),
                       "3se": float(K.SE_K * dif.std(ddof=1) / np.sqrt(len(dif)))}
        tab[str(b)] = cell
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(K.SEEDS[:3]))
    ap.add_argument("--bs", type=int, nargs="+", default=[40, 4])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {"checks": [], "rows": [], "power": power_table(), "calib": {}}
    tau = cal_at(N_KV)["c_tau"]
    res["calib"] = {"n_kv": N_KV, "c_tau": tau, "rho_main": K.R_MAIN}
    print(f"=== probe26n（tau={tau:.4f}，b∈{args.bs}，种子 {args.seeds}）===")

    # 工作例：ΔJ 的精确式 vs 暴力构造
    D0 = M14.build_stream_b(40, args.seeds[0], D_MODEL, N_KV, R_PRES, theta_of(0.9))
    we = worked_example(lambda: make_core(D_MODEL, args.seeds[0], K.R0), D0)
    res["worked"] = {"worst_rel": we["worst_rel"], "n": len(we["rows"]),
                     "sample": we["rows"][:4]}
    ok = we["worst_rel"] <= 1e-9
    res["checks"].append({"id": "P26N-1", "ok": bool(ok),
                          "msg": f"ΔJ 精确式 vs 暴力 $W'=W^0+U'V'^\\top$：最坏相对误差 {we['worst_rel']:.2e}（≤1e-9）"})
    print(f"  [P26N-1] ΔJ 一致性：最坏相对误差 {we['worst_rel']:.2e} {'OK' if ok else '★FAIL'}")

    for b in args.bs:
        for s in args.seeds:
            r = probe_unit(s, b, tau)
            res["rows"].append(r)
            a = r["arms"]
            print(f"  b={b:2d}(m={r['m']:2d}) seed={s:3d} | A0 ret1 {a['A0']['ret1']:.4f} iss {a['A0']['issued']:4d} | "
                  + " | ".join(f"{tg} ret1 {a[tg]['ret1']:.4f} iss {a[tg]['issued']:4d} "
                               f"gate {a[tg]['gate_rate']:.3f} q̄ {a[tg]['qbar']:.2f} "
                               + (f"ΔJ<0 {a[tg]['dj_pos_frac']:.3f} " if 'dj_pos_frac' in a[tg] else "")
                               + (f"agree {a[tg]['agree_vs_tau']:.3f}" if 'agree_vs_tau' in a[tg] else "")
                               for tg in ("A", "H", "J", "W", "R")))
    res["passed"] = all(c["ok"] for c in res["checks"])
    p = os.path.join(OUT, "probe26n_feas.json")
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  → {p}  passed={res['passed']}")
    print("  功效（D-45 的 30 种子）：" + "; ".join(
        f"m={20+int(b)}: ret1(A) {c['A']['mean']:.4f} 3se {c['A']['3se']:.4f} / ret1(B) {c['B']['mean']:.4f} "
        f"3se {c['B']['3se']:.4f} / 配对 B−A {c['B-A']['mean']:+.4f} 3se {c['B-A']['3se']:.4f}"
        for b, c in res["power"].items()))


if __name__ == "__main__":
    main()
