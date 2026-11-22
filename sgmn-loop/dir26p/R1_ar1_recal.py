r"""DIR-26-P · R1_ar1_recal.py —— 阈值跨输入过程的**重新标定**（卡片 rev 1.1 冻结）

卡片：`revolve/DIR-26P/DIRECTION_rev26p_v15.md`（rev 1.0 → 1.1 增补 A26P-12 → 1.2 订正 A26P-2 期望 md5 → 1.3 留档 DEF-26P-4/5，**阈值未动**）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，**零改动**）
派生自：`dir26o/R1_ar1.py`（DIR-26-O 的**正式轮版本**，md5 0bab194adf0f6fc9c4cc0845b77af3da；
DEF-26O-8/9/10 的后期修订**不改主轮数据**）

唯一改动（相对 `R1_ar1.py`）——按纪律「只改测量层与独立变量」：
  ① **独立变量**：把 $\tau$ 从「iid 标定的 $\tau_{\rm iid}$」换成「**AR(1) pilot 标定的 $\tau_{\rm AR}$**」
     （同一 `SEED_PILOT`、同一跳过率 $\rho=0.40$、同一分位算法；$\varphi=0$ 必须**逐位**等于
     `cal_at(60)["c_tau"]` ⇒ 复刻正确，Stage A A26P-4 复核）；
  ② **测量层**：新增 `gain = ret1(tau) - ret1(E0)`、`gain_recov = gain(tau_AR) - gain(tau_iid)`
     （**同种子配对**）、跨种子分位数（p50/p90/p99）、各 $\varphi$ 的 $\tau_{\rm AR}$ 与决策一致率（报告）。
  **不改**：AR(1) 输入（`RA.build_stream_ar1` 逐字沿用）、写流顺序、读流、口径 O / 控制口径 X、判据算法、种子表、装置。

★ **成本设计（卡片 §2.3）**：`E0` 臂**复用** DIR-26-O 正式轮的归档读数（位级锚点保证同一写流），
本地每单元只跑 **2 条新驱动**（`tau_iid` / `tau_AR`）⇒ 240 条新驱动 ≈ 3.0 h。
★ **配对性**：`tau_iid` 与 DIR-26-O 的 `tau` 臂**逐位相等**（同种子、同流、同一 $\tau$）⇒ 配对差只来自 $\tau$。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

for _p in ("dir26p", "dir26o", "dir26n", "dir26m", "dir26l", "dir26k", "dir26j", "dir26i", "dir26h",
           "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_ar1 as RA
import R1_gating as G
import R1_gating12 as K
import R1_gating14 as M14
import probe26n_feas as PR
from R1_gating import D_MODEL, SEED_PILOT, drive, make_core
from R1_gating5 import theta_of
from R1_gating12 import R_HO4, R_HO8, cal_at, measure_prec, read_stream
from R1_gating7 import md5
from rerun_missing import measure_hit_bounded

N_KV = RA.N_KV
N_READ = RA.N_READ
PHIS = RA.PHIS
SEEDS = RA.SEEDS
RHO = 0.40
REC_THRESH = 0.050
SIGN_MIN = 21
CAL_OK_GAP = 0.05
CAL_OK_REP = 0.85
T_UNIT = RA.T_UNIT
D26O = "/home/ec2-user/sgmn-loop/dir26o/out/DIR26Or11_formal.json"
D26O_B4 = "/home/ec2-user/sgmn-loop/dir26o/out/DIR26Or11_formal_b4.json"
OUT = "/home/ec2-user/sgmn-loop/dir26p/out"


# ══════════════════════════════════════════════ 归档读取（E0 复用 + 锚点）
def load_archive():
    r"""DIR-26-O 正式轮的归档行，按**世界**索引：$(n\_kv, seed, world) \to$ 行。

    ★ DEF-26O-9/10：DIR-26-O 主轮的行标签写成 `b=60`，而**恒等世界**在归档里记作 `b=40`
      （`build_stream_ar1(60,...)` 与 `build_stream_ar1(40,...)` 写流**逐位相同**：折叠条件 `b<40` 均不触发）
      ⇒ 这里按**世界**归一（40 ≡ 60），否则查不到行。
    """
    out = {}
    # ★ DEF-26P-8：Stage C 的 b=4 冗余世界归档在**另一个文件**里（`DIR26Or11_formal_b4.json`），
    #   一并载入才能复用 E0 与做位级锚点。
    files = [D26O] + ([D26O_B4] if os.path.exists(D26O_B4) else [])
    for fp in files:
        for r in json.load(open(fp))["rows"]:
            b = int(r["b"])
            world = 40 if b in (40, N_KV) else b
            # ★ DEF-26P-5：**索引必须含 φ** —— 同一 seed 有 4 个 φ 行，漏掉 φ 会互相覆盖
            out[(int(r["n_kv"]), int(r["seed"]), world, float(r["phi"]))] = r
    return out


ARCH = load_archive()


# ══════════════════════════════════════════════ AR(1) 重新标定（复刻 calibrate_w，只改输入流）
def calibrate_ar(phi):
    r"""与 `R1_gating5.calibrate_w` **同逻辑**（同 `SEED_PILOT`、同分位、同 `gate_ok`），只把输入流换成 AR(1)。

    ★ `phi==0` 必须**逐位**等于 `cal_at(N_KV)["c_tau"]`（Stage A A26P-4 的复核对象）。
    """
    D = RA.build_stream_ar1(40, SEED_PILOT, D_MODEL, N_KV, RA.R_PRES, theta_of(0.9), phi)
    c = make_core(D_MODEL, SEED_PILOT, K.R0)
    _, log = drive(c, D, "A")
    allc = [e["cos"] for e in log]
    tau = float(np.quantile(allc, 1.0 - RHO))
    skip = [e for e in log if e["cos"] > tau]
    kep = [e for e in log if e["cos"] <= tau]
    rate = len(skip) / len(allc)
    m_skip = float(np.median([e["cos"] for e in skip])) if skip else float("nan")
    m_keep = float(np.median([e["cos"] for e in kep])) if kep else float("nan")
    frac_rep = float(np.mean([e["round"] > 0 for e in skip])) if skip else float("nan")
    ok = bool(abs(rate - RHO) <= 0.05 and (m_skip - m_keep) >= CAL_OK_GAP and frac_rep >= CAL_OK_REP)
    return {"c_tau": tau, "skip_rate": rate, "gap": m_skip - m_keep,
            "skip_frac_repeat": frac_rep, "gate_ok": ok, "phi": float(phi)}


# ══════════════════════════════════════════════ 单元（测量层逐字复制 RA.run_unit 的口径）
def run_unit(seed, phi, tau_iid, tau_ar, b=N_KV):
    r"""一条 $(s,\varphi)$ 单元 = **2 条新驱动**：`tau_iid` / `tau_AR`；`E0` 复用归档（位级锚点保证同一写流）。"""
    D = RA.build_stream_ar1(b, seed, D_MODEL, int(N_KV), RA.R_PRES, theta_of(0.9), phi)
    assert int(D["n"]) == N_KV, f"写流 N 不符：{D['n']} != {N_KV}"
    RS8, RS4 = read_stream(seed, R_HO8), read_stream(seed, R_HO4)
    proto, vb = D["proto"][:N_READ], D["vb"][:N_READ]
    arch = ARCH.get((N_KV, int(seed), 40 if b in (40, N_KV) else int(b), float(phi)))
    assert arch is not None, f"归档缺 seed={seed} b={b}"
    out = {"n_kv": N_KV, "n_read": N_READ, "seed": int(seed), "phi": float(phi), "b": int(b),
           "m": N_READ + int(b), "T": T_UNIT, "arms": {}, "diag": D["diag"]}

    def rec(tag, core, issued, extra=None):
        r"""★ 与 `RA.run_unit` 的 `rec` **逐字同口径**（口径 O / 控制口径 X / `P_norm`）。"""
        W = core._W().detach()
        P = W - core.W0
        m8 = measure_hit_bounded(P, RS8["K"], vb, N_READ, R_HO8)
        p4 = measure_prec(P, RS4["K"], vb, N_READ, R_HO4)
        mx = measure_hit_bounded(P, RS8["K"], RS8["vb"], N_READ, R_HO8)
        assert m8["ret1"] == measure_prec(P, RS8["K"], vb, N_READ, R_HO8)["ho_mean"], "μ/精度口径不符"
        mp = G.metrics_plas(W, core.W0, proto, vb, N_READ)
        d = {"issued": int(issued), "wsig": K.wsig(W),
             "W_unchanged": bool(torch.equal(core._W().detach(), W)),
             "clean": mp["ret1"], "P_norm": mp["P_norm"], "cosP": mp["cosP"],
             "mu": m8["mu"], "mu_sd": m8["mu_sd"], "mu_read": m8["n_read"],
             "ret1": m8["ret1"], "ret1_maj": m8["ret1_maj"], "maj_need": m8["maj_need"],
             "ho8_mean": m8["ret1"], "ho8_maj": m8["ret1_maj"],
             "ho4_mean": p4["ho_mean"], "ho4_maj": p4["ho_maj"],
             "mu_ret1_consistent": m8["mu_ret1_consistent"],
             "X": {"mu": mx["mu"], "mu_sd": mx["mu_sd"], "ret1": mx["ret1"]}}
        if extra:
            d.update(extra)
        out["arms"][tag] = d

    def drive_tau(tau, tag):
        c = make_core(D_MODEL, seed, K.R0)
        ki, lg, _dg = PR.drive_rule(c, D, "A", tau=tau)
        sk = [e for e in lg if not e["issued"]]
        rec(tag, c, ki, {"gate_rate": len(sk) / len(lg),
                         "skip_frac_repeat": (float(np.mean([e["round"] > 0 for e in sk])) if sk else float("nan")),
                         "qbar": ki / N_KV, "log_n": len(lg)})
        return lg

    lg_i = drive_tau(tau_iid, "tau_iid")
    lg_a = drive_tau(tau_ar, "tau_AR")
    # E0 复用归档（字段逐字搬运，锚点由 A26P-6/7 守住）
    e0 = arch["arms"]["E0"]
    out["arms"]["E0"] = {k: e0[k] for k in e0}
    out["arms"]["E0"]["reused_from"] = "DIR-26-O formal (b=40)"
    out["arms"]["tau_iid"]["gain"] = out["arms"]["tau_iid"]["ret1"] - out["arms"]["E0"]["ret1"]
    out["arms"]["tau_AR"]["gain"] = out["arms"]["tau_AR"]["ret1"] - out["arms"]["E0"]["ret1"]
    out["arms"]["tau_AR"]["gain_recov"] = out["arms"]["tau_AR"]["gain"] - out["arms"]["tau_iid"]["gain"]
    out["arms"]["tau_AR"]["agree_vs_iid"] = float(np.mean(
        [e1["issued"] == e2["issued"] for e1, e2 in zip(lg_i, lg_a)]))
    return out


# ══════════════════════════════════════════════ 判据骨架
def _pair(vals_a, vals_b):
    a, b = np.asarray(vals_a, float), np.asarray(vals_b, float)
    d = a - b
    n = len(d)
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    return {"mean": float(np.mean(d)), "sd": sd, "se": se, "k": 3.0 * se, "n": n,
            "pos": int((d > 0).sum()), "neg": int((d < 0).sum()),
            "vals": [round(float(x), 6) for x in d]}


def crit(rows, calib):
    r"""判决（卡片 §1.1 / §2.2 冻结）。"""
    seeds = sorted({int(r["seed"]) for r in rows})
    by = {(int(r["seed"]), float(r["phi"])): r for r in rows}
    phis = sorted({float(r["phi"]) for r in rows})
    for s in seeds:
        for p in phis:
            assert (s, p) in by, f"缺单元 seed={s} phi={p}"
    v = {"n_seed": len(seeds), "phis": phis, "seeds": seeds,
         "calib": {str(p): calib[p] for p in calib}}
    # ── P1：标定有效（gate_ok 全 True）
    v["P1"] = {"gate_ok": {str(p): calib[p]["gate_ok"] for p in calib},
               "ok": all(calib[p]["gate_ok"] for p in calib)}
    # ── 锚点：tau_iid ≡ DIR-26-O 的 tau（位级）⇒ E0 复用成立、配对成立
    # ★ DEF-26P-7：锚点按**世界**查（主轮 b=60 ≡ 恒等世界 40；Stage C 为 b=4），
    #   且 φ=0 档不在时（Stage C 只跑 φ=0.9）跳过锚点而**不**误判。
    AF_TAU = ("issued", "ret1", "wsig")
    ncmp, bads = 0, []
    worlds = sorted({int(r["b"]) for r in rows})
    world = 40 if worlds == [40] or worlds == [N_KV] else worlds[0]
    has0 = 0.0 in phis
    if has0:
        for s in seeds:
            a = ARCH.get((N_KV, s, world, 0.0))
            if not a:
                bads.append(f"{s}: DIR-26-O 缺")
                continue
            for f in AF_TAU:
                ncmp += 1
                if by[(s, 0.0)]["arms"]["tau_iid"][f] != a["arms"]["tau"][f]:
                    bads.append(f"{s}/tau_iid.{f}")
    v["anchor"] = {"ok": (not bads) if has0 else True, "n_cmp": ncmp, "n_bad": len(bads),
                   "bad": bads[:8], "world": world, "has_phi0": bool(has0)}
    v["pair_ok"] = v["anchor"]["ok"]
    # ── P2：恢复判据（φ=0.9）
    p_main = 0.9 if 0.9 in phis else phis[-1]
    d_rec = _pair([by[(s, p_main)]["arms"]["tau_AR"]["gain_recov"] for s in seeds], [0.0] * len(seeds))
    d_ret = _pair([by[(s, p_main)]["arms"]["tau_AR"]["ret1"] - by[(s, p_main)]["arms"]["tau_iid"]["ret1"]
                   for s in seeds], [0.0] * len(seeds))
    v["P2"] = {"phi": p_main, "gain_recov": d_rec, "delta_ret1": d_ret,
               "ok": bool(abs(d_rec["mean"]) >= REC_THRESH and max(d_rec["pos"], d_rec["neg"]) >= SIGN_MIN)}
    # ── 诊断
    diag = {}
    for p in phis:
        diag[str(p)] = {
            "tau_ar": calib[p]["c_tau"],
            "gain_iid": float(np.mean([by[(s, p)]["arms"]["tau_iid"]["gain"] for s in seeds])),
            "gain_ar": float(np.mean([by[(s, p)]["arms"]["tau_AR"]["gain"] for s in seeds])),
            "gain_recov": _pair([by[(s, p)]["arms"]["tau_AR"]["gain_recov"] for s in seeds], [0.0] * len(seeds)),
            "agree": float(np.mean([by[(s, p)]["arms"]["tau_AR"]["agree_vs_iid"] for s in seeds])),
            "issued_ar": float(np.mean([by[(s, p)]["arms"]["tau_AR"]["issued"] for s in seeds])),
            "issued_iid": float(np.mean([by[(s, p)]["arms"]["tau_iid"]["issued"] for s in seeds])),
            "ret1_ar": float(np.mean([by[(s, p)]["arms"]["tau_AR"]["ret1"] for s in seeds])),
            "ret1_iid": float(np.mean([by[(s, p)]["arms"]["tau_iid"]["ret1"] for s in seeds])),
        }
    v["diag"] = diag
    # ── 通道
    if not v["P1"]["ok"]:
        ch = "CALIB-DEGEN / IND"
    elif not v["anchor"]["ok"]:
        ch = "IND"
    elif v["P2"]["ok"]:
        ch = "MISMATCH"
    else:
        ch = "DIR-DAMAGED"
    v["channel"] = ch
    v["verdict_ok"] = bool(ch == "MISMATCH")
    return v


def chk(checks, cid, ok, detail):
    checks.append({"id": cid, "ok": bool(ok), "msg": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}  {detail}")
    return bool(ok)


# ══════════════════════════════════════════════ Stage A
def stageA():
    v = {"checks": []}
    v["dev_md5"] = md5("/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py")
    chk(v["checks"], "A26P-1", v["dev_md5"] == M14.DEV_MD5, f"装置 md5 {v['dev_md5']}")
    m = md5("/home/ec2-user/sgmn-loop/dir26o/R1_ar1.py")
    chk(v["checks"], "A26P-2", m == "a3c0981bbf8e09a92b2b5411742d76f1",
        f"参照脚本 `dir26o/R1_ar1.py`（档案最终版）md5={m}（须等于 a3c0981bbf8e09a92b2b5411742d76f1）")

    calib = {}
    tau_iid = cal_at(N_KV)["c_tau"]
    v["tau_iid"] = tau_iid
    for p in PHIS:
        calib[p] = calibrate_ar(p)
    v["calib"] = {str(p): calib[p] for p in calib}
    chk(v["checks"], "A26P-3", all(calib[p]["gate_ok"] for p in PHIS),
        "AR(1) 标定 gate_ok：" + ", ".join(f"φ{p}:{calib[p]['gate_ok']}(τ={calib[p]['c_tau']:.4f})" for p in PHIS))
    chk(v["checks"], "A26P-4", abs(calib[0.0]["c_tau"] - tau_iid) < 1e-12,
        f"φ=0 的 τ_AR={calib[0.0]['c_tau']:.6f} == cal_at(60)={tau_iid:.6f}")
    t1 = calibrate_ar(0.9)["c_tau"]
    t2 = calibrate_ar(0.9)["c_tau"]
    chk(v["checks"], "A26P-5", t1 == t2, f"φ=0.9 的 τ_AR 两口径逐位相等（{t1:.6f} == {t2:.6f}）")
    # A26P-6 / A26P-7：配对性 + 位级锚点（5 种子预演；全 30 在正式轮 crit）
    ncmp, bads = 0, []
    for s in SEEDS[:5]:
        a = ARCH.get((N_KV, s, 40, 0.0))
        u = run_unit(s, 0.0, tau_iid, calib[0.0]["c_tau"])
        for f in ("issued", "ret1", "wsig"):
            ncmp += 1
            if u["arms"]["tau_iid"][f] != a["arms"]["tau"][f]:
                bads.append(f"{s}/tau_iid.{f}")
    chk(v["checks"], "A26P-6", not bads,
        f"配对性：φ=0 的 tau_iid 与 DIR-26-O 的 tau 比较 {ncmp} 项，坏 {len(bads)} {bads[:3]}")
    chk(v["checks"], "A26P-7", not bads, f"位级锚点（E0 复用成立；全 30 种子在正式轮 crit 复核）")
    # A26P-8：确定性
    u1 = run_unit(SEEDS[0], 0.6, tau_iid, calib[0.6]["c_tau"])
    u2 = run_unit(SEEDS[0], 0.6, tau_iid, calib[0.6]["c_tau"])
    same = all(u1["arms"][t][f] == u2["arms"][t][f] for t in ("tau_iid", "tau_AR")
               for f in ("issued", "ret1", "mu", "wsig"))
    chk(v["checks"], "A26P-8", same, "同 (seed,φ,τ) 跑两次逐位相同")
    # A26P-9：口径 O 护栏
    chk(v["checks"], "A26P-9", all(u1["arms"][t]["mu_ret1_consistent"] for t in ("E0", "tau_iid", "tau_AR")),
        "每单元 μ/精度口径一致")
    # A26P-10：标定量程（诊断）
    ratios = {p: calib[p]["c_tau"] / tau_iid for p in PHIS}
    inband = all(0.5 <= r <= 2.0 for r in ratios.values())
    chk(v["checks"], "A26P-10", inband,
        "τ_AR/τ_iid 比值（诊断）：" + ", ".join(f"φ{p}:{ratios[p]:.3f}" for p in PHIS))
    # A26P-12（rev 1.1 增补）：测量层不漂移 —— 本地 rec ≡ RA.run_unit 的 tau 臂
    ref = RA.run_unit(SEEDS[0], 0.3, tau_iid)
    mine = run_unit(SEEDS[0], 0.3, tau_iid, calib[0.3]["c_tau"])
    drift = [f for f in ("issued", "ret1", "mu", "clean", "wsig", "ho4_mean", "ho8_mean", "mu_ret1_consistent")
             if mine["arms"]["tau_iid"][f] != ref["arms"]["tau"][f]]
    chk(v["checks"], "A26P-12", not drift,
        f"本地测量层 ≡ `RA.run_unit` 的 tau 臂（逐位比较 {len(drift)} 项漂移 {drift[:4]}）")
    chk(v["checks"], "A26P-11", len(v["checks"]) >= 11,
        f"断言集合非空（{len(v['checks'])} 条）且逐条 PASS（纪律 51：空断言集不得记 PASS）")
    v["passed"] = bool(v["checks"] and all(c["ok"] for c in v["checks"]))
    return v


# ══════════════════════════════════════════════ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("stageA", "smoke", "formal"), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--phis", type=float, nargs="+", default=None)
    ap.add_argument("--bs", type=int, nargs="+", default=[N_KV])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if args.stage == "stageA":
        v = stageA()
        json.dump(v, open(f"{OUT}/DIR26Pr11_stageA.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        for c in v["checks"]:
            print(("  ✓ " if c["ok"] else "  ✗ ") + c["id"] + " " + c["msg"])
        print(f"=== Stage A {'PASS' if v['passed'] else 'FAIL'}（{sum(c['ok'] for c in v['checks'])}/{len(v['checks'])}）"
              f" {time.time() - t0:.1f}s ===")
        sys.exit(0 if v["passed"] else 1)

    seeds = args.seeds or (list(SEEDS[:2]) if args.stage == "smoke" else list(SEEDS))
    phis = tuple(args.phis) if args.phis else PHIS
    bs = args.bs
    tau_iid = cal_at(N_KV)["c_tau"]
    calib = {p: calibrate_ar(p) for p in phis}
    res = {"checks": [], "rows": [], "calib": {"tau_iid": tau_iid, "tau_ar": {str(p): calib[p] for p in phis}},
           "stage": args.stage, "meta": {"PHIS": list(phis), "SEEDS": list(seeds), "T": T_UNIT,
                                         "card": "DIRECTION_rev26p_v15.md",
                                         "thresholds": {"REC_THRESH": REC_THRESH, "SIGN_MIN": SIGN_MIN}}}
    print(f"=== {args.stage}（φ∈{phis} × {len(seeds)} 种子 × 2 新驱动，τ_iid={tau_iid:.4f}）===")
    for s in seeds:
        for p in phis:
            for b in bs:
              tt = time.time()
              u = run_unit(s, p, tau_iid, calib[p]["c_tau"], b=b)
              res["rows"].append(u)
              ar = u["arms"]
              print(f"  b={b:2d} seed={s:3d} φ={p:.1f} | E0 {ar['E0']['ret1']:.4f} | "
                  f"τ_iid {ar['tau_iid']['ret1']:.4f}/g {ar['tau_iid']['gain']:+.4f}/n {ar['tau_iid']['issued']:.0f} | "
                  f"τ_AR {ar['tau_AR']['ret1']:.4f}/g {ar['tau_AR']['gain']:+.4f}/rec {ar['tau_AR']['gain_recov']:+.4f}/n {ar['tau_AR']['issued']:.0f} | "
                  f"{time.time() - tt:.0f}s", flush=True)
    if args.stage == "formal":
        res["verdicts"] = crit(res["rows"], calib)
        res["passed"] = bool(res["verdicts"]["verdict_ok"])
        vv = res["verdicts"]
        print("\n===== 判决 =====")
        print(f"  P1 标定有效 ok={vv['P1']['ok']}")
        print(f"  P2 恢复判据 φ={vv['P2']['phi']}: gain_recov {vv['P2']['gain_recov']['mean']:+.4f} "
              f"(3se {vv['P2']['gain_recov']['k']:.4f}, +{vv['P2']['gain_recov']['pos']}/-{vv['P2']['gain_recov']['neg']}) "
              f"P2 {vv['P2']['ok']}")
        for p, d in vv["diag"].items():
            print(f"  φ={p}: τ_AR={d['tau_ar']:.4f} gain_iid={d['gain_iid']:+.4f} gain_ar={d['gain_ar']:+.4f} "
                  f"recov={d['gain_recov']['mean']:+.4f} agree={d['agree']:.3f} issued {d['issued_iid']:.0f}→{d['issued_ar']:.0f}")
        print(f"  锚点：{vv['anchor']['n_cmp']} 项、坏 {vv['anchor']['n_bad']}")
        print(f"  ★ 通道 = {vv['channel']}")
    else:
        res["verdicts"] = {"note": "smoke/Stage A 不出判决"}
        res["passed"] = None
    p = f"{OUT}/DIR26Pr11_{args.stage}{args.tag}.json"
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  → {p}  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
