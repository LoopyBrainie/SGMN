r"""DIR-26-N · R1_gating15.py —— $\alpha$ 的**产生规则**（卡片 rev 1.1 冻结）

卡片：`revolve/DIR-26-N/DIRECTION_rev26n_v11.md`（rev 1.0 → **1.1**：只订正锚点值对数 1110 → 300，见 §5）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，**零改动**）

唯一改动（相对 `dir26m/R1_gating14.py`）——按纪律「只改测量层与独立变量」：
  ① **独立变量**：把 $B$ 臂的门控谓词换成四条**产生规则**（$\tau$ 标定基线 / H 历史中位数 / R 相对残差 /
     J 局部目标符号），$n_{\rm kv}{=}60$ 固定、$b\in\{40,4\}$（世界值模式数 $m=20+b$；同一 `build_stream_b`）；
  ② **臂集收缩**：每单元只跑 5 条驱动 —— $\varnothing$、$\tau$、H、R、J（本轮判据不用 $G/R_\mu$ 口径，
     P/Fu/Uu/Ku 不跑）；
  ③ 判据量：$\mathrm{ret}_1=\,$`measure_hit_bounded`（口径 O，DEC-38）、控制口径 X（外来值读）逐字沿用。

★ 门控谓词的**唯一实现**在 `dir26n/probe26n_feas.drive_rule`（探针即参考实现，本文件只 import 它）：
  这样"规则定义"只有一份，Stage A 用"跑两次 log 逐位相同"（A26N-6）守住因果性。
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

for _p in ("dir26n", "dir26m", "dir26l", "dir26k", "dir26j", "dir26i", "dir26h", "dir26g",
           "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating12 as K
import R1_gating14 as M14
import probe26n_feas as PR
from R1_gating import D_MODEL, drive, make_core
from R1_gating5 import theta_of
from R1_gating12 import (R_PRES, R_HO4, R_HO8, SE_K, cal_at, measure_prec, read_stream)
from R1_gating7 import md5
from rerun_missing import measure_hit_bounded

N_READ = 20
N_KV = 60
BS = (40, 4)
SEEDS = K.SEEDS
D45 = "/home/ec2-user/sgmn-loop/dir26m/out/DIR26Mr10_formal.json"
RULES3 = ("H", "R", "J")                      # 判决对象（P1 未过的退化规则单独留档）
T_UNIT = R_PRES * N_KV * 2 * G.M               # 960
OUT = "/home/ec2-user/sgmn-loop/dir26n/out"
# ── 判据阈值（卡片 §2.2 冻结）──
P1_BAND = (0.20, 0.75)
P1_MIN_ISSUED = 240
A_GATE = 0.15
P2_FLOOR = -0.05
P3_FLOOR = 0.10
P3_SIGN_MIN = 27
CTL_D24 = 0.20
CTL_GATE60 = (0.30, 0.50)
SKIP_REP_MIN = 0.85


def _logsig(log):
    r"""门控 log 的**内容指纹**（因果性检查用：同一 seed/b/规则跑两次必须同指纹）。"""
    return hashlib.md5(json.dumps([[e["round"], e["item"], round(e["cos"], 12), e["issued"]]
                                   for e in log]).encode()).hexdigest()[:16]


# ══════════════════════════════════════════════ 单元
def run_unit(seed, b, tau, n_kv=N_KV):
    r"""一条 (seed, b) 单元 = 5 条驱动：$\varnothing$ / $\tau$ / H / R / J。"""
    D = M14.build_stream_b(b, seed, D_MODEL, int(n_kv), R_PRES, theta_of(0.9))
    assert int(D["n"]) == n_kv, f"写流 N 不符：{D['n']} != {n_kv}"
    RS8, RS4 = read_stream(seed, R_HO8), read_stream(seed, R_HO4)
    proto, vb = D["proto"][:N_READ], D["vb"][:N_READ]
    out = {"n_kv": int(n_kv), "n_read": N_READ, "seed": seed, "b": int(b),
           "m": N_READ + int(b), "T": R_PRES * int(n_kv) * 2 * G.M, "arms": {}}

    def rec(tag, core, issued, gate=None):
        W = core._W().detach()
        P = W - core.W0
        m8 = measure_hit_bounded(P, RS8["K"], vb, N_READ, R_HO8)
        p4 = measure_prec(P, RS4["K"], vb, N_READ, R_HO4)
        mx = measure_hit_bounded(P, RS8["K"], RS8["vb"], N_READ, R_HO8)      # 控制口径 X（外来值）
        # ★ DEF-26N-3：口径 O 的 ret1 只能与**同抖动档**（R_HO8）的精度口径比；拿 R_HO4 的 ho_mean 比是错的。
        assert m8["ret1"] == measure_prec(P, RS8["K"], vb, N_READ, R_HO8)["ho_mean"], "μ/精度口径不符（同抖动档）"
        out["arms"][tag] = {
            "issued": int(issued), "wsig": K.wsig(W),
            "W_unchanged": bool(torch.equal(core._W().detach(), W)),
            "clean": G.metrics_plas(W, core.W0, proto, vb, N_READ)["ret1"],
            "mu": m8["mu"], "mu_sd": m8["mu_sd"], "mu_read": m8["n_read"],
            "ret1": m8["ret1"], "ret1_maj": m8["ret1_maj"], "maj_need": m8["maj_need"],
            "ho8_mean": m8["ret1"], "ho8_maj": m8["ret1_maj"],
            "ho4_mean": p4["ho_mean"], "ho4_maj": p4["ho_maj"],
            "mu_ret1_consistent": m8["mu_ret1_consistent"],
            "X": {"mu": mx["mu"], "mu_sd": mx["mu_sd"], "ret1": mx["ret1"]},
        }
        if gate:
            out["arms"][tag].update(gate)

    c = make_core(D_MODEL, seed, K.R0)
    kA, _ = drive(c, D, "A")
    rec("E0", c, kA, {"gate_rate": 0.0, "skip_frac_repeat": float("nan"), "qbar": kA / n_kv})

    logs = {}
    for mode in ("A",) + RULES3:
        c = make_core(D_MODEL, seed, K.R0)
        ki, lg, diag = PR.drive_rule(c, D, mode, tau=tau)
        sk = [e for e in lg if not e["issued"]]
        gate = {"gate_rate": len(sk) / len(lg),
                "skip_frac_repeat": (float(np.mean([e["round"] > 0 for e in sk])) if sk else float("nan")),
                "qbar": ki / n_kv, "log_n": len(lg)}
        if diag["dj_n"]:
            gate["dj_neg_frac"] = diag["dj_neg"] / diag["dj_n"]
        rec("tau" if mode == "A" else mode, c, ki, gate)
        logs[mode] = lg

    # 一致性诊断（相对标定基线）
    for mode in RULES3:
        a, g = logs["A"], logs[mode]
        out["arms"][mode]["agree_vs_tau"] = float(np.mean([e1["issued"] == e2["issued"] for e1, e2 in zip(a, g)]))
        out["arms"][mode]["both_write"] = float(np.mean([e1["issued"] and e2["issued"] for e1, e2 in zip(a, g)]))
        out["arms"][mode]["neither_write"] = float(np.mean([(not e1["issued"]) and (not e2["issued"])
                                                            for e1, e2 in zip(a, g)]))
    out["qbar"] = out["arms"]["tau"]["qbar"]
    out["log_sig"] = {m: _logsig(logs[m]) for m in logs}
    return out


# ══════════════════════════════════════════════ 判据骨架
def _pair(vals_a, vals_b):
    a, b = np.asarray(vals_a, float), np.asarray(vals_b, float)
    d = a - b
    n = len(d)
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    return {"mean": float(np.mean(d)), "sd": sd, "se": se, "k": SE_K * se, "n": n,
            "pos": int((d > 0).sum()), "vals": [round(float(x), 6) for x in d]}


def crit(rows):
    r"""判决（卡片 §2.2/§2.3 冻结）。返回逐规则读数 + 通道。"""
    by = {b: sorted([r for r in rows if r["b"] == b], key=lambda r: r["seed"]) for b in BS}
    assert len(by[40]) == len(by[4]) == len(rows) // 2, "两档种子数不等"
    seeds = [r["seed"] for r in by[40]]
    assert seeds == [r["seed"] for r in by[4]], "两档种子不一致（必须同种子配对）"

    def col(b, tag, f):
        return [r["arms"][tag][f] for r in by[b]]

    v = {"n_seed": len(seeds), "rules": {}}
    for R in RULES3:
        gate24 = float(np.mean(col(4, R, "gate_rate")))
        gate60 = float(np.mean(col(40, R, "gate_rate")))
        iss24 = float(np.mean(col(4, R, "issued")))
        d60 = _pair(col(40, R, "ret1"), col(40, "tau", "ret1"))
        d24 = _pair(col(4, R, "ret1"), col(4, "E0", "ret1"))
        p1 = bool(P1_BAND[0] <= gate24 <= P1_BAND[1] and iss24 >= P1_MIN_ISSUED)
        dgate = gate24 - gate60
        v["rules"][R] = {
            "gate60": gate60, "gate24": gate24, "dgate": dgate, "issued24": iss24,
            "P1": p1, "a": bool(dgate >= A_GATE),
            "d60": d60, "d24": d24,
            "P2": bool(d60["mean"] >= P2_FLOOR),
            "P3": bool(d24["mean"] >= P3_FLOOR and d24["pos"] >= P3_SIGN_MIN),
            "ret1_60": float(np.mean(col(40, R, "ret1"))), "ret1_24": float(np.mean(col(4, R, "ret1"))),
            "skip_rep_24": float(np.nanmean([x for x in col(4, R, "skip_frac_repeat") if x == x])),
            "agree_tau_24": float(np.mean(col(4, R, "agree_vs_tau"))),
        }
        if "dj_neg_frac" in by[40][0]["arms"][R]:
            v["rules"][R]["dj_neg_frac"] = [by[b][0]["arms"][R]["dj_neg_frac"] for b in BS]
    # 正对照
    ctl24 = _pair(col(4, "tau", "ret1"), col(4, "E0", "ret1"))
    g60 = float(np.mean(col(40, "tau", "gate_rate")))
    v["control"] = {"d24_tau": ctl24, "gate60_tau": g60,
                    "ok": bool(ctl24["mean"] >= CTL_D24 and CTL_GATE60[0] <= g60 <= CTL_GATE60[1])}
    # J ≡ 全写？（A26N-7：正式轮复核）
    j_all = all(r["arms"]["J"]["issued"] == T_UNIT and r["arms"]["J"]["wsig"] == r["arms"]["E0"]["wsig"] for r in rows)
    v["j_always_write"] = bool(j_all)

    X = [R for R in RULES3 if v["rules"][R]["P1"]]
    v["pool"] = X
    der = [R for R in X if v["rules"][R]["a"] and v["rules"][R]["P2"] and v["rules"][R]["P3"]]
    gate_only = [R for R in X if v["rules"][R]["a"]]
    if not v["control"]["ok"]:
        ch = "PREMISE-ABSENT"
    elif not X:
        ch = "PREMISE-ABSENT"
    elif der:
        ch = "DERIVED"
    elif gate_only:
        ch = "GATE-ONLY"
    else:
        ch = "CALIB-DEP"
    # 锚点（270 值对）：b=40 的 τ 臂 vs D-45 的 B 臂 —— 逐位相同才允许裁决（否则读数不可比）
    arch = {(r["n_kv"], r["seed"]): r for r in json.load(open(D45))["rows"] if r["b"] == 40}
    AF = ("issued", "ret1", "ho4_mean", "ho4_maj", "ho8_mean", "clean", "mu", "wsig")
    ncmp, bads = 0, []
    for r in by[40]:
        a = arch.get((N_KV, r["seed"]))
        if not a:
            bads.append(f"{r['seed']}: D-45 缺该种子")
            continue
        for f in AF:
            ncmp += 1
            if r["arms"]["tau"][f] != a["arms"]["B"][f]:
                bads.append(f"{r['seed']}/tau.{f}")
        ncmp += 1
        if r["arms"]["tau"]["qbar"] != a["qbar"]:
            bads.append(f"{r['seed']}/qbar")
    v["anchor"] = {"ok": not bads, "n_cmp": ncmp, "n_bad": len(bads), "bad": bads[:8]}
    if not v["anchor"]["ok"]:
        ch = "IND"

    v["channel"] = ch
    v["derived"] = der
    v["gate_only"] = gate_only
    v["verdict_ok"] = bool(ch == "DERIVED")
    return v


def chk(checks, cid, ok, detail):
    r"""Stage A 断言（本地版：显式收集，不用 R1_gating7 的模块级 `res`）。"""
    checks.append({"id": cid, "ok": bool(ok), "msg": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}  {detail}")
    return bool(ok)


# ══════════════════════════════════════════════ Stage A
def stageA():
    v = {"checks": [], "rows": []}
    v["dev_md5"] = md5("/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py")
    chk(v["checks"], "A26N-1", v["dev_md5"] == M14.DEV_MD5, f"装置 md5 {v['dev_md5']}（期望 {M14.DEV_MD5}）")

    tau = cal_at(N_KV)["c_tau"]
    chk(v["checks"], "A26N-8a", CTL_GATE60[0] <= 0.4 <= CTL_GATE60[1], f"c_tau(60)={tau:.4f}（运行时取，禁止手抄）")
    v["ref_md5"] = md5(PR.__file__)          # 只记录（门控谓词的唯一实现），不作断言

    s0 = SEEDS[0]
    # A26N-3：τ 臂与 R1_gating12.drive(...,"B",tau) 逐位相等
    D40 = M14.build_stream_b(40, s0, D_MODEL, N_KV, R_PRES, theta_of(0.9))
    c1 = make_core(D_MODEL, s0, K.R0)
    k1, _l1 = drive(c1, D40, "B", tau=tau)
    c2 = make_core(D_MODEL, s0, K.R0)
    k2, l2, _d2 = PR.drive_rule(c2, D40, "A", tau=tau)
    same = (k1 == k2 and K.wsig(c1._W().detach()) == K.wsig(c2._W().detach()))
    chk(v["checks"], "A26N-3", same, f"τ 臂与 `R1_gating12.drive(...,'B',tau)` 逐位相同（issued {k1}=={k2}）")

    # A26N-4：E0 臂与 drive(...,"A") 逐位相等
    c3 = make_core(D_MODEL, s0, K.R0)
    k3, _l3 = drive(c3, D40, "A")
    c4 = make_core(D_MODEL, s0, K.R0)
    k4, _l4 = drive(c4, D40, "A")
    chk(v["checks"], "A26N-4", k3 == k4 == T_UNIT and K.wsig(c3._W().detach()) == K.wsig(c4._W().detach()),
        f"$\varnothing$ 臂 issued={k3}（=T={T_UNIT}）且两次驱动 wsig 相同")

    # A26N-5：工作例（ΔJ 便宜式 vs 暴力构造）
    we = PR.worked_example(lambda: make_core(D_MODEL, s0, K.R0), D40)
    worst = 0.0
    for r in we["rows"]:
        jmag = max(abs(r["dj_brute"]), 1.0)
        worst = max(worst, abs(r["dj"] - r["dj_brute"]) / jmag)
    chk(v["checks"], "A26N-5", worst <= 1e-6, f"ΔJ 便宜式 vs 暴力 $W'=W^0+U'V'^\\top$：最坏 $|diff|/\\max(|J|,1)$ = {worst:.2e}（≤1e-6）")

    # A26N-6：因果性（跑两次 log 逐位相同）
    sigs = {}
    for mode in RULES3:
        cs = []
        for _ in range(2):
            c = make_core(D_MODEL, s0, K.R0)
            ki, lg, _dd = PR.drive_rule(c, D40, mode, tau=tau)
            cs.append(_logsig(lg))
        sigs[mode] = (cs[0], cs[1], ki)
    chk(v["checks"], "A26N-6", all(a == b for a, b, _ in sigs.values()),
        f"三条规则各跑两次 log 逐位相同（{ {m: a == b for m, (a, b, _) in sigs.items()} }）")

    # A26N-7 / A26N-8b / A26N-9 / A26N-10：一个 E0+四规则单元在 s0/b=40 上
    u = run_unit(s0, 40, tau)
    v["rows"].append(u)
    ar = u["arms"]
    chk(v["checks"], "A26N-7", ar["J"]["issued"] == T_UNIT and ar["J"]["wsig"] == ar["E0"]["wsig"],
        f"J 臂 issued={ar['J']['issued']}、wsig==E0 ⇒ J≡全写（ΔJ<0 占比 {ar['J'].get('dj_neg_frac'):.3f}）")
    u4 = run_unit(s0, 4, tau)
    v["rows"].append(u4)
    a4 = u4["arms"]
    chk(v["checks"], "A26N-8b",
        CTL_GATE60[0] <= ar["tau"]["gate_rate"] <= CTL_GATE60[1]
        and all(0.05 <= a4[m]["gate_rate"] <= 0.95 for m in ("H", "R")),
        f"量程：gate(τ;60)={ar['tau']['gate_rate']:.3f}∈{CTL_GATE60}；"
        f"gate(H;24)={a4['H']['gate_rate']:.3f}、gate(R;24)={a4['R']['gate_rate']:.3f}（J 由 A26N-7 判为退化）")
    gating_arms = [m for m in ("tau",) + RULES3 if ar[m]["skip_frac_repeat"] == ar[m]["skip_frac_repeat"]]
    chk(v["checks"], "A26N-9", all(ar[m]["skip_frac_repeat"] >= SKIP_REP_MIN for m in gating_arms)
        and "tau" in gating_arms,
        "跳写中重复占比 " + ", ".join(f"{m}:{ar[m]['skip_frac_repeat']:.3f}" for m in gating_arms)
        + f"（无跳写的退化臂由 A26N-7 处理：{[m for m in RULES3 if m not in gating_arms]}）")
    chk(v["checks"], "A26N-10", all(abs(ar[m]["X"]["mu"]) > 0 for m in ("E0",) + ("tau",) + RULES3)
        and all(ar[m]["mu_ret1_consistent"] for m in ("E0", "tau") + RULES3),
        "控制口径 X 有量纲且 μ/精度口径一致")

    # A26N-2（**单种子预演**；全 30 种子的 270 值对锚点在正式轮 `crit` 里复核）
    arch = {(r["n_kv"], r["seed"]): r for r in json.load(open(D45))["rows"] if r["b"] == 40}
    ncmp, bads = 0, []
    miss = [seed for seed in SEEDS if (N_KV, seed) not in arch]
    if miss:
        bads.append(f"D-45 缺种子 {miss}")
    a0 = arch[(N_KV, s0)]
    for f in ("issued", "ret1", "ho4_mean", "ho4_maj", "ho8_mean", "clean", "mu", "wsig"):
        ncmp += 1
        if ar["tau"][f] != a0["arms"]["B"][f]:
            bads.append(f"tau.{f}: {ar['tau'][f]} != {a0['arms']['B'][f]}")
    for f in ("qbar",):
        ncmp += 1
        if ar["tau"][f] != a0[f]:
            bads.append(f"{f}: {ar['tau'][f]} != {a0[f]}")
    chk(v["checks"], "A26N-2", not bads, f"锚点：b=40 的 τ 臂 8+1 字段与 D-45 的 B 臂逐位相同（比较 {ncmp} 项，坏 {len(bads)} 项 {bads[:4]}）")

    v["tau"] = tau
    v["passed"] = all(c["ok"] for c in v["checks"])
    return v


# ══════════════════════════════════════════════ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("stageA", "smoke", "formal"), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--bs", type=int, nargs="+", default=list(BS))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if args.stage == "stageA":
        v = stageA()
        json.dump(v, open(f"{OUT}/DIR26Nr11_stageA.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        for c in v["checks"]:
            print(("  ✓ " if c["ok"] else "  ✗ ") + c["id"] + " " + c["msg"])
        print(f"=== Stage A {'PASS' if v['passed'] else 'FAIL'}（{sum(c['ok'] for c in v['checks'])}/{len(v['checks'])}）"
              f" {time.time() - t0:.1f}s ===")
        sys.exit(0 if v["passed"] else 1)

    seeds = args.seeds or (list(SEEDS[:2]) if args.stage == "smoke" else list(SEEDS))
    bs = args.bs or list(BS)
    tau = cal_at(N_KV)["c_tau"]
    res = {"checks": [], "rows": [], "calib": {"c_tau": tau, "n_kv": N_KV},
           "stage": args.stage, "meta": {"RULES3": list(RULES3), "SEEDS": list(seeds), "BS": list(bs),
                                         "T": T_UNIT, "card": "DIRECTION_rev26n_v11.md",
                                         "thresholds": {"P1_BAND": P1_BAND, "P1_MIN_ISSUED": P1_MIN_ISSUED,
                                                        "A_GATE": A_GATE, "P2_FLOOR": P2_FLOOR,
                                                        "P3_FLOOR": P3_FLOOR, "P3_SIGN_MIN": P3_SIGN_MIN,
                                                        "CTL_D24": CTL_D24, "CTL_GATE60": CTL_GATE60}}}
    print(f"=== {args.stage}（b∈{bs} × {len(seeds)} 种子 × 5 臂，tau={tau:.4f}）===")
    for b in bs:
        for s in seeds:
            tt = time.time()
            u = run_unit(s, b, tau)
            res["rows"].append(u)
            ar = u["arms"]
            print(f"  b={b:2d} seed={s:3d} | E0 {ar['E0']['ret1']:.4f} | "
                  f"τ {ar['tau']['ret1']:.4f}/{ar['tau']['gate_rate']:.3f} | "
                  + " ".join(f"{m} {ar[m]['ret1']:.4f}/{ar[m]['gate_rate']:.3f}" for m in RULES3)
                  + f" | {time.time() - tt:.0f}s", flush=True)
    if args.stage == "formal":
        res["verdicts"] = crit(res["rows"])
        res["passed"] = bool(res["verdicts"]["verdict_ok"])
        vv = res["verdicts"]
        print("\n===== 判决 =====")
        for m, d in vv["rules"].items():
            print(f"  {m}: gate60 {d['gate60']:.3f}→gate24 {d['gate24']:.3f} (Δ {d['dgate']:+.3f}) | "
                  f"P1 {d['P1']} (a) {d['a']} | Δ60 {d['d60']['mean']:+.4f} (3se {d['d60']['k']:.4f}) P2 {d['P2']} | "
                  f"Δ24 {d['d24']['mean']:+.4f} (3se {d['d24']['k']:.4f}, 正 {d['d24']['pos']}/{d['d24']['n']}) P3 {d['P3']}")
        print(f"  正对照 Δ24(τ) {vv['control']['d24_tau']['mean']:+.4f}、gate60(τ) {vv['control']['gate60_tau']:.3f} ⇒ {vv['control']['ok']}")
        print(f"  J≡全写：{vv['j_always_write']}；判决集合 {vv['pool']}；派生 {vv['derived']}")
        print(f"  ★ 通道 = {vv['channel']}")
    else:
        res["verdicts"] = {"note": "smoke/Stage A 不出判决（纪律 §5）"}
        res["passed"] = None
    p = f"{OUT}/DIR26Nr11_{args.stage}.json"
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  → {p}  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
