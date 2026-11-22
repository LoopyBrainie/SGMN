r"""DIR-26-M · R1_gating14.py —— 剂量的**自变量**：世界中不同值模式数 $m$，还是条目数/预算？

卡片：`revolve/DIR-26-M/DIRECTION_rev26m_v10.md`（rev 1.0 冻结）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，零改动）

唯一改动（相对 `dir26k/R1_gating12.py`）——按纪律「只改测量层与独立变量」：
  ① **独立变量**：$n_{\rm kv}{=}60$ **固定**；把 40 个**非读条目**的值模式折叠到
     $b\in\{40,20,4\}$ 个（$\mathrm{pat}(i)=N_{\rm read}+((i-N_{\rm read})\bmod b)$，读集恒有 $\mathrm{pat}(i)=i$）
     ⇒ 世界中不同值模式数 $m=20+b\in\{60,40,24\}$；
  ② **预算外生冻结**（这是与探针 `probe26m_feas.py` 的实质差别，DEF-26M-1）：
     `cnt` 改为从 $b{=}40$（原世界）的**同一 seed** 取，记 `cnt_ref` ⇒ 三档的 $k_B$、逐条目计数 $q_i$、
     R 臂 keep 位置、日程剖面**逐位相同**，唯一变化的是写进 $W$ 的值内容；
  ③ 判据量：$\Delta_1=G_{b=20}-G_{b=40}$、$\Delta_2=G_{b=4}-G_{b=20}$（**同种子配对**，`_eff`）。
     $b{=}40$ 且 `cnt_ref=None` 时 `run_unit_m` **逐位等于** `R1_gating12.run_unit`（Stage A 证明）。

其余（装置层、六臂、读流、$\tau$ 标定、抖动指标、剖面规则、落盘字段）全部 import 同一对象。
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

for _p in ("dir26m", "dir26l", "dir26k", "dir26j", "dir26i", "dir26h", "dir26g", "dir26f",
           "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating12 as K
from R1_gating import (D_MODEL, N_DIST, ETA, M, S_KEY, build_stream as ORIG_BUILD, drive,
                       make_core, metrics_plas, unit)
from R1_gating2 import keep_positions, item_counts
from R1_gating5 import theta_of
from R1_gating7 import build_keep, realized_profile, PROFILE, md5, chk, res
from R1_gating10 import ret_jitter_margin
from R1_gating12 import (ARMS6, DEV_MD5, NU_CAP_MAX, QBAR_BAND, QMAX_CAP, R0, R_HO4, R_HO8,
                         R_MAIN, R_PRES, SE_K, cal_at, measure_prec, read_stream, wsig)
from rerun_missing import measure_hit_bounded        # DEC-38：只换假断言，数值同源

N_READ = 20
N_KV = 60
BS = (40, 20, 4)                     # 非读条目的不同值模式数 ⇒ 世界值模式数 m = 20 + b
SEEDS = K.SEEDS                      # 与 D-43 完全相同的 30 个（10 共享 + 20 全新）
D43 = "/home/ec2-user/sgmn-loop/dir26k/out/DIR26Kr10_formal.json"


# ══════════════════════════════════════════════ 独立变量：值模式折叠
def pat_of(b, n=N_KV):
    idx = torch.arange(int(n))
    if b >= n - N_READ:
        return idx                                    # 恒等（b=40）
    return torch.where(idx < N_READ, idx, N_READ + (idx - N_READ) % int(b))


def build_stream_b(b, seed, d, n, r_pres, theta):
    r"""读集（$i<N_{\rm read}$）**逐位不变**；非读条目的 $v^a/v^b$ 折叠到 $b$ 个模式。键 $K$ 一律不动。"""
    D = ORIG_BUILD(seed, d, n, r_pres, theta)
    if int(n) == N_KV and b < N_KV - N_READ:
        pat = pat_of(b, n)
        D["va"] = D["va"][pat].clone()
        D["vb"] = D["vb"][pat].clone()
        D["pat"] = pat
    return D


# ══════════════════════════════════════════════ 单元（逐字派生自 `R1_gating12.run_unit`）
def run_unit_m(seed, tau, b, cnt_ref=None, n_kv=N_KV):
    n_kv = int(n_kv)
    D = build_stream_b(b, seed, D_MODEL, n_kv, R_PRES, theta_of(0.9))
    assert int(D["n"]) == n_kv, f"写流 N 不符：{D['n']} != {n_kv}"
    RS8, RS4 = read_stream(seed, R_HO8), read_stream(seed, R_HO4)
    proto = D["proto"][:N_READ]
    out = {"n_kv": n_kv, "n_read": N_READ, "seed": seed, "b": int(b), "T": R_PRES * n_kv * 2 * M,
           "arms": {}, "prof": {}, "k": {}, "qbar": 0.0}

    vb = D["vb"][:N_READ]

    def rec(tag, core, issued):
        W = core._W().detach()
        W_before = W.clone()
        P = W - core.W0
        m8 = measure_hit_bounded(P, RS8["K"], vb, N_READ, R_HO8)     # 判据口径 O（DEC-38 护栏）
        p4 = measure_prec(P, RS4["K"], vb, N_READ, R_HO4)
        mx = measure_hit_bounded(P, RS8["K"], RS8["vb"], N_READ, R_HO8)   # 控制口径 X（外来值）
        assert m8["ret1"] == measure_prec(P, RS8["K"], vb, N_READ, R_HO8)["ho_mean"], "μ/精度口径不符"
        out["arms"][tag] = {
            "issued": issued, "wsig": wsig(W),
            "W_unchanged": bool(torch.equal(core._W().detach(), W_before)),
            "clean": metrics_plas(W, core.W0, proto, vb, N_READ)["ret1"],
            "mu": m8["mu"], "mu_sd": m8["mu_sd"], "mu_read": m8["n_read"],
            "ret1": m8["ret1"], "ret1_maj": m8["ret1_maj"], "maj_need": m8["maj_need"],
            "ho8_mean": m8["ret1"], "ho8_maj": m8["ret1_maj"],
            "ho4_mean": p4["ho_mean"], "ho4_maj": p4["ho_maj"],
            "mu_ret1_consistent": m8["mu_ret1_consistent"],
            "X": {"mu": mx["mu"], "mu_sd": mx["mu_sd"], "ret1": mx["ret1"]},
        }

    c = make_core(D_MODEL, seed, R0); kA, _ = drive(c, D, "A"); rec("A", c, kA)
    c = make_core(D_MODEL, seed, R0); kB, logB = drive(c, D, "B", tau=tau); rec("B", c, kB)
    cnt_own = item_counts({t for t, e in enumerate(logB) if e["issued"]}, n_kv)
    cnt = list(cnt_own) if cnt_ref is None else [int(x) for x in cnt_ref]
    assert len(cnt) == n_kv, f"cnt 长度不符：{len(cnt)} != {n_kv}"
    out["k"] = {"A": kA, "B": kB, "B_frozen": int(sum(cnt))}
    out["qbar"] = kB / n_kv                                  # 本世界自己的门控读数（诊断）
    out["qbar_used"] = sum(cnt) / n_kv                       # ★ 实际驱动预算（= 冻结值）
    out["qmax"] = int(max(cnt))
    out["counts"] = cnt
    out["counts_own"] = cnt_own
    out["cnt_frozen"] = bool(cnt_ref is not None)
    out["nu_cap"] = float(np.mean(np.array(cnt[:N_READ]) >= QMAX_CAP))
    out["W0sig"] = wsig(make_core(D_MODEL, seed, R0)._W().detach())

    c = make_core(D_MODEL, seed, R0)
    kP, _ = drive(c, D, "R", keep=keep_positions(n_kv, R_PRES, sum(cnt))); rec("P", c, kP)

    keeps = {}
    for pk in ("K", "U", "F"):
        keep, per_item = build_keep(n_kv, R_PRES, cnt, PROFILE[pk])
        keeps[pk] = keep
        tag = f"{pk}u"
        c = make_core(D_MODEL, seed, R0)
        kk, _ = drive(c, D, "R", keep=keep); rec(tag, c, kk)
        out["prof"][tag] = realized_profile(keep, n_kv, R_PRES)
        out["arms"][tag]["sum_ok"] = (kk == sum(cnt))
        out["arms"][tag]["quota_ok"] = all(sum(per_item[i]) == cnt[i] for i in range(n_kv))
        out["arms"][tag]["keep_n"] = len(keep)
    out["mirror_ok"] = (len(keeps["F"]) == len(keeps["K"])
                        and all(abs(out["prof"]["Fu"][i] - out["prof"]["Ku"][R_PRES - 1 - i]) < 1e-12
                                for i in range(R_PRES)))
    out["sched_dist"] = float(np.abs(np.array(out["prof"]["Fu"]) - np.array(out["prof"]["Ku"])).sum())
    out["G_O"] = -(out["arms"]["Fu"]["mu"] - out["arms"]["Ku"]["mu"])
    out["U0_O"] = out["arms"]["Uu"]["mu"]
    out["R_O"] = out["G_O"] / out["U0_O"]
    out["G_X"] = -(out["arms"]["Fu"]["X"]["mu"] - out["arms"]["Ku"]["X"]["mu"])
    out["U0_X"] = out["arms"]["Uu"]["X"]["mu"]
    return out


# ══════════════════════════════════════════════ 判据骨架（派生自 `R1_gating12._eff/_band`）
def _eff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    return {"mean": float(np.mean(d)), "sd": sd, "se": se, "k": SE_K * se, "n": n,
            "sig": bool(np.isfinite(se) and abs(float(np.mean(d))) >= SE_K * se),
            "vals": [round(float(x), 6) for x in d]}


def _band(x):
    x = np.asarray(x, float)
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1)) / np.sqrt(len(x))
    return m, se, SE_K * se


def _tree(G40, G20, G4, R40=None, R20=None, R4=None):
    r"""五通道判决树（卡 §4.1）。Δ 为**同种子配对**差、共同基准 $b{=}40$；阈值 $k=3\mathrm{se}(\Delta)$。

    ★ DEC-40：判决只给**远点** $\Delta_2=G_{m=24}-G_{m=60}$；近步 $\Delta_1=G_{m=40}-G_{m=60}$ 只用单侧约束
      $\Delta_1\le+3\mathrm{se}$ 防"非单调也判 DIV"，不要求它显著（探针实测近步仅 $\sim0.5\times3\mathrm{se}$）。
    ★ DEC-41：本轮把 **标尺分解** 升级为判决的一部分（$R_\mu=G/\mu_{U0}$ 的远点差 $\Delta_2^R$），
      因为 D-42/D-43 已建立"剂量 ≠ 标尺"，而值模式折叠改变的是写世界内容、不是读集
      ⇒ 若只有 $G$ 动而 $R_\mu$ 不动，那只是标尺的影子，不能叫"剂量随多样性长"。
    """
    d1 = _eff(G20, G40)
    d2 = _eff(G4, G40)
    down = d2["mean"] <= -d2["k"]
    up = d2["mean"] >= d2["k"]
    dR = _eff(R4, R40) if R4 is not None and R40 is not None else None
    if down and dR is not None and dR["mean"] >= dR["k"]:
        chan = "DIV-B"       # 绝对剂量随多样性下降，**相对**剂量反而上升（塌的是标尺）
    elif down and dR is not None and abs(dR["mean"]) <= dR["k"]:
        chan = "DIV-A"       # 绝对剂量降、相对剂量不动 ⇒ 整体标尺同向塌
    elif down:
        chan = "DIV-C"       # 绝对剂量降、相对剂量也降（真·差异剂量）
    elif up:
        chan = "REV"         # 反向：模式越少剂量越大
    elif abs(d2["mean"]) <= d2["k"] and (dR is None or dR["sig"]):
        chan = "SCALE"       # 绝对剂量不动、只有标尺动
    elif abs(d2["mean"]) <= d2["k"] and (dR is None or not dR["sig"]):
        chan = "FLAT"        # 两者都不动 ⇒ 多样性不是自变量
    else:
        chan = "PART"
    return {"D1": d1, "D2": d2, "DR": dR, "channel": chan, "mono_ok": bool(d1["mean"] <= d1["k"]),
            "g": {40: float(np.mean(G40)), 20: float(np.mean(G20)), 4: float(np.mean(G4))}}


def crit(rows):
    v = {}
    by = {b: sorted([r for r in rows if r["b"] == b], key=lambda r: r["seed"]) for b in BS}
    used = [r["seed"] for r in by[40]]
    assert all([r["seed"] for r in by[b]] == used for b in BS), "三档的种子序列不一致"
    assert len(used) >= 5, "种子过少"

    Gp = {b: [r["G_O"] for r in by[b]] for b in BS}
    Gx = {b: [r["G_X"] for r in by[b]] for b in BS}
    U0 = {b: [r["U0_O"] for r in by[b]] for b in BS}
    Rp = {b: [r["R_O"] for r in by[b]] for b in BS}

    # 护栏 (c) 锚点：b=40 档与 D-43 存档（n_kv=60）逐位相同
    arch = {(r["n_kv"], r["seed"]): r for r in json.load(open(D43))["rows"]}
    ncmp, bads = 0, []
    for r in by[40]:
        a = arch.get((N_KV, r["seed"]))
        if not a:
            bads.append(f"{r['seed']}: D-43 缺该种子")
            continue
        for a_ in ARMS6:
            for f in ("ho4_mean", "ho4_maj", "ho8_mean", "clean", "mu"):
                ncmp += 1
                if r["arms"][a_][f] != a["arms"][a_][f]:
                    bads.append(f"{r['seed']}/{a_}.{f}")
        for f in ("qbar", "qmax", "mirror_ok", "sched_dist", "G_O", "U0_O", "G_X"):
            ncmp += 1
            if r[f] != a[f]:
                bads.append(f"{r['seed']}/{f}")
    v["c_anchor"] = {"ok": not bads, "n_cmp": ncmp, "n_bad": len(bads), "bad": bads[:8]}

    # 护栏 (f) 预算冻结：三档逐种子 k_B、逐条目 cnt、日程剖面必须**完全相等**
    frz, bad2 = 0, []
    for s in used:
        base = [r for r in by[40] if r["seed"] == s][0]
        frz += 1
        for b in (20, 4):
            r = [x for x in by[b] if x["seed"] == s][0]
            if r["counts"] != base["counts"]:
                bad2.append(f"{s}/b={b}: cnt 不同")
            if r["k"]["B_frozen"] != base["k"]["B_frozen"]:
                bad2.append(f"{s}/b={b}: k_B 不同")
            if r["sched_dist"] != base["sched_dist"]:
                bad2.append(f"{s}/b={b}: 日程距离不同")
            for a_ in ARMS6:
                if r["prof"].get(a_) and base["prof"].get(a_) and r["prof"][a_] != base["prof"][a_]:
                    bad2.append(f"{s}/b={b}: {a_} 剖面不同")
    v["f_frozen"] = {"ok": not bad2, "n_seed": frz, "n_bad": len(bad2), "bad": bad2[:8],
                     "kB": {b: round(float(np.mean([r["k"]["B_frozen"] for r in by[b]])), 1) for b in BS},
                     "qbar_used": {b: round(float(np.mean([r["qbar_used"] for r in by[b]])), 3) for b in BS}}

    # 护栏 (d) 构造恒等
    ident = all(r["arms"][t]["sum_ok"] and r["arms"][t]["quota_ok"] for r in rows for t in ("Fu", "Uu", "Ku"))
    v["d_identity"] = {"ok": bool(ident and all(r["mirror_ok"] for r in rows)), "ident": bool(ident),
                       "mirror": bool(all(r["mirror_ok"] for r in rows))}
    # 护栏 (e) 触顶稀释
    v["e_nucap"] = {"ok": all(r["nu_cap"] <= NU_CAP_MAX for r in rows),
                    "nu": {b: round(float(np.mean([r["nu_cap"] for r in by[b]])), 3) for b in BS}}
    # 护栏 (g) 门控臂自己的 q̄（诊断：折叠改变门控读数多少；不进判决）
    v["g_qbar_own"] = {b: {"mean": round(float(np.mean([r["qbar"] for r in by[b]])), 3),
                           "ratio_vs_b40": round(float(np.mean([r["qbar"] for r in by[b]]))
                                                 / float(np.mean([r["qbar"] for r in by[40]])), 3)} for b in BS}
    # 护栏 (m) 控制口径 X 塌到 0
    mX = {}
    for b in BS:
        gx, mo = Gx[b], Gp[b]
        se = float(np.std(gx, ddof=1)) / np.sqrt(len(gx))
        mX[b] = {"G_X": round(float(np.mean(gx)), 6), "se": round(se, 6),
                 "ratio": round(abs(float(np.mean(gx))) / max(abs(float(np.mean(mo))), 1e-12), 4)}
    v["m_control"] = {"ok": all(abs(x["G_X"]) <= SE_K * x["se"] and x["ratio"] <= 0.25
                                for x in mX.values()), "cells": mX}

    # 标尺条款：U0 三档无差（3se）；基准取 b=40
    dU1 = _eff(U0[20], U0[40])
    dU2 = _eff(U0[4], U0[40])
    v["scale"] = {"ok": True, "level": "读数（DEC-41：本轮标尺允许动；护栏在结构层 A26M-3）",
                  "dU1": dU1["mean"], "k1": dU1["k"], "dU2": dU2["mean"], "k2": dU2["k"],
                  "dU2_over_3se": round(dU2["mean"] / dU2["k"], 3) if dU2["k"] else None,
                  "levels": {b: round(float(np.mean(U0[b])), 5) for b in BS},
                  "R_mu": {b: round(float(np.mean(Rp[b])), 4) for b in BS}}

    v["levels"] = {b: {"G": round(float(np.mean(Gp[b])), 5), "sd": round(float(np.std(Gp[b], ddof=1)), 5),
                       "U0": round(float(np.mean(U0[b])), 5), "R_mu": round(float(np.mean([r["R_O"] for r in by[b]])), 4),
                       "G_X": round(float(np.mean(Gx[b])), 5),
                       "k_B": round(float(np.mean([r["k"]["B_frozen"] for r in by[b]])), 1),
                       "qbar": round(float(np.mean([r["qbar"] for r in by[b]])), 3)} for b in BS}

    main = _tree(Gp[40], Gp[20], Gp[4], Rp[40], Rp[20], Rp[4])
    guards = ("c_anchor", "d_identity", "e_nucap", "f_frozen", "m_control")
    gok = all(v[g]["ok"] for g in guards)
    main["ok"] = bool(main["channel"].startswith("DIV") and gok)
    v["main"] = main
    v["guards_ok"] = gok
    v["n_seeds"] = len(used)
    v["verdict"] = ("DIV" if main["ok"] else
                    ("guards failed" if not gok else f"not DIV ({main['channel']})"))
    return v


# ══════════════════════════════════════════════ Stage A
def stage_a():
    print("=== Stage A（A26M-1 … A26M-14） ===")
    chk("A26M-1", md5("/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py") == DEV_MD5,
        f"装置 md5 == {DEV_MD5}")

    pairs = [(sys.modules["R1_gating"], "build_stream", ORIG_BUILD),
             (sys.modules["R1_gating"], "drive", drive),
             (sys.modules["R1_gating"], "make_core", make_core),
             (sys.modules["R1_gating"], "metrics_plas", metrics_plas),
             (sys.modules["R1_gating2"], "keep_positions", keep_positions),
             (sys.modules["R1_gating2"], "item_counts", item_counts),
             (sys.modules["R1_gating7"], "build_keep", build_keep),
             (sys.modules["R1_gating7"], "realized_profile", realized_profile),
             (sys.modules["R1_gating12"], "read_stream", read_stream),
             (sys.modules["R1_gating12"], "measure_prec", measure_prec),
             (sys.modules["R1_gating12"], "cal_at", cal_at),
             (sys.modules["R1_gating12"], "wsig", wsig)]
    bad = [f"{m.__name__}.{n}" for m, n, o in pairs if getattr(m, n) is not o]
    chk("A26M-2", not bad, f"12 个复用对象逐项同源（不同源：{bad}）")

    th = theta_of(0.9)
    struct = []
    D0 = ORIG_BUILD(SEEDS[0], D_MODEL, N_KV, R_PRES, th)
    for b in BS:
        D1 = build_stream_b(b, SEEDS[0], D_MODEL, N_KV, R_PRES, th)
        same = (torch.equal(D1["K"], D0["K"]) and torch.equal(D1["proto"], D0["proto"])
                and torch.equal(D1["va"][:N_READ], D0["va"][:N_READ])
                and torch.equal(D1["vb"][:N_READ], D0["vb"][:N_READ]))
        m_val = int(len(torch.unique(torch.round(D1["va"] * 1e6), dim=0)))
        struct.append({"b": b, "read_same": bool(same), "keys_same": bool(torch.equal(D1["K"], D0["K"])),
                       "m_val": m_val, "expect": N_READ + min(b, N_KV - N_READ)})
    chk("A26M-3", all(s["read_same"] and s["keys_same"] for s in struct),
        f"三档读集（K/proto/v^a/v^b）与键流逐位不变：{[(s['b'], s['read_same'], s['keys_same']) for s in struct]}")
    chk("A26M-4", all(s["m_val"] == s["expect"] for s in struct),
        f"世界值模式数 m=20+b 实测：{[(s['b'], s['m_val'], s['expect']) for s in struct]}")
    D40 = build_stream_b(40, SEEDS[0], D_MODEL, N_KV, R_PRES, th)
    chk("A26M-5", torch.equal(D40["va"], D0["va"]) and torch.equal(D40["vb"], D0["vb"]),
        "b=40 折叠是**恒等**（与未折叠逐位相同）")

    tau = cal_at(N_KV)["c_tau"]
    chk("A26M-6", 0.0 < tau < 1.0, f"c_tau(60)={tau:.4f}（三档共用同一冻结值）")
    chk("A26M-7", K.G5.N_DIST == N_DIST, f"N_DIST 已复位 {K.G5.N_DIST}")

    # A26M-8：派生单元在 b=40 / cnt_ref=None 时与 G12.run_unit **逐位相同**
    s8 = SEEDS[0]
    K.measure_hit = measure_hit_bounded
    g_ref = K.run_unit(N_KV, s8, tau)
    g_new = run_unit_m(s8, tau, 40, cnt_ref=None)
    diff = []
    for a_ in ARMS6:
        for f in ("issued", "mu", "ret1", "ho4_mean", "ho8_mean", "clean", "wsig", "W_unchanged"):
            if g_ref["arms"][a_][f] != g_new["arms"][a_][f]:
                diff.append(f"{a_}.{f}")
    for f in ("G_O", "U0_O", "G_X", "qbar", "qmax", "sched_dist", "mirror_ok", "counts"):
        if g_ref[f] != g_new[f]:
            diff.append(f)
    chk("A26M-8", not diff, f"派生单元 b=40/cnt_ref=None 与 `R1_gating12.run_unit` **逐位相同**（差异 {diff[:6]}）")

    # A26M-9：cnt_ref 生效 ⇒ k_B/剖面逐位冻结，且 qbar_used 与参考一致
    g20a = run_unit_m(s8, tau, 20, cnt_ref=g_ref["counts"])
    chk("A26M-9", g20a["k"]["B_frozen"] == g_ref["k"]["B"] and g20a["counts"] == g_ref["counts"]
        and g20a["sched_dist"] == g_ref["sched_dist"],
        f"外生 cnt 生效：k_B {g20a['k']['B_frozen']}=={g_ref['k']['B']}、cnt/日程逐位相同")

    # A26M-10：折叠确实改变了门控读数（否则独立变量没动）
    chk("A26M-10", g20a["qbar"] != g_ref["qbar"] or g20a["k"]["B"] != g_ref["k"]["B"],
        f"折叠改变了写侧门控读数（q̄ {g_ref['qbar']:.3f} → {g20a['qbar']:.3f}；"
        f"k_B_own {g_ref['k']['B']} → {g20a['k']['B']}）⇒ 独立变量真的在动")

    # A26M-11：G 分量与两条差值恒等式
    ok11 = abs(g_new["G_O"] - (-(g_new["arms"]["Fu"]["mu"] - g_new["arms"]["Ku"]["mu"]))) < 1e-15
    chk("A26M-11", ok11 and abs(g_new["R_O"] - g_new["G_O"] / g_new["U0_O"]) < 1e-12,
        "G=-（μ_F−μ_K）、R=G/μ_U0 恒等式成立")

    # A26M-12：装置层不变性（三单元素臂同 W0 指纹）
    chk("A26M-12", g_new["W0sig"] == g20a["W0sig"], "三档共用同一初值指纹 W0sig（同 seed）")

    # A26M-13：六臂读后 W 未被测量层改动
    chk("A26M-13", all(g_new["arms"][a_]["W_unchanged"] for a_ in ARMS6),
        "六臂读后 W 逐位不变（测量层无副作用）")

    # A26M-14：判据骨架在工作例上可复算（_eff 的 3se 与手算一致）
    d = _eff([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
    se = float(np.std([0.5, 1.5, 2.5], ddof=1)) / np.sqrt(3)
    chk("A26M-14", abs(d["se"] - se) < 1e-12 and abs(d["k"] - SE_K * se) < 1e-12,
        f"_eff 的 se/k 与手算一致（se={d['se']:.6f}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="DIR26Mr10")
    ap.add_argument("--seeds", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    outp = args.out or f"/home/ec2-user/sgmn-loop/dir26m/out/{args.tag}_{args.stage}.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    K.measure_hit = measure_hit_bounded

    if args.stage == "stageA":
        stage_a()
        ok = all(c["ok"] for c in res["checks"])
        print(f"\n{'★ Stage A 全部通过' if ok else '✗ Stage A 有 FAIL'}"
              f"（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        res["stage"], res["passed"] = "stageA", ok
    else:
        seeds = SEEDS[:args.seeds] if args.seeds else (SEEDS[:2] if args.stage == "smoke" else SEEDS)
        tau = cal_at(N_KV)["c_tau"]
        res["calib"] = {"c_tau": tau, "n_kv": N_KV}
        rows, done = [], set()
        if os.path.exists(outp):
            try:
                old = json.load(open(outp))
                rows = old.get("rows", [])
                done = {(r["b"], r["seed"]) for r in rows}
                print(f"  [resume] 已有 {len(rows)} 行")
            except Exception:
                rows, done = [], set()
        print(f"=== {args.stage}（b∈{BS} × {len(seeds)} 种子，cnt 外生冻结，tau={tau:.4f}）===")
        for seed in seeds:
            ref = run_unit_m(seed, tau, 40, cnt_ref=None)          # 原世界 = 参考 cnt
            if (40, seed) not in done:
                rows.append(ref); done.add((40, seed))
                json.dump({"rows": rows}, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
                print(f"  b=40 seed={seed} | G {ref['G_O']:+.5f} U0 {ref['U0_O']:+.5f} "
                      f"k_B {ref['k']['B']} q̄ {ref['qbar']:.3f}", flush=True)
            for b in (20, 4):
                if (b, seed) in done:
                    continue
                g = run_unit_m(seed, tau, b, cnt_ref=ref["counts"])
                rows.append(g); done.add((b, seed))
                json.dump({"rows": rows}, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
                print(f"  b={b:>2} seed={seed} | G {g['G_O']:+.5f} (b=40 的 {ref['G_O']:+.5f}) "
                      f"Δ {g['G_O']-ref['G_O']:+.5f} | k_B {g['k']['B_frozen']} q̄_own {g['qbar']:.3f} "
                      f"dist {g['sched_dist']:.3f} ν_cap {g['nu_cap']:.3f}", flush=True)
        rows.sort(key=lambda r: (r["b"], r["seed"]))
        res["rows"] = rows
        if args.stage == "formal":
            json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
            print(f"  [落盘] rows 已写入 {outp}（判据前）")
            try:
                res["verdicts"] = crit(rows)
                res["passed"] = res["verdicts"]["main"]["ok"]
            except Exception as e:
                res["verdicts"] = {"error": f"{type(e).__name__}: {e}"}
                res["passed"] = False
                print(f"  ✗ 判据异常：{type(e).__name__}: {e}")

    res["meta"] = {"wall_seconds": round(time.time() - t0, 1), "device": "cpu", "d": D_MODEL,
                   "n_kv": N_KV, "n_read": N_READ, "bs": list(BS), "r0": R0, "eta": ETA, "m": M,
                   "s_key": S_KEY, "r_pres": R_PRES, "r_ho": [R_HO4, R_HO8], "rho_main": R_MAIN,
                   "seeds": list(SEEDS), "qmax_cap": QMAX_CAP,
                   "profile": {"F": PROFILE["F"], "U0": PROFILE["U"], "K": PROFILE["K"]},
                   "stage": args.stage, "card": "DIR-26-M rev1.0",
                   "indep_var": "值模式折叠 pat(i)=20+((i-20) mod b)，读集 pat(i)=i 恒等",
                   "budget": "cnt 取自 b=40 同种子（外生冻结）"}
    res["stage"] = args.stage
    json.dump(res, open(outp, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"  [落盘] {outp}（{time.time()-t0:.1f}s）")


if __name__ == "__main__":
    main()
