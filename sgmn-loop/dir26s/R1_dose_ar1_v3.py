r"""DIR-26-S · R1_dose_ar1.py —— AR(1) 输入下，D-43 的剂量 $G$ 是否仍在原值上？

卡片：`revolve/DIR-26-S/DIRECTION_rev26s_v12.md`（rev 1.2 冻结；rev 1.0/1.1 留档）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 1a242d02c530307f6c76646ca13469ef，**零改动**）

派生纪律（只改独立变量与测量层增量）：
  ① **独立变量** $\varphi$：**只**替换**写流**的构造（DEF-26S-9）—— `_dispatch` 按调用点分派：
     $n==n_{\rm kv}{=}60$ 走 `R1_ar1.build_stream_ar1(..., φ)`；$n==20$ 的**读流保持原 `build_stream`**（标尺不变）。
  ② **臂集合**：`B`（必需：$\varphi{=}0$ 时三臂的配额由它的 `item_counts` 决定）+ `Fu/Uu/Ku`；`A`/`P` 与
     `R1_gating12.run_unit` **逐字相同地**保留（不改测量层）。
  ③ **测量层增量**：`phi`、`cnt0` 冻结（$\varphi{=}0$ 的 `B` 臂配额，两档共用）、逐种子配对差、Go 规则、功率落盘。
  **不改**：写流顺序、每条目写次数、值张量、读流、口径 O / 控制口径 X、剖面定义、$\tau$ 的运行时标定（`cal_at(60)`）。

★ ★ **rev 2.0（DIR-26-S′ 续卡）**：本文件是 `R1_dose_ar1.py` 的**逐字派生**，**只改两处常量**：
  ① `N_CAP` 40 → **52**（卡 `revolve/DIR-26S/DIRECTION_rev26s_v13.md` §5.4 / DEF-26S-12）；
  ② `EXTRA_SEEDS` 追加 **12** 个**开跑前冻结**的种子名（DEF-26S-13；步长 7 续接 20270721）。
  ⇒ ★ **判据、阈值、δ\*、Go 门限、3se 公式、通道定义、`run_unit_dose`/`crit`/`stage_a`/`go_rule` 全部一字未改**
  （逐字派生 ⇒ 位级可复现；旧文件 `R1_dose_ar1.py` **留档不改**）。
  ⇒ ★ 行键仍是 (φ, seed) ⇒ 已冻结的 40 行（30 个 φ=0 + 10 个 φ=0.9）**断点续用**。

★ **rev 2.1（DEF-26S-20）：本文件 = `R1_dose_ar1_v2.py` 的逐字派生，只改**一条护栏**。**
  老护栏（继承自 `dir26k/R1_gating12.py` 的 `measure_hit`）：
  ```
  assert (m["mu"] > 0) == (m["ret1"] > 0.5), "符号与命中不一致"
  ```
  ⇒ ★ **它不是恒等式**：`mu` 是 160 个边缘值的**均值符号**，`ret1` 是**多数符号**。
  反例（rev 2.0 正式轮第 90 行实际撞上）：`mu = +0.005133818`（>0）而 `ret1 = 0.49375`（=79/160<0.5）
  ⇒ 79 个命中但幅度更大、81 个未命中但幅度更小 ⇒ 均值符号与多数符号**合法地**分道扬镳。
  真正的恒等式是**逐样本**的：$m_{i,\rho}>0 \iff \arg\max$ 命中
  ⇒ ⇒ 改为断言 $\mathrm{frac}(m>0)$ 与 `ret1` 在**平局容差**内相等（见 `measure_hit_v2`）。
  ★ **这不是放宽**：新断言是**恒等式**（更严），旧断言是**启发式**（会误杀合法边界样本）。
  ★ **判据一字未动**：H26S 只用**均值**（$\Delta G$、$\Delta\mu_{U0}$、$\Delta R_\mu$）；`ret1` 从不进入判据。
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

for _p in ("dir26s", "dir26r", "dir26o", "dir26k", "dir26j", "dir26i", "dir26h",
           "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G
import R1_gating2 as G2
import R1_gating5 as G5
import R1_gating12 as K12
import R1_ar1 as RA
from R1_gating import D_MODEL, M, N_DIST as N_DIST_BASE, drive, make_core, unit
from R1_gating2 import item_counts
from R1_gating5 import theta_of
from R1_gating7 import PROFILE, build_keep, chk, md5, realized_profile, res
from R1_gating10 import ret_jitter_margin
from R1_gating12 import (N_READ, QMAX_CAP, R0, R_HO4, R_HO8, R_PRES,
                         SEEDS, SE_K, cal_at, measure_hit, measure_prec, read_stream, wsig)

# ══════════════════════════════════════════ rev 2.1：逐样本恒等式护栏（DEF-26S-20）
def _hit_pos_frac(P, Kq, vq, n_read, r_q):
    r"""$\mathrm{frac}(\{m_{i,\rho}>0\})$ 与**平局计数**。

    ★ 与 `R1_gating10.ret_jitter_margin` **同一套算式**（同一 $S$、同一 `other.max()`），
    只额外统计「$m>0$」与「$m=0$」的个数 ⇒ 不改变任何 $\mu$ / `ret1` 的值。
    """
    vbU = unit(vq)
    n_pos = n_tie = 0
    for rho in range(r_q):
        Y = P @ Kq[rho].T
        S = unit(Y.T) @ vbU.T
        for i in range(n_read):
            row = S[i]
            other = torch.cat([row[:i], row[i + 1:]]) if i + 1 < n_read else row[:i]
            mm = float(row[i] - other.max())
            if mm > 0:
                n_pos += 1
            elif mm == 0:
                n_tie += 1
    n = n_read * r_q
    return n_pos, n_tie, n


def measure_hit_v2(P, Kq, vq, n_read, r_q):
    r"""rev 2.1 的命中护栏：**逐样本恒等式**替代 rev 2.0 的均值/多数符号启发式。

    允许的唯一偏差来自**平局**：平局时 `argmax` 取首列 ⇒ 若恰好命中则 `hit=True` 但 $m=0$（不计入正）
    ⇒ ⇒ 允许 `ret1 - frac(m>0) <= n_tie / n`。
    """
    m = ret_jitter_margin(P, Kq, vq, n_read, r_q)
    assert m["n_read"] == n_read * r_q, f"μ 样本数不符：{m['n_read']} != {n_read}×{r_q}"
    n_pos, n_tie, n = _hit_pos_frac(P, Kq, vq, n_read, r_q)
    pos_frac = n_pos / float(n)
    gap = float(m["ret1"]) - pos_frac                      # 只允许 ≥0（平局方向）
    assert gap >= -1e-12, (f"逐样本恒等式被破坏（命中多于正边缘）：ret1 {m['ret1']} < "
                          f"frac(m>0) {pos_frac}（n_pos {n_pos}/{n}）")
    assert gap <= (n_tie + 1e-12) / n, (f"逐样本恒等式被破坏（偏差超出平局容差）："
                                        f"ret1 {m['ret1']} - frac(m>0) {pos_frac} = {gap:.6e} "
                                        f"> n_tie/n = {n_tie}/{n}")
    m["pos_frac"] = pos_frac
    m["n_tie"] = int(n_tie)
    return m


measure_hit = measure_hit_v2                              # ★ 全局重绑：所有调用点自动生效

# ══════════════════════════════════════════════ 冻结常数（卡 §1.1 / §2.1 / §5.5）
N_KV = 60
PHIS = (0.0, 0.9)
DELTA_STAR = 0.00162                 # D-43 的 Δ1 = G35 − G20（归档判为 SIG 的较小步）
GO_TOL = DELTA_STAR / 2              # 0.00081 —— Go 门限（3se(ΔG) ≤ GO_TOL）
# ★ rev 2.0（卡 §5.4 / DEF-26S-12）：**上限 40→52** —— 冻结公式用冻结 pilot sd 算出的
#   n_required = ceil((3*0.0019371/0.00081)^2) = 52 ⇒ 上限是当时唯一的阻塞点。
#   ★ 公式、门限、3se 口径、INFEASIBLE-POWER 含义**一字未改**。
N_CAP = 52
# ★ rev 2.0（卡 §5.5 / DEF-26S-13）：追加种子 22 个 = rev 1.2 的 10 个 + **新冻结 12 个**
#   （步长 7 续接 20270721；全仓零碰撞已核）。★ 种子名**开跑前冻结**，不按结果挑选。
EXTRA_SEEDS_10 = [20270519, 20270526, 20270602, 20270609, 20270616,
                  20270623, 20270630, 20270707, 20270714, 20270721]   # §5.5 rev 1.2 追加 10
EXTRA_SEEDS_12 = [20270728, 20270804, 20270811, 20270818, 20270825, 20270901,
                  20270908, 20270915, 20270922, 20270929, 20271006, 20271013]  # ★ rev 2.0 新冻结 12
EXTRA_SEEDS = EXTRA_SEEDS_10 + EXTRA_SEEDS_12
DEV = "/home/ec2-user/sgmn-loop/dir22/sgmn_module_hebb_v7.py"
DERIVED = {"R1_gating": "/home/ec2-user/sgmn-loop/dir26/R1_gating.py",
           "R1_gating12": "/home/ec2-user/sgmn-loop/dir26k/R1_gating12.py",
           "R1_gating7": "/home/ec2-user/sgmn-loop/dir26g/R1_gating7.py",
           "R1_ar1": "/home/ec2-user/sgmn-loop/dir26o/R1_ar1.py"}
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
DERIVED_MD5 = {"R1_gating": "5f6513d5a1885b256201af072cfdfb2b",
               "R1_gating12": "16bbb8df05f4074e053ea94da53fa0ca",
               "R1_gating7": "837b3dec199e534551e81c50b37e9da3",
               "R1_ar1": "a3c0981bbf8e09a92b2b5411742d76f1"}
D43 = "/home/ec2-user/sgmn-loop/dir26k/out/DIR26Kr10_formal.json"
OUT = "/home/ec2-user/sgmn-loop/dir26s/out"
ROWS_F = "DIR26S_rows.json"
EDGE_TOL = 3e-4                      # A26S-3：边缘矩相对极差
RHO_HI = 0.85                        # A26S-4：φ=0.9 的 60 条目均值下界（D-47 交付 0.8999 −0.05）
RHO_LO = 0.10                        # A26S-4：φ=0 的上界
NU_CAP_MAX = 0.10
PILOT_N = 10                         # S3 的 pilot 种子数（§5.6）

# ══════════════════════════════════════════════ 写流分派（DEF-26S-9）
_ORIG_BUILD = K12.build_stream
_PHI = {"cur": 0.0}


def _dispatch(seed, d, n, r_pres, theta):
    r"""写流（$n=n_{\rm kv}$）走 AR(1)；读流（$n=N_{\rm read}$）走**原** `build_stream`。"""
    if int(n) == N_KV:
        return RA.build_stream_ar1(N_KV, seed, d, int(n), r_pres, theta, _PHI["cur"])
    return _ORIG_BUILD(seed, d, n, r_pres, theta)


def set_phi(phi):
    _PHI["cur"] = float(phi)
    K12.build_stream = _dispatch
    G.build_stream = _dispatch


def build_write(seed, phi, n_kv=N_KV):
    set_phi(phi)
    return _dispatch(int(seed), D_MODEL, int(n_kv), R_PRES, theta_of(0.9))


def tsig(t):
    return hashlib.md5(np.ascontiguousarray(t.detach().numpy(), dtype=np.float32).tobytes()).hexdigest()[:16]


def keysig(D):
    return {"K": tsig(D["K"]), "proto": tsig(D["proto"]), "va": tsig(D["va"]), "vb": tsig(D["vb"])}


# ══════════════════════════════════════════════ 单元（逐字派生自 R1_gating12.run_unit）
def run_unit_dose(seed, phi, tau, cnt0=None, n_kv=N_KV):
    r"""一条 (seed, φ) 单元 = 6 驱动（A/B/P/Fu/Uu/Ku），与 `K12.run_unit` 逐字相同，**只两处改动**：
    ① 写流来自 AR(1) 分派；② 三臂配额 `cnt` 可外部传入（`cnt0` 冻结，卡 §6-2）。"""
    seed, n_kv = int(seed), int(n_kv)
    tau = float(tau["c_tau"]) if isinstance(tau, dict) else float(tau)
    D = build_write(seed, phi, n_kv)
    assert int(D["n"]) == n_kv, f"写流 N 不符：{D['n']} != {n_kv}"
    RS8, RS4 = read_stream(seed, R_HO8), read_stream(seed, R_HO4)      # 读流：原 iid（标尺不变）
    proto = D["proto"][:N_READ]
    out = {"n_kv": n_kv, "n_read": N_READ, "seed": seed, "phi": float(phi),
           "T": R_PRES * n_kv * 2 * M, "arms": {}, "prof": {}, "k": {}, "qbar": 0.0,
           "diag": D["diag"], "keysig": keysig(D)}

    vb = D["vb"][:N_READ]

    def rec(tag, core, issued):
        W = core._W().detach()
        W_before = W.clone()
        P = W - core.W0
        m8 = measure_hit(P, RS8["K"], vb, N_READ, R_HO8)          # 判据口径 O
        p4 = measure_prec(P, RS4["K"], vb, N_READ, R_HO4)         # ho4 精度（报告）
        mx = measure_hit(P, RS8["K"], RS8["vb"], N_READ, R_HO8)   # 控制口径 X（外来值）
        # ★ 卡 A26S-10：一致性必须与**同一条读流、同一 R_HO8** 的精度口径比
        consistent = (m8["ret1"] == measure_prec(P, RS8["K"], vb, N_READ, R_HO8)["ho_mean"])
        out["arms"][tag] = {
            "issued": issued, "wsig": wsig(W),
            "W_unchanged": bool(torch.equal(core._W().detach(), W_before)),
            "clean": G.metrics_plas(W, core.W0, proto, vb, N_READ)["ret1"],
            "mu": m8["mu"], "mu_sd": m8["mu_sd"], "mu_read": m8["n_read"],
            "ret1": m8["ret1"], "ret1_maj": m8["ret1_maj"], "maj_need": m8["maj_need"],
            "ho8_mean": m8["ret1"], "ho8_maj": m8["ret1_maj"],
            "ho4_mean": p4["ho_mean"], "ho4_maj": p4["ho_maj"],
            "mu_ret1_consistent": bool(consistent),
            "X": {"mu": mx["mu"], "mu_sd": mx["mu_sd"], "ret1": mx["ret1"]},
        }

    c = make_core(D_MODEL, seed, R0); kA, _ = drive(c, D, "A"); rec("A", c, kA)
    c = make_core(D_MODEL, seed, R0); kB, logB = drive(c, D, "B", tau=tau); rec("B", c, kB)
    if cnt0 is None:
        cnt = item_counts({t for t, e in enumerate(logB) if e["issued"]}, n_kv)
        out["cnt_source"] = "B@this_phi"
    else:
        cnt = [int(x) for x in cnt0]
        out["cnt_source"] = "frozen_cnt0"
    out["k"] = {"A": kA, "B": kB}
    out["qbar"] = kB / n_kv
    out["qmax"] = int(max(cnt))
    out["counts"] = cnt
    out["nu_cap"] = float(np.mean(np.array(cnt[:N_READ]) >= QMAX_CAP))
    out["W0sig"] = wsig(make_core(D_MODEL, seed, R0)._W().detach())

    c = make_core(D_MODEL, seed, R0)
    kP, _ = drive(c, D, "R", keep=G2.keep_positions(n_kv, R_PRES, kB)); rec("P", c, kP)

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


# ══════════════════════════════════════════════ 行存档（断点续跑）
def rows_path():
    return os.path.join(OUT, ROWS_F)


def load_rows():
    p = rows_path()
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    return {(float(r["phi"]), int(r["seed"])): r for r in d["rows"]}


def save_rows(by):
    os.makedirs(OUT, exist_ok=True)
    json.dump({"rows": [by[k] for k in sorted(by)]}, open(rows_path(), "w"))


# ══════════════════════════════════════════════ Stage A
def stage_a(by):
    print("=== Stage A（A26S-1…14；A26S-6 在 S2 之后） ===")
    t0 = time.time()
    chk("A26S-1", md5(DEV) == DEV_MD5, f"装置 md5 == {DEV_MD5}")
    got = {k: md5(p) for k, p in DERIVED.items()}
    chk("A26S-2", got == DERIVED_MD5, f"派生链 md5 与冻结值一致：{ {k: v[:8] for k, v in got.items()} }")

    s0 = SEEDS[0]
    tau = cal_at(N_KV)
    chk("A26S-tau", G5.N_DIST == N_DIST_BASE and bool(tau["gate_ok"]),
        f"τ 由 cal_at(60) 运行时取得（卡 §6-8 禁止手抄）：c_tau={tau['c_tau']:.4f} "
        f"skip={tau['skip_rate']:.3f} gap={tau['gap']:+.3f} rep={tau['skip_frac_repeat']:.3f}；"
        f"N_DIST 已复位 {G5.N_DIST} == {N_DIST_BASE}")

    D0, D9 = build_write(s0, 0.0), build_write(s0, 0.9)
    k0, k9 = keysig(D0), keysig(D9)

    m0, m9 = D0["diag"]["km"], D9["diag"]["km"]
    rel = {f: abs(m9[f] - m0[f]) / max(abs(m0[f]), 1e-30) for f in ("k2", "cosp", "k4")}
    edge_ok = all(v <= EDGE_TOL for v in rel.values())
    chk("A26S-3", edge_ok, "边缘不变（W2 前半）：" + " ".join(f"{f}: {m0[f]:.10f} → {m9[f]:.10f} (rel {v:.2e})"
        for f, v in rel.items()) + f" ≤ {EDGE_TOL}")

    ru0, ru9 = D0["diag"]["rho_u"], D9["diag"]["rho_u"]
    chk("A26S-4", ru9 >= RHO_HI and ru0 <= RHO_LO,
        f"前提护栏：60 条目均值的抖动 lag-1 相关 ρ̄_u(0.9)={ru9:.4f} ≥ {RHO_HI}、ρ̄_u(0)={ru0:.4f} ≤ {RHO_LO}"
        f"（依据 D-47 交付值 0.8999；带下界 −0.05）")

    same_pv = all(k0[f] == k9[f] for f in ("proto", "va", "vb"))
    chk("A26S-5", same_pv and k0["K"] != k9["K"],
        f"配对（W2 后半）：proto/va/vb 逐位{same_pv and '相等' or '不等'} "
        f"({k0['proto'][:8]}/{k9['proto'][:8]})，K {'不同' if k0['K'] != k9['K'] else '相同'}"
        f" ({k0['K'][:8]}/{k9['K'][:8]})")

    # ── 单元级断言：φ=0.9 跑一次（A26S-8/9/10）+ φ=0 跑两次（A26S-12）
    g9 = run_unit_dose(s0, 0.9, tau)
    three = ("Fu", "Uu", "Ku")
    iso = (len({g9["W0sig"]}) == 1
           and len({g9["arms"][t]["issued"] for t in three}) == 1
           and len({g9["arms"][t]["keep_n"] for t in three}) == 1
           and all(g9["arms"][t]["sum_ok"] and g9["arms"][t]["quota_ok"] for t in three))
    chk("A26S-8", iso, f"三臂副作用隔离：W0 指纹 {g9['W0sig'][:8]}、k {g9['arms']['Fu']['issued']}、"
        f"keep_n {g9['arms']['Fu']['keep_n']}、sum/quota 全 True；q̄ {g9['qbar']:.3f} q_max {g9['qmax']}")

    chk("A26S-9", g9["mirror_ok"] and g9["sched_dist"] > 0,
        f"剖面镜像 {g9['mirror_ok']}（≤1e-12）、日程距离 {g9['sched_dist']:.4f} > 0；"
        f"F {[round(x, 3) for x in g9['prof']['Fu']]} vs K {[round(x, 3) for x in g9['prof']['Ku']]}")

    cons = all(v["mu_ret1_consistent"] for v in g9["arms"].values())
    chk("A26S-10", cons, f"口径 O 护栏：六臂 mu_ret1_consistent 全 True = {cons}")

    # A26S-7 工作例 W1：手工复核判据量（φ=0.9、Fu/Ku 两臂、seed=SEEDS[0]）
    w1 = worked_example(s0, g9)
    chk("A26S-7", w1["ok"], w1["detail"])
    res["worked"] = w1

    a = run_unit_dose(s0, 0.0, tau)
    by[(0.0, int(s0))] = a
    save_rows(by)
    b = run_unit_dose(s0, 0.0, tau)
    det = all(a[f] == b[f] for f in ("G_O", "U0_O", "R_O", "qmax", "mirror_ok", "W0sig")) \
        and all(a["arms"][t]["wsig"] == b["arms"][t]["wsig"] for t in ("Fu", "Uu", "Ku"))
    chk("A26S-12", det, f"确定性：同 (seed={s0},φ=0) 两次 G_O {a['G_O']:+.12f} / {b['G_O']:+.12f} 逐位"
        f"{'相同' if det else '不同'}；三臂 W 指纹亦一致")

    print(f"  [REPORT] A26S-11（rev 1.1 起为纯报告字段、不判）φ=0 s0：q̄ {a['qbar']:.3f} "
          f"q_max {a['qmax']} ν_cap {a['nu_cap']:.3f}；G_O {a['G_O']:+.6f} U0_O {a['U0_O']:+.6f} "
          f"R_O {a['R_O']:+.4f}（量程由 A26S-6 的 90 值对锚点覆盖）")

    chk("A26S-14", len(res["checks"]) >= 10, f"断言集合非空：{len(res['checks'])} 条（纪律 ㊱）")
    print(f"  Stage A 用时 {time.time() - t0:.1f} s")
    return all(c["ok"] for c in res["checks"])


def worked_example(seed, g9):
    r"""W1：打印 3×3 的 cos(y_i,v_j) 阵与逐行 max_{j≠i}，手工重算 μ、G、R_μ 并断言逐位相等。"""
    D = build_write(seed, 0.9)
    RS8 = read_stream(seed, R_HO8)
    cnt = g9["counts"]
    vb = D["vb"][:N_READ]
    vbU = unit(vb)
    rows, tbl, mus = {}, [], {}
    for pk in ("F", "K"):
        keep, _ = build_keep(N_KV, R_PRES, cnt, PROFILE[pk])
        c = make_core(D_MODEL, seed, R0)
        drive(c, D, "R", keep=keep)
        P = c._W().detach() - c.W0
        m = ret_jitter_margin(P, RS8["K"], vb, N_READ, R_HO8)
        ms = []
        for rho in range(R_HO8):
            Y = P @ RS8["K"][rho].T
            S = unit(Y.T) @ vbU.T
            for i in range(N_READ):
                row = S[i]
                other = torch.cat([row[:i], row[i + 1:]]) if i + 1 < N_READ else row[:i]
                ms.append(float(row[i] - other.max()))
            if rho == 0 or rho == 1:
                for i in (0, 7, 19):
                    row = S[i]
                    other = torch.cat([row[:i], row[i + 1:]])
                    tbl.append({"arm": pk, "rho": rho, "i": i,
                                "cos": [round(float(x), 6) for x in row],
                                "max_other": round(float(other.max()), 6),
                                "m": round(float(row[i] - other.max()), 6)})
        mus[pk] = {"mu_re": float(np.mean(ms)), "mu_ref": m["mu"], "n": len(ms), "n_ref": m["n_read"]}
    G_re = -(mus["F"]["mu_re"] - mus["K"]["mu_re"])
    G_ok = all(mus[p]["mu_re"] == mus[p]["mu_ref"] and mus[p]["n"] == mus[p]["n_ref"] for p in mus)
    g_ok = G_re == g9["G_O"]
    R_re = G_re / g9["U0_O"]
    r_ok = R_re == g9["R_O"]
    detail = (f"W1：μ_F {mus['F']['mu_re']:+.12f}=={mus['F']['mu_ref']:+.12f} "
              f"μ_K {mus['K']['mu_re']:+.12f}=={mus['K']['mu_ref']:+.12f}（160 样本）；"
              f"G 重算 {G_re:+.12f} vs 脚本 {g9['G_O']:+.12f}；R_μ 重算 {R_re:+.12f} vs {g9['R_O']:+.12f}；"
              f"抽查 (arm={tbl[0]['arm']},ρ={tbl[0]['rho']},i={tbl[0]['i']}) cos 行 {tbl[0]['cos']} "
              f"max_other {tbl[0]['max_other']:+.6f} ⇒ m {tbl[0]['m']:+.6f}")
    return {"ok": bool(G_ok and g_ok and r_ok), "detail": detail, "table": tbl, "mus": mus,
            "G_re": G_re, "R_re": R_re}


# ══════════════════════════════════════════════ Stage A26S-6（位级锚点，90 值对）
def anchor_90(by):
    ref = {(r["n_kv"], r["n_read"], r["seed"]): r
           for r in json.load(open(D43))["rows"] if r["n_kv"] == N_KV}
    ncmp, bad = 0, []
    for s in SEEDS:
        r = by.get((0.0, int(s)))
        if r is None:
            bad.append(f"{s}:missing")
            continue
        R = ref[(N_KV, N_READ, int(s))]
        for f in ("G_O", "U0_O", "R_O"):
            ncmp += 1
            if r[f] != R[f]:
                bad.append(f"{s}.{f}: {r[f]!r} != {R[f]!r}")
    return {"ok": (not bad and ncmp == 3 * len(SEEDS)), "n_cmp": ncmp, "n_seeds": len(SEEDS), "bad": bad[:8]}


# ══════════════════════════════════════════════ 判据
def _eff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": float(np.mean(d)), "sd": sd, "se": se, "k": SE_K * se, "n": n,
            "sig": bool(np.isfinite(se) and abs(float(np.mean(d))) >= SE_K * se),
            "vals": [round(float(x), 8) for x in d]}


def go_rule(sd_dg):
    n_req = int(math.ceil((SE_K * sd_dg / GO_TOL) ** 2)) if sd_dg > 0 else 10 ** 9
    if n_req <= 30:
        n_use = 30
    elif n_req <= N_CAP:
        n_use = n_req
    else:
        n_use = None
    return {"sd_dG_pilot": sd_dg, "n_required": n_req, "n_use": n_use,
            "go": n_use is not None, "tol": GO_TOL, "delta_star": DELTA_STAR,
            "pilot_n": PILOT_N, "cap": N_CAP}


def crit(by, go, pool=None):
    v = {}
    seeds = [int(s) for s in (pool or SEEDS)]
    have0 = [s for s in seeds if (0.0, s) in by]
    have9 = [s for s in seeds if (0.9, s) in by]
    used = [s for s in seeds if s in have0 and s in have9]
    v["n_rows"] = {"phi0": len(have0), "phi9": len(have9), "paired": len(used)}

    anc = anchor_90(by)
    v["a_anchor"] = anc

    edge = {}
    for s in used[:PILOT_N]:
        d0, d9 = by[(0.0, s)]["diag"], by[(0.9, s)]["diag"]
        edge[s] = {f: abs(d9["km"][f] - d0["km"][f]) / max(abs(d0["km"][f]), 1e-30) for f in ("k2", "cosp", "k4")}
    edge_max = max((max(x.values()) for x in edge.values()), default=float("inf"))
    v["b_edge"] = {"ok": bool(edge_max <= EDGE_TOL), "max_rel": edge_max}

    ru = {s: (by[(0.0, s)]["diag"]["rho_u"], by[(0.9, s)]["diag"]["rho_u"]) for s in used}
    ru_lo = min((x[1] for x in ru.values()), default=float("nan"))
    ru_hi = max((x[0] for x in ru.values()), default=float("nan"))
    v["c_rho"] = {"ok": bool(ru_lo >= RHO_HI and ru_hi <= RHO_LO), "rho0_max": ru_hi, "rho9_min": ru_lo,
                  "band": [RHO_HI, RHO_LO]}

    ident = all(by[k]["arms"][t]["sum_ok"] and by[k]["arms"][t]["quota_ok"]
                for k in by for t in ("Fu", "Uu", "Ku"))
    v["d_identity"] = {"ok": bool(ident and all(by[k]["mirror_ok"] and by[k]["sched_dist"] > 0 for k in by))}
    v["e_cnt"] = {"ok": all(by[(0.9, s)]["cnt_source"] == "frozen_cnt0" for s in have9)
                  and all(by[(0.0, s)]["cnt_source"] == "B@this_phi" for s in have0),
                  "note": "φ=0.9 的三臂配额必须来自 φ=0 的 B 臂（卡 §6-2）"}
    v["f_consistent"] = {"ok": all(x["mu_ret1_consistent"] for k in by for x in by[k]["arms"].values())}
    v["g_nucap"] = {"ok": all(by[k]["nu_cap"] <= NU_CAP_MAX for k in by)}
    guards = all(v[k]["ok"] for k in ("a_anchor", "b_edge", "c_rho", "d_identity", "e_cnt", "f_consistent", "g_nucap"))

    dG = _eff([by[(0.9, s)]["G_O"] for s in used], [by[(0.0, s)]["G_O"] for s in used])
    dU = _eff([by[(0.9, s)]["U0_O"] for s in used], [by[(0.0, s)]["U0_O"] for s in used])
    dR = _eff([by[(0.9, s)]["R_O"] for s in used], [by[(0.0, s)]["R_O"] for s in used])
    v["dG"], v["dU0"], v["dR"] = dG, dU, dR

    lv = {}
    for p, tag in ((0.0, "phi0"), (0.9, "phi9")):
        G_ = [by[(p, s)]["G_O"] for s in seeds if (p, s) in by]
        U_ = [by[(p, s)]["U0_O"] for s in seeds if (p, s) in by]
        R_ = [by[(p, s)]["R_O"] for s in seeds if (p, s) in by]
        lv[tag] = {"G": float(np.mean(G_)), "sd_G": float(np.std(G_, ddof=1)), "n": len(G_),
                   "U0": float(np.mean(U_)), "R": float(np.mean(R_)),
                   "G_min": float(np.min(G_)), "G_max": float(np.max(G_)),
                   "U0_min": float(np.min(U_)), "U0_max": float(np.max(U_)),
                   "R_min": float(np.min(R_)), "R_max": float(np.max(R_)),
                   "rho_u": float(np.mean([by[(p, s)]["diag"]["rho_u"] for s in seeds if (p, s) in by])),
                   "qbar": float(np.mean([by[(p, s)]["qbar"] for s in seeds if (p, s) in by]))}
    v["levels"] = lv

    gx = {tag: {"G_X": float(np.mean([by[(p, s)]["G_X"] for s in seeds if (p, s) in by])),
                "U0_X": float(np.mean([by[(p, s)]["U0_X"] for s in seeds if (p, s) in by]))}
          for p, tag in ((0.0, "phi0"), (0.9, "phi9"))}
    gx["ratio"] = abs(gx["phi9"]["G_X"]) / max(abs(lv["phi9"]["G"]), 1e-12)
    v["m_control"] = gx

    if not go["go"]:
        chan = "INFEASIBLE-POWER"
    elif not guards:
        chan = "IND"
    elif dG["sig"]:
        chan = "DOS"
    else:
        chan = "INV"
    v["channel"] = chan
    v["falsified"] = {"F26S-1": bool(dG["sig"]), "F26S-2_ind": bool(not guards)}
    v["report_labels"] = {"RATIO-REPORT": bool(dR["sig"]), "SCALE-REPORT": bool((not dR["sig"]) and dU["sig"])}
    v["dose_step_for_scale"] = {"delta_star": DELTA_STAR, "go_tol": GO_TOL,
                                "dG_vs_tol": abs(dG["mean"]) / GO_TOL if GO_TOL else float("nan")}
    return v


# ══════════════════════════════════════════════ 主流程
def run_rows(seeds, phi, tau, by, cnt0_of=None):
    for s in seeds:
        if (float(phi), int(s)) in by:
            print(f"  [skip] φ={phi} seed={s}")
            continue
        t = time.time()
        cnt0 = None if cnt0_of is None else cnt0_of.get(int(s))
        r = run_unit_dose(s, phi, tau, cnt0=cnt0)
        by[(float(phi), int(s))] = r
        save_rows(by)
        print(f"  [done] φ={phi} seed={s}  G_O {r['G_O']:+.6f} U0_O {r['U0_O']:+.6f} "
              f"R_O {r['R_O']:+.4f} ρ̄_u {r['diag']['rho_u']:.4f} q̄ {r['qbar']:.2f} ({time.time() - t:.1f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["s1", "s2", "s3", "s4", "s1s3", "all"], required=True)
    ap.add_argument("--tag", default="DIR26S")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t00 = time.time()
    by = load_rows()
    tau = cal_at(N_KV)
    print(f"τ(60) = {tau['c_tau']:.4f}（cal_at 运行时取得）；已有行 {len(by)}")

    if args.stage in ("s1", "s1s3", "all"):
        ok = stage_a(by)
        json.dump(res, open(f"{OUT}/{args.tag}_stageA.json", "w"), indent=1)
        print(f"{'★ Stage A 全 PASS' if ok else '✗ Stage A 有 FAIL'}"
              f"（{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}）")
        if not ok:
            print("⇒ 阻断：不进入 S2"); return

    if args.stage in ("s2", "s1s3", "all"):
        print("=== S2：φ=0 全 30 种子（φ=0 的 B 臂配额 = cnt0） ===")
        run_rows(SEEDS, 0.0, tau, by)
        anc = anchor_90(by)
        json.dump(anc, open(f"{OUT}/{args.tag}_anchor90.json", "w"), indent=1)
        print(f"  A26S-6 位级锚点：{anc['n_cmp']} 值对（{anc['n_seeds']} 种子 × 3 字段）"
              f" ⇒ {'PASS' if anc['ok'] else 'FAIL ' + str(anc['bad'])}")
        if not anc["ok"]:
            print("⇒ 阻断：锚点失败，不进入 S3"); return

    if args.stage in ("s3", "s1s3", "all"):
        print(f"=== S3：φ=0.9 的 pilot 前 {PILOT_N} 种子（cnt0 由 S2 传入） ===")
        by = load_rows()
        cnt0_of = {int(s): by[(0.0, int(s))]["counts"] for s in SEEDS if (0.0, int(s)) in by}
        run_rows(SEEDS[:PILOT_N], 0.9, tau, by, cnt0_of)
        by = load_rows()
        pilot = [int(s) for s in SEEDS[:PILOT_N] if (0.9, int(s)) in by and (0.0, int(s)) in by]
        d = [by[(0.9, s)]["G_O"] - by[(0.0, s)]["G_O"] for s in pilot]
        sd_dg = float(np.std(d, ddof=1))
        go = go_rule(sd_dg)
        go["pairs"] = [round(x, 8) for x in d]
        go["seeds"] = pilot
        go["note"] = "pilot 只贡献 sd；均值不得进入任何阈值（决定十六-2）"
        json.dump(go, open(f"{OUT}/DIR26S_power.json", "w"), indent=1)
        print(f"  pilot sd(ΔG) = {sd_dg:.6f} ⇒ n_required {go['n_required']} ⇒ n_use {go['n_use']}"
              f" ⇒ Go {'PASS' if go['go'] else '**FAIL ⇒ INFEASIBLE-POWER，不跑**'}")
        print(f"  (3se@n_use = {SE_K * sd_dg / math.sqrt(go['n_use'] or N_CAP):.6f} vs 门限 {GO_TOL})")

    if args.stage in ("s4", "all"):
        go = json.load(open(f"{OUT}/DIR26S_power.json"))
        if not go["go"]:
            print("⇒ S4 冻结不跑（Go 失败）；本方向记 INFEASIBLE-POWER")
            return
        by = load_rows()
        n_use = int(go["n_use"])
        pool = (list(SEEDS) + list(EXTRA_SEEDS))[:n_use]
        extra = [s for s in pool if s not in SEEDS]
        if extra:
            print(f"=== S4a：追加种子 {extra} 的 φ=0（产生各自的 cnt0） ===")
            run_rows(extra, 0.0, tau, by)
            by = load_rows()
        cnt0_of = {int(s): by[(0.0, int(s))]["counts"] for s in pool if (0.0, int(s)) in by}
        print(f"=== S4：φ=0.9 补齐到 n={n_use} ===")
        run_rows([s for s in pool if s not in SEEDS[:PILOT_N]], 0.9, tau, by, cnt0_of)
        by = load_rows()
        v = crit(by, go, pool)
        json.dump(v, open(f"{OUT}/{args.tag}_verdict.json", "w"), indent=1)
        print(f"  通道 = **{v['channel']}**")
        print(f"  ΔG {v['dG']['mean']:+.6f} ± 3se {v['dG']['k']:.6f}（n={v['dG']['n']}）"
              f" | ΔU0 {v['dU0']['mean']:+.6f} ± {v['dU0']['k']:.6f} | ΔR {v['dR']['mean']:+.6f} ± {v['dR']['k']:.6f}")
        print(f"  护栏：锚点 {v['a_anchor']['ok']} 边缘 {v['b_edge']['ok']} ρ̄_u {v['c_rho']['ok']} "
              f"恒等 {v['d_identity']['ok']} cnt0 {v['e_cnt']['ok']} 一致 {v['f_consistent']['ok']} "
              f"ν_cap {v['g_nucap']['ok']}")
        print(f"  报告标签：{v['report_labels']}")
    print(f"总用时 {time.time() - t00:.1f} s")


if __name__ == "__main__":
    main()
