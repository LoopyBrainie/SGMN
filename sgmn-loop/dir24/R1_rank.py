#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIR-24 · R1（**`W⁰` 的构造：初始增量的秩 $r_0$**）

判据见 `../revolve/DIR-24/DIRECTION_rev24_v10.md`（**rev 1.0 冻结**）：H24 四子句 (a)–(d) + 诊断条款 (e)
+ Stage A `A24-1…A24-8`。**阈值一字不改。**

装置：`dir24/sgmn_module_hebb_v7.py` = `dir22/`（= `dir23/`）同名文件，**md5 完全相同**（DEC-09 零改动）。
秩约束**不改装置**：只把 `U_0,V_0` 的首 $r_0$ 列之外置零 ⇒ 由 §3.3 因子步形成不变子空间
⇒ $\\operatorname{rank}W_t\\le r_0$ 是**精确不变式**（DEC-15；工作算例 6/6 水平实测）。

派生纪律：本脚本**不 import 上游脚本**；`unit` / `item_z` / `best_q` / `_norm_cdf` / `sign_test_p`
从 `dir23/R1_scope.py` **逐字复制**（`patch_r1_rank.py` 用 AST 逐字核验，不一致即拒跑）。
`build_A` 从 `dir23/R1_scope.py` 派生：**去掉 $S$ 相关字段**（$u$）、把 $d$ 与 $N_{\\text{pair}}$ 从常量改为参数
—— 这正是卡片 §6 允许的"只改测量层与自变量"。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sgmn_module_hebb_v7 import HebbianCore   # noqa: E402

# ══════════════════════════════════════════════════════════ 常量（卡片 §6 冻结）
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]
ALPHA, LAM, ETA, M = 1.0, 0.05, 0.1, 2
D_LEVELS = [896, 64]                       # 第二轴：896 参与主判据；64 只服务 (e)
N_PAIR = {896: 35, 64: 12}                 # 值的数目 = 2*N_PAIR（70 / 24）
R0_LEVELS = {896: [0, 1, 2, 8, 35, 70, 896], 64: [0, 1, 2, 8, 24, 64]}
#   ↑ $r_0=0$ = §3.2 全矩阵参考步（唯一可表达的空记忆，A24-5）；$r_0=d$ = 装置默认满矩阵。
#   ★ d=64 上 $r_0\\in\\{35,70,896\\}$ 在数学上不可表达（$r_0\\le d$）⇒ 该轴用自身的 6 个水平（DEC-18）。
N_SLOTS = 1024
DIR_TAG = "DIR24r10"
RTOL = 1e-6    # ★ rev 1.1（DEF-24-3）：秩与伪逆统一用**相对容差**。$\kappapprox1.8	imes10^5$
#  使默认容差把满秩臂算成 894（float64 同样 $\sigma_{\min}=2.7$e-6 ⇒ 与精度无关）；rtol=1e-6 下
#  秩与 $\sigma$ 谱一致（$r_0\le70$ 精确、$r_0=d$ 为 896），且投影残差回落到浮点噪声。

res = {"dir": DIR_TAG, "checks": [], "rows": [], "meta": {}}


# ══════════════════════════════════════════════════════════ helpers（逐字自 dir23/R1_scope.py）
def _norm_cdf(x):
    """标准正态 CDF（无 scipy ⇒ 用 math.erf）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sign_test_p(d):
    d = [x for x in d if abs(x) > 1e-15]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0
    k = sum(1 for x in d if x > 0)
    pk = math.comb(n, k) * 0.5 ** n
    p = sum(math.comb(n, i) * 0.5 ** n for i in range(n + 1)
            if math.comb(n, i) * 0.5 ** n <= pk + 1e-12)
    return min(1.0, p), k, n


def unit(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-30)


def item_z(n, dev="cpu"):
    """条目身份编码：单位圆上 n 等分 ⇒ 落在 2 维子空间 ⇒ 线性控制器**可泛化**到未见条目。"""
    th = torch.arange(n, dtype=torch.float32) * (2 * math.pi / n)
    return torch.stack([torch.cos(th), torch.sin(th)], -1).to(dev)


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


# ══════════════════════════════════════════════════════════ 族 A（派生自 dir23/build_A：去掉 $S$ 字段、$d$ 参数化）
def build_A(seed, d):
    """族 A · 冲突：$k_a=k_b$（共线），$v_a\\perp v_b$，对间键**精确正交**（标准基）。"""
    g = torch.Generator().manual_seed(seed)
    n = N_PAIR[d]
    K = torch.zeros(n, d)
    for i in range(n):
        K[i, i] = 1.0                                  # 对间精确正交（A24-8）
    va = unit(torch.randn(n, d, generator=g))
    r = torch.randn(n, d, generator=g)
    vb = unit(r - (r * va).sum(-1, keepdim=True) * va)  # 与 va 正交
    target_a = torch.arange(n) % 2 == 0                 # 35 对 ⇒ 18/17；12 对 ⇒ 6/6（A24-7）
    return {"K": K, "va": va, "vb": vb, "target_a": target_a, "n": n}


# ══════════════════════════════════════════════════════════ 装置构造（秩约束只在初值支撑上）
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


# ══════════════════════════════════════════════════════════ 求值
def eval_rank(d, seed, r0):
    """读：信息论口径（DEC-10）$q^*=\\arg\\min\\|Wq-t\\|$ ⇒ $y=\\mathrm{Proj}_{\\operatorname{range}W}(t)$。"""
    D = build_A(seed, d)
    core = make_core(d, seed, r0)
    info = write_A(core, D)
    W = core._W().detach()
    rank_W = int(torch.linalg.matrix_rank(W, rtol=RTOL).item())
    P = W @ torch.linalg.pinv(W, rtol=RTOL)
    W64 = W.double()                                            # ★ rev 1.1（DEF-24-3）：两个区段分开核
    P64 = W64 @ torch.linalg.pinv(W64, rtol=RTOL)               #   低秩区：核**投影恒等式**（float64 表述）
    a_ok, b_ok = [], []
    margins, resid, both = [], 0.0, []                       # ★ rev 1.1（DEF-24-4）：主判据=**每条目一个目标**
    for i in range(D["n"]):                                  #   （$target_a[i]=i\%2=0$ ⇒ 18 问 a / 17 问 b，35 条），
        mbr = 0 if bool(D["target_a"][i]) else 1              #   与 DIR-23 的 `eval_A` 逐字同口径；
        for mm in (0, 1):                                    #   另一成员只作**报告项** `acc_both_members`
            t = D["va"][i] if mm == 0 else D["vb"][i]
            q = best_q(core, "hebbian", [t])
            y = core.read(q.unsqueeze(0))[0]
            mg = float(y @ unit(D["va"][i]) - y @ unit(D["vb"][i]))
            ok = (mg > 0) if mm == 0 else (mg < 0)
            both.append(ok)
            if mm == mbr:
                margins.append(mg)
                if r0 < d:                                   # 低秩区：核**投影恒等式**
                    resid = max(resid, float((W64 @ q.double() - P64 @ t.double()).norm()))
                a_ok.append(ok) if mbr == 0 else b_ok.append(ok)
    n = D["n"]
    acc = float(np.mean(a_ok + b_ok))
    acc_both = float(np.mean(both))
    if r0 == 0:
        delta_uv = {"du": 0.0, "dv": 0.0}                          # 参考步无因子（A24-6 不适用）
    else:
        delta_uv = {"du": float(core.U.norm()), "dv": float(core.V.norm())}
    return {"d": d, "seed": seed, "r0": r0, "rank_W": rank_W, "acc": acc,
            "acc_a": float(np.mean(a_ok)), "acc_b": float(np.mean(b_ok)),
            "acc_both_members": acc_both, "n_items": len(a_ok) + len(b_ok),
            "margin_mean": float(np.mean(margins)), "margin_abs_mean": float(np.mean(np.abs(margins))),
            "proj_resid_max": resid, "min_abs_margin": float(np.min(np.abs(margins))),
            "rank_snaps": info["rank_snaps"],
            "n_pairs": n, "n_ask_a": int(sum(D["target_a"].tolist())),
            "n_ask_b": int(n - sum(D["target_a"].tolist())),
            "u_norm": delta_uv["du"], "v_norm": delta_uv["dv"],
            "fit": "info-ls"}


# ══════════════════════════════════════════════════════════ 断言工具
def chk(cid, ok, detail):
    res["checks"].append({"id": cid, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}  {detail}")
    return bool(ok)


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


# ══════════════════════════════════════════════════════════ Stage A
def stage_a():
    print("\n=== Stage A（A24-1 … A24-8） ===")
    dev = HERE / "sgmn_module_hebb_v7.py"
    up = HERE.parent / "dir22" / "sgmn_module_hebb_v7.py"
    chk("A24-1", md5(dev) == md5(up) == "1a242d02c530307f6c76646ca13469ef",
        f"装置 md5 {md5(dev)} == dir22 {md5(up)}")

    # A24-3 写序系数闭式（参考步；与 d 无关）
    gam = 1.0 - ETA * (ALPHA + LAM)
    ca_an, cb_an = ETA * ALPHA * (1 + gam) * gam ** 2, ETA * ALPHA * (1 + gam)
    got = {}
    for d in D_LEVELS:
        g = torch.Generator().manual_seed(3)
        c = HebbianCore(d, d, ALPHA, LAM, ETA, g, "cpu", factorized=False, n_entries=8, d_model_sig=8)
        c.W.zero_(); c.W0.zero_()
        va = torch.zeros(d); va[0] = 1.0
        vb = torch.zeros(d); vb[1] = 1.0
        k = torch.zeros(d); k[0] = 1.0
        for mbr, v in ((0, va), (1, vb)):
            for _ in range(M):
                c.write(k, v)
        col = c.W[:, 0]
        got[d] = (float(col @ va), float(col @ vb))
    ok3 = all(abs(got[d][0] - ca_an) / ca_an <= 1e-3 and abs(got[d][1] - cb_an) / cb_an <= 1e-3
              for d in D_LEVELS)
    chk("A24-3", ok3, f"c_a={got[896][0]:.6f} vs {ca_an:.6f}; c_b={got[896][1]:.6f} vs {cb_an:.6f}"
                      f"; d=64 逐位相同={got[64] == got[896]}")

    # A24-8 键结构
    ok8 = True
    for d in D_LEVELS:
        D = build_A(SEEDS[0], d)
        K = D["K"]
        off = (K @ K.T) - torch.diag(torch.diag(K @ K.T))
        rows_ok = bool(torch.all((K > 0).sum(-1) == 1) and torch.all(K[K > 0] == 1.0))
        ok8 = ok8 and float(off.abs().max()) <= 1e-12 and rows_ok
    chk("A24-8", ok8, "每行是标准基向量（唯一非零=1.0）且对间 |k_i·k_j|<=1e-12（d=896 与 d=64）；对内同一行由 write_A 结构保证")

    # A24-2 / A24-4 / A24-5 / A24-6 / A24-7（真跑小规模端点：1 种子全水平）
    ok2 = ok4 = ok5 = ok6 = ok7 = True
    d2det, d4det, d5det, d6det, d7det = [], [], [], [], []
    for d in D_LEVELS:
        for r0 in R0_LEVELS[d]:
            r = eval_rank(d, SEEDS[0], r0)
            if r0 >= 1:
                if r["rank_W"] != r0:
                    ok2 = False; d2det.append(f"d{d} r0{r0}->rank{r['rank_W']}")
                if max(r["rank_snaps"]) > r0:
                    ok2 = False; d2det.append(f"d{d} r0{r0} snap{max(r['rank_snaps'])}")
                if r["u_norm"] <= 0 or r["v_norm"] <= 0:
                    ok6 = False; d6det.append(f"d{d} r0{r0}")
            if r0 == 0 and r["rank_W"] != r["n_pairs"]:
                ok5 = False; d5det.append(f"d{d} rank{r['rank_W']}!=n_pairs{r['n_pairs']}")
            if r0 < d:
                if r["proj_resid_max"] > 1e-4:
                    ok4 = False; d4det.append(f"d{d} r0{r0} resid{r['proj_resid_max']:.2e}")
            else:
                if r["min_abs_margin"] < 0.90:
                    ok4 = False; d4det.append(f"d{d} r0{r0} min|margin|{r['min_abs_margin']:.3f}")
            if r["n_items"] != r["n_pairs"] or abs(r["n_ask_a"] - r["n_ask_b"]) > 1:
                ok7 = False; d7det.append(f"d{d} {r['n_ask_a']}/{r['n_ask_b']}")
    chk("A24-2", ok2, f"秩不变性（rtol={RTOL}）：每 (d,r0>=1) 末态 rank=r0 且 5 端点 rank<=r0 {d2det}")
    chk("A24-4", ok4, f"低秩区 ||Wq*-Proj t||<=1e-4（float64 表述，pinv rtol={RTOL}）；满秩区 min|margin|>=0.90 {d4det}")
    chk("A24-5", ok5, f"r0=0 走全矩阵参考步且 rank=n_pairs {d5det}")
    chk("A24-6", ok6, f"r0>=1 末态 ||U||>0 且 ||V||>0 {d6det}")
    chk("A24-7", ok7, f"平衡目标 18/17（d=896）与 6/6（d=64） {d7det}")
    return all(c["ok"] for c in res["checks"])


# ══════════════════════════════════════════════════════════ 判据
def criteria(rows):
    """(a) 单调非减 · (b) 低秩端不饱和 · (c) 饱和刻度=值的数目 · (d) 满秩端点 · (e) d 依赖诊断。"""
    A = [r for r in rows if r["d"] == 896]
    lv = R0_LEVELS[896]
    acc = {(r["r0"], r["seed"]): r["acc"] for r in A}
    nondec = tot = 0
    for k in range(len(lv) - 1):
        for s in SEEDS:
            tot += 1
            nondec += int(acc[(lv[k + 1], s)] >= acc[(lv[k], s)] - 1e-12)
    a_ok = nondec >= 24
    b1 = all(acc[(1, s)] <= 0.65 for s in SEEDS)
    b2 = all(acc[(1, s)] <= acc[(896, s)] - 0.25 for s in SEEDS)
    r_star = None
    for r0 in lv:
        if all(acc[(r0, s)] >= 0.90 for s in SEEDS):
            r_star = r0
            break
    c_ok = r_star is not None and r_star <= 70
    d_ok = all(acc[(896, s)] >= 0.95 for s in SEEDS)
    E = [r for r in rows if r["d"] == 64 and r["r0"] == 0]
    e_gap = float(np.mean([r["acc"] for r in E])) - float(np.mean([acc[(0, s)] for s in SEEDS]))
    verdicts = {
        "a_monotone": {"ok": a_ok, "nondec": nondec, "total": tot,
                       "detail": f"相邻比较非减 {nondec}/{tot}（阈值 24/30）"},
        "b_lowrank": {"ok": b1 and b2, "acc_r0_1": [acc[(1, s)] for s in SEEDS],
                      "detail": "acc(r0=1)<=0.65 且 <= acc(896)-0.25"},
        "c_saturation": {"ok": c_ok, "r_star": r_star,
                         "detail": f"r*=min{{r0: acc>=0.90}}={r_star}（阈值 <=70）"},
        "d_fullrank": {"ok": d_ok, "acc_896": [acc[(896, s)] for s in SEEDS], "detail": "acc(896)>=0.95 5/5"},
        "e_d_dependence": {"ok": e_gap >= 0.15, "gap": e_gap, "clauses": False,
                           "detail": f"acc(r0=0,d=64)-acc(r0=0,d=896)={e_gap:.4f}（阈值 >=0.15，**不计入主判据**）"},
    }
    main = verdicts["a_monotone"]["ok"] and verdicts["b_lowrank"]["ok"] and \
        verdicts["c_saturation"]["ok"] and verdicts["d_fullrank"]["ok"]
    verdicts["main"] = {"ok": main, "falsified": not main}
    return verdicts


# ══════════════════════════════════════════════════════════ 主
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stageA", "smoke", "formal"], required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    res["meta"] = {"stage": a.stage, "seeds": SEEDS, "d_levels": D_LEVELS,
                   "r0_levels": R0_LEVELS, "n_pair": N_PAIR,
                   "consts": {"alpha": ALPHA, "lam": LAM, "eta": ETA, "M": M}}

    if a.stage == "stageA":
        ok = stage_a()
        res["meta"]["stageA_all_pass"] = ok
    elif a.stage == "smoke":
        print("\n=== smoke（1 种子 × 1 水平 × d=896；**不出判据**） ===")
        r = eval_rank(896, SEEDS[0], 8)
        res["rows"].append(r)
        print("  " + json.dumps({k: r[k] for k in ("d", "seed", "r0", "acc", "rank_W", "proj_resid_max")}))
    else:
        print("\n=== 正式轮（7 水平 × d=896 + 6 水平 × d=64，5 种子） ===")
        for d in D_LEVELS:
            for r0 in R0_LEVELS[d]:
                for s in SEEDS:
                    r = eval_rank(d, s, r0)
                    res["rows"].append(r)
                    print(f"  d={d} r0={r0:3d} seed={s} acc={r['acc']:.4f} rank={r['rank_W']:3d}"
                          f" mA={r['margin_abs_mean']:.4f} resid={r['proj_resid_max']:.2e}")
        res["verdicts"] = criteria(res["rows"])

    res["meta"]["wall_seconds"] = round(time.time() - t0, 1)
    out = pathlib.Path(a.out) if a.out else HERE / "out" / f"{DIR_TAG}_{a.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}  ({res['meta']['wall_seconds']}s)")
    if a.stage == "stageA":
        print("\n★ Stage A " + ("全部通过" if res["meta"]["stageA_all_pass"] else "**有 FAIL，阻断正式轮**"))
        sys.exit(0 if res["meta"]["stageA_all_pass"] else 2)


if __name__ == "__main__":
    main()
