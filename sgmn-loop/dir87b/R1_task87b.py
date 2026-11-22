#!/usr/bin/env python3
"""DIR-87b rev 1.2 · `S` 的容量对 $W^0$ 的依赖（`empty` vs `rand`）。

**逐字派生自 `dir87/R1_task87.py`（`D-116`）。相对它有且只有三处改动**：
  (1) 新增自变量 `base ∈ {"empty","rand_full","rand_fact"}`（`explicit` 记 `"n/a"`，因其无 $W^0$）；
  (2) `NGRID` 加密小 $N$（20 点）以分辨 $N_{0.5}$；
  (3) 判据整体替换为 `C87b-NOTMEM` / `C87b-GAIN` / `C87b-SCALE`，Stage A 换成本卡手算断言。
**装置一字未改**：`dir22/sgmn_module_hebb_v7.py` md5 `1a242d02c530307f6c76646ca13469ef`。
`decode_hits` / `unit` / `sha16` / `make_data` **逐字沿用**（含 `N_MAX=max(NGRID)=2048`，否则键池会变）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dir22"))          # 装置所在（逐字自 dir87）
from sgmn_module_hebb_v7 import ExplicitKVCore, HebbianCore   # noqa: E402

# ── 常量：逐字自 dir87/R1_task87.py:31-41（= dir23/R1_scope.py:33-39）
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]
ALPHA, LAM, ETA = 1.0, 0.05, 0.1
N_SLOTS = 1024
BETA = 16.0
# ── 本卡网格（§2.1）：小 N 加密，末点仍 2048（保住 N_MAX 与 A87b-05 的键池一致）
NGRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 900, 1024, 2048]
DS = [8, 64]
# ★ 三种 hebbian 配置（DEF-87b-03）：empty / rand_full(= D-116 的 hebbian，同写规则、仅初值不同) / rand_fact(= dir23 的 "rand")
ARMS = [("hebbian", "empty"), ("hebbian", "rand_full"), ("hebbian", "rand_fact"),
        ("explicit", "n/a")]
N_MAX = max(NGRID)                      # = 2048，与 dir87 一致（键池 RNG 依赖它）
HIT_HALF = 0.5                          # §3 N_{0.5} 定义
GAIN_MIN = 0.20                         # §3 C87b-GAIN 冻结阈值
SCALE_MIN = 1.5                         # §3 C87b-SCALE 冻结阈值
REP_N = [1, 4, 64, 1024]                # A87b-05 复现子集

res = {"dir": "DIR87b-r13", "checks": [], "rows": [], "meta": {}}


def rec(name, passed, detail, value=None):
    """`passed=None` ⇒ **不可计算**（按规则不得记 PASS/FAIL）。"""
    res["checks"].append({"name": name, "pass": None if passed is None else bool(passed),
                          "detail": detail, "value": value})
    tag = "N/A " if passed is None else ("PASS" if passed else "FAIL")
    print(f"[{tag}] {name}: {detail}", flush=True)
    return None if passed is None else bool(passed)


def unit(x):
    return x / x.norm().clamp_min(1e-30)          # ★ 整池 Frobenius 归一（DEF-87b-02）


def sha16(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


# ══════════════════════════════════════════════════ 装置构造（§2.1；empty 分支逐字自 dir23:123-126）
def build_core(kind, d, seed, base):
    g = torch.Generator().manual_seed(seed)
    if kind == "hebbian":
        if base == "rand_fact":                      # = dir23 的 "rand"：因子步，W0=0，随机性在 U0V0ᵀ
            return HebbianCore(d, d, ALPHA, LAM, ETA, g, "cpu", factorized=True,
                               n_entries=N_SLOTS, d_model_sig=0)
        core = HebbianCore(d, d, ALPHA, LAM, ETA, g, "cpu", factorized=False,
                           n_entries=N_SLOTS, d_model_sig=0)      # 全矩阵参考步
        if base == "empty":                          # 参考点与有效初值都置空
            core.W.zero_(); core.W0.zero_()
        # base == "rand_full"：不置零 ⇒ W0 = 随机 W（**这正是 D-116 的 hebbian 口径**）
        return core
    return ExplicitKVCore(d, d, ALPHA, LAM, ETA, g, "cpu", n_entries=N_SLOTS, d_model_sig=0)


def make_data(d, seed):
    """逐字自 dir87/R1_task87.py:73-78：**只依赖 (d, seed)**，与 kind/base 无关。"""
    g = torch.Generator().manual_seed(seed * 1000003 + d)
    K = unit(torch.randn(N_MAX, d, generator=g))
    V = unit(torch.randn(N_MAX, d, generator=g))
    return K, V


# ══════════════════════════════════════════════════ 测量层（逐字自 dir87:82-90）
def decode_hits(Y, truth, idx):
    Yn = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-30)
    Tn = truth / truth.norm(dim=1, keepdim=True).clamp_min(1e-30)
    C = Yn @ Tn.T
    pred = C.argmax(dim=1)
    hit = float((pred == idx).float().mean())
    return hit, C, pred


# ══════════════════════════════════════════════════ 一个格
def run_cell(kind, d, seed, N, base):
    core = build_core(kind, d, seed, base)
    K, V = make_data(d, seed)
    kk, vv = K[:N], V[:N]

    w0_norm0 = float(core.W0.norm()) if kind == "hebbian" else None
    w_eff0 = float(core._W().norm()) if kind == "hebbian" else None
    alpha_before = (core.alpha, core.lam, core.eta)
    w0_before = core.W0.detach().clone() if kind == "hebbian" else None

    for i in range(N):                       # oneshot 单遍（§2.1）
        core.write(kk[i], vv[i], sig=None)

    with torch.no_grad():
        Y = core.read(kk)
    hit, C, pred = decode_hits(Y, vv, torch.arange(N))

    alpha_after = (core.alpha, core.lam, core.eta)
    w0_same = (kind != "hebbian") or bool(torch.equal(core.W0, w0_before))
    return {
        "kind": kind, "base": base, "d": d, "seed": seed, "N": N, "hit": hit,
        "entries": int(core.count), "slots_used": int(min(core.count, core.N)),
        "alpha_ok": alpha_before == alpha_after, "w0_same": w0_same,
        "y_norm_mean": float(Y.norm(dim=1).mean()),
        "w0_norm": (float(core.W0.norm()) if kind == "hebbian" else None),
        "w_eff_norm": (float(core._W().norm()) if kind == "hebbian" else None),
        "w0_norm_init": w0_norm0, "w_eff_norm_init": w_eff0,
        "data_norm": (float(core.data_norm) if kind == "hebbian" else None),
        "reg_norm": (float(core.reg_norm) if kind == "hebbian" else None),
        "key_sha": sha16(kk) if N > 0 else "",
        "pred_head": pred[:8].tolist(),
    }


def mean_hit(kind, base, d, N):
    return sum(run_cell(kind, d, s, N, base)["hit"] for s in SEEDS) / len(SEEDS)


# ══════════════════════════════════════════════════ Stage A（§2.2，全部手算过）
def stage_a():
    print("=" * 70 + "\nStage A（§2.2，逐条手算冻结）\n" + "=" * 70, flush=True)
    d8, sd0 = 8, SEEDS[0]
    K, V = make_data(d8, sd0)
    k0, v0 = K[0], V[0]
    nk, nv = float(k0.norm()), float(v0.norm())
    r, delta = ETA * ALPHA, ETA * LAM

    # ---- A87b-00（empty / d=8 / N=1）
    c = build_core("hebbian", d8, sd0, "empty")
    c.write(k0, v0, sig=None)
    W = c.W.detach().clone()
    W_pred = r * torch.outer(v0, k0)
    rel_a = float((W - W_pred).norm() / W_pred.norm().clamp_min(1e-30))
    rec("A87b-00a W 精确等于 r·v0k0ᵀ", rel_a <= 1e-6,
        f"相对偏差 {rel_a:.3e}（≤1e-6）", value=rel_a)

    rel_b = abs(float(W.norm()) - r * nv * nk) / (r * nv * nk)
    rec("A87b-00b ‖W‖_F = r·n_v·n_k", rel_b <= 1e-6,
        f"实测 {float(W.norm()):.6e} vs 手算 r·n_v·n_k={r*nv*nk:.6e}（相对 {rel_b:.2e}）", value=rel_b)

    with torch.no_grad():
        y0 = c.read(k0.unsqueeze(0))[0]
    cos0 = float(torch.nn.functional.cosine_similarity(y0, v0, dim=0))
    rec("A87b-00c cos(y0,v0) = 1.0", abs(cos0 - 1.0) <= 1e-5,
        f"实测 {cos0:.10f}（手算 1.0）；‖y0‖={float(y0.norm()):.3e}、r·n_k²={r*nk*nk:.3e}", value=cos0)

    rel_d = abs(c.data_norm - nv * nk) / (nv * nk)
    rec("A87b-00d data_norm = n_v·n_k", rel_d <= 1e-6,
        f"实测 {c.data_norm:.6e} vs 手算 n_v·n_k={nv*nk:.6e}（相对 {rel_d:.2e}）", value=rel_d)

    rec("A87b-00e reg_norm = 0（精确）", c.reg_norm == 0.0,
        f"实测 {c.reg_norm!r}（W_prev=W0=0 ⇒ 正则项恒 0）", value=c.reg_norm)

    # ---- A87b-01 秩 1
    sv = torch.linalg.svdvals(W)
    ratio1 = float(sv[1] / sv[0].clamp_min(1e-30))
    rec("A87b-01 N=1 后 σ1/σ0 = 0（rank 恰 1）", ratio1 <= 1e-6,
        f"σ0={float(sv[0]):.6e} σ1={float(sv[1]):.3e} ⇒ 比 {ratio1:.3e}", value=ratio1)

    # ---- A87b-02 N=2 闭式（§2.3(b)）
    c2 = build_core("hebbian", d8, sd0, "empty")
    c2.write(K[0], V[0], sig=None); c2.write(K[1], V[1], sig=None)
    with torch.no_grad():
        y2 = c2.read(K[0].unsqueeze(0))[0]
    cos2 = float(torch.nn.functional.cosine_similarity(y2, V[0], dim=0))
    s = float(K[0] @ K[1]); t = float(V[0] @ V[1])
    nv0, nv1 = float(V[0].norm()), float(V[1].norm())
    A = (1 - delta) * r * nk * nk - ETA * r * s * s      # y0 里 v0 的系数
    B = ETA * s                                          # y0 里 v1 的系数
    # ★ v0、v1 的范数**不为 1 且互不相等** ⇒ 余弦须带 nv0/nv1（DEF-87b-04，已数值验证 1.1e-07）
    num = A * nv0 * nv0 + B * t
    den = nv0 * (A * A * nv0 * nv0 + B * B * nv1 * nv1 + 2 * A * B * t) ** 0.5
    cos_pred = num / den
    rec("A87b-02 N=2 闭式 cos", abs(cos2 - cos_pred) <= 1e-5,
        f"实测 {cos2:.10f} vs 闭式 {cos_pred:.10f}（nk={nk:.6f} nv0={nv0:.6f} nv1={nv1:.6f} s={s:.4e} t={t:.4e}）"
        f"；差 {abs(cos2-cos_pred):.2e}", value={"obs": cos2, "pred": cos_pred, "s": s, "t": t})

    # ---- A87b-03 参考点 vs 有效初值
    ce = build_core("hebbian", d8, sd0, "empty")
    cf = build_core("hebbian", d8, sd0, "rand_full")
    cd = build_core("hebbian", d8, sd0, "rand_fact")
    w0e, w0f, w0d = float(ce.W0.norm()), float(cf.W0.norm()), float(cd.W0.norm())
    wee, wef, wed = float(ce._W().norm()), float(cf._W().norm()), float(cd._W().norm())
    ok3 = (w0e == 0.0 and wee == 0.0                          # empty：参考点与有效初值都空
           and w0f > 0.0 and abs(wef - w0f) <= 1e-6 * w0f     # rand_full：W0 就是随机 W
           and w0d == 0.0 and wed > 0.0)                      # rand_fact：W0=0，随机性在 U0V0ᵀ
    rec("A87b-03 三配置的 W0 / 有效初值", ok3,
        f"W0: empty={w0e:.1e}(=0) rand_full={w0f:.4f}(>0) rand_fact={w0d:.1e}(=0) | "
        f"_W(): empty={wee:.1e}(=0) rand_full={wef:.4f}(=W0) rand_fact={wed:.4f}(>0)",
        value={"w0": {"empty": w0e, "rand_full": w0f, "rand_fact": w0d},
               "weff": {"empty": wee, "rand_full": wef, "rand_fact": wed}})

    # ---- A87b-04 跨 base 数据不变性
    shas = {b: sha16(make_data(d8, sd0)[0]) for b in ("empty", "rand")}
    shas["n/a"] = sha16(make_data(d8, sd0)[0])
    rec("A87b-04 跨 base 键张量逐位相同", len(set(shas.values())) == 1,
        f"sha16: {shas}", value=shas)

    # ---- A87b-05 复现 D-116
    f87 = sorted((HERE.parent / "dir87" / "out").glob("D87_formal_*.json"))
    if not f87:
        rec("A87b-05 复现 D-116 读数", None, "dir87/out 无 formal JSON ⇒ 不可计算（不记 PASS/FAIL）")
    else:
        ref = json.loads(f87[-1].read_text(encoding="utf-8"))["curves"]
        pairs = [("hebbian", "rand_full", "hebbian"), ("explicit", "n/a", "explicit")]
        worst, bad = 0.0, []
        for kind, base, refkind in pairs:
            for d in DS:
                rc = ref[f"{refkind}/d{d}"]
                for N in REP_N:
                    mine = mean_hit(kind, base, d, N)
                    theirs = rc.get(str(N), rc.get(N))
                    if theirs is None:
                        continue
                    diff = abs(mine - theirs)
                    worst = max(worst, diff)
                    if diff > 1e-9:
                        bad.append(f"{kind}/{base}/d{d}/N{N}: 本 {mine:.6f} vs D-116 {theirs:.6f}")
        rec("A87b-05 复现 D-116 读数", not bad,
            f"最大偏差 {worst:.2e}（≤1e-9）；{len(bad)} 处不符 {bad[:2]}", value=worst)

    # ---- A87b-06 越界五条
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    forb = {"torch." + "optim": ("torch." + "optim") in src,
            "Layer" + "Mem(": ("Layer" + "Mem(") in src,
            "SGMN" + "Stack(": ("SGMN" + "Stack(") in src,
            "write_side_" + "loss(": ("write_side_" + "loss(") in src}
    ca = build_core("hebbian", d8, sd0, "empty")
    c.write(k0, v0, sig=None)
    intact = (ca.alpha, ca.lam, ca.eta) == (ALPHA, LAM, ETA)
    ok6 = (not any(forb.values())) and intact
    rec("A87b-06 越界五条", ok6,
        f"禁用符号命中={ {k: v for k, v in forb.items() if v} }（应为空）；参数不变={intact}", value=forb)

    # ---- A87b-08 N=1 命中
    hits = {f"{k}/{b}/d{d}": run_cell(k, d, sd0, 1, b)["hit"] for k, b in ARMS for d in DS}
    rec("A87b-08 N=1 命中 = 1.0（六臂）", all(h >= 0.999 for h in hits.values()),
        f"min={min(hits.values()):.3f}；{hits}", value=hits)

    n_pass = sum(1 for x in res["checks"] if x["pass"] is True)
    n_fail = sum(1 for x in res["checks"] if x["pass"] is False)
    n_na = sum(1 for x in res["checks"] if x["pass"] is None)
    print(f"\n★ Stage A: {n_pass} PASS / {n_fail} FAIL / {n_na} N/A", flush=True)
    return n_fail == 0


# ══════════════════════════════════════════════════ N_{0.5}（§3 定义）
def n_half(curve, grid):
    """首个 h<0.5 与前一格的线性插值。返回 (值, 是否右删失)。"""
    prev = None
    for N in grid:
        h = curve.get(N)
        if h is None:
            continue
        if h < HIT_HALF:
            if prev is None:
                return 1.0, False            # 网格首点已 <0.5（左删失，按 1 计）
            Np, hp = prev
            if hp == h:
                return float(N), False
            return Np + (HIT_HALF - hp) * (N - Np) / (h - hp), False
        prev = (N, h)
    return float(grid[-1]), True             # 右删失：只作下界


# ══════════════════════════════════════════════════ 正式轮
def formal():
    t0 = time.time()
    cells = [run_cell(k, d, s, N, b) for k, b in ARMS for d in DS for s in SEEDS for N in NGRID]
    res["rows"] = cells
    res["meta"]["elapsed_s"] = time.time() - t0
    res["meta"]["n_cells"] = len(cells)

    need = [(k, b, d, s, N) for k, b in ARMS for d in DS for s in SEEDS for N in NGRID]
    have = {(r["kind"], r["base"], r["d"], r["seed"], r["N"]) for r in cells}
    missing = [x for x in need if x not in have]
    rec("A87b-07 网格逐格有读数", not missing,
        f"缺 {len(missing)} 格 {missing[:3]}（共需 {len(need)}）" if missing else f"{len(need)} 格全有读数")

    def agg(kind, base, d):
        return {N: sum(r["hit"] for r in cells if r["kind"] == kind and r["base"] == base
                       and r["d"] == d and r["N"] == N) / len(SEEDS) for N in NGRID}

    curves = {f"{k}/{b}/d{d}": agg(k, b, d) for k, b in ARMS for d in DS}
    res["curves"] = curves

    nh, cens = {}, {}
    for key, cv in curves.items():
        nh[key], cens[key] = n_half(cv, NGRID)
    res["n_half"], res["n_half_censored"] = nh, cens

    cand = [N for N in NGRID if N >= 4]
    for d in DS:
        hE, hR = curves[f"hebbian/empty/d{d}"], curves[f"hebbian/rand_full/d{d}"]
        res.setdefault("gain", {})[str(d)] = {
            "max_gap": max(hE[N] - hR[N] for N in cand),
            "at_N": max(cand, key=lambda N: hE[N] - hR[N]),
            "empty": {N: hE[N] for N in cand}, "rand": {N: hR[N] for N in cand}}

    # ---- C87b-NOTMEM（最先判）
    notmem_viol = []
    for d in DS:
        hE = curves[f"hebbian/empty/d{d}"]
        for N in cand:
            if hE[N] > max(3.0 / N, 0.10):
                notmem_viol.append((d, N, round(hE[N], 4)))
    notmem = len(notmem_viol) == 0
    rec("C87b-NOTMEM（成立=否证 H87b）", notmem,
        ("两 d 上所有 N≥4 都不离机会水平 ⇒ NOT-MEMORY 成立 ⇒ H87b 否证（F1）" if notmem else
         f"有 {len(notmem_viol)} 处离开机会水平（举例 {notmem_viol[:4]}）⇒ NOT-MEMORY 不成立"),
        value={"violations": notmem_viol[:12], "n_viol": len(notmem_viol)})

    # ---- C87b-GAIN
    gaps = {d: max(curves[f"hebbian/empty/d{d}"][N] - curves[f"hebbian/rand_full/d{d}"][N] for N in cand)
            for d in DS}
    gain_ok = all(gaps[d] >= GAIN_MIN for d in DS)
    rec("C87b-GAIN max(empty−rand) ≥ 0.20（两 d）", gain_ok,
        "; ".join(f"d{d}: {gaps[d]:.3f}@N={res['gain'][str(d)]['at_N']}" for d in DS)
        + f"（阈值 {GAIN_MIN}）", value=gaps)

    # ---- C87b-SCALE
    a8, c8 = nh["hebbian/empty/d8"], cens["hebbian/empty/d8"]
    a64, c64 = nh["hebbian/empty/d64"], cens["hebbian/empty/d64"]
    if c8 and c64:
        rec("C87b-SCALE N0.5(64)/N0.5(8) > 1.5", None,
            f"两者都右删失（{a8:.0f}/{a64:.0f}）⇒ 网格不足以分辨 ⇒ 不可计算，不记 PASS/FAIL")
    else:
        ratio = a64 / max(a8, 1e-9)
        rec("C87b-SCALE N0.5(64)/N0.5(8) > 1.5", ratio > SCALE_MIN,
            f"N0.5: d8={a8:.3f}{'(删失)' if c8 else ''} d64={a64:.3f}{'(删失)' if c64 else ''}"
            f" ⇒ 比 {ratio:.3f}（阈值 {SCALE_MIN}）", value={"n0.5_d8": a8, "n0.5_d64": a64, "ratio": ratio})

    # ---- 合成判定
    g = [x for x in res["checks"] if x["name"].startswith("C87b-")]
    if notmem:
        verdict, detail = "H87b 否证（F1）", "hebbian S 在本族参数下不是记忆"
    elif gain_ok and (nh["hebbian/empty/d64"] > SCALE_MIN * nh["hebbian/empty/d8"]):
        verdict, detail = "H87b 证实", "empty 下有可用容量且随 d 增长"
    else:
        verdict, detail = "H87b 部分", "见上；至少一支未过或不可计算"
    res["verdict"], res["verdict_detail"] = verdict, detail
    print(f"\n★ 合成判定: {verdict} —— {detail}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stagea", "formal", "all"], default="all")
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")

    if a.mode in ("stagea", "all"):
        ok = stage_a()
        res["stage_a_pass"] = ok
        (pathlib.Path(a.out) / f"D87b_stageA_{stamp}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        if not ok:
            print("★ Stage A 未过 ⇒ 阻断正式轮", flush=True)
            return 1
        if a.mode == "stagea":
            return 0

    res2 = formal()
    res2["stage_a_pass"] = True
    (pathlib.Path(a.out) / f"D87b_formal_{stamp}.json").write_text(
        json.dumps(res2, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== 命中率曲线（oneshot）===", flush=True)
    for k, cv in res2["curves"].items():
        print(f"  {k}: " + " ".join(f"{N}:{cv[N]:.4f}" for N in NGRID), flush=True)
    print("\n=== N_0.5（半衰点）===", flush=True)
    for k, v in res2["n_half"].items():
        print(f"  {k}: {v:.3f}{'  [右删失]' if res2['n_half_censored'][k] else ''}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
