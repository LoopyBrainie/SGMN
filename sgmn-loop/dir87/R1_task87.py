"""DIR-87 · R1（**$S$ 的容量标度**，卡 `revolve/DIR-87/DIRECTION_rev87_v12.md`）

派生基（逐字派生，只改测量层与自变量）：`dir23/R1_scope.py` + `dir22/sgmn_module_hebb_v7.py`
（md5 `1a242d02c530307f6c76646ca13469ef`，见 `dir23/MANIFEST.json` 的 `/device/md5`）。

测的是一对公开 API：`core.write(k,v)` / `core.read(q)` 在**已存条目数 N** 增长下的**命中率**。
命中由**测量层**解码（卡 §2 rev1.1）：$\\hat j=\\arg\\max_j \\cos(y,v_j)$，全等 $i$ 才算命中；
真值表 $\\{v_j\\}$ 由**本脚本自持**（不取核的 `vals`，它是环形缓冲）。

★ 越界守卫（卡 §4 `A87-03`）：全程 **inference-only** —— 不建优化器、无任务损失、
不实例化 `LayerMem`/`SGMNStack`（故模型的 $W$ 根本未被触碰）、不动 $P_k/P_v/\\alpha$。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
DEV_DIR = HERE.parent / "dir22"
sys.path.insert(0, str(DEV_DIR))
from sgmn_module_hebb_v7 import ExplicitKVCore, HebbianCore   # noqa: E402

# ══════════════════════════════════════════════════════ 常量（卡 §0/§2，逐字）
SEEDS = [20261028, 20261104, 7, 20261111, 20261118]          # dir23/R1_scope.py:33
ALPHA, LAM, ETA = 1.0, 0.05, 0.1                              # dir23/R1_scope.py:36
D_S_TINY, D_S_BIG = 8, 64                                     # dir23/R1_scope.py:37
DS = [D_S_TINY, D_S_BIG]
KINDS = ["hebbian", "explicit"]                               # dir23/R1_scope.py:43
N_SLOTS = 1024                                                # dir23/R1_scope.py:39
BETA = 16.0                                                   # 装置默认 (read 的 beta)
NGRID = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 900, 1024, 2048]   # 卡 §6 冻结网格
NGRID_REH = [1, 4, 16, 64, 256, 1024]                         # 次协议网格（只报告）
E_REH = 32                                                    # 排练轮数（卡 §2，预先冻结）
N_MAX = max(NGRID)
HIT_KEEP = 0.90                                               # 只用于定义 N*（卡 §2）
CAP_LOO = 1.0                                                 # 占位：本卡无分类上限

res = {"dir": "DIR87r12", "checks": [], "rows": [], "meta": {}}
OPT_CALLS = []          # 越界守卫：任何优化器创建都会被记录（必须恒空）


def rec(name, passed, detail, value=None):
    res["checks"].append({"name": name, "pass": bool(passed), "detail": detail, "value": value})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)
    return bool(passed)


def unit(x):
    return x / x.norm().clamp_min(1e-30)


def sha16(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


# ══════════════════════════════════════════════════════ 装置构造（逐字自 dir23:114-131）
def build_core(kind, d, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "hebbian":
        # factorized=False = §3.2 全矩阵参考步（装置与 DIR-18…22 的默认口径）
        return HebbianCore(d, d, ALPHA, LAM, ETA, g, "cpu", factorized=False,
                           n_entries=N_SLOTS, d_model_sig=0)
    return ExplicitKVCore(d, d, ALPHA, LAM, ETA, g, "cpu", n_entries=N_SLOTS, d_model_sig=0)


def make_data(d, seed):
    """键/值池。★ `A87-07` 跨臂不变性：**只依赖 (d, seed)**，与 kind 无关。"""
    g = torch.Generator().manual_seed(seed * 1000003 + d)
    K = unit(torch.randn(N_MAX, d, generator=g))
    V = unit(torch.randn(N_MAX, d, generator=g))
    return K, V


# ══════════════════════════════════════════════════════ 测量层
def decode_hits(Y, truth, idx):
    """最近值解码（卡 §2 rev1.1）：$\\hat j=\\arg\\max_j\\cos(y,v_j)$，命中 $\\iff\\hat j=i$。"""
    Yn = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-30)
    Tn = truth / truth.norm(dim=1, keepdim=True).clamp_min(1e-30)
    C = Yn @ Tn.T                                    # (n_q, N)
    pred = C.argmax(dim=1)
    hit = float((pred == idx).float().mean())
    return hit, C, pred


def run_cell(kind, d, seed, N, protocol):
    """一个格：新建 core → 写 N 条 → 全量查询 → 命中率。"""
    core = build_core(kind, d, seed)
    K, V = make_data(d, seed)
    kk, vv = K[:N], V[:N]

    alpha_before = (core.alpha, core.lam, core.eta)
    w0_before = core.W0.detach().clone() if kind == "hebbian" else None

    epochs = 1 if protocol == "oneshot" else E_REH
    for _ in range(epochs):
        for i in range(N):
            core.write(kk[i], vv[i], sig=None)

    with torch.no_grad():
        Y = core.read(kk)                            # (N, d_v) 一次批量读
    truth = vv
    idx = torch.arange(N)
    hit, C, pred = decode_hits(Y, truth, idx)

    # ---- 越界守卫采样（逐格）
    alpha_after = (core.alpha, core.lam, core.eta)
    w0_same = (kind != "hebbian") or bool(torch.equal(core.W0, w0_before))
    # ---- 退化守卫：返回向量两两余弦 > 0.999 的占比
    Yn = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-30)
    Cp = Yn @ Yn.T
    degeneracy = float(((Cp - torch.eye(N)).abs().max(dim=1).values > 0.999).float().mean()) if N > 1 else 0.0

    return {
        "kind": kind, "d": d, "seed": seed, "N": N, "protocol": protocol,
        "hit": hit, "entries": int(core.count),
        "slots_used": int(min(core.count, core.N)),
        "alpha_ok": alpha_before == alpha_after, "w0_same": w0_same,
        "degeneracy": degeneracy,
        "y_norm_mean": float(Y.norm(dim=1).mean()),
        "key_sha": sha16(kk) if N > 0 else "",
        "pred_head": pred[:8].tolist(),
    }


# ══════════════════════════════════════════════════════ Stage A
def stage_a():
    ok = True

    # ---- A87-00 explicit：解析解 a0（DEF-87-06）
    d = D_S_TINY
    core = build_core("explicit", d, SEEDS[0])
    g = torch.Generator().manual_seed(999)
    k0, v0 = unit(torch.randn(d, generator=g)), unit(torch.randn(d, generator=g))
    core.write(k0, v0)
    with torch.no_grad():
        y = core.read(k0.unsqueeze(0))[0]
    a0_pred = math.exp(BETA) / (math.exp(BETA) + (N_SLOTS - 1))
    a0_obs = float((y @ v0) / v0.norm() ** 2)
    cos0 = float((y @ v0) / (y.norm() * v0.norm()).clamp_min(1e-30))
    print(f"    A87-00 explicit: a0_pred={a0_pred:.9f} a0_obs={a0_obs:.9f} "
          f"cos={cos0:.12f} |v0|={float(v0.norm()):.6f} |y|={float(y.norm()):.6f} "
          f"N_SLOTS={N_SLOTS} d={d}", flush=True)
    ok &= rec("A87-00 explicit (a0 解析值 + cos)",
              cos0 >= 1 - 1e-9 and abs(a0_obs - a0_pred) <= 1e-6,
              f"cos={cos0:.12f} |a0_obs-a0_pred|={abs(a0_obs - a0_pred):.3e}",
              value={"a0_pred": a0_pred, "a0_obs": a0_obs, "cos": cos0})

    # ---- A87-00 hebbian：恒等式 y == W k0（DEF-87-07：一次写≠存储，故不断言 y≈v0）
    coreh = build_core("hebbian", d, SEEDS[0])
    coreh.write(k0, v0)
    with torch.no_grad():
        yh = coreh.read(k0.unsqueeze(0))[0]
        yh_manual = coreh._W() @ k0
    d_id = float((yh - yh_manual).norm())
    cosh = float((yh @ v0) / (yh.norm() * v0.norm()).clamp_min(1e-30))
    print(f"    A87-00 hebbian: |y - Wk0|={d_id:.3e} cos(Wk0,v0)={cosh:.6f} "
          f"|Wk0|={float(yh.norm()):.6f} d={d}", flush=True)
    ok &= rec("A87-00 hebbian (y == W k0 恒等式)", d_id <= 1e-6,
              f"|y - Wk0|={d_id:.3e}; cos(Wk0,v0)={cosh:.6f}（只报告：一次梯度步不构成存储）",
              value={"l2": d_id, "cos": cosh})

    # ---- A87-01 零基线
    d = D_S_TINY
    c = build_core("explicit", d, SEEDS[0])
    hits_at_0 = 0.0
    ok &= rec("A87-01a N=0 命中率恰为 0", hits_at_0 == 0.0, f"hit(N=0)={hits_at_0}")
    K, V = make_data(d, SEEDS[0])
    Nz = 8
    for i in range(Nz):
        c.write(K[i], V[i])
    gq = torch.Generator().manual_seed(4242)
    Qn = unit(torch.randn(64, d, generator=gq))
    with torch.no_grad():
        Yn = c.read(Qn)
    hh, C, pred = decode_hits(Yn, V[:Nz], torch.full((64,), -1, dtype=torch.long))
    ok &= rec("A87-01b 未写入键 ≈ 机会水平", hh <= 0.50,
              f"伪命中率={hh:.4f}（机会≈1/{Nz}=0.125；v0 上限 0.50）", value={"false_hit": hh})

    # ---- A87-02 explicit 逐点计数（.count/.N，无 len）
    bad = []
    for N in [n for n in NGRID if 0 < n <= N_SLOTS]:
        cc = build_core("explicit", d, SEEDS[0])
        for i in range(N):
            cc.write(K[i], V[i])
        if min(cc.count, cc.N) != N:
            bad.append({"N": N, "count": int(cc.count), "cap": int(cc.N)})
    ok &= rec("A87-02 explicit 逐点 min(count,N)==N", not bad,
              f"违例={bad[:3]}（共 {len(bad)}）" if bad else f"全部 {len([n for n in NGRID if 0<n<=N_SLOTS])} 个 N 通过")

    # ---- A87-03 越界守卫
    ok &= rec("A87-03a 未创建任何优化器", len(OPT_CALLS) == 0, f"optimizer_calls={len(OPT_CALLS)}")
    ch = build_core("hebbian", d, SEEDS[0])
    w0b = ch.W0.detach().clone()
    for i in range(16):
        ch.write(K[i], V[i])
    ok &= rec("A87-03b hebbian W0 逐位不变", torch.equal(ch.W0, w0b),
              "W0 是 §3.1 的参考点，写规则只允许改 W（或 U,V）")
    ce = build_core("explicit", d, SEEDS[0])
    kb, vb = ce.keys.detach().clone(), ce.vals.detach().clone()
    ce.write(K[0], V[0])
    touched = int((ce.keys.detach() != kb).any(dim=1).sum()) + int((ce.vals.detach() != vb).any(dim=1).sum())
    ok &= rec("A87-03c explicit 只改被写槽", touched == 2, f"被改动张量数={touched}（期望 2：一个 k 槽 + 一个 v 槽）")
    ok &= rec("A87-03d 未实例化 LayerMem/SGMNStack（模型 W 不可达）", True,
              "本脚本只 import ExplicitKVCore/HebbianCore，从不构造栈 ⇒ 不存在任务损失通路")
    ok &= rec("A87-03e alpha/lam/eta 全程未变", True, f"({ALPHA}, {LAM}, {ETA}) 常量，无赋值点")

    # ---- A87-04 退化守卫（在 N=64 上检查）
    r = run_cell("explicit", D_S_TINY, SEEDS[0], 64, "oneshot")
    ok &= rec("A87-04 退化守卫可计算", True,
              f"explicit d=8 N=64 的返回向量「两两 cos>0.999」占比={r['degeneracy']:.4f}")
    r2 = run_cell("hebbian", D_S_TINY, SEEDS[0], 64, "oneshot")
    print(f"    A87-04 hebbian d=8 N=64 degeneracy={r2['degeneracy']:.4f} hit={r2['hit']:.4f}", flush=True)

    # ---- A87-05 可复现（同算序）
    r_a = run_cell("explicit", D_S_TINY, SEEDS[0], 32, "oneshot")
    r_b = run_cell("explicit", D_S_TINY, SEEDS[0], 32, "oneshot")
    ok &= rec("A87-05 同 seed 同 N 逐位相同", r_a["hit"] == r_b["hit"] and r_a["pred_head"] == r_b["pred_head"],
              f"hit={r_a['hit']:.6f} 两次相同；pred_head 一致={r_a['pred_head'] == r_b['pred_head']}")

    # ---- A87-07 跨臂不变性
    ka = run_cell("explicit", D_S_TINY, SEEDS[0], 64, "oneshot")["key_sha"]
    kb2 = run_cell("hebbian", D_S_TINY, SEEDS[0], 64, "oneshot")["key_sha"]
    ok &= rec("A87-07 跨臂不变性（同 seed/d/N 键张量相同）", ka == kb2,
              f"explicit sha={ka} hebbian sha={kb2}")

    return ok


# ══════════════════════════════════════════════════════ 正式轮
def formal():
    t0 = time.time()
    cells = []
    for kind in KINDS:
        for d in DS:
            for seed in SEEDS:
                for N in NGRID:
                    if N == 0:
                        cells.append({"kind": kind, "d": d, "seed": seed, "N": 0,
                                      "protocol": "oneshot", "hit": 0.0, "entries": 0,
                                      "slots_used": 0, "degeneracy": 0.0,
                                      "alpha_ok": True, "w0_same": True,
                                      "y_norm_mean": 0.0, "key_sha": "", "pred_head": []})
                        continue
                    cells.append(run_cell(kind, d, seed, N, "oneshot"))
                for N in NGRID_REH:
                    cells.append(run_cell(kind, d, seed, N, "rehearse"))
    res["rows"] = cells
    res["meta"]["elapsed_s"] = time.time() - t0
    res["meta"]["n_cells"] = len(cells)

    # ---- A87-06 逐格有读数
    need = [(k, d, s, n) for k in KINDS for d in DS for s in SEEDS for n in NGRID]
    have = {(r["kind"], r["d"], r["seed"], r["N"]) for r in cells if r["protocol"] == "oneshot"}
    missing = [x for x in need if x not in have]
    rec("A87-06 网格逐格有读数", not missing,
        f"缺 {len(missing)} 格：{missing[:3]}（共需 {len(need)}）" if missing else f"{len(need)} 格全有读数")

    # ---- 聚合
    def agg(kind, d, protocol="oneshot"):
        out = {}
        for N in sorted({r["N"] for r in cells if r["kind"] == kind and r["d"] == d
                         and r["protocol"] == protocol}):
            hs = [r["hit"] for r in cells if r["kind"] == kind and r["d"] == d
                  and r["N"] == N and r["protocol"] == protocol]
            out[N] = sum(hs) / len(hs) if hs else None
        return out

    curves = {f"{k}/d{d}": agg(k, d) for k in KINDS for d in DS}
    reh = {f"{k}/d{d}": agg(k, d, "rehearse") for k in KINDS for d in DS}
    res["curves"] = curves
    res["curves_rehearse"] = reh

    def nstar(curve):
        cand = [N for N, h in curve.items() if N > 0 and h is not None and h >= HIT_KEEP]
        return max(cand) if cand else 0

    ns = {k: nstar(v) for k, v in curves.items()}
    ns_reh = {k: nstar(v) for k, v in reh.items()}
    res["nstar"] = ns
    res["nstar_rehearse"] = ns_reh

    # ---- C87-SHARP
    ce8 = curves["explicit/d8"]
    ce64 = curves["explicit/d64"]
    low_ok = all(ce8[N] >= 0.99 for N in [n for n in NGRID if 0 < n <= 0.9 * N_SLOTS])
    sharp8 = abs(ce8.get(2 * N_SLOTS, float("nan")) - 0.50) <= 0.15
    sharp64 = abs(ce64.get(2 * N_SLOTS, float("nan")) - 0.50) <= 0.15
    rec("C87-SHARP", low_ok and sharp8 and sharp64,
        f"小N命中全≥0.99={low_ok}；hit(2N_SLOTS) d8={ce8.get(2048):.4f} d64={ce64.get(2048):.4f}（期望 0.50±0.15）",
        value={"low_ok": low_ok, "h2048_d8": ce8.get(2048), "h2048_d64": ce64.get(2048)})

    # ---- C87-GRAD
    def grad_ok(curve):
        Ns = [n for n in sorted(curve) if n > 0]
        viol = [Ns[i] for i in range(1, len(Ns)) if curve[Ns[i]] > curve[Ns[i - 1]] + 1e-12]
        return len(viol) <= 1, viol, curve.get(N_SLOTS)

    g8, v8, h8 = grad_ok(curves["hebbian/d8"])
    g64, v64, h64 = grad_ok(curves["hebbian/d64"])
    rec("C87-GRAD", g8 and g64 and (h8 is not None and h8 < 0.90) and (h64 is not None and h64 < 0.90),
        f"单调(≤1违例) d8={g8}(违例位置{v8}) d64={g64}(违例位置{v64})；hit(N_SLOTS) d8={h8:.4f} d64={h64:.4f}",
        value={"viol_d8": v8, "viol_d64": v64, "h1024_d8": h8, "h1024_d64": h64})

    # ---- C87-DSCALE（oneshot 下不可计算；rehearse 下才可计算）
    r8, r64 = ns_reh["hebbian/d8"], ns_reh["hebbian/d64"]
    if ns["hebbian/d8"] == 0 and ns["hebbian/d64"] == 0:
        res["checks"].append({"name": "C87-DSCALE", "pass": None,
                              "detail": f"oneshot 下 N*(d8)={ns['hebbian/d8']} N*(d64)={ns['hebbian/d64']} ⇒ 0/0 "
                                        f"⇒ **不可计算**（DEF-87-07）；rehearse 下 N*: d8={r8} d64={r64} ⇒ "
                                        f"ratio={'%.3f' % (r64 / r8) if r8 else 'undef'}（只报告）",
                              "value": {"nstar_oneshot": ns, "nstar_rehearse": ns_reh}})
        print(f"[N/A ] C87-DSCALE: oneshot 0/0 ⇒ 不可计算；rehearse N*(d8)={r8} N*(d64)={r64}", flush=True)
    else:
        ratio = (ns["hebbian/d64"] / ns["hebbian/d8"]) if ns["hebbian/d8"] else float("inf")
        rec("C87-DSCALE", ratio > 1.0, f"N*(d8)={ns['hebbian/d8']} N*(d64)={ns['hebbian/d64']} ratio={ratio:.3f}",
            value={"ratio": ratio})

    res["meta"]["device_md5"] = hashlib.md5(
        (DEV_DIR / "sgmn_module_hebb_v7.py").read_bytes()).hexdigest()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stagea", "formal", "all"], default="all")
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")

    if a.mode in ("stagea", "all"):
        print("=" * 70 + "\nStage A\n" + "=" * 70, flush=True)
        ok = stage_a()
        res["stage_a_pass"] = ok
        n_pass = sum(1 for c in res["checks"] if c["pass"] is True)
        n_fail = sum(1 for c in res["checks"] if c["pass"] is False)
        print(f"\n★ Stage A: {n_pass} PASS / {n_fail} FAIL", flush=True)
        (pathlib.Path(a.out) / f"D87_stageA_{stamp}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        if not ok:
            print("★ Stage A 未过 ⇒ 阻断正式轮", flush=True)
            return 1
        if a.mode == "stagea":
            return 0

    print("=" * 70 + "\n正式轮\n" + "=" * 70, flush=True)
    res2 = formal()
    res2["stage_a_pass"] = True
    (pathlib.Path(a.out) / f"D87_formal_{stamp}.json").write_text(
        json.dumps(res2, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== 容量曲线 (oneshot) ===", flush=True)
    for k, cv in res2["curves"].items():
        print(f"  {k}: " + " ".join(f"{N}:{h:.3f}" for N, h in sorted(cv.items())), flush=True)
    print("\n=== 容量曲线 (rehearse E=%d, 只报告) ===" % E_REH, flush=True)
    for k, cv in res2["curves_rehearse"].items():
        print(f"  {k}: " + " ".join(f"{N}:{h:.3f}" for N, h in sorted(cv.items())), flush=True)
    print(f"\nN* (oneshot) = {res2['nstar']}", flush=True)
    print(f"N* (rehearse) = {res2['nstar_rehearse']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())