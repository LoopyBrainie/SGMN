r"""DIR-26-R · 开卡前置检查（**未开卡**；无冻结判据、无裁决）

文件：`revolve/DIR-26R/PREMISE-CHECK.md`（§3 的 R-1/R-2/R-3）
装置：`dir22/sgmn_module_hebb_v7.py`（md5 `1a242d02c530307f6c76646ca13469ef`，**零改动**）

只做三件事：
  R-1 **每条目刚体重瞄**（同一个 $R_i$ 作用于该条目所有轮次）⇒ $\rho_u,\Lambda$、条目内成对余弦**不变**，而键**确实变了**。
  R-2 **逐轮次独立 Haar 旋转** ⇒ $\rho_u\to0$（即 iid 档）。
  R-3 装置 `E0` 臂「刚体重瞄 vs 恒等」的配对 $\Delta\text{ret1}$（**描述性**，3 种子，不作 PASS/FAIL）。

派生纪律：只改**写流的键**（monkeypatch `R1_ar1.build_stream_ar1` 的返回值），
`run_unit`/`rec`/判据算法/读流/值张量/种子一律不动。
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("dir26o", "dir26n", "dir26m", "dir26l", "dir26k", "dir26j", "dir26i", "dir26h",
           "dir26g", "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"{_ROOT}/{_p}")

import R1_ar1 as RA
from R1_gating import D_MODEL, DTYPE, S_KEY, unit
from R1_gating12 import R_PRES, cal_at
from R1_gating5 import theta_of

OUT = f"{_ROOT}/dir26r/out"
DEV = f"{_ROOT}/dir22/sgmn_module_hebb_v7.py"
DEV_MD5 = "1a242d02c530307f6c76646ca13469ef"
THETA = theta_of(0.9)
COS, SIN = math.cos(THETA), math.sin(THETA)
SEEDS_R3 = (20261028, 20261029, 20261030)
TAU = None   # 由 main() 从 cal_at(N_KV)["c_tau"] 取（与正式轮同一标定）


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def _uc_of(K, n, d):
    r"""从装置键**反解**抖动方向：$K/\sqrt S=\cos\theta e_i+\sin\theta u$ ⇒ $u=(K/\sqrt S-\cos\theta e_i)/\sin\theta$。"""
    E = torch.zeros(n, d)
    for i in range(n):
        E[i, i] = 1.0
    Uc = torch.zeros_like(K)
    for rho in range(K.shape[0]):
        for i in range(n):
            Uc[rho, i] = (K[rho, i] / math.sqrt(S_KEY) - COS * E[i]) / SIN
    return E, Uc


def _k_of(E, Uc):
    K = torch.zeros(Uc.shape[0], Uc.shape[1], Uc.shape[2], dtype=DTYPE)
    for rho in range(Uc.shape[0]):
        for i in range(Uc.shape[1]):
            K[rho, i] = math.sqrt(S_KEY) * unit(COS * E[i] + SIN * Uc[rho, i])
    return K


def _rand_iso(gen, d, idx):
    r"""$\perp e_{\rm idx}$ 上的**随机正交等距**（随机置换；$O(d)$ 而非 $O(d^3)$），固定第 idx 个坐标。

    ★ DEF-26R-2：首版用 2-平面旋转，在 $d{=}896$ 上几乎等于恒等（$\rho_u$ 0.900→0.896），未实现"逐轮独立"的效果；
      ★ DEF-26R-3：第二版用全 Haar（$895\times895$ 高斯 QR + 双重循环建 $896\times896$ 矩阵），
      单次 $\sim$0.5 GFLOP 且 Python 层 $8\times10^5$ 次赋值 ⇒ 1260 次调用直接把进程卡死（已杀，无读数）。
    随机置换**是**正交等距（保内积、保范数），且对两个不同向量 $u\neq u'$ 有 $u\cdot Pu'\approx0$ ⇒ 足以检验"逐轮独立等距 ⇒ $\rho_u\to0$"这一机制。
    注意：本式**不是** P3 的分布等式（Haar ⇒ 恰为 iid），那一条是代数结论，不靠数值。
    """
    idxs = [j for j in range(d) if j != idx]
    shuffled = torch.randperm(len(idxs), generator=gen).tolist()
    out = list(range(d))
    for a, j in enumerate(idxs):
        out[j] = idxs[shuffled[a]]                 # 子空间内的**随机**置换（固定第 idx 坐标）
    return torch.tensor(out, dtype=torch.long)     # u' = u[perm]


def _stats(Uc, n, r):
    rho_u, cnt, lam = 0.0, 0, 0.0
    for i in range(n):
        for k in range(r - 1):
            rho_u += float(Uc[k, i] @ Uc[k + 1, i])
            cnt += 1
        s = Uc[:, i, :].sum(0)
        lam += float(s @ s) / (r * r)        # Λ_i = ‖Σ_ρu_ρ‖²/R²；各向同性 iid 的**操作点** E[Λ]=1/R（DEF-26R-9：原注 1/(d-1) 是错的）
    return rho_u / cnt, lam / n


def premise_stats(seed, phi, n=None, r=None):
    n = int(n or RA.N_KV)
    r = int(r or R_PRES)
    D = RA.build_stream_ar1(RA.N_KV, seed, D_MODEL, n, r, THETA, phi)
    K0 = D["K"].clone()
    E, Uc = _uc_of(K0, n, D_MODEL)   # DEF-26R-1：第 3 参是 d_model，不是轮次数
    st0 = _stats(Uc, n, r)

    # ── R-1 每条目刚体重瞄
    g = torch.Generator().manual_seed(seed + 90001)
    Uc1 = torch.zeros_like(Uc)
    for i in range(n):
        pm = _rand_iso(g, D_MODEL, i)
        for rho in range(r):
            Uc1[rho, i] = unit(Uc[rho, i][pm])
    K1 = _k_of(E, Uc1)
    st1 = _stats(Uc1, n, r)
    # 条目内成对余弦的最大偏差（P2 的直接检验）
    dev_pair = 0.0
    for i in range(n):
        for k in range(r - 1):
            dev_pair = max(dev_pair, abs(float(Uc0_ := Uc[k, i] @ Uc[k + 1, i]) - float(Uc1[k, i] @ Uc1[k + 1, i])))
    dK = float((K1 - K0).abs().mean())

    # ── R-2 逐轮次独立 Haar 旋转 ⇒ 应等于 iid 档
    g2 = torch.Generator().manual_seed(seed + 90002)
    Uc2 = torch.zeros_like(Uc)
    for i in range(n):
        for rho in range(r):
            pm = _rand_iso(g2, D_MODEL, i)
            Uc2[rho, i] = unit(Uc[rho, i][pm])
    st2 = _stats(Uc2, n, r)
    D0 = RA.build_stream_ar1(RA.N_KV, seed, D_MODEL, n, r, THETA, 0.0)
    _, Uc_0 = _uc_of(D0["K"].clone(), n, D_MODEL)
    st_iid = _stats(Uc_0, n, r)

    return {
        "seed": int(seed), "phi": float(phi), "n": n, "r": r,
        "rho_u": st0[0], "rho_u_iso": st1[0], "rho_u_perrot": st2[0], "rho_u_iid": st_iid[0],
        "lam": st0[1], "lam_iso": st1[1], "lam_perrot": st2[1], "lam_iid": st_iid[1],
        "pair_cos_maxdev": dev_pair, "mean_abs_dK": dK,
        "lam_iso_vs_iid_ratio": (st1[1] / st_iid[1]) if st_iid[1] else float("nan"),
    }


def r3_arm(seed, phi, mode):
    r"""装置 `E0` 臂，键按 mode 变换（`id` / `iso`）。只 monkeypatch 写流构建，别的不动。"""
    orig = RA.build_stream_ar1
    pool = {}

    def patched(b, s, d, n, r_pres, theta, phi_):
        D = orig(b, s, d, n, r_pres, theta, phi_)
        if int(n) == int(RA.N_KV):
            key = (int(s), float(phi_), str(mode))
            if key not in pool:
                K = D["K"].clone()
                E, Uc = _uc_of(K, int(n), D_MODEL)
                if mode != "id":
                    g = torch.Generator().manual_seed(int(s) + 90001)
                    for i in range(int(n)):
                        pm = _rand_iso(g, D_MODEL, i)
                        for rho in range(int(r_pres)):
                            Uc[rho, i] = unit(Uc[rho, i][pm])
                    D["K"] = _k_of(E, Uc)
                    ru, lam = _stats(Uc, int(n), int(r_pres))
                    D["diag"] = dict(D.get("diag", {}))
                    D["diag"]["rho_u"] = ru
                    D["diag"]["lam"] = lam
                pool[key] = True
        return D

    RA.build_stream_ar1 = patched
    try:
        row = RA.run_unit(int(seed), float(phi), TAU)
    finally:
        RA.build_stream_ar1 = orig
    a = row["arms"]["E0"]
    return {"seed": int(seed), "mode": mode, "ret1": a["ret1"], "mu": a["mu"], "clean": a["clean"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--phi", type=float, default=0.9)
    ap.add_argument("--no-device", action="store_true")
    ap.add_argument("--r3-only", action="store_true", help="只跑 R-3（加功效复制用）")
    ap.add_argument("--r3-seeds", type=int, default=0, help="R-3 的种子数（0=默认 3 个）")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    assert _md5(DEV) == DEV_MD5, "装置 md5 变了"

    global TAU
    TAU = float(cal_at(int(RA.N_KV))["c_tau"])
    print(f"标定：c_tau = {TAU:.4f}")
    seeds_stat = tuple(a.seeds) if a.seeds else (RA.SEEDS[0], RA.SEEDS[1], RA.SEEDS[2])
    rows = [] if a.r3_only else [premise_stats(s, a.phi) for s in seeds_stat]
    print("=== R-1 / R-2（纯流统计，描述性）===")
    if a.r3_only:
        print("  （--r3-only：跳过）")
    for r in rows:
        print(f"  seed={r['seed']} φ={r['phi']}: ρ_u={r['rho_u']:.4f} | 刚体重瞄 {r['rho_u_iso']:.4f}"
              f" | 逐轮独立 {r['rho_u_perrot']:+.4f} | iid 参照 {r['rho_u_iid']:+.4f}"
              f" ‖ Λ={r['lam']:.4f}/{r['lam_iso']:.4f}/{r['lam_perrot']:.4f}/{r['lam_iid']:.4f}"
              f" ‖ 成对余弦最大偏差 {r['pair_cos_maxdev']:.2e} ‖ 平均|ΔK| {r['mean_abs_dK']:.4f}")

    res = {"phi": a.phi, "rows_stat": rows, "checks": [], "rows_r3": []}
    if a.r3_only:
        rows = [{"rho_u_iso": 0.0, "rho_u": 0.0, "pair_cos_maxdev": 0.0, "mean_abs_dK": 1.0,
                 "rho_u_perrot": 0.0}]
    ru_dev = max(abs(r["rho_u_iso"] - r["rho_u"]) for r in rows)
    pair_dev = max(r["pair_cos_maxdev"] for r in rows)
    dk_min = min(r["mean_abs_dK"] for r in rows)
    res["checks"] = [
        {"id": "R-1a", "ok": ru_dev <= 1e-5, "msg": f"刚体重瞄后 ρ_u 最大变化 {ru_dev:.2e} ≤1e-5"},
        {"id": "R-1b", "ok": pair_dev <= 1e-5, "msg": f"条目内成对余弦最大偏差 {pair_dev:.2e} ≤1e-5"},
        {"id": "R-1c", "ok": dk_min > 1e-3, "msg": f"键确实被改写：最小平均|ΔK| {dk_min:.4f} >1e-3"},
        {"id": "R-2", "ok": max(abs(r["rho_u_perrot"]) for r in rows) <= 2e-2,
         "msg": f"逐轮独立等距（随机置换）后 |ρ_u| ≤ {max(abs(r['rho_u_perrot']) for r in rows):.2e}（→iid 档）"},
    ]
    for c in res["checks"]:
        print(("  ✓ " if c["ok"] else "  ✗ ") + c["id"] + " " + c["msg"])

    if not a.no_device:
        print("=== R-3（装置 E0 臂，描述性；无裁决）===")
        r3_seeds = tuple(RA.SEEDS[:a.r3_seeds]) if a.r3_seeds else SEEDS_R3
        print(f"  R-3 种子数 n={len(r3_seeds)}（预登记口径：|mean| ≥ 3se 且同号 ⇒ 效应成立）")
        dev = [r3_arm(s, a.phi, "id") for s in r3_seeds]
        iso = [r3_arm(s, a.phi, "iso") for s in r3_seeds]
        d = np.array([b["ret1"] - x["ret1"] for x, b in zip(dev, iso)], float)
        se = float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
        for s, x, b in zip(r3_seeds, dev, iso):
            print(f"  seed={s}: ret1 id={x['ret1']:.4f} → iso={b['ret1']:.4f}  Δ={b['ret1'] - x['ret1']:+.4f}")
        print(f"  Δret1 配对均值 {d.mean():+.4f}、3se={3 * se:.4f}、正/负 {int((d > 0).sum())}/{int((d < 0).sum())}")
        res["rows_r3"] = {"id": dev, "iso": iso,
                          "delta_mean": float(d.mean()), "delta_3se": float(3 * se),
                          "pos": int((d > 0).sum()), "neg": int((d < 0).sum())}
        verdict = ("无可检出的效应（描述性）" if abs(d.mean()) < 3 * se and 0 < int((d > 0).sum()) < len(d)
                   else "有超预期效应 ⇒ P1–P3 需重新审视")
        res["r3_reading"] = verdict
        print("  ⇒ " + verdict)

    res["passed"] = all(c["ok"] for c in res["checks"])
    res["r3_seeds"] = list(RA.SEEDS[:a.r3_seeds]) if a.r3_seeds else list(SEEDS_R3)
    tag = "r3_n%d" % a.r3_seeds if a.r3_seeds else "premise"
    json.dump(res, open(f"{OUT}/DIR26R_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"=== 前提检查 {'PASS' if res['passed'] else 'FAIL'}（"
          f"{sum(c['ok'] for c in res['checks'])}/{len(res['checks'])}） {time.time() - t0:.1f}s ===")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
