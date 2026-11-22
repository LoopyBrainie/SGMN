#!/opt/pytorch/bin/python
r"""DIR-52 · 实现：**AR(1) 后重优势放大的轮次定位**。

卡：`revolve/DIR-52/DIRECTION_rev52_v10.md`（rev 1.0，判据与全部数值一次冻结）。

★ **逐字复用**（零改动）：`dir26s/R1_dose_ar1_v3.py::{build_write,run_unit_dose}`（frozen `cnt0`）·
  `dir26g/R1_gating7.py::{build_keep,PROFILE}` · `dir26/R1_gating.py::drive` ·
  `dir26i/R1_gating10.py::ret_jitter_margin`（唯一判据算法）。
★ **唯一增量** = 把 `keep` **截断到轮边界**（`{t : t < ρ·n·2M}`）后 `drive` 并读 `μ(ρ)`。

判据量（卡 §2）：
  `EARLY = ΔG(2)`，`TOTAL = ΔG(4)`，`LATE = TOTAL − EARLY`，`SHAPE := EARLY − LATE`
  其中 `ΔG(ρ) = [μ_{0.9,K}(ρ) − μ_{0.9,F}(ρ)] − [μ_{0}(ρ) − μ_{0,F}(ρ)]`

用法：`R1_task52.py --seed0 62000 --n 100 --out dir52/out/DIR52r10_shard0.jsonl`
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

for _p in ("dir26s", "dir26r", "dir26o", "dir26k", "dir26j", "dir26i", "dir26h", "dir26g",
           "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G                      # noqa: E402
import R1_gating2 as G2                    # noqa: E402
import R1_gating7 as G7                    # noqa: E402
import R1_gating12 as K12                  # noqa: E402
import R1_dose_ar1_v3 as V3                # noqa: E402
from R1_gating import D_MODEL, M, drive, make_core                  # noqa: E402
from R1_gating7 import build_keep, PROFILE                          # noqa: E402
from R1_gating12 import (N_READ, R_HO8, R_PRES, cal_at, read_stream,
                         ret_jitter_margin, wsig)                   # noqa: E402

N_KV = V3.N_KV
PHIS = (0.0, 0.9)
RHO_STOP = 4                      # 只测 ρ=2 与 ρ=4（ρ=1 结构性为 0；ρ=3 不进判据）
TAUS = {}


def trunc_keep(keep, rho):
    r"""把 `keep` 截到「只呈现前 `rho` 轮」：`stream_idx < rho·n·2M`（卡 §3 的唯一增量）。"""
    ub = rho * N_KV * 2 * M
    return {t for t in keep if t < ub}


def read_mu(seed, D, keep, vb, RS8, W0):
    r"""驱动 + 读 `μ`（纯读）。返回 `(mu, W_cos_sig, W_norm)`。"""
    c = make_core(D_MODEL, seed, K12.R0)
    drive(c, D, "R", keep=keep)
    W = c._W().detach()
    P = W - c.W0
    m = ret_jitter_margin(P, RS8["K"], vb, N_READ, R_HO8)
    return m["mu"], wsig(W), float(W.norm())


def run_seed(seed):
    r"""一个种子 = 两档 `φ` × 两臂 `F/K` × `{ρ=0,2,4}` 读数（`ρ=4` 即完整呈现）。"""
    tau = TAUS["tau"]
    D0 = V3.build_write(seed, 0.0, N_KV)
    RS8 = read_stream(seed, R_HO8)
    vb = D0["vb"][:N_READ]
    out = {"seed": seed, "phi": {}, "c_tau": float(tau["c_tau"])}

    # ── φ=0：B 臂给出 cnt0（frozen 来源）；φ=0.9 复用
    unit0 = V3.run_unit_dose(seed, 0.0, tau, cnt0=None)
    cnt0 = [int(x) for x in unit0["counts"]]
    out["cnt_source_phi0"] = unit0["cnt_source"]
    out["cnt0"] = cnt0

    for phi, cnt_src, cnt in ((0.0, "B@this_phi", cnt0), (0.9, "frozen_cnt0", cnt0)):
        D = D0 if phi == 0.0 else V3.build_write(seed, phi, N_KV)
        rec = {"cnt_source": cnt_src, "mu": {}, "wsig": {}, "wnorm": {}}
        # ρ=0：未写基座
        c0 = make_core(D_MODEL, seed, K12.R0)
        m0 = ret_jitter_margin(c0._W().detach() - c0.W0, RS8["K"], vb, N_READ, R_HO8)
        rec["mu"]["0"] = m0["mu"]
        for pk in ("F", "K"):
            keep, per_item = build_keep(N_KV, R_PRES, cnt, PROFILE[pk])
            assert len(keep) == sum(cnt), (pk, len(keep), sum(cnt))
            rec.setdefault("keep_n", {})[pk] = {}
            rec["keep_n"][pk][str(RHO_STOP)] = len(keep)
            if phi == 0.0:
                # ★ 复用 `run_unit_dose` 已算好的 Fu/Ku 全量读数（= ρ=4，省 2 次 drive）
                rec["mu"].setdefault(pk, {})[str(RHO_STOP)] = unit0["arms"][pk + "u"]["mu"]
                rec["wsig"].setdefault(pk, {})[str(RHO_STOP)] = unit0["arms"][pk + "u"]["wsig"]
                rec["wnorm"].setdefault(pk, {})[str(RHO_STOP)] = float("nan")
            for rho in ((2,) if phi == 0.0 else (2, RHO_STOP)):
                kt = trunc_keep(keep, rho)
                rec["keep_n"][pk][str(rho)] = len(kt)
                mu, sig, wn = read_mu(seed, D, kt, vb, RS8, None)
                rec["mu"].setdefault(pk, {})[str(rho)] = mu
                rec["wsig"].setdefault(pk, {})[str(rho)] = sig
                rec["wnorm"].setdefault(pk, {})[str(rho)] = wn
        out["phi"][str(phi)] = rec

    # ── 判据量
    def G_of(phi_rec, rho):
        return phi_rec["mu"]["K"][str(rho)] - phi_rec["mu"]["F"][str(rho)]

    # ★ ΔG(0) ≡ 0 是**结构性恒等**（两臂在未写时共用同一个 W0 ⇒ G_φ(0) = μ_base − μ_base = 0）
    dG = {"0": 0.0,
          "2": G_of(out["phi"]["0.9"], 2) - G_of(out["phi"]["0.0"], 2),
          "4": G_of(out["phi"]["0.9"], 4) - G_of(out["phi"]["0.0"], 4)}
    early = dG["2"] - dG["0"]
    total = dG["4"] - dG["0"]
    out["dG"] = dG
    out["EARLY"] = early
    out["TOTAL"] = total
    out["LATE"] = total - early
    out["SHAPE"] = early - (total - early)
    out["identity_err"] = abs((early + (total - early)) - total)
    out["finite"] = bool(np.isfinite([early, total, out["SHAPE"]]).all())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="r10")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    TAUS["tau"] = cal_at(N_KV)
    done = set()
    if a.resume and os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            line = line.strip()
            if line:
                done.add(json.loads(line)["seed"])
    t0 = time.time()
    with open(a.out, "a" if a.resume else "w", encoding="utf-8") as f:
        for i in range(a.n):
            s = a.seed0 + i
            if s in done:
                continue
            row = run_seed(s)
            row["tag"] = a.tag
            f.write(json.dumps(row, ensure_ascii=False) + "\n")     # ★ 算完即落盘（纪律 108）
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"[shard {a.seed0}] {i+1}/{a.n}  {time.time()-t0:.1f}s", flush=True)
    print(f"DONE seed0={a.seed0} n={a.n} → {a.out}  墙钟 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
