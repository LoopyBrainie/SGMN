#!/opt/pytorch/bin/python
r"""DIR-52 · 工作例（卡 §2.4）—— 开卡后、Stage A 内的**必过**三项逐位核验。

① `μ(4)`（截断到 4 轮）必须与**未截断**的同装置读数**逐位相同**（`max|Δ| = 0`）
   ⇒ 证明「截断到 4 轮 ≡ 完整呈现」，即唯一增量没有改坏装置；
② `Σ_ρ ΔδG_ρ − ΔG(4) ≤ 1e-15`（卡 §2.3 的算术恒等式）；
③ `ΔδG_1` 与 `0` **逐位相同**（卡 §2.2 的结构恒等式 F-52-a）。
"""
import json
import os
import sys

import numpy as np
import torch

for _p in ("dir26s", "dir26r", "dir26o", "dir26k", "dir26j", "dir26i", "dir26h", "dir26g",
           "dir26f", "dir26e", "dir26c", "dir26b", "dir26", "dir22", "dir52"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_p}")

import R1_gating as G                      # noqa: E402
import R1_gating7 as G7                    # noqa: E402
import R1_gating12 as K12                  # noqa: E402
import R1_dose_ar1_v3 as V3                # noqa: E402
import R1_task52 as T52                    # noqa: E402
from R1_gating import D_MODEL, M, drive, make_core     # noqa: E402
from R1_gating7 import build_keep, PROFILE             # noqa: E402
from R1_gating12 import N_READ, R_HO8, R_PRES, cal_at, read_stream, ret_jitter_margin  # noqa: E402

SEED = 61000


def mu_at(seed, D, keep, vb, RS8):
    c = make_core(D_MODEL, seed, K12.R0)
    drive(c, D, "R", keep=keep)
    P = c._W().detach() - c.W0
    return ret_jitter_margin(P, RS8["K"], vb, N_READ, R_HO8)["mu"]


def main():
    tau = cal_at(V3.N_KV)
    out = {"seed": SEED, "checks": {}}
    unit = V3.run_unit_dose(SEED, 0.0, tau, cnt0=None)
    cnt = [int(x) for x in unit["counts"]]
    D = V3.build_write(SEED, 0.0, V3.N_KV)
    RS8 = read_stream(SEED, R_HO8)
    vb = D["vb"][:N_READ]

    # ① 截断到 4 轮 ≡ 完整呈现（与 run_unit_dose 的 Fu/Ku 逐位比）
    d1 = 0.0
    for pk in ("F", "K"):
        keep, _ = build_keep(V3.N_KV, R_PRES, cnt, PROFILE[pk])
        kt = T52.trunc_keep(keep, R_PRES)
        assert len(kt) == len(keep) == sum(cnt), (len(kt), len(keep), sum(cnt))
        mu_t = mu_at(SEED, D, kt, vb, RS8)
        mu_full = unit["arms"][pk + "u"]["mu"]
        d1 = max(d1, abs(mu_t - mu_full))
    out["checks"]["A"] = {"desc": "μ(4) 截断 ≡ 完整呈现（逐位）", "max_abs_diff": d1, "pass": d1 == 0.0}

    # ③ ΔδG_1 ≡ 0：第 1 轮的两档 φ 写流必须逐位相同
    D9 = V3.build_write(SEED, 0.9, V3.N_KV)
    out["checks"]["C_keys"] = {
        "desc": "第 0 轮键与值对两档 φ 逐位相同（F-52-a 的结构恒等式）",
        "K0_equal": bool(torch.equal(D["K"][0], D9["K"][0])),
        "va_equal": bool(torch.equal(D["va"], D9["va"])),
        "vb_equal": bool(torch.equal(D["vb"], D9["vb"])),
    }
    out["checks"]["C_keys"]["pass"] = (out["checks"]["C_keys"]["K0_equal"]
                                      and out["checks"]["C_keys"]["va_equal"]
                                      and out["checks"]["C_keys"]["vb_equal"])
    d3 = 0.0
    for pk in ("F", "K"):
        keep, _ = build_keep(V3.N_KV, R_PRES, cnt, PROFILE[pk])
        k1 = T52.trunc_keep(keep, 1)
        m0 = mu_at(SEED, D, k1, vb, RS8)
        m9 = mu_at(SEED, D9, k1, vb, RS8)
        d3 = max(d3, abs(m0 - m9))
    out["checks"]["C"] = {"desc": "ΔδG_1 ≡ 0（ρ=1 两档 φ 读数逐位相同）", "max_abs_diff": d3, "pass": d3 == 0.0}

    # ② 算术恒等式
    def G_of(Dx, rho):
        keepf, _ = build_keep(V3.N_KV, R_PRES, cnt, PROFILE["F"])
        keepk, _ = build_keep(V3.N_KV, R_PRES, cnt, PROFILE["K"])
        return (mu_at(SEED, Dx, T52.trunc_keep(keepk, rho), vb, RS8)
                - mu_at(SEED, Dx, T52.trunc_keep(keepf, rho), vb, RS8))
    G0, G9 = {r: G_of(D, r) for r in (1, 2, 4)}, {r: G_of(D9, r) for r in (1, 2, 4)}
    dG = {r: G9[r] - G0[r] for r in (1, 2, 4)}
    inc = {1: dG[1], 2: dG[2] - dG[1], 4: dG[4] - dG[2]}
    err = abs(sum(inc.values()) - dG[4])
    out["checks"]["B"] = {"desc": "Σ_ρ ΔδG_ρ = ΔG(4)（算术恒等式）", "dG": dG, "inc": inc,
                          "abs_err": err, "pass": err <= 1e-15}
    out["pass"] = all(v["pass"] for k, v in out["checks"].items())
    os.makedirs("/home/ec2-user/sgmn-loop/dir52/out", exist_ok=True)
    json.dump(out, open("/home/ec2-user/sgmn-loop/dir52/out/DIR52_workex.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in out["checks"].items():
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}  {v['desc']}")
        if not v["pass"]:
            print("     ", json.dumps(v, ensure_ascii=False)[:300])
    print("工作例总体：", "PASS" if out["pass"] else "FAIL")
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
