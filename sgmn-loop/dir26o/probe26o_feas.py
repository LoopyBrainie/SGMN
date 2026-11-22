r"""DIR-26-O · 探针（**探针级，非裁决**）：AR(1) 输入的效应量级、功效核算与单位成本。

只跑 3 种子 × 4 档 φ × 2 臂（E0/τ），输出：
  ① 逐种子配对差 $\Delta{\rm ret}_1,\Delta\mu,\Delta{\rm issued}$（对 φ=0）与**效应量级**；
  ② 抖动相关 $\rho_u$ 与键相关 $\rho_k$（随 φ 的走向）；
  ③ 键矩的边缘不变性（相对极差）；
  ④ 每驱动成本（供正式轮预算核算）。
★ 探针**不得**用于调卡片阈值（卡片 rev 1.1 已冻结）。
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26o")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26n")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26m")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26l")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26k")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26j")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26i")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26h")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26g")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26f")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26e")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26c")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26b")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir26")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir22")

import R1_ar1 as RA

OUT = "/home/ec2-user/sgmn-loop/dir26o/out"
SEEDS = RA.SEEDS[:3]


def main():
    os.makedirs(OUT, exist_ok=True)
    tau = RA.cal_at(RA.N_KV)["c_tau"]
    rows = []
    t0 = time.time()
    for s in SEEDS:
        for p in RA.PHIS:
            tt = time.time()
            u = RA.run_unit(s, p, tau)
            u["sec"] = time.time() - tt
            rows.append(u)
            print(f"  seed={s} φ={p:.1f} | E0 {u['arms']['E0']['ret1']:.4f}/mu {u['arms']['E0']['mu']:+.4f} | "
                  f"τ {u['arms']['tau']['ret1']:.4f}/mu {u['arms']['tau']['mu']:+.4f}/gate {u['arms']['tau']['gate_rate']:.3f} | "
                  f"ρ_u {u['diag']['rho_u']:.3f} ρ_k {u['diag']['rho_k']:.3f} | {u['sec']:.0f}s", flush=True)
    by = {(r["seed"], r["phi"]): r for r in rows}
    seeds = sorted(SEEDS)
    out = {"tau": tau, "seeds": seeds, "phis": list(RA.PHIS), "rows": rows,
           "cost_per_drive_s": float(np.mean([r["sec"] for r in rows]))}
    for arm in RA.ARMS:
        for f in ("ret1", "mu", "issued"):
            out[f"{arm}|{f}"] = {}
            for p in RA.PHIS[1:]:
                d = np.array([by[(s, p)]["arms"][arm][f] - by[(s, 0.0)]["arms"][arm][f] for s in seeds], float)
                out[f"{arm}|{f}"][str(p)] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                                            "pos": int((d > 0).sum()), "neg": int((d < 0).sum()),
                                            "vals": [round(float(x), 6) for x in d]}
    km = {f: {str(p): float(np.mean([by[(s, p)]["diag"]["km"][f] for s in seeds])) for p in RA.PHIS}
          for f in ("k2", "cosp", "k4", "kd")}
    out["kmom"] = km
    out["kmom_rel_range"] = {f: float((max(v.values()) - min(v.values())) / abs(sum(v.values()) / len(v)))
                              for f, v in km.items()}
    out["rho_u"] = {str(p): float(np.mean([by[(s, p)]["diag"]["rho_u"] for s in seeds])) for p in RA.PHIS}
    out["rho_k"] = {str(p): float(np.mean([by[(s, p)]["diag"]["rho_k"] for s in seeds])) for p in RA.PHIS}
    p = f"{OUT}/probe26o_feas.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n===== 探针摘要（**非裁决**）=====")
    print(f"  τ={tau:.4f}；每驱动 {out['cost_per_drive_s']:.1f}s；总 {time.time() - t0:.0f}s")
    print(f"  ρ_u: {out['rho_u']}")
    print(f"  ρ_k: {out['rho_k']}")
    print(f"  键矩相对极差: {out['kmom_rel_range']}")
    for arm in RA.ARMS:
        for f in ("ret1", "mu"):
            d = out[f"{arm}|{f}"]
            print(f"  {arm}|{f}: " + "  ".join(f"φ={p}: {d[p]['mean']:+.4f}(+{d[p]['pos']}/-{d[p]['neg']})"
                                               for p in ("0.3", "0.6", "0.9")))
    print(f"  → {p}")


if __name__ == "__main__":
    main()
