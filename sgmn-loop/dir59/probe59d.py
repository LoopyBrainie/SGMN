#!/opt/pytorch/bin/python
r"""DIR-59 · **task 级探针 59D** —— 3 臂 × 3 格，为「$P_v$ 的其他来源」建立任务损失层的门槛

口径（`dir59/PROBE59_REV1.1.md §4`）：只报 $L$ 与由它导出的 $R=L_{\rm NONE}-L_{\rm OWN}$、$\Delta=L_{\rm FOREIGN}-L_{\rm OWN}$，
**不写判据、不产出裁决**。协议逐字用 D-76/DIR-53 的（250 步、lr 0.03、`n_rep=16`、96 训练实例、400 评估实例）。

用法（trainer）：
  `/opt/pytorch/bin/python dir59/probe59d.py --stagea`            # 自检 + 种子不相交 + 落 stageA JSON
  `/opt/pytorch/bin/python dir59/probe59d.py --cell C1 --arm OWN` # 跑一格一臂
  `/opt/pytorch/bin/python dir59/probe59d.py --summarize`         # 汇总（只报数）
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir59")
import R1_task59 as T59      # noqa: E402

OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir59/out")
# ══════════ 三格（**新段**，与 DIR-53 的 `20262201..20262205` 不重叠）
PV_SEEDS = {"C1": 20263301, "C2": 20263302, "C3": 20263303}
CELLS = ("C1", "C2", "C3")
ARMS = ("OWN", "FOREIGN", "NONE")
# ══════════ 训练/评估实例段（**新段**：`59800..59895` / `59900..60299`）
TRAIN = list(range(59800, 59896))
EVAL = list(range(59900, 60300))
TRAIN_OFF = [TRAIN[0], TRAIN[-1]]
EVAL_OFF = [EVAL[0], EVAL[-1]]
HISTORY = [(10000, 10399), (11000, 11549), (12000, 13095), (13500, 14511), (15000, 15899),
           (16000, 16411), (17000, 18911), (20400, 20999), (21000, 21111), (21500, 22063),
           (22100, 22103), (22500, 22899), (23000, 23003), (23990, 24399), (25000, 25019),
           (25200, 26199), (26200, 27199), (27400, 28399), (28800, 28819), (29400, 30419),
           (31000, 31199), (31400, 32419), (33000, 36999), (41000, 41039), (51000, 52999),
           (53000, 53019), (55000, 55023), (56000, 56095), (57000, 57095), (58000, 58463),
           (59000, 59095), (59500, 59595), (59600, 59631), (61000, 61007), (62000, 62399)]


def pv_foreign(cell):
    r"""外源 = **循环的下一格**的 $P_v$（同分布、同形状、逐位可取）。"""
    return PV_SEEDS[CELLS[(CELLS.index(cell) + 1) % len(CELLS)]]


def run(cell, arm):
    t0 = time.time()
    pv_own, pv_for = T59._pv1_for(PV_SEEDS[cell]), T59._pv1_for(pv_foreign(cell))
    if arm == "OWN":
        pv, alpha = pv_own, None
    elif arm == "FOREIGN":
        pv, alpha = pv_for, None
    elif arm == "NONE":
        pv, alpha = pv_own, 0.0
    else:
        raise ValueError(arm)
    print(f"[{cell}/{arm}] 训头（learned+ctrl，{T59.STEPS} 步，n={len(TRAIN)}）…", flush=True)
    phi, meta = T59.train_arm59("learned", TRAIN, shared_pv=pv, alpha=alpha, train_ctrl=True)
    print(f"[{cell}/{arm}] 训练完成 {time.time()-t0:.0f}s 梯度 {meta['gradnorms']}", flush=True)
    a = T59.eval_arm59("learned", phi, EVAL, shared_pv=pv, alpha=alpha)
    print(f"[{cell}/{arm}] L={a['mean_L']:.6f} (sd {a['sd_L']:.4f}) |q|={a['q_norm']:.4f} "
          f"Wmax={a['Wmax']:.4f} cos_q_k={a['cos_q_k']:.4f}", flush=True)
    rec = {"cell": cell, "arm": arm, "pv_seed": PV_SEEDS[cell],
           "pv_seed_used": (pv_foreign(cell) if arm == "FOREIGN" else PV_SEEDS[cell]),
           "alpha": (0.0 if arm == "NONE" else 1.0),
           "n_train": len(TRAIN), "train_off": TRAIN_OFF,
           "n_eval": len(EVAL), "eval_off": EVAL_OFF,
           "steps": T59.STEPS, "lr": T59.LR, "n_rep": T59.N_REP,
           "pv_sha": hashlib.sha256(pv.numpy().tobytes()).hexdigest(),
           "pv_fro": float(pv.norm()), "pv_cos_own_foreign": float(
               (pv_own / pv_own.norm() * (pv_for / pv_for.norm())).sum()),
           "mean_L": a["mean_L"], "sd_L": a["sd_L"], "L_series": a["L_series"],
           "q_norm": a["q_norm"], "y_norm": a["y_norm"], "m_abs": a["m_abs"],
           "Wmax": a["Wmax"], "cos_q_k": a["cos_q_k"], "align_q_k": a["align_q_k"],
           "gradnorms": meta["gradnorms"], "dWq1_from_init": meta["dWq1_from_init"],
           "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"PROBE59D_{cell}_{arm}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    print(f"[{cell}/{arm}] → {OUT}/PROBE59D_{cell}_{arm}.json", flush=True)


def stagea():
    g = T59.selftest()
    g["seed_disjoint_train"] = all(not (a <= s <= b) for a, b in HISTORY for s in (TRAIN[0], TRAIN[-1]))
    g["seed_disjoint_eval"] = all(not (a <= s <= b) for a, b in HISTORY for s in (EVAL[0], EVAL[-1]))
    g["seed_disjoint_cells"] = all(not (a <= s <= b) for a, b in HISTORY for s in PV_SEEDS.values())
    g["cells_distinct"] = len(set(PV_SEEDS.values())) == len(PV_SEEDS)
    g["train_eval_disjoint"] = not set(TRAIN) & set(EVAL)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PROBE59D_stageA.json").write_text(json.dumps(g, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    ok = g["all_pass"] and g["seed_disjoint_train"] and g["seed_disjoint_eval"] \
        and g["seed_disjoint_cells"] and g["cells_distinct"] and g["train_eval_disjoint"]
    print(json.dumps({k: v for k, v in g.items() if k != "md5"}, ensure_ascii=False, indent=1))
    print("★ 59D Stage A 全部通过" if ok else "★★ 59D Stage A 失败")
    return 0 if ok else 1


def summarize():
    rows = []
    for cell in CELLS:
        rec = {}
        for arm in ARMS:
            p = OUT / f"PROBE59D_{cell}_{arm}.json"
            if p.exists():
                rec[arm] = json.loads(p.read_text(encoding="utf-8"))
        if len(rec) != 3:
            print(f"[{cell}] 缺臂：{sorted(rec)}")
            continue
        L = {a: rec[a]["mean_L"] for a in ARMS}
        # 逐评估种子配对：三臂的 L_series 顺序相同（同 EVAL 列表、同头初值）
        d_own_none = [x - y for x, y in zip(rec["OWN"]["L_series"], rec["NONE"]["L_series"])]
        d_for_own = [x - y for x, y in zip(rec["FOREIGN"]["L_series"], rec["OWN"]["L_series"])]
        se = lambda v: statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")
        R = L["NONE"] - L["OWN"]
        rows.append({"cell": cell, "L_OWN": L["OWN"], "L_FOREIGN": L["FOREIGN"], "L_NONE": L["NONE"],
                     "R": R, "Delta": L["FOREIGN"] - L["OWN"],
                     "D_own_none": statistics.fmean(d_own_none), "se_D_own_none": se(d_own_none),
                     "3se_D_own_none": 3 * se(d_own_none),
                     "D_for_own": statistics.fmean(d_for_own), "se_D_for_own": se(d_for_own),
                     "3se_D_for_own": 3 * se(d_for_own),
                     "Delta_over_R": (L["FOREIGN"] - L["OWN"]) / R if R else float("nan"),
                     "3se_over_020R": (3 * se(d_for_own)) / (0.20 * R) if R else float("nan"),
                     "cos_pv": rec["OWN"]["pv_cos_own_foreign"]})
    print(f"{'cell':5s} {'L_OWN':>9s} {'L_FOR':>9s} {'L_NONE':>9s} {'R':>8s} {'Δ':>8s} "
          f"{'Δ/R':>7s} {'3se/0.2R':>8s} {'3se(D_own-none)':>15s}")
    for r in rows:
        print(f"{r['cell']:5s} {r['L_OWN']:9.5f} {r['L_FOREIGN']:9.5f} {r['L_NONE']:9.5f} "
              f"{r['R']:8.5f} {r['Delta']:+8.5f} {r['Delta_over_R']:+7.4f} "
              f"{r['3se_over_020R']:8.3f} {r['3se_D_own_none']:15.5f}")
    (OUT / "PROBE59D_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stagea", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cell")
    ap.add_argument("--arm")
    a = ap.parse_args()
    if a.list:
        for c in CELLS:
            print(c, PV_SEEDS[c], "foreign→", pv_foreign(c))
    elif a.stagea:
        sys.exit(stagea())
    elif a.summarize:
        summarize()
    elif a.cell and a.arm:
        run(a.cell, a.arm)
    else:
        ap.print_help()