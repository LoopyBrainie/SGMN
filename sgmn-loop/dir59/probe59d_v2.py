#!/opt/pytorch/bin/python
r"""DIR-59 · **task 级探针 59D · rev 2** —— 修 `DEF-59-2`（循环外源 ⇒ 结构恒等）与 `DEF-59-3`（漏 `shared_pk`）

与 `probe59d.py`（rev 1，**原样留档**）的**全部差异**：
  ① `DEF-59-3`：`shared_pk = pk1_for(cell)`（**逐字照 `dir53/R1_task53.py` 的 `pk1_for`**：`randn*s_h*KEY_SCALE`）
     —— rev 1 传的是 `shared_pk=None`，偏离 D-76 协议 ⇒ 读出 $L\approx20\text{–}25$（对比 DIR-53 的 1.7–2.4），
     且"记忆有害"是**漏共享键映射**的假象。
  ② `DEF-59-2`：外源 $P_v$ 改用**独立种子段** `20263311..13`（rev 1 用"循环下一格"⇒ 三个 $P_v$ 各当一次 own/foreign
     ⇒ $\sum_j\Delta_j\equiv0$ 的**结构恒等**；`alpha=0` 臂与 $P_v$ 无关 ⇒ 三次 `NONE` 逐位相同——已实测
     `C2_OWN` 与 `C1_FOREIGN` 的 `mean_L`/`sd`/`|q|`/`Wmax`/`cos` **全部逐位同**）。
  ③ `NONE` 只跑一次（$P_v$、$P_k$ 都不进 $\alpha{=}0$ 的写步）⇒ 七次训练而非九次。
  ④ `OWN` 臂额外评估 `rand_content=True`（= DIR-53 的 `OFF_rc` 位置）⇒ **内容锁定**诊断。

用法：`--stagea` · `--cell C?` `--arm OWN|FOREIGN|NONE` · `--summarize`
"""
import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir59")
import R1_task59 as T59      # noqa: E402
import R1_task33 as R        # noqa: E402

OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir59/out")
PK1_SEEDS = {"C1": 20263321, "C2": 20263322, "C3": 20263323}   # ★ 新段（DIR-53 用 20262201..05）
PV_OWN_SEEDS = {"C1": 20263301, "C2": 20263302, "C3": 20263303}
PV_FOR_SEEDS = {"C1": 20263311, "C2": 20263312, "C3": 20263313}  # ★ 独立（非循环）
CELLS = ("C1", "C2", "C3")
ARMS = ("OWN", "FOREIGN", "NONE")
TRAIN = list(range(59800, 59896))
EVAL = list(range(59900, 60300))
HISTORY = [(10000, 10399), (11000, 11549), (12000, 13095), (13500, 14511), (15000, 15899),
           (16000, 16411), (17000, 18911), (20400, 20999), (21000, 21111), (21500, 22063),
           (22100, 22103), (22500, 22899), (23000, 23003), (23990, 24399), (25000, 25019),
           (25200, 26199), (26200, 27199), (27400, 28399), (28800, 28819), (29400, 30419),
           (31000, 31199), (31400, 32419), (33000, 36999), (41000, 41039), (51000, 52999),
           (53000, 53019), (55000, 55023), (56000, 56095), (57000, 57095), (58000, 58463),
           (59000, 59095), (59500, 59595), (59600, 59631), (61000, 61007), (62000, 62399)]


def pk1_for(cell):
    r"""**逐字照 `dir53/R1_task53.py::pk1_for`**（`randn(D_H,D_K)*s_h*KEY_SCALE`）。"""
    g = torch.Generator().manual_seed(int(PK1_SEEDS[cell]))
    s_h = 1.0 / math.sqrt(T59.D_H)
    return torch.randn(T59.D_H, T59.D_K, generator=g) * s_h * R.KEY_SCALE


def run(cell, arm):
    t0 = time.time()
    pk = pk1_for(cell)
    pv_own, pv_for = T59._pv1_for(PV_OWN_SEEDS[cell]), T59._pv1_for(PV_FOR_SEEDS[cell])
    if arm == "OWN":
        pv, alpha = pv_own, None
    elif arm == "FOREIGN":
        pv, alpha = pv_for, None
    else:
        pv, alpha = pv_own, 0.0
    print(f"[{cell}/{arm}] 训头（learned+ctrl，{T59.STEPS} 步，n={len(TRAIN)}，"
          f"shared_pk={PK1_SEEDS[cell]}，pv={PV_OWN_SEEDS[cell] if arm!='FOREIGN' else PV_FOR_SEEDS[cell]}，"
          f"alpha={'0(地板)' if arm=='NONE' else 1})", flush=True)
    phi, meta = T59.train_arm59("learned", TRAIN, shared_pk=pk, shared_pv=pv, alpha=alpha,
                               train_ctrl=True)
    print(f"[{cell}/{arm}] 训练完成 {time.time()-t0:.0f}s 梯度 {meta['gradnorms']}", flush=True)
    a = T59.eval_arm59("learned", phi, EVAL, shared_pk=pk, shared_pv=pv, alpha=alpha)
    print(f"[{cell}/{arm}] L={a['mean_L']:.6f} (sd {a['sd_L']:.4f}) |q|={a['q_norm']:.4f} "
          f"|y|={a['y_norm']:.4f} m̄={a['m_abs']:.4f} Wmax={a['Wmax']:.4f} cos={a['cos_q_k']:.4f}",
          flush=True)
    extra = {}
    if arm == "OWN":
        rc = T59.eval_arm59("learned", phi, EVAL, shared_pk=pk, shared_pv=pv, rand_content=True)
        extra["OWN_rc"] = {k: v for k, v in rc.items() if k != "L_series"}
        extra["L_series_OWN_rc"] = rc["L_series"]
        print(f"[{cell}/{arm}] OWN_rc（内容置换）L={rc['mean_L']:.6f} (sd {rc['sd_L']:.4f})", flush=True)
    rec = {"cell": cell, "arm": arm, "protocol": "D-76 (shared_pk + qsrc=learned + tctrl)",
           "pk1_seed": PK1_SEEDS[cell], "pk1_sha1": hashlib.sha1(pk.numpy().tobytes()).hexdigest(),
           "pv_seed": (PV_FOR_SEEDS[cell] if arm == "FOREIGN" else PV_OWN_SEEDS[cell]),
           "alpha": (0.0 if arm == "NONE" else 1.0),
           "n_train": len(TRAIN), "train_off": [TRAIN[0], TRAIN[-1]],
           "n_eval": len(EVAL), "eval_off": [EVAL[0], EVAL[-1]],
           "steps": T59.STEPS, "lr": T59.LR, "n_rep": T59.N_REP,
           "pv_sha": hashlib.sha256(pv.numpy().tobytes()).hexdigest(), "pv_fro": float(pv.norm()),
           "mean_L": a["mean_L"], "sd_L": a["sd_L"], "L_series": a["L_series"],
           "q_norm": a["q_norm"], "y_norm": a["y_norm"], "m_abs": a["m_abs"], "Wmax": a["Wmax"],
           "cos_q_k": a["cos_q_k"], "align_q_k": a["align_q_k"],
           "gradnorms": meta["gradnorms"], "dWq1_from_init": meta["dWq1_from_init"],
           "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"PROBE59Dv2_{cell}_{arm}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{cell}/{arm}] → {OUT}/PROBE59Dv2_{cell}_{arm}.json", flush=True)


def stagea():
    g = T59.selftest()
    bad = lambda s: any(a <= s <= b for a, b in HISTORY)
    g["seed_disjoint"] = {"train": not bad(TRAIN[0]) and not bad(TRAIN[-1]),
                          "eval": not bad(EVAL[0]) and not bad(EVAL[-1]),
                          "pk1": all(not bad(v) for v in PK1_SEEDS.values()),
                          "pv_own": all(not bad(v) for v in PV_OWN_SEEDS.values()),
                          "pv_for": all(not bad(v) for v in PV_FOR_SEEDS.values())}
    g["segments_distinct"] = len({*PK1_SEEDS.values(), *PV_OWN_SEEDS.values(), *PV_FOR_SEEDS.values()}) == 9
    # ★ DEF-59-2/3 的直接守卫
    pv_o = [T59._pv1_for(PV_OWN_SEEDS[c]) for c in CELLS]
    pv_f = [T59._pv1_for(PV_FOR_SEEDS[c]) for c in CELLS]
    g["A59Dv2-1_pv_disjoint"] = all(abs(float((a / a.norm() * (b / b.norm())).sum())) < 0.999
                                    for a in pv_o for b in pv_o if a is not b)
    g["A59Dv2-2_foreign_indep"] = all(hashlib.sha256(a.numpy().tobytes()).hexdigest() !=
                                      hashlib.sha256(b.numpy().tobytes()).hexdigest()
                                      for a in pv_o + pv_f for b in pv_o + pv_f if a is not b)
    g["A59Dv2-3_no_cycle"] = {c: (PV_OWN_SEEDS[c], PV_FOR_SEEDS[c]) for c in CELLS}
    g["A59Dv2-4_pk1_scale"] = {c: float(pk1_for(c).norm()) for c in CELLS}
    g["A59Dv2-4_pass"] = all(0.5 < v < 5.0 for v in g["A59Dv2-4_pk1_scale"].values())
    g["A59Dv2-5_train_eval"] = not set(TRAIN) & set(EVAL)
    g["all_pass"] = g["all_pass"] and all(g["seed_disjoint"].values()) and g["segments_distinct"] \
        and g["A59Dv2-1_pv_disjoint"] and g["A59Dv2-2_foreign_indep"] and g["A59Dv2-4_pass"] \
        and g["A59Dv2-5_train_eval"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PROBE59Dv2_stageA.json").write_text(json.dumps(g, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print(json.dumps({k: v for k, v in g.items() if k != "md5"}, ensure_ascii=False, indent=1))
    print("★ 59Dv2 Stage A 全部通过" if g["all_pass"] else "★★ 59Dv2 Stage A 失败")
    return 0 if g["all_pass"] else 1


def summarize():
    none_p = OUT / "PROBE59Dv2_C1_NONE.json"
    if not none_p.exists():
        print("缺 NONE（地板）"); return []
    none = json.loads(none_p.read_text(encoding="utf-8"))
    se = lambda v: statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")
    rows = []
    for cell in CELLS:
        o_p, f_p = OUT / f"PROBE59Dv2_{cell}_OWN.json", OUT / f"PROBE59Dv2_{cell}_FOREIGN.json"
        if not (o_p.exists() and f_p.exists()):
            print(f"[{cell}] 缺臂"); continue
        o, f = (json.loads(p.read_text(encoding="utf-8")) for p in (o_p, f_p))
        Lo, Lf, Ln = o["mean_L"], f["mean_L"], none["mean_L"]
        d_fo = [x - y for x, y in zip(f["L_series"], o["L_series"])]
        d_on = [x - y for x, y in zip(o["L_series"], none["L_series"])]
        R = Ln - Lo
        rows.append({"cell": cell, "L_OWN": Lo, "L_FOREIGN": Lf, "L_NONE": Ln,
                     "R": R, "Delta": Lf - Lo, "D_own_none": statistics.fmean(d_on),
                     "3se_D_own_none": 3 * se(d_on), "3se_D_for_own": 3 * se(d_fo),
                     "Delta_over_R": (Lf - Lo) / R if R else float("nan"),
                     "3se_over_020R": (3 * se(d_fo)) / (0.20 * R) if R else float("nan"),
                     "L_OWN_rc": o.get("OWN_rc", {}).get("mean_L"),
                     "rc_over_own": (o["OWN_rc"]["mean_L"] / Lo) if o.get("OWN_rc") else None})
    if rows:
        print(f"{'cell':4s} {'L_OWN':>9s} {'L_FOR':>9s} {'L_NONE':>9s} {'R':>8s} {'Δ':>8s} "
              f"{'Δ/R':>7s} {'3se/0.2R':>8s} {'3seΔ':>8s} {'3se(D_on)':>10s} {'L_OWN_rc':>9s}")
        for r in rows:
            print(f"{r['cell']:4s} {r['L_OWN']:9.5f} {r['L_FOREIGN']:9.5f} {r['L_NONE']:9.5f} "
                  f"{r['R']:8.5f} {r['Delta']:+8.5f} {r['Delta_over_R']:+7.4f} "
                  f"{r['3se_over_020R']:8.3f} {r['3se_D_for_own']:8.5f} "
                  f"{r['3se_D_own_none']:10.5f} "
                  f"{(f'{r[chr(76)+chr(95)+chr(79)+chr(87)+chr(78)+chr(95)+chr(114)+chr(99)]:.4f}' if r['L_OWN_rc'] else '—'):>9s}")
        (OUT / "PROBE59Dv2_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stagea", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--cell")
    ap.add_argument("--arm")
    a = ap.parse_args()
    if a.stagea:
        sys.exit(stagea())
    elif a.summarize:
        summarize()
    elif a.cell and a.arm:
        run(a.cell, a.arm)
    else:
        ap.print_help()
