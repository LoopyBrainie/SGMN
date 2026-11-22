#!/opt/pytorch/bin/python
r"""DIR-59 · **task 级探针 59D · rev 3** —— 2×2 交叉：**写步的 $P_v$** × **任务目标的 $P_v$**

★ 为什么 rev 2 不够（`DEF-59-4`）：rev 2 的 `OWN`/`FOREIGN` 把 $P_v$ **对称**地同时用于写步与目标
  ⇒ 两者只是"同分布的两次抽样"，测的是**draw-to-draw 方差**，**不是**"记忆是否绑定到写它的那个值映射"。
  真正可判的问题需要**错配**：
    · `AB` = 写步用 $P_v^A$、任务目标用 $P_v^B$ ⇒ **来源错配**（记忆按 A 写，任务要 B）
    · `BA` = 写步用 $P_v^B$、任务目标用 $P_v^A$ ⇒ 反向错配
  2×2（每格）：`AA`（= rev2 `OWN`，已有）· `AB`（新）· `BA`（新）· `BB`（= rev2 `FOREIGN`，已有）
  ⇒ **主判据对象 = "写源"主效应**：$AA$ vs $BA$（目标都是 A）与 $BB$ vs $AB$（目标都是 B）；
  ⇒ **阳性对照 = "目标"主效应**：$AA$ vs $AB$（写源都是 A）—— 若目标换掉而不变差，则记忆里的值内容根本没被任务读到，
     等价型断言无功效（前提不成立）。
★ 只跑缺的 6 次（3 格 × {AB, BA}）；`AA`/`BB` 直接读 `PROBE59Dv2_{cell}_{OWN,FOREIGN}.json`。
"""
import argparse
import hashlib
import json
import pathlib
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir59")
import R1_task59 as T59        # noqa: E402
import probe59d_v2 as P2       # noqa: E402  （种子段/协议/Stage A 全部复用）

OUT = pathlib.Path("/home/ec2-user/sgmn-loop/dir59/out")
CELLS, TRAIN, EVAL = P2.CELLS, P2.TRAIN, P2.EVAL
ARMS = ("AB", "BA")


def run(cell, arm):
    t0 = time.time()
    pk = P2.pk1_for(cell)
    pv_a, pv_b = T59._pv1_for(P2.PV_OWN_SEEDS[cell]), T59._pv1_for(P2.PV_FOR_SEEDS[cell])
    pv_w, pv_t = (pv_a, pv_b) if arm == "AB" else (pv_b, pv_a)
    print(f"[{cell}/{arm}] 训头（learned+ctrl，{T59.STEPS} 步，shared_pk={P2.PK1_SEEDS[cell]}，"
          f"写源={'A' if arm=='AB' else 'B'}，目标={'B' if arm=='AB' else 'A'}）", flush=True)
    phi, meta = T59.train_arm59("learned", TRAIN, shared_pk=pk, shared_pv=pv_w, alpha=None,
                                train_ctrl=True, target_pv=pv_t)
    a = T59.eval_arm59("learned", phi, EVAL, shared_pk=pk, shared_pv=pv_w, target_pv=pv_t)
    print(f"[{cell}/{arm}] L={a['mean_L']:.6f} (sd {a['sd_L']:.4f}) |q|={a['q_norm']:.4f} "
          f"|y|={a['y_norm']:.4f} Wmax={a['Wmax']:.4f}  {time.time()-t0:.0f}s", flush=True)
    rec = {"cell": cell, "arm": arm, "protocol": "D-76 + target_pv 解耦",
           "pk1_seed": P2.PK1_SEEDS[cell], "pk1_sha1": hashlib.sha1(pk.numpy().tobytes()).hexdigest(),
           "pv_write_seed": (P2.PV_OWN_SEEDS[cell] if arm == "AB" else P2.PV_FOR_SEEDS[cell]),
           "pv_target_seed": (P2.PV_FOR_SEEDS[cell] if arm == "AB" else P2.PV_OWN_SEEDS[cell]),
           "alpha": 1.0, "n_train": len(TRAIN), "n_eval": len(EVAL),
           "train_off": [TRAIN[0], TRAIN[-1]], "eval_off": [EVAL[0], EVAL[-1]],
           "steps": T59.STEPS, "lr": T59.LR, "n_rep": T59.N_REP,
           "pv_write_sha": hashlib.sha256(pv_w.numpy().tobytes()).hexdigest(),
           "pv_target_sha": hashlib.sha256(pv_t.numpy().tobytes()).hexdigest(),
           "mean_L": a["mean_L"], "sd_L": a["sd_L"], "L_series": a["L_series"],
           "q_norm": a["q_norm"], "y_norm": a["y_norm"], "m_abs": a["m_abs"], "Wmax": a["Wmax"],
           "cos_q_k": a["cos_q_k"], "gradnorms": meta["gradnorms"],
           "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"PROBE59Dv3_{cell}_{arm}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{cell}/{arm}] → PROBE59Dv3_{cell}_{arm}.json", flush=True)


def summarize():
    se = lambda v: statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")
    rows = []
    for cell in CELLS:
        d = {}
        for arm, fn in (("AA", f"PROBE59Dv2_{cell}_OWN.json"), ("BB", f"PROBE59Dv2_{cell}_FOREIGN.json"),
                        ("AB", f"PROBE59Dv3_{cell}_AB.json"), ("BA", f"PROBE59Dv3_{cell}_BA.json")):
            p = OUT / fn
            if p.exists():
                d[arm] = json.loads(p.read_text(encoding="utf-8"))
        if len(d) != 4:
            print(f"[{cell}] 缺臂 {sorted(d)}"); continue
        L = {k: v["mean_L"] for k, v in d.items()}
        def pd(x, y):
            dif = [p - q for p, q in zip(d[x]["L_series"], d[y]["L_series"])]
            return statistics.fmean(dif), se(dif), 3 * se(dif)
        src_at_A = pd("BA", "AA")   # 目标 A：写源 B − 写源 A  （正 = 错源更差）
        src_at_B = pd("AB", "BB")   # 目标 B：写源 A − 写源 B
        tgt_at_A = pd("AB", "AA")   # 写源 A：目标 B − 目标 A
        tgt_at_B = pd("BB", "BA")   # 写源 B：目标 A − 目标 B
        rows.append({"cell": cell, **{f"L_{k}": v for k, v in L.items()},
                     "src_at_A": src_at_A[0], "3se_src_at_A": src_at_A[2],
                     "src_at_B": src_at_B[0], "3se_src_at_B": src_at_B[2],
                     "src_mean": (src_at_A[0] + src_at_B[0]) / 2,
                     "tgt_at_A": tgt_at_A[0], "3se_tgt_at_A": tgt_at_A[2],
                     "tgt_at_B": tgt_at_B[0], "3se_tgt_at_B": tgt_at_B[2],
                     "tgt_mean": (tgt_at_A[0] + tgt_at_B[0]) / 2,
                     "interaction": (src_at_A[0] - src_at_B[0]) / 2})
    if rows:
        print(f"{'cell':4s} {'L_AA':>8s} {'L_AB':>8s} {'L_BA':>8s} {'L_BB':>8s} | "
              f"{'src@A':>8s} {'±3se':>7s} {'src@B':>8s} {'±3se':>7s} | "
              f"{'tgt@A':>8s} {'±3se':>7s} {'tgt@B':>8s} {'±3se':>7s}")
        for r in rows:
            print(f"{r['cell']:4s} {r['L_AA']:8.4f} {r['L_AB']:8.4f} {r['L_BA']:8.4f} {r['L_BB']:8.4f} | "
                  f"{r['src_at_A']:+8.4f} {r['3se_src_at_A']:7.4f} {r['src_at_B']:+8.4f} "
                  f"{r['3se_src_at_B']:7.4f} | {r['tgt_at_A']:+8.4f} {r['3se_tgt_at_A']:7.4f} "
                  f"{r['tgt_at_B']:+8.4f} {r['3se_tgt_at_B']:7.4f}")
        (OUT / "PROBE59Dv3_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--cell"); ap.add_argument("--arm")
    a = ap.parse_args()
    if a.summarize:
        summarize()
    elif a.cell and a.arm:
        run(a.cell, a.arm)
    else:
        ap.print_help()
