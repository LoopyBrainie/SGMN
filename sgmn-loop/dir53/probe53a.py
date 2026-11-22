#!/opt/pytorch/bin/python
r"""DIR-53 · **开卡前探针**：$P_k$ 的「离线学习」是否比「固定随机」更好？

★ 本脚本**不产生判据**（见 `dir53/PROBE53.md` §0）：它只量效应与方差，供正式卡**冻结阈值**。
★ 装置**逐字复用**：`dir31/R1_grad.py`、`dir33/R1_task33.py`、`dir37/R1_task37.py`、
  `dir40/R1_task40.py::trace40 / init_params / eval_arm40`。**本文件不改任何装置/任务层文件**。

用法：
  /opt/pytorch/bin/python dir53/probe53a.py            # 单臂（--pk 与 --arm 指定）
  /opt/pytorch/bin/python dir53/probe53a.py --merge     # 汇总（只打印，不写裁决）
"""
import argparse
import json
import math
import os
import pathlib
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
for _d in ("dir31", "dir33", "dir37", "dir38", "dir39", "dir40"):
    sys.path.insert(0, f"/home/ec2-user/sgmn-loop/{_d}")
import R1_task33 as R      # 任务层（逐字未改）
import R1_task37 as T37    # 测量层（_mk）
import R1_task40 as T40    # trace40 / init_params / eval_arm40

OUT = "/home/ec2-user/sgmn-loop/dir53/out"
SEED0 = 20261111
D_H, D_K = R.D_H, R.D_K
N_REP = 16
N_TRAIN, N_EVAL = 32, 100
STEPS, LR = 250, 0.03
PK_SEEDS = (20262201, 20262202, 20262203)   # 3 张独立抽取（探针档；非正式种子段）
TRAIN_OFF = list(range(21000, 21000 + N_TRAIN))
EVAL_OFF = list(range(21500, 21500 + N_EVAL))
TRAIN = [SEED0 + o for o in TRAIN_OFF]
EVAL = [SEED0 + o for o in EVAL_OFF]


def fixed_pk(seed):
    r"""与装置**逐字同式**（`dir31/R1_grad.build` + `dir33/R1_task33.build` 的 `KEY_SCALE`）。"""
    g = torch.Generator().manual_seed(int(seed))
    s_h = 1.0 / math.sqrt(D_H)
    return torch.randn(D_H, D_K, generator=g) * s_h * R.KEY_SCALE


def _row_norms(P):
    return P.norm(dim=1, keepdim=True).clamp_min(1e-300)


def train_probe(pk0, arm, steps=STEPS, lr=LR):
    r"""`FIX`：只训头/控制器（$P_k$ 冻结）；`OFF`：$P_k$ 一起训 + **行范数投影**。"""
    init = T40.init_params()
    phi = {"g1": init["g1"], "Wo1": init["Wo1"], "bo1": init["bo1"],
           "Wq1": init["Wq1"], "bq1": init["bq1"]}
    pk = pk0.detach().clone()
    params = [phi[n] for n in ("g1", "Wo1", "bo1", "Wq1", "bq1")]
    if arm == "OFF":
        pk.requires_grad_(True)
        params = params + [pk]
    opt = torch.optim.Adam(params, lr=lr)
    target = _row_norms(pk0).detach().clone()      # ★ 每行的目标范数（= 初值行范数）
    hist, gn = [], {}
    for _ in range(steps):
        opt.zero_grad()
        for sd in TRAIN:
            P = T37._mk(sd, phi, train=True)
            L, _ = T40.trace40(P, N_REP, "learned", sd, phi, shared_pk1=pk)
            (L / len(TRAIN)).backward()
        if arm == "OFF" and pk.grad is not None:
            gn["pk"] = max(gn.get("pk", 0.0), float(pk.grad.norm()))
        opt.step()
        if arm == "OFF":
            with torch.no_grad():                   # ★ 行范数投影：只改方向
                pk.mul_(target / _row_norms(pk))
        if (len(hist) % 10 == 0) or (_ == steps - 1):   # 每 10 步记一次（省一半算力）
            with torch.no_grad():
                tot = 0.0
                for sd in TRAIN:
                    P = T37._mk(sd, phi, train=True)
                    L, _ = T40.trace40(P, N_REP, "learned", sd, phi, shared_pk1=pk)
                    tot += float(L)
                hist.append(tot / len(TRAIN))
    dPk = float((pk.detach() - pk0).norm() / pk0.norm().clamp_min(1e-300))
    return phi, pk.detach(), {"L_series": hist, "rel_dPk": dPk, "grad_pk": gn.get("pk", 0.0)}


def run(pk_seed, arm):
    t0 = time.time()
    pk0 = fixed_pk(pk_seed)
    phi, pk, meta = train_probe(pk0, arm)
    ev = T40.eval_arm40("learned", phi, EVAL, shared_pk=pk)
    ev_rc = T40.eval_arm40("learned", phi, EVAL, shared_pk=pk, rand_content=True)
    tr = T40.eval_arm40("learned", phi, TRAIN, shared_pk=pk)
    out = {"pk_seed": pk_seed, "arm": arm, "n_train": N_TRAIN, "n_eval": N_EVAL,
           "steps": STEPS, "lr": LR,
           "row_norm_projection": (arm == "OFF"),
           "eval": {k: v for k, v in ev.items() if k != "L_series"},
           "eval_rand_content": {k: v for k, v in ev_rc.items() if k != "L_series"},
           "train_eval": {k: v for k, v in tr.items() if k != "L_series"},
           "L_series_eval": ev["L_series"],
           "first20_mean": statistics.fmean(meta["L_series"][:20]),
           "last20_mean": statistics.fmean(meta["L_series"][-20:]),
           "rel_dPk": meta["rel_dPk"], "grad_pk_max": meta["grad_pk"],
           "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"P53_pk{pk_seed}_{arm}.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[pk{pk_seed}/{arm}] eval L={ev['mean_L']:.6f} (sd {ev['sd_L']:.4f}) "
          f"rc L={ev_rc['mean_L']:.6f} train L={tr['mean_L']:.6f} "
          f"L {out['first20_mean']:.4f}→{out['last20_mean']:.4f} "
          f"rel_dPk={meta['rel_dPk']:.4f} {time.time()-t0:.0f}s", flush=True)
    return out


def merge():
    rows = []
    for p in sorted(pathlib.Path(OUT).glob("P53_pk*_*.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        print("没有读数"); return
    by = {}
    for r in rows:
        by.setdefault(r["pk_seed"], {})[r["arm"]] = r
    print(f"{'pk_seed':>9} {'L_FIX':>10} {'L_OFF':>10} {'dL':>10} {'L_OFFrc':>10} {'dL_rc':>10} {'rel_dPk':>8}")
    dl = []
    for s, d in sorted(by.items()):
        if "FIX" in d and "OFF" in d:
            lf, lo = d["FIX"]["eval"]["mean_L"], d["OFF"]["eval"]["mean_L"]
            rc = d["OFF"]["eval_rand_content"]["mean_L"]
            dl.append(lf - lo)
            print(f"{s:>9} {lf:10.6f} {lo:10.6f} {lf-lo:10.6f} {rc:10.6f} {rc-lo:10.6f} "
                  f"{d['OFF']['rel_dPk']:8.4f}")
    if len(dl) > 1:
        sd = statistics.stdev(dl)
        print(f"\nΔL: mean={statistics.fmean(dl):+.6f}  sd={sd:.6f}  n={len(dl)}")
        print(f"   ⇒ 正式轮 n=400 的 3se(ΔL) 预测 ≈ {3*sd/math.sqrt(400):.6f}"
              f"（若正式档 n_eval=400；探针 sd 只作量级输入）")
        print(f"   ⇒ 正式轮 n=200 的 3se(ΔL) 预测 ≈ {3*sd/math.sqrt(200):.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pk", type=int, default=None)
    ap.add_argument("--arm", choices=("FIX", "OFF"), default=None)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    else:
        assert a.pk is not None and a.arm is not None, "需要 --pk 与 --arm"
        run(a.pk, a.arm)