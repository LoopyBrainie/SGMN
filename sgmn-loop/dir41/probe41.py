#!/opt/pytorch/bin/python
r"""DIR-41 · **开卡前探针**（**不进判据**）：共享 `Pk1` vs 逐实例 `Pk1`，在**匹配头**协议下各跑一格。

派生自 `dir40/R1_task40.py`（**逐字**：trace / 初值 / 训练协议 / 评估口径），**只改**：
  ① 种子段（探针专用，全新）；
  ② `shared_pk1` 的**参数化**（DIR-40 写死一个种子 ⇒ 本探针与正式轮各需多张图）；
  ③ 「格」的编排（`S1`/`S2` = 共享图；`PI` = 装置自己的逐实例 `Pk1`）。

★ 目的（冻结在 `PROBE41.md`）：
  (i)  **关键未知**：逐实例格在**匹配头**协议下**到底学不学**
      （DIR-39 用的是「一份 oracle 头」⇒ 失败**不可归因** ⇒ D-63-CORR 的悬置项）；
  (ii) 共享格在**缩小规模**（`n_train=96`）下**效应是否仍在**（决定正式轮的规模冻结）；
  (iii) 两张不同共享图之间的差异量级（探针只给下限，分布由正式轮的 5 抽取给出）。

★ 纪律 97：探针**不得**产生判据裁决；读数只是**探针档**（纪律 91/92）。
用法：smoke | run --cell {S1,S2,PI} --arm {oracle,rand,learned} | merge
"""
import json
import math
import os
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir37")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir38")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir39")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir40")
import R1_task40 as T40            # ★ DIR-40 的测量层（逐字复用）

OUT = "/home/ec2-user/sgmn-loop/dir41/out"
SEED0 = T40.SEED0
N_REP = T40.N_REP
D_H, D_K = T40.D_H, T40.D_K
KEY_SCALE = T40.R.KEY_SCALE

# ══════════ 探针种子段（**全新**；与正式轮、与全部历史段不相交）
P_TRAIN = [SEED0 + 20500 + i for i in range(96)]      # 训练段 20500..20595
P_EVAL = [SEED0 + 20600 + i for i in range(400)]       # 评估段 20600..20999
P_SMOKE_TRAIN = [SEED0 + 20400 + i for i in range(4)]
P_SMOKE_EVAL = [SEED0 + 20410 + i for i in range(2)]

# ══════════ 探针的共享图种子（与 DIR-40 的 20261112、与正式轮的 5 张图都不同）
PROBE_PK1_SEEDS = {"S1": 20261001, "S2": 20261988}
CELLS = ("S1", "S2", "PI")


def shared_pk1_for(seed):
    r"""**装置 revision** 的参数化版本（DIR-40 `shared_pk1()` 的逐字推广）。

    ★ **与装置源码逐字核对过**（Advisor 的硬要求）：
      装置 `dir31/R1_grad.py::build`：`s_h = 1/√D_H`；`P["Pk{l}"] = rn(D_H, D_K, scale=s_h)`，
      而 `rn(*shape, scale) = randn(*shape, generator=g) * scale`
      ⇒ 装置逐实例 `Pk1` 的**边缘律** = `randn(D_H,D_K)/√D_H`；
      任务层 `dir33/R1_task33.py::build`：`P["Pk1"] = P["Pk1"] * KEY_SCALE`
      ⇒ 装置的有效 `Pk1` = `randn(D_H,D_K)/√D_H × KEY_SCALE`。
      ⇒ 本函数**逐字复刻该边缘律**（独立生成器、独立抽取位置 ⇒ 与任何单个装置实例的
        `Pk1` **不逐位相同**，但**同分布**）⇒ ⇒ 只改「是否跨实例共享」，不改分布。
    """
    g = torch.Generator().manual_seed(int(seed))
    return (torch.randn(D_H, D_K, generator=g) / math.sqrt(D_H)) * KEY_SCALE


def cell_pk1(cell):
    r"""格 → 共享图；`PI` ⇒ `None`（**装置自己的逐实例 `Pk1`**）。"""
    if cell == "PI":
        return None
    return shared_pk1_for(PROBE_PK1_SEEDS[cell])


def run_cell(cell, arm, steps=T40.STEPS, train_seeds=None, eval_seeds=None):
    spk = cell_pk1(cell)
    qsrc, tctrl = {"oracle": ("oracle", False), "rand": ("rand", False),
                   "learned": ("learned", True)}[arm]
    t0 = time.time()
    phi, meta = T40.train_arm40(qsrc, train_seeds, shared_pk=spk, train_ctrl=tctrl, steps=steps)
    arms = {arm: T40.eval_arm40(qsrc, phi, eval_seeds, shared_pk=spk)}
    if arm == "learned":
        arms["learned_rc"] = T40.eval_arm40(qsrc, phi, eval_seeds, shared_pk=spk, rand_content=True)
    for k, a in arms.items():
        print(f"[{cell}/{arm}] {k:11s} L={a['mean_L']:.6f} (sd {a['sd_L']:.4f}) "
              f"cos={a['cos_q_k']:.4f} |q|={a['q_norm']:.4f} m̄={a['m_abs']:.4f} "
              f"Wmax={a['Wmax']:.4f}", flush=True)
    print(f"[{cell}/{arm}] 训练 {time.time()-t0:.0f}s  梯度 {meta['gradnorms']} "
          f"|ΔWq1|={meta['dWq1_from_init']:.6f}", flush=True)
    out = {"cell": cell, "arm": arm, "n_train": len(train_seeds), "n_eval": len(eval_seeds),
           "steps": steps, "n_rep": N_REP, "pk1_seed": PROBE_PK1_SEEDS.get(cell, "device-own"),
           "arms": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_series")}
                    for k, v in arms.items()},
           "L_series": {k: v["L_series"] for k, v in arms.items()},
           "gradnorms": meta["gradnorms"], "dWq1_from_init": meta["dWq1_from_init"],
           "secs": time.time() - t0, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"DIR41_probe_{cell}_{arm}.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def smoke():
    out = {"note": "探针 smoke：只证明可执行，不给判据"}
    for cell in CELLS:
        spk = cell_pk1(cell)
        for arm in ("oracle", "rand", "learned"):
            phi, meta = T40.train_arm40(
                {"oracle": "oracle", "rand": "rand", "learned": "learned"}[arm],
                P_SMOKE_TRAIN, shared_pk=spk,
                train_ctrl=(arm == "learned"), steps=3)
            e = T40.eval_arm40({"oracle": "oracle", "rand": "rand", "learned": "learned"}[arm],
                               phi, P_SMOKE_EVAL, shared_pk=spk)
            print(f"  {cell}/{arm:8s} L={e['mean_L']:.6f} cos={e['cos_q_k']:.4f} "
                  f"Wmax={e['Wmax']:.4f} |ΔWq1|={meta['dWq1_from_init']:.3e}", flush=True)
    print("探针 smoke 完成（**无判据裁决**）")
    return out


def merge():
    """汇总探针读数（**只报数，不裁决**）。"""
    cells = {}
    for cell in CELLS:
        arms, series = {}, {}
        for arm in ("oracle", "rand", "learned"):
            p = os.path.join(OUT, f"DIR41_probe_{cell}_{arm}.json")
            if not os.path.exists(p):
                print(f"★ 缺文件：{p}")
                continue
            d = json.load(open(p, encoding="utf-8"))
            for k, v in d["arms"].items():
                arms[k] = v
                series[k] = d["L_series"][k]
        if "learned" not in arms or "rand" not in arms or "oracle" not in arms:
            continue
        R_head = arms["rand"]["mean_L"] - arms["oracle"]["mean_L"]
        numer = arms["rand"]["mean_L"] - arms["learned"]["mean_L"]
        pg = T40.paired(series["rand"], series["learned"])
        pc = T40.paired(series["learned_rc"], series["learned"]) if "learned_rc" in series else None
        cells[cell] = {"arms": {k: {"mean_L": v["mean_L"], "sd_L": v["sd_L"],
                                    "cos_q_k": v["cos_q_k"], "q_norm": v["q_norm"],
                                    "m_abs": v["m_abs"], "Wmax": v["Wmax"]}
                                for k, v in arms.items()},
                       "R": R_head, "delta": numer, "closure_CL": (numer / R_head if R_head > 0 else None),
                       "floor_0.20R": 0.20 * R_head, "paired_rand_learned": pg,
                       "paired_learnedrc_learned": pc}
    print("\n══ DIR-41 探针汇总（**探针档 · 无判据裁决**） ══")
    for cell, c in cells.items():
        print(f"\n── 格 {cell} ──")
        for k, v in c["arms"].items():
            print(f"   L[{k:11s}] = {v['mean_L']:12.6f}  cos={v['cos_q_k']:+.4f}  "
                  f"|q|={v['q_norm']:.4f}  m̄={v['m_abs']:.4f}")
        pg = c["paired_rand_learned"]
        print(f"   R = {c['R']:.6f}   Δ = L(rand) − L(learned) = {c['delta']:.6f}   "
              f"闭合率 CL = Δ/R = {c['closure_CL']:.4f}   阈 0.20·R = {c['floor_0.20R']:.6f}")
        print(f"   配对 rand−learned: mean={pg['mean']:.6f} 3se={pg['3se']:.6f} "
              f"⇒ {abs(pg['mean'])/pg['3se'] if pg['3se']>0 else float('nan'):.2f}×3se")
        if c["paired_learnedrc_learned"]:
            pc = c["paired_learnedrc_learned"]
            print(f"   内容性 learned_rc−learned: mean={pc['mean']:.6f} 3se={pc['3se']:.6f} "
                  f"⇒ {pc['mean']/pc['3se'] if pc['3se']>0 else float('nan'):.2f}×3se")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "DIR41_probe_summary.json"), "w") as f:
        json.dump(cells, f, ensure_ascii=False, indent=1)
    return cells


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if what == "smoke":
        json.dump(smoke(), open(os.path.join(OUT, "DIR41_probe_smoke.json"), "w"),
                  ensure_ascii=False, indent=1)
        sys.exit(0)
    if what == "run":
        rest = [a for a in sys.argv[2:] if not a.startswith("--")]
        cell = rest[0]
        arm = rest[1]
        run_cell(cell, arm, train_seeds=P_TRAIN, eval_seeds=P_EVAL)
        sys.exit(0)
    if what == "merge":
        merge()
        sys.exit(0)
    print("用法：smoke | run --cell {S1,S2,PI} --arm {oracle,rand,learned} | merge")
    sys.exit(2)
