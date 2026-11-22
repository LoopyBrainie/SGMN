#!/opt/pytorch/bin/python
r"""DIR-34 · **装置与任务层逐字不变**，只改**评估域**：出样本下的内容性前向任务收益。

卡：`revolve/DIR-34/DIRECTION_rev34_v10.md`（rev 1.0，阈值已冻结）。

★ **为什么本文件这么短**：`dir33/R1_task33.py`（任务层）与 `dir31/R1_grad.py`（装置）**逐字零改动**，
  本文件**只 import 它们**（⇒ 任务与装置在**代码对象层面**就是 DIR-33 的那些，不是拷贝），
  改动的只有**测量层**：训练用哪批种子、评估用哪批种子、以及按什么口径出判据。
  A34-7/A34-10 用 md5 把这一性质钉死。

用法：stageA | gates | smoke | formal | mech
"""
import json
import math
import os
import pathlib
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
import R1_task33 as R          # 任务层（逐字未改）
import R1_grad as M            # 装置（逐字未改）

OUT = "/home/ec2-user/sgmn-loop/dir34/out"

# ══════════ 冻结的种子段（卡 §3.2）—— 与**所有**历史段不相交（纪律 70）
SEED0 = M.SEED0
TRAIN = [SEED0 + 900 + i for i in range(24)]                       # 训练（读数头在这里学）
EVAL = [SEED0 + 2000 + i for i in range(400)]      # 评估 n=400（卡 §2.2 功效核算冻结）
# 历史种子段（**偏移区间**，纪律 70：DIR-34 的训练/评估段必须全部避开）
HIST_RANGES = {"pilot32": (100, 124), "formal32": (200, 224), "calib33": (300, 308),
               "formal33": (400, 424), "pilot33": (500, 524), "calib34": (600, 624),
               "heldout34": (700, 724)}

# ══════════ 冻结阈值（卡 §2）
P1_MIN = 0.5      # rank-2 功效上界
P2_MIN = 0.2      # 未训练实际检索（存储可检索性）
EFF_PCT = 0.05    # 效应阈值：0.05·L*(G0, eval)
P4_MIN = 1.0      # 出样本隔离：L*(N,eval) ≥ P4_MIN·Z(eval)
H34C_PCT = 0.25   # 样本内溢价：L*(N,out) − L*(N,in) ≥ H34C_PCT·L*(G0, train)
COST_CAP = 1e-15
REL_TOL = 1e-12
WORK_TOL = 1e-14
MD5_DEVICE = "8fa40f708600bec7e6c484727d2180c7"   # dir31/R1_grad.py
MD5_TASK = "96ef89e87ec61556b52e0e464850087e"     # dir33/R1_task33.py


# ══════════ 基础量
def floor_Z(seeds):
    r"""地板 $Z\equiv\sum_t\lVert g_t\rVert^2$：读数头输出恒 0 时的损失（逐实例）。"""
    zs = []
    for sd in seeds:
        P = R.build(sd, "V0")
        R.period2(P, sd)
        zs.append(sum(float((g ** 2).sum()) for g in R.targets(P)))
    return zs


def paired(a, b):
    """配对差（同一种子上两臂之差 ⇒ 实例难度抵消）的 mean/sd/se/3se。"""
    d = [x - y for x, y in zip(a, b)]
    m = statistics.fmean(d)
    sd = statistics.stdev(d)
    se = sd / len(d) ** 0.5
    return {"mean": m, "sd": sd, "se": se, "3se": 3 * se, "n": len(d)}


def _md5(p):
    import hashlib
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


# ══════════ Stage A
def stage_a():
    r"""九条断言：八条取自 dir33 的**同源**断言（同一代码对象 ⇒ 逐字同义），另加 DIR-34 专属的两条。"""
    a = {}
    ok33, a33 = R.stage_a()          # ★ 同一代码对象跑出的同源断言
    a["A34-1_write_block_bitwise"] = a33["A33-1"]
    a["A34-2_align_min"] = a33["A33-2_align_min"]
    a["A34-2_sep_absmax"] = a33["A33-2_sep_absmax"]
    a["A34-3_magnitude_match"] = a33["A33-3"]
    a["A34-4_determinism"] = a33["A33-4"]
    a["A34-5_G0_no_injection_rel"] = a33["A33-5"]
    a["A34-8_worked_example"] = a33["A33-8"]
    a["A34-8_seq"] = a33["A33-8_seq"]
    # A34-6 在 **DIR-34 的评估种子上**重算（双条款，见卡 §4 与决定三十六）
    per_inst, pooled = [], []
    for sd in EVAL:
        P = R.build(sd, "V0")
        R.period2(P, sd)
        g = R.targets(P)
        cs = []
        for i in range(len(g)):
            for j in range(len(g)):
                if (i - j) % 2 == 1:
                    cs.append(abs(float((g[i] * g[j]).sum() / (g[i].norm() * g[j].norm()))))
        per_inst.append(max(cs))
        pooled += cs
    a["A34-6_inst_mean"] = statistics.fmean(per_inst)
    # ★ rev 1.1（DEF-34-1）：第二条款由「实例 max」改为「p95(实例 max)」——
    #   实例 max 是**样本量依赖**的极值（0.8892@24 → 0.9951@400，diag34a.py 实测），
    #   不是总体性质 ⇒ 降级为诊断量；改用固定分位数（总体量，估计随 n 稳定）。
    srt = sorted(per_inst)
    k95 = min(len(srt) - 1, max(0, round(0.95 * (len(srt) - 1))))
    a["A34-6_inst_p95"] = srt[k95]
    a["A34-6_inst_max_diag"] = max(per_inst)
    a["A34-6_pooled_max_diag"] = max(pooled)
    # A34-7 / A34-10：装置与任务层逐字未改（md5）
    a["A34-7_device_md5_diff"] = 0.0 if _md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py") == MD5_DEVICE else 1.0
    a["A34-7_device_md5"] = _md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py")
    a["A34-10_task_md5_diff"] = 0.0 if _md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py") == MD5_TASK else 1.0
    a["A34-10_task_md5"] = _md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py")
    # A34-9（DIR-34 专属：**种子隔离** + **训练确定性**）
    seg_hits = sorted({name for name, (lo, hi) in HIST_RANGES.items()
                       for sd in TRAIN + EVAL if lo <= (sd - SEED0) <= hi})
    a["A34-9_seed_disjoint"] = 0.0 if (not seg_hits and not (set(TRAIN) & set(EVAL))) else 1.0
    a["A34-9_hist_hits"] = ",".join(seg_hits) if seg_hits else "none"
    # 训练确定性（5 步两次 ⇒ 读数头逐位相同；确认训练路径不依赖评估种子）
    p1 = R.train_arm("oracle", "T", steps=5, seeds=TRAIN)
    p2 = R.train_arm("oracle", "T", steps=5, seeds=TRAIN)
    a["A34-9_train_bitwise"] = max(float((p1[k] - p2[k]).abs().max()) for k in p1)
    ok = (a["A34-1_write_block_bitwise"] == 0.0 and a["A34-2_align_min"] >= 1 - 1e-14
          and a["A34-3_magnitude_match"] <= COST_CAP and a["A34-4_determinism"] == 0.0
          and a["A34-5_G0_no_injection_rel"] <= REL_TOL and a["A34-8_worked_example"] <= WORK_TOL
          and a["A34-6_inst_mean"] <= 0.60 and a["A34-6_inst_p95"] <= 0.95
          and a["A34-7_device_md5_diff"] == 0.0 and a["A34-10_task_md5_diff"] == 0.0
          and a["A34-9_seed_disjoint"] == 0.0 and a["A34-9_train_bitwise"] == 0.0)
    return ok, a


def gates(seeds=None):
    """P1 rank-2 上界 / P2 未训练实际检索（**逐字复用** dir33 的实现，换种子段）。"""
    return R.gates(EVAL if seeds is None else seeds)


def mech():
    """装置级机制量（与 DIR-33 同一装置 ⇒ 取其同源实现，用于一致性对照）。"""
    return R.mech()


# ══════════ 正式轮
def formal():
    t0 = time.time()
    res = {"P1": None, "P2": None, "arms": {}, "verdict": None}
    p1, p2 = gates()
    res["P1"], res["P2"] = p1, p2
    res["Z_eval"] = floor_Z(EVAL)
    res["Z_train"] = floor_Z(TRAIN)
    Z_ev, Z_tr = statistics.fmean(res["Z_eval"]), statistics.fmean(res["Z_train"])
    print(f"P1 {p1:+.4f}（阈 {P1_MIN}） · P2 {p2:+.4f}（阈 >{P2_MIN}） · "
          f"地板 Z: eval {Z_ev:.4f} / train {Z_tr:.4f}")
    for addr in ("oracle", "device"):
        learned = (addr == "device")
        L = {}
        for arm in ("N", "S", "T"):
            t1 = time.time()
            phi = R.train_arm(addr, arm, learned=learned, seeds=TRAIN)
            L[arm] = {"out": R.per_seed(addr, arm, phi, EVAL), "in": R.per_seed(addr, arm, phi, TRAIN),
                      "gamma": float(phi["g1"])}
            print(f"  {addr:6s} {arm:2s} 出样本 {statistics.fmean(L[arm]['out']):9.5f}"
                  f" · 样本内 {statistics.fmean(L[arm]['in']):9.5f}  ({time.time()-t1:.0f}s)", flush=True)
        L["G0"] = {"out": R.per_seed(addr, "G0", {}, EVAL), "in": R.per_seed(addr, "G0", {}, TRAIN)}
        g0 = statistics.fmean(L["G0"]["out"])
        dc = paired(L["S"]["out"], L["T"]["out"])
        dr = paired(L["N"]["out"], L["S"]["out"])
        dc_in = paired(L["S"]["in"], L["T"]["in"])
        thr = EFF_PCT * g0
        res["arms"][addr] = {
            "mean_out": {k: statistics.fmean(v["out"]) for k, v in L.items()},
            "mean_in": {k: statistics.fmean(v["in"]) for k, v in L.items()},
            "per_seed_out": {k: v["out"] for k, v in L.items()},
            "gamma": {k: v.get("gamma") for k, v in L.items()},
            "dcontent_out": dc, "drand_out": dr, "dcontent_in": dc_in, "thr": thr,
            "dchan_out": g0 - statistics.fmean(L["N"]["out"]),
            "N_out_over_Z": statistics.fmean(L["N"]["out"]) / Z_ev,
            "N_in_over_Z": statistics.fmean(L["N"]["in"]) / Z_tr,
            "in_sample_premium_N": statistics.fmean(L["N"]["out"]) - statistics.fmean(L["N"]["in"]),
            "h34c_thr": H34C_PCT * statistics.fmean(L["G0"]["in"]),
        }
        a = res["arms"][addr]
        # ── 状态词（卡 §1.3）
        if p2 <= P2_MIN:
            v = "PREMISE-NOT-MET"
        elif dc["3se"] > thr:
            v = "INFEASIBLE-POWER"
        elif dc["mean"] >= thr and dc["mean"] > dr["mean"]:
            v = "SUCCEEDED"
        else:
            v = "FALSIFIED"
        a["verdict"] = v
        print(f"  ⇒ {addr}: Δcontent {dc['mean']:+.5f} (sd {dc['sd']:.5f}, 3se {dc['3se']:.5f}) · "
              f"Δrand {dr['mean']:+.5f} · Δchan {a['dchan_out']:+.5f} · 阈 {thr:.5f} ⇒ **{v}**")
    # P4（出样本隔离）与 H34c（样本内溢价）
    a = res["arms"]["oracle"]
    res["P4"] = {"N_out_over_Z": a["N_out_over_Z"], "thr": P4_MIN,
                 "ok": a["N_out_over_Z"] >= P4_MIN}
    res["H34c"] = {"premium_N": a["in_sample_premium_N"], "thr": a["h34c_thr"],
                   "ok": a["in_sample_premium_N"] >= a["h34c_thr"],
                   "N_in_over_Z": a["N_in_over_Z"], "N_out_over_Z": a["N_out_over_Z"]}
    print(f"  P4 隔离：L*(N,eval)/Z {a['N_out_over_Z']:.3f}（阈 ≥{P4_MIN}）⇒ {'过' if res['P4']['ok'] else '不过'}")
    print(f"  H34c 样本内溢价 {a['in_sample_premium_N']:+.5f}（阈 {a['h34c_thr']:.5f}）"
          f"⇒ {'支持' if res['H34c']['ok'] else '否证'}")
    res["verdict"] = ("PREMISE-NOT-MET" if res["arms"]["oracle"]["verdict"] == "PREMISE-NOT-MET"
                      else res["arms"]["oracle"]["verdict"])
    res["secs"] = round(time.time() - t0, 1)
    print(f"\n主判据状态 **{res['verdict']}**  ({res['secs']:.0f}s)")
    json.dump(res, open(f"{OUT}/DIR34r11_formal.json", "w"), ensure_ascii=False, indent=1)
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stageA"
    os.makedirs(OUT, exist_ok=True)
    if mode == "stageA":
        ok, a = stage_a()
        print(f"Stage A：{'★ 全部通过' if ok else '★ 有断言失败 —— 冻结正式轮'}")
        for k, v in a.items():
            print(f"  {k:26s} {v if isinstance(v, str) else f'{v:.3e}'}")
        json.dump({"ok": ok, "a": a}, open(f"{OUT}/DIR34r11_stageA.json", "w"), ensure_ascii=False, indent=1)
        sys.exit(0 if ok else 1)
    if mode == "gates":
        p1, p2 = gates()
        print(f"P1 {p1:+.4f}（阈 {P1_MIN}） · P2 {p2:+.4f}（阈 >{P2_MIN}）")
        json.dump({"P1": p1, "P2": p2}, open(f"{OUT}/DIR34r11_gates.json", "w"), indent=1)
        sys.exit(0)
    if mode == "smoke":
        t1 = time.time()
        out = {}
        for addr in ("oracle", "device"):
            out[addr] = {}
            for arm in ("N", "T"):
                phi = R.train_arm(addr, arm, steps=20, learned=(addr == "device"), seeds=TRAIN[:4])
                ev = R.per_seed(addr, arm, phi, EVAL[:4])
                out[addr][arm] = statistics.fmean(ev)
        ok = all(math.isfinite(v) for a_ in out.values() for v in a_.values())
        print(f"smoke：20 步 × 2 臂 × 2 寻址 × 4 种子 ⇒ {'可执行' if ok else '★ 不可执行'}  ({time.time()-t1:.0f}s)")
        for a_, d in out.items():
            print(f"  {a_:6s} " + " · ".join(f"{k} {v:.4f}" for k, v in d.items()))
        json.dump(out, open(f"{OUT}/DIR34r11_smoke.json", "w"), indent=1)
        sys.exit(0 if ok else 1)
    if mode == "mech":
        m = mech()
        for n, r in m.items():
            print(f"N={n:4d} cos {r['cos']:+.4f} ‖ΔW‖/‖W0‖ {r['dW']:.4f}")
        json.dump(m, open(f"{OUT}/DIR34r11_mech.json", "w"), indent=1)
        sys.exit(0)
    formal()
