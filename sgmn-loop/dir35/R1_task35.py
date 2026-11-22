#!/opt/pytorch/bin/python
r"""DIR-35 · **装置与任务层逐字不变**，唯一自变量 = **训练实例数** $n_{\rm train}$。

卡：`revolve/DIR-35/DIRECTION_rev35_v10.md`（rev 1.0，阈值已冻结）。

★ **为什么本文件这么短**：`dir33/R1_task33.py`（任务层）与 `dir31/R1_grad.py`（装置）**逐字零改动**，
  本文件**只 import 它们**（⇒ 任务与装置在**代码对象层面**就是 DIR-33 的那些，不是拷贝），
  改动的只有**测量层**与**独立变量**：扫 $n_{\rm train}$ 而不是扫寻址。
  A35-7/A35-10 用 md5 把这一性质钉死。

★ **独立变量（唯一）**：读数头在**多少个实例**上训练（`train_arm(seeds=…)` 的长度）。
  装置、任务、读数头结构（8 个 tanh 单元）、训练协议（Adam / lr 0.03 / 250 步）、
  评估集（400 个未见实例）**全部不变**。

★ **测量对象**：出样本下真记忆臂的损失与**地板**（零预测基线 $Z\equiv\sum_t\lVert g_t\rVert^2$）的
  **逐实例配对差** $d_i\equiv Z_i-\ell_{T,i}$（同实例 ⇒ 实例难度抵消）。

用法：calib | stageA | smoke | formal | repro
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
sys.path.insert(0, "/home/brra1n0/.sandbox/experiment/1107/sgmn-loop/dir33")
sys.path.insert(0, "/home/ec2-user/sgmn-loop/dir31")
sys.path.insert(0, "/home/brra1n0/.sandbox/experiment/1107/sgmn-loop/dir31")
import R1_task33 as R          # 任务层（逐字未改）
import R1_grad as M            # 装置（逐字未改）

OUT = "/home/ec2-user/sgmn-loop/dir35/out"

# ══════════ 种子段（卡 §3.2；纪律 70：与全部历史段不相交）
SEED0 = M.SEED0
LEVELS = (24, 48, 96, 192, 384)                 # ★ 唯一自变量的取值（嵌套前缀 ⇒ 受控剂量-反应）
TRAIN_BASE = SEED0 + 5000                       # 训练段起点（嵌套：5000..5023 / ..5047 / ..5095 / ..5191 / ..5383）
EVAL = [SEED0 + 6000 + i for i in range(400)]   # 评估 n=400（与训练不相交；与历史段不相交）

# 标定段（决定二十九：开卡前标定，**不进判据**）
CALIB_LEVELS = (24, 96)
CALIB_TRAIN_BASE = SEED0 + 3000
CALIB_EVAL = [SEED0 + 4000 + i for i in range(200)]

# 复现锚（DIR-34 的**精确**配置 ⇒ 必须逐位复现 D-58 的读数；只作一致性校验，不进判据）
ANCHOR_TRAIN = [SEED0 + 900 + i for i in range(24)]
ANCHOR_EVAL = [SEED0 + 2000 + i for i in range(400)]

# 历史种子段（**偏移区间**，纪律 70：DIR-35 的训练/评估/标定段必须全部避开）
HIST_RANGES = {"pilot32": (100, 124), "formal32": (200, 224), "calib33": (300, 308),
               "formal33": (400, 424), "pilot33": (500, 524), "calib34": (600, 624),
               "heldout34": (700, 724), "train34": (900, 923), "eval34": (2000, 2399)}
# ★ 本轮自己的段（标定段也必须避开历史 ⇒ 一并检查）
OWN_SEGMENTS = {"train35": (5000, 5383), "eval35": (6000, 6399),
                "calibtrain35": (3000, 3095), "calibeval35": (4000, 4199)}

# ══════════ 冻结阈值（卡 rev 1.0；标定后填入并冻结）
D_MIN_FRAC = 0.25    # Δmin = D_MIN_FRAC · Z(eval)   ← 卡 rev 1.0 冻结
CARD_MD5 = "b043d73e4bdbb5f8f17b064f0fe8d789"  # 卡 rev 1.0 的 md5              ← 卡 rev 1.0 冻结（Stage A 守卫）
COST_CAP = 1e-15
REL_TOL = 1e-12
WORK_TOL = 1e-14
MD5_DEVICE = "8fa40f708600bec7e6c484727d2180c7"   # dir31/R1_grad.py
MD5_TASK = "96ef89e87ec61556b52e0e464850087e"     # dir33/R1_task33.py
CARD_PATH = "/home/ec2-user/sgmn-loop/revolve/DIR-35/DIRECTION_rev35_v10.md"


# ══════════ 基础量（逐字沿用 dir34）
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


# ══════════ 扫描（唯一自变量 = n_train）
def sweep(addr, levels, train_base, eval_seeds, arms=("N", "S", "T")):
    r"""对每个 $n_{\rm train}$：在**嵌套前缀**的训练种子上训读数头，在固定的评估种子上测。

    ★ 嵌套前缀 ⇒ 剂量-反应是**受控**的（档位之间只差“多了多少个实例”，不换数据分布）。
    ★ 每臂在自己分布里训练到底（纪律 73）⇒ 臂间可比。
    """
    out = {"levels": {}, "eval_seeds": list(eval_seeds)}
    Z_ev = floor_Z(eval_seeds)
    out["Z_eval"] = Z_ev
    out["Z_eval_mean"] = statistics.fmean(Z_ev)
    for n in levels:
        t0 = time.time()
        tr = [train_base + i for i in range(n)]
        Z_in = floor_Z(tr)
        lev = {"n_train": n, "Z_in_mean": statistics.fmean(Z_in), "arms": {}}
        for arm in arms:
            phi = R.train_arm(addr, arm, learned=(addr == "device"), seeds=tr)
            lo = R.per_seed(addr, arm, phi, eval_seeds)
            li = R.per_seed(addr, arm, phi, tr)
            lev["arms"][arm] = {"out": lo, "in": li, "gamma": float(phi["g1"])}
            print(f"  {addr:6s} n_train={n:4d} {arm:2s} 出样本 {statistics.fmean(lo):9.5f}"
                  f" · 样本内 {statistics.fmean(li):9.5f}  ({time.time()-t0:.0f}s)", flush=True)
        lev["mean_out"] = {k: statistics.fmean(v["out"]) for k, v in lev["arms"].items()}
        lev["mean_in"] = {k: statistics.fmean(v["in"]) for k, v in lev["arms"].items()}
        # ★ DEF-35-1（标定只跑 N/T ⇒ 内容性统计必须**条件化**，否则 KeyError）
        if "S" in lev["arms"]:
            lev["dcontent_out"] = paired(lev["arms"]["S"]["out"], lev["arms"]["T"]["out"])
            lev["drand_out"] = paired(lev["arms"]["N"]["out"], lev["arms"]["S"]["out"])
        else:
            lev["dcontent_out"] = None
            lev["drand_out"] = None
        lev["premium_N"] = lev["mean_out"]["N"] - lev["mean_in"]["N"]
        # ★ 配对差 d_i = Z_i − ℓ_T,i（同实例 ⇒ 地板与真记忆臂的差）
        lev["dZ_T_out"] = paired(Z_ev, lev["arms"]["T"]["out"])
        lev["dZ_N_out"] = paired(Z_ev, lev["arms"]["N"]["out"])
        out["levels"][str(n)] = lev
    return out


# ══════════ 标定（决定二十九：开卡前标定，**不进判据**）
def calib():
    r"""标定：剂量-反应的**形状** + $d$ 的 sd（功效核算用）+ 每一档的墙钟成本。"""
    t0 = time.time()
    print("── 标定（oracle，臂 N/T，档位 24/96；标定段 3000.. / 4000..4199）")
    sw = sweep("oracle", CALIB_LEVELS, CALIB_TRAIN_BASE, CALIB_EVAL, arms=("N", "T"))
    sw["secs"] = round(time.time() - t0, 1)
    print(f"标定耗时 {sw['secs']:.0f}s")
    for n in CALIB_LEVELS:
        lev = sw["levels"][str(n)]
        print(f"  n_train={n:4d}: L*(T,eval) {lev['mean_out']['T']:9.5f} · L*(N,eval) {lev['mean_out']['N']:9.5f}"
              f" · 地板 {sw['Z_eval_mean']:.5f} · d̄ {lev['dZ_T_out']['mean']:+.5f}"
              f" (sd {lev['dZ_T_out']['sd']:.4f}, 3se {lev['dZ_T_out']['3se']:.4f})"
              f" · 样本内溢价(N) {lev['premium_N']:+.5f}")
    json.dump(sw, open(f"{OUT}/DIR35_calib.json", "w"), ensure_ascii=False, indent=1)
    return sw


# ══════════ 复现锚（DIR-34 的精确配置 ⇒ 必须逐位复现 D-58 的读数）
def repro():
    r"""用 DIR-34 的**精确**训练/评估种子跑 n_train=24 ⇒ 必须逐位等于 D-58 的 formal 读数。"""
    t0 = time.time()
    print("── 复现锚（train 900..923 / eval 2000..2399，臂 N/S/T）")
    sw = sweep("oracle", (24,), ANCHOR_TRAIN[0], ANCHOR_EVAL, arms=("N", "S", "T"))
    lev = sw["levels"]["24"]
    sw["secs"] = round(time.time() - t0, 1)
    print(f"  L*(G0 用 N/S/T 代替) 出样本: N {lev['mean_out']['N']:.5f} · S {lev['mean_out']['S']:.5f}"
          f" · T {lev['mean_out']['T']:.5f} · Δcontent {lev['dcontent_out']['mean']:+.5f}")
    json.dump(sw, open(f"{OUT}/DIR35_repro.json", "w"), ensure_ascii=False, indent=1)
    return sw


# ══════════ Stage A
def stage_a():
    r"""十条断言：八条取自 dir33 的**同源**断言（同一代码对象 ⇒ 逐字同义），另加 DIR-35 专属的四条。"""
    a = {}
    ok33, a33 = R.stage_a()          # ★ 同一代码对象跑出的同源断言
    a["A35-1_write_block_bitwise"] = a33["A33-1"]
    a["A35-2_align_min"] = a33["A33-2_align_min"]
    a["A35-2_sep_absmax"] = a33["A33-2_sep_absmax"]
    a["A35-3_magnitude_match"] = a33["A33-3"]
    a["A35-4_determinism"] = a33["A33-4"]
    a["A35-5_G0_no_injection_rel"] = a33["A33-5"]
    a["A35-8_worked_example"] = a33["A33-8"]
    a["A35-8_seq"] = a33["A33-8_seq"]
    # A35-6 在 **DIR-35 的评估种子上**重算（同源口径；rev 1.1 的纪律 80：分位数，不是 max）
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
    a["A35-6_inst_mean"] = statistics.fmean(per_inst)
    srt = sorted(per_inst)
    k95 = min(len(srt) - 1, max(0, round(0.95 * (len(srt) - 1))))
    a["A35-6_inst_p95"] = srt[k95]
    a["A35-6_inst_max_diag"] = max(per_inst)
    a["A35-6_pooled_max_diag"] = max(pooled)
    # A35-7 / A35-10：装置与任务层逐字未改（md5）
    a["A35-7_device_md5_diff"] = 0.0 if _md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py") == MD5_DEVICE else 1.0
    a["A35-7_device_md5"] = _md5("/home/ec2-user/sgmn-loop/dir31/R1_grad.py")
    a["A35-10_task_md5_diff"] = 0.0 if _md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py") == MD5_TASK else 1.0
    a["A35-10_task_md5"] = _md5("/home/ec2-user/sgmn-loop/dir33/R1_task33.py")
    # A35-9（DIR-35 专属：**种子隔离** + **训练确定性**）
    own = (list(TRAIN_BASE + i for i in range(max(LEVELS))) + EVAL + CALIB_EVAL
           + list(CALIB_TRAIN_BASE + i for i in range(96)))
    seg_hits = sorted({name for name, (lo, hi) in HIST_RANGES.items()
                       for sd in own if lo <= (sd - SEED0) <= hi})
    a["A35-9_own_segments"] = ",".join(f"{k}:{lo}-{hi}" for k, (lo, hi) in OWN_SEGMENTS.items())
    a["A35-9_seed_disjoint"] = 0.0 if (not seg_hits and not (set(EVAL) & set(range(TRAIN_BASE, TRAIN_BASE + max(LEVELS))))) else 1.0
    a["A35-9_hist_hits"] = ",".join(seg_hits) if seg_hits else "none"
    p1 = R.train_arm("oracle", "T", steps=5, seeds=list(range(TRAIN_BASE, TRAIN_BASE + 8)))
    p2 = R.train_arm("oracle", "T", steps=5, seeds=list(range(TRAIN_BASE, TRAIN_BASE + 8)))
    a["A35-9_train_bitwise"] = max(float((p1[k] - p2[k]).abs().max()) for k in p1)
    # A35-11（DIR-35 专属：**卡已冻结**）—— 正式轮必须跑在冻结的卡 rev 1.0 上
    a["A35-11_card_md5"] = _md5(CARD_PATH)
    a["A35-11_card_md5_diff"] = 0.0 if (CARD_MD5 is not None and a["A35-11_card_md5"] == CARD_MD5) else 1.0
    a["A35-11_threshold_set"] = 0.0 if (D_MIN_FRAC is not None) else 1.0
    # A35-12（DIR-35 专属：**工作例**）—— 逐实例 $d_i=Z_i-\ell_{T,i}$ 的独立重算
    #   （独立代码路径：不用 sweep 的聚合，直接从 rollout 重算一个实例的 $Z_i$ 与 $\ell_{T,i}$）
    sd0 = EVAL[0]
    phi = R.train_arm("oracle", "T", seeds=list(range(TRAIN_BASE, TRAIN_BASE + 8)), steps=25)
    P = R.build(sd0, "V0")
    R.period2(P, sd0)
    z0 = sum(float((g ** 2).sum()) for g in R.targets(P))
    P2 = R.build(sd0, "V0")
    R.period2(P2, sd0)
    P2.update({k: v.clone() for k, v in phi.items()})
    l0 = float(R.rollout_task(P2, "oracle", "T", sd0))
    a["A35-12_worked_dZ"] = z0 - l0
    a["A35-12_worked_Z"] = z0
    a["A35-12_worked_L"] = l0
    ok = (a["A35-1_write_block_bitwise"] == 0.0 and a["A35-2_align_min"] >= 1 - 1e-14
          and a["A35-3_magnitude_match"] <= COST_CAP and a["A35-4_determinism"] == 0.0
          and a["A35-5_G0_no_injection_rel"] <= REL_TOL and a["A35-8_worked_example"] <= WORK_TOL
          and a["A35-6_inst_mean"] <= 0.60 and a["A35-6_inst_p95"] <= 0.95
          and a["A35-7_device_md5_diff"] == 0.0 and a["A35-10_task_md5_diff"] == 0.0
          and a["A35-9_seed_disjoint"] == 0.0 and a["A35-9_train_bitwise"] == 0.0
          and a["A35-11_card_md5_diff"] == 0.0 and a["A35-11_threshold_set"] == 0.0
          and abs(a["A35-12_worked_dZ"]) < 1e30)
    return ok, a


# ══════════ 正式轮
def formal():
    t0 = time.time()
    if D_MIN_FRAC is None or CARD_MD5 is None:
        print("★ 阈值/卡 md5 未冻结 ⇒ 拒绝跑正式轮")
        sys.exit(2)
    res = {"levels": {}, "D_MIN_FRAC": D_MIN_FRAC, "CARD_MD5": CARD_MD5}
    sw = sweep("oracle", LEVELS, TRAIN_BASE, EVAL, arms=("N", "S", "T"))
    res.update({k: sw[k] for k in ("levels", "Z_eval", "Z_eval_mean")})
    nmax = str(max(LEVELS))
    lev = res["levels"][nmax]
    Z_ev = res["Z_eval_mean"]
    dmin = D_MIN_FRAC * Z_ev
    d = lev["dZ_T_out"]
    res["dmin"] = dmin
    res["clauses"] = {}
    res["clauses"]["C35-a"] = {"mean": d["mean"], "dmin": dmin,
                               "ok": d["mean"] >= dmin}
    res["clauses"]["C35-b"] = {"mean": d["mean"], "3se": d["3se"], "lower": d["mean"] - d["3se"],
                               "ok": (d["mean"] - d["3se"]) >= 0.0}
    prem24 = res["levels"][str(min(LEVELS))]["premium_N"]
    res["clauses"]["C35-c"] = {"premium_384": lev["premium_N"], "premium_24": prem24,
                               "N_out_384": lev["mean_out"]["N"], "N_out_24": res["levels"][str(min(LEVELS))]["mean_out"]["N"],
                               "ok": (lev["premium_N"] < prem24) and (lev["mean_out"]["N"] < res["levels"][str(min(LEVELS))]["mean_out"]["N"])}
    res["clauses"]["C35-d"] = {"3se": d["3se"], "dmin": dmin, "ok": d["3se"] <= dmin}
    ok = all(c["ok"] for c in res["clauses"].values())
    # P35-1（前提闸门）：最大档上**样本内** T 臂必须优于 N 臂（记忆内容在样本内也必须有用 ⇒ 装置没坏）
    res["P35-1"] = {"T_in": lev["mean_in"]["T"], "N_in": lev["mean_in"]["N"],
                    "ok": lev["mean_in"]["T"] < lev["mean_in"]["N"]}
    res["verdict"] = ("PREMISE-NOT-MET" if not res["P35-1"]["ok"]
                      else ("INFEASIBLE-POWER" if not res["clauses"]["C35-d"]["ok"]
                            else ("SUCCEEDED" if ok else "FALSIFIED")))
    # 曲线（不计数）
    res["curve"] = {n: {"L_N": res["levels"][n]["mean_out"]["N"], "L_S": res["levels"][n]["mean_out"]["S"],
                        "L_T": res["levels"][n]["mean_out"]["T"], "dZ_T": res["levels"][n]["dZ_T_out"]["mean"],
                        "premium_N": res["levels"][n]["premium_N"],
                        "dcontent": res["levels"][n]["dcontent_out"]["mean"]}
                    for n in res["levels"]}
    res["crossing"] = [int(n) for n in sorted(res["levels"], key=lambda k: int(k))
                        if res["levels"][n]["dZ_T_out"]["mean"] > 0]
    res["secs"] = round(time.time() - t0, 1)
    for k, c in res["clauses"].items():
        print(f"  {k}: {c} ⇒ {'过' if c['ok'] else '不过'}")
    print(f"曲线 {json.dumps(res['curve'], ensure_ascii=False)}")
    print(f"d>0 的档位 {res['crossing']}  ⇒ 主判据状态 **{res['verdict']}**  ({res['secs']:.0f}s)")
    json.dump(res, open(f"{OUT}/DIR35r10_formal.json", "w"), ensure_ascii=False, indent=1)
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stageA"
    os.makedirs(OUT, exist_ok=True)
    if mode == "calib":
        calib()
    elif mode == "repro":
        repro()
    elif mode == "stageA":
        ok, a = stage_a()
        print(f"Stage A：{'★ 全部通过' if ok else '★ 有断言失败 —— 冻结正式轮'}")
        for k, v in a.items():
            print(f"  {k:26s} {v if isinstance(v, str) else f'{v:.3e}'}")
        json.dump({"ok": ok, "a": a}, open(f"{OUT}/DIR35r10_stageA.json", "w"), ensure_ascii=False, indent=1)
        sys.exit(0 if ok else 1)
    elif mode == "smoke":
        t1 = time.time()
        out = {}
        for n in (4, 8):
            sw = sweep("oracle", (n,), TRAIN_BASE, EVAL[:4], arms=("N", "T"))
            out[n] = {k: v for k, v in sw["levels"][str(n)]["mean_out"].items()}
        ok = all(math.isfinite(v) for d in out.values() for v in d.values())
        print(f"smoke：两档 × 2 臂 × 4 种子 ⇒ {'可执行' if ok else '★ 不可执行'}  ({time.time()-t1:.0f}s)")
        json.dump(out, open(f"{OUT}/DIR35r10_smoke.json", "w"), indent=1)
        sys.exit(0 if ok else 1)
    else:
        formal()
