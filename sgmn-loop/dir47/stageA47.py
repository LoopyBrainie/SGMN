#!/opt/pytorch/bin/python
r"""DIR-47 · Stage A（**硬门**）：卡 §7 的 `A47-1…7`。任一失败 ⇒ 正式轮不得启动。

用法：`stageA47.py --out dir47/out/DIR47r10_stageA.json`
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys

import torch

torch.set_default_dtype(torch.float64)

ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task47 as T47      # noqa: E402
import workex47 as WX        # noqa: E402

CARD = os.path.join(ROOT, "revolve/DIR-47/DIRECTION_rev47_v10.md")
CARD_MD5 = "58cb94d44327ae00ca2236d6e56d67d2"
FORMAL = list(range(25200, 25700))
res, ok_all = {}, True


def rep(tag, ok, detail):
    global ok_all
    res[tag] = {"ok": bool(ok), "detail": detail}
    ok_all = ok_all and bool(ok)
    print(("  ✓ " if ok else "  ✗ ") + tag + " " + detail, flush=True)


print("=== DIR-47 Stage A ===", flush=True)
SEEDS = [25200, 25250, 25300, 25400]

# ── A47-1 等写入数（数出来）+ A47-5 计数恒等式
rows = {(sd, a): T47.trace_one(sd, a) for sd in SEEDS for a in T47.ARMS}
nw = {k: v["n_writes"] for k, v in rows.items()}
ok1 = all(v == 13 for v in nw.values()) and len({v for v in nw.values()}) == 1
rep("A47-1", ok1, f"n_writes 全部 = 13（{len(nw)} 个 (seed,arm) 对，唯一值 = {sorted(set(nw.values()))}）")
ok5 = all(len(v["hit"]) + len(v["rest"]) == 7 and v["n_read"] + v["n_skip"] == 7
          and len(v["slots"]) == v["n_read"] for v in rows.values())
rep("A47-5", ok5, "每行 len(hit)+len(rest)==7 且 n_read+n_skip==7 且 len(slots)==n_read")

# ── A47-2 工作例（独立第二链，逐位）
worst, bad = 0.0, 0
for sd in SEEDS:
    for arm in T47.ARMS:
        ind, pk_i = WX.indep_run(sd, arm)
        got = rows[(sd, arm)]
        bad += 0 if pk_i == got["picks"] else 1
        for s, v in ind.items():
            d = abs(got["slots"][str(s)] - v)
            worst = max(worst, d)
            bad += 0 if d <= 1e-12 else 1
rep("A47-2", bad == 0, f"{len(SEEDS)*2} 次运行 × 7 槽位 = {len(SEEDS)*2*7} 单元；max|Δ| = {worst:.3e}；不一致 {bad}")

# ── A47-3 两臂 picks 相同 + 前缀 K/V 逐位相同
pk_same = all(rows[(sd, "nor")]["picks"] == rows[(sd, "rep_on")]["picks"] for sd in SEEDS)
rng_ok = all(1 <= p <= 7 for v in rows.values() for p in v["picks"])
worst_kv = 0.0
for sd in SEEDS:
    P = R.build(sd, "V0")
    K1, V1 = T47.build_KV(sd, P)
    K2, V2 = T47.build_KV(sd, P)      # 同 seed 同函数 ⇒ 必须逐位相同（也顺带验无隐藏状态）
    for t in range(T47.CTX + T47.S_MEAS):
        worst_kv = max(worst_kv, float((K1[t] - K2[t]).abs().max()), float((V1[t] - V2[t]).abs().max()))
rep("A47-3", pk_same and rng_ok and worst_kv == 0.0,
    f"两臂 picks 逐位相同={pk_same}；1≤p≤7={rng_ok}；前缀 K/V max|Δ|={worst_kv:.3e}")

# ── A47-4 退化锚：R=0 时两臂逐位相同
_R0, _M0 = T47.R_EXTRA, T47.M_TOT
try:
    T47.R_EXTRA, T47.M_TOT = 0, T47.CTX + T47.S_MEAS
    d0 = 0.0
    for sd in SEEDS:
        a0, b0 = T47.trace_one(sd, "nor"), T47.trace_one(sd, "rep_on")
        for s in a0["slots"]:
            d0 = max(d0, abs(a0["slots"][s] - b0["slots"][s]))
finally:
    T47.R_EXTRA, T47.M_TOT = _R0, _M0
rep("A47-4", d0 == 0.0, f"R=0 时两臂逐槽位 max|Δ| = {d0:.3e}（设计自检）")

# ── A47-6 种子段隔离（扫全部既往 jsonl 的 seed 字段）
used = set()
for f in glob.glob(os.path.join(ROOT, "dir*/out/*.jsonl")):
    try:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if isinstance(o, dict) and isinstance(o.get("seed"), int):
                    used.add(o["seed"])
    except OSError:
        pass
inter = sorted(set(FORMAL) & used)
rep("A47-6", not inter, f"正式段 {FORMAL[0]}..{FORMAL[-1]}（500）vs 既往已落盘 seed {len(used)} 个；交集 {len(inter)} 个 {inter[:5]}")

# ── A47-7 卡 md5 + 装置常数
md5 = hashlib.md5(open(CARD, "rb").read()).hexdigest()
consts = (T47.N_REP, M.ALPHA, M.LAM, M.ETA, R.D_H)
want = (16, 1.0, 0.05, 0.2, 8)
rep("A47-7", md5 == CARD_MD5 and all(abs(a - b) < 1e-15 for a, b in zip(consts, want)),
    f"卡 md5={md5}；N_REP/ALPHA/LAM/ETA/D_H={consts}")

print(f"\n{'★ Stage A 全部通过' if ok_all else '✗ Stage A 失败 ⇒ 正式轮不得启动'}", flush=True)
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"card_md5": CARD_MD5, "checks": res, "all_pass": ok_all}, open(a.out, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
sys.exit(0 if ok_all else 1)
