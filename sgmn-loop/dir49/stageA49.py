#!/opt/pytorch/bin/python
r"""DIR-49 · Stage A（**硬门**）：卡 §7 的 `A49-1…9`。任一失败 ⇒ 正式轮不得启动。

用法：`stageA49.py --out dir49/out/DIR49r10_stageA.json`
"""
import argparse, glob, hashlib, json, math, os, sys
import torch
torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir49"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task46 as T46      # noqa: E402
import R1_task48 as T48      # noqa: E402
import R1_task49 as T49      # noqa: E402
import workex49 as WX        # noqa: E402

CARD = os.path.join(ROOT, "revolve/DIR-49/DIRECTION_rev49_v12.md")
CARD_MD5 = "3977442d90524aa85875e5dd66b476fb"
FORMAL = list(range(29400, 30400))
SEEDS = [29400 + 25 * i for i in range(40)]
res, ok_all = {}, True


def rep(tag, ok, detail):
    global ok_all
    res[tag] = {"ok": bool(ok), "detail": detail}
    ok_all = ok_all and bool(ok)
    print(("  ✓ " if ok else "  ✗ ") + tag + " " + detail, flush=True)


print("=== DIR-49 Stage A ===", flush=True)
rows = {(sd, a): T49.trace_one(sd, a) for sd in SEEDS for a in T49.ARMS}

# A49-1 写入数
nw = {v["n_writes"] for v in rows.values()}
rep("A49-1", nw == {12}, f"每臂每实例写入数集合 = {sorted(nw)}（应为 [12]）")

# A49-2 工作例
bad, wc = 0, 0.0
for sd in SEEDS[:3]:
    for arm in T49.ARMS:
        ind = WX.indep_run(sd, arm)
        got = rows[(sd, arm)]
        for k in ("ov_k", "ov_v", "c0", "c1", "cN"):
            if ind[k] is None or got[k] is None:
                continue
            d = abs(ind[k] - got[k]); wc = max(wc, d)
            bad += 0 if d <= 1e-9 else 1
        for k in ("k_tgt", "v_tgt"):
            d = max(abs(x - y) for x, y in zip(ind[k], got[k]))
            bad += 0 if d == 0.0 else 1
rep("A49-2", bad == 0, f"工作例 3 实例 × 4 臂；max|Δ| = {wc:.3e}（容差 1e-9）；不一致 {bad}")

# A49-3 四臂 ov_v 逐位相同 + c0 逐位相同
ovv_bad = c0_bad = 0
for sd in SEEDS:
    ovvs = [rows[(sd, a)]["ov_v"] for a in T49.ARMS]
    c0s = [rows[(sd, a)]["c0"] for a in T49.ARMS]
    if len(set(ovvs)) != 1:
        ovv_bad += 1
    if len(set(c0s)) != 1:
        c0_bad += 1
rep("A49-3", ovv_bad == 0 and c0_bad == 0,
    f"四臂 ov_v 逐位相同违例 {ovv_bad}/{len(SEEDS)}；四臂 c0 逐位相同违例 {c0_bad}/{len(SEEDS)}")

# A49-4 ov_k 分档
ovk_ortho_max = max(rows[(sd, "ortho")]["ov_k"] for sd in SEEDS)
ovk_same_min = min(rows[(sd, "d1.0")]["ov_k"] for sd in SEEDS)
mean_by_arm = {a: sum(rows[(sd, a)]["ov_k"] for sd in SEEDS) / len(SEEDS) for a in T49.ARMS}
ordered = mean_by_arm["ortho"] < mean_by_arm["d0.0"] < mean_by_arm["d0.5"] < mean_by_arm["d1.0"]
rep("A49-4", ovk_ortho_max <= 1e-5 and ovk_same_min >= 0.999 and ordered,
    f"ortho 的 ov_k 最大 {ovk_ortho_max:.2e}（≤1e-5）；d1.0 的 ov_k 最小 {ovk_same_min:.6f}（≥0.999）；"
    f"均值序 {({k: round(v,4) for k,v in mean_by_arm.items()})} 有序={ordered}")

# A49-5 目标 (k,v) 四臂逐位相同
tgt_bad = 0
for sd in SEEDS:
    for k in ("k_tgt", "v_tgt"):
        vals = [tuple(rows[(sd, a)][k]) for a in T49.ARMS]
        if len(set(vals)) != 1:
            tgt_bad += 1
rep("A49-5", tgt_bad == 0, f"目标 (k_tgt,v_tgt) 四臂逐位相同违例 {tgt_bad}")

# A49-6 种子段隔离
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
rep("A49-6", not inter, f"正式段 {FORMAL[0]}..{FORMAL[-1]}（1000）vs 既往已落盘 seed {len(used)} 个；交集 {len(inter)} 个 {inter[:5]}")

# A49-7 卡 md5 + 常数
md5 = hashlib.md5(open(CARD, "rb").read()).hexdigest()
consts_ok = (T49.N_WRITES == 12 and T48.N_REP == 16 and abs(M.ALPHA - 1.0) < 1e-15
             and abs(M.LAM - 0.05) < 1e-15 and abs(M.ETA - 0.2) < 1e-15
             and R.D_H == 8 and R.D_K == 16 and T49.N_INT == 5 and T49.N_AFTER == 7)
rep("A49-7", md5 == CARD_MD5 and consts_ok,
    f"卡 md5={md5}；N_WRITES/N_REP/ALPHA/LAM/ETA/D_H/D_K/N_INT/N_AFTER="
    f"{T49.N_WRITES}/{T48.N_REP}/{M.ALPHA}/{M.LAM}/{M.ETA}/{R.D_H}/{R.D_K}/{T49.N_INT}/{T49.N_AFTER}")

# A49-8 DEF-48-4 回归：固定 NMARK ⇒ 跨长度前缀逐位相同；并记录反例
kv_ok, old_bug = 0.0, 0.0
for sd in SEEDS[:8]:
    _, Ka, Va = T48.build_KV(sd, T49.CTX + 8)
    _, Kb, Vb = T48.build_KV(sd, T49.CTX + 32)
    for tt in range(T49.CTX + 8):
        kv_ok = max(kv_ok, float((Ka[tt] - Kb[tt]).abs().max()), float((Va[tt] - Vb[tt]).abs().max()))
    a16 = T46.content(sd, T49.CTX + 8, "perstep"); a32 = T46.content(sd, T49.CTX + 32, "perstep")
    old_bug = max(old_bug, float((a16 - a32[:T49.CTX + 8]).abs().max()))
rep("A49-8", kv_ok == 0.0 and old_bug > 0.0,
    f"跨长度 K/V 前缀 max|Δ|={kv_ok:.3e}；★ 反例（按各长度生成）max|Δ|={old_bug:.3e} > 0 ⇒ DEF-48-4 已治")

# A49-9 判据量可算 + 方向自检
n_bad = sum(1 for sd in SEEDS for a in T49.ARMS if rows[(sd, a)]["rel1"] is None)
n_bad_inst = sum(1 for sd in SEEDS if any(rows[(sd, a)]["rel1"] is None for a in T49.ARMS))
de = []
for sd in SEEDS:
    o = rows[(sd, "ortho")]["rel1"]; s = rows[(sd, "d1.0")]["rel1"]
    if o is not None and s is not None:
        de.append(o - s)
nbad_frac = n_bad / (len(SEEDS) * len(T49.ARMS))
de_mean = sum(de) / len(de) if de else None
rep("A49-9", nbad_frac <= 0.10 and de_mean is not None and de_mean > 0,
    f"n_bad(判据口径: rel1 不可算)={nbad_frac:.4f} ≤ 0.10（实例口径 {n_bad_inst}/{len(SEEDS)}）；"
    f"方向自检 DELTA_END 均值 = {de_mean:.4f} > 0（不入判据）")

# A49-10 ★ `DEF-49-4` 回归：`cN` 非有限**不得**成为 `rel1` 不可算的原因
#   不变量（正确表述）：凡 `(c0,c1)` 有限且 `c0>0` 的行，`rel1` 必须可算 —— **无论 `cN` 是否非有限**。
n_ok_in = n_ok_out = 0
viol = 0
for sd in SEEDS:
    for a in T49.ARMS:
        row = rows[(sd, a)]
        good_in = row["finite_c0c1"] and math.isfinite(row["c0"]) and row["c0"] > 0
        if good_in:
            n_ok_in += 1
            if row["rel1"] is None:
                viol += 1
        else:
            n_ok_out += 1
n_cN_bad = sum(1 for sd in SEEDS for a in T49.ARMS if rows[(sd, a)]["bad_report"])
rep("A49-10", viol == 0,
    f"(c0,c1) 有限且 c0>0 的 {n_ok_in} 行：rel1 全部可算（违例 {viol}）；"
    f"(c0,c1) 不可用的 {n_ok_out} 行（rel1 不可算属正当）；cN 非有限共 {n_cN_bad} 行 ⇒ `cN` **不**污染判据量（`DEF-49-4` 已修）")

print(f"\n{'★ Stage A 全部通过' if ok_all else '✗ Stage A 失败 ⇒ 正式轮不得启动'}", flush=True)
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"card_md5": CARD_MD5, "checks": res, "all_pass": ok_all},
          open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
sys.exit(0 if ok_all else 1)
