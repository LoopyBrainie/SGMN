#!/opt/pytorch/bin/python
r"""DIR-49 · 开卡前诊断 A：**键重叠度的自然分布**与「目标损伤 vs 键重叠度」是否单调。

★ 任务卡红线①要求：先量 `|k_tgtᵀ k_new|` 的分布，若 90% 落在同一窄带 ⇒ 本方向**不可自然分层**。
★ 本探针一次性给三个读数：
  (1) 自然键重叠度分布（`|cos|` 的分位数）；
  (2) 自然值重叠度分布（`|cos(v_tgt, v_new)|`）；
  (3) **单次干扰损伤** `rel = c(1)/c(0)`（写一个新对之后目标的相对余弦）**对** 键重叠度的关系
      （Pearson/Spearman + 分层均值）—— 若天然就有单调性 ⇒ 可直接用 `perstep` 分层；
      若没有 ⇒ 必须**构造**重叠度（改内容相关性）。

用法：`probe49a.py --n 400 --out dir49/out/DIR49_probe_a.json`
"""
import argparse, json, math, os, sys
import torch
torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_task48 as T48      # noqa: E402   （含 DEF-48-4 的 NMARK 修复与写循环）


def cosv(a, b):
    den = float(a.norm() * b.norm())
    return float(a @ b) / den if den > 0 else float("nan")


def one(sd, d_off):
    r"""ctx(3) + 目标(1)，然后写**一个**新对（偏移 `d_off`），返回重叠度与损伤。"""
    P, K, V = T48.build_KV(sd, T48.NMARK)
    U, Vm = P["U1"].clone(), P["V1"].clone()
    for t in range(T48.CTX + 1):
        U, Vm = T48.step_pair(U, Vm, K[t], V[t], 1.0)
    kt, vt = K[T48.CTX], V[T48.CTX]
    c0, w0 = T48.read_target(U, Vm, kt, vt)
    if not (math.isfinite(c0) and c0 > 0):
        return None
    tn = T48.CTX + d_off
    kn, vn = K[tn], V[tn]
    ov_k = abs(cosv(kt, kn))
    ov_v = abs(cosv(vt, vn))
    ov_j = ov_k * ov_v
    U2, V2 = T48.step_pair(U, Vm, kn, vn, 1.0)
    c1, w1 = T48.read_target(U2, V2, kt, vt)
    if not (math.isfinite(c1) and math.isfinite(w1)):
        return {"undef": True, "ov_k": ov_k, "ov_v": ov_v, "ov_j": ov_j}
    return {"undef": False, "ov_k": ov_k, "ov_v": ov_v, "ov_j": ov_j,
            "rel": c1 / c0, "c0": c0, "c1": c1, "wnorm_ratio": w1 / w0}


def q(xs, p):
    s = sorted(xs)
    if not s:
        return None
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=29000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for i in range(a.n):
        sd = a.seed0 + i
        for d_off in (1, 2, 4, 8):        # 多个写入偏移 ⇒ 天然的重叠度样本
            o = one(sd, d_off)
            if o is None:
                continue
            o["seed"] = sd; o["d_off"] = d_off
            rows.append(o)
    fin = [r for r in rows if not r.get("undef")]
    ovk = [r["ov_k"] for r in fin]
    ovv = [r["ov_v"] for r in fin]
    rel = [r["rel"] for r in fin]
    out = {"n_inst": a.n, "seed0": a.seed0, "n_rows": len(rows), "n_undef": len(rows) - len(fin),
           "ov_k": {p: q(ovk, p) for p in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)},
           "ov_v": {p: q(ovv, p) for p in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)},
           "rel": {"mean": sum(rel) / len(rel), "q10": q(rel, .1), "q50": q(rel, .5), "q90": q(rel, .9)},
           "pearson_rel_vs_ovk": pearson(ovk, rel), "spearman_rel_vs_ovk": spearman(ovk, rel),
           "pearson_rel_vs_ovv": pearson(ovv, rel), "spearman_rel_vs_ovv": spearman(ovv, rel),
           "pearson_rel_vs_ovj": pearson([r["ov_j"] for r in fin], rel),
           "spearman_rel_vs_ovj": spearman([r["ov_j"] for r in fin], rel),
           "rel_by_doff": {}}
    for d in (1, 2, 4, 8):
        sub = [r["rel"] for r in fin if r["d_off"] == d]
        out["rel_by_doff"][str(d)] = {"n": len(sub), "mean": (sum(sub) / len(sub)) if sub else None}
    # ★ 分层：按 ov_k 的**固定**分位（0.1/0.9）看 rel 均值差（诊断用，不作判据）
    lo, hi = q(ovk, 0.10), q(ovk, 0.90)
    band = sum(1 for x in ovk if lo <= x <= hi) / len(ovk)
    s_lo = [r["rel"] for r in fin if r["ov_k"] <= lo]
    s_hi = [r["rel"] for r in fin if r["ov_k"] >= hi]
    out["band_10_90_frac"] = band
    out["rel_low_ovk"] = {"n": len(s_lo), "mean": (sum(s_lo) / len(s_lo)) if s_lo else None}
    out["rel_high_ovk"] = {"n": len(s_hi), "mean": (sum(s_hi) / len(s_hi)) if s_hi else None}
    out["low_high_diff"] = ((sum(s_hi) / len(s_hi)) - (sum(s_lo) / len(s_lo))) if s_lo and s_hi else None
    print(json.dumps(out, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
