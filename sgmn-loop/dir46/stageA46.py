#!/opt/pytorch/bin/python
r"""DIR-46 · Stage A 硬门（`A46-1 … A46-6`；卡 §7）。任一失败 ⇒ 正式轮**不得**启动。

★ `A46-2` 直接复用 `workex46`（独立手写实现）⇒ **不**引入第二份测量逻辑。
★ 本文件**不改**任何阈值（卡 §8 红线 4）。
"""
import json
import math
import os
import sys

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46"):
    sys.path.insert(0, os.path.join(ROOT, _d))

import R1_grad as M             # noqa: E402
import R1_task33 as R           # noqa: E402
import R1_task46 as T           # noqa: E402  ★ 被测实现（`A46-1/3/5` 用它的测量层）
import workex46 as WX           # noqa: E402  ★ 独立手写实现（`A46-2`）

OUT = os.path.join(ROOT, "dir46", "out")
PROBE_JSON = os.path.join(OUT, "DIR46_probe_a.json")
STAGE_SEEDS = [M.SEED0 + 23800 + i for i in range(12)]     # 诊断段（与 probe46a 逐字相同）
ATOL = 1e-12
FORMAL_SEED0 = M.SEED0 + 24000
N_FORMAL = 400
# ★ `A46-6`：既往**全部**启用过的种子段（含 DIR-33/41 的训练段与 DIR-42…45 的各诊断段）
USED = [(0, 99), (400, 423), (500, 523), (21000, 21095), (21500, 21899),
        (22500, 22511), (23100, 23111), (23200, 23211), (23300, 23311),
        (23400, 23411), (23500, 23511), (23600, 23611), (23700, 23711), (23800, 23811)]
res = {"meta": {"card": "rev 1.0", "card_md5": "19ace86942e89c0c4608b07174a2770a",
                "atol": ATOL, "stage_seeds": STAGE_SEEDS,
                "formal_seed0": FORMAL_SEED0, "n_formal": N_FORMAL}, "checks": []}


def add(name, ok, detail):
    res["checks"].append({"id": name, "ok": bool(ok), "detail": detail})
    print(f"  {'✓' if ok else '✗'} {name}: {detail}")
    return ok


def main():
    ok_all = True
    ref = json.load(open(PROBE_JSON, encoding="utf-8"))

    # ── A46-1 位级锚（诊断种子、N_w=8、两模式、逐实例）
    rows, worst = [], 0.0
    for mode in T.MODES:
        cell = next(c for c in ref["cells"] if c["mode"] == mode and c["n_w"] == 8)
        for r in cell["rows"]:
            mine = T.trace_one(r["seed"], mode, 8)
            d = (0.0 if (mine["C"] is None or r["C"] is None) else abs(mine["C"] - r["C"]))
            dw = (0.0 if (mine["wnorm"] is None or r["wnorm"] is None)
                  else abs(mine["wnorm"] - r["wnorm"]))
            bad = (mine["finite"] != r["finite"]) or ((mine["C"] is None) != (r["C"] is None))
            worst = max(worst, d, dw)
            rows.append({"mode": mode, "seed": r["seed"], "dC": d, "dW": dw, "mismatch": bad})
    n_bad = sum(1 for x in rows if x["mismatch"] or x["dC"] > ATOL or x["dW"] > ATOL)
    ok_all &= add("A46-1", n_bad == 0,
                  f"{len(rows)} 实例（2 模式 × 12）× N_w=8：不一致 {n_bad}，最大|Δ| {worst:.3e} ≤ ATOL {ATOL:g}"
                  if worst < 1e-15 else f"{len(rows)} 实例：不一致 {n_bad}，最大|Δ| {worst:.3e}")

    # ── A46-2 独立实现一致（复用 `workex46` 的三格 × 12 实例）
    ok2, bad2, worst2 = 0, [], 0.0
    for mode, n_w in WX.CASES:
        cell = next(c for c in ref["cells"] if c["mode"] == mode and c["n_w"] == n_w)
        for r in cell["rows"]:
            g = WX.c_indep(r["seed"], mode, n_w)
            dc = (0.0 if (g["C"] is None or r["C"] is None) else abs(g["C"] - r["C"]))
            worst2 = max(worst2, dc)
            if (g["finite"] != r["finite"]) or ((g["C"] is None) != (r["C"] is None)) or dc > ATOL:
                bad2.append((mode, n_w, r["seed"], dc))
            else:
                ok2 += 1
    ok_all &= add("A46-2", not bad2, f"独立实现 {ok2}/{ok2+len(bad2)} 一致（最大|Δ|{worst2:.3e}）"
                  + (f"；不一致 {bad2[:3]}" if bad2 else ""))

    # ── A46-3 判据非空：C(2) ≥ C_HIGH（两模式，诊断种子）
    a3 = {}
    for mode in T.MODES:
        cs = [T.trace_one(sd, mode, T.ANCHOR_NW)["C"] for sd in STAGE_SEEDS]
        cs = [c for c in cs if c is not None]
        a3[mode] = {"mean": sum(cs) / len(cs), "n": len(cs)}
    good3 = all(v["mean"] >= 0.9 and v["n"] == len(STAGE_SEEDS) for v in a3.values())
    ok_all &= add("A46-3", good3, " ".join(f"{m}: C(2)={v['mean']:+.5f}(n={v['n']})" for m, v in a3.items()))

    # ── A46-4 `k_0=0` 恒等式：ALPHA=0 的 trace 与「两步都用 k=0 的同一对」逐位相同
    P = R.build(STAGE_SEEDS[0], "V0")
    a = T.content(STAGE_SEEDS[0], 2, "perstep")
    H = [torch.tanh(a[t] @ P["A1"] + P["b1"]) for t in range(2)]
    K = [torch.tanh((torch.zeros(T.D_H) if s == 0 else H[s - 1]) @ P["Pk1"]) for s in range(2)]
    Vv = [torch.tanh(H[s] @ P["Pv1"]) for s in range(2)]

    def _run(alpha, kseq, vseq):
        U, V = P["U1"].clone(), P["V1"].clone()
        for k, v in zip(kseq, vseq):
            for _ in range(T.N_REP):
                W = M._W(U, V)
                G = alpha * torch.outer(W @ k - v, k) + M.LAM * W
                U = U - M.ETA * (G @ V)
                V = V - M.ETA * (G.t() @ U)
        return M._W(U, V)

    z = torch.zeros_like(K[0])     # ★ DEF-46-4：`k` 的维度是 `D_K`（`tanh(h@Pk1)` 之后），不是 `D_H`
    WA = _run(0.0, [K[0], K[1]], [Vv[0], Vv[1]])          # ALPHA=0
    WB = _run(M.ALPHA, [z, z], [Vv[0], Vv[1]])            # 两步都写「k=0 的那一对」
    d4 = float((WA - WB).abs().max())
    WC = _run(M.ALPHA, [z, z], [-Vv[0], Vv[1]])           # ★ 附：v 取反也必须同值（k=0 时 v 不进梯度）
    d4b = float((WB - WC).abs().max())
    ok_all &= add("A46-4", d4 <= ATOL,
                  f"ALPHA=0 vs 只写 k=0 对：max|ΔW|={d4:.3e} ≤ {ATOL:g}；"
                  f"附 v→−v 不变性 max|ΔW|={d4b:.3e}")

    # ── A46-5 删失计数自洽（探针行 + 实现抽样）
    bad5 = []
    for c in ref["cells"]:
        for r in c["rows"]:
            if not r["finite"]:
                continue
            if r["n_read"] + r["n_skip"] != r["n_w"] - 1:
                bad5.append(("probe", c["mode"], c["n_w"], r["seed"]))
    for mode in T.MODES:
        for sd in STAGE_SEEDS[:3]:
            row = T.one_instance(sd, mode)
            for k, cc in row["cells"].items():
                if cc["n_read"] + cc["n_skip"] != int(k) - 1:
                    bad5.append(("impl", mode, k, sd))
    ok_all &= add("A46-5", not bad5, f"计数恒等式 n_read+n_skip=N_w−1：违规 {len(bad5)} 处"
                  + (f"（{bad5[:3]}）" if bad5 else ""))

    # ── A46-6 种子不相交
    formal = set(range(FORMAL_SEED0, FORMAL_SEED0 + N_FORMAL))
    hit = [(lo, hi) for lo, hi in USED if formal & set(range(M.SEED0 + lo, M.SEED0 + hi + 1))]
    ok_all &= add("A46-6", not hit, f"正式段 offsets {24000}..{24000+N_FORMAL-1} 与既往 14 段交集：{hit or '空'}")

    res["all_pass"] = bool(ok_all)
    res["A46-3_detail"] = a3
    p = os.path.join(OUT, "DIR46r10_stageA.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(("★ Stage A 全过" if ok_all else "★★ Stage A 有失败 ⇒ 正式轮不得启动") + f" → {p}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
