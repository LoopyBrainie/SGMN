#!/opt/pytorch/bin/python
r"""DIR-49 · 开卡前诊断 B：**混淆排查 + 构造剂量臂可行性**。

★ 本探针回答三件事：
  (1) **配对对比的效应量与分辨力**：每实例取"键重叠最大 vs 最小"的两个写入偏移，配对差 `Δ = rel_hi − rel_lo` 的均值与 `3se`；
  (2) **混淆排查**：`ov_k` 在实例内是否与（a）**值重叠** `ov_v`、（b）**写入残差** `|W k_n − v_n|` 相关？
      若 `ov_k` 与残差强相关 ⇒ 损伤可能来自"残差大"而非"键重叠"⇒ 必须构造剂量臂；
  (3) **构造剂量臂**可行性：把"干扰对"的**键**按剂量 `w` 从自然键连续旋到目标键（内容混合），验证
      `ov_k(w)` 单调、`rel(w)` 单调下降 ⇒ 若成立，可作**因果剂量-反应**（键重叠是唯一被操纵的量）。

用法：`probe49b.py --n 300 --out dir49/out/DIR49_probe_b.json`
"""
import argparse, json, math, os, sys
import torch
torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_task48 as T48      # noqa: E402

OFFS = (1, 2, 4, 8)
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)


def cosv(a, b):
    den = float(a.norm() * b.norm())
    return float(a @ b) / den if den > 0 else float("nan")


def norm_rows(x):
    return x / x.norm(dim=1, keepdim=True) * math.sqrt(x.shape[1])


def write_ctx_target(sd):
    r"""写 ctx(3) + 目标(索引 3)；返回 (P, K, V, U, Vm, c0)。"""
    P, K, V = T48.build_KV(sd, T48.NMARK)
    U, Vm = P["U1"].clone(), P["V1"].clone()
    for t in range(T48.CTX + 1):
        U, Vm = T48.step_pair(U, Vm, K[t], V[t], 1.0)
    kt, vt = K[T48.CTX], V[T48.CTX]
    c0, w0 = T48.read_target(U, Vm, kt, vt)
    return P, K, V, U, Vm, kt, vt, c0, w0


def natural_rows(sd):
    r"""自然臂：四个写入偏移，各记 `ov_k / ov_v / 残差 / rel`。"""
    P, K, V, U, Vm, kt, vt, c0, w0 = write_ctx_target(sd)
    rows = []
    if not (math.isfinite(c0) and c0 > 0):
        return rows
    for d in OFFS:
        tn = T48.CTX + d
        kn, vn = K[tn], V[tn]
        Wm = T48.M._W(U, Vm)
        res = float((Wm @ kn - vn).norm())
        ovk = abs(cosv(kt, kn)); ovv = abs(cosv(vt, vn))
        U2, V2 = T48.step_pair(U, Vm, kn, vn, 1.0)
        c1, _ = T48.read_target(U2, V2, kt, vt)
        rows.append({"d_off": d, "ov_k": ovk, "ov_v": ovv, "res": res,
                     "rel": (c1 / c0) if (math.isfinite(c1) and c0 > 0) else None})
    return rows


def dose_rows(sd):
    r"""构造剂量臂：干扰对索引 `n = CTX+2 = 5`，其**键**由 `a_4` 决定、`值**由 `a_5` 决定。
    ★ 目标对(索引 3)的 `(k,v)` 由 `a_2,a_3` 决定 ⇒ **剂量不动目标**。
    ★ 剂量只改 `a_4`：`a_4(w) = norm((1-w)·a_base + w·a_tgt)`，其中 `a_tgt = a_2`（产生目标键的内容）。
      `w=1` ⇒ `a_4 = a_2` ⇒ 干扰键 = 目标键（地址重合）；`w=0` ⇒ 自然。
    ★ `a_5` 不动 ⇒ 干扰**值**在所有剂量上相同（值重叠被控住）。
    """
    P, K, V = T48.build_KV(sd, T48.NMARK)
    # 重算 K/V，使 a_4 被替换（a_4 影响 k_5 与 v_4）
    a = T48.content(sd, T48.NMARK)
    A1, b1 = P["A1"], P["b1"]
    Pk1, Pv1 = P["Pk1"], P["Pv1"]
    D_H = T48.D_H
    out = []
    base4, tgt = a[T48.CTX + 1].clone(), a[T48.CTX - 1].clone()   # a_4, a_2
    for w in DOSES:
        a2 = a.clone()
        a2[T48.CTX + 1] = norm_rows((1 - w) * base4.unsqueeze(0) + w * tgt.unsqueeze(0))[0]
        H = [torch.tanh(a2[t] @ A1 + b1) for t in range(T48.NMARK)]
        Kk = [torch.tanh((torch.zeros(D_H) if t == 0 else H[t - 1]) @ Pk1) for t in range(T48.NMARK)]
        Vv = [torch.tanh(H[t] @ Pv1) for t in range(T48.NMARK)]
        U, Vm = P["U1"].clone(), P["V1"].clone()
        for t in range(T48.CTX + 1):
            U, Vm = T48.step_pair(U, Vm, Kk[t], Vv[t], 1.0)
        kt, vt = Kk[T48.CTX], Vv[T48.CTX]
        c0, _ = T48.read_target(U, Vm, kt, vt)
        if not (math.isfinite(c0) and c0 > 0):
            continue
        n = T48.CTX + 2
        kn, vn = Kk[n], Vv[n]
        ovk = abs(cosv(kt, kn)); ovv = abs(cosv(vt, vn))
        Wm = T48.M._W(U, Vm)
        res = float((Wm @ kn - vn).norm())
        U2, V2 = T48.step_pair(U, Vm, kn, vn, 1.0)
        c1, _ = T48.read_target(U2, V2, kt, vt)
        out.append({"w": w, "ov_k": ovk, "ov_v": ovv, "res": res,
                    "rel": (c1 / c0) if math.isfinite(c1) else None})
    return out


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def sd3(v):
    v = [x for x in v if x is not None]
    n = len(v)
    if n < 2:
        return None
    m = sum(v) / n
    s = (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5
    return 3 * s / math.sqrt(n)


def pear(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=29100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    deltas, ovk_res, ovk_ovv, nat = [], [], [], []
    dose_acc = {w: [] for w in DOSES}
    n_bad_nat = n_bad_dose = 0
    for i in range(a.n):
        sd = a.seed0 + i
        rs = natural_rows(sd)
        if len(rs) == len(OFFS) and all(r["rel"] is not None for r in rs):
            nat.append(rs)
            hi = max(rs, key=lambda r: r["ov_k"]); lo = min(rs, key=lambda r: r["ov_k"])
            deltas.append(hi["rel"] - lo["rel"])
            ovk_res.append(pear([r["ov_k"] for r in rs], [r["res"] for r in rs]))
            ovk_ovv.append(pear([r["ov_k"] for r in rs], [r["ov_v"] for r in rs]))
        else:
            n_bad_nat += 1
        ds = dose_rows(sd)
        if len(ds) == len(DOSES) and all(d["rel"] is not None for d in ds):
            for d in ds:
                dose_acc[d["w"]].append(d)
        else:
            n_bad_dose += 1
    out = {"n": a.n, "seed0": a.seed0, "natural_bad": n_bad_nat, "dose_bad": n_bad_dose,
           "paired_delta": {"n": len(deltas), "mean": mean(deltas), "3se": sd3(deltas)},
           "within_instance_pear_ovk_res": {"mean": mean(ovk_res)},
           "within_instance_pear_ovk_ovv": {"mean": mean(ovk_ovv)},
           "dose": {}}
    for w in DOSES:
        ds = dose_acc[w]
        out["dose"][str(w)] = {"n": len(ds), "ov_k": mean([d["ov_k"] for d in ds]),
                               "ov_v": mean([d["ov_v"] for d in ds]),
                               "res": mean([d["res"] for d in ds]),
                               "rel": mean([d["rel"] for d in ds]),
                               "rel_3se": sd3([d["rel"] for d in ds])}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
