#!/opt/pytorch/bin/python
r"""DIR-49 · **开卡前标定**：给判据量定点估计与 `3se`（阈值冻结前必须拿到）。

★ 复用 `probe49c` 的构造（正交锚 / 剂量旋钮），但：
  ① 每实例只写一次 ctx+目标，然后**分叉**各剂量臂（省时且保证配对）；
  ② 记录 `res`（写入残差）以核对"伤害不是残差驱动"；
  ③ 统计**实例内相邻配对差的符号比例**（单调性判据量）。

用法：`calib49.py --n 300 --out dir49/out/DIR49_calib.json`
"""
import argparse, json, math, os, sys, time
import torch
torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_task48 as T48      # noqa: E402
import probe49c as PC        # noqa: E402

ARM_KEYS = ("ortho", "d0.0", "d0.5", "d1.0")
ORDER = ["ortho", "d0.0", "d0.5", "d1.0"]      # ov_k 递增序


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def sd3(v):
    v = [x for x in v if x is not None]
    n = len(v)
    if n < 2:
        return None
    m = sum(v) / n
    return 3 * ((sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5) / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=29300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    per = {k: {"ov_k": [], "ov_v": [], "res": [], "rel1": [], "relN": []} for k in ARM_KEYS}
    mono_pos, mono_tot = 0, 0
    diffs = []
    nbad = 0
    for i in range(a.n):
        sd = a.seed0 + i
        P, K, V, U, Vm, kt, vt, c0 = PC.setup(sd)
        if not (math.isfinite(c0) and c0 > 0):
            nbad += 1
            continue
        a_full = T48.content(sd, T48.NMARK).clone()
        base4 = a_full[T48.CTX + 1].clone(); tgt2 = a_full[T48.CTX - 1].clone()
        x_ortho, ov_ortho = PC.ortho_a4(P, a_full, kt)
        row = {}
        for w, key in ((0.0, "d0.0"), (0.5, "d0.5"), (1.0, "d1.0")):
            a4 = PC.norm_rows(((1 - w) * base4.unsqueeze(0) + w * tgt2.unsqueeze(0)))[0]
            row[key] = PC.run_arm(sd, a4)
        row["ortho"] = PC.run_arm(sd, x_ortho)
        if any((row[k] is None) or row[k].get("undef") for k in ARM_KEYS):
            nbad += 1
            continue
        for k in ARM_KEYS:
            r = row[k]
            per[k]["ov_k"].append(r["ov_k"]); per[k]["ov_v"].append(r["ov_v"])
            per[k]["rel1"].append(r["rel1"]); per[k]["relN"].append(r["relN"])
        sgn_ok = True
        for i2 in range(len(ORDER) - 1):
            d = row[ORDER[i2]]["rel1"] - row[ORDER[i2 + 1]]["rel1"]
            diffs.append(d)
            if d <= 0:
                sgn_ok = False
        mono_tot += 1
        mono_pos += 1 if sgn_ok else 0
    out = {"n": a.n, "seed0": a.seed0, "n_bad": nbad, "n_use": mono_tot,
           "wall_s": time.time() - t0, "arms": {}}
    for k in ARM_KEYS:
        out["arms"][k] = {"n": len(per[k]["rel1"]),
                          "ov_k": mean(per[k]["ov_k"]), "ov_k_max": max(per[k]["ov_k"]) if per[k]["ov_k"] else None,
                          "ov_v": mean(per[k]["ov_v"]),
                          "rel1": mean(per[k]["rel1"]), "rel1_3se": sd3(per[k]["rel1"]),
                          "relN": mean(per[k]["relN"]), "relN_3se": sd3(per[k]["relN"])}
    out["mono_frac_all3"] = mono_pos / mono_tot if mono_tot else None
    out["mono_frac_3se"] = sd3([1.0] * mono_pos + [0.0] * (mono_tot - mono_pos))
    out["adj_diff_mean"] = mean(diffs); out["adj_diff_3se"] = sd3(diffs)
    # 判据量候选
    d_span = [per["ortho"]["rel1"][i] - per["d1.0"]["rel1"][i] for i in range(len(per["d1.0"]["rel1"]))]
    out["rel_ortho_minus_same"] = {"mean": mean(d_span), "3se": sd3(d_span)}
    out["ovv_span_ortho_vs_same"] = abs(mean(per["ortho"]["ov_v"]) - mean(per["d1.0"]["ov_v"]))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
