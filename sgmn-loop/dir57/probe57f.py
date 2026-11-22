#!/opt/pytorch/bin/python
r"""DIR-57 · **前提探针 F**：给**卡**提供"可预注册阈值"的**唯一**依据——**读出误差**口径的量级与功效。

★ 为什么换口径（`PROBE57E` 的决定性事实）：单笔写入后 `W` 近似 rank-1 ⇒ `Wk' ∥ v` 对**任意**键成立
   （未写键余弦 **44.7%** 落在 `|cos|>0.9`，q05 −0.983 / q95 +0.987）⇒ **方向不含寻址信息**；
   寻址信息在**幅度**（`‖Wk_a‖/median‖Wk'‖` 比值中位 **2.59**）。
   ⇒ 唯一"既含方向又含幅度"、且**退化时自动归零**的功能量是**读出误差**
   $$E_i=\|W k_i-v_P\|,\qquad E_{\rm null}=\mathrm{median}_{k'\in\text{未写}}\|W k'-v_P\|,\qquad S_i=E_{\rm null}-E_i .$$
   ★ 为什么 $S$ 能自动拒绝"塌缩"：若 `W→0` 则 `E_i→‖v_P‖=1` 且 `E_null→1` ⇒ `S→0`（不会像裸余弦那样给出 1.0）。

★ 四臂（同一实例、同一起跑状态；m=4 把键两两 `|cos|≤0.3`，全部同挂 `v_P`）+ 干扰臂：
   `m1` / `m2` / `m4`：写 1/2/4 把同值键；`m4_int8`：写 4 把后再写 **8 条异值**干扰（新键 + 与 `v_P` 不平行的新值）。
★ 输出（供卡预注册）：各臂 `E_i`/`E_null`/`S_i` 的**中位、sd、3se@96**；跨键 `ΔS` 的离散（`max|ΔS_i−ΔS_j|`）。
用法：`/opt/pytorch/bin/python dir57/probe57f.py` → `dir57/out/PROBE57F.json`
"""
import json
import os
import statistics
import sys

import torch

torch.set_default_dtype(torch.float64)
ROOT = "/home/ec2-user/sgmn-loop"
for _d in ("dir31", "dir33", "dir46", "dir47", "dir48", "dir51"):
    sys.path.insert(0, os.path.join(ROOT, _d))
import R1_grad as M          # noqa: E402
import R1_task33 as R        # noqa: E402
import R1_task48 as T48      # noqa: E402
import R1_task51 as T51      # noqa: E402

OUT = os.path.join(ROOT, "dir57", "out")
SEED0 = 20261117
SEEDS = [SEED0 + 58400 + i for i in range(64)]
OFF_K, OFF_V, OFF_R, OFF_INT = 57501, 57502, 57503, 57504
N_UNWRITTEN = 5
N_INT = 8


def write_pair(U, Vm, k, v):
    return T48.step_pair(U, Vm, k, v, 1.0)


def key_batch(g, m, max_cos=0.3):
    ks = []
    while len(ks) < m:
        k = R.KEY_SCALE * T51.gen_vec(g, M.D_K)
        if all(abs(float((k @ q) / (k.norm() * q.norm()))) <= max_cos for q in ks):
            ks.append(k)
    return ks


def val_near_orth(vP, g, max_cos=0.3):
    while True:
        v = T51.gen_vec(g, M.D_V)
        if abs(float(v @ vP)) <= max_cos:
            return v


def errs(U, Vm, ks, vP):
    W = M._W(U, Vm)
    return [float((W @ k - vP).norm()) for k in ks]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sd in SEEDS:
        P = R.build(sd, "V0")
        U0, V0 = P["U1"].clone(), P["V1"].clone()
        gk = torch.Generator().manual_seed(sd + OFF_K)
        gv = torch.Generator().manual_seed(sd + OFF_V)
        gr = torch.Generator().manual_seed(sd + OFF_R)
        gi = torch.Generator().manual_seed(sd + OFF_INT)
        ks = key_batch(gk, 4)
        vP = T51.gen_vec(gv, M.D_V)
        kUn = key_batch(gr, N_UNWRITTEN)
        kInt = key_batch(gi, N_INT)
        vInt = [val_near_orth(vP, gi) for _ in range(N_INT)]

        arm = {}
        # m1
        U, Vm = write_pair(U0, V0, ks[0], vP)
        e = errs(U, Vm, [ks[0]], vP)[0]
        en = statistics.median(errs(U, Vm, kUn, vP))
        arm["m1"] = {"E": e, "E_null": en, "S": en - e}
        # m2
        U, Vm = write_pair(U, Vm, ks[1], vP)
        e2 = errs(U, Vm, ks[:2], vP)
        en2 = statistics.median(errs(U, Vm, kUn, vP))
        arm["m2"] = {"E": e2, "E_null": en2, "S": [en2 - x for x in e2]}
        # m4
        U, Vm = write_pair(U, Vm, ks[2], vP)
        U, Vm = write_pair(U, Vm, ks[3], vP)
        e4 = errs(U, Vm, ks, vP)
        en4 = statistics.median(errs(U, Vm, kUn, vP))
        arm["m4"] = {"E": e4, "E_null": en4, "S": [en4 - x for x in e4]}
        # m4 + 8 条异值干扰
        for j in range(N_INT):
            U, Vm = write_pair(U, Vm, kInt[j], vInt[j])
        e4i = errs(U, Vm, ks, vP)
        en4i = statistics.median(errs(U, Vm, kUn, vP))
        arm["m4_int8"] = {"E": e4i, "E_null": en4i, "S": [en4i - x for x in e4i],
                          "dS": [arm["m4"]["S"][i] - (en4i - e4i[i]) for i in range(4)],
                          "dispS": max(e4i) - min(e4i),
                          "disp_dS": max(abs((en4i - e4i[i]) - (en4i - e4i[j]))
                                         for i in range(4) for j in range(4))}
        arm["_meta"] = {"sigma_m4": [float(x) for x in torch.linalg.svdvals(M._W(U, Vm).detach())[:3]]}
        rows.append({"seed": sd, **arm})

    def flat(path):
        vals = []
        for x in rows:
            d = x
            for p in path.split("."):
                d = d[p]
            vals.extend(d if isinstance(d, list) else [d])
        return vals

    def stat(vals):
        sd_ = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return {"median": statistics.median(vals), "mean": statistics.fmean(vals), "sd": sd_,
                "q05": sorted(vals)[int(0.05 * (len(vals) - 1))],
                "q95": sorted(vals)[int(0.95 * (len(vals) - 1))],
                "three_se_96": 3 * sd_ / (96 ** 0.5)}

    res = {"probe": "57f", "n_instances": len(rows), "n_unwritten": N_UNWRITTEN, "n_int": N_INT,
           "arms": {a: {"E": stat(flat(f"{a}.E")), "E_null": stat(flat(f"{a}.E_null")),
                        **({"S": stat(flat(f"{a}.S"))} if a != "m1" else {})}
                    for a in ("m1", "m2", "m4", "m4_int8")},
           "m1_S": stat(flat("m1.S")),
           "m4_S_by_position": [stat([x["m4"]["S"][i] for x in rows]) for i in range(4)],
           "int8": {"dispS": stat([x["m4_int8"]["dispS"] for x in rows]),
                    "dS_by_position": [stat([x["m4_int8"]["dS"][i] for x in rows]) for i in range(4)],
                    "frac_all_S_positive": sum(1 for x in rows if all(s > 0 for s in x["m4_int8"]["S"])) / len(rows)},
           "sigma_m4_median": [statistics.median([x["_meta"]["sigma_m4"][i] for x in rows]) for i in range(3)],
           "rows": rows}
    with open(os.path.join(OUT, "PROBE57F.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    for a in ("m1", "m2", "m4", "m4_int8"):
        r = res["arms"][a]
        print("── %-9s E 中位 %.4f · E_null 中位 %.4f · S 中位 %.4f（sd %.4f，3se@96 %.4f）"
              % (a, r["E"]["median"], r["E_null"]["median"],
                 r.get("S", res["m1_S"])["median"], r.get("S", res["m1_S"])["sd"],
                 r.get("S", res["m1_S"])["three_se_96"]))
    print("── m4 逐位置 S：%s" % ["%.4f" % s["median"] for s in res["m4_S_by_position"]])
    print("── 干扰后：全键 S>0 的实例 %d/%d · ΔS 逐位置 %s · dispS 中位 %.4f"
          % (round(res["int8"]["frac_all_S_positive"] * len(rows)), len(rows),
             ["%.3f" % s["median"] for s in res["int8"]["dS_by_position"]],
             res["int8"]["dispS"]["median"]))
    print("── m4 后 σ = [%.3f, %.3f, %.3f]" % tuple(res["sigma_m4_median"]))


if __name__ == "__main__":
    main()