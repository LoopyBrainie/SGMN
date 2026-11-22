#!/opt/pytorch/bin/python
r"""DIR-58 前提探针 A —— **S 的时间衰减曲线是否存在且非退化**（框架 §7.2「时间与情节状态 · S 的更新、检索、容量和遗忘」）。

★ 探针纪律（`loop.md`）：**不写阈值、不写判据、不产出裁决**——只把"能不能问"变成装置事实。
★ 与 D-79 的边界：D-79 测的是**写入后**的读数；本探针测**无数据写入（α=0）**、只保留 λ 正则时的**时间衰减**。
★ 与 D-67（DIR-22）的边界：DIR-22 在**旧装置**（`dir22/sgmn_module_hebb_v7.py`）上证明了**解析不变量** $D=(1-\eta\lambda)^m$；
  本探针在**当前装置**（`dir31/33/48`，$W=UV^\top$ 因子化）上问的是**可测曲线**（半衰期、动态范围、单调性），不是同一件事。
★ 读出量（D-79 纪律 140）：$E=\|Wk-v\|$、$E_{\rm null}$=5 把未写键中位、$S=E_{\rm null}-E$；**不得**用 `cos`。
"""
import json
import os
import statistics as st
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

OUT = os.path.join(ROOT, "dir58", "out")
OFF_K, OFF_V, OFF_R = 59501, 59502, 59503
SEEDS = list(range(59000, 59032))
QS = [0, 1, 2, 4, 8, 16, 32, 64]      # 以「对」为单位（1 对 = N_REP = 16 内步）


def key_batch(g, m, max_cos=0.30):
    ks = []
    while len(ks) < m:
        k = R.KEY_SCALE * T51.gen_vec(g, M.D_K)
        if all(abs(float((k @ q) / (k.norm() * q.norm()))) <= max_cos for q in ks):
            ks.append(k)
    return ks


def read_errors(U, Vm, ks, vP):
    W = M._W(U, Vm)
    return [float((W @ k - vP).norm()) for k in ks]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sd in SEEDS:
        P = R.build(sd, "V0")
        gk = torch.Generator().manual_seed(sd + OFF_K)
        gv = torch.Generator().manual_seed(sd + OFF_V)
        gr = torch.Generator().manual_seed(sd + OFF_R)
        ks = key_batch(gk, 4)
        vP = T51.gen_vec(gv, M.D_V)
        kUn = key_batch(gr, 5)
        U0, V0 = T48.step_pair(P["U1"].clone(), P["V1"].clone(), ks[0], vP, 1.0)
        W0 = M._W(U0, V0).detach()
        n0 = float(W0.norm())
        rec = {"seed": sd, "W0_norm": n0, "T": [], "E": [], "E_null": [], "S": [],
               "W_norm": [], "E_new_key": []}
        for q in QS:
            U, Vm = U0.clone(), V0.clone()
            for _ in range(q):                       # 每次 = N_REP 内步，α=0
                U, Vm = T48.step_pair(U, Vm, ks[0], vP, 0.0)
            E = read_errors(U, Vm, [ks[0]], vP)[0]
            En = st.median(read_errors(U, Vm, kUn, vP))
            W = M._W(U, Vm).detach()
            rec["T"].append(q * T48.N_REP)
            rec["E"].append(E); rec["E_null"].append(En); rec["S"].append(En - E)
            rec["W_norm"].append(float(W.norm()))
            rec["E_new_key"].append(read_errors(U, Vm, [ks[1]], vP)[0])
        rows.append(rec)

    def col(f, i):
        return [r[f][i] for r in rows]

    res = {"probe": "58a", "n_seeds": len(rows), "qs": QS, "n_rep": T48.N_REP,
           "device": {"ETA": M.ETA, "LAM": M.LAM, "D_K": M.D_K, "D_V": M.D_V,
                      "KEY_SCALE": R.KEY_SCALE},
           "curve": {}, "rows": rows}
    for i, q in enumerate(QS):
        S = col("S", i); E = col("E", i); En = col("E_null", i)
        wn = col("W_norm", i); en = col("E_new_key", i)
        res["curve"]["q%d" % q] = {
            "T": q * T48.N_REP,
            "S_med": st.median(S), "S_min": min(S), "S_q05": sorted(S)[1],
            "S_pos_frac": sum(1 for x in S if x > 0) / len(S),
            "E_med": st.median(E), "E_null_med": st.median(En),
            "W_ratio_med": st.median([wn[j] / rows[j]["W0_norm"] for j in range(len(rows))]),
            "E_new_key_med": st.median(en)}
    # 单调性 / 半衰期（按中位曲线；以 q=0 的 S 为基准）
    Sm = [res["curve"]["q%d" % q]["S_med"] for q in QS]
    mono = all(Sm[i + 1] <= Sm[i] + 1e-12 for i in range(len(Sm) - 1))
    half = None
    for i in range(len(Sm) - 1):
        if Sm[i] >= Sm[0] / 2 >= Sm[i + 1]:
            half = QS[i] + (QS[i + 1] - QS[i]) * (Sm[i] - Sm[0] / 2) / (Sm[i] - Sm[i + 1])
            break
    ratio16 = [res["curve"]["q%d" % q]["W_ratio_med"] for q in QS]
    res["summary"] = {
        "S0_med": Sm[0], "S_last_med": Sm[-1], "S_min_over_curve": min(Sm),
        "dynamic_range": (Sm[0] - Sm[-1]) / Sm[0] if Sm[0] else None,
        "monotone_nonincreasing": mono,
        "half_life_pairs": half, "half_life_steps": (half * T48.N_REP) if half else None,
        "W_ratio_at_q1": ratio16[1], "W_ratio_at_q8": ratio16[QS.index(8)],
        "analytic_per_pair": (1 - M.ETA * M.LAM) ** T48.N_REP,
        "S0_all_positive": all(res["curve"]["q0"]["S_pos_frac"] == 1.0 for _ in [0])}
    with open(os.path.join(OUT, "PROBE58A.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("探针 58A：%d 种子 · η=%.3f λ=%.3f · 每「对」%d 内步" % (len(rows), M.ETA, M.LAM, T48.N_REP))
    print(" q    T     S_med      S_min    S>0     E_med   E_null  ‖W‖/‖W0‖  E(未写键)")
    for q in QS:
        c = res["curve"]["q%d" % q]
        print("%2d %4d  %+8.4f  %+8.4f  %5.3f   %6.4f  %6.4f   %6.4f   %7.4f"
              % (q, c["T"], c["S_med"], c["S_min"], c["S_pos_frac"], c["E_med"],
                 c["E_null_med"], c["W_ratio_med"], c["E_new_key_med"]))
    s = res["summary"]
    print("── S0=%.4f · S_end=%.4f · 动态范围=%.3f · 单调不增=%s · 半衰期=%.2f 对（%.0f 内步）"
          % (s["S0_med"], s["S_last_med"], s["dynamic_range"], s["monotone_nonincreasing"],
             s["half_life_pairs"] or -1, s["half_life_steps"] or -1))
    print("── ‖W‖ 每对保留比：q1=%.5f · q8=%.5f · 解析 (1−ηλ)^16=%.5f"
          % (s["W_ratio_at_q1"], s["W_ratio_at_q8"], s["analytic_per_pair"]))


if __name__ == "__main__":
    main()