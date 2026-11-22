#!/opt/pytorch/bin/python
r"""DIR-58 前提探针 D —— 为开卡**把两个新阈值的量先量出来**（仍不写判据、不裁决）：

  ① **配对分离量**（修 `DEF-58-3`：零对照改为**同种子配对**，而不是跨种子范围）：
     同一种子、同一把未写键 `k_I` 与同一个未写值 `v_I`：
       `S_ctrl(q)` = **只衰减**臂里对 `(k_I,v_I)` 的读数（从未写过）；
       `S_new(q)`  = **1 次写入后**对 `(k_I,v_I)` 的读数（写完 + (q−1) 对衰减）；
       `Δ_s(q) = S_new(q) − S_ctrl(q)` ⇒ 报中位、`3se`、`Pr(Δ>0)`。
  ② **半衰期随水平变化**（`H58-a` 的"非常数"子句）：同一条只衰减曲线上，
     从 `q=0` 的水平测得的 $q_{1/2}$ 与从 `q=2` 的水平测得的 $q_{1/2}$ 之比。
     （若 S 是指数衰减，二者应相等；比值 >1 就说明**没有常数半衰期**。）
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
import probe58b as P58       # noqa: E402

OUT = os.path.join(ROOT, "dir58", "out")
SEEDS = list(range(59000, 59032))
QS = [0, 1, 2, 4, 8]
QW = [1, 2, 4, 8]            # 臂 W 的 q=0 与 q=1 构造同态（DEF-58-1）⇒ 只用 q>=1


def qhalf(S, qs, i0):
    """从 qs[i0] 的水平起，线性内插到一半处；不足则 None。"""
    tgt = S[i0] / 2
    for i in range(i0, len(S) - 1):
        if S[i] >= tgt >= S[i + 1]:
            return qs[i] + (qs[i + 1] - qs[i]) * (S[i] - tgt) / (S[i] - S[i + 1])
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sd in SEEDS:
        P = R.build(sd, "V0")
        gk = torch.Generator().manual_seed(sd + P58.OFF_K)
        gv = torch.Generator().manual_seed(sd + P58.OFF_V)
        gr = torch.Generator().manual_seed(sd + P58.OFF_R)
        gi = torch.Generator().manual_seed(sd + P58.OFF_I)
        ks = P58.key_batch(gk, 4)
        vP = T51.gen_vec(gv, M.D_V)
        kUn = P58.key_batch(gr, 5)
        kI = P58.key_batch(gi, 1)[0]
        vI = None
        while vI is None:
            v = T51.gen_vec(gi, M.D_V)
            if abs(float(v @ vP)) <= 0.30:
                vI = v
        U1, V1 = T48.step_pair(P["U1"].clone(), P["V1"].clone(), ks[0], vP, 1.0)
        rec = {"seed": sd, "d": {}, "w": {}, "ctrl": {}}
        for q in QS:                                     # 臂 D（只衰减）
            U, Vm = U1.clone(), V1.clone()
            for _ in range(q):
                U, Vm = T48.step_pair(U, Vm, ks[0], vP, 0.0)
            En = st.median(P58.read_errors(U, Vm, kUn, vP))
            rec["d"]["q%d" % q] = En - P58.read_errors(U, Vm, [ks[0]], vP)[0]
            rec["ctrl"]["q%d" % q] = En - P58.read_errors(U, Vm, [kI], vI)[0]
        for q in QW:                                     # 臂 W（1 次写入 + q−1 对衰减）
            U, Vm = U1.clone(), V1.clone()
            U, Vm = T48.step_pair(U, Vm, kI, vI, 1.0)
            for _ in range(q - 1):
                U, Vm = T48.step_pair(U, Vm, ks[0], vP, 0.0)
            En = st.median(P58.read_errors(U, Vm, kUn, vP))
            rec["w"]["q%d" % q] = En - P58.read_errors(U, Vm, [kI], vI)[0]
        SD = [rec["d"]["q%d" % q] for q in QS]
        rec["q12_from0"] = qhalf(SD, QS, 0)
        rec["q12_from2"] = qhalf(SD, QS, 2)
        rows.append(rec)

    def med3(v):
        v = [x for x in v if x is not None]
        if not v:
            return None
        m = st.median(v)
        se = st.stdev(v) / (len(v) ** 0.5) if len(v) > 1 else float("nan")
        return {"n": len(v), "med": m, "se": se, "3se": 3 * se, "min": min(v), "max": max(v),
                "q05": sorted(v)[max(0, int(0.05 * len(v)) - 1)] if v else None,
                "pos_frac": sum(1 for x in v if x > 0) / len(v)}

    res = {"probe": "58d", "n_seeds": len(rows), "qs": QS, "qw": QW,
           "paired_delta": {("q%d" % q): med3([r["w"]["q%d" % q] - r["ctrl"]["q%d" % q]
                                               for r in rows if ("q%d" % q) in r["w"]]) for q in QW},
           "ctrl": {("q%d" % q): med3([r["ctrl"]["q%d" % q] for r in rows]) for q in QS},
           "s_new": {("q%d" % q): med3([r["w"]["q%d" % q] for r in rows]) for q in QW},
           "half_life": {"from_q0": med3([r["q12_from0"] for r in rows]),
                         "from_q2": med3([r["q12_from2"] for r in rows])},
           "rows": rows}
    ratios = [r["q12_from2"] / r["q12_from0"] for r in rows
              if r["q12_from0"] and r["q12_from2"]]
    res["half_life"]["ratio"] = med3(ratios)
    with open(os.path.join(OUT, "PROBE58D.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("探针 58D：%d 种子 · 配对分离量 Δ = S_new − S_ctrl（同种子同键同值）" % len(rows))
    print(" q    中位 Δ    3se     Pr(Δ>0)   S_new中位  S_ctrl中位")
    for q in QW:
        d = res["paired_delta"]["q%d" % q]
        print("%2d  %+8.4f  %6.4f    %5.2f     %+7.4f   %+7.4f"
              % (q, d["med"], d["3se"], d["pos_frac"],
                 res["s_new"]["q%d" % q]["med"], res["ctrl"]["q%d" % q]["med"]))
    h = res["half_life"]
    print("── 半衰期（对）：从 q=0 起 %.3f（n=%d）· 从 q=2 起 %.3f（n=%d）· 比值 %.3f ± %.3f（3se）· 范围 [%.2f, %.2f]"
          % (h["from_q0"]["med"], h["from_q0"]["n"], h["from_q2"]["med"], h["from_q2"]["n"],
             h["ratio"]["med"], h["ratio"]["3se"], h["ratio"]["min"], h["ratio"]["max"]))
    print("   比值 q05 = %.2f · Pr(ratio>1) = %.2f" % (h["ratio"]["q05"], h["ratio"]["pos_frac"]))


if __name__ == "__main__":
    main()