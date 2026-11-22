#!/opt/pytorch/bin/python
r"""DIR-58 前提探针 C —— 两件补充事实（仍不写判据）：
  ① **未写键的 `S_new` 零对照**：在**只衰减**臂里，对"从未写过的键 + 从未写过的值"算
     $S_{\rm ctrl}=E_{\rm null}-\|Wk_I-v_I\|$。若 $\approx0$，则"新关联可读"确实**只能**由写入产生
     ⇒ 探针 58B 的 `S_new` 是一个**可用**的通道分离量。
  ② **`q*` 的逐种子分布**（不只用中位）：$q^\ast$ = 用臂 D 曲线内插出与臂 W 同 `S_old` 的衰减时间。
     若逐种子 $q^\ast$ 分散极大，则"1 次写入 ≈ 常数对数的衰减"**不成立**（只能当噪声讲）。
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
import probe58b as P58       # noqa: E402  复用同一 key/读出定义（同文件同式）

OUT = os.path.join(ROOT, "dir58", "out")
SEEDS = list(range(59000, 59032))
QS = [0, 1, 2, 4, 8]


def interp_qstar(D, W):
    for i in range(len(D) - 1):
        if D[i] >= W >= D[i + 1]:
            return QS[i] + (QS[i + 1] - QS[i]) * (D[i] - W) / (D[i] - D[i + 1])
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    ctrl = {"q0": [], "q4": []}
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
        for q in (0, 4):
            U, Vm = U1.clone(), V1.clone()
            for _ in range(q):
                U, Vm = T48.step_pair(U, Vm, ks[0], vP, 0.0)
            En = st.median(P58.read_errors(U, Vm, kUn, vP))
            E_nw = P58.read_errors(U, Vm, [kI], vI)[0]      # 未写键 + 未写值
            ctrl["q%d" % q].append({"seed": sd, "S_ctrl": En - E_nw, "E_null": En, "E_nw": E_nw})

    B = json.load(open(os.path.join(OUT, "PROBE58B.json")))
    qstars = {q: [] for q in QS}
    for r in B["rows"]:
        D = [r["d"]["q%d" % q]["S_old"] for q in QS]
        for q in QS:
            w = r["w"]["q%d" % q]["S_old"]
            qs_ = interp_qstar(D, w)
            qstars[q].append(qs_)

    res = {"probe": "58c", "n_seeds": len(SEEDS),
           "ctrl_never_written": {k: {"S_med": st.median([x["S_ctrl"] for x in v]),
                                      "S_min": min(x["S_ctrl"] for x in v),
                                      "S_max": max(x["S_ctrl"] for x in v),
                                      "E_nw_med": st.median([x["E_nw"] for x in v]),
                                      "E_null_med": st.median([x["E_null"] for x in v])}
                                  for k, v in ctrl.items()},
           "qstar_per_seed": {("q%d" % q): {"n": len([x for x in qstars[q] if x is not None]),
                                            "med": st.median([x for x in qstars[q] if x is not None]) if any(qstars[q]) else None,
                                            "min": min([x for x in qstars[q] if x is not None], default=None),
                                            "max": max([x for x in qstars[q] if x is not None], default=None),
                                            "frac_in_range": sum(1 for x in qstars[q] if x is not None) / len(SEEDS)}
                                        for q in QS},
           "rows_ctrl": ctrl}
    with open(os.path.join(OUT, "PROBE58C.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("探针 58C：未写键 S_ctrl 零对照")
    for k in ("q0", "q4"):
        c = res["ctrl_never_written"][k]
        print("  %s：S_ctrl 中位 %+.4f（min %+.4f / max %+.4f）· E_nw %.4f · E_null %.4f"
              % (k, c["S_med"], c["S_min"], c["S_max"], c["E_nw_med"], c["E_null_med"]))
    print("q*（臂 W 的一次写入 ≈ 多少对纯衰减；逐种子）")
    for q in QS:
        s = res["qstar_per_seed"]["q%d" % q]
        print("  q=%d：可内插 %d/32 · 中位 %s · 范围 [%s, %s]"
              % (q, s["n"], ("%.2f" % s["med"]) if s["med"] is not None else "—",
                 ("%.2f" % s["min"]) if s["min"] is not None else "—",
                 ("%.2f" % s["max"]) if s["max"] is not None else "—"))


if __name__ == "__main__":
    main()